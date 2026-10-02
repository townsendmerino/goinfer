//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
)

// MC3 (docs/tasks/task-concurrency-2026-09.md): batched decode — one decode token for up to batchMaxSeqs sequences in
// ONE command buffer, each sequence on its own MC1 resident KV slot at its own position, and every logit bit-identical
// to production's single-token forward on that sequence (TestMC3Step_bitIdentical). Everything that is per-row work
// runs production's own kernels, per sequence: norm + int8 quantisation, RoPE, the KV store into that sequence's slot,
// attention over that slot (attention_fa at or above attnFADepthFloor, exactly as a single-token step would plan it),
// ctx quantisation, SwiGLU + quantisation, the final norm. The five matmuls run once for all sequences, on the
// matrix units, reproducing production's arithmetic:
//   - W4A8 projections (mc3_bt: qkv +bias, o +residual, gate|up; mc3_btd: down +residual): an 8-token fragment column,
//     weights built in registers through thread_elements(), each 32-k group's integer sum exact in the f32 fragment
//     (nibble-8 and the int8 activations are exact in half), accumulated per lane residue in the decode GEMV's lane
//     order and reduced by simd_sum's own xor tree 1, 2, 4, 8, 16 (batchSimdSumTreeOK checks that tree at build);
//   - the int8 LM head (mc3_lm): slab integers converted and summed in int32 — the same integer in any order.
// Measured: docs/measurements/concurrency-mc3-s0-2026-09-26.md (kernels), concurrency-mc3-s1-2026-09-26.md (in sequence:
// B = 4 at 1.73–1.84x one stream's aggregate; B = 1 loses, so the decoder calls this at B >= batchMinSeqs only).

// batchMaxSeqs is the fragment's token width: every batched matmul computes 8 token columns.
const batchMaxSeqs = 8

// batchMinSeqs is the smallest batch the decoder should send here. At B = 1 the step costs 1.6–1.95x a production
// token (the fragment's cost does not shrink with the batch); B = 2 measured 0.98–1.10x (S1), about even, and it
// spares the per-token KV-slot rebinding two interleaved single-token decodes would pay.
const batchMinSeqs = 2

// batchKernelsHeader is what batchKernels needs from allKernels' preamble: the Metal header and the rows kernels' unroll
// pragma (kernels.go), spelled identically.
const batchKernelsHeader = `#include <metal_stdlib>
using namespace metal;
#define SA_ROWS_UNROLL _Pragma("clang loop unroll(full)")
`

const batchKernels = `
inline void mc3_frag(ushort lane, thread ushort& fm, thread ushort& fn) {
    const ushort qid = lane >> 2;
    fm = (qid & 4) + ((lane >> 1) & 3);
    fn = (qid & 2)*2 + (lane & 1)*2;
}

// aT[k][t] = half(aq[t*K + k]) for t < M, 0 above: the MMA kernels' right operand from M rows of int8 activations.
kernel void mc3_pack(device const char* aq[[buffer(0)]], device half* aT[[buffer(1)]], constant uint& M[[buffer(2)]],
    constant uint& K[[buffer(3)]], uint gid[[thread_position_in_grid]]) {
    uint k = gid >> 3u, t = gid & 7u;
    if (k >= K) return;
    aT[gid] = t < M ? half(float(aq[t*K + k])) : 0.0h;
}
// the LM head's permuted layout (see mc3_lm): logical row j = 32s + 8kb + kk holds physical k = 32s + 8(kk/2) + 2kb + kk%2.
kernel void mc3_pack_lm(device const char* aq[[buffer(0)]], device half* aT[[buffer(1)]], constant uint& M[[buffer(2)]],
    constant uint& K[[buffer(3)]], uint gid[[thread_position_in_grid]]) {
    uint j = gid >> 3u, t = gid & 7u;
    if (j >= K) return;
    uint s = j >> 5u, kb = (j >> 3u) & 3u, kk = j & 7u;
    uint k = s*32u + 8u*(kk >> 1u) + 2u*kb + (kk & 1u);
    aT[gid] = t < M ? half(float(aq[t*K + k])) : 0.0h;
}

// mc3_bt: gemv_w4a8_sa_rows (mode 0), _sa_bias_rows (1) and _sa_resid_rows (2) for 8 activation rows. Residue r = g mod
// 32 of the GEMV's lane-strided group loop keeps its own partial (r = 8*sgid + slot); the partials reduce by simd_sum's
// tree — slots ^1/^2/^4 in registers, simdgroups ^8/^16 through Q — so each output is the GEMV's simd_sum exactly.
template <uint FB>
kernel void mc3_bt(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    device const float* bias[[buffer(8)]], constant uint& mode[[buffer(9)]],
    threadgroup float2* Q[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    ushort fm, fn; mc3_frag(lane, fm, fn);
    const ushort sh0 = 4u*fn, sh1 = 4u*(fn+1u);
    float2 P[FB][8];
    device const uint4* wrow[FB];
    device const half* srow[FB];
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm; wrow[b] = wq + row*G; srow[b] = sct + row*G;
        for (ushort i=0;i<8;i++) P[b][i] = float2(0.0f);
    }
    const uint nc = (G + 31u) >> 5u;
    for (uint c = 0; c < nc; c++) {
        SA_ROWS_UNROLL for (ushort i = 0; i < 8; i++) {
            const uint g = c*32u + sgid*8u + i;
            if (g < G) {
                simdgroup_half8x8 A[4];
                for (ushort kb=0; kb<4; kb++) simdgroup_load(A[kb], aT + (g*32u + kb*8u)*8u, 8);
                for (uint b=0;b<FB;b++) {
                    uint4 w = wrow[b][g];
                    float s = float(srow[b][g]);
                    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);
                    for (ushort kb=0; kb<4; kb++) {
                        uint word = w[kb];
                        simdgroup_half8x8 Wf;
                        Wf.thread_elements()[0] = half(int((word >> sh0) & 0xFu) - 8);
                        Wf.thread_elements()[1] = half(int((word >> sh1) & 0xFu) - 8);
                        simdgroup_multiply_accumulate(acc, Wf, A[kb], acc);
                    }
                    P[b][i].x += acc.thread_elements()[0] * s;
                    P[b][i].y += acc.thread_elements()[1] * s;
                }
            }
        }
    }
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 q = ((P[b][0] + P[b][1]) + (P[b][2] + P[b][3])) + ((P[b][4] + P[b][5]) + (P[b][6] + P[b][7]));
        Q[(sgid*FB + b)*32u + lane] = q;
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid != 0) return;
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 v = (Q[(0u*FB + b)*32u + lane] + Q[(1u*FB + b)*32u + lane]) + (Q[(2u*FB + b)*32u + lane] + Q[(3u*FB + b)*32u + lane]);
        uint row = f0 + b*8u + fm;
        for (ushort e = 0; e < 2; e++) {
            uint t = fn + e;
            if (t >= M) continue;
            float vv = e == 0 ? v.x : v.y;
            if (mode == 1u) out[t*N + row] = vv*asc[t] + bias[row];
            else if (mode == 2u) out[t*N + row] += vv*asc[t];
            else out[t*N + row] = vv*asc[t];
        }
    }
    }
}
template [[host_name("mc3_bt_fb2")]] kernel decltype(mc3_bt<2>) mc3_bt<2>;

// mc3_btd: gemv_w4a8_resid_staged for 8 rows. That kernel sums per 8-k WORD (lane L: words L, L+32, ...), so each k
// block's MMA runs into its own zero fragment (exactly that word's integer sum) and residues are words.
template <uint FB>
kernel void mc3_btd(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    threadgroup float2* Q[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    ushort fm, fn; mc3_frag(lane, fm, fn);
    const ushort sh0 = 4u*fn, sh1 = 4u*(fn+1u);
    float2 P[FB][8];
    device const uint4* wrow[FB];
    device const half* srow[FB];
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm; wrow[b] = wq + row*G; srow[b] = sct + row*G;
        for (ushort i=0;i<8;i++) P[b][i] = float2(0.0f);
    }
    const uint nc = (G + 7u) >> 3u;
    for (uint c = 0; c < nc; c++) {
        SA_ROWS_UNROLL for (ushort j = 0; j < 2; j++) {
            const uint g = c*8u + sgid*2u + j;
            if (g < G) {
                simdgroup_half8x8 A[4];
                for (ushort kb=0; kb<4; kb++) simdgroup_load(A[kb], aT + (g*32u + kb*8u)*8u, 8);
                for (uint b=0;b<FB;b++) {
                    uint4 w = wrow[b][g];
                    float s = float(srow[b][g]);
                    SA_ROWS_UNROLL for (ushort kb=0; kb<4; kb++) {
                        uint word = w[kb];
                        simdgroup_half8x8 Wf;
                        Wf.thread_elements()[0] = half(int((word >> sh0) & 0xFu) - 8);
                        Wf.thread_elements()[1] = half(int((word >> sh1) & 0xFu) - 8);
                        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);
                        simdgroup_multiply_accumulate(acc, Wf, A[kb], acc);
                        P[b][j*4+kb].x += acc.thread_elements()[0] * s;
                        P[b][j*4+kb].y += acc.thread_elements()[1] * s;
                    }
                }
            }
        }
    }
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 q = ((P[b][0] + P[b][1]) + (P[b][2] + P[b][3])) + ((P[b][4] + P[b][5]) + (P[b][6] + P[b][7]));
        Q[(sgid*FB + b)*32u + lane] = q;
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid != 0) return;
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 v = (Q[(0u*FB + b)*32u + lane] + Q[(1u*FB + b)*32u + lane]) + (Q[(2u*FB + b)*32u + lane] + Q[(3u*FB + b)*32u + lane]);
        uint row = f0 + b*8u + fm;
        if (fn < M) out[fn*N + row] += v.x * asc[fn];
        if (fn+1u < M) out[(fn+1u)*N + row] += v.y * asc[fn+1u];
    }
    }
}
template [[host_name("mc3_btd_fb2")]] kernel decltype(mc3_btd<2>) mc3_btd<2>;

// mc3_lm: gemv_w8a8_coal for 8 rows. A 32-k slab's four MMAs stay below 2^24 (32*128*128), so each lane converts the
// slab's exact integer and sums it in int32 — production's integer, in any order. Inside a slab k is permuted, the same
// way for both operands, so each lane's weights are 8 contiguous bytes (mc3_pack_lm lays the activations out to match).
template <uint FB>
kernel void mc3_lm(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const half* aTp[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    ushort fm, fn; mc3_frag(lane, fm, fn);
    const uint f0 = (tgid*4u + sgid) * (8u*FB);
    int2 acc[FB];
    device const uint2* wrow[FB];
    for (uint b=0;b<FB;b++) { acc[b] = int2(0); wrow[b] = reinterpret_cast<device const uint2*>(bq + (f0 + b*8u + fm)*K) + (fn >> 1); }
    const uint S = K >> 5u;
    for (uint s = 0; s < S; s++) {
        simdgroup_half8x8 A[4];
        for (ushort kb=0; kb<4; kb++) simdgroup_load(A[kb], aTp + (s*32u + kb*8u)*8u, 8);
        for (uint b=0;b<FB;b++) {
            uint2 w = wrow[b][s*4u];
            char4 cx = as_type<char4>(w.x), cy = as_type<char4>(w.y);
            simdgroup_half8x8 W0, W1, W2, W3;
            W0.thread_elements()[0] = half(cx.x); W0.thread_elements()[1] = half(cx.y);
            W1.thread_elements()[0] = half(cx.z); W1.thread_elements()[1] = half(cx.w);
            W2.thread_elements()[0] = half(cy.x); W2.thread_elements()[1] = half(cy.y);
            W3.thread_elements()[0] = half(cy.z); W3.thread_elements()[1] = half(cy.w);
            simdgroup_float8x8 f = make_filled_simdgroup_matrix<float,8,8>(0.0f);
            simdgroup_multiply_accumulate(f, W0, A[0], f);
            simdgroup_multiply_accumulate(f, W1, A[1], f);
            simdgroup_multiply_accumulate(f, W2, A[2], f);
            simdgroup_multiply_accumulate(f, W3, A[3], f);
            acc[b] += int2(int(f.thread_elements()[0]), int(f.thread_elements()[1]));
        }
    }
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm;
        if (fn < M) out[fn*N + row] = float(acc[b].x) * asc[fn] * bsc[row];
        if (fn+1u < M) out[(fn+1u)*N + row] = float(acc[b].y) * asc[fn+1u] * bsc[row];
    }
}
template [[host_name("mc3_lm_fb4")]] kernel decltype(mc3_lm<4>) mc3_lm<4>;

// mc3_sstree: the build-time check that simd_sum IS the xor tree 1, 2, 4, 8, 16 the kernels above reproduce. Writes, per
// simdgroup, simd_sum(v) and the tree's own value; batchSimdSumTreeOK compares them bit for bit.
kernel void mc3_sstree(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    float v = in[gid];
    float b = v;
    b += simd_shuffle_xor(b, 1); b += simd_shuffle_xor(b, 2); b += simd_shuffle_xor(b, 4);
    b += simd_shuffle_xor(b, 8); b += simd_shuffle_xor(b, 16);
    float s = simd_sum(v);
    if (lane == 0) { uint w = gid >> 5; out[w*2] = s; out[w*2+1] = b; }
}
`

// batchState is a resident's MC3 batched-step machinery: pipelines, per-row scratch for batchMaxSeqs sequences, and
// per-sequence uniforms. nil when the resident cannot batch (batchIneligible).
type batchState struct {
	pack, packLM, bt, btd, lm Pipeline
	xB, qkvB, guB, logitsB    Buffer // f32 rows
	aqB, cqB, mqB, dqB        Buffer // int8 rows
	aScB, cScB, mScB, dScB    Buffer // one f32 per row
	aT, aTp                   Buffer // half [K][8]
	uM, uQKV, uGU, uV         Buffer
	uMode                     [3]Buffer
	uPos, uNKeys, uQTemp      []Buffer
	uFANSplit                 []Buffer
	// per-row uniforms and outputs of a device draw (ForwardSample's gumbel dispatches, run on a row of logitsB)
	uGInvT, uGK0, uGK1, uGD0, uGD1 []Buffer
	gOut                           Buffer // [batchMaxSeqs] int32 ids
	// S3: per-row positions and Q temperature scales for mc3_rope2_rows, and per-row attention outputs, so the ctx
	// quantisation runs over every row at once
	posB, qtB, ctxB           Buffer
	nKeysB, rowmapB, slotOffB Buffer   // per-row key counts; the rows mc3_attention_rows serves; each row's slot offset
	uCount                    []Buffer // uCount[k] holds k (0..batchMaxSeqs): a row count for a dispatch
	noBias                    Buffer
	qkvRows, nHhd, kOff, vOff int
	// S4: qkv / gate|up run as per-row production GEMVs at B <= rowsQKV / rowsGU, and as the fragment above that (0:
	// always the fragment). Set by calibrateRows at build; tests set them to force either path.
	rowsQKV, rowsGU int
	// steps / seqs / maxSeqs count completed batched steps, the sequences they served, and the largest batch — read by
	// tests to confirm a concurrent run really batched. Written only inside forwardMulti, which the decoder serialises.
	steps, seqs, maxSeqs int
}

// batchSeq is one sequence of a batched step: its resident KV slot, the position it decodes, and its input embedding.
type batchSeq struct {
	slot, pos int
	emb       []float32
	draw      *decoder.ResidentBatchDraw // non-nil: draw this row's next token on-device (ForwardSample's draw)
}

// batchIneligible is why this resident cannot run a batched step ("" if it can): MC3 covers the plain dense W4A8 decode
// path — the family flags every per-row kernel in forwardMulti assumes — on at least two resident KV slots.
func (r *resident) batchIneligible() string {
	switch {
	case len(r.kvSlotBufs) < 2:
		return "one resident KV slot"
	case !r.kvContig || r.kvF32:
		return "KV slots that are not one f16 allocation per layer"
	case r.moe != nil || r.g4moe != nil:
		return "MoE"
	case r.w8:
		return "int8 weights (the batched step's kernels read int4; docs/tasks/task-metal-int8-2026-10.md, slice 3)"
	case r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm || r.decodeLaneW4F16 || r.nonGatedMLP ||
		r.outBias || r.qkNorm || r.learnedPos || r.attnSink:
		return "a family variant the batched step does not reproduce"
	}
	g0 := r.layers[0].geom
	if g0 == nil {
		return "no attention geometry"
	}
	for _, L := range r.layers {
		if L.qGate || L.delta != nil || L.geom == nil || L.geom.kEqV || L.window != 0 ||
			L.geom.hd != g0.hd || L.geom.kvDim != g0.kvDim || L.geom.nKV != g0.nKV {
			return "a layer the batched step does not reproduce"
		}
	}
	qkvRows := r.nH*g0.hd + 2*g0.kvDim
	for _, c := range []struct{ n, by int }{{qkvRows, 16}, {r.H, 16}, {2 * r.I, 16}, {r.V, 128}, {r.H, 32},
		{r.nH * g0.hd, 32}, {r.I, 32}} {
		if c.n%c.by != 0 {
			return fmt.Sprintf("a width (%d) the batched kernels do not tile (%% %d)", c.n, c.by)
		}
	}
	return ""
}

// buildBatch compiles and allocates the batched step when the resident is eligible and this GPU's simd_sum is the tree
// the kernels reproduce; otherwise it leaves r.batch nil (the decoder then never batches). It compiles with the build's
// own fast-math setting (r.preciseMath), so the batched kernels round like the kernels they must match.
func (r *resident) buildBatch() {
	if why := r.batchIneligible(); why != "" {
		return
	}
	compile := r.d.CompileLibrary
	if r.preciseMath {
		compile = r.d.CompileLibraryPrecise
	}
	lib, err := compile(batchKernelsHeader+batchKernels, MSL3_1) // its own small library: allKernels is compiled once, above
	if err != nil {
		fmt.Fprintf(os.Stderr, "metal: batched decode off (compile: %v)\n", err)
		return
	}
	p := func(n string) Pipeline {
		pp, e := r.d.NewComputePipeline(lib, n)
		if e != nil {
			panic(fmt.Sprintf("metal pipeline %s: %v", n, e))
		}
		return pp
	}
	if !batchSimdSumTreeOK(r.d, p("mc3_sstree")) {
		fmt.Fprintf(os.Stderr, "metal: batched decode off — this GPU's simd_sum is not the xor tree the batched kernels reproduce, so their output would not be bit-identical\n")
		return
	}
	d, g0 := r.d, r.layers[0].geom
	b := &batchState{nHhd: r.nH * g0.hd}
	b.qkvRows = b.nHhd + 2*g0.kvDim
	b.kOff, b.vOff = b.nHhd*4, (b.nHhd+g0.kvDim)*4
	b.pack, b.packLM, b.bt, b.btd, b.lm = p("mc3_pack"), p("mc3_pack_lm"), p("mc3_bt_fb2"), p("mc3_btd_fb2"), p("mc3_lm_fb4")
	const B = batchMaxSeqs
	maxK := max(r.H, b.nHhd, r.I)
	b.xB, b.qkvB, b.guB, b.logitsB = d.NewBufferLen(B*r.H), d.NewBufferLen(B*b.qkvRows), d.NewBufferLen(B*2*r.I), d.NewBufferLen(B*r.V)
	b.aqB, b.cqB, b.mqB, b.dqB = d.NewBufferBytes(B*r.H), d.NewBufferBytes(B*b.nHhd), d.NewBufferBytes(B*r.H), d.NewBufferBytes(B*r.I)
	b.aScB, b.cScB, b.mScB, b.dScB = d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B)
	b.aT, b.aTp = NewBufferU16s(d, make([]uint16, maxK*8)), NewBufferU16s(d, make([]uint16, r.H*8))
	b.uM = NewBufferU32(d, 1)
	b.uQKV, b.uGU, b.uV = NewBufferU32(d, uint32(b.qkvRows)), NewBufferU32(d, uint32(2*r.I)), NewBufferU32(d, uint32(r.V))
	b.uMode = [3]Buffer{NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferU32(d, 2)}
	b.noBias = d.NewBufferLen(max(r.H, 2*r.I))
	for range B {
		b.uPos = append(b.uPos, NewBufferU32(d, 0))
		b.uNKeys = append(b.uNKeys, NewBufferU32(d, 1))
		b.uQTemp = append(b.uQTemp, NewBufferFloats(d, []float32{1}))
		b.uFANSplit = append(b.uFANSplit, NewBufferU32(d, 1))
		b.uGInvT = append(b.uGInvT, NewBufferFloats(d, []float32{1}))
		b.uGK0, b.uGK1 = append(b.uGK0, NewBufferU32(d, 0)), append(b.uGK1, NewBufferU32(d, 0))
		b.uGD0, b.uGD1 = append(b.uGD0, NewBufferU32(d, 0)), append(b.uGD1, NewBufferU32(d, 0))
	}
	b.gOut = d.NewBufferLen(B)
	b.posB, b.qtB, b.ctxB = d.NewBufferLen(B), NewBufferFloats(d, make([]float32, B)), d.NewBufferLen(B*b.nHhd)
	b.nKeysB, b.rowmapB, b.slotOffB = d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B)
	for k := 0; k <= B; k++ {
		b.uCount = append(b.uCount, NewBufferU32(d, uint32(k)))
	}
	r.batch = b
	r.calibrateRows()
}

// batchTGBytes is mc3_bt's / mc3_btd's threadgroup memory at FB = 2: the Q exchange.
const batchTGBytes = 4 * 2 * 32 * 8

// calibrateRows decides, once, at which batch sizes qkv and gate|up are cheaper as B per-row production GEMVs than as
// one fragment dispatch (MC3 S4, docs/tasks/task-concurrency-2026-09.md). The fragment's cost is fixed for any B up to
// 8, and a GEMV's is per row. On the 7B's large shapes the GEMV is bandwidth-bound and the fragment ALU-bound, so two
// GEMVs beat one fragment (gate|up 0.87 against 1.17 ms). Both run on the real weights of the first layers, one layer's
// weights per dispatch so the cache does not serve them, median of 7 command buffers after 2 warm-ups. The choice
// changes speed only: the per-row path is production's own kernel on each row.
func (r *resident) calibrateRows() {
	b := r.batch
	n := min(8, r.nL)
	b.uM.SetU32(batchMaxSeqs)
	pq, nq := saRowsPick(r.pSABias, r.pSABiasRows, b.qkvRows, r.gemvRows.qkv)
	pg, ng := saRowsPick(r.pSA, r.pSARows, 2*r.I, r.gemvRows.gu)
	arms := []func(e *Encoder, L *residLayer){
		func(e *Encoder, L *residLayer) { // qkv, fragment
			e.DispatchTG(b.bt, b.qkvRows/16*128, 128, batchTGBytes, L.qkvW, L.qkvS, b.aT, b.aScB, b.qkvB, r.uH, b.uQKV, b.uM, L.qkvBias, b.uMode[1])
		},
		func(e *Encoder, L *residLayer) { // qkv, one production GEMV
			e.DispatchTG(pq, nq, 256, r.H*2, L.qkvW, L.qkvS, b.aqB, b.aScB, b.qkvB, L.qkvBias, r.uH)
		},
		func(e *Encoder, L *residLayer) { // gate|up, fragment
			e.DispatchTG(b.bt, 2*r.I/16*128, 128, batchTGBytes, L.guW, L.guS, b.aT, b.mScB, b.guB, r.uH, b.uGU, b.uM, b.noBias, b.uMode[0])
		},
		func(e *Encoder, L *residLayer) { // gate|up, one production GEMV
			e.DispatchTG(pg, ng, 256, r.H*2, L.guW, L.guS, b.mqB, b.mScB, b.guB, r.uH)
		},
	}
	// The arms interleave rep by rep, so both of a pair see the same GPU state, and each keeps its fastest command
	// buffer after 3 warm-ups: a first use of a pipeline, or a clock still ramping, only ever makes a buffer slower.
	best := make([]float64, len(arms))
	for rep := range 10 {
		for a, enc := range arms {
			e := r.q.Begin()
			for l := range n {
				enc(e, &r.layers[l])
			}
			e.End()
			if e.Err() != nil {
				return // keep the fragment everywhere (rowsQKV = rowsGU = 0)
			}
			if t := (e.GPUEnd() - e.GPUStart()) / float64(n); rep >= 3 && (best[a] == 0 || t < best[a]) {
				best[a] = t
			}
		}
	}
	rows := func(frag, gemv float64) int {
		k := 0
		for B := batchMinSeqs; B <= batchMaxSeqs && frag > 0 && gemv > 0 && float64(B)*gemv < frag; B++ {
			k = B
		}
		return k
	}
	b.rowsQKV, b.rowsGU = rows(best[0], best[1]), rows(best[2], best[3])
	fmt.Fprintf(os.Stderr, "metal: batched step: qkv as per-row GEMVs at B <= %d (GEMV %.3f ms, fragment %.3f ms), gate|up at B <= %d (%.3f, %.3f); 0 = always the fragment\n",
		b.rowsQKV, best[1]*1e3, best[0]*1e3, b.rowsGU, best[3]*1e3, best[2]*1e3)
}

// batchSimdSumTreeOK runs mc3_sstree over random vectors spanning 24 binades (where different addition trees round
// differently) and reports whether simd_sum equalled the xor tree 1, 2, 4, 8, 16 on every one.
func batchSimdSumTreeOK(d *Device, p Pipeline) bool {
	const W = 4096 // simdgroups
	in := make([]float32, W*32)
	seed := uint32(12345)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	for i := range in {
		m := float64(rnd())/float64(1<<32) - 0.5
		in[i] = float32(math.Ldexp(m, int(rnd()%24)-12))
	}
	ib, ob := NewBufferFloats(d, in), d.NewBufferLen(W*2)
	q := d.NewCommandQueue()
	e := q.Begin()
	e.DispatchTG(p, W*32, 256, 0, ib, ob)
	e.End()
	if e.Err() != nil {
		return false
	}
	o := ob.Floats()[:W*2]
	for w := 0; w < W; w++ {
		if math.Float32bits(o[w*2]) != math.Float32bits(o[w*2+1]) {
			return false
		}
	}
	return true
}

// canUseAttnFAAt is canUseAttnFA for a given key count rather than the one planNKeys reports: the batched step plans
// each sequence's attention for its own depth, exactly as that sequence's single-token step would.
func (r *resident) canUseAttnFAAt(l, nKeys int) bool {
	if !r.decodeAttnFA || r.attnFAPartial == (Buffer{}) {
		return false
	}
	if r.sandwich || r.postOnly || r.parallelBlock || r.attnSink || r.kvI8 {
		return false
	}
	if r.loraLayers != nil {
		return false
	}
	L := &r.layers[l]
	if L.moe != nil || L.g4moe != nil || L.delta != nil || L.qGate || L.window != 0 {
		return false
	}
	g := L.geom
	return g != nil && g.hd == 128 && attnFAGroupOK(r.nH, g.nKV) && nKeys >= r.attnFAFloor()
}

// forwardMulti runs one decode token for each of seqs — 1..batchMaxSeqs sequences, each on its own resident KV slot at
// its own position — in one command buffer. It returns each row's logits (fresh slices), or, for a row that carries a
// draw, the id ForwardSample would have drawn (ids[m] >= 0, logits[m] nil); ids[m] is -1 for a logits row. It never
// touches the bound slot (r.kc/r.vc); every sequence's KV is addressed through kvSlotBufs. The pipelined executor is
// stopped first: a command buffer it pre-encoded would otherwise run after this one's writes with its own stale plan.
func (r *resident) forwardMulti(seqs []batchSeq) (logits [][]float32, ids []int, err error) {
	return r.forwardMultiInto(seqs, false)
}

// forwardMultiInto is forwardMulti; with argmaxOnly it returns no logits, only each row's argmax, read in place from
// the step's logits buffer (no per-row copy). That is the same id the copied row's argmax gives when no logit transform
// applies, and the caller falls back to the full rows when one does (softcap, a logit scale).
func (r *resident) forwardMultiInto(seqs []batchSeq, argmaxOnly bool) (logits [][]float32, ids []int, err error) {
	b := r.batch
	if b == nil {
		return nil, nil, fmt.Errorf("metal: batched decode unavailable")
	}
	B := len(seqs)
	if B < 1 || B > batchMaxSeqs {
		return nil, nil, fmt.Errorf("metal: batched step of %d sequences (1..%d)", B, batchMaxSeqs)
	}
	if r.loraLayers != nil {
		return nil, nil, fmt.Errorf("metal: batched step with an adapter bound")
	}
	// A slot may carry several rows only as ONE sequence's consecutive positions, in row order (a speculative verify:
	// every row's K/V is stored before its attention runs, layer by layer, so each row sees the rows below it as
	// sequential decode would). Two sequences on one slot would overwrite each other.
	last := map[int]int{}
	for _, q := range seqs {
		if q.slot < 0 || q.slot >= len(r.kvSlotBufs) {
			return nil, nil, fmt.Errorf("metal: batched step: slot %d out of range", q.slot)
		}
		if p, ok := last[q.slot]; ok && q.pos != p+1 {
			return nil, nil, fmt.Errorf("metal: batched step: slot %d repeated at position %d after %d (rows on one slot must be consecutive positions of one sequence)", q.slot, q.pos, p)
		}
		last[q.slot] = q.pos
		if q.pos < 0 || q.pos >= r.ctxCap {
			return nil, nil, fmt.Errorf("metal: batched step: position %d outside the resident context %d", q.pos, r.ctxCap)
		}
		if len(q.emb) != r.H {
			return nil, nil, fmt.Errorf("metal: batched step: embedding length %d != hidden %d", len(q.emb), r.H)
		}
		if q.draw != nil && !r.SampleAvailable() {
			return nil, nil, fmt.Errorf("metal: batched step: a device draw on a resident that cannot sample on-device")
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.stopExec()
	if r.attnFANKV > 0 {
		r.uAttnFAG.SetU32(uint32(r.nH / r.attnFANKV))
	}
	b.uM.SetU32(uint32(B))
	for m, q := range seqs {
		copy(b.xB.Floats()[m*r.H:(m+1)*r.H], q.emb)
		r.addLearnedPosTo(b.xB.Floats()[m*r.H:(m+1)*r.H], q.pos)
		b.uPos[m].SetU32(uint32(q.pos))
		b.uNKeys[m].SetU32(uint32(q.pos + 1))
		scale := float32(1)
		if r.attnTempBeta != 0 {
			scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(q.pos)/r.attnTempOrigMaxPos)))
		}
		b.uQTemp[m].Floats()[0] = scale
		b.posB.U32s()[m] = uint32(q.pos)
		b.nKeysB.U32s()[m] = uint32(q.pos + 1)
		b.slotOffB.U32s()[m] = uint32(q.slot * r.kvSlotBytes[0] / 2) // f16 elements; batchIneligible: uniform layers
		b.qtB.Floats()[m] = scale
		if r.attnFANKV > 0 {
			b.uFANSplit[m].SetU32(uint32(r.attnFASplitFor(q.pos+1, r.attnFANKV)))
		}
	}
	H, I, nHhd := r.H, r.I, b.nHhd
	// Buffer.At SETS the byte offset from the allocation's start — it does not add to one a view already carries — so a
	// sub-row offset is always computed whole (row*n + off), never as row.At(off).
	f32 := func(buf Buffer, m, n int) Buffer { return buf.At(4 * m * n) }
	pack := func(e *Encoder, aq Buffer, K int, uK Buffer) { e.Dispatch(b.pack, K*8, 256, aq, b.aT, b.uM, uK) }
	const tgb = batchTGBytes
	// S4: below the calibrated sizes, qkv and gate|up run production's GEMV once per row (the row's int8 activations
	// and scale in, its output row out), which is production's arithmetic by definition.
	rowsQKV, rowsGU := B <= b.rowsQKV, B <= b.rowsGU
	pq, nq := saRowsPick(r.pSABias, r.pSABiasRows, b.qkvRows, r.gemvRows.qkv)
	pg, ng := saRowsPick(r.pSA, r.pSARows, 2*I, r.gemvRows.gu)
	// Each row's attention plan, once per step: batchIneligible admits only uniform layers without a window, so a row's
	// plan depends on its depth alone. Rows on the per-head kernel run in one dispatch (rowmapB); a row at attention_fa
	// depth keeps its own two dispatches.
	faRow := make([]bool, B)
	nPlain := 0
	for m, q := range seqs {
		faRow[m] = r.canUseAttnFAAt(0, q.pos+1)
		if !faRow[m] {
			b.rowmapB.U32s()[nPlain] = uint32(m)
			nPlain++
		}
	}
	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		g := L.geom
		// the per-row kernels run as one dispatch over all B rows (S3; batch_rows.go): norm+quant, RoPE, ctx quant,
		// SwiGLU+quant. The KV store and attention address each sequence's own slot, so they stay per sequence.
		e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.preNorm, b.aqB, b.aScB, r.uH, r.uEps, r.uAddOne)
		if rowsQKV {
			for m := range B {
				e.DispatchTG(pq, nq, 256, H*2, L.qkvW, L.qkvS, b.aqB.At(m*H), b.aScB.At(4*m), f32(b.qkvB, m, b.qkvRows), L.qkvBias, r.uH)
			}
		} else {
			pack(e, b.aqB, H, r.uH)
			e.DispatchTG(b.bt, b.qkvRows/16*128, 128, tgb, L.qkvW, L.qkvS, b.aT, b.aScB, b.qkvB, r.uH, b.uQKV, b.uM, L.qkvBias, b.uMode[1])
		}
		e.Dispatch(r.pRope2Rows, B*(r.nH*g.half+g.nKV*g.half), 64, b.qkvB, L.invf, g.uHd, b.posB, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, b.qtB, b.uQKV, b.uM)
		// KV store and attention, each row over its OWN slot: a layer's slots are one allocation (kvContig), reached
		// as the slot-0 base (kvSlotBufs[0], offset 0) plus the row's element offset (slotOffB). One dispatch stores
		// every row's K/V; one runs every per-head-kernel row's attention.
		kcAll, vcAll := r.kvSlotBufs[0].kc[l], r.kvSlotBufs[0].vc[l]
		e.Dispatch(r.pKvRows, B*g.kvDim, 64, b.qkvB.At(b.kOff), b.qkvB.At(b.vOff), kcAll, vcAll, g.uKvDim, b.posB, b.uQKV, b.uM, b.slotOffB)
		for m, q := range seqs {
			if !faRow[m] {
				continue
			}
			sb := r.kvSlotBufs[q.slot]
			nSplit := r.attnFASplitFor(q.pos+1, g.nKV)
			shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
			e.DispatchTG(r.pAttnFA, g.nKV*nSplit*128, 128, shmBytes, b.qkvB.At(4*m*b.qkvRows), sb.kc[l], sb.vc[l], r.attnFAPartial,
				g.uNKV, r.uAttnFAG, b.uNKeys[m], r.uScale, L.uWindow, b.uFANSplit[m])
			e.Dispatch(r.pAttnFACombine, r.nH*g.hd, g.hd, r.attnFAPartial, f32(b.ctxB, m, nHhd), r.uAttnFAG, g.uHd, b.uFANSplit[m])
		}
		if nPlain > 0 {
			e.Dispatch(r.pAttnRows, nPlain*r.nH*tgReduceAttn, tgReduceAttn, b.qkvB, kcAll, vcAll, b.ctxB, r.uNH, g.uNKV, g.uHd,
				b.nKeysB, r.uScale, L.uWindow, L.attnSinks, L.uHasSink, b.uCount[nPlain], b.rowmapB, b.slotOffB)
		}
		e.Dispatch(r.pQvRows, B*256, 256, b.ctxB, b.cqB, b.cScB, g.uNHhd)
		pack(e, b.cqB, nHhd, g.uNHhd)
		e.DispatchTG(b.bt, H/16*128, 128, tgb, L.oW, L.oS, b.aT, b.cScB, b.xB, g.uNHhd, r.uH, b.uM, b.noBias, b.uMode[2])
		e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.postNorm, b.mqB, b.mScB, r.uH, r.uEps, r.uAddOne)
		if rowsGU {
			for m := range B {
				e.DispatchTG(pg, ng, 256, H*2, L.guW, L.guS, b.mqB.At(m*H), b.mScB.At(4*m), f32(b.guB, m, 2*I), r.uH)
			}
		} else {
			pack(e, b.mqB, H, r.uH)
			e.DispatchTG(b.bt, 2*I/16*128, 128, tgb, L.guW, L.guS, b.aT, b.mScB, b.guB, r.uH, b.uGU, b.uM, b.noBias, b.uMode[0])
		}
		e.Dispatch(r.pSwRows, B*256, 256, b.guB, b.guB, b.dqB, b.dScB, r.uI, r.uAct)
		pack(e, b.dqB, I, r.uI)
		e.DispatchTG(b.btd, H/16*128, 128, tgb, L.dW, L.dS, b.aT, b.dScB, b.xB, r.uI, r.uH, b.uM)
	}
	e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, r.finalNorm, b.aqB, b.aScB, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(b.packLM, H*8, 256, b.aqB, b.aTp, b.uM, r.uH)
	e.DispatchTG(b.lm, r.V/128*128, 128, 0, r.lmW, r.lmS, b.aTp, b.aScB, b.logitsB, r.uH, b.uV, b.uM)
	// Device draws: ForwardSample's two gumbel dispatches, exactly, on each drawing row of logitsB (the same bits as
	// r.logits would hold) with that row's uniforms. The partial buffers are reused row after row — dispatches in one
	// encoder run in order.
	greedyDraw := make([]bool, B)
	for m, q := range seqs {
		if q.draw == nil {
			continue
		}
		invT := float32(1 / q.draw.Temperature)
		if math.IsInf(float64(invT), 0) { // ForwardSample's own shortcut: the argmax, as greedy
			greedyDraw[m] = true
			continue
		}
		b.uGInvT[m].Floats()[0] = invT
		b.uGK0[m].SetU32(uint32(q.draw.Seed))
		b.uGK1[m].SetU32(uint32(q.draw.Seed >> 32))
		b.uGD0[m].SetU32(uint32(q.draw.Draw))
		b.uGD1[m].SetU32(uint32(q.draw.Draw >> 32))
		const gbThreads = 256
		const gbShmBytes = gbThreads * 2 * 4
		e.DispatchTG(r.pGumbel1, r.gumbelNB*gbThreads, gbThreads, gbShmBytes,
			b.logitsB.At(4*m*r.V), r.uGumbelV, b.uGInvT[m], b.uGK0[m], b.uGK1[m], b.uGD0[m], b.uGD1[m],
			r.gumbelBKey, r.gumbelBIdx)
		e.DispatchTG(r.pGumbel2, gbThreads, gbThreads, gbShmBytes,
			r.gumbelBKey, r.gumbelBIdx, r.uGumbelNB, b.gOut.At(4*m))
	}
	e.End()
	r.recordExecErr(e.Err())
	if err := r.takeExecErr(); err != nil {
		return nil, nil, err
	}
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()
	b.steps, b.seqs, b.maxSeqs = b.steps+1, b.seqs+B, max(b.maxSeqs, B)
	logits, ids = make([][]float32, B), make([]int, B)
	lg := b.logitsB.Floats()
	gid := b.gOut.U32s()
	if argmaxOnly {
		for m := range seqs {
			ids[m] = argmaxF32(lg[m*r.V : (m+1)*r.V])
		}
		return nil, ids, nil
	}
	for m, q := range seqs {
		ids[m] = -1
		if q.draw != nil {
			id := int32(gid[m])
			if greedyDraw[m] || id < 0 { // ForwardSample's fallbacks: the argmax of the raw row
				id = int32(argmaxF32(lg[m*r.V : (m+1)*r.V]))
			}
			ids[m] = int(id)
			continue
		}
		row := make([]float32, r.V)
		copy(row, lg[m*r.V:(m+1)*r.V])
		if r.finalSoftcap > 0 {
			softcapParallel(row, r.finalSoftcap)
		}
		if r.logitScale != 0 && r.logitScale != 1 {
			for i := range row {
				row[i] *= r.logitScale
			}
		}
		logits[m] = row
	}
	return logits, ids, nil
}

// BatchStepRange is decoder.ResidentBatchStepper's: the batch sizes StepBatch serves well (batchMinSeqs..the smaller
// of batchMaxSeqs and the slot count), or (0, 0) when this resident cannot batch.
func (a *metalResident) BatchStepRange() (lo, hi int) {
	if a.r.batch == nil {
		return 0, 0
	}
	return batchMinSeqs, min(batchMaxSeqs, len(a.r.kvSlotBufs))
}

// StepBatch is decoder.ResidentBatchStepper's: one decode token for each sequence, each on its own KV slot.
func (a *metalResident) StepBatch(seqs []decoder.ResidentBatchSeq) ([]decoder.ResidentBatchOut, error) {
	bs := make([]batchSeq, len(seqs))
	for i, q := range seqs {
		bs[i] = batchSeq{slot: q.Slot, pos: q.Pos, emb: q.Emb, draw: q.Draw}
	}
	logits, ids, err := a.r.forwardMulti(bs)
	if err != nil {
		return nil, err
	}
	out := make([]decoder.ResidentBatchOut, len(seqs))
	for i := range out {
		out[i] = decoder.ResidentBatchOut{Logits: logits[i], ID: ids[i]}
	}
	return out, nil
}

// stepVerifyCost is the step-kernel verify's measured cost in single-token steps, indexed by rows verified: the
// conservative end of TestMC3Verify_rowCost on the M1 Pro (the 7B at depth 2048, the dearest cell of both models; the
// 1.5B at depth 128 reads 1.59 / 1.71 / 1.76 at 2 / 4 / 8 rows). 1 row is production's own Forward; 3 and 5-7 are
// interpolated; 9 rows are an 8-row step plus one Forward. docs/tasks/task-concurrency-2026-09.md, MC4.
var stepVerifyCost = []float64{0, 1, 1.79, 1.95, 2.11, 2.25, 2.38, 2.52, 2.65, 3.65}

var (
	_ decoder.VerifyCostReporter = (*metalResident)(nil)
)

// VerifyCost (decoder.VerifyCostReporter): the argmax-only verify's cost curve when this resident can run it on the
// step kernels, nil otherwise.
func (a *metalResident) VerifyCost() []float64 {
	if a.r == nil || a.r.batch == nil || len(a.r.kvSlotBufs) == 0 || a.r.finalSoftcap > 0 || (a.r.logitScale != 0 && a.r.logitScale != 1) {
		return nil
	}
	return stepVerifyCost
}

// PrefillLastNArgmax is the n-gram loop's argmax-only verify (decoder spec_ngram.go, idsVerify) on the step kernels:
// the rows are consecutive positions of the bound slot's sequence from startPos, run by forwardMulti, which is
// bit-identical to production decode for such rows (TestMC3Verify_sameSlotRowsBitIdentical), in pieces of at most
// batchMaxSeqs, the fragment's width. A one-row piece runs production's own Forward, so a round with no draft costs
// what plain decode does. Each id is the row's first maximum, as the decoder's own argmax.
func (a *metalResident) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	r := a.r
	if a.VerifyCost() == nil {
		return nil, fmt.Errorf("metal: no step-kernel verify on this resident")
	}
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	ids := make([]int, 0, len(embeddings))
	for from := 0; from < len(embeddings); {
		n := min(batchMaxSeqs, len(embeddings)-from)
		if n == 1 {
			lg, err := a.Forward(embeddings[from], startPos+from)
			if err != nil {
				return nil, err
			}
			ids = append(ids, argmaxF32(lg))
		} else {
			seqs := make([]batchSeq, n)
			for i := range n {
				seqs[i] = batchSeq{slot: r.kvSlot, pos: startPos + from + i, emb: embeddings[from+i]}
			}
			_, got, err := r.forwardMultiInto(seqs, true)
			if err != nil {
				return nil, err
			}
			ids = append(ids, got...)
		}
		from += n
	}
	return ids, nil
}

// promptStepMaxAboveFloor is the longest suffix the above-floor half of the step route takes (promptStepAboveFloor).
// T1.10 measured the step at 0.40× / 0.80× / 1.56× the pass at K = 16 / 32 / 64 on the 1.5B and the 7B
// (docs/tasks/task-metal-audit-2026-10.md, Batch A), so it stops at the last K measured faster.
const promptStepMaxAboveFloor = 32

// promptStepAboveFloor turns on E-P01's above-floor half: a suffix of up to promptStepMaxAboveFloor tokens on a prompt
// that reaches the floor also runs on the step. OFF, and test-only, until the owner decides: unlike the below-floor half
// it changes numerics. The pass computes a suffix's K/V from f16 activations and the step from decode's int8 ones, so
// a reuse turn would no longer equal a cold prefill of the same prompt (TestPromptStepAboveFloor_reuseVsCold measures
// by how much), and a chunked prefill's short tail would differ from the whole pass unless the decoder keeps tails
// longer than promptStepMaxAboveFloor.
var promptStepAboveFloor = false

// promptStepOK reports whether PrefillLast takes the step route for n tokens at startPos (E-P01, audit-metal-2026-09-30):
// the resident runs the step-kernel verify (VerifyCost: the batched step exists, and no logit transform separates its
// rows from Forward's), and the prompt ends below the fast-prefill floor, where the batched pass declines and the
// decoder would run the sequential loop. The step's rows are that loop's bits, so this route changes no output.
//
// Above the floor the step is faster than the pass up to K = 32 too, but there it would replace the pass's numerics with
// decode's: promptStepAboveFloor, off. A short suffix there is also a chunked prefill's tail, and on the step it differed
// from the whole pass the chunks must equal (TestMC5_prefillChunkInvariance's C = 81 case: 20480 of 20480 logits).
func (a *metalResident) promptStepOK(n, startPos, floor int) bool {
	if a.r.promptStepOff || n < 2 || a.VerifyCost() == nil {
		return false
	}
	if floor > 0 && startPos+n < floor {
		return true
	}
	return promptStepAboveFloor && n <= promptStepMaxAboveFloor
}

// prefillByStep is PrefillLast on the step kernels (E-P01): the prompt's positions run as consecutive rows of the bound
// slot's sequence, batchMaxSeqs at a time, exactly as PrefillLastNArgmax runs a verify, and the last row's logits come
// back. forwardMulti is bit-identical to production decode for such rows (TestMC3Verify_sameSlotRowsBitIdentical,
// TestMC3Step_promptInRowsBitIdentical), so this is the sequential path's numerics at about a quarter of its time
// (T1.10: 0.22× on the 1.5B, 0.25× on the 7B). Pieces before the last read only their argmax, which skips the
// per-row logits copy; a one-row last piece runs production's own Forward. Cancellation is checked between pieces.
func (a *metalResident) prefillByStep(ctx context.Context, embeddings [][]float32, startPos int) (logits []float32, err error) {
	r := a.r
	defer func() {
		if p := recover(); p != nil {
			logits, err = nil, fmt.Errorf("metal: step prefill aborted: %v", p)
		}
	}()
	for from := 0; from < len(embeddings); {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n := min(batchMaxSeqs, len(embeddings)-from)
		last := from+n == len(embeddings)
		if n == 1 {
			lg, err := a.Forward(embeddings[from], startPos+from)
			if err != nil {
				return nil, err
			}
			logits = lg
		} else {
			seqs := make([]batchSeq, n)
			for i := range n {
				seqs[i] = batchSeq{slot: r.kvSlot, pos: startPos + from + i, emb: embeddings[from+i]}
			}
			rows, _, err := r.forwardMultiInto(seqs, !last)
			if err != nil {
				return nil, err
			}
			if last {
				logits = rows[n-1]
			}
		}
		from += n
	}
	if err := r.takeExecErr(); err != nil {
		return nil, err
	}
	return logits, nil
}
