//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// mc3StepKernels are MC3 S1's batched matmuls (docs/tasks/task-concurrency-2026-09.md; S0:
// docs/measurements/concurrency-mc3-s0-2026-09-26.md), TEST-ONLY: S0's bit-identical 8-row kernels with the production
// epilogues, plus the activation packers. mc3_bt reproduces gemv_w4a8_sa_rows / _bias_rows / _resid_rows (mode 0 / 1 /
// 2), mc3_btd gemv_w4a8_resid_staged, mc3_lm gemv_w8a8_coal — each on 8 activation rows at once, each output equal to
// the production kernel's on that row. Their bit-identity rests on simd_sum being the xor tree 1, 2, 4, 8, 16
// (TestMC3SimdSumTree), except mc3_lm, which is integer-exact in any order.
const mc3StepKernels = `
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
`

// mc3Step is S1's test-only batched decode step on a resident: pipelines, per-row scratch for up to 8 sequences, and
// per-sequence uniforms.
type mc3Step struct {
	r                         *resident
	pack, packLM, bt, btd, lm Pipeline
	xB, qkvB, guB, logitsB    Buffer // f32 rows
	aqB, cqB, mqB, dqB        Buffer // int8 rows
	aScB, cScB, mScB, dScB    Buffer // one f32 per row
	aT, aTp                   Buffer // half [K][8]
	uM, uQKV, uGU, uV, uMode  [3]Buffer
	uPos, uNKeys, uQTemp      []Buffer
	noBias                    Buffer
	qkvRows, nHhd, kOff, vOff int
	gpuMS                     float64 // the last step's command-buffer GPU time
}

type mc3Seq struct {
	slot, pos int
	emb       []float32
}

func newMC3Step(t *testing.T, r *resident) *mc3Step {
	t.Helper()
	g0 := r.layers[0].geom
	s := &mc3Step{r: r, nHhd: r.nH * g0.hd}
	s.qkvRows = s.nHhd + 2*g0.kvDim
	s.kOff, s.vOff = s.nHhd*4, (s.nHhd+g0.kvDim)*4
	for _, L := range r.layers {
		if L.geom == nil || L.geom.hd != g0.hd || L.geom.kvDim != g0.kvDim {
			t.Skip("MC3 S1 assumes uniform attention geometry")
		}
	}
	for _, chk := range []struct {
		name  string
		n, by int
	}{{"qkv rows", s.qkvRows, 16}, {"hidden", r.H, 16}, {"2*inter", 2 * r.I, 16}, {"vocab", r.V, 128}} {
		if chk.n%chk.by != 0 {
			t.Skipf("MC3 S1 kernels need %s %% %d == 0 (got %d)", chk.name, chk.by, chk.n)
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	d := r.d
	lib, err := d.CompileLibrary(allKernels+"\n"+mc3StepKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p := func(n string) Pipeline {
		pp, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("pipeline %s: %v", n, err)
		}
		return pp
	}
	s.pack, s.packLM, s.bt, s.btd, s.lm = p("mc3_pack"), p("mc3_pack_lm"), p("mc3_bt_fb2"), p("mc3_btd_fb2"), p("mc3_lm_fb4")
	const B = 8
	maxK := max(r.H, s.nHhd, r.I)
	s.xB, s.qkvB, s.guB, s.logitsB = d.NewBufferLen(B*r.H), d.NewBufferLen(B*s.qkvRows), d.NewBufferLen(B*2*r.I), d.NewBufferLen(B*r.V)
	s.aqB, s.cqB, s.mqB, s.dqB = d.NewBufferBytes(B*r.H), d.NewBufferBytes(B*s.nHhd), d.NewBufferBytes(B*r.H), d.NewBufferBytes(B*r.I)
	s.aScB, s.cScB, s.mScB, s.dScB = d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B)
	s.aT, s.aTp = NewBufferU16s(d, make([]uint16, maxK*8)), NewBufferU16s(d, make([]uint16, r.H*8))
	s.uM[0] = NewBufferU32(d, 1)
	s.uQKV[0], s.uGU[0], s.uV[0] = NewBufferU32(d, uint32(s.qkvRows)), NewBufferU32(d, uint32(2*r.I)), NewBufferU32(d, uint32(r.V))
	s.uMode[0], s.uMode[1], s.uMode[2] = NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferU32(d, 2)
	s.noBias = d.NewBufferLen(max(r.H, 2*r.I))
	for range B {
		s.uPos = append(s.uPos, NewBufferU32(d, 0))
		s.uNKeys = append(s.uNKeys, NewBufferU32(d, 1))
		s.uQTemp = append(s.uQTemp, NewBufferFloats(d, []float32{1}))
	}
	return s
}

// step runs one decode token for each sequence (each on its own resident KV slot, at its own position) in ONE command
// buffer, and returns each sequence's logits. Per sequence it dispatches production's own kernels for everything but
// the five matmuls (norm+quant, RoPE, KV store, attention, ctx quant, SwiGLU+quant, final norm), so each row's data
// is exactly production's; the matmuls run once for all rows.
func (s *mc3Step) step(t *testing.T, seqs []mc3Seq) [][]float32 {
	r := s.r
	B := len(seqs)
	if B < 1 || B > 8 {
		t.Fatalf("mc3 step: %d sequences (1..8)", B)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.stopExec()
	r.curNKeys = 0 // no attention_fa: S1 runs below attnFADepthFloor, where production's plan is the shipped kernel
	s.uM[0].SetU32(uint32(B))
	for m, q := range seqs {
		if q.pos+1 >= attnFADepthFloor {
			t.Fatalf("mc3 step: position %d is past attnFADepthFloor (S1 does not reproduce attention_fa's plan)", q.pos)
		}
		copy(s.xB.Floats()[m*r.H:(m+1)*r.H], q.emb)
		s.uPos[m].SetU32(uint32(q.pos))
		s.uNKeys[m].SetU32(uint32(q.pos + 1))
	}
	H, I, nHhd := r.H, r.I, s.nHhd
	// Buffer.At SETS the byte offset from the allocation's start; it does not add to one a view already carries, so a
	// sub-row offset is computed whole (row*n + off), never as row.At(off).
	f32 := func(b Buffer, m, n int) Buffer { return b.At(4 * m * n) }
	i8 := func(b Buffer, m, n int) Buffer { return b.At(m * n) }
	pack := func(e *Encoder, aq Buffer, K int, uK Buffer) {
		e.Dispatch(s.pack, K*8, 256, aq, s.aT, s.uM[0], uK)
	}
	const tgb = 4 * 2 * 32 * 8 // mc3_bt/mc3_btd FB=2: the Q exchange
	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		g := L.geom
		for m := range B {
			r.encodeNorm(e, f32(s.xB, m, H), L.preNorm, L.preNormBias, i8(s.aqB, m, H), f32(s.aScB, m, 1))
		}
		pack(e, s.aqB, H, r.uH)
		e.DispatchTG(s.bt, s.qkvRows/16*128, 128, tgb, L.qkvW, L.qkvS, s.aT, s.aScB, s.qkvB, r.uH, s.uQKV[0], s.uM[0], L.qkvBias, s.uMode[1])
		for m, q := range seqs {
			sb := r.kvSlotBufs[q.slot]
			qkv := f32(s.qkvB, m, s.qkvRows)
			e.Dispatch(r.pRope2, r.nH*g.half+g.nKV*g.half, 64, qkv, L.invf, g.uHd, s.uPos[m], g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, s.uQTemp[m])
			row := 4 * m * s.qkvRows
			e.Dispatch(r.pKv, g.kvDim, 64, s.qkvB.At(row+s.kOff), s.qkvB.At(row+s.vOff), sb.kc[l], sb.vc[l], g.uKvDim, s.uPos[m])
			e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, qkv, sb.kc[l], sb.vc[l], r.ctx, r.uNH, g.uNKV, g.uHd, s.uNKeys[m], r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
			e.Dispatch(r.pQv, 256, 256, r.ctx, i8(s.cqB, m, nHhd), f32(s.cScB, m, 1), g.uNHhd)
		}
		pack(e, s.cqB, nHhd, g.uNHhd)
		e.DispatchTG(s.bt, H/16*128, 128, tgb, L.oW, L.oS, s.aT, s.cScB, s.xB, g.uNHhd, r.uH, s.uM[0], s.noBias, s.uMode[2])
		for m := range B {
			r.encodeNorm(e, f32(s.xB, m, H), L.postNorm, L.postNormBias, i8(s.mqB, m, H), f32(s.mScB, m, 1))
		}
		pack(e, s.mqB, H, r.uH)
		e.DispatchTG(s.bt, 2*I/16*128, 128, tgb, L.guW, L.guS, s.aT, s.mScB, s.guB, r.uH, s.uGU[0], s.uM[0], s.noBias, s.uMode[0])
		for m := range B {
			e.Dispatch(r.pSw, 256, 256, f32(s.guB, m, 2*I), s.guB.At(4*(m*2*I+I)), i8(s.dqB, m, I), f32(s.dScB, m, 1), r.uI, r.uAct)
		}
		pack(e, s.dqB, I, r.uI)
		e.DispatchTG(s.btd, H/16*128, 128, tgb, L.dW, L.dS, s.aT, s.dScB, s.xB, r.uI, r.uH, s.uM[0])
	}
	for m := range B {
		r.encodeNorm(e, f32(s.xB, m, H), r.finalNorm, r.finalNormBias, i8(s.aqB, m, H), f32(s.aScB, m, 1))
	}
	e.Dispatch(s.packLM, H*8, 256, s.aqB, s.aTp, s.uM[0], r.uH)
	e.DispatchTG(s.lm, r.V/128*128, 128, 0, r.lmW, r.lmS, s.aTp, s.aScB, s.logitsB, r.uH, s.uV[0], s.uM[0])
	e.End()
	if err := e.Err(); err != nil {
		t.Fatalf("mc3 step: %v", err)
	}
	s.gpuMS = (e.GPUEnd() - e.GPUStart()) * 1e3
	out := make([][]float32, B)
	lg := s.logitsB.Floats()
	for m := range B {
		out[m] = append([]float32(nil), lg[m*r.V:(m+1)*r.V]...)
	}
	return out
}

// mc3Resident loads the S1 checkpoint with `slots` resident KV slots at a 1024-token context, for the plain dense
// W4A8 decode path S1 covers.
func mc3Resident(t *testing.T, slots int) (*decoder.Model, *resident) {
	t.Helper()
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_MC3_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 1024, ResidentKVSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	if r.moe != nil || r.g4moe != nil || r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm ||
		r.decodeLaneW4F16 || r.nonGatedMLP || r.outBias || r.loraLayers != nil || r.qkNorm || r.learnedPos {
		r.Close()
		t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
	}
	for _, L := range r.layers {
		if L.qGate || L.delta != nil || L.geom == nil || L.geom.kEqV {
			r.Close()
			t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
		}
	}
	if got := len(r.kvSlotBufs); got < slots {
		r.Close()
		t.Fatalf("resident allocated %d KV slots, %d requested (the fit clamp)", got, slots)
	}
	return m, r
}

// mc3Emb is token id's (scaled) input embedding.
func mc3Emb(r *resident, id int) []float32 {
	e := make([]float32, r.H)
	r.embed.Row(id, e)
	if r.embedScale > 1 {
		for i := range e {
			e[i] *= r.embedScale
		}
	}
	return e
}

// mc3Fill writes history ids at positions 0.. into KV slot `slot` through production's layer-major batch path, in
// chunks of 64.
func mc3Fill(t *testing.T, r *resident, slot int, ids []int) {
	t.Helper()
	if err := r.useKVSlot(slot); err != nil {
		t.Fatal(err)
	}
	for c := 0; c < len(ids); c += 64 {
		var embs [][]float32
		for _, id := range ids[c:min(c+64, len(ids))] {
			embs = append(embs, mc3Emb(r, id))
		}
		if _, err := r.ForwardBatch(embs, c); err != nil {
			t.Fatalf("fill slot %d: %v", slot, err)
		}
	}
}

// TestMC3Step_bitIdentical is S1's identity check: 4 sequences at different depths, each on its own KV slot, stepped
// together 12 times teacher-forced, against production's single-token forward (ForwardEmb) run per sequence on a TWIN
// slot filled with the same history. Every logit must match bit for bit, every step.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3Step_bitIdentical$' -v ./metal/
func TestMC3Step_bitIdentical(t *testing.T) {
	const B, steps = 4, 12
	_, r := mc3Resident(t, 2*B)
	defer r.Close()
	s := newMC3Step(t, r)
	depths := []int{5, 23, 40, 300}
	seed := uint32(1234567)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
		mc3Fill(t, r, B+m, ids)
	}
	totalDiff := 0
	for st := 0; st < steps; st++ {
		seqs := make([]mc3Seq, B)
		ref := make([][]float32, B)
		for m := range B {
			seqs[m] = mc3Seq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			if err := r.useKVSlot(B + m); err != nil {
				t.Fatal(err)
			}
			ref[m] = append([]float32(nil), r.ForwardEmb(seqs[m].emb, seqs[m].pos)...)
		}
		got := s.step(t, seqs)
		for m := range B {
			diff, worst := 0, 0.0
			for i := range got[m] {
				if math.Float32bits(got[m][i]) != math.Float32bits(ref[m][i]) {
					diff++
					worst = math.Max(worst, math.Abs(float64(got[m][i]-ref[m][i])))
				}
			}
			if diff != 0 {
				t.Errorf("step %d sequence %d (pos %d): %d of %d logits differ from production (max |diff| %.3g)", st, m, seqs[m].pos, diff, r.V, worst)
			}
			totalDiff += diff
		}
	}
	t.Logf("%d sequences x %d steps at depths %v: %d logits differ from production's single-token forward", B, steps, depths, totalDiff)
}

// TestMC3Step_throughput is S1's in-sequence cost (exploratory): one batched step of B = 1, 2, 4, 8 sequences, each on
// its own slot at depth D, against production's single-token decode at the same depth, GPU time per command buffer,
// arms interleaved per rep with rotating order. Reports aggregate = B x production / step(B).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_DEPTHS=128,512] go test -count=1 -run '^TestMC3Step_throughput$' -v ./metal/
func TestMC3Step_throughput(t *testing.T) {
	const maxB = 8
	_, r := mc3Resident(t, maxB+1)
	defer r.Close()
	s := newMC3Step(t, r)
	depths := []int{128, 512}
	if v := os.Getenv("GOINFER_METAL_MC3_DEPTHS"); v != "" {
		depths = nil
		for _, f := range strings.Split(v, ",") {
			var n int
			fmt.Sscan(strings.TrimSpace(f), &n)
			depths = append(depths, n)
		}
	}
	const reps, tokens = 7, 12
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc3-s1 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(7654321)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	for _, D := range depths {
		for sl := 0; sl <= maxB; sl++ {
			ids := make([]int, D)
			for i := range ids {
				ids[i] = rnd()
			}
			mc3Fill(t, r, sl, ids)
		}
		hb("depth %d: %d slots filled", D, maxB+1)
		arms := []int{0, 1, 2, 4, 8} // 0 = production single-token decode on slot maxB
		per := map[int][]float64{}   // arm -> per-rep median GPU ms per step
		for rep := 0; rep < reps; rep++ {
			for k := range arms {
				a := arms[(k+rep)%len(arms)]
				var ms []float64
				for tok := 0; tok < tokens; tok++ {
					emb := mc3Emb(r, rnd())
					if a == 0 {
						if err := r.useKVSlot(maxB); err != nil {
							t.Fatal(err)
						}
						r.ForwardEmb(emb, D) // rewrites position D each token: the depth stays D
						ms = append(ms, (r.gpuEnd-r.gpuStart)*1e3)
						continue
					}
					seqs := make([]mc3Seq, a)
					for m := range a {
						seqs[m] = mc3Seq{slot: m, pos: D, emb: emb}
					}
					s.step(t, seqs)
					ms = append(ms, s.gpuMS)
				}
				per[a] = append(per[a], med(ms))
			}
		}
		line := fmt.Sprintf("depth %d: production %.3f ms/token", D, med(per[0]))
		for _, a := range arms[1:] {
			var ratio []float64
			for i := range per[a] {
				ratio = append(ratio, float64(a)*per[0][i]/per[a][i])
			}
			sort.Float64s(ratio)
			line += fmt.Sprintf("  | B=%d step %.3f ms (%.2fx a token), aggregate %.3fx [%.3f-%.3f]", a, med(per[a]), med(per[a])/med(per[0]), med(ratio), ratio[0], ratio[len(ratio)-1])
		}
		hb("%s", line)
	}
}
