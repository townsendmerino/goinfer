//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"runtime"
	"sort"
	"testing"
	"time"
)

// R19 (docs/tasks/red-october.md): a test-only prototype of Metal's prefill attention in the shape of MLX 0.32.0's
// steel_attention (bq32 bk16 bd128 wm4 wn1), per the read in docs/measurements/metal-prefill-attn-2026-09-27.md.
// TestMetalPrefillDecomp compiles it and compares it with attention_prefill_fused when GOINFER_METAL_R19=1.
//
// Differences from attention_prefill_fused, each from the read:
//   - one threadgroup per (query head, 32 query rows); its 4 simdgroups own 8 rows each, and a 16-key K/V block is staged
//     ONCE in padded threadgroup memory and read by all 32 rows (the fused kernel re-read K/V per simdgroup);
//   - the O accumulator stays in registers, 16 8x8 fragments per simdgroup, rescaled in place (the fused kernel kept it
//     in threadgroup memory and rescaled it on 8 lanes after every head-dim tile, with two barriers each);
//   - the online softmax runs on every lane through the fragment lane map (mc3_frag: this lane holds row fm, columns
//     fn and fn+1 of every fragment; the 4 lanes of a row differ in lane bits 0 and 3, so two simd_shuffle_xor reduce
//     a row), in exp2 with log2e folded into the score scale (the fused kernel ran it serially on 8 lanes);
//   - P is written straight into the left-operand fragments of PV, and blocks past a row's causal (or window) limit
//     are skipped, the mask applied only where a block crosses it;
//   - two threadgroup barriers per 16 keys.
//
// Head dim 128 only (R19_HD); the harness runs it on the 1.5B. Not bit-identical to the fused kernel (a different
// accumulation order and exp2 for exp): R19's precondition 1 is set A's §3.2 pooled gate.
const r19Kernels = `
#include <metal_stdlib>
using namespace metal;

#define R19_BQ 32
#define R19_BK 16
#define R19_HD 128
#define R19_LD (R19_HD + 8)

kernel void attention_prefill_steel(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& M[[buffer(11)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]], ushort tid[[thread_index_in_threadgroup]]) {
    threadgroup half Ks[R19_BK * R19_LD];
    threadgroup half Vs[R19_BK * R19_LD];

    const uint nRB = (M + R19_BQ - 1u) / R19_BQ;
    const uint qh = tgid / nRB, rb = tgid % nRB;
    if (qh >= nH) return;
    const uint kvDim = nKV * R19_HD, kvh = qh / (nH / nKV), qDim = nH * R19_HD;
    const uint tr0 = rb * R19_BQ;                 // the threadgroup's first query row
    const uint r0 = tr0 + uint(sgid) * 8u;        // this simdgroup's first query row
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    const uint myRow = r0 + fm;                   // the query row this lane's fragment elements belong to
    const uint myKeys = startPos + min(myRow, M - 1u) + 1u;               // causal: keys [0, myKeys)
    const uint myWin = (window > 0u && myKeys > window) ? myKeys - window : 0u;

    // The threadgroup's key range: its first row's window start to its last row's causal end.
    const uint tLast = min(tr0 + R19_BQ - 1u, M - 1u);
    const uint jEnd = startPos + tLast + 1u;
    const uint firstKeys = startPos + tr0 + 1u;
    const uint jStart = ((window > 0u && firstKeys > window) ? firstKeys - window : 0u) / R19_BK * R19_BK;
    // This simdgroup's own range, to skip blocks it has nothing in.
    const uint sLast = min(r0 + 7u, M - 1u);
    const uint sEnd = startPos + sLast + 1u;
    const uint sFirstKeys = startPos + min(r0, M - 1u) + 1u;
    const uint sStart = (window > 0u && sFirstKeys > window) ? sFirstKeys - window : 0u;

    const float sl2 = scale * 1.4426950408889634f;
    // A simdgroup whose 8 rows are all past M (the last block's tail) loads no Q (the qkv buffer is padded to 8 rows,
    // not 32) and computes nothing; it still helps stage K/V and meets every barrier.
    const bool active = r0 < M;
    simdgroup_half8x8 qT[R19_HD / 8];
    if (active) {
        for (ushort kk = 0; kk < R19_HD / 8; kk++) {
            simdgroup_load(qT[kk], qkv + r0 * qStride + qh * R19_HD + kk * 8u, qStride);
        }
    }
    simdgroup_float8x8 oAcc[R19_HD / 8];
    for (ushort cc = 0; cc < R19_HD / 8; cc++) oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float mRow = -INFINITY, lRow = 0.0f;

    for (uint j0 = jStart; j0 < jEnd; j0 += R19_BK) {
        // Stage K and V for keys j0 .. j0+15: 2048 halves each, four half4 per thread; keys past jEnd read as zero.
        for (ushort i = 0; i < 4; i++) {
            uint idx = uint(tid) + uint(i) * 128u;
            uint key = idx >> 5, d4 = (idx & 31u) * 4u;
            uint j = j0 + key;
            half4 kv4 = half4(0.0h), vv4 = half4(0.0h);
            if (j < jEnd) {
                kv4 = *(device const half4*)(kc + j * kvDim + kvh * R19_HD + d4);
                vv4 = *(device const half4*)(vc + j * kvDim + kvh * R19_HD + d4);
            }
            *(threadgroup half4*)(Ks + key * R19_LD + d4) = kv4;
            *(threadgroup half4*)(Vs + key * R19_LD + d4) = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && j0 < sEnd && j0 + R19_BK > sStart) {
            // S = Q K^T for this simdgroup's 8 rows and the block's 16 keys (two 8x8 fragments).
            simdgroup_float8x8 S[2];
            for (ushort sub = 0; sub < 2; sub++) {
                S[sub] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (ushort kk = 0; kk < R19_HD / 8; kk++) {
                    simdgroup_half8x8 kTile;
                    simdgroup_load(kTile, Ks + (sub * 8u) * R19_LD + kk * 8u, R19_LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S[sub], qT[kk], kTile, S[sub]);
                }
            }
            // This lane's four scores: row myRow, keys j0 + {fn, fn+1, 8+fn, 9+fn}.
            float s[4];
            s[0] = S[0].thread_elements()[0] * sl2; s[1] = S[0].thread_elements()[1] * sl2;
            s[2] = S[1].thread_elements()[0] * sl2; s[3] = S[1].thread_elements()[1] * sl2;
            const bool edge = (j0 + R19_BK > startPos + r0 + 1u) || (j0 < myWin + R19_BK) || (myRow >= M);
            if (edge) {
                uint js[4] = {j0 + fn, j0 + fn + 1u, j0 + 8u + fn, j0 + 9u + fn};
                for (ushort e = 0; e < 4; e++) {
                    if (js[e] >= myKeys || js[e] < myWin || myRow >= M) s[e] = -INFINITY;
                }
            }
            float bmax = max(max(s[0], s[1]), max(s[2], s[3]));
            bmax = max(bmax, simd_shuffle_xor(bmax, 1));
            bmax = max(bmax, simd_shuffle_xor(bmax, 8));
            float mNew = max(mRow, bmax);
            float factor = 1.0f;
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (mNew > -INFINITY) {
                factor = (mRow > -INFINITY) ? exp2(mRow - mNew) : 0.0f;
                for (ushort e = 0; e < 4; e++) p[e] = (s[e] > -INFINITY) ? exp2(s[e] - mNew) : 0.0f;
                mRow = mNew;
            }
            float bsum = (p[0] + p[1]) + (p[2] + p[3]);
            bsum += simd_shuffle_xor(bsum, 1);
            bsum += simd_shuffle_xor(bsum, 8);
            lRow = lRow * factor + bsum;
            simdgroup_half8x8 P[2];
            P[0].thread_elements()[0] = half(p[0]); P[0].thread_elements()[1] = half(p[1]);
            P[1].thread_elements()[0] = half(p[2]); P[1].thread_elements()[1] = half(p[3]);
            for (ushort cc = 0; cc < R19_HD / 8; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                for (ushort sub = 0; sub < 2; sub++) {
                    simdgroup_half8x8 vTile;
                    simdgroup_load(vTile, Vs + (sub * 8u) * R19_LD + cc * 8u, R19_LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P[sub], vTile, oAcc[cc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (myRow < M) {
        const float inv = lRow > 0.0f ? 1.0f / lRow : 0.0f;
        device half* o = out + myRow * qDim + qh * R19_HD + fn;
        for (ushort cc = 0; cc < R19_HD / 8; cc++) {
            o[cc * 8u] = half(oAcc[cc].thread_elements()[0] * inv);
            o[cc * 8u + 1u] = half(oAcc[cc].thread_elements()[1] * inv);
        }
    }
}
`

// runR19Phase is R19's comparison inside TestMetalPrefillDecomp, at one prompt length M, per the pre-registration in
// docs/tasks/red-october.md R19:
//  1. the prototype's attention output against attention_prefill_fused's on the last layer's real inputs (the buffers
//     a full replay leaves): relative L2 error, max |diff|, cosine;
//  2. a full replay with the prototype as every layer's attention: logits against PrefillLast's (cosine, argmax);
//  3. the registered metric: per arm, the attention category in sequence (full − full-without-attention), both arms
//     interleaved rep by rep, current ÷ prototype.
func runR19Phase(t *testing.T, r *resident, M, reps int, hb func(string, ...any), cats []decompCat,
	full func(e *Encoder), resetX func(), gpuMs func(func(e *Encoder)) float64, ref []float32, ctxF Buffer,
	protoOn *bool, protoPipe *Pipeline) {
	if r.layers[0].geom.hd != 128 {
		hb("K=%d R19 skipped: the prototype is head dim 128 only (this model: %d)", M, r.layers[0].geom.hd)
		return
	}
	func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary(r19Kernels, MSL3_1)
		if err != nil {
			t.Fatalf("compile R19 prototype: %v", err)
		}
		if *protoPipe, err = r.d.NewComputePipeline(lib, "attention_prefill_steel"); err != nil {
			t.Fatalf("R19 pipeline: %v", err)
		}
	}()
	attn := -1
	for ci, c := range cats {
		if c.name == "attention" {
			attn = ci
		}
	}
	last := r.nL - 1
	// (1) one layer, same inputs, both kernels
	*protoOn = false
	resetX()
	gpuMs(full)
	runOne := func(proto bool) []float32 {
		*protoOn = proto
		gpuMs(func(e *Encoder) { cats[attn].enc(e, last) })
		*protoOn = false
		u := ctxF.U16s()
		out := make([]float32, M*r.nH*r.layers[0].geom.hd)
		for i := range out {
			out[i] = f16ToF32(u[i])
		}
		return out
	}
	cur, pro := runOne(false), runOne(true)
	var num, den, dot, nb, maxAbs float64
	for i := range cur {
		d := float64(cur[i]) - float64(pro[i])
		num += d * d
		den += float64(cur[i]) * float64(cur[i])
		nb += float64(pro[i]) * float64(pro[i])
		dot += float64(cur[i]) * float64(pro[i])
		maxAbs = math.Max(maxAbs, math.Abs(d))
	}
	hb("K=%d R19 layer %d attention, prototype vs fused on the same inputs: relative L2 %.3g, max |diff| %.3g, cosine %.9f",
		M, last, math.Sqrt(num/den), maxAbs, dot/math.Sqrt(den*nb))
	// (2) full replay with the prototype everywhere
	*protoOn = true
	resetX()
	gpuMs(full)
	*protoOn = false
	pl := r.logits.Floats()[:r.V]
	dot, na, nr := 0.0, 0.0, 0.0
	am, bm := 0, 0
	for i := range pl {
		dot += float64(pl[i]) * float64(ref[i])
		na += float64(pl[i]) * float64(pl[i])
		nr += float64(ref[i]) * float64(ref[i])
		if pl[i] > pl[am] {
			am = i
		}
		if ref[i] > ref[bm] {
			bm = i
		}
	}
	hb("K=%d R19 full-replay logits with the prototype: cosine %.9f against PrefillLast, argmax %d vs %d", M, dot/math.Sqrt(na*nr), am, bm)
	// (3) the registered metric
	without := func(e *Encoder) {
		for l := 0; l < r.nL; l++ {
			for ci, c := range cats {
				if !c.once && ci != attn {
					c.enc(e, l)
				}
			}
		}
		for _, c := range cats {
			if c.once {
				c.enc(e, 0)
			}
		}
	}
	var ratios, curMs, proMs []float64
	for rep := 0; rep < reps; rep++ {
		order := []bool{false, true}
		if rep%2 == 1 {
			order = []bool{true, false}
		}
		cost := map[bool]float64{}
		for _, arm := range order {
			*protoOn = arm
			resetX()
			f := gpuMs(full)
			resetX()
			w := gpuMs(without)
			*protoOn = false
			cost[arm] = f - w
		}
		curMs, proMs = append(curMs, cost[false]), append(proMs, cost[true])
		ratios = append(ratios, cost[false]/cost[true])
		hb("K=%d R19 rep %d/%d: attention in sequence — fused %.1f ms, prototype %.1f ms → %.2fx", M, rep+1, reps, cost[false], cost[true], ratios[rep])
	}
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	hb("K=%d R19 attention in sequence (median of %d paired reps): fused %.1f ms, prototype %.1f ms → %.3fx (per rep %v)",
		M, reps, med(curMs), med(proMs), med(ratios), ratios)
	// (4) Burst vs sustained (R19 precondition 2 grades the sustained number): the attention category alone over every
	// layer, back to back, then after 2 s of idle, both kernels.
	only := func(e *Encoder) {
		for l := 0; l < r.nL; l++ {
			cats[attn].enc(e, l)
		}
	}
	sus, idle := map[bool][]float64{}, map[bool][]float64{}
	for rep := 0; rep < reps; rep++ {
		for _, arm := range []bool{false, true} {
			*protoOn = arm
			gpuMs(only)
			sus[arm] = append(sus[arm], gpuMs(only))
			time.Sleep(2 * time.Second)
			idle[arm] = append(idle[arm], gpuMs(only))
			*protoOn = false
		}
	}
	hb("K=%d R19 attention alone: fused sustained %.1f ms, after 2 s idle %.1f (%.2f); prototype sustained %.1f, after idle %.1f (%.2f)",
		M, med(sus[false]), med(idle[false]), med(idle[false])/med(sus[false]), med(sus[true]), med(idle[true]), med(idle[true])/med(sus[true]))
}
