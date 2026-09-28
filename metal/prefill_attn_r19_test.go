//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"sort"
	"testing"
	"time"
)

// R19 (docs/tasks/red-october.md): Metal prefill attention at depth. The prototype selected by the exploratory runs and
// graded by the confirmation run is production's attention_prefill_steel since its wiring (metal/prefill.go); what
// stays here is the harness phase that compares production against the retired attention_prefill_fused.

// runR19Phase is R19's comparison inside TestMetalPrefillDecomp, at one prompt length M, per the pre-registration in
// docs/tasks/red-october.md R19. The "prototype" arm is production's attention (attention_prefill_steel); the other
// arm, switched on by protoOn, is the retired attention_prefill_fused:
//  1. the two kernels' attention output on the last layer's real inputs (the buffers a full replay leaves): relative
//     L2 error, max |diff|, cosine;
//  2. a full replay on the fused kernel: logits against PrefillLast's, which now runs the steel kernel;
//  3. the registered metric: per arm, the attention category in sequence (full − full-without-attention), both arms
//     interleaved rep by rep, fused ÷ steel;
//  4. burst vs sustained, the attention category alone.
func runR19Phase(t *testing.T, r *resident, M, reps int, hb func(string, ...any), cats []decompCat,
	full func(e *Encoder), resetX func(), gpuMs func(func(e *Encoder)) float64, ref []float32, ctxF Buffer,
	protoOn *bool, steelInProduction bool) {
	if !steelInProduction {
		hb("K=%d R19 skipped: production does not run attention_prefill_steel on this model (head dim %d)", M, r.layers[0].geom.hd)
		return
	}
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
	pro, cur := runOne(false), runOne(true) // false = production (steel), true = the retired fused kernel
	var num, den, dot, nb, maxAbs float64
	for i := range cur {
		d := float64(cur[i]) - float64(pro[i])
		num += d * d
		den += float64(cur[i]) * float64(cur[i])
		nb += float64(pro[i]) * float64(pro[i])
		dot += float64(cur[i]) * float64(pro[i])
		maxAbs = math.Max(maxAbs, math.Abs(d))
	}
	hb("K=%d R19 layer %d attention, steel (production) vs fused (retired) on the same inputs: relative L2 %.3g, max |diff| %.3g, cosine %.9f",
		M, last, math.Sqrt(num/den), maxAbs, dot/math.Sqrt(den*nb))
	// (2) full replay on the retired fused kernel everywhere, against PrefillLast (which runs the steel kernel)
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
	hb("K=%d R19 full-replay logits on the fused kernel: cosine %.9f against PrefillLast (steel), argmax %d vs %d", M, dot/math.Sqrt(na*nr), am, bm)
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
		curMs, proMs = append(curMs, cost[true]), append(proMs, cost[false]) // cur = fused (retired), pro = steel
		ratios = append(ratios, cost[true]/cost[false])
		hb("K=%d R19 rep %d/%d: attention in sequence — fused %.1f ms, steel %.1f ms → %.2fx", M, rep+1, reps, cost[true], cost[false], ratios[rep])
	}
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	hb("K=%d R19 attention in sequence (median of %d paired reps): fused %.1f ms, steel %.1f ms → %.3fx (per rep %v)",
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
	hb("K=%d R19 attention alone: fused sustained %.1f ms, after 2 s idle %.1f (%.2f); steel sustained %.1f, after idle %.1f (%.2f)",
		M, med(sus[true]), med(idle[true]), med(idle[true])/med(sus[true]), med(sus[false]), med(idle[false]), med(idle[false])/med(sus[false]))
}
