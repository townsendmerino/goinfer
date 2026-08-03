//go:build cuda

package cuda

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4MoE_residentParity is Split-B task 2c's END-TO-END gate. Every primitive underneath is
// pinned in isolation and the pos-0 per-branch parity (TestGemma4MoE_localize) proves the wiring, so
// this asserts kernel correctness at the logit level (pos 0) and then CHARACTERIZES the multi-position
// int4-vs-int4 drift with two calibrated instruments instead of a picked-below-observed floor:
//
//  1. ROUTING AGREEMENT at EVERY position, not just pos 0. rn derives from h derives from the KV
//     cache, so as attention drifts a top-k flip at position N becomes possible even though pos 0 is
//     clean — and a flipped expert reads IDENTICAL to accumulation in a cosine (same shape, same
//     "grows with position"). If resident idx == CPU idx at every position, accumulation is the only
//     explanation left; a flip means the 0.87 has a discrete component.
//  2. A CALIBRATED curve: CUDA-int4-vs-CPU-int4 vs CPU-int4-vs-CPU-f32 at the same positions. The
//     latter is "as well as int4 arithmetic can agree with f32". CUDA-vs-CPU-int4 (same weights, only
//     W4A8 activation rounding differs) should track it or sit ABOVE it. If CUDA drops FASTER than the
//     fixture's own quantization curve, that's a real divergence no conditioning explains.
func TestGemma4MoE_residentParity(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_G4_CAPTURE", "1")
	const dir = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_moe_forward.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED gemma4 MoE with env on — admission regressed")
	}
	r := rf.(*cudaResident)
	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(dir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	// 16 positions: pos 7 was the only inversion AND the endpoint (max accumulation), so run out to
	// 2× to see whether the CUDA-vs-CPUint4 / CPUint4-vs-f32 gap STABILIZES or keeps widening — a
	// widening gap at the tail would flake the tolerance for reasons unrelated to a bug.
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200}

	// CPU int4 over the prompt, capturing its per-decision idx (the routing reference).
	decoder.SetRouterCaptureForTest(true)
	defer decoder.SetRouterCaptureForTest(false)
	cpu4 := make([][]float32, len(prompt))
	c4cache := mc4.NewCache(len(prompt))
	for i, tok := range prompt {
		l, err := mc4.ForwardForTest(tok, c4cache)
		if err != nil {
			t.Fatalf("cpu int4 pos %d: %v", i, err)
		}
		cpu4[i] = append([]float32(nil), l...)
	}
	idxCpu4, _ := decoder.RouterCaptureForTest()
	idxCpu4 = append([][]int(nil), idxCpu4...)                             // snapshot before the f32 run appends more
	marginCpu4 := append([]float32(nil), decoder.RouterMarginForTest()...) // per-decision top-k boundary margin, same order

	// CPU f32 over the prompt (the conditioning reference; idx not needed).
	cpuF := make([][]float32, len(prompt))
	cFcache := mcF.NewCache(len(prompt))
	for i, tok := range prompt {
		l, err := mcF.ForwardForTest(tok, cFcache)
		if err != nil {
			t.Fatalf("cpu f32 pos %d: %v", i, err)
		}
		cpuF[i] = append([]float32(nil), l...)
	}

	// Resident int4 over the prompt (g4capIdx accumulates per decision).
	cuda := make([][]float32, len(prompt))
	for i, tok := range prompt {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		cuda[i] = append([]float32(nil), l...)
	}

	// ---- curves: CUDA-vs-CPU-int4 vs the fixture's own CPU-int4-vs-f32 quantization curve ----
	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }
	cVs4, c4VsF := make([]float64, len(prompt)), make([]float64, len(prompt))
	pos0 := 0.0
	for i := range prompt {
		cVs4[i] = cos(cpu4[i], cuda[i])  // CUDA-int4 vs CPU-int4 (only W4A8 activation rounding + reduction differ)
		c4VsF[i] = cos(cpuF[i], cpu4[i]) // CPU-int4 vs CPU-f32 (the "as well as int4 arithmetic can agree" bound)
		if i == 0 {
			pos0 = cVs4[i]
		}
		t.Logf("  pos %2d  CUDA-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cuda=%d cpu4=%d",
			i, cVs4[i], c4VsF[i], argmaxF(cuda[i]), argmaxF(cpu4[i]))
	}

	// ---- MARGIN-GATED routing agreement (the reusable cross-backend MoE instrument) ----
	//
	// Unconditional resident-idx == CPU-idx is an INVALID gate for MoE, and the 26B established why:
	// a top-k router is a DISCRETE function of a continuously-drifting input, so where two selected
	// experts are near-tied in router probability, the tiny W4A8-activation delta between resident and
	// CPU legitimately FLIPS the selection — a different-but-correct expert, not a bug (the resident
	// and CPU routers are each bit-exact given their own input; only the input differs by rounding).
	// Past such a flip the two backends compute different experts, so a hidden-state cosine CLIFFS and
	// per-position argmax vs CPU goes to noise; neither is a defect. But at WIDE margin a flip is NOT
	// explainable by rounding — it means the dispatch fed the wrong activation, or the router itself
	// diverged — a real bug. So gate on the margin: assert index agreement only where the top-k
	// boundary margin (smallest-selected minus largest-rejected softmax prob) exceeds a threshold;
	// below it, record the disagreement as expected sensitivity rather than failing.
	//
	// THRESHOLD (marginGate = 0.01), chosen from the MEASURED margin distribution, which is bimodal by
	// ~2 orders of magnitude (metal/gemma4_moe_noisefloor_test.go + metal/gemma4_26b_routing_test.go):
	//   - THIS fixture, gemma4-moe-tiny (nE=4, top-2): min margin 0.2679 — every decision well-separated
	//   - real width, 26B (nE=128, top-8): flips sit at 0.00115, matched at 0.00218 — the near-tie band
	//   - the degenerate control, gemma4-moe-kv-tiny: 0.0001 — routing is a coin-flip, non-gating
	// 0.01 sits 5x above the near-tie band (0.002) and 27x below this fixture's min (0.268) — an order
	// of magnitude clear of both regimes, so it is robust to per-arch softmax-scale drift. On moe-tiny
	// every margin is >> 0.01, so this gate stays FULLY STRICT here (a real nE=4 dispatch bug still
	// fails); the sensitivity exemption only ever fires at real width, where unconditional agreement is
	// the wrong bar. To reuse on another MoE family, confirm its well-separated band still clears 0.01
	// (wider nE compresses margins) and re-pick from that family's distribution if it does not.
	const marginGate = 0.01
	nMoE := len(idxCpu4) / len(prompt)
	if len(r.g4capIdx) != len(idxCpu4) {
		t.Fatalf("decision-count mismatch: resident %d vs cpu %d", len(r.g4capIdx), len(idxCpu4))
	}
	routeAgree, sensitivityFlips := true, 0
	for k := range idxCpu4 {
		if sameExpertSet(idxCpu4[k], r.g4capIdx[k]) {
			continue
		}
		mg := float32(math.Inf(1)) // no margin captured ⇒ treat as wide ⇒ assert (fail-closed)
		if k < len(marginCpu4) {
			mg = marginCpu4[k]
		}
		if float64(mg) > marginGate {
			routeAgree = false
			t.Errorf("ROUTING FLIP at decision %d (pos %d, moe-layer %d), margin %.4f > %.4f — a WELL-SEPARATED "+
				"expert disagrees, NOT int4 sensitivity: cpu idx %v vs resident %v (wrong activation dispatched, or "+
				"the router diverged)", k, k/nMoE, k%nMoE, mg, marginGate, idxCpu4[k], r.g4capIdx[k])
		} else {
			sensitivityFlips++
			t.Logf("near-tie flip at decision %d (pos %d, moe-layer %d), margin %.4f ≤ %.4f — expected int4/int8 "+
				"input-drift sensitivity, NOT gated (per-position cosine/argmax vs CPU is invalid past here)",
				k, k/nMoE, k%nMoE, mg, marginGate)
		}
	}
	if routeAgree && sensitivityFlips == 0 {
		t.Logf("routing agreement: resident idx == CPU-int4 idx at ALL %d decisions — pure accumulation, no flip", len(idxCpu4))
	} else if routeAgree {
		t.Logf("routing agreement: %d/%d decisions matched; %d near-tie flips (margin ≤ %.2f) exempted as sensitivity — "+
			"NO well-separated flip, so no dispatch/router bug", len(idxCpu4)-sensitivityFlips, len(idxCpu4), sensitivityFlips, marginGate)
	}

	mean := func(v []float64) float64 {
		s := 0.0
		for _, x := range v {
			s += x
		}
		return s / float64(len(v))
	}
	meanCuda, meanCpu := mean(cVs4), mean(c4VsF)
	t.Logf("gemma4 MoE resident (16 pos): pos0=%.6f | mean CUDA-vs-CPUint4=%.6f  CPUint4-vs-f32=%.6f",
		pos0, meanCuda, meanCpu)

	// ---- gates ----
	if pos0 < 0.97 {
		t.Errorf("pos-0 cosine %.6f < 0.97 — kernel divergence at the first token (GOINFER_G4_CAPTURE / "+
			"TestGemma4MoE_localize to localize)", pos0)
	}
	// CALIBRATED, RUN-LEVEL. A per-position CUDA ≥ CPUint4-vs-f32 gate is too literal: the two curves
	// measure DIFFERENT perturbations (CUDA differs from CPU only in W4A8 activation rounding; the
	// baseline is the full int4 weight quantization), so they legitimately CROSS position-to-position
	// (run to 16 and CUDA dips under at pos 7/9/15 — with routing bit-equal at all 32 decisions, i.e.
	// no flip, those are conditioning, not bugs). The property that survives a prompt/length change is
	// the run mean: CUDA must agree with CPU-int4 AT LEAST AS WELL, on average, as int4 agrees with
	// f32 — the activation perturbation is smaller than the weight one, so this holds by construction
	// and by a wide margin (~0.95 vs ~0.87). A real divergence (CUDA dropping FASTER than the fixture's
	// own quantization across the run) sinks the mean below the baseline; conditioning cannot.
	if meanCuda < meanCpu {
		t.Errorf("mean CUDA-vs-CPUint4 %.6f < mean CPUint4-vs-f32 %.6f — CUDA agrees with CPU-int4 WORSE than "+
			"int4 agrees with f32, i.e. it diverges faster than the fixture's own quantization: a real bug, not conditioning",
			meanCuda, meanCpu)
	}
}
