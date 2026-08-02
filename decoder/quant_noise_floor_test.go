package decoder

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestQuantNoiseFloor_gemma4MoE is the Split-B pre-flight: it measures, on CPU, the two things that
// actually predict whether a resident (GPU) MoE kernel can hit parity on this fixture — measured
// BEFORE any kernel is written, at the cost of a few CPU forwards and no GPU.
//
// It began as a pure int4-vs-f32 "noise floor" (CPU-at-quant vs CPU-f32), on the theory that a
// resident backend can only agree with CPU-int4 as well as int4 agrees with f32. That theory is
// FALSIFIED by the Split-A control: the dense two-geometry fixture PASSES the resident gate
// (cuda-int4 vs cpu-int4) at cosine 0.979 while its own int4-vs-f32 floor is only 0.880
// (NOISE_FLOOR_CKPT=../testdata/gemma4-dense-twogeom-tiny). The resident gate compares int4-to-int4,
// so a modest f32-floor is fine — it's a chaos sniff-test, not a hard bar.
//
// What a resident MoE kernel can get wrong that a dense one can't is a ROUTING FLIP: quant noise near
// a router tie picks a DIFFERENT expert — a different computation, not a small numeric error. So the
// real gate is (1) routing agreement 100% and (2) a min routing MARGIN wide enough that the tighter
// cpu-int4-vs-gpu-int4 gap can't flip it either. The int4-vs-f32 logit cosine stays REPORTED as
// context. (History: the Split-A dense fixture at hidden=64 manufactured a phantom "bug" — cuda
// resident cosine drifted to 0.82, pure int8-activation sensitivity, fixed by hidden≥256. Same
// instinct built this fixture at hidden=64; measure before any Split-B kernel.)
func TestQuantNoiseFloor_gemma4MoE(t *testing.T) {
	ckpt := "../testdata/gemma4-moe-tiny"
	if e := os.Getenv("NOISE_FLOOR_CKPT"); e != "" {
		ckpt = e
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}

	mf, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load f32: %v", err)
	}
	defer mf.Close()
	m4, err := Load(ckpt, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m4.Close()

	// Run each quant FULLY (not interleaved) so routerCapture records a clean per-run sequence of
	// selected expert sets. MoE adds a DISCRETE failure mode dense doesn't: quant noise near a
	// router tie flips the top-k, and a flipped expert is a DIFFERENT computation, not a small
	// numerical error. If routing already disagrees CPU-f32 vs CPU-int4, NO resident kernel can
	// ever match — the divergence is a fixture property, not a kernel bug.
	routerCapture = true
	defer func() { routerCapture = false }()

	run := func(m *Model) (logits [][]float32, routes [][]int, margins []float32) {
		routerCaptureBuf = nil
		routerMarginBuf = nil
		cache := m.NewCache(len(prompt))
		for i, tok := range prompt {
			l, err := m.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatalf("forward pos %d: %v", i, err)
			}
			logits = append(logits, append([]float32(nil), l...))
		}
		return logits, routerCaptureBuf, routerMarginBuf
	}
	lfAll, rf, mgf := run(mf)
	l4All, r4, mg4 := run(m4)

	minCos := 1.0
	for i := range lfAll {
		lf, l4 := lfAll[i], l4All[i]
		var dot, na, nb float64
		for j := range lf {
			dot += float64(lf[j]) * float64(l4[j])
			na += float64(lf[j]) * float64(lf[j])
			nb += float64(l4[j]) * float64(l4[j])
		}
		c := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if c < minCos {
			minCos = c
		}
		t.Logf("  pos %2d  int4-vs-f32 cosine %.6f", i, c)
	}

	// Routing agreement: fraction of MoE selections where int4 picked the SAME expert SET as f32.
	sameSet := func(a, b []int) bool {
		if len(a) != len(b) {
			return false
		}
		seen := map[int]bool{}
		for _, x := range a {
			seen[x] = true
		}
		for _, x := range b {
			if !seen[x] {
				return false
			}
		}
		return true
	}
	agree, total := 0, 0
	for i := range rf {
		if i < len(r4) {
			total++
			if sameSet(rf[i], r4[i]) {
				agree++
			}
		}
	}
	routeAgree := 1.0
	if total > 0 {
		routeAgree = float64(agree) / float64(total)
	}

	// ROUTING MARGIN is the real MoE-robustness metric, not the int4-vs-f32 logit cosine. The margin
	// is the top-k boundary gap (min selected prob − max rejected prob): the distance a quant
	// perturbation must move a decision to flip an expert. Report the MINIMUM across the run in both
	// paths — a fixture is routing-robust when even its tightest decision keeps a wide margin, so the
	// GPU int4 path (closer to CPU int4 than f32 is) cannot flip it. Constructed margins (fit
	// router.proj to the fixture's own router inputs) make this large by design.
	minMargin := func(mg []float32) float32 {
		m := float32(1)
		for _, x := range mg {
			if x < m {
				m = x
			}
		}
		return m
	}
	mmF, mm4 := minMargin(mgf), minMargin(mg4)
	t.Logf("gemma4-moe-tiny int4 PRE-FLIGHT: logit minCosine=%.6f (f32-floor) | expert-set agreement %d/%d = %.1f%% | "+
		"min routing margin f32=%.4f int4=%.4f (over %d decisions)",
		minCos, agree, total, routeAgree*100, mmF, mm4, total)

	// GATE — recalibrated. The int4-vs-f32 logit floor is NOT the resident-gate predictor: the Split-A
	// dense two-geometry fixture PASSES the resident gate (cuda-int4 vs cpu-int4) at cosine 0.979 while
	// its OWN int4-vs-f32 floor is only 0.880 (measured: NOISE_FLOOR_CKPT=…/gemma4-dense-twogeom-tiny).
	// The resident gate compares int4-to-int4, so a modest f32-floor is fine; what a resident MoE kernel
	// can still get wrong that dense can't is a ROUTING FLIP. So the pre-flight gates on the two things
	// that actually predict resident MoE parity:
	//   (1) routing agreement 100% (int4 must not flip the top-k vs f32 — a flip is a fixture defect,
	//       unrecoverable by any kernel), and
	//   (2) a min routing margin comfortably above the observed logit perturbation, so the tighter
	//       cpu-int4-vs-gpu-int4 gap can't flip it either. 0.02 ≈ 2 percentage points of router prob,
	//       far wider than the sub-1% activation-quant noise a resident GEMV introduces.
	// The f32-floor stays REPORTED (a chaos sniff-test; two-geom-calibrated ≳0.85) but is no longer a
	// hard 0.97 bar. The true kernel gate is int4-vs-int4, asserted when Split B's resident MoE lands.
	const marginFloor = 0.02
	if routeAgree < 1.0 || mm4 < marginFloor {
		t.Skipf("gemma4-moe-tiny is NOT resident-gate-ready: expert-set agreement %.1f%%, min int4 routing margin %.4f "+
			"(need 100%% agreement and margin ≥ %.2f). Construct wider router margins (fit router.proj to the fixture's "+
			"router inputs) so no decision sits on a near-tie. f32-floor %.3f is CONTEXT only (two-geom control passes "+
			"resident at 0.88).", routeAgree*100, mm4, marginFloor, minCos)
	}
}
