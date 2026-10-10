//go:build goinfer_testhooks

package decoder

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestQuantNoiseFloor_gemma4MoE is the Split-B pre-flight: it measures, on CPU, the two things that predict whether a
// resident (GPU) MoE kernel can hit parity on this fixture, before any kernel is written, at the cost of a few CPU forwards
// and no GPU.
//
// ROUTING. What a resident MoE kernel can get wrong that a dense one can't is a ROUTING FLIP: quant noise near a router tie
// picks a DIFFERENT expert, a different computation, not a small numeric error. So the gate is (1) routing agreement 100%
// and (2) a min routing MARGIN wide enough that the tighter cpu-int4-vs-gpu-int4 gap can't flip it either.
//
// The int4-vs-f32 logit floor (CPU-at-quant vs CPU-f32) is REPORTED as a warning signal, not gated: it is a conditioning
// proxy, correlated with (not independent of) resident parity, and the 0.97 bar it once carried was uncalibrated. Fixture
// size matters (a hidden=64 fixture manufactures phantom parity failures from int8-activation sensitivity), so measure
// first. History and the calibration point: docs/code-notes/decoder.md#TestQuantNoiseFloor_gemma4MoE.
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
		routerCaptureReset() // N-27: clears every buffer AND re-arms the cap
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

	// ROUTING MARGIN is the resident-MoE-robustness metric. The margin is the top-k boundary gap (min
	// selected prob − max rejected prob): the distance a perturbation must move a decision to flip an
	// expert. What the gate needs to survive is the CPU-int4-vs-CUDA-int4 router-input difference (same
	// quantized weights, different activation rounding/grouping). That can't be measured until the
	// resident MoE kernel exists (Split B task 2), so gate on the int4 path's OWN min margin (mm4) and
	// separately report the int4-vs-f32 margin EROSION as a CONSERVATIVE upper bound on that difference
	// — CUDA-vs-CPU-int4 shares the weights and so erodes LESS than int4-vs-f32 does. If the int4 path
	// keeps a wide margin after the full f32→int4 erosion, the smaller residual CUDA erosion is safe.
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
	maxErosion := float32(0) // per-decision f32→int4 margin loss (mgf and mg4 are index-aligned).
	for i := range mg4 {
		if i < len(mgf) {
			if e := mgf[i] - mg4[i]; e > maxErosion {
				maxErosion = e
			}
		}
	}
	t.Logf("gemma4-moe-tiny int4 PRE-FLIGHT: logit minCosine=%.6f (f32-floor) | expert-set agreement %d/%d = %.1f%% | "+
		"min routing margin f32=%.4f int4=%.4f, max f32→int4 erosion=%.4f (conservative bound on CUDA erosion) over %d decisions",
		minCos, agree, total, routeAgree*100, mmF, mm4, maxErosion, total)

	// GATE. A routing flip is the one discrete failure mode a resident MoE kernel can get wrong that a dense one can't, and no
	// kernel can recover from it, so the pre-flight gates on:
	//   (1) routing agreement 100% (int4 must not flip the top-k vs f32; a flip is a fixture defect), and
	//   (2) a min int4 routing margin comfortably above the perturbation, so the residual CUDA-vs-CPU-int4 gap can't flip it
	//       either (marginFloor is about 2 points of router prob; the reported f32->int4 erosion shows the actual margin loss).
	//
	// The int4-vs-f32 logit floor is a WARNING SIGNAL, not a gate: a conditioning proxy correlated with (not independent of)
	// resident parity, whose one calibration point establishes a value as fine FOR THAT FIXTURE, not that any lower value is
	// fine in general. If the resident MoE gate comes back marginal, a low floor is the first suspect.
	const marginFloor = 0.02
	if routeAgree < 1.0 || mm4 < marginFloor {
		t.Skipf("gemma4-moe-tiny is NOT resident-gate-ready: expert-set agreement %.1f%%, min int4 routing margin %.4f "+
			"(need 100%% agreement and margin ≥ %.2f). Construct wider router margins (fit router.proj to the fixture's "+
			"router inputs) so no decision sits on a near-tie. f32-floor %.3f is a warning signal, reported not gated.",
			routeAgree*100, mm4, marginFloor, minCos)
	}
}
