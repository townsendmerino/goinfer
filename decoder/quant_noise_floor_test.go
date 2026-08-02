package decoder

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestQuantNoiseFloor_gemma4MoE establishes the QUANTIZATION NOISE FLOOR of a fixture BEFORE it is
// used to gate a resident (GPU) backend at that quant: CPU-at-quant vs CPU-f32 on the same tokens.
// A resident backend can only agree with the CPU int4 path as well as int4 agrees with f32 — so if
// this floor is already below the resident tolerance, the fixture CANNOT gate that quant, no matter
// how correct the kernel is. Checking it costs two CPU forwards and no GPU.
//
// This exists because the Split-A dense fixture at hidden=64 manufactured a phantom "bug": the
// CUDA resident cosine drifted to 0.82, which was pure int8-activation sensitivity, not a defect —
// visible immediately here as a bad CPU int4-vs-f32 floor, had it been run first. The Gemma-4 MoE
// fixture was built with the same instinct (hidden=64), so measure it before any Split-B kernel.
func TestQuantNoiseFloor_gemma4MoE(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
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

	run := func(m *Model) (logits [][]float32, routes [][]int) {
		routerCaptureBuf = nil
		cache := m.NewCache(len(prompt))
		for i, tok := range prompt {
			l, err := m.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatalf("forward pos %d: %v", i, err)
			}
			logits = append(logits, append([]float32(nil), l...))
		}
		return logits, routerCaptureBuf
	}
	lfAll, rf := run(mf)
	l4All, r4 := run(m4)

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
	t.Logf("gemma4-moe-tiny int4 NOISE FLOOR: logit minCosine=%.6f | expert-set agreement %d/%d = %.1f%%",
		minCos, agree, total, routeAgree*100)

	// This is a PRE-FLIGHT check for Split B, not yet a hard gate (no resident MoE runner exists on
	// this fixture yet). When Split B lands the resident MoE path, its parity gate MUST assert this
	// floor is ABOVE its tolerance and routing agreement is 100% — else the fixture, not the kernel,
	// is what's failing. Today the fixture flunks both (near-tie routing int4 flips), so skip loudly
	// with the fix, rather than red a gate that has no kernel behind it.
	if routeAgree < 1.0 || minCos < 0.97 {
		t.Skipf("gemma4-moe-tiny is NOT resident-gate-ready: logit floor %.3f, expert-set agreement %.1f%%. "+
			"int4 flips the router's top-k vs f32 — a FIXTURE property (near-tie routing), not a kernel bug. "+
			"Split B's first task: a routing-robust MoE fixture (fewer/structured experts, larger hidden) that "+
			"clears BOTH checks, THEN kernels.", minCos, routeAgree*100)
	}
}
