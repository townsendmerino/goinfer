//go:build cuda

package cuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSlidingWindowLongContext is the ONE end-to-end proof that CUDA's sliding window agrees
// with the CPU's WHILE THE WINDOW IS ACTUALLY ENGAGED.
//
// Everything else stops short of this:
//   - TestSlidingWindowAttention proves the KERNEL (windowed-over-N == full-over-last-W, exact).
//   - TestBackendResidentWired asserts the WIRING (the window reaches every layer).
//   - but both run short prompts, where winStart = max(0, nKeys-W) = 0 — the window is INERT,
//     so neither can catch a winStart that disagrees with the CPU's.
//
// WHY A TINY FIXTURE. This test used to drive the real phi3-mini-4k, whose window is 2047: it
// needed win+40 = 2087 forwards on BOTH the CPU and the GPU of a 3.8B model, took 15-25
// minutes, and so never actually completed — it was skipped in practice and gated nothing. A
// gate that cannot finish is not a gate.
//
// The window's SPAN is not a property worth scaling: winStart = max(pos-W+1, 0) is the same
// arithmetic at W=16 as at W=4096. testdata/mistral-tiny-window is a seeded 1.9 MB Mistral with
// sliding_window=16, so 56 forwards cover the identical logic in ~1s, with ~40 positions PAST
// the window where winStart is > 0 and MOVING — which is the whole point. It also closes a
// second gap: the README claims Mistral runs GPU-resident, and this is the only place a Mistral
// checkpoint is actually run resident.
//
// (Only safetensors can test this at all: the released Mistral and Phi-3 GGUF conversions DROP
// sliding_window — Mistral-7B-v0.1's is converted as general.architecture="llama" with no key —
// so a GGUF loads full-attention and proves nothing.)
func TestSlidingWindowLongContext(t *testing.T) {
	const path = "../testdata/mistral-tiny-window"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no tiny windowed fixture at %s (regenerate: scripts/pin_mistral_tiny_window.py)", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	win := mc.SlidingWindowResident()
	if win <= 0 {
		t.Fatalf("fixture declares no window (got %d) — the whole point of this fixture is that "+
			"config.json carries sliding_window; regenerate it", win)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED a windowed Mistral — admission says sliding-window is " +
			"implemented, so either the feature gate or the declaration is wrong")
	}
	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	// Run well past the window so winStart > 0 and MOVES for the tail positions.
	n := win + 40
	_, _, _, _, _, _, vocab := mc.Dims()
	prompt := make([]int, n)
	var s uint32 = 7
	for i := range prompt {
		s = s*1664525 + 1013904223
		prompt[i] = int(s>>8) % (vocab - 1)
	}

	cache := mcpu.NewCache(n + 2)
	exact, hard, checked := 0, 0, 0
	worst, minCos := 0.0, 1.0
	for i, tok := range prompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		cudaL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		// Only the tail matters: before pos >= win the window is inert on BOTH sides, so those
		// positions would pass even with a broken winStart.
		if i < win {
			continue
		}
		checked++
		var dot, na, nb float64
		for j := range cpuL {
			dot += float64(cpuL[j]) * float64(cudaL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(cudaL[j]) * float64(cudaL[j])
		}
		if c := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30); c < minCos {
			minCos = c
		}
		ca, ga := argmaxF(cpuL), argmaxF(cudaL)
		if ca == ga {
			exact++
			continue
		}
		lo, hi := cpuL[0], cpuL[0]
		for _, v := range cpuL {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
		if gap > worst {
			worst = gap
		}
		// The repo's 3% near-tie rule (gpu/kv_i8_parity_test.go). A winStart that disagrees with
		// the CPU's attends over the wrong span entirely — that is not a near-tie.
		if gap > 0.03 {
			hard++
			t.Errorf("pos %d (window ENGAGED, winStart=%d): CPU=%d CUDA=%d gap=%.3f%% > 3%% — "+
				"CUDA's window disagrees with the CPU's WindowStart", i, i+1-win, ca, ga, gap*100)
		}
	}
	if checked == 0 {
		t.Fatalf("no positions were checked past the window (win=%d, n=%d) — the test ran but "+
			"gated NOTHING, which is exactly the failure mode of the 2047-window version", win, n)
	}
	t.Logf("window=%d engaged over %d tail positions (of %d): %d/%d exact | worst near-tie %.3f%% | "+
		"min cosine %.6f | hard fails %d", win, checked, n, exact, checked, worst*100, minCos, hard)
	if hard == 0 {
		t.Logf("WINDOW GATE GREEN: CUDA's sliding window matches the CPU with the window actually " +
			"engaged, on a real Mistral checkpoint")
	}
}
