//go:build cuda

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSlidingWindowLongContext is the ONE end-to-end proof that CUDA's sliding window agrees
// with the CPU's on a real model *while the window is actually engaged*.
//
// Everything else stops short of this:
//   - TestSlidingWindowAttention proves the KERNEL (windowed-over-N == full-over-last-W, exact).
//   - TestBackendResidentWired asserts the WIRING (2047 reaches all 32 layers).
//   - but both parity gates run 8-token prompts, where winStart = max(0, 8-2047) = 0 — the
//     window is inert, so they cannot catch a winStart that disagrees with the CPU.
//
// Finding a checkpoint that can test this is harder than it sounds: the released GGUF
// conversions DROP the window. Phi-3's GGUF has no phi3.attention.sliding_window key, and
// Mistral-7B-v0.1's GGUF is converted as general.architecture="llama" with no key either — so
// both load full-attention and prove nothing. Only safetensors carry config.json's
// sliding_window. Phi-3-mini-4k safetensors (2047, all layers local) is the vehicle; it is
// driven by raw token ids here, which also sidesteps that checkpoint's unloadable tokenizer.
//
// Slow (a few thousand forwards on both CPU and GPU) — skipped under -short.
func TestSlidingWindowLongContext(t *testing.T) {
	if testing.Short() {
		t.Skip("long-context window proof: thousands of CPU forwards")
	}
	path := os.ExpandEnv("$HOME/models/phi3-mini-4k")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no phi3-mini-4k safetensors")
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	win := mc.SlidingWindowResident()
	if win <= 0 {
		t.Skipf("checkpoint declares no window (got %d) — nothing to prove", win)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident declined")
	}
	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	// Run past the window so winStart > 0 for the tail positions.
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
	worst := 0.0
	for i, tok := range prompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		cudaL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		// Only the tail matters: before pos >= win the window is inert on both sides.
		if i < win {
			continue
		}
		checked++
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
		if gap > 0.03 {
			hard++
			t.Errorf("pos %d (window ENGAGED, winStart=%d): CPU=%d CUDA=%d gap=%.3f%% > 3%% — "+
				"CUDA's window disagrees with the CPU's WindowStart", i, i+1-win, ca, ga, gap*100)
		}
	}
	t.Logf("window=%d engaged over %d tail positions (of %d): %d/%d exact | worst near-tie %.3f%% | hard fails %d",
		win, checked, n, exact, checked, worst*100, hard)
	if hard == 0 && checked > 0 {
		t.Logf("WINDOW GATE GREEN: CUDA's sliding window matches the CPU with the window actually engaged")
	}
}
