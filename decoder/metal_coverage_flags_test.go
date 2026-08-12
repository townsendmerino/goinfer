package decoder

import (
	"os"
	"testing"
)

// TestMetalCoverageFlags validates the decoder-side arch detection the Metal resident
// decoder depends on (docs/metal-coverage-validation-prompt.md §1). Metal is darwin-only,
// so it cannot run here — but the biggest risk in the Mac's coverage work is NOT the
// kernels, it's whether the decoder detects these flags from a REAL checkpoint. A silently
// false HasQKNorm() on a Qwen3 GGUF means Metal skips QK-norm and emits wrong output.
//
// This asserts nothing it cannot prove; it reports the ACTUAL values per family so the Mac
// knows what "correct" looks like before it runs. Skips per-family when the checkpoint is
// absent (CI has none of these).
func TestMetalCoverageFlags(t *testing.T) {
	requireHeavyModel(t) // loads real ~/models checkpoints (qwen3-1.7b, phi3-mini-4k, ...)
	cases := []struct {
		family string
		path   string
		// expectations from the checkpoint's own config.json (see the report)
		wantQKNorm, wantPartialRotary bool
		wantWindow                    int // 0 = none
	}{
		{"qwen3", os.ExpandEnv("$HOME/models/qwen3-1.7b-q8_0.gguf"), true, false, 0},
		{"qwen3-safetensors", os.ExpandEnv("$HOME/models/qwen3-1.7b-bf16"), true, false, 0},
		// Phi-3-mini-4k: the safetensors config.json declares sliding_window: 2047, which
		// phi3Architecture now honours. The released GGUF conversion DROPS the
		// phi3.attention.sliding_window key entirely (verified by reading the file's metadata:
		// no key matching "sliding"), so that file honestly declares no window and stays
		// full-attention. The two paths legitimately differ — this pins that asymmetry.
		{"phi3", os.ExpandEnv("$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"), false, false, 0},
		{"phi3-safetensors", os.ExpandEnv("$HOME/models/phi3-mini-4k"), false, false, 2047},
	}
	for _, c := range cases {
		t.Run(c.family, func(t *testing.T) {
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no checkpoint at %s", c.path)
			}
			m, err := Load(c.path, Options{Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()

			H, nL, nH, nKV, hd, I, vocab := m.Dims()
			qkn := m.HasQKNorm()
			pr := m.PartialRotary()
			win := m.SlidingWindowResident()
			addOne := m.RMSAddOne()
			elig := m.DecodeRunnerEligible()
			invF := len(m.RopeInvFreq())

			t.Logf("%s: H=%d L=%d nH=%d nKV=%d hd=%d I=%d vocab=%d", c.family, H, nL, nH, nKV, hd, I, vocab)
			t.Logf("  HasQKNorm=%v  PartialRotary=%v  SlidingWindow=%d  RMSAddOne=%v  EmbedScale=%v",
				qkn, pr, win, addOne, m.EmbedScaleResident())
			t.Logf("  len(RopeInvFreq)=%d (headDim/2=%d ⇒ %s)  DecodeRunnerEligible=%v",
				invF, hd/2, map[bool]string{true: "FULL rotary", false: "PARTIAL rotary"}[invF == hd/2], elig)

			// Per-layer QK-norm tensors — Metal reads these; empty ⇒ it silently skips the norm.
			if qkn {
				bad := 0
				for l := range nL {
					lw := &m.Weights().Layers[l]
					if len(lw.QNorm) != hd || len(lw.KNorm) != hd {
						bad++
					}
				}
				if bad > 0 {
					t.Errorf("  QK-norm claimed but %d/%d layers lack len==headDim(%d) QNorm/KNorm", bad, nL, hd)
				} else {
					t.Logf("  QNorm/KNorm present, len==headDim(%d), on all %d layers ✓", hd, nL)
				}
			}
			// Sliding window: report which layers are local (Metal advances start per-layer).
			if win > 0 {
				local := 0
				for l := range nL {
					if m.LayerIsLocalResident(l) {
						local++
					}
				}
				t.Logf("  window=%d; LayerIsLocalResident true on %d/%d layers", win, local, nL)
			}

			// Flag the deltas vs what the Mac's prompt expects.
			if qkn != c.wantQKNorm {
				t.Errorf("  HasQKNorm=%v want %v — DECODER DETECTION BUG", qkn, c.wantQKNorm)
			}
			if pr != c.wantPartialRotary {
				t.Errorf("  PartialRotary=%v want %v — DECODER DETECTION BUG", pr, c.wantPartialRotary)
			}
			if win != c.wantWindow {
				t.Errorf("  SlidingWindow=%d want %d — DECODER DETECTION BUG", win, c.wantWindow)
			}
			if !elig {
				t.Errorf("  DecodeRunnerEligible=false — Metal would NEVER build a resident for %s", c.family)
			}
			// The GGUF/safetensors window asymmetry is real, not a bug: the released GGUF
			// conversion drops the sliding_window key, so each file gets what it declares.
			if c.family == "phi3" {
				t.Logf("  note: full attention because this GGUF carries no phi3.attention.sliding_window " +
					"key (the conversion dropped it); the safetensors sibling declares 2047 and windows.")
			}
		})
	}
}
