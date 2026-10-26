package audio

import (
	"math"
	"testing"
)

// TestGemma4Counts pins the valid-frame and soft-token counts to the ones measured against transformers 5.19.0's
// extractor and processor (goinfer's audio-gate0 spec, §1 and §3), including the edges: under one frame, and the
// 30 s truncation.
func TestGemma4Counts(t *testing.T) {
	for _, c := range []struct{ samples, frames, tokens int }{
		{16000, 99, 25}, {16001, 100, 25}, {8000, 49, 13}, {160, 0, 0}, {100, 0, 0},
		{37920, 236, 59}, {240000, 1499, 375}, {500000, 2999, 750},
	} {
		f := Gemma4ValidFrames(c.samples)
		if f != c.frames || Gemma4SoftTokens(f) != c.tokens {
			t.Errorf("%d samples: %d frames, %d tokens; want %d, %d", c.samples, f, Gemma4SoftTokens(f), c.frames, c.tokens)
		}
	}
}

// TestGemma4Features_edges: the lowest mel filter is all zero, so bin 0 is ln(0.001) on every frame (spec §1.8), and
// audio too short for a frame is refused rather than turned into an empty tower input.
func TestGemma4Features_edges(t *testing.T) {
	x := make([]float32, 4000)
	for i := range x {
		x[i] = float32(0.5 * math.Sin(float64(i)*0.3))
	}
	f, T, err := Gemma4Features(x)
	if err != nil {
		t.Fatal(err)
	}
	for r := range T {
		if d := math.Abs(float64(f[r*gemma4Mels]) - math.Log(gemma4MelFloor)); d > 1e-6 {
			t.Fatalf("frame %d bin 0 = %g, want ln(0.001)", r, f[r*gemma4Mels])
		}
	}
	if _, _, err := Gemma4Features(make([]float32, 160)); err == nil {
		t.Fatal("160 samples (no frame) accepted")
	}
}
