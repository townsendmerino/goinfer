package vision

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

// TestPreprocess_zeroMaxPixelsUsesDefaultCap is audit C-06's first half. A
// Config built without naming MaxPixels — the natural literal, and what the
// zero value gives — used to disable the decompression-bomb guard entirely.
func TestPreprocess_zeroMaxPixelsUsesDefaultCap(t *testing.T) {
	if got := Gemma3().MaxPixels; got != DefaultMaxPixels {
		t.Errorf("Gemma3().MaxPixels = %d, want DefaultMaxPixels %d", got, DefaultMaxPixels)
	}
	// The assertion that matters: an UNSET cap resolves to the default, not to
	// "no limit". Against the old `cfg.MaxPixels > 0 &&` guard the effective cap
	// was 0 and every image passed however large.
	for _, tc := range []struct{ set, want int }{
		{0, DefaultMaxPixels},  // unset -> default, NOT unlimited
		{-1, DefaultMaxPixels}, // negative is not a backdoor either
		{16, 16},               // an explicit cap is honoured
		{1 << 30, 1 << 30},     // including a deliberately large one
	} {
		if got := (Config{MaxPixels: tc.set}).maxPixels(); got != tc.want {
			t.Errorf("Config{MaxPixels: %d}.maxPixels() = %d, want %d", tc.set, got, tc.want)
		}
	}
	// End to end: a small image still passes with the cap unset.
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 8, 8))); err != nil {
		t.Fatal(err)
	}
	if _, err := Preprocess(buf.Bytes(), Config{Size: 896, Std: [3]float32{1, 1, 1}}); err != nil {
		t.Fatalf("small image rejected under an unset MaxPixels: %v", err)
	}
}

// TestAveragePool_partialBucketUsesActualOccupancy is C-06's second half: the
// pooler divided every bucket by k*k regardless of how many patches landed in
// it, so an under-filled bucket was scaled toward zero rather than averaged.
func TestAveragePool_partialBucketUsesActualOccupancy(t *testing.T) {
	const hidden, k = 4, 3
	e := &Gemma4Encoder{}
	e.Cfg.HiddenSize = hidden
	e.Cfg.PoolingKernelSize = k

	// A 3x3 grid (one bucket) with only 3 of 9 positions present. Every present
	// patch is all-ones, so the correct mean is 1.0 in every channel; dividing
	// by k*k=9 would give 3/9 = 0.3333.
	pos := [][2]int{{0, 0}, {1, 0}, {2, 0}, {0, 2}} // maxX=2,maxY=2 -> 3x3 grid
	h := make([]float32, len(pos)*hidden)
	for i := range h {
		h[i] = 1
	}
	pooled, pw, ph, err := e.averagePool(h, pos, len(pos))
	if err != nil {
		t.Fatal(err)
	}
	if pw != 1 || ph != 1 {
		t.Fatalf("pool grid = %dx%d, want 1x1", pw, ph)
	}
	for d, got := range pooled {
		if got != 1 {
			t.Errorf("channel %d = %v, want 1 (mean of the 4 present patches); "+
				"%v is what dividing by k*k=%d gives", d, got, float32(len(pos))/float32(k*k), k*k)
		}
	}
}
