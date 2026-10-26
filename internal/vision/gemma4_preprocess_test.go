package vision

import "testing"

// TestGemma4AspectRatioSize_matchesRealFormula cross-checks the Go port against
// get_aspect_ratio_preserving_size (image_processing_gemma4.py) computed directly
// in Python for the same inputs (non-degenerate aspect ratios — the extreme-ratio
// edge-case clamp differs slightly and is a documented, deferred simplification;
// see Gemma4AspectRatioSize's doc comment).
func TestGemma4AspectRatioSize_matchesRealFormula(t *testing.T) {
	cases := []struct {
		h, w, maxSoftTokens int
		wantH, wantW        int
	}{
		{1000, 800, 280, 864, 672},
		{1920, 1080, 280, 1056, 576},
		{100, 100, 70, 384, 384},
		{500, 2000, 560, 528, 2256},
		{96, 96, 70, 384, 384},
	}
	for _, c := range cases {
		gotH, gotW := Gemma4AspectRatioSize(c.h, c.w, c.maxSoftTokens, 16, 3)
		if gotH != c.wantH || gotW != c.wantW {
			t.Errorf("Gemma4AspectRatioSize(%d,%d,%d) = (%d,%d), want (%d,%d)",
				c.h, c.w, c.maxSoftTokens, gotH, gotW, c.wantH, c.wantW)
		}
	}
}

// TestGemma4Preprocess_shapeAndDeterminism checks the end-to-end preprocessing
// path (decode -> resize -> unfold) produces the expected shapes and is
// deterministic on a synthetic solid-color image, without asserting pixel-exact
// parity against PIL's bicubic resize (this repo's existing SigLIP preprocessing
// has the same documented gap — see preprocess.go's package doc comment).
func TestGemma4Preprocess_shapeAndDeterminism(t *testing.T) {
	img := solidPNG(t, 96, 96, 128, 64, 32)
	patches, positionIDs, err := Gemma4Preprocess(img, 70)
	if err != nil {
		t.Fatalf("Gemma4Preprocess: %v", err)
	}
	// 96x96 at max_soft_tokens=70 -> target 384x384 (per the cross-check above)
	// -> grid 24x24=576 patches of 3*16*16=768 each.
	wantPatches := 24 * 24
	if len(positionIDs) != wantPatches {
		t.Fatalf("len(positionIDs) = %d, want %d", len(positionIDs), wantPatches)
	}
	if len(patches) != wantPatches*768 {
		t.Fatalf("len(patches) = %d, want %d", len(patches), wantPatches*768)
	}
	for _, v := range patches {
		if v < 0 || v > 1 {
			t.Fatalf("patch value %g out of [0,1] range", v)
		}
	}
	patches2, positionIDs2, err := Gemma4Preprocess(img, 70)
	if err != nil {
		t.Fatalf("Gemma4Preprocess (2nd call): %v", err)
	}
	for i := range patches {
		if patches[i] != patches2[i] {
			t.Fatalf("non-deterministic patch value at %d: %g vs %g", i, patches[i], patches2[i])
		}
	}
	for i := range positionIDs {
		if positionIDs[i] != positionIDs2[i] {
			t.Fatalf("non-deterministic position id at %d: %v vs %v", i, positionIDs[i], positionIDs2[i])
		}
	}
}

func TestGemma4Preprocess_rejectsUnsupportedBudget(t *testing.T) {
	img := solidPNG(t, 32, 32, 10, 10, 10)
	if _, _, err := Gemma4Preprocess(img, 100); err == nil {
		t.Fatal("expected an error for max_soft_tokens=100 (not one of {70,140,280,560,1120})")
	}
}

func BenchmarkGemma4Unfold(b *testing.B) {
	const targetH, targetW, patchSize = 864, 672, 16
	hwc := make([]float32, targetH*targetW*3)
	for i := range hwc {
		hwc[i] = float32(i%100) * 0.01
	}
	b.SetBytes(int64(targetH * targetW * 3 * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_, _ = gemma4Unfold(hwc, targetH, targetW, patchSize)
	}
}
