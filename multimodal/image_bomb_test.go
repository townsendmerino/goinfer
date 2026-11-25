package multimodal

import (
	"strings"
	"testing"
)

// TestLfm2VLAndPixtral_rejectDecompressionBombArea pins that LFM2-VL and Pixtral enforce the 32 MP pixel-area
// limit before image.Decode runs. 16384x16384 (268 MP) clears the per-axis <= 16384 check, so without the
// pixel-area guard image.Decode runs on the forged IHDR and fails with a PNG stream/zlib error rather than
// the pre-decode "out of range" rejection.
func TestLfm2VLAndPixtral_rejectDecompressionBombArea(t *testing.T) {
	bomb := pngWithForgedIHDR(t, 16384, 16384)
	if _, err := Lfm2VLPreprocess(bomb, Lfm2VLPreprocessConfig{}); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("Lfm2VLPreprocess(16384x16384) err = %v, want pre-decode 'out of range' rejection", err)
	}
	if _, _, _, err := PixtralPreprocess(bomb, PixtralPreprocessConfig{}); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("PixtralPreprocess(16384x16384) err = %v, want pre-decode 'out of range' rejection", err)
	}
}
