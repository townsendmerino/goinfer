package multimodal

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

// TestQwenPreprocess_inputPixelLimit_M15 gates M-15: an input image whose raw dimensions exceed
// qwenMaxInputPixels is rejected right after decode, before the h*w*3 allocation — a small
// compressible PNG that decodes to a huge canvas can no longer drive a ~GB allocation per
// request ahead of the model mutex. A normal-sized image still preprocesses.
func TestQwenPreprocess_inputPixelLimit_M15(t *testing.T) {
	cfg := QwenDefaultPreprocess()
	enc := func(w, h int) []byte {
		var buf bytes.Buffer
		if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	// A blank 12000x12000 PNG (144 MP) compresses tiny but would allocate ~1.7 GB; must be refused.
	if _, _, err := QwenPreprocess(enc(12000, 12000), cfg); err == nil {
		t.Error("144 MP image accepted (M-15): must reject above the input-pixel limit")
	}
	// A realistic image still works.
	if _, _, err := QwenPreprocess(enc(224, 224), cfg); err != nil {
		t.Errorf("small image rejected: %v", err)
	}
}
