package embeddinggemma2

import (
	"strings"
	"testing"
)

func TestSetImageResize(t *testing.T) {
	var e Encoder
	if e.ImageResizeMode() != defaultImageResize {
		t.Fatalf("default %q, want %q", e.ImageResizeMode(), defaultImageResize)
	}
	for _, r := range []ImageResize{ResizeBilinear, ResizeBicubic, "BICUBIC"} {
		if err := e.SetImageResize(r); err != nil || e.ImageResizeMode() != ImageResize(strings.ToLower(string(r))) {
			t.Fatalf("SetImageResize(%q): %v, mode %q", r, err, e.ImageResizeMode())
		}
	}
	if err := e.SetImageResize("lanczos"); err == nil {
		t.Fatal("SetImageResize(lanczos) accepted")
	}
}
