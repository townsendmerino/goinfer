package vision

import (
	"bytes"
	"image"
	"image/png"
	"math"
	"testing"
)

// solidPNG encodes a w×h opaque PNG of a single RGB colour.
func solidPNG(t *testing.T, w, h int, r, g, b uint8) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = r, g, b, 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func cfg(size int) Config {
	return Config{Size: size, Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}, MaxPixels: 1 << 20}
}

// TestPreprocess_solidColor: resizing a solid colour is exact (every output
// pixel is that colour), so the normalized value is known in closed form —
// validates the rescale + normalize math and the CHW layout independent of the
// (parity-pending) interpolation kernel.
func TestPreprocess_solidColor(t *testing.T) {
	const size = 8
	cases := []struct {
		r, g, b uint8
	}{{0, 0, 0}, {255, 255, 255}, {128, 64, 200}}
	for _, c := range cases {
		pv, err := Preprocess(solidPNG(t, 20, 13, c.r, c.g, c.b), cfg(size))
		if err != nil {
			t.Fatalf("Preprocess: %v", err)
		}
		if len(pv.Data) != 3*size*size || pv.Size != size {
			t.Fatalf("shape: len=%d size=%d, want %d / %d", len(pv.Data), pv.Size, 3*size*size, size)
		}
		want := [3]float32{
			(float32(c.r)/255 - 0.5) / 0.5,
			(float32(c.g)/255 - 0.5) / 0.5,
			(float32(c.b)/255 - 0.5) / 0.5,
		}
		plane := size * size
		for ch := range 3 {
			for i := range plane {
				if got := pv.Data[ch*plane+i]; math.Abs(float64(got-want[ch])) > 1e-4 {
					t.Fatalf("rgb(%d,%d,%d) ch %d idx %d = %v, want %v", c.r, c.g, c.b, ch, i, got, want[ch])
				}
			}
		}
	}
}

// TestPreprocess_rangeAndDeterminism: arbitrary content stays in [-1,1] for the
// SigLIP mean/std, and the same bytes preprocess identically.
func TestPreprocess_rangeAndDeterminism(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 37, 19))
	for i := range img.Pix {
		img.Pix[i] = byte((i * 7) % 256)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	a, err := Preprocess(buf.Bytes(), cfg(16))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range a.Data {
		if v < -1.0001 || v > 1.0001 {
			t.Fatalf("value %v at %d outside [-1,1]", v, i)
		}
	}
	b, err := Preprocess(buf.Bytes(), cfg(16))
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("non-deterministic at %d: %v vs %v", i, a.Data[i], b.Data[i])
		}
	}
}

// TestPreprocess_guardsAndErrors: the decompression-bomb guard and malformed
// input both produce a typed error, never a panic.
func TestPreprocess_guardsAndErrors(t *testing.T) {
	// Oversized image vs a tiny MaxPixels → rejected before the (already-decoded)
	// bitmap matters; the guard reads the header dims.
	small := Config{Size: 8, Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}, MaxPixels: 16}
	if _, err := Preprocess(solidPNG(t, 64, 64, 10, 20, 30), small); err == nil {
		t.Error("expected a pixel-limit error for a 64x64 image with MaxPixels=16")
	}
	for name, data := range map[string][]byte{
		"empty":         nil,
		"garbage":       []byte("not an image at all"),
		"truncated-png": solidPNG(t, 8, 8, 1, 2, 3)[:20],
	} {
		if _, err := Preprocess(data, cfg(8)); err == nil {
			t.Errorf("%s: expected a decode error, got nil", name)
		}
	}
}
