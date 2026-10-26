package vision

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

// FuzzPreprocess feeds arbitrary bytes to the image preprocessing path
// (campaign Track-2, extended to image input): a typed error or valid
// pixel_values, never a panic/OOM/hang. A small size + tight pixel cap keep each
// exec's decode allocation bounded.
func FuzzPreprocess(f *testing.F) {
	// Seeds: a couple of valid tiny PNGs + obvious junk.
	add := func(w, h int) {
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = byte(i)
		}
		var buf bytes.Buffer
		_ = png.Encode(&buf, img)
		f.Add(buf.Bytes())
	}
	add(1, 1)
	add(8, 5)
	f.Add([]byte("not an image"))
	f.Add([]byte{})

	c := Config{Size: 16, Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}, MaxPixels: 1 << 16}
	f.Fuzz(func(t *testing.T, data []byte) {
		pv, err := Preprocess(data, c)
		if err != nil {
			return // typed error is the correct outcome for hostile/garbage bytes
		}
		if pv == nil || len(pv.Data) != 3*c.Size*c.Size {
			t.Fatalf("ok return but bad pixel_values: %v", pv)
		}
	})
}
