//go:build gpu

package gpu

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/vision"
)

func TestVisionEncoder_timing(t *testing.T) {
	dir := os.Getenv("HOME") + "/models/gemma-3-4b-it"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("no gemma-3-4b-it")
	}
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	enc, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.GPUWeights()
	if err != nil {
		t.Fatal(err)
	}
	tu := time.Now()
	ve, err := c.NewVisionEncoder(w)
	if err != nil {
		t.Fatal(err)
	}
	defer ve.Close()
	t.Logf("upload weights: %v", time.Since(tu).Round(time.Millisecond))
	pixels := make([]float32, 3*896*896)
	for i := range pixels {
		pixels[i] = float32(i%255)/255 - 0.5
	}
	patches, _ := enc.GridPatches(pixels)
	if _, err := ve.ForwardPatches(patches); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	if _, err := ve.ForwardPatches(patches); err != nil {
		t.Fatal(err)
	}
	t.Logf("resident GPU encoder.ForwardPatches (real gemma-3-4b-it): %v", time.Since(t0).Round(time.Millisecond))
}
