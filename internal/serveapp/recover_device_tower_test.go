package serveapp

import (
	"errors"
	"strings"
	"testing"
)

// A device tower that panics (an allocation failure inside aikit's Qwen2.5-VL tower) must fail the request with its reason, not the process.
func TestRecoverDeviceTower(t *testing.T) {
	got, err := recoverDeviceTower("Qwen2.5-VL", func() ([]float32, error) { return []float32{1, 2}, nil })
	if err != nil || len(got) != 2 {
		t.Fatalf("a healthy forward: got %v, %v", got, err)
	}
	want := errors.New("plain error")
	if _, err := recoverDeviceTower("Qwen2.5-VL", func() ([]float32, error) { return nil, want }); !errors.Is(err, want) {
		t.Errorf("an ordinary error must pass through untouched, got %v", err)
	}
	out, err := recoverDeviceTower("Qwen2.5-VL", func() ([]float32, error) {
		panic("cuda: device allocation failed (typed-len, 75294720 bytes): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")
	})
	if out != nil || err == nil || !strings.Contains(err.Error(), "Qwen2.5-VL tower failed") || !strings.Contains(err.Error(), "OUT_OF_MEMORY") ||
		!strings.Contains(err.Error(), "-vision-device cpu") {
		t.Errorf("a panic must become an error naming the family, the cause and the way out, got %v, %v", out, err)
	}
}

// Every family's tower runs through withFeatureCache, so a tower that panics there (Gemma 3's SigLIP, Gemma 4's image or audio tower, on any device) fails the
// request with an error, not the server. Before 2026-10-07 only Qwen2.5-VL's forward was recovered.
func TestWithFeatureCache_recoversAnyTower(t *testing.T) {
	lm := &loadedModel{}
	vi := visionInput{features: func() ([]float32, error) { panic("metal: device allocation failed (out of memory)") }}
	out, err := func() (out []float32, err error) {
		defer func() {
			if r := recover(); r != nil {
				err = errors.New("panic escaped")
			}
		}()
		return lm.withFeatureCache(vi, []byte("an image"), "Gemma 4 audio").features()
	}()
	if out != nil || err == nil || !strings.Contains(err.Error(), "Gemma 4 audio tower failed") || !strings.Contains(err.Error(), "-vision-device cpu") {
		t.Errorf("a panicking tower: got %v, %v; want an error naming the family and the way out", out, err)
	}
}
