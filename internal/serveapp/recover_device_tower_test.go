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
	if out != nil || err == nil || !strings.Contains(err.Error(), "Qwen2.5-VL vision tower failed on the device") || !strings.Contains(err.Error(), "OUT_OF_MEMORY") ||
		!strings.Contains(err.Error(), "-vision-device cpu") {
		t.Errorf("a panic must become an error naming the family, the cause and the way out, got %v, %v", out, err)
	}
}
