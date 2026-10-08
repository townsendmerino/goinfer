package serveapp

import (
	"os"
	"path/filepath"
	"testing"
)

// The S4 addendum (docs/tasks/task-multimodal-support-2026-10.md): -vision-quant is unset by default so an explicit f32 is distinguishable from the default.
// Unset keeps today's choice (Gemma 3's device tower on cuda/webgpu is int8); an explicit f32 selects the float32 tower; int8 stays int8 everywhere.
func TestTowerInt8_gemma3ExplicitF32(t *testing.T) {
	cases := []struct {
		quant, backend string
		want           bool
	}{
		{"", "cuda", true}, {"", "webgpu", true}, {"", "metal", false}, {"", "cpu", false}, {"", "", false}, // unset: unchanged
		{"f32", "cuda", false}, {"f32", "webgpu", false}, {"f32", "metal", false}, {"f32", "cpu", false}, // explicit f32: float32 everywhere
		{"int8", "cuda", true}, {"int8", "webgpu", true}, {"int8", "metal", true}, {"int8", "cpu", true}, // int8 when asked
	}
	for _, c := range cases {
		if got := towerInt8("gemma3", c.quant, c.backend); got != c.want {
			t.Errorf("towerInt8(gemma3, %q, %q) = %v, want %v", c.quant, c.backend, got, c.want)
		}
	}
}

// Gemma 3's float32 SigLIP tower is priced into the resident plan only when it will really run float32 on CUDA: the default (int8) reserves nothing, an explicit f32
// reserves about 2 GB (27 layers of hidden 1152 / intermediate 4304 in float32 plus the 4096-patch scratch).
func TestTowerReserve_gemma3(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"gemma3","vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_hidden_layers":27,"patch_size":14,"image_size":896,"num_channels":3}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	mk := func(quant, backend string) config {
		c := config{visionPath: dir, visionQuant: quant}
		c.load.Backend = backend
		return c
	}
	if got := towerReserve(mk("", "cuda"), dir); got != 0 {
		t.Errorf("default (int8) reserve = %d, want 0", got)
	}
	if got := towerReserve(mk("int8", "cuda"), dir); got != 0 {
		t.Errorf("explicit int8 reserve = %d, want 0", got)
	}
	got := float64(towerReserve(mk("f32", "cuda"), dir)) / 1e9
	t.Logf("gemma3 f32 tower reserve: %.2f GB", got)
	if got < 1.9 || got > 2.5 {
		t.Errorf("explicit f32 reserve %.2f GB, want 1.9-2.5", got)
	}
	if got := towerReserve(mk("f32", "metal"), dir); got != 0 {
		t.Errorf("a non-cuda backend reserves %d, want 0", got)
	}
}
