package serveapp

import (
	"os"
	"path/filepath"
	"testing"
)

// towerVRAMEstimate prices a CUDA vision tower from its checkpoint's vision_config (S4 Gate 0 step 4): float32 weights plus peak scratch at the family's largest image.
// The three real configs below are the ones in ~/models (Gemma 4 E2B, Qwen3.5-0.8B, GLM-OCR), and the bounds are the Gate 0 arithmetic (weights ~0.67 / 0.35 / 1.62 GB).
func TestTowerVRAMEstimate(t *testing.T) {
	cases := []struct {
		name, mt, cfg string
		maxPixels     int
		lo, hi        float64 // GB
	}{
		{"gemma4 E2B", "gemma4", `{"vision_config":{"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"patch_size":16,"position_embedding_size":10240,"default_output_length":280,"pooling_kernel_size":3}}`, 0, 0.80, 0.95},
		{"qwen3.5-0.8b", "qwen3_5", `{"vision_config":{"depth":12,"hidden_size":768,"in_channels":3,"intermediate_size":3072,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 0.60, 0.75},
		{"glm-ocr, ceiling capped to 1.5 MP for the reserve", "glm_ocr", `{"vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 2.3, 2.7},
		{"qwen2.5-vl-3b (measured 2758 MiB of weights + 270 MiB of scratch per 4096 patches, at the 8192-patch ceiling: 3298 MiB = 3.46 GB)", "qwen2_5_vl", `{"vision_config":{"depth":32,"hidden_size":1280,"in_channels":3,"intermediate_size":3420,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 3.40, 3.60},
		{"qwen3-vl-2b (S10 on CUDA: 1158 MiB of weights resident after an image, the scratch released after each call; this is the peak, 1.21 GB of weights + 0.43 GB of scratch at the 1024-token cap)", "qwen3_vl", `{"vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 1.55, 1.75},
		{"glm-ocr capped at 1 MP", "glm_ocr", `{"vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, 1_000_000, 1.8, 2.2},
	}
	for _, c := range cases {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(c.cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		d, ok := readTowerDims(dir)
		if !ok {
			t.Fatalf("%s: the config did not parse into tower dims", c.name)
		}
		got := float64(towerVRAMEstimate(c.mt, d, c.maxPixels)) / 1e9
		t.Logf("%s: %.2f GB", c.name, got)
		if got < c.lo || got > c.hi {
			t.Errorf("%s: estimate %.2f GB, want %.2f-%.2f", c.name, got, c.lo, c.hi)
		}
	}
	if towerVRAMEstimate("llama", towerDims{hidden: 1, inter: 1, layers: 1, patch: 1}, 0) != 0 {
		t.Error("an unknown model type has no tower to price")
	}
}

// towerReserve is zero unless the tower will run on CUDA: any other backend, -vision-device cpu, an int8 tower, a GGUF, a checkpoint with no vision_config, or a binary that
// registers no CUDA tower for the family (this package's test binary imports none).
func TestTowerReserve_zeroCases(t *testing.T) {
	dir := t.TempDir()
	cfgJSON := `{"model_type":"gemma4","vision_config":{"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"patch_size":16,"position_embedding_size":10240,"default_output_length":280,"pooling_kernel_size":3}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	cuda := config{visionPath: dir}
	cuda.load.Backend = "cuda"
	cpuDev := cuda
	cpuDev.visionDevice = "cpu"
	metal := cuda
	metal.load.Backend = "metal"
	for name, c := range map[string]config{"cuda but no CUDA tower registered in this binary": cuda, "-vision-device cpu": cpuDev, "metal": metal} {
		if got := towerReserve(c, dir); got != 0 {
			t.Errorf("%s: reserve %d, want 0", name, got)
		}
	}
	if got := towerReserve(cuda, filepath.Join(dir, "model.gguf")); got != 0 {
		t.Errorf("an unresolved path: reserve %d, want 0", got)
	}
}
