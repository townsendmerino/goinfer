package serveapp

import (
	"os"
	"path/filepath"
	"testing"
)

// The S4 addendum (docs/tasks/task-multimodal-support-2026-10.md): -vision-quant is unset by default so an explicit f32 is distinguishable from the default (which matters for the
// fallback, attachGemma3Tower). Unset is float32 on CUDA since 2026-10-08 (owner) and int8 on WebGPU, which has no float32 device tower; an explicit f32 selects the float32 tower;
// int8 stays int8 everywhere.
func TestTowerInt8_gemma3ExplicitF32(t *testing.T) {
	cases := []struct {
		quant, backend string
		want           bool
	}{
		{"", "cuda", false}, {"", "webgpu", true}, {"", "metal", false}, {"", "cpu", false}, {"", "", false}, // unset: float32 on cuda (the owner's 2026-10-08 default), int8 on webgpu
		{"f32", "cuda", false}, {"f32", "webgpu", false}, {"f32", "metal", false}, {"f32", "cpu", false}, // explicit f32: float32 everywhere
		{"int8", "cuda", true}, {"int8", "webgpu", true}, {"int8", "metal", true}, {"int8", "cpu", true}, // int8 when asked
	}
	for _, c := range cases {
		if got := towerInt8("gemma3", c.quant, c.backend); got != c.want {
			t.Errorf("towerInt8(gemma3, %q, %q) = %v, want %v", c.quant, c.backend, got, c.want)
		}
	}
}

// Gemma 3's SigLIP tower is priced into the resident plan on CUDA: the default (float32 since 2026-10-08) and an explicit f32 reserve about 2 GB (27 layers of hidden 1152 / intermediate
// 4304 in float32 plus the 4096-patch scratch; measured at about 1.7 GiB, so the figure is generous), an explicit int8 about 558 MiB.
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
	// S18: the int8 device tower (an explicit int8, and the fallback) is priced too, at about the 558 MiB it holds
	if got := float64(towerReserve(mk("int8", "cuda"), dir)) / (1 << 20); got < 520 || got > 640 {
		t.Errorf("int8 reserve = %.0f MiB, want 520-640", got)
	}
	// the default is float32 now, and prices as an explicit f32 does
	for _, q := range []string{"", "f32"} {
		got := float64(towerReserve(mk(q, "cuda"), dir)) / 1e9
		t.Logf("gemma3 float32 tower reserve (quant %q): %.2f GB", q, got)
		if got < 1.9 || got > 2.5 {
			t.Errorf("float32 reserve (quant %q) %.2f GB, want 1.9-2.5", q, got)
		}
	}
	if got := towerReserve(mk("f32", "metal"), dir); got != 0 {
		t.Errorf("a non-cuda backend reserves %d, want 0", got)
	}
}

// TestTowerReserve_everyDeviceTower is S18's G-S18c (docs/tasks/task-multimodal-support-2026-10.md): towerReserve is nonzero for every family that has a DEVICE tower under its backend, int8 included, and
// zero where the tower runs on the CPU. A table over family x -vision-quant x backend. Red on the code before S18 for Gemma 3 under cuda, where the shipped default (the int8 device tower) was priced at zero.
func TestTowerReserve_everyDeviceTower(t *testing.T) {
	old := cudaTowerRegistered // stand in for a cuda build, whose blank import fills the tower registries
	cudaTowerRegistered = func(string) bool { return true }
	t.Cleanup(func() { cudaTowerRegistered = old })
	cfgs := map[string]string{
		"gemma3":     `{"model_type":"gemma3","vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_hidden_layers":27,"patch_size":14,"image_size":896,"num_channels":3}}`,
		"gemma4":     `{"model_type":"gemma4","vision_config":{"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"patch_size":16,"position_embedding_size":10240,"default_output_length":280,"pooling_kernel_size":3}}`,
		"qwen3_5":    `{"model_type":"qwen3_5","vision_config":{"depth":12,"hidden_size":768,"in_channels":3,"intermediate_size":3072,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}`,
		"glm_ocr":    `{"model_type":"glm_ocr","vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`,
		"qwen2_5_vl": `{"model_type":"qwen2_5_vl","vision_config":{"depth":32,"hidden_size":1280,"in_channels":3,"intermediate_size":3420,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`,
	}
	for fam, js := range cfgs {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(js), 0o644); err != nil {
			t.Fatal(err)
		}
		for _, quant := range []string{"", "f32", "int8"} {
			for _, backend := range []string{"cuda", "metal", "webgpu", "cpu"} {
				c := config{visionPath: dir, visionQuant: quant}
				c.load.Backend = backend
				got := towerReserve(c, dir)
				// a device tower on CUDA: every family in float32; Gemma 3 also as the int8 device tower (the shipped default and -vision-quant int8)
				wantDevice := backend == "cuda" && (towerInt8(fam, quant, "cuda") == false || fam == "gemma3")
				if wantDevice && got <= 0 {
					t.Errorf("%s quant %q on %s has a device tower but towerReserve = %d", fam, quant, backend, got)
				}
				if !wantDevice && got != 0 {
					t.Errorf("%s quant %q on %s has no device tower but towerReserve = %d", fam, quant, backend, got)
				}
			}
		}
	}
	// the int8 figure is calibrated against the 558 MiB measured on the card
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgs["gemma3"]), 0o644)
	c := config{visionPath: dir, visionQuant: "int8"} // explicit: the default is float32 now
	c.load.Backend = "cuda"
	if mib := float64(towerReserve(c, dir)) / (1 << 20); mib < 520 || mib > 640 {
		t.Errorf("gemma3 int8 reserve %.0f MiB, want 520-640 (measured 558)", mib)
	}
}
