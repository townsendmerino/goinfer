package multimodal

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// G0b of P8a (docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md): for a grid-aligned image
// pixel_values and grid_thw are BIT-EXACT to the numpy transcription of HF's Qwen2VLImageProcessor
// (fused normalize + patchify, scripts/qwen35vl_images.py), and the loader reads the real Qwen3.5
// budget from size.*.
func TestQwen3Preprocess_bitExact(t *testing.T) {
	raw, err := os.ReadFile("../testdata/qwen35vl_preprocess_golden.json")
	if err != nil {
		t.Skipf("no golden — run scripts/pin_qwen35_vl_preprocess.py (%v)", err)
	}
	img, err := os.ReadFile("../testdata/qwen35vl_preprocess_image.png")
	if err != nil {
		t.Skipf("no image (%v)", err)
	}
	var g struct {
		PatchSize   int        `json:"patch_size"`
		MergeSize   int        `json:"merge_size"`
		TemporalPS  int        `json:"temporal_patch_size"`
		Mean        [3]float32 `json:"image_mean"`
		Std         [3]float32 `json:"image_std"`
		GridTHW     [][3]int   `json:"grid_thw"`
		PixelValues []float32  `json:"pixel_values"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	cfg := QwenPreprocessConfig{
		PatchSize: g.PatchSize, MergeSize: g.MergeSize, TemporalPatchSize: g.TemporalPS,
		MinPixels: 1024, MaxPixels: 1 << 24, // 64x96 is grid-aligned and inside the budget: resize is a no-op
		Mean: g.Mean, Std: g.Std, FusedNormalize: true,
	}
	pv, grid, err := QwenPreprocess(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if grid != g.GridTHW[0] {
		t.Fatalf("grid %v, HF %v", grid, g.GridTHW[0])
	}
	if len(pv) != len(g.PixelValues) {
		t.Fatalf("len %d, HF %d", len(pv), len(g.PixelValues))
	}
	bad := 0
	for i := range pv {
		if math.Float32bits(pv[i]) != math.Float32bits(g.PixelValues[i]) {
			if bad < 3 {
				t.Errorf("pixel_values[%d] = %v (bits %08x), HF %v (bits %08x)", i, pv[i], math.Float32bits(pv[i]), g.PixelValues[i], math.Float32bits(g.PixelValues[i]))
			}
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d values not bit-identical", bad, len(pv))
	}

	// The unfused arithmetic is NOT what HF's torchvision backend does; prove the flag matters, so the
	// bit-exact claim above is a property of the fused form and not of an input where both agree.
	cfg.FusedNormalize = false
	pv2, _, _ := QwenPreprocess(img, cfg)
	diff := 0
	for i := range pv2 {
		if math.Float32bits(pv2[i]) != math.Float32bits(g.PixelValues[i]) {
			diff++
		}
	}
	if diff == 0 {
		t.Error("unfused normalize also matches bit-for-bit on this image: the fused/unfused distinction is untested by it")
	}
	t.Logf("fused: %d/%d bit-identical; unfused differs in %d values", len(pv), len(pv), diff)
}

func TestLoadQwen3PreprocessConfig(t *testing.T) {
	write := func(body string) string {
		d := t.TempDir()
		if err := os.WriteFile(filepath.Join(d, "preprocessor_config.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return d
	}
	const good = `{"size":{"longest_edge":16777216,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`
	c, err := LoadQwen3PreprocessConfig(write(good))
	if err != nil {
		t.Fatal(err)
	}
	if c.MinPixels != 65536 || c.MaxPixels != 16777216 || c.PatchSize != 16 || c.MergeSize != 2 || c.TemporalPatchSize != 2 ||
		c.Mean != [3]float32{.5, .5, .5} || c.Std != [3]float32{.5, .5, .5} || !c.FusedNormalize {
		t.Errorf("Qwen3.5 config misread: %+v", c)
	}
	// The trap this loader exists for: the Qwen2.5-VL loader on the SAME file keeps its own budget.
	old, err := LoadQwenPreprocessConfig(write(good))
	if err != nil {
		t.Fatal(err)
	}
	if old.MinPixels == 65536 || old.MaxPixels == 16777216 {
		t.Errorf("Qwen2.5-VL loader unexpectedly read size.*: %+v — the trap note in the loader's doc is stale", old)
	}
	for name, body := range map[string]string{
		"min_pixels keys only (Qwen2.5-VL style)": `{"min_pixels":3136,"max_pixels":12845056,"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[.5,.5,.5],"image_std":[.5,.5,.5]}`,
		"no patch_size": `{"size":{"longest_edge":16777216,"shortest_edge":65536},"temporal_patch_size":2,"merge_size":2,"image_mean":[.5,.5,.5],"image_std":[.5,.5,.5]}`,
		"no mean":       `{"size":{"longest_edge":16777216,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_std":[.5,.5,.5]}`,
		"zero std":      `{"size":{"longest_edge":16777216,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[.5,.5,.5],"image_std":[.5,0,.5]}`,
	} {
		if _, err := LoadQwen3PreprocessConfig(write(body)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := LoadQwen3PreprocessConfig(t.TempDir()); err == nil {
		t.Error("missing preprocessor_config.json accepted")
	}
}

// TestLoadQwen3PreprocessConfig_real08B asserts the three released configs' numbers on the real
// 0.8B dir when present: 65536 / 16777216 / 16 / 2.
func TestLoadQwen3PreprocessConfig_real08B(t *testing.T) {
	dir := filepath.Join(os.Getenv("HOME"), "models", "qwen3.5-0.8b")
	if _, err := os.Stat(filepath.Join(dir, "preprocessor_config.json")); err != nil {
		t.Skipf("no real 0.8B at %s", dir)
	}
	c, err := LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if c.MinPixels != 65536 || c.MaxPixels != 16777216 || c.PatchSize != 16 || c.MergeSize != 2 || c.TemporalPatchSize != 2 {
		t.Errorf("real 0.8B config: %+v", c)
	}
}
