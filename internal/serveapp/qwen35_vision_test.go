package serveapp

import (
	"os"
	"path/filepath"
	"testing"
)

func writeQwen35Dir(t *testing.T, config, pre string) string {
	t.Helper()
	d := t.TempDir()
	if config != "" {
		if err := os.WriteFile(filepath.Join(d, "config.json"), []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if pre != "" {
		if err := os.WriteFile(filepath.Join(d, "preprocessor_config.json"), []byte(pre), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return d
}

const (
	q35Pre  = `{"size":{"longest_edge":16777216,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`
	q35Conf = `{"model_type":"qwen3_5","vision_config":{"depth":12,"deepstack_visual_indexes":[]}}`
)

// TestIsQwen35VisionDir: auto-discovery must recognise a Qwen3.5 checkpoint that carries a usable
// tower and stay SILENT (false, not an error) on everything else, so a text-only copy or a
// Qwen3-VL DeepStack checkpoint never fails serve's startup.
func TestIsQwen35VisionDir(t *testing.T) {
	for _, tc := range []struct {
		name        string
		config, pre string
		want        bool
	}{
		{"qwen3_5 with tower and preprocessor", q35Conf, q35Pre, true},
		{"qwen3_5_moe", `{"model_type":"qwen3_5_moe","vision_config":{"depth":27,"deepstack_visual_indexes":[]}}`, q35Pre, true},
		{"text-only copy: no vision_config", `{"model_type":"qwen3_5"}`, q35Pre, false},
		{"text-only copy: no preprocessor_config.json", q35Conf, "", false},
		{"Qwen2.5-VL-style preprocessor (min_pixels keys) is not a Qwen3.5 tower", q35Conf, `{"min_pixels":3136,"max_pixels":100,"patch_size":14,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`, false},
		{"DeepStack tower (Qwen3-VL proper) is refused", `{"model_type":"qwen3_5","vision_config":{"depth":27,"deepstack_visual_indexes":[8,16,24]}}`, q35Pre, false},
		{"another family", `{"model_type":"qwen2_5_vl","vision_config":{"depth":32}}`, q35Pre, false},
		{"no config.json", "", q35Pre, false},
	} {
		if got := isQwen35VisionDir(writeQwen35Dir(t, tc.config, tc.pre)); got != tc.want {
			t.Errorf("%s: isQwen35VisionDir = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestSetupQwen35Vision: the cap lowers max_pixels to exactly 1024 merged tokens, leaves the
// checkpoint's own min alone, and does NOT read the tower's weights at startup — the directory has
// none, so any load would fail — and the tower's eventual load error is reported, not swallowed.
func TestSetupQwen35Vision(t *testing.T) {
	dir := writeQwen35Dir(t, q35Conf, q35Pre)
	tower, pp, err := setupQwen35Vision(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if want := 1024 * 32 * 32; pp.MaxPixels != want {
		t.Errorf("MaxPixels %d, want the 1024-token cap %d (checkpoint says 16777216)", pp.MaxPixels, want)
	}
	if pp.MinPixels != 65536 || pp.PatchSize != 16 || pp.MergeSize != 2 || !pp.FusedNormalize {
		t.Errorf("checkpoint values not carried through: %+v", pp)
	}
	if tower.enc != nil || tower.err != nil {
		t.Fatal("setup touched the tower's weights: it must load lazily")
	}
	if _, err := tower.encoder(); err == nil {
		t.Error("encoder() on a directory with no weights returned no error")
	}
	if _, err2 := tower.encoder(); err2 == nil || err2.Error() != tower.err.Error() {
		t.Error("a failed load must be remembered, not retried into a different answer")
	}
	// A cap must never RAISE a smaller checkpoint budget.
	small := writeQwen35Dir(t, q35Conf, `{"size":{"longest_edge":262144,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`)
	if _, pp2, err := setupQwen35Vision(small, false); err != nil || pp2.MaxPixels != 262144 {
		t.Errorf("a checkpoint budget below the cap must be kept: %+v %v", pp2, err)
	}
	if _, _, err := setupQwen35Vision(writeQwen35Dir(t, `{"model_type":"qwen3_5"}`, q35Pre), false); err == nil {
		t.Error("a text-only directory was accepted as a vision tower")
	}
}
