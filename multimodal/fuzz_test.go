package multimodal

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// An uploaded WAV is attacker-supplied, and its length decides whether a request is refused before it is decoded, so Samples must be exact on any
// bytes: a header that under-reports would let a clip past its limit.
func FuzzParseWAV(f *testing.F) {
	for _, rate := range []int{4000, 16000, 44100, 383999} {
		for _, ch := range []int{1, 2, 8} {
			wav := wavBytes(make([]float32, 8*ch), ch, rate, false)
			f.Add(wav)
			f.Add(wav[:len(wav)-1])
		}
	}
	f.Add(wavBytes(make([]float32, 16), 2, 48000, true))
	f.Add([]byte("RIFF"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		w, err := ParseWAV(data)
		samples, derr := DecodeWAVAnyRate(data)
		if (err == nil) != (derr == nil) {
			t.Fatalf("ParseWAV says %v, DecodeWAVAnyRate says %v", err, derr)
		}
		_, _ = DecodeWAV(data)
		if err != nil {
			return
		}
		if w.Samples() != len(samples) {
			t.Fatalf("Samples %d, decoded %d", w.Samples(), len(samples))
		}
	})
}

// A preprocessor_config.json ships in a model directory, and what it accepts sizes the allocations of every image the model is sent.
// Both Qwen loaders refuse the sizes no released config has.
func TestQwenPreprocessConfigs_refuseSizesNoRealModelHas(t *testing.T) {
	type sizes struct{ shortest, longest, patch, merge, temporal int64 }
	const norm = `"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]`
	qwen3 := func(s sizes) error {
		_, err := ParseQwen3PreprocessConfig([]byte(fmt.Sprintf(`{"size":{"shortest_edge":%d,"longest_edge":%d},"patch_size":%d,"merge_size":%d,"temporal_patch_size":%d,%s}`,
			s.shortest, s.longest, s.patch, s.merge, s.temporal, norm)))
		return err
	}
	qwen25 := func(s sizes) error {
		dir := t.TempDir()
		raw := fmt.Sprintf(`{"min_pixels":%d,"max_pixels":%d,"patch_size":%d,"merge_size":%d,"temporal_patch_size":%d,%s}`,
			s.shortest, s.longest, s.patch, s.merge, s.temporal, norm)
		if err := os.WriteFile(filepath.Join(dir, "preprocessor_config.json"), []byte(raw), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := LoadQwenPreprocessConfig(dir)
		return err
	}
	for name, s := range map[string]sizes{
		"Qwen3-VL, Qwen3.5 and Qwen3.6": {65536, 16777216, 16, 2, 2},
		"Qwen2.5-VL":                    {3136, 12845056, 14, 2, 2},
		"GLM-OCR":                       {12544, 9633792, 14, 2, 2},
	} {
		if err := qwen3(s); err != nil {
			t.Errorf("%s's sizes: the Qwen3 loader refused them: %v", name, err)
		}
		if err := qwen25(s); err != nil {
			t.Errorf("%s's sizes: the Qwen2.5-VL loader refused them: %v", name, err)
		}
	}
	if _, err := LoadQwenPreprocessConfig(t.TempDir()); err != nil {
		t.Errorf("the Qwen2.5-VL defaults, for a directory without the file, were refused: %v", err)
	}
	for name, s := range map[string]sizes{
		"temporal patch sizes an output no slice can hold": {65536, 1 << 24, 16, 2, 1 << 50},
		"temporal patch past the released 2":               {65536, 1 << 24, 16, 2, 4},
		"patch times merge wraps to zero":                  {65536, 1 << 24, 1 << 32, 1 << 32, 2},
		"patch alone past the limit":                       {65536, 1 << 24, 1 << 15, 2, 2},
		"patch times merge past the limit, each within it": {65536, 1 << 24, 64, 32, 2},
		"budget past the largest released":                 {65536, 1<<24 + 1, 16, 2, 2},
		"minimum above the maximum":                        {1 << 24, 1 << 16, 16, 2, 2},
	} {
		if qwen3(s) == nil {
			t.Errorf("%s: the Qwen3 loader accepted it", name)
		}
		if qwen25(s) == nil {
			t.Errorf("%s: the Qwen2.5-VL loader accepted it", name)
		}
	}
}

// A request's image is untrusted. smart_resize floors a short side to one cell without shrinking the long side, so a thin image
// resizes to many times the budget: under the Qwen3.5 config, 1 x 1048576 pixels becomes a 32 x 4194304 grid, gigabytes of
// pixel_values. QwenPreprocess refuses it where HF's processor does, past an aspect ratio of 200.
func TestQwenPreprocess_refusesTheAspectRatiosHFRefuses(t *testing.T) {
	cfg := Qwen35FamilyPreprocessConfig()
	thin := func(w int) []byte {
		var b bytes.Buffer
		if err := png.Encode(&b, image.NewGray(image.Rect(0, 0, w, 1))); err != nil {
			t.Fatal(err)
		}
		return b.Bytes()
	}
	if _, _, err := QwenPreprocess(thin(qwenMaxAspect), cfg); err != nil {
		t.Errorf("1 x %d, at the limit, was refused: %v", qwenMaxAspect, err)
	}
	if _, _, err := QwenPreprocess(thin(qwenMaxAspect+1), cfg); err == nil {
		t.Errorf("1 x %d, past the limit, was accepted", qwenMaxAspect+1)
	}
	// The sizes themselves, by arithmetic: preprocessing either would allocate gigabytes if the refusal were missing.
	if _, _, err := qwenResizeGrid(1, 1<<20, cfg); err == nil {
		t.Error("1 x 1048576 under the Qwen3.5 config was accepted")
	}
	// At the aspect limit, a config with the largest cell the loaders accept still resizes past the released budget.
	big := cfg
	big.PatchSize, big.MergeSize = 32, 32
	if _, _, err := qwenResizeGrid(300, 60000, big); err == nil {
		t.Error("300 x 60000 under a 1024-pixel cell was accepted")
	}
}

// Whatever ParseQwen3PreprocessConfig accepts must resize any image the input limit admits to no more pixels and frames than the
// Qwen3.5 config's largest image, and must preprocess a small image into the grid it reports, or fail with an error.
func FuzzParseQwen3PreprocessConfig(f *testing.F) {
	shipped := Qwen35FamilyPreprocessConfig()
	most := int64(shipped.MaxPixels) * int64(shipped.TemporalPatchSize)
	cfg := func(shortest, longest, patch, merge, temporal int64) string {
		return fmt.Sprintf(`{"size":{"shortest_edge":%d,"longest_edge":%d},"patch_size":%d,"merge_size":%d,"temporal_patch_size":%d,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`,
			shortest, longest, patch, merge, temporal)
	}
	configs := []string{
		qwen35FamilyPreprocessorJSON,
		cfg(65536, 1<<24, 32, 32, 2), // the largest cell the limits accept
		cfg(65536, 1<<24, 16, 2, 4),  // past the released temporal patch
		cfg(65536, 1<<25, 16, 2, 2),  // past the released budget
		`{"size":{"shortest_edge":1,"longest_edge":1},"patch_size":1,"merge_size":1,"temporal_patch_size":1,"image_mean":[0,0,0],"image_std":[1,1,1]}`,
		`{"size":{"shortest_edge":1,"longest_edge":1},"patch_size":9223372036854775807,"merge_size":9223372036854775807,"temporal_patch_size":1,"image_mean":[0,0,0],"image_std":[1,1,1]}`,
		`{"size":{"shortest_edge":2147483647,"longest_edge":1},"patch_size":3,"merge_size":5,"temporal_patch_size":7,"image_mean":[1e30,0,0],"image_std":[1e-30,1,1]}`,
		`{"image_mean":[0],"image_std":[0,0,0]}`, `{}`, `[`,
	}
	for _, c := range configs {
		for _, hw := range [][2]uint32{{9, 16}, {1, 1 << 20}, {1 << 25, 1}, {300, 60000}, {5792, 5792}} {
			f.Add([]byte(c), hw[0], hw[1])
		}
	}
	var img bytes.Buffer
	if err := png.Encode(&img, image.NewNRGBA(image.Rect(0, 0, 16, 9))); err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, raw []byte, h, w uint32) {
		cfg, err := ParseQwen3PreprocessConfig(raw)
		if err != nil {
			return
		}
		// The bound, by arithmetic on the grid QwenPreprocess would allocate for: preprocessing an image at a legal maximum takes seconds.
		if h > 0 && w > 0 && uint64(h)*uint64(w) <= qwenMaxInputPixels {
			if hb, wb, err := qwenResizeGrid(int(h), int(w), cfg); err == nil {
				factor := cfg.PatchSize * cfg.MergeSize
				if hb <= 0 || wb <= 0 || hb%factor != 0 || wb%factor != 0 {
					t.Fatalf("%dx%d resizes to %dx%d, not whole %d-pixel cells", w, h, wb, hb, factor)
				}
				if got := int64(hb) * int64(wb) * int64(cfg.TemporalPatchSize); got > most {
					t.Fatalf("%dx%d resizes to %dx%d over %d frames, %d pixels, past the %d of a released config's largest image", w, h, wb, hb,
						cfg.TemporalPatchSize, got, most)
				}
			}
		}
		// The grid it reports, on a small image with the budget capped for the same reason.
		cfg.MaxPixels, cfg.MinPixels = min(cfg.MaxPixels, 1<<14), min(cfg.MinPixels, 1<<14)
		out, grid, err := QwenPreprocess(img.Bytes(), cfg)
		if err != nil {
			return
		}
		if want := grid[0] * grid[1] * grid[2] * 3 * cfg.TemporalPatchSize * cfg.PatchSize * cfg.PatchSize; len(out) != want {
			t.Fatalf("%d values for grid %v, want %d", len(out), grid, want)
		}
	})
}
