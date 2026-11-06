package multimodal

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
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
func TestParseQwen3PreprocessConfig_refusesSizesNoRealModelHas(t *testing.T) {
	const good = `"patch_size":16,"merge_size":2,"temporal_patch_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]`
	config := func(shortest, longest, patch, merge, temporal int64) []byte {
		return []byte(fmt.Sprintf(`{"size":{"shortest_edge":%d,"longest_edge":%d},"patch_size":%d,"merge_size":%d,"temporal_patch_size":%d,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`,
			shortest, longest, patch, merge, temporal))
	}
	if _, err := ParseQwen3PreprocessConfig([]byte(`{"size":{"shortest_edge":65536,"longest_edge":16777216},` + good + `}`)); err != nil {
		t.Fatalf("a real config was refused: %v", err)
	}
	if _, err := ParseQwen3PreprocessConfig(config(65536, 16777216, 16, 2, 2)); err != nil {
		t.Fatalf("this test's own config is not a real one: %v", err)
	}
	for name, raw := range map[string][]byte{
		"temporal patch sizes an output no slice can hold": config(65536, 16777216, 16, 2, 1<<50),
		"patch times merge wraps to zero":                  config(65536, 16777216, 1<<32, 1<<32, 2),
		"patch times merge floors every image huge":        config(65536, 16777216, 1<<15, 1<<15, 2),
		"longest edge above the input limit":               config(65536, qwenMaxInputPixels+1, 16, 2, 2),
		"shortest edge above the longest":                  config(1<<24, 1<<16, 16, 2, 2),
	} {
		if _, err := ParseQwen3PreprocessConfig(raw); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// Whatever ParseQwen3PreprocessConfig accepts must preprocess a small image into the grid it reports, or fail with an error. The target
// pixels are capped because a legal maximum resizes for seconds, and the limits themselves are the test above's.
func FuzzParseQwen3PreprocessConfig(f *testing.F) {
	for _, s := range []string{
		`{"size":{"shortest_edge":65536,"longest_edge":16777216},"patch_size":16,"merge_size":2,"temporal_patch_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`,
		`{"size":{"shortest_edge":1,"longest_edge":1},"patch_size":1,"merge_size":1,"temporal_patch_size":1,"image_mean":[0,0,0],"image_std":[1,1,1]}`,
		`{"size":{"shortest_edge":1,"longest_edge":1},"patch_size":9223372036854775807,"merge_size":9223372036854775807,"temporal_patch_size":1,"image_mean":[0,0,0],"image_std":[1,1,1]}`,
		`{"size":{"shortest_edge":2147483647,"longest_edge":1},"patch_size":3,"merge_size":5,"temporal_patch_size":7,"image_mean":[1e30,0,0],"image_std":[1e-30,1,1]}`,
		`{"image_mean":[0],"image_std":[0,0,0]}`, `{}`, `[`,
	} {
		f.Add([]byte(s))
	}
	var img bytes.Buffer
	if err := png.Encode(&img, image.NewNRGBA(image.Rect(0, 0, 16, 9))); err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		cfg, err := ParseQwen3PreprocessConfig(raw)
		if err != nil {
			return
		}
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
