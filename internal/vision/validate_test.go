package vision

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEncoderConfig_validate (H8): a config whose dims would ÷0 or mis-partition
// is rejected at load, not left to panic in e.grid / headDim / Forward.
func TestEncoderConfig_validate(t *testing.T) {
	good := EncoderConfig{
		HiddenSize: 32, IntermediateSize: 64, NumHiddenLayers: 2,
		NumAttentionHeads: 4, NumChannels: 3, ImageSize: 32, PatchSize: 8,
	}
	if err := good.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	bad := map[string]func(*EncoderConfig){
		"hidden 0":          func(c *EncoderConfig) { c.HiddenSize = 0 },
		"inter 0":           func(c *EncoderConfig) { c.IntermediateSize = 0 },
		"heads 0":           func(c *EncoderConfig) { c.NumAttentionHeads = 0 },
		"hidden%heads":      func(c *EncoderConfig) { c.NumAttentionHeads = 5 }, // 32%5≠0
		"patch 0":           func(c *EncoderConfig) { c.PatchSize = 0 },
		"image 0":           func(c *EncoderConfig) { c.ImageSize = 0 },
		"image%patch":       func(c *EncoderConfig) { c.ImageSize = 30 }, // 30%8≠0
		"channels negative": func(c *EncoderConfig) { c.NumChannels = -1 },
		"layers negative":   func(c *EncoderConfig) { c.NumHiddenLayers = -1 },
	}
	for name, mutate := range bad {
		c := good
		mutate(&c)
		if err := c.validate(); err == nil {
			t.Errorf("%s: validate accepted an invalid config", name)
		}
	}
}

// TestQwenEncoderConfig_validate (H8): same for the Qwen tower's config.
func TestQwenEncoderConfig_validate(t *testing.T) {
	good := QwenEncoderConfig{
		Depth: 2, HiddenSize: 32, IntermediateSize: 64, NumHeads: 2, InChans: 3,
		PatchSize: 14, SpatialMergeSize: 2, TemporalPatchSize: 2, OutHiddenSize: 64,
		WindowSize: 112,
	}
	if err := good.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	bad := map[string]func(*QwenEncoderConfig){
		"hidden 0":         func(c *QwenEncoderConfig) { c.HiddenSize = 0 },
		"heads 0":          func(c *QwenEncoderConfig) { c.NumHeads = 0 },
		"hidden%heads":     func(c *QwenEncoderConfig) { c.NumHeads = 5 }, // 32%5≠0
		"merge 0":          func(c *QwenEncoderConfig) { c.SpatialMergeSize = 0 },
		"patch 0":          func(c *QwenEncoderConfig) { c.PatchSize = 0 },
		"out 0":            func(c *QwenEncoderConfig) { c.OutHiddenSize = 0 },
		"window too small": func(c *QwenEncoderConfig) { c.WindowSize = 10 }, // < merge*patch=28 → vmws 0
		"headDim%4":        func(c *QwenEncoderConfig) { c.NumHeads = 16 },   // 32/16=2, 2%4≠0 (rotary ÷0)
		"temporal 0":       func(c *QwenEncoderConfig) { c.TemporalPatchSize = 0 },
		"bad act":          func(c *QwenEncoderConfig) { c.HiddenAct = "gelu" }, // mlp hardcodes silu
	}
	for name, mutate := range bad {
		c := good
		mutate(&c)
		if err := c.validate(); err == nil {
			t.Errorf("%s: validate accepted an invalid config", name)
		}
	}
}

// writeMiniSafetensors writes a one-tensor F32 safetensors file (name→shape).
func writeMiniSafetensors(t *testing.T, path, name string, shape []int, nElem int) {
	t.Helper()
	raw := make([]byte, 4*nElem) // zeros; values don't matter for a shape check
	header := map[string]any{
		name: map[string]any{"dtype": "F32", "shape": shape, "data_offsets": []int{0, len(raw)}},
	}
	hb, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8, 8+len(hb)+len(raw))
	binary.LittleEndian.PutUint64(out, uint64(len(hb)))
	out = append(out, hb...)
	out = append(out, raw...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestLoadEncoder_shapeMismatchIsError (H7): with a valid config but a
// mis-shaped patch_embedding tensor, LoadEncoder returns an error rather than
// panicking deep in the loader. (The full parity path needs testdata/siglip-tiny.)
func TestLoadEncoder_shapeMismatchIsError(t *testing.T) {
	dir := t.TempDir()
	cfg := EncoderConfig{
		HiddenSize: 32, IntermediateSize: 64, NumHiddenLayers: 1,
		NumAttentionHeads: 4, NumChannels: 3, ImageSize: 32, PatchSize: 8,
	}
	cb, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cb, 0o644); err != nil {
		t.Fatal(err)
	}
	// patch_embedding.weight should be [hidden, C, P, P] = [32,3,8,8]; write [1]
	// instead. The loader reads it first, so this exercises the H7 shape check.
	writeMiniSafetensors(t, filepath.Join(dir, "model.safetensors"),
		"embeddings.patch_embedding.weight", []int{1}, 1)

	_, err := LoadEncoder(dir, false)
	if err == nil {
		t.Fatal("LoadEncoder with a mis-shaped patch_embedding: got nil error, want a shape error")
	}
	if !strings.Contains(err.Error(), "shape") {
		t.Errorf("error %q does not mention shape", err)
	}
}
