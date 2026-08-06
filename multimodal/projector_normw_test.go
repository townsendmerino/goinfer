package multimodal

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSafetensorsF32 writes a minimal safetensors file with the given F32 tensors (row-major,
// 1-D by len). Enough for LoadProjector, which only reads two named tensors as F32.
func writeSafetensorsF32(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()
	type entry struct {
		Dtype   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	// deterministic order not required by the format; TensorF32 looks up by name.
	for name, vals := range tensors {
		start := len(data)
		buf := make([]byte, len(vals)*4)
		for i, v := range vals {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		data = append(data, buf...)
		header[name] = entry{Dtype: "F32", Shape: []int{len(vals)}, Offsets: []int{start, len(data)}}
	}
	hb, _ := json.Marshal(header)
	var out []byte
	out = binary.LittleEndian.AppendUint64(out, uint64(len(hb)))
	out = append(out, hb...)
	out = append(out, data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestLoadProjector_normWLength_C23 gates C-23: LoadProjector must reject a checkpoint whose
// mm_soft_emb_norm.weight is shorter than vision_hidden, rather than load cleanly and panic
// "index out of range" inside Forward on the first image request. The projection weight is kept
// valid so the failure is isolated to the norm tensor.
func TestLoadProjector_normWLength_C23(t *testing.T) {
	const vh, th = 8, 4
	cfg := `{"mm_tokens_per_image":4,"text_config":{"hidden_size":4},` +
		`"vision_config":{"hidden_size":8,"image_size":4,"patch_size":2,"layer_norm_eps":1e-6}}`
	proj := make([]float32, vh*th) // correct length

	mk := func(normLen int) error {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		writeSafetensorsF32(t, filepath.Join(dir, "model.safetensors"), map[string][]float32{
			"multi_modal_projector.mm_soft_emb_norm.weight":    make([]float32, normLen),
			"multi_modal_projector.mm_input_projection_weight": proj,
		})
		_, err := LoadProjector(dir)
		return err
	}

	if err := mk(vh); err != nil {
		t.Fatalf("correct-length normW should load, got %v", err)
	}
	err := mk(vh - 2) // short norm tensor
	if err == nil {
		t.Fatal("short mm_soft_emb_norm.weight loaded cleanly — C-23: must reject at load, not panic in Forward")
	}
	if !strings.Contains(err.Error(), "mm_soft_emb_norm") {
		t.Errorf("wrong error for short normW: %v", err)
	}
}
