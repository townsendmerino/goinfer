//go:build cuda

package cuda

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// writeTinyDrafter writes a 1-layer BF16 DFlash checkpoint (hidden 8) to a temp dir and returns the dir.
func writeTinyDrafter(t *testing.T) string {
	t.Helper()
	const h, hd, inter = 8, 4, 16
	dir := t.TempDir()
	cfg := fmt.Sprintf(`{"hidden_size": %d, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": %d,
	  "intermediate_size": %d, "rms_norm_eps": 1e-6, "rope_theta": 10000,
	  "dflash_config": {"block_size": 4, "mask_token_id": 7, "target_layer_ids": [0, 1]}}`, h, hd, inter)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	shapes := map[string][]int{
		"fc.weight": {h, 2 * h}, "hidden_norm.weight": {h}, "norm.weight": {h},
		"layers.0.self_attn.q_proj.weight": {2 * hd, h}, "layers.0.self_attn.k_proj.weight": {hd, h},
		"layers.0.self_attn.v_proj.weight": {hd, h}, "layers.0.self_attn.o_proj.weight": {h, 2 * hd},
		"layers.0.mlp.gate_proj.weight": {inter, h}, "layers.0.mlp.up_proj.weight": {inter, h},
		"layers.0.mlp.down_proj.weight": {h, inter}, "layers.0.self_attn.q_norm.weight": {hd},
		"layers.0.self_attn.k_norm.weight": {hd}, "layers.0.input_layernorm.weight": {h},
		"layers.0.post_attention_layernorm.weight": {h},
	}
	header := map[string]any{}
	var blob []byte
	for name, shape := range shapes {
		n := 1
		for _, d := range shape {
			n *= d
		}
		off := len(blob)
		for range n {
			blob = binary.LittleEndian.AppendUint16(blob, 0x3f80) // bf16 1.0
		}
		header[name] = map[string]any{"dtype": "BF16", "shape": shape, "data_offsets": []int{off, len(blob)}}
	}
	hj, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := append(binary.LittleEndian.AppendUint64(nil, uint64(len(hj))), hj...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), append(file, blob...), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestAttachDrafter_refusesReleasedWeights: a DFlashDrafter whose host weights were released (ReleaseHostWeights) is refused
// before anything is read or uploaded, rather than uploading zero matrices that draft noise.
func TestAttachDrafter_refusesReleasedWeights(t *testing.T) {
	d, err := decoder.LoadDFlashDrafter(writeTinyDrafter(t))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if err := d.ReleaseHostWeights(); err != nil {
		t.Fatal(err)
	}
	_, err = (&cudaResident{}).AttachDrafter(d)
	if err == nil || !strings.Contains(err.Error(), "released") {
		t.Fatalf("AttachDrafter on a released drafter: %v, want a refusal naming the release", err)
	}
}
