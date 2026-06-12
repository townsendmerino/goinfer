package multimodal

import (
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// openWeights mmaps a Gemma 3 VL checkpoint's safetensors: the multi-shard set
// named by model.safetensors.index.json when present (a real HF VL checkpoint like
// gemma-3-4b-it ships the projector inside the model shards), else the single
// model.safetensors (the tiny pinned checkpoint under testdata/). Both yield the
// same SafetensorsFile, read via embed.SafetensorsFile.TensorF32. The encoder half
// lives in aikit/vision now; the projector keeps this thin local opener since it
// reads the multi_modal_projector.* tensors from the same checkpoint.
func openWeights(dir string) (*embed.SafetensorsFile, error) {
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(idx); err == nil {
		return embed.OpenSafetensorsShardedMmap(idx)
	}
	return embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
}
