package vision

import (
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// openWeights mmaps a checkpoint's safetensors: the multi-shard set named by
// model.safetensors.index.json when present (a real HF VL checkpoint like
// gemma-3-4b-it ships its tower inside the model shards), else the single
// model.safetensors (the tiny pinned tower under testdata/). Both yield the same
// SafetensorsFile, read via embed.SafetensorsFile.TensorF32 (F32/BF16/F16 dispatch).
func openWeights(dir string) (*embed.SafetensorsFile, error) {
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(idx); err == nil {
		return embed.OpenSafetensorsShardedMmap(idx)
	}
	return embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
}

// tensorPrefix reports the namespace a real HF VL checkpoint nests a tower under:
// "" when the bare `probe` tensor exists (the tiny pinned layout), else `nested`
// (e.g. "vision_tower.vision_model.") when the prefixed name is present. Lets one
// loader read both the stripped tiny checkpoints and a full HF VL safetensors.
func tensorPrefix(st *embed.SafetensorsFile, probe, nested string) string {
	if _, err := st.Tensor(probe); err == nil {
		return ""
	}
	return nested
}
