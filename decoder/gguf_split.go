package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"

	"github.com/townsendmerino/aikit/embed"
)

// ggufShardName matches llama-gguf-split's shard naming, "<prefix>-00001-of-00003.gguf": the prefix, the shard number
// and the shard count.
var ggufShardName = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// GGUFShards returns the files a GGUF path stands for: the path itself for a single file, or every shard of a split
// GGUF when path is its first shard (task-checkpoint-fetch-2026-09.md P4). The model is named by its first shard, the
// one that carries the metadata. A later shard is refused, and so is a set with a shard missing from the directory,
// naming it, so an interrupted pull reads as "re-run the pull" and not as a corrupt model.
func GGUFShards(path string) ([]string, error) {
	m := ggufShardName.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return []string{path}, nil
	}
	no, _ := strconv.Atoi(m[2])
	count, _ := strconv.Atoi(m[3])
	if count < 1 || no < 1 || no > count {
		return nil, fmt.Errorf("%s: shard %d of %d is not a valid split GGUF name", path, no, count)
	}
	first := filepath.Join(filepath.Dir(path), fmt.Sprintf("%s-%05d-of-%05d.gguf", m[1], 1, count))
	if no != 1 {
		return nil, fmt.Errorf("%s is shard %d of a split GGUF: load it by its first shard, %s", path, no, first)
	}
	shards := make([]string, count)
	for i := range shards {
		p := filepath.Join(filepath.Dir(path), fmt.Sprintf("%s-%05d-of-%05d.gguf", m[1], i+1, count))
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("split GGUF %s: shard %d of %d (%s) is missing; a pull of the model fetches the whole set and resumes an interrupted one", path, i+1, count, filepath.Base(p))
		}
		shards[i] = p
	}
	return shards, nil
}

// OpenGGUFMmap maps a GGUF model, single-file or split: the split set a first shard names is opened as one file
// (embed.OpenGGUFSplitMmap), so every reader of a model's tensors sees the whole model either way.
func OpenGGUFMmap(path string) (*embed.GGUFFile, error) {
	shards, err := GGUFShards(path)
	if err != nil {
		return nil, err
	}
	return embed.OpenGGUFSplitMmap(shards)
}

// GGUFFileBytes is a GGUF model's size on disk: the file's own, or for a split set's first shard the sum of every
// shard's. Size checks price the whole model with it: the fit guard's mapped-source term, the sidecar's disk check.
// ok is false when the path or a shard cannot be read.
func GGUFFileBytes(path string) (int64, bool) {
	shards, err := GGUFShards(path)
	if err != nil {
		return 0, false
	}
	var n int64
	for _, p := range shards {
		fi, err := os.Stat(p)
		if err != nil {
			return 0, false
		}
		n += fi.Size()
	}
	return n, true
}
