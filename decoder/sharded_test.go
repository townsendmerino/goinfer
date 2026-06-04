package decoder

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// G1 sharded-loader acceptance: load a multi-shard copy of the 270M checkpoint
// and reproduce the M1 sampled-tensor checksums — proving the index.json +
// multi-mmap + name-resolution path yields byte-equal tensors to the single
// file. The fixture is per-machine (~536 MB, gitignored); regenerate with:
//
//	.venv/bin/python scripts/shard_checkpoint.py
const gemmaShardedDir = "../testdata/gemma-3-270m-sharded"

func TestLoadWeights_shardedChecksums(t *testing.T) {
	g := loadGemmaGolden(t) // skips if golden absent
	if _, err := os.Stat(filepath.Join(gemmaShardedDir, shardIndexFile)); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no sharded fixture at %s — generate with scripts/shard_checkpoint.py", gemmaShardedDir)
	}

	w, err := LoadWeights(gemmaShardedDir)
	if err != nil {
		t.Fatalf("LoadWeights(sharded): %v", err)
	}
	defer w.st.Close() // munmaps every shard

	// Same checksum bar the single-file M1 test uses; tensors are spread across
	// shards by the round-robin split, so this exercises cross-shard resolution.
	checkSampledChecksums(t, w, g)
}
