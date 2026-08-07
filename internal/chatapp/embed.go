//go:build embed && !prequant

package chatapp

import (
	_ "embed"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
)

// loadEmbedded loads the baked-in GGUF: in-memory by default (parsed straight
// from the binary image, no temp file), or to a temp file under --model-tmp.
func loadEmbedded(useTmp bool, opts decoder.Options) (*session, error) {
	if useTmp {
		progress("writing model → temp file…")
		path, cleanup, err := embeddedModelTemp()
		if err != nil {
			return nil, err
		}
		defer cleanup()
		return loadFromPath(path, opts)
	}
	raw, err := embeddedModelBytes()
	if err != nil {
		return nil, err
	}
	return loadFromBytes(raw, opts)
}

// modelGGUF is the GGUF model baked into the binary — stored UNCOMPRESSED. The
// q4 weights are high-entropy (zstd shaved only ~3%), so compression bought
// almost nothing on size while costing inflate time + a full-size heap buffer on
// every launch. Uncompressed, the in-memory path parses the model straight from
// this image-mapped slice with no heap copy. The file is a BUILD INPUT generated
// by build-embed.sh (gitignored, not committed — it exceeds GitHub's 100 MB file
// cap). Without -tags embed it is not compiled, so the default build carries no
// model.
//
//go:embed model.gguf
var modelGGUF []byte

// hasEmbeddedModel reports that this build has a baked-in model.
const hasEmbeddedModel = true

// embeddedModelBytes returns the baked-in model. This is the default path: the
// returned slice is the binary's own image-mapped data (zero-copy), loaded with
// no filesystem access — so the binary runs on a read-only / FROM-scratch
// filesystem, and peak RAM is just the resident weights (no inflate buffer).
func embeddedModelBytes() ([]byte, error) {
	return modelGGUF, nil
}

// embeddedModelTemp writes the baked-in model to a temp .gguf and returns its
// path (the --model-tmp / GOINFER_MODEL_TMP opt-out). The model then mmaps from
// disk instead of the heap during the build — marginally lower peak RAM on huge
// models, at the cost of needing a writable temp dir. cleanup removes the file;
// call it once the model is built (the weights are fresh copies after load).
func embeddedModelTemp() (path string, cleanup func(), err error) {
	f, err := os.CreateTemp("", "goinfer-model-*.gguf")
	if err != nil {
		return "", nil, fmt.Errorf("embed: create temp file: %w", err)
	}
	if _, err := f.Write(modelGGUF); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", nil, fmt.Errorf("embed: write temp model: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(f.Name())
		return "", nil, fmt.Errorf("embed: close temp model: %w", err)
	}
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}
