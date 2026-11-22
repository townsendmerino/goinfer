package decoder

import (
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// A GGUF that carries llama.cpp's prism.hadamard.* metadata has its weights in a Hadamard-rotated basis that needs a matching activation
// transform on every forward pass; a correct dequant of such a file still generates garbage, in whatever tensor type it is stored.
// The loader declines it by name, through buildGGUFWeights (the loader's entry), before any tensor is read; a file without the key is
// not declined by the guard.
func TestGGUF_hadamardRotatedWeightsDeclinedByName(t *testing.T) {
	base := func(extra ...ggufKV) *embed.GGUFFile {
		kvs := append([]ggufKV{
			kvStr("general.architecture", "qwen35"),
			kvU32("qwen35.embedding_length", 8),
			kvU32("qwen35.block_count", 1),
		}, extra...)
		g, err := embed.OpenGGUFBytes(buildGGUF(kvs, []ggufTensorDecl{embedTensor(8, 32)}))
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		t.Cleanup(func() { g.Close() })
		return g
	}
	for _, key := range []string{"prism.hadamard.version", "prism.hadamard.block_size"} {
		_, err := buildGGUFWeights(base(kvU32(key, 1)), quantNone, false, true, false, nil)
		if !errors.Is(err, ErrUnsupportedModel) {
			t.Fatalf("%s: err = %v, want ErrUnsupportedModel", key, err)
		}
		for _, want := range []string{"Hadamard", key} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("%s: the decline %q does not name %q", key, err, want)
			}
		}
	}
	// The control: the same file without the key fails (it is not a complete model), but not as a rotated file.
	if _, err := buildGGUFWeights(base(), quantNone, false, true, false, nil); err != nil && strings.Contains(err.Error(), "Hadamard") {
		t.Errorf("an unrotated file was declined as rotated: %v", err)
	}
}
