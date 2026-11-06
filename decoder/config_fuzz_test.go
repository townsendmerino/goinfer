package decoder

import (
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

// A checkpoint's config.json is untrusted, and every family's adapter runs on it. FuzzGGUFConfig is this for a GGUF; this is the other door.
func FuzzLoadConfig(f *testing.F) {
	seeds, err := filepath.Glob(filepath.Join("..", "testdata", "*-tiny", "config.json"))
	if err != nil {
		f.Fatal(err)
	}
	if len(seeds) < 10 {
		f.Fatalf("%d seed configs: the fixtures are not where this looks", len(seeds))
	}
	for _, p := range seeds {
		raw, err := os.ReadFile(p)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(raw)
	}
	f.Add([]byte(`{"model_type":"llama","num_hidden_layers":2147483647,"hidden_size":-1}`))
	f.Add([]byte(`{"model_type":"gemma4","text_config":{"layer_types":[1]},"num_hidden_layers":1e308}`))
	f.Add([]byte(`{"model_type":"","rope_scaling":{"type":3},"rope_parameters":[]}`))
	f.Add([]byte(`{"hidden_size":Infinity,"model_type":NaN}`))
	f.Fuzz(func(t *testing.T, raw []byte) {
		c, err := loadConfig(fstest.MapFS{"config.json": {Data: raw}}, "config.json")
		if err != nil {
			return
		}
		_, _, _ = resolveArchitecture(c)
	})
}
