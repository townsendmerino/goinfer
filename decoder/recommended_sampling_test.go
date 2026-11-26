package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"testing/fstest"
)

func rsVal(r *RecommendedSampling) any {
	if r == nil {
		return nil
	}
	return *r
}

// TestRecommendedSampling_sources: an HF directory reads its own generation_config.json; a bare .gguf reads only its own
// metadata, never a generation_config.json in its folder; and a bundle carries the values, held to the same ranges when
// read back.
func TestRecommendedSampling_sources(t *testing.T) {
	gc := []byte(`{"do_sample": true, "temperature": 0.3, "top_k": 7}`)

	cfg, err := loadConfig(fstest.MapFS{
		"d/config.json":            {Data: []byte(`{"model_type": "llama", "vocab_size": 8}`)},
		"d/generation_config.json": {Data: gc},
	}, "d/config.json")
	if err != nil {
		t.Fatal(err)
	}
	if want := (&RecommendedSampling{Source: "generation_config.json", Temperature: new(0.3), TopK: new(7)}); !reflect.DeepEqual(cfg.Sampling, want) {
		t.Errorf("HF directory: got %+v, want %+v", rsVal(cfg.Sampling), *want)
	}

	raw, err := os.ReadFile("../testdata/gemma1-tiny.gguf")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "gemma1-tiny.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), gc, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Load(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if s := m.Config().Sampling; s != nil {
		t.Errorf("a bare .gguf with no general.sampling keys took %+v from its folder's generation_config.json", *s)
	}

	m.w.Cfg.Sampling = &RecommendedSampling{Source: "gguf general.sampling", Temperature: new(0.6), TopP: new(1.5), TopK: new(20)}
	blob, err := SerializeWeights(m.w, "sampling-test")
	if err != nil {
		t.Fatal(err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatal(err)
	}
	if want := (&RecommendedSampling{Source: "gguf general.sampling", Temperature: new(0.6), TopK: new(20)}); !reflect.DeepEqual(w2.Cfg.Sampling, want) {
		t.Errorf("bundle round trip: got %+v, want %+v (top_p 1.5 dropped on read)", rsVal(w2.Cfg.Sampling), *want)
	}
}
