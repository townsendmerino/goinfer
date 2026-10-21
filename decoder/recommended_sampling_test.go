package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"testing/fstest"

	"github.com/townsendmerino/aikit/embed"
)

func fptr(v float64) *float64 { return &v }
func iptr(v int) *int         { return &v }

// TestSamplingFromGenerationConfig: the fields are read as transformers writes them, do_sample false is greedy, and a
// value that changes nothing or is out of range is dropped.
func TestSamplingFromGenerationConfig(t *testing.T) {
	for _, c := range []struct {
		name, file string
		want       *RecommendedSampling
	}{
		{"qwen2.5", `{"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05, "eos_token_id": [151645, 151643]}`,
			&RecommendedSampling{Source: "generation_config.json", Temperature: fptr(0.7), TopP: fptr(0.8), TopK: iptr(20), RepetitionPenalty: fptr(1.05)}},
		{"gemma3, no temperature", `{"do_sample": true, "top_p": 0.95, "top_k": 64}`,
			&RecommendedSampling{Source: "generation_config.json", TopP: fptr(0.95), TopK: iptr(64)}},
		{"do_sample false is greedy", `{"do_sample": false, "temperature": 0.7, "top_k": 5}`,
			&RecommendedSampling{Source: "generation_config.json", Temperature: fptr(0)}},
		{"no-op values dropped", `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0}`,
			&RecommendedSampling{Source: "generation_config.json", Temperature: fptr(1)}},
		{"out of range dropped", `{"temperature": -1, "top_p": 1.5, "min_p": 2, "top_k": 2.5}`, nil},
		{"min_p and presence_penalty", `{"min_p": 0.05, "presence_penalty": 1.5}`,
			&RecommendedSampling{Source: "generation_config.json", MinP: fptr(0.05), PresencePenalty: fptr(1.5)}},
		{"no sampling keys", `{"bos_token_id": 1, "eos_token_id": 2}`, nil},
		{"Python NaN", `{"temperature": NaN, "top_k": 20}`, &RecommendedSampling{Source: "generation_config.json", TopK: iptr(20)}},
		{"unparseable", `{"temperature": `, nil},
	} {
		got := samplingFromGenerationConfig(fstest.MapFS{"m/generation_config.json": {Data: []byte(c.file)}}, "m/generation_config.json")
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: got %+v, want %+v", c.name, rsVal(got), rsVal(c.want))
		}
	}
	if got := samplingFromGenerationConfig(fstest.MapFS{}, "generation_config.json"); got != nil {
		t.Errorf("absent file: got %+v, want nil", *got)
	}
}

func rsVal(r *RecommendedSampling) any {
	if r == nil {
		return nil
	}
	return *r
}

// TestGGUFSampling: llama.cpp's general.sampling.* keys, float32 values read at their stored decimal, a signed top_k.
func TestGGUFSampling(t *testing.T) {
	g := &embed.GGUFFile{Metadata: map[string]any{
		"general.sampling.temp":           float32(1.0),
		"general.sampling.top_p":          float32(0.95),
		"general.sampling.top_k":          int32(64),
		"general.sampling.min_p":          float32(0.05),
		"general.sampling.penalty_repeat": float32(1.1),
	}}
	want := &RecommendedSampling{Source: "gguf general.sampling", Temperature: fptr(1), TopP: fptr(0.95), TopK: iptr(64), MinP: fptr(0.05), RepetitionPenalty: fptr(1.1)}
	if got := ggufSampling(g); !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", rsVal(got), *want)
	}
	if got := ggufSampling(&embed.GGUFFile{Metadata: map[string]any{"general.sampling.top_k": int32(-1)}}); got != nil {
		t.Errorf("top_k -1 alone: got %+v, want nil", *got)
	}
	if got := ggufSampling(&embed.GGUFFile{Metadata: map[string]any{"general.architecture": "qwen2"}}); got != nil {
		t.Errorf("no keys: got %+v, want nil", *got)
	}
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
	if want := (&RecommendedSampling{Source: "generation_config.json", Temperature: fptr(0.3), TopK: iptr(7)}); !reflect.DeepEqual(cfg.Sampling, want) {
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

	m.w.Cfg.Sampling = &RecommendedSampling{Source: "gguf general.sampling", Temperature: fptr(0.6), TopP: fptr(1.5), TopK: iptr(20)}
	blob, err := SerializeWeights(m.w, "sampling-test")
	if err != nil {
		t.Fatal(err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatal(err)
	}
	if want := (&RecommendedSampling{Source: "gguf general.sampling", Temperature: fptr(0.6), TopK: iptr(20)}); !reflect.DeepEqual(w2.Cfg.Sampling, want) {
		t.Errorf("bundle round trip: got %+v, want %+v (top_p 1.5 dropped on read)", rsVal(w2.Cfg.Sampling), *want)
	}
}
