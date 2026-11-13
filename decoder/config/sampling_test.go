package config

import (
	"reflect"
	"testing"
	"testing/fstest"
)

func fptr(v float64) *float64 { return &v }
func iptr(v int) *int         { return &v }

func rsVal(r *RecommendedSampling) any {
	if r == nil {
		return nil
	}
	return *r
}

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
