package ggufconfig

import (
	"reflect"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

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
	want := &RecommendedSampling{Source: "gguf general.sampling", Temperature: new(1.0), TopP: new(0.95), TopK: new(64), MinP: new(0.05), RepetitionPenalty: new(1.1)}
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
