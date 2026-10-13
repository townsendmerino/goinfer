//go:build cuda

package cuda

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCudaKVF16Decline_table pins which architectures the CUDA f16 KV cache takes and the reason each other one is held at f32: the decision is made before pricing,
// so a family that slipped through would be sized for f16 and then run kernels that read f32.
func TestCudaKVF16Decline_table(t *testing.T) {
	dense := func(name string) decoder.KVFacts { return decoder.KVFacts{Family: name} }
	cases := []struct {
		name string
		f    decoder.KVFacts
		o    decoder.Options
		want string // "" = f16 allowed; else a substring of the decline
	}{
		{"qwen2 dense", dense("qwen2"), decoder.Options{}, ""},
		{"llama dense", dense("llama"), decoder.Options{}, ""},
		{"mistral dense", dense("mistral"), decoder.Options{}, ""},
		{"qwen3 dense", dense("qwen3"), decoder.Options{}, ""},
		{"phi3 dense", dense("phi3"), decoder.Options{}, ""},
		{"gemma3 not cleared", dense("gemma3"), decoder.Options{}, "keeps f32"},
		{"qwen3_5 not cleared", dense("qwen3_5"), decoder.Options{}, "keeps f32"},
		{"qwen2 MoE", decoder.KVFacts{Family: "qwen2", MoE: true}, decoder.Options{}, "MoE"},
		{"qwen2 paged experts", dense("qwen2"), decoder.Options{MoECacheExperts: true}, "paged-experts"},
		{"mistral sliding window", decoder.KVFacts{Family: "mistral", SlidingWindow: true}, decoder.Options{}, "sliding-window"},
		{"llama pairwise rope", decoder.KVFacts{Family: "llama", PairwiseRoPE: true}, decoder.Options{}, "pairwise RoPE"},
		{"qwen2 m-rope", decoder.KVFacts{Family: "qwen2", MRoPE: true}, decoder.Options{}, "m-RoPE"},
		{"llama mla", decoder.KVFacts{Family: "llama", MLA: true}, decoder.Options{}, "MLA"},
		{"qwen2 hybrid", decoder.KVFacts{Family: "qwen2", Hybrid: true}, decoder.Options{}, "not covered"},
	}
	for _, c := range cases {
		got := cudaKVF16Decline(c.f, c.o)
		switch {
		case c.want == "" && got != "":
			t.Errorf("%s: declined (%q), want f16", c.name, got)
		case c.want != "" && !strings.Contains(got, c.want):
			t.Errorf("%s: decline = %q, want it to contain %q", c.name, got, c.want)
		}
		if c.want == "" && !cudaKVF16Cleared(c.f) {
			t.Errorf("%s: allowed by the decline check but not in the cleared set", c.name)
		}
	}
}
