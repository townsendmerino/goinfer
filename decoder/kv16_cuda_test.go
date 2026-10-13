package decoder

import (
	"strings"
	"testing"
)

// TestCudaKVF16Decline_table pins which architectures the CUDA f16 KV cache takes and the reason each other one is held at f32: the decision is
// made before pricing (kv16_cuda.go), so a family that slipped through would be sized for f16 and then run kernels that read f32.
func TestCudaKVF16Decline_table(t *testing.T) {
	dense := func(name string) *Architecture { return &Architecture{Name: name} }
	cases := []struct {
		name string
		a    *Architecture
		o    Options
		want string // "" = f16 allowed; else a substring of the decline
	}{
		{"qwen2 dense", dense("qwen2"), Options{}, ""},
		{"llama dense", dense("llama"), Options{}, ""},
		{"mistral dense", dense("mistral"), Options{}, ""},
		{"qwen3 dense", dense("qwen3"), Options{}, ""},
		{"phi3 dense", dense("phi3"), Options{}, ""},
		{"nil", nil, Options{}, "no architecture"},
		{"gemma3 not cleared", dense("gemma3"), Options{}, "keeps f32"},
		{"qwen3_5 not cleared", dense("qwen3_5"), Options{}, "keeps f32"},
		{"qwen2 MoE", &Architecture{Name: "qwen2", MoE: &MoEConfig{}}, Options{}, "MoE"},
		{"qwen2 paged experts", dense("qwen2"), Options{MoECacheExperts: true}, "paged-experts"},
		{"mistral sliding window", &Architecture{Name: "mistral", SlidingWindow: 4096}, Options{}, "sliding-window"},
		{"llama pairwise rope", &Architecture{Name: "llama", ropeInterleave: true}, Options{}, "pairwise RoPE"},
		{"qwen2 m-rope", &Architecture{Name: "qwen2", MRopeSection: []int{16, 24, 24}}, Options{}, "m-RoPE"},
		{"llama mla", &Architecture{Name: "llama", mla: &mlaParams{}}, Options{}, "MLA"},
	}
	for _, c := range cases {
		got := cudaKVF16Decline(c.a, c.o)
		switch {
		case c.want == "" && got != "":
			t.Errorf("%s: declined (%q), want f16", c.name, got)
		case c.want != "" && !strings.Contains(got, c.want):
			t.Errorf("%s: decline = %q, want it to contain %q", c.name, got, c.want)
		}
		if cleared := cudaKVF16Cleared(c.a); c.want == "" && !cleared {
			t.Errorf("%s: allowed by the decline check but not in the cleared set", c.name)
		}
	}
}

// TestKVF16For_offCUDA: the new CUDA resolution leaves every other backend as it was. WebGPU and the CPU honour "f16" and nothing else makes
// them f16, and "f32" is always f32.
func TestKVF16For_offCUDA(t *testing.T) {
	a := &Architecture{Name: "qwen2"}
	for _, backend := range []string{"", "cpu", "webgpu"} {
		for _, kv := range []string{"", "auto", "f32"} {
			if (Options{Backend: backend, KVPrecision: kv}).kvF16For(a) {
				t.Errorf("backend %q -kv %q resolved to f16", backend, kv)
			}
		}
		if !(Options{Backend: backend, KVPrecision: "f16"}).kvF16For(a) {
			t.Errorf("backend %q -kv f16 did not resolve to f16", backend)
		}
	}
	if (Options{Backend: "cuda", KVPrecision: "f32"}).kvF16For(a) {
		t.Error("cuda -kv f32 resolved to f16")
	}
}
