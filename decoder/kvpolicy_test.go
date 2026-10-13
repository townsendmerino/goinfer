package decoder

import "testing"

// TestKVFacts_readTheArchitecture: the facts a backend's KV policy sees are the architecture's own properties, one per disqualifier.
func TestKVFacts_readTheArchitecture(t *testing.T) {
	for _, c := range []struct {
		name string
		a    *Architecture
		want KVFacts
	}{
		{"dense", &Architecture{Name: "qwen2"}, KVFacts{Family: "qwen2"}},
		{"moe", &Architecture{Name: "qwen2", MoE: &MoEConfig{}}, KVFacts{Family: "qwen2", MoE: true}},
		{"window", &Architecture{Name: "mistral", SlidingWindow: 4096}, KVFacts{Family: "mistral", SlidingWindow: true}},
		{"pairwise", &Architecture{Name: "llama", ropeInterleave: true}, KVFacts{Family: "llama", PairwiseRoPE: true}},
		{"mrope", &Architecture{Name: "qwen2", MRopeSection: []int{16, 24, 24}}, KVFacts{Family: "qwen2", MRoPE: true}},
		{"mla", &Architecture{Name: "deepseek", mla: &mlaParams{}}, KVFacts{Family: "deepseek", MLA: true}},
	} {
		if got := c.a.kvFacts(); got != c.want {
			t.Errorf("%s: kvFacts = %+v, want %+v", c.name, got, c.want)
		}
	}
}

// TestKVF16For_resolution: a registered policy resolves "f16" / "auto" / "" / "f32"; with none, only an explicit "f16" is f16.
func TestKVF16For_resolution(t *testing.T) {
	a := &Architecture{Name: "qwen2"}
	for _, backend := range []string{"", "cpu", "webgpu", "nopolicy"} {
		for _, kv := range []string{"", "auto", "f32"} {
			if (Options{Backend: backend, KVPrecision: kv}).kvF16For(a) {
				t.Errorf("backend %q -kv %q resolved to f16 with no policy", backend, kv)
			}
		}
		if !(Options{Backend: backend, KVPrecision: "f16"}).kvF16For(a) {
			t.Errorf("backend %q -kv f16 did not resolve to f16", backend)
		}
	}
	auto := true
	RegisterKVF16Policy("kvtest", KVF16Policy{
		Decline: func(f KVFacts, _ Options) string {
			if f.Family != "qwen2" {
				return "not covered"
			}
			return ""
		},
		Auto: func() bool { return auto },
	})
	defer delete(kvPolicies, "kvtest")
	for _, c := range []struct {
		kv   string
		a    *Architecture
		auto bool
		want bool
	}{
		{"", a, true, true}, {"auto", a, true, true}, {"f16", a, true, true}, {"f32", a, true, false},
		{"", a, false, false}, {"auto", a, false, false}, {"f16", a, false, true},
		{"", &Architecture{Name: "gemma3"}, true, false}, {"f16", &Architecture{Name: "gemma3"}, true, false},
	} {
		auto = c.auto
		if got := (Options{Backend: "kvtest", KVPrecision: c.kv}).kvF16For(c.a); got != c.want {
			t.Errorf("policy auto=%v -kv %q on %s: f16 = %v, want %v", c.auto, c.kv, c.a.Name, got, c.want)
		}
	}
	if r := kvF16Decline("kvtest", &Architecture{Name: "gemma3"}, Options{}); r != "not covered" {
		t.Errorf("decline reason = %q", r)
	}
}
