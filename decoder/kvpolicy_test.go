package decoder

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

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

// TestResolveKV_f16Resolution: CUDA's policy resolves "f16" / "auto" / "" / "f32" by family and by its auto switch; another resident backend
// gives f16 for an explicit "f16" and nothing else.
func TestResolveKV_f16Resolution(t *testing.T) {
	a := &Architecture{Name: "qwen2", NumLayers: 1}
	for _, backend := range []string{"webgpu", "nopolicy"} {
		for _, kv := range []string{"", "auto", "f32"} {
			if got, _ := ResolveKV(backend, a, Options{KVPrecision: kv}); got == "f16" {
				t.Errorf("backend %q -kv %q resolved to f16 with no policy", backend, kv)
			}
		}
		if got, _ := ResolveKV(backend, a, Options{KVPrecision: "f16"}); got != "f16" {
			t.Errorf("backend %q -kv f16 resolved to %q", backend, got)
		}
	}
	saved := kvPolicies["cuda"]
	defer func() { kvPolicies["cuda"] = saved }()
	auto := true
	RegisterKVF16Policy("cuda", KVF16Policy{Decline: CUDAKVF16Decline, Auto: func() bool { return auto }})
	gemma := &Architecture{Name: "gemma3", NumLayers: 1}
	for _, c := range []struct {
		kv   string
		a    *Architecture
		auto bool
		want string
	}{
		{"", a, true, "f16"}, {"auto", a, true, "f16"}, {"f16", a, true, "f16"}, {"f32", a, true, "f32"},
		{"", a, false, "f32"}, {"auto", a, false, "f32"}, {"f16", a, false, "f16"},
		{"", gemma, true, "f32"}, {"f16", gemma, true, "f32"}, {"i8", a, true, "f32"},
	} {
		auto = c.auto
		if got, _ := ResolveKV("cuda", c.a, Options{KVPrecision: c.kv}); got != c.want {
			t.Errorf("auto=%v -kv %q on %s: %q, want %q", c.auto, c.kv, c.a.Name, got, c.want)
		}
	}
	if r := kvF16Decline("cuda", gemma, Options{}); r == "" {
		t.Error("gemma3 on CUDA: no decline reason")
	}
}

// TestCUDAKVF16Decline_table pins which architectures the CUDA f16 KV cache takes and the reason each other one is held at f32: the decision is
// made before pricing, so a family that slipped through would be sized for f16 and then run kernels that read f32.
func TestCUDAKVF16Decline_table(t *testing.T) {
	dense := func(name string) KVFacts { return KVFacts{Family: name} }
	cases := []struct {
		name string
		f    KVFacts
		o    Options
		want string // "" = f16 allowed; else a substring of the decline
	}{
		{"qwen2 dense", dense("qwen2"), Options{}, ""},
		{"llama dense", dense("llama"), Options{}, ""},
		{"mistral dense", dense("mistral"), Options{}, ""},
		{"qwen3 dense", dense("qwen3"), Options{}, ""},
		{"phi3 dense", dense("phi3"), Options{}, ""},
		{"gemma3 not cleared", dense("gemma3"), Options{}, "keeps f32"},
		{"qwen3_5 not cleared", dense("qwen3_5"), Options{}, "keeps f32"},
		{"qwen2 MoE", KVFacts{Family: "qwen2", MoE: true}, Options{}, "MoE"},
		{"qwen2 paged experts", dense("qwen2"), Options{MoECacheExperts: true}, "paged-experts"},
		{"mistral sliding window", KVFacts{Family: "mistral", SlidingWindow: true}, Options{}, "sliding-window"},
		{"llama pairwise rope", KVFacts{Family: "llama", PairwiseRoPE: true}, Options{}, "pairwise RoPE"},
		{"qwen2 m-rope", KVFacts{Family: "qwen2", MRoPE: true}, Options{}, "m-RoPE"},
		{"llama mla", KVFacts{Family: "llama", MLA: true}, Options{}, "MLA"},
		{"qwen2 hybrid", KVFacts{Family: "qwen2", Hybrid: true}, Options{}, "not covered"},
	}
	for _, c := range cases {
		got := CUDAKVF16Decline(c.f, c.o)
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

// TestResidentBuilds_readTheResolver: Metal's and WebGPU's resident builds take their KV precision from
// Model.ResidentKVFor (ResolveKV), never from the raw request (KVCacheF16, KVCacheI8), so a precision the resolver
// changes reaches the build with the banner, the fit guard and the request log.
func TestResidentBuilds_readTheResolver(t *testing.T) {
	raw := regexp.MustCompile(`\.KVCache(F16|I8)\(\)`)
	builds := map[string]bool{}
	for _, dir := range []string{"../metal", "../gpu"} {
		files, err := filepath.Glob(filepath.Join(dir, "*.go"))
		if err != nil || len(files) == 0 {
			t.Fatalf("no Go files in %s (%v)", dir, err)
		}
		for _, f := range files {
			if strings.HasSuffix(f, "_test.go") {
				continue
			}
			b, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			if loc := raw.FindIndex(b); loc != nil {
				t.Errorf("%s reads the raw KV request (%s); a resident build reads ResidentKVFor", f, b[loc[0]:loc[1]])
			}
			if strings.Contains(string(b), ".ResidentKVFor(") {
				builds[dir] = true
			}
		}
	}
	for _, dir := range []string{"../metal", "../gpu"} {
		if !builds[dir] {
			t.Errorf("no file in %s reads ResidentKVFor: its build takes its KV precision from somewhere else", dir)
		}
	}
}
