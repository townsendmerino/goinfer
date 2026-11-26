package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"strings"
	"testing"
)

// TestKVFacts_readTheArchitecture: the facts a backend's KV policy sees are the architecture's own properties, one per disqualifier.
func TestKVFacts_readTheArchitecture(t *testing.T) {
	for _, c := range []struct {
		name string
		a    *darch.Architecture
		want darch.KVFacts
	}{
		{"dense", &darch.Architecture{Name: "qwen2"}, darch.KVFacts{Family: "qwen2"}},
		{"moe", &darch.Architecture{Name: "qwen2", MoE: &MoEConfig{}}, darch.KVFacts{Family: "qwen2", MoE: true}},
		{"window", &darch.Architecture{Name: "mistral", SlidingWindow: 4096}, darch.KVFacts{Family: "mistral", SlidingWindow: true}},
		{"pairwise", &darch.Architecture{Name: "llama", RoPEInterleave: true}, darch.KVFacts{Family: "llama", PairwiseRoPE: true}},
		{"mrope", &darch.Architecture{Name: "qwen2", MRopeSection: []int{16, 24, 24}}, darch.KVFacts{Family: "qwen2", MRoPE: true}},
		{"mla", &darch.Architecture{Name: "deepseek", MLA: &archparts.MLAParams{}}, darch.KVFacts{Family: "deepseek", MLA: true}},
	} {
		if got := c.a.KVFacts(); got != c.want {
			t.Errorf("%s: kvFacts = %+v, want %+v", c.name, got, c.want)
		}
	}
}

// TestResolveKV_f16Resolution: CUDA's policy resolves "f16" / "auto" / "" / "f32" by family and by its auto switch; another resident backend
// gives f16 for an explicit "f16" and nothing else.
func TestResolveKV_f16Resolution(t *testing.T) {
	a := &darch.Architecture{Name: "qwen2", NumLayers: 1}
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
	RegisterKVF16Policy("cuda", KVF16Policy{Decline: CUDAKVF16Decline, Auto: func() bool { return auto }, AutoDecline: CUDAKVF16AutoDecline})
	gemma := &darch.Architecture{Name: "gemma3", NumLayers: 1, SlidingWindow: 1024}
	undelivered := &darch.Architecture{Name: "laguna", NumLayers: 1}
	moe := &darch.Architecture{Name: "qwen3_moe", NumLayers: 1, MoE: &MoEConfig{}}
	pairwise := &darch.Architecture{Name: "cohere", NumLayers: 1, RoPEInterleave: true}
	hybrid := &darch.Architecture{Name: "qwen3_5", NumLayers: 1, Qwen35: &archparts.Qwen35Params{}} // a group whose G4 has not cleared it
	for _, c := range []struct {
		kv   string
		a    *darch.Architecture
		auto bool
		want string
	}{
		{"", a, true, "f16"}, {"auto", a, true, "f16"}, {"f16", a, true, "f16"}, {"f32", a, true, "f32"},
		{"", a, false, "f32"}, {"auto", a, false, "f32"}, {"f16", a, false, "f16"},
		{"", gemma, true, "f32"}, {"auto", gemma, true, "f32"}, {"f16", gemma, true, "f16"}, {"f16", undelivered, true, "f32"}, {"i8", a, true, "f32"},
		{"", moe, true, "f16"}, {"auto", moe, false, "f32"}, {"f16", moe, false, "f16"},
		{"", pairwise, true, "f32"}, {"auto", pairwise, true, "f32"}, {"f16", pairwise, true, "f16"},
		{"", hybrid, true, "f32"}, {"auto", hybrid, true, "f32"}, {"f16", hybrid, true, "f16"}, {"f16", hybrid, false, "f16"},
	} {
		auto = c.auto
		if got, _ := ResolveKV("cuda", c.a, Options{KVPrecision: c.kv}); got != c.want {
			t.Errorf("auto=%v -kv %q on %s: %q, want %q", c.auto, c.kv, c.a.Name, got, c.want)
		}
	}
	auto = true
	if _, why := ResolveKV("cuda", a, Options{}); why != "auto is f16 on CUDA for this family" {
		t.Errorf("qwen2 auto on CUDA: reason %q, want the plain one", why)
	}
	if r := kvF16Decline("cuda", undelivered, Options{}); r == "" {
		t.Error("laguna on CUDA: no decline reason")
	}
}

// TestCUDAKVF16Decline_table pins which architectures the CUDA f16 KV cache takes and the reason each other one is held at f32: the decision is
// made before pricing, so a family that slipped through would be sized for f16 and then run kernels that read f32.
func TestCUDAKVF16Decline_table(t *testing.T) {
	dense := func(name string) darch.KVFacts { return darch.KVFacts{Family: name} }
	cases := []struct {
		name string
		f    darch.KVFacts
		o    Options
		want string // "" = f16 allowed; else a substring of the decline
	}{
		{"qwen2 dense", dense("qwen2"), Options{}, ""},
		{"llama dense", dense("llama"), Options{}, ""},
		{"mistral dense", dense("mistral"), Options{}, ""},
		{"qwen3 dense", dense("qwen3"), Options{}, ""},
		{"phi3 dense", dense("phi3"), Options{}, ""},
		{"gemma3", darch.KVFacts{Family: "gemma3", SlidingWindow: true}, Options{}, ""},
		{"gemma2", darch.KVFacts{Family: "gemma2", SlidingWindow: true}, Options{}, ""},
		{"gpt-oss", darch.KVFacts{Family: "gpt-oss", MoE: true, SlidingWindow: true}, Options{}, ""},
		{"flint", darch.KVFacts{Family: "flint", MoE: true, SlidingWindow: true}, Options{}, ""},
		{"laguna not delivered", dense("laguna"), Options{}, "keeps f32"},
		{"qwen3_5", darch.KVFacts{Family: "qwen3_5", Hybrid: true}, Options{}, ""},
		{"gemma4", darch.KVFacts{Family: "gemma4", Hybrid: true, SlidingWindow: true}, Options{}, ""},
		{"lfm2 not delivered", darch.KVFacts{Family: "lfm2", Hybrid: true}, Options{}, "keeps f32"},
		{"qwen2 MoE", darch.KVFacts{Family: "qwen2", MoE: true}, Options{}, ""},
		{"qwen2_moe", darch.KVFacts{Family: "qwen2_moe", MoE: true}, Options{}, ""},
		{"qwen3_moe", darch.KVFacts{Family: "qwen3_moe", MoE: true}, Options{}, ""},
		{"mixtral", darch.KVFacts{Family: "mixtral", MoE: true}, Options{}, ""},
		{"qwen3_moe paged experts", darch.KVFacts{Family: "qwen3_moe", MoE: true}, Options{MoECacheExperts: true}, ""},
		{"granitemoe not delivered", darch.KVFacts{Family: "granitemoe", MoE: true}, Options{}, "keeps f32"},
		{"mistral sliding window", darch.KVFacts{Family: "mistral", SlidingWindow: true}, Options{}, ""},
		{"llama pairwise rope", darch.KVFacts{Family: "llama", PairwiseRoPE: true}, Options{}, ""},
		{"cohere", darch.KVFacts{Family: "cohere", PairwiseRoPE: true}, Options{}, ""},
		{"cohere2", darch.KVFacts{Family: "cohere2", PairwiseRoPE: true, SlidingWindow: true}, Options{}, ""},
		{"glm_ocr pairwise m-rope", darch.KVFacts{Family: "glm_ocr", PairwiseRoPE: true, MRoPE: true}, Options{}, "keeps f32"},
		{"qwen2 m-rope", darch.KVFacts{Family: "qwen2", MRoPE: true}, Options{}, ""},
		{"qwen2_5_vl", darch.KVFacts{Family: "qwen2_5_vl", MRoPE: true}, Options{}, ""},
		{"qwen3_vl_moe", darch.KVFacts{Family: "qwen3_vl_moe", MRoPE: true, MoE: true}, Options{}, ""},
		{"llama mla", darch.KVFacts{Family: "llama", MLA: true}, Options{}, ""},
		{"deepseek_v2", darch.KVFacts{Family: "deepseek_v2", MLA: true, MoE: true}, Options{}, ""},
		{"bailing_hybrid not delivered", darch.KVFacts{Family: "bailing_hybrid", MLA: true}, Options{}, "keeps f32"},
		{"qwen2 hybrid", darch.KVFacts{Family: "qwen2", Hybrid: true}, Options{}, ""},
	}
	for _, c := range cases {
		got := CUDAKVF16Decline(c.f, c.o)
		switch {
		case c.want == "" && got != "":
			t.Errorf("%s: declined (%q), want f16", c.name, got)
		case c.want != "" && !strings.Contains(got, c.want):
			t.Errorf("%s: decline = %q, want it to contain %q", c.name, got, c.want)
		}
		if c.want == "" && !cudaKVF16Delivered(c.f) {
			t.Errorf("%s: allowed by the decline check but not in the delivered set", c.name)
		}
	}
}

// TestCUDAKVF16AutoDecline_table pins which delivered models `--kv auto` keeps at f32 on CUDA: a group delivered on request whose quality
// gate (G4) has not passed stays f32 under auto, and a cleared family is not held back.
func TestCUDAKVF16AutoDecline_table(t *testing.T) {
	for _, c := range []struct {
		name string
		f    darch.KVFacts
		o    Options
		auto bool
	}{
		{"qwen2 dense", darch.KVFacts{Family: "qwen2"}, Options{}, true},
		{"phi3 dense", darch.KVFacts{Family: "phi3"}, Options{}, true},
		{"qwen3_moe", darch.KVFacts{Family: "qwen3_moe", MoE: true}, Options{}, true},
		{"qwen2_moe", darch.KVFacts{Family: "qwen2_moe", MoE: true}, Options{}, true},
		{"mixtral", darch.KVFacts{Family: "mixtral", MoE: true}, Options{}, true},
		{"qwen2 with MoE", darch.KVFacts{Family: "qwen2", MoE: true}, Options{}, true},
		{"qwen3_moe paged experts", darch.KVFacts{Family: "qwen3_moe", MoE: true}, Options{MoECacheExperts: true}, true},
		{"qwen3_5_moe: hybrid and MoE", darch.KVFacts{Family: "qwen3_5_moe", Hybrid: true, MoE: true}, Options{}, false},
		{"gemma3", darch.KVFacts{Family: "gemma3", SlidingWindow: true}, Options{}, false},
		{"gemma2", darch.KVFacts{Family: "gemma2", SlidingWindow: true}, Options{}, false},
		{"mistral with a window", darch.KVFacts{Family: "mistral", SlidingWindow: true}, Options{}, false},
		{"gpt-oss: sliding window and MoE", darch.KVFacts{Family: "gpt-oss", MoE: true, SlidingWindow: true}, Options{}, false},
		{"cohere", darch.KVFacts{Family: "cohere", PairwiseRoPE: true}, Options{}, false},
		{"cohere2: sliding window and pairwise RoPE", darch.KVFacts{Family: "cohere2", PairwiseRoPE: true, SlidingWindow: true}, Options{}, false},
		{"gemma4: sliding window and hybrid", darch.KVFacts{Family: "gemma4", Hybrid: true, SlidingWindow: true}, Options{}, false},
		{"qwen2_5_vl", darch.KVFacts{Family: "qwen2_5_vl", MRoPE: true}, Options{}, true},
		{"qwen3_vl", darch.KVFacts{Family: "qwen3_vl", MRoPE: true}, Options{}, true},
		{"qwen3_vl_moe", darch.KVFacts{Family: "qwen3_vl_moe", MRoPE: true, MoE: true}, Options{}, true},
		{"qwen3_5", darch.KVFacts{Family: "qwen3_5", Hybrid: true}, Options{}, false},
		{"deepseek_v3", darch.KVFacts{Family: "deepseek_v3", MLA: true}, Options{}, false},
	} {
		if got := CUDAKVF16AutoDecline(c.f, c.o); (got == "") != c.auto {
			t.Errorf("%s: auto decline %q, want auto f16 %v", c.name, got, c.auto)
		}
	}
	if got := CUDAKVF16AutoDecline(darch.KVFacts{Family: "qwen3_5", Hybrid: true}, Options{}); !strings.Contains(got, "did not pass") {
		t.Errorf("qwen3_5: auto decline %q does not publish the hybrids' failed gate", got)
	}
	if got := CUDAKVF16AutoDecline(darch.KVFacts{Family: "gemma3", SlidingWindow: true}, Options{}); !strings.Contains(got, "did not pass") {
		t.Errorf("gemma3: auto decline %q does not publish the sliding-window group's failed gate", got)
	}
	// The three the owner narrowed auto away from (D-KV4-10) keep f32 under auto, each naming what would clear it.
	for _, c := range []struct {
		f    darch.KVFacts
		want string
	}{
		{darch.KVFacts{Family: "gemma3", SlidingWindow: true}, "passing K=8,000 cell"},
		{darch.KVFacts{Family: "cohere2", PairwiseRoPE: true, SlidingWindow: true}, "passing K=8,000 cell"},
		{darch.KVFacts{Family: "gpt-oss", MoE: true, SlidingWindow: true}, "sinks G4 cell"},
	} {
		if got := CUDAKVF16AutoDecline(c.f, Options{}); !strings.Contains(got, c.want) {
			t.Errorf("%s: auto decline %q, want it to name %q", c.f.Family, got, c.want)
		}
	}
}

// TestResidentSpec_matchesLegacyAccessors verifies that Model.ResidentSpec() populates all sub-specs
// and agrees with the accessors (MoE, MLA, Qwen35).
func TestResidentSpec_matchesLegacyAccessors(t *testing.T) {
	m := &Model{
		w: &darch.Weights{
			Arch: &darch.Architecture{
				Name:      "flint",
				AttnScale: 0.125,
				MoE: &MoEConfig{
					NumExperts:            8,
					TopK:                  2,
					IntermediateDim:       1024,
					SharedIntermediateDim: 512,
					RouterSigmoid:         false,
					NormTopKProb:          true,
					SharedUngated:         false,
					RoutedScale:           1.0,
					NGroup:                1,
					TopkGroup:             1,
				},
				MLA: &archparts.MLAParams{
					QLoRARank:      1536,
					KVLoRARank:     512,
					QKNopeHeadDim:  128,
					QKRopeHeadDim:  64,
					VHeadDim:       128,
					RoPEInterleave: true,
				},
				Qwen35: &archparts.Qwen35Params{
					ConvKernel:    4,
					KeyHeadDim:    64,
					ValueHeadDim:  64,
					NumKeyHeads:   8,
					NumValueHeads: 8,
					AttnGate:      true,
				},
			},
		},
	}
	spec := m.ResidentSpec()
	if spec.MoE == nil || spec.MLA == nil || spec.Qwen35 == nil {
		t.Fatalf("ResidentSpec() missing sub-spec: %+v", spec)
	}
	if mo, ok := m.BackendView().MoE(); !ok || mo != *spec.MoE {
		t.Errorf("ResidentSpec().MoE = %+v, disagrees with MoE() = %+v", *spec.MoE, mo)
	}
	if mla, ok := m.BackendView().MLA(); !ok || mla != *spec.MLA {
		t.Errorf("ResidentSpec().MLA = %+v, disagrees with MLA() = %+v", *spec.MLA, mla)
	}
	if q, ok := m.BackendView().Qwen35(); !ok || q != *spec.Qwen35 {
		t.Errorf("ResidentSpec().Qwen35 = %+v, disagrees with Qwen35() = %+v", *spec.Qwen35, q)
	}
}
