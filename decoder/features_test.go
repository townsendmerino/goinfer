package decoder

import (
	"slices"
	"testing"
)

// archFeatureProfile is the feature set each REGISTERED architecture is expected to require.
// Every registry entry must appear here — TestResidentAdmission_registryCovered fails if a new
// arch lands unclassified, which is the recurrence guard: an unclassified arch would otherwise
// inherit whatever admission it happened to get and could run with features silently dropped.
//
// Only features that reach a backend matter for admission; decodeRunnerEligible already refuses
// several archs upstream (own-forward families, softcapped/sandwich/GPT-2 shapes), but they are
// classified here anyway so the taxonomy stays honest about what each family needs.
var archFeatureProfile = map[string][]ResidentFeature{
	// dense, plain — the subset every backend implements
	"qwen2":      {},
	"qwen2_5_vl": {},
	"llama":      {},
	// NOTE phi3 is config-DEPENDENT: Phi-4 and the released GGUFs are plain dense, while the
	// Phi-3-mini-4k safetensors declares sliding_window: 2047 and so also needs
	// FeatSlidingWindow. This table states the BASE profile; the authoritative requirement is
	// derived per-model at load (RequiredResidentFeatures), which is what admission uses. The
	// table's job is to force a new arch to be classified, not to be the runtime check.
	"phi3":             {},
	"gpt2":             {FeatLearnedPos, FeatNonGatedMLP, FeatOutBias},
	"mistral":          {FeatSlidingWindow},
	"qwen3":            {FeatQKNorm},
	"mellum":           {FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow},
	"mixtral":          {FeatMoE},
	"qwen2_moe":        {FeatMoE},
	"glm4_moe":         {FeatMoE, FeatPartialRotary, FeatQKNorm},
	"deepseek_v2":      {FeatMLA, FeatMoE},
	"deepseek_v3":      {FeatMLA, FeatMoE},
	"kimi_k2":          {FeatMLA, FeatMoE},
	"qwen3_5_moe":      {FeatMoE, FeatQKNorm},
	"llama4_text":      {FeatMoE},
	"nemotron_h":       {FeatNonGatedMLP, FeatSSM},
	"granitemoehybrid": {FeatLogitScale, FeatMoE, FeatSSM},
	"qwen3_5_moe_text": {FeatMoE, FeatQKNorm},
	// Gemma: embed-scale + (1+w) + sandwich norms (+ softcap on 2/3). Refused upstream.
	"gemma3":      {FeatEmbedScale, FeatRMSAddOne, FeatSandwichNorm},
	"gemma3_text": {FeatEmbedScale, FeatRMSAddOne, FeatSandwichNorm},
	"gemma4":      {FeatEmbedScale, FeatRMSAddOne, FeatSandwichNorm},
}

// TestResidentAdmission_registryCovered is THE recurrence guard. Every architecture in the
// registry must be classified in archFeatureProfile. A new family that lands without a profile
// fails here — forcing the author to state which features it needs, which in turn decides
// (via the subset check) which backends may run it. Without this, a new arch silently inherits
// admission and mis-runs on whichever backend lacks its features.
func TestResidentAdmission_registryCovered(t *testing.T) {
	for name := range registry {
		if _, ok := archFeatureProfile[name]; !ok {
			t.Errorf("architecture %q is registered but has no feature profile — classify it in "+
				"archFeatureProfile (decoder/features_test.go) so admission can refuse backends "+
				"that do not implement what it needs", name)
		}
	}
	for name := range archFeatureProfile {
		if _, ok := registry[name]; !ok {
			t.Errorf("archFeatureProfile has %q, which is not a registered architecture — stale entry", name)
		}
	}
}

// TestResidentAdmission_matrix asserts the (arch × backend) admission decision for every
// registered arch: a backend is admitted ONLY when it implements every feature the arch needs.
// Hardware-free — it reads the declared sets, so it runs in CI with no GPU.
func TestResidentAdmission_matrix(t *testing.T) {
	backends := []string{"cuda", "webgpu", "metal"}
	for _, name := range slices.Sorted(mapKeys(archFeatureProfile)) {
		req := archFeatureProfile[name]
		var admits []string
		for _, be := range backends {
			impl, ok := ResidentBackendFeatures[be]
			if !ok {
				t.Fatalf("backend %q has no declared feature set", be)
			}
			missing := missingFeatures(req, impl)
			admitted := len(missing) == 0
			// The invariant: admission ⇔ every required feature is implemented.
			for _, r := range req {
				if admitted && !impl[r] {
					t.Errorf("%s/%s: admitted but does NOT implement %q — this is the silent-wrong-output bug", be, name, r)
				}
			}
			if admitted {
				admits = append(admits, be)
			}
		}
		t.Logf("%-12s needs %-58v → admitted by %v", name, req, admits)
	}
}

// TestResidentBackendFeatures_noOverclaim pins each backend's declared coverage. A backend
// adding a claim here without shipping the kernel is exactly the lie the admission gate exists
// to catch, so widening a set must be a deliberate, reviewed edit — not a drive-by.
func TestResidentBackendFeatures_noOverclaim(t *testing.T) {
	want := map[string][]ResidentFeature{
		// cgo-free CUDA implements the plain dense block ONLY (verified against the kernels:
		// rmsnorm_quant has no (1+w); rope hardcodes half = hd/2; attention starts at key 0;
		// nothing reads QNorm/KNorm).
		"cuda": {},
		"webgpu": {
			FeatQKNorm, FeatPartialRotary, FeatSlidingWindow, FeatPerLayerRoPE, FeatRopeMscale,
			FeatMoE, FeatMLA, FeatSSM, FeatNonGatedMLP, FeatLogitScale, FeatRMSAddOne,
		},
		"metal": {FeatQKNorm, FeatSlidingWindow, FeatPartialRotary},
	}
	for be, exp := range want {
		got := ResidentBackendFeatures[be]
		if len(got) != len(exp) {
			t.Errorf("%s declares %d features, expected %d — if a kernel really landed, update this "+
				"test deliberately; %v", be, len(got), len(exp), got)
		}
		for _, f := range exp {
			if !got[f] {
				t.Errorf("%s no longer declares %q", be, f)
			}
		}
	}
	// Every declared feature must exist in the taxonomy (catches a typo'd string constant).
	known := map[ResidentFeature]bool{
		FeatQKNorm: true, FeatSlidingWindow: true, FeatPartialRotary: true, FeatPerLayerRoPE: true,
		FeatRopeMscale: true, FeatRMSAddOne: true, FeatEmbedScale: true, FeatLogitSoftcap: true,
		FeatSandwichNorm: true, FeatNonGatedMLP: true, FeatLearnedPos: true, FeatOutBias: true,
		FeatLogitScale: true, FeatMoE: true, FeatMLA: true, FeatSSM: true,
	}
	for be, set := range ResidentBackendFeatures {
		for f := range set {
			if !known[f] {
				t.Errorf("%s declares unknown feature %q", be, f)
			}
		}
	}
}

// TestResidentFeatures_derivation checks the arch→feature derivation directly: set one flag,
// expect one feature. If a future Architecture field changes the math, it needs a case here.
func TestResidentFeatures_derivation(t *testing.T) {
	base := func() *Architecture {
		return &Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: NormPre2}
	}
	cases := []struct {
		name string
		mut  func(*Architecture)
		want ResidentFeature
	}{
		{"qk-norm", func(a *Architecture) { a.QKNorm = true }, FeatQKNorm},
		{"sliding-window", func(a *Architecture) { a.SlidingWindow = 4096 }, FeatSlidingWindow},
		{"partial-rotary", func(a *Architecture) { a.RotaryDim = 64 }, FeatPartialRotary},
		{"rms-add-one", func(a *Architecture) { a.RMSAddOne = true }, FeatRMSAddOne},
		{"embed-scale", func(a *Architecture) { a.EmbedScale = 32 }, FeatEmbedScale},
		{"softcap", func(a *Architecture) { a.FinalLogitSoftcap = 30 }, FeatLogitSoftcap},
		{"sandwich", func(a *Architecture) { a.NormPlacement = NormSandwich4 }, FeatSandwichNorm},
		{"non-gated", func(a *Architecture) { a.NonGatedMLP = true }, FeatNonGatedMLP},
		{"learned-pos", func(a *Architecture) { a.LearnedPosEmbed = true }, FeatLearnedPos},
		{"out-bias", func(a *Architecture) { a.OutBias = true }, FeatOutBias},
		{"logit-scale", func(a *Architecture) { a.LogitScale = 8 }, FeatLogitScale},
		{"moe", func(a *Architecture) { a.MoE = &MoEConfig{} }, FeatMoE},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := base()
			c.mut(a)
			got := a.residentFeatures()
			if !slices.Contains(got, c.want) {
				t.Errorf("derived %v, want it to include %q", got, c.want)
			}
			// A plain dense arch must require nothing — else every backend declines everything.
			if plain := base().residentFeatures(); len(plain) != 0 {
				t.Errorf("plain dense arch requires %v, want none", plain)
			}
		})
	}
}

func mapKeys[K comparable, V any](m map[K]V) func(func(K) bool) {
	return func(yield func(K) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}
