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
	"gpt2":             {FeatLayerNorm, FeatLearnedPos, FeatNonGatedMLP, FeatOutBias},
	"cohere":           {FeatLayerNorm, FeatLogitScale, FeatParallelBlock},
	"cohere2":          {FeatLayerNorm, FeatLogitScale, FeatNoPE, FeatParallelBlock, FeatSlidingWindow},
	"mistral":          {FeatSlidingWindow},
	"qwen3":            {FeatQKNorm},
	"mellum":           {FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow},
	"mixtral":          {FeatMoE},
	"qwen2_moe":        {FeatMoE, FeatMoEGatedShared}, // sigmoid-gated always-on shared expert (CUDA declines it)
	"glm4_moe":         {FeatMoE, FeatPartialRotary, FeatQKNorm},
	"deepseek_v2":      {FeatMLA, FeatMoE},
	"deepseek_v3":      {FeatMLA, FeatMoE},
	"kimi_k2":          {FeatMLA, FeatMoE},
	"qwen3_5_moe":      {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne},
	"llama4_text":      {FeatMoE},
	"gpt_oss":          {FeatAttnSink, FeatMoE, FeatOutBias, FeatRopeMscale, FeatSlidingWindow},
	"nemotron_h":       {FeatNonGatedMLP, FeatSSM},
	"granitemoehybrid": {FeatLogitScale, FeatMoE, FeatSSM},
	"qwen3_5_moe_text": {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne},
	// Gemma — VERIFIED against the real checkpoints via RequiredResidentFeatures (an earlier
	// hand-written guess here was wrong on three counts: it missed per-layer-rope / qk-norm /
	// sliding-window, and claimed rms-add-one for gemma4, which has RMSAddOne=false).
	// gemma3/gemma3_text are resident on CUDA. gemma4 needs the FINAL-logit softcap (30) — one
	// host-side tanh, which CUDA now ships (FeatFinalLogitSoftcap, 9a-P2) — not the attention
	// softcap. So it is feature-compatible with CUDA; its OWN forward (per-layer head_dim / K=V)
	// is what the resident geometry bridge addresses, and dense admission is env-gated
	// (GOINFER_GEMMA4_RESIDENT) in decodeRunnerEligible — a separate gate from this feature set.
	"gemma3":      {FeatEmbedScale, FeatGatedGELU, FeatPerLayerRoPE, FeatQKNorm, FeatRMSAddOne, FeatSandwichNorm, FeatSlidingWindow},
	"gemma3_text": {FeatEmbedScale, FeatGatedGELU, FeatPerLayerRoPE, FeatQKNorm, FeatRMSAddOne, FeatSandwichNorm, FeatSlidingWindow},
	"gemma4":      {FeatEmbedScale, FeatFinalLogitSoftcap, FeatGatedGELU, FeatPerLayerRoPE, FeatQKNorm, FeatSandwichNorm, FeatSlidingWindow},
	// gemma4_text is the 26B-A4B MoE variant: gemma4's feature set + FeatMoE. It is
	// FEATURE-compatible with CUDA, but Gemma 4's MoE is the parallel dense‖MoE shape the
	// generic FeatMoE kernel cannot express, so decodeRunnerEligible declines a.MoE != nil
	// until Split B lands the delta — the feature set is necessary, not sufficient.
	"gemma4_text": {FeatEmbedScale, FeatFinalLogitSoftcap, FeatGatedGELU, FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatSandwichNorm, FeatSlidingWindow},
	// gemma4_unified_text — the real unified checkpoints' text_config model_type;
	// same feature set as gemma4_text (K=V globals are a loader detail, not a
	// resident feature).
	"gemma4_unified_text": {FeatEmbedScale, FeatFinalLogitSoftcap, FeatGatedGELU, FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatSandwichNorm, FeatSlidingWindow},
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
		// cgo-free CUDA: dense + QK-norm + sliding window + the Gemma set + partial rotary + MoE
		// (routed via mixtral-tiny, ungated shared expert via glm-tiny) + the FINAL-logit softcap
		// (host-side tanh in step(), 9a-P2 — Gemma 4). Still no per-layer rotary WIDTH, no YaRN
		// mscale, no ATTENTION softcap (FeatAttnLogitSoftcap — a per-layer kernel), no MLA/SSM, and
		// no GATED shared expert (Qwen2-MoE) — now expressed as FeatMoEGatedShared (which CUDA does
		// NOT declare), so the Qwen2-MoE decline is in the shared taxonomy, not a hand-coded check.
		"cuda": {FeatQKNorm, FeatSlidingWindow, FeatPartialRotary, FeatRMSAddOne, FeatSandwichNorm, FeatGatedGELU, FeatEmbedScale, FeatPerLayerRoPE, FeatMoE, FeatFinalLogitSoftcap},
		"webgpu": {
			FeatQKNorm, FeatPartialRotary, FeatSlidingWindow, FeatPerLayerRoPE, FeatRopeMscale,
			FeatMoE, FeatMoEGatedShared, FeatMLA, FeatSSM, FeatNonGatedMLP, FeatLogitScale, FeatRMSAddOne,
		},
		"metal": {FeatQKNorm, FeatSlidingWindow, FeatPartialRotary, FeatMoE, FeatMoEGatedShared, FeatSandwichNorm, FeatGatedGELU, FeatRMSAddOne, FeatEmbedScale, FeatPerLayerRoPE},
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
		FeatRopeMscale: true, FeatRMSAddOne: true, FeatEmbedScale: true,
		FeatAttnLogitSoftcap: true, FeatFinalLogitSoftcap: true,
		FeatSandwichNorm: true, FeatGatedGELU: true, FeatNonGatedMLP: true, FeatLearnedPos: true,
		FeatOutBias: true, FeatLogitScale: true, FeatMoE: true, FeatMoEGatedShared: true,
		FeatMLA: true, FeatSSM: true, FeatLayerNorm: true, FeatParallelBlock: true, FeatNoPE: true,
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
	// A plain dense Llama/Qwen2 block. Act is set EXPLICITLY: ActGeluTanh is ActKind's zero
	// value (iota), so a default-constructed Architecture reads as GELU, not SiLU.
	base := func() *Architecture {
		return &Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: NormPre2, Act: ActSiLU}
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
		{"final-softcap", func(a *Architecture) { a.FinalLogitSoftcap = 30 }, FeatFinalLogitSoftcap},
		{"attn-softcap", func(a *Architecture) { a.AttnLogitSoftcap = 30 }, FeatAttnLogitSoftcap},
		{"sandwich", func(a *Architecture) { a.NormPlacement = NormSandwich4 }, FeatSandwichNorm},
		{"layer-norm", func(a *Architecture) { a.Norm = NormLayer }, FeatLayerNorm},
		{"parallel-block", func(a *Architecture) { a.NormPlacement = NormParallel }, FeatParallelBlock},
		{"nope", func(a *Architecture) { a.layerNoPE = func(int) bool { return true } }, FeatNoPE},
		{"gated-gelu", func(a *Architecture) { a.Act = ActGeluTanh }, FeatGatedGELU},
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

// TestResidentFeatures_derivationMatchesProfile ties the two sources of truth together (C6):
// the hand-written archFeatureProfile (the classification forcing-function) and residentFeatures()
// (the derivation that drives the generated hardware matrix AND the runtime RequiredResidentFeatures).
// They must agree for every registered arch — otherwise they silently disagree, as they did on
// Mellum (a layer-0-only yarn-mscale sample missed its full-layer YaRN) and qwen2_moe (the gated
// shared expert). Hardware-free — reads declared sets, runs in CI with no GPU.
func TestResidentFeatures_derivationMatchesProfile(t *testing.T) {
	for name := range archFeatureProfile {
		cfg := representativeConfig(name)
		if cfg == nil {
			t.Errorf("%q: has a feature profile but no representativeConfig — add one so the derivation is cross-checked", name)
			continue
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			t.Errorf("%q: resolveArchitecture: %v", name, err)
			continue
		}
		got := arch.residentFeatures()
		want := slices.Clone(archFeatureProfile[name])
		slices.Sort(want)
		if !slices.Equal(got, want) {
			t.Errorf("%q: residentFeatures()=%v but archFeatureProfile=%v — derivation and hand table DISAGREE (fix whichever is wrong)", name, got, want)
		}
	}
}

// TestResidentMoECapacity_routerCap is the M22 taxonomy gate (kimi_k2 doc-drift). WebGPU/CUDA score
// experts into a fixed-size array (256) and groups into 32; a model past that routes on only the
// first N — plausible-looking wrong output — so ResidentEligible must decline it even though every
// FEATURE it needs is implemented. This is what keeps the generated hardware matrix honest: before
// residentBackendMoECap, the feature-only predicate showed Kimi K2 (384 experts) WebGPU-resident
// while the runtime (gpu/backend.go, M22) declined it.
func TestResidentMoECapacity_routerCap(t *testing.T) {
	arch, _, err := resolveArchitecture(representativeConfig("kimi_k2"))
	if err != nil {
		t.Fatalf("resolve kimi_k2: %v", err)
	}
	if arch.MoE == nil {
		t.Fatal("kimi_k2 arch has no MoE — representativeConfig regressed")
	}
	if arch.MoE.NumExperts <= 256 {
		t.Fatalf("representativeConfig(kimi_k2) has %d experts; must carry the real >256 count or this "+
			"gate is vacuous (see the C6/Mellum representativeConfig-accuracy lesson)", arch.MoE.NumExperts)
	}

	// The decline must be the CAP, not a missing feature: WebGPU implements everything kimi needs.
	if miss := missingFeatures(arch.residentFeatures(), ResidentBackendFeatures["webgpu"]); len(miss) != 0 {
		t.Fatalf("kimi_k2 is missing WebGPU features %v — this test can no longer isolate the router cap", miss)
	}
	if ResidentEligible(arch, "webgpu") {
		t.Error("kimi_k2 (384 experts) must NOT be WebGPU-resident-eligible — the router kernel caps at 256 (M22)")
	}
	if residentMoECapacityOK(arch, "webgpu") {
		t.Error("residentMoECapacityOK(kimi, webgpu) = true; 384 > 256 must fail")
	}

	// Boundary: at/under the cap the same arch is admitted (proves it's the count, not the family).
	atCap := *arch.MoE
	atCap.NumExperts = 256
	capped := *arch
	capped.MoE = &atCap
	if !residentMoECapacityOK(&capped, "webgpu") {
		t.Error("256 experts is at the cap and must be admitted (off-by-one in the router-cap check)")
	}
	over := atCap
	over.NumExperts = 257
	capped.MoE = &over
	if residentMoECapacityOK(&capped, "webgpu") {
		t.Error("257 experts must decline (cap is 256)")
	}

	// A backend with no fixed-size router cap is unaffected (dense archs and cap-less backends pass).
	dense, _, _ := resolveArchitecture(representativeConfig("qwen3"))
	if !residentMoECapacityOK(dense, "webgpu") {
		t.Error("a dense (non-MoE) arch must never be declined by the router cap")
	}
}
