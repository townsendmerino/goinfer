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
	// SmolLM3: llama-shaped plus per-layer NoPE (FeatNoPE). No resident backend declares
	// FeatNoPE at all yet (cohere2, the only other family needing it, is also CPU-only), so
	// this is CPU-only too — not a new gap, the SAME undeclared feature cohere2 already has.
	"smollm3": {FeatNoPE},
	// Olmo 3: NormPostOnly + QKNormWhole, both brand new this pass, plus the standard
	// sliding-window/YaRN features every backend already has. Neither new feature is declared
	// anywhere, so this is CPU-only regardless of the rest.
	// FeatPerLayerRoPE fires too: YaRN applies to full_attention layers only, so the local
	// (sliding) and global (full) inv-freq tables genuinely differ (see olmo3Architecture's own
	// comment on the real config's flat-rope_scaling-applies-to-full-only finding).
	"olmo3": {FeatPerLayerRoPE, FeatPostOnlyNorm, FeatQKNormWhole, FeatRopeMscale, FeatSlidingWindow},
	// InternLM3 is a llama alias: same descriptor, so the same (empty) feature profile.
	// Its dynamic-NTK rope resolves to no scaling at all in-window, so it does not even
	// need FeatRopeMscale.
	// InternLM2 needs no resident FEATURE the others lack: its differences are all in the
	// loader (renamed tensors + the grouped wqkv split), so by the time a descriptor exists it
	// is a llama. Same empty profile, same backends.
	"internlm2": {},
	"internlm3": {},
	// Dense Granite 4.2: llama-shaped, and EMPTY on purpose, not by omission — checked against
	// all three released sizes (3b/8b/30b), which all ship embedding_multiplier and
	// logits_scaling at their identity value 1.0 (FeatEmbedScale/FeatLogitScale only trigger
	// above/away-from 1, per their own derivation in this file), and residual_multiplier is
	// forced to 1.0 by validateGraniteDense. attention_multiplier is NOT gated by any
	// ResidentFeature flag at all — it is baked into AttnScale, which every backend already
	// reads generically regardless of value. RequiredResidentFeatures still derives the true
	// per-model requirement, so a future release with a non-identity multiplier is caught
	// there even though this table (correctly) shows nothing.
	"granite": {},
	// NOTE phi3 is config-DEPENDENT: Phi-4 and the released GGUFs are plain dense, while the
	// Phi-3-mini-4k safetensors declares sliding_window: 2047 and so also needs
	// FeatSlidingWindow. This table states the BASE profile; the authoritative requirement is
	// derived per-model at load (RequiredResidentFeatures), which is what admission uses. The
	// table's job is to force a new arch to be classified, not to be the runtime check.
	"phi3":    {},
	"gpt2":    {FeatLayerNorm, FeatLearnedPos, FeatNonGatedMLP, FeatOutBias},
	"cohere":  {FeatLayerNorm, FeatLogitScale, FeatParallelBlock},
	"cohere2": {FeatLayerNorm, FeatLogitScale, FeatNoPE, FeatParallelBlock, FeatSlidingWindow},
	"mistral": {FeatSlidingWindow},
	// Ministral 3: no sliding window on the real releases (confirmed null), so the only feature
	// this family needs at all is the new attn-temp query scale — CPU-only until a resident
	// backend implements it (see FeatAttnTemp's own comment). Otherwise a plain GQA+YaRN model
	// needing nothing else any backend lacks.
	"mistral3":   {FeatAttnTemp},
	"ministral3": {FeatAttnTemp},
	"qwen3":      {FeatQKNorm},
	"mellum":     {FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow},
	"mixtral":    {FeatMoE},
	"qwen2_moe":  {FeatMoE, FeatMoEGatedShared}, // sigmoid-gated always-on shared expert
	// qwen3_moe: qwen3's QK-norm attention + a routed MoE FFN with NO shared expert (confirmed
	// against the real Qwen3-30B-A3B config.json), so unlike qwen2_moe it does NOT need
	// FeatMoEGatedShared.
	"qwen3_moe": {FeatMoE, FeatQKNorm},
	"glm4_moe":  {FeatMoE, FeatPartialRotary, FeatQKNorm},
	// Laguna: sigmoid-routed MoE with an UNGATED shared expert (so FeatMoE, not
	// FeatMoEGatedShared), QK-norm, partial rotary on the full-attention layers, a
	// sliding/full interleave with per-layer RoPE bases, YaRN mscale on the full
	// layers, and the family-specific output gate. The XS generations carry all of
	// these; M.1 drops the sliding-window half, but this table states the family's
	// BASE profile and RequiredResidentFeatures derives the per-model truth.
	"laguna":      {FeatAttnOutputGate, FeatMoE, FeatPartialRotary, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow},
	"deepseek_v2": {FeatMLA, FeatMoE},
	"deepseek_v3": {FeatMLA, FeatMoE},
	"kimi_k2":     {FeatMLA, FeatMoE},
	"qwen3_5_moe": {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
	// Qwen3.8 dense (qwen3_5): the MoE sibling's profile MINUS the two MoE features. The
	// remaining three are unchanged and each was checked against the released 27B rather
	// than inherited — partial rotary (0.25 × head_dim 256 = 64 < 256), q_norm/k_norm on
	// every softmax layer, and Gemma-style (1+w) RMSNorm. The DeltaNet mixer itself needs
	// no feature here because decodeRunnerEligible refuses the whole arch.qwen35 family
	// upstream (own forward, not yet bridged) — the same posture qwen3_5_moe has.
	"qwen3_5":      {FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
	"qwen3_5_text": {FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
	// Qwen3-Next: same profile as qwen3_5_moe (verified, not assumed — its MoE block
	// (Qwen3NextSparseMoeBlock(Qwen2MoeSparseMoeBlock): pass) directly inherits
	// Qwen2-MoE's gated-shared-expert combination, and its RMSNorm
	// (Qwen3NextRMSNorm(Gemma3RMSNorm): pass) inherits Gemma's (1+w)).
	"qwen3_next":       {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
	"llama4_text":      {FeatMoE},
	"gpt_oss":          {FeatAttnSink, FeatMoE, FeatOutBias, FeatRopeMscale, FeatSlidingWindow},
	"nemotron_h":       {FeatNonGatedMLP, FeatSSM},
	"granitemoehybrid": {FeatLogitScale, FeatMoE, FeatSSM},
	// LFM2/LFM2.5: FeatShortConv is what keeps this CPU-only. Strip it and the profile is
	// {FeatQKNorm} — which every resident backend implements, so all three would admit a
	// family none of them can run. Same shape as laguna's FeatAttnOutputGate above.
	"lfm2":             {FeatQKNorm, FeatShortConv},
	"qwen3_5_moe_text": {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
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
// admissionGolden is the INDEPENDENT, hand-reviewed (arch → backends that admit it as resident) matrix.
// The old TestResidentAdmission_matrix was tautological — it re-derived admission from the same
// missingFeatures/residentBackendFeatures it was checking, so its only assertion was unreachable and it
// stayed green even if a backend's feature set were emptied (audit G-01). This golden is maintained by
// hand against docs + intent, so a change to archFeatureProfile or residentBackendFeatures that alters
// which family a backend runs resident fails the test and forces a deliberate golden edit + review.
// Regenerate the CANDIDATE list with `go test -run TestResidentAdmission_matrix -v` (the T logs it),
// then hand-verify each row before pasting — do NOT auto-write it, that would restore the tautology.
var admissionGolden = map[string][]string{
	"cohere":              {},
	"cohere2":             {},
	"deepseek_v2":         {"webgpu"},
	"deepseek_v3":         {"webgpu"},
	"gemma3":              {"cuda", "metal"},
	"gemma3_text":         {"cuda", "metal"},
	"gemma4":              {"cuda", "metal"},
	"gemma4_text":         {"cuda", "metal"},
	"gemma4_unified_text": {"cuda", "metal"},
	"glm4_moe":            {"cuda", "metal", "webgpu"},
	// Laguna: NO resident backend. Its softplus attention output gate and per-layer
	// query-head count are unimplemented everywhere, and both are silent failures if
	// skipped (the gate multiplies the whole attention context; a wrong head count
	// mis-shapes q/o). CPU-only until a bridge lands.
	"laguna": {},
	// lfm2: no resident backend implements the gated short conv (FeatShortConv) or its
	// rolling window, so every one declines. CPU-only until a bridge lands.
	"lfm2": {},
	"gpt2": {"metal"},
	// gpt_oss reaches BOTH backends on real end-to-end evidence now, which is not how it looked
	// for most of G7's life. metal declared on the tiny fixture (TestGptOssResidentParity, cosine
	// 0.9989); cuda declared 2026-08-31 on the REAL 20B, resident on an 8 GB card through
	// --moe-cache-experts (7/8 argmax-exact, min cosine 0.996392). The cuda half is the stronger
	// evidence of the two — an odd inversion worth noting, since the tiny fixture stopped
	// reproducing the last defect it had to catch.
	"gpt_oss":          {"cuda", "metal"},
	"granitemoehybrid": {"webgpu"},
	"kimi_k2":          {"webgpu"},
	"llama":            {"cuda", "metal", "webgpu"},
	"smollm3":          {}, // FeatNoPE undeclared everywhere, same as cohere2
	"olmo3":            {}, // FeatPostOnlyNorm/FeatQKNormWhole undeclared everywhere -- both new this pass
	"internlm2":        {"cuda", "metal", "webgpu"},
	"internlm3":        {"cuda", "metal", "webgpu"},
	// Dense Granite 4.2: empty feature profile (see archFeatureProfile's note), so it is
	// admitted everywhere llama is — same backends, same reason.
	"granite":     {"cuda", "metal", "webgpu"},
	"llama4_text": {"cuda", "metal", "webgpu"},
	// mellum reaches cuda and metal by the SAME coupling and with VERY DIFFERENT evidence, so
	// the two are not interchangeable rows:
	//   - cuda (added 2026-08-31, G7): FeatRopeMscale was declared for gpt-oss's YaRN, and
	//     mellumArchitecture needs exactly {MoE, PerLayerRoPE, QKNorm, RopeMscale, SlidingWindow}
	//     — the other four were already declared, so the fifth admitted Mellum for free. That
	//     coupling was PRE-REGISTERED as a trap and then discharged by measurement rather than
	//     waived: TestMellumResidentParityCUDA on a real 4-layer slice gives 7/11 argmax-exact,
	//     0 hard fails, min cosine 0.994600. Mellum on CUDA is a MEASURED admission.
	//   - metal: still the undischarged version of that same side effect (G10) — no Mellum
	//     checkpoint was reachable there, so it was resolved by an owner call. G11 tracks the
	//     real-weight Metal proof. It is here because it IS what the code does, not because it
	//     is trusted.
	"mellum":  {"cuda", "metal", "webgpu"},
	"mistral": {"cuda", "metal", "webgpu"},
	// Ministral 3: CPU-only. FeatAttnTemp is brand new to this pass and no backend declares it,
	// so every one correctly declines rather than silently dropping the attn-temp scale.
	"mistral3":   {},
	"ministral3": {},
	"mixtral":    {"cuda", "metal", "webgpu"},
	"nemotron_h": {"webgpu"},
	"phi3":       {"cuda", "metal", "webgpu"},
	"qwen2":      {"cuda", "metal", "webgpu"},
	"qwen2_5_vl": {"cuda", "metal", "webgpu"},
	"qwen2_moe":  {"cuda", "metal", "webgpu"}, // cuda joined 2026-08-20 (the gate weight, not a kernel)
	"qwen3":      {"cuda", "metal", "webgpu"},
	// qwen3_moe needs {FeatMoE, FeatQKNorm} — strictly WEAKER than qwen2_moe's
	// {FeatMoE, FeatMoEGatedShared} (no shared expert to gate) and the same FeatQKNorm
	// qwen3 dense already clears on all three backends. Since qwen2_moe and qwen3 are
	// both already admitted everywhere on those exact features, qwen3_moe's subset is
	// too — no new kernel needed, same generic MoE + QK-norm dispatch every backend
	// already has.
	"qwen3_moe": {"cuda", "metal", "webgpu"},
	// The Gated-DeltaNet family collapses to the backends that implement BOTH the recurrence AND
	// the fused attention output gate (FeatDeltaNet) — the whole point of adding it as one taxon
	// bundling both departures rather than two. CUDA and WebGPU landed it first (2026-08-19/20);
	// Metal landed the recurrence kernels + the qGate softmax-layer wiring + a whole-model gate
	// against qwen3_5-tiny (worst cosine 0.9886, drift 0.0081, replay-after-Reset self-cosine
	// 1.0 — TestQwen35ResidentParityMetal) and now admits too. CUDA still lacks it.
	"qwen3_5":          {"cuda", "metal", "webgpu"},
	"qwen3_5_text":     {"cuda", "metal", "webgpu"},
	"qwen3_5_moe":      {"cuda", "metal", "webgpu"},
	"qwen3_5_moe_text": {"cuda", "metal", "webgpu"},
	"qwen3_next":       {"cuda", "metal", "webgpu"}, // same arch.qwen35 != nil bridge qwen3_5_moe uses — reused directly, not a new one
}

func TestResidentAdmission_matrix(t *testing.T) {
	backends := []string{"cuda", "metal", "webgpu"}
	seen := map[string]bool{}
	for _, name := range slices.Sorted(mapKeys(archFeatureProfile)) {
		req := archFeatureProfile[name]
		var admits []string
		for _, be := range backends {
			impl, ok := residentBackendFeatures[be]
			if !ok {
				t.Fatalf("backend %q has no declared feature set", be)
			}
			if len(missingFeatures(req, impl)) == 0 {
				admits = append(admits, be)
			}
		}
		slices.Sort(admits)
		t.Logf("%-20s → admitted by %v", name, admits) // the candidate row, for regenerating the golden

		want, ok := admissionGolden[name]
		if !ok {
			t.Errorf("%s: no admissionGolden row — a new family must be added to the reviewed golden", name)
			continue
		}
		seen[name] = true
		wantSorted := slices.Clone(want)
		slices.Sort(wantSorted)
		if !slices.Equal(admits, wantSorted) {
			t.Errorf("%s: admitted by %v, golden says %v — a backend's feature set or the arch's needs changed; "+
				"verify this is intended and update admissionGolden", name, admits, wantSorted)
		}
	}
	for name := range admissionGolden {
		if !seen[name] {
			t.Errorf("admissionGolden has a stale row %q not in archFeatureProfile", name)
		}
	}
}

// TestResidentBackendFeatures_noOverclaim pins each backend's declared coverage. A backend
// adding a claim here without shipping the kernel is exactly the lie the admission gate exists
// to catch, so widening a set must be a deliberate, reviewed edit — not a drive-by.
func TestResidentBackendFeatures_noOverclaim(t *testing.T) {
	want := map[string][]ResidentFeature{
		// cgo-free CUDA: dense + QK-norm + sliding window + the Gemma set + partial rotary + MoE
		// (routed via mixtral-tiny, ungated shared expert via glm-tiny) + the FINAL-logit softcap
		// (host-side tanh in step(), 9a-P2 — Gemma 4). FeatRopeMscale landed 2026-08-31 (G7): the
		// YaRN attention_factor is folded into cos/sin inside all three rope kernels (glue.cu,
		// gemv_fwd.cu, prefill_batched.cu) with per-layer wiring from RopeMscaleLayer, and it is
		// gated by TestRopeMscale plus a real-weight TestMellumResidentParityCUDA (4-layer slice:
		// 7/11 argmax-exact, 0 hard fails, min cosine 0.994600) — evidence first, declaration
		// after. FeatAttnSink + FeatOutBias joined 2026-08-31 (G7), and ONLY after a real
		// gpt-oss-20b forward ran resident on an 8 GB card via --moe-cache-experts: 7/8
		// argmax-exact, min cosine 0.996392 (TestGptOssResidentParityCUDA). 2224441 declared
		// FeatAttnSink once on kernel-level evidence and was correctly reverted; the difference
		// this time is the end-to-end run, which found three silent wiring defects no kernel test
		// could see. Still no per-layer rotary WIDTH, no ATTENTION softcap (FeatAttnLogitSoftcap —
		// a per-layer kernel), no MLA/SSM, and no GATED shared expert (Qwen2-MoE)... which is now
		// expressed as FeatMoEGatedShared (which CUDA does NOT declare), so the Qwen2-MoE decline
		// is in the shared taxonomy, not a hand-coded check.
		"cuda": {FeatQKNorm, FeatSlidingWindow, FeatPartialRotary, FeatRMSAddOne, FeatSandwichNorm,
			FeatGatedGELU, FeatEmbedScale, FeatPerLayerRoPE, FeatMoE, FeatFinalLogitSoftcap,
			FeatDeltaNet, FeatMoEGatedShared, FeatRopeMscale, FeatAttnSink, FeatOutBias},
		"webgpu": {
			FeatQKNorm, FeatPartialRotary, FeatSlidingWindow, FeatPerLayerRoPE, FeatRopeMscale,
			FeatMoE, FeatMoEGatedShared, FeatMLA, FeatSSM, FeatDeltaNet, FeatNonGatedMLP, FeatLogitScale,
			FeatRMSAddOne,
		},
		// GPT-2 (2026-08-18): LayerNorm/non-gated-MLP/learned-pos/out-bias all landed as real
		// kernels + full BuildResident/encodeLayer/encodeAttention wiring, validated end-to-end
		// against the real checkpoint (TestGPT2ResidentParity, min cosine 0.999) — not a
		// declaration ahead of the evidence. gpt-oss's attention sink + clamped-SwiGLU MoE +
		// custom router + YaRN rope mscale also landed (TestGptOssResidentParity, min cosine
		// 0.9989 on the tiny fixture) — FeatRopeMscale ALSO admits Mellum as a documented side
		// effect (see docs/queue-correctness.md G10/G11): declared anyway on explicit user call,
		// Mellum's own real-weight Metal gate is tracked as G11, not yet landed. Gated-DeltaNet
		// mixer + fused attn output gate (deltanet.go/deltanet_kernels.go) landed with its own
		// end-to-end whole-model gate (TestQwen35ResidentParityMetal, qwen3_5-tiny: worst cosine
		// 0.9886, drift 0.0081, replay-after-Reset self-cosine 1.0) — not ahead of it.
		"metal": {
			FeatQKNorm, FeatSlidingWindow, FeatPartialRotary, FeatMoE, FeatMoEGatedShared, FeatSandwichNorm,
			FeatGatedGELU, FeatRMSAddOne, FeatEmbedScale, FeatPerLayerRoPE, FeatFinalLogitSoftcap,
			FeatLayerNorm, FeatNonGatedMLP, FeatLearnedPos, FeatOutBias, FeatRopeMscale, FeatAttnSink,
			FeatDeltaNet,
		},
	}
	for be, exp := range want {
		got := residentBackendFeatures[be]
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
		FeatAttnSink: true, FeatGemma4EModel: true, FeatDeltaNet: true, // N-12: were omitted, so declaring either failed with a misleading "unknown feature"
	}
	for be, set := range residentBackendFeatures {
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
		{"gemma4-e-model-ple", func(a *Architecture) { a.gemma4 = &gemma4Params{HiddenSizePerLayerInput: 256} }, FeatGemma4EModel},
		{"gemma4-e-model-sharedkv", func(a *Architecture) { a.gemma4 = &gemma4Params{SharedKVLayers: 2} }, FeatGemma4EModel},
		{"gemma4-e-model-ffnperlayer", func(a *Architecture) { a.gemma4 = &gemma4Params{FFNPerLayer: []int{1, 2}} }, FeatGemma4EModel},
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
	if miss := missingFeatures(arch.residentFeatures(), residentBackendFeatures["webgpu"]); len(miss) != 0 {
		t.Fatalf("kimi_k2 is missing WebGPU features %v — this test can no longer isolate the router cap", miss)
	}
	// The cap was raised 256 -> 512 (MOE_MAX_E / MAXE) so Kimi-K2's 384 is now ADMITTED on the two
	// backends whose router scratch was widened. This is the point of the change: K2 is the shipped
	// family the old cap declined.
	if !residentMoECapacityOK(arch, "webgpu") {
		t.Error("kimi_k2 (384 experts) must now pass the WebGPU router cap (raised to 512)")
	}
	if !residentMoECapacityOK(arch, "cuda") {
		t.Error("kimi_k2 (384 experts) must now pass the CUDA router cap (raised to 512)")
	}
	// ...and WebGPU is where that actually matters: it is the only backend declaring FeatMLA, so
	// K2 is genuinely resident-eligible there now. On cuda/metal it still declines, on FEATURES not
	// capacity — asserted below so a future MLA-on-CUDA leg does not mistake this for already-done.
	if !ResidentEligible(arch, "webgpu") {
		t.Error("kimi_k2 must be WebGPU-resident-eligible once the router cap admits 384")
	}
	if ResidentEligible(arch, "cuda") {
		t.Error("kimi_k2 must still decline on CUDA — it needs FeatMLA, which CUDA does not implement")
	}
	if miss := missingFeatures(arch.residentFeatures(), residentBackendFeatures["cuda"]); len(miss) == 0 {
		t.Error("the CUDA decline for kimi_k2 must be a FEATURE decline (MLA); if this is empty the " +
			"cap is now the only guard and the assertion above is testing the wrong thing")
	}

	// Metal is DECLARED in the cap map for the first time (its shader really is 256 and rejects
	// above it, metal/moe.go:206) — the map previously claimed "absent = uncapped", which was false.
	if residentMoECapacityOK(arch, "metal") {
		t.Error("kimi_k2 (384 experts) must fail Metal's router cap — its shader is still float score[256]")
	}

	// Boundary: at/under the cap the same arch is admitted (proves it's the count, not the family).
	atCap := *arch.MoE
	atCap.NumExperts = 512
	capped := *arch
	capped.MoE = &atCap
	if !residentMoECapacityOK(&capped, "webgpu") {
		t.Error("512 experts is at the cap and must be admitted (off-by-one in the router-cap check)")
	}
	over := atCap
	over.NumExperts = 513
	capped.MoE = &over
	if residentMoECapacityOK(&capped, "webgpu") {
		t.Error("513 experts must decline (cap is 512)")
	}
	if residentMoECapacityOK(&capped, "cuda") {
		t.Error("513 experts must decline on CUDA too (MOE_MAX_E is 512; past it moe_route writes out of bounds)")
	}
	// Kimi-K3's 896 must STILL decline everywhere — the cap was deliberately not stretched to an
	// unbuilt family, and a future reader must not assume otherwise.
	k3 := atCap
	k3.NumExperts = 896
	capped.MoE = &k3
	for _, be := range []string{"webgpu", "cuda", "metal"} {
		if residentMoECapacityOK(&capped, be) {
			t.Errorf("896 experts (Kimi-K3 class) must decline on %s — 512 was chosen deliberately", be)
		}
	}

	// A backend with no fixed-size router cap is unaffected (dense archs and cap-less backends pass).
	dense, _, _ := resolveArchitecture(representativeConfig("qwen3"))
	if !residentMoECapacityOK(dense, "webgpu") {
		t.Error("a dense (non-MoE) arch must never be declined by the router cap")
	}
}
