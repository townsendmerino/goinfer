package decoder

import (
	"slices"
	"strings"
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
	// qwen3_vl (text only): aliases plain qwen3's attention shape (qwen3_vlArchitecture calls qwen3Architecture), so it
	// needs the same FeatQKNorm and nothing else: MRopeSection/MRopeInterleaved only diverge from scalar RoPE on an image
	// path that cache.mropePos never takes for ordinary Generate, same reasoning as qwen2_5_vl's empty profile above.
	"qwen3_vl": {FeatQKNorm},
	// qwen3_asr (S14.3): Qwen3 exactly (qwen3_asrArchitecture calls qwen3Architecture; the checkpoint's interleaved m-RoPE section is dropped because audio positions are sequential), so the same
	// FeatQKNorm and nothing else.
	"qwen3_asr": {FeatQKNorm},
	// voxtral (S14.4b): Llama exactly (voxtralArchitecture calls llamaArchitecture), so the same empty profile.
	"voxtral": {},
	"llama":   {},
	// SmolLM3: llama-shaped plus per-layer NoPE (FeatNoPE), its only required feature (see the admission golden below).
	// cohere2 also needs FeatNoPE, plus FeatLayerNorm/FeatLogitScale/FeatParallelBlock.
	"smollm3": {FeatNoPE},
	// Olmo 3: NormPostOnly + QKNormWhole, plus the standard sliding-window/YaRN features. FeatPerLayerRoPE IS needed
	// (docs/audit-2026-09-10.md G-03/G-04's disposition): YaRN applies to full_attention layers only, sliding_attention gets
	// plain RoPE at the same theta, genuinely different local/global tables. `Olmo3RotaryEmbedding.forward` in the pinned
	// transformers takes an explicit layer_type argument and returns different (cos, sin) per call, so a single shared table
	// cannot be what runs. See registry.go's olmo3Architecture comment for the full account.
	"olmo3": {FeatPostOnlyNorm, FeatQKNormWhole, FeatRopeMscale, FeatSlidingWindow, FeatPerLayerRoPE},
	// Olmo Hybrid: qwen3_5's own FeatDeltaNet (shared math), plus olmo3's FeatPostOnlyNorm/FeatQKNormWhole (its
	// full-attention layers only, but the arch-level check reads NormPlacement/QKNormWhole model-wide, not per-layer) and
	// FeatNoPE (layerNoPE set unconditionally: the release has no RoPE at all). Its full-attention layer uses a plain q
	// projection, not qwen3.5's double-width q-gate (Architecture.qwen35.AttnGate).
	"olmo_hybrid": {FeatDeltaNet, FeatNoPE, FeatPostOnlyNorm, FeatQKNormWhole},
	// InternLM3 is a llama alias: same descriptor, so the same (empty) feature profile.
	// Its dynamic-NTK rope resolves to no scaling at all in-window, so it does not even
	// need FeatRopeMscale.
	// InternLM2 needs no resident FEATURE the others lack: its differences are all in the
	// loader (renamed tensors + the grouped wqkv split), so by the time a descriptor exists it
	// is a llama. Same empty profile, same backends.
	"internlm2": {},
	"internlm3": {},
	// GLM-OCR text decoder: Gemma's FOUR-norm sandwich placement (FeatSandwichNorm) over plain RMSNorm and a plain SwiGLU
	// MLP, GQA with no bias or QK-norm, full-width rotary. NOTE what the feature taxonomy would otherwise miss: the rotation
	// is PAIRWISE (GPT-J) over m-RoPE sections and the generic resident rope kernels are NeoX half-split, so the family needs
	// FeatPairwiseRoPE (scalar rows, decode) AND FeatPairwiseMRoPE (image-block prefill); only a backend with pairwise rope
	// kernels declares them. See TestGlmOcr_residentDeclined.
	"glm_ocr":      {FeatPairwiseMRoPE, FeatPairwiseRoPE, FeatSandwichNorm},
	"glm_ocr_text": {FeatPairwiseMRoPE, FeatPairwiseRoPE, FeatSandwichNorm},
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
	"phi3": {},
	"gpt2": {FeatLayerNorm, FeatLearnedPos, FeatNonGatedMLP, FeatOutBias},
	// Cohere's rotation is GPT-J PAIRWISE (rope_gptj), not NeoX, so it needs FeatPairwiseRoPE: a resident backend that runs
	// it on NeoX kernels is wrong on the real checkpoints (flat-fixture gates are blind to it). Backends without a pairwise
	// rope variant decline these families.
	"cohere":  {FeatLayerNorm, FeatLogitScale, FeatPairwiseRoPE, FeatParallelBlock},
	"cohere2": {FeatLayerNorm, FeatLogitScale, FeatNoPE, FeatPairwiseRoPE, FeatParallelBlock, FeatSlidingWindow},
	"mistral": {FeatSlidingWindow},
	// Ministral 3: no sliding window on the real releases, so the only feature this family needs is the attn-temp query
	// scale (see FeatAttnTemp's own comment). Otherwise a plain GQA+YaRN model.
	"mistral3":   {FeatAttnTemp},
	"ministral3": {FeatAttnTemp},
	"qwen3":      {FeatQKNorm},
	"mellum":     {FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow},
	"mixtral":    {FeatMoE},
	"qwen2_moe":  {FeatMoE, FeatMoEGatedShared}, // sigmoid-gated always-on shared expert
	// qwen3_moe: qwen3's QK-norm attention + a routed MoE FFN with NO shared expert (confirmed
	// against the real Qwen3-30B-A3B config.json), so unlike qwen2_moe it does NOT need
	// FeatMoEGatedShared.
	"qwen3_moe":    {FeatMoE, FeatQKNorm},
	"qwen3_vl_moe": {FeatMoE, FeatQKNorm}, // qwen3_moe's needs; m-RoPE is the qwen3_vl image path's, as for qwen3_vl
	"glm4_moe":     {FeatMoE, FeatPartialRotary, FeatQKNorm},
	// Laguna: sigmoid-routed MoE with an UNGATED shared expert (so FeatMoE, not
	// FeatMoEGatedShared), QK-norm, partial rotary on the full-attention layers, a
	// sliding/full interleave with per-layer RoPE bases, YaRN mscale on the full
	// layers, and the family-specific output gate. The XS generations carry all of
	// these; M.1 drops the sliding-window half, but this table states the family's
	// BASE profile and RequiredResidentFeatures derives the per-model truth.
	"laguna": {FeatAttnOutputGate, FeatMoE, FeatPartialRotary, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow},
	// spark2_5: same attention-output-gate CPU-only forcing function as Laguna (FeatAttnOutputGate: sigmoid here, softplus
	// there; no resident backend implements either). It also derives FeatGatedGELU (!NonGatedMLP && Act != ActSiLU): the
	// first family combining a gated MLP with ActGelu (exact erf). Whether CUDA's glu_quant is correct for that activation
	// on the gated path is UNVERIFIED, and masked only because FeatAttnOutputGate keeps the family CPU-only
	// (docs/code-notes/decoder.md#archFeatureProfile.spark2_5).
	"spark2_5":    {FeatAttnOutputGate, FeatGatedGELU, FeatPartialRotary, FeatPerLayerRoPE, FeatSlidingWindow},
	"deepseek_v2": {FeatMLA, FeatMoE},
	"deepseek_v3": {FeatMLA, FeatMoE},
	"kimi_k2":     {FeatMLA, FeatMoE},
	// Bailing Hybrid (Ling 3.0): FeatKDA is declared by no backend, so the family is CPU-only although FeatMLA and FeatMoE
	// are ordinary DeepSeek-shaped features.
	"bailing_hybrid": {FeatMLA, FeatMoE, FeatKDA},
	"qwen3_5_moe":    {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
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
	"lfm2_vl":          {FeatQKNorm, FeatShortConv}, // S10: LFM2-VL's text_config is lfm2
	"qwen3_5_moe_text": {FeatMoE, FeatMoEGatedShared, FeatPartialRotary, FeatQKNorm, FeatRMSAddOne, FeatDeltaNet},
	// Gemma, VERIFIED against the real checkpoints via RequiredResidentFeatures. gemma4 needs the FINAL-logit softcap (one
	// host-side tanh, FeatFinalLogitSoftcap), not the attention softcap; its own forward (per-layer head_dim / K=V) is what
	// the resident geometry bridge addresses, a separate gate from this feature set. Gemma 1 / CodeGemma and Gemma 2
	// (gemma12.go): Gemma 2 needs the attention-score softcap, a per-layer kernel no resident backend ships, so it is
	// admitted nowhere and runs on the CPU until one does.
	"gemma":       {FeatEmbedScale, FeatGatedGELU, FeatRMSAddOne},
	"gemma2":      {FeatAttnLogitSoftcap, FeatEmbedScale, FeatFinalLogitSoftcap, FeatGatedGELU, FeatRMSAddOne, FeatSandwichNorm, FeatSlidingWindow},
	"gemma3":      {FeatEmbedScale, FeatGatedGELU, FeatPerLayerRoPE, FeatQKNorm, FeatRMSAddOne, FeatSandwichNorm, FeatSlidingWindow},
	"gemma3_text": {FeatEmbedScale, FeatGatedGELU, FeatPerLayerRoPE, FeatQKNorm, FeatRMSAddOne, FeatSandwichNorm, FeatSlidingWindow},
	"gemma4":      {FeatEmbedScale, FeatFinalLogitSoftcap, FeatGatedGELU, FeatPerLayerRoPE, FeatQKNorm, FeatSandwichNorm, FeatSlidingWindow},
	// gemma4_text is the 26B-A4B MoE variant: gemma4's feature set + FeatMoE. Gemma 4's MoE is the parallel dense‖MoE shape
	// the generic FeatMoE kernel cannot express, so the feature set is necessary, not sufficient: admission also consults
	// residentGemma4MoEBackends (features.go).
	"gemma4_text": {FeatEmbedScale, FeatFinalLogitSoftcap, FeatGatedGELU, FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatSandwichNorm, FeatSlidingWindow},
	// gemma4_unified_text is the real unified checkpoints' text_config model_type; same feature set as gemma4_text
	// (K=V globals are a loader detail, not a resident feature).
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

// admissionGolden is the INDEPENDENT, hand-reviewed (arch → backends that admit it as resident) matrix. A golden
// re-derived from missingFeatures/residentBackendFeatures would be tautological (it would stay green even if a
// backend's feature set were emptied), so this one is maintained by hand against docs + intent: a change to
// archFeatureProfile or residentBackendFeatures that alters which family a backend runs resident fails the test and
// forces a deliberate golden edit + review. Regenerate the CANDIDATE list with
// `go test -run TestResidentAdmission_matrix -v` (the T logs it), then hand-verify each row before pasting: do NOT
// auto-write it, that would restore the tautology.
var admissionGolden = map[string][]string{
	// Cohere / Command-R + Cohere2 / Command-R7B: FeatParallelBlock, FeatLogitScale and FeatLayerNorm (and for cohere2
	// FeatNoPE/FeatSlidingWindow) are declared on cuda and metal. Their rotation is GPT-J pairwise and WebGPU's rope kernel
	// is NeoX half-split, so webgpu declines (FeatPairwiseRoPE): a NeoX rotation reads near-zero cosine on the real
	// checkpoints while flat 0.02-std fixtures read 0.9997 (docs/tasks/task-metal-pairwise-rope-2026-10.md).
	"cohere":      {"cuda", "metal"},
	"cohere2":     {"cuda", "metal"},
	"deepseek_v2": {"cuda", "webgpu"},
	"deepseek_v3": {"cuda", "webgpu"},
	// gemma3 (uniform head_dim) reaches webgpu for real (testdata/gemma3-vl-tiny's text tower,
	// gpu.TestGemma3ResidentParityWebGPU). gemma4/gemma4_text/gemma4_unified_text are FEATURE-compatible there, but dense
	// Gemma 4's local/global head_dim split needs a per-layer geometry seam this feature-list model cannot express
	// (decoder.Model.PerLayerGeomOK, a runtime check outside the ResidentFeature taxonomy), so the REAL runtime
	// (decoder.ResidentEligible, TestGemma4Admission_unconditional) declines gemma4 on webgpu while this golden shows it
	// feature-admitted: it intentionally tracks the feature-only model, as with the MoE-cap precedent (deepseek_v2/kimi_k2
	// below, admitted by feature but capped elsewhere).
	"gemma":               {"cuda", "metal", "webgpu"},
	"gemma2":              {},
	"gemma3":              {"cuda", "metal", "webgpu"},
	"gemma3_text":         {"cuda", "metal", "webgpu"},
	"gemma4":              {"cuda", "metal", "webgpu"},
	"gemma4_text":         {"cuda", "metal", "webgpu"},
	"gemma4_unified_text": {"cuda", "metal", "webgpu"},
	"glm4_moe":            {"cuda", "metal", "webgpu"},
	// Laguna: NO resident backend. Its softplus attention output gate and per-layer
	// query-head count are unimplemented everywhere, and both are silent failures if
	// skipped (the gate multiplies the whole attention context; a wrong head count
	// mis-shapes q/o). CPU-only until a bridge lands.
	"laguna": {},
	// spark2_5: same reasoning as laguna directly above — its sigmoid attention output gate is
	// unimplemented on every resident backend, and skipping it silently would multiply the wrong
	// (unmodified) context into o_proj. CPU-only until a bridge lands.
	"spark2_5": {},
	// lfm2: no resident backend implements the gated short conv (FeatShortConv) or its
	// rolling window, so every one declines. CPU-only until a bridge lands.
	"lfm2":    {},
	"lfm2_vl": {}, // the same lfm2 decoder
	"gpt2":    {"metal"},
	// gpt_oss reaches ALL THREE backends, each on end-to-end evidence: metal on the tiny fixture (TestGptOssResidentParity);
	// cuda on the real 20B, resident on an 8 GB card through --moe-cache-experts (TestGptOssResidentParityCUDA), the stronger
	// evidence of the two; webgpu with the sink threaded through every attention kernel plus gpu/moe.go's MoE kernels
	// (gpu.TestGptOssResidentParityWebGPU).
	"gpt_oss":          {"cuda", "metal", "webgpu"},
	"granitemoehybrid": {"webgpu"},
	"kimi_k2":          {"cuda", "webgpu"},
	"bailing_hybrid":   {}, // FeatKDA undeclared everywhere -- new this pass
	// GLM-OCR: cuda (cuda.TestGlmOcrResidentParityCUDA) and metal (metal.TestGlmOcrResidentParityMetal), which have pairwise
	// rope kernels; WebGPU has only NeoX rope kernels and declines (FeatPairwiseRoPE + FeatPairwiseMRoPE). Admitted with the
	// NeoX kernels it read resident-vs-CPU cosine -0.34 on glm-ocr-tiny.
	"glm_ocr":      {"cuda", "metal"},
	"glm_ocr_text": {"cuda", "metal"},
	"llama":        {"cuda", "metal", "webgpu"},
	// FeatNoPE is declared on cuda and metal (RopeInvFreqLayer zeroes the NoPE layers' invFreq table, no new kernel):
	// smollm3's only required feature.
	"smollm3": {"cuda", "metal"},
	// Olmo 3 / Olmo Hybrid: FeatPostOnlyNorm + FeatQKNormWhole are declared on cuda and metal (the pre-norm skip reuses
	// quant_vec, the whole-vector qk-norm reuses the per-head qk_norm kernel with a collapsed grid). olmo3 needs only those
	// two beyond already-declared features; olmo_hybrid's FeatDeltaNet+FeatNoPE were already declared, so this is its only
	// remaining requirement. Its full-attention layer uses a plain q projection, not qwen3.5's double-width q-gate
	// (Architecture.qwen35.AttnGate).
	"olmo3":       {"cuda", "metal"},
	"olmo_hybrid": {"cuda", "metal"},
	"internlm2":   {"cuda", "metal", "webgpu"},
	"internlm3":   {"cuda", "metal", "webgpu"},
	// Dense Granite 4.2: empty feature profile (see archFeatureProfile's note), so it is
	// admitted everywhere llama is — same backends, same reason.
	"granite":     {"cuda", "metal", "webgpu"},
	"llama4_text": {"cuda", "metal", "webgpu"},
	// mellum reaches cuda and metal by the SAME coupling with VERY DIFFERENT evidence, so the two rows are not
	// interchangeable. cuda: FeatRopeMscale (declared for gpt-oss's YaRN) was the fifth of mellumArchitecture's required
	// features {MoE, PerLayerRoPE, QKNorm, RopeMscale, SlidingWindow}, so it admitted Mellum for free; that coupling was
	// pre-registered as a trap and discharged by measurement (TestMellumResidentParityCUDA on a real 4-layer slice).
	// metal: still the undischarged version of that side effect (G10): no real Mellum checkpoint was reachable there, so
	// it was resolved by an owner call, and the real-weight Metal proof is open (G11). It is here because it IS what the
	// code does, not because it is trusted.
	"mellum":  {"cuda", "metal", "webgpu"},
	"mistral": {"cuda", "metal", "webgpu"},
	// Ministral 3: FeatAttnTemp (rope_kv/rope2's qTempScale param, Q-only, post-rotation) is declared on cuda and metal and is
	// its only required feature; not webgpu, where it is unimplemented.
	"mistral3":   {"cuda", "metal"},
	"ministral3": {"cuda", "metal"},
	"mixtral":    {"cuda", "metal", "webgpu"},
	"nemotron_h": {"webgpu"},
	"phi3":       {"cuda", "metal", "webgpu"},
	"qwen2":      {"cuda", "metal", "webgpu"},
	"qwen2_5_vl": {"cuda", "metal", "webgpu"},
	"qwen3_vl":   {"cuda", "metal", "webgpu"}, // text-only (P8 Phase 0); same admission as plain qwen3
	"voxtral":    {"cuda", "metal", "webgpu"}, // the text decoder is plain llama; the audio turn (GenerateAudio) prefills and decodes on the CPU weights, and no resident build has been run on it (S14.4b)
	"qwen3_asr":  {"cuda", "metal", "webgpu"}, // the text decoder is plain qwen3; the audio turn (GenerateAudio) prefills and decodes on the CPU weights whatever resident is built
	"qwen2_moe":  {"cuda", "metal", "webgpu"}, // cuda joined 2026-08-20 (the gate weight, not a kernel)
	"qwen3":      {"cuda", "metal", "webgpu"},
	// qwen3_moe needs {FeatMoE, FeatQKNorm} — strictly WEAKER than qwen2_moe's
	// {FeatMoE, FeatMoEGatedShared} (no shared expert to gate) and the same FeatQKNorm
	// qwen3 dense already clears on all three backends. Since qwen2_moe and qwen3 are
	// both already admitted everywhere on those exact features, qwen3_moe's subset is
	// too — no new kernel needed, same generic MoE + QK-norm dispatch every backend
	// already has.
	"qwen3_moe":    {"cuda", "metal", "webgpu"},
	"qwen3_vl_moe": {"cuda", "metal", "webgpu"}, // the same admission as qwen3_moe
	// The Gated-DeltaNet family collapses to the backends that implement BOTH the recurrence AND the fused attention output
	// gate (FeatDeltaNet): one taxon bundling both departures rather than two. Metal's declaration is gated by
	// TestQwen35ResidentParityMetal.
	"qwen3_5":          {"cuda", "metal", "webgpu"},
	"qwen3_5_text":     {"cuda", "metal", "webgpu"},
	"qwen3_5_moe":      {"cuda", "metal", "webgpu"},
	"qwen3_5_moe_text": {"cuda", "metal", "webgpu"},
	"qwen3_next":       {"cuda", "metal", "webgpu"}, // same arch.qwen35 != nil bridge qwen3_5_moe uses — reused directly, not a new one
}

// TestResidentAdmission_matrix asserts the (arch × backend) admission decision for every
// registered arch: a backend is admitted ONLY when it implements every feature the arch needs.
// Hardware-free — it reads the declared sets, so it runs in CI with no GPU.
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
		// cgo-free CUDA. Declaration follows evidence: FeatRopeMscale (the YaRN attention_factor folded into cos/sin inside the
		// rope kernels, per-layer wiring from RopeMscaleLayer) is gated by TestRopeMscale plus the real-weight
		// TestMellumResidentParityCUDA; FeatAttnSink + FeatOutBias were declared only after a real gpt-oss-20b forward ran
		// resident (TestGptOssResidentParityCUDA), because a declaration on kernel-level evidence alone was reverted once and
		// the end-to-end run found three silent wiring defects no kernel test could see. Still undeclared: per-layer rotary
		// WIDTH, the ATTENTION softcap (FeatAttnLogitSoftcap, a per-layer kernel) and FeatSSM; the Qwen2-MoE decline is the
		// gated shared expert (FeatMoEGatedShared) being in the shared taxonomy, not a hand-coded check.
		"cuda": {FeatQKNorm, FeatSlidingWindow, FeatPartialRotary, FeatRMSAddOne, FeatSandwichNorm,
			FeatGatedGELU, FeatEmbedScale, FeatPerLayerRoPE, FeatMoE, FeatFinalLogitSoftcap,
			FeatDeltaNet, FeatMoEGatedShared, FeatRopeMscale, FeatAttnSink, FeatOutBias,
			FeatNoPE, FeatAttnTemp, FeatPostOnlyNorm, FeatQKNormWhole,
			FeatLayerNorm, FeatParallelBlock, FeatLogitScale,
			FeatMLA,
			// FeatPairwiseRoPE/FeatPairwiseMRoPE: GPT-J pairwise rotation (cuda/rope_pairwise.cu), declared with the peaked-attention
			// gates (cuda.TestPairwiseRoPEResidentParityCUDA, cuda.TestGlmOcrResidentParityCUDA) and the real Aya/R7B gate.
			// FeatGemma4EModel: the PLE branch from existing kernels, KV-shared layers aliasing their source's cache, per-layer FFN
			// widths (docs/tasks/task-multimodal-support-2026-10.md).
			FeatPairwiseRoPE, FeatPairwiseMRoPE, FeatGemma4EModel},
		// WebGPU: FeatGatedGELU is a genuinely new kernel pair and FeatAttnSink threads the sink through every attention
		// kernel plus gpu/moe.go's MoE kernels; the others reuse existing kernels or host-side wiring.
		"webgpu": {
			FeatQKNorm, FeatPartialRotary, FeatSlidingWindow, FeatPerLayerRoPE, FeatRopeMscale,
			FeatMoE, FeatMoEGatedShared, FeatMLA, FeatSSM, FeatDeltaNet, FeatNonGatedMLP, FeatLogitScale,
			FeatRMSAddOne,
			FeatEmbedScale, FeatFinalLogitSoftcap, FeatSandwichNorm, FeatGatedGELU, FeatOutBias, FeatAttnSink,
		},
		// Metal: declaration follows evidence. FeatRopeMscale also admits Mellum as a documented side effect
		// (docs/queue-correctness.md G10/G11): declared on an explicit owner call, with Mellum's own real-weight Metal gate
		// still open (G11). The Gated-DeltaNet mixer + fused attn output gate landed with its own whole-model gate
		// (TestQwen35ResidentParityMetal), not ahead of it.
		"metal": {
			FeatQKNorm, FeatSlidingWindow, FeatPartialRotary, FeatMoE, FeatMoEGatedShared, FeatSandwichNorm,
			FeatGatedGELU, FeatRMSAddOne, FeatEmbedScale, FeatPerLayerRoPE, FeatFinalLogitSoftcap,
			FeatLayerNorm, FeatNonGatedMLP, FeatLearnedPos, FeatOutBias, FeatRopeMscale, FeatAttnSink,
			FeatDeltaNet,
			FeatNoPE, FeatAttnTemp, FeatPostOnlyNorm, FeatQKNormWhole,
			FeatParallelBlock, FeatLogitScale,
			FeatGemma4EModel,
			// GPT-J pairwise rotation (rope_pw, rope2_pw, rope_f16_pw, rope_mrope_f16_pw; docs/tasks/task-metal-pairwise-rope-2026-10.md).
			FeatPairwiseRoPE, FeatPairwiseMRoPE,
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
		// Every feature added to the taxonomy must be added here too, or this test goes red on the declaring backend (N-12).
		FeatAttnTemp: true, FeatPostOnlyNorm: true, FeatQKNormWhole: true,
		FeatPairwiseRoPE: true, FeatPairwiseMRoPE: true,
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
		{"pairwise-rope", func(a *Architecture) { a.ropeInterleave = true }, FeatPairwiseRoPE},
		{"pairwise-mrope", func(a *Architecture) { a.ropeInterleave = true; a.MRopeSection = []int{16, 24, 24} }, FeatPairwiseMRoPE},
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

// TestPairwiseRoPE_derivationScope pins what FeatPairwiseRoPE is NOT: MLA carries its own interleave
// flag into mla.cu (gated by FeatMLA, where pairwise rope already works), and qwen35's forward passes
// interleave=false to ropeAt whatever ropeInterleave says. Neither may newly need the feature, or
// DeepSeek/Kimi/Qwen3.5 admission would change under it. And PairwiseRoPEResident must read the same
// predicate the feature derives from.
func TestPairwiseRoPE_derivationScope(t *testing.T) {
	base := func() *Architecture {
		return &Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: NormPre2, Act: ActSiLU, ropeInterleave: true}
	}
	if !slices.Contains(base().residentFeatures(), FeatPairwiseRoPE) {
		t.Fatal("a generic ropeInterleave arch must need FeatPairwiseRoPE")
	}
	mla := base()
	mla.mla = &mlaParams{ropeInterleave: true}
	if slices.Contains(mla.residentFeatures(), FeatPairwiseRoPE) {
		t.Error("an MLA arch must not need FeatPairwiseRoPE (mla.cu carries its own interleave)")
	}
	q35 := base()
	q35.qwen35 = &qwen35Params{}
	if slices.Contains(q35.residentFeatures(), FeatPairwiseRoPE) {
		t.Error("qwen35 forwards with interleave=false and must not need FeatPairwiseRoPE")
	}
	for _, a := range []*Architecture{base(), mla, q35} {
		m := &Model{w: &Weights{arch: a}}
		if got, want := m.PairwiseRoPEResident(), slices.Contains(a.residentFeatures(), FeatPairwiseRoPE); got != want {
			t.Errorf("PairwiseRoPEResident()=%v but the feature derivation says %v", got, want)
		}
	}
}

// TestPairwiseRoPE_declineNamesCause: the families that rotate pairwise are DECLINED on every backend that has only
// NeoX rope kernels, and the decline reason names the missing feature, so `serve check` / DecodePath tell the operator
// why instead of "arch is not eligible".
func TestPairwiseRoPE_declineNamesCause(t *testing.T) {
	for _, fam := range []string{"cohere", "cohere2", "glm_ocr"} {
		arch, _, err := resolveArchitecture(representativeConfig(fam))
		if err != nil {
			t.Fatalf("%s: %v", fam, err)
		}
		for _, be := range []string{"webgpu"} {
			if ResidentEligible(arch, be) {
				t.Errorf("%s admitted on %s: only NeoX rope kernels there", fam, be)
				continue
			}
			why := residentGateReason(arch, be)
			if !strings.Contains(why, string(FeatPairwiseRoPE)) {
				t.Errorf("%s on %s: decline reason %q does not name %q", fam, be, why, FeatPairwiseRoPE)
			}
		}
		for _, be := range []string{"cuda", "metal"} {
			if !ResidentEligible(arch, be) {
				t.Errorf("%s declined on %s: %s", fam, be, residentGateReason(arch, be))
			}
		}
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

// TestResidentFeatures_derivationMatchesProfile ties the two sources of truth together (C6): the hand-written
// archFeatureProfile (the classification forcing-function) and residentFeatures() (the derivation that drives the
// generated hardware matrix AND the runtime RequiredResidentFeatures). They must agree for every registered arch, or
// they silently disagree (a layer-0-only yarn-mscale sample missed Mellum's full-layer YaRN; the gated shared expert
// of qwen2_moe). Hardware-free: reads declared sets, runs in CI with no GPU.
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

// TestResidentMoECapacity_routerCap is the M22 taxonomy gate. WebGPU/CUDA score experts into a fixed-size array and
// groups into 32; a model past the cap routes on only the first N (plausible-looking wrong output), so
// ResidentEligible must decline it even though every FEATURE it needs is implemented. This is what keeps the generated
// hardware matrix honest: a feature-only predicate would show Kimi K2 (384 experts) resident where the runtime
// (gpu/backend.go) declines it.
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
	// The cap is 512 (MOE_MAX_E / MAXE) on the two backends whose router scratch was widened, so Kimi-K2's 384 is
	// ADMITTED there: K2 is the shipped family a 256 cap would decline.
	if !residentMoECapacityOK(arch, "webgpu") {
		t.Error("kimi_k2 (384 experts) must now pass the WebGPU router cap (raised to 512)")
	}
	if !residentMoECapacityOK(arch, "cuda") {
		t.Error("kimi_k2 (384 experts) must now pass the CUDA router cap (raised to 512)")
	}
	// ...and WebGPU + CUDA are where that actually matters: both declare FeatMLA, so
	// K2 is genuinely resident-eligible there now. On metal it still declines on both features
	// (needs FeatMLA) and router capacity (shader 256).
	if !ResidentEligible(arch, "webgpu") {
		t.Error("kimi_k2 must be WebGPU-resident-eligible once the router cap admits 384")
	}
	if !ResidentEligible(arch, "cuda") {
		t.Error("kimi_k2 must now be resident-eligible on CUDA with FeatMLA declared")
	}

	// Metal is DECLARED in the cap map (its router shader is a fixed 256-entry array and rejects above it, metal/moe.go's
	// router kernel); an absent entry would read "uncapped", which is false.
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
