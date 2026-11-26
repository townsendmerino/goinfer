package decoder

import (
	"slices"
	"strings"
	"testing"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// archFeatureProfile is the feature set each REGISTERED architecture is expected to require.
// Every registry entry must appear here — TestResidentAdmission_registryCovered fails if a new
// arch lands unclassified, which is the recurrence guard: an unclassified arch would otherwise
// inherit whatever admission it happened to get and could run with features silently dropped.
//
// Only features that reach a backend matter for admission; decodeRunnerEligible already refuses
// several archs upstream (own-forward families, softcapped/sandwich/GPT-2 shapes), but they are
// classified here anyway so the taxonomy stays honest about what each family needs.
var archFeatureProfile = map[string][]darch.ResidentFeature{
	// dense, plain — the subset every backend implements
	"qwen2":      {},
	"qwen2_5_vl": {},
	// qwen3_vl (text only): aliases plain qwen3's attention shape (families.Qwen3VLArchitecture calls qwen3Architecture), so it
	// needs the same FeatQKNorm and nothing else: MRopeSection/MRopeInterleaved only diverge from scalar RoPE on an image
	// path that cache.mropePos never takes for ordinary Generate, same reasoning as qwen2_5_vl's empty profile above.
	"qwen3_vl": {darch.FeatQKNorm},
	// qwen3_asr (S14.3): Qwen3 exactly (qwen3_asrArchitecture calls qwen3Architecture; the checkpoint's interleaved m-RoPE section is dropped because audio positions are sequential), so the same
	// FeatQKNorm and nothing else.
	"qwen3_asr": {darch.FeatQKNorm},
	// voxtral (S14.4b): Llama exactly (voxtralArchitecture calls llamaArchitecture), so the same empty profile.
	"voxtral": {},
	"llama":   {},
	// SmolLM3: llama-shaped plus per-layer NoPE (FeatNoPE), its only required feature (see the admission golden below).
	// cohere2 also needs FeatNoPE, plus FeatLayerNorm/FeatLogitScale/FeatParallelBlock.
	"smollm3": {darch.FeatNoPE},
	// Olmo 3: NormPostOnly + QKNormWhole, plus the standard sliding-window/YaRN features. FeatPerLayerRoPE IS needed
	// (docs/audit-2026-09-10.md G-03/G-04's disposition): YaRN applies to full_attention layers only, sliding_attention gets
	// plain RoPE at the same theta, genuinely different local/global tables. `Olmo3RotaryEmbedding.forward` in the pinned
	// transformers takes an explicit layer_type argument and returns different (cos, sin) per call, so a single shared table
	// cannot be what runs. See registry.go's olmo3Architecture comment for the full account.
	"olmo3": {darch.FeatPostOnlyNorm, darch.FeatQKNormWhole, darch.FeatRopeMscale, darch.FeatSlidingWindow, darch.FeatPerLayerRoPE},
	// Olmo Hybrid: qwen3_5's own FeatDeltaNet (shared math), plus olmo3's FeatPostOnlyNorm/FeatQKNormWhole (its
	// full-attention layers only, but the arch-level check reads NormPlacement/QKNormWhole model-wide, not per-layer) and
	// FeatNoPE (layerNoPE set unconditionally: the release has no RoPE at all). Its full-attention layer uses a plain q
	// projection, not qwen3.5's double-width q-gate (Architecture.qwen35.AttnGate).
	"olmo_hybrid": {darch.FeatDeltaNet, darch.FeatNoPE, darch.FeatPostOnlyNorm, darch.FeatQKNormWhole},
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
	"glm_ocr":      {darch.FeatPairwiseMRoPE, darch.FeatPairwiseRoPE, darch.FeatSandwichNorm},
	"glm_ocr_text": {darch.FeatPairwiseMRoPE, darch.FeatPairwiseRoPE, darch.FeatSandwichNorm},
	// Dense Granite 4.2: llama-shaped, and EMPTY on purpose, not by omission — checked against
	// all three released sizes (3b/8b/30b), which all ship embedding_multiplier and
	// logits_scaling at their identity value 1.0 (FeatEmbedScale/FeatLogitScale only trigger
	// above/away-from 1, per their own derivation in this file), and residual_multiplier is
	// forced to 1.0 by ValidateGraniteDense. attention_multiplier is NOT gated by any
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
	"gpt2": {darch.FeatLayerNorm, darch.FeatLearnedPos, darch.FeatNonGatedMLP, darch.FeatNonGatedGELUTanh, darch.FeatOutBias},
	// Cohere's rotation is GPT-J PAIRWISE (rope_gptj), not NeoX, so it needs FeatPairwiseRoPE: a resident backend that runs
	// it on NeoX kernels is wrong on the real checkpoints (flat-fixture gates are blind to it). Backends without a pairwise
	// rope variant decline these families.
	"cohere":  {darch.FeatLayerNorm, darch.FeatLogitScale, darch.FeatPairwiseRoPE, darch.FeatParallelBlock},
	"cohere2": {darch.FeatLayerNorm, darch.FeatLogitScale, darch.FeatNoPE, darch.FeatPairwiseRoPE, darch.FeatParallelBlock, darch.FeatSlidingWindow},
	"mistral": {darch.FeatSlidingWindow},
	// Ministral 3: no sliding window on the real releases, so the only feature this family needs is the attn-temp query
	// scale (see FeatAttnTemp's own comment). Otherwise a plain GQA+YaRN model.
	"mistral3":   {darch.FeatAttnTemp},
	"ministral3": {darch.FeatAttnTemp},
	"qwen3":      {darch.FeatQKNorm},
	"mellum":     {darch.FeatMoE, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatRopeMscale, darch.FeatSlidingWindow},
	"mixtral":    {darch.FeatMoE},
	"qwen2_moe":  {darch.FeatMoE, darch.FeatMoEGatedShared}, // sigmoid-gated always-on shared expert
	// qwen3_moe: qwen3's QK-norm attention + a routed MoE FFN with NO shared expert (confirmed
	// against the real Qwen3-30B-A3B config.json), so unlike qwen2_moe it does NOT need
	// FeatMoEGatedShared.
	"qwen3_moe":    {darch.FeatMoE, darch.FeatQKNorm},
	"qwen3_vl_moe": {darch.FeatMoE, darch.FeatQKNorm}, // qwen3_moe's needs; m-RoPE is the qwen3_vl image path's, as for qwen3_vl
	"glm4_moe":     {darch.FeatMoE, darch.FeatPartialRotary, darch.FeatQKNorm},
	// Laguna: sigmoid-routed MoE with an UNGATED shared expert (so FeatMoE, not
	// FeatMoEGatedShared), QK-norm, partial rotary on the full-attention layers, a
	// sliding/full interleave with per-layer RoPE bases, YaRN mscale on the full
	// layers, and the family-specific output gate. The XS generations carry all of
	// these; M.1 drops the sliding-window half, but this table states the family's
	// BASE profile and RequiredResidentFeatures derives the per-model truth.
	"laguna": {darch.FeatAttnOutputGateSoftplus, darch.FeatMoE, darch.FeatPartialRotary, darch.FeatPerLayerQueryHeads, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatRopeMscale, darch.FeatSlidingWindow},
	// Flint: sigmoid-routed MoE with an UNGATED shared expert (FeatMoE), per-head QK RMSNorm (FeatQKNorm), per-query-head
	// attention sinks (FeatAttnSink), and a sliding/full attention schedule where the full-attention layers are NoPE
	// (FeatNoPE, FeatSlidingWindow).
	"flint": {darch.FeatAttnSink, darch.FeatMoE, darch.FeatNoPE, darch.FeatQKNorm, darch.FeatSlidingWindow},
	// spark2_5: an attention output gate as Laguna has, through sigmoid here (FeatAttnOutputGateSigmoid) where Laguna's is
	// softplus; no resident backend implements either, and a gated MLP with exact-erf GELU (FeatGatedGELUErf), which no backend
	// implements either: every gated kernel has GELU-tanh and SiLU only.
	"spark2_5":    {darch.FeatAttnOutputGateSigmoid, darch.FeatGatedGELUErf, darch.FeatPartialRotary, darch.FeatPerLayerRoPE, darch.FeatSlidingWindow},
	"deepseek_v2": {darch.FeatMLA, darch.FeatMoE},
	"deepseek_v3": {darch.FeatMLA, darch.FeatMoE},
	"kimi_k2":     {darch.FeatMLA, darch.FeatMoE},
	// Bailing Hybrid (Ling 3.0): FeatKDA is declared by no backend, so the family is CPU-only although FeatMLA and FeatMoE
	// are ordinary DeepSeek-shaped features.
	"bailing_hybrid": {darch.FeatMLA, darch.FeatMoE, darch.FeatKDA},
	"qwen3_5_moe":    {darch.FeatMoE, darch.FeatMoEGatedShared, darch.FeatPartialRotary, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatDeltaNet},
	// Qwen3.8 dense (qwen3_5): the MoE sibling's profile MINUS the two MoE features. The
	// remaining three are unchanged and each was checked against the released 27B rather
	// than inherited — partial rotary (0.25 × head_dim 256 = 64 < 256), q_norm/k_norm on
	// every softmax layer, and Gemma-style (1+w) RMSNorm. The DeltaNet mixer itself needs
	// no feature here because decodeRunnerEligible refuses the whole arch.qwen35 family
	// upstream (own forward, not yet bridged) — the same posture qwen3_5_moe has.
	"qwen3_5":      {darch.FeatPartialRotary, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatDeltaNet},
	"qwen3_5_text": {darch.FeatPartialRotary, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatDeltaNet},
	// Qwen3-Next: same profile as qwen3_5_moe (verified, not assumed — its MoE block
	// (Qwen3NextSparseMoeBlock(Qwen2MoeSparseMoeBlock): pass) directly inherits
	// Qwen2-MoE's gated-shared-expert combination, and its RMSNorm
	// (Qwen3NextRMSNorm(Gemma3RMSNorm): pass) inherits Gemma's (1+w)).
	"qwen3_next":       {darch.FeatMoE, darch.FeatMoEGatedShared, darch.FeatPartialRotary, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatDeltaNet},
	"llama4_text":      {darch.FeatMoE},
	"gpt_oss":          {darch.FeatAttnSink, darch.FeatMoE, darch.FeatOutBias, darch.FeatRopeMscale, darch.FeatSlidingWindow},
	"nemotron_h":       {darch.FeatNonGatedMLP, darch.FeatNonGatedReLU2, darch.FeatSSM},
	"granitemoehybrid": {darch.FeatLogitScale, darch.FeatMoE, darch.FeatSSM},
	// LFM2/LFM2.5: FeatShortConv is what keeps this CPU-only. Strip it and the profile is
	// {FeatQKNorm} — which every resident backend implements, so all three would admit a
	// family none of them can run. Same shape as laguna's FeatAttnOutputGateSoftplus above.
	"lfm2":             {darch.FeatQKNorm, darch.FeatShortConv},
	"lfm2_vl":          {darch.FeatQKNorm, darch.FeatShortConv}, // S10: LFM2-VL's text_config is lfm2
	"qwen3_5_moe_text": {darch.FeatMoE, darch.FeatMoEGatedShared, darch.FeatPartialRotary, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatDeltaNet},
	// Gemma, VERIFIED against the real checkpoints via RequiredResidentFeatures. gemma4 needs the FINAL-logit softcap (one
	// host-side tanh, FeatFinalLogitSoftcap), not the attention softcap; its own forward (per-layer head_dim / K=V) is what
	// the resident geometry bridge addresses, a separate gate from this feature set. Gemma 1 / CodeGemma and Gemma 2
	// (gemma12.go): Gemma 2 needs the attention-score softcap, a per-layer kernel no resident backend ships, so it is
	// admitted nowhere and runs on the CPU until one does.
	"gemma":       {darch.FeatEmbedScale, darch.FeatGatedGELU, darch.FeatRMSAddOne},
	"gemma2":      {darch.FeatAttnLogitSoftcap, darch.FeatEmbedScale, darch.FeatFinalLogitSoftcap, darch.FeatGatedGELU, darch.FeatRMSAddOne, darch.FeatSandwichNorm, darch.FeatSlidingWindow},
	"gemma3":      {darch.FeatEmbedScale, darch.FeatGatedGELU, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatSandwichNorm, darch.FeatSlidingWindow},
	"gemma3_text": {darch.FeatEmbedScale, darch.FeatGatedGELU, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatRMSAddOne, darch.FeatSandwichNorm, darch.FeatSlidingWindow},
	"gemma4":      {darch.FeatEmbedScale, darch.FeatFinalLogitSoftcap, darch.FeatGatedGELU, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatSandwichNorm, darch.FeatSlidingWindow},
	// gemma4_text is the 26B-A4B MoE variant: gemma4's feature set + FeatMoE. Gemma 4's MoE is the parallel dense‖MoE shape
	// the generic FeatMoE kernel cannot express, so the feature set is necessary, not sufficient: admission also consults
	// residentGemma4MoEBackends (features.go).
	"gemma4_text": {darch.FeatEmbedScale, darch.FeatFinalLogitSoftcap, darch.FeatGatedGELU, darch.FeatMoE, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatSandwichNorm, darch.FeatSlidingWindow},
	// gemma4_unified_text is the real unified checkpoints' text_config model_type; same feature set as gemma4_text
	// (K=V globals are a loader detail, not a resident feature).
	"gemma4_unified_text": {darch.FeatEmbedScale, darch.FeatFinalLogitSoftcap, darch.FeatGatedGELU, darch.FeatMoE, darch.FeatPerLayerRoPE, darch.FeatQKNorm, darch.FeatSandwichNorm, darch.FeatSlidingWindow},
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
	// Cohere / Command-R + Cohere2 / Command-R7B: FeatParallelBlock, FeatLogitScale, FeatLayerNorm and FeatPairwiseRoPE (and for
	// cohere2 FeatNoPE/FeatSlidingWindow), declared on every backend (gpu.TestCohereResidentParityWebGPU on peaked weights,
	// gpu.TestPairwiseRoPERealWebGPU on the real Command-R7B). A NeoX rotation reads near-zero cosine on the real checkpoints
	// while flat 0.02-std fixtures read 0.9997.
	"cohere":      {"cuda", "metal", "webgpu"},
	"cohere2":     {"cuda", "metal", "webgpu"},
	"deepseek_v2": {"cuda", "metal", "webgpu"},
	"deepseek_v3": {"cuda", "metal", "webgpu"},
	"flint":       {"cuda", "metal", "webgpu"},
	// gemma3 (uniform head_dim) reaches webgpu for real (testdata/gemma3-vl-tiny's text tower,
	// gpu.TestGemma3ResidentParityWebGPU). gemma4/gemma4_text/gemma4_unified_text are FEATURE-compatible there, but dense
	// Gemma 4's local/global head_dim split needs a per-layer geometry seam this feature-list model cannot express
	// (decoder.Model.PerLayerGeomOK, a runtime check outside the ResidentFeature taxonomy), so the REAL runtime
	// (decoder.ResidentEligible, TestGemma4Admission_unconditional) declines gemma4 on webgpu while this golden shows it
	// feature-admitted: it intentionally tracks the feature-only model, as with the MoE-cap precedent (deepseek_v2/kimi_k2
	// below, admitted by feature but capped elsewhere).
	"gemma":               {"cuda", "metal", "webgpu"},
	"gemma2":              {"cuda", "metal"}, // the attention-score softcap: Metal and CUDA since OP1; webgpu does not declare it
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
	"granitemoehybrid": {"webgpu"},                  // feature-admitted on WebGPU; Metal's and CUDA's tables have no residual scale (residual_multiplier); the runner-shape bring-up gate keeps it on the CPU everywhere (hardware matrix)
	"kimi_k2":          {"cuda", "metal", "webgpu"}, // feature-admitted; the 384-expert router cap keeps it off Metal (hardware matrix)
	"bailing_hybrid":   {},                          // FeatKDA undeclared everywhere -- new this pass
	// GLM-OCR: every backend has pairwise rope kernels (cuda.TestGlmOcrResidentParityCUDA, metal.TestGlmOcrResidentParityMetal,
	// gpu.TestGlmOcrResidentParityWebGPU). On NeoX kernels it read resident-vs-CPU cosine -0.34 on glm-ocr-tiny.
	"glm_ocr":      {"cuda", "metal", "webgpu"},
	"glm_ocr_text": {"cuda", "metal", "webgpu"},
	"llama":        {"cuda", "metal", "webgpu"},
	// FeatNoPE is declared on cuda, metal and webgpu (RopeInvFreqLayer zeroes the NoPE layers' invFreq table, no new kernel):
	// smollm3's only required feature.
	"smollm3": {"cuda", "metal", "webgpu"},
	// Olmo 3 / Olmo Hybrid: FeatPostOnlyNorm + FeatQKNormWhole are declared on cuda, metal and webgpu (the pre-norm skip reuses
	// quant, the whole-vector qk-norm reuses the per-head qk_norm kernel with a collapsed grid). olmo3 needs only those
	// two beyond already-declared features; olmo_hybrid's FeatDeltaNet+FeatNoPE are also declared on webgpu.
	// Its full-attention layer uses a plain q projection, not qwen3.5's double-width q-gate
	// (Architecture.qwen35.AttnGate).
	"olmo3":       {"cuda", "metal", "webgpu"},
	"olmo_hybrid": {"cuda", "metal", "webgpu"}, // each claims DeltaNet's widened write gate (NegEigval) and its own output-norm epsilon
	"internlm2":   {"cuda", "metal", "webgpu"},
	"internlm3":   {"cuda", "metal", "webgpu"},
	// Dense Granite 4.2: empty feature profile (see archFeatureProfile's note), so it is
	// admitted everywhere llama is — same backends, same reason.
	"granite":     {"cuda", "metal", "webgpu"},
	"llama4_text": {"webgpu"}, // Metal's and CUDA's tables have no de-interleaving (mlaRope) rotation, L2 QK-norm or Llama 4 MoE form; the runner shape keeps it on the CPU everywhere
	// mellum reaches cuda and metal by the SAME coupling with VERY DIFFERENT evidence, so the two rows are not
	// interchangeable. cuda: FeatRopeMscale (declared for gpt-oss's YaRN) was the fifth of mellumArchitecture's required
	// features {MoE, PerLayerRoPE, QKNorm, RopeMscale, SlidingWindow}, so it admitted Mellum for free; that coupling was
	// pre-registered as a trap and discharged by measurement (TestMellumResidentParityCUDA on a real 4-layer slice).
	// metal: still the undischarged version of that side effect (G10): no real Mellum checkpoint was reachable there, so
	// it was resolved by an owner call, and the real-weight Metal proof is open (G11). It is here because it IS what the
	// code does, not because it is trusted.
	"mellum":  {"cuda", "metal", "webgpu"},
	"mistral": {"cuda", "metal", "webgpu"},
	// Ministral 3: FeatAttnTemp is its only required feature: cuda and metal scale Q after rotation (rope_kv/rope2's
	// qTempScale), webgpu folds the temperature into the attention score scale (gpu.TestMinistral3ResidentParityWebGPU).
	"mistral3":   {"cuda", "metal", "webgpu"},
	"ministral3": {"cuda", "metal", "webgpu"},
	"mixtral":    {"cuda", "metal", "webgpu"},
	"nemotron_h": {"cuda", "metal", "webgpu"}, // Metal since OP2b step 2, CUDA since OP2b step 2d's declaration, int4 only (buildMamba declines int8 on both)
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
// registered arch: a backend is admitted ONLY when it implements every feature the arch needs, or, for a backend whose
// admission reads an operation table, every operation and parameter of the arch's program and the features outside it.
// Hardware-free — it reads the declared sets and tables, so it runs in CI with no GPU.
func TestResidentAdmission_matrix(t *testing.T) {
	backends := []string{"cuda", "metal", "webgpu"}
	seen := map[string]bool{}
	for _, name := range slices.Sorted(mapKeys(archFeatureProfile)) {
		req := archFeatureProfile[name]
		var admits []string
		for _, be := range backends {
			if residentBackendTables[be] != nil {
				// A table backend's admission reads the program, so it is asked of the registry's representative arch.
				a, _, err := resolveArchitecture(representativeConfig(name))
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if residentCovers(a, nil, be) {
					admits = append(admits, be)
				}
				continue
			}
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

// TestMetalTable_noOverclaim pins Metal's operation table (program.Metal) as TestResidentBackendFeatures_noOverclaim pins the
// feature sets: a kind, an FFN form or activation, or a parameter added without its kernel is the lie admission exists to
// catch, so widening the table is a deliberate, reviewed edit. Each entry's evidence is in docs/code-notes/decoder.md,
// "residentBackendFeatures.metal.removed".
func TestMetalTable_noOverclaim(t *testing.T) {
	m := program.Metal
	kinds := []program.OpKind{program.OpNorm, program.OpProjQKV, program.OpQKNorm, program.OpRope, program.OpAttnTemp,
		program.OpAttention, program.OpProjO, program.OpFFN, program.OpResidual, program.OpMLA, program.OpMamba2, program.OpDeltaNet,
		program.OpGatedAttn, program.OpGemma4Attn, program.OpGemma4MoE, program.OpPLE, program.OpLayerScale}
	if !slices.Equal(m.Kinds, kinds) {
		t.Errorf("metal's kinds are %v, want %v", m.Kinds, kinds)
	}
	ffn := map[program.FFNForm][]program.ActKind{
		program.FfnGated:       {program.ActSiLU, program.ActGeluTanh},
		program.FfnNonGated:    {program.ActGeluTanh, program.ActSiLU, program.ActReLU2},
		program.FfnMoE:         {program.ActSiLU, program.ActGeluTanh},
		program.FfnGptOss:      {program.ActSiLU, program.ActGeluTanh},
		program.FfnGemma4Dense: {program.ActSiLU, program.ActGeluTanh},
	}
	if len(m.FFN) != len(ffn) {
		t.Errorf("metal has %d FFN forms, want %d: %v", len(m.FFN), len(ffn), m.FFN)
	}
	for f, acts := range ffn {
		if !slices.Equal(m.FFN[f], acts) {
			t.Errorf("metal's %v activations are %v, want %v", f, m.FFN[f], acts)
		}
	}
	if !slices.Equal(m.QKNorms, []program.QKNormForm{program.QKRMS}) || !slices.Equal(m.Ropes, []program.RopeKernel{program.RopeAtKernel}) {
		t.Errorf("metal's QK-norm forms %v and rotation kernels %v, want the RMS form and the ropeAt kernel only", m.QKNorms, m.Ropes)
	}
	if !m.PairwiseRope {
		t.Errorf("metal no longer claims GPT-J pairwise rotation (rope_pw, rope2_pw, rope_f16_pw, rope_mrope_f16_pw)")
	}
	if !(m.AttnWindow && m.AttnSoftcap && m.QKNormWhole && m.SandwichNorm && m.PostOnlyNorm && m.ParallelBlock && m.LayerNorm &&
		m.OutBias && m.LearnedPos) {
		t.Errorf("metal's decode table no longer claims a placement or form its kernels run: %+v", *m)
	}
	// The multi-sequence step's table: one layer shape, pre-norm attention and a gated dense FFN (metal/batch.go).
	s := program.MetalStep
	stepKinds := []program.OpKind{program.OpNorm, program.OpProjQKV, program.OpQKNorm, program.OpRope, program.OpAttnTemp,
		program.OpAttention, program.OpProjO, program.OpFFN, program.OpResidual}
	if !slices.Equal(s.Kinds, stepKinds) || len(s.FFN) != 1 ||
		!slices.Equal(s.FFN[program.FfnGated], []program.ActKind{program.ActSiLU, program.ActGeluTanh}) ||
		!slices.Equal(s.QKNorms, []program.QKNormForm{program.QKRMS}) || !slices.Equal(s.Ropes, []program.RopeKernel{program.RopeAtKernel}) {
		t.Errorf("metal's step table changed its kinds, FFN forms, QK-norm forms or rotations: %+v", *s)
	}
	if s.PairwiseRope || s.AttnChunk || s.AttnWindow || s.AttnSoftcap || s.QKNormWhole || s.ResidualScale || s.DeltaNegEigval ||
		s.DeltaONormEps || s.SandwichNorm || s.PostOnlyNorm || s.ParallelBlock || s.LayerNorm || s.OutBias || s.LearnedPos {
		t.Errorf("metal's step table claims a parameter its per-row kernels do not run: %+v", *s)
	}
	if m.AttnChunk || m.ResidualScale {
		t.Errorf("metal claims a chunked attention %v or a residual scale %v; neither has a kernel", m.AttnChunk, m.ResidualScale)
	}
	if !m.DeltaNegEigval || !m.DeltaONormEps { // delta_gates' betaScale and the gated norm's own epsilon
		t.Errorf("metal claims DeltaNet's widened write gate %v and its own output-norm epsilon %v; both have kernels", m.DeltaNegEigval, m.DeltaONormEps)
	}
	if residentBackendTables["metal"] != m {
		t.Errorf("Metal's admission does not read program.Metal")
	}
	if _, ok := residentBackendFeatures["metal"]; ok {
		t.Errorf("metal declares a feature set beside its table; one vocabulary decides its admission")
	}
}

// TestResidentBackendFeatures_noOverclaim pins each backend's declared coverage. A backend
// adding a claim here without shipping the kernel is exactly the lie the admission gate exists
// to catch, so widening a set must be a deliberate, reviewed edit — not a drive-by.
func TestResidentBackendFeatures_noOverclaim(t *testing.T) {
	want := map[string][]darch.ResidentFeature{
		// WebGPU: FeatGatedGELU is a genuinely new kernel pair and FeatAttnSink threads the sink through every attention
		// kernel plus gpu/moe.go's MoE kernels; the others reuse existing kernels or host-side wiring.
		"webgpu": {
			darch.FeatQKNorm, darch.FeatPartialRotary, darch.FeatSlidingWindow, darch.FeatPerLayerRoPE, darch.FeatRopeMscale,
			darch.FeatMoE, darch.FeatMoEGatedShared, darch.FeatMLA, darch.FeatSSM, darch.FeatDeltaNet, darch.FeatNonGatedMLP, darch.FeatNonGatedReLU2, darch.FeatLogitScale,
			darch.FeatRMSAddOne,
			darch.FeatEmbedScale, darch.FeatFinalLogitSoftcap, darch.FeatSandwichNorm, darch.FeatGatedGELU, darch.FeatOutBias, darch.FeatAttnSink,
			darch.FeatGemma4EModel,                                        // gpu's TestGemma4EModel_webgpuResidentParity and its planted defects
			darch.FeatNoPE, darch.FeatPostOnlyNorm, darch.FeatQKNormWhole, // Olmo Hybrid / Olmo 3 / SmolLM3 (gpu.TestOlmoHybridResidentParityWebGPU)
			darch.FeatAttnTemp,                              // folded into the attention score scale (gpu.TestMinistral3ResidentParityWebGPU)
			darch.FeatPairwiseRoPE, darch.FeatPairwiseMRoPE, // the pairwise rope twins (gpu.TestGlmOcrResidentParityWebGPU, kernels.TestPairwiseTwins_matchGPTJ)
			darch.FeatLayerNorm, darch.FeatParallelBlock, // LayerNormQuantWGSL and the shared MLP input (gpu.TestCohereResidentParityWebGPU)
		},
	}
	// CUDA's admission reads its operation table; the evidence each
	// feature was declared on is docs/code-notes/decoder.md#residentBackendFeatures.cuda.entry.
	if residentBackendTables["cuda"] != program.CUDA {
		t.Errorf("CUDA's admission does not read program.CUDA")
	}
	if _, ok := residentBackendFeatures["cuda"]; ok {
		t.Errorf("cuda declares a feature set beside its table; one vocabulary decides its admission")
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
	known := map[darch.ResidentFeature]bool{
		darch.FeatQKNorm: true, darch.FeatSlidingWindow: true, darch.FeatPartialRotary: true, darch.FeatPerLayerRoPE: true,
		darch.FeatRopeMscale: true, darch.FeatRMSAddOne: true, darch.FeatEmbedScale: true,
		darch.FeatAttnLogitSoftcap: true, darch.FeatFinalLogitSoftcap: true,
		darch.FeatSandwichNorm: true, darch.FeatGatedGELU: true, darch.FeatNonGatedMLP: true, darch.FeatLearnedPos: true,
		darch.FeatOutBias: true, darch.FeatLogitScale: true, darch.FeatMoE: true, darch.FeatMoEGatedShared: true,
		darch.FeatMLA: true, darch.FeatSSM: true, darch.FeatLayerNorm: true, darch.FeatParallelBlock: true, darch.FeatNoPE: true,
		darch.FeatAttnSink: true, darch.FeatGemma4EModel: true, darch.FeatDeltaNet: true, // N-12: were omitted, so declaring either failed with a misleading "unknown feature"
		// Every feature added to the taxonomy must be added here too, or this test goes red on the declaring backend (N-12).
		darch.FeatAttnTemp: true, darch.FeatPostOnlyNorm: true, darch.FeatQKNormWhole: true,
		darch.FeatPairwiseRoPE: true, darch.FeatPairwiseMRoPE: true,
		darch.FeatGatedGELUErf: true, darch.FeatNonGatedGELUErf: true, // FeatGatedActUnknown is left out on purpose: it names no kernel
		darch.FeatNonGatedGELUTanh: true, darch.FeatNonGatedSiLU: true, darch.FeatNonGatedReLU2: true, // FeatNonGatedActUnknown likewise
		darch.FeatAttnOutputGateSoftplus: true, darch.FeatAttnOutputGateSigmoid: true, darch.FeatPerLayerQueryHeads: true,
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
	base := func() *darch.Architecture {
		return &darch.Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: darch.NormPre2, Act: darch.ActSiLU}
	}
	cases := []struct {
		name string
		mut  func(*darch.Architecture)
		want darch.ResidentFeature
	}{
		{"qk-norm", func(a *darch.Architecture) { a.QKNorm = true }, darch.FeatQKNorm},
		{"sliding-window", func(a *darch.Architecture) { a.SlidingWindow = 4096 }, darch.FeatSlidingWindow},
		{"partial-rotary", func(a *darch.Architecture) { a.RotaryDim = 64 }, darch.FeatPartialRotary},
		{"rms-add-one", func(a *darch.Architecture) { a.RMSAddOne = true }, darch.FeatRMSAddOne},
		{"embed-scale", func(a *darch.Architecture) { a.EmbedScale = 32 }, darch.FeatEmbedScale},
		{"final-softcap", func(a *darch.Architecture) { a.FinalLogitSoftcap = 30 }, darch.FeatFinalLogitSoftcap},
		{"attn-softcap", func(a *darch.Architecture) { a.AttnLogitSoftcap = 30 }, darch.FeatAttnLogitSoftcap},
		{"sandwich", func(a *darch.Architecture) { a.NormPlacement = darch.NormSandwich4 }, darch.FeatSandwichNorm},
		{"layer-norm", func(a *darch.Architecture) { a.Norm = darch.NormLayer }, darch.FeatLayerNorm},
		{"parallel-block", func(a *darch.Architecture) { a.NormPlacement = darch.NormParallel }, darch.FeatParallelBlock},
		{"nope", func(a *darch.Architecture) { a.LayerNoPE = func(int) bool { return true } }, darch.FeatNoPE},
		{"gated-gelu", func(a *darch.Architecture) { a.Act = darch.ActGeluTanh }, darch.FeatGatedGELU},
		{"gated-gelu-erf", func(a *darch.Architecture) { a.Act = darch.ActGelu }, darch.FeatGatedGELUErf},
		{"gated-act-unknown", func(a *darch.Architecture) { a.Act = darch.ActReLU2 }, darch.FeatGatedActUnknown},
		{"non-gated", func(a *darch.Architecture) { a.NonGatedMLP = true }, darch.FeatNonGatedMLP},
		{"non-gated-gelu-erf", func(a *darch.Architecture) { a.NonGatedMLP = true; a.Act = darch.ActGelu }, darch.FeatNonGatedGELUErf},
		{"non-gated-gelu-tanh", func(a *darch.Architecture) { a.NonGatedMLP = true; a.Act = darch.ActGeluTanh }, darch.FeatNonGatedGELUTanh},
		{"non-gated-silu", func(a *darch.Architecture) { a.NonGatedMLP = true }, darch.FeatNonGatedSiLU},
		{"non-gated-relu2", func(a *darch.Architecture) { a.NonGatedMLP = true; a.Act = darch.ActReLU2 }, darch.FeatNonGatedReLU2},
		{"non-gated-act-unknown", func(a *darch.Architecture) { a.NonGatedMLP = true; a.Act = ActKind(99) }, darch.FeatNonGatedActUnknown},
		{"attn-output-gate-sigmoid", func(a *darch.Architecture) { a.AttnGate = darch.GateSigmoid }, darch.FeatAttnOutputGateSigmoid},
		{"attn-output-gate-softplus", func(a *darch.Architecture) { a.Laguna = &archparts.LagunaParams{} }, darch.FeatAttnOutputGateSoftplus},
		{"per-layer-query-heads", func(a *darch.Architecture) { a.Laguna = &archparts.LagunaParams{HeadsPerLayer: []int{48, 64}} }, darch.FeatPerLayerQueryHeads},
		{"learned-pos", func(a *darch.Architecture) { a.LearnedPosEmbed = true }, darch.FeatLearnedPos},
		{"out-bias", func(a *darch.Architecture) { a.OutBias = true }, darch.FeatOutBias},
		{"logit-scale", func(a *darch.Architecture) { a.LogitScale = 8 }, darch.FeatLogitScale},
		{"moe", func(a *darch.Architecture) { a.MoE = &MoEConfig{} }, darch.FeatMoE},
		{"pairwise-rope", func(a *darch.Architecture) { a.RoPEInterleave = true }, darch.FeatPairwiseRoPE},
		{"pairwise-mrope", func(a *darch.Architecture) { a.RoPEInterleave = true; a.MRopeSection = []int{16, 24, 24} }, darch.FeatPairwiseMRoPE},
		{"gemma4-e-model-ple", func(a *darch.Architecture) { a.Gemma4 = &archparts.Gemma4Params{HiddenSizePerLayerInput: 256} }, darch.FeatGemma4EModel},
		{"gemma4-e-model-sharedkv", func(a *darch.Architecture) { a.Gemma4 = &archparts.Gemma4Params{SharedKVLayers: 2} }, darch.FeatGemma4EModel},
		{"gemma4-e-model-ffnperlayer", func(a *darch.Architecture) { a.Gemma4 = &archparts.Gemma4Params{FFNPerLayer: []int{1, 2}} }, darch.FeatGemma4EModel},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := base()
			c.mut(a)
			got := a.ResidentFeatures()
			if !slices.Contains(got, c.want) {
				t.Errorf("derived %v, want it to include %q", got, c.want)
			}
			// A plain dense arch must require nothing — else every backend declines everything.
			if plain := base().ResidentFeatures(); len(plain) != 0 {
				t.Errorf("plain dense arch requires %v, want none", plain)
			}
		})
	}
}

// TestPairwiseRoPE_derivationScope pins what FeatPairwiseRoPE is NOT: MLA carries its own interleave
// flag into mla.cu (gated by FeatMLA, where pairwise rope already works), and qwen35's forward passes
// interleave=false to rope.At whatever ropeInterleave says. Neither may newly need the feature, or
// DeepSeek/Kimi/Qwen3.5 admission would change under it. And PairwiseRoPEResident must read the same
// predicate the feature derives from.
func TestPairwiseRoPE_derivationScope(t *testing.T) {
	base := func() *darch.Architecture {
		return &darch.Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: darch.NormPre2, Act: darch.ActSiLU, RoPEInterleave: true}
	}
	if !slices.Contains(base().ResidentFeatures(), darch.FeatPairwiseRoPE) {
		t.Fatal("a generic ropeInterleave arch must need FeatPairwiseRoPE")
	}
	mla := base()
	mla.MLA = &archparts.MLAParams{RoPEInterleave: true}
	if slices.Contains(mla.ResidentFeatures(), darch.FeatPairwiseRoPE) {
		t.Error("an MLA arch must not need FeatPairwiseRoPE (mla.cu carries its own interleave)")
	}
	q35 := base()
	q35.Qwen35 = &archparts.Qwen35Params{}
	if slices.Contains(q35.ResidentFeatures(), darch.FeatPairwiseRoPE) {
		t.Error("qwen35 forwards with interleave=false and must not need FeatPairwiseRoPE")
	}
	for _, a := range []*darch.Architecture{base(), mla, q35} {
		m := &Model{w: &darch.Weights{Arch: a}}
		if got, want := m.BackendView().PairwiseRoPE(), slices.Contains(a.ResidentFeatures(), darch.FeatPairwiseRoPE); got != want {
			t.Errorf("PairwiseRoPEResident()=%v but the feature derivation says %v", got, want)
		}
	}
}

// TestPairwiseRoPE_admittedOnEveryBackend: every resident backend declares the pairwise rope kernels and what the pairwise
// families need besides, so Cohere, Cohere2 and GLM-OCR are admitted on each, and no decline names pairwise-rope.
func TestPairwiseRoPE_admittedOnEveryBackend(t *testing.T) {
	for _, fam := range []string{"cohere", "cohere2", "glm_ocr"} {
		arch, _, err := resolveArchitecture(representativeConfig(fam))
		if err != nil {
			t.Fatalf("%s: %v", fam, err)
		}
		if !slices.Contains(arch.ResidentFeatures(), darch.FeatPairwiseRoPE) {
			t.Fatalf("%s does not need %s: the fixture no longer exercises the pairwise rotation", fam, darch.FeatPairwiseRoPE)
		}
		for _, be := range []string{"cuda", "metal", "webgpu"} {
			why := residentGateReason(arch, be)
			if strings.Contains(why, string(darch.FeatPairwiseRoPE)) {
				t.Errorf("%s on %s: the decline %q still names %q", fam, be, why, darch.FeatPairwiseRoPE)
			}
			if !ResidentEligible(arch, be) {
				t.Errorf("%s declined on %s: %s", fam, be, why)
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
		got := arch.ResidentFeatures()
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
	if miss := missingFeatures(arch.ResidentFeatures(), residentBackendFeatures["webgpu"]); len(miss) != 0 {
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

// TestFeatureSplits_wrongHalfDeclinesByName is OP2's gate for the two feature splits: a backend that declares the wrong half of a split, the other activation of the same structure, must
// decline the family and name the half it lacks.
func TestFeatureSplits_wrongHalfDeclinesByName(t *testing.T) {
	for _, c := range []struct {
		family       string
		declare, not darch.ResidentFeature // the wrong half the backend declares, and the half the family needs
		with         []darch.ResidentFeature
	}{
		{"gpt2", darch.FeatNonGatedSiLU, darch.FeatNonGatedGELUTanh, []darch.ResidentFeature{darch.FeatNonGatedMLP}},
		{"nemotron_h", darch.FeatNonGatedGELUTanh, darch.FeatNonGatedReLU2, []darch.ResidentFeature{darch.FeatNonGatedMLP}},
		{"laguna", darch.FeatAttnOutputGateSigmoid, darch.FeatAttnOutputGateSoftplus, []darch.ResidentFeature{darch.FeatPerLayerQueryHeads}},
		{"spark2_5", darch.FeatAttnOutputGateSoftplus, darch.FeatAttnOutputGateSigmoid, nil},
	} {
		arch, _, err := resolveArchitecture(representativeConfig(c.family))
		if err != nil {
			t.Fatalf("%s: %v", c.family, err)
		}
		// Every feature the family needs, except the half under test, which is swapped for the wrong one.
		set := map[darch.ResidentFeature]bool{c.declare: true}
		for _, f := range arch.ResidentFeatures() {
			if f != c.not {
				set[f] = true
			}
		}
		for _, f := range c.with {
			set[f] = true
		}
		missing := missingFeatures(arch.ResidentFeatures(), set)
		if !slices.Equal(missing, []darch.ResidentFeature{c.not}) {
			t.Errorf("%s on a backend declaring %s and not %s: missing %v, want exactly [%s]", c.family, c.declare, c.not, missing, c.not)
		}
		set[c.not] = true
		if missing := missingFeatures(arch.ResidentFeatures(), set); len(missing) != 0 {
			t.Errorf("%s with the right half declared: still missing %v", c.family, missing)
		}
	}
}
