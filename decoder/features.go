package decoder

import (
	"fmt"
	"maps"
	"slices"
)

// ResidentFeature is one architecture capability a resident (GPU) decode path must implement to run a model correctly.
//
// The failure this taxonomy prevents is silent: a backend that admits a model needing a feature it has not implemented raises no
// error, it drops the feature and emits wrong logits. Admission is therefore a subset check: RequiredResidentFeatures(model) must be
// contained in the backend's implemented set, else decline to the staged/CPU path.
//
// Requirements are derived from the loaded Architecture's own flags, never a hand-maintained per-arch list, so a newly registered arch is
// classified automatically. Adding a field to Architecture that changes the math means adding it here too; the registry-driven test
// (features_test.go) is what makes forgetting expensive. This layer sits above the arch flags, so it cannot catch an arch that fails to
// claim a feature it needs: that is a registry bug, caught by per-family parity. History: docs/code-notes/decoder.md#ResidentFeature.
type ResidentFeature string

const (
	FeatQKNorm            ResidentFeature = "qk-norm"             // per-head Q/K RMSNorm before RoPE (Qwen3, GLM, Mellum)
	FeatSlidingWindow     ResidentFeature = "sliding-window"      // windowed attention (Mistral, Mellum, Phi-3-mini)
	FeatPartialRotary     ResidentFeature = "partial-rotary"      // RotaryDim < HeadDim (GLM-dense, some Phi)
	FeatPerLayerRoPE      ResidentFeature = "per-layer-rope"      // inv-freq/mscale differ per layer type (Mellum)
	FeatRopeMscale        ResidentFeature = "yarn-mscale"         // YaRN attention_factor != 1 (Mellum, long-ctx)
	FeatRMSAddOne         ResidentFeature = "rms-add-one"         // Gemma's (1+w) RMSNorm offset
	FeatEmbedScale        ResidentFeature = "embed-scale"         // √hidden embedding multiplier (Gemma)
	FeatAttnLogitSoftcap  ResidentFeature = "attn-logit-softcap"  // per-layer attention-score softcap (Gemma 2) — a real attention kernel
	FeatFinalLogitSoftcap ResidentFeature = "final-logit-softcap" // softcap·tanh(logits/softcap) applied ONCE after the LM head, host-side (Gemma 2/4)
	FeatSandwichNorm      ResidentFeature = "sandwich-norm"       // post-attn / post-MLP norms (Gemma)
	FeatLayerNorm         ResidentFeature = "layer-norm"          // mean-subtracting LayerNorm instead of RMSNorm (Cohere, GPT-2)
	FeatParallelBlock     ResidentFeature = "parallel-block"      // one shared input norm; attn+MLP sum into a single residual add (Cohere, GPT-J)
	FeatNoPE              ResidentFeature = "nope"                // per-layer NoPE: some layers skip RoPE entirely (Cohere2 global layers, Llama-4 iRoPE)
	FeatGatedGELU         ResidentFeature = "gated-gelu"          // gated MLP whose activation is GELU-tanh, not SiLU (Gemma)
	FeatNonGatedMLP       ResidentFeature = "non-gated-mlp"       // up→act→down, no gate (GPT-2, Nemotron relu²)
	FeatLearnedPos        ResidentFeature = "learned-pos"         // learned position embeddings, no RoPE (GPT-2)
	FeatOutBias           ResidentFeature = "out-bias"            // additive bias on the attn output proj (GPT-2)
	FeatLogitScale        ResidentFeature = "logit-scale"         // logits_scaling divisor (Granite)
	FeatMoE               ResidentFeature = "moe"                 // sparse mixture-of-experts FFN
	FeatMoEGatedShared    ResidentFeature = "moe-gated-shared"    // sigmoid-GATED always-on shared expert (Qwen2-MoE); ungated (GLM/DeepSeek) needs only FeatMoE
	FeatMLA               ResidentFeature = "mla"                 // latent-KV attention (DeepSeek, Kimi)
	FeatSSM               ResidentFeature = "ssm"                 // Mamba-2 mixer (Granite-4.0-H, Nemotron-H)
	// FeatDeltaNet bundles the two departures of the Gated-DeltaNet hybrids, which always co-occur: the recurrent delta rule replacing softmax
	// attention on 3 of every 4 layers (a fixed-size per-head matrix state: no KV cache, not position-truncatable), and attn_output_gate on the
	// remaining layers (q_proj emits [query ‖ gate] per head at double width; the context is scaled by sigmoid(gate) before o_proj). A backend
	// implementing one without the other would be admitted and silently run half the model wrong. The gate is not FeatAttnOutputGate's (a
	// separate g_proj through softplus), so declaring one must not admit the other.
	FeatDeltaNet ResidentFeature = "deltanet" // Gated-DeltaNet mixer + fused attn output gate
	// FeatAttnSink bundles gpt-oss's three departures: the learned per-head sink in the softmax denominator, the clamped interleaved-SwiGLU
	// expert, and a router whose bias reaches the mixing weight rather than only the selection. A backend declares all three or none, and only
	// after a real gpt-oss forward has run end to end on it: kernel-level parity is not end-to-end parity. History and the per-backend status:
	// docs/code-notes/decoder.md#FeatAttnSink.
	FeatAttnSink       ResidentFeature = "attn-sink"        // gpt-oss: softmax sink, clamped-SwiGLU expert, bias-weighted router (see above)
	FeatAttnOutputGate ResidentFeature = "attn-output-gate" // attention output gate applied before o_proj (Laguna: ctx *= softplus(g_proj·h); Spark-X2.5: sigmoid), plus Laguna's per-layer query head count; a resident runner would skip both
	FeatShortConv      ResidentFeature = "short-conv"       // LFM2 gated short-convolution mixer replacing attention on most layers (depthwise causal conv, a per-layer rolling window of the last K-1 inputs: stateful)
	FeatGemma4EModel   ResidentFeature = "gemma4-e-model"   // Gemma-4 E2B/E4B: per-layer embeddings (PLE, hidden_size_per_layer_input>0), cross-layer shared KV, per-layer FFN width; runLayersGemma4 injects PLE per layer. A backend without the PLE branch would skip it and mis-run
	// FeatAttnTemp is the Llama4-style attention-temperature query scale (AttnTempBeta/AttnTempOrigMaxPos; Model.AttnTempScale,
	// AttnTempParams). A path that does not apply it gives plausible-but-wrong logits at exactly the context lengths past
	// original_max_position_embeddings it exists for, and the model otherwise needs nothing a backend lacks, so without this feature it
	// would be admitted and mis-run.
	FeatAttnTemp ResidentFeature = "attn-temp"
	// FeatPostOnlyNorm is NormPostOnly: no pre-norm at all, the sublayer's output is normalized before the residual add. Not
	// FeatSandwichNorm, which normalizes both the input and the output.
	FeatPostOnlyNorm ResidentFeature = "post-only-norm"
	// FeatQKNormWhole is QK-norm over the full projected q/k vector (one RMSNorm over num_heads*head_dim) rather than per head: a different
	// statistic and a differently shaped weight tensor, so its own feature, not a variant of FeatQKNorm. An implementation that reuses the
	// per-head kernel with its grid collapsed to one whole-vector Q block and one K block is valid only for MHA (nH==nKV) and must decline
	// otherwise.
	FeatQKNormWhole ResidentFeature = "qk-norm-whole"
	// FeatKDA is Kimi Delta Attention's linear-attention mixer: a delta-rule recurrence like Gated DeltaNet (FeatDeltaNet) but with a
	// per-channel decay (one value per row of the state matrix) where Gated DeltaNet's is one scalar per head. A different recurrence, so its
	// own feature, not a FeatDeltaNet variant.
	FeatKDA ResidentFeature = "kda"
	// FeatPairwiseRoPE is GPT-J pairwise rotation (dims 2d, 2d+1) in the scalar decode/prefill rope (Cohere, Cohere2, Aya, GLM-OCR's text
	// rows). Every generic resident rope kernel is the NeoX half-split rotation of pairs (d, d+half); a runner on those is exact at position 0
	// (the rotation is the identity there) and wrong at every later one, with no error and fluent-looking logits. A backend declares it only
	// with pairwise rope kernels, bound in place of the NeoX ones when Model.PairwiseRoPEResident(); its gate must use peaked attention and
	// non-flat weights, because flat 0.02-std fixtures hide a wrong rotation. MLA carries its own interleave flag and is gated by FeatMLA.
	// History: docs/code-notes/decoder.md#FeatPairwiseRoPE; the kernel port notes are in docs/measurements/cuda-pairwise-rope-2026-10-01.md.
	FeatPairwiseRoPE ResidentFeature = "pairwise-rope"
	// FeatPairwiseMRoPE is pairwise rotation on a family that also carries m-RoPE sections (ropeInterleave with MRopeSection set; GLM-OCR):
	// the image-block prefill rotates each frequency by its own (t,h,w) component through the pairwise pairs, and the generic m-RoPE
	// image-prefill kernels are NeoX. Needed in addition to FeatPairwiseRoPE, since the family's text rows and decode use the scalar kernel.
	FeatPairwiseMRoPE ResidentFeature = "pairwise-mrope"
)

// residentFeatures derives the features this architecture actually needs from its own flags.
func (a *Architecture) residentFeatures() []ResidentFeature {
	var f []ResidentFeature
	add := func(need bool, x ResidentFeature) {
		if need {
			f = append(f, x)
		}
	}
	// FeatQKNorm is the per-head kernel; QKNormWhole needs FeatQKNormWhole instead (added below), not both: a backend implementing only the
	// per-head kernel must not be admitted for the whole-vector families.
	add(a.QKNorm && !a.QKNormWhole, FeatQKNorm)
	add(a.SlidingWindow > 0, FeatSlidingWindow)
	// MLA (DeepSeek/Kimi) carries rope on a decoupled qk_rope slice handled inside the MLA kernel (FeatMLA), so its RotaryDim<HeadDim is not
	// the generic partial-rope path: do not double-count it as FeatPartialRotary.
	add(a.mla == nil && a.RotaryDim != 0 && a.RotaryDim < a.HeadDim, FeatPartialRotary)
	add(!a.ropeUniform(), FeatPerLayerRoPE)
	// YaRN attention_factor on ANY layer, not just layer 0: Mellum interleaves 3:1 and puts YaRN only on its full_attention layers, so a
	// layer-0 sample misses it. Mirrors ropeUniform's all-layer loop.
	yarnMscale := false
	for i := 0; i < a.NumLayers; i++ {
		if a.ropeMscale(i) != 1 {
			yarnMscale = true
			break
		}
	}
	add(yarnMscale, FeatRopeMscale)
	add(a.RMSAddOne, FeatRMSAddOne)
	add(a.AttnTempBeta != 0, FeatAttnTemp)
	add(a.EmbedScale > 1, FeatEmbedScale)
	// The attention-score softcap is a per-layer kernel; the final-logit softcap is one host-side tanh after the LM head (like
	// FeatEmbedScale's √hidden). Backends declare each separately: Gemma 4 needs only the latter.
	add(a.AttnLogitSoftcap != 0, FeatAttnLogitSoftcap)
	add(a.FinalLogitSoftcap != 0, FeatFinalLogitSoftcap)
	// Gate on the SPECIFIC placement, not "anything but Pre2" — NormParallel is a
	// third placement whose kernel is FeatParallelBlock, not sandwich norms.
	add(a.NormPlacement == NormSandwich4, FeatSandwichNorm)
	add(a.NormPlacement == NormPostOnly, FeatPostOnlyNorm)
	add(a.QKNormWhole, FeatQKNormWhole)
	add(a.NormPlacement == NormParallel, FeatParallelBlock)
	// Per-layer NoPE (Cohere2 global layers skip RoPE). A backend that ropes every layer would corrupt the NoPE layers, so it is a distinct
	// implemented-or-decline capability, not FeatPerLayerRoPE (differing rope tables).
	add(a.layerNoPE != nil, FeatNoPE)
	// Mean-subtracting LayerNorm (Cohere, GPT-2) is a distinct kernel from RMSNorm.
	add(a.Norm != NormRMS, FeatLayerNorm)
	// The MLP activation, not just the gate's presence: a glue kernel that hardcodes SwiGLU would run Gemma's gated GELU-tanh block silently
	// wrong. Scoped to gated archs on purpose: a non-gated arch's activation (GPT-2 gelu, Nemotron relu²) is implied by FeatNonGatedMLP.
	// Tested against SiLU rather than for GeluTanh deliberately: ActGeluTanh is ActKind's zero value, so an arch that forgets to set Act
	// lands here and is declined (correct but slow); matching on GeluTanh would let a forgotten Act through onto a SwiGLU kernel.
	add(!a.NonGatedMLP && a.Act != ActSiLU, FeatGatedGELU)
	add(a.NonGatedMLP, FeatNonGatedMLP)
	add(a.LearnedPosEmbed, FeatLearnedPos)
	add(a.OutBias, FeatOutBias)
	add(a.LogitScale != 0 && a.LogitScale != 1, FeatLogitScale)
	add(a.MoE != nil, FeatMoE)
	// A sigmoid-gated always-on shared expert (Qwen2-MoE: out += sigmoid(SharedGate·h)·shared(h)) is a distinct kernel from the ungated add
	// (GLM/DeepSeek: out += shared(h)), which needs only FeatMoE.
	add(a.MoE != nil && a.MoE.SharedIntermediateDim > 0 && !a.MoE.SharedUngated, FeatMoEGatedShared)
	add(a.mla != nil, FeatMLA)
	add(a.granite != nil || a.nemotron != nil, FeatSSM)
	add(a.qwen35 != nil, FeatDeltaNet)
	add(a.kda != nil, FeatKDA)
	// Pairwise rotation (Cohere/Cohere2/Aya/GLM-OCR) and, on top of it, pairwise m-RoPE sections
	// (GLM-OCR). pairwiseRoPE is the one predicate the resident accessor reads too, so the feature
	// and the kernel selection cannot drift.
	add(a.pairwiseRoPE(), FeatPairwiseRoPE)
	add(a.pairwiseRoPE() && len(a.MRopeSection) > 0, FeatPairwiseMRoPE)
	add(a.lfm2 != nil, FeatShortConv)
	add(a.gptoss != nil, FeatAttnSink)
	// Laguna's attention output gate and per-layer query-head count, and Spark-X2.5's sigmoid gate (arch.AttnGate == GateSigmoid): a generic
	// attention-output gate that a resident runner would silently skip. hasAttnOutputGate is the predicate the forward dispatches on too, so
	// the feature and the math cannot drift.
	add(a.hasAttnOutputGate(), FeatAttnOutputGate)
	// Gemma-4 E-model (E2B/E4B) shape: PLE, cross-layer shared KV and variable per-layer FFN width, all co-present. PLE alone catches every
	// real E-model; the shared-KV and FFN disjuncts make the decline complete against a PLE-less variant.
	add(a.gemma4 != nil && (a.gemma4.HiddenSizePerLayerInput > 0 || a.gemma4.SharedKVLayers > 0 || len(a.gemma4.FFNPerLayer) > 0), FeatGemma4EModel)
	slices.Sort(f)
	return f
}

// pairwiseRoPE reports whether the generic (non-MLA) scalar rope is GPT-J pairwise (dims 2d, 2d+1)
// rather than NeoX half-split. MLA carries its own interleave on mlaParams and is gated by FeatMLA;
// qwen35 passes interleave=false to ropeAt whatever ropeInterleave says (forward_qwen35.go), so it is
// excluded to match what the CPU path actually runs.
func (a *Architecture) pairwiseRoPE() bool {
	return a.ropeInterleave && a.mla == nil && a.qwen35 == nil
}

// ropeUniform reports whether every layer shares one inv-freq table and mscale. False ⇒ the
// backend must dispatch RoPE per layer type (Mellum's YaRN-on-global vs default-local).
func (a *Architecture) ropeUniform() bool {
	if a.NumLayers <= 1 {
		return true
	}
	base := a.ropeInvFreq(0)
	m0 := a.ropeMscale(0)
	for i := 1; i < a.NumLayers; i++ {
		if a.ropeMscale(i) != m0 {
			return false
		}
		inv := a.ropeInvFreq(i)
		if len(inv) != len(base) {
			return false
		}
		for j := range inv {
			if inv[j] != base[j] {
				return false
			}
		}
	}
	return true
}

// RequiredResidentFeatures returns the features a resident backend must implement to run this
// model correctly. Derived from the arch — see ResidentFeature.
func (m *Model) RequiredResidentFeatures() []ResidentFeature { return m.w.arch.residentFeatures() }

// MissingResidentFeatures returns the required features that `implemented` does not contain,
// sorted. Empty ⇒ a backend declaring that set can run m correctly, and admission is granted.
// Backends call this instead of hand-rolling their own decline list, so the taxonomy has one
// source of truth (three copies is how the bug recurs).
func (m *Model) MissingResidentFeatures(implemented map[ResidentFeature]bool) []ResidentFeature {
	return missingFeatures(m.w.arch.residentFeatures(), implemented)
}

func missingFeatures(required []ResidentFeature, implemented map[ResidentFeature]bool) []ResidentFeature {
	var missing []ResidentFeature
	for _, r := range required {
		if !implemented[r] {
			missing = append(missing, r)
		}
	}
	return missing
}

// ResidentEligible reports whether backend can run architecture a on its resident (GPU) decode path: the capability predicate, from arch
// flags alone. It composes the two gates every resident admission applies: the arch is a shape the runner supports
// (decodeRunnerEligible) and the backend implements every feature the arch needs (residentGateReason). The runtime and the
// hardware-matrix generator derive from the same pieces, so the published table cannot disagree with what a backend can run.
//
// It is capability, not runtime admission: load-time policy such as Nemotron's int4-only precision gate lives in
// Model.DecodeRunnerEligible, and the matrix footnotes it. "Arch flags only" is also not strictly true of granite: its admission gate is
// inside Architecture.decodeRunnerEligible and reads GOINFER_SSM_RESIDENT (the model's snapshot, or the live environment for an
// Architecture built without one). The matrix generator pins that variable empty, so the published table shows granite CPU-only on every
// backend (docs/hardware-matrix.md footnote); docs/code-notes/decoder.md#ResidentEligible.
func ResidentEligible(a *Architecture, backend string) bool {
	if _, ok := residentBackendFeatures[backend]; !ok {
		return false
	}
	return a.decodeRunnerEligible() && residentGateReason(a, backend) == ""
}

// residentGateReason is ResidentEligible's gates after the runner-shape one: the backend implements every feature the arch needs, its MoE
// router is big enough, and it has the per-layer geometry and Gemma 4 MoE seams the arch uses. It returns why it declines, "" when all
// admit. One implementation serves the model-free predicate (the hardware matrix) and the load path (Model.residentAdmission), so the
// table and the runtime cannot disagree and a decline reaches `serve check` / DecodePath with its real cause.
func residentGateReason(a *Architecture, backend string) string {
	return residentGateReasonAct(a, backend, false)
}

// residentGateReasonAct is residentGateReason for a model whose resident projections run W8A8 with per-32 activation scales (actSafe):
// that clears ActivationQuantHazard on a backend that implements it (cuda, actgroup.cu), since the hazard is the per-vector activation
// scale. The hardware matrix passes false (the default configuration); residentAdmission passes the model's own.
func residentGateReasonAct(a *Architecture, backend string, actSafe bool) string {
	impl, ok := residentBackendFeatures[backend]
	if !ok {
		return fmt.Sprintf("backend %q declares no resident feature set", backend)
	}
	// No resident GPU backend has an f32-activation projection, so a family whose output int8 activations destroy runs on the CPU at
	// weight-only precision instead of fast and wrong. Checked here rather than in decodeRunnerDecline so the generated matrix shows it.
	if why := ActivationQuantHazard(a.Name); why != "" && !(actSafe && (backend == "cuda" || backend == "metal")) {
		return "every resident " + backend + " projection quantizes activations to int8, and " + why
	}
	if missing := missingFeatures(a.residentFeatures(), impl); len(missing) > 0 {
		return fmt.Sprintf("%s does not implement %v, which this model needs", backend, missing)
	}
	if !residentMoECapacityOK(a, backend) {
		c := residentBackendMoECap[backend]
		return fmt.Sprintf("MoE with %d experts / %d expert groups exceeds %s's router capacity (%d / %d)",
			a.MoE.NumExperts, a.MoE.NGroup, backend, c.experts, c.groups)
	}
	if !residentPerLayerGeomOK(a, backend) {
		return fmt.Sprintf("per-layer attention geometry (head_dim %d on local layers, %d on global) is not implemented on %s",
			a.HeadDim, a.gemma4.GlobalHeadDim, backend)
	}
	if !residentGemma4MoEOK(a, backend) {
		return fmt.Sprintf("Gemma 4's parallel dense+MoE FFN is not implemented on %s", backend)
	}
	return ""
}

// residentBackendMoECap is the router-kernel capacity of each backend whose MoE scoreboard is a fixed-size array: the numeric twin of
// the feature sets (a feature is "implemented at all", a cap "implemented up to N"). A model past the cap would route on only the first
// N experts or index groups out of bounds, with plausible-looking wrong output and no error, so it declines to the staged/CPU path. An
// absent entry means no fixed-size router. Each value mirrors a constant in the backend's kernel source (gpu/moe.go MAXE, cuda/moe.cu
// MOE_MAX_E/MOE_MAX_G, metal/moe.go score arrays): change both together. Raising cuda's cap changes moe_route's scratch in the frozen
// cuda/testdata/moe.ptx, which must be regenerated at the pinned toolchain (cuda/testdata/REGEN.md); raising metal's needs Mac
// validation. History: docs/code-notes/decoder.md#residentBackendMoECap.
var residentBackendMoECap = map[string]struct{ experts, groups int }{
	"webgpu": {experts: 512, groups: 32}, // gpu/moe.go: MAXE 512, array<f32,512> score/sel / array<f32,32> gscore
	"cuda":   {experts: 512, groups: 64}, // cuda/moe.cu: MOE_MAX_E 512 / MOE_MAX_G 64. 256→512 raised deliberately (see below); groups was 32→64 by audit M-17
	"metal":  {experts: 256, groups: 64}, // metal/moe.go: float score[256]/sel[256], gscore[64]/keep[64]; guarded at build (moe.go:375-376)
}

// ResidentBackendMoECap returns backend's declared router-kernel capacity; ok is false for a backend with no fixed-size router.
//
// Exported for the backends to read in their own admission, instead of hardcoding a copy: two copies drift, and ResidentEligible then
// admits what BuildResident declines. History: docs/code-notes/decoder.md#ResidentBackendMoECap.
func ResidentBackendMoECap(backend string) (experts, groups int, ok bool) {
	c, ok := residentBackendMoECap[backend]
	return c.experts, c.groups, ok
}

// residentMoECapacityOK reports whether backend's router kernel can route arch's MoE; true for a dense arch or a backend without a
// fixed-size router.
func residentMoECapacityOK(a *Architecture, backend string) bool {
	cap, ok := residentBackendMoECap[backend]
	if !ok || a.MoE == nil {
		return true
	}
	if cap.experts > 0 && a.MoE.NumExperts > cap.experts {
		return false
	}
	if cap.groups > 0 && a.MoE.NGroup > cap.groups {
		return false
	}
	return true
}

// residentPerLayerGeomBackends declares which resident backends implement per-layer attention geometry: a layer's own head_dim and KV
// head count genuinely differing from another's (Gemma 4's local/global split, HeadDim vs gemma4.GlobalHeadDim), not just a per-layer
// RoPE table (FeatPerLayerRoPE, which this is not).
//
// It is not a ResidentFeature: no other family needs it, and Gemma 3 and dense Gemma 4 otherwise derive identical feature sets, so the
// taxonomy cannot name it. A backend without the seam would be admitted by the feature check alone and fail on upload, which is the
// decline this predicate makes deliberate. History: docs/code-notes/decoder.md#residentPerLayerGeomBackends.
var residentPerLayerGeomBackends = map[string]bool{"cuda": true, "metal": true, "webgpu": true}

// residentPerLayerGeomOK reports whether backend implements the per-layer geometry a's layers
// actually need. true for every arch that doesn't vary (the overwhelming majority); false only
// when the arch's layers genuinely differ AND the backend lacks the seam.
func residentPerLayerGeomOK(a *Architecture, backend string) bool {
	if a.gemma4 == nil || a.gemma4.GlobalHeadDim <= 0 || a.gemma4.GlobalHeadDim == a.HeadDim {
		return true // uniform head_dim across every layer — no backend needs a special seam
	}
	return residentPerLayerGeomBackends[backend]
}

// residentGemma4MoEBackends declares which resident backends implement Gemma 4's parallel dense+MoE FFN (enable_moe_block,
// gemma4MoeMLP): the joint dense‖MoE bridge, not just the per-layer attention geometry of dense Gemma 4.
var residentGemma4MoEBackends = map[string]bool{"cuda": true, "metal": true}

func residentGemma4MoEOK(a *Architecture, backend string) bool {
	if a.gemma4 == nil || a.MoE == nil {
		return true
	}
	return residentGemma4MoEBackends[backend]
}

// PerLayerGeomOK is residentPerLayerGeomOK's Model-level twin, exported so a backend's BuildResident can check it directly: that
// admission is hand-rolled from MissingResidentFeatures, not ResidentEligible, and the feature check alone would admit dense Gemma 4
// onto a backend without the per-layer geometry seam (see residentPerLayerGeomBackends).
func (m *Model) PerLayerGeomOK(backend string) bool { return residentPerLayerGeomOK(m.w.arch, backend) }

// Gemma4MoEOK is residentGemma4MoEOK's Model-level twin, exported so a resident backend's own
// BuildResident can check it directly.
func (m *Model) Gemma4MoEOK(backend string) bool { return residentGemma4MoEOK(m.w.arch, backend) }

// HasPerLayerGeometry reports whether the model's layers genuinely differ in attention geometry
// (e.g. Gemma 4's local/global head_dim split: 256 vs 512).
func (m *Model) HasPerLayerGeometry() bool {
	a := m.w.arch
	return a.gemma4 != nil && a.gemma4.GlobalHeadDim > 0 && a.gemma4.GlobalHeadDim != a.HeadDim
}

// ResidentBackendFeatures returns a copy of the feature set a resident backend implements (nil if the backend is unknown). The copy keeps
// the source map read-only from outside the package: a caller or third-party init() cannot add a feature claim its kernels do not
// implement, nor trigger a concurrent map write during a Load. The package's own admission path reads the unexported map.
func ResidentBackendFeatures(backend string) map[ResidentFeature]bool {
	src := residentBackendFeatures[backend]
	if src == nil {
		return nil
	}
	out := make(map[ResidentFeature]bool, len(src))
	maps.Copy(out, src)
	return out
}

// residentBackendFeatures declares what each resident backend's decode path implements. It lives here, not in the backends, for one
// source of truth (hand-maintained copies are how the silent-wrong-output bug recurs) and for testability: the backends are build-tagged,
// so a test that could see their sets could not run in CI, while the registry-driven admission gate (features_test.go) checks every
// (arch × backend) pair with no GPU present.
//
// A backend adds an entry only when it ships the kernel that implements it, and only with an end-to-end gate on the real path:
// overclaiming here is exactly the lie the gate exists to catch. Per-entry history: docs/code-notes/decoder.md#residentBackendFeatures and the sections named after it.
var residentBackendFeatures = map[string]map[ResidentFeature]bool{
	// cgo-free CUDA (cuda/). FeatMoE covers the routed block (router, stacked experts, every routing flavour the route kernel handles) and
	// the ungated shared expert; the sigmoid-gated one is FeatMoEGatedShared. FeatPartialRotary and the shared expert were declared together
	// because every partial-rotary arch (glm4_moe) also has a shared expert, so neither is independently reachable: glm-tiny
	// (TestGLMResidentParity) is the joint gate. Per-layer rotary width is not implemented, only a per-layer base. FeatMLA's gate
	// (TestMLAResidentParityCUDA, deepseek-tiny with n_group=2, topk_group=1) must compare the full generated sequence: a transposed
	// nGroup/topkGroup argument passed a first-token comparison.
	"cuda": {
		FeatGemma4EModel:      true, // PLE branch in segBFFN, KV-shared layers aliasing their source cache, per-layer FFN widths
		FeatQKNorm:            true, // qk_norm kernel — per-head Q/K RMSNorm before RoPE (Qwen3)
		FeatSlidingWindow:     true, // attention `window` uniform, per-layer via LayerIsLocalResident
		FeatPartialRotary:     true, // rope_kv rhalf = rotaryDim/2 + un-rotated tail cached (GLM/Phi)
		FeatRMSAddOne:         true, // (1+w) offset, threaded through rmsnorm_quant/fused_rms_*/qk_norm
		FeatSandwichNorm:      true, // rmsnorm_f32 on each sublayer output (breaks the accum epilogue)
		FeatGatedGELU:         true, // glu_quant `act` — GeGLU as well as SwiGLU
		FeatEmbedScale:        true, // √hidden applied host-side in embedResident
		FeatFinalLogitSoftcap: true, // softcap·tanh(logits/softcap) host-side after readback (finalSoftcap)
		FeatPerLayerRoPE:      true, // per-layer invFreq buffer (Gemma local 10k vs global 1M base)
		FeatMoE:               true, // moe_route + indexed stacked experts + ungated shared expert
		// The sigmoid-gated shared expert (Qwen-MoE) is gated by qwen3_5_moe-tiny (TestQwen35ResidentParityCUDA), whose MoE block is
		// Qwen2-MoE's. Declaring it therefore also admits qwen2_moe on cuda, which has no fixture of its own: its admission rests on that
		// inheritance. Add one if that stops being good enough.
		FeatMoEGatedShared: true,
		// Gated-DeltaNet: the deltanet.ptx mixer (conv ring, delta rule, gated norm) plus the family's fused double-width q_proj and sigmoid
		// output gate, declared with its end-to-end gate (TestQwen35ResidentParityCUDA). It admits the dense sibling, and with
		// FeatMoEGatedShared the MoE ones.
		FeatDeltaNet: true,
		// FeatRopeMscale: YaRN's attention_factor, folded into cos/sin by rope, rope_kv and rope_kv_batched and threaded per layer from
		// Model.RopeMscaleLayer via cudaLayer.mscale. Declared only after the kernels' scale parameter was proven in isolation (TestRopeMscale)
		// and then end to end on real weights (TestMellumResidentParityCUDA). Declaring it admits Mellum: mellumArchitecture needs exactly
		// {FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow} and cuda already declared the other four.
		FeatRopeMscale: true,
		// gpt-oss: FeatAttnSink and FeatOutBias are declared together because the family needs both. FeatAttnSink is the kernels in
		// cuda/gptoss_act.cu with sinkArg threaded into both attention launches and launchGluSplitExpert dispatched from the MoE expert loop.
		// FeatOutBias needed no new kernel: gemv_quant.cu and the batched gemv_w4a8_rn fold the o_proj bias in before the accumulate select.
		// Declared only after a real gpt-oss-20b forward ran on this path (TestGptOssResidentParityCUDA, resident on an 8 GB card via
		// --moe-cache-experts); kernel-level parity is not end-to-end parity, and an earlier declaration on kernel evidence was reverted. The
		// defects found getting there were each a term the wiring dropped, not a kernel computing wrongly, so no kernel test could see them.
		FeatAttnSink: true,
		FeatOutBias:  true,
		// SmolLM3's NoPE layers get an all-zero per-layer invFreq table instead of a new kernel path (RopeInvFreqLayerResident folds it in for
		// every backend that reads it). Identity rotation at invFreq==0 holds only when mscale==1 on those layers, true of every layerNoPE
		// family admitted so far.
		FeatNoPE: true,
		// Ministral 3's post-RoPE query scale is folded into rope_kv's launch (a qTempScale parameter, applied to Q only, after the rotation;
		// Model.AttnTempScale/AttnTempParams supply the value); rope_kv_batched recomputes it per row device-side. beta==0 makes it 1, an exact
		// no-op for every other family. Gate: TestMinistral3ResidentParityCUDA.
		FeatAttnTemp: true,
		// Olmo 3 / Olmo Hybrid's no-pre-norm placement (Model.PostOnlyNormResident): segA quantizes the raw residual (quant_vec) instead of
		// running rmsnorm_quant, and the sandwich post-norm dispatch widens to sandwich || postOnly. Requires the fused QKV path off, which
		// bakes in a real pre-norm weight.
		FeatPostOnlyNorm: true,
		// FeatQKNormWhole reuses the per-head qk_norm kernel with its grid collapsed to one Q block and one K block (nH=1, nKV=1,
		// hd=nH_orig*hd_orig): no new kernel. Valid only for MHA (nH==nKV); BuildResident declines otherwise.
		FeatQKNormWhole: true,
		// Cohere/Command-R and Cohere2. FeatLayerNorm is a new kernel (layernorm_quant, cuda/glue.cu), bias-free only: a LayerNorm family with a
		// learned bias needs its own kernel. FeatParallelBlock reuses segA's pre-attn norm+quant as the MLP input (Model.ParallelBlockResident),
		// a sequencing change. FeatLogitScale is a host-side multiply after readback via Model.LogitScaleResident, not GraniteResidentParams'
		// copy of the same arch field.
		FeatLayerNorm:     true,
		FeatParallelBlock: true,
		FeatLogitScale:    true,
		FeatMLA:           true, // latent-KV attention (DeepSeek, Kimi)
		// GPT-J pairwise rotation via cuda/rope_pairwise.cu (rope_kv_pw, rope_kv_batched_pw, rope_kv_mrope_batched_pw), bound in place of the NeoX
		// rope pipelines when Model.PairwiseRoPEResident(). Gated by a peaked-attention test that goes red on the NeoX kernels
		// (TestPairwiseRoPEResidentParityCUDA), TestGlmOcrResidentParityCUDA and the real-checkpoint TestCohereRealResidentParityCUDA (heavy).
		FeatPairwiseRoPE:  true,
		FeatPairwiseMRoPE: true,
	},

	// WebGPU (gpu/): the richest runner — the levers in docs/gpu-residency-coverage.md.
	"webgpu": {
		FeatQKNorm:         true, // per-head QK-norm before RoPE
		FeatPartialRotary:  true, // rotary_dim < head_dim
		FeatSlidingWindow:  true, // per-layer windowed start
		FeatPerLayerRoPE:   true, // differing invFreq per layer type
		FeatRopeMscale:     true, // YaRN attention_factor
		FeatMoE:            true, // router / stacked experts / shared expert
		FeatMoEGatedShared: true, // sharedGatedCombine — sigmoid-gated shared expert (gpu/moe.go)
		FeatMLA:            true, // latent-KV attention
		FeatSSM:            true, // Mamba-2 engine (Granite-4.0-H, Nemotron-H)
		FeatDeltaNet:       true, // Gated-DeltaNet engine + fused attn output gate (gpu/deltanet.go)
		FeatNonGatedMLP:    true, // relu2Quant (Nemotron-H squared-ReLU)
		FeatLogitScale:     true, // Granite logits_scaling — folded into the lm_head weight scale at BuildResident, not a host-side postcap
		FeatRMSAddOne:      true, // (1+w) RMS offset
		// The Gemma3/Gemma4/gpt-oss set. FeatEmbedScale needs no gpu/ code: embedResident applies √hidden host-side for every backend before
		// resident.Forward.
		FeatEmbedScale: true,
		// FeatFinalLogitSoftcap: applySoftcap in gpu/softcap.go, host-side after readback.
		FeatFinalLogitSoftcap: true,
		// FeatSandwichNorm reuses the rmsnormF32 closure; it defeats the fused gemvAdd residual epilogue at the o-proj and MLP down-proj sites.
		FeatSandwichNorm: true,
		// FeatGatedGELU: the gegluShaderWGSL/gegluQuantWGSL kernel pair (gpu/layer.go, gpu/decodefuse.go). It clamps the tanh argument to ±15
		// before calling tanh, as Metal's does: without the clamp tanh overflows.
		FeatGatedGELU: true,
		// FeatOutBias: gpt-oss's o_proj bias, composed from a bare gemv then biasAdd (the general vec[i] += other[i] kernel); no new kernel.
		FeatOutBias: true,
		// FeatAttnSink: the learned sink threaded through every attention pipeline (attn, attn-keys, attn-f16, attn-i8 and the wide variants),
		// the clamped interleaved-SwiGLU expert (gptossGluQuantWGSL) and a router whose bias reaches the mixing weight (routeGptOssWGSL;
		// moeRouteWGSL's contract is wrong for this family). The sink needs its own per-layer uniform (HS in gpu/attention.go): hasSink is a
		// per-layer property that the geometry-cached P uniform's geomFor dedup cannot carry. Declared after a real gpt-oss forward ran
		// resident end to end (TestGptOssResidentParityWebGPU).
		FeatAttnSink: true,
	},

	// cgo-free Metal (metal/). It still declines MLA and SSM.
	//
	// FeatRopeMscale is shared with Mellum (the same required set minus FeatAttnSink/FeatOutBias), so declaring it for gpt-oss's YaRN also
	// admits Mellum on Metal, which is validated only on a real 4-layer weight slice (TestMellumResidentParity, skipped when the slice is
	// absent), not a full checkpoint. The flag is one boolean and reversible; if you are chasing a Mellum-on-Metal bug, start here.
	"metal": {
		FeatQKNorm:            true, // qk_norm kernels
		FeatSlidingWindow:     true, // attention window uniform
		FeatPartialRotary:     true, // rope rhalf = rotaryDim/2
		FeatMoE:               true, // moe_route + indexed stacked-expert W4A8 GEMVs + shared expert (metal/moe.go)
		FeatMoEGatedShared:    true, // shared_gate_combine — sigmoid-gated shared expert (metal/moe.go)
		FeatSandwichNorm:      true, // rmsnorm_f32 on each sublayer output (Gemma)
		FeatGatedGELU:         true, // GeGLU — clamped-tanh geglu (glu_act)
		FeatRMSAddOne:         true, // (1+w) RMS offset
		FeatEmbedScale:        true, // √hidden embedding multiplier (embedResident)
		FeatPerLayerRoPE:      true, // per-layer invFreq (Gemma local 10k vs global 1M base)
		FeatFinalLogitSoftcap: true, // softcap·tanh(logits/softcap) host-side after readback (metal/model.go: finalizeLogits)
		FeatLayerNorm:         true, // layernorm_quant — mean-centered norm+quant (GPT-2, generalized with a hasBias flag for Cohere)
		FeatNonGatedMLP:       true, // act_quant — up→act→down, no gate (GPT-2)
		FeatLearnedPos:        true, // addLearnedPos — host-side wpe[pos] add, RoPE dispatch skipped (GPT-2)
		FeatOutBias:           true, // gemv_w4a8_sa_bias_resid (o-proj) — GPT-2's attention output bias
		FeatRopeMscale:        true, // rope kernel's scale param (kernels.go), proven in isolation (TestRope_mscale) and end-to-end via gpt-oss's YaRN (TestGptOssResidentParity) — ALSO admits Mellum, see note above
		FeatAttnSink:          true, // attention sink term + clamped-SwiGLU MoE + custom router (gpt-oss) — TestGptOssResidentParity
		FeatDeltaNet:          true, // Gated-DeltaNet mixer + fused attn output gate (deltanet.go/deltanet_kernels.go) — TestQwen35ResidentParityMetal
		FeatNoPE:              true, // SmolLM3 NoPE layers — all-zero invFreq (RopeInvFreqLayer), no new kernel; see that function's comment
		FeatAttnTemp:          true, // Ministral 3 post-RoPE query scale (rope2's qTempScale param, Q-only, after rotation) — Model.AttnTempScale
		FeatPostOnlyNorm:      true, // Olmo 3 / Olmo Hybrid no-pre-norm — quant_vec on the raw residual instead of rmsnorm_quant; sandwich's post-norm dispatch widened to sandwich||postOnly
		FeatQKNormWhole:       true, // qk_norm's grid collapsed to one Q block + one K block (nH=1,nKV=1,hd=nH_orig*hd_orig) — no new kernel; MHA only
		// Cohere/Command-R and Cohere2. FeatLayerNorm is GPT-2's layernorm_quant with hasBias=0 (r.uLNHasBias). FeatParallelBlock reuses
		// encodeAttention's pre-attn r.aq/r.aSc as the MLP's gate|up input in encodeLayer instead of re-normalizing after the attention add
		// (Model.ParallelBlockResident). FeatLogitScale is a host-side multiply in finalizeLogits via Model.LogitScaleResident.
		FeatParallelBlock: true,
		FeatLogitScale:    true,
		// The Gemma 4 E-model shape: PLE inputs arrive as the embedding row's tail (ResidentEmbedLen) and run through encodePLE; a KV-shared
		// layer aliases its source's cache (residLayer.kvShared); each dense layer carries its FFN width (residLayer.ffnI). The f16 prefill and
		// the MC3 batched step decline it.
		FeatGemma4EModel: true,
		// GPT-J pairwise rotation via the twins rope_pw, rope2_pw, rope_f16_pw and rope_mrope_f16_pw, bound in place of the NeoX pipelines when
		// Model.PairwiseRoPEResident(); the MC3 batched step declines a pairwise model. Gated by TestRopePairwise_*, the peaked-attention
		// resident gates that go red on the NeoX kernels (TestPairwiseRoPEResidentParityMetal, TestGlmOcrResidentParityMetal) and real checkpoints.
		FeatPairwiseRoPE:  true,
		FeatPairwiseMRoPE: true,
	},
}
