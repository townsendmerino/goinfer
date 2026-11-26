package arch

import (
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
	FeatNonGatedMLP       ResidentFeature = "non-gated-mlp"       // up→act→down, no gate (GPT-2, Nemotron relu²); its activation is a feature of its own (FeatNonGatedGELUTanh and the rest)
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
	// implementing one without the other would be admitted and silently run half the model wrong. The gate is not FeatAttnOutputGateSoftplus's (a
	// separate g_proj through softplus), so declaring one must not admit the other.
	FeatDeltaNet ResidentFeature = "deltanet" // Gated-DeltaNet mixer + fused attn output gate
	// FeatAttnSink bundles gpt-oss's three departures: the learned per-head sink in the softmax denominator, the clamped interleaved-SwiGLU
	// expert, and a router whose bias reaches the mixing weight rather than only the selection. A backend declares all three or none, and only
	// after a real gpt-oss forward has run end to end on it: kernel-level parity is not end-to-end parity. History and the per-backend status:
	// docs/code-notes/decoder.md#FeatAttnSink.
	FeatAttnSink ResidentFeature = "attn-sink" // gpt-oss: softmax sink, clamped-SwiGLU expert, bias-weighted router (see above)
	// The attention output gate before o_proj, one feature per activation (a backend that builds one gate must not admit the
	// other), and Laguna's per-layer query-head count, a separate property a gate kernel does not cover.
	FeatAttnOutputGateSoftplus ResidentFeature = "attn-output-gate-softplus" // ctx *= softplus(g_proj·h) (Laguna)
	FeatAttnOutputGateSigmoid  ResidentFeature = "attn-output-gate-sigmoid"  // ctx *= sigmoid(g_proj·h) (Spark-X2.5)
	FeatPerLayerQueryHeads     ResidentFeature = "per-layer-query-heads"     // the query head count varies by layer (Laguna XS)
	FeatShortConv              ResidentFeature = "short-conv"                // LFM2 gated short-convolution mixer replacing attention on most layers (depthwise causal conv, a per-layer rolling window of the last K-1 inputs: stateful)
	FeatGemma4EModel           ResidentFeature = "gemma4-e-model"            // Gemma-4 E2B/E4B: per-layer embeddings (PLE, hidden_size_per_layer_input>0), cross-layer shared KV, per-layer FFN width; the Gemma 4 program walk injects PLE per layer. A backend without the PLE branch would skip it and mis-run
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
	// with pairwise rope kernels, bound in place of the NeoX ones when Model.BackendView().PairwiseRoPE(); its gate must use peaked attention and
	// non-flat weights, because flat 0.02-std fixtures hide a wrong rotation. MLA carries its own interleave flag and is gated by FeatMLA.
	// History: docs/code-notes/decoder.md#FeatPairwiseRoPE; the kernel port notes are in docs/measurements/cuda-pairwise-rope-2026-10-01.md.
	FeatPairwiseRoPE ResidentFeature = "pairwise-rope"
	// FeatPairwiseMRoPE is pairwise rotation on a family that also carries m-RoPE sections (ropeInterleave with MRopeSection set; GLM-OCR):
	// the image-block prefill rotates each frequency by its own (t,h,w) component through the pairwise pairs, and the generic m-RoPE
	// image-prefill kernels are NeoX. Needed in addition to FeatPairwiseRoPE, since the family's text rows and decode use the scalar kernel.
	FeatPairwiseMRoPE ResidentFeature = "pairwise-mrope"
	// FeatGatedGELUErf is a gated MLP whose activation is the exact erf GELU (ActGelu), not FeatGatedGELU's tanh approximation. The
	// gated kernels select by GatedAct's ordinal and know GELU-tanh and SiLU only, so any other ordinal runs one of those two
	// with no error. A backend declares this only with an erf branch in every gated kernel it runs and an end-to-end gate.
	FeatGatedGELUErf ResidentFeature = "gated-gelu-erf"
	// FeatNonGatedGELUErf is the same activation in a non-gated MLP (a GPT-2 checkpoint declaring "gelu" rather than "gelu_new"),
	// needed in addition to FeatNonGatedMLP. A separate feature from FeatGatedGELUErf because it is a separate kernel.
	FeatNonGatedGELUErf ResidentFeature = "non-gated-gelu-erf"
	// The other non-gated activations, one feature each, needed in addition to FeatNonGatedMLP: a backend's non-gated kernel
	// implements some and must not be handed another's ordinal. FeatNonGatedActUnknown names no kernel, as FeatGatedActUnknown.
	FeatNonGatedGELUTanh   ResidentFeature = "non-gated-gelu-tanh"
	FeatNonGatedSiLU       ResidentFeature = "non-gated-silu"
	FeatNonGatedReLU2      ResidentFeature = "non-gated-relu2"
	FeatNonGatedActUnknown ResidentFeature = "non-gated-act-unknown"
	// FeatGatedActUnknown is what a gated MLP derives when its activation is none of SiLU, GELU-tanh and exact GELU. It names no
	// kernel, so no backend may declare it (TestGatedAct_onlyTanhAndSiLUAdmitted): it exists so a new ActKind declines until
	// residentFeatures gives it a feature of its own.
	FeatGatedActUnknown ResidentFeature = "gated-act-unknown"
)

// ResidentFeatures derives the features this architecture actually needs from its own flags.
func (a *Architecture) ResidentFeatures() []ResidentFeature {
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
	add(a.MLA == nil && a.RotaryDim != 0 && a.RotaryDim < a.HeadDim, FeatPartialRotary)
	add(!a.ropeUniform(), FeatPerLayerRoPE)
	// YaRN attention_factor on ANY layer, not just layer 0: Mellum interleaves 3:1 and puts YaRN only on its full_attention layers, so a
	// layer-0 sample misses it. Mirrors ropeUniform's all-layer loop.
	yarnMscale := false
	for i := 0; i < a.NumLayers; i++ {
		if a.RoPEMscale(i) != 1 {
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
	add(a.LayerNoPE != nil, FeatNoPE)
	// Mean-subtracting LayerNorm (Cohere, GPT-2) is a distinct kernel from RMSNorm.
	add(a.Norm != NormRMS, FeatLayerNorm)
	// The MLP activation, not just the gate's presence: a glue kernel that hardcodes SwiGLU would run Gemma's gated GELU-tanh block silently
	// wrong. One feature per gated activation, and the default arm is deliberate: a gated activation without a feature of its own must
	// not derive one a backend declares, or GatedAct hands its ordinal to a kernel that knows only GELU-tanh and SiLU. An arch
	// that forgets to set Act reads as ActGeluTanh (ActKind's zero value) on the CPU and on a backend alike. A non-gated arch
	// derives one feature per activation in the same way.
	switch {
	case a.NonGatedMLP:
		switch a.Act {
		case ActGeluTanh:
			add(true, FeatNonGatedGELUTanh)
		case ActSiLU:
			add(true, FeatNonGatedSiLU)
		case ActReLU2:
			add(true, FeatNonGatedReLU2)
		case ActGelu:
			add(true, FeatNonGatedGELUErf)
		default:
			add(true, FeatNonGatedActUnknown)
		}
	case a.Act == ActSiLU:
	case a.Act == ActGeluTanh:
		add(true, FeatGatedGELU)
	case a.Act == ActGelu:
		add(true, FeatGatedGELUErf)
	default:
		add(true, FeatGatedActUnknown)
	}
	add(a.NonGatedMLP, FeatNonGatedMLP)
	add(a.LearnedPosEmbed, FeatLearnedPos)
	add(a.OutBias, FeatOutBias)
	add(a.LogitScale != 0 && a.LogitScale != 1, FeatLogitScale)
	add(a.MoE != nil, FeatMoE)
	// A sigmoid-gated always-on shared expert (Qwen2-MoE: out += sigmoid(SharedGate·h)·shared(h)) is a distinct kernel from the ungated add
	// (GLM/DeepSeek: out += shared(h)), which needs only FeatMoE.
	add(a.MoE != nil && a.MoE.SharedIntermediateDim > 0 && !a.MoE.SharedUngated, FeatMoEGatedShared)
	add(a.MLA != nil, FeatMLA)
	add(a.Granite != nil || a.Nemotron != nil, FeatSSM)
	add(a.Qwen35 != nil, FeatDeltaNet)
	add(a.KDA != nil, FeatKDA)
	// Pairwise rotation (Cohere/Cohere2/Aya/GLM-OCR) and, on top of it, pairwise m-RoPE sections
	// (GLM-OCR). pairwiseRoPE is the one predicate the resident accessor reads too, so the feature
	// and the kernel selection cannot drift.
	add(a.PairwiseRoPE(), FeatPairwiseRoPE)
	add(a.PairwiseRoPE() && len(a.MRopeSection) > 0, FeatPairwiseMRoPE)
	add(a.LFM2 != nil, FeatShortConv)
	add(a.HasAttnSinks(), FeatAttnSink)
	// The attention output gate, by the activation the forward applies (hasAttnOutputGate's two cases: Laguna's softplus, keyed on
	// arch.laguna, and arch.AttnGate == GateSigmoid), so the features and the math cannot drift; and Laguna's per-layer query-head
	// count where its config gives one.
	add(a.Laguna != nil, FeatAttnOutputGateSoftplus)
	add(a.AttnGate == GateSigmoid, FeatAttnOutputGateSigmoid)
	add(a.Laguna != nil && len(a.Laguna.HeadsPerLayer) > 0, FeatPerLayerQueryHeads)
	// Gemma-4 E-model (E2B/E4B) shape: PLE, cross-layer shared KV and variable per-layer FFN width, all co-present. PLE alone catches every
	// real E-model; the shared-KV and FFN disjuncts make the decline complete against a PLE-less variant.
	add(a.Gemma4 != nil && (a.Gemma4.HiddenSizePerLayerInput > 0 || a.Gemma4.SharedKVLayers > 0 || len(a.Gemma4.FFNPerLayer) > 0), FeatGemma4EModel)
	slices.Sort(f)
	return f
}

// PairwiseRoPE reports whether the generic (non-MLA) scalar rope is GPT-J pairwise (dims 2d, 2d+1)
// rather than NeoX half-split. MLA carries its own interleave on mlaParams and is gated by FeatMLA;
// qwen35 passes interleave=false to rope.At whatever ropeInterleave says (forward_qwen35.go), so it is
// excluded to match what the CPU path actually runs.
func (a *Architecture) PairwiseRoPE() bool {
	return a.RoPEInterleave && a.MLA == nil && a.Qwen35 == nil
}

// ropeUniform reports whether every layer shares one inv-freq table and mscale. False ⇒ the
// backend must dispatch RoPE per layer type (Mellum's YaRN-on-global vs default-local).
func (a *Architecture) ropeUniform() bool {
	if a.NumLayers <= 1 {
		return true
	}
	base := a.RoPEInvFreq(0)
	m0 := a.RoPEMscale(0)
	for i := 1; i < a.NumLayers; i++ {
		if a.RoPEMscale(i) != m0 {
			return false
		}
		inv := a.RoPEInvFreq(i)
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

// FeaturesOutsideProgram is the features a needs that its program does not express, so a table cannot admit them: the
// per-layer query-head count, a shape, and the gated activation of Gemma 4's MoE block, whose operation carries none. No table
// backend implements them. FeatGatedGELU, the tanh form every Gemma 4 checkpoint uses, is implemented by the block.
func (a *Architecture) FeaturesOutsideProgram() []ResidentFeature {
	var out []ResidentFeature
	for _, f := range a.ResidentFeatures() {
		switch {
		case f == FeatPerLayerQueryHeads:
		case (f == FeatGatedGELUErf || f == FeatGatedActUnknown) && a.Gemma4 != nil && a.MoE != nil:
		default:
			continue
		}
		out = append(out, f)
	}
	return out
}
