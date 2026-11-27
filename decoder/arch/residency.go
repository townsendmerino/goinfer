package arch

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// RunnerShapeDeclineSSM is runnerShapeDecline with the GOINFER_SSM_RESIDENT knob given (ssmResident), not read: Admit's form.
func (a *Architecture) RunnerShapeDeclineSSM(ssmResident bool) string {
	// Own-forward families (their non-uniform forward the uniform-layer runners can't express).
	// Only Gemma 4 (dense, env-gated) is bridged so far; the rest keep the staged path. A
	// bridged family does NOT early-return true — it FALLS THROUGH to the common feature/shape
	// checks below, so a later decline COMPOSES instead of being short-circuited past. That is
	// the trap the old blanket `return false` set for whoever narrows the next family here: move
	// one line to a conditional admit and it silently skips every check below it. Structuring the
	// bridged case as a fall-through makes that mistake un-writable (cf. the geometry port).
	switch {
	case a.GPTOSS != nil:
		// gpt-oss falls through to the common checks below, same discipline as gemma4: the sink
		// (FeatAttnSink) and the MoE bias-in-combine kernels are backend feature gates, not an
		// arch-level decline, so a backend without them declines there instead of here.
	case a.Gemma4 != nil:
		// Gemma 4 is admitted UNCONDITIONALLY: do not re-gate a family on an env var without a gate behind it (the flag reads
		// as caution and functions as a coverage hole). The gates are cuda.TestGemma4MoEScaled_residentParity and
		// TestGemma4DenseScaled_residentParity. Dense and MoE both fall through to the common checks below; softcap and the
		// E-model's PLE (FeatGemma4EModel) are per-backend feature gates. The parallel dense‖MoE FFN runs on cuda's own path,
		// routed around the generic MoE checks via HasGemma4MoE.
	case a.Qwen35 != nil:
		// Gated-DeltaNet hybrids (qwen3_5, qwen3_5_moe, qwen3_next) fall through, same discipline as gemma4/gpt-oss: the
		// recurrence and the fused output gate are a BACKEND capability (FeatDeltaNet), so a backend without them declines at
		// the feature gate, and falling through is what keeps the checks below composing. The MoE siblings also need FeatMoE,
		// from the ordinary MoE checks.
	case a.Llama4 != nil:
		return archparts.RunnerShapeOwnForward
	case a.LFM2 != nil:
		// LFM2/LFM2.5: its short-convolution layers (opShortConv) are not bridged. Declining HERE and not only via FeatShortConv is
		// deliberate: decodeRunnerEligible is an ARCH-SHAPE predicate, and 22 of 30 layers run a gated short
		// convolution with a rolling window that no uniform-layer runner can express, a property of the family
		// rather than of any backend's kernels. It also keeps the capability matrix honest: its GPU-resident
		// column is generated from this predicate, and without this case LFM2 would fall through to
		// ropeResidentCompatible and read "yes".
		return archparts.RunnerShapeOwnForward
	case a.Inkling2 != nil:
		// Inkling 2: every visit is one opInkling2 (attention-residual mixes, relative-bias attention with per-token sinks,
		// short convolutions, loop controllers) that no resident implements. Declined here as an arch shape, like LFM2, so a
		// backend admitting by feature set (webgpu) cannot read it as a plain attention+MoE model.
		return archparts.RunnerShapeOwnForward
	}
	// Granite-4.0-H resident SSM hybrid: its own mixer-kind path (Mamba-2 ⊕
	// attention), so it bypasses the standard GQA gates below. Admitted only
	// under ssmResident (GOINFER_SSM_RESIDENT, ssmResidentEnv).
	if a.Granite != nil {
		if !ssmResident {
			return archparts.RunnerShapeBringUp
		}
		return ""
	}
	// Nemotron-H resident (dense squared-ReLU hybrid): single-op-per-block Mamba-2 / NoPE-GQA / relu² MLP,
	// reusing the granite SSM engine. Arch-eligible; the int4-default-vs-int8-opt-in precision gate is applied
	// at the Model level (DecodeRunnerEligible).
	//
	// Nemotron 3 Nano adds a fourth block kind (MoE FFN, arch.MoE != nil). cuda and metal build no MoE block
	// and their per-layer switch has no default, so an unhandled kind would leave that layer's buffers nil: a
	// crash-or-garbage risk, not a clean decline. Decline here, at the one predicate every backend's admission
	// funnels through; decodeRunnerDecline overrides it for webgpu, which implements the block.
	if a.Nemotron != nil {
		if a.MoE != nil {
			return archparts.RunnerShapeNemoMoE
		}
		return ""
	}
	if a.MoE != nil && !a.moeResidentEligible() {
		return archparts.RunnerShapeMoE
	}
	// FFN / norm / head constraints common to both attention paths (sandwich norms, logit softcap, NonGatedMLP,
	// LearnedPosEmbed, OutBias) are all representable, and a backend that lacks one declines via the feature gate
	// (decoder/features.go), not here: this stays an ARCH-shape predicate, because a capability blocked here for everyone
	// blocks the family for a backend that implements it. GPT-2, the only LearnedPosEmbed family, has no RoPE, so
	// ropeResidentCompatible passes vacuously.
	// MLA (DeepSeek/Kimi) runs its own latent-attention path (gpu/mla.go), so the GQA/RoPE/QK-norm/sliding-window checks
	// don't apply; its decoupled RoPE rides a separate qk_rope slice, not HeadDim.
	if a.MLA != nil {
		return ""
	}
	// Standard GQA attention: QK-norm (Qwen3/GLM/Mellum) is handled (per-head RMSNorm
	// before RoPE), q/k/v bias (Qwen2) and the (1+w) RMS offset too. Partial RoPE
	// (GLM/Phi rotary_dim < HeadDim) is handled, sliding window, and per-layer-type RoPE
	// (Mellum YaRN-on-global vs default-local invFreq + mscale). The last gate is
	// ropeResidentCompatible below.
	if !a.ropeResidentCompatible() {
		return archparts.RunnerShapeRoPE
	}
	return ""
}

// ropeResidentCompatible is a coarse ARCH-level screen on the GENERIC (finalizeRoPE) local/global inv-freq tables:
// they must share a length. Gemma passes it trivially (both come from one model-level rd), and that is NOT proof the
// runner ropes Gemma correctly: its REAL per-layer tables (Layer(i).RopeInvFreq) differ in length, an inequality
// this check never sees. The real invariant (each bound invFreq buffer has exactly rhalf entries) is asserted per layer
// where the buffer meets the geometry (cuda/backend.go).
func (a *Architecture) ropeResidentCompatible() bool {
	return len(a.RoPEInvFreqLocal) == len(a.RoPEInvFreqGlobal)
}

// moeResidentEligible reports whether this arch's MoE is a shape the GPU resident
// runner handles. The route kernel now covers every routing flavor — softmax/sigmoid,
// optional selection bias, norm_topk_prob, routed scaling, and DeepSeek group-limited
// top-k (NGroup > 1) — plus the always-on shared expert and dense
// prefix layers (FirstKDense > 0), so any int8-stackable MoE qualifies.
func (a *Architecture) moeResidentEligible() bool {
	return a.MoE != nil
}
