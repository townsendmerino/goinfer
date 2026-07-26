package decoder

import "sort"

// ResidentFeature is one architecture capability a resident (GPU) decode path must implement
// in order to run a model CORRECTLY.
//
// The failure mode this taxonomy exists to prevent is SILENT. A backend that admits a model
// needing a feature it has not implemented raises no error — it simply drops the feature and
// emits wrong logits (docs/metal-model-coverage.md found exactly this: Qwen3 running with
// QK-norm ignored, Mistral running full-attention past its window). Admission must therefore
// be a subset check: RequiredResidentFeatures(model) ⊆ the backend's implemented set, else
// decline to the staged/CPU path.
//
// Requirements are DERIVED from the loaded Architecture's own flags — never a hand-maintained
// per-arch list. That is the point: a newly registered arch is classified automatically, and a
// backend that has not implemented its features declines by default instead of mis-running.
// Adding a field to Architecture that changes the math means adding it here too; the
// registry-driven test (features_test.go) is what makes forgetting expensive.
//
// Note this layer sits ABOVE the arch flags, so it cannot catch an arch that fails to CLAIM a
// feature it needs — e.g. phi3Architecture once dropped sliding_window entirely, which made
// every path (CPU included) silently wrong. That class is a registry bug, caught by
// per-family parity, not by admission.
type ResidentFeature string

const (
	FeatQKNorm         ResidentFeature = "qk-norm"          // per-head Q/K RMSNorm before RoPE (Qwen3, GLM, Mellum)
	FeatSlidingWindow  ResidentFeature = "sliding-window"   // windowed attention (Mistral, Mellum, Phi-3-mini)
	FeatPartialRotary  ResidentFeature = "partial-rotary"   // RotaryDim < HeadDim (GLM-dense, some Phi)
	FeatPerLayerRoPE   ResidentFeature = "per-layer-rope"   // inv-freq/mscale differ per layer type (Mellum)
	FeatRopeMscale     ResidentFeature = "yarn-mscale"      // YaRN attention_factor != 1 (Mellum, long-ctx)
	FeatRMSAddOne      ResidentFeature = "rms-add-one"      // Gemma's (1+w) RMSNorm offset
	FeatEmbedScale     ResidentFeature = "embed-scale"      // √hidden embedding multiplier (Gemma)
	FeatLogitSoftcap   ResidentFeature = "logit-softcap"    // attention / final logit softcap (Gemma 2/3)
	FeatSandwichNorm   ResidentFeature = "sandwich-norm"    // post-attn / post-MLP norms (Gemma)
	FeatGatedGELU      ResidentFeature = "gated-gelu"       // gated MLP whose activation is GELU-tanh, not SiLU (Gemma)
	FeatNonGatedMLP    ResidentFeature = "non-gated-mlp"    // up→act→down, no gate (GPT-2, Nemotron relu²)
	FeatLearnedPos     ResidentFeature = "learned-pos"      // learned position embeddings, no RoPE (GPT-2)
	FeatOutBias        ResidentFeature = "out-bias"         // additive bias on the attn output proj (GPT-2)
	FeatLogitScale     ResidentFeature = "logit-scale"      // logits_scaling divisor (Granite)
	FeatMoE            ResidentFeature = "moe"              // sparse mixture-of-experts FFN
	FeatMoEGatedShared ResidentFeature = "moe-gated-shared" // sigmoid-GATED always-on shared expert (Qwen2-MoE); ungated (GLM/DeepSeek) needs only FeatMoE
	FeatMLA            ResidentFeature = "mla"              // latent-KV attention (DeepSeek, Kimi)
	FeatSSM            ResidentFeature = "ssm"              // Mamba-2 mixer (Granite-4.0-H, Nemotron-H)
)

// residentFeatures derives the features this architecture actually needs from its own flags.
func (a *Architecture) residentFeatures() []ResidentFeature {
	var f []ResidentFeature
	add := func(need bool, x ResidentFeature) {
		if need {
			f = append(f, x)
		}
	}
	add(a.QKNorm, FeatQKNorm)
	add(a.SlidingWindow > 0, FeatSlidingWindow)
	// MLA (DeepSeek/Kimi) carries rope on a decoupled qk_rope slice handled INSIDE the MLA kernel
	// (FeatMLA), so its RotaryDim<HeadDim is not the generic partial-rope path — don't double-count
	// it as FeatPartialRotary (C6: the hand table lists MLA archs as [mla moe]).
	add(a.mla == nil && a.RotaryDim != 0 && a.RotaryDim < a.HeadDim, FeatPartialRotary)
	add(!a.ropeUniform(), FeatPerLayerRoPE)
	// YaRN attention_factor on ANY layer, not just layer 0 (C6). Mellum interleaves 3:1 and puts
	// YaRN only on its full_attention layers — layer 0 is a sliding/default layer (mscale 1), so a
	// layer-0 sample missed it and over-admitted Mellum2 as resident on backends that don't apply
	// the full-layer scaling. Mirrors ropeUniform's all-layer loop just above.
	yarnMscale := false
	for i := 0; i < a.NumLayers; i++ {
		if a.ropeMscale(i) != 1 {
			yarnMscale = true
			break
		}
	}
	add(yarnMscale, FeatRopeMscale)
	add(a.RMSAddOne, FeatRMSAddOne)
	add(a.EmbedScale > 1, FeatEmbedScale)
	add(a.FinalLogitSoftcap != 0 || a.AttnLogitSoftcap != 0, FeatLogitSoftcap)
	add(a.NormPlacement != NormPre2, FeatSandwichNorm)
	// The MLP ACTIVATION, not just the gate's presence. A backend whose glue kernel hardcodes
	// SwiGLU would run Gemma's gated GELU-tanh block silently wrong — the exact failure this
	// taxonomy exists to catch, and the one it missed until CUDA went looking at Gemma. Scoped to
	// GATED archs on purpose: a non-gated arch's activation (GPT-2 gelu, Nemotron relu²) is
	// already implied by FeatNonGatedMLP, whose kernel is written for that family's activation.
	//
	// Tested against SiLU rather than for GeluTanh deliberately: ActGeluTanh is ActKind's ZERO
	// value, so an arch that forgets to set Act reads as GELU and lands here — declined, CPU
	// fallback, correct-but-slow. That is the safe direction; matching on GeluTanh would instead
	// let a forgotten Act sail through onto a SwiGLU kernel.
	add(!a.NonGatedMLP && a.Act != ActSiLU, FeatGatedGELU)
	add(a.NonGatedMLP, FeatNonGatedMLP)
	add(a.LearnedPosEmbed, FeatLearnedPos)
	add(a.OutBias, FeatOutBias)
	add(a.LogitScale != 0 && a.LogitScale != 1, FeatLogitScale)
	add(a.MoE != nil, FeatMoE)
	// A sigmoid-GATED always-on shared expert (Qwen2-MoE: out += sigmoid(SharedGate·h)·shared(h))
	// is a distinct kernel from the ungated add (GLM/DeepSeek: out += shared(h)). CUDA implements
	// only the ungated combine and DECLINES the gated one (cuda/backend.go); Metal and WebGPU
	// implement both. Splitting it out moves that decline from a hand-coded backend check into the
	// shared taxonomy, so the hardware matrix matches admission.
	add(a.MoE != nil && a.MoE.SharedIntermediateDim > 0 && !a.MoE.SharedUngated, FeatMoEGatedShared)
	add(a.mla != nil, FeatMLA)
	add(a.granite != nil || a.nemotron != nil, FeatSSM)
	sort.Slice(f, func(i, j int) bool { return f[i] < f[j] })
	return f
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

// ResidentEligible reports whether `backend` can run architecture `a` on its resident (GPU)
// decode path — the CAPABILITY predicate, model-free (arch flags only). It is exactly the two
// gates every resident admission already applies, composed in one place: the arch is a shape the
// runner supports (decodeRunnerEligible) AND the backend implements every feature the arch needs
// (the shared taxonomy). The runtime and the hardware-matrix generator both derive from these
// same pieces, so the published table can never disagree with what a backend can run.
//
// Scope note (deliberate, matches capability_matrix's GPUResident = decodeRunnerEligible): this is
// arch-level CAPABILITY, not a runtime admission. The runtime additionally applies load-time
// POLICY that a model-free predicate cannot know — the Nemotron int4-only / GOINFER_SSM_RESIDENT
// precision gate (Model.DecodeRunnerEligible). Those are precision choices, not "can this backend
// run this family", so the matrix shows capability and footnotes the policy.
func ResidentEligible(a *Architecture, backend string) bool {
	impl, ok := ResidentBackendFeatures[backend]
	if !ok {
		return false
	}
	return a.decodeRunnerEligible() &&
		len(missingFeatures(a.residentFeatures(), impl)) == 0 &&
		residentMoECapacityOK(a, backend)
}

// residentBackendMoECap is the router-kernel capacity of each backend whose MoE scoreboard is a
// FIXED-SIZE array — the numeric twin of ResidentBackendFeatures (a feature is "implemented at
// all"; a cap is "implemented up to N"). gpu/moe.go scores into array<f32,256> and its group-limited
// path into array<f32,32>/array<bool,32>; cuda/backend.go rejects >256 identically. A model past the
// cap would route on only the first N experts (or index groups out of bounds) — plausible-looking
// WRONG output, no error — so it must decline to the staged/CPU path. This is exactly the runtime
// guard M22 added in gpu/backend.go BuildResident; declared here too so the hardware-matrix
// generator derives the same answer the runtime gives (the C6 one-source-of-truth discipline — a
// feature-only predicate silently over-admitted Kimi K2's 384 experts as WebGPU-resident). Absent
// entry = no fixed-size router cap (a backend that declines these archs on features never reaches
// this — Metal/CUDA decline Kimi on FeatMLA).
var residentBackendMoECap = map[string]struct{ experts, groups int }{
	"webgpu": {experts: 256, groups: 32},
	"cuda":   {experts: 256, groups: 32},
}

// residentMoECapacityOK reports whether backend's router kernel can route arch's MoE (M22). True for
// a dense arch or a backend without a fixed-size router.
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

// ResidentBackendFeatures declares what each resident backend's decode path implements.
//
// These live HERE, not in the backends, for two reasons. First, one source of truth: three
// hand-maintained copies of this logic is precisely how the silent-wrong-output bug recurs
// (Metal had it; CUDA had it; the audit found them independently). Second, testability — the
// backends are build-tagged (`-tags cuda`, `-tags gpu`, darwin-only metal), so a test that
// could see their sets could not run in CI. Declared here, the registry-driven admission gate
// (features_test.go) checks every (arch × backend) pair with no GPU present.
//
// A backend adds an entry ONLY when it ships the kernel that implements it. Overclaiming here
// is exactly the lie the gate exists to catch.
var ResidentBackendFeatures = map[string]map[ResidentFeature]bool{
	// cgo-free CUDA (cuda/): the dense Qwen2/Llama block, plus QK-norm, sliding window, the
	// Gemma set ((1+w) RMS, sandwich norms, GeGLU, embed scale, per-layer RoPE base), partial
	// rotary, and MoE (routed + ungated shared expert). Still NOT implemented: per-layer rotary
	// WIDTH (only per-layer base); no YaRN mscale; no logit softcap; no MLA/SSM.
	//
	// FeatMoE covers the ROUTED block (router + stacked experts + every routing flavour the
	// route kernel handles) AND the always-on UNGATED shared expert (GLM/DeepSeek). The GATED
	// shared expert (Qwen-MoE's sigmoid(SharedGate·h) scaling) is NOT wired — BuildResident
	// declines it at load, since no committed fixture gates it end to end and FeatMoE is one flag
	// that cannot express the sub-shape. That decline is the honest "admitted, but this variant
	// is not wired" in a table whose whole job is to not lie.
	//
	// FeatPartialRotary and the shared expert land together on purpose: every partial-rotary arch
	// (glm4_moe) also has a shared expert, so neither is independently reachable — glm-tiny is the
	// joint end-to-end gate (TestGLMResidentParity), and declaring partial rotary before the
	// shared expert existed would have admitted glm onto a path no model could exercise.
	"cuda": {
		FeatQKNorm:        true, // qk_norm kernel — per-head Q/K RMSNorm before RoPE (Qwen3)
		FeatSlidingWindow: true, // attention `window` uniform, per-layer via LayerIsLocalResident
		FeatPartialRotary: true, // rope_kv rhalf = rotaryDim/2 + un-rotated tail cached (GLM/Phi)
		FeatRMSAddOne:     true, // (1+w) offset, threaded through rmsnorm_quant/fused_rms_*/qk_norm
		FeatSandwichNorm:  true, // rmsnorm_f32 on each sublayer output (breaks the accum epilogue)
		FeatGatedGELU:     true, // glu_quant `act` — GeGLU as well as SwiGLU
		FeatEmbedScale:    true, // √hidden applied host-side in embedResident
		FeatPerLayerRoPE:  true, // per-layer invFreq buffer (Gemma local 10k vs global 1M base)
		FeatMoE:           true, // moe_route + indexed stacked experts + ungated shared expert
	},

	// WebGPU (gpu/): the richest runner — the levers in docs/gpu-residency-coverage.md.
	"webgpu": {
		FeatQKNorm:         true, // C1  per-head QK-norm before RoPE
		FeatPartialRotary:  true, // C5  rotary_dim < head_dim
		FeatSlidingWindow:  true, // C6  per-layer windowed start
		FeatPerLayerRoPE:   true, // C7  differing invFreq per layer type
		FeatRopeMscale:     true, // C7  YaRN attention_factor
		FeatMoE:            true, // C3a-d router / stacked experts / shared expert
		FeatMoEGatedShared: true, // sharedGatedCombine — sigmoid-gated shared expert (gpu/moe.go)
		FeatMLA:            true, // C4a-d latent-KV attention
		FeatSSM:            true, // Mamba-2 engine (Granite-4.0-H, Nemotron-H)
		FeatNonGatedMLP:    true, // relu2Quant (Nemotron-H squared-ReLU)
		FeatLogitScale:     true, // Granite logits_scaling
		FeatRMSAddOne:      true, // (1+w) RMS offset
	},

	// cgo-free Metal (metal/): dense Qwen2/Llama plus qk-norm, sliding-window, partial-rotary,
	// MoE (router + stacked experts + shared expert; metal/moe.go), and the full Gemma set —
	// sandwich norms, GeGLU, (1+w) RMS, √hidden embed scale, per-layer RoPE base. Gemma parity
	// was gated on the GELU-tanh overflow fix (glu_act clamp, 38a2b7c): logit cosine 0.818→0.994.
	// Still declines YaRN mscale, MLA and SSM.
	"metal": {
		FeatQKNorm:         true, // qk_norm kernels
		FeatSlidingWindow:  true, // attention window uniform
		FeatPartialRotary:  true, // rope rhalf = rotaryDim/2
		FeatMoE:            true, // moe_route + indexed stacked-expert W4A8 GEMVs + shared expert (metal/moe.go)
		FeatMoEGatedShared: true, // shared_gate_combine — sigmoid-gated shared expert (metal/moe.go)
		FeatSandwichNorm:   true, // rmsnorm_f32 on each sublayer output (Gemma)
		FeatGatedGELU:      true, // GeGLU — clamped-tanh geglu (glu_act, 38a2b7c)
		FeatRMSAddOne:      true, // (1+w) RMS offset
		FeatEmbedScale:     true, // √hidden embedding multiplier (embedResident)
		FeatPerLayerRoPE:   true, // per-layer invFreq (Gemma local 10k vs global 1M base)
	},
}
