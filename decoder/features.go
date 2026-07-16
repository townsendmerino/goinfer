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
	FeatQKNorm        ResidentFeature = "qk-norm"        // per-head Q/K RMSNorm before RoPE (Qwen3, GLM, Mellum)
	FeatSlidingWindow ResidentFeature = "sliding-window" // windowed attention (Mistral, Mellum, Phi-3-mini)
	FeatPartialRotary ResidentFeature = "partial-rotary" // RotaryDim < HeadDim (GLM-dense, some Phi)
	FeatPerLayerRoPE  ResidentFeature = "per-layer-rope" // inv-freq/mscale differ per layer type (Mellum)
	FeatRopeMscale    ResidentFeature = "yarn-mscale"    // YaRN attention_factor != 1 (Mellum, long-ctx)
	FeatRMSAddOne     ResidentFeature = "rms-add-one"    // Gemma's (1+w) RMSNorm offset
	FeatEmbedScale    ResidentFeature = "embed-scale"    // √hidden embedding multiplier (Gemma)
	FeatLogitSoftcap  ResidentFeature = "logit-softcap"  // attention / final logit softcap (Gemma 2/3)
	FeatSandwichNorm  ResidentFeature = "sandwich-norm"  // post-attn / post-MLP norms (Gemma)
	FeatNonGatedMLP   ResidentFeature = "non-gated-mlp"  // up→act→down, no gate (GPT-2, Nemotron relu²)
	FeatLearnedPos    ResidentFeature = "learned-pos"    // learned position embeddings, no RoPE (GPT-2)
	FeatOutBias       ResidentFeature = "out-bias"       // additive bias on the attn output proj (GPT-2)
	FeatLogitScale    ResidentFeature = "logit-scale"    // logits_scaling divisor (Granite)
	FeatMoE           ResidentFeature = "moe"            // sparse mixture-of-experts FFN
	FeatMLA           ResidentFeature = "mla"            // latent-KV attention (DeepSeek, Kimi)
	FeatSSM           ResidentFeature = "ssm"            // Mamba-2 mixer (Granite-4.0-H, Nemotron-H)
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
	add(a.RotaryDim != 0 && a.RotaryDim < a.HeadDim, FeatPartialRotary)
	add(!a.ropeUniform(), FeatPerLayerRoPE)
	add(a.ropeMscale(0) != 1, FeatRopeMscale)
	add(a.RMSAddOne, FeatRMSAddOne)
	add(a.EmbedScale > 1, FeatEmbedScale)
	add(a.FinalLogitSoftcap != 0 || a.AttnLogitSoftcap != 0, FeatLogitSoftcap)
	add(a.NormPlacement != NormPre2, FeatSandwichNorm)
	add(a.NonGatedMLP, FeatNonGatedMLP)
	add(a.LearnedPosEmbed, FeatLearnedPos)
	add(a.OutBias, FeatOutBias)
	add(a.LogitScale != 0 && a.LogitScale != 1, FeatLogitScale)
	add(a.MoE != nil, FeatMoE)
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
	// cgo-free CUDA (cuda/): the plain dense Qwen2/Llama block, plus per-head QK-norm and
	// sliding-window attention. Still NOT implemented: rmsnorm_quant has no (1+w) offset, the
	// rope kernel hardcodes half = hd/2 (no partial rotary / per-layer table / YaRN factor),
	// and there is no MoE/MLA/SSM path. Partial rotary is deliberately absent: its only arch
	// (glm4_moe) also needs MoE, so implementing it today would unlock nothing.
	"cuda": {
		FeatQKNorm:        true, // qk_norm kernel — per-head Q/K RMSNorm before RoPE (Qwen3)
		FeatSlidingWindow: true, // attention `window` uniform, per-layer via LayerIsLocalResident
	},

	// WebGPU (gpu/): the richest runner — the levers in docs/gpu-residency-coverage.md.
	"webgpu": {
		FeatQKNorm:        true, // C1  per-head QK-norm before RoPE
		FeatPartialRotary: true, // C5  rotary_dim < head_dim
		FeatSlidingWindow: true, // C6  per-layer windowed start
		FeatPerLayerRoPE:  true, // C7  differing invFreq per layer type
		FeatRopeMscale:    true, // C7  YaRN attention_factor
		FeatMoE:           true, // C3a-d router / stacked experts / shared expert
		FeatMLA:           true, // C4a-d latent-KV attention
		FeatSSM:           true, // Mamba-2 engine (Granite-4.0-H, Nemotron-H)
		FeatNonGatedMLP:   true, // relu2Quant (Nemotron-H squared-ReLU)
		FeatLogitScale:    true, // Granite logits_scaling
		FeatRMSAddOne:     true, // (1+w) RMS offset
	},

	// cgo-free Metal (metal/): dense Qwen2/Llama plus the three coverage levers it shipped
	// (docs/metal-model-coverage.md). Declines embed-scale and YaRN mscale.
	"metal": {
		FeatQKNorm:        true, // qk_norm kernels
		FeatSlidingWindow: true, // attention window uniform
		FeatPartialRotary: true, // rope rhalf = rotaryDim/2
	},
}
