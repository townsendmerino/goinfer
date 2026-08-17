package decoder

import (
	"encoding/json"
	"fmt"
	"math"
)

// RoPE frequency scaling. HF's rope_scaling object
// transforms the base inverse-frequency table so a model trained at one context
// length serves a longer one. Only the variants whose math this file implements
// load; the rest (yarn, longrope/su, dynamic) are rejected loudly by the
// adapter rather than loaded with the wrong positional frequencies.

// ropeScalingKind enumerates the rope_scaling.rope_type values this build
// supports. "none" is the implicit default (no rope_scaling object).
type ropeScalingKind int

const (
	ropeScaleNone   ropeScalingKind = iota
	ropeScaleLinear                 // positions divided by factor (inv_freq /= factor)
	ropeScaleLlama3                 // Llama-3.1+/3.2 piecewise wavelength interpolation
	ropeScaleYarn                   // YaRN NTK-by-parts interpolation (Mellum, long-context Qwen)
)

// ropeScaling holds the resolved scaling parameters. Only the fields the active
// kind needs are populated.
type ropeScaling struct {
	kind   ropeScalingKind
	factor float64 // linear + llama3 + yarn

	// llama3 only:
	lowFreqFactor  float64
	highFreqFactor float64

	origMaxPosition float64 // llama3 + yarn

	// yarn only:
	betaFast float64
	betaSlow float64
	mscale   float64 // attention_factor applied to cos/sin (1.0 for non-yarn)
	truncate bool    // floor/ceil the correction range (HF default true); gpt-oss sets false (exact — truncating shifts inv_freq by ~2e-4)
}

// parseRopeScaling reads config.json's rope_scaling object. A null/empty object
// means no scaling (returns nil, nil). An object whose rope_type this build
// doesn't implement is an error — the caller surfaces it so the load fails
// loudly instead of producing wrong frequencies. HF has used both "rope_type"
// and the legacy "type" key; both are accepted.
func parseRopeScaling(raw json.RawMessage) (*ropeScaling, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var obj struct {
		RopeType        string   `json:"rope_type"`
		Type            string   `json:"type"`
		Factor          float64  `json:"factor"`
		LowFreqFactor   float64  `json:"low_freq_factor"`
		HighFreqFactor  float64  `json:"high_freq_factor"`
		OrigMaxPosition float64  `json:"original_max_position_embeddings"`
		BetaFast        *float64 `json:"beta_fast"`
		BetaSlow        *float64 `json:"beta_slow"`
		AttentionFactor *float64 `json:"attention_factor"`
		Truncate        *bool    `json:"truncate"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("rope_scaling: %w", err)
	}
	kind := obj.RopeType
	if kind == "" {
		kind = obj.Type
	}
	switch kind {
	case "linear":
		if obj.Factor <= 0 {
			return nil, fmt.Errorf("rope_scaling(linear): factor must be >0, got %v", obj.Factor)
		}
		return &ropeScaling{kind: ropeScaleLinear, factor: obj.Factor}, nil
	case "llama3":
		if obj.Factor <= 0 || obj.HighFreqFactor == obj.LowFreqFactor || obj.OrigMaxPosition <= 0 {
			return nil, fmt.Errorf("rope_scaling(llama3): bad params factor=%v low=%v high=%v origMax=%v",
				obj.Factor, obj.LowFreqFactor, obj.HighFreqFactor, obj.OrigMaxPosition)
		}
		return &ropeScaling{
			kind: ropeScaleLlama3, factor: obj.Factor,
			lowFreqFactor: obj.LowFreqFactor, highFreqFactor: obj.HighFreqFactor,
			origMaxPosition: obj.OrigMaxPosition,
		}, nil
	case "yarn":
		return newYarnScaling(obj.Factor, obj.OrigMaxPosition, obj.BetaFast, obj.BetaSlow, obj.AttentionFactor, obj.Truncate)
	default:
		return nil, fmt.Errorf("rope_scaling rope_type=%q unsupported (have: linear, llama3, yarn; longrope/dynamic are a follow-up)", kind)
	}
}

// ropeLayerSpec is one attention type's resolved RoPE: its base (rope_theta)
// and optional scaling (nil = plain RoPE). Used by parseRopeParameters for the
// per-attention-type rope_parameters form.
type ropeLayerSpec struct {
	base    float64
	scaling *ropeScaling
	// partial is this attention type's partial_rotary_factor, read from the SAME
	// inner object (Laguna keys it per layer type: 0.5 on full_attention, 1.0 on
	// sliding_attention, so the two types rotate different widths). 0 ⇒ absent;
	// the caller falls back to the top-level config field. Mellum leaves it 0.
	partial float64
}

// parseRopeParameters reads the per-attention-type rope_parameters object
// (Mellum): {"full_attention": {...}, "sliding_attention": {...}}, where each
// inner object carries rope_theta plus a rope_scaling-style rope_type + params.
// Returns the full (global) and sliding (local) specs. A missing rope_parameters
// is (nil, nil, nil) — the caller falls back to the flat rope_theta/rope_scaling.
//
// A MISSING sliding_attention returns (full, nil, nil) rather than an error: a
// family whose config drops layer_types entirely is all-full-attention and ships
// only the one key (Laguna M.1 does exactly this, where the XS generations ship
// both). Callers that REQUIRE both — mellum — already check for a nil sliding
// themselves and say so in their own error, so relaxing it here cannot loosen them.
func parseRopeParameters(raw json.RawMessage) (full, sliding *ropeLayerSpec, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil, nil
	}
	var obj struct {
		Full    json.RawMessage `json:"full_attention"`
		Sliding json.RawMessage `json:"sliding_attention"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, nil, fmt.Errorf("rope_parameters: %w", err)
	}
	if full, err = parseRopeSpec("full_attention", obj.Full); err != nil {
		return nil, nil, err
	}
	if len(obj.Sliding) == 0 {
		return full, nil, nil
	}
	if sliding, err = parseRopeSpec("sliding_attention", obj.Sliding); err != nil {
		return nil, nil, err
	}
	return full, sliding, nil
}

// parseRopeSpec reads one inner rope_parameters object: rope_theta + a
// rope_type ("default" → plain; "yarn"/"linear"/"llama3" → scaling).
func parseRopeSpec(name string, raw json.RawMessage) (*ropeLayerSpec, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("rope_parameters: missing %s", name)
	}
	var head struct {
		RopeType  string  `json:"rope_type"`
		RopeTheta float64 `json:"rope_theta"`
		Partial   float64 `json:"partial_rotary_factor"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("rope_parameters(%s): %w", name, err)
	}
	if head.RopeTheta <= 0 {
		return nil, fmt.Errorf("rope_parameters(%s): rope_theta must be >0", name)
	}
	spec := &ropeLayerSpec{base: head.RopeTheta, partial: head.Partial}
	switch head.RopeType {
	case "", "default":
		// plain RoPE, no scaling
	default:
		sc, err := parseRopeScaling(raw) // reuses the rope_type dispatch (yarn/linear/llama3)
		if err != nil {
			return nil, fmt.Errorf("rope_parameters(%s): %w", name, err)
		}
		spec.scaling = sc
	}
	return spec, nil
}

// parseRopeFlat reads the qwen3_5_moe flat rope_parameters object
// {rope_type, rope_theta, partial_rotary_factor, …}: one ropeLayerSpec (base +
// optional scaling) plus the partial-rotary fraction (HF defaults it to 0.25
// inside this object, so it is read here rather than from the top-level config).
func parseRopeFlat(raw json.RawMessage) (spec *ropeLayerSpec, partialRotary float64, err error) {
	spec, err = parseRopeSpec("rope_parameters", raw)
	if err != nil {
		return nil, 0, err
	}
	var head struct {
		PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	}
	_ = json.Unmarshal(raw, &head)
	return spec, head.PartialRotaryFactor, nil
}

// newYarnScaling builds a YaRN ropeScaling, mirroring HF
// _compute_yarn_parameters' defaults: beta_fast 32, beta_slow 1, and an
// attention_factor of get_mscale(factor) = 0.1·ln(factor)+1 when not given.
func newYarnScaling(factor, origMax float64, betaFast, betaSlow, attnFactor *float64, truncate *bool) (*ropeScaling, error) {
	if factor <= 0 || origMax <= 0 {
		return nil, fmt.Errorf("rope_scaling(yarn): bad params factor=%v origMax=%v", factor, origMax)
	}
	bf, bs := 32.0, 1.0
	if betaFast != nil && *betaFast != 0 {
		bf = *betaFast
	}
	if betaSlow != nil && *betaSlow != 0 {
		bs = *betaSlow
	}
	ms := 1.0
	if attnFactor != nil {
		ms = *attnFactor
	} else if factor > 1 {
		ms = 0.1*math.Log(factor) + 1.0
	}
	trunc := true // HF's _compute_yarn_parameters default
	if truncate != nil {
		trunc = *truncate
	}
	return &ropeScaling{
		kind: ropeScaleYarn, factor: factor, origMaxPosition: origMax,
		betaFast: bf, betaSlow: bs, mscale: ms, truncate: trunc,
	}, nil
}

// yarnGetMscale is HF's yarn_get_mscale: 1.0 at scale ≤ 1, else 0.1·m·ln(scale)+1.
// DeepSeek's cos/sin attention_factor is the RATIO of this at the two mscale knobs
// (mscale / mscale_all_dim); equal knobs (V2-Lite) ⇒ 1.0.
func yarnGetMscale(scale, m float64) float64 {
	if scale <= 1 {
		return 1.0
	}
	return 0.1*m*math.Log(scale) + 1.0
}

// computeInvFreq builds the per-dimension inverse-frequency table RoPE rotates
// with: invFreq[d] = base^(-2d/rotaryDim) for d in [0, rotaryDim/2), with any
// scaling transform applied. rotaryDim is the number of rotated dims (== head
// dim for full rotary, < head dim for Phi's partial_rotary_factor).
func computeInvFreq(base float64, rotaryDim int, sc *ropeScaling) []float64 {
	half := rotaryDim / 2
	inv := make([]float64, half)
	for d := range half {
		inv[d] = math.Pow(base, -float64(2*d)/float64(rotaryDim))
	}
	if sc == nil {
		return inv
	}
	switch sc.kind {
	case ropeScaleLinear:
		for d := range inv {
			inv[d] /= sc.factor
		}
	case ropeScaleLlama3:
		applyLlama3Scaling(inv, sc)
	case ropeScaleYarn:
		applyYarnScaling(inv, base, rotaryDim, sc)
	}
	return inv
}

// applyYarnScaling reproduces HF's _compute_yarn_parameters inv_freq (NTK-by-
// parts): high-frequency dims extrapolate (kept as-is), low-frequency dims
// interpolate (divided by factor), and a linear ramp over the correction range
// [low, high] blends the two. `inv` enters as the plain extrapolation table
// (base^-2d/dim); the per-dim attention_factor mscale is applied separately at
// rotation time (see ropeMscale). truncate=true (floor/ceil the range), as in
// every config we target. rotaryDim is the full dim (the ramp spans dim/2).
func applyYarnScaling(inv []float64, base float64, rotaryDim int, sc *ropeScaling) {
	dim := float64(rotaryDim)
	corr := func(numRot float64) float64 {
		return (dim * math.Log(sc.origMaxPosition/(numRot*2*math.Pi))) / (2 * math.Log(base))
	}
	// truncate=true (HF default) floors/ceils the range to integer dims; gpt-oss
	// uses truncate=false (the raw fractional range), which matches HF to ~1e-8
	// where truncating is ~2e-4 off.
	low, high := corr(sc.betaFast), corr(sc.betaSlow)
	if sc.truncate {
		low, high = math.Floor(low), math.Ceil(high)
	}
	low = math.Max(low, 0)
	high = math.Min(high, dim-1)
	if low == high {
		high += 0.001 // prevent singularity (matches HF linear_ramp_factor)
	}
	for d := range inv {
		ramp := (float64(d) - low) / (high - low)
		ramp = math.Min(math.Max(ramp, 0), 1)
		extrapFactor := 1 - ramp // inv_freq_extrapolation_factor
		extrap := inv[d]         // plain (extrapolation) frequency
		interp := extrap / sc.factor
		inv[d] = interp*(1-extrapFactor) + extrap*extrapFactor
	}
}

// applyLlama3Scaling reproduces HF's _compute_llama3_parameters: split the
// frequency band by wavelength into a high-frequency part (kept), a
// low-frequency part (divided by factor), and a middle part (smoothly
// interpolated between the two). Wavelength = 2π / inv_freq.
func applyLlama3Scaling(inv []float64, sc *ropeScaling) {
	lowWavelen := sc.origMaxPosition / sc.lowFreqFactor   // long wavelength threshold
	highWavelen := sc.origMaxPosition / sc.highFreqFactor // short wavelength threshold
	for d, f := range inv {
		wavelen := 2 * math.Pi / f
		switch {
		case wavelen < highWavelen:
			// high frequency: unchanged
		case wavelen > lowWavelen:
			inv[d] = f / sc.factor
		default:
			smooth := (sc.origMaxPosition/wavelen - sc.lowFreqFactor) / (sc.highFreqFactor - sc.lowFreqFactor)
			inv[d] = (1-smooth)*(f/sc.factor) + smooth*f
		}
	}
}
