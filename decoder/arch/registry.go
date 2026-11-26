package arch

import (
	"fmt"
	"math"
)

// ValidateResolved catches descriptor fields an adapter left at their zero value when zero is not a legal setting. Every adapter builds an
// Architecture by hand from a struct literal, so a field simply omitted is a compile-clean, load-clean, silently wrong model. Checked here
// rather than in each adapter to cover the families nobody has written yet.
//
//   - AttnScale 0 makes every q·k score 0, so softmax returns a UNIFORM average over the context. Invisible at one token (softmax of a
//     single element is 1.0 whatever the scale) and in any greedy smoke test that only reads argmax. An arch that genuinely wants no
//     attention scaling sets 1.0.
//   - NormEps 0 divides by rsqrt(variance) with no floor: not merely imprecise, it rescales the norm output on a small variance.
func (a *Architecture) ValidateResolved() error {
	switch {
	// !(x > 0) rather than x <= 0: NaN fails EVERY comparison, so `a.AttnScale <= 0` is false for NaN and would wave it through. gemma3 with a
	// negative query_pre_attn_scalar produces exactly that, and +Inf arrives from num_attention_heads: 0 → Pow(0, -0.5). Both give a forward
	// that runs and returns garbage rather than one that refuses.
	case !(a.AttnScale > 0) || math.IsInf(a.AttnScale, 0):
		return fmt.Errorf("decoder(%s): AttnScale=%v must be finite and >0 (adapter omitted it, or a config value made it NaN/Inf; 1/sqrt(head_dim) is the usual value, 1.0 means deliberately unscaled)", a.Name, a.AttnScale)
	case a.Norm == NormRMS && !(a.NormEps > 0):
		return fmt.Errorf("decoder(%s): NormEps=%v must be >0 (adapter omitted it, or read the wrong config key)", a.Name, a.NormEps)
	}

	// Position information must come from somewhere. finalizeRoPE treats RoPEGlobalBase <= 0 as "no tables" and rope.Apply is a silent no-op on
	// an empty table, so an adapter that never reads rope_theta loads clean and generates fluent, position-blind text (and drops YaRN with it).
	// transformers >= 5.10 nests rope_theta under rope_parameters, so an adapter that reads only the flat key fails silently.
	//
	// The legitimate ways to have no global RoPE table are named explicitly rather than inferred, so a new family that forgot cannot look like
	// one of them: GPT-2 has learned positions, Nemotron-H encodes NoPE layers as base 0, MLA carries its own decoupled rope dims, and Olmo
	// Hybrid's released checkpoint has none at all (NoPositionEncoding; its rope_parameters is {"rope_theta": null}).
	if !a.LearnedPosEmbed && !a.NoPositionEncoding && a.Nemotron == nil && a.MLA == nil && len(a.RoPEInvFreqGlobal) == 0 {
		return fmt.Errorf("decoder(%s): no position information — RoPEGlobalBase=%v yields no "+
			"inv-freq table, and the arch is not learned-position, Nemotron-H or MLA. The adapter "+
			"most likely did not read rope_theta (transformers >=5.10 nests it under "+
			"rope_parameters); a model loaded this way generates fluent but position-blind text",
			a.Name, a.RoPEGlobalBase)
	}

	// Zero dims. Every one of these is a divisor, a slice length or both somewhere in the forward; num_key_value_heads: 0 is the sharpest:
	// `group := nH/nKV` is an integer divide by zero, a panic in the decode goroutine rather than a load error.
	for _, d := range []struct {
		name string
		v    int
	}{
		{"HiddenDim", a.HiddenDim}, {"NumLayers", a.NumLayers}, {"NumHeads", a.NumHeads},
		{"NumKVHeads", a.NumKVHeads}, {"HeadDim", a.HeadDim}, {"VocabSize", a.VocabSize},
	} {
		if d.v <= 0 {
			return fmt.Errorf("decoder(%s): %s=%d must be >0", a.Name, d.name, d.v)
		}
	}
	if a.NumHeads%a.NumKVHeads != 0 {
		return fmt.Errorf("decoder(%s): NumHeads=%d is not a multiple of NumKVHeads=%d; "+
			"grouped-query attention divides one into the other", a.Name, a.NumHeads, a.NumKVHeads)
	}
	return nil
}
