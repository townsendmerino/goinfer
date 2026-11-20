//go:build darwin

package metal

import (
	"github.com/townsendmerino/goinfer/decoder"
)

// The window group's bounded pool. A window group keeps only each sequence's
// recent positions (kvRetain), so its pool can hold decoder.MetalKVWindowBlocks a slot instead of the context's blocks, as long as no
// pass writes more rows into it than that holds. windowPoolBlocks decides, from the model alone, whether every pass that writes the
// group is bounded; the build sizes the pool by it and the plan (decoder.Model.MetalWindowBlocks) prices it by the same function,
// registered here at init. A condition that does not hold leaves the pool the context's size, as before.
func init() { decoder.RegisterMetalWindowPool(windowPoolBlocks) }

// windowCondition is one reason a build's passes are not all bounded: the first one that does not hold keeps the pool full-size. They
// are exactly the ways PrefillBlockRows reports 0 (an unsplit prompt) or a route writes without blocking, kept to what the model alone
// answers; a family a route might take that the model cannot rule out is excluded.
type windowCondition struct {
	name  string
	holds func(m *decoder.Model) bool
}

var windowConditions = []windowCondition{
	{"kv-not-f32", func(m *decoder.Model) bool { return m.KVDeliveredFor("metal") != "f32" }},
	{"no-moe", func(m *decoder.Model) bool {
		bv := m.BackendView()
		_, moe := bv.MoE()
		return !moe && !bv.HasGemma4MoE()
	}},
	{"no-mla", func(m *decoder.Model) bool {
		_, ok := m.BackendView().MLA()
		return !ok
	}},
	{"no-recurrent", func(m *decoder.Model) bool {
		_, nL, _, _, _, _, _ := m.Dims()
		for l := range nL {
			if len(m.StateSpecs(l)) > 0 {
				return false
			}
		}
		return true
	}},
	{"no-attn-sink", func(m *decoder.Model) bool { return !m.BackendView().HasAttnSinks() }},
	{"no-learned-pos", func(m *decoder.Model) bool { return !m.BackendView().LearnedPos() }},
	{"rms-norm", func(m *decoder.Model) bool { return !m.BackendView().LayerNorm() }},
	{"uniform-geometry", func(m *decoder.Model) bool { return !m.HasPerLayerGeometry() }},
	{"no-e-model", func(m *decoder.Model) bool {
		return m.BackendView().Gemma4PLEDim() == 0 && len(m.KVGeometry().Shared) == 0
	}},
	{"no-drafter-kv", func(m *decoder.Model) bool { return m.ExtraResidentKVPerPosition() == 0 }},
	{"prefill-features", func(m *decoder.Model) bool { return len(m.MissingResidentFeatures(prefillFeatures)) == 0 }},
	{"prefill-steps", func(m *decoder.Model) bool { return prefillStepsDecline(m.Program()) == "" }},
	{"not-q4k", func(m *decoder.Model) bool { return m.Quant() != "q4k" }},
	{"fast-prefill", func(m *decoder.Model) bool {
		fast, _ := m.Knob("GOINFER_METAL_FAST_PREFILL")
		batched, _ := m.Knob("GOINFER_METAL_BATCHED_PREFILL")
		return !m.ExactPrefill() && metalFastPrefillEnabled(fast, batched)
	}},
	{"aligned", func(m *decoder.Model) bool {
		h, _, nH, _, hd, inter, _ := m.Dims()
		return h%32 == 0 && (nH*hd)%32 == 0 && inter%32 == 0
	}},
	{"head-dim", func(m *decoder.Model) bool { _, _, _, _, hd, _, _ := m.Dims(); return hd <= 256 }},
}

// kvWindowPoolOffForTest keeps every window group the context's size, as KV5-7c-i did.
var kvWindowPoolOffForTest bool

// kvWindowPoolForceForTest bounds every window group whatever the conditions say, for a test of the bounded pool on a fixture whose
// prefill is not blocked (its passes are one row each).
var kvWindowPoolForceForTest bool

// kvWindowIgnoreForTest names a condition to treat as holding: the plant TestKVWindow_planAndBuildAgree must catch for each condition
// that guards a route that is not bounded.
var kvWindowIgnoreForTest string

// windowConditionsFailing names the conditions that keep m's window pool the context's size.
func windowConditionsFailing(m *decoder.Model) []string {
	var out []string
	for _, c := range windowConditions {
		if !c.holds(m) {
			out = append(out, c.name)
		}
	}
	return out
}

// windowPoolBlocks is the blocks a slot's window group holds for m: decoder.MetalKVWindowBlocks of its window when m has a window group
// beside a full one and every condition holds, else 0 (the context's blocks).
func windowPoolBlocks(m *decoder.Model) int {
	if kvWindowPoolOffForTest {
		return 0
	}
	g := m.KVGeometry()
	span, _ := kvWindowSpan(g)
	if span == 0 || len(kvWindowLayers(g)) == 0 {
		return 0
	}
	if !kvWindowPoolForceForTest {
		for _, c := range windowConditions {
			if c.name != kvWindowIgnoreForTest && !c.holds(m) {
				return 0
			}
		}
	}
	return decoder.MetalKVWindowBlocks(span)
}
