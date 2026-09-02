package decoder

import (
	"math"
	"strings"
	"testing"
)

// validArch returns an Architecture that PASSES validateResolved, so each sub-test below can
// break exactly one thing. Without this they would pass for whichever reason fired first.
//
// It does NOT call finalizeRoPE — the caller does, mirroring resolveArchitecture's real order
// (adapter → finalizeRoPE → validateResolved). That matters: finalizeRoPE returns early on
// base <= 0 WITHOUT clearing an existing table, so a fixture that finalized at a good base and
// was then mutated keeps its old table and the RoPE guard cannot fire. Harmless in production,
// where finalizeRoPE runs once on a fresh descriptor — but it made the first draft of this test
// report a passing guard that had not run.
func validArch() *Architecture {
	return &Architecture{
		Name: "probe", Norm: NormRMS, NormEps: 1e-5, AttnScale: 0.25,
		HiddenDim: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16, VocabSize: 32,
		RoPEGlobalBase: 10000,
	}
}

// finalized is validArch() put through the step resolveArchitecture performs before validating.
func finalized(a *Architecture) *Architecture { a.finalizeRoPE(); return a }

// M-06: validateResolved checked two fields and waved through three more classes of zero that
// look legal. Each of these loads CLEAN today and fails later — in the forward, or not at all.
func TestValidateResolved_theZerosThatLookLegal(t *testing.T) {
	// The baseline must pass, or every assertion below is vacuous.
	if err := finalized(validArch()).validateResolved(); err != nil {
		t.Fatalf("the valid fixture is rejected: %v — every sub-test below would pass for the "+
			"wrong reason", err)
	}

	for name, tc := range map[string]struct {
		mutate func(*Architecture)
		names  string
	}{
		// NaN is the sharp one: `AttnScale <= 0` is FALSE for NaN, so the old guard accepted it.
		// gemma3 with a negative query_pre_attn_scalar produces exactly this.
		"NaN AttnScale": {func(a *Architecture) { a.AttnScale = math.NaN() }, "AttnScale"},
		// +Inf arrives from num_attention_heads: 0 → Pow(0, -0.5).
		"Inf AttnScale":  {func(a *Architecture) { a.AttnScale = math.Inf(1) }, "AttnScale"},
		"zero HiddenDim": {func(a *Architecture) { a.HiddenDim = 0 }, "HiddenDim"},
		"zero NumLayers": {func(a *Architecture) { a.NumLayers = 0 }, "NumLayers"},
		"zero NumHeads":  {func(a *Architecture) { a.NumHeads = 0 }, "NumHeads"},
		"zero HeadDim":   {func(a *Architecture) { a.HeadDim = 0 }, "HeadDim"},
		"zero VocabSize": {func(a *Architecture) { a.VocabSize = 0 }, "VocabSize"},
		// The one that panics rather than misbehaves: `group := nH/nKV` is an integer divide.
		"zero NumKVHeads": {func(a *Architecture) { a.NumKVHeads = 0 }, "NumKVHeads"},
		// Not a zero, but the same divisor: 5 heads over 2 kv-heads has no grouping.
		"NumHeads not a multiple of NumKVHeads": {
			func(a *Architecture) { a.NumHeads, a.NumKVHeads = 5, 2 }, "multiple"},
	} {
		t.Run(name, func(t *testing.T) {
			a := finalized(validArch())
			tc.mutate(a)
			err := a.validateResolved()
			if err == nil {
				t.Fatalf("accepted; the guard did not fire")
			}
			if !strings.Contains(err.Error(), tc.names) {
				t.Errorf("error %q does not name %q", err, tc.names)
			}
		})
	}
}

// The RoPE half, which is the quietest of the three: finalizeRoPE treats base <= 0 as "no
// tables" and applyRoPE is a silent no-op on an empty table, so the model generates fluent,
// POSITION-BLIND text. gpt-oss and llama4 read only the flat rope_theta, and transformers
// >=5.10 nests it under rope_parameters — a loud error for llama/mistral/qwen3, silence here.
func TestValidateResolved_positionInformationMustExist(t *testing.T) {
	a := validArch()
	a.RoPEGlobalBase = 0
	a.finalizeRoPE() // tables stay nil — never populated, as on a real load
	err := a.validateResolved()
	if err == nil {
		t.Fatal("accepted an arch with no RoPE table and no learned positions — it would load " +
			"clean and generate position-blind text (M-06)")
	}
	if !strings.Contains(err.Error(), "position") {
		t.Errorf("error %q does not say what is missing", err)
	}

	// THE THREE LEGITIMATE EXEMPTIONS must still pass, or this guard breaks real families.
	// Named explicitly in the guard rather than inferred, so a family that simply FORGOT to
	// read rope_theta cannot pass by resembling one of them.
	t.Run("GPT-2 learned positions", func(t *testing.T) {
		a := validArch()
		a.RoPEGlobalBase, a.LearnedPosEmbed = 0, true
		a.finalizeRoPE()
		if err := a.validateResolved(); err != nil {
			t.Errorf("learned-position arch rejected: %v", err)
		}
	})
	t.Run("Nemotron-H NoPE layers", func(t *testing.T) {
		a := validArch()
		a.RoPEGlobalBase, a.nemotron = 0, &nemotronParams{}
		a.finalizeRoPE()
		if err := a.validateResolved(); err != nil {
			t.Errorf("nemotron arch rejected: %v", err)
		}
	})
	t.Run("MLA decoupled rope", func(t *testing.T) {
		a := validArch()
		a.RoPEGlobalBase, a.mla = 0, &mlaParams{}
		a.finalizeRoPE()
		if err := a.validateResolved(); err != nil {
			t.Errorf("MLA arch rejected: %v", err)
		}
	})
}
