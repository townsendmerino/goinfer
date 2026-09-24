package decoder

import (
	"os"
	"testing"
)

// The phase-2a contract (docs/tasks/task-env-config-2026-09.md): a model's knobs are read ONCE, at Load,
// Options.Knobs overrides the environment for that model alone, and two models in one process can differ.
func TestKnobs_snapshotAtLoad_overridePerModel(t *testing.T) {
	t.Setenv(knobFusedAttention, "0")
	t.Setenv(knobOptFwdMaxTemp, "1.5")
	env, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()
	over, err := Load(tinyFixture(t), Options{Backend: "cpu", Knobs: map[string]string{
		knobFusedAttention:   "1",
		"GOINFER_NOT_A_KNOB": "x", // ignored: knobs.go is the list
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer over.Close()

	if env.knobs.fusedAttention() {
		t.Error("GOINFER_FUSED_ATTENTION=0 in the environment at Load: the model must snapshot it off")
	}
	if !over.knobs.fusedAttention() {
		t.Error("Options.Knobs set it to 1: the override must win over the environment, for this model")
	}
	if got := over.knobs.optFwdTempCap(); got != 1.5 {
		t.Errorf("a knob Options.Knobs does not name must still come from the environment: cap %g, want 1.5", got)
	}
	if env.w.arch.knobs != env.knobs {
		t.Error("the Architecture must share its model's snapshot — the deep forward helpers read it there")
	}
	if _, ok := over.knobs.val["GOINFER_NOT_A_KNOB"]; ok {
		t.Error("an unknown Options.Knobs name must be ignored, not stored")
	}

	// Pinning one model does not touch the other.
	setKnob(t, env, knobFusedAttention, "1")
	unsetKnob(t, over, knobFusedAttention)
	if !env.knobs.fusedAttention() || !over.knobs.fusedAttention() {
		t.Error("setKnob/unsetKnob: both models should now read fused on (unset = default on)")
	}
	if v, ok := over.knobs.lookup(knobFusedAttention); ok || v != "" {
		t.Errorf("unsetKnob must read as absent, got %q set=%v", v, ok)
	}
}

// A hand-built structure with no snapshot (nil *knobSet) reads the live environment, exactly as before.
func TestKnobs_nilSnapshotReadsEnvironment(t *testing.T) {
	var k *knobSet
	t.Setenv(knobAttnGrouped, "0")
	if k.attnGrouped() {
		t.Error("nil snapshot: GOINFER_ATTN_GROUPED=0 must read off")
	}
	os.Unsetenv(knobAttnGrouped)
	if !k.attnGrouped() {
		t.Error("nil snapshot: unset must read as the default (on)")
	}
}

func TestKnobs_pinUnknownPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("pinning a name that is not a knob must panic, not silently no-op")
		}
	}()
	snapshotKnobs(nil).pin("GOINFER_NOT_A_KNOB", "1", true)
}
