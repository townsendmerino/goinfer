//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"os"
)

// knobDrift is the phase-2a tripwire (docs/tasks/task-env-config-2026-09.md). A knob is read once, at Load;
// a test that sets its environment variable AFTER loading a model and expects the loaded model to follow
// would, silently, now compare a path with itself — an identity A/B passing for the wrong reason. In test
// builds every snapshot read therefore checks the live environment and panics when it has moved for a knob
// that was not set through Options.Knobs or SetKnobForTest. Production builds compile knobDrift to nothing.
func knobDrift(k *knobSet, name string) {
	if k.pinned[name] {
		return
	}
	live, liveSet := os.LookupEnv(name)
	if live != k.val[name] || liveSet != k.set[name] {
		panic(fmt.Sprintf("decoder: %s changed after this model was loaded (snapshot %q set=%v, environment now %q set=%v). "+
			"Knobs are read once per model, at Load (docs/tasks/task-env-config-2026-09.md). A test that A/Bs a loaded "+
			"model must use decoder.SetKnobForTest(m, %q, value) instead of t.Setenv — or set the variable before Load.",
			name, k.val[name], k.set[name], live, liveSet, name))
	}
}

// SetKnobForTest sets one of a loaded model's knobs (knobs.go's knobNames) — the per-model replacement for
// t.Setenv-ing the variable after Load. It returns a func that restores the previous value.
func SetKnobForTest(m *Model, name, value string) (restore func()) {
	if m.knobs == nil {
		panic("decoder.SetKnobForTest: model has no knob snapshot (not built by Load)")
	}
	return m.knobs.pin(name, value, true)
}

// UnsetKnobForTest makes a loaded model behave as if name were not set at all.
func UnsetKnobForTest(m *Model, name string) (restore func()) {
	if m.knobs == nil {
		panic("decoder.UnsetKnobForTest: model has no knob snapshot (not built by Load)")
	}
	return m.knobs.pin(name, "", false)
}

// knobTB is the part of *testing.T SetKnobEnvForTest needs (decoder does not import testing in hooks).
type knobTB interface {
	Helper()
	Setenv(key, value string)
	Cleanup(func())
}

// SetKnobEnvForTest is the drop-in replacement for t.Setenv(name, value) in a test that has ALREADY loaded
// m: it sets the environment (for any backend still reading it — CUDA/Metal until their own phases) AND
// pins m's snapshot, so the decoder follows and the drift tripwire stays quiet. Both are undone at the end
// of the test.
func SetKnobEnvForTest(t knobTB, m *Model, name, value string) {
	t.Helper()
	t.Setenv(name, value)
	t.Cleanup(SetKnobForTest(m, name, value))
}

// UnsetKnobEnvForTest is SetKnobEnvForTest for "not set at all" (the test's os.Unsetenv).
func UnsetKnobEnvForTest(t knobTB, m *Model, name string) {
	t.Helper()
	t.Setenv(name, "")
	os.Unsetenv(name)
	t.Cleanup(UnsetKnobForTest(m, name))
}
