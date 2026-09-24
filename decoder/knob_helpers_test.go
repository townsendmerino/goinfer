package decoder

import (
	"os"
	"testing"
)

// setKnob is the in-package form of SetKnobEnvForTest (knobs_drift_testhooks.go), usable from untagged tests:
// the replacement for t.Setenv(name, value) in a test that has already loaded m. It sets the environment and
// pins m's snapshot (knobs are read once per model, at Load — docs/tasks/task-env-config-2026-09.md), so the
// loaded model follows exactly as it did when these knobs were read per call. Undone at the end of the test.
func setKnob(t testing.TB, m *Model, name, value string) {
	t.Helper()
	t.Setenv(name, value)
	t.Cleanup(m.knobs.pin(name, value, true))
}

// unsetKnob is setKnob for "not set at all".
func unsetKnob(t testing.TB, m *Model, name string) {
	t.Helper()
	t.Setenv(name, "")
	os.Unsetenv(name)
	t.Cleanup(m.knobs.pin(name, "", false))
}
