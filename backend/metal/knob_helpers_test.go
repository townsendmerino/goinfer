//go:build darwin

package metal

import (
	"os"
	"testing"
)

// setResidentKnob overrides one per-call operator knob on an already-built resident, for the rest of the test.
// Knobs are read from the model's Load-time snapshot (docs/tasks/task-env-config-2026-09.md, phase 4), so a
// t.Setenv after the build no longer reaches the resident — an A/B written that way would silently compare a path
// with itself. This works untagged (unlike decoder.SetKnobEnvForTest), and only for knobs read per call
// (r.knobValue); a knob read at build needs its value before Load (t.Setenv, or Options.Knobs).
func setResidentKnob(t testing.TB, r *resident, name, value string) {
	t.Helper()
	prev := r.knob
	r.knob = func(n string) (string, bool) {
		if n == name {
			return value, true
		}
		if prev == nil {
			return os.LookupEnv(n)
		}
		return prev(n)
	}
	t.Cleanup(func() { r.knob = prev })
}
