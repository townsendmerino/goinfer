//go:build !goinfer_testhooks

package decoder

// knobDrift is the test-build tripwire (knobs_drift_testhooks.go); production builds read the snapshot only.
func knobDrift(*knobSet, string) {}
