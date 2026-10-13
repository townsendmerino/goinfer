package decoder

import "testing"

// slowTest marks a test that takes ten seconds or more (the 990-test run of 2026-10-13 spent 52% of its CPU time in five of them). `go test -short` skips it, which is
// what the day loop runs (scripts/touched.py); CI, the pre-tag tier and the night queue run without -short, so every one of them still runs before anything ships.
// A skip is not a pass: the day loop is not a gate.
func slowTest(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("slow (10 s or more): runs in CI and the night queue; -short is the day loop")
	}
}
