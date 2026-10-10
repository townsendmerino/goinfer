//go:build gpu

package gpu

import (
	"os"
	"testing"
)

// requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an
// explicit env opt-in: the package-local twin of decoder/heavytest_test.go's helper, same
// GOINFER_HEAVY_TESTS key, so `GOINFER_HEAVY_TESTS=1 go test ./...` runs every backend's heavy
// tests uniformly. A model happening to be on disk is not a request to run a multi-GB test
// (deciding by path existence alone once fired the whole zoo opportunistically and thrashed the
// box); the per-test os.Stat skip stays as a second guard, so opting in on a bare box is still
// harmless.
//
// It takes testing.TB so the measurement harnesses can be Benchmarks: they report numbers and
// assert nothing, and as Test* a green suite would include a PASS that proves nothing about the
// win they map.
func requireHeavyModel(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}
