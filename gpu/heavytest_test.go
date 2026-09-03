//go:build gpu

package gpu

import (
	"os"
	"testing"
)

// requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an explicit
// env opt-in — the package-local twin of decoder/heavytest_test.go's helper, same GOINFER_HEAVY_TESTS
// key, so `GOINFER_HEAVY_TESTS=1 go test ./...` runs every backend's heavy tests uniformly.
//
// The bug this closes: the gpu (WebGPU) real-model tests decided whether to run by PATH EXISTENCE
// alone (os.ExpandEnv("$HOME/models/...") → skip if absent, else Load()). On the Ryzen box with the
// model zoo present, `go test ./gpu/` fired the whole zoo opportunistically — climbing to ~75 GB RSS
// and thrashing, never finishing. The asset happening to be on disk is not a request to run a
// multi-GB test. The per-test os.Stat skip stays as a second guard, so opting in on a bare box is
// still harmless.
// requireHeavyModel takes testing.TB so the measurement harnesses can be Benchmarks (G-10):
// they report numbers and assert nothing, and shipping them as Test* made a green suite include
// a PASS that proves nothing about the win they map.
func requireHeavyModel(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}
