//go:build darwin

package metal

import (
	"os"
	"testing"
)

// requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an explicit
// env opt-in — the package-local twin of decoder/heavytest_test.go's helper, same GOINFER_HEAVY_TESTS
// key, so `GOINFER_HEAVY_TESTS=1 go test ./...` runs every backend's heavy tests uniformly.
//
// Without it, a box with the model zoo present fires every real-model test opportunistically on `go test ./metal/`: dozens
// of GB-scale loads that blow the 10-minute timeout. The asset happening to be on disk is not a request to run a multi-GB
// test. The per-test os.Stat skip stays as a second guard, so opting in on a bare box is still harmless.
func requireHeavyModel(t *testing.T) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}
