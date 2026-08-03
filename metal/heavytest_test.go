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
// The bug this closes: metal's real-model tests decided whether to run by PATH EXISTENCE alone
// (os.ExpandEnv("$HOME/models/...") → skip if absent, else Load()). On a box with the model zoo
// present, `go test ./metal/` fired them all opportunistically — dozens of GB-scale loads that
// blew the 10-minute timeout (and one, TestGemmaBisect_PerLayer, panics on the qwen control's
// hidden dim). The asset happening to be on disk is not a request to run a multi-GB test. The
// per-test os.Stat skip stays as a second guard, so opting in on a bare box is still harmless.
func requireHeavyModel(t *testing.T) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}
