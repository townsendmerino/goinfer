package decoder

import (
	"os"
	"testing"
)

// requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an explicit env opt-in.
//
// Path existence is not a gate: the asset happening to exist is not a request to run a multi-GB test. A test that decided by
// os.Stat alone fires opportunistically on a box with the model zoo present, reloading a checkpoint per test and thrashing
// swap.
//
// Policy (same shape as GOINFER_MOE_GIW / GOINFER_SSM_RESIDENT): a test that loads a multi-GB model must OPT IN via
// GOINFER_HEAVY_TESTS. Default `go test ./decoder/` skips them regardless of what is on disk, so the package is runnable
// everywhere; set GOINFER_HEAVY_TESTS=1 when you mean to run them. The per-test os.Stat skip stays as a second guard, so
// opting in on a bare box is still harmless.
func requireHeavyModel(t *testing.T) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}
