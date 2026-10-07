//go:build cuda

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an explicit
// env opt-in — the package-local twin of decoder/heavytest_test.go's helper, same GOINFER_HEAVY_TESTS
// key, so `GOINFER_HEAVY_TESTS=1 go test -tags cuda ./...` runs every backend's heavy tests
// uniformly.
//
// The bug this closes: the cuda real-model tests decided whether to run by PATH EXISTENCE alone
// (os.ExpandEnv("$HOME/models/...") → skip if absent, else Load()). On a box with the model zoo
// present, the suite fired them all opportunistically instead of on request. The asset happening to
// be on disk is not a request to run a multi-GB test. The per-test os.Stat skip stays as a second
// guard, so opting in on a bare box is still harmless.
func requireHeavyModel(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}

// fourBLoadOpts is the load a heavy test of the default 4B target (~/models/qwen3-4b, int4) uses. It pins the resident
// context at 8192, the default before R19 (2026-10-01) raised the candidate to 16384: unpinned, the 4B's KV at ~16k
// positions plus the margin fills the 8 GB card, the resident build declines to the CPU path ("default resident context 16029
// positions does not fit"), or builds and leaves AttachDrafter no room for the drafter's weights (CUDA_ERROR_OUT_OF_MEMORY). Found by the
// 2026-10-06 night gate: the gate-gpu heavy tier lost its drafter, block-spec and attention tests to it, and the first one's panic took the rest of the
// package's run with it.
func fourBLoadOpts() decoder.Options {
	return decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 8192}
}

// mustResident returns mc's CUDA resident or fails THIS test. The bare mc.ResidentForwardForTest().(*cudaResident)
// panics when the build declined (nil interface), and a panic kills the whole `go test` process: the 2026-10-06
// night gate lost every test after the first decline to it, which the verdict then listed as 20 FAILs.
func mustResident(t testing.TB, mc *decoder.Model) *cudaResident {
	t.Helper()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("the model did not build a CUDA resident (got %T): the build declined; see the [resident] line above", mc.ResidentForwardForTest())
	}
	return r
}
