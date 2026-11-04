//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// These helpers live behind goinfer_testhooks because mustResident calls Model.ResidentForwardForTest, which exists only
// under that tag; in heavytest_test.go (tag cuda alone) they broke the kernel-level suite's build, which gate gpu's group
// 2a compiles without testhooks.

// fourBLoadOpts is the load a heavy test of the default 4B target (~/models/qwen3-4b, int4) uses. It pins the resident
// context at 8192: unpinned, the 4B's KV at ~16k positions plus the margin fills the 8 GB card, so the resident build
// declines to the CPU path ("default resident context 16029 positions does not fit"), or builds and leaves AttachDrafter
// no room for the drafter's weights (CUDA_ERROR_OUT_OF_MEMORY), and the first panic takes the rest of the package's run
// with it.
func fourBLoadOpts() decoder.Options {
	return decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 8192}
}

// mustResident returns mc's CUDA resident or fails THIS test. The bare mc.ResidentForwardForTest().(*cudaResident) panics
// when the build declined (nil interface), and a panic kills the whole `go test` process, taking every later test with it.
func mustResident(t testing.TB, mc *decoder.Model) *cudaResident {
	t.Helper()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("the model did not build a CUDA resident (got %T): the build declined; see the [resident] line above", mc.ResidentForwardForTest())
	}
	return r
}

// cosineF32 is the cosine similarity of a and b in float64, 0 when either is all zeros.
func cosineF32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
