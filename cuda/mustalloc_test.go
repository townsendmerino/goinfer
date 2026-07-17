//go:build cuda

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// requireDeviceAndFixture skips a GPU parity test when it cannot run, so it never Fatals in an
// environment that is not a bug. It exists because the CI `cuda` job runs `-tags cuda` with NO
// device and the tiny fixtures are NOT committed (blanket *.safetensors gitignore), and
// crucially decoder.Load(Backend:"cuda") SUCCEEDS with no GPU — it declines residency
// gracefully — so a parity test that just loads and asserts would blow up on the missing
// fixture or the nil resident instead of skipping. The device check must come FIRST: it is the
// one that catches CI (no GPU), before the load can fail on an absent fixture.
func requireDeviceAndFixture(t *testing.T, dir string) {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	// A GPU box that lacks the (uncommitted) fixture should also skip, not Fatal. config.json is
	// the one file present whenever the fixture is; the weights sit beside it.
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("no fixture at %s (uncommitted; regenerate with the matching scripts/pin_*.py)", dir)
	}
}

// mustAlloc allocates device memory and FAILS THE TEST on error, instead of the
// `buf, _ := gc.Alloc[T](...)` that used to be the norm here.
//
// WHY THIS MATTERS MORE THAN IT LOOKS. A dropped alloc error is how an out-of-memory
// condition disguises itself as a numerics bug. gc.Alloc returns (nil, err) when the card
// is full; drop the err and the nil buffer reads back as ZEROS, so the assertion that
// fires is "cosine 0.000000 — layout/unpack mismatch". That sentence sent two people
// hunting a kernel bug for a day while the real cause was a VRAM leak saturating an 8 GB
// card mid-suite (d8e81cb). The kernels were never wrong; the memory was gone, and every
// test lied about why.
//
// The rule this encodes: a RESOURCE failure must say it is a resource failure. Tests may
// legitimately skip when a GPU is absent or too small — but they must never silently
// compute on nothing and report the result as a correctness verdict.
func mustAlloc[T gc.Supported](t *testing.T, cx *gc.Context, n int) *gc.Buffer[T] {
	t.Helper()
	b, err := gc.Alloc[T](cx, n)
	if err != nil {
		t.Fatalf("gc.Alloc[%T](%d) failed: %v — this is a RESOURCE failure (GPU memory exhausted?), "+
			"NOT a kernel/parity bug. Check for stray processes holding the card and re-run.",
			*new(T), n, err)
	}
	if b == nil {
		t.Fatalf("gc.Alloc[%T](%d) returned a nil buffer with no error — reads would silently "+
			"return zeros and surface as a bogus cosine failure", *new(T), n)
	}
	return b
}

// mustStream creates a stream or fails loudly. A dropped error here surfaces later as
// "cuda: nil stream" from an unrelated launch, which reads as a kernel problem.
func mustStream(t *testing.T, cx *gc.Context) *gc.Stream {
	t.Helper()
	s, err := cx.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v — a RESOURCE/context failure, not a kernel bug", err)
	}
	return s
}
