//go:build cuda

package cuda

import (
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

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
