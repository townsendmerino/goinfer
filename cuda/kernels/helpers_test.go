//go:build cuda

package kernels

import (
	"context"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// mustAlloc allocates device memory and fails the test on error. A dropped alloc error reads back as zeros, so an
// out-of-memory card would otherwise surface as a numerics failure (docs/code-notes/cuda.md#mustAlloc).
func mustAlloc[T gc.Supported](t testing.TB, cx *gc.Context, n int) *gc.Buffer[T] {
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

// mustStream creates a stream or fails the test. A dropped error surfaces later as "cuda: nil stream" from an
// unrelated launch.
func mustStream(t testing.TB, cx *gc.Context) *gc.Stream {
	t.Helper()
	s, err := cx.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v — a RESOURCE/context failure, not a kernel bug", err)
	}
	return s
}

// kvBlockRows is the rows of one KV block, as the kernels' KV_ROW in cuda/kvpage.cuh hard-codes it.
const kvBlockRows = 64

// gcKVIdentityTab is the block table that maps block i to block i, for a contiguous cache of positions rows
// (KV_ROW(t, k) == k for every k < positions), as a device buffer. A kernel test that hands a KV kernel a plain
// [pos][kvDim] buffer passes it as the table. The context is synchronised after the copy: a pageable copy returns
// once staged, and a launch on another stream could read the table before it lands.
func gcKVIdentityTab(t testing.TB, cx *gc.Context, positions int) *gc.Buffer[int32] {
	t.Helper()
	ids := make([]int32, max(1, (positions+kvBlockRows-1)/kvBlockRows))
	for i := range ids {
		ids[i] = int32(i)
	}
	b, err := gc.Alloc[int32](cx, len(ids))
	if err != nil {
		t.Fatalf("identity KV table: %v", err)
	}
	if err := b.CopyFrom(context.Background(), ids); err != nil {
		t.Fatalf("identity KV table: %v", err)
	}
	if err := cx.Synchronize(context.Background()); err != nil {
		t.Fatalf("identity KV table: %v", err)
	}
	return b
}
