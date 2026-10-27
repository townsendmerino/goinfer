//go:build cuda

package kernels

import (
	"context"
	"math"
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

// f16rne rounds a float32 through f16 with round-to-nearest-even, as the device's __float2half_rn does. It is not
// cuda/kernels.go's f32tof16, which rounds half up on purpose for the int4 group scales.
func f16rne(x float32) float32 {
	b := math.Float32bits(x)
	sign := b & 0x80000000
	exp := int32((b>>23)&0xFF) - 127
	if exp > 15 {
		return math.Float32frombits(sign | 0x7F800000)
	}
	if exp < -24 {
		return math.Float32frombits(sign)
	}
	shift := uint(13)
	if exp < -14 {
		shift = uint(13 + (-14 - exp))
	}
	m := b & 0x7FFFFF
	if exp < -14 {
		m |= 0x800000
	}
	half := m >> shift
	rem := m & ((1 << shift) - 1)
	mid := uint32(1) << (shift - 1)
	if rem > mid || (rem == mid && half&1 == 1) {
		half++
	}
	if exp < -14 {
		v := float32(half) * float32(math.Pow(2, -24))
		if sign != 0 {
			v = -v
		}
		return v
	}
	out := sign | uint32(exp+127)<<23 | (half << shift)
	if half<<shift >= 0x800000 {
		out = sign | uint32(exp+1+127)<<23
	}
	return math.Float32frombits(out)
}

// kvIdentityIDs is the block table that maps block i to block i, for a contiguous cache of positions rows.
func kvIdentityIDs(positions int) []int32 {
	ids := make([]int32, max(1, (positions+kvBlockRows-1)/kvBlockRows))
	for i := range ids {
		ids[i] = int32(i)
	}
	return ids
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// f16BitsToF32 widens a binary16 pattern (exact).
func f16BitsToF32(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	e, m := int(h>>10)&0x1F, float32(h&0x3FF)
	switch e {
	case 0:
		return sign * m * float32(math.Ldexp(1, -24))
	case 0x1F:
		if m == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	return sign * (1 + m/1024) * float32(math.Ldexp(1, e-15))
}
