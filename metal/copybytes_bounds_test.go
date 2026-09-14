//go:build darwin

package metal

import (
	"testing"

	"github.com/townsendmerino/aikit/gpu"
)

// TestCopyBytesToU32Buf_oversizedSrcPanics gates N-33 (audit-metal-2026-09-12.md):
// copyBytesToU32Buf used to unsafe.Slice-reinterpret dst and Go-copy src into it, which silently
// truncates an oversized src with no error — the exact gap gpu.Upload's own bounds check exists to
// close. Confirmed red without the fix: the old implementation returned normally on an oversized
// src (truncating it into dst) instead of panicking.
func TestCopyBytesToU32Buf_oversizedSrcPanics(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const nWords = 4
	dst := gpu.NewBufferLenOf[uint32](d, nWords)

	oversized := make([]byte, nWords*4+8) // 8 bytes past the buffer's actual capacity
	defer func() {
		if recover() == nil {
			t.Fatal("copyBytesToU32Buf did not panic on an oversized src — the bounds check did not fire")
		}
	}()
	copyBytesToU32Buf(dst, oversized)
}

// TestCopyBytesToU32Buf_exactFitStillWorks pins the ordinary (in-bounds) case still round-trips
// correctly through gpu.Upload, not just that the oversized case now panics.
func TestCopyBytesToU32Buf_exactFitStillWorks(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const nWords = 4
	dst := gpu.NewBufferLenOf[uint32](d, nWords)

	src := make([]byte, nWords*4)
	for i := range src {
		src[i] = byte(i + 1)
	}
	copyBytesToU32Buf(dst, src)

	got := dst.U32s()
	want := []uint32{0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("word %d = %#x, want %#x", i, got[i], w)
		}
	}
}
