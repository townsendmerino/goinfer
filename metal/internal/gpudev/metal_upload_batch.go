//go:build darwin

package gpudev

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// HostCopy is one host→device pair for UploadBatch: Src's bytes go into Dst at
// Dst's bind offset (Buffer.At).
//
// Src is []byte rather than a generic []T because a batch is heterogeneous by
// nature — the motivating caller loads an expert's f32 weights and its f16 scales
// in the same pass — and a generic batch would force one element type across all
// of them. Upload[T] stays generic; this is the byte-level sibling.
type HostCopy struct {
	Dst Buffer
	Src []byte
}

// UploadBatch uploads every pair.
//
// ON METAL THERE IS NO SYNCHRONIZE TO AMORTIZE, WHICH IS THE WHOLE POINT OF THE
// CUDA VERSION — so this buys nothing here and does not pretend to. Apple Silicon
// is UMA: Upload is already a plain copy() into selContents-mapped memory with no
// transfer and no fence, so a batch of N is N copies either way. The saving on CUDA
// is ~240 device synchronizes per MoE decode token (~3.6 ms of 64 ms); the
// equivalent number here is zero, because there were never any.
//
// It exists so a consumer targeting both backends writes one call rather than a
// build-tagged branch, and so the two backends validate identically. See
// cuda_upload_batch.go for the measurements and the reasoning.
//
// Every pair is validated before any bytes move, matching the CUDA half: a bad pair
// fails without leaving the batch half-applied.
func UploadBatch(copies []HostCopy) error {
	for i, c := range copies {
		if err := checkHostCopy(c); err != nil {
			return fmt.Errorf("metal: upload %d of %d: %w", i, len(copies), err)
		}
	}
	for _, c := range copies {
		if len(c.Src) == 0 {
			continue
		}
		dst := unsafe.Slice(objc.Send[*byte](c.Dst.id, selContents), int(c.Dst.capacityBytes()))
		copy(dst[c.Dst.off:], c.Src)
	}
	return nil
}

// checkHostCopy mirrors cuda_upload_batch.go's checks and error vocabulary so a
// consumer sees the same failures on either backend. There is no context to compare
// on Metal and no mapped-host distinction to make (all UMA memory is mapped), so the
// bounds check is the whole of it.
func checkHostCopy(c HostCopy) error {
	if c.Dst.id == 0 {
		return fmt.Errorf("metal: upload into a nil buffer")
	}
	if got, want := c.Dst.capacityBytes(), uint64(c.Dst.off)+uint64(len(c.Src)); uint64(got) < want {
		return fmt.Errorf("metal: upload of %d bytes at offset %d overruns a %d-byte buffer",
			len(c.Src), c.Dst.off, got)
	}
	return nil
}
