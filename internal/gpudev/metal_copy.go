//go:build darwin

package gpudev

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// DeviceCopy is one src→dst pair for CopyDeviceBatch. Both buffers are read at
// their bind offsets (Buffer.At), so a pair can move a sub-range of a larger
// allocation without a separate view type.
type DeviceCopy struct {
	Dst   Buffer
	Src   Buffer
	Bytes int
}

// CopyDevice copies nBytes from src into dst, device to device. Both buffers are
// read at their bind offsets.
//
// ON METAL THIS BUYS NOTHING, AND THAT IS THE POINT. Apple Silicon is UMA: an
// MTLBuffer's contents are already host-addressable, which is why Upload and
// Download here are plain copy() over selContents-mapped memory with no transfer
// at all. A device-to-device copy is therefore a memcpy between two mapped
// slices — the PCIe round trip this verb exists to eliminate on CUDA does not
// exist on this backend, so there is no speedup to report and none is claimed.
//
// It is here for API symmetry. The alternative is that every consumer targeting
// both backends writes a build-tagged branch around a call that is available on
// one of them, which is a worse outcome than a function that is merely
// unremarkable on darwin. The CUDA half carries the measurements and the
// synchronization hazard; see cuda_copy.go.
//
// Output contract: overwrites dst[:nBytes] (at its bind offset); do not pre-zero.
func CopyDevice(dst, src Buffer, nBytes int) error {
	if err := checkDeviceCopy(dst, src, nBytes); err != nil {
		return err
	}
	if nBytes == 0 {
		return nil
	}
	copyMapped(dst, src, nBytes)
	return nil
}

// CopyDeviceBatch runs every copy in order, coalescing adjacent pairs first. On
// CUDA this form exists to pay one synchronize instead of N and to collapse
// dispatches; here there is nothing to synchronize and a dispatch is just a
// memcpy call, so it is close to a plain loop — kept so the two backends present
// the same surface, perform the same number of copies, and a caller's batch code
// compiles and behaves the same on either.
//
// Every pair is validated before any bytes move, matching the CUDA half: a bad
// pair fails without leaving the batch half-applied.
func CopyDeviceBatch(copies []DeviceCopy) error {
	for i, c := range copies {
		if err := checkDeviceCopy(c.Dst, c.Src, c.Bytes); err != nil {
			return fmt.Errorf("metal: device copy %d of %d: %w", i, len(copies), err)
		}
	}
	for _, c := range coalesce(copies) {
		copyMapped(c.Dst, c.Src, c.Bytes)
	}
	return nil
}

// adjacent reports whether b continues a exactly — same MTLBuffer on both sides,
// and b's src/dst ranges each start where a's ended. Coalescing buys far less
// here than on CUDA (there is no dispatch to save, only a few memcpy calls), but
// it is kept identical so the two backends agree on how many copies a batch
// performs and a shared test can assert that on either.
func adjacent(a, b DeviceCopy) bool {
	return a.Src.id == b.Src.id && a.Dst.id == b.Dst.id &&
		a.Src.off+uintptr(a.Bytes) == b.Src.off &&
		a.Dst.off+uintptr(a.Bytes) == b.Dst.off
}

// copyMapped is the memcpy itself. Go's copy() handles overlap correctly (it is
// memmove semantics), which matters because dst and src may be views into the
// SAME MTLBuffer at different offsets — a caller snapshotting one region of a
// combined allocation into another is exactly the shape this verb is for.
func copyMapped(dst, src Buffer, nBytes int) {
	sp := unsafe.Slice(objc.Send[*byte](src.id, selContents), int(src.capacityBytes()))
	dp := unsafe.Slice(objc.Send[*byte](dst.id, selContents), int(dst.capacityBytes()))
	copy(dp[dst.off:dst.off+uintptr(nBytes)], sp[src.off:src.off+uintptr(nBytes)])
}

// checkDeviceCopy validates one pair, mirroring cuda_copy.go's checks and error
// vocabulary so a consumer sees the same failures on either backend. There is no
// context to compare on Metal and no mapped-host distinction to make (all UMA
// memory is mapped), so the bounds checks are the whole of it.
func checkDeviceCopy(dst, src Buffer, nBytes int) error {
	if nBytes < 0 {
		return fmt.Errorf("metal: device copy of %d bytes (negative)", nBytes)
	}
	if src.id == 0 || dst.id == 0 {
		return fmt.Errorf("metal: device copy with a nil buffer")
	}
	if got, want := src.capacityBytes(), uint64(src.off)+uint64(nBytes); uint64(got) < want {
		return fmt.Errorf("metal: device copy reads %d bytes at offset %d from a %d-byte buffer", nBytes, src.off, got)
	}
	if got, want := dst.capacityBytes(), uint64(dst.off)+uint64(nBytes); uint64(got) < want {
		return fmt.Errorf("metal: device copy writes %d bytes at offset %d to a %d-byte buffer", nBytes, dst.off, got)
	}
	return nil
}
