//go:build linux

package gpudev

import (
	"fmt"
)

// DeviceCopy is one src→dst pair for CopyDeviceBatch. Both buffers are read at
// their bind offsets (Buffer.At), so a pair can move a sub-range of a larger
// allocation without a separate view type.
type DeviceCopy struct {
	Dst   Buffer
	Src   Buffer
	Bytes int
}

// CopyDevice copies nBytes from src into dst, device to device, without a host
// round trip. Both buffers are read at their bind offsets. It returns only after
// the copy has completed — see the synchronization note below, which is the whole
// reason this function does more than forward to gocudrv.
//
// WHY THIS EXISTS. Upload (host→device) and Download (device→host) were the only
// transfer verbs, so moving bytes between two device buffers meant a PCIe round
// trip through host memory. goinfer measured that at 8.1 ms for ~20 MiB of
// recurrent decoder state on an RTX 2070 SUPER — about one whole resident decode
// step, a full token's work, to snapshot state that was already on the device.
// The same bytes device-to-device cost ~446 µs. That is ~18×, and it moves the
// operation from "costs a token" to "costs a few percent of one".
//
// MEASURED HERE, 2070 SUPER, idle box, median of 5: one contiguous 20 MiB copy
// takes 0.127 ms — 165 GB/s of bytes copied, 331 GB/s of memory TRAFFIC, 74% of
// the card's 448 GB/s ceiling. Quote whichever you mean and say which: a DtoD
// copy reads n and writes n, so the two conventions differ by exactly 2× and the
// hardware ceiling bounds the traffic one. See trafficGBs in cuda_copy_test.go —
// that factor of two is why an earlier version of the test reported 37% for a
// copy running at 74%.
//
// THE SYNCHRONIZATION IS NOT INCIDENTAL — READ THIS BEFORE "OPTIMIZING" IT AWAY.
// The underlying gocudrv call dispatches cuMemcpyDtoD_v2, which the CUDA driver
// defines as ASYNCHRONOUS with respect to the host for device-to-device copies.
// It returns once the copy is enqueued, not once it is done. gocudrv's own
// CopyToDeviceAt doc comment claims "It blocks until the copy completes"; that
// claim does not match the driver's contract, and measurement agrees with the
// driver: timing the call alone on a 20 MiB copy reports ~9 µs of dispatch rather
// than ~116 µs of transfer, which works out to 8145 GB/s on a card whose physical
// ceiling is 448 GB/s. Not a suspicious number — an impossible one.
//
// So this verb synchronizes explicitly, and its contract is "the bytes are there
// when it returns". That matches Upload, which also synchronizes, and it means a
// caller cannot accidentally read a half-copied buffer. A caller who genuinely
// wants overlap should reach for CopyDeviceBatch, which pays one synchronize for
// many copies rather than one each.
//
// Output contract: overwrites dst[:nBytes] (at its bind offset); do not pre-zero.
func CopyDevice(dst, src Buffer, nBytes int) error {
	if err := checkDeviceCopy(dst, src, nBytes); err != nil {
		return err
	}
	if nBytes == 0 {
		return nil
	}
	// Pre-sync for the same write-after-read hazard Buffer.upload documents
	// (audit C-01): a queued launch may still be reading dst.
	if err := syncCopyContext(dst, src); err != nil {
		return err
	}
	if err := src.b.CopyToDeviceAt(bg, int(dst.off), dst.b, int(src.off), nBytes); err != nil {
		return err
	}
	return syncCopyContext(dst, src)
}

// CopyDeviceBatch issues every copy and then synchronizes ONCE, which is the
// form the motivating consumer actually needs and the reason it is not just a
// loop over CopyDevice at the call site.
//
// WHY THE BATCH FORM EARNS ITS KEEP. The single-copy number flatters what callers
// see. One contiguous 20 MiB device-to-device copy runs at ~347 GB/s (78% of a
// 2070 SUPER's 448 GB/s peak). The real consumer does not issue one copy — it
// issues 36 (18 layers × two buffers, 73 KB and 1 MiB each), and that shape runs
// at ~174 GB/s of traffic, 39% of peak (reproduced here: 172.5 GB/s median of 5,
// against 331 for the contiguous shape), because a small copy is dominated by its ~9 µs of
// dispatch rather than by bandwidth. Batching removes 35 of the 36 synchronizes,
// and coalesce (copy_coalesce.go) removes whole dispatches wherever the caller's
// pairs are already adjacent. Benchmark both shapes before quoting a number
// (BenchmarkCopyDevice_contiguous vs BenchmarkCopyDevice_manySmall).
//
// Every pair is validated before any CUDA call, so a bad pair fails without
// leaving some copies issued and others not.
func CopyDeviceBatch(copies []DeviceCopy) error {
	if len(copies) == 0 {
		return nil
	}
	for i, c := range copies {
		if err := checkDeviceCopy(c.Dst, c.Src, c.Bytes); err != nil {
			return fmt.Errorf("cuda: device copy %d of %d: %w", i, len(copies), err)
		}
	}
	var issued bool
	for i, c := range coalesce(copies) {
		if c.Bytes == 0 {
			continue
		}
		if err := c.Src.b.CopyToDeviceAt(bg, int(c.Dst.off), c.Dst.b, int(c.Src.off), c.Bytes); err != nil {
			return fmt.Errorf("cuda: device copy %d (after coalescing) of %d issued: %w", i, len(copies), err)
		}
		issued = true
	}
	if !issued {
		return nil
	}
	return syncCopyContext(copies[0].Dst, copies[0].Src)
}

// adjacent reports whether b continues a exactly — same buffers on both sides,
// and b's src/dst ranges each start where a's ended.
func adjacent(a, b DeviceCopy) bool {
	return a.Src.b == b.Src.b && a.Dst.b == b.Dst.b &&
		a.Src.off+uintptr(a.Bytes) == b.Src.off &&
		a.Dst.off+uintptr(a.Bytes) == b.Dst.off
}

// checkDeviceCopy validates one pair up front, in aikit's error vocabulary rather
// than gocudrv's. gocudrv does check context and range itself and would return
// ErrContextMismatch / ErrOutOfRange, but its errors name neither buffer nor the
// offsets involved, and by the time a caller sees one it cannot tell which of 36
// batched pairs was wrong.
func checkDeviceCopy(dst, src Buffer, nBytes int) error {
	if nBytes < 0 {
		return fmt.Errorf("cuda: device copy of %d bytes (negative)", nBytes)
	}
	// A mapped-host Buffer carries a device-usable pointer into pinned HOST memory
	// (NewMappedHostBuffer) and has no gocudrv buffer behind it. Copying to or from
	// one is a PCIe transfer wearing a device-copy's clothes — exactly the cost this
	// verb exists to avoid — so it is refused by name rather than silently served.
	if src.raw != 0 || dst.raw != 0 {
		return fmt.Errorf("cuda: device copy with a mapped-host buffer — those bytes live in host RAM, " +
			"so this would be a PCIe transfer, not a device-to-device copy; use Download/Upload if that is what you want")
	}
	if src.b == nil || dst.b == nil {
		return fmt.Errorf("cuda: device copy with a nil buffer")
	}
	if src.cx != nil && dst.cx != nil && src.cx != dst.cx {
		return fmt.Errorf("cuda: device copy between buffers on different contexts")
	}
	if got, want := src.b.Bytes(), uint64(src.off)+uint64(nBytes); got < want {
		return fmt.Errorf("cuda: device copy reads %d bytes at offset %d from a %d-byte buffer", nBytes, src.off, got)
	}
	if got, want := dst.b.Bytes(), uint64(dst.off)+uint64(nBytes); got < want {
		return fmt.Errorf("cuda: device copy writes %d bytes at offset %d to a %d-byte buffer", nBytes, dst.off, got)
	}
	return nil
}

// syncCopyContext blocks until issued copies have completed. Either buffer's
// context will do (checkDeviceCopy has already established they agree); a Buffer
// built without one simply has nothing to wait on.
func syncCopyContext(dst, src Buffer) error {
	if cx := src.cx; cx != nil {
		noteWait(cx)
		return cx.Synchronize(bg)
	}
	if cx := dst.cx; cx != nil {
		noteWait(cx)
		return cx.Synchronize(bg)
	}
	return nil
}
