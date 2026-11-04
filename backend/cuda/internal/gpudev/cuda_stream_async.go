//go:build linux

package gpudev

import (
	"fmt"
	"runtime"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// Event is a CUDA event: a point in one queue's order that another queue (Queue.Wait) or the host
// (Event.Sync) can wait on WITHOUT draining anything else. Value type; released with the Device.
//
// This is the primitive Upload/UploadBatch's full-context synchronize stands in for. Those exist
// because a null-stream copy has no ordering against a CU_STREAM_NON_BLOCKING queue at all, so the
// only safe thing a synchronous copy can do is wait for everything. A copy issued ON a queue
// (UploadAsyncAt) is stream-ordered and needs no such wait; an event is how a consumer on a
// different queue, or the host, waits for exactly that copy and nothing more.
type Event struct {
	e *gc.Event
	d *Device // counts the event's waits
}

// NewEvent creates a timing-disabled event (the cheapest kind; Elapsed is not offered).
func (d *Device) NewEvent() (Event, error) {
	if d == nil || d.cx == nil {
		return Event{}, fmt.Errorf("cuda: NewEvent on a released device")
	}
	e, err := d.cx.NewEvent(gc.WithEventDisableTiming())
	if err != nil {
		return Event{}, fmt.Errorf("cuda: NewEvent: %w", err)
	}
	d.TrackObj(e)
	return Event{e: e, d: d}, nil
}

// Record marks this queue's current position in ev. Re-recording moves the mark; a Wait or Sync
// already issued against the previous mark keeps it.
func (q Queue) Record(ev Event) error {
	if ev.e == nil {
		return fmt.Errorf("cuda: Record of a nil event")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: Record on a queue with no stream")
	}
	return ev.e.Record(q.s)
}

// Wait makes every later submission on this queue wait, on the device, for ev's recorded position.
// The host does not block. An event never recorded is complete by CUDA's definition, so Wait on it
// is a no-op.
func (q Queue) Wait(ev Event) error {
	if ev.e == nil {
		return fmt.Errorf("cuda: Wait on a nil event")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: Wait on a queue with no stream")
	}
	return q.s.WaitEvent(ev.e)
}

// Sync blocks the host until ev's recorded position has executed — and only that. Work queued
// after the Record keeps running.
func (ev Event) Sync() error {
	if ev.e == nil {
		return fmt.Errorf("cuda: Sync of a nil event")
	}
	if ev.d != nil {
		ev.d.waits.Add(1)
	}
	return ev.e.Synchronize(bg)
}

// UploadAsyncAt enqueues an H2D copy of n bytes from the pinned src at srcOff into dst at its bind
// offset, on THIS queue's stream, and returns without synchronizing.
//
// Stream-ordered: a kernel launched later on this queue observes the bytes; a consumer on another
// queue needs Record on this queue and Wait on that one. The host must not modify
// src[srcOff:srcOff+n] until the copy has executed (Event.Sync, or a later drain of this queue).
// Nothing about the DESTINATION is checked against work in flight: a kernel still reading dst on
// another queue is the caller's write-after-read hazard to order, exactly as with any stream-
// ordered copy.
//
// It differs from UploadAsync in taking an offset on BOTH sides. gocudrv's exported async copy
// insists the pinned source and the device buffer be the same length, which rules out the case this
// exists for — a slice of a large pinned stack into a slot of a large device buffer — so the copy
// goes through the raw driver from the calling thread (see onThread).
func (q Queue) UploadAsyncAt(dst Buffer, src *HostBuffer[uint8], srcOff, n int) error {
	if dst.b == nil {
		return fmt.Errorf("cuda: UploadAsyncAt into a nil or mapped-host buffer")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: UploadAsyncAt on a queue with no stream")
	}
	if src == nil || src.h == nil {
		return fmt.Errorf("cuda: UploadAsyncAt from a nil pinned buffer")
	}
	if n <= 0 {
		return nil
	}
	if srcOff < 0 || srcOff+n > src.Len() {
		return fmt.Errorf("cuda: UploadAsyncAt source [%d,%d) overruns a %d-byte pinned buffer", srcOff, srcOff+n, src.Len())
	}
	if got, want := dst.b.Bytes(), uint64(dst.off)+uint64(n); got < want {
		return fmt.Errorf("cuda: UploadAsyncAt of %d bytes at offset %d overruns a %d-byte buffer", n, dst.off, got)
	}
	s := src.h.Slice()
	dptr := dst.b.DevicePtr() + cudasys.CUdeviceptr(dst.off)
	err := q.d.onThread(func(drv *cudasys.Driver) error {
		return cudaresult.MemcpyHtoDAsync(drv, dptr, &s[srcOff], uint64(n), q.s.Raw())
	})
	runtime.KeepAlive(src)
	return err
}

// UploadAsyncAtFrom is UploadAsyncAt for a MappedHostBuffer source instead of a HostBuffer —
// the same stream-ordered offset H2D, working from either of MappedHostBuffer's two origins
// (NewMappedHostBuffer's own allocation, or RegisterMappedHostBuffer's caller-owned pin). Added
// alongside RegisterMappedHostBuffer: C′'s expert-slot DMA (goinfer's dmaExpertSlot) reads its
// pinned source through a MappedHostBuffer, not a bare HostBuffer, and UploadAsyncAt's existing
// signature is kept as shipped (gpu/v0.33.2) rather than widened, so this is additive, not a
// breaking change to released surface.
func (q Queue) UploadAsyncAtFrom(dst Buffer, src *MappedHostBuffer, srcOff, n int) error {
	if dst.b == nil {
		return fmt.Errorf("cuda: UploadAsyncAtFrom into a nil or mapped-host buffer")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: UploadAsyncAtFrom on a queue with no stream")
	}
	if src == nil {
		return fmt.Errorf("cuda: UploadAsyncAtFrom from a nil MappedHostBuffer")
	}
	s := src.Bytes()
	if s == nil {
		return fmt.Errorf("cuda: UploadAsyncAtFrom from a closed or empty MappedHostBuffer")
	}
	if n <= 0 {
		return nil
	}
	if srcOff < 0 || srcOff+n > len(s) {
		return fmt.Errorf("cuda: UploadAsyncAtFrom source [%d,%d) overruns a %d-byte pinned buffer", srcOff, srcOff+n, len(s))
	}
	if got, want := dst.b.Bytes(), uint64(dst.off)+uint64(n); got < want {
		return fmt.Errorf("cuda: UploadAsyncAtFrom of %d bytes at offset %d overruns a %d-byte buffer", n, dst.off, got)
	}
	dptr := dst.b.DevicePtr() + cudasys.CUdeviceptr(dst.off)
	err := q.d.onThread(func(drv *cudasys.Driver) error {
		return cudaresult.MemcpyHtoDAsync(drv, dptr, &s[srcOff], uint64(n), q.s.Raw())
	})
	runtime.KeepAlive(src)
	return err
}

// ZeroAsync enqueues a stream-ordered zero of n bytes of dst from its bind offset, on this queue.
// The queue-ordered replacement for "Upload a slice of zeros" (two full syncs) when the buffer's
// last reader and next writer are both on this queue.
func (q Queue) ZeroAsync(dst Buffer, n int) error {
	if dst.b == nil {
		return fmt.Errorf("cuda: ZeroAsync of a nil or mapped-host buffer")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: ZeroAsync on a queue with no stream")
	}
	if n <= 0 {
		return nil
	}
	if got, want := dst.b.Bytes(), uint64(dst.off)+uint64(n); got < want {
		return fmt.Errorf("cuda: ZeroAsync of %d bytes at offset %d overruns a %d-byte buffer", n, dst.off, got)
	}
	dptr := dst.b.DevicePtr() + cudasys.CUdeviceptr(dst.off)
	return q.d.onThread(func(drv *cudasys.Driver) error {
		return cudaresult.MemsetD8Async(drv, dptr, 0, uint64(n), q.s.Raw())
	})
}

// Host returns the pinned host buffer backing m — the source form UploadAsyncAt takes, so a
// mapped stack can be DMA'd by slices as well as read zero-copy.
func (m *MappedHostBuffer) Host() *HostBuffer[uint8] {
	if m == nil {
		return nil
	}
	return m.hb
}

// onThread runs fn with the device's context current on the CALLING OS thread.
//
// gocudrv marshals every wrapped driver call onto a private command thread and exposes no offset
// async copy or memset, so the two raw calls above are issued directly. Every wrapped call returns
// only after the command thread has enqueued it, so a raw call issued afterwards lands on the
// stream in program order — the ordering a caller reads off the source is the ordering the stream
// gets. A CUDA context may be current on any number of threads at once; the previous binding is
// restored on the way out so a caller's own thread state is untouched.
func (d *Device) onThread(fn func(drv *cudasys.Driver) error) error {
	if d == nil || d.cx == nil {
		return fmt.Errorf("cuda: raw driver call on a released device")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	drv := d.cx.Driver()
	prev, err := cudaresult.CtxGetCurrent(drv)
	if err != nil {
		return err
	}
	cur := d.cx.Raw()
	if prev == cur {
		return fn(drv)
	}
	if err := cudaresult.CtxSetCurrent(drv, cur); err != nil {
		return err
	}
	err = fn(drv)
	if rerr := cudaresult.CtxSetCurrent(drv, prev); rerr != nil && err == nil {
		err = fmt.Errorf("cuda: restoring the caller's context: %w", rerr)
	}
	return err
}
