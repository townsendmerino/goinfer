//go:build darwin

package gpudev

import "testing"

// TestEncoder_useResidencySet gates M-14 (goinfer audit-metal-2026-09-12.md): Encoder.UseResidencySet
// is a NEW entry point (macOS 15+'s per-encoder useResidencySet:, an alternative to
// Queue.AddResidencySet's queue-wide attach), and it must actually make a dispatch against a
// residency-set-only buffer complete correctly — a set that is committed and made resident but
// never attached to the encoder that touches its buffer would either abort the command buffer or
// (worse, on hardware that tolerates it) run against unvalidated memory.
//
// Compares three arms against a plain unmanaged dispatch (no residency set at all): the existing
// queue-wide AddResidencySet, the new per-encoder UseResidencySet, and — the actual scenario M-14
// exists for — a SECOND encoder on the same queue that does NOT call UseResidencySet, proving the
// per-encoder attach really is scoped to just the encoder it was called on and not silently queue-
// wide under the hood.
func TestEncoder_useResidencySet(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	if !ResidencySetsSupported() {
		t.Skip("MTLResidencySet unavailable (macOS <15)")
	}

	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void bump(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] += 1.0f;
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "bump")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()

	newSet := func(bufs ...Buffer) ResidencySet {
		rs, err := d.NewResidencySet()
		if err != nil {
			t.Fatalf("NewResidencySet: %v", err)
		}
		for _, b := range bufs {
			rs.Add(b)
		}
		rs.Commit()
		rs.RequestResidency()
		return rs
	}

	dispatch := func(e *Encoder, buf Buffer) {
		e.Dispatch(pipe, 8, 8, buf)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("command buffer aborted: %v", err)
		}
	}

	t.Run("plain, no residency set", func(t *testing.T) {
		buf := d.NewBufferLen(8)
		dispatch(q.Begin(), buf)
		got := buf.Floats()
		if got[0] != 1 {
			t.Errorf("buf[0] = %v, want 1", got[0])
		}
	})

	t.Run("queue-wide AddResidencySet", func(t *testing.T) {
		buf := d.NewBufferLen(8)
		rs := newSet(buf)
		q.AddResidencySet(rs)
		dispatch(q.Begin(), buf)
		if got := buf.Floats()[0]; got != 1 {
			t.Errorf("buf[0] = %v, want 1", got)
		}
	})

	t.Run("per-encoder UseResidencySet", func(t *testing.T) {
		buf := d.NewBufferLen(8)
		rs := newSet(buf)
		q2 := d.NewCommandQueue() // fresh queue: no lingering AddResidencySet from the prior subtest
		e := q2.Begin()
		e.UseResidencySet(rs)
		dispatch(e, buf)
		if got := buf.Floats()[0]; got != 1 {
			t.Errorf("buf[0] = %v, want 1", got)
		}
	})

	t.Run("second dispatch on the SAME queue without UseResidencySet still completes", func(t *testing.T) {
		// The scenario M-14 exists for: a queue with NOTHING attached at the queue level, one
		// encoder opts into the set via UseResidencySet, a SIBLING encoder on the same queue
		// touches an UNRELATED plain buffer and must be entirely unaffected by the first one's
		// residency set (proving the attach really is per-encoder, not silently promoted queue-wide).
		q3 := d.NewCommandQueue()
		setBuf := d.NewBufferLen(8)
		rs := newSet(setBuf)
		e1 := q3.Begin()
		e1.UseResidencySet(rs)
		dispatch(e1, setBuf)

		plainBuf := d.NewBufferLen(8)
		dispatch(q3.Begin(), plainBuf)

		if got := setBuf.Floats()[0]; got != 1 {
			t.Errorf("setBuf[0] = %v, want 1", got)
		}
		if got := plainBuf.Floats()[0]; got != 1 {
			t.Errorf("plainBuf[0] = %v, want 1", got)
		}
	})
}
