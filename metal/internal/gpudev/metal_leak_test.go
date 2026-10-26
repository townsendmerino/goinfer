//go:build darwin

package gpudev

import "testing"

// metal_leak_test.go gives the device layer its own leak coverage, now that it lives
// in aikit rather than goinfer: the ledger (allocs/objs) must reach 0 on release, and
// per-Device retains must stay isolated (M24/C5). These assert on the ledger — the
// deterministic signal goinfer switched to after RSS proved unreliable under macOS
// memory compression — via LedgerLen (and, in-package here, the raw device id).
// They need a real Metal device, so they skip when there is none.

const leakKernelSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void noop(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i]; }`

func newLeakDevice(t *testing.T) *Device {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	return d
}

// TestLedger_buffers: NewBuffer* records each alloc; ReleaseBuf swap-removes exactly
// one (no double-free / no phantom); ReleaseAll drains to zero.
func TestLedger_buffers(t *testing.T) {
	d := newLeakDevice(t)
	defer d.ReleaseObjects()
	defer d.ReleaseAll()

	const n = 5
	bufs := make([]Buffer, n)
	for i := range bufs {
		bufs[i] = d.NewBufferLen(4)
	}
	if a, _ := d.LedgerLen(); a != n {
		t.Fatalf("after %d allocs, ledger allocs=%d, want %d", n, a, n)
	}
	// Release a MIDDLE buffer — exercises the swap-remove, not just a tail pop.
	// (ReleaseBuf's contract is release-once; the swap-remove is what keeps the
	// later ReleaseAll from double-freeing it.)
	d.ReleaseBuf(bufs[2])
	if a, _ := d.LedgerLen(); a != n-1 {
		t.Errorf("after ReleaseBuf, allocs=%d, want %d", a, n-1)
	}
	d.ReleaseAll()
	if a, _ := d.LedgerLen(); a != 0 {
		t.Errorf("after ReleaseAll, allocs=%d, want 0", a)
	}
}

// TestLedger_objects: the command queue, pipeline, and library are tracked as objs;
// ReleaseObjects clears them AND nils the device id (the field access is exactly why
// this assertion belongs in aikit, in-package, rather than goinfer).
func TestLedger_objects(t *testing.T) {
	d := newLeakDevice(t)
	lib, err := d.CompileLibrary(leakKernelSrc, MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	_ = d.NewCommandQueue()
	if _, err := d.NewComputePipeline(lib, "noop"); err != nil {
		t.Fatal(err)
	}
	if _, o := d.LedgerLen(); o != 3 { // library + queue + pipeline
		t.Errorf("objs=%d, want 3 (library+queue+pipeline)", o)
	}
	if d.id == 0 {
		t.Fatal("device id nil before ReleaseObjects")
	}
	d.ReleaseObjects()
	if _, o := d.LedgerLen(); o != 0 {
		t.Errorf("after ReleaseObjects, objs=%d, want 0", o)
	}
	if d.id != 0 {
		t.Error("ReleaseObjects did not nil the device id")
	}
	// Idempotent: a second ReleaseObjects is a no-op (id already nil).
	d.ReleaseObjects()
}

// TestCurrentAllocatedSize_reflectsRealAllocation proves CurrentAllocatedSize tracks the
// device's actual native allocation, independent of this package's own ledger — the gap
// LedgerLen cannot cover (goinfer audit-metal-2026-09-12 G-06): ReleaseAll/ReleaseObjects
// clear allocs/objs to nil BEFORE sending release to each id, not because those releases
// succeeded, so a leak test asserting only LedgerLen()==0 cannot distinguish "everything was
// actually freed" from "the release loop silently did nothing." This grows and shrinks real
// GPU memory and checks MTLDevice's own reported total moves with it.
func TestCurrentAllocatedSize_reflectsRealAllocation(t *testing.T) {
	d := newLeakDevice(t)
	defer d.ReleaseObjects()

	base := d.CurrentAllocatedSize()

	const n, bytesPerBuf = 8, 1 << 20 // 8 x 1 MiB
	bufs := make([]Buffer, n)
	for i := range bufs {
		bufs[i] = d.NewBufferBytes(bytesPerBuf)
	}
	grown := d.CurrentAllocatedSize()
	if want := base + uint64(n*bytesPerBuf); grown < want {
		t.Fatalf("CurrentAllocatedSize after %d x %d-byte allocs = %d, want >= %d (base %d)",
			n, bytesPerBuf, grown, want, base)
	}

	d.ReleaseAll()
	after := d.CurrentAllocatedSize()
	if after >= grown {
		t.Fatalf("CurrentAllocatedSize after ReleaseAll = %d, want < %d (the grown value) — "+
			"release did not actually free device memory even though it would (LedgerLen()==0)", after, grown)
	}
	if after > base+bytesPerBuf { // allocator/alignment slack, not an exact match to base
		t.Errorf("CurrentAllocatedSize after ReleaseAll = %d, want close to base %d — real memory not fully released", after, base)
	}
	_ = bufs // referenced only for the alloc calls above; ReleaseAll already freed them
}

// TestLedger_perDeviceIsolation: MTLCreateSystemDefaultDevice +1-retains the shared
// system device per call, so each *Device owns an independent ledger and retain.
// Closing one must leave the other's ledger and device id intact (M24).
func TestLedger_perDeviceIsolation(t *testing.T) {
	d1 := newLeakDevice(t)
	d2, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("second device: %v", err)
	}
	defer d2.ReleaseObjects()
	defer d2.ReleaseAll()

	_ = d1.NewBufferLen(4)
	_ = d2.NewBufferLen(4)
	_ = d2.NewBufferLen(4)

	// Close d1 fully.
	d1.ReleaseAll()
	d1.ReleaseObjects()
	if a, o := d1.LedgerLen(); a != 0 || o != 0 {
		t.Errorf("closed d1 ledger = %d/%d, want 0/0", a, o)
	}
	// d2 is untouched: its ledger and its device retain both survive.
	if a, _ := d2.LedgerLen(); a != 2 {
		t.Errorf("d2 allocs = %d after closing d1, want 2 (untouched)", a)
	}
	if d2.id == 0 {
		t.Error("closing d1 nilled d2's device id — per-Device isolation broken")
	}
}
