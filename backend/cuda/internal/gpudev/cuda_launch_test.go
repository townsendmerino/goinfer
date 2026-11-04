//go:build linux

package gpudev

import (
	"math"
	"strings"
	"testing"
)

// cuda_launch_test.go covers the surface a TUNED kernel set needs and the ANN
// proving path does not: scalar kernel args passed by value, hand-picked launch
// geometry with dynamic shared memory, pinned host readback, and async
// launch-many-then-sync-once. Together these are what let a consumer express a
// whole decode loop importing only this package — no gocudrv type in any
// signature. All against a real device; each test skips cleanly without one.

// TestCUDA_argNull locks the null-pointer bias arg the tuned decode path needs. A zero
// Buffer is NOT a null pointer: Arg(Buffer{}) marshals through gocudrv's Arg, which rejects
// a nil buffer handle (ErrNilBuffer) — so a kernel that binds an absent optional buffer and
// guards on `if (bias) …` cannot be handed "no bias" that way. ArgNull passes CUdeviceptr 0,
// which marshals cleanly. (The end-to-end guard-on-null EXECUTION is validated by goinfer's
// real decode kernels — the optional-bias path — since there is no nvcc on this box to
// compile a bespoke guard kernel here.)
func TestCUDA_argNull(t *testing.T) {
	d, q, p := setup(t, "saxpy")
	n := 8
	dx := NewBufferLenOf[float32](d, n)
	// A zero Buffer is rejected at the arg-marshaling stage — this is exactly why ArgNull exists.
	err := q.Launch(p, Grid1D(n, 256), Arg(dx), Arg(Buffer{}), ArgValue(float32(1)), ArgValue(int32(n)))
	if err == nil || !strings.Contains(err.Error(), "nil buffer") {
		t.Fatalf("Arg(Buffer{}) should be rejected as a nil buffer; got err=%v", err)
	}
	// ArgNull is a real, initialized arg (a null CUdeviceptr), so it does not hit that path.
	if ArgNull().a == nil {
		t.Fatal("ArgNull produced an uninitialized KernelArg")
	}
}

// setup reaches the device and builds a pipeline from smoke.ptx, or skips.
func setup(t *testing.T, kernel string) (*Device, Queue, Pipeline) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(d.ReleaseObjects)
	lib, err := d.CompileLibrary(smokePTX)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	p, err := d.NewComputePipeline(lib, kernel)
	if err != nil {
		t.Fatalf("pipeline %q: %v", kernel, err)
	}
	return d, d.NewCommandQueue(), p
}

// TestCUDA_scalarArgs proves a by-value scalar actually reaches the kernel, mixed
// positionally with buffer args: saxpy's coefficient `a` and count `n` are
// ArgValue, the vectors are Arg. A scalar that failed to marshal would leave y
// unchanged or garbage — both caught here, since the expected value depends on a.
func TestCUDA_scalarArgs(t *testing.T) {
	d, q, p := setup(t, "saxpy")

	const n = 8
	const a = float32(2.5)
	x := make([]float32, n)
	y := make([]float32, n)
	want := make([]float32, n)
	for i := range x {
		x[i] = float32(i + 1)        // 1..8
		y[i] = float32(10 * (i + 1)) // 10..80
		want[i] = a*x[i] + y[i]
	}
	dx := NewBufferOf(d, x)
	dy := NewBufferOf(d, y)

	if err := q.Launch(p, Grid1D(n, 256), Arg(dy), Arg(dx), ArgValue(a), ArgValue(int32(n))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, n)
	if err := Download(dy, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("y[%d] = %v, want %v (scalar arg did not reach the kernel?)", i, got[i], want[i])
		}
	}
	t.Logf("saxpy a=%v: %v", a, got)

	// The gate must be able to fail: the SAME launch with a different coefficient
	// must produce different output, proving the assertion is sensitive to `a`
	// rather than passing on any value.
	if err := q.Launch(p, Grid1D(n, 256), Arg(dy), Arg(dx), ArgValue(float32(-1)), ArgValue(int32(n))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	after := make([]float32, n)
	if err := Download(dy, after); err != nil {
		t.Fatalf("Download: %v", err)
	}
	same := true
	for i := range after {
		if after[i] != got[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("break-it-first vacuous: changing the scalar coefficient changed nothing")
	}
}

// TestCUDA_explicitGeometrySharedMem proves GridOne's hand-picked geometry and
// DYNAMIC shared-memory sizing reach the driver: blocksum reduces the whole vector
// in ONE block, staging partials in shared memory the caller sized. Run1D cannot
// express this — it derives a multi-block grid and never sets shared memory.
func TestCUDA_explicitGeometrySharedMem(t *testing.T) {
	d, q, p := setup(t, "blocksum")

	const n, block = 1000, 256
	x := make([]float32, n)
	var want float64
	for i := range x {
		x[i] = float32(i + 1)
		want += float64(x[i])
	}
	dx := NewBufferOf(d, x)
	out := d.NewBufferLen(1)

	cfg := GridOne(block, block*4) // 1 block, `block` threads, block*4 bytes shared
	if cfg.GridX != 1 || cfg.BlockX != block || cfg.SharedMemBytes != block*4 {
		t.Fatalf("GridOne built %+v", cfg)
	}
	if err := q.Launch(p, cfg, Arg(dx), Arg(out), ArgValue(int32(n))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, 1)
	if err := Download(out, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if math.Abs(float64(got[0])-want) > 1e-3 {
		t.Errorf("blocksum = %v, want %v", got[0], want)
	}
	t.Logf("blocksum(1..%d) in 1 block with %d B shared = %v", n, block*4, got[0])

	// Under-sizing the shared memory must FAIL the launch rather than silently
	// reading past the allocation — the gate that the shared-mem field is really
	// being sent to the driver and not dropped on the floor.
	huge := GridOne(block, 1<<30) // 1 GiB of dynamic shared memory: far past any limit
	if err := q.Launch(p, huge, Arg(dx), Arg(out), ArgValue(int32(n))); err == nil {
		_ = q.Sync() //nolint:errcheck // draining the unexpectedly-accepted launch before the assertion below
		t.Error("a 1 GiB dynamic shared-memory request was accepted — SharedMemBytes is not reaching the driver")
	}
}

// TestCUDA_hostBufferRoundTrip covers the pinned-host readback path: allocate
// page-locked memory, run a kernel, copy the device result straight into it, and
// read it off the Slice view. This is the shape a hot readback uses (a vocab-sized
// logits vector every token) — one copy into reused pinned memory, no allocation.
func TestCUDA_hostBufferRoundTrip(t *testing.T) {
	d, q, p := setup(t, "saxpy")

	const n = 64
	const a = float32(3)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i)
	}
	dx := NewBufferOf(d, x)
	dy := NewBufferOf(d, make([]float32, n)) // zeros ⇒ y = a*x

	hb, err := NewHostBuffer[float32](d, n)
	if err != nil {
		t.Fatalf("NewHostBuffer: %v", err)
	}
	if hb.Len() != n {
		t.Fatalf("HostBuffer.Len = %d, want %d", hb.Len(), n)
	}
	// The pinned slice is ordinary host memory — writable, and stale content must
	// be overwritten by the readback (a no-op copy would leave the sentinel).
	for i := range hb.Slice() {
		hb.Slice()[i] = -999
	}

	if err := q.Launch(p, Grid1D(n, 128), Arg(dy), Arg(dx), ArgValue(a), ArgValue(int32(n))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := ReadToHost(dy, hb); err != nil {
		t.Fatalf("ReadToHost: %v", err)
	}
	got := hb.Slice()
	for i := range got {
		if want := a * x[i]; got[i] != want {
			t.Fatalf("pinned[%d] = %v, want %v", i, got[i], want)
		}
	}
	t.Logf("pinned readback of %d floats OK (head: %v)", n, got[:4])

	// Close must free the pinned memory AND drop it from the ledger, so
	// ReleaseObjects neither double-frees it nor miscounts.
	_, objsBefore := d.LedgerLen()
	if err := hb.Close(); err != nil {
		t.Errorf("HostBuffer.Close: %v", err)
	}
	if _, objsAfter := d.LedgerLen(); objsAfter != objsBefore-1 {
		t.Errorf("ledger objs %d → %d after HostBuffer.Close, want a drop of 1", objsBefore, objsAfter)
	}
	if err := hb.Close(); err != nil { // idempotent
		t.Errorf("second HostBuffer.Close: %v", err)
	}
}

// TestCUDA_asyncLaunchThenSync proves the launch-many-then-sync-at-a-boundary
// model: many Launches enqueue with NO per-call sync, one Sync waits for all of
// them, and because they share a stream they execute in issue order. `scale`
// applied k times composes to s^k exactly — a result that only holds if every
// launch ran, and ran in order.
func TestCUDA_asyncLaunchThenSync(t *testing.T) {
	d, q, p := setup(t, "scale")

	const n, reps = 1024, 12
	const s = float32(2)
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	dx := NewBufferOf(d, x)

	// reps launches, no sync between them.
	for range reps {
		if err := q.Launch(p, Grid1D(n, 256), Arg(dx), ArgValue(s), ArgValue(int32(n))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
	}
	// ONE sync for the whole batch.
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	want := float32(math.Pow(float64(s), reps)) // 2^12 = 4096
	got := make([]float32, n)
	if err := Download(dx, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range got {
		if got[i] != want {
			t.Fatalf("x[%d] = %v after %d async launches, want %v (a launch was dropped or ran out of order)", i, got[i], reps, want)
		}
	}
	t.Logf("%d async launches + 1 Sync: x = %v (= %v^%d)", reps, want, s, reps)
}

// TestCUDA_launchRejectsBadInput covers the guards on the new entry points, so a
// malformed launch is an error rather than an undefined-behavior dispatch.
func TestCUDA_launchRejectsBadInput(t *testing.T) {
	d, q, p := setup(t, "saxpy")
	dx := NewBufferOf(d, []float32{1, 2, 3, 4})

	for _, tc := range []struct {
		name string
		cfg  LaunchConfig
		p    Pipeline
		args []KernelArg
	}{
		{"zero geometry", LaunchConfig{}, p, []KernelArg{Arg(dx)}},
		{"zero block", LaunchConfig{GridX: 1, GridY: 1, GridZ: 1}, p, []KernelArg{Arg(dx)}},
		{"nil pipeline", Grid1D(4, 4), Pipeline{}, []KernelArg{Arg(dx)}},
		{"uninitialized arg", Grid1D(4, 4), p, []KernelArg{{}}},
		{"Grid1D(0)", Grid1D(0, 256), p, []KernelArg{Arg(dx)}},
		{"GridOne(0)", GridOne(0, 0), p, []KernelArg{Arg(dx)}},
	} {
		if err := q.Launch(tc.p, tc.cfg, tc.args...); err == nil {
			t.Errorf("%s: Launch returned nil error, want a rejection", tc.name)
		}
	}
}

// TestCUDA_typedBufferVerbs round-trips every scalar element type a consumer
// actually allocates — float32 activations, int32 indices, uint16 f16 scales,
// uint32 packed weights — through the generic verbs. A per-type method set would
// have to grow a method for each; these cover all of them, which is what keeps a
// consumer's port to this layer mechanical.
func TestCUDA_typedBufferVerbs(t *testing.T) {
	d, _, _ := setup(t, "vadd")

	t.Run("float32", func(t *testing.T) {
		src := []float32{-1.5, 0, 2.25, 1e9}
		got := make([]float32, len(src))
		if err := Download(NewBufferOf(d, src), got); err != nil {
			t.Fatalf("round-trip: %v", err)
		}
		for i := range src {
			if got[i] != src[i] {
				t.Errorf("[%d] = %v, want %v", i, got[i], src[i])
			}
		}
	})
	t.Run("int32", func(t *testing.T) {
		src := []int32{-2147483648, -1, 0, 7, 2147483647}
		got := make([]int32, len(src))
		if err := Download(NewBufferOf(d, src), got); err != nil {
			t.Fatalf("round-trip: %v", err)
		}
		for i := range src {
			if got[i] != src[i] {
				t.Errorf("[%d] = %v, want %v", i, got[i], src[i])
			}
		}
	})
	t.Run("uint16", func(t *testing.T) {
		src := []uint16{0, 1, 0x3c00, 0xffff} // f16 bit patterns
		got := make([]uint16, len(src))
		if err := Download(NewBufferOf(d, src), got); err != nil {
			t.Fatalf("round-trip: %v", err)
		}
		for i := range src {
			if got[i] != src[i] {
				t.Errorf("[%d] = %v, want %v", i, got[i], src[i])
			}
		}
	})
	t.Run("uint32-alloc-then-upload", func(t *testing.T) {
		src := []uint32{0xdeadbeef, 0, 0xffffffff}
		b := NewBufferLenOf[uint32](d, len(src))
		if b.Len() != len(src) {
			t.Fatalf("Len = %d, want %d", b.Len(), len(src))
		}
		if err := Upload(b, src); err != nil {
			t.Fatalf("Upload: %v", err)
		}
		got := make([]uint32, len(src))
		if err := Download(b, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		for i := range src {
			if got[i] != src[i] {
				t.Errorf("[%d] = %#x, want %#x", i, got[i], src[i])
			}
		}
	})

	// Overrunning the allocation must be an error, not a silent stomp.
	small := NewBufferLenOf[float32](d, 2)
	if err := Upload(small, make([]float32, 8)); err == nil {
		t.Error("Upload past the end of a buffer was accepted")
	}
	if err := Download(small, make([]float32, 8)); err == nil {
		t.Error("Download past the end of a buffer was accepted")
	}
}

// goinferLaunchShape is a COMPILE-TIME lock on the calling convention a tuned
// decode path needs, transcribed from goinfer/cuda's resident.go: a kernel taking
// buffers and by-value scalars positionally interleaved, on hand-picked geometry
// with dynamic shared memory, enqueued async on a persistent queue. If a future
// change to this package breaks that shape, this file stops compiling — which is
// the point. It is never called; it only has to typecheck.
//
// Every gocudrv symbol goinfer uses has a counterpart here, so a port names only
// this package: gc.Arg→Arg, gc.ArgValue→ArgValue, gc.KernelArg→KernelArg,
// gc.LaunchConfig→LaunchConfig, gc.Function→Pipeline, gc.Stream→Queue,
// gc.Alloc→NewBufferLenOf, gc.CopyHtoD→Upload, gc.CopyDtoH→Download,
// gc.AllocHost→NewHostBuffer, gc.ArgDevicePtr→Arg(b.At(off)),
// gc.Init/gc.GetDevice→CreateSystemDefaultDevice, gc.Context→Device.Context.
func goinferLaunchShape(d *Device, q Queue, fRms Pipeline, hidden int, eps float32, addOne int32) error {
	// Allocation of the element types a decode path mixes.
	src := NewBufferLenOf[float32](d, hidden)
	nrm := NewBufferLenOf[float32](d, hidden)
	qOut := NewBufferLenOf[int32](d, hidden/4)
	sOut := NewBufferLenOf[uint16](d, hidden/32)

	// The representative launch: 1 block, 256 threads, (hidden+256)*4 bytes of
	// dynamic shared memory; scalars by value between buffer args; async.
	if err := q.Launch(fRms, GridOne(256, (hidden+256)*4),
		Arg(src), Arg(nrm),
		ArgValue(int32(hidden)), ArgValue(eps), ArgValue(addOne),
		Arg(qOut), Arg(sOut)); err != nil {
		return err
	}
	// A second launch on derived 1-D geometry, and a sub-view bind (base+offset).
	if err := q.Launch(fRms, Grid1D(hidden, 256), Arg(src.At(4*hidden/2)), ArgValue(int32(hidden))); err != nil {
		return err
	}
	// One sync at the boundary, then a pinned readback.
	if err := q.Sync(); err != nil {
		return err
	}
	pinned, err := NewHostBuffer[float32](d, hidden)
	if err != nil {
		return err
	}
	defer func() { _ = pinned.Close() }()
	if err := ReadToHost(src, pinned); err != nil {
		return err
	}
	_ = pinned.Slice()
	_ = d.Context() // escape hatch for driver surface this layer does not wrap
	return Upload(nrm, make([]float32, hidden))
}

var _ = goinferLaunchShape
