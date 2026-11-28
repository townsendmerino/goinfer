// Package winfault measures how far below a frame a recovered nil dereference writes into its goroutine's stack.
// It has no non-test code. On Windows the system delivers a hardware fault on the faulting goroutine's own stack, so
// the exception record and context it pushes there count toward that depth, and the runtime sets aside a fixed number
// of bytes for them (stackSystem). TestFaultFrameDepth takes the reading. It faults on purpose, so it runs only with
// -probe; TestMeasure_readsAKnownDepth checks the instrument everywhere and takes no fault.
package winfault

import (
	"flag"
	"runtime"
	"runtime/debug"
	"testing"
	"unsafe"
)

var probe = flag.Bool("probe", false, "take the fault frame reading (TestFaultFrameDepth)")

const (
	span      = 64 << 10 // bytes of stack below the anchor that are patterned, then scanned
	near      = 1024     // the nearest bytes are left alone: the frames of fill and deepest, and of what they call, live there
	fillByte  = 0xA5
	burnBytes = 4096 // the array burn writes, the known depth
	slack     = 2048 // how far past burnBytes a reading of burn may be: its frame, sum's, and the caller's below the anchor

	// descend recurses depthFrames frames and probes in frame probeFrame on the way back up.
	depthFrames = 300
	probeFrame  = 100
)

// work is what measure runs between laying the pattern and reading it back.
type work int

const (
	burnKnown work = iota // a callee that writes a burnBytes array
	softPanic             // a recovered Go panic: the recover path with no hardware fault
	hardFault             // a recovered nil dereference
)

var sink int

//go:noinline
func addr(p unsafe.Pointer) uintptr { return uintptr(p) }

// local returns the address of one of its own locals. Two calls from one frame agree when the compiler keeps such a
// local on the stack, and differ when it moves it to the heap (as -asan and -d=checkptr=2 do).
//
//go:noinline
func local() uintptr {
	var x byte
	return addr(unsafe.Pointer(&x))
}

// fill patterns the stack from near to span bytes below top. It writes nothing and returns false unless top is a
// stack local of a frame just above its own, which is the only case where that range is this goroutine's unused stack.
//
//go:noinline
func fill(top unsafe.Pointer) bool {
	var here byte
	d := addr(top) - addr(unsafe.Pointer(&here))
	first, second := local(), local()
	if d == 0 || d > near/2 || first != second {
		return false
	}
	for off := near; off < span; off++ {
		*(*byte)(unsafe.Add(top, -off)) = fillByte
	}
	return true
}

// deepest returns the largest offset below top whose byte no longer holds the pattern, or 0 when every byte from near
// to span still does.
//
//go:noinline
func deepest(top unsafe.Pointer) int {
	for off := span - 1; off >= near; off-- {
		if *(*byte)(unsafe.Add(top, -off)) != fillByte {
			return off
		}
	}
	return 0
}

//go:noinline
func sum(b *[burnBytes]byte) (s int) {
	for _, v := range b {
		s += int(v)
	}
	return s
}

//go:noinline
func burn() {
	var buf [burnBytes]byte
	for i := range buf {
		buf[i] = ^byte(fillByte)
	}
	sink += sum(&buf)
}

//go:noinline
func nilPtr() *int { return nil }

// fault panics and recovers: by a nil dereference when hard, else by a Go panic.
//
//go:noinline
func fault(hard bool) {
	defer func() { _ = recover() }()
	if hard {
		sink = *nilPtr()
	}
	panic("soft")
}

// measure patterns the stack below its own frame, runs w in a callee, and returns how deep that callee and everything
// under it wrote. ok is false when the pattern could not be laid or the stack moved meanwhile. The caller must have
// span bytes of stack to spare below this frame: descend arranges that.
//
//go:noinline
func measure(w work) (depth int, ok bool) {
	var anchor byte
	top := unsafe.Pointer(&anchor)
	before := addr(top)
	if !fill(top) {
		return 0, false
	}
	switch w {
	case burnKnown:
		burn()
	case softPanic:
		fault(false)
	case hardFault:
		fault(true)
	}
	if addr(top) != before {
		return 0, false
	}
	return deepest(top), true
}

// descend recurses n frames of about 1 KiB each and, on the way back up, calls f in frame at. The stack below f has
// then been grown and released, and f sits on more than a quarter of the stack, so a collection would not shrink it.
//
//go:noinline
func descend(n, at int, f func()) byte {
	var pad [1024]byte
	pad[n&1023] = byte(n)
	var r byte
	if n > 0 {
		r = descend(n-1, at, f)
	}
	if n == at {
		f()
	}
	return r + pad[n&1023]
}

// quiet runs a collection and then holds the collector off for the test, so no stack is shrunk while a pattern lies
// below a live frame.
func quiet(t *testing.T) {
	runtime.GC()
	old := debug.SetGCPercent(-1)
	t.Cleanup(func() { debug.SetGCPercent(old) })
}

// The instrument's own check: a callee that writes a burnBytes array reads at least that deep and not much deeper.
func TestMeasure_readsAKnownDepth(t *testing.T) {
	quiet(t)
	best, n := span, 0
	descend(depthFrames, probeFrame, func() {
		for range 8 {
			if d, ok := measure(burnKnown); ok {
				best, n = min(best, d), n+1
			}
		}
	})
	if n == 0 {
		t.Fatal("no reading: the pattern could not be laid below the probe's frame")
	}
	if best < burnBytes || best >= burnBytes+slack {
		t.Fatalf("a callee wrote a %d-byte array and the shallowest of %d readings is %d bytes below the anchor; want %d to %d",
			burnBytes, n, best, burnBytes, burnBytes+slack-1)
	}
	t.Logf("known depth: a callee's %d-byte array reads %d bytes (%d readings)", burnBytes, best, n)
}

// The reading, in bytes below a local of the faulting function's caller: nil_deref is the shallowest of 16 recovered
// nil dereferences, nil_deref_max the deepest (a preemption can add to one), and go_panic the shallowest of 16
// recovered Go panics. A depth of 0 is under near. On Linux and macOS a fault is handled on the signal stack, so the
// two differ only by the runtime's own frames.
func TestFaultFrameDepth(t *testing.T) {
	if !*probe {
		t.Skip("a diagnostic that faults on purpose: run it with -args -probe")
	}
	quiet(t)
	hard, hardMax, nHard := span, 0, 0
	soft, nSoft := span, 0
	descend(depthFrames, probeFrame, func() {
		for range 16 {
			if d, ok := measure(hardFault); ok {
				hard, hardMax, nHard = min(hard, d), max(hardMax, d), nHard+1
			}
			if d, ok := measure(softPanic); ok {
				soft, nSoft = min(soft, d), nSoft+1
			}
		}
	})
	if nHard == 0 || nSoft == 0 {
		t.Fatalf("no reading: %d nil dereferences and %d Go panics of 16 each; the pattern could not be laid below the probe's frame", nHard, nSoft)
	}
	t.Logf("fault frame depth: nil_deref=%d go_panic=%d nil_deref_max=%d bytes, 0 meaning under %d (%d and %d readings, %s/%s, %s)",
		hard, soft, hardMax, near, nHard, nSoft, runtime.GOOS, runtime.GOARCH, runtime.Version())
}
