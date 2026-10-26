//go:build linux

package gpudev

import (
	"runtime"
	"strings"
	"testing"

	"github.com/eitamring/gocudrv/cudaresult"
)

// TestCUDA_launchDirect: a pipeline carries its CUfunction (so LaunchDirect is not
// silently falling back to Launch), a direct launch from a bound thread computes what
// Launch computes, including through a bind offset, and a direct launch on the null
// stream from a thread with no context current is refused, not dropped.
func TestCUDA_launchDirect(t *testing.T) {
	d, q, p := setup(t, "saxpy")
	if p.raw == 0 {
		t.Fatal("the pipeline has no CUfunction: gocudrv's Function layout changed, and LaunchDirect would only ever take Launch's path")
	}

	const n = 8
	const a = float32(2.5)
	x := make([]float32, 2*n)
	y := make([]float32, 2*n)
	for i := range x {
		x[i] = float32(i + 1)
		y[i] = float32(10 * (i + 1))
	}
	run := func(direct bool) []float32 {
		dx := NewBufferOf(d, x)
		dy := NewBufferOf(d, y)
		dyOff := dy.At(n * 4) // the upper half: a bind offset
		args := []KernelArg{Arg(dyOff), Arg(dx), ArgValue(a), ArgValue(int32(n))}
		var err error
		if direct {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			if err = d.BindCurrentThread(); err != nil {
				t.Fatalf("BindCurrentThread: %v", err)
			}
			err = q.LaunchDirect(p, Grid1D(n, 256), args...)
		} else {
			err = q.Launch(p, Grid1D(n, 256), args...)
		}
		if err != nil {
			t.Fatalf("launch (direct=%v): %v", direct, err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, 2*n)
		if err := Download(dy, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		return got
	}
	want, got := run(false), run(true)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("y[%d]: %v direct, %v through Launch", i, got[i], want[i])
		}
	}
	if want[0] != y[0] || want[n] == y[n] {
		t.Fatalf("the reference launch did not write only the offset half: %v", want)
	}

	if err := q.LaunchDirect(p, Grid1D(n, 256), Arg(Buffer{}), Arg(NewBufferOf(d, x)), ArgValue(a), ArgValue(int32(n))); err == nil ||
		!strings.Contains(err.Error(), "nil buffer") {
		t.Fatalf("a nil buffer argument should be refused; got %v", err)
	}

	errc := make(chan error, 1)
	go func() {
		// A pooled thread may still carry a binding from an earlier locked goroutine, so clear it. Never unlocked: the
		// runtime retires the thread.
		runtime.LockOSThread()
		dx := NewBufferOf(d, x)
		if err := cudaresult.CtxSetCurrent(d.cx.Driver(), 0); err != nil {
			errc <- err
			return
		}
		errc <- Queue{d: d}.LaunchDirect(p, Grid1D(n, 256), Arg(dx), Arg(dx), ArgValue(a), ArgValue(int32(n)))
	}()
	if err := <-errc; err == nil || !strings.Contains(err.Error(), "BindCurrentThread") {
		t.Fatalf("a direct launch on the null stream from an unbound thread should be refused; got %v", err)
	}
}
