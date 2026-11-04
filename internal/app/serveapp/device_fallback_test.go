package serveapp

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/multimodal"
)

type fakeGridAcc struct {
	closed bool
}

func (f *fakeGridAcc) Name() string { return "fake" }
func (f *fakeGridAcc) Hidden([]float32, [][3]int) ([]float32, error) {
	return nil, nil
}
func (f *fakeGridAcc) Close() error { f.closed = true; return nil }

// A device tower that runs out of memory is closed and the tower is CPU from then on; any other error is not the fallback's.
func TestDeviceFallback(t *testing.T) {
	var f deviceFallback
	oom := errors.New("cuda: device allocation failed (typed-len, 75294720 bytes): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")

	acc := &fakeGridAcc{}
	var a multimodal.GridTowerAccelerator = acc
	calls := 0
	out, err := f.run("X", false, &a, func(x multimodal.GridTowerAccelerator) ([]float32, error) {
		calls++
		if x != nil {
			return nil, oom
		}
		return []float32{7}, nil
	})
	if err != nil || len(out) != 1 || out[0] != 7 || calls != 2 || !acc.closed || a != nil {
		t.Fatalf("OOM: out %v err %v calls %d closed %v acc %v; want the CPU result after one device try, the accelerator closed and cleared", out, err, calls, acc.closed, a)
	}
	// from then on the accelerator is nil: one call, on the CPU
	calls = 0
	if out, err = f.run("X", false, &a, func(x multimodal.GridTowerAccelerator) ([]float32, error) {
		calls++
		if x != nil {
			t.Error("a cleared accelerator must not be offered again")
		}
		return []float32{8}, nil
	}); err != nil || out[0] != 8 || calls != 1 {
		t.Fatalf("after the fallback: out %v err %v calls %d", out, err, calls)
	}

	// An ordinary error passes through, and the accelerator stays.
	acc2 := &fakeGridAcc{}
	a = acc2
	boom := errors.New("bad grid")
	if _, err = f.run("X", false, &a, func(multimodal.GridTowerAccelerator) ([]float32, error) { return nil, boom }); !errors.Is(err, boom) || acc2.closed || a == nil {
		t.Errorf("a non-memory error: err %v closed %v acc %v; want it passed through with the accelerator kept", err, acc2.closed, a)
	}
	if !isDeviceMemoryError(oom) || isDeviceMemoryError(boom) || isDeviceMemoryError(nil) {
		t.Error("isDeviceMemoryError misclassifies")
	}
}

// Under -require-backend a device tower's memory failure fails the request: no CPU run, the accelerator kept (the fallback must not run the CPU tower
// when the user asked never to leave the backend).
func TestDeviceFallback_requireBackendRefuses(t *testing.T) {
	var f deviceFallback
	oom := errors.New("cuda: device allocation failed (typed-len, 75294720 bytes): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")
	acc := &fakeGridAcc{}
	var a multimodal.GridTowerAccelerator = acc
	cpuRan := false
	out, err := f.run("X", true, &a, func(x multimodal.GridTowerAccelerator) ([]float32, error) {
		if x == nil {
			cpuRan = true
			return []float32{7}, nil
		}
		return nil, oom
	})
	if err == nil || out != nil || cpuRan || acc.closed || a == nil || !strings.Contains(err.Error(), "-require-backend") || !errors.Is(err, oom) {
		t.Fatalf("require: out %v err %v cpuRan %v closed %v acc %v; want the request refused naming -require-backend, no CPU run, the accelerator kept", out, err, cpuRan, acc.closed, a)
	}
}

// A device tower that panics inside run is an error, and a device-memory panic falls back like a returned allocation failure.
func TestDeviceFallback_recoversPanics(t *testing.T) {
	var f deviceFallback
	acc := &fakeGridAcc{}
	var a multimodal.GridTowerAccelerator = acc
	out, err := func() (out []float32, err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("panic escaped run: %v", r)
			}
		}()
		return f.run("X", false, &a, func(x multimodal.GridTowerAccelerator) ([]float32, error) {
			if x != nil {
				panic("metal: MTLBuffer allocation failed: device allocation failed (out of memory)")
			}
			return []float32{9}, nil
		})
	}()
	if err != nil || len(out) != 1 || out[0] != 9 || !acc.closed || a != nil {
		t.Fatalf("a device-memory panic: out %v err %v closed %v acc %v; want the CPU result with the accelerator closed", out, err, acc.closed, a)
	}
}

type fakeQwenResident struct{ attached, closed bool }

func (f *fakeQwenResident) ResidentEnabled() bool { return f.attached && !f.closed }
func (f *fakeQwenResident) Close()                { f.closed = true }

// The Qwen2.5-VL tower's fallback: without -require-backend a device-memory failure detaches the tower and reruns on the CPU; with it, the request fails and the
// tower stays attached.
func TestQwenDeviceForward_requireBackend(t *testing.T) {
	oom := errors.New("cuda: device allocation failed: CUDA_ERROR_OUT_OF_MEMORY")
	for _, require := range []bool{false, true} {
		tw := &fakeQwenResident{attached: true}
		calls := 0
		out, err := qwenDeviceForward(tw, require, func() ([]float32, error) {
			calls++
			if tw.ResidentEnabled() {
				return nil, oom
			}
			return []float32{3}, nil
		})
		if require {
			if err == nil || out != nil || calls != 1 || tw.closed || !strings.Contains(err.Error(), "-require-backend") {
				t.Errorf("require: out %v err %v calls %d closed %v; want refused after one device try, the tower kept", out, err, calls, tw.closed)
			}
		} else if err != nil || len(out) != 1 || calls != 2 || !tw.closed {
			t.Errorf("no require: out %v err %v calls %d closed %v; want the CPU result after detaching", out, err, calls, tw.closed)
		}
	}
}
