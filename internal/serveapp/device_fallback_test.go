package serveapp

import (
	"errors"
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
	out, err := f.run("X", &a, func(x multimodal.GridTowerAccelerator) ([]float32, error) {
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
	if out, err = f.run("X", &a, func(x multimodal.GridTowerAccelerator) ([]float32, error) {
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
	if _, err = f.run("X", &a, func(multimodal.GridTowerAccelerator) ([]float32, error) { return nil, boom }); !errors.Is(err, boom) || acc2.closed || a == nil {
		t.Errorf("a non-memory error: err %v closed %v acc %v; want it passed through with the accelerator kept", err, acc2.closed, a)
	}
	if !isDeviceMemoryError(oom) || isDeviceMemoryError(boom) || isDeviceMemoryError(nil) {
		t.Error("isDeviceMemoryError misclassifies")
	}
}
