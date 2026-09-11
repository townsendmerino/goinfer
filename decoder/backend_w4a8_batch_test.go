package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// fakeBatchBackend4 is a minimal Backend that also implements QuantBatchBackend4, recording
// whether it was called and letting the test control whether it accepts (true) or declines
// (false) — the same decline contract a real GPU backend uses (M>1, a GPU error, etc.).
type fakeBatchBackend4 struct {
	called bool
	accept bool
}

func (f *fakeBatchBackend4) Name() string                              { return "fake" }
func (f *fakeBatchBackend4) MatmulBT(a, b, dst []float32, M, K, N int) {}
func (f *fakeBatchBackend4) Close() error                              { return nil }
func (f *fakeBatchBackend4) MatmulW4A8Batch(a []float32, M, K, group int, ops []linalg.W4A8Op) bool {
	f.called = true
	return f.accept
}

// TestMatmulW4A8Batch_routesThroughQuantBatchBackend4 is P-16's wiring gate (audit-2026-09-10):
// matmulW4A8Batch must route through a QuantBatchBackend4 when the active Backend implements it,
// and fall back to the CPU kernel (linalg.MatmulBTW4A8Batch) when it does not, or when the
// backend itself declines — the same contract matmulW8A8Batch already has, extended to int4.
// A fake backend proves the ROUTING decision directly, independent of any real GPU kernel (that
// numeric correctness is gpu/backend_w4a8_batch_test.go's job, on real hardware).
func TestMatmulW4A8Batch_routesThroughQuantBatchBackend4(t *testing.T) {
	const N, K, group = 4, 64, 32
	nGroups := K / group
	a := make([]float32, K)
	op := linalg.W4A8Op{W4: make([]byte, N*K/2), Scales: make([]float32, N*nGroups), Dst: make([]float32, N), N: N}

	t.Run("backend accepts: CPU kernel must NOT also run", func(t *testing.T) {
		be := &fakeBatchBackend4{accept: true}
		// Poison Dst with a sentinel the CPU kernel would overwrite — proves the CPU fallback
		// did NOT also run after the backend accepted (the backend's own copy is a no-op here
		// since the fake never writes Dst, so Dst staying at the sentinel IS the proof).
		for i := range op.Dst {
			op.Dst[i] = -999
		}
		var ws linalg.Workspace
		matmulW4A8Batch(be, &ws, a, 1, K, group, []linalg.W4A8Op{op})
		if !be.called {
			t.Fatal("QuantBatchBackend4.MatmulW4A8Batch was never called — matmulW4A8Batch did not " +
				"check the interface at all")
		}
		for i, v := range op.Dst {
			if v != -999 {
				t.Fatalf("Dst[%d] = %v, want untouched sentinel -999 — the CPU fallback ran even "+
					"though the backend accepted", i, v)
			}
		}
	})

	t.Run("backend declines: CPU kernel must run", func(t *testing.T) {
		be := &fakeBatchBackend4{accept: false}
		for i := range op.Dst {
			op.Dst[i] = -999
		}
		var ws linalg.Workspace
		matmulW4A8Batch(be, &ws, a, 1, K, group, []linalg.W4A8Op{op})
		if !be.called {
			t.Fatal("QuantBatchBackend4.MatmulW4A8Batch was never called")
		}
		allSentinel := true
		for _, v := range op.Dst {
			if v != -999 {
				allSentinel = false
			}
		}
		if allSentinel {
			t.Fatal("Dst is still all-sentinel after a decline — the CPU fallback never ran")
		}
	})

	t.Run("backend does not implement QuantBatchBackend4: CPU kernel must run", func(t *testing.T) {
		be := &fakeMatmulBTBackend{} // implements only Backend, no batch interface
		for i := range op.Dst {
			op.Dst[i] = -999
		}
		var ws linalg.Workspace
		matmulW4A8Batch(be, &ws, a, 1, K, group, []linalg.W4A8Op{op})
		allSentinel := true
		for _, v := range op.Dst {
			if v != -999 {
				allSentinel = false
			}
		}
		if allSentinel {
			t.Fatal("Dst is still all-sentinel — the CPU fallback never ran for a backend with no " +
				"QuantBatchBackend4 implementation")
		}
	})
}

// fakeMatmulBTBackend implements only the bare Backend interface — no quantized batch support at
// all — so matmulW4A8Batch's type-assertion must fail and fall through to the CPU kernel.
type fakeMatmulBTBackend struct{}

func (fakeMatmulBTBackend) Name() string                              { return "fake-plain" }
func (fakeMatmulBTBackend) MatmulBT(a, b, dst []float32, M, K, N int) {}
func (fakeMatmulBTBackend) Close() error                              { return nil }
