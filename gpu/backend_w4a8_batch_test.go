//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// buildW4A8Op builds one random int4 projection (decoder's native 2-nibble/byte packed layout)
// of shape [N,K] and its linalg.W4A8Op, sharing K with the caller's activation.
func buildW4A8Op(rng *rand.Rand, N, K int) (linalg.W4A8Op, []float32) {
	rowBytes := (K + 1) / 2
	bQ4 := make([]byte, N*rowBytes)
	for i := range bQ4 {
		bQ4[i] = byte(rng.Intn(256))
	}
	nGroups := (K + w4a8GroupSize - 1) / w4a8GroupSize
	scales := make([]float32, N*nGroups)
	for i := range scales {
		scales[i] = 0.01 + rng.Float32()*0.05
	}
	dst := make([]float32, N)
	return linalg.W4A8Op{W4: bQ4, Scales: scales, Dst: dst, N: N}, dst
}

// TestWebGPUBackend_MatmulW4A8Batch_matchesCPU is the batched int4 twin of
// TestWebGPUBackend_MatmulW4A8_matchesCPU: three ops sharing one activation, the way the
// qkv/gate-up call sites in decoder/attention.go and decoder/mlp.go build them. It compares the
// one-submit GPU batch against linalg.MatmulBTW4A8Batch, the CPU reference matmulW4A8Batch falls
// back to when no backend implements QuantBatchBackend4.
func TestWebGPUBackend_MatmulW4A8Batch_matchesCPU(t *testing.T) {
	be, err := newWebGPUBackend("decoder")
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer be.Close()

	const K = 1536 // K%32==0: the fast (byte-identical) upload path, matching real q/k/v shapes
	const group = w4a8GroupSize
	shapes := []int{1536, 256, 256} // q (full width), k, v (GQA-narrower) — mirrors qkvShapes
	rng := rand.New(rand.NewSource(11))

	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}

	gpuOps := make([]linalg.W4A8Op, len(shapes))
	cpuOps := make([]linalg.W4A8Op, len(shapes))
	for i, n := range shapes {
		op, _ := buildW4A8Op(rng, n, K)
		gpuOps[i] = op
		// Independent Dst backing arrays: the GPU and CPU runs must not alias each other's output.
		cpuOps[i] = linalg.W4A8Op{W4: op.W4, Scales: op.Scales, Dst: make([]float32, n), N: n}
	}

	if !be.MatmulW4A8Batch(a, 1, K, group, gpuOps) {
		t.Fatal("MatmulW4A8Batch declined — expected it to run on a real device")
	}

	var ws linalg.Workspace
	linalg.MatmulBTW4A8Batch(&ws, a, 1, K, group, cpuOps)

	for i := range shapes {
		var dot, na, nb, maxAbs float64
		for j := range gpuOps[i].Dst {
			g, c := float64(gpuOps[i].Dst[j]), float64(cpuOps[i].Dst[j])
			if d := math.Abs(g - c); d > maxAbs {
				maxAbs = d
			}
			dot += g * c
			na += g * g
			nb += c * c
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		t.Logf("op %d (N=%d): cosine=%.6f maxAbs=%.4e", i, shapes[i], cos, maxAbs)
		if cos < 0.999 {
			t.Errorf("op %d diverges from CPU: cosine=%.6f maxAbs=%.4e", i, cos, maxAbs)
		}
	}
}

// TestWebGPUBackend_MatmulW4A8Batch_declinesPrefill mirrors
// TestWebGPUBackend_MatmulW4A8_declinesPrefill: no int4 tiled/prefill kernel on this backend, so
// the batch path must decline (not silently run M=1 math against a multi-row activation) for M>1.
func TestWebGPUBackend_MatmulW4A8Batch_declinesPrefill(t *testing.T) {
	be, err := newWebGPUBackend("decoder")
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer be.Close()

	const N, K, M = 4, 64, 3
	rng := rand.New(rand.NewSource(13))
	op, _ := buildW4A8Op(rng, N, K)
	a := make([]float32, M*K)
	if be.MatmulW4A8Batch(a, M, K, w4a8GroupSize, []linalg.W4A8Op{op}) {
		t.Error("MatmulW4A8Batch must decline for M>1 (no int4 prefill kernel on this backend)")
	}
}

// TestWebGPUBackend_MatmulW4A8Batch_oneSubmit pins that MatmulW4A8Batch and MatmulW4A8 share one
// upload/cache path: a prior MatmulW4A8 call on op 0's bytes must make the batch call on the same
// bytes a cache hit, with the same cached resident weight (identity, via residentW4A8For's key
// map) and no re-upload. It does not count GPU submits itself, so the one-submit property of the
// batch is not asserted here.
func TestWebGPUBackend_MatmulW4A8Batch_oneSubmit(t *testing.T) {
	be, err := newWebGPUBackend("decoder")
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer be.Close()

	const N, K = 8, 1536
	rng := rand.New(rand.NewSource(17))
	op, dst := buildW4A8Op(rng, N, K)
	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}

	if !be.MatmulW4A8(a, op.W4, linalg.F32ToF16Scales(op.Scales), w4a8GroupSize, dst, 1, K, N) {
		t.Fatal("MatmulW4A8 declined — expected it to run on a real device")
	}
	if len(be.q4resident) != 1 {
		t.Fatalf("q4resident has %d entries after one MatmulW4A8 call, want 1", len(be.q4resident))
	}
	cached := be.q4resident[&op.W4[0]]

	dst2 := make([]float32, N)
	op2 := linalg.W4A8Op{W4: op.W4, Scales: op.Scales, Dst: dst2, N: N}
	if !be.MatmulW4A8Batch(a, 1, K, w4a8GroupSize, []linalg.W4A8Op{op2}) {
		t.Fatal("MatmulW4A8Batch declined — expected it to run on a real device")
	}
	if len(be.q4resident) != 1 {
		t.Errorf("q4resident has %d entries after MatmulW4A8Batch on the SAME weight bytes, want 1 — "+
			"the batch path re-uploaded instead of reusing MatmulW4A8's cached resident weight", len(be.q4resident))
	}
	if be.q4resident[&op.W4[0]] != cached {
		t.Error("MatmulW4A8Batch cached a DIFFERENT *q4Resident for the same weight bytes — " +
			"the two entry points are not sharing residentW4A8For's cache")
	}
}
