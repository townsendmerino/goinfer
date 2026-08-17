//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttnBlockFull_nonCausal gates the drafter's attention kernel against the one it was copied
// from, using a property that needs no CPU reference and cannot pass by accident.
//
// attn_batched masks row m at nKeys = startPos+m+1 (causal). attn_block_full uses nKeys =
// startPos+M for every row (non-causal), which is what the drafter's block requires:
//
//	decoder/dflash.go, blockTrunk.layer — "Non-causal: every block query attends every context
//	key AND every block key, including positions after itself. No mask, by construction."
//
// Two assertions, and they pin it from both sides:
//
//	LAST ROW IDENTICAL — at m = M-1 the causal bound startPos+(M-1)+1 EQUALS the non-causal
//	startPos+M, so the two kernels see the same keys and must agree BIT-FOR-BIT. This is the
//	strong one: it proves the copy changed nothing except the mask (same float4 K read, same
//	d-order, same two-pass softmax).
//
//	EARLIER ROWS DIFFER — at m < M-1 the causal kernel sees fewer keys, so the outputs must NOT
//	match. Without this, a kernel that silently kept the causal bound would pass the first check.
func TestAttnBlockFull_nonCausal(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}

	const (
		nH       = 4
		nKV      = 2
		hd       = 64
		startPos = 40 // "context" length
		M        = 8  // block width
		scale    = 0.125
		window   = 0
	)
	qDim, kvDim, nKeys := nH*hd, nKV*hd, startPos+M

	q := make([]float32, M*qDim)
	kc := make([]float32, nKeys*kvDim)
	vc := make([]float32, nKeys*kvDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
	}
	for i := range kc {
		kc[i] = float32(math.Cos(float64(i) * 0.21))
		vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
	}

	var causal, full []float32
	err = r.do(func() error {
		qb, kb, vb := r.af(len(q)), r.af(len(kc)), r.af(len(vc))
		outA, outB := r.af(M*qDim), r.af(M*qDim)
		if e := gpu.Upload(qb, q); e != nil {
			return e
		}
		if e := gpu.Upload(kb, kc); e != nil {
			return e
		}
		if e := gpu.Upload(vb, vc); e != nil {
			return e
		}
		cfg := LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((nKeys + 128) * 4)}
		args := func(dst Buffer) []gpu.KernelArg {
			return []gpu.KernelArg{Arg(qb), Arg(kb), Arg(vb),
				gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(int32(startPos)), gpu.ArgValue(float32(scale)),
				gpu.ArgValue(int32(window)), gpu.ArgValue(int32(M)), Arg(dst)}
		}
		if e := r.launch(r.bAttn, cfg, args(outA)...); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return fmt.Errorf("attn_batched (causal reference) failed: %w", e)
		}
		// Compiled HERE, not bound at model load. The kernel has no production consumer yet
		// (the resident drafter path is not built), and TestPipelineLint_boundKernelsAreLaunched
		// exists precisely to stop a kernel being NVRTC-compiled into every model load while
		// nothing launches it — that was gemv_w4a8_batched's exact history. When the drafter
		// path lands it binds this at load; until then the only launch site is this gate.
		bmod, e := r.dev.CompileLibrary(attnBlockPTX)
		if e != nil {
			return e
		}
		pipe, e := r.dev.NewComputePipeline(bmod, "attn_block_full")
		if e != nil {
			return e
		}
		if e := r.launch(pipe, cfg, args(outB)...); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return fmt.Errorf("attn_block_full failed: %w", e)
		}
		causal, full = make([]float32, M*qDim), make([]float32, M*qDim)
		if e := gpu.Download(outA, causal); e != nil {
			return e
		}
		return gpu.Download(outB, full)
	})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}

	// 1. last row: same key set, so bit-for-bit equality.
	last := M - 1
	for d := 0; d < qDim; d++ {
		a, b := causal[last*qDim+d], full[last*qDim+d]
		if a != b {
			t.Fatalf("row %d (the row where causal and non-causal see the SAME keys) differs at %d: "+
				"causal %v vs full %v — the copy changed more than the mask", last, d, a, b)
		}
	}
	t.Logf("row %d: bit-identical across %d values (same key set) — the copy changed only the mask", last, qDim)

	// 2. every earlier row must differ: it sees strictly more keys now.
	for m := 0; m < last; m++ {
		same := true
		for d := 0; d < qDim; d++ {
			if causal[m*qDim+d] != full[m*qDim+d] {
				same = false
				break
			}
		}
		if same {
			t.Errorf("row %d is IDENTICAL to causal — the mask did not widen (row should see %d keys, not %d)",
				m, nKeys, startPos+m+1)
		}
	}
	t.Logf("rows 0..%d all differ from causal — every block row now attends all %d keys", last-1, nKeys)
}
