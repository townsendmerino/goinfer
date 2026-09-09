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

// requireCUDAResident loads a real (small) model just far enough to get a live *cudaResident —
// a device, stream and executor — the same minimal setup TestAttnBlockFull_nonCausal uses. The
// weights themselves are untouched by these tests; every kernel launch below uses synthetic Q/K/V.
func requireCUDAResident(t *testing.T) *cudaResident {
	t.Helper()
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
	t.Cleanup(func() { mc.Close() })
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	if !r.imgPrefillReady {
		t.Skip("attn_img_batched did not load on this build")
	}
	return r
}

// attnImgBatchedArgs builds the argument list attn_img_batched and attn_batched share (identical
// prefix; attn_img_batched appends imgStart/imgEnd). window=0 throughout unless a test overrides it.
func attnImgBatchedArgs(q, kc, vc Buffer, nH, nKV, hd, startPos int, scale float32, window, M int, ctx Buffer, imgStart, imgEnd int) []gpu.KernelArg {
	return []gpu.KernelArg{Arg(q), Arg(kc), Arg(vc),
		gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
		gpu.ArgValue(int32(startPos)), gpu.ArgValue(scale),
		gpu.ArgValue(int32(window)), gpu.ArgValue(int32(M)), Arg(ctx), ArgNull(),
		gpu.ArgValue(int32(imgStart)), gpu.ArgValue(int32(imgEnd))}
}

// TestAttnImgBatched_blockBidirectionalAndOutsideMatchesCausal is attn_img_batched's twin of
// TestAttnBlockFull_nonCausal, scoped to a SUB-RANGE rather than the whole M — the property that
// needs no CPU reference and cannot pass by accident, proven from both sides:
//
//   - OUTSIDE the block (before imgStart, and at/after imgEnd): the mask formula is UNCHANGED
//     from attn_batched's causal one, so output must be BIT-IDENTICAL to attn_batched's own —
//     catches a kernel that widened too much (leaked bidirectionality outside the block).
//   - The block's LAST row (pos=imgEnd-1): causal nKeys there already equals imgEnd (pos+1 ==
//     imgEnd), so the two kernels see the SAME keys and must agree bit-for-bit — same "boundary
//     equality" proof TestAttnBlockFull_nonCausal uses.
//   - Every OTHER row inside the block must DIFFER from causal — it now sees strictly more keys
//     (up to imgEnd, not just pos+1). Without this, a kernel that silently kept the causal bound
//     for interior rows would still pass the boundary-equality check above.
func TestAttnImgBatched_blockBidirectionalAndOutsideMatchesCausal(t *testing.T) {
	r := requireCUDAResident(t)

	const (
		nH  = 4
		nKV = 2
		hd  = 64
		// startPos=10, M=24 → rows m=0..23 sit at absolute positions pos=10..33. imgStart/imgEnd
		// are ABSOLUTE positions (pos, not row index m) — the kernel compares pos, matching
		// decoder/kvcache.go's attendHi(pos) convention. Rows m=8..15 (pos=18..25) are the block.
		startPos = 10
		M        = 24
		scale    = 0.125
		window   = 0
		imgStart = 18 // = startPos + 8 (row m=8)
		imgEnd   = 26 // = startPos + 16 (row m=15 is the last in-block row)
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

	var causal, img []float32
	err := r.do(func() error {
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
		cfgCausal := LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((nKeys + 128) * 4)}
		causalArgs := []gpu.KernelArg{Arg(qb), Arg(kb), Arg(vb),
			gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
			gpu.ArgValue(int32(startPos)), gpu.ArgValue(float32(scale)),
			gpu.ArgValue(int32(window)), gpu.ArgValue(int32(M)), Arg(outA), ArgNull()}
		if e := r.launch(r.bAttn, cfgCausal, causalArgs...); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return fmt.Errorf("attn_batched (causal reference) failed: %w", e)
		}
		cfgImg := cfgCausal // same M/nH grid, same shared-mem bound (no window here, nKeys<=startPos+M either way)
		imgArgs := attnImgBatchedArgs(qb, kb, vb, nH, nKV, hd, startPos, scale, window, M, outB, imgStart, imgEnd)
		if e := r.launch(r.bAttnImg, cfgImg, imgArgs...); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return fmt.Errorf("attn_img_batched failed: %w", e)
		}
		causal, img = make([]float32, M*qDim), make([]float32, M*qDim)
		if e := gpu.Download(outA, causal); e != nil {
			return e
		}
		return gpu.Download(outB, img)
	})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}

	rowsDiffer := func(m int) bool {
		for d := 0; d < qDim; d++ {
			if causal[m*qDim+d] != img[m*qDim+d] {
				return true
			}
		}
		return false
	}

	// Row indices, not absolute positions: pos = startPos+m, and imgStart/imgEnd above are
	// absolute (the kernel's own convention, matching attendHi(pos)) — so the block's row-index
	// span is [imgStart-startPos, imgEnd-startPos).
	blockStartRow, blockEndRow := imgStart-startPos, imgEnd-startPos

	for m := 0; m < blockStartRow; m++ {
		if rowsDiffer(m) {
			t.Errorf("row %d (before the block) must be bit-identical to causal — the mask leaked bidirectionality outside the block", m)
		}
	}
	for m := blockEndRow; m < M; m++ {
		if rowsDiffer(m) {
			t.Errorf("row %d (after the block) must be bit-identical to causal — the mask leaked bidirectionality outside the block", m)
		}
	}
	t.Logf("rows [0,%d) and [%d,%d) all bit-identical to causal", blockStartRow, blockEndRow, M)

	last := blockEndRow - 1
	if rowsDiffer(last) {
		t.Errorf("row %d (the block's LAST row, where causal nKeys already equals imgEnd) must be bit-identical to causal", last)
	} else {
		t.Logf("row %d: bit-identical (same key set as causal at the block's boundary)", last)
	}

	for m := blockStartRow; m < last; m++ {
		if !rowsDiffer(m) {
			t.Errorf("row %d (inside the block, before its last row) is IDENTICAL to causal — "+
				"the mask did not widen for this interior row", m)
		}
	}
	t.Logf("rows [%d,%d) all differ from causal — every interior block row attends the whole block", blockStartRow, last)
}

// TestAttnImgBatched_windowDecoupled is the direct proof for the plan's top risk: that
// attn_img_batched's sliding-window start is derived from the row's PLAIN CAUSAL key count, never
// from the image-widened one (cuda/attn_img_prefill.cu's header). A naive port of attn_batched's
// own COUPLED formula (winStart = nKeys-window, using the WIDENED nKeys) would move winStart later
// than the CPU reference's decoupled one — excluding real keys the CPU reference (decoder/kvcache.go
// attendHi+WindowStart, computed independently) attends to.
//
// Poison-key method (no bit-exact CPU reference needed): place an extreme-magnitude key/value at a
// position that the CORRECT decoupled formula includes but the WRONG coupled formula would exclude.
// If the kernel is correct, that key dominates the softmax and the output row is driven to (near)
// the poison value; if it silently used the coupled formula instead, the poison key is never
// attended and the output stays near the small, ordinary values every other key carries.
func TestAttnImgBatched_windowDecoupled(t *testing.T) {
	r := requireCUDAResident(t)

	const (
		nH       = 1
		nKV      = 1
		hd       = 64
		scale    = 0.125
		window   = 8
		startPos = 0
		imgStart = 20
		imgEnd   = 32 // imgLen=12
		M        = 40
	)
	// Row under test: pos = imgStart (the row minimizing the decoupled window's own winStart,
	// per imgBlockMaxNWin's derivation — imgStart > window-1 here, 20 > 7).
	const testRow = imgStart
	// Decoupled (correct): winStart = pos-window+1 = 20-8+1 = 13, nKeys = imgEnd = 32 → attends [13,32).
	// Coupled (wrong, naive port of attn_batched): winStart = nKeys-window = 32-8 = 24 → attends [24,32).
	const poisonKey = 15 // in [13,24): correct-includes, wrong-excludes.
	if !(13 <= poisonKey && poisonKey < 24) {
		t.Fatal("test setup: poisonKey must lie in the decoupled-vs-coupled gap")
	}

	qDim, kvDim, nKeys := nH*hd, nKV*hd, M // startPos=0, so nKeys cache rows = M

	q := make([]float32, M*qDim)
	kc := make([]float32, nKeys*kvDim)
	vc := make([]float32, nKeys*kvDim)
	// Ordinary small keys/values everywhere...
	for i := range kc {
		kc[i] = float32(math.Cos(float64(i)*0.21)) * 0.1
		vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.1
	}
	// ...q at the test row set to all-ones, dot-producted against an all-100s poison key: score =
	// 100*hd*scale = 800, versus ordinary keys scoring O(0.01) — softmax puts effectively all mass
	// on the poison key if it is attended at all.
	for d := 0; d < hd; d++ {
		q[testRow*qDim+d] = 1
		kc[poisonKey*kvDim+d] = 100
		vc[poisonKey*kvDim+d] = 999
	}

	var out []float32
	err := r.do(func() error {
		qb, kb, vb := r.af(len(q)), r.af(len(kc)), r.af(len(vc))
		outB := r.af(M * qDim)
		if e := gpu.Upload(qb, q); e != nil {
			return e
		}
		if e := gpu.Upload(kb, kc); e != nil {
			return e
		}
		if e := gpu.Upload(vb, vc); e != nil {
			return e
		}
		// Shared memory sized via imgBlockMaxNWin, exactly as prefillCore's launch site computes it —
		// this test would itself fail to launch (or read out of bounds) if that formula under-sizes.
		causalMaxNWin := startPos + M
		if window > 0 && window < causalMaxNWin {
			causalMaxNWin = window
		}
		maxNWin := imgBlockMaxNWin(causalMaxNWin, window, imgStart, imgEnd)
		cfg := LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((maxNWin + 128) * 4)}
		args := attnImgBatchedArgs(qb, kb, vb, nH, nKV, hd, startPos, scale, window, M, outB, imgStart, imgEnd)
		if e := r.launch(r.bAttnImg, cfg, args...); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return fmt.Errorf("attn_img_batched failed: %w", e)
		}
		out = make([]float32, M*qDim)
		return gpu.Download(outB, out)
	})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}

	got := out[testRow*qDim]
	if got < 500 {
		t.Fatalf("row %d (pos=%d, window=%d, block=[%d,%d)) output[0] = %v, want close to the poison "+
			"value 999 — the kernel's window start was NOT decoupled from the widened key count "+
			"(it excluded key %d, which the CPU reference's independent WindowStart/attendHi split "+
			"attends to)", testRow, imgStart, window, imgStart, imgEnd, got, poisonKey)
	}
	t.Logf("row %d output[0] = %v (near the poison value 999) — the decoupled window correctly "+
		"reached key %d, which a naive coupled formula would have excluded", testRow, got, poisonKey)
}
