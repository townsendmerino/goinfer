//go:build cuda

package cuda

import "github.com/townsendmerino/aikit/gpu"

// prefillSplitKVBudget bounds the score scratch of the row-batched exact attention: a launch group takes as many rows as fit
// it (at least one), so the scratch does not grow with the pass's rows.
const prefillSplitKVBudget = 64 << 20

// prefillSplitKVForceForTest, when set (tests only), routes every exact-attention layer of a batched pass to the row-batched
// kernels whatever its span, so a test can hold them to attn_batched below the shared-memory bound.
var prefillSplitKVForceForTest bool

// loadPrefillSplitKV binds prefill_splitkv.cu's three kernels, all or none: a pass routes to them only when every one loaded.
func (r *cudaResident) loadPrefillSplitKV() {
	mod, e := r.dev.CompileLibrary(r.kvPTX(prefillSplitKVPTX, prefillSplitKVKV16PTX))
	if e != nil {
		return
	}
	ok := true
	load := func(dst *Pipeline, name string) {
		if p, le := r.dev.NewComputePipeline(mod, name); le == nil {
			*dst = p
		} else {
			ok = false
		}
	}
	load(&r.psScores, "splitkv_scores_rows")
	load(&r.psSoftmax, "splitkv_softmax_rows")
	load(&r.psVsum, "splitkv_vsum_rows")
	if !ok {
		r.psScores, r.psSoftmax, r.psVsum = Pipeline{}, Pipeline{}, Pipeline{}
	}
}

func (r *cudaResident) prefillSplitKVReady() bool {
	return r.psScores != (Pipeline{}) && r.psSoftmax != (Pipeline{}) && r.psVsum != (Pipeline{})
}

// prefillSplitKVServes reports whether an exact-attention layer whose widest row attends maxNWin keys takes the row-batched
// kernels: exactly where attn_batched's shared memory cannot hold the row at the prefill's limit (prefillAttnFits).
func (r *cudaResident) prefillSplitKVServes(maxNWin int) bool {
	return r.prefillSplitKVReady() && (!r.prefillAttnFits(maxNWin) || prefillSplitKVForceForTest)
}

// prefillShmemOptInOffForTest, when set (tests only), makes optInPrefillShmem a no-op, so a load keeps the default limit.
var prefillShmemOptInOffForTest bool

// optInPrefillShmem opts the loaded exact prefill attention kernels (attn_batched, attn_img_batched, attn_span_batched) in to the
// most dynamic shared memory each can take beside its static shared memory, and sets prefillShmem to the smallest. Any kernel that
// cannot go past the default, or any failed opt-in, leaves prefillShmem 0 (the default limit for all). Decode and the drafter
// are not opted in: they keep splitKVRequired's bound.
func (r *cudaResident) optInPrefillShmem() {
	if prefillShmemOptInOffForTest || r.bAttn == (Pipeline{}) {
		return
	}
	ps := []*Pipeline{&r.bAttn, &r.bAttnImg, &r.bAttnSpan}
	limit := 0
	for _, p := range ps {
		if *p == (Pipeline{}) {
			continue
		}
		n := p.MaxDynamicSharedBytes(r.dev)
		if n <= singleBlockAttnShmemLimit {
			return
		}
		if limit == 0 || n < limit {
			limit = n
		}
	}
	for _, p := range ps {
		if *p != (Pipeline{}) && p.SetMaxDynamicSharedBytes(limit) != nil {
			return
		}
	}
	r.prefillShmem = limit
}

// prefillAttnShmemLimit is the dynamic shared memory the exact prefill attention kernels may request: what optInPrefillShmem
// set, or singleBlockAttnShmemLimit when it set nothing.
func (r *cudaResident) prefillAttnShmemLimit() int {
	if r.prefillShmem > 0 {
		return r.prefillShmem
	}
	return singleBlockAttnShmemLimit
}

// prefillAttnFits reports whether the exact prefill attention can hold a row attending nWin keys in shared memory.
func (r *cudaResident) prefillAttnFits(nWin int) bool {
	return attnShmemBytes(nWin) <= r.prefillAttnShmemLimit()
}

// prefillPastBoundExact reports whether a prompt whose last row attends keys keys has an exact-attention layer past the default
// 48 KiB shared-memory bound (splitKVRequired, window-clamped), with no fused attention serving its head dim. prefillCore runs
// every pass of such a prompt on the exact kernels, so it is bit-identical to the per-token path. The bound stays the default
// one where the opt-in moves the attention's (prefillAttnFits): a prompt between the two keeps the exact GEMM (KS-B15, owner
// decision A, docs/tasks/task-kernel-survey-2026-10.md). The tensor-core GEMM beside the row
// kernels missed its fidelity bar there; enabling it past the bound needs a fidelity gate of its own first
// (docs/measurements/kv3-tiled-exact-2026-10-13.md §11). The test hook that forces the route below the bound does not count.
func (r *cudaResident) prefillPastBoundExact(keys int) bool {
	if !r.prefillSplitKVReady() {
		return false
	}
	for l := range r.layers {
		Ly := &r.layers[l]
		maxNWin := keys
		if Ly.window > 0 && int(Ly.window) < maxNWin {
			maxNWin = int(Ly.window)
		}
		if !splitKVRequired(maxNWin) {
			continue
		}
		if p, _ := r.attnFusedFor(Ly.hd); !r.fastAttn || p == (Pipeline{}) {
			return true
		}
	}
	return false
}

// prefillSplitKVPlan sizes the score scratch of an ordinary pass of M rows whose last row attends keys keys: stride is the widest
// row over the layers that route to the row-batched kernels (0 when none does), and rows the rows one launch group takes. fused
// says whether the fused attention serves a head dim in this pass, as checkPrefillShmem's argument does.
func (r *cudaResident) prefillSplitKVPlan(M, keys int, fused func(hd int) bool, blocks [][2]int) (stride, rows int) {
	for l := range r.layers {
		Ly := &r.layers[l]
		maxNWin := keys
		if Ly.window > 0 && int(Ly.window) < maxNWin {
			maxNWin = int(Ly.window)
		}
		if len(blocks) > 0 {
			maxNWin = imgBlocksMaxNWin(maxNWin, int(Ly.window), blocks) // the widest row an image block gives this layer, as attn_img_batched sizes it
		}
		if !r.prefillSplitKVServes(maxNWin) || (fused != nil && fused(Ly.hd)) {
			continue
		}
		stride = max(stride, maxNWin)
	}
	if stride == 0 {
		return 0, 0
	}
	return stride, min(M, max(1, prefillSplitKVBudget/4/(r.nH*stride)))
}

// PrefillSpanScratchBytes (decoder.ResidentPrefillSpanScratch, matched structurally) is what a pass of rows rows whose last row
// attends keys keys allocates beyond PrefillScratchBytes(rows): the row-batched exact attention's scores and inverses, when a
// layer routes to it. It asks the fused question as the pass will for an ordinary last-row prefill at that length, and
// TestKV3_prefillScratchBytes_matchesThePass holds the sum to the pass.
func (r *cudaResident) PrefillSpanScratchBytes(rows, keys int) int64 {
	if rows <= 0 || keys <= 0 {
		return 0
	}
	fused := func(hd int) bool {
		if r.dnet != nil || r.eModel || r.deepPlan != nil || !r.fastAttn || rows < attnFusedMinRows || r.prefillPastBoundExact(keys) ||
			keys < fastPrefillFloorFor(r.knobValue("GOINFER_CUDA_FAST_PREFILL_FLOOR")) {
			return false
		}
		p, _ := r.attnFusedFor(hd)
		return p != (Pipeline{})
	}
	stride, R := r.prefillSplitKVPlan(rows, keys, fused, nil)
	if stride == 0 {
		return 0
	}
	return scratchBufBytes(R*r.nH*stride) + scratchBufBytes(R*r.nH)
}

// prefillSplitKVAttn is the exact attention of layer l for the M rows of a pass at startPos, written to cctxB as attn_batched
// writes it: three launches a group of R rows (scores over keys, the softmax at attn_batched's block width, the V fold over
// output dims). scB is [R][nH][stride] floats and invB [R][nH].
func (r *cudaResident) prefillSplitKVAttn(l, startPos, M int, qBb, cctxB, scB, invB Buffer, stride, R int, imgBlk Buffer, nBlocks int) error {
	Ly := &r.layers[l]
	const dTile = 32
	// The image-block table of attn_img_batched's contract ([start,end) int32 pairs), or null with nBlocks 0 for the causal case.
	blocksArg := ArgNull()
	if nBlocks > 0 {
		blocksArg = Arg(imgBlk)
	}
	for row0 := 0; row0 < M; row0 += R {
		n := min(R, M-row0)
		widest := startPos + row0 + n // the last row of the group attends this many keys before its window
		if Ly.window > 0 && int(Ly.window) < widest {
			widest = int(Ly.window)
		}
		if nBlocks > 0 {
			widest = stride // a row inside an image block sees past its own position: launch the plan's widest span, the surplus tiles return at once
		}
		if e := r.launch(r.psScores, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((widest + 127) / 128), GridZ: uint32(n), BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(Ly.hd * 4)},
			Arg(qBb), Arg(r.kvK(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(startPos)), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(row0)), gpu.ArgValue(r.attnScale),
			Arg(scB), gpu.ArgValue(int32(stride)), blocksArg, gpu.ArgValue(int32(nBlocks))); e != nil {
			return e
		}
		// Block 128: the softmax partition and ladder are attn_batched's only at that width.
		if e := r.launch(r.psSoftmax, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(n), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
			Arg(scB), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(row0)),
			gpu.ArgValue(int32(stride)), Arg(invB), r.sinkArg(l), blocksArg, gpu.ArgValue(int32(nBlocks))); e != nil {
			return e
		}
		if e := r.launch(r.psVsum, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((Ly.hd + dTile - 1) / dTile), GridZ: uint32(n), BlockX: dTile, BlockY: 1, BlockZ: 1},
			Arg(scB), Arg(r.kvV(l)), Arg(r.kvT(l)), Arg(invB), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(startPos)), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(row0)), gpu.ArgValue(int32(stride)), Arg(cctxB), blocksArg, gpu.ArgValue(int32(nBlocks))); e != nil {
			return e
		}
		r.psLaunches++
	}
	return nil
}
