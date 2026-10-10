//go:build cuda

package cuda

import (
	"slices"

	"github.com/townsendmerino/aikit/gpu"
)

// CUDA expert-major MoE prefill (docs/queue-performance.md P20, docs/measurements/p20-expert-locality-2026-09-21.md).
// Generic MoE path only (Ly.isMoE, moeMLPPre/moeMLPPost's shape); gemma4's parallel dense||MoE FFN is
// moe_expert_major_gemma4.go.
//
// Mechanism: prefill.go's per-row MoE FFN loop admits/DMAs a routed expert once per (row, routing rank), re-fetching the
// same expert many times per chunk because only cacheSlots experts fit and rows are visited in positional order. This
// reorders processing (not admission capacity) to group by expert: route every row in the chunk first, bucket (row,
// rank) pairs by expert, then for each distinct expert admit/DMA it once and run every row assigned to it. No new
// GEMV/SwiGLU/down-proj kernel: gemv_w4a8_moe(_wacc) and glu_quant are the per-row path's own kernels (moe.ptx, the
// audited artifact, is untouched), called with a dedicated one-entry "current slot" index instead of the live r.slotIdx.
//
// Bit-identity is required: float addition is not associative, and expert iteration order has nothing to do with any one
// row's rank order. So every (row, rank) expert output goes into a separate rank-indexed scratch (moeScratch[rank],
// written by the unchanged gemv_w4a8_moe_wacc into a buffer that starts at exactly zero, so 0+x is exact), and each row
// is then folded in rank order by topK sequential residual_batched launches on one stream, so launch order is per-row
// term order for every row. This is decoder/mlp.go's moeMLPBatch technique, restated for a stream-ordered GPU.
//
// It refuses (falls through to the per-row path, as moeMLPBatch does) on a shared expert (Ly.hasShared), on any
// per-expert bias table (gpt-oss: its bias lookup is keyed by the live per-token slot index, which this scheme does not
// use the same way and was never verified against), and on gpt-oss's own route kernel.

// prefillExpertMajorEnabled reports whether the expert-major MoE prefill restructuring is on: default on, and
// GOINFER_CUDA_MOE_EXPERT_MAJOR=0 restores the per-row path (an escape hatch and A/B handle, as GOINFER_MOE_EXPERT_MAJOR
// is for the CPU's moeExpertMajor in decoder/mlp.go). The default rests on bit-identity and no measured regression
// (docs/measurements/p20-expert-major-m26-2026-09-21.md).
func (r *cudaResident) prefillExpertMajorEnabled() bool {
	return r.knobValue("GOINFER_CUDA_MOE_EXPERT_MAJOR") != "0"
}

// cudaMoeExpertMajorRuns counts the layers/chunks that actually took this path: a non-vacuity counter, so a silent
// refusal cannot pass as a green (TestMoEExpertMajor_bitIdentical).
var cudaMoeExpertMajorRuns int64

// moeExpertMajorState is the working set of one prefillMoEExpertMajorRun: the per-row quantized activations and routing
// weights, and the rank-indexed accumulators that are folded in rank order.
type moeExpertMajorState struct {
	M, hidden, moeInter, topK int
	mqAll, mScAll, wgtAll     Buffer   // per-row scratch: mqAll[row]=quantized activation, mScAll[row]=its scale, wgtAll[row*topK+j]=routing weight
	scratch                   []Buffer // rank-indexed [M,hidden] accumulators, len topK
	hostIdx                   []uint32 // [M*topK], this chunk's routed expert ids
	emSlot                    Buffer   // 1-element: the CURRENTLY-resident expert's cache slot, for gemv_w4a8_moe's idx[0]
}

// prefillMoEExpertMajorEligible reports whether layer Ly can take the expert-major path at all, checked once per layer
// before the row loop so a decline builds no state.
func (r *cudaResident) prefillMoEExpertMajorEligible(Ly *cudaLayer) bool {
	if !r.prefillExpertMajorEnabled() || !Ly.isMoE || Ly.g4moe || Ly.hasShared || r.gptOssRoute != (Pipeline{}) {
		return false
	}
	if r.expBiasArg(Ly) != (Buffer{}) || r.expDownBiasArg(Ly) != (Buffer{}) {
		return false // gpt-oss per-expert bias table — not verified against the one-entry slot scheme below
	}
	if !r.cacheExperts || Ly.expCache == nil {
		return false // the mechanism this exists for (bounded slots forcing repeat DMA) requires C′ to be on
	}
	return true
}

// prefillMoEExpertMajorRun replaces the per-row `for m := range M { segBFFN; layerTail }` loop for one MoE layer of one
// prefill chunk: route every row, bucket by expert, run the expert-grouped compute into rank-indexed scratch, and fold
// each row in rank order into xB. Callers must have confirmed prefillMoEExpertMajorEligible(Ly). Once called it returns
// an error rather than declining (loadRoutedExperts's propagate-don't-swallow convention): a mid-chunk failure has
// nowhere safe to fall back to without redoing routing.
func (r *cudaResident) prefillMoEExpertMajorRun(ctx interface{ Err() error }, Ly *cudaLayer, xB Buffer, M, hidden int) error {
	st := &moeExpertMajorState{M: M, hidden: hidden, moeInter: r.moeInter, topK: r.topK}
	var scratch []Buffer
	af := func(n int) Buffer { b := r.af(n); scratch = append(scratch, b); return b }
	ai := func(n int) Buffer { b := r.ai(n); scratch = append(scratch, b); return b }
	au := func(n int) Buffer { b := r.au32(n); scratch = append(scratch, b); return b }
	defer func() {
		for _, b := range scratch {
			r.dev.ReleaseBuf(b)
		}
	}()

	st.mqAll = ai(M * hidden / 4)
	st.mScAll = af(M)
	st.wgtAll = af(M * r.topK)
	idxAll := au(M * r.topK) // the route kernel writes each row's expert ids here; read back ONCE after the loop (audit R-22)
	st.emSlot = r.au32(1)
	st.scratch = make([]Buffer, r.topK)
	for j := range st.scratch {
		st.scratch[j] = af(M * hidden)
		// The expert kernel accumulates into these, so they must start at zero. Zeroed on r.stream, where every kernel below
		// launches: no host copy of a zero slice and no context sync, and the buffer is fresh, so no earlier reader needs
		// waiting for.
		if e := r.stream.ZeroAsync(st.scratch[j], M*hidden*4); e != nil {
			return e
		}
	}
	st.hostIdx = make([]uint32, M*r.topK)

	// --- Phase 1: route every row, storing its quantized activation directly into mqAll/mScAll (r.rms and the
	// router GEMV both take explicit destination buffers, so this is a plain retarget, no extra copy). ---
	for m := range M {
		if e := ctx.Err(); e != nil {
			return e
		}
		xm := xB.At(m * hidden * 4)
		qOff, sOff := st.mqAll.At(m*hidden/4*4), st.mScAll.At(m*4)
		if e := r.rms(xm, Ly.postNorm, qOff, sOff); e != nil {
			return e
		}
		if e := r.launch(r.fRouterGemv, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(Ly.routerW), Arg(qOff), Arg(sOff), gpu.ArgValue(int32(r.nE)),
			gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits)); e != nil {
			return e
		}
		if e := r.launch(r.fRoute, onecfg(1, 0),
			Arg(r.rLogits), Arg(Ly.routerB), Arg(idxAll.At(m*r.topK*4)), Arg(st.wgtAll.At(m*r.topK*4)),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
			gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
			gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup))); e != nil {
			return e
		}
		// No per-row Sync or Download: each row's route output lands in its own slice of idxAll/wgtAll, and the loop is launches
		// only, in stream order. r.rLogits, the one scratch every row reuses, is written by row m's router GEMV and read by row
		// m's route kernel before row m+1's GEMV overwrites it. The weights never come to the host; it only buckets by expert
		// id.
	}
	if e := r.stream.Sync(); e != nil {
		return e
	}
	if e := gpu.Download(idxAll, st.hostIdx); e != nil {
		return e
	}

	// --- Phase 2: bucket (row, rank) by expert id, ascending — any fixed order works, since fold order below is
	// controlled entirely by RANK, not by this iteration order. ---
	type slot struct{ row, rank int }
	byExpert := map[uint32][]slot{}
	for m := range M {
		for j := 0; j < r.topK; j++ {
			e := st.hostIdx[m*r.topK+j]
			byExpert[e] = append(byExpert[e], slot{m, j})
		}
	}
	experts := make([]uint32, 0, len(byExpert))
	for e := range byExpert {
		experts = append(experts, e)
	}
	slices.Sort(experts)

	c := Ly.expCache
	gu := 2 * r.moeInter
	for _, e := range experts {
		if e := ctx.Err(); e != nil {
			return e
		}
		slotNo, hit := c.admit(e)
		if !hit {
			r.expBatch = r.expBatch[:0]
			r.appendExpertSlot(&Ly.expGU, int(e), slotNo)
			r.appendExpertSlot(&Ly.expDown, int(e), slotNo)
			if ue := gpu.UploadBatch(r.expBatch); ue != nil {
				c.unadmit(slotNo, e)
				return ue
			}
		}
		if ue := gpu.Upload(st.emSlot, []uint32{uint32(slotNo)}); ue != nil {
			return ue
		}
		for _, s := range byExpert[e] {
			qOff, sOff := st.mqAll.At(s.row*hidden/4*4), st.mScAll.At(s.row*4)
			if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(Ly.expGU.W), Arg(qOff), Arg(Ly.expGU.ws16), Arg(sOff), Arg(st.emSlot), gpu.ArgValue(int32(0)),
				gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
				Arg(r.moeGU)); e != nil {
				return e
			}
			if e := r.launchGluSplitExpert(r.moeGU, r.moeInter, r.moeQ, r.moeSc, r.moeScr, Buffer{}, 0); e != nil {
				return e
			}
			wgtOff := st.wgtAll.At((s.row*r.topK + s.rank) * 4)
			dstOff := st.scratch[s.rank].At(s.row * hidden * 4)
			if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc), Arg(st.emSlot), Arg(wgtOff),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.hidden)),
				gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)), Arg(dstOff)); e != nil {
				return e
			}
		}
	}

	// --- Phase 3: fold each row in RANK order (topK sequential launches, each covering all M rows — same stream,
	// so launch order IS per-row term order for every row, matching the per-row path's own j=0..topK-1 fold). ---
	for j := 0; j < r.topK; j++ {
		if e := r.launch(r.bRes, LaunchConfig{GridX: uint32((M*hidden + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(xB), Arg(st.scratch[j]), gpu.ArgValue(int32(M*hidden))); e != nil {
			return e
		}
	}
	cudaMoeExpertMajorRuns++
	return nil
}
