//go:build cuda

package cuda

import (
	"sort"

	"github.com/townsendmerino/aikit/gpu"
)

// R11/P20 — CUDA expert-major MoE prefill (docs/queue-performance.md P20, docs/tasks/red-october.md R11(b),
// docs/measurements/p20-expert-locality-2026-09-21.md). GENERIC MoE PATH ONLY (Ly.isMoE, moeMLPPre/moeMLPPost's
// own shape) — gemma4's parallel dense‖MoE FFN (gemma4MoeMLPPre/Post) is NOT covered here; that is a separate,
// structurally similar extension, not attempted in this pass (M26, the model the locality measurement used,
// therefore does NOT yet benefit from this — a real, named remainder, not an oversight).
//
// MECHANISM: cuda/prefill.go's per-row MoE FFN loop admits/DMAs a routed expert once per (row, routing rank),
// which the P20 locality measurement found re-fetches the SAME expert a mean of ~69x per 512-row chunk on M26
// (a different model's C′ config, but the mechanism is model-independent) because only cacheSlots (as few as 10
// on a tight card) can be resident at once and rows are visited in POSITIONAL order, not grouped by expert. This
// reorders PROCESSING (not admission capacity) to group by expert: route every row in the chunk first, bucket
// (row, rank) pairs by expert, then for each DISTINCT expert admit/DMA it ONCE and run every row assigned to it
// before moving on. No new GEMV/SwiGLU/down-proj kernel: gemv_w4a8_moe(_wacc) and glu_quant are the SAME kernels
// the per-row path already uses (moe.ptx is the audited 12.6.85 artifact and is NOT touched), called with a
// dedicated one-entry "current slot" index instead of the live per-token r.slotIdx.
//
// BIT-IDENTITY (required, not optional — the brief's own words): float addition is not associative, so folding
// rank-major (today: for each row, rank 0..topK-1 in order) must produce the IDENTICAL term sequence as folding
// expert-major would if experts happened to be visited in a different order per row (they do — expert iteration
// order has nothing to do with any one row's own rank order). The fix, mirrored EXACTLY from the CPU precedent
// (decoder/mlp.go's moeMLPBatch, P18): compute every (row, rank) expert output into a SEPARATE, rank-indexed
// scratch buffer (moeScratch[rank], written via the UNCHANGED gemv_w4a8_moe_wacc kernel into a buffer that
// starts at exactly zero — 0+x is exact in IEEE754, so this is not merely close to a plain write, it IS one),
// then fold each row in RANK order at the end via topK sequential residual_batched launches (rank 0, then 1, ...
// — same CUDA stream, so launch order IS per-row term order, for every row, regardless of what order the
// experts were computed in). This is the CPU's own documented technique, restated for a stream-ordered GPU
// instead of a single-threaded Go loop.
//
// REFUSES (falls through to the per-row path, exactly as CPU's moeMLPBatch does) on: a shared expert
// (Ly.hasShared — a different combine shape, not attempted here, matching CPU's own refusal), any per-expert
// bias table (gpt-oss; its bias lookup is keyed by the LIVE per-token slot index this rewrite does not use in
// the same way, and it was never verified against this scheme), and gpt-oss's own route kernel.

// prefillExpertMajorEnabled reports whether the expert-major MoE prefill restructuring is on.
// DEFAULT ON since 2026-09-21 (docs/measurements/p20-expert-major-m26-2026-09-21.md — the gemma4
// extension, cuda/moe_expert_major_gemma4.go, measured 2.66x/2.50x/2.39x/2.26x at M=512/2048/4096/8012
// on the real M26, sequential control unmoved within 0.3% noise), mirroring the CPU precedent this
// build mirrors throughout (decoder/mlp.go's moeExpertMajor, P18: "GOINFER_MOE_EXPERT_MAJOR=0
// restores the per-row path... an escape hatch and an A/B handle, not a user setting"). The generic
// (non-gemma4) path's own measured win (Mellum2, 3.5-4.3%) is real but small — it rides the same
// default because it is bit-identical and never measured a regression, not because it was the case
// this default was chosen for.
// GOINFER_CUDA_MOE_EXPERT_MAJOR=0 restores the per-row path.
func (r *cudaResident) prefillExpertMajorEnabled() bool {
	return r.knobValue("GOINFER_CUDA_MOE_EXPERT_MAJOR") != "0"
}

// moeExpertMajorRuns counts chunks/layers that actually took this path — a non-vacuity counter, the same
// discipline TestMoEExpertMajor_bitIdentical's CPU precedent uses, so a silent refusal cannot pass as a green.
var cudaMoeExpertMajorRuns int64

// prefillMoEExpertMajorRow is the CALLER'S per-row hook, replacing the ordinary segBFFN+layerTail pair for one
// row of the batched MoE prefill loop — but it does NOT run the FFN itself; it only ROUTES this row and stashes
// its quantized activation, deferring compute to prefillMoEExpertMajorFlush once every row in the chunk has been
// routed. Declines (returns ok=false) on the FIRST row it sees anything it does not handle, so the caller can
// fall back to the ordinary per-row path for the WHOLE chunk before any expert-major state exists.
type moeExpertMajorState struct {
	M, hidden, moeInter, topK int
	mqAll, mScAll, wgtAll     Buffer   // per-row scratch: mqAll[row]=quantized activation, mScAll[row]=its scale, wgtAll[row*topK+j]=routing weight
	scratch                   []Buffer // rank-indexed [M,hidden] accumulators, len topK
	hostIdx                   []uint32 // [M*topK], this chunk's routed expert ids
	hostWgt                   []float32
	emSlot                    Buffer // 1-element: the CURRENTLY-resident expert's cache slot, for gemv_w4a8_moe's idx[0]
}

// prefillMoEExpertMajorEligible reports whether layer Ly can take the expert-major path at all — checked ONCE
// per layer before the chunk's row loop starts, so a decline costs nothing (no state is built).
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

// prefillMoEExpertMajorRun replaces the per-row `for m := range M { segBFFN; layerTail }` loop for one MoE
// layer of one prefill chunk, end to end: route every row, bucket by expert, run the expert-grouped compute
// into rank-indexed scratch, then fold each row in rank order into xB. Callers must have already confirmed
// prefillMoEExpertMajorEligible(Ly); this returns an error rather than declining once called, matching
// loadRoutedExperts's own "propagate, don't swallow" convention — a mid-chunk failure has nowhere safe to
// fall back to without redoing routing, and every allocation/launch here mirrors ones the per-row path already
// makes successfully for the same layer.
func (r *cudaResident) prefillMoEExpertMajorRun(ctx interface{ Err() error }, Ly *cudaLayer, xB Buffer, M, hidden int) error {
	st := &moeExpertMajorState{M: M, hidden: hidden, moeInter: r.moeInter, topK: r.topK}
	var scratch []Buffer
	af := func(n int) Buffer { b := r.af(n); scratch = append(scratch, b); return b }
	ai := func(n int) Buffer { b := r.ai(n); scratch = append(scratch, b); return b }
	defer func() {
		for _, b := range scratch {
			r.dev.ReleaseBuf(b)
		}
	}()

	st.mqAll = ai(M * hidden / 4)
	st.mScAll = af(M)
	st.wgtAll = af(M * r.topK)
	st.emSlot = r.au32(1)
	st.scratch = make([]Buffer, r.topK)
	zero := make([]float32, M*hidden)
	for j := range st.scratch {
		st.scratch[j] = af(M * hidden)
		if e := gpu.Upload(st.scratch[j], zero); e != nil {
			return e
		}
	}
	st.hostIdx = make([]uint32, M*r.topK)
	st.hostWgt = make([]float32, M*r.topK)

	// --- Phase 1: route every row, storing its quantized activation directly into mqAll/mScAll (r.rms and the
	// router GEMV both take explicit destination buffers, so this is a plain retarget, no extra copy). ---
	for m := 0; m < M; m++ {
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
			Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
			gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
			gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup))); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.Download(r.rIdx, st.hostIdx[m*r.topK:(m+1)*r.topK]); e != nil {
			return e
		}
		if e := gpu.Download(r.rWgt, st.hostWgt[m*r.topK:(m+1)*r.topK]); e != nil {
			return e
		}
	}
	if e := gpu.Upload(st.wgtAll, st.hostWgt); e != nil {
		return e
	}

	// --- Phase 2: bucket (row, rank) by expert id, ascending — any fixed order works, since fold order below is
	// controlled entirely by RANK, not by this iteration order. ---
	type slot struct{ row, rank int }
	byExpert := map[uint32][]slot{}
	for m := 0; m < M; m++ {
		for j := 0; j < r.topK; j++ {
			e := st.hostIdx[m*r.topK+j]
			byExpert[e] = append(byExpert[e], slot{m, j})
		}
	}
	experts := make([]uint32, 0, len(byExpert))
	for e := range byExpert {
		experts = append(experts, e)
	}
	sort.Slice(experts, func(i, j int) bool { return experts[i] < experts[j] })

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
