//go:build cuda

package cuda

import (
	"sort"

	"github.com/townsendmerino/aikit/gpu"
)

// Gemma-4 extension of cuda/moe_expert_major.go's expert-major restructuring (R11/P20, this is the
// item docs/queue-performance.md's P20 entry names as "OPEN, ORPHANED: the gemma4-specific extension
// this needs to actually reach M26" — see docs/measurements/p20-expert-major-2026-09-21.md).
//
// gemma4MoeMLPPre/Post (cuda/resident.go) is a PARALLEL dense‖MoE FFN, structurally different from
// the generic moeMLPPre/Post the other file covers:
//
//	x1 = postFFNNorm1( mlpDown( geluTanh(mlpGate·xd)·(mlpUp·xd) ) )      xd = preFFNNorm(h)   [dense]
//	rn = rmsnorm_nw(h); logits = RouterProjScaled·rn; idx,wgt = route    wgt *= perExpertScale[idx]
//	x2 = postFFNNorm2( Σ_j wgt[j]·expertDown_j(geluTanh(gu_j)·up_j) )    xe = preFFNNorm2(h)  [MoE]
//	h  = (h + postFFNNorm(x1 + x2)) · layerScalar                                             [join]
//
// Same mechanism as the generic file (route every row first, bucket by expert, admit/DMA each
// distinct expert once, fold each row's topK contributions in RANK ORDER via sequential
// residual_batched launches so bit-identity holds by the same argument — see that file's header for
// the full proof), but the EXPERT LOOP here accumulates into a PER-ROW g4x2 buffer (not directly
// into the residual xB), because gemma4's join (normF32(g4x2) -> x1+=x2 -> normF32 -> h+=comb ->
// *layerScalar) still has to run per row afterward. The dense branch (x1) and the router's
// pre-fold activation (xe) are computed in the SAME pre-pass, per row, into their own per-row
// scratch, since both are needed again only after the (deferred) expert-major fold completes.

// prefillGemma4ExpertMajorEligible mirrors prefillMoEExpertMajorEligible's shape for g4moe layers.
// No shared-expert / gpt-oss-bias exclusion needed here — gemma4's join has neither.
func (r *cudaResident) prefillGemma4ExpertMajorEligible(Ly *cudaLayer) bool {
	return prefillExpertMajorEnabled() && Ly.g4moe && r.cacheExperts && Ly.expCache != nil
}

// prefillGemma4ExpertMajorRun replaces the per-row `for m := range M { segBFFN; layerTail }` loop
// for one gemma4 MoE layer of one prefill chunk. See cuda/moe_expert_major.go's header for the
// bit-identity argument this reuses unchanged (fold in rank order, one sequential launch per rank,
// same CUDA stream).
func (r *cudaResident) prefillGemma4ExpertMajorRun(ctx interface{ Err() error }, Ly *cudaLayer, xB Buffer, M, hidden int) error {
	var scratch []Buffer
	af := func(n int) Buffer { b := r.af(n); scratch = append(scratch, b); return b }
	ai := func(n int) Buffer { b := r.ai(n); scratch = append(scratch, b); return b }
	defer func() {
		for _, b := range scratch {
			r.dev.ReleaseBuf(b)
		}
	}()

	g4x1All := af(M * hidden) // dense branch output, one row per token, held until the join
	mqAll := ai(M * hidden / 4)
	mScAll := af(M)
	wgtAll := af(M * r.topK)
	emSlot := r.au32(1)
	rankScratch := make([]Buffer, r.topK) // this layer's Σ_j wgt[j]·expertDown_j(...), per rank
	zero := make([]float32, M*hidden)
	for j := range rankScratch {
		rankScratch[j] = af(M * hidden)
		if e := gpu.Upload(rankScratch[j], zero); e != nil {
			return e
		}
	}
	hostIdx := make([]uint32, M*r.topK)
	hostWgt := make([]float32, M*r.topK)
	nullBias := ArgNull()

	// --- Phase 1: per row, the dense branch (-> g4x1All) and the router+MoE-input-norm (-> hostIdx/
	// hostWgt, mqAll/mScAll). Identical arithmetic to gemma4MoeMLPPre, only the destinations differ. ---
	for m := 0; m < M; m++ {
		if e := ctx.Err(); e != nil {
			return e
		}
		xm := xB.At(m * hidden * 4)
		if e := r.rms(xm, Ly.g4preFFN, r.mq, r.mSc); e != nil {
			return e
		}
		if e := r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0); e != nil {
			return e
		}
		if e := r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0); e != nil {
			return e
		}
		if e := r.launch(r.fSw, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
			gpu.ArgValue(int32(r.inter)), gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
			return e
		}
		x1m := g4x1All.At(m * hidden * 4)
		if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, x1m, 0); e != nil {
			return e
		}
		if e := r.normF32(x1m, Ly.g4postFFN1); e != nil {
			return e
		}

		if e := r.launch(r.fRmsNW, onecfg(256, 256*4), Arg(xm), Arg(r.g4rn), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps)); e != nil {
			return e
		}
		if e := r.launch(r.fRouterF32, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(Ly.routerW), Arg(r.g4rn), gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits)); e != nil {
			return e
		}
		if e := r.launch(r.fRoute, onecfg(1, 0), Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)),
			gpu.ArgValue(float32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
			return e
		}
		if e := r.launch(r.fScaleWgt, LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(r.topK), BlockY: 1, BlockZ: 1},
			Arg(r.rWgt), Arg(r.rIdx), Arg(Ly.perExpertScaleB), gpu.ArgValue(int32(r.topK))); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.Download(r.rIdx, hostIdx[m*r.topK:(m+1)*r.topK]); e != nil {
			return e
		}
		if e := gpu.Download(r.rWgt, hostWgt[m*r.topK:(m+1)*r.topK]); e != nil {
			return e
		}
		// xe = preFFNNorm2(h), the MoE branch's own input norm — straight into this row's slot.
		if e := r.rms(xm, Ly.g4preFFN2, mqAll.At(m*hidden/4*4), mScAll.At(m*4)); e != nil {
			return e
		}
	}
	if e := gpu.Upload(wgtAll, hostWgt); e != nil {
		return e
	}

	// --- Phase 2: bucket (row, rank) by expert id; admit/DMA each distinct expert once; run every
	// row assigned to it into rankScratch[rank]. Identical shape to moe_expert_major.go Phase 2. ---
	type slot struct{ row, rank int }
	byExpert := map[uint32][]slot{}
	for m := 0; m < M; m++ {
		for j := 0; j < r.topK; j++ {
			byExpert[hostIdx[m*r.topK+j]] = append(byExpert[hostIdx[m*r.topK+j]], slot{m, j})
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
		if ue := gpu.Upload(emSlot, []uint32{uint32(slotNo)}); ue != nil {
			return ue
		}
		for _, s := range byExpert[e] {
			qOff, sOff := mqAll.At(s.row*hidden/4*4), mScAll.At(s.row*4)
			if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(Ly.expGU.W), Arg(qOff), Arg(Ly.expGU.ws16), Arg(sOff), Arg(emSlot), gpu.ArgValue(int32(0)),
				gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
				Arg(r.moeGU)); e != nil {
				return e
			}
			if e := r.launchGluSplitExpert(r.moeGU, r.moeInter, r.moeQ, r.moeSc, r.moeScr, Buffer{}, 0); e != nil {
				return e
			}
			wgtOff := wgtAll.At((s.row*r.topK + s.rank) * 4)
			dstOff := rankScratch[s.rank].At(s.row * hidden * 4)
			if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc), Arg(emSlot), Arg(wgtOff),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.hidden)),
				gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)), Arg(dstOff)); e != nil {
				return e
			}
		}
	}

	// --- Phase 3: fold every row's topK contributions into g4x2All, IN RANK ORDER. ---
	g4x2All := af(M * hidden)
	if e := gpu.Upload(g4x2All, zero); e != nil {
		return e
	}
	for j := 0; j < r.topK; j++ {
		if e := r.launch(r.bRes, LaunchConfig{GridX: uint32((M*hidden + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(g4x2All), Arg(rankScratch[j]), gpu.ArgValue(int32(M*hidden))); e != nil {
			return e
		}
	}

	// --- Phase 4: per row, the rest of gemma4MoeMLPPost's join — cheap (a handful of O(hidden)
	// kernels), not the DMA-heavy part, so a plain per-row loop over the batched buffers is fine. ---
	for m := 0; m < M; m++ {
		if e := ctx.Err(); e != nil {
			return e
		}
		xm := xB.At(m * hidden * 4)
		x1m := g4x1All.At(m * hidden * 4)
		x2m := g4x2All.At(m * hidden * 4)
		if e := r.normF32(x2m, Ly.g4postFFN2); e != nil {
			return e
		}
		if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x1m), Arg(x2m), gpu.ArgValue(int32(r.hidden))); e != nil { // x1 += x2
			return e
		}
		if e := r.normF32(x1m, Ly.g4postFFN); e != nil { // x1 = postFFNNorm(x1 + x2)
			return e
		}
		if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(xm), Arg(x1m), gpu.ArgValue(int32(r.hidden))); e != nil { // x = h + comb
			return e
		}
		if e := r.launch(r.fScaleVec, g1cfg(r.hidden, 256), Arg(xm), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(r.hidden))); e != nil {
			return e
		}
	}
	cudaMoeExpertMajorRuns++
	return nil
}
