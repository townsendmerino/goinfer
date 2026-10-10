package decoder

import (
	"github.com/townsendmerino/aikit/linalg"
)

// decodeScratch holds the per-stream float buffers the single-token forward reuses across decode steps, so steady-state decode
// allocates almost nothing per layer. One lives on each KVCache; a cache is one generation stream, so the buffers are never shared
// concurrently.
//
// Liveness (why these few buffers suffice): within a layer, `norm` is the normalized input to attention, then re-derived as the
// input to the MLP (sequential, no overlap); `sub` is the attention output added to the residual, then the MLP output (also
// sequential). The residual `h` persists across the layer loop; the rest are local to one matmul/attention.
type decodeScratch struct {
	knobs    *knobSet  // the owning model's knob snapshot (from its Architecture); nil = live environment
	h        []float32 // [hidden] residual stream (overwritten by the embedding each step)
	norm     []float32 // [hidden] normalized layer input (pre-attn, then pre-mlp)
	sub      []float32 // [hidden] attention output, then MLP output (added to h)
	sub2     []float32 // [hidden] parallel-block (Cohere) MLP output, held while attn output lives in sub
	q        []float32 // [qDim]
	k, v     []float32 // [kvDim]
	ctx      []float32 // [qDim] attention context before the O-projection
	gate     []float32 // [inter] gatedMLP gate
	up       []float32 // [inter] gatedMLP up
	scores   []float32 // [>=nKeys] attention scores; grown on demand as context extends
	attnGate []float32 // [>=nH or nH*hd] Laguna attention output-gate scratch (g_proj result, pre-softplus); nil for every other family
	logits   []float32 // [vocab]

	// headPool is per-head worker scratch for single-token attention, which routes through attendBatchedHeads (K=1) with the same
	// acc64 matmul as batched prefill, so the discrete MoE router never sees a prefill/decode numerical discontinuity. It is a pool,
	// not a single set, so concurrent per-head workers never share mutable scratch.
	headPool []headWorkerScratch

	// localK/localV assemble a local (ring) layer's [base, pos] read window for K=1 decode (resident ring history plus this token's
	// K/V); at most W rows.
	localK, localV []float32

	ws        *linalg.Workspace // W8A8/W4A8 activation-quant scratch (zero-alloc Into/Batch)
	qkvOps    [3]linalg.W8A8Op  // reused q/k/v batch ops
	gateUpOps [2]linalg.W8A8Op  // reused gate/up batch ops
	qkvOpsW4  [3]linalg.W4A8Op  // reused q/k/v W4A8 batch ops (audit R-06)
	guOpsW4   [2]linalg.W4A8Op  // reused gate/up W4A8 batch ops (audit R-06)
	fusedPair fusedPairScratch  // per-worker serial Workspaces for the fused gate+up+SwiGLU path (cpu_gateup_fused.go)

	loraTmp []float32 // [>=r] compute-time LoRA A·x scratch (#7); nil until an adapter is active

	// MoE decode scratch (moeMLP): backs the router-logits/accumulator/expert-gate-up buffers moeMLP would otherwise allocate every
	// call, layer after layer (nE+2·hidden+2·sc floats), the bulk of MoE decode's per-token allocation. nil for non-MoE
	// architectures; moeMLP allocates when passed a nil scr (the batched-prefill call site, which has no cache.scr).
	moeLogits         []float32 // [NumExperts]
	moeOut, moeExpOut []float32 // [HiddenDim]
	// moeHQ is moeMLP's per-layer quantization of h, shared by the layer's experts; its buffers are reused.
	moeHQ w4a8Act
	// hq is a decode layer's one quantization of its normed row for the projections that share it (q/k/v, then gate/up). Rewritten by
	// each use before it is read.
	hq w4a8Act
	// gateUpQ is gatedMLPFusedGateUp's one quantization of h, read by every worker.
	gateUpQ        linalg.ActQ
	moeGate, moeUp []float32 // [max(IntermediateDim, SharedIntermediateDim)]
}

// loraBuf returns a length-r scratch for the compute-time LoRA A·x intermediate,
// growing the (per-stream, reused) backing array once on demand.
func (s *decodeScratch) loraBuf(r int) []float32 {
	if cap(s.loraTmp) < r {
		s.loraTmp = make([]float32, r)
	}
	return s.loraTmp[:r]
}

func newDecodeScratch(a *Architecture) *decodeScratch {
	// maxHeads, not NumHeads: a family with per-layer query heads (Laguna) must size
	// q/ctx for its WIDEST layer, since one scratch is reused across every layer.
	qDim, kvDim := a.maxHeads()*a.HeadDim, a.NumKVHeads*a.HeadDim
	// aikit's opt-in worker pool (Workspace.SetWorkers) is deliberately not used: it measured neutral-to-slightly-slower than the
	// spawn path (the batch=1 fork/join cost is a floor, not pool-fixable).
	//
	// The W8A8 (int8) decode matmuls run through this Workspace (matmulInto), so its per-Workspace threshold, not the process
	// global, is what makes int8 decode parallelize. Setting it here means every decode stream (library Load, serve, tests) gets it
	// automatically and race-free, the way the int4 path self-configures per call (weightmat.go int4ParThreshold). See tune.go
	// DefaultDecodeParallelThreshold.
	ws := &linalg.Workspace{}
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	s := &decodeScratch{
		knobs:  a.knobs,
		h:      make([]float32, a.HiddenDim),
		norm:   make([]float32, a.HiddenDim),
		sub:    make([]float32, a.HiddenDim),
		sub2:   make([]float32, a.HiddenDim),
		q:      make([]float32, qDim),
		k:      make([]float32, kvDim),
		v:      make([]float32, kvDim),
		ctx:    make([]float32, qDim),
		gate:   make([]float32, a.IntermediateDim),
		up:     make([]float32, a.IntermediateDim),
		logits: make([]float32, a.VocabSize),
		ws:     ws,
	}
	if a.MoE != nil {
		sc := max(a.MoE.SharedIntermediateDim, a.MoE.IntermediateDim)
		s.moeLogits = make([]float32, a.MoE.NumExperts)
		s.moeOut = make([]float32, a.HiddenDim)
		s.moeExpOut = make([]float32, a.HiddenDim)
		s.moeGate = make([]float32, sc)
		s.moeUp = make([]float32, sc)
	}
	return s
}

// scoresBuf returns a length-n scores buffer, reusing the backing array when it is large enough. It grows with headroom
// (growCap): n is the key count, which rises by one a token, so growing to exactly n would reallocate and re-zero the buffer every
// token. The caller writes all n scores before reading any, so the spare capacity's contents never matter.
func (s *decodeScratch) scoresBuf(n int) []float32 {
	if cap(s.scores) < n {
		s.scores = make([]float32, growCap(cap(s.scores), n))
	}
	return s.scores[:n]
}

// growCap is the capacity a per-token buffer grows to when it needs n: at least n, and at least double the old capacity, so a
// buffer that grows by one row a token reallocates O(log n) times instead of every token.
func growCap(old, n int) int {
	return max(n, 2*old, 256)
}

// ctxBuf returns the attention context buffer, length n, from the scratch, for the own-forward families' per-layer attention;
// attendQuery clears it before accumulating, so it need not be zeroed. The caller consumes it (the o-projection) before the next
// layer asks again.
func (s *decodeScratch) ctxBuf(n int) []float32 {
	if len(s.ctx) < n {
		s.ctx = make([]float32, n)
	}
	return s.ctx[:n]
}

// gateBuf returns a length-n buffer for Laguna's attention output gate (the
// g_proj result, before softplus), grown once on demand. n is nH or nH*hd
// depending on the gate's granularity, so it is sized from the weight's row
// count rather than assumed.
func (s *decodeScratch) gateBuf(n int) []float32 {
	if cap(s.attnGate) < n {
		s.attnGate = make([]float32, n)
	}
	return s.attnGate[:n]
}

// headWorkerScratch holds one worker's per-head attention scratch: qh/scores/ch (+ avAcc) for the acc64 path's independent
// per-head compute, and kh/vt for the f32 path's per-kvh K/V gather. The f32 path is the default for prefill above
// fastAttnMinPrompt, so kh/vt are touched on every prefill of a real prompt, each worker gathering into its own pair; the budget
// in prefillAttnWorkersK charges 2*nKeys*hd per slot for them.
type headWorkerScratch struct {
	qh, kh, vt, scores, ch []float32
	avAcc                  []float64
	// groupScores/groupCtx/groupAvAcc are the grouped-kernel scratch: one MatmulQKAcc64Group/MatmulAVAcc64Group call covers
	// attnGroupedNEONSize query heads, so they are sized ×attnGroupedNEONSize the single-head scores/ch/avAcc (see headWorkerPool).
	// In attendGroupedLayer they are each worker's temp buffer for its key-range (QK) or dim-range (AV) slice; groupScoresCombined,
	// used only on pool[0], is the shared full-width [group,nKeys] buffer each worker's slice is copied into at its own disjoint
	// offset (a copy, never a sum, so no float add is reassociated). The context needs no combined buffer: each worker's dim slice is
	// scattered straight into ctx.
	groupScores, groupCtx []float32
	groupAvAcc            []float64
	groupScoresCombined   []float32
	// mmWS is this slot's private matmul Workspace, threshold pinned so high that MatmulBT through it is always serial. The f32
	// attention path fans out over query heads and MatmulBT also fans out over its output columns; nesting them oversubscribes (6 head
	// workers each spawning GOMAXPROCS column goroutines), so the head-parallel arm takes its matmul through here and keeps the
	// fan-out at one level. Numerically inert: MatmulBT's per-element result is independent of fan-out width by contract (aikit
	// TestParallelWidth_bitIdentical), so both arms share one golden.
	mmWS *linalg.Workspace
	// fused is this slot's fused-attention scratch, or nil when the fused schedule is off. Per worker, not shared: each worker gathers
	// its own kv head's V into vBlk, so a shared buffer would be a data race producing silently wrong output rather than a crash.
	fused *fusedScratch
}

// serialMMWorkspace returns a Workspace whose matmuls never fan out. 1<<62 is
// far above any reachable MAC count, so parallelCols always takes its serial
// branch. (SetWorkers(1) would not do: the spawn path still allocates and
// joins one goroutine per call.)
func serialMMWorkspace() *linalg.Workspace {
	ws := &linalg.Workspace{}
	ws.SetThreshold(1 << 62)
	return ws
}

// maxAttnWorkers caps the head-parallel fan-out at the P-core count, not GOMAXPROCS: the Mac's 2 E-cores measured harmful for this
// class of work (docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md).
const maxAttnWorkers = 6

// attnScoreTileBytes is the per-slot budget for the `scores` buffer, which sets the query-row tile. scores is tile*nKeys floats,
// so pinning its bytes makes a slot's cost linear in nKeys rather than quadratic in prompt length, which is what lets the pool
// keep fanning out on a long prompt. 8 MiB is comfortably L2/SLC-resident on the machines this runs on and leaves the tile large
// enough that per-tile call overhead is noise.
const attnScoreTileBytes = 8 << 20

// attnRowTile returns how many QUERY rows one attention pass handles. Never
// below 1, never above K. GOINFER_ATTN_ROW_TILE overrides it — an A/B handle,
// and how the bit-identity test forces the untiled shape to compare against.
func attnRowTile(K, nKeys int) int { return attnRowTileK(nil, K, nKeys) }

// attnRowTileK is attnRowTile reading the given model's knobs (nil = the live environment).
func attnRowTileK(k *knobSet, K, nKeys int) int {
	if n, ok := k.positiveInt(knobAttnRowTile); ok {
		return min(n, K)
	}
	if nKeys < 1 || K < 1 {
		return max(1, K)
	}
	return max(1, min(K, attnScoreTileBytes/(nKeys*4)))
}

// prefillAttnScratchBudget caps the total per-head scratch a single batched (prefill) sweep may hold across its worker pool. The
// dominant slot buffer, `scores`, is K*nKeys floats, quadratic in prompt length, so an unbudgeted fan-out to maxAttnWorkers would
// hold hundreds of MB at a few thousand tokens and GBs at 8k. The worker count therefore falls back toward serial as the prompt
// grows rather than the allocation growing without bound; quadratic memory for a constant-factor speedup is not a trade made
// silently.
const prefillAttnScratchBudget = 256 << 20 // 256 MiB

// prefillAttnWorkers sizes the head-parallel pool for a K>1 (prefill) sweep: the P-core cap, the head count and the scratch
// budget above, whichever binds first. It returns 1 (serial) when even two slots would not fit. GOINFER_PREFILL_ATTN_WORKERS
// overrides it (an A/B handle and escape hatch); 1 forces the serial path.
func prefillAttnWorkers(K, nKeys, hd, nH int) int { return prefillAttnWorkersK(nil, K, nKeys, hd, nH) }

// prefillAttnWorkersK is prefillAttnWorkers reading the given model's knobs (nil = the live environment).
func prefillAttnWorkersK(k *knobSet, K, nKeys, hd, nH int) int {
	if n, ok := k.positiveInt(knobPrefillWorkers); ok {
		return min(n, maxAttnWorkers)
	}
	if nH < 1 {
		return 1
	}
	// Per slot, in bytes, at the tiled size: scores (tile*nKeys) + kh + vt (nKeys*hd each) + qh + ch (tile*hd each), all float32;
	// avAcc is hd float64, noise beside these. Tiling is why this is linear in nKeys, and therefore why long prompts still fan out.
	t := attnRowTileK(k, K, nKeys)
	perSlot := 4 * (t*nKeys + 2*nKeys*hd + 2*t*hd)
	// newHeadWorkerPool allocates a fusedScratch (sBlk+tmp+acc+mRun+lRun+vBlk) alongside the materialized shape above whenever the
	// fused schedule is enabled, so the budget must count it or it oversubscribes (mostly vBlk, hd*nKeys floats, the same order as
	// kh/vt). It is charged whenever fusion could apply, not only when this call will use it, matching the budget's other worst-case
	// assumptions. vt itself goes unused once fusion is active, but dropping it needs the caller's useAcc64/treeMask state, which is
	// unavailable here, and getting it wrong risks a nil vt where fusion does not apply.
	if k.fusedAttention() {
		perSlot += 4 * (t*fusedKeyBlock + 2*t*hd + 2*t + hd*nKeys)
	}
	if perSlot <= 0 {
		return 1
	}
	n := prefillAttnScratchBudget / perSlot
	return max(1, min(n, min(maxAttnWorkers, nH)))
}

// headWorkerPool returns n (capped at maxAttnWorkers) independent per-head scratch sets sized for this call's K/nKeys/hd, growing
// each slot's backing arrays once as decode extends (grow-only, per slot), so concurrent per-head workers never share mutable
// scratch.
func (s *decodeScratch) headWorkerPool(n, K, nKeys, hd int, wantFused, useAcc64 bool) []headWorkerScratch {
	if n > maxAttnWorkers {
		n = maxAttnWorkers
	}
	if n < 1 {
		n = 1
	}
	if len(s.headPool) < n {
		grown := make([]headWorkerScratch, n)
		copy(grown, s.headPool)
		s.headPool = grown
	}
	for i := range s.headPool[:n] {
		p := &s.headPool[i]
		if p.mmWS == nil {
			p.mmWS = serialMMWorkspace()
		}
		// The fused scratch is allocated only when the caller will use it, and grown once. Decode runs with acc64, where
		// attendBatchedHeads computes fusedOK = !useAcc64 && treeMask == nil and never reads ws.fused, so an unconditional per-call
		// allocation churned 4*hd*nKeys bytes per slot per layer per token (zero-filled, never read, linear in context) through the GC.
		if wantFused && s.knobs.fusedAttention() {
			if t := attnRowTileK(s.knobs, K, nKeys); p.fused == nil || !p.fused.fits(t, hd, nKeys) {
				p.fused = newFusedScratch(t, hd, nKeys)
			}
		} else {
			p.fused = nil
		}
		if c := K * hd; cap(p.qh) < c {
			p.qh = make([]float32, c)
			p.ch = make([]float32, c)
		}
		if cap(p.avAcc) < hd {
			p.avAcc = make([]float64, hd)
		}
		// kh/vt are unused whenever useAcc64 is true: the acc64 kernels (MatmulQKAcc64/MatmulAVAcc64) read keys/vals directly with
		// strided addressing, skipping the gather. The check is on useAcc64 itself, not assumed (headWorkerPool's only caller,
		// causalAttention, passes acc64 := true), so a future non-acc64 caller still grows them on its own first call.
		if !useAcc64 {
			if c := nKeys * hd; cap(p.kh) < c {
				g := max(2*cap(p.kh), c) // headroom: nKeys grows by 1 each decode step
				p.kh = make([]float32, g)
				p.vt = make([]float32, g)
			}
		}
		if c := K * nKeys; cap(p.scores) < c {
			g := max(2*cap(p.scores), c)
			p.scores = make([]float32, g)
		}
		if c := K * nKeys * attnGroupedNEONSize; cap(p.groupScores) < c {
			g := max(2*cap(p.groupScores), c)
			p.groupScores = make([]float32, g)
		}
		if c := K * hd * attnGroupedNEONSize; cap(p.groupCtx) < c {
			p.groupCtx = make([]float32, c)
			p.groupAvAcc = make([]float64, c)
		}
		// groupScoresCombined (Arm B) is grown lazily inside
		// attendGroupedLayer itself instead of here: its size depends on nKV,
		// which headWorkerPool's callers don't thread through this signature,
		// and attendGroupedLayer already has it captured. Only pool[0] (the
		// "leader" slot — see headWorkerScratch's own doc) ever uses it.
	}
	return s.headPool[:n]
}

// newHeadWorkerPool builds n (capped at maxAttnWorkers) fresh headWorkerScratch entries sized for one call's K/nKeys/hd: the
// batched (M=K>1) forward's one-shot sibling of decodeScratch.headWorkerPool. That path has no cache.scr to grow and reuse across
// calls, so it allocates per call rather than growing once.
//
// wantFused is the caller's promise that every call using this pool will have fusedOK true (attendBatchedHeads:
// !useAcc64 && cache.treeMask == nil), i.e. useAcc64 and cache.treeMask are fixed for the pool's whole lifetime. Under that
// promise vt and scores are skipped whenever fusion is also enabled (knobSet.fusedAttention): vt is unused once fusion is active,
// and scores (tile*nKeys, larger than vt's nKeys*hd) is unused whenever the fused path is taken, since attendTileFused writes ch
// directly. `fused`'s own allocation stays gated on knobSet.fusedAttention alone, so callers that pass wantFused=false because
// they legitimately mix useAcc64 states against one pool (TestA3FanoutUtilization) keep exercising the same arm. A caller that
// does not hold the promise must pass false: getting it wrong risks a nil-slice access in the prefill hot path every model goes
// through.
func newHeadWorkerPool(n, K, nKeys, hd int, wantFused bool) []headWorkerScratch {
	return newHeadWorkerPoolK(nil, n, K, nKeys, hd, wantFused)
}

// newHeadWorkerPoolK is newHeadWorkerPool reading the given model's knobs (nil = the live environment).
func newHeadWorkerPoolK(k *knobSet, n, K, nKeys, hd int, wantFused bool) []headWorkerScratch {
	if n > maxAttnWorkers {
		n = maxAttnWorkers
	}
	if n < 1 {
		n = 1
	}
	// Slots are sized for one row tile, not the whole prompt: attendOneHead walks its query rows in tiles of exactly this many, so
	// anything larger would be allocated and never touched.
	t := attnRowTileK(k, K, nKeys)
	skipMaterialized := wantFused && k.fusedAttention()
	pool := make([]headWorkerScratch, n)
	for i := range pool {
		p := headWorkerScratch{
			mmWS:  serialMMWorkspace(),
			fused: fusedIfEnabled(k, t, hd, nKeys),
			qh:    make([]float32, t*hd),
			kh:    make([]float32, nKeys*hd),
			ch:    make([]float32, t*hd),
			avAcc: make([]float64, hd),
		}
		if !skipMaterialized {
			p.vt = make([]float32, nKeys*hd)
			p.scores = make([]float32, t*nKeys)
		}
		pool[i] = p
	}
	return pool
}

// localBufs returns two length-n scratch slices (grown once on demand) for assembling a local layer's contiguous read window in
// the K=1 decode path.
func (s *decodeScratch) localBufs(n int) (lk, lv []float32) {
	if cap(s.localK) < n {
		g := max(2*cap(s.localK), n) // headroom: window grows each step; avoid a per-token realloc+zero
		s.localK = make([]float32, g)
		s.localV = make([]float32, g)
	}
	return s.localK[:n], s.localV[:n]
}
