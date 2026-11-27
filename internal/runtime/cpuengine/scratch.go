package cpuengine

import (
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// DecodeScratch holds the per-stream float buffers the single-token forward reuses across decode steps, so steady-state decode
// allocates almost nothing per layer. One lives on each Cache; a cache is one generation stream, so the buffers are never shared
// concurrently.
//
// Liveness (why these few buffers suffice): within a layer, `norm` is the normalized input to attention, then re-derived as the
// input to the MLP (sequential, no overlap); `sub` is the attention output added to the residual, then the MLP output (also
// sequential). The residual `h` persists across the layer loop; the rest are local to one matmul/attention.
type DecodeScratch struct {
	Knobs    KnobSource // the owning model's knob snapshot (from its Architecture); nil = live environment
	h        []float32  // [hidden] residual stream (overwritten by the embedding each step)
	norm     []float32  // [hidden] normalized layer input (pre-attn, then pre-mlp)
	sub      []float32  // [hidden] attention output, then MLP output (added to h)
	sub2     []float32  // [hidden] parallel-block (Cohere) MLP output, held while attn output lives in sub
	q        []float32  // [qDim]
	k, v     []float32  // [kvDim]
	ctx      []float32  // [qDim] attention context before the O-projection
	gate     []float32  // [inter] gatedMLPAct gate
	up       []float32  // [inter] gatedMLPAct up
	Scores   []float32  // [>=nKeys] attention scores; grown on demand as context extends
	attnGate []float32  // [>=nH or nH*hd] Laguna attention output-gate scratch (g_proj result, pre-softplus); nil for every other family
	logits   []float32  // [vocab]
	logitsN  []float32  // [K*vocab] a batched pass's logits, a verify round's or a prefill's last row (Cache.headLogits), grown on demand

	// headPool is per-head worker scratch for single-token attention, which routes through attendBatchedHeads (K=1) with the same
	// acc64 matmul as batched prefill, so the discrete MoE router never sees a prefill/decode numerical discontinuity. It is a pool,
	// not a single set, so concurrent per-head workers never share mutable scratch.
	HeadPool []HeadWorkerScratch

	// localK/localV assemble a local (ring) layer's [base, pos] read window for K=1 decode (resident ring history plus this token's
	// K/V); at most W rows.
	localK, localV []float32

	WS        *linalg.Workspace // W8A8/W4A8 activation-quant scratch (zero-alloc Into/Batch)
	qkvOps    [3]linalg.BatchOp // reused q/k/v batch ops
	gateUpOps [2]linalg.BatchOp // reused gate/up batch ops
	qkvOpsW4  [3]linalg.BatchOp // reused q/k/v W4A8 batch ops
	guOpsW4   [2]linalg.BatchOp // reused gate/up W4A8 batch ops
	gptOss    *gptOssScratch    // gpt-oss MoE scratch (gptOssMoEWith), made on first use
	fusedPair fusedPairScratch  // per-worker serial Workspaces for the fused gate+up+SwiGLU path (cpu_gateup_fused.go)

	loraTmp []float32 // [>=r] compute-time LoRA A·x scratch (#7); nil until an adapter is active

	// MoE decode scratch (moeMLP): backs the router-logits/accumulator/expert-gate-up buffers moeMLP would otherwise allocate every
	// call, layer after layer (nE+2·hidden+2·sc floats), the bulk of MoE decode's per-token allocation. nil for non-MoE
	// architectures; moeMLP allocates when passed a nil scr (the batched-prefill call site, which has no cache.scr).
	moeLogits         []float32 // [NumExperts]
	moeOut, moeExpOut []float32 // [HiddenDim]
	// moeHQ is moeMLP's per-layer quantization of h, shared by the layer's experts; its buffers are reused.
	moeHQ W4A8Act
	// hq is a decode layer's one quantization of its normed row for the projections that share it (q/k/v, then gate/up). Rewritten by
	// each use before it is read.
	hq W4A8Act
	// gateUpQ is gatedMLPFusedGateUp's one quantization of h, read by every worker.
	gateUpQ        linalg.ActQ
	moeGate, moeUp []float32 // [max(IntermediateDim, SharedIntermediateDim)]

	ffnMid []float32 // nonGatedMLPWith's [inter] activation, grown on first use

	mla mlaScratch  // MLA absorb attention's working set; empty until an MLA layer runs
	G4  gemma4Token // Gemma 4's per-position state, reset by newGemma4Token each token; empty for every other family
}

// mlaScratch is mlaAttentionAbsorb's per-stream working set: the roped query, the absorbed query, the per-key scores and
// normalized latents, the rope key, the rank-space value sum and the context. Each buffer is reused across layers and tokens and
// grows to the longest context seen.
type mlaScratch struct {
	qRope, qNopeAbs, scores, cnAll, krj, wsum, ctx []float32
}

// scratchF32 returns (*b)[:n], first growing *b with headroom (growCap) when it holds less, since the key-sized buffers rise by
// one row a token. The contents are stale: the caller writes or clears every element it reads.
func scratchF32(b *[]float32, n int) []float32 {
	if cap(*b) < n {
		*b = make([]float32, growCap(cap(*b), n))
	}
	return (*b)[:n]
}

// loraBuf returns a length-r scratch for the compute-time LoRA A·x intermediate,
// growing the (per-stream, reused) backing array once on demand.
func (s *DecodeScratch) loraBuf(r int) []float32 {
	if cap(s.loraTmp) < r {
		s.loraTmp = make([]float32, r)
	}
	return s.loraTmp[:r]
}

func NewDecodeScratch(a *darch.Architecture, knobs KnobSource) *DecodeScratch {
	// aikit's opt-in worker pool (Workspace.SetWorkers) is deliberately not used: it measured neutral-to-slightly-slower than the
	// spawn path (the batch=1 fork/join cost is a floor, not pool-fixable).
	//
	// The W8A8 (int8) decode matmuls run through this Workspace (matmulInto), so its per-Workspace threshold, not the process
	// global, is what makes int8 decode parallelize. Setting it here means every decode stream (library Load, serve, tests) gets it
	// automatically and race-free, the way the int4 path self-configures per call (weightmat.go int4ParThreshold). See tune.go
	// DefaultDecodeParallelThreshold.
	ws := &linalg.Workspace{}
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	s := &DecodeScratch{Knobs: knobs, WS: ws}
	for _, r := range DecodeScratchRows(a) {
		*s.row(r.Name) = make([]float32, r.Elems)
	}
	return s
}

// scratchRow is one f32 row of a decode stream's fixed scratch: the DecodeScratch field that holds it and its element count.
type scratchRow struct {
	Name  string
	Elems int
}

// DecodeScratchRows is the fixed rows NewDecodeScratch allocates for a, by field name: the one list the constructor allocates from and
// Model.CPUDecodeScratch prices, so the two cannot disagree. maxHeads, not NumHeads, sizes q and ctx (a family with per-layer query
// heads reuses one buffer across layers); the MoE rows exist only for a mixture-of-experts architecture. Buffers grown on demand
// (scores, the batched logits, the per-head pool, the local window, the quantization scratch) are not here.
func DecodeScratchRows(a *darch.Architecture) []scratchRow {
	H, qDim, kvDim := a.HiddenDim, a.MaxHeads()*a.HeadDim, a.NumKVHeads*a.HeadDim
	rows := []scratchRow{{"h", H}, {"norm", H}, {"sub", H}, {"sub2", H}, {"q", qDim}, {"k", kvDim}, {"v", kvDim}, {"ctx", qDim},
		{"gate", a.IntermediateDim}, {"up", a.IntermediateDim}, {"logits", a.VocabSize}}
	if a.MoE != nil {
		sc := max(a.MoE.SharedIntermediateDim, a.MoE.IntermediateDim)
		rows = append(rows, scratchRow{"moeLogits", a.MoE.NumExperts}, scratchRow{"moeOut", H}, scratchRow{"moeExpOut", H},
			scratchRow{"moeGate", sc}, scratchRow{"moeUp", sc})
	}
	return rows
}

// row is the field a scratchRow name stands for.
func (s *DecodeScratch) row(name string) *[]float32 {
	switch name {
	case "h":
		return &s.h
	case "norm":
		return &s.norm
	case "sub":
		return &s.sub
	case "sub2":
		return &s.sub2
	case "q":
		return &s.q
	case "k":
		return &s.k
	case "v":
		return &s.v
	case "ctx":
		return &s.ctx
	case "gate":
		return &s.gate
	case "up":
		return &s.up
	case "logits":
		return &s.logits
	case "moeLogits":
		return &s.moeLogits
	case "moeOut":
		return &s.moeOut
	case "moeExpOut":
		return &s.moeExpOut
	case "moeGate":
		return &s.moeGate
	case "moeUp":
		return &s.moeUp
	}
	panic("decoder: no decode scratch row named " + name)
}

// ScoresBuf returns a length-n scores buffer, reusing the backing array when it is large enough. It grows with headroom
// (growCap): n is the key count, which rises by one a token, so growing to exactly n would reallocate and re-zero the buffer every
// token. The caller writes all n scores before reading any, so the spare capacity's contents never matter.
func (s *DecodeScratch) ScoresBuf(n int) []float32 {
	if cap(s.Scores) < n {
		s.Scores = make([]float32, growCap(cap(s.Scores), n))
	}
	return s.Scores[:n]
}

// growCap is the capacity a per-token buffer grows to when it needs n: at least n, and at least double the old capacity, so a
// buffer that grows by one row a token reallocates O(log n) times instead of every token.
func growCap(old, n int) int {
	return max(n, 2*old, 256)
}

// ctxBuf returns the attention context buffer, length n, from the scratch, for the own-forward families' per-layer attention;
// attendQuery clears it before accumulating, so it need not be zeroed. The caller consumes it (the o-projection) before the next
// layer asks again.
func (s *DecodeScratch) ctxBuf(n int) []float32 {
	if len(s.ctx) < n {
		s.ctx = make([]float32, n)
	}
	return s.ctx[:n]
}

// gateBuf returns a length-n buffer for Laguna's attention output gate (the
// g_proj result, before softplus), grown once on demand. n is nH or nH*hd
// depending on the gate's granularity, so it is sized from the weight's row
// count rather than assumed.
func (s *DecodeScratch) gateBuf(n int) []float32 {
	if cap(s.attnGate) < n {
		s.attnGate = make([]float32, n)
	}
	return s.attnGate[:n]
}

// HeadWorkerScratch holds one worker's per-head attention scratch: qh/scores/ch (+ avAcc) for the acc64 path's independent
// per-head compute, and kh/vt for the f32 path's per-kvh K/V gather. The f32 path is the default for prefill above
// fastAttnMinPrompt, so kh/vt are touched on every prefill of a real prompt, each worker gathering into its own pair; the budget
// in prefillAttnWorkersK charges 2*nKeys*hd per slot for them.
type HeadWorkerScratch struct {
	QH, KH, VT, Scores, CH []float32
	AvAcc                  []float64
	// groupScores/groupCtx/groupAvAcc are the grouped-kernel scratch: one MatmulQKAcc64Group/MatmulAVAcc64Group call covers
	// AttnGroupedNEONSize query heads, so they are sized ×AttnGroupedNEONSize the single-head scores/ch/avAcc (see HeadWorkerPool).
	// In kvAttend.groupedLayer they are each worker's temp buffer for its key-range (QK) or dim-range (AV) slice; groupScoresCombined,
	// used only on pool[0], is the shared full-width [group,nKeys] buffer each worker's slice is copied into at its own disjoint
	// offset (a copy, never a sum, so no float add is reassociated). The context needs no combined buffer: each worker's dim slice is
	// scattered straight into ctx.
	GroupScores, GroupCtx []float32
	GroupAvAcc            []float64
	GroupScoresCombined   []float32
	// mmWS is this slot's private matmul Workspace, threshold pinned so high that MatmulBT through it is always serial. The f32
	// attention path fans out over query heads and MatmulBT also fans out over its output columns; nesting them oversubscribes (6 head
	// workers each spawning GOMAXPROCS column goroutines), so the head-parallel arm takes its matmul through here and keeps the
	// fan-out at one level. Numerically inert: MatmulBT's per-element result is independent of fan-out width by contract (aikit
	// TestParallelWidth_bitIdentical), so both arms share one golden.
	mmWS *linalg.Workspace
	// fused is this slot's fused-attention scratch, or nil when the fused schedule is off. Per worker, not shared: each worker gathers
	// its own kv head's V into vBlk, so a shared buffer would be a data race producing silently wrong output rather than a crash.
	Fused *fusedScratch
}

// SerialMMWorkspace returns a Workspace whose matmuls never fan out. 1<<62 is
// far above any reachable MAC count, so parallelCols always takes its serial
// branch. (SetWorkers(1) would not do: the spawn path still allocates and
// joins one goroutine per call.)
func SerialMMWorkspace() *linalg.Workspace {
	ws := &linalg.Workspace{}
	ws.SetThreshold(1 << 62)
	return ws
}

// MaxAttnWorkers caps the head-parallel fan-out at the P-core count, not GOMAXPROCS: the Mac's 2 E-cores measured harmful for this
// class of work (docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md).
const MaxAttnWorkers = 6

// AttnScoreTileBytes is the per-slot budget for the `scores` buffer, which sets the query-row tile. scores is tile*nKeys floats,
// so pinning its bytes makes a slot's cost linear in nKeys rather than quadratic in prompt length, which is what lets the pool
// keep fanning out on a long prompt. 8 MiB is comfortably L2/SLC-resident on the machines this runs on and leaves the tile large
// enough that per-tile call overhead is noise.
const AttnScoreTileBytes = 8 << 20

// AttnRowTile returns how many QUERY rows one attention pass handles. Never
// below 1, never above K. GOINFER_ATTN_ROW_TILE overrides it — an A/B handle,
// and how the bit-identity test forces the untiled shape to compare against.
func AttnRowTile(K, nKeys int) int { return AttnRowTileK(nil, K, nKeys) }

// AttnRowTileK is attnRowTile reading the given model's knobs (nil = the live environment).
func AttnRowTileK(k KnobSource, K, nKeys int) int {
	if n, ok := knobPositiveInt(k, KnobAttnRowTile); ok {
		return min(n, K)
	}
	if nKeys < 1 || K < 1 {
		return max(1, K)
	}
	return max(1, min(K, AttnScoreTileBytes/(nKeys*4)))
}

// PrefillAttnScratchBudget caps the total per-head scratch a single batched (prefill) sweep may hold across its worker pool. The
// dominant slot buffer, `scores`, is K*nKeys floats, quadratic in prompt length, so an unbudgeted fan-out to MaxAttnWorkers would
// hold hundreds of MB at a few thousand tokens and GBs at 8k. The worker count therefore falls back toward serial as the prompt
// grows rather than the allocation growing without bound; quadratic memory for a constant-factor speedup is not a trade made
// silently.
const PrefillAttnScratchBudget = 256 << 20 // 256 MiB

// PrefillAttnWorkers sizes the head-parallel pool for a K>1 (prefill) sweep: the P-core cap, the head count and the scratch
// budget above, whichever binds first. It returns 1 (serial) when even two slots would not fit. GOINFER_PREFILL_ATTN_WORKERS
// overrides it (an A/B handle and escape hatch); 1 forces the serial path.
func PrefillAttnWorkers(K, nKeys, hd, nH int) int { return PrefillAttnWorkersK(nil, K, nKeys, hd, nH) }

// PrefillAttnWorkersK is prefillAttnWorkers reading the given model's knobs (nil = the live environment).
func PrefillAttnWorkersK(k KnobSource, K, nKeys, hd, nH int) int {
	if n, ok := knobPositiveInt(k, KnobPrefillWorkers); ok {
		return min(n, MaxAttnWorkers)
	}
	if nH < 1 {
		return 1
	}
	// Per slot, in bytes, at the tiled size: scores (tile*nKeys) + kh + vt (nKeys*hd each) + qh + ch (tile*hd each), all float32;
	// avAcc is hd float64, noise beside these. Tiling is why this is linear in nKeys, and therefore why long prompts still fan out.
	t := AttnRowTileK(k, K, nKeys)
	perSlot := 4 * (t*nKeys + 2*nKeys*hd + 2*t*hd)
	// NewHeadWorkerPool allocates a fusedScratch (sBlk+tmp+acc+mRun+lRun+vBlk) alongside the materialized shape above whenever the
	// fused schedule is enabled, so the budget must count it or it oversubscribes (mostly vBlk, hd*nKeys floats, the same order as
	// kh/vt). It is charged whenever fusion could apply, not only when this call will use it, matching the budget's other worst-case
	// assumptions. vt itself goes unused once fusion is active, but dropping it needs the caller's useAcc64/treeMask state, which is
	// unavailable here, and getting it wrong risks a nil vt where fusion does not apply.
	if knobFusedAttention(k) {
		perSlot += 4 * (t*FusedKeyBlock + 2*t*hd + 2*t + hd*nKeys)
	}
	if perSlot <= 0 {
		return 1
	}
	n := PrefillAttnScratchBudget / perSlot
	return max(1, min(n, min(MaxAttnWorkers, nH)))
}

// HeadWorkerPool returns n (capped at MaxAttnWorkers) independent per-head scratch sets sized for this call's K/nKeys/hd, growing
// each slot's backing arrays once as decode extends (grow-only, per slot), so concurrent per-head workers never share mutable
// scratch.
func (s *DecodeScratch) HeadWorkerPool(n, K, nKeys, hd int, wantFused, useAcc64 bool) []HeadWorkerScratch {
	if n > MaxAttnWorkers {
		n = MaxAttnWorkers
	}
	if n < 1 {
		n = 1
	}
	if len(s.HeadPool) < n {
		grown := make([]HeadWorkerScratch, n)
		copy(grown, s.HeadPool)
		s.HeadPool = grown
	}
	for i := range s.HeadPool[:n] {
		p := &s.HeadPool[i]
		if p.mmWS == nil {
			p.mmWS = SerialMMWorkspace()
		}
		// The fused scratch is allocated only when the caller will use it, and grown once. Decode runs with acc64, where
		// attendBatchedHeads computes fusedOK = !useAcc64 && treeMask == nil and never reads ws.Fused, so an unconditional per-call
		// allocation churned 4*hd*nKeys bytes per slot per layer per token (zero-filled, never read, linear in context) through the GC.
		if wantFused && knobFusedAttention(s.Knobs) {
			if t := AttnRowTileK(s.Knobs, K, nKeys); p.Fused == nil || !p.Fused.fits(t, hd, nKeys) {
				p.Fused = newFusedScratch(t, hd, nKeys)
			}
		} else {
			p.Fused = nil
		}
		if c := K * hd; cap(p.QH) < c {
			p.QH = make([]float32, c)
			p.CH = make([]float32, c)
		}
		if cap(p.AvAcc) < hd {
			p.AvAcc = make([]float64, hd)
		}
		// kh/vt are unused whenever useAcc64 is true: the acc64 kernels (MatmulQKAcc64/MatmulAVAcc64) read keys/vals directly with
		// strided addressing, skipping the gather. The check is on useAcc64 itself, not assumed (HeadWorkerPool's callers,
		// attendToken and the multi-sequence walker, pass acc64 true), so a future non-acc64 caller still grows them on its own first call.
		if !useAcc64 {
			if c := nKeys * hd; cap(p.KH) < c {
				g := max(2*cap(p.KH), c) // headroom: nKeys grows by 1 each decode step
				p.KH = make([]float32, g)
				p.VT = make([]float32, g)
			}
		}
		if c := K * nKeys; cap(p.Scores) < c {
			g := max(2*cap(p.Scores), c)
			p.Scores = make([]float32, g)
		}
		if c := K * nKeys * AttnGroupedNEONSize; cap(p.GroupScores) < c {
			g := max(2*cap(p.GroupScores), c)
			p.GroupScores = make([]float32, g)
		}
		if c := K * hd * AttnGroupedNEONSize; cap(p.GroupCtx) < c {
			p.GroupCtx = make([]float32, c)
			p.GroupAvAcc = make([]float64, c)
		}
		// groupScoresCombined (Arm B) is grown lazily inside
		// kvAttend.groupedLayer itself instead of here: its size depends on nKV,
		// which HeadWorkerPool's callers don't thread through this signature,
		// and kvAttend.groupedLayer already has it captured. Only pool[0] (the
		// "leader" slot — see HeadWorkerScratch's own doc) ever uses it.
	}
	return s.HeadPool[:n]
}

// headScoresPool is HeadWorkerPool for the per-head scalar attention (attendSinksToken, attendQueryP): n slots (capped at
// MaxAttnWorkers) whose scores are nKeys long. It grows nothing else, so the score buffers are all it adds.
func (s *DecodeScratch) headScoresPool(n, nKeys int) []HeadWorkerScratch {
	n = min(max(n, 1), MaxAttnWorkers)
	if len(s.HeadPool) < n {
		grown := make([]HeadWorkerScratch, n)
		copy(grown, s.HeadPool)
		s.HeadPool = grown
	}
	for i := range s.HeadPool[:n] {
		p := &s.HeadPool[i]
		if cap(p.Scores) < nKeys {
			p.Scores = make([]float32, growCap(cap(p.Scores), nKeys))
		}
		p.Scores = p.Scores[:nKeys]
	}
	return s.HeadPool[:n]
}

// NewHeadWorkerPool builds n (capped at MaxAttnWorkers) fresh HeadWorkerScratch entries sized for one call's K/nKeys/hd: the
// batched (M=K>1) forward's one-shot sibling of DecodeScratch.HeadWorkerPool. That path has no cache.scr to grow and reuse across
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
func NewHeadWorkerPool(n, K, nKeys, hd int, wantFused bool) []HeadWorkerScratch {
	return newHeadWorkerPoolK(nil, n, K, nKeys, hd, wantFused)
}

// NewHeadWorkerPoolK is NewHeadWorkerPool reading the given model's knobs (nil = the live environment).
func newHeadWorkerPoolK(k KnobSource, n, K, nKeys, hd int, wantFused bool) []HeadWorkerScratch {
	if n > MaxAttnWorkers {
		n = MaxAttnWorkers
	}
	if n < 1 {
		n = 1
	}
	// Slots are sized for one row tile, not the whole prompt: kvAttend.oneHead walks its query rows in tiles of exactly this many, so
	// anything larger would be allocated and never touched.
	t := AttnRowTileK(k, K, nKeys)
	skipMaterialized := wantFused && knobFusedAttention(k)
	pool := make([]HeadWorkerScratch, n)
	for i := range pool {
		p := HeadWorkerScratch{
			mmWS:  SerialMMWorkspace(),
			Fused: fusedIfEnabled(k, t, hd, nKeys),
			QH:    make([]float32, t*hd),
			KH:    make([]float32, nKeys*hd),
			CH:    make([]float32, t*hd),
			AvAcc: make([]float64, hd),
		}
		if !skipMaterialized {
			p.VT = make([]float32, nKeys*hd)
			p.Scores = make([]float32, t*nKeys)
		}
		pool[i] = p
	}
	return pool
}

// LocalBufs returns two length-n scratch slices (grown once on demand) for assembling a local layer's contiguous read window in
// the K=1 decode path.
func (s *DecodeScratch) LocalBufs(n int) (lk, lv []float32) {
	if cap(s.localK) < n {
		g := max(2*cap(s.localK), n) // headroom: window grows each step; avoid a per-token realloc+zero
		s.localK = make([]float32, g)
		s.localV = make([]float32, g)
	}
	return s.localK[:n], s.localV[:n]
}

// HeadLogits returns n floats the cache owns for a batched pass's LM-head output, valid until the next batched prefill or
// verify on this cache. It is not the decode logits buffer, so a prefill's row outlives the decode forwards after it. nil when
// the cache has no decode scratch (one built outside Model.NewCache), and the caller allocates.
func (c *Cache) HeadLogits(n int) []float32 {
	if c == nil || c.Scr == nil {
		return nil
	}
	if cap(c.Scr.logitsN) < n {
		c.Scr.logitsN = make([]float32, n)
	}
	return c.Scr.logitsN[:n]
}

// GateSlice returns the FFN gate scratch buffer.
func (s *DecodeScratch) GateSlice() []float32 { return s.gate }

// UpSlice returns the FFN up scratch buffer.
func (s *DecodeScratch) UpSlice() []float32 { return s.up }

// NewGateUpScratch allocates a minimal DecodeScratch with gate and up slices of length inter (used by tests).
func NewGateUpScratch(inter int) *DecodeScratch {
	return &DecodeScratch{gate: make([]float32, inter), up: make([]float32, inter)}
}
