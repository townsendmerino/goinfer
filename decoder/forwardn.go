package decoder

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// cpuFastAttention reports whether the live environment enables f32 prefill attention (GOINFER_CPU_FAST_ATTENTION; on unless
// set to "0", which is what --cpu-exact-prefill sets). It is the reading for a hand-built caller with no Model; a loaded model
// reads its own snapshot through (*Model).cpuFastAttention. It is read only here: every consumer receives the answer as an
// explicit argument, so no path can pick it up by accident (see runLayersFromEmbedN).
//
// Enabling it gives up two of the exact (acc64) path's guarantees for the model: spec-decode verify == sequential greedy
// (structurally prevented from applying there) and decode == prefill. A long prompt can therefore produce a different response at
// temperature 0 than under the exact kernel; decode itself is untouched. It also gives up split-invariance: f32 reassociation makes
// a prompt's KV depend on how the prompt was chunked, so a Session (a warm prefix, then a divergent suffix) stops matching a
// one-shot generate over the same tokens. That divergence is accepted (operator decision). TestSessionFastAttnDivergence pins
// it; the equality itself is gated under the exact kernel by TestSessionNgramSpecParity.
//
// It is floored by prompt length (fastAttnMinPrompt). MoE is deliberately not excluded, and nothing here checks the arch.
// Evidence, measured figures and the decision record: docs/code-notes/decoder.md#cpuFastAttention.
func cpuFastAttention() bool { return (*knobSet)(nil).cpuFastAttention() }

// cpuFastAttention is the per-model form: off when this model was loaded with
// Options.ExactPrefill, else the env var's answer.
func (m *Model) cpuFastAttention() bool { return !m.exactPrefill && m.knobs.cpuFastAttention() }

// ExactPrefill reports whether this model was loaded with Options.ExactPrefill — prompt ingestion
// must take each backend's bit-exact path. Backends consult it next to their own env var.
func (m *Model) ExactPrefill() bool { return m.exactPrefill }

// fastAttnMinPrompt is the prompt length (K, the suffix being prefilled) below which f32 prefill attention is not used, even
// when enabled: attention is O(K*nKeys), so the win grows with K while the divergence does not, and a short prompt gets a
// different answer for almost nothing.
//
// 512 is a judgement, not a measured crossover (the smallest K measured with a win over ~15%); a sweep that finds one should move
// it and say so. It is deliberately not configurable: a knob would be another way for prefill numerics to vary between two runs
// of the same build. Measurements: docs/code-notes/decoder.md#fastAttnMinPrompt.
const fastAttnMinPrompt = 512

// attnHeadsParThreshold is the per-call work (K*nKeys, the QKᵀ/AV size driver) below which attendBatchedHeads runs its heads
// serially through pool[0] instead of fanning out: the same "small work stays serial" rule int4ParThreshold applies. It is 0, so
// the fan-out is never skipped on size. Campaign record: docs/completed/task-attention-decode-cost.md (move (a)).
const attnHeadsParThreshold = 0

// attnGroupedNEONSize is the query-head group size (NumHeads/NumKVHeads) the aikit grouped acc64 kernels
// (MatmulQKAcc64Group/MatmulAVAcc64Group) have a NEON port for. Any other ratio still runs correctly through the Go fallback but
// slower than separate per-head calls, so attendGroupedHeads is gated to exactly this size rather than trusting the kernel's own
// dispatch to save a mismatched caller.
const attnGroupedNEONSize = 6

// attnGroupedNEONBlock is the NEON grouped kernels' own block width (8 keys per QK block, 8 dims per AV block). runSplitAligned
// rounds each worker's slice up to a multiple of it so no slice falls through to the much slower plain-Go grouped kernel for a
// ragged remainder.
const attnGroupedNEONBlock = 8

// attnGroupedMinKeys is the number of attended keys below which the per-head path runs unchanged. 128 is the smallest depth
// either R13 record measured a win at: a conservative floor, not the cache probe's own crossover (the cache already dedups a GQA
// group's shared reads below roughly K=1024). Records: docs/measurements/r13-distinct-bytes-probe-2026-09-19.md and
// docs/measurements/r13-neon-kernel-ab-2026-09-20.md.
const attnGroupedMinKeys = 128

// knobSet.attnGrouped (knobs.go) reports whether the grouped-kernel decode path may run; on unless GOINFER_ATTN_GROUPED=0, which
// restores the per-head path. The path also needs attnGroupedKernels (a per-architecture default, on for arm64 only).

// attnGroupedRuns counts attendGroupedHeads/attendGroupedLayer dispatches. Bit-identity hides dispatch inertness (a grouped path
// that never runs passes every correctness gate), so a test asserts it nonzero above attnGroupedMinKeys and zero below;
// moeExpertMajorRuns (mlp.go) is the same idiom.
var attnGroupedRuns int64

// canBatchN reports whether the batched M=K path applies: K>1 on a family without its own sequential forward, with a gated MLP
// and no learned position embedding (GPT-2-style models take the sequential fallback). Sparse MoE batches its attention, but its
// FFN stays per row because the router picks different experts per token.
func (m *Model) canBatchN(K int) bool {
	// The program decides (program.batchEligible), not a family list: a restated list once fell a family behind, and a 2-token
	// prompt ran the dense attention stack over conv layers that load no q/k/v/o.
	return K > 1 && m.w.Embed.Rows() != 0 && m.program().BatchEligible()
}

// hasRecurrentState reports whether this model carries state mutated in place per token (a conv window, an SSM state, a
// linear-attention state), which no positional rewind can restore. It is the arch-side view of KVCache.hasRecurrentState(), read
// from the dispatch table's Recurrent bit so it can be asked before any cache exists (the resident path has none).
//
// One predicate, read everywhere (specRollbackSafe, resident_reuse.go, generate_vl.go): a second hand-written family list is how
// a family gets missed at a consumer. A new state kind is added to the dispatch table once and every caller follows.
func (m *Model) hasRecurrentState() bool {
	// Nil-safe because residentReuseLen is reachable from a Model with no weights loaded (its unit test builds one). A model with no
	// arch has no family and so no recurrent state.
	if m == nil || m.w == nil || m.w.arch == nil {
		return false
	}
	return m.w.arch.recurrent()
}

// specRollbackSafe reports whether speculative decode's rollback (KVCache.TruncateTo after a partial accept) correctly restores
// this model's state. It is true for softmax/GQA and MLA, whose state lives in the cache and is resliced. It is false for every
// family hasRecurrentState marks Recurrent: TruncateTo does not roll back their in-place state, so verifying a K-token block
// over-advances it and the next round decodes from it, a silent distribution bug rather than a crash. It is also false for
// windowed models (see below). The n-gram speculative entry points refuse unsafe models and the caller falls back to plain
// decode; no checkpoint-at-block-start/restore path exists.
func (m *Model) specRollbackSafe() bool { return m.specRollbackSafeVia(nil, 0) }

// specRollbackSafeVia is specRollbackSafe for a loop that verifies through fold, a resident's recurrent fold path, in passes of up
// to width rows (BlockSpec; nil for a loop that rolls back with TruncateTo or ForwardN). A recurrent model is admitted only there:
// the fold path never writes the committed recurrent state during a verify and folds only the committed rows into it
// (ResidentRecurrentSpec, SP2). The resident reports rows only when it serves every recurrent layer, and the model's own layers must
// all be Gated DeltaNet (RecurrentSpecArenaBytes), the kind the path is built for.
func (m *Model) specRollbackSafeVia(fold ResidentRecurrentSpec, width int) bool {
	a := m.w.arch
	if m.hasRecurrentState() && (fold == nil || width < 1 || width > fold.RecurrentSpecRows() || RecurrentSpecArenaBytes(a, 1) == 0) {
		return false
	}
	// A staged sliding-window cache stores local layers in physical rings. Once a ring wraps (context > window), a rollback of more
	// than one position cannot restore the evicted positions, so verify would read stale history and diverge. The resident
	// positional path is itself safe, but the speculative loops can use the staged ring even when m.resident != nil (and a resident
	// CAS loss falls back to staged mid-flight), so windowed models are refused for speculation unconditionally; the cost is plain
	// decode. That keeps the staged rollback sites (KVCache.TruncateTo) to ring-free caches, where it is always exact. Re-enabling
	// windowed speculation means consuming an exact-rollback bool at each rollback site.
	if a.SlidingWindow > 0 {
		return false
	}
	return true
}

// forwardLayersN runs the embedding + all transformer layers + final norm over
// the K tokens in ids — appended as the next K positions of cache — and returns
// the [K, HiddenDim] post-final-norm hidden states (the LM head is the caller's).
// The weight matmuls run at M=K, so each weight streams from memory once and is
// reused across the K rows (aikit's column-blocked W8A8 kernel); attention stays
// per-position and causal. Bit-identical to K sequential forwards. Assumes
// canBatchN(len(ids)) — callers check.
func (m *Model) forwardLayersN(reqCtx context.Context, ids []int, cache *KVCache, fastAttn bool) ([]float32, error) {
	t0 := costStart(cache)
	h := m.embedN(ids)
	if cache.cost != nil {
		m.countEmbed(cache, len(ids), false, t0)
	}
	return m.runLayersFromEmbedN(reqCtx, h, cache, fastAttn)
}

// embedN returns the [K*HiddenDim] embedding rows for ids (the per-row token embed
// times any embedding scale) — the text input to the batched stack.
func (m *Model) embedN(ids []int) []float32 {
	return m.embedNInto(make([]float32, len(ids)*m.w.arch.HiddenDim), ids)
}

// embedNInto is embedN into h, which holds exactly len(ids)*HiddenDim floats.
func (m *Model) embedNInto(h []float32, ids []int) []float32 {
	hidden := m.w.arch.HiddenDim
	for i, id := range ids {
		m.w.Embed.Row(id, h[i*hidden:i*hidden+hidden])
	}
	if emb := m.program().Embed; emb.Scale != 0 && emb.Scale != 1 {
		s := float32(emb.Scale)
		for i := range h {
			h[i] *= s
		}
	}
	return h
}

// runLayersFromEmbedN runs the batched layer stack over K pre-embedded rows
// h [K*HiddenDim] — the batched analog of the streaming runLayersFromEmbed, so a
// multimodal caller can inject projected vision embeddings at <image> positions
// (and SetImageBlocks for the bidirectional mask) before the forward. Text rows
// must already carry the embedding scale; injected vision rows are raw (HF's
// masked_scatter overwrites the scaled placeholder embed). Returns the [K,
// HiddenDim] post-final-norm hidden states (the LM head is the caller's).
func (m *Model) runLayersFromEmbedN(reqCtx context.Context, h []float32, cache *KVCache, fastAttn bool) ([]float32, error) {
	return m.runLayersFromEmbedNFloor(reqCtx, h, cache, fastAttn, len(h)/m.w.arch.HiddenDim, nil)
}

// runLayersFromEmbedNFloor is runLayersFromEmbedN with the fast-attention floor keyed on floorK, the length of the whole
// suffix being prefilled, rather than on this pass's rows: a chunked prefill (Model.prefillLogits) passes every chunk the
// suffix's length, so a short last chunk keeps the suffix's attention path and its rows equal a single pass's.
func (m *Model) runLayersFromEmbedNFloor(reqCtx context.Context, h []float32, cache *KVCache, fastAttn bool, floorK int, ps *prefillPassScratch) ([]float32, error) {
	return m.walkBatch(reqCtx, m.program(), h, cache, fastAttn, floorK, ps, nil)
}

// attnTask is one unit of work handed to attnWorkerPool: run fn over the
// caller's own [lo,hi) slice, tagged with worker index w so fn can index its
// own per-worker scratch (pool[w] in attendGroupedLayer).
type attnTask struct {
	fn        func(w, lo, hi int)
	w, lo, hi int
	done      *sync.WaitGroup
}

// attnWorkerPool is a small set of long-lived goroutines, parked on a channel between rounds and shared by every decode step's
// attention fan-out. It replaced a fresh goroutine per split: a decoded token does up to 28 layers x 2 kv heads x 2 phases of
// fan-outs, and the spawn overhead alone outweighed most of the grouped kernel's savings. A persistent pool pays goroutine
// creation once; each round is a channel send to a parked goroutine. Record:
// docs/measurements/r13-served-decode-2026-09-20.md.
type attnWorkerPool struct {
	tasks chan attnTask
}

func newAttnWorkerPool(n int) *attnWorkerPool {
	p := &attnWorkerPool{tasks: make(chan attnTask, n)}
	for range n {
		go func() {
			for t := range p.tasks {
				t.fn(t.w, t.lo, t.hi)
				t.done.Done()
			}
		}()
	}
	return p
}

// globalAttnWorkerPool is process-wide, not per-Model: total attention fan-out parallelism is bounded by maxAttnWorkers across
// every concurrent request, the same oversubscription discipline applied within one call.
var globalAttnWorkerPool = sync.OnceValue(func() *attnWorkerPool {
	return newAttnWorkerPool(maxAttnWorkers)
})

// runSplitAligned fans work over the independent-output range [0,n) into up to `workers` contiguous, non-overlapping slices, each
// width rounded up to a multiple of align, running fn(w, lo, hi) once per non-empty slice: worker 0 inline, the rest via the
// persistent globalAttnWorkerPool. fn must write only to worker w's own scratch (pool[w]) and to its own disjoint [lo,hi) region of
// any shared output; no combining is performed, so callers must not need one.
//
// The alignment matters: aikit's NEON grouped kernels process their range in 8-wide blocks and fall back to the plain-Go grouped
// kernel (6-10x slower) for a remainder that does not fill a block. Equal-width splitting would give every worker a ragged
// remainder; rounding up keeps each slice block-aligned at the cost of slightly uneven work.
func runSplitAligned(workers, n, align int, fn func(w, lo, hi int)) {
	if workers <= 1 || n < workers {
		fn(0, 0, n)
		return
	}
	pool := globalAttnWorkerPool()
	var wg sync.WaitGroup
	per := (n + workers - 1) / workers
	if align > 1 {
		per = ((per + align - 1) / align) * align
	}
	for w := 1; w < workers; w++ {
		lo, hi := w*per, min((w+1)*per, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		pool.tasks <- attnTask{fn: fn, w: w, lo: lo, hi: hi, done: &wg}
	}
	hi0 := min(per, n)
	fn(0, 0, hi0)
	wg.Wait()
}

// attendTileFor is attendTileForK reading the live environment (tests that build scratch by hand).
func attendTileFor(ws *headWorkerScratch, K, nKeys, hd int) int {
	return attendTileForK(nil, ws, K, nKeys, hd)
}

// attendTileForK returns how many query rows attendOneHead may process at once with this slot: the cache-sized tile from
// attnRowTileK, clamped to what the slot's buffers hold. The tile is a property of the pool, not of the call: the pool is sized
// once from maxKeys = startPos+K, but a local layer whose ring has wrapped assembles a shorter window and attnRowTile is inverse
// in nKeys, so the per-layer tile can exceed the qh the slot was allocated and the Q gather would slice past its length. The
// slot's capacity is the binding constraint, so the slot decides: with kt <= tile and nKeys <= maxKeys every other slot buffer
// (scores, ch, kh, vt) fits once qh does. It also covers hand-built scratch in the ring tests.
func attendTileForK(k *knobSet, ws *headWorkerScratch, K, nKeys, hd int) int {
	tile := attnRowTileK(k, K, nKeys)
	if hd < 1 {
		return max(1, tile)
	}
	if rows := len(ws.qh) / hd; rows >= 1 && tile > rows {
		tile = rows
	}
	return max(1, tile)
}

// attendBatchedHeads computes grouped-query causal attention for K query positions at once, per head: per KV head it gathers
// K_head and V_headᵀ once (f32 path), then per query head scores = Q·Kᵀ, a scaled causal/window-masked softmax per row (masked
// entries zeroed so they drop out of the next matmul), and ctx = scores·V scattered into ctx[K,qDim].
//
// With useAcc64 the dots accumulate in f64 and are bit-identical to the per-token walker's attendToken. The f32 path
// (MatmulBT) reassociates the reduction, so its parity is argmax-exact plus cosine, the standard the GPU residency attention also
// meets. The softmax exp stays per row in float64. Scratch slices (qh, kh, vt, scores, ch) are caller-owned and reused across
// layers.
//
// keys/vals are the contiguous K/V the gather reads, physical row 0 holding absolute key position `base`: cache.Keys(layer) at
// base 0 for a global layer; for a local (sliding-window) layer an assembled [base, startPos+K) window, so the ring's wrap is
// invisible here and the math is byte-identical to append-forever. Per-query masking stays in absolute positions
// (WindowStart/attendHi), mapped to physical columns by s-base.
func attendBatchedHeads(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, global bool, arch *Architecture, useAcc64 bool, pool []headWorkerScratch) {
	attendBatchedHeadsP(q, ctx, keys, vals, base, cache, layer, startPos, K, arch.attnParamsFor(cache, global), arch, useAcc64, pool)
}

// attendBatchedHeadsP is attendBatchedHeads with the window, scale and softcap given, not read from arch: the program walker's
// form (attentionParams). arch supplies the shapes and the knobs.
func attendBatchedHeadsP(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, ap attentionParams, arch *Architecture, useAcc64 bool, pool []headWorkerScratch) {
	// headsAt, not NumHeads: Laguna varies the query head count per layer (its KV heads stay uniform, so group is per-layer too).
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim, qDim := nKV*hd, nH*hd
	group := nH / nKV
	scale := ap.Scale
	nKeys := len(keys) / kvDim
	if arch.gptoss != nil && ap.Sinks == nil {
		panic(fmt.Sprintf("decoder: layer %d: gpt-oss attention without its sinks (the caller built attentionParams without LayerWeights.AttnSinks)", layer))
	}
	// sinkMax and sinkSum add query head qh's learned sink to a softmax: to the max-shift, then exp(sink − max) to the denominator. They
	// do nothing when ap.Sinks is nil, so every other family runs the same operations as before.
	sinkMax := func(qh int, maxS float64) float64 {
		if ap.Sinks != nil {
			if sl := float64(ap.Sinks[qh]); sl > maxS {
				return sl
			}
		}
		return maxS
	}
	sinkSum := func(qh int, maxS, sum float64) float64 {
		if ap.Sinks != nil {
			return sum + math.Exp(float64(ap.Sinks[qh])-maxS)
		}
		return sum
	}
	// MoE routing is discontinuous: an f32 QKᵀ reassociation can flip a top-k expert at a near-tie and cascade, so the acc64 kernels
	// accumulate each dot in f64 (bit-identical to the sequential f64 reference); dense MLPs tolerate the f32 error. There is no
	// shared `matmul` variable: acc64 calls MatmulQKAcc64/MatmulAVAcc64 directly (strided, no gather), and the f32 path is handed its
	// matmul per head (package-level MatmulBT when the head loop is serial, the worker's serial Workspace when not).

	// The fused schedule is eligible only on the f32 path (it would break acc64's bit-identity), without a tree mask or logit softcap.
	// Each worker uses its own ws.fused, never a shared one, since each gathers a different kv head's V.
	fusedOK := !useAcc64 && cache.treeMask == nil && ap.Softcap == 0
	// attendOneHead runs one query head's QKᵀ → softmax → scores·V → scatter into ctx using ws's scratch. It runs concurrently
	// across heads and is bit-identical regardless of pool slot, goroutine or finish order: each head's reduction order is
	// unchanged and its ctx write is a disjoint qhead*hd slice. mm is the f32 matmul to use (package-level MatmulBT on the serial
	// arm, the worker's own serial Workspace on the head-parallel arm); it is unused on the acc64 path.
	attendOneHead := func(qhead int, ws *headWorkerScratch, mm func(a, b, dst []float32, M, K, N int)) {
		var fs *fusedScratch
		if fusedOK {
			fs = ws.fused
		}
		kvh := qhead / group
		qh, scores, ch, avAcc := ws.qh, ws.scores, ws.ch, ws.avAcc
		// Walk the query rows in tiles. Every step below is row-wise, so splitting rows splits independent outputs, which is what
		// bit-identity permits. No key-dimension split happens here and none may: it would re-associate the softmax denominator and the AV
		// fold, exactly what acc64 exists to prevent. The point is memory: scores is tile*nKeys, not K*nKeys, so a worker slot stops
		// growing with the square of the prompt and the pool can still fan out on a long prompt.
		//
		// `i` indexes the tile; `gi` is the global row. Positions and masks use `gi` (startPos+gi, treeRowPos[gi], treeMask[gi]), buffers
		// use `i`. The tile comes from attendTileForK, not attnRowTile: the slot's capacity binds, and recomputing it from this layer's key
		// count panicked a warm windowed session.
		tile := attendTileForK(arch.knobs, ws, K, nKeys, hd)
		for t0 := 0; t0 < K; t0 += tile {
			kt := min(tile, K-t0)
			for i := range kt { // gather this tile's Q_head [kt,hd]
				b := (t0+i)*qDim + qhead*hd
				copy(qh[i*hd:i*hd+hd], q[b:b+hd])
			}
			// The fused schedule, when enabled and applicable, replaces the whole QKᵀ/softmax/scores·V sequence for this tile and keeps the
			// score block resident instead of materializing kt x nKeys (fusedattn.go). It declines, and falls through, for acc64 (whose
			// bit-identity it would break) and for tree attention (whose per-(row,column) mask is not the contiguous [lo,hi] bound it handles).
			if fs != nil {
				for i := range kt {
					gi := t0 + i
					pos := startPos + gi
					fs.lo[i] = ap.Start(pos) - base
					fs.hi[i] = cache.attendHi(pos) - base
				}
				if attendTileFused(mm, qh[:kt*hd], ws.kh[:nKeys*hd], fs.vBlk, ch[:kt*hd],
					fs.sBlk, fs.tmp, fs.acc, fs.mRun, fs.lRun,
					kt, hd, nKeys, scale, fs.lo[:kt], fs.hi[:kt]) {
					for i := range kt { // scatter this tile's ctx_head into ctx[K,qDim]
						b := (t0+i)*qDim + qhead*hd
						copy(ctx[b:b+hd], ch[i*hd:i*hd+hd])
					}
					continue
				}
			}
			// QKᵀ: scores[kt,nKeys] = Q_head[kt,hd] · K_head[nKeys,hd]ᵀ. Acc64 reads keys directly (row stride kvDim, element stride 1),
			// skipping the kh gather. MatmulQKAcc64 interleaves 8 keys' dot products as concurrent f64 accumulator chains to hide FMA latency;
			// each key's own d-order fold is unchanged, so the result is bit-identical to the plain strided reduction
			// (TestAttendStrided_matchesGatherReference checks it at this stride).
			if useAcc64 {
				linalg.MatmulQKAcc64(qh[:kt*hd], keys, scores[:kt*nKeys], kt, hd, nKeys, kvh*hd, kvDim)
			} else {
				mm(qh[:kt*hd], ws.kh[:nKeys*hd], scores[:kt*nKeys], kt, hd, nKeys)
			}
			// Scaled, masked softmax per query row; zero the out-of-range entries
			// so they contribute nothing to the scores·V matmul below.
			for i := range kt {
				gi := t0 + i // global row: positions and masks are indexed by it, buffers by i
				pos := startPos + gi
				if cache.treeRowPos != nil {
					pos = cache.treeRowPos[gi]
				}
				rowS := scores[i*nKeys : i*nKeys+nKeys]
				// TREE attention (05): row i attends to the whole committed prefix
				// [loP, batchCol0) plus only its ancestor batch columns (treeMask[i][j]).
				if cache.treeMask != nil {
					loP := ap.Start(pos) - base
					batchCol0 := startPos - base
					allowed := func(s int) bool {
						if s < batchCol0 {
							return s >= loP
						}
						j := s - batchCol0
						return j < K && cache.treeMask[gi][j]
					}
					maxS := math.Inf(-1)
					for s := range nKeys {
						if allowed(s) {
							sc := softcapScore(float64(rowS[s])*scale, ap.Softcap)
							rowS[s] = float32(sc)
							if sc > maxS {
								maxS = sc
							}
						}
					}
					maxS = sinkMax(qhead, maxS)
					var sum float64
					for s := range nKeys {
						if allowed(s) {
							e := math.Exp(float64(rowS[s]) - maxS)
							rowS[s] = float32(e)
							sum += e
						} else {
							rowS[s] = 0
						}
					}
					sum = sinkSum(qhead, maxS, sum)
					inv := 1.0 / sum
					for s := range nKeys {
						if rowS[s] != 0 {
							rowS[s] = float32(float64(rowS[s]) * inv)
						}
					}
					continue
				}
				// Absolute attend range [start, hi]; map to physical columns by −base
				// (base = absolute position of column 0). hi is the inclusive upper
				// key bound: pos for a causal text query, or the image-block end for a
				// bidirectional image position (so it also attends to the block's
				// future tokens). Equals pos with no image blocks — inert for text.
				loP := ap.Start(pos) - base
				hiP := cache.attendHi(pos) - base
				if loP < 0 {
					loP = 0
				}
				if hiP >= nKeys {
					hiP = nKeys - 1
				}
				if loP <= hiP {
					active := rowS[loP : hiP+1]
					_ = active[len(active)-1]
					maxS := math.Inf(-1)
					for s, v := range active {
						sc := softcapScore(float64(v)*scale, ap.Softcap)
						active[s] = float32(sc)
						if sc > maxS {
							maxS = sc
						}
					}
					maxS = sinkMax(qhead, maxS)
					var sum float64
					for s, v := range active {
						e := math.Exp(float64(v) - maxS)
						active[s] = float32(e)
						sum += e
					}
					sum = sinkSum(qhead, maxS, sum)
					inv := 1.0 / sum
					clear(rowS[:loP])
					for s, v := range active {
						active[s] = float32(float64(v) * inv)
					}
					clear(rowS[hiP+1 : nKeys])
				} else {
					clear(rowS[:nKeys])
				}
			}
			// scores·V: ctx_head[kt,hd] = scores[kt,nKeys] · V_head[nKeys,hd]. Acc64 reads vals directly, "as if transposed" (row stride 1,
			// element stride kvDim), skipping a vt gather and transpose. MatmulAVAcc64 reads V rows contiguously (keys-outer, dims-inner) into
			// hd independent f64 accumulators; each dim's accumulator sees the same key-ascending adds as the strided walk, so it is
			// bit-identical.
			//
			// A one-row tile (every K=1 decode) writes this head's context straight into ctx: MatmulAVAcc64 overwrites its destination and the
			// head's hd floats at ctx[t0*qDim+qhead*hd] are contiguous, so the scratch `ch` and the scatter copy are dead work. A multi-row
			// tile's [kt,hd] result is not contiguous in ctx (row stride qDim) and keeps the scatter.
			direct := useAcc64 && kt == 1
			if direct {
				b := t0*qDim + qhead*hd
				linalg.MatmulAVAcc64(scores[:nKeys], vals, ctx[b:b+hd], avAcc, 1, nKeys, hd, kvh*hd, kvDim)
			} else if useAcc64 {
				linalg.MatmulAVAcc64(scores[:kt*nKeys], vals, ch[:kt*hd], avAcc, kt, nKeys, hd, kvh*hd, kvDim)
			} else {
				mm(scores[:kt*nKeys], ws.vt[:hd*nKeys], ch[:kt*hd], kt, nKeys, hd)
			}
			if !direct {
				for i := range kt { // scatter this tile's ctx_head into ctx[K,qDim]
					b := (t0+i)*qDim + qhead*hd
					copy(ctx[b:b+hd], ch[i*hd:i*hd+hd])
				}
			}
		}
	}

	// attendGroupedHeads is the grouped-kernel path: one MatmulQKAcc64Group/MatmulAVAcc64Group pair covers all attnGroupedNEONSize
	// query heads of one kv head, instead of that many attendOneHead calls re-reading the same K/V rows. The caller must have checked
	// K==1 (decode; M>1 is not wired), no tree mask (its per-(row,column) mask needs attendOneHead's own branch), and
	// group == attnGroupedNEONSize.
	//
	// Q for these heads is contiguous in q's [K,qDim] layout (and so is ctx), so no gather is needed. Softmax stays per head, row by
	// row, identical to attendOneHead's non-tree branch: grouping shares the QKᵀ and scores·V loads and folds, never the reduction
	// inside each head's softmax.
	attendGroupedHeads := func(kvh int, ws *headWorkerScratch) {
		qh0 := kvh * group
		a := q[qh0*hd : qh0*hd+group*hd]
		gScores := ws.groupScores[:group*nKeys]
		linalg.MatmulQKAcc64Group(a, keys, gScores, group, hd, nKeys, kvh*hd, kvDim)
		pos := startPos // K==1 is a precondition, so this call's only row is gi=0
		loP := ap.Start(pos) - base
		hiP := cache.attendHi(pos) - base
		if loP < 0 {
			loP = 0
		}
		if hiP >= nKeys {
			hiP = nKeys - 1
		}
		for g := range group {
			rowS := gScores[g*nKeys : g*nKeys+nKeys]
			if loP <= hiP {
				active := rowS[loP : hiP+1]
				_ = active[len(active)-1]
				maxS := math.Inf(-1)
				for s, v := range active {
					sc := softcapScore(float64(v)*scale, ap.Softcap)
					active[s] = float32(sc)
					if sc > maxS {
						maxS = sc
					}
				}
				maxS = sinkMax(qh0+g, maxS)
				var sum float64
				for s, v := range active {
					e := math.Exp(float64(v) - maxS)
					active[s] = float32(e)
					sum += e
				}
				sum = sinkSum(qh0+g, maxS, sum)
				inv := 1.0 / sum
				clear(rowS[:loP])
				for s, v := range active {
					active[s] = float32(float64(v) * inv)
				}
				clear(rowS[hiP+1 : nKeys])
			} else {
				clear(rowS[:nKeys])
			}
		}
		// MatmulAVAcc64Group overwrites a [group, hd] destination and this kv group's query heads are contiguous in ctx, so it writes
		// straight into ctx with no scratch copy.
		gAvAcc := ws.groupAvAcc[:group*hd]
		linalg.MatmulAVAcc64Group(gScores, vals, ctx[qh0*hd:qh0*hd+group*hd], gAvAcc, group, nKeys, hd, kvh*hd, kvDim)
		atomic.AddInt64(&attnGroupedRuns, 1)
	}
	// attnGroupedOK is this call's eligibility for the grouped path, checked once per call since none of it depends on qhead.
	// treeMask excludes speculative verify (attendGroupedHeads has no per-(row,column) mask); K != 1 excludes prefill/batched M>1
	// (not wired); attnGroupedKernels is the per-architecture platform gate.
	attnGroupedOK := useAcc64 && K == 1 && cache.treeMask == nil && attnGroupedKernels &&
		group == attnGroupedNEONSize && nKeys >= attnGroupedMinKeys && arch.knobs.attnGrouped()
	// runHeadRange walks qhead across [h0,h1), taking the grouped path for any run of attnGroupedNEONSize heads that starts on a
	// kv-group boundary and fits inside [h0,h1); a worker whose range splits a kv group uses attendOneHead for that group's heads.
	runHeadRange := func(ws *headWorkerScratch, h0, h1 int) {
		qhead := h0
		for qhead < h1 {
			if attnGroupedOK && qhead%group == 0 && qhead+group <= h1 {
				attendGroupedHeads(qhead/group, ws)
				qhead += group
				continue
			}
			attendOneHead(qhead, ws, nil)
			qhead++
		}
	}

	// attendGroupedLayer is the grouped path for a pool of more than one slot (decode, K==1). Arm A (runHeadRange/attendGroupedHeads)
	// needs one worker to own a whole kv group's heads and so collapses worker count to nKV; this keeps the pool busy and still uses
	// the grouped kernel by splitting within each kv head's group: QK by key range, softmax by row, AV by dim range, the three phases
	// joined in between.
	//
	// All three are independent-output splits, not reduction splits (QK reduces over d and is split by key; AV reduces over keys and
	// is split by dim), so no floating-point combining is needed: each worker's slice lands in a disjoint region of the shared buffer
	// by plain copy. The result is bit-identical to the per-head path and to Arm A; TestAttendGroupedLayer_manyWorkers is the check.
	// The softmax phase must stay split: see the note on it below.
	attendGroupedLayer := func(pool []headWorkerScratch) {
		splitWorkers := max(min(len(pool), maxAttnWorkers), 1)
		leader := &pool[0]
		if c := nKV * group * nKeys; cap(leader.groupScoresCombined) < c { // grows by nKV*group a token: headroom (R-17)
			leader.groupScoresCombined = make([]float32, growCap(cap(leader.groupScoresCombined), c))
		}
		fullScores := leader.groupScoresCombined[:nKV*group*nKeys]
		fullCtx := ctx[:nKV*group*hd] // R-17: kv head kvh's group is ctx[kvh*group*hd:(kvh+1)*group*hd], the layout the old combined scratch was copied into at the end

		runSplitAligned(splitWorkers, nKeys, attnGroupedNEONBlock, func(w, lo, hi int) {
			ws := &pool[w]
			sliceLen := hi - lo
			slice := ws.groupScores[:group*sliceLen]
			for kvh := range nKV {
				qh0 := kvh * group
				a := q[qh0*hd : qh0*hd+group*hd]
				linalg.MatmulQKAcc64Group(a, keys, slice, group, hd, sliceLen, kvh*hd+lo*kvDim, kvDim)
				dst := fullScores[kvh*group*nKeys : (kvh+1)*group*nKeys]
				for g := range group {
					copy(dst[g*nKeys+lo:g*nKeys+hi], slice[g*sliceLen:(g+1)*sliceLen])
				}
			}
		})

		pos := startPos
		loP := ap.Start(pos) - base
		hiP := cache.attendHi(pos) - base
		if loP < 0 {
			loP = 0
		}
		if hiP >= nKeys {
			hiP = nKeys - 1
		}
		// Softmax is split by row (one kv head's one query head) across the workers, as the per-head path already does inside each head's
		// worker. Running it serially on this goroutine was measured slower than not grouping at all; do not collapse it
		// (docs/code-notes/decoder.md#attendGroupedLayer.softmax).
		runSplitAligned(splitWorkers, nKV*group, 1, func(w, lo, hi int) {
			for row := lo; row < hi; row++ {
				kvh, g := row/group, row%group
				rowS := fullScores[kvh*group*nKeys+g*nKeys : kvh*group*nKeys+g*nKeys+nKeys]
				if loP <= hiP {
					active := rowS[loP : hiP+1]
					_ = active[len(active)-1]
					maxS := math.Inf(-1)
					for s, v := range active {
						sc := softcapScore(float64(v)*scale, ap.Softcap)
						active[s] = float32(sc)
						if sc > maxS {
							maxS = sc
						}
					}
					maxS = sinkMax(kvh*group+g, maxS)
					var sum float64
					for s, v := range active {
						e := math.Exp(float64(v) - maxS)
						active[s] = float32(e)
						sum += e
					}
					sum = sinkSum(kvh*group+g, maxS, sum)
					inv := 1.0 / sum
					clear(rowS[:loP])
					for s, v := range active {
						active[s] = float32(float64(v) * inv)
					}
					clear(rowS[hiP+1 : nKeys])
				} else {
					clear(rowS[:nKeys])
				}
			}
		})

		runSplitAligned(splitWorkers, hd, attnGroupedNEONBlock, func(w, d0, d1 int) {
			ws := &pool[w]
			sliceHd := d1 - d0
			sliceCtx := ws.groupCtx[:group*sliceHd]
			sliceAcc := ws.groupAvAcc[:group*sliceHd]
			for kvh := range nKV {
				rows := fullScores[kvh*group*nKeys : (kvh+1)*group*nKeys]
				linalg.MatmulAVAcc64Group(rows, vals, sliceCtx, sliceAcc, group, nKeys, sliceHd, kvh*hd+d0, kvDim)
				dst := fullCtx[kvh*group*hd : (kvh+1)*group*hd]
				for g := range group {
					copy(dst[g*hd+d0:g*hd+d1], sliceCtx[g*sliceHd:(g+1)*sliceHd])
				}
			}
		})

		atomic.AddInt64(&attnGroupedRuns, int64(nKV))
	}

	if !useAcc64 {
		// f32 path (the default for prefill above fastAttnMinPrompt). gatherKV fills ws's kh/vt for one kv head. It is a pure function of
		// (keys, vals, kvh), so two workers gathering the same kvh into their own buffers produce identical bytes, which is why the serial
		// and fan-out arms below are bit-identical, not merely close (TestAttendF32Fanout_bitIdentical).
		gatherKV := func(ws *headWorkerScratch, kvh int) {
			// The fused schedule needs V block-major (a key-range slice of the [hd, nKeys]
			// layout is not contiguous), so the layout is chosen here at gather time rather
			// than re-transposed per block.
			if fusedOK && ws.fused != nil {
				gatherKVFused(ws.kh, ws.fused.vBlk, keys, vals, kvh, hd, kvDim, nKeys)
				return
			}
			for s := range nKeys {
				kvBase := s*kvDim + kvh*hd
				copy(ws.kh[s*hd:s*hd+hd], keys[kvBase:kvBase+hd])
				vrow := vals[kvBase : kvBase+hd]
				_ = vrow[hd-1]
				_ = ws.vt[(hd-1)*nKeys+s]
				for d := range hd {
					ws.vt[d*nKeys+s] = vrow[d]
				}
			}
		}
		// Fan out over query heads, as the acc64 path does. Every pool slot already owns a full-size kh/vt pair (prefillAttnWorkersK
		// budgets 2*nKeys*hd per slot), so a worker gathers into its own buffers and nothing is shared across the group. Heads are
		// assigned in contiguous runs so a worker walks whole kv groups and re-gathers only when kvh changes: at most nKV + workers
		// gathers, not nH. MatmulBT already fans out over output columns internally; head-level fan-out is what parallelises the gather,
		// the softmax and the scatter.
		workers := 1
		if len(pool) > 1 && nH > 1 && K*nKeys >= attnHeadsParThreshold {
			workers = min(len(pool), nH)
		}
		if workers <= 1 {
			// Serial arm: one slot, and the matmul keeps its OWN column-level
			// fan-out (package-level MatmulBT) since no head-level fan-out is
			// competing with it. This is the pre-A3 behaviour exactly.
			ws := &pool[0]
			for kvh := range nKV {
				gatherKV(ws, kvh)
				for g := range group {
					attendOneHead(kvh*group+g, ws, linalg.MatmulBT)
				}
			}
			return
		}
		var wg sync.WaitGroup
		headsPer := (nH + workers - 1) / workers
		for w := 1; w < workers; w++ {
			h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
			if h0 >= h1 {
				continue
			}
			wg.Add(1)
			go func(w, h0, h1 int) {
				defer wg.Done()
				ws := &pool[w]
				lastKVH := -1
				for qhead := h0; qhead < h1; qhead++ {
					if kvh := qhead / group; kvh != lastKVH {
						gatherKV(ws, kvh)
						lastKVH = kvh
					}
					// Serial matmul: the fan-out is at the head level here, and
					// nesting MatmulBT's column fan-out inside it would
					// oversubscribe. Bit-identical to the column-parallel form
					// by MatmulBT's width contract.
					attendOneHead(qhead, ws, ws.mmWS.MatmulBT)
				}
			}(w, h0, h1)
		}
		h1_0 := min(headsPer, nH)
		ws0 := &pool[0]
		lastKVH0 := -1
		for qhead := range h1_0 {
			if kvh := qhead / group; kvh != lastKVH0 {
				gatherKV(ws0, kvh)
				lastKVH0 = kvh
			}
			attendOneHead(qhead, ws0, ws0.mmWS.MatmulBT)
		}
		wg.Wait()
		return
	}

	// With more than one pool slot, Arm B (attendGroupedLayer) takes priority over Arm A, whose one-worker-per-group rule would strand
	// the other slots idle. With one slot Arm B has nothing to parallelise and the serial arm below already takes the grouped kernel
	// (runHeadRange's [0,nH) range spans whole groups), so Arm B is skipped rather than adding fork-join cost.
	if attnGroupedOK && len(pool) > 1 {
		attendGroupedLayer(pool)
		return
	}

	// The acc64 path (the one every live caller takes): the nH query heads are independent (disjoint ctx writes, no shared mutable
	// state; kh/vt are unused here). Below the fan-out floor, or with one pool slot or one head, run serially through pool[0]: a
	// fork-join costs real time the per-head kernels have already shrunk to a few microseconds a head at shallow depth.
	if len(pool) <= 1 || nH <= 1 || K*nKeys < attnHeadsParThreshold {
		runHeadRange(&pool[0], 0, nH)
		return
	}
	workers := min(len(pool), nH)
	var wg sync.WaitGroup
	headsPer := (nH + workers - 1) / workers
	for w := 1; w < workers; w++ {
		h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
		if h0 >= h1 {
			continue
		}
		wg.Add(1)
		go func(w, h0, h1 int) {
			defer wg.Done()
			runHeadRange(&pool[w], h0, h1)
		}(w, h0, h1)
	}
	h1_0 := min(headsPer, nH)
	runHeadRange(&pool[0], 0, h1_0)
	wg.Wait()
}

// lmHeadN projects M post-final-norm hidden rows (h is [M, HiddenDim]) to logits
// [M, VocabSize] (+ final-logit softcap), at M=K so the head weights stream once.
func (m *Model) lmHeadN(h []float32, M int) []float32 {
	return m.lmHeadNInto(h, M, nil)
}

// lmHeadNInto is lmHeadN writing into dst (M*VocabSize long; nil allocates).
func (m *Model) lmHeadNInto(h []float32, M int, dst []float32) []float32 {
	hp := &m.program().Head
	logits := dst
	if logits == nil {
		logits = make([]float32, M*m.w.arch.VocabSize)
	}
	if hp.Tied {
		matmul(m.be, &m.w.Embed, h, logits, M)
	} else {
		matmul(m.be, &m.w.LMHead, h, logits, M)
	}
	if hp.Softcap > 0 {
		sc := float32(hp.Softcap)
		for j, val := range logits {
			logits[j] = sc * float32(math.Tanh(float64(val/sc)))
		}
	}
	// logit_scale (Cohere multiplier stored as goinfer's reciprocal; Granite logits_scaling divisor): mirrors logitsFromHidden's tail.
	// The sequential path applies it, so batched prefill/verify must too or forwardN diverges for any LogitScale family.
	if hp.LogitScale != 0 && hp.LogitScale != 1 {
		inv := float32(1 / hp.LogitScale)
		for j := range logits {
			logits[j] *= inv
		}
	}
	return logits
}

// forwardN runs a batched forward over ids and returns the logits at every
// position ([K][VocabSize]) — used by the speculative verifier. Bit-identical to
// K sequential forwards. Falls back to sequential for the non-batched archs. The
// rows may be the cache's own buffer, valid until the next batched prefill or verify on cache.
func (m *Model) forwardN(reqCtx context.Context, ids []int, cache *KVCache) ([][]float32, error) {
	// Speculative VERIFY: never fast, whatever the operator asked for. See forwardNAttn.
	return m.forwardNAttn(reqCtx, ids, cache, false)
}

// forwardNAttn is forwardN with the attention-kernel choice made by the caller. Speculative verify must run the exact kernel
// (fastAttn=false) on both arms or its equality argument collapses. A caller that must produce KV identical to Generate's prefill
// passes cpuFastAttention(), as prefillLogits does; otherwise "token-identical to plain greedy" stops holding at temperature 0 for
// any prompt over the fastAttnMinPrompt floor.
func (m *Model) forwardNAttn(reqCtx context.Context, ids []int, cache *KVCache, fastAttn bool) ([][]float32, error) {
	K := len(ids)
	if K == 0 {
		return nil, nil
	}
	// Tree verify needs the batched path: the sequential fallback ignores treeRowPos/
	// treeMask, so tree nodes would be attended as a linear chain (wrong parents) —
	// error rather than silently mis-verify.
	if cache.treeMask != nil && !m.canBatchN(K) {
		return nil, fmt.Errorf("decoder.forwardN: tree verify unsupported on this arch (not batchable)")
	}
	// Compute-time LoRA is wired only into the sequential forward, so an active adapter must take the M=1 path (as prefillLogits
	// does): the batched verify would project every position with the base model and commit base K/V, silently verifying drafts
	// against the wrong model.
	if cache.lora != nil || !m.canBatchN(K) {
		out := make([][]float32, K)
		for i, id := range ids {
			l, err := m.forward(id, cache)
			if err != nil {
				return nil, err
			}
			out[i] = append([]float32(nil), l...) // forward reuses scr.logits — copy
		}
		return out, nil
	}
	h, err := m.forwardLayersN(reqCtx, ids, cache, fastAttn)
	if err != nil {
		return nil, err
	}
	vocab := m.w.arch.VocabSize
	t0 := costStart(cache)
	logits := m.lmHeadNInto(h, K, cache.headLogits(K*vocab))
	if cache.cost != nil {
		m.countHead(cache, K, t0)
	}
	out := make([][]float32, K)
	for i := range K {
		out[i] = logits[i*vocab : i*vocab+vocab]
	}
	return out, nil
}

// prefillLogits processes the whole prompt and returns the logits at its last position (the seed for the first generated token),
// in the cache's own buffer (KVCache.headLogits), valid until the next batched prefill or verify on cache.
// On the batched archs it runs the layers at M=len(prompt) in one pass (each weight streamed once, reused across all positions)
// and the LM head on the last position only. Otherwise it falls back to sequential runLayers + forward. Bit-identical to the
// sequential prefill when the exact kernel is in use; the cache is filled with the whole prompt either way.
func (m *Model) prefillLogits(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	// Compute-time LoRA (#7) is wired only into the per-token walker (projQKVToken
	// + gatedMLPAct), so an active adapter takes the M=1 path — the prompt's K/V must carry
	// the delta or decode would continue a base-projected context. The RAM-density win
	// (N adapters share one base) is unaffected; only adapter'd prefill speed regresses.
	if cache.lora != nil || !m.canBatchN(len(prompt)) {
		for _, id := range prompt[:len(prompt)-1] {
			// The sequential fallback checks the context per token: it has no layer batch to bound, and a LoRA'd or non-batchable arch
			// prefills here.
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if _, err := m.runLayers(id, cache); err != nil {
				return nil, err
			}
		}
		return m.forward(prompt[len(prompt)-1], cache)
	}
	h, err := m.prefillPassesN(ctx, len(prompt), cache, func(dst []float32, lo, hi int) {
		t0 := costStart(cache)
		m.embedNInto(dst, prompt[lo:hi])
		if cache.cost != nil {
			m.countEmbed(cache, hi-lo, false, t0)
		}
	}, nil)
	if err != nil {
		return nil, err
	}
	cache.reserveRows(cache.planRows - cache.Pos()) // a one-pass prefill reserves the request's generation here
	hidden := m.w.arch.HiddenDim
	t0 := costStart(cache)
	logits := m.lmHeadNInto(h[len(h)-hidden:], 1, cache.headLogits(m.w.arch.VocabSize)) // the LM head on the last row only
	if cache.cost != nil {
		m.countHead(cache, 1, t0)
	}
	if len(prompt) >= prefillReleaseRows {
		releaseHostMemory()
	}
	return logits, nil
}

// prefillPassesN runs the n rows of a prefill through the batched layers, in passes of at most prefillChunkRows rows when the
// cache allows it (prefillChunkable) and in one otherwise. embed fills dst with the prefill's rows [lo, hi), embedded and with any
// image features spliced in. keep, if non-nil, sees each pass's output rows, which start at row lo and are valid only during the
// call. It returns the last pass's output rows, post final norm.
//
// A sliding-window layer's fast attention is not chunk-invariant (G40-1: its output depends on where the pass's key window starts),
// so a cache with one keeps the single pass whenever the prefill takes fast attention. A cut never falls inside an image block
// (KVCache.prefillCut), so every block's rows share a pass.
func (m *Model) prefillPassesN(ctx context.Context, n int, cache *KVCache, embed func(dst []float32, lo, hi int), keep func(lo int, rows []float32)) ([]float32, error) {
	fast := m.cpuFastAttention() && n >= fastAttnMinPrompt
	C := n
	if cache.prefillChunkable() && !(cache.localAny && fast) {
		C = prefillChunkRows
	}
	hidden, start := m.w.arch.HiddenDim, cache.Pos()
	// One working set for every chunk, sized by the first (the largest, unless an image block stretches a later one) and the
	// whole prefill's keys (G40's amendment).
	ps := &prefillPassScratch{poolKeys: start + n}
	var h []float32
	for c0, i := 0, 0; c0 < n; i++ {
		c1 := cache.prefillCut(start, c0, min(c0+C, n))
		if i == 1 {
			cache.reserveRows(max(start+n, cache.planRows) - cache.Pos()) // the first chunk taught the cache each layer's width
			if prefillChunkDone != nil {
				prefillChunkDone(-1) // the KV reserve, which a gate on working buffers reads apart from them
			}
		}
		dst := ps.take(&ps.h, (c1-c0)*hidden)
		embed(dst, c0, c1)
		var err error
		atomic.AddInt64(&prefillPasses, 1)
		if h, err = m.runLayersFromEmbedNFloor(ctx, dst, cache, m.cpuFastAttention(), n, ps); err != nil {
			return nil, err
		}
		if keep != nil {
			keep(c0, h)
		}
		if prefillChunkDone != nil {
			prefillChunkDone(i)
		}
		c0 = c1
	}
	if prefillScratchSeen != nil {
		prefillScratchSeen(ps)
	}
	return h, nil
}

// prefillScratchSeen, when set (tests only), sees the working set a prefill's passes used, after the last one.
var prefillScratchSeen func(ps *prefillPassScratch)

// prefillPassScratch is a batched pass's working set. A chunked prefill (Model.prefillLogits) makes one and passes it to
// every chunk, so the chunks after the first allocate no working buffers; any other caller passes nil and gets fresh ones.
// take clears what it returns, so a chunk starts from the state a fresh allocation gives.
type prefillPassScratch struct {
	h, norm, q, k, v, ctx, att, gate, up, mlpOut, alk, alv, gbuf []float32
	ws                                                           linalg.Workspace
	hq                                                           w4a8Act
	pool                                                         []headWorkerScratch
	poolWant                                                     bool
	poolKeys                                                     int                // the whole prefill's keys, which the attention pool is sized for
	moe                                                          *moeBatchScratch   // the MoE expert-major buffers; nil until a MoE layer runs
	qwen35                                                       *qwen35PassScratch // qwen3.5's mixer buffers; nil until such a layer runs
}

// moeScratch returns the pass's MoE expert-major buffers, starting them on first use so a dense pass carries none.
func (ps *prefillPassScratch) moeScratch() *moeBatchScratch {
	if ps.moe == nil {
		ps.moe = &moeBatchScratch{}
	}
	return ps.moe
}

// take returns (*buf)[:n], cleared, growing *buf when it holds less.
func (ps *prefillPassScratch) take(buf *[]float32, n int) []float32 {
	if cap(*buf) < n {
		*buf = make([]float32, n)
		return *buf
	}
	b := (*buf)[:n]
	clear(b)
	return b
}

// prefillChunkRows is how many rows one pass of a long text prefill holds (docs/tasks/task-cpu-prefill-memory-2026-10.md,
// G40): the pass's working set is sized by its rows, so a chunk bounds it. A variable so a test can cross chunk boundaries on a
// tiny fixture; 2048 in production.
var prefillChunkRows = 2048

// prefillChunkDone, when set (tests only), is called after each chunk of a chunked prefill with its index, and with -1 right
// after the KV reserve: the G40-T0 gates read the allocator there.
var prefillChunkDone func(chunk int)

// prefillPasses counts the batched passes prefillLogits runs, so a test can see that a chunked prefill really chunked: a
// bit-identity gate cannot tell a prefill that never split from one that split correctly.
var prefillPasses int64

// prefillReleaseRows is the prompt length from which prefillLogits returns freed memory to the OS once the prefill is done: a
// long prompt's scratch is garbage by then, and the runtime would otherwise keep it mapped. Shorter prompts skip the forced
// collection.
const prefillReleaseRows = 4096

// prefillLogitsVL prefills a multimodal prompt and returns the last-position
// logits: the text token embeddings with the projected vision embeddings
// (imageEmbeds, [imgLen*HiddenDim]) substituted at the <image> placeholder run
// [imgPos, imgPos+imgLen), under a bidirectional attention mask over that block
// (so the image tokens see each other). Requires the batched path — the
// bidirectional image-block attention lives in attendBatchedHeads. The injected
// embeddings are RAW (the projector output), matching HF's masked_scatter, which
// overwrites the scaled placeholder embed. See docs/multimodal.md §4–5.
func (m *Model) prefillLogitsVL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageEmbeds, cache, false)
}

// prefillLogitsVLSpans is prefillLogitsVL for several images (S11): each span is its own bidirectional block, and
// imageEmbeds holds every span's rows concatenated in span order.
// causal (Pixtral, S10) splices the spans without a bidirectional block: plain causal attention over the prompt.
func (m *Model) prefillLogitsVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	var hN []float32
	var err error
	if causal && m.w.arch.lfm2 != nil {
		hN, err = m.prefillHiddenLFM2VL(ctx, ids, spans, imageEmbeds, cache)
	} else {
		hN, err = m.prefillVLRows(ctx, ids, spans, imageEmbeds, cache, causal, false)
	}
	if err != nil {
		return nil, err
	}
	hidden := m.w.arch.HiddenDim
	return m.lmHeadNInto(hN[len(hN)-hidden:], 1, cache.headLogits(m.w.arch.VocabSize)), nil
}

// prefillHiddenVLSpans runs prefillLogitsVLSpans's prefill and returns the final hidden rows of every position (before
// the LM head), so a test can grade every position, not only the last, through the production path.
func (m *Model) prefillHiddenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	if causal && m.w.arch.lfm2 != nil { // S10, LFM2-VL: lfm2 has no batched pass; its own forward, one row at a time
		return m.prefillHiddenLFM2VL(ctx, ids, spans, imageEmbeds, cache)
	}
	return m.prefillVLRows(ctx, ids, spans, imageEmbeds, cache, causal, true)
}

// prefillVLRows runs the image prefill of prefillHiddenVLSpans in passes (prefillPassesN), each embedding its own rows with the
// raw projected features spliced in. It returns every position's rows when all, else the last pass's.
func (m *Model) prefillVLRows(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal, all bool) ([]float32, error) {
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: multimodal prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	if !causal {
		cache.SetImageBlocks(imageSpanBlocks(spans))
	}
	return m.prefillSpliced(ctx, ids, spans, imageEmbeds, cache, all)
}

// prefillSpliced runs ids through prefillPassesN with each pass's image rows replaced by their features (raw, no embed scale),
// returning every position's rows when all, else the last pass's.
func (m *Model) prefillSpliced(ctx context.Context, ids []int, spans []ImageSpan, feats []float32, cache *KVCache, all bool) ([]float32, error) {
	hidden := m.w.arch.HiddenDim
	var out []float32
	var keep func(lo int, rows []float32)
	if all {
		out = make([]float32, len(ids)*hidden)
		keep = func(lo int, rows []float32) { copy(out[lo*hidden:], rows) }
	}
	h, err := m.prefillPassesN(ctx, len(ids), cache, func(dst []float32, lo, hi int) {
		m.embedNInto(dst, ids[lo:hi])
		spliceImageSpansRange(dst, spans, feats, hidden, lo, hi)
	}, keep)
	if all {
		return out, err
	}
	return h, err
}

// prefillHiddenLFM2VL is prefillHiddenVLSpans for lfm2 (LFM2-VL, S10): the text embeddings with the image rows spliced
// in, each run through the per-token walker in order (the conv windows see every row, image rows included), and each
// position's hidden state final-normed, as the batched entry returns them.
func (m *Model) prefillHiddenLFM2VL(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden)
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := h[i*hidden : (i+1)*hidden]
		out, err := m.runLayersFromEmbed(row, cache)
		if err != nil {
			return nil, err
		}
		m.finalNorm(out)
	}
	return h, nil
}

// prefillLogitsQwenVL prefills a Qwen2.5-VL multimodal prompt and returns the
// last-position logits: text embeddings with the merged vision features
// (imageFeats, [imgLen*HiddenDim] from the ViT+merger) substituted at the <image>
// run [imgPos,imgPos+imgLen), under m-RoPE 3D positions (mropePos, one (t,h,w) per
// absolute sequence position). Two deltas vs prefillLogitsVL (Gemma 3): the image
// tokens attend CAUSALLY — no bidirectional image block (Qwen's bidirectionality is
// inside the ViT, not the decoder) — and the rotary uses m-RoPE (ropeAt reads
// cache.mropePos). The merged features are RAW (no embed scale), matching HF's
// scatter into inputs_embeds.
func (m *Model) prefillLogitsQwenVL(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsQwenVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageFeats, mropePos, cache)
}

// prefillLogitsQwenVLSpans is prefillLogitsQwenVL for several images (S11): every span's merged rows, concatenated in
// span order in imageFeats, replace that span's placeholder run, and mropePos covers them all (mropePositions with one
// grid per image).
func (m *Model) prefillLogitsQwenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageFeats []float32, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	if m.w.arch.qwen35 != nil { // the Gated-DeltaNet hybrids take their own prefill (batched when qwen35BatchNAnyPos applies)
		return m.prefillLogitsQwen35VL(ctx, ids, imageFeats, spans, mropePos, cache)
	}
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: Qwen2.5-VL prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, err
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	cache.mropePos = mropePos                         // ropeAt switches to m-RoPE for this prefill
	cache.mropeDelta = mropeDelta(mropePos, len(ids)) // decode past the prefill rotates at seqPos+delta
	hN, err := m.prefillSpliced(ctx, ids, spans, imageFeats, cache, false)
	if err != nil {
		return nil, err
	}
	return m.lmHeadNInto(hN[len(hN)-hidden:], 1, cache.headLogits(m.w.arch.VocabSize)), nil
}

// addDeepstack adds DeepStack set l to the rows of h ([K*hidden], positions startPos..startPos+K) that fall in the image
// run (S10): HF's hidden_states[visual_pos_masks] += deepstack_visual_embeds[l], after decoder layer l.
func addDeepstack(h []float32, ds *deepstackRows, l, startPos, K, hidden int) {
	set := ds.rows[l]
	off := 0 // the first row of this span within the set
	for _, s := range ds.spans {
		for p := max(s.Pos, startPos); p < min(s.Pos+s.Len, startPos+K); p++ {
			dst := h[(p-startPos)*hidden : (p-startPos+1)*hidden]
			src := set[(off+p-s.Pos)*hidden : (off+p-s.Pos+1)*hidden]
			for j := range dst {
				dst[j] += src[j]
			}
		}
		off += s.Len
	}
}
