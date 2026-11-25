//go:build darwin

package metal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// execState holds the pipelined logits executor's channels and the latched command-buffer abort error.
type execState struct {
	execReq  chan execJob
	execAck  chan []float32
	execDone chan struct{} // closed when execLoop returns — Close waits on this before freeing
	execErr  error
}

// batchFwdState holds the pre-allocated buffers for layer-major ForwardBatch calls.
type batchFwdState struct {
	batchCap                                 int
	batchX                                   Buffer
	batchLogits                              Buffer
	batchUPos, batchUNKeys, batchUQTempScale []Buffer
}

// recordExecErr latches the first command-buffer abort a forward path observes.
func (r *resident) recordExecErr(err error) {
	if err != nil && r.execErr == nil {
		r.execErr = err
	}
}

// takeExecErr returns and clears the latched command-buffer error. The adapter calls
// it after each forward and returns the error in place of the (stale) logits.
func (r *resident) takeExecErr() error {
	err := r.execErr
	r.execErr = nil
	return err
}

type execJob struct {
	emb     []float32
	pos     int
	ropePos int
	noHead  bool
}

// loadEmbedRow dequantizes token id's embedding into the shared input buffer and applies the arch's embedding scale
// (Gemma's sqrt(hidden), FeatEmbedScale; a no-op, <= 1, for every other family): the complete
// token-id-to-layer-0-input step. It exists so the two id-taking entry points (Forward, ForwardArgmax) cannot disagree
// with production, where decoder.embedResident applies the scale on the ForwardEmb path. The snapshot golden drives
// exactly these two methods, so a regression confined to this seam would be invisible to the Metal suite's one
// absolute gate.
func (r *resident) loadEmbedRow(id, pos int) {
	dst := r.x.Floats()
	r.embed.Row(id, dst) // CPU dequant embedding into the shared buffer
	if r.embedScale > 1 {
		for i := range dst {
			dst[i] *= r.embedScale
		}
	}
	r.addLearnedPos(pos)
}

// addLearnedPosTo adds wpe[pos] to dst: GPT-2's learned absolute position embedding, host-side (CPU dequant, like the
// token embedding lookup) since it happens once per token before any GPU dispatch, mirroring decoder/model.go's
// ResidentForward (arch.LearnedPosEmbed: h[i] += wpe[pos][i]). A no-op for every family without FeatLearnedPos.
// addLearnedPos applies it to the input embedding already in r.x.
func (r *resident) addLearnedPosTo(dst []float32, pos int) {
	if !r.learnedPos {
		return
	}
	pe := make([]float32, len(dst))
	r.posEmbed.Row(pos, pe)
	for i := range dst {
		dst[i] += pe[i]
	}
}

func (r *resident) addLearnedPos(pos int) {
	r.addLearnedPosTo(r.x.Floats(), pos)
}

// ensureBatchCap ensures r.batchX, r.batchLogits, and per-token uniform buffers have capacity for at least n tokens. A growth gives
// the rows it replaces back; the caller has stopped the executor, so nothing still reads them.
func (r *resident) ensureBatchCap(n int) {
	if n <= r.batchCap {
		return
	}
	c := 16
	for c < n {
		c *= 2
	}
	for _, b := range []Buffer{r.batchX, r.batchLogits} {
		if b != (Buffer{}) {
			r.d.ReleaseBuf(b)
		}
	}
	held := len(r.batchUPos)
	r.batchUPos = append(r.batchUPos, make([]Buffer, c-held)...)
	r.batchUNKeys = append(r.batchUNKeys, make([]Buffer, c-held)...)
	r.batchUQTempScale = append(r.batchUQTempScale, make([]Buffer, c-held)...)
	for _, sb := range r.batchRows(c, held) {
		sb.set(sb.alloc(r.d))
	}
	r.batchCap = c
}

// setPos writes pos/nKeys/qTempScale into their uniform buffers in place — called at every
// forward entry point right before encoding or committing, mirroring uPos/uNKeys's existing
// per-call-mutation pattern. qTempScale (Ministral 3, FeatAttnTemp) is the SAME formula
// decoder.Model.AttnTempScale computes host-side; recomputed here rather than round-tripping
// through the decoder package because resident has no *decoder.Model reference, only the raw
// params fetched once at build time (attnTempBeta/attnTempOrigMaxPos). attnTempBeta==0 (every
// family without this feature) always gives exactly 1 without evaluating the division at all —
// same guard as decoder/attention.go's sequential path, load-bearing: attnTempOrigMaxPos is 0
// for those families, and pos/0 would poison every Q with NaN otherwise.
func (r *resident) setPos(pos int, ropePos ...int) {
	rp := pos
	if len(ropePos) > 0 {
		rp = ropePos[0]
	}
	r.curNKeys = pos + 1
	r.kvWriteMust(r.kvSlot, pos, pos+1) // the token's K/V row; the entry points map it first, where an error can go back
	r.writePosUniforms(posUniforms{r.uPos, r.uNKeys, r.uRopePos, r.uQTempScale, r.uAttnFANSplit}, pos, rp)
}

// posUniforms is one set of the per-token uniforms a decode command buffer binds: setPos writes the resident's own
// set, and the greedy chain (greedy_chain.go) alternates two of its own so a buffer can be committed while the one
// before it still reads its set.
type posUniforms struct{ uPos, uNKeys, uRopePos, uQTempScale, uFANSplit Buffer }

// writePosUniforms writes token pos's values into set u: the one formula for both setPos and the chain.
func (r *resident) writePosUniforms(u posUniforms, pos, rp int) {
	u.uPos.SetU32(uint32(pos))
	u.uNKeys.SetU32(uint32(pos + 1))
	u.uRopePos.SetU32(uint32(rp))
	// uAttnFAG and uAttnFANSplit are written here, not at the per-layer dispatch site (see their field comment):
	// this happens before this buffer commits, not during the next buffer's encode-ahead, the same argument
	// uPos/uNKeys/uRopePos above rely on. attnFANKV == 0 means no layer is attention_fa-eligible: the buffers stay
	// unused (canUseAttnFA gates on it too), but skip the divide by zero regardless.
	if r.attnFANKV > 0 {
		r.uAttnFAG.SetU32(uint32(r.nH / r.attnFANKV))
		u.uFANSplit.SetU32(uint32(r.attnFASplitFor(pos+1, r.attnFANKV)))
	}
	scale := float32(1)
	if r.attnTempBeta != 0 {
		scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(pos)/r.attnTempOrigMaxPos)))
	}
	u.uQTempScale.Floats()[0] = scale
}

// Forward runs token `id` at absolute position `pos` and returns logits[V]. The whole
// layer stack + LM head is encoded into ONE command buffer, one commit/wait.
func (r *resident) Forward(id, pos int) []float32 {
	// Pin to one OS thread for the whole call: the NSAutoreleasePool (begin/end) is
	// per-OS-thread, and Go can migrate goroutines mid-call — draining a pool on a
	// different thread than it was pushed is UB (intermittent SIGSEGV). Same discipline
	// the CUDA backend's LockOSThread executor uses.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmbedRow(id, pos)
	return r.forwardLogits(pos)
}

// ForwardEmb is Forward given a precomputed embedding[H] (the decoder.ResidentForward shape) instead of a token id: it
// copies the embedding into the shared input buffer and skips the internal lookup. This is the production path:
// decoder.embedResident does the lookup and applies any embed scale before calling in. Numerically identical to
// Forward(id, pos) when emb is the scaled embedding row (Forward applies the scale in loadEmbedRow).
func (r *resident) ForwardEmb(emb []float32, pos int) []float32 {
	return r.ForwardEmbMRoPE(emb, pos, pos)
}

// ForwardEmbMRoPE is ForwardEmb with decoupled rotation position (ropePos) and KV cache position (pos).
func (r *resident) ForwardEmbMRoPE(emb []float32, pos, ropePos int) []float32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.addLearnedPos(pos)
	if r.g4moe != nil && r.g4moe.paged { // synchronous expert paging: per-layer submit+wait, staged experts
		return r.forwardLogitsPaged(pos, ropePos)
	}
	if r.moe != nil && r.moe.paged { // generic MoE's twin (moe.go) — same mechanism, generalized
		return r.forwardLogitsMoEPaged(pos, ropePos)
	}
	return r.forwardLogits(pos, ropePos)
}

// forwardHiddenNoHead encodes the trunk (all layers plus final norm) for one token at absolute position pos, writing
// this position's K/V into the resident cache as ForwardEmb does, but never dispatching the LM head:
// decoder.ResidentPrefillSubsystem never needs logits, and the head is the most expensive matmul in a forward. want is false
// for every position but the last of a HiddenLast sequence: those still need their K/V written (attention over the
// whole sequence, matching hiddenLastBatched's causal chain) but their hidden state is never read, so nil comes back
// without the int8-to-float32 dequant loop.
// The returned hidden state is exactly what feeds the head on the decode path (encodeNorm's rmsnorm_quant or
// layernorm_quant output in r.aq/r.aSc), dequantized; it matches forwardHeadForTest's act.
func (r *resident) forwardHiddenNoHead(emb []float32, pos int, want bool) ([]float32, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.addLearnedPos(pos)
	r.setPos(pos)
	e := r.q.Begin()
	r.encodeTrunkInto(e) // layers → final norm → r.aq/r.aSc; no head dispatch
	e.End()
	r.recordExecErr(e.Err())
	if !want {
		return nil, nil
	}
	q, sc := r.aq.Int8s(), r.aSc.Floats()[0]
	out := make([]float32, r.H)
	for i := range out {
		out[i] = float32(q[i]) * sc
	}
	return out, nil
}

// forwardLogits encodes the trunk + full lm head and reads back logits[V]. Caller must hold
// the OS thread and have filled r.x with the input embedding.
func (r *resident) forwardLogits(pos int, ropePos ...int) []float32 {
	rp := pos
	if len(ropePos) > 0 {
		rp = ropePos[0]
	}
	r.setPos(pos, rp)
	e := r.q.Begin()
	r.encodeTrunkInto(e)                                                           // 28 layers → final norm → r.aq/r.aSc
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH) // full lm head (int8 — logit-critical)
	e.End()
	r.recordExecErr(e.Err())
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()
	r.finalizeLogits()
	return r.logitsHost
}

// finalizeLogits copies the device logits into the host buffer and applies Gemma's final-logit
// softcap host-side (softcap·tanh(logits/softcap)) — the single site FeatFinalLogitSoftcap
// declares on this backend, exactly as the CPU path (logitsFromHidden) and CUDA (resident.go)
// do it. finalSoftcap is 0 for every non-softcapped family (no-op). The softcap is MONOTONIC, so
// ForwardArgmax's on-device argmax needs it not — only the full-logits paths (sampling,
// temperature, parity) do, and both readback sites (forwardLogits, execLoop) route through here.
func (r *resident) finalizeLogits() {
	copy(r.logitsHost, r.logits.Floats())
	if r.finalSoftcap > 0 {
		softcapParallel(r.logitsHost, r.finalSoftcap)
	}
	// Cohere/Command-R's logits_scaling (FeatLogitScale): a plain host-side multiply, the same
	// site and shape as the softcap just above. 0 or 1 for every non-FeatLogitScale family
	// (BackendModelView.LogitScale's own no-op value) — a single serial pass, since a multiply is
	// memory- not compute-bound at any vocab size this repo has seen (unlike softcap's tanh).
	if r.logitScale != 0 && r.logitScale != 1 {
		for i, v := range r.logitsHost {
			r.logitsHost[i] = v * r.logitScale
		}
	}
}

// softcapParallel applies Gemma's final-logit softcap sc*tanh(x/sc) in place, across cores. Every element is
// independent and math.Tanh is deterministic, so the split is byte-identical to the serial form (disjoint writes, no
// reduction or ordering; TestMetalSoftcapParallel_bitIdentical). Serial below the goroutine-spawn threshold (as
// parallelF32ToF16).
// It does not skip on greedy decode, although argmax is softcap-invariant: finalizeLogits calls it whenever
// r.finalSoftcap > 0, on every execLoop token whatever the sampling mode. The paths that skip it never read
// r.logitsHost (ForwardArgmax, and the greedy chain, which greedyChainWhyNot refuses for an embed-scaled family), so a
// Gemma family pays this 262k-wide tanh loop on every decode token regardless of temperature.
func softcapParallel(logits []float32, softcap float32) {
	sc := softcap
	n := len(logits)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if n < 8192 || workers <= 1 {
		for j, v := range logits {
			logits[j] = sc * float32(math.Tanh(float64(v/sc)))
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Go(func() {
			for j := lo; j < hi; j++ {
				logits[j] = sc * float32(math.Tanh(float64(logits[j]/sc)))
			}
		})
	}
	wg.Wait()
}

func (r *resident) ensureExec() {
	if r.execReq == nil {
		r.execReq = make(chan execJob)
		r.execAck = make(chan []float32)
		r.execDone = make(chan struct{})
		go r.execLoop()
	}
}

// ForwardEmbPipe is ForwardEmb through the pipelined executor (encode-ahead) — the production
// decode path. Returns logits[V] (reused buffer; consume before the next call). Synchronous to
// the caller (one job in, one logits out), but the executor overlaps the next token's encode
// with this token's GPU execution.
func (r *resident) ForwardEmbPipe(emb []float32, pos int) []float32 {
	return r.ForwardEmbMRoPEPipe(emb, pos, pos)
}

// ForwardEmbMRoPEPipe is ForwardEmbMRoPE through the pipelined executor (encode-ahead).
func (r *resident) ForwardEmbMRoPEPipe(emb []float32, pos, ropePos int) []float32 {
	if r.g4moe != nil && r.g4moe.paged {
		// Paging tears each MoE layer into two submits with a host readback between — the encode-ahead
		// executor (one static command buffer/token) cannot express it. Fall back to the synchronous
		// paged path; there is no pipelining to lose (Step-0: paging is submit-bound, not encode-bound).
		return r.ForwardEmbMRoPE(emb, pos, ropePos)
	}
	if r.moe != nil && r.moe.paged { // generic MoE's twin — same reasoning, same fallback
		return r.ForwardEmbMRoPE(emb, pos, ropePos)
	}
	r.ensureExec()
	r.execReq <- execJob{emb: emb, pos: pos, ropePos: ropePos, noHead: false}
	return <-r.execAck
}

// ForwardEmbNoLogitsPipe is ForwardNoLogits through the pipelined executor (encode-ahead) —
// overlapping token t+1's trunk encode with token t's GPU execution while skipping the LM head
// dispatch and logits readback.
func (r *resident) ForwardEmbNoLogitsPipe(emb []float32, pos int) {
	if (r.g4moe != nil && r.g4moe.paged) || (r.moe != nil && r.moe.paged) {
		// The synchronous paged path (no executor, ForwardEmbMRoPEPipe's reason), with the final norm and LM head
		// skipped: a prompt token's logits are discarded, and on the Gemma 4 26B the head reads ~0.74 GB of int8 a token
		// (task-m26-mac-2026-10.md). The layers, and so the KV, are the same calls either way.
		r.pagedNoHead = true
		r.ForwardEmb(emb, pos)
		r.pagedNoHead = false
		return
	}
	r.ensureExec()
	r.execReq <- execJob{emb: emb, pos: pos, ropePos: pos, noHead: true}
	<-r.execAck
}

// encodeTrunkCB builds an uncommitted command buffer with only the trunk (no LM head),
// autoreleasing into the executor's long-lived pool.
func (r *resident) encodeTrunkCB() *Encoder {
	e := r.q.BeginNP()
	r.encodeTrunkInto(e)
	e.FinishEncoding()
	return e
}

// encodeLogitsCB builds a complete, un-committed command buffer (trunk + full lm head) with no
// per-call pool — it autoreleases into the executor's long-lived pool.
func (r *resident) encodeLogitsCB() *Encoder {
	e := r.q.BeginNP()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.FinishEncoding()
	return e
}

// execLoop is the pinned executor: pipeline commit(t), pre-encode(t+1), wait(t). One shared autorelease pool, drained
// every drainEvery tokens, with a one-token non-overlapped hiccup so no un-committed command buffer is live across the
// drain (which keeps the pool LIFO-safe).
//
// A buffer is encoded for a key count, not just a head mode: the attention plan (attnPlan) is baked in at encode time.
// Each buffer is encoded for the job it will run (the current job's own key count when encoded fresh, pos+2 when
// pre-encoded for the predicted next job), and a pre-encoded buffer whose plan is not the arriving job's is dropped
// uncommitted and re-encoded. So a new request, the attnFADepthFloor crossing, or a toggle can never run another
// position's plan (TestExecutorAttnPlan; docs/measurements/metal-decode-attn-r17-2026-09-25.md).
func (r *resident) execLoop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(r.execDone) // Close waits here: no command buffer is live once this returns
	const drainEvery = 64
	pool := NewARPool()
	var cur *Encoder
	var curNoHead bool
	var curPlan attnPlan
	count := 0
	encodeFor := func(noHead bool, nKeys int) *Encoder {
		r.encNKeys = nKeys
		defer func() { r.encNKeys = 0 }()
		if noHead {
			return r.encodeTrunkCB()
		}
		return r.encodeLogitsCB()
	}
	for job := range r.execReq {
		var out []float32
		func() {
			curCommitted := false
			defer func() {
				if p := recover(); p != nil {
					if curCommitted && cur != nil {
						cur.WaitDone()
					}
					cur = nil
					pool.Drain()
					pool = NewARPool()
					count = 0
					mem.RepanicBug(p)
					if err, ok := p.(error); ok {
						r.recordExecErr(fmt.Errorf("metal: executor panicked: %w", err))
					} else {
						r.recordExecErr(fmt.Errorf("metal: executor panicked: %v", p))
					}
					out = nil
				}
			}()
			want := r.attnPlanFor(job.pos + 1)
			if cur != nil && (curNoHead != job.noHead || curPlan != want) {
				// Pre-encoded for a different head mode or attention plan than this job needs: drop it
				// uncommitted, with the pool it was encoded into.
				pool.Drain()
				pool = NewARPool()
				cur = nil
				count = 0
			}
			if cur != nil && r.d.Tracking() && !cur.Tracked() && costPlantForTest != 2 {
				// Encoded ahead before a cost counter was attached: drop it and encode it again under the counter, so the first
				// counted token attributes its dispatches (cost.go).
				pool.Drain()
				pool = NewARPool()
				cur = nil
				count = 0
			}
			if cur == nil {
				cur, curNoHead, curPlan = encodeFor(job.noHead, job.pos+1), job.noHead, want
			}
			r.loadEmb(job.emb)       // this token's embedding + pos (set at commit time, not encode)
			r.addLearnedPos(job.pos) // GPT-2: += wpe[pos], same commit-time placement as the copy above
			r.setPos(job.pos, job.ropePos)
			cur.Commit()
			curCommitted = true

			count++
			drain := count%drainEvery == 0
			var next *Encoder
			var nextNoHead bool
			var nextPlan attnPlan
			if !drain {
				next, nextNoHead, nextPlan = encodeFor(job.noHead, job.pos+2), job.noHead, r.attnPlanFor(job.pos+2)
			}
			cur.WaitDone()
			curCommitted = false
			r.recordExecErr(cur.Err()) // set BEFORE the execAck send so the adapter's read is ordered
			r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = cur.GPUStart(), cur.GPUEnd(), cur.KernStart(), cur.KernEnd()
			if !job.noHead {
				r.finalizeLogits()
				out = r.logitsHost
			}

			if drain { // no un-committed cb live now → safe to drain the shared pool
				pool.Drain()
				pool = NewARPool()
				cur = nil
			} else {
				cur, curNoHead, curPlan = next, nextNoHead, nextPlan
			}
		}()
		r.execAck <- out
	}
	pool.Drain()
}

// stopExec shuts down the executor goroutine (if started) and blocks until it has returned, which is what makes
// freeing safe: closing the channel only signals, the loop may still be waiting on an in-flight command buffer, and
// releasing a buffer it references is a use-after-free.
func (r *resident) stopExec() {
	r.stopChain() // an open greedy chain has command buffers in flight on r.q too (greedy_chain.go)
	if r.execReq != nil {
		close(r.execReq)
		r.execReq = nil
		<-r.execDone
	}
}

// slotBuffers returns the paged MoE slot-pool buffers, g4moe's and the generic MoE's: GPU-read-only (phase 2 reads
// them; the pread stage CPU-writes their contents), so safe to pin resident. Both pools hold the buffer category the
// residency-set bisect found the win in (the pread-invalidated slot buffers; see buildResident).
func (r *resident) slotBuffers() []Buffer {
	var out []Buffer
	for l := range r.layers {
		// A pool's storage is one contiguous buffer per field (not N per-slot objects), so pinning it is one 4-buffer
		// add per pool regardless of N.
		if p := r.layers[l].g4moe; p != nil && p.pool != nil {
			out = append(out, p.pool.guW, p.pool.guS, p.pool.dW, p.pool.dS)
		}
		if p := r.layers[l].moe; p != nil && p.pool != nil {
			out = append(out, p.pool.guW, p.pool.guS, p.pool.dW, p.pool.dS)
		}
	}
	return out
}

// kvBuffers returns the per-layer KV cache buffers — GPU-WRITTEN by attention every token.
func (r *resident) kvBuffers() []Buffer {
	out := make([]Buffer, 0, len(r.kc)+len(r.vc)+len(r.ks)+len(r.vs))
	out = append(out, r.kc...)
	out = append(out, r.vc...)
	out = append(out, r.ks...)
	out = append(out, r.vs...)
	out = append(out, r.kvTab)
	if r.kvWin != nil {
		out = append(out, r.kvWin.kvTab)
	}
	return out
}

// scratchBuffers returns the per-token GPU-WRITTEN intermediates (attention/MLP staging, MoE scratch,
// logits, and the per-token uniform buffers). NOT the KV cache (see kvBuffers).
func (r *resident) scratchBuffers() []Buffer {
	out := []Buffer{
		r.x, r.aq, r.aSc, r.ctx, r.cq, r.cSc, r.oO, r.mq, r.mSc, r.dq, r.dSc, r.dO,
		r.logits, r.qkv, r.gu, r.part, r.tok, r.uPos, r.uNKeys, r.uRopePos,
	}
	if g := r.g4moe; g != nil {
		out = append(out, g.rLogits, g.rIdx, g.rWgt, g.g4x1, g.g4x2, g.g4rn)
	}
	return out
}

// writtenBuffers is every GPU-written buffer (KV cache + scratch) — the set that MUST be excluded when
// pinning "all read-only" buffers (pinning a written member regresses the writing phase).
func (r *resident) writtenBuffers() []Buffer {
	return append(r.kvBuffers(), r.scratchBuffers()...)
}

// kvSlotBuf is one resident KV slot: its block table in each group, a view of that group's kvTab (kt the primary group's, ktw
// the window group's).
type kvSlotBuf struct {
	kt, ktw Buffer
}

// kvBlocksFor is how many 64-row blocks hold positions rows.
func kvBlocksFor(positions int) int {
	return (positions + decoder.KVBlockRows - 1) / decoder.KVBlockRows
}

// kvScrambleForTest hands blocks out alternately from the two ends of the pool and skips each slot's home block (kvTake), so a
// sequence's consecutive blocks are physically far apart and out of order. Bit-identity against the in-order layout is what shows every
// kernel addressing KV goes through the table.
var kvScrambleForTest = false

// kvHostRow is the row position pos of the bound slot occupies in layer l's pool, counted from the pool's start as the host sees
// its contents (the null block is rows 0 to 63): the block the slot's table in l's group names, at the position's offset inside it.
// Positions pos to the end of its block are the consecutive rows after it.
func (r *resident) kvHostRow(l, pos int) int {
	g := r.kvGroupOf(l)
	return int(g.kvTabU32[r.kvSlot*g.kvTabBlocks+pos/decoder.KVBlockRows])*decoder.KVBlockRows + pos%decoder.KVBlockRows
}

// kvElemBytes is one element of the K/V cache: 1 for int8, 4 for f32, 2 for f16.
// kvLayerWant is what one layer's kernels address in the KV pools: an MLA layer's latent row (latent elements, f32), an attention
// layer's K and V rows (kvDim elements in nKV heads, at the KV precision), or none, for a layer that reads another's KV (shared) or holds no
// pool.
type kvLayerWant struct {
	latent, kvDim, nKV int
	shared             bool
}

// kvCheckRows checks layer l's declared per-position rows (decoder.Model.LayerBuffers) against want, and returns the width the layer's
// pool is sized from (0 for none). Every row's name, width, heads and precision must be what the kernels address, with no row missing,
// extra or declared twice: a pool sized from the list and written by kernels sized otherwise would overflow. State entries are not rows.
func kvCheckRows(l int, bufs []program.Buf, want kvLayerWant) (int, error) {
	rows := map[string]program.Buf{}
	for _, b := range bufs {
		if b.Unit != program.PerPosition {
			continue
		}
		if _, again := rows[b.Name]; again {
			return 0, fmt.Errorf("metal: layer %d: the program declares %s rows twice", l, b.Name)
		}
		rows[b.Name] = b
	}
	var need map[string]program.Buf
	switch {
	case want.shared || (want.latent == 0 && want.kvDim == 0):
		need = map[string]program.Buf{}
	case want.latent > 0:
		need = map[string]program.Buf{"latent": {Prec: program.PrecF32, Elems: want.latent}}
	default:
		need = map[string]program.Buf{"k": {Prec: program.PrecKV, Elems: want.kvDim, Heads: want.nKV}, "v": {Prec: program.PrecKV, Elems: want.kvDim, Heads: want.nKV}}
	}
	for name, b := range rows {
		w, ok := need[name]
		switch {
		case !ok:
			return 0, fmt.Errorf("metal: layer %d: the program declares %s rows, which its kernels have no pool for", l, name)
		case b.Elems != w.Elems || b.Prec != w.Prec || (w.Heads > 0 && b.Heads != w.Heads):
			return 0, fmt.Errorf("metal: layer %d: the program declares %s rows of %d elements in %d heads at precision %d; the kernels address %d in %d at %d",
				l, name, b.Elems, b.Heads, b.Prec, w.Elems, w.Heads, w.Prec)
		}
	}
	for name := range need {
		if _, ok := rows[name]; !ok {
			return 0, fmt.Errorf("metal: layer %d: the program declares no %s rows", l, name)
		}
	}
	if want.latent > 0 {
		return rows["latent"].Elems, nil
	}
	return rows["k"].Elems, nil
}

func (r *resident) kvElemBytes() int {
	switch {
	case r.kvI8:
		return 1
	case r.kvF32:
		return 4
	}
	return 2
}

// kvSlotCount is how many resident KV slots were allocated (at least 1).
func (r *resident) kvSlotCount() int { return max(1, len(r.kvSlotBufs)) }

// useKVSlot binds resident KV slot i: every later encode reads and writes its buffers. The executor is stopped first,
// because a command buffer it pre-encoded for the next token baked in the previous slot's buffers (execLoop encodes
// token t+1 while t runs).
func (r *resident) useKVSlot(i int) error {
	if i < 0 || i >= r.kvSlotCount() {
		return fmt.Errorf("metal: KV slot %d out of range (%d allocated)", i, r.kvSlotCount())
	}
	if len(r.kvSlotBufs) == 0 || i == r.kvSlot {
		return nil
	}
	r.stopExec()
	r.kvBindTables(i)
	for l := range r.layers { // a recurrent layer's window and state are the slot's own (allocRecurrentSlots)
		if D := r.layers[l].delta; D != nil && i < len(D.slotWin) {
			D.win, D.state = D.slotWin[i], D.slotState[i]
		}
		if M := r.layers[l].mamba; M != nil && i < len(M.slotWin) {
			M.win, M.ssm = M.slotWin[i], M.slotSSM[i]
		}
	}
	r.kvSlot = i
	return nil
}

// drainMoEPrefetch joins any background MoE expert prefetch goroutines across all layers so no goroutine
// writes into pool buffers or reads giwFile after abort or during Close.
func (r *resident) drainMoEPrefetch() {
	for l := range r.layers {
		if g := r.layers[l].g4moe; g != nil && g.pool != nil {
			g.pool.drainPrefetch()
		}
		if m := r.layers[l].moe; m != nil && m.pool != nil {
			m.pool.drainPrefetch()
		}
	}
}

// Close stops the executor, waits for it, then releases every MTLBuffer this resident allocated, so it must be called
// once the model is unloaded: purego has no ARC and Metal has no context-destroy to reclaim in bulk, and cmd/serve loads
// and unloads models at runtime, so an unclosed resident leaks its weights, per-layer KV and MoE experts until the
// process exits. It returns error to satisfy io.Closer and match cuda/gpu's Close() error; teardown cannot fail, so it
// always returns nil. Idempotent: ReleaseAll empties the ledger, so a second Close is a no-op.
//
// Close must not run while a request still holds the model: that is a use-after-free (on CUDA, a driver SIGSEGV that
// kills the server). Serve's unload drains in-flight holders first (internal/serveapp/admin.go, handleAdminUnload;
// ); keep that ordering for any new caller.
func (r *resident) Close() error {
	r.stopExec()
	r.drainMoEPrefetch()
	if r.g4moe != nil && r.g4moe.giwFile != nil {
		_ = r.g4moe.giwFile.Close() // the pread-staging fd (GOINFER_MOE_PREAD)
		r.g4moe.giwFile = nil
	}
	if r.moe != nil && r.moe.giwFile != nil {
		_ = r.moe.giwFile.Close() // the generic-MoE path's twin (moe.go)
		r.moe.giwFile = nil
	}
	if r.d != nil {
		r.d.ReleaseAll()     // every MTLBuffer
		r.d.ReleaseObjects() // command queue, ~40 pipelines, 1-2 libraries, and the MTLDevice
	}
	return nil
}

// LastGPUTimes returns, for the last Forward, the GPU-busy window and the kernel window
// (incl scheduling) in seconds — GPUEnd-GPUStart and kernelEnd-kernelStart from the command
// buffer. wall - gpuBusy is the per-token host bubble (Step 0 of the headroom decision tree).
func (r *resident) LastGPUTimes() (gpuBusy, kernTotal float64) {
	return r.gpuEnd - r.gpuStart, r.kernEnd - r.kernStart
}

// ForwardArgmax runs the identical trunk but replaces the full lm head and logits readback with the fused block-argmax
// (per-tile (maxLogit, rowIdx), then argmax_finish, then a 4-byte token). It returns argmax(Forward's logits), first
// max wins, without materializing the logit vector. Production greedy decode does not call it: Metal has no
// ResidentDecodeSubsystem implementation, so greedy runs through the greedy chain (ResidentDecodeSubsystem, greedy_chain.go) when
// it is available and otherwise through the full-logits ForwardEmbPipe with a host argmax. Only tests and gates call
// this.
func (r *resident) ForwardArgmax(id, pos int) uint32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmbedRow(id, pos) // includes the arch embed scale — see loadEmbedRow
	if r.V%8 != 0 {
		// gemv_w8a8_amax derives its output row from tgid*(tgs>>5)+sgid, as the SA-family decode kernels do (see
		// buildResident's width checks): dispatchThreads: makes the last threadgroup non-uniform whenever the dispatch
		// total is not a multiple of the threadgroup size, and that formula silently miscomputes for a non-uniform
		// group, rewriting an already-written row while the true tail row is never computed. Padding the weight is not
		// a safe workaround here: a padding row's score depends on the activation, which is unknown at load time, so
		// it could win the argmax. Route a non-%8 vocab (GPT-2's 50257 is the first) through the full-logits path
		// instead: gemv_w8a8_coal addresses its row directly from threadgroup_position_in_grid, which Metal guarantees
		// correct whatever the threadgroup's uniformity. Slower, but correct.
		return uint32(argmaxF32(r.forwardLogits(pos)))
	}
	// setPos, not a direct uPos/uNKeys write: the same gap as the paged MoE forwards (moe.go, gemma4_moe.go). It
	// matters only for FeatAttnTemp, which no family combines with this dispatch today, so it is currently a
	// no-op, as in the paged case.
	r.setPos(pos)
	e := r.q.Begin()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8Amax, (r.V)*32, 256, r.aq, r.aSc, r.lmW, r.lmS, r.part, r.uH) // tile partials (int8 head)
	e.Dispatch(r.pArgFinish, 256, 256, r.part, r.tok, r.uP)                           // reduce tiles → token
	e.End()
	r.recordExecErr(e.Err())
	return uint32(headID(r.tok.U32(), r.V))
}

// headID is the fused argmax head's id as a token. The head (gemv_w8a8_amax or gemv_w8f32_amax, then argmax_finish) writes an
// id at or past V only for a row whose every logit is NaN, and argmaxF32 gives 0 for that row.
func headID(id uint32, V int) int {
	if int64(id) >= int64(V) {
		return 0
	}
	return int(id)
}

// argmaxF32 is internal/sampling's Argmax, mirrored here to avoid a package dependency for one
// function: first-max-wins (strict >), the same tie-break metal's own argmax_finish kernel uses
// (cv>v||(cv==v&&ci<idx) — lower index wins a tie) and decoder's CPU path uses.
func argmaxF32(v []float32) int {
	best, bi := v[0], 0
	for i, x := range v[1:] {
		if x > best {
			best, bi = x, i+1
		}
	}
	return bi
}

// ForwardBatch runs N embeddings at consecutive positions starting at startPos,
// encoding all N token forward passes into a SINGLE Metal command buffer and
// single compute command encoder using a LAYER-MAJOR dispatch schedule (weights
// streamed once per layer across all N tokens), returning all N logits vectors.
func (r *resident) ForwardBatch(embeddings [][]float32, startPos int) ([][]float32, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	n := len(embeddings)
	if n == 0 {
		return nil, nil
	}

	// This path drives encodeAttentionResidualWith with its own per-token r.batchUNKeys[m] buffers, never through
	// setPos, so r.curNKeys (planNKeys, canUseAttnFA's depth gate) would otherwise stay stale from whatever a
	// prior single-token call set. Force it to 0 for the whole batch rather than dispatch attention_fa off a stale
	// depth: attention_fa targets decode (M=1), so the shipped kernel is the intended choice here too. The
	// executor does not read this zero: it encodes each job for its own key count (execLoop).
	r.curNKeys = 0

	// Drain and stop the pipelined executor so we have exclusive, synchronous access to the queue
	// and no in-flight or pre-encoded command buffers conflict with our batch.
	r.stopExec()
	if err := r.kvWrite(r.kvSlot, startPos, startPos+n); err != nil {
		return nil, err
	}

	r.ensureBatchCap(n)

	// Populate batch input embeddings and per-token uniforms up front.
	for m, emb := range embeddings {
		if len(emb) != r.H {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != hidden %d", m, len(emb), r.H)
		}
		dst := r.batchX.Floats()[m*r.H : (m+1)*r.H]
		copy(dst, emb)
		pos := startPos + m
		if r.learnedPos {
			r.addLearnedPosTo(dst, pos)
		}
		r.batchUPos[m].SetU32(uint32(pos))
		r.batchUNKeys[m].SetU32(uint32(pos + 1))
		scale := float32(1)
		if r.attnTempBeta != 0 {
			scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(pos)/r.attnTempOrigMaxPos)))
		}
		r.batchUQTempScale[m].Floats()[0] = scale
	}

	// Encode all N tokens in LAYER-MAJOR order into a SINGLE command buffer and single compute encoder.
	// Layer l's weights stay hot in GPU cache/memory across all N tokens instead of streaming the
	// full multi-GB model weights N separate times.
	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		for m := range n {
			r.encodeLayerResidualWith(e, l, r.batchX.At(m*r.H*4), r.batchUPos[m], r.batchUNKeys[m], r.batchUQTempScale[m])
		}
	}
	for m := range n {
		r.encodeNorm(e, r.batchX.At(m*r.H*4), r.finalNorm, r.finalNormBias, r.aq, r.aSc)
		e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.batchLogits.At(m*r.V*4), r.uH)
	}
	e.End()

	r.recordExecErr(e.Err())
	if err := r.takeExecErr(); err != nil {
		return nil, err
	}
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()

	// Read back and finalize logits for each token.
	out := make([][]float32, n)
	bLogits := r.batchLogits.Floats()
	for m := range n {
		row := make([]float32, r.V)
		copy(row, bLogits[m*r.V:(m+1)*r.V])
		if r.finalSoftcap > 0 {
			softcapParallel(row, r.finalSoftcap)
		}
		if r.logitScale != 0 && r.logitScale != 1 {
			for j, v := range row {
				row[j] = v * r.logitScale
			}
		}
		out[m] = row
	}

	return out, nil
}
