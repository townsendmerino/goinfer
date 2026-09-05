//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// errPrefillDeclined marks an ARCH/GEOMETRY decline from the batched path (MoE, K=V, non-int4,
// non-uniform geometry, or missing batched kernels) — as opposed to a real compute/cap error. The
// batched forward covers the plain dense unfused family only; callers that can fall back to the
// sequential per-token path (ForwardN for spec verify, model.go for prefill) test errors.Is(err,
// errPrefillDeclined) to distinguish "this arch can't batch, use the slow path" (recoverable) from
// "the batched kernels failed" (propagate). Wrapped into each decline so the message stays specific.
var errPrefillDeclined = errors.New("cuda prefill: batched path declined (arch/geometry)")

// errPrefillOOM narrows errPrefillDeclined to the one decline that is a function of M rather than of
// the model: the [M, inter]-sized scratch prefillCore allocates did not fit beside the weights and
// the KV. It is wrapped alongside errPrefillDeclined (so every existing errors.Is check keeps its
// meaning) purely so prefillChunked can tell "this prompt is too long for one pass, halve it" from
// "this model can never batch", which must not be retried at all.
var errPrefillOOM = errors.New("cuda prefill: M-sized scratch did not fit")

// prefillDefaultChunk is the default number of prompt rows per batched pass.
//
// WHY CHUNKING EXISTS. prefillCore's scratch is O(M·inter): at Qwen2.5-7B's inter=18944 it is
// ~278 KB per row, so an 8k prompt asks for 2.28 GB on top of the weights and the KV. MEASURED on
// this box (RTX 2070 SUPER, 8 GB, qwen2.5-7b-instruct-q4_k_m at int4, ResidentContext=8192, 1.96 GB
// free after load — docs/measurements/prefill-chunking-2026-09-04/):
//
//	M=512   batched     2.776 ms/token      M=512   sequential  12.485 ms/token
//	M=2048  batched     3.543 ms/token      M=2048  sequential  13.850 ms/token
//	M=4096  batched     4.440 ms/token
//	M=8012  DECLINED — cuMemAlloc_v2 CUDA_ERROR_OUT_OF_MEMORY on the 607 MB gate buffer
//
// So the batched path passed its LOAD-time report ("batched (one weight-stationary CUDA pass)") and
// then declined every prompt long enough to need it, falling back — silently, since
// residentPrefillSeed discards the decline — to the ~4.5× slower per-token loop. The failure grows
// with the prompt: it is exactly the deep-context cell where TTFT matters most that lost the path.
//
// WHY 512. The per-token cost above is a + b·(average attended keys), and attention is charged per
// position against its own prefix whatever the chunking, so chunk size buys nothing there — it only
// sets how many times each weight is re-read. Fitting the three batched points gives a ≈ 2.52
// ms/token of weight+glue work and b ≈ 1.0 µs/key; at 512 rows each weight is already amortized
// 512-fold, which is within a hair of the M→∞ limit, and the scratch is ~146 MB rather than 2.3 GB.
// A 2048-token prompt therefore costs the same chunked as it did in one pass (predicted 3.54 vs
// measured 3.543 ms/token), so this is not a trade against the lengths that already worked.
// GOINFER_PREFILL_CHUNK overrides it (0 or unset = this default).
const prefillDefaultChunk = 512

// prefillMinChunk is the floor prefillChunked halves down to before giving up and letting the caller
// take the sequential path. At 32 rows each weight is still read once per 32 tokens rather than once
// per token, so the floor is set by diminishing returns and not by correctness; below it the batched
// path's fixed per-pass cost stops paying for itself.
const prefillMinChunk = 32

// prefillProf accumulates PrefillLast's per-category GPU time (test-only). The boundaries are stream
// syncs, so the category sum slightly exceeds the pipelined wall time (lost launch overlap) — it
// attributes where the time goes, not the fully-overlapped total.
type prefillProf struct {
	gemv, attn, glue time.Duration
}

type profCat int

const (
	gemvCat profCat = iota
	attnCat
	glueCat
)

// profTic syncs the stream and returns a start time when profiling is on; a no-op zero Time otherwise.
func (r *cudaResident) profTic() time.Time {
	if r.prof == nil {
		return time.Time{}
	}
	_ = r.stream.Sync()
	return time.Now()
}

// profToc syncs the stream and adds the elapsed time to the named category (no-op when profiling off).
func (r *cudaResident) profToc(cat profCat, t0 time.Time) {
	if r.prof == nil {
		return
	}
	_ = r.stream.Sync()
	d := time.Since(t0)
	switch cat {
	case gemvCat:
		r.prof.gemv += d
	case attnCat:
		r.prof.attn += d
	case glueCat:
		r.prof.glue += d
	}
}

// PrefillLast (decoder.Prefiller) ingests a whole prompt in ONE weight-stationary pass and returns the
// logits for the last token — the batched (M=len) counterpart of the sequential ForwardNoLogits loop.
// It fixes the ~128-token Ollama crossover: goinfer's sequential prefill reads every weight once PER
// PROMPT TOKEN (weight-bandwidth-bound at ~6 ms/token), while this reads each weight once for all M
// tokens (the gemv_w4a8_batched amortization). The K/V it writes is BIT-IDENTICAL to the sequential
// path row-for-row (every batched kernel is the M=1 kernel with an M dimension, per-row math verbatim),
// so decode from the last prompt token stays byte-identical.
//
// It handles the plain dense unfused forward only (Llama/Qwen2/Mistral-class): rmsnorm→Q/K/V→rope+kv→
// causal windowed attention→o-proj→rmsnorm→gate/up→swiglu→down. Anything it does not cover — MoE, the
// Gemma parallel dense‖MoE, sandwich norms, per-head QK-norm, K=V (Gemma), int8 weights, non-uniform
// per-layer geometry, or a prompt past the KV cap — returns an error so decoder/model.go falls back to
// the sequential KV-only prefill (which is correct for every family). Uniform-only is enforced against
// layer 0; a non-uniform family trips the guard and declines rather than reading a wrong stride.
// PrefillLast ingests a whole prompt in one batched pass, returning the last token's logits.
func (r *cudaResident) PrefillLast(embeddings [][]float32, startPos int) ([]float32, error) {
	return r.prefillChunked(embeddings, startPos)
}

// prefillChunkRows is the row budget for one batched pass — prefillDefaultChunk unless
// GOINFER_PREFILL_CHUNK says otherwise. An unparseable or non-positive value is ignored rather than
// failing the request: this is a tuning knob on a path that has a correct fallback, so a typo in it
// must not be the thing that takes a model off the fast path.
func prefillChunkRows() int {
	if n, err := strconv.Atoi(os.Getenv("GOINFER_PREFILL_CHUNK")); err == nil && n > 0 {
		return n
	}
	return prefillDefaultChunk
}

// prefillChunked runs the prompt through the batched path in passes of at most prefillChunkRows()
// rows, returning the LAST row's logits. Every pass writes its own K/V at absolute positions
// startPos+i…, and attention reads the cache — so pass k attends the keys passes 0…k-1 wrote exactly
// as one M=len pass would have, and the result is bit-identical to the unchunked path
// (TestPrefillChunked_bitIdentical). What changes is only the peak scratch, which is what the
// unchunked path ran out of.
//
// A chunk that OOMs is retried at half the width from the SAME position: the passes already done are
// committed to the positional KV and stay valid, so a retry re-enters at the boundary rather than
// restarting. Only errPrefillOOM is retried; a static decline is M-independent and would spin, so it
// is checked once up front and returned.
//
// NOT CHUNKED when a batched hidden-state capture is armed (r.capBTaps): those taps record the
// residual for ALL M rows of one pass, and a chunked run would leave a block drafter holding the last
// chunk's rows only. That combination is not reachable today — the capture is armed by the verify
// entry points, not by PrefillLast — but a partial capture would be a silent wrong answer rather than
// a slow one, so it declines to one pass instead of being merely documented as unreachable.
func (r *cudaResident) prefillChunked(embeddings [][]float32, startPos int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	chunk := prefillChunkRows()
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned // a previous prompt already found the default too wide for this card
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, e
	}
	if M <= chunk || len(r.capBTaps) > 0 {
		outs, _, err := r.prefillCore(embeddings, startPos, tailLastLogits)
		if err != nil {
			return nil, err
		}
		return outs[len(outs)-1], nil
	}
	for i := 0; i < M; {
		n := chunk
		if i+n > M {
			n = M - i
		}
		last := i+n == M
		tail := tailKVOnly
		if last {
			tail = tailLastLogits
		}
		outs, _, err := r.prefillCore(embeddings[i:i+n], startPos+i, tail)
		if err != nil {
			if errors.Is(err, errPrefillOOM) && chunk > prefillMinChunk {
				chunk = max(chunk/2, prefillMinChunk)
				r.prefillChunkCap.Store(int64(chunk)) // remember: do not re-OOM on the next prompt
				continue                              // same i: nothing was committed by the pass that failed to allocate
			}
			return nil, err
		}
		if last {
			return outs[len(outs)-1], nil
		}
		i += n
	}
	// Unreachable: the i+n==M pass returns above. Kept as a real error rather than a panic so a
	// future edit to the loop bounds surfaces as a decline the caller can survive.
	return nil, fmt.Errorf("cuda prefill: chunked loop ended without heading the last row (M=%d)", M)
}

// PrefillLastN is the D1 (speculative-decode) verify primitive: the SAME batched pass, but returns
// the logits at ALL M positions (row m = the target's prediction for position startPos+m+1). The
// batched layer stack is bit-identical to sequential per position (TestPrefillLast_e2e), and the
// final norm + LM head is applied per row exactly as PrefillLast applies it to the last — so each
// row's logits equal a sequential Forward's, which is what makes greedy accept lossless.
func (r *cudaResident) PrefillLastN(embeddings [][]float32, startPos int) ([][]float32, error) {
	outs, _, err := r.prefillCore(embeddings, startPos, tailAllLogits)
	return outs, err
}

// PrefillLastNArgmax is the spec-decode VERIFY primitive: the same batched pass, returning only
// each row's argmax token id — which is all the accept decision needs.
//
// It exists because PrefillLastN's per-row tail re-reads the LM head's weights ONCE PER ROW
// (~389 M params, ~195 MB at int4), measured at 1.046 ms marginal per row against a 0.934 ms
// single-row head — no amortization at all, in the one place the batched pass exists to provide
// it. This tail instead runs ONE batched final-norm, ONE batched head GEMV over all M rows, and M
// argmax reductions, then reads back 4 bytes per row instead of 608 KB.
//
// LOSSLESSNESS: the accept decision compares the drafted token against the target's argmax, so
// only the ARGMAX must match the sequential path — not the logits bit-for-bit. That is a strictly
// weaker requirement than PrefillLastN's, and it is what makes batching the head admissible at
// all. TestPrefillLastNArgmax_matchesPerRow gates it.
func (r *cudaResident) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	_, ids, err := r.prefillCore(embeddings, startPos, tailAllArgmax)
	return ids, err
}

// PrefillSeedArgmax satisfies decoder.ResidentSeedArgmax: the same batched forward and the same
// batched capture, but the head runs over ONE row.
//
// The block-spec prompt seed asked for M rows of argmax and read only the last. At vocab 151,936 a
// 2048-token prompt therefore allocated 1.24 GB of VRAM for the batched logits, a 1.24 GB host
// slice and a 1.24 GB D2H, ran the head GEMV over 2048 rows and a single-threaded host argmax over
// 311M floats — for one token id. logitsB is grow-only, so each longer prompt also abandoned its
// predecessor; a 4096-token prompt on an 8 GB card OOM'd inside the executor, and per backend.go's
// A13 a context driven to refusal and kept in use can afterwards launch kernels that "return
// SUCCESS and execute NOTHING" (audit-2026-09-02 C-12).
//
// tailLastLogits, not a new kernel path: it ALREADY heads the last row only, and the seed's other
// requirement — the batched capture — comes from the layer loop either way. One row of logits is
// vocab floats (~0.6 MB) against M x vocab.
func (r *cudaResident) PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error) {
	outs, _, err := r.prefillCore(embeddings, startPos, tailLastLogits)
	if err != nil {
		return 0, err
	}
	if len(outs) == 0 || outs[len(outs)-1] == nil {
		return 0, fmt.Errorf("cuda prefill: seed argmax got no logits for the last row")
	}
	row := outs[len(outs)-1]
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	return best, nil
}

// prefillStaticDecline reports why the batched path can't run for THIS model, or nil when it can. It
// covers exactly the MODEL-dependent guards — arch, per-layer geometry, weight kind, kernel
// availability — and deliberately not the prompt-dependent one (checkCap, which needs M/startPos).
// That split is what lets PrefillPath answer at LOAD time from the same code prefillCore enforces at
// call time: one source of truth, so the startup line can never drift from the actual decline.
//
// qk-norm (per-head Q/K RMSNorm) and Gemma sandwich norms are batched (qk_norm_batched /
// rmsnorm_f32_batched) — neither declines. backend.go asserts qNorm/kNorm length == headDim before
// residency (⇒ present when r.qkNorm), and the per-layer K=V / non-uniform / int4 checks still catch
// a Gemma-4-class layer this dense-batched path can't stride. MoE and the Gemma parallel dense‖MoE
// take the sequential path.
func (r *cudaResident) prefillStaticDecline() error {
	if !r.prefillReady {
		return fmt.Errorf("cuda prefill: batched kernels unavailable: %w", errPrefillDeclined)
	}
	// MoE is no longer a categorical refusal: a MoE layer's FFN runs ROW BY ROW off the batched
	// residual (prefillCore), so the attention half batches and the routed experts keep the exact
	// per-token sequence decode uses. What must still decline are the PER-TOKEN DEBUG SEAMS, which
	// that row loop would fire M times per layer instead of once: hidCapTaps records one residual
	// per tap per TOKEN for a block drafter, and layerCap appends one snapshot per layer for the
	// divergence probe. Both would silently return M× the rows their consumers expect. Declining
	// sends those runs down the sequential path, where their semantics are the ones they were
	// written against.
	if len(r.hidCapTaps) > 0 {
		return fmt.Errorf("cuda prefill: per-token hidden-state taps are armed: %w", errPrefillDeclined)
	}
	if r.layerCap {
		return fmt.Errorf("cuda prefill: per-layer residual capture is armed: %w", errPrefillDeclined)
	}
	// PER-LAYER geometry, not layer 0's hoisted and asserted uniform. The batched launches bind
	// each layer's own hd/nKV/qDim/kvDim/rhalf exactly as the decode launches already do, and the
	// M-sized scratch is sized by the MAX across layers — so a family whose layers differ (Gemma-4:
	// 5 of its 30 are full_attention with a different KV width) strides correctly instead of being
	// refused. What made the old uniform assertion necessary was hoisting L0 into every launch;
	// remove the hoist and the assertion has nothing left to protect.
	for l := range r.layers {
		Ly := &r.layers[l]
		// K=V (Gemma-4 global layers) is handled in the batched pass the same ~10 lines decode
		// handles it in (segA): a second k-projection GEMV into the V buffer, then a scale-less
		// v_norm over it BEFORE rope rotates k. Both kernels already take an M dimension, so this
		// costs no new kernel — but it does need the unit-weight buffer segA uses, which is
		// allocated only when some layer is kEqV. Refuse rather than bind a null weight.
		if Ly.kEqV && r.vNormUnit == (Buffer{}) {
			return fmt.Errorf("cuda prefill: K=V layer at %d but no v_norm unit weight: %w", l, errPrefillDeclined)
		}
		if k := nonBatchableKind(Ly); k != "" {
			return fmt.Errorf("cuda prefill: %s weight at layer %d needs the sequential path: %w", k, l, errPrefillDeclined)
		}
	}
	return nil
}

// prefillMaxGeom returns the largest qDim and kvDim across the layers — the row strides the M-sized
// Q/K/V/context scratch must be allocated at once geometry is per-layer. Each launch still binds its
// OWN layer's dims, so a narrower layer simply uses a prefix of each row; the buffers are per-pass
// scratch with no cross-layer meaning, so that is safe and is what lets one allocation serve a
// non-uniform stack.
func (r *cudaResident) prefillMaxGeom() (maxQDim, maxKvDim int) {
	for l := range r.layers {
		maxQDim = max(maxQDim, r.layers[l].qDim)
		maxKvDim = max(maxKvDim, r.layers[l].kvDim)
	}
	return maxQDim, maxKvDim
}

// checkPrefillShmem is prefillStaticDecline's PROMPT-dependent twin (V-05, docs/review-2026-09-04.md):
// the single-block batched-prefill attention launch sizes its dynamic shared memory the same way
// decode's does, (maxNWin+128)*4 bytes, but — unlike decode — has no split-KV fallback kernel, so
// there is no case where exceeding singleBlockAttnShmemLimit is survivable here. Without this check
// the launch itself failed at the driver, prefillCore returned an unnamed error, and the caller
// (decoder/model.go's PrefillLast handling) silently fell through to the ~9x-slower sequential
// per-token path with nothing distinguishing "declined" from "crashed". Checked per layer because a
// sliding-window layer's maxNWin is clamped to its own window and may stay under the limit even when
// a global layer in the SAME model does not — mirrors the launch site's own per-layer maxNWin
// computation in prefillCore exactly, so this can never decline a shape the launch would have run,
// or miss one it would have failed.
func (r *cudaResident) checkPrefillShmem(startPos, M int) error {
	for l := range r.layers {
		Ly := &r.layers[l]
		maxNWin := startPos + M
		if Ly.window > 0 && int(Ly.window) < maxNWin {
			maxNWin = int(Ly.window)
		}
		if splitKVRequired(maxNWin) {
			return fmt.Errorf("cuda prefill: layer %d attention at %d attended keys needs %d B of "+
				"shared memory, past this device's %d B limit — batched prefill has no split-KV "+
				"path, so this prompt length needs the sequential path: %w",
				l, maxNWin, attnShmemBytes(maxNWin), singleBlockAttnShmemLimit, errPrefillDeclined)
		}
	}
	return nil
}

// nonBatchableKind returns the weight kind of the layer's first projection the batched GEMVs can't
// handle, or "" when every projection is int4 or int8. The batched path dispatches per projection
// (bGemvB): int4 → gemv_w4a8_batched/_rn (group-scaled float accumulate), int8 → gemv_w8a8_batched
// (exact int32, §C6). A uniformly-int4, uniformly-int8, or MIXED int4mix bundle all batch; anything
// else (e.g. a native/f32 projection) declines. Naming the kind turns "declined" into an actionable
// startup message.
func nonBatchableKind(Ly *cudaLayer) string {
	// CHECK EXACTLY THE PROJECTIONS THIS LAYER'S BATCHED PATH WILL BIND, and no others. Two of them
	// are legitimately ABSENT on shapes that now reach here, and an absent weight has kind "" —
	// which this function used to return and its caller reads as "no problem" (`if k := …; k != ""`).
	// That made the check pass vacuously rather than catch anything, so tightening the sentinel
	// without narrowing the list would swap a silent hole for a wrong refusal:
	//
	//   - Ly.v is absent on a K=V layer (V is derived from the k projection), and
	//   - Ly.g/u/d are absent on a pure-MoE layer, whose FFN runs per row through doG on the
	//     expert stacks and never touches a dense gate/up/down.
	//
	// The expert stacks themselves are deliberately NOT checked: the per-row MoE FFN issues exactly
	// decode's launches on exactly decode's weights, so whatever kind decode accepts, it accepts
	// here. Only the M-wide GEMVs (bGemvB, int4/int8 only) constrain anything, and those are the
	// attention projections plus a dense layer's gate/up/down.
	ws := []cudaWQ{Ly.q, Ly.k, Ly.o}
	if !Ly.kEqV {
		ws = append(ws, Ly.v)
	}
	if !Ly.g4moe && !Ly.isMoE {
		ws = append(ws, Ly.g, Ly.u, Ly.d)
	}
	for _, w := range ws {
		if w.kind != "int4" && w.kind != "int8" {
			if w.kind == "" {
				return "absent/unquantized"
			}
			return w.kind
		}
	}
	return ""
}

// PrefillPath (decoder.PrefillPathReporter) answers, at load, whether this model will get the batched
// prefill — before a single request has been served. Batched prefill now covers int4 AND int8 bundles
// (§C6); it still declines a native/f32 projection or a non-uniform/K=V geometry. Before int8 batched
// prefill landed, a dense model loaded at int8int8 built a fully resident decode path (looked healthy,
// decoded at 0.7× int4) but every prompt took the sequential per-token prefill — measured 1.73 s vs
// 0.19 s on a 300-token prompt (9×), 20× the CPU (4.56 vs 0.22 CPU-s), no compute hotspot: the executor
// spin-waiting through 300 sequential launches instead of one pass.
func (r *cudaResident) PrefillPath() (bool, string) {
	err := r.prefillStaticDecline()
	if err == nil {
		// Say ROWS PER PASS, not "one pass". The report is read as a promise about how a long prompt
		// is ingested, and a prompt past the chunk width is now several weight-stationary passes over
		// the positional KV rather than one — same numbers (TestPrefillChunked_bitIdentical), bounded
		// scratch. Claiming "one pass" was what let the O(M·inter) OOM decline hide behind a green
		// startup line for every prompt long enough to matter.
		return true, fmt.Sprintf("batched (weight-stationary CUDA passes of up to %d rows)", prefillChunkRows())
	}
	// Detail without the wrapped sentinel, which says nothing a user can act on.
	detail := strings.TrimPrefix(strings.TrimSuffix(err.Error(), ": "+errPrefillDeclined.Error()), "cuda prefill: ")
	if len(r.layers) > 0 {
		if k := nonBatchableKind(&r.layers[0]); k != "" {
			return false, fmt.Sprintf("sequential — batched prefill needs int4 or int8 projections (%s weights) — ~9× slower TTFT, 20× CPU on a 300-token prompt", k)
		}
	}
	return false, "sequential — " + detail + " (slower TTFT: one forward per prompt token)"
}

// prefillCore runs the batched (M=len) forward. allLogits=false heads only the last row (PrefillLast);
// allLogits=true heads every row (PrefillLastN, spec-decode verify).
// prefill tail modes: what the batched pass does after the layer stack.
const (
	tailLastLogits = iota // head the LAST row only (PrefillLast)
	tailAllLogits         // head every row, per-row loop, full logits (PrefillLastN)
	tailAllArgmax         // batched head over all rows, return argmax ids (PrefillLastNArgmax)
	tailKVOnly            // no head at all: the pass exists only to commit K/V (prefillChunked's
	// non-final chunks). It is the batched twin of the sequential path's
	// ForwardNoLogits — the final norm, the ~389 M-parameter head GEMV and the
	// [M, hidden] readback are all dead work for a chunk whose logits nobody reads.
)

func (r *cudaResident) prefillCore(embeddings [][]float32, startPos int, tail int) ([][]float32, []int, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, nil, e
	}
	if e := r.checkCap(startPos, M); e != nil {
		return nil, nil, e
	}
	if e := r.checkPrefillShmem(startPos, M); e != nil {
		return nil, nil, e
	}
	maxQDim, maxKvDim := r.prefillMaxGeom()
	hidden, inter := r.hidden, r.inter

	var outs [][]float32
	var ids []int
	err := r.do(func() error {
		r.launchErr = nil // N-04: clear the sticky accumulator first (like launchToken), so a prior
		// decode's discarded launch error isn't re-reported by this prefill.
		// --- M-sized scratch (device), freed at the end.
		//
		// The free list and its defer are registered BEFORE the first allocation, and each buffer
		// joins the list as it is created (audit C-24). Allocation PANICS on OOM per
		// gpu.NewBufferLenOf's contract, and at M=3000 this is hundreds of MB, so a partial
		// allocation is the expected failure on a nearly-full card — not a rare one. Building the
		// list first and deferring after (the previous shape) freed nothing at all when allocation
		// #10 of 17 panicked, because the defer had not been registered yet. That leaked only
		// because the panic used to kill the process anyway; now that runJob recovers it into a
		// decline, the leak would be real, repeatable, and would push the NEXT prompt closer to OOM.
		var scratch []Buffer
		defer func() {
			for _, b := range scratch {
				r.dev.ReleaseBuf(b)
			}
		}()
		af := func(n int) Buffer { b := r.af(n); scratch = append(scratch, b); return b }
		ai := func(n int) Buffer { b := r.ai(n); scratch = append(scratch, b); return b }

		xB := af(M * hidden)
		aqB, aScB := ai(M*hidden/4), af(M)
		qBb, kBb, vBb := af(M*maxQDim), af(M*maxKvDim), af(M*maxKvDim)
		cctxB := af(M * maxQDim)
		cqB, cScB := ai(M*maxQDim/4), af(M)
		mqB, mScB := ai(M*hidden/4), af(M)
		gOb, uOb := af(M*inter), af(M*inter)
		dqB, dScB, dScrB := ai(M*inter/4), af(M), af(M*inter)
		// Sandwich families (Gemma) norm the attention / MLP sublayer output BEFORE adding it to the
		// residual, so the o-proj and down GEMVs write a temp instead of accumulating in place. One
		// [M, hidden] buffer, reused for both (the two uses are sequential).
		var sbB Buffer
		if r.sandwich {
			sbB = af(M * hidden)
		}
		residMN := uint32((M*hidden + 255) / 256)

		// Upload the M embeddings contiguously as xB[M, hidden] (already FeatEmbedScale-scaled by the
		// caller, exactly as Forward/ForwardNoLogits receive them).
		xhost := make([]float32, M*hidden)
		for m, e := range embeddings {
			copy(xhost[m*hidden:(m+1)*hidden], e)
		}
		if e := gpu.Upload(xB, xhost); e != nil {
			return e
		}

		for l := 0; l < r.nLayers; l++ {
			Ly := &r.layers[l]
			// PER-LAYER, read here and bound below — never L0's hoisted and assumed uniform.
			hd, nKV, qDim, rhalf := Ly.hd, Ly.nKV, Ly.qDim, Ly.rhalf
			ropeN := r.nH*rhalf + nKV*rhalf + nKV*(hd-2*rhalf)
			qb, kb, vb := ArgNull(), ArgNull(), ArgNull()
			if Ly.hasBias {
				qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
			}
			// segA: rmsnorm+quant (glue), then Q/K/V GEMVs. Category timers (r.prof) sync r.stream at
			// each group boundary; nil in production, so the launch sequence is otherwise unchanged.
			t := r.profTic()
			if e := r.bRmsB(xB, Ly.preNorm, hidden, aqB, aScB, M); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			t = r.profTic()
			if e := r.bGemvB(Ly.q, aqB, aScB, qb, qBb, M, 0); e != nil {
				return e
			}
			if e := r.bGemvB(Ly.k, aqB, aScB, kb, kBb, M, 0); e != nil {
				return e
			}
			if Ly.kEqV {
				// K=V (Gemma-4 global layers): this layer has NO v_proj. V is v_norm(the RAW
				// pre-RoPE k_proj output), so recompute the k projection into the V buffer here and
				// normalize it below, before rope_kv_batched rotates k. Mirrors segA's decode path
				// launch for launch; a SECOND GEMV rather than a copy of kBb because that is what
				// decode does, and the two paths must not differ by so much as an operation order.
				if e := r.bGemvB(Ly.k, aqB, aScB, kb, vBb, M, 0); e != nil {
					return e
				}
			} else if e := r.bGemvB(Ly.v, aqB, aScB, vb, vBb, M, 0); e != nil {
				return e
			}
			r.profToc(gemvCat, t)
			// per-head Q/K RMSNorm BEFORE rope (Qwen3): in place on qBb/kBb, one block per (head,token).
			// Bit-identical to the decode qk_norm applied per token (same f64 reduction, same addOne).
			if r.qkNorm {
				t = r.profTic()
				addOne := int32(0)
				if r.rmsAddOne {
					addOne = 1
				}
				if e := r.launch(r.bQKN, LaunchConfig{GridX: uint32(r.nH + nKV), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
					Arg(qBb), Arg(kBb), Arg(Ly.qNorm), Arg(Ly.kNorm),
					gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
					gpu.ArgValue(r.eps), gpu.ArgValue(addOne), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
				r.profToc(glueCat, t)
			}
			if Ly.kEqV {
				// Scale-less v_norm over the raw k sitting in vBb, BEFORE rope rotates k — segA's
				// decode launch with an M dimension added. nH=0 makes qk_norm_batched treat every
				// block as a K-head (base = v + m*kvDim + h*hd), and vNormUnit is a unit weight so
				// addOne=0 gives a pure RMS scale with no learned gain.
				t = r.profTic()
				if e := r.launch(r.bQKN, LaunchConfig{GridX: uint32(nKV), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
					Arg(vBb), Arg(vBb), Arg(r.vNormUnit), Arg(r.vNormUnit),
					gpu.ArgValue(int32(0)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
					gpu.ArgValue(r.eps), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
				r.profToc(glueCat, t)
			}
			// rope + kv-store (glue): token m at absolute position startPos+m; rotates q/k, writes K/V.
			t = r.profTic()
			if e := r.launch(r.bRopeKV, LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(qBb), Arg(kBb), Arg(vBb), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.vc[l]),
				gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rhalf)), gpu.ArgValue(int32(M)),
				gpu.ArgValue(Ly.mscale)); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			// causal + per-row sliding-window attention; block 128 matches the M=1 attention reduce.
			maxNWin := startPos + M
			if Ly.window > 0 && int(Ly.window) < maxNWin {
				maxNWin = int(Ly.window)
			}
			t = r.profTic()
			if e := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((maxNWin + 128) * 4)},
				Arg(qBb), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)),
				gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(r.attnScale),
				// N-10: r.sinkArg(l), not ArgNull(). The decode launches thread the gpt-oss
				// learned sink through and this one hard-coded null — unreachable today only
				// because every gpt-oss model is MoE and MoE declines batched prefill, which
				// is a property of a DIFFERENT check and not something this call site should
				// depend on.
				gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB), r.sinkArg(l)); e != nil {
				return e
			}
			r.profToc(attnCat, t)
			// segB: ctx-quant (glue), o-proj (gemv, accum into residual), MLP.
			t = r.profTic()
			if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
				Arg(cctxB), gpu.ArgValue(int32(qDim)), Arg(cqB), Arg(cScB), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			t = r.profTic()
			if r.sandwich {
				// o-proj → temp (accum=0), Gemma post-attn RMSNorm per row, then add to residual.
				// Mirrors segB's decode sandwich path (o-proj → normF32 → residual) exactly, per row.
				if e := r.bGemvB(Ly.o, cqB, cScB, r.oBiasArg(Ly), sbB, M, 0); e != nil {
					return e
				}
				if e := r.bNormF32B(sbB, Ly.postAttnNorm, hidden, M); e != nil {
					return e
				}
				if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
					Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
					return e
				}
			} else if e := r.bGemvB(Ly.o, cqB, cScB, r.oBiasArg(Ly), xB, M, 1); e != nil {
				return e
			}
			r.profToc(gemvCat, t)

			// --- FFN. Dense batches; MoE runs ROW BY ROW off the batched residual. ---
			//
			// The routed-expert GEMVs are indexed by a DEVICE-side routing decision that differs per
			// token, so there is no M-wide form of them without an expert-major gather and a new
			// kernel (queue-performance P20 step 2). What there IS, for free, is the attention half
			// above: on Gemma-4-26B-A4B the attention projections are ~45% of the per-token weight
			// traffic and the dense FFN branch another ~25%, all of it re-read once per token on the
			// sequential path and once per PASS here.
			//
			// aikit/gpu.Buffer.At gives a zero-copy sub-view that binds as a raw device pointer, so
			// row m of the batched residual IS a valid single-row residual for the existing per-token
			// FFN chain — segBFFN → layerTail → segC, the same calls decode makes, in the same order,
			// including the g4x2 accumulator clear and the C′ routed-expert DMA. Nothing about the
			// expert path changes; it simply no longer drags the attention weights along with it.
			//
			// gC=false: prefill never replays captured graphs (a graph bakes r.x, and these rows are
			// not r.x). The per-token debug seams layerTail also carries — hidCapTaps, layerCap —
			// would fire M times per layer here, which is why prefillStaticDecline refuses a model
			// with either armed rather than quietly returning M× the rows they expect.
			if Ly.g4moe || Ly.isMoE {
				t = r.profTic()
				for m := 0; m < M; m++ {
					xm := xB.At(m * hidden * 4)
					if e := r.segBFFN(Ly, l, xm); e != nil {
						return e
					}
					if e := r.layerTail(Ly, l, false, xm); e != nil {
						return e
					}
				}
				r.profToc(gemvCat, t)
			} else {
				t = r.profTic()
				if e := r.bRmsB(xB, Ly.postNorm, hidden, mqB, mScB, M); e != nil {
					return e
				}
				r.profToc(glueCat, t)
				t = r.profTic()
				if e := r.bGemvB(Ly.g, mqB, mScB, ArgNull(), gOb, M, 0); e != nil {
					return e
				}
				if e := r.bGemvB(Ly.u, mqB, mScB, ArgNull(), uOb, M, 0); e != nil {
					return e
				}
				r.profToc(gemvCat, t)
				t = r.profTic()
				if e := r.launch(r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
					Arg(gOb), Arg(uOb), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
					gpu.ArgValue(r.act), Arg(dqB), Arg(dScB), Arg(dScrB), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
				r.profToc(glueCat, t)
				t = r.profTic()
				if r.sandwich {
					// down → temp (accum=0), Gemma post-MLP RMSNorm per row, then add to residual.
					if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), sbB, M, 0); e != nil {
						return e
					}
					if e := r.bNormF32B(sbB, Ly.postMLPNorm, hidden, M); e != nil {
						return e
					}
					if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
						Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
						return e
					}
				} else if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), xB, M, 1); e != nil {
					return e
				}
				r.profToc(gemvCat, t)
			}
			// BATCHED HIDDEN-STATE CAPTURE (P10). The per-token seam (capVec) syncs and
			// downloads once per TAP PER TOKEN; a block drafter needs the taps for every token
			// the verify commits, so on this path that is 5 taps x M tokens of stalls. Here the
			// residual for all M rows is already in xB, so one download per tap covers the whole
			// block — 5 downloads per verify instead of 5*M.
			if len(r.capBTaps) > 0 {
				for slot, tap := range r.capBTaps {
					if tap != l {
						continue
					}
					if e := r.stream.Sync(); e != nil {
						return e
					}
					buf := make([]float32, M*hidden)
					if e := gpu.Download(xB, buf); e != nil {
						return e
					}
					r.capBOut[slot] = buf
					break
				}
			}
		}

		// Final norm + LM head, per row — copy xB[m] into the M=1 scratch and reuse the exact Forward
		// tail, so each row's logits are bit-identical to a sequential Forward at position startPos+m
		// (given identical residual, which the KV/logits gate checks). allLogits=false heads only the
		// last row (the crossover-fixing PrefillLast); allLogits=true heads every row (verify). Drain
		// the layer launches first: they run on r.stream, and the DtoH below is not ordered after it.
		if e := r.stream.Sync(); e != nil {
			return e
		}
		// KV-only chunk: the layer stack has run and its K/V is committed, which is the entire point
		// of a non-final chunk. Return before the [M, hidden] readback and the head. The sync above
		// has already drained the launches, so r.launchErr is complete here.
		if tail == tailKVOnly {
			return r.launchErr
		}
		if e := gpu.Download(xB, xhost); e != nil {
			return e
		}
		if tail == tailAllArgmax {
			return r.batchedHeadArgmax(xB, aqB, aScB, M, &ids)
		}
		first := M - 1
		if tail == tailAllLogits {
			first = 0
		}
		outs = make([][]float32, M)
		for m := first; m < M; m++ {
			if e := gpu.Upload(r.x, xhost[m*hidden:(m+1)*hidden]); e != nil {
				return e
			}
			if e := r.rms(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
				return e
			}
			if e := r.doG(r.lmW, r.aq, r.aSc, ArgNull(), r.logits, 0); e != nil {
				return e
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
				return e
			}
			outs[m] = append([]float32(nil), r.logitsHost...)
		}
		return r.launchErr
	})
	if err != nil {
		// An OOM inside the job arrives as a recovered panic (runJob, audit C-24) carrying aikit
		// MustBuf's "device allocation failed" message. For prefill specifically that is a DECLINE, not
		// a request failure: the sequential per-token path needs no M-sized scratch and will serve this
		// prompt. Match that OOM SENTINEL, not any "panicked" (audit R-20): a future programming-bug
		// panic in the batched path must surface as a real error, not be silently absorbed into the
		// ~9×-slower sequential path. Errors that are already declines (static guards, checkCap) keep
		// their own wrapping.
		if strings.Contains(err.Error(), "device allocation failed") && !errors.Is(err, errPrefillDeclined) {
			return nil, nil, fmt.Errorf("cuda prefill: out of device memory for M=%d scratch (%w; %w): %v", M, errPrefillDeclined, errPrefillOOM, err)
		}
		return nil, nil, err
	}
	// Final-logit softcap (Gemma) — host-side, exactly as step(). No-op (0) for the dense families
	// this path serves, but kept so the contract matches Forward if a softcapped dense arch appears.
	for _, out := range outs {
		applySoftcap(out, r.finalSoftcap)
	}
	return outs, ids, nil
}

// batchedHeadArgmax is tailAllArgmax's tail: ONE batched final-norm, ONE batched head GEMV over
// all M rows, M argmax reductions, then a 4-bytes-per-row readback.
//
// The win it exists for: the per-row tail issues the head as an M=1 GEMV per row, so the head's
// ~389 M parameters are re-read from VRAM M times. Measured, that marginal row costs 1.046 ms
// against a 0.934 ms single-row head — no amortization whatsoever, in the one place the batched
// pass exists to provide it.
//
// Buffers are allocated lazily HERE because af/ai need r.dev's context current, which holds on
// the executor thread this runs on. They are sized to M and reused; a wider block reallocates
// once. The old buffers are left to the device ledger rather than freed mid-job, which is the
// same lifetime the rest of the resident scratch has.
func (r *cudaResident) batchedHeadArgmax(xB, aqB, aScB Buffer, M int, out *[]int) error {
	if M > r.logitsBCap {
		// RELEASE BEFORE GROWING. This was grow-only: each larger prompt abandoned the previous
		// buffer to the device ledger, bounded only by 2*ctxCap*vocab*4 B — about 5 GB at the 4096
		// default, on top of the live one (audit-2026-09-02 C-12).
		if r.logitsBCap > 0 {
			r.dev.ReleaseBuf(r.logitsB)
		}
		r.logitsB = r.af(M * r.vocab)
		r.logitsBCap = M
	}
	// Batched final norm + quant over all M rows, exactly as the layer loop norms its input.
	if e := r.bRmsB(xB, r.finalNorm, r.hidden, aqB, aScB, M); e != nil {
		return e
	}
	// ONE head GEMV for all M rows: the weights are read once instead of M times.
	if e := r.bGemvB(r.lmW, aqB, aScB, ArgNull(), r.logitsB, M, 0); e != nil {
		return e
	}
	// The argmax is taken on the HOST, not by M launches of argmax_reduce over slices of the
	// batched logits. gocudrv exposes no buffer view or offset (the same limitation noted for the
	// SwiGLU halves at resident.go), so a per-row reduction would need either a batched argmax
	// kernel or M single-row buffers. Neither is worth it here: the win being chased is the M
	// weight READS of a 195 MB head, and a host argmax over M x vocab costs ~1.1 ms against the
	// ~6.4 ms that recovers. A batched argmax kernel is a later, separate ~1 ms.
	if e := r.stream.Sync(); e != nil {
		return e
	}
	host := make([]float32, M*r.vocab)
	if e := gpu.Download(r.logitsB, host); e != nil {
		return e
	}
	ids := make([]int, M)
	for m := 0; m < M; m++ {
		row := host[m*r.vocab : (m+1)*r.vocab]
		bi, bv := 0, row[0]
		for i, v := range row {
			if v > bv {
				bi, bv = i, v
			}
		}
		ids[m] = bi
	}
	*out = ids
	return r.launchErr
}

// bRmsB launches rmsnorm_quant_batched over M rows (shared = [blockDim]+[hidden]).
func (r *cudaResident) bRmsB(x, w Buffer, N int, qOut, sOut Buffer, M int) error {
	return r.launch(r.bRms, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((256 + N) * 4)},
		Arg(x), Arg(w), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()),
		Arg(qOut), Arg(sOut))
}

// bNormF32B is the batched counterpart of normF32 (Gemma sandwich post-norm): a plain in-place
// f32 RMSNorm of an [M, H] sublayer output, one block per row (grid.y = m), blockDim 256 to match
// the decode reduction tree. No-op when the arch declares no sandwich norms (empty weight buffer).
func (r *cudaResident) bNormF32B(x, w Buffer, H, M int) error {
	if w.Len() == 0 {
		return nil
	}
	return r.launch(r.bNormF32, LaunchConfig{GridX: 1, GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(x), Arg(w), gpu.ArgValue(int32(H)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()), gpu.ArgValue(int32(M)))
}

// rnBlockRows must equal RN in gemv_w4a8_rn.cu — each warp computes this many output rows, so the grid
// covers ceil(N/rnBlockRows) warps. Bit-identical for any RN; 2 is the profiled knee (halves the L1TEX
// load count → halves the scoreboard stall, 4.41→3.38 ms, at the 64-reg / 100%-occupancy limit).
const rnBlockRows = 2

// bGemvB launches the register-blocked gemv_w4a8_rn (RN rows/warp). Bit-identical to the M=1 GEMV
// (each row keeps its own facc across all K, one warp-reduce), just with each activation load reused
// across RN rows. Grid = ceil(N/RN) warps → ceil(that/8) blocks of 256 threads.
func (r *cudaResident) bGemvB(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	switch wt.kind {
	case "int4":
		warps := (wt.N + rnBlockRows - 1) / rnBlockRows
		cfg := LaunchConfig{GridX: uint32((warps + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		return r.launch(r.bRN, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
			gpu.ArgValue(int32(M)), Arg(dst), gpu.ArgValue(accum))
	case "int8":
		// Batched W8A8 (§C6). One warp per output row (8 warps/block), same layout as doG's int8
		// GEMV (wt.ws = per-row f32 scale, K/4 int words). Bit-identical to gemv_w8a8_fwd by
		// construction — exact int32 accumulation, so tiling M cannot change any element.
		cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		return r.launch(r.bW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), gpu.ArgValue(int32(M)),
			Arg(dst), gpu.ArgValue(accum))
	default:
		return fmt.Errorf("cuda prefill: batched GEMV is int4/int8-only, got %q", wt.kind)
	}
}
