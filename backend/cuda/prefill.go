//go:build cuda

package cuda

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// errPrefillDeclined marks an arch or geometry decline from the batched path, which covers the plain dense unfused
// family only, as opposed to a compute or cap error. A caller that can fall back to the sequential path tests
// errors.Is(err, errPrefillDeclined) and recovers; any other error propagates. It is wrapped into each decline so the
// message stays specific.
var errPrefillDeclined = errors.New("cuda prefill: batched path declined (arch/geometry)")

// errPrefillOOM narrows errPrefillDeclined to the one decline that depends on M rather than the model: the [M,
// inter]-sized scratch prefillCore allocates did not fit. It is wrapped alongside errPrefillDeclined so prefillChunked
// can tell "halve the chunk" from "this model can never batch", which must not be retried.
var errPrefillOOM = errors.New("cuda prefill: M-sized scratch did not fit")

// prefillDefaultChunk is the default number of prompt rows per batched pass. prefillCore's scratch is O(M*inter) (~278
// KB per row at inter=18944), so one pass over a long prompt can fail to allocate beside the weights and the KV. At 512
// rows each weight is already amortized close to the M-to-infinity limit and the scratch stays ~146 MB.
// GOINFER_PREFILL_CHUNK overrides it (0 or unset = this default). The measurements and the fit:
// docs/code-notes/cuda.md#prefillDefaultChunk.
const prefillDefaultChunk = 512

// prefillMinChunk is the floor prefillChunked halves down to before declining to the sequential path. It is a
// diminishing-returns floor, not a correctness one: below it the batched pass's fixed per-pass cost stops paying for
// itself.
const prefillMinChunk = 32

// prefillAlignRows is the grid a pass's start stays on: the fused attention's key tile (attnFusedBM). Under a sliding window a
// fused pass that starts off it does not write the bits the whole prompt writes (docs/measurements/kv1-2026-10-13.md).
const prefillAlignRows = 64

// prefillTailCut is how many rows a pass of n rows gives to a last pass that would have rem rows, fewer than attnFusedMinRows.
// It is a whole prefillAlignRows when the pass keeps attnFusedMinRows after it, so the last pass starts on the grid the passes
// before it started on; a pass too narrow for that (a chunk under 80 rows) gives only what the last pass lacks. A pass that
// would itself drop under attnFusedMinRows gives nothing (a chunk under 32 rows, set by GOINFER_PREFILL_CHUNK): both passes
// cannot be fast class there, and the cut would empty the pass when the chunk is under 16.
func prefillTailCut(n, rem int) int {
	if n-prefillAlignRows >= attnFusedMinRows {
		return prefillAlignRows
	}
	if cut := attnFusedMinRows - rem; n-cut >= attnFusedMinRows {
		return cut
	}
	return 0
}

// prefillProf accumulates PrefillLast's per-category GPU time (test-only). The boundaries are stream
// syncs, so the category sum slightly exceeds the pipelined wall time (lost launch overlap) — it
// attributes where the time goes, not the fully-overlapped total.
type prefillProf struct {
	gemv, attn, glue time.Duration
	rec              time.Duration // the Gated-DeltaNet recurrence (prefillDeltaNetRows), apart from attention
	moe              time.Duration // the whole routed-expert FFN of an MoE layer (route, expert DMA, expert GEMVs, fold), which gemv no longer includes
}

type profCat int

const (
	gemvCat profCat = iota
	attnCat
	glueCat
	recCat
	moeCat
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
	case recCat:
		r.prof.rec += d
	case moeCat:
		r.prof.moe += d
	}
}

// PrefillLast (decoder.Prefiller) ingests a whole prompt in batched weight-stationary passes (prefillChunked) and
// returns the last token's logits: the batched counterpart of the sequential ForwardNoLogits loop, reading each weight
// once per pass rather than once per token. The K/V it writes is bit-identical to the sequential path row for row, so
// decode afterwards is byte-identical.
//
// It covers the plain dense unfused forward only. Anything else (see prefillStaticDecline) or a prompt past the KV cap
// returns an error wrapping errPrefillDeclined, and decoder/model.go falls back to the sequential KV-only prefill.
// Uniform per-layer geometry is checked against layer 0, so a non-uniform family declines rather than reading a wrong
// stride.
func (r *cudaResident) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return r.prefillChunked(ctx, embeddings, startPos, tailLastLogits, nil)
}

// markLatePlantForTest calls the mark's callback one pass late, when the slot's ring has moved on: the host snapshot store's planted defect, set by tests only.
var markLatePlantForTest bool

// PrefillMarkAlign (decoder.ResidentPrefillSubsystem) is the width of this prefill's passes: a mark fires only at startPos plus a multiple of it.
func (r *cudaResident) PrefillMarkAlign() int { return r.prefillPassRows() }

// PrefillLastMarked (decoder.ResidentPrefillSubsystem) is PrefillLast that calls at(mark) between two of its passes, after the one that ends exactly at
// position mark and before the next starts: the prompt's first mark rows are written, a windowed layer's ring holds what a prompt of mark ids would
// leave, and nothing newer has moved it. A prompt of one pass, or a mark that is not at a pass boundary, calls nothing.
func (r *cudaResident) PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error) {
	return r.prefillChunkedMarked(ctx, embeddings, startPos, tailLastLogits, nil, mark, at)
}

// HiddenLast (decoder.ResidentPrefillSubsystem) is prefillChunked's twin for embedding requests: same chunking, same K/V,
// bit-identical per-row math, but the final tail returns the last row's post-final-norm hidden state and never
// dispatches the LM head, which an embedder does not need (decoder/embed.go's HiddenLast). startPos is taken so it
// shares prefillChunked's chunk bookkeeping.
func (r *cudaResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return r.prefillChunked(ctx, embeddings, startPos, tailHiddenLast, nil)
}

// ResidualAll (decoder.ResidentPrefillSubsystem) returns every row's residual stream after the last layer and before the
// final norm, for a head that reads every position. The passes are prefillChunked's, except that each chunk keeps its
// rows and all of them take the exact kernels (as does every tail but an ordinary single-row prefill, and any DeltaNet
// model). The rows are the f32 residual the pass already downloads, so nothing is quantized and the decoder's f32 final
// norm finishes the job.
func (r *cudaResident) ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) ([][]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	chunk := r.prefillPassRows()
	if e := r.prefillStaticDecline(); e != nil {
		return nil, e
	}
	if M <= chunk || len(r.capBTaps) > 0 {
		outs, _, err := r.prefillCore(ctx, embeddings, startPos, tailResidualAll, nil, nil, nil)
		return outs, err
	}
	r.chunkOrdinary, r.chunkPromptLen = false, startPos+M
	defer func() { r.chunkOrdinary, r.chunkPromptLen = false, 0 }()
	all := make([][]float32, 0, M)
	for i := 0; i < M; {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n := min(chunk, M-i)
		outs, _, err := r.prefillCore(ctx, embeddings[i:i+n], startPos+i, tailResidualAll, nil, nil, nil)
		if err != nil {
			if errors.Is(err, errPrefillOOM) && chunk > prefillMinChunk {
				chunk = max(chunk/2, prefillMinChunk)
				r.prefillChunkCap.Store(int64(chunk))
				continue // same i: the failed pass committed nothing
			}
			return nil, err
		}
		all = append(all, outs...)
		i += n
	}
	return all, nil
}

// prefillChunked runs the prompt through the batched path in passes of at most prefillPassRows() rows and returns the
// last row's tail output: logits for tailLastLogits, the post-final-norm hidden state for tailHiddenLast. mropePos is
// nil except for PrefillMRoPELast; otherwise a row's rotation position equals its KV position. Each pass writes K/V at
// absolute positions startPos+i and attention reads the cache, so the result is bit-identical to one unchunked pass
// (TestPrefillChunked_bitIdentical); chunking only bounds peak scratch. Every chunk but the last runs tailKVOnly, since
// its output is never read.
//
// A chunk that OOMs is retried at half the width from the same position (the earlier passes stay committed in the
// positional KV). Only errPrefillOOM is retried: a static decline is independent of M and would spin, so it is checked
// once up front. Any other failure after the first pass returns a decoder.PrefillPartial naming the first unwritten position.
//
// With a batched hidden-state capture armed (r.capBTaps) each pass records its own rows, and the passes' rows are joined in
// order, so a block drafter gets every row of the prompt whatever the chunking (one pass of a long prompt does not fit
// beside a drafter on an 8 GB card).
func (r *cudaResident) prefillChunked(ctx context.Context, embeddings [][]float32, startPos int, finalTail int, mropePos [][3]int) ([]float32, error) {
	return r.prefillChunkedMarked(ctx, embeddings, startPos, finalTail, mropePos, 0, nil)
}

// prefillChunkedMarked is prefillChunked that calls at(mark) after the non-final pass that ends at position mark (none with mark <= 0). at runs on the
// caller's goroutine with no pass in flight, so a snapshot it takes is an executor job ordered after the pass that wrote the rows.
func (r *cudaResident) prefillChunkedMarked(ctx context.Context, embeddings [][]float32, startPos int, finalTail int, mropePos [][3]int, mark int, at func(pos int)) ([]float32, error) {
	return r.prefillChunkedRows(ctx, len(embeddings), func(lo, hi int) [][]float32 { return embeddings[lo:hi] }, startPos, finalTail, mropePos, mark, at)
}

// PrefillLastRows (decoder.ResidentPrefillSubsystem) is PrefillLast for n rows at startPos that reads the suffix's embeddings a pass at a time from rows, so the host holds one
// pass of them. A pass that is retried at half the width asks for its range again; each range is used before the next is read.
func (r *cudaResident) PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	return r.prefillChunkedRows(ctx, n, rows, startPos, tailLastLogits, nil, 0, nil)
}

// prefillChunkedRows is prefillChunkedMarked over m rows read by rows(lo, hi): rows [lo, hi) of the suffix, valid until the next call.
func (r *cudaResident) prefillChunkedRows(ctx context.Context, M int, rows func(lo, hi int) [][]float32, startPos int, finalTail int, mropePos [][3]int, mark int, at func(pos int)) ([]float32, error) {
	markLateFired := false
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	// Checked at entry, before the single-pass branch: a prompt that fits one chunk skips the loop, and without this check
	// that case would be uncancellable.
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	chunk := r.prefillPassRows()
	if e := r.prefillStaticDecline(); e != nil {
		return nil, e
	}
	if M <= chunk {
		outs, _, err := r.prefillCore(ctx, rows(0, M), startPos, finalTail, nil, mropePos, nil)
		if err != nil {
			return nil, err
		}
		return outs[len(outs)-1], nil
	}
	// capt joins the passes' captured rows per tap; r.capBOut holds the whole prompt's when the last pass returns.
	var capt [][]float32
	if len(r.capBTaps) > 0 {
		capt = make([][]float32, len(r.capBTaps))
		defer func() { r.capBOut = capt }()
	}
	// The non-final passes are tailKVOnly; they belong to this prompt's prefill and take the kernels its FINAL tail takes (see chunkOrdinary).
	r.chunkOrdinary = finalTail == tailLastLogits
	r.chunkPromptLen = startPos + M
	defer func() { r.chunkOrdinary, r.chunkPromptLen = false, 0 }()
	pass := 0
	for i := 0; i < M; {
		// Between passes: bounds a cancelled request to one chunk's work.
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n := chunk
		if i+n > M {
			n = M - i
		}
		// A prefill never ends in a pass of fewer than attnFusedMinRows rows: such a pass runs the exact kernels while every row
		// before it ran the fast ones, so the prompt would depend on where its last chunk fell (the 16-row rule), and past
		// singleBlockAttnShmemLimit keys the exact attention is not launchable at all. The rows come from this pass
		// (prefillTailCut), not the last one's; a chunk under 32 rows keeps its short last pass.
		if rem := M - (i + n); rem > 0 && rem < attnFusedMinRows {
			n -= prefillTailCut(n, rem)
		}
		last := i+n == M
		tail := tailKVOnly
		if last {
			tail = finalTail
		}
		pass++
		var outs [][]float32
		var err error
		if pass == prefillDeclinePassForTest {
			err = fmt.Errorf("cuda prefill: pass %d declined by the test seam: %w", pass, errPrefillDeclined)
		} else {
			outs, _, err = r.prefillCore(ctx, rows(i, i+n), startPos+i, tail, nil, mropePos, nil)
		}
		if err != nil {
			if errors.Is(err, errPrefillOOM) && chunk > prefillMinChunk {
				chunk = max(chunk/2, prefillMinChunk)
				r.prefillChunkCap.Store(int64(chunk)) // remember: do not re-OOM on the next prompt
				continue                              // same i: nothing was committed by the pass that failed to allocate
			}
			return nil, prefillPartialAt(startPos, i, err)
		}
		if prefillPassDoneForTest != nil {
			prefillPassDoneForTest()
		}
		for slot := range capt {
			capt[slot] = append(capt[slot], r.capBOut[slot]...)
		}
		if last {
			return outs[len(outs)-1], nil
		}
		i += n
		if at != nil && mark > 0 {
			if pos := startPos + i; markLatePlantForTest {
				if pos > mark && !markLateFired {
					markLateFired = true
					at(mark)
				}
			} else if pos == mark {
				at(mark)
			}
		}
	}
	// Unreachable: the i+n==M pass returns above. Kept as a real error rather than a panic so a
	// future edit to the loop bounds surfaces as a decline the caller can survive.
	return nil, fmt.Errorf("cuda prefill: chunked loop ended without heading the last row (M=%d)", M)
}

// prefillPartialAt is a failed pass's error as prefillChunkedMarked returns it: once the call's first i rows from startPos are committed it is
// a decoder.PrefillPartial, so the caller resumes after them rather than at startPos, which a windowed layer's base may already have passed.
func prefillPartialAt(startPos, i int, err error) error {
	if i == 0 {
		return err
	}
	return &decoder.PrefillPartial{Next: startPos + i, Err: err}
}

// prefillDeclinePassForTest is the 1-based pass of a chunked prefill that declines without running, set by tests only.
var prefillDeclinePassForTest int

// PrefillLastN is the speculative-decode verify primitive: the same batched pass, returning logits at all M positions
// (row m predicts position startPos+m+1). The final norm and LM head run per row exactly as in the sequential path, so
// each row's logits equal a sequential Forward's and greedy accept is lossless (TestPrefillLast_e2e).
func (r *cudaResident) PrefillLastN(embeddings [][]float32, startPos int) ([][]float32, error) {
	outs, _, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllLogits, nil, nil, nil)
	return outs, err
}

// PrefillLastNArgmax is the spec-decode verify primitive returning only each row's argmax token id, which is all the
// accept decision needs. PrefillLastN's tail re-reads the LM head once per row; this one runs ONE batched final norm,
// ONE batched head GEMV over all M rows and M argmax reductions, and reads back 4 bytes per row. It is lossless because
// only the argmax must match the sequential path, not the logits bit for bit (TestPrefillLastNArgmax_matchesPerRow).
func (r *cudaResident) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	_, ids, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllArgmax, nil, nil, nil)
	return ids, err
}

// PrefillSeedArgmax satisfies decoder.ResidentRuntimeSubsystem: the same batched forward and capture, but the head runs over
// ONE row (tailLastLogits), since the block-spec prompt seed needs only the last row's argmax. Heading all M rows
// allocates M x vocab of logits on device and host (1.24 GB at a 2048-token prompt, vocab 151,936), and an OOM there can
// leave the context launching kernels that "return SUCCESS and execute NOTHING" (prefillChunkCap in resident.go).
func (r *cudaResident) PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error) {
	// No context: decoder.ResidentRuntimeSubsystem carries none, so a cancelled block-spec seed runs to completion. Known gap;
	// closing it is an interface change (docs/code-notes/cuda.md#PrefillSeedArgmax: cancellation gap). The passes are
	// PrefillLast's, so the seed writes the K/V and state a plain prefill of the prompt writes.
	row, err := r.prefillChunked(context.Background(), embeddings, startPos, tailLastLogits, nil)
	if err != nil {
		return 0, err
	}
	if row == nil {
		return 0, fmt.Errorf("cuda prefill: seed argmax got no logits for the last row")
	}
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	return best, nil
}

// Tail modes: what the batched pass does after the layer stack.
const (
	tailLastLogits = iota // head the LAST row only (PrefillLast)
	tailAllLogits         // head every row, per-row loop, full logits (PrefillLastN)
	tailAllArgmax         // batched head over all rows, return argmax ids (PrefillLastNArgmax)
	tailKVOnly            // no head at all: the pass exists only to commit K/V (prefillChunked's
	// non-final chunks), the batched twin of ForwardNoLogits: the final norm, the head GEMV and the [M, hidden] readback
	// would be dead work for a chunk whose logits nobody reads.
	tailHiddenLast // head the LAST row only, but with the norm instead of the head (HiddenLast,
	// an embedder never needs the head): it dequantizes r.aq/r.aSc, the per-row final-norm+quant the head would read, into a
	// float32 hidden vector. Never batched across rows, like tailLastLogits.
	tailResidualAll // return EVERY row's residual stream after the last layer, BEFORE the final norm (ResidualAll): the f32 [M, hidden] block the
	// batched pass already downloads for its tails. No norm, no quantization, no head: the decoder applies the final norm on the host in f32.
)

// choosePassKernels sets the pass's kernel class before it runs: passPromptLen (the prompt the fast-prefill floor is judged on),
// forceExactKernels, and a step's prefill chunk's own class.
func (r *cudaResident) choosePassKernels(startPos, M, tail int, rows []stepRow) {
	// The fast-prefill floor is judged on the WHOLE prompt, so record it once here rather than
	// letting each selector see only this chunk's M (prefillChunked passes <=512 rows at a time).
	r.passPromptLen = max(r.chunkPromptLen,
		// a chunk of a longer prompt: the floor is the prompt's, not this pass's
		startPos+M)
	// Every tail except tailLastLogits (an ordinary single-row prefill) needs decode-identical numerics (HiddenLast's
	// bit-identity contract, speculative verify's "verify == sequential greedy"), so the fast L2/L3 levers, cosine-close but
	// not proven bit-identical, must not engage. forceExactKernels is a field, not a parameter, to avoid threading it
	// through bGemvB's call sites (see its comment in resident.go).
	r.forceExactKernels = tail != tailLastLogits && !(tail == tailKVOnly && r.chunkOrdinary) ||
		// Gated-DeltaNet: the fast levers' projection error feeds the recurrent state and compounds token after token, so they
		// stay off.
		r.recurrent() ||
		// Gemma 4 E-model: the fast levers have no fidelity evidence on this family, and a real E2B prompt above the floor
		// diverged from the sequential path. It batches on the exact kernels, bit-identical to decode at every length, until the
		// levers pass a fidelity gate of their own.
		r.eModel ||
		// Qwen3-VL's DeepStack prefill: the levers have no fidelity evidence on image rows carrying DeepStack sets (the first
		// real reading was clearly worse with them on), so such a pass runs the exact kernels, as the E-model does, until they
		// have.
		r.deepPlan != nil ||
		// Per-32 activations (the q4k lane): the levers have no fidelity evidence on it, and on the exact kernels its pass is
		// bit-identical to the sequential path (prefill_g32.cu).
		r.actG32 ||
		// a prompt with a layer on the row-batched attention past its bound (prefillPastBoundExact)
		r.prefillPastBoundExact(r.passPromptLen)
	if c := &r.stepChunk; rows != nil && c.rows > 0 {
		// A step's prefill chunk: the floor is its prompt's, and its class is the one its own pass takes as a pass of an ordinary
		// prompt (chunkOrdinary), so only the model-wide reasons above make it exact. The step's other rows stay exact.
		r.passPromptLen = c.promptLen
		c.exact = c.exact || r.recurrent() || r.eModel || r.deepPlan != nil || r.actG32 || r.prefillPastBoundExact(r.passPromptLen)
	}
}

// prefillScratchInter is the dense FFN width a pass's M-row scratch is sized for.
func (r *cudaResident) prefillScratchInter() int {
	inter := r.inter
	if r.eModel {
		// An E-model's dense layers differ in width (double-wide FFNs in the KV-shared tail): the M-row scratch is the widest, each layer's launches take its own (ffnI).
		for l := range r.layers {
			inter = max(inter, r.layers[l].ffnI)
		}
	}
	return inter
}

// prefillCore runs the batched (M=len) forward; tail selects what follows the layer stack. blocks (bidirectional image
// blocks) and mropePos (whole-prompt m-RoPE positions) are nil for every other caller. rows is the multi-sequence mode
// (batchstep.go): non-nil means row m is one decode token of its own sequence, KV slot rows[m].slot at position
// rows[m].pos, rather than prompt row startPos+m. The layer stack is unchanged; rope, the KV store and attention run per
// row through decode's own gap (decodeAttnGap) with that row's slot bound; the tail must be tailAllLogits, whose per-row
// head is decode's, and a row with a draw ends in ForwardSample's pick. With stepChunk set, the first rows are instead one
// pass of a prompt's prefill in its slot: its GEMVs and attention are that pass's, in its class (raggedChunk).
func (r *cudaResident) prefillCore(ctx context.Context, embeddings [][]float32, startPos int, tail int, blocks [][2]int, mropePos [][3]int, rows []stepRow) ([][]float32, []int, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, nil, e
	}
	// img: this pass's attention is the image-block kernel's (attn_img_batched, or attn_span_batched in a span pass, which takes
	// it even when no span falls in this pass).
	img := len(blocks) > 0 || r.spanPass
	if img && r.attnCap != 0 {
		return nil, nil, fmt.Errorf("cuda prefill: an image block on a model with an attention softcap; attn_img_batched has no cap")
	}
	if e := r.prefillEModelDecline(embeddings, blocks, mropePos, rows); e != nil {
		return nil, nil, e
	}
	if e := r.prefillRoomDecline(startPos, M, tail, blocks, mropePos, rows); e != nil {
		return nil, nil, e
	}
	// Image-block prefill (decoder.ResidentPrefillSubsystem); empty blocks is the "no block" case. checkPrefillShmemImgBlocks is
	// a separate call from checkPrefillShmem above, not a replacement: TestPrefillCoreAndDraftBlockCallTheShmemGuards pins
	// prefillCore's call to checkPrefillShmem by name (see the header of cuda/attn_img_prefill.cu).
	if img {
		if !r.imgPrefillReady || r.spanPass && !r.spanPrefillReady {
			return nil, nil, fmt.Errorf("cuda prefill: image-block batched kernel unavailable: %w", errPrefillDeclined)
		}
		if e := r.checkPrefillShmemImgBlocks(startPos, M, blocks); e != nil {
			return nil, nil, e
		}
	}
	// m-RoPE prefill (decoder.ResidentPrefillSubsystem); nil mropePos means not m-RoPE, as in decoder/rope.go's ropeAt. No
	// shared-memory widening check: m-RoPE changes the rotation angle, not attention geometry.
	if mropePos != nil && !r.mropePrefillReady {
		return nil, nil, fmt.Errorf("cuda prefill: m-RoPE batched kernel unavailable: %w", errPrefillDeclined)
	}
	r.choosePassKernels(startPos, M, tail, rows)
	maxQDim, maxKvDim := r.prefillMaxGeom()
	hidden, inter := r.hidden, r.prefillScratchInter()

	var outs [][]float32
	var ids []int
	err := r.doOn(mem.LineScratch, func() error {
		defer func() { r.forceExactKernels = false }()
		p := &r.pass
		p.reset(r, ctx, embeddings, startPos, tail, blocks, mropePos, rows, img, maxQDim, maxKvDim, hidden, inter)
		defer p.release()
		r.launchErr = nil // clear the sticky accumulator first (like launchToken), so a prior
		// decode's discarded launch error isn't re-reported by this prefill.
		var psFused func(hd int) bool // the fused question of an ordinary pass, for the row-batched attention's plan below
		if rows == nil {
			// Inside the pass, where passPromptLen and forceExactKernels are set: the guard asks useAttnFused exactly what the launch
			// will, so a layer the fused kernel serves is not held to the exact kernel's span bound. An image-block pass never takes it.
			var fused func(hd int) bool
			if !img {
				fused = func(hd int) bool { _, ok := r.useAttnFused(hd, M); return ok }
			}
			// The guard holds only the layers attn_batched (or its image twin) would run past its bound: with the row-batched exact
			// attention loaded, an ordinary pass has none (prefillSplitKVServes takes them).
			if !r.prefillSplitKVReady() {
				if e := r.checkPrefillShmem(startPos, M, fused); e != nil {
					return e
				}
			}
			if !img {
				psFused = fused
			}
		}
		if e := p.mapKV(); e != nil {
			return e
		}
		p.psFused = psFused
		if mropePos != nil {
			p.mropeWin = mropePosWindow(mropePos, startPos, M)
		}
		if e := p.allocScratch(); e != nil {
			return e
		}
		if e := p.uploadInputs(); e != nil {
			return e
		}

		for l := 0; l < r.nLayers; l++ {
			if e := p.layer(l); e != nil {
				return e
			}
		}

		r.stepChunk.split = false // the head is decode's on every row (batchedHeadFull), a chunk's last row included
		// Drain the layer launches first: they run on r.stream and the DtoH below is not ordered after it. The tails below reuse
		// the exact Forward tail, so each row's logits are bit-identical to a sequential Forward at position startPos+m (given
		// an identical residual, which the KV/logits gate checks).
		if e := r.stream.Sync(); e != nil {
			return e
		}
		// KV-only chunk: the K/V is committed, which is all a non-final chunk is for. Return before the [M, hidden] readback and
		// the head; the sync above drained the launches, so r.launchErr is complete.
		if tail == tailKVOnly {
			return r.launchErr
		}
		var e error
		switch tail {
		case tailResidualAll:
			e = p.tailResidualAll()
		case tailAllArgmax:
			e = p.tailAllArgmax()
		case tailAllLogits:
			e = p.tailAllLogits()
		default:
			e = p.tailLastRow()
		}
		outs, ids = p.outs, p.ids
		return e
	})
	if err != nil {
		// An OOM inside the job arrives as a recovered panic (runJob) carrying aikit MustBuf's "device allocation failed"
		// message. For prefill that is a decline, not a request failure: the sequential path needs no M-sized scratch. Match
		// that OOM sentinel, not any "panicked": a programming-bug panic in the batched path must surface as an error, not be
		// absorbed into the slower path. Errors that are already declines keep their own wrapping.
		if mem.DeviceAllocFailed(err) && !errors.Is(err, errPrefillDeclined) {
			return nil, nil, fmt.Errorf("cuda prefill: out of device memory for M=%d scratch (%w; %w): %v", M, errPrefillDeclined, errPrefillOOM, err)
		}
		return nil, nil, err
	}
	// Final-logit softcap and logit scale, host-side, as step() does: the one place they are applied to a returned logits
	// row, a step row included, so nothing before this point may apply them (TestCUDAStepBatch_matchesForward). Skipped for
	// tailHiddenLast and tailResidualAll: outs holds hidden states there, not logits, and a logit-only transform would
	// silently corrupt them.
	if tail == tailHiddenLast || tail == tailResidualAll {
		return outs, ids, nil
	}
	for _, out := range outs {
		applySoftcap(out, r.finalSoftcap)
		applyLogitScale(out, r.logitScale) // step() applies both
	}
	return outs, ids, nil
}
