//go:build cuda

package cuda

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
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
// fused pass that starts off it does not write the bits the whole prompt writes (docs/measurements/kv1-2026-10-13.md, G2b).
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

// prefillImageDefaultChunk is PrefillImageLast's row budget: one image's tokens plus its chat text, a bounded quantity,
// unlike an open-ended prompt. PrefillImageLast never chunks, so M past the budget is an outright decline. A card where
// 2048 is too wide still declines cleanly (prefillCore wraps a device OOM as errPrefillDeclined/errPrefillOOM) and the
// caller reuses the computed vision features. GOINFER_PREFILL_IMAGE_CHUNK overrides it (0 or unset = this default). The
// basis for 2048: docs/code-notes/cuda.md#prefillImageDefaultChunk.
const prefillImageDefaultChunk = 2048

// prefillImageChunkRows is prefillImageDefaultChunk unless GOINFER_PREFILL_IMAGE_CHUNK parses to a positive integer; an
// unusable override is ignored, as in prefillChunkRows.
func prefillImageChunkRows(v string) int {
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return n
	}
	return prefillImageDefaultChunk
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

// markLatePlantForTest calls the mark's callback one pass late, when the slot's ring has moved on: KV7 step 2's planted defect, set by tests only.
var markLatePlantForTest bool

// PrefillMarkAlign (decoder.ResidentPrefillSubsystem) is the width of this prefill's passes: a mark fires only at startPos plus a multiple of it.
func (r *cudaResident) PrefillMarkAlign() int { return r.prefillPassRows() }

// PrefillLastMarked (decoder.ResidentPrefillSubsystem) is PrefillLast that calls at(mark) between two of its passes, after the one that ends exactly at
// position mark and before the next starts: the prompt's first mark rows are written, a windowed layer's ring holds what a prompt of mark ids would
// leave, and nothing newer has moved it. A prompt of one pass, or a mark that is not at a pass boundary, calls nothing.
func (r *cudaResident) PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error) {
	return r.prefillChunkedMarked(ctx, embeddings, startPos, tailLastLogits, nil, mark, at)
}

// PrefillImageLast satisfies decoder.ResidentPrefillSubsystem: PrefillLast for a turn whose prompt carries a bidirectional
// image block [imgStart,imgEnd) (docs/multimodal.md; decoder/kvcache.go's SetImageBlocks/attendHi is the CPU reference
// it must match bit for bit). embeddings already carry the spliced vision features.
//
// The whole prompt must fit one pass: a bidirectional block split across a chunk boundary is unverified, so this never
// chunks. The row budget is prefillImageChunkRows, not prefillChunkRows. Any decline (kernel unavailable, invalid range,
// M past the budget) wraps errPrefillDeclined, and decoder.GenerateVL then falls back to the CPU prefill plus UploadKV.
func (r *cudaResident) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return r.PrefillImageBlocksLast(ctx, embeddings, startPos, [][2]int{{imgStart, imgEnd}})
}

// PrefillImageBlocksLast (decoder.ResidentPrefillSubsystem) is PrefillImageLast for several blocks, each [start, end),
// in order and disjoint, each attending bidirectionally within itself.
func (r *cudaResident) PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	prev := 0
	for _, b := range blocks {
		if b[0] < prev || b[1] <= b[0] || b[1] > startPos+M {
			return nil, fmt.Errorf("cuda prefill: image blocks %v invalid for %d rows at startPos %d: %w",
				blocks, M, startPos, errPrefillDeclined)
		}
		prev = b[1]
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("cuda prefill: no image block: %w", errPrefillDeclined)
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	chunk := prefillImageChunkRows(r.knobValue("GOINFER_PREFILL_IMAGE_CHUNK"))
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned
	}
	if M > chunk {
		return nil, fmt.Errorf("cuda prefill: image prefill needs the whole %d-row prompt in one "+
			"%d-row chunk (no chunked path for a bidirectional block): %w", M, chunk, errPrefillDeclined)
	}
	outs, _, err := r.prefillCore(ctx, embeddings, startPos, tailLastLogits, blocks, nil, nil)
	if err != nil {
		// An OOM here cannot be retried narrower (an image block must land in one pass and the OOM is a function of this M).
		// Leaving prefillChunkCap unlearned would repeat it on every image turn, and a context repeatedly driven to
		// out-of-memory can afterwards launch kernels that "return SUCCESS and execute NOTHING" (prefillChunkCap in
		// resident.go). So halve the budget for the next image call, to the prefillChunked floor.
		if errors.Is(err, errPrefillOOM) && chunk > prefillMinChunk {
			r.prefillChunkCap.Store(int64(max(chunk/2, prefillMinChunk)))
		}
		return nil, err
	}
	return outs[len(outs)-1], nil
}

// PrefillSpansLast (decoder.ResidentPrefillSubsystem) is the batched prefill of a Gemma 4 bidirectional prompt: spans are each
// image's [start, end), absolute, in order and disjoint, and attention is attn_span_batched's (gemma4AttendRange). The
// passes are prefillPassRows wide and cut outside the spans, so a span longer than a pass gets a pass of its own. Every
// pass takes attn_span_batched, with or without a span in it, so where the passes fall does not change a bit. An OOM is
// retried narrower from the same row, as in prefillChunked; any other failure after the first pass is a
// decoder.PrefillPartial. Declines: an E-model, an f16 KV cache (attn_span_batched's f16 twin passes exact-input identity, but no
// end-to-end f16 span gate has run, so f16 keeps the CPU prefill and upload), a sliding layer that would
// need the row-batched kernels, and whatever prefillCore declines.
func (r *cudaResident) PrefillSpansLast(ctx context.Context, embeddings [][]float32, startPos int, spans [][2]int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	switch {
	case r.eModel:
		return nil, fmt.Errorf("cuda prefill: span prefill on a Gemma 4 E-model: %w", errPrefillDeclined)
	case r.kvF16:
		return nil, fmt.Errorf("cuda prefill: span prefill on an f16 KV cache is not gated end to end yet: %w", errPrefillDeclined)
	case r.kvI8:
		return nil, fmt.Errorf("cuda prefill: span prefill on an int8 KV cache: %w", errPrefillDeclined)
	case !r.spanPrefillReady:
		return nil, fmt.Errorf("cuda prefill: attn_span_batched unavailable: %w", errPrefillDeclined)
	}
	prev := startPos
	for _, b := range spans {
		if b[0] < prev || b[1] <= b[0] || b[1] > startPos+M {
			return nil, fmt.Errorf("cuda prefill: spans %v invalid for %d rows at startPos %d: %w", spans, M, startPos, errPrefillDeclined)
		}
		prev = b[1]
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, e
	}
	chunk := r.prefillPassRows()
	r.spanPass, r.chunkOrdinary, r.chunkPromptLen = true, true, startPos+M
	defer func() { r.spanPass, r.chunkOrdinary, r.chunkPromptLen = false, false, 0 }()
	for i := 0; i < M; {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n := spanPassRows(spans, startPos+i, min(chunk, M-i))
		last := i+n == M
		tail := tailKVOnly
		if last {
			tail = tailLastLogits
		}
		outs, _, err := r.prefillCore(ctx, embeddings[i:i+n], startPos+i, tail, spansIn(spans, startPos+i, startPos+i+n), nil, nil)
		if err != nil {
			if errors.Is(err, errPrefillOOM) && chunk > prefillMinChunk {
				chunk = max(chunk/2, prefillMinChunk)
				r.prefillChunkCap.Store(int64(chunk))
				continue // same i: the failed pass committed nothing
			}
			return nil, prefillPartialAt(startPos, i, err)
		}
		if last {
			return outs[len(outs)-1], nil
		}
		i += n
	}
	return nil, fmt.Errorf("cuda prefill: span loop ended without heading the last row (M=%d)", M)
}

// spanPassRows is the width of a span prefill's pass that starts at absolute position from and would take want rows: cut
// before a span the cut would split, or, when the pass starts at that span, widened to its end (the CPU path's
// cutOutsideBlocks).
func spanPassRows(spans [][2]int, from, want int) int {
	end := from + want
	for _, b := range spans {
		if b[0] < end && end < b[1] {
			if b[0] > from {
				return b[0] - from
			}
			return b[1] - from
		}
	}
	return want
}

// spansIn is the spans that lie inside the pass [lo, hi), shifted by spanDefectForTest's plants: a span prefill never cuts
// inside a span, so each one is wholly in one pass.
func spansIn(spans [][2]int, lo, hi int) [][2]int {
	var out [][2]int
	if spanDefectForTest == spanDefectCausal {
		return nil
	}
	for _, b := range spans {
		if b[0] >= lo && b[1] <= hi {
			if spanDefectForTest == spanDefectShortEnd {
				b[1]--
			}
			out = append(out, b)
		}
	}
	return out
}

// spanDefectForTest plants a defect for the span prefill's gate (G-S19b) to catch: the mask left causal, each span's end one
// row short, or Gemma 3's window rule (no reach-back to the span start). Set by tests only.
var spanDefectForTest int

const (
	spanDefectCausal = iota + 1
	spanDefectShortEnd
	spanDefectNoReachBack
)

// PrefillMRoPELast satisfies decoder.ResidentPrefillSubsystem: PrefillLast for Qwen2.5-VL under m-RoPE 3D positions
// (docs/multimodal.md; decoder/rope.go's applyMRoPE/mropePositions is the CPU reference, bit for bit). embeddings carry
// the spliced vision features. mropePos is mropePositions' output, one (t,h,w) triple per ABSOLUTE position over the
// WHOLE prompt; it is never pre-sliced because mropePosWindow slices it inside prefillCore.
//
// Unlike PrefillImageLast this chunks, through prefillChunked: image tokens attend causally and each row's rotation is
// independent, so a chunk boundary breaks nothing.
func (r *cudaResident) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if len(mropePos) != startPos+M {
		return nil, fmt.Errorf("cuda prefill: mropePos len %d, want %d (startPos+M): %w",
			len(mropePos), startPos+M, errPrefillDeclined)
	}
	if !r.mropePrefillReady {
		return nil, fmt.Errorf("cuda prefill: m-RoPE batched kernel unavailable: %w", errPrefillDeclined)
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	return r.prefillChunked(ctx, embeddings, startPos, tailLastLogits, mropePos)
}

// cudaDeepPlan is the DeepStack sets one PrefillMRoPEDeepstackLast call carries: the image run's absolute start and length, and one [n*hidden] set per decoder layer that gets one.
type cudaDeepPlan struct {
	start, n int
	sets     [][]float32
}

// cudaDeepstackPrefillOn is the production switch for the resident DeepStack prefill. False sends a Qwen3-VL image turn
// back to the CPU prefill and upload, exactly as before the resident path existed. It is on by owner decision, over a
// registered gate FAIL (a small logits-cosine difference on the 896-pixel image, no located defect); the gates set it
// explicitly and restore it. Do not change the default without reading docs/code-notes/cuda.md#cudaDeepstackPrefillOn.
var cudaDeepstackPrefillOn = true

// deepDefectForTest plants a defect for the DeepStack gate to catch: 1 the sets not added, 2 each set one layer late, 3
// the sets added to the text rows too.
var deepDefectForTest int

const (
	deepDefectNotAdded = iota + 1
	deepDefectOneLayerLate
	deepDefectTextRows
)

// PrefillMRoPEDeepstackLast satisfies decoder.ResidentPrefillSubsystem: PrefillMRoPELast that also adds DeepStack
// set l to the image rows [imgStart, imgStart+imgLen) after layer l, as the CPU prefill does. Only plain dense layers
// are claimed; a layer with an MoE, DeltaNet or Gemma 4 branch declines and the turn keeps the CPU prefill and upload.
func (r *cudaResident) PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) ([]float32, error) {
	if !cudaDeepstackPrefillOn {
		return nil, fmt.Errorf("cuda prefill: the resident DeepStack prefill is switched off: %w", errPrefillDeclined)
	}
	if imgLen <= 0 || len(deep) == 0 || len(deep) > r.nLayers {
		return nil, fmt.Errorf("cuda prefill: %d DeepStack sets over %d image rows for %d layers: %w", len(deep), imgLen, r.nLayers, errPrefillDeclined)
	}
	for l := range r.layers {
		if Ly := &r.layers[l]; Ly.isMoE || Ly.g4moe || Ly.isDeltaNet() {
			return nil, fmt.Errorf("cuda prefill: DeepStack is claimed for plain dense layers only (layer %d is not one): %w", l, errPrefillDeclined)
		}
	}
	for l, set := range deep {
		if len(set) != imgLen*r.hidden {
			return nil, fmt.Errorf("cuda prefill: DeepStack set %d has %d floats, want %d rows of %d", l, len(set), imgLen, r.hidden)
		}
	}
	r.deepPlan = &cudaDeepPlan{start: imgStart, n: imgLen, sets: deep}
	defer func() { r.deepPlan = nil }()
	return r.PrefillMRoPELast(ctx, embeddings, startPos, mropePos)
}

// HybridMRoPEPrefill satisfies decoder.ResidentHybridMRoPEPrefill: for a Gated-DeltaNet hybrid (Qwen3.5+),
// PrefillMRoPELast builds the recurrent state itself, because prefillCore runs the DeltaNet recurrence over the spliced
// rows (prefillDeltaNetRows) and rotates the full-attention layers by the m-RoPE positions in the model's layout
// (rope_kv_mrope_batched, mode 1 for Qwen3.5's interleaved one); gated by
// TestGenerateQwenVL_hybridResidentPrefillMatchesCPU. A hybrid with MoE layers is not claimed (no MoE hybrid image gate
// exists) and stays on the CPU prefill.
func (r *cudaResident) HybridMRoPEPrefill() bool {
	if !r.mropePrefillReady {
		return false
	}
	delta := false
	for l := range r.layers {
		if r.layers[l].isMoE || r.layers[l].g4moe {
			return false
		}
		delta = delta || r.layers[l].isDeltaNet()
	}
	return delta
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
		// before it ran the fast ones, so the prompt would depend on where its last chunk fell (the 16-row rule, KV0's invariance
		// record), and past singleBlockAttnShmemLimit keys the exact attention is not launchable at all. The rows come from this
		// pass (prefillTailCut), not the last one's; a chunk under 32 rows keeps its short last pass.
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

// prefillStaticDecline reports why the batched path cannot run for THIS model, or nil. It covers the model-dependent
// guards (arch, per-layer geometry, weight kind, kernel availability) and deliberately not the prompt-dependent one
// (checkCap needs M/startPos), so PrefillPath can answer at load time from the code prefillCore enforces at call time.
// Per-head QK norm and Gemma sandwich norms are batched and do not decline (backend.go asserts qNorm/kNorm length ==
// headDim before residency); MoE and the Gemma parallel dense||MoE take the sequential path.
func (r *cudaResident) prefillStaticDecline() error {
	if err := r.prefillFlagsDecline(); err != nil {
		return err
	}
	// Every layer must have batched steps (cudaPrefillStepsOf); prefillFlagsDecline declines every model that has a layer without.
	for l := range r.layers {
		if len(r.layers[l].prefillSteps) == 0 {
			return fmt.Errorf("cuda prefill: layer %d has no batched steps: %w", l, errPrefillDeclined)
		}
	}
	return nil
}

// prefillFlagsDecline is prefillStaticDecline's guards read from the layers' flags and buffers, without the check on their batched
// steps; TestPrefillSteps_equalTheFlags compares the two.
func (r *cudaResident) prefillFlagsDecline() error {
	if !r.prefillReady {
		return fmt.Errorf("cuda prefill: batched kernels unavailable: %w", errPrefillDeclined)
	}
	if r.actG32 {
		// The batched quantizers and GEMMs read one activation scale per row; per-32 prompts take
		// the sequential decode path, which runs the per-32 kernels.
		return fmt.Errorf("cuda prefill: per-32 activation quantization has no batched kernels yet: %w", errPrefillDeclined)
	}
	// MoE batches the attention half; its FFN runs row by row off the batched residual (prefillCore) and keeps decode's
	// exact per-token expert sequence. The per-token debug seams would fire M times per layer in that row loop, so they
	// decline: hidCapTaps (one residual per tap per token, for a block drafter) and layerCap (one snapshot per layer, for
	// the divergence probe) would silently return Mx the rows their consumers expect. The guard is scoped to MoE on purpose:
	// prefillCore calls layerTail only inside the per-row MoE FFN loop, and a block drafter arms hidCapTaps and verifies
	// through the batched path on a DENSE model (TestDFlashRoundComposition, TestDFlashCompositionResidual), so widening it
	// breaks that.
	if r.moe || r.gemma4Moe {
		if len(r.hidCapTaps) > 0 {
			return fmt.Errorf("cuda prefill: MoE per-row FFN with per-token hidden-state taps armed "+
				"would record M rows per tap instead of one: %w", errPrefillDeclined)
		}
		if r.layerCap {
			return fmt.Errorf("cuda prefill: MoE per-row FFN with per-layer residual capture armed "+
				"would append M snapshots per layer: %w", errPrefillDeclined)
		}
	}
	// Gated-DeltaNet layers, whose conv ring and matrix state must advance one token at a time in order, are handled inside
	// the pass: prefillCore branches on Ly.isDeltaNet() and runs the recurrence through row-batched twins of decode's kernels.
	// Dispatch is on the layer's kind, not on which
	// weights happen to be absent (a dense attention stack once ran over conv layers that load no q/k/v/o), and
	// nonBatchableKind checks the projections each kind binds. ForwardN (spec verify) stays sequential for this family:
	// verify rewinds and recurrent state does not.
	//
	// Geometry is per layer: the batched launches bind each layer's own hd/nKV/qDim/kvDim/rhalf, and the M-sized scratch is
	// sized by the maximum across layers (prefillMaxGeom), so a family whose layers differ strides correctly.
	for l := range r.layers {
		Ly := &r.layers[l]
		// K=V layers (Gemma 4 global) batch as decode does: a second k-projection GEMV into the V buffer, then a scale-less
		// v_norm over it before rope rotates k. That needs the unit-weight buffer, allocated only when some layer is kEqV, so
		// refuse rather than bind a null weight.
		if Ly.vNorm && r.vNormUnit == (Buffer{}) {
			return fmt.Errorf("cuda prefill: v_norm layer at %d but no v_norm unit weight: %w", l, errPrefillDeclined)
		}
		if k := nonBatchableKind(Ly); k != "" {
			return fmt.Errorf("cuda prefill: %s weight at layer %d needs the sequential path: %w", k, l, errPrefillDeclined)
		}
	}
	return nil
}

// prefillMaxGeom returns the largest qDim and kvDim across layers: the row strides the M-sized Q/K/V/context scratch is
// allocated at. Each launch binds its own layer's dims, so a narrower layer uses a prefix of each row; the scratch has
// no cross-layer meaning, which is what lets one allocation serve a non-uniform stack.
func (r *cudaResident) prefillMaxGeom() (maxQDim, maxKvDim int) {
	for l := range r.layers {
		maxQDim = max(maxQDim, r.layers[l].qDim)
		maxKvDim = max(maxKvDim, r.layers[l].kvDim)
	}
	return maxQDim, maxKvDim
}

// checkPrefillShmem is prefillStaticDecline's prompt-dependent twin. The exact batched attention (attn_batched and its image
// twin) sizes its dynamic shared memory as decode's does, (maxNWin+128)*4 bytes, so a launch past singleBlockAttnShmemLimit
// is not survivable and must be a named decline; otherwise the launch fails in the driver and the caller silently falls to
// the sequential path. prefillCore asks it only when the row-batched exact attention did not load: with it, an ordinary pass and an image-block
// pass both have no layer attn_batched or its image twin would run past the bound (prefillSplitKVServes). Checked per layer, because a sliding-window layer's maxNWin is clamped to its
// window. It mirrors prefillCore's per-layer maxNWin exactly, so it neither declines a shape the launch would run nor misses
// one it would fail.
//
// fused says, for a layer's head dimension, whether the fused attention (attn_fused*) serves it in this pass: that kernel
// stages 64 keys at a time and its shared memory does not grow with the span, so such a layer needs nothing of it. nil means
// no layer is fused-served, the exact kernel's bound on every layer.
func (r *cudaResident) checkPrefillShmem(startPos, M int, fused func(hd int) bool) error {
	for l := range r.layers {
		Ly := &r.layers[l]
		maxNWin := startPos + M
		if Ly.window > 0 && int(Ly.window) < maxNWin {
			maxNWin = int(Ly.window)
		}
		if splitKVRequired(maxNWin) {
			if fused != nil && fused(Ly.hd) {
				continue
			}
			return fmt.Errorf("cuda prefill: layer %d attention at %d attended keys needs %d B of "+
				"shared memory, past this device's %d B limit — the row-batched exact attention does not serve "+
				"this pass (its module did not load) and neither does the fused attention (it needs head dim 64 "+
				"or 128, an ordinary last-row prefill, 16 rows or more, no image block), so this prompt length needs the sequential path: %w",
				l, maxNWin, attnShmemBytes(maxNWin), singleBlockAttnShmemLimit, errPrefillDeclined)
		}
	}
	return nil
}

// checkPrefillShmemImg is checkPrefillShmem's image-aware twin (decoder.ResidentPrefillSubsystem): it sizes the check with
// launchgeom.ImgBlockMaxNWin, the formula prefillCore's attn_img_batched launch uses, so it neither declines a shape the launch
// would run nor misses one it would fail. It is a separate call from checkPrefillShmem, not a replacement (see
// prefillCore).
func (r *cudaResident) checkPrefillShmemImg(startPos, M, imgStart, imgEnd int) error {
	return r.checkPrefillShmemImgBlocks(startPos, M, [][2]int{{imgStart, imgEnd}})
}

// checkPrefillShmemImgBlocks is checkPrefillShmemImg for several image blocks, sized as prefillCore's launch is
// (launchgeom.ImgBlocksMaxNWin).
func (r *cudaResident) checkPrefillShmemImgBlocks(startPos, M int, blocks [][2]int) error {
	for l := range r.layers {
		Ly := &r.layers[l]
		causalMaxNWin := startPos + M
		window := int(Ly.window)
		if window > 0 && window < causalMaxNWin {
			causalMaxNWin = window
		}
		maxNWin := launchgeom.ImgBlocksMaxNWin(causalMaxNWin, window, blocks)
		if splitKVRequired(maxNWin) && !r.prefillSplitKVServes(maxNWin) {
			return fmt.Errorf("cuda prefill: layer %d image-block attention at %d attended keys "+
				"needs %d B of shared memory, past this device's %d B limit: %w",
				l, maxNWin, attnShmemBytes(maxNWin), singleBlockAttnShmemLimit, errPrefillDeclined)
		}
	}
	return nil
}

// nonBatchableKind returns the weight kind of the layer's first projection the batched GEMVs cannot handle, or "" when
// every projection is int4 or int8 (bGemvB: int4 uses gemv_w4a8_batched/_rn, int8 gemv_w8a8_batched; a uniform or mixed
// int4/int8 bundle all batch). Naming the kind makes the startup message actionable.
func nonBatchableKind(Ly *cudaLayer) string {
	// A Mamba-2 mixer and a feed-forward-only block have no batched kernels: their prompts take the sequential path, named here
	// and not left to the absent q/k/v weights reading as "absent/unquantized".
	switch Ly.mixer {
	case decoder.MixerMamba:
		return "Mamba-2 mixer"
	case decoder.MixerNone:
		return "feed-forward-only block"
	}
	// Check exactly the projections this layer's batched path binds, and no others. An absent weight has kind "", which the
	// caller reads as "no problem", so checking an absent weight passes vacuously and tightening the sentinel would turn
	// that into a wrong refusal; narrow the list instead. Ly.v is absent on a K=V layer (V is derived from k), and Ly.g/u/d
	// on a pure-MoE layer (its FFN runs per row on the expert stacks). The expert stacks are deliberately not checked: the
	// per-row MoE FFN issues decode's launches on decode's weights, so whatever kind decode accepts it accepts here.
	ws := []cudaWQ{Ly.q, Ly.o}
	if !Ly.kvShared { // a KV-shared layer (Gemma 4 E-model) has no k_proj or v_proj: it attends over its source's cache
		ws = append(ws, Ly.k)
		if !Ly.kEqV {
			ws = append(ws, Ly.v)
		}
	}
	if Ly.pleGate.N > 0 || Ly.pleProj.N > 0 { // the E-model per-layer embedding branch runs through the batched GEMVs too
		ws = append(ws, Ly.pleGate, Ly.pleProj)
	}
	if Ly.isDeltaNet() {
		// A Gated-DeltaNet layer has no q/k/v/o; prefillDeltaNetRows binds these five instead.
		ws = []cudaWQ{Ly.dnQKV, Ly.dnB, Ly.dnA, Ly.dnZ, Ly.dnOut}
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

// PrefillPath (decoder.PrefillPathReporter) answers at load, before a request is served, whether this model gets the
// batched prefill, and says why not. It declines what prefillStaticDecline declines, including a native/f32 projection.
func (r *cudaResident) PrefillPath() (bool, string) {
	err := r.prefillStaticDecline()
	if err == nil {
		// Say rows per pass, not "one pass": the report is read as a promise about how a long prompt is ingested, and a prompt
		// past the chunk width is several weight-stationary passes (TestPrefillChunked_bitIdentical).
		return true, fmt.Sprintf("batched (weight-stationary CUDA passes of up to %d rows)%s",
			r.prefillPassRows(), r.fusedAttnNote())
	}
	// Detail without the wrapped sentinel, which says nothing a user can act on.
	detail := strings.TrimPrefix(strings.TrimSuffix(err.Error(), ": "+errPrefillDeclined.Error()), "cuda prefill: ")
	for l := range r.layers {
		if r.layers[l].mixer == decoder.MixerMamba {
			// Not a precision problem: the Mamba-2 recurrence has no batched kernel, so a prompt runs one forward per token at any quant.
			return false, "sequential — the Mamba-2 mixer has no batched kernel, so a prompt runs one forward per token (slower TTFT)"
		}
	}
	if len(r.layers) > 0 {
		if k := nonBatchableKind(&r.layers[0]); k != "" {
			return false, fmt.Sprintf("sequential — batched prefill needs int4 or int8 projections (%s weights) — ~9× slower TTFT, 20× CPU on a 300-token prompt", k)
		}
	}
	return false, "sequential — " + detail + " (slower TTFT: one forward per prompt token)"
}

// fusedAttnNote describes the attention kernel PrefillPath reports on. It states the condition, not a verdict: the
// kernel depends on head dim and on M, and M is a call-time property PrefillPath never sees, so a flat "fused" would be
// the same untestable promise as "one pass".
func (r *cudaResident) fusedAttnNote() string {
	if !r.fastAttn && !r.fastGemm {
		return ""
	}
	if r.eModel {
		// prefillCore forces the exact kernels for an E-model at every length; saying "fused" here would be the untestable promise this note avoids.
		return "; exact kernels (the fast levers are not fidelity-gated on Gemma 4 E-models)"
	}
	if !r.fastAttn {
		return "; weight term: gemm_w4a8_mma (L3) where M>=" + strconv.Itoa(gemmMMAMinRows) +
			"; attention: attn_batched (exact)"
	}
	if r.bAttnFused64 == (Pipeline{}) && r.bAttnFused128 == (Pipeline{}) {
		return "; fused attention REQUESTED but its module did not load — attn_batched (exact) throughout"
	}
	var served []string
	for _, hd := range []int{64, 128} {
		if p, _ := r.attnFusedFor(hd); p != (Pipeline{}) {
			served = append(served, strconv.Itoa(hd))
		}
	}
	return fmt.Sprintf("; attention: attn_fused (L2, head dim %s, M>=%d) else attn_batched (exact)",
		strings.Join(served, "/"), attnFusedMinRows)
}

// fastPrefillEnabled reports which fast prefill kernels are selected, per lever: L2 (fused attention) and L3
// (tensor-core GEMM). They are separately selectable so each can be measured against the exact path alone, and so a
// fidelity failure of the combined path can be attributed to a kernel.
//
// Default on above fastPrefillFloor, and `=0` is a complete undo. The default rests on the fidelity gate passing at
// every cell at or above that floor on both bench models and failing below it, which is why the floor is 512 and not
// lower; that failure stands on the record (docs/code-notes/cuda.md#fastPrefillEnabled,
// ). The exact path (attn_batched, gemv_w4a8_rn) stays bit-identical to the M=1
// decode kernels, is what spec-decode verify and the parity gates run, and serves every shape the fast kernels decline
// (hd not in {64,128}, M below the mma row floors, int8 bundles, K%32 != 0, every prompt below the floor).
//
//	GOINFER_CUDA_FAST_PREFILL  unset | 1 | true   both levers — THE DEFAULT, above the floor
//	                           0 | false | off    neither: the exact path everywhere (complete undo)
//	                           attn                L2 only (fused attention; exact GEMV)
//	                           gemm                L3 only (tensor-core GEMM; exact attention)
func fastPrefillEnabled(v string) (attn, gemm bool) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "false", "off":
		return false, false
	case "attn":
		return true, false
	case "gemm":
		return false, true
	}
	// Unset, "1", "true" or anything else: both levers. An unrecognised value takes the default, not off: a typo'd knob must
	// not silently change which kernels serve production, and PrefillPath reports what is in use.
	return true, true
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

// mropePosWindow returns the [startPos, startPos+M) slice of the WHOLE-prompt, absolute-indexed mropePos. prefillChunked
// passes embeddings chunk-relative (embeddings[i:i+n]) but mropePos whole on every call, so slicing by the chunk loop's
// own i instead of the absolute startPos silently reintroduces the wrong rotation, and no real-checkpoint gate whose
// prompt is too short to chunk would see it (cuda/mropepos_window_test.go covers exactly this).
func mropePosWindow(mropePos [][3]int, startPos, M int) [][3]int {
	return mropePos[startPos : startPos+M]
}

// prefillEModelDecline is a Gemma 4 E-model's decline from a pass, nil for any other model.
func (r *cudaResident) prefillEModelDecline(embeddings [][]float32, blocks [][2]int, mropePos [][3]int, rows []stepRow) error {
	if r.eModel {
		// Gemma 4 E-model: text prompts batch. A step (its rows' PLE tails and KV slots are not staged by the step path), an
		// image block and m-RoPE (no E-model has it) stay sequential.
		if rows != nil || len(blocks) > 0 || mropePos != nil {
			return fmt.Errorf("cuda prefill: Gemma 4 E-model batches text prompts only (no batched multi-conversation step, image block or m-RoPE yet): %w", errPrefillDeclined)
		}
		for m, e := range embeddings {
			if len(e) != r.embLen {
				// An E-model row is [hidden ‖ nLayers*P] (launchToken refuses the same by length): a hidden-sized row would be read as a full one with a short tail.
				return fmt.Errorf("cuda prefill: Gemma 4 E-model embedding row %d has %d floats, want %d (hidden %d + %d layers x PLE width %d)", m, len(e), r.embLen, r.hidden, r.nLayers, r.pleP)
			}
		}
	}
	return nil
}

// prefillRoomDecline is the pass's position and KV checks: a step's rows (one decode token each, at its own position), or an
// ordinary pass against the cap and the windowed KV's room.
func (r *cudaResident) prefillRoomDecline(startPos, M, tail int, blocks [][2]int, mropePos [][3]int, rows []stepRow) error {
	if rows != nil {
		// A step: each row is one decode token at its own position, and its attention is decode's (whose own launch
		// checks the shared-memory limit per row), so the whole-pass checks below do not apply.
		if len(rows) != M || tail != tailAllLogits || len(blocks) > 0 || mropePos != nil {
			return fmt.Errorf("cuda step: %d rows for %d embeddings, tail %d — a step is all-logits, text only", len(rows), M, tail)
		}
		for _, rw := range rows {
			if e := r.checkCap(rw.pos, 1); e != nil {
				return e
			}
		}
	} else if e := r.checkCap(startPos, M); e != nil {
		return e
	} else if !r.kvRoomFor(startPos, M) {
		return fmt.Errorf("cuda prefill: %d rows at position %d do not fit the windowed KV (window %d + %d slack): %w", M, startPos, r.kvWindow, r.kvSlack, errPrefillDeclined)
	}
	return nil
}

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
		// a prompt with a layer on the row-batched attention past its bound (prefillPastBoundExact)
		r.prefillPastBoundExact(r.passPromptLen)
	if c := &r.stepChunk; rows != nil && c.rows > 0 {
		// A step's prefill chunk: the floor is its prompt's, and its class is the one its own pass takes as a pass of an ordinary
		// prompt (chunkOrdinary), so only the model-wide reasons above make it exact. The step's other rows stay exact.
		r.passPromptLen = c.promptLen
		c.exact = c.exact || r.recurrent() || r.eModel || r.deepPlan != nil || r.prefillPastBoundExact(r.passPromptLen)
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

// batchedHeadArgmax is tailAllArgmax's tail: ONE batched final norm, ONE batched head GEMV over all M rows, M argmax
// reductions, then a 4-bytes-per-row readback. The per-row tail issues the head as an M=1 GEMV per row, re-reading the
// head's weights M times with no amortization.
//
// Buffers are allocated lazily here because af/ai need r.dev's context current, which holds on the executor thread this
// runs on. They are sized to M and reused; a wider block releases the old ones and reallocates.
func (r *cudaResident) batchedHeadArgmax(xB, aqB, aScB Buffer, M int, out *[]int) error {
	defer r.onLine(mem.LineScratch)()
	if M > r.logitsBCap {
		// Release before growing: abandoning the old buffer to the device ledger would strand each shorter prompt's M x vocab
		// logits.
		if r.logitsBCap > 0 {
			r.dev.ReleaseBuf(r.logitsB)
			r.dev.ReleaseBuf(r.logitsBIdx)
		}
		r.logitsB = r.af(M * r.vocab)
		r.logitsBIdx = r.ai(M)
		r.logitsBCap = M
	}
	// Batched final norm + quant over all M rows, exactly as the layer loop norms its input.
	if e := r.bNormB(xB, r.finalNorm, r.hidden, aqB, aScB, M); e != nil {
		return e
	}
	// ONE head GEMV for all M rows: the weights are read once instead of M times.
	if e := r.bGemvB(r.lmW, aqB, aScB, ArgNull(), r.logitsB, M, 0); e != nil {
		return e
	}
	// The argmax is taken on the device (argmax_rows) and M ints come back, not M x vocab logits to the host
	// (docs/measurements/r14-drafter-argmax-2026-09-22.md).
	ids := make([]int, M)
	if e := r.argmaxRows(r.logitsB, r.logitsBIdx, M, ids); e != nil {
		return e
	}
	*out = ids
	return r.launchErr
}

// batchedHeadFull is prefillCore's batched head for tailAllLogits, where every row's FULL logits are needed
// (PrefillLastN, speculative verify, the multi-sequence StepBatch), unlike batchedHeadArgmax. It runs ONE batched
// norm+quant (bNormB) and ONE batched GEMV (bGemvB) over all M rows instead of M single-row calls, so the lm_head
// weights are read once; bGemvB's kernels are bit-identical to the per-row GEMVs by construction.
//
// Everything after the GEMV stays per row. A step row with an on-device draw goes through the unmodified
// stepDraw/gumbelPick kernels, which read r.logits, so that row's slice is copied there first (gpu.CopyDevice). Every
// other row is downloaded to the pinned host buffer (r.logitsPinned/r.logitsHost) one at a time. The rows it returns are
// raw head output: prefillCore applies the softcap and logit scale to every logits row after the job. Rows before from (a
// ragged step's prefill chunk, raggedChunk.headFrom) are not headed and come back nil.
func (r *cudaResident) batchedHeadFull(xB, aqB, aScB Buffer, M int, rows []stepRow, from int) (outs [][]float32, ids []int, err error) {
	n := M - from
	if n <= 0 {
		return make([][]float32, M), nil, nil // a step of a non-final chunk alone: no row is read
	}
	defer r.onLine(mem.LineScratch)()
	if n > r.logitsBCap { // sized by the headed rows, so a chunk's unread rows hold no logits
		if r.logitsBCap > 0 {
			r.dev.ReleaseBuf(r.logitsB)
			r.dev.ReleaseBuf(r.logitsBIdx)
		}
		r.logitsB = r.af(n * r.vocab)
		r.logitsBIdx = r.ai(n)
		r.logitsBCap = n
	}
	if e := r.bNormB(xB.At(from*r.hidden*4), r.finalNorm, r.hidden, aqB, aScB, n); e != nil {
		return nil, nil, e
	}
	if e := r.bGemvB(r.lmW, aqB, aScB, ArgNull(), r.logitsB, n, 0); e != nil {
		return nil, nil, e
	}
	r.headRows += n
	outs = make([][]float32, M)
	for m := from; m < M; m++ {
		row := r.logitsB.At((m - from) * r.vocab * 4)
		if rows != nil && rows[m].draw != nil {
			// stepDraw/gumbelPick read r.logits, not a caller-supplied buffer — copy this row's
			// slice into it rather than changing kernels the per-row path already proved correct.
			if e := gpu.CopyDevice(r.logits, row, r.vocab*4); e != nil {
				return nil, nil, e
			}
			if ids == nil {
				ids = make([]int, M)
			}
			id, e := r.stepDraw(rows[m].draw)
			if e != nil {
				return nil, nil, e
			}
			ids[m] = id
			continue
		}
		if e := r.stream.Sync(); e != nil {
			return nil, nil, e
		}
		if e := gpu.ReadToHost(row, r.logitsPinned); e != nil {
			return nil, nil, e
		}
		outs[m] = append([]float32(nil), r.logitsHost...)
	}
	return outs, ids, r.launchErr
}

// attnTempRows returns rope_kv_batched's per-row query-scale table for rows at positions [startPos, startPos+M): each
// entry is launchToken's float64 expression rounded once to float32, so a batched row's Q matches decode bit for bit. A
// family without an attention temperature gets a NULL argument, which the kernel reads as scale 1. The table is rebuilt
// only when (startPos, M) changes, so every layer of one prefill shares one upload.
func (r *cudaResident) attnTempRows(startPos, M int) (gpu.KernelArg, error) {
	defer r.onLine(mem.LineScratch)()
	if r.attnTempBeta == 0 {
		return ArgNull(), nil
	}
	if M > r.qTempRowsCap {
		if r.qTempRowsCap > 0 {
			r.dev.ReleaseBuf(r.qTempRowsB)
		}
		r.qTempRowsB, r.qTempRowsCap, r.qTempRowsKey = r.af(M), M, [2]int{-1, -1}
	}
	if key := [2]int{startPos, M}; key != r.qTempRowsKey {
		rows := make([]float32, M)
		for m := range rows {
			rows[m] = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(startPos+m)/r.attnTempOrigMaxPos)))
		}
		if e := gpu.Upload(r.qTempRowsB, rows); e != nil {
			return ArgNull(), e
		}
		r.qTempRowsKey = key
	}
	return Arg(r.qTempRowsB), nil
}

// bNormB dispatches the batched pre-GEMV norm+quant for M rows — layernorm_quant_batched
// (Cohere's bias-free mean-centered LayerNorm) when r.layerNorm, else the default rmsnorm_quant_batched —
// mirroring r.norm's decode-path dispatch.
func (r *cudaResident) bNormB(x, w Buffer, N int, qOut, sOut Buffer, M int) error {
	if r.layerNorm {
		return r.bLayerNormQuantB(x, w, N, M, qOut, sOut)
	}
	return r.bRmsB(x, w, N, qOut, sOut, M)
}

func (r *cudaResident) bLayerNormQuantB(x, w Buffer, N, M int, qOut, sOut Buffer) error {
	if r.bLN != (Pipeline{}) && r.zeroBias != (Buffer{}) {
		return r.launch(r.bLN, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + N) * 4)},
			Arg(x), Arg(w), Arg(r.zeroBias), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), Arg(qOut), Arg(sOut))
	}
	// Fallback: loop M rows with r.fLN (layernorm_quant from glue.ptx)
	for m := range M {
		xm := x.At(m * N * 4)
		qm := qOut.At(m * N) // N bytes per row (N/4 int32 words)
		sm := sOut.At(m * 4) // 4 bytes per float32 scale
		if e := r.launch(r.fLN, onecfg(256, (N+256)*4),
			Arg(xm), Arg(w), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), Arg(qm), Arg(sm)); e != nil {
			return e
		}
	}
	return nil
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

// The fused attention kernel's launch geometry (package launchgeom has the contracts).
const (
	attnFusedBM      = launchgeom.AttnFusedBM
	attnFusedThreads = launchgeom.AttnFusedThreads
	attnFusedKPAD    = launchgeom.AttnFusedKPAD
	attnFusedMinRows = launchgeom.AttnFusedMinRows
)

// fastPrefillFloor is the prompt length below which neither fast lever engages. It sits at a depth that was actually
// measured: L2 is slower than the exact path at K=128 and wins from K=512, and the fidelity gate passes at K>=512 and
// fails at K=256 for the two levers combined. Moving it DOWN requires a passing fidelity-gate cell at the new depth, not
// an argument that the curve looks smooth (docs/code-notes/cuda.md#fastPrefillFloor,
// docs/measurements/prefill-l2l3-phase3-2026-09-05.md).
const fastPrefillFloor = 512

// fastPrefillFloorFor returns the floor, allowing an experiment to move it. Set
// GOINFER_CUDA_FAST_PREFILL_FLOOR to override; 0 disables the floor entirely (both levers engage
// at any length), which is how the sub-floor cells are measured at all.
func fastPrefillFloorFor(v string) int {
	if v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return fastPrefillFloor
}

// aboveFastPrefillFloor reports whether THIS PROMPT is long enough for the fast levers. It reads
// passPromptLen (the whole prompt) rather than M (this chunk) — see the field's note.
func (r *cudaResident) aboveFastPrefillFloor() bool {
	return r.passPromptLen >= fastPrefillFloorFor(r.knobValue("GOINFER_CUDA_FAST_PREFILL_FLOOR"))
}

// FastPrefillFloor is ResidentCaps' FastPrefillFloor (decoder.ResidentCaps): the prompt length from which this resident's batched prefill runs the fast (non-exact) kernels, 0 when it never does (both levers
// off, or neither kernel module loaded, in which case every selection site falls back to the exact path). GOINFER_CUDA_FAST_PREFILL_FLOOR=0 means "fast at any length", reported as 1: 0 is
// the record's "never". The decoder uses it to decline reuse of a short prefix whose rows an exact-kernel request computed (decoder.declineShortLeadReuse).
func (r *cudaResident) FastPrefillFloor() int {
	attn := r.fastAttn && (r.bAttnFused64 != (Pipeline{}) || r.bAttnFused128 != (Pipeline{}))
	gemm := r.fastGemm && r.bGemmMMA != (Pipeline{})
	if !attn && !gemm {
		return 0
	}
	return max(1, fastPrefillFloorFor(r.knobValue("GOINFER_CUDA_FAST_PREFILL_FLOOR")))
}

// attnFusedShmem is the dynamic shared memory attn_fused needs: Ksh[BN][hd+KPAD] plus Vtsh[hd][BN+KPAD], in halves. It
// is constant in K, unlike the exact path's (maxNWin+128)*4 scratch, so a layer it serves is not held to
// singleBlockAttnShmemLimit's span bound (checkPrefillShmem).
func attnFusedShmem(hd int) uint32 {
	return uint32(2 * (attnFusedBM*(hd+attnFusedKPAD) + hd*(attnFusedBM+attnFusedKPAD)))
}

// attnFusedFor returns the fused pipeline serving this head dim, or the zero Pipeline. hd 64 and 128 are the two
// instantiations attn_fused.cu emits; any other hd, or a build where the module did not load, is served by attn_batched,
// which handles every shape, so there is no special case to write.
func (r *cudaResident) attnFusedFor(hd int) (Pipeline, uint32) {
	switch hd {
	case 64:
		return r.bAttnFused64, attnFusedShmem(64)
	case 128:
		return r.bAttnFused128, attnFusedShmem(128)
	}
	return Pipeline{}, 0
}

// useAttnFused is the ONE place the L2 kernel is chosen, so the fallback cannot drift between call sites. Every "no"
// means attn_batched, the exact path, bit-identical to decode and what spec-decode verify and the parity gates run. That
// is a guarantee only because forceExactKernels (set by prefillCore) makes it an unconditional "no" for any tail but
// tailLastLogits, independent of M/K/position; the M/floor gates alone (attnFusedMinRows, aboveFastPrefillFloor) could
// be met by verify's own M by coincidence of constants.
//
// It returns only the shared-memory size, NOT the pipeline: the launch site names r.bAttnFused64 / r.bAttnFused128
// explicitly. Handing back a Pipeline in a local variable would hide WHICH kernel runs from every static reader,
// including TestPipelineLint_boundKernelsAreLaunched, which flags a field bound at every model load and launched by
// nothing.
func (r *cudaResident) useAttnFused(hd, M int) (uint32, bool) {
	if r.forceExactKernels || !r.fastAttn || r.attnCap != 0 || M < attnFusedMinRows || !r.aboveFastPrefillFloor() {
		return 0, false
	}
	p, sh := r.attnFusedFor(hd)
	if p == (Pipeline{}) {
		return 0, false
	}
	return sh, true
}

// gemmMMAMinRows: tensor cores lose below a warp's worth of rows, so M<16 takes gemv_w4a8_rn, the exact path. 16 is two
// m8n8k16 M-tiles. The kernel's tile geometry is launchgeom's GemmMMA*.
const gemmMMAMinRows = 16

// useGemmMMA is the ONE place the L3 kernel is chosen. Every "no" means gemv_w4a8_rn, the exact path, bit-identical to
// the M=1 decode GEMV and what the parity gates run. forceExactKernels makes that unconditional for any prefillCore tail
// but tailLastLogits (speculative verify's own M could otherwise cross the shape gates by coincidence).
//
// The K constraint is not defensive padding: the kernel contracts 32 elements per group scale and 8 per packed weight
// word, so a K that is not a multiple of 32 would misalign the group-scale fold. The exact path serves such a shape
// correctly.
func (r *cudaResident) useGemmMMA(kind string, K, M int) bool {
	return !r.forceExactKernels && r.fastGemm && kind == "int4" && M >= gemmMMAMinRows &&
		r.aboveFastPrefillFloor() && r.bGemmMMA != (Pipeline{}) && K%32 == 0
}

// bGemvB launches the batched GEMV for a projection. int4 uses gemm_w4a8_mma when useGemmMMA selects it, else the
// register-blocked gemv_w4a8_rn (RN rows per warp), bit-identical to the M=1 GEMV because each row keeps its own
// accumulator across all K. int8 uses gemv_w8a8_batched, bit-identical to gemv_w8a8_fwd by construction (exact int32
// accumulation, so tiling M changes no element).
func (r *cudaResident) bGemvB(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	if c := r.stepChunk; c.split && c.rows > 0 {
		// A ragged step's prefill chunk, rows [0, c.rows): the launch its own pass makes, in its class; the step's other rows
		// follow on the exact kernels. Rows are K activation bytes, one scale and N outputs apart (bQuant's and the GEMVs' layout).
		exact := r.forceExactKernels
		r.forceExactKernels = c.exact
		e := r.bGemvRows(wt, a, as, bias, dst, c.rows, accum)
		r.forceExactKernels = exact
		if e != nil || c.rows == M {
			return e
		}
		return r.bGemvRows(wt, a.At(c.rows*wt.K), as.At(c.rows*4), bias, dst.At(c.rows*wt.N*4), M-c.rows, accum)
	}
	return r.bGemvRows(wt, a, as, bias, dst, M, accum)
}

// bGemvRows is bGemvB's launch over M rows in the class forceExactKernels and the pass allow.
func (r *cudaResident) bGemvRows(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	switch wt.kind {
	case "int4":
		if r.useGemmMMA(wt.kind, wt.K, M) {
			// L3: block tile 64(M) x 64(N), 4 warps, activation panel staged in shared.
			gx, gy := launchgeom.GemmMMAGrid(wt.N, M)
			cfg := LaunchConfig{GridX: gx, GridY: gy, GridZ: 1,
				BlockX: launchgeom.GemmMMAThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: launchgeom.GemmMMAShmem()}
			r.fastGemmLaunches++
			return r.launch(r.bGemmMMA, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
				gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
				gpu.ArgValue(int32(M)), Arg(dst), gpu.ArgValue(accum))
		}
		cfg := LaunchConfig{GridX: uint32(launchgeom.GemvRNGridX(wt.N)), GridY: 1, GridZ: 1, BlockX: launchgeom.GemvRNThreads, BlockY: 1, BlockZ: 1}
		return r.launch(r.bRN, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
			gpu.ArgValue(int32(M)), Arg(dst), gpu.ArgValue(accum))
	case "int8":
		// Batched W8A8: one warp per output row (8 warps per block), the layout of doG's int8 GEMV (wt.ws = per-row f32 scale,
		// K/4 int words).
		cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		return r.launch(r.bW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), gpu.ArgValue(int32(M)),
			Arg(dst), gpu.ArgValue(accum))
	default:
		return fmt.Errorf("cuda prefill: batched GEMV is int4/int8-only, got %q", wt.kind)
	}
}
