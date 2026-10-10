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

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
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
}

type profCat int

const (
	gemvCat profCat = iota
	attnCat
	glueCat
	recCat
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

// PrefillImageLast satisfies decoder.ResidentImagePrefill: PrefillLast for a turn whose prompt carries a bidirectional
// image block [imgStart,imgEnd) (docs/multimodal.md; decoder/kvcache.go's SetImageBlocks/attendHi is the CPU reference
// it must match bit for bit). embeddings already carry the spliced vision features.
//
// The whole prompt must fit one pass: a bidirectional block split across a chunk boundary is unverified, so this never
// chunks. The row budget is prefillImageChunkRows, not prefillChunkRows. Any decline (kernel unavailable, invalid range,
// M past the budget) wraps errPrefillDeclined, and decoder.GenerateVL then falls back to the CPU prefill plus UploadKV.
func (r *cudaResident) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return r.PrefillImageBlocksLast(ctx, embeddings, startPos, [][2]int{{imgStart, imgEnd}})
}

// PrefillImageBlocksLast (decoder.ResidentImageBlocksPrefill) is PrefillImageLast for several blocks, each [start, end),
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

// PrefillMRoPELast satisfies decoder.ResidentMRoPEPrefill: PrefillLast for Qwen2.5-VL under m-RoPE 3D positions
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
// explicitly and restore it. Do not change the default without reading docs/code-notes/cuda.md#cudaDeepstackPrefillOn
// and docs/tasks/task-multimodal-support-2026-10.md.
var cudaDeepstackPrefillOn = true

// deepDefectForTest plants a defect for the DeepStack gate to catch: 1 the sets not added, 2 each set one layer late, 3
// the sets added to the text rows too.
var deepDefectForTest int

const (
	deepDefectNotAdded = iota + 1
	deepDefectOneLayerLate
	deepDefectTextRows
)

// PrefillMRoPEDeepstackLast satisfies decoder.ResidentMRoPEDeepstackPrefill: PrefillMRoPELast that also adds DeepStack
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
		if Ly := &r.layers[l]; Ly.isMoE || Ly.g4moe || Ly.isDeltaNet {
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
		delta = delta || r.layers[l].isDeltaNet
	}
	return delta
}

// prefillChunkRows is prefillDefaultChunk unless GOINFER_PREFILL_CHUNK parses to a positive integer. An unusable value
// is ignored, not an error: this is a tuning knob on a path with a correct fallback, and a typo must not take a model
// off the fast path.
func prefillChunkRows(v string) int {
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return n
	}
	return prefillDefaultChunk
}

// HiddenLast (decoder.ResidentHiddenLast) is prefillChunked's twin for embedding requests: same chunking, same K/V,
// bit-identical per-row math, but the final tail returns the last row's post-final-norm hidden state and never
// dispatches the LM head, which an embedder does not need (decoder/embed.go's HiddenLast). startPos is taken so it
// shares prefillChunked's chunk bookkeeping.
func (r *cudaResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return r.prefillChunked(ctx, embeddings, startPos, tailHiddenLast, nil)
}

// ResidualAll (decoder.ResidentResidualAll) returns every row's residual stream after the last layer and before the
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
	chunk := prefillChunkRows(r.knobValue("GOINFER_PREFILL_CHUNK"))
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned
	}
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

// prefillChunked runs the prompt through the batched path in passes of at most prefillChunkRows() rows and returns the
// last row's tail output: logits for tailLastLogits, the post-final-norm hidden state for tailHiddenLast. mropePos is
// nil except for PrefillMRoPELast; otherwise a row's rotation position equals its KV position. Each pass writes K/V at
// absolute positions startPos+i and attention reads the cache, so the result is bit-identical to one unchunked pass
// (TestPrefillChunked_bitIdentical); chunking only bounds peak scratch. Every chunk but the last runs tailKVOnly, since
// its output is never read.
//
// A chunk that OOMs is retried at half the width from the same position (the earlier passes stay committed in the
// positional KV). Only errPrefillOOM is retried: a static decline is independent of M and would spin, so it is checked
// once up front.
//
// Not chunked while a batched hidden-state capture is armed (r.capBTaps): the taps record the residual for all M rows of
// one pass, and a chunked run would leave a block drafter holding the last chunk's rows only, a silent wrong answer. It
// runs as one pass instead.
func (r *cudaResident) prefillChunked(ctx context.Context, embeddings [][]float32, startPos int, finalTail int, mropePos [][3]int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	// Checked at entry, before the single-pass branch: a prompt that fits one chunk skips the loop, and without this check
	// that case would be uncancellable.
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	chunk := prefillChunkRows(r.knobValue("GOINFER_PREFILL_CHUNK"))
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned // a previous prompt already found the default too wide for this card
	}
	if r.kvWin {
		chunk = min(chunk, kvWindowSlack+1) // a windowed layer's slack bounds the rows one pass can add (kvRoomFor)
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, e
	}
	if M <= chunk || len(r.capBTaps) > 0 {
		outs, _, err := r.prefillCore(ctx, embeddings, startPos, finalTail, nil, mropePos, nil)
		if err != nil {
			return nil, err
		}
		return outs[len(outs)-1], nil
	}
	// The non-final passes are tailKVOnly; they belong to this prompt's prefill and take the kernels its FINAL tail takes (see chunkOrdinary).
	r.chunkOrdinary = finalTail == tailLastLogits
	r.chunkPromptLen = startPos + M
	defer func() { r.chunkOrdinary, r.chunkPromptLen = false, 0 }()
	for i := 0; i < M; {
		// Between passes: bounds a cancelled request to one chunk's work.
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n := chunk
		if i+n > M {
			n = M - i
		}
		last := i+n == M
		tail := tailKVOnly
		if last {
			tail = finalTail
		}
		outs, _, err := r.prefillCore(ctx, embeddings[i:i+n], startPos+i, tail, nil, mropePos, nil)
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

// PrefillSeedArgmax satisfies decoder.ResidentSeedArgmax: the same batched forward and capture, but the head runs over
// ONE row (tailLastLogits), since the block-spec prompt seed needs only the last row's argmax. Heading all M rows
// allocates M x vocab of logits on device and host (1.24 GB at a 2048-token prompt, vocab 151,936), and an OOM there can
// leave the context launching kernels that "return SUCCESS and execute NOTHING" (prefillChunkCap in resident.go).
func (r *cudaResident) PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error) {
	// No context: decoder.ResidentSeedArgmax carries none, so a cancelled block-spec seed runs to completion. Known gap;
	// closing it is an interface change (docs/code-notes/cuda.md#PrefillSeedArgmax: cancellation gap).
	outs, _, err := r.prefillCore(context.Background(), embeddings, startPos, tailLastLogits, nil, nil, nil)
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

// prefillStaticDecline reports why the batched path cannot run for THIS model, or nil. It covers the model-dependent
// guards (arch, per-layer geometry, weight kind, kernel availability) and deliberately not the prompt-dependent one
// (checkCap needs M/startPos), so PrefillPath can answer at load time from the code prefillCore enforces at call time.
// Per-head QK norm and Gemma sandwich norms are batched and do not decline (backend.go asserts qNorm/kNorm length ==
// headDim before residency); MoE and the Gemma parallel dense||MoE take the sequential path.
func (r *cudaResident) prefillStaticDecline() error {
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
	// the pass: prefillCore branches on Ly.isDeltaNet and runs the recurrence through row-batched twins of decode's kernels
	// (prefillDeltaNetRows, docs/tasks/task-cuda-deltanet-prefill-2026-09.md). Dispatch is on the layer's kind, not on which
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

// checkPrefillShmem is prefillStaticDecline's prompt-dependent twin. Batched-prefill attention sizes its dynamic shared
// memory as decode's does, (maxNWin+128)*4 bytes, but has no split-KV fallback, so exceeding singleBlockAttnShmemLimit
// is not survivable and must be a named decline; otherwise the launch fails in the driver and the caller silently falls
// to the sequential path. Checked per layer, because a sliding-window layer's maxNWin is clamped to its window. It
// mirrors prefillCore's per-layer maxNWin exactly, so it neither declines a shape the launch would run nor misses one it
// would fail.
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

// imgBlockMaxNWin widens a layer's causal shared-memory row count (causalMaxNWin) for a bidirectional image block
// [imgStart,imgEnd). checkPrefillShmemImg and prefillCore's attn_img_batched launch both call it, so the check and the
// allocation cannot drift apart. No block (imgEnd<=imgStart) returns causalMaxNWin. A query row inside the block sees
// keys [max(pos-window+1,0), imgEnd) when windowed and [0, imgEnd) when not (decoder/kvcache.go's attendHi; derivation
// in the header of cuda/attn_img_prefill.cu): the widest row is imgEnd when the block starts at or before window-1, else
// imgLen+window-1. Unwindowed, the block never exceeds startPos+M, because imgEnd<=startPos+M (the whole prompt fits one
// chunk).
func imgBlockMaxNWin(causalMaxNWin, window, imgStart, imgEnd int) int {
	if imgEnd <= imgStart {
		return causalMaxNWin
	}
	blockMax := imgEnd
	if window > 0 {
		if imgStart > window-1 {
			blockMax = (imgEnd - imgStart) + window - 1
		}
	}
	return max(causalMaxNWin, blockMax)
}

// imgBlocksMaxNWin is imgBlockMaxNWin over several blocks: each row sees at most its own block, so the widest window is
// the widest over the blocks.
func imgBlocksMaxNWin(causalMaxNWin, window int, blocks [][2]int) int {
	n := causalMaxNWin
	for _, b := range blocks {
		n = max(n, imgBlockMaxNWin(causalMaxNWin, window, b[0], b[1]))
	}
	return n
}

// checkPrefillShmemImg is checkPrefillShmem's image-aware twin (decoder.ResidentImagePrefill): it sizes the check with
// imgBlockMaxNWin, the formula prefillCore's attn_img_batched launch uses, so it neither declines a shape the launch
// would run nor misses one it would fail. It is a separate call from checkPrefillShmem, not a replacement (see
// prefillCore).
func (r *cudaResident) checkPrefillShmemImg(startPos, M, imgStart, imgEnd int) error {
	return r.checkPrefillShmemImgBlocks(startPos, M, [][2]int{{imgStart, imgEnd}})
}

// checkPrefillShmemImgBlocks is checkPrefillShmemImg for several image blocks, sized as prefillCore's launch is
// (imgBlocksMaxNWin).
func (r *cudaResident) checkPrefillShmemImgBlocks(startPos, M int, blocks [][2]int) error {
	for l := range r.layers {
		Ly := &r.layers[l]
		causalMaxNWin := startPos + M
		window := int(Ly.window)
		if window > 0 && window < causalMaxNWin {
			causalMaxNWin = window
		}
		maxNWin := imgBlocksMaxNWin(causalMaxNWin, window, blocks)
		if splitKVRequired(maxNWin) {
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
	if Ly.isDeltaNet {
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
			prefillChunkRows(r.knobValue("GOINFER_PREFILL_CHUNK")), r.fusedAttnNote())
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
// docs/completed/task-prefill-gap.md §3). The exact path (attn_batched, gemv_w4a8_rn) stays bit-identical to the M=1
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

// prefillCore runs the batched (M=len) forward; tail selects what follows the layer stack. blocks (bidirectional image
// blocks) and mropePos (whole-prompt m-RoPE positions) are nil for every other caller. rows is the multi-sequence mode
// (batchstep.go): non-nil means row m is one decode token of its own sequence, KV slot rows[m].slot at position
// rows[m].pos, rather than prompt row startPos+m. The layer stack is unchanged; rope, the KV store and attention run per
// row through decode's own gap (decodeAttnGap) with that row's slot bound; the tail must be tailAllLogits, whose per-row
// head is decode's, and a row with a draw ends in ForwardSample's pick.
func (r *cudaResident) prefillCore(ctx context.Context, embeddings [][]float32, startPos int, tail int, blocks [][2]int, mropePos [][3]int, rows []stepRow) ([][]float32, []int, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, nil, e
	}
	if r.eModel {
		// Gemma 4 E-model: text prompts batch. A step (its rows' PLE tails and KV slots are not staged by the step path), an
		// image block and m-RoPE (no E-model has it) stay sequential.
		if rows != nil || len(blocks) > 0 || mropePos != nil {
			return nil, nil, fmt.Errorf("cuda prefill: Gemma 4 E-model batches text prompts only (no MC3 step, image block or m-RoPE yet): %w", errPrefillDeclined)
		}
		for m, e := range embeddings {
			if len(e) != r.embLen {
				// An E-model row is [hidden ‖ nLayers*P] (launchToken refuses the same by length): a hidden-sized row would be read as a full one with a short tail.
				return nil, nil, fmt.Errorf("cuda prefill: Gemma 4 E-model embedding row %d has %d floats, want %d (hidden %d + %d layers x PLE width %d)", m, len(e), r.embLen, r.hidden, r.nLayers, r.pleP)
			}
		}
	}
	if rows != nil {
		// A step: each row is one decode token at its own position, and its attention is decode's (whose own launch
		// checks the shared-memory limit per row), so the whole-pass checks below do not apply.
		if len(rows) != M || tail != tailAllLogits || len(blocks) > 0 || mropePos != nil {
			return nil, nil, fmt.Errorf("cuda step: %d rows for %d embeddings, tail %d — a step is all-logits, text only", len(rows), M, tail)
		}
		for _, rw := range rows {
			if e := r.checkCap(rw.pos, 1); e != nil {
				return nil, nil, e
			}
		}
	} else if e := r.checkCap(startPos, M); e != nil {
		return nil, nil, e
	} else if !r.kvRoomFor(startPos, M) {
		return nil, nil, fmt.Errorf("cuda prefill: %d rows at position %d do not fit the windowed KV (window %d + %d slack): %w", M, startPos, r.kvWindow, kvWindowSlack, errPrefillDeclined)
	}
	if rows == nil {
		if e := r.checkPrefillShmem(startPos, M); e != nil {
			return nil, nil, e
		}
	}
	// Image-block prefill (decoder.ResidentImagePrefill); empty blocks is the "no block" case. checkPrefillShmemImgBlocks is
	// a separate call from checkPrefillShmem above, not a replacement: TestPrefillCoreAndDraftBlockCallTheShmemGuards pins
	// prefillCore's call to checkPrefillShmem by name (see the header of cuda/attn_img_prefill.cu).
	if len(blocks) > 0 {
		if !r.imgPrefillReady {
			return nil, nil, fmt.Errorf("cuda prefill: image-block batched kernel unavailable: %w", errPrefillDeclined)
		}
		if e := r.checkPrefillShmemImgBlocks(startPos, M, blocks); e != nil {
			return nil, nil, e
		}
	}
	// m-RoPE prefill (decoder.ResidentMRoPEPrefill); nil mropePos means not m-RoPE, as in decoder/rope.go's ropeAt. No
	// shared-memory widening check: m-RoPE changes the rotation angle, not attention geometry.
	if mropePos != nil && !r.mropePrefillReady {
		return nil, nil, fmt.Errorf("cuda prefill: m-RoPE batched kernel unavailable: %w", errPrefillDeclined)
	}
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
		// stay off (docs/tasks/task-cuda-deltanet-prefill-2026-09.md).
		r.dnet != nil ||
		// Gemma 4 E-model: the fast levers have no fidelity evidence on this family, and a real E2B prompt above the floor
		// diverged from the sequential path. It batches on the exact kernels, bit-identical to decode at every length, until the
		// levers pass a fidelity gate of their own (docs/tasks/task-multimodal-support-2026-10.md, G3p).
		r.eModel ||
		// Qwen3-VL's DeepStack prefill: the levers have no fidelity evidence on image rows carrying DeepStack sets (the first
		// real reading was clearly worse with them on), so such a pass runs the exact kernels, as the E-model does, until they
		// have.
		r.deepPlan != nil
	maxQDim, maxKvDim := r.prefillMaxGeom()
	hidden, inter := r.hidden, r.inter
	if r.eModel {
		// An E-model's dense layers differ in width (double-wide FFNs in the KV-shared tail): the M-row scratch is the widest, each layer's launches take its own (ffnI).
		for l := range r.layers {
			inter = max(inter, r.layers[l].ffnI)
		}
	}

	var outs [][]float32
	var ids []int
	err := r.do(func() error {
		defer func() { r.forceExactKernels = false }()
		r.launchErr = nil // clear the sticky accumulator first (like launchToken), so a prior
		// decode's discarded launch error isn't re-reported by this prefill.
		if rows != nil {
			for _, rw := range rows {
				if e := r.kvEnsureSlot(rw.slot, rw.pos, 1); e != nil {
					return e
				}
			}
		} else if e := r.kvEnsure(startPos, M); e != nil {
			return e
		}
		// M-sized scratch (device), freed at the end. The free list and its defer are registered BEFORE the first allocation,
		// and each buffer joins the list as it is created: allocation panics on OOM (gpu.NewBufferLenOf's contract), runJob
		// recovers the panic into a decline, and at M=3000 this is hundreds of MB, so a partial allocation is an expected
		// failure on a nearly full card and must not leak.
		var scratch []Buffer
		defer func() {
			for _, b := range scratch {
				r.dev.ReleaseBuf(b)
			}
		}()
		// At least one element: a pure-MoE family (qwen3_5_moe's intermediate_size=0) has no dense FFN, so its
		// gate/up/down scratch is zero-sized and unused — and a zero-length device allocation is refused.
		af := func(n int) Buffer { b := r.af(max(n, 1)); scratch = append(scratch, b); return b }
		ai := func(n int) Buffer { b := r.ai(max(n, 1)); scratch = append(scratch, b); return b }

		xB := af(M * hidden)
		aqB, aScB := ai(M*hidden/4), af(M)
		qBb, kBb, vBb := af(M*maxQDim), af(M*maxKvDim), af(M*maxKvDim)
		cctxB := af(M * maxQDim)
		cqB, cScB := ai(M*maxQDim/4), af(M)
		mqB, mScB := ai(M*hidden/4), af(M)
		gOb, uOb := af(M*inter), af(M*inter)
		dqB, dScB, dScrB := ai(M*inter/4), af(M), af(M*inter)
		// Gated-DeltaNet families (qwen3_5 / qwen3_5_moe / qwen3_next): the M-row buffers the recurrence reads row
		// by row, and the softmax layers' double-width q projection and output gate.
		var dnB dnPrefillBufs
		if r.dnet != nil {
			dp := r.dnet
			dnB = dnPrefillBufs{
				mix: af(M * dp.convDim), bt: af(M * dp.nv), at: af(M * dp.nv), z: af(M * dp.valueDim),
				conv: af(M * dp.convDim), qn: af(M * dp.keyDim), kn: af(M * dp.keyDim), headP: af(M * dp.nv * 2),
				core: af(M * dp.valueDim), g: af(M * dp.valueDim), gq: ai(M * dp.valueDim / 4), gsc: af(M),
				qg: af(M * 2 * maxQDim), gate: af(M * maxQDim),
			}
			if startPos == 0 {
				// A prompt from position 0 starts a fresh recurrence (not positional: no rewind restores it),
				// as ForwardN's sequential path and decode's Reset do.
				if e := r.resetState(); e != nil {
					return e
				}
			}
		}
		// Sandwich families (Gemma) norm the attention / MLP sublayer output BEFORE adding it to the
		// residual, so the o-proj and down GEMVs write a temp instead of accumulating in place. One
		// [M, hidden] buffer, reused for both (the two uses are sequential).
		var sbB Buffer
		if r.sandwich || r.postOnly || r.parallelBlock {
			sbB = af(M * hidden)
		}
		residMN := uint32((M*hidden + 255) / 256)
		// Qwen3-VL's DeepStack sets for the rows of THIS pass that fall in the image run: uploaded once, added to the residual
		// after each layer (the loop below). A pass holding none of the image run uploads nothing.
		var dsBufs []Buffer
		dsOff, dsN := 0, 0
		if ds := r.deepPlan; ds != nil {
			if a, b := max(ds.start, startPos), min(ds.start+ds.n, startPos+M); a < b {
				imgOff, imgN := (a-startPos)*hidden, (b-a)*hidden
				dsOff, dsN = imgOff, imgN
				if deepDefectForTest == deepDefectTextRows {
					dsOff, dsN = 0, M*hidden
				}
				for _, set := range ds.sets {
					rows := set[(a-ds.start)*hidden : (b-ds.start)*hidden]
					if deepDefectForTest == deepDefectTextRows { // planted defect: the set added to the text rows too (the image rows' values cycled over the whole pass)
						rows = make([]float32, M*hidden)
						for i := range rows {
							rows[i] = set[(a-ds.start)*hidden+i%imgN]
						}
					}
					buf := af(dsN)
					if e := gpu.Upload(buf, rows); e != nil {
						return e
					}
					dsBufs = append(dsBufs, buf)
				}
			}
		}
		// Gemma 4 E-model per-layer embedding inputs: pleB holds every row's tail REORDERED to [layer][row][P], so layer l's slice is one contiguous
		// [M, P] operand (glu_quant_batched's `up`, row stride P), and pleGB is the gate GEMV's [M, P] output.
		var pleB, pleGB Buffer
		if r.pleP > 0 {
			P := r.pleP
			pleHost := make([]float32, r.nLayers*M*P)
			for m, e := range embeddings {
				for l := 0; l < r.nLayers; l++ {
					copy(pleHost[(l*M+m)*P:(l*M+m+1)*P], e[hidden+l*P:hidden+(l+1)*P])
				}
			}
			pleB, pleGB = af(len(pleHost)), af(M*P)
			if e := gpu.Upload(pleB, pleHost); e != nil {
				return e
			}
		}

		// Upload the M embeddings contiguously as xB[M, hidden] (already FeatEmbedScale-scaled by the
		// caller, exactly as Forward/ForwardNoLogits receive them).
		xhost := make([]float32, M*hidden)
		for m, e := range embeddings {
			copy(xhost[m*hidden:(m+1)*hidden], e)
		}
		if e := gpu.Upload(xB, xhost); e != nil {
			return e
		}

		// Every image block's [start, end), uploaded once for every layer's attn_img_batched launch.
		var imgBlk Buffer
		if len(blocks) > 0 {
			flat := make([]int32, 0, 2*len(blocks))
			for _, b := range blocks {
				flat = append(flat, int32(b[0]), int32(b[1]))
			}
			imgBlk = ai(len(flat))
			if e := gpu.Upload(imgBlk, flat); e != nil {
				return e
			}
		}
		// m-RoPE: this pass's per-row (t,h,w) rotation triples, uploaded once and reused by every layer's
		// rope_kv_mrope_batched launch (slicing: mropePosWindow).
		var rposT, rposH, rposW Buffer
		if mropePos != nil {
			window := mropePosWindow(mropePos, startPos, M)
			pt, ph, pw := make([]int32, M), make([]int32, M), make([]int32, M)
			for i, p := range window {
				pt[i], ph[i], pw[i] = int32(p[0]), int32(p[1]), int32(p[2])
			}
			rposT, rposH, rposW = ai(M), ai(M), ai(M)
			if e := gpu.Upload(rposT, pt); e != nil {
				return e
			}
			if e := gpu.Upload(rposH, ph); e != nil {
				return e
			}
			if e := gpu.Upload(rposW, pw); e != nil {
				return e
			}
		}

		for l := 0; l < r.nLayers; l++ {
			Ly := &r.layers[l]
			// PER-LAYER, read here and bound below — never L0's hoisted and assumed uniform.
			hd, nKV, qDim, rhalf := Ly.hd, Ly.nKV, Ly.qDim, Ly.rhalf
			// A KV-shared layer rotates Q only (nKV=0 leaves rope_kv's K and V work empty, so nothing is stored through the alias into its source's cache), as
			// decodeAttnGap does. The attention launch keeps the real nKV, which is how it reads the source's cache.
			ropeNKV := nKV
			if Ly.kvShared && !g4KeepSharedKVStoreForTest {
				ropeNKV = 0
			}
			ropeN := r.nH*rhalf + ropeNKV*rhalf + ropeNKV*(hd-2*rhalf)
			ffn := inter
			if r.eModel {
				ffn = Ly.ffnI
			}
			qb, kb, vb := ArgNull(), ArgNull(), ArgNull()
			if Ly.hasBias {
				qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
			}
			t := r.profTic()
			if Ly.isDeltaNet {
				// Gated-DeltaNet mixer (docs/tasks/task-cuda-deltanet-prefill-2026-09.md): replaces segA + rope +
				// attention + o-proj, as deltaNetMixer does in decode — the projections batched over the M rows,
				// the recurrence row by row in order. The FFN below is the shared one.
				if e := r.prefillDeltaNetRows(Ly, xB, aqB, aScB, M, dnB); e != nil { // profiles its own categories
					return e
				}
			} else {
				// segA: rmsnorm+quant (glue), then Q/K/V GEMVs. Category timers (r.prof) sync r.stream at
				// each group boundary; nil in production, so the launch sequence is otherwise unchanged.
				t = r.profTic()
				if r.postOnly {
					// Olmo 3/Olmo Hybrid: no pre-norm at all — quantize the raw residual
					if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
						Arg(xB), gpu.ArgValue(int32(hidden)), Arg(aqB), Arg(aScB), gpu.ArgValue(int32(M))); e != nil {
						return e
					}
				} else {
					if e := r.bNormB(xB, Ly.preNorm, hidden, aqB, aScB, M); e != nil {
						return e
					}
				}
				r.profToc(glueCat, t)
				t = r.profTic()
				if Ly.qGate {
					// attn_output_gate (qwen3_5 softmax layers): q_proj emits [query ‖ gate] per head at double
					// width. Decode's split, one launch over all M rows: M rows are M·nH contiguous heads.
					if e := r.bGemvB(Ly.q, aqB, aScB, qb, dnB.qg, M, 0); e != nil {
						return e
					}
					if e := r.launch(r.dnQSplit, g1cfg(M*qDim, 256),
						Arg(dnB.qg), Arg(qBb), Arg(dnB.gate),
						gpu.ArgValue(int32(M*qDim)), gpu.ArgValue(int32(hd))); e != nil {
						return e
					}
				} else if e := r.bGemvB(Ly.q, aqB, aScB, qb, qBb, M, 0); e != nil {
					return e
				}
				if Ly.kvShared {
					// No k_proj and no v_proj: K/V are the source layer's, already in the aliased cache.
				} else if e := r.bGemvB(Ly.k, aqB, aScB, kb, kBb, M, 0); e != nil {
					return e
				} else if Ly.kEqV {
					// K=V (Gemma 4 global layers): no v_proj. V is v_norm(the RAW pre-RoPE k_proj output), so copy the k projection into the
					// V buffer and normalize it below, before rope_kv_batched rotates k, as segA's decode path does. kBb and vBb are [M,
					// kvDim] row-major, so one contiguous copy of M*kvDim floats moves every row.
					if e := r.copyF32(kBb, vBb, M*Ly.kvDim); e != nil {
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
					qkNH, qkNKV, qkHD := r.nH, nKV, hd
					if r.qkNormWhole {
						qkNH, qkNKV, qkHD = 1, 1, r.nH*hd
					}
					kNormW := Ly.kNorm
					if Ly.kvShared {
						// Q only: nKV=0 launches no K blocks, and a shared layer has no k_norm weight to bind (an empty Buffer is a nil deref).
						qkNKV, kNormW = 0, Ly.qNorm
					}
					if e := r.launch(r.bQKN, LaunchConfig{GridX: uint32(qkNH + qkNKV), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
						Arg(qBb), Arg(kBb), Arg(Ly.qNorm), Arg(kNormW),
						gpu.ArgValue(int32(qkNH)), gpu.ArgValue(int32(qkNKV)), gpu.ArgValue(int32(qkHD)),
						gpu.ArgValue(r.eps), gpu.ArgValue(addOne), gpu.ArgValue(int32(M))); e != nil {
						return e
					}
					r.profToc(glueCat, t)
				}
				if Ly.vNorm {
					// Scale-less v_norm over vBb (the raw k on a K=V layer, the v_proj output otherwise), BEFORE rope rotates k — segA's
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
				if rows != nil {
					// Multi-sequence step: rope, the KV store and attention per sequence through decode's own gap, each row on its own slot
					// at its own position; the batched stack resumes at the ctx-quant below.
					t = r.profTic()
					if e := r.stepAttnRows(Ly, l, rows, qBb, kBb, vBb, cctxB); e != nil {
						return e
					}
					r.profToc(attnCat, t)
				} else if e := func() error {
					// rope + kv-store (glue): token m at absolute position startPos+m; rotates q/k, writes K/V.
					t = r.profTic()
					ropeCfg := LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
					ropeArgs := []gpu.KernelArg{
						Arg(qBb), Arg(kBb), Arg(vBb), Arg(Ly.invF), Arg(r.kvK(l)), Arg(r.kvV(l)),
						gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(ropeNKV)), gpu.ArgValue(int32(hd)),
						gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rhalf)), gpu.ArgValue(int32(M)),
						gpu.ArgValue(Ly.mscale),
					}
					var ropeErr error
					if mropePos != nil {
						mropeArgs := append(append([]gpu.KernelArg{}, ropeArgs...),
							Arg(rposT), Arg(rposH), Arg(rposW), gpu.ArgValue(r.mropeSec0), gpu.ArgValue(r.mropeSec1))
						if r.mropeTakesMode { // rope_kv_mrope_batched takes the layout mode; the pairwise twin has no such argument
							mropeArgs = append(mropeArgs, gpu.ArgValue(r.mropeMode))
						}
						ropeErr = r.launch(r.bRopeKVMRoPE, ropeCfg, mropeArgs...)
					} else {
						// qTempRows (Ministral 3, FeatAttnTemp): this launch covers M rows at different positions, so each row's post-RoPE query
						// scale comes from a host-built table using launchToken's own float64 expression, which keeps a batched row
						// bit-identical to decode. Not carried into the m-RoPE branch above: rope_kv_mrope_batched has no attention temperature,
						// and no family needs both.
						qTemp, qe := r.attnTempRows(startPos, M)
						if qe != nil {
							return qe
						}
						ropeErr = r.launch(r.bRopeKV, ropeCfg, append(append([]gpu.KernelArg{}, ropeArgs...), qTemp)...)
					}
					if ropeErr != nil {
						return ropeErr
					}
					r.profToc(glueCat, t)
					// causal + per-row sliding-window attention; block 128 matches the M=1 attention reduce.
					maxNWin := startPos + M
					if Ly.window > 0 && int(Ly.window) < maxNWin {
						maxNWin = int(Ly.window)
					}
					if len(blocks) > 0 {
						// Widened by imgBlocksMaxNWin, the formula checkPrefillShmemImgBlocks uses; the two must not drift, or this allocation
						// under-sizes the launch it covers.
						maxNWin = imgBlocksMaxNWin(maxNWin, int(Ly.window), blocks)
					}
					t = r.profTic()
					// L2: the fused kernel when it serves this (hd, M), else attn_batched (see useAttnFused for the selection rule). The
					// argument list is identical for all of them, so it is built once; the launches differ only in pipeline, grid and shared
					// memory. Each names its pipeline field directly (see useAttnFused for why a local variable will not do).
					attnArgs := []gpu.KernelArg{
						Arg(qBb), Arg(r.kvK(l)), Arg(r.kvV(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)),
						gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(r.attnScale),
						// r.sinkArg(l), not ArgNull(): decode threads the gpt-oss learned sink through, and this launch must not depend on a
						// different check declining gpt-oss.
						gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB), r.sinkArg(l),
					}
					var attnErr error
					// Flash-decode lane for speculative verify (attn-decode-fa-verify-PREREGISTERED.md): rows [laneFrom, M) of an all-rows
					// verify batch are served by the multi-row lane, bit-identical to the M=1 lane at each row's position
					// (TestFlashDecodeRowsBitIdentical), so a verified position scores as plain lane decode scores it. Rows below laneFrom
					// are under the attended-span floor and take the exact path, as their M=1 decode would. laneFrom == M means no lane rows
					// (the common case); laneFrom == 0 skips the exact launch.
					laneFrom := r.verifyLaneFrom(l, startPos, M, tail, len(blocks) > 0)
					// Image block first, unconditionally, before useAttnFused is consulted: attn_fused's tile-level aggregates assume
					// monotonic per-row nKeys across a 64-row tile, which an image block breaks, so the exact attn_img_batched is used
					// instead. Checking the block after useAttnFused would silently route a long image prompt through the incompatible fused
					// kernel.
					if laneFrom == 0 {
						// every row is a lane row: the exact attention launch is skipped and the lane below writes the whole context buffer
					} else if len(blocks) > 0 {
						imgArgs := append(append([]gpu.KernelArg{}, attnArgs...),
							Arg(imgBlk), gpu.ArgValue(int32(len(blocks))))
						attnErr = r.launch(r.bAttnImg, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1,
							BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)}, imgArgs...)
					} else if fsh, use := r.useAttnFused(hd, M); use {
						fcfg := LaunchConfig{GridX: uint32(r.nH),
							GridY: uint32((M + attnFusedBM - 1) / attnFusedBM), GridZ: 1,
							BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: fsh}
						// Default tile (attn-fused-tile128-default-PREREGISTERED.md): hd128 layers with no sliding window run the 128-row-tile
						// kernel, bit-identical to the 64x64 one for window == 0. hd64 stays 64x64, and so do windowed layers: their key-tile
						// grouping starts at the block's first row, so a taller block is not bit-identical there.
						// GOINFER_CUDA_ATTN_FUSED_TILE=64x64 forces 64x64.
						tile := r.attnTile
						if tile == 0 && hd == 128 && Ly.window <= 0 && r.bAttnBM128hd128 != (Pipeline{}) {
							tile = 3
						}
						if tile > 0 {
							// Diagnostic query-tile arms (attn-fused-tile-PREREGISTERED.md): 32 / 32 / 128 rows. Name each pipeline at its launch:
							// TestPipelineLint_boundKernelsAreLaunched keys on that.
							bm, bn := 32, 64
							if tile == 2 {
								bn = 32
							} else if tile == 3 {
								bm = 128
							}
							fcfg.GridY = uint32((M + bm - 1) / bm)
							fcfg.BlockX = uint32(bm / 16 * 32)
							fcfg.SharedMemBytes = uint32(2 * (bn*(hd+attnFusedKPAD) + hd*(bn+attnFusedKPAD)))
							switch {
							case tile == 1 && hd == 64:
								attnErr = r.launch(r.bAttnBM32x64hd64, fcfg, attnArgs...)
							case tile == 1:
								attnErr = r.launch(r.bAttnBM32x64hd128, fcfg, attnArgs...)
							case tile == 2 && hd == 64:
								attnErr = r.launch(r.bAttnBM32x32hd64, fcfg, attnArgs...)
							case tile == 2:
								attnErr = r.launch(r.bAttnBM32x32hd128, fcfg, attnArgs...)
							case hd == 64:
								attnErr = r.launch(r.bAttnBM128hd64, fcfg, attnArgs...)
							default:
								attnErr = r.launch(r.bAttnBM128hd128, fcfg, attnArgs...)
							}
							r.fastAttnLaunches++
							if tile == 3 {
								r.tile128Launches++
							}
						} else if hd == 64 {
							attnErr = r.launch(r.bAttnFused64, fcfg, attnArgs...)
							r.fastAttnLaunches++
						} else {
							attnErr = r.launch(r.bAttnFused128, fcfg, attnArgs...)
							r.fastAttnLaunches++
						}
					} else {
						attnErr = r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1,
							BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)}, attnArgs...)
					}
					if attnErr == nil && laneFrom < M {
						attnErr = r.flashVerifyAttn(l, startPos, laneFrom, M, qBb, cctxB)
					}
					if attnErr != nil {
						return attnErr
					}
					r.profToc(attnCat, t)
					return nil
				}(); e != nil {
					return e
				}
				// segB: ctx-quant (glue), o-proj (gemv, accum into residual), MLP.
				t = r.profTic()
				if Ly.qGate { // ctx *= sigmoid(gate), before o_proj — segB's launch over all M rows
					if e := r.launch(r.dnAttnGate, g1cfg(M*qDim, 256),
						Arg(cctxB), Arg(dnB.gate), gpu.ArgValue(int32(M*qDim))); e != nil {
						return e
					}
				}
				if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
					Arg(cctxB), gpu.ArgValue(int32(qDim)), Arg(cqB), Arg(cScB), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
				r.profToc(glueCat, t)
				t = r.profTic()
				if r.sandwich || r.postOnly || r.parallelBlock {
					// o-proj → temp (accum=0), post-attn RMSNorm per row, then add to residual.
					// For parallelBlock (Cohere), postAttnNorm is unallocated (Len==0), so bNormF32B no-ops.
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
			} // !Ly.isDeltaNet

			// FFN: dense batches; MoE runs row by row off the batched residual. The routed-expert GEMVs are indexed by a device-side
			// routing decision that differs per token, so there is no M-wide form without an expert-major gather (the expert-major
			// path below); the batched attention half still reads its weights once per pass. gpu.Buffer.At gives a zero-copy
			// sub-view, so row m of the batched residual is a valid single-row residual for the per-token FFN chain (segBFFN,
			// layerTail, segC: decode's calls in decode's order, including the g4x2 accumulator clear and the C' routed-expert DMA).
			//
			// gC=false: prefill never replays captured graphs (a graph bakes r.x, and these rows are not r.x). layerTail also
			// carries per-token debug seams (hidCapTaps, layerCap) that would fire M times per layer here; prefillStaticDecline
			// refuses a model with either armed.
			if Ly.g4moe || Ly.isMoE {
				t = r.profTic()
				// Expert-major MoE (moe_expert_major.go, moe_expert_major_gemma4.go;
				// docs/measurements/p20-expert-locality-2026-09-21.md): route, bucket and admit once per distinct expert instead of once
				// per (row, rank), when eligible. Checked once per layer, so ineligible layers pay one field-and-map read.
				if !Ly.g4moe && r.prefillMoEExpertMajorEligible(Ly) {
					if e := r.prefillMoEExpertMajorRun(ctx, Ly, xB, M, hidden); e != nil {
						return e
					}
				} else if Ly.g4moe && r.prefillGemma4ExpertMajorEligible(Ly) {
					if e := r.prefillGemma4ExpertMajorRun(ctx, Ly, xB, M, hidden); e != nil {
						return e
					}
				} else {
					for m := range M {
						// Between rows: a MoE chunk is M sequential per-token FFNs, so without this check a cancelled chunk still runs all of
						// them. Checking per row keeps cancellation at roughly one token's granularity.
						if e := ctx.Err(); e != nil {
							return e
						}
						xm := xB.At(m * hidden * 4)
						if e := r.segBFFN(Ly, l, xm); e != nil {
							return e
						}
						if e := r.layerTail(Ly, l, false, xm); e != nil {
							return e
						}
					}
				}
				r.profToc(gemvCat, t)
			} else {
				t = r.profTic()
				// segBFFN's postOnlyHere: a Gated-DeltaNet layer in a post-only model (Olmo Hybrid) replaces only the
				// attention half, so its FFN takes the normal pre-MLP norm and in-place down projection.
				postOnlyHere := r.postOnly && !Ly.isDeltaNet
				gqB, gScB := mqB, mScB
				if r.parallelBlock {
					// Cohere/Command-R: reuse segA's shared input norm (aqB/aScB) — MLP consumes the SAME
					// normed activation already computed, with zero norm dispatch.
					gqB, gScB = aqB, aScB
				} else if postOnlyHere {
					// Olmo 3/Olmo Hybrid: no pre-MLP norm — quantize raw residual xB directly.
					if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
						Arg(xB), gpu.ArgValue(int32(hidden)), Arg(mqB), Arg(mScB), gpu.ArgValue(int32(M))); e != nil {
						return e
					}
				} else {
					if e := r.bNormB(xB, Ly.postNorm, hidden, mqB, mScB, M); e != nil {
						return e
					}
				}
				r.profToc(glueCat, t)
				t = r.profTic()
				if e := r.bGemvB(Ly.g, gqB, gScB, ArgNull(), gOb, M, 0); e != nil {
					return e
				}
				if e := r.bGemvB(Ly.u, gqB, gScB, ArgNull(), uOb, M, 0); e != nil {
					return e
				}
				r.profToc(gemvCat, t)
				t = r.profTic()
				if e := r.launch(r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
					Arg(gOb), Arg(uOb), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(ffn)),
					gpu.ArgValue(r.act), Arg(dqB), Arg(dScB), Arg(dScrB), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
				r.profToc(glueCat, t)
				t = r.profTic()
				if r.sandwich || postOnlyHere || r.parallelBlock {
					// down → temp (accum=0), Gemma/Olmo post-MLP RMSNorm per row, then add to residual.
					// For parallelBlock, postMLPNorm is unallocated (Len==0), so bNormF32B no-ops.
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
					if r.pleP > 0 && !g4SkipPLEForTest {
						// Gemma 4 E-model per-layer embedding branch, segBFFN's own order over M rows (cuda/resident.go): quantize the RAW residual, the gate GEMV,
						// geluTanh(gate) * this layer's slice of the row tails (glu_quant_batched act 0 with pleB's contiguous [M, P] slice as `up`), the projection
						// GEMV, the post-norm, the residual add. All existing batched kernels. A pleLayerShiftForTest defect reads another layer's slice (modulo
						// the layer count here: decode reads past the row, which has no batched twin).
						P, pl := r.pleP, ((l+pleLayerShiftForTest)%r.nLayers+r.nLayers)%r.nLayers
						if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
							Arg(xB), gpu.ArgValue(int32(hidden)), Arg(mqB), Arg(mScB), gpu.ArgValue(int32(M))); e != nil {
							return e
						}
						if e := r.bGemvB(Ly.pleGate, mqB, mScB, ArgNull(), pleGB, M, 0); e != nil {
							return e
						}
						if e := r.launch(r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
							Arg(pleGB), Arg(pleB.At(pl*M*P*4)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(P)),
							gpu.ArgValue(int32(0)), Arg(dqB), Arg(dScB), Arg(dScrB), gpu.ArgValue(int32(M))); e != nil {
							return e
						}
						if e := r.bGemvB(Ly.pleProj, dqB, dScB, ArgNull(), sbB, M, 0); e != nil {
							return e
						}
						if e := r.bNormF32B(sbB, Ly.postPLENorm, hidden, M); e != nil {
							return e
						}
						if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
							Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
							return e
						}
					}
					// Gemma 4 dense per-layer output scalar, segB's decode twin. fScaleVec is a pure elementwise dst[i]*=s, so it batches
					// over the flattened M*hidden buffer in one launch, like bRes above. TestPrefillNonUniform_bitIdentical exists to catch
					// batched prefill omitting it while decode has it.
					if Ly.layerScalar != 0 && !g4DropLayerScalarForTest {
						if e := r.launch(r.fScaleVec, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
							Arg(xB), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(M*hidden))); e != nil {
							return e
						}
					}
				} else if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), xB, M, 1); e != nil {
					return e
				}
				r.profToc(gemvCat, t)
			}
			// DeepStack: after decoder layer l, set l is added to the image rows, as the CPU prefill does (addDeepstack; HF's
			// hidden_states[visual_pos_masks] += deepstack_visual_embeds[l]).
			if dsN > 0 && deepDefectForTest != deepDefectNotAdded {
				k := l
				if deepDefectForTest == deepDefectOneLayerLate {
					k = l - 1 // set l-1 added after layer l: every set one layer late, set 0 never early
				}
				if k >= 0 && k < len(dsBufs) {
					if e := r.launch(r.bRes, LaunchConfig{GridX: uint32((dsN + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
						Arg(xB.At(dsOff*4)), Arg(dsBufs[k]), gpu.ArgValue(int32(dsN))); e != nil {
						return e
					}
				}
			}
			// Batched hidden-state capture (block drafter): the per-token seam (capVec) syncs and downloads once per tap per token.
			// Here the residual for all M rows is already in xB, so one download per tap covers the whole block.
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
		if tail == tailResidualAll {
			// The only tail that reads the whole residual on the host: the argmax and all-logits heads read xB on the device, and
			// the last-row tails copy one row device-to-device. Downloading all M rows for them was pageable D2H after every pass
			// for nothing.
			if e := gpu.Download(xB, xhost); e != nil {
				return e
			}
			outs = make([][]float32, M)
			for m := range outs {
				outs[m] = append([]float32(nil), xhost[m*hidden:(m+1)*hidden]...)
			}
			return r.launchErr
		}
		if tail == tailAllArgmax {
			return r.batchedHeadArgmax(xB, aqB, aScB, M, &ids)
		}
		outs = make([][]float32, M)
		if tail == tailAllLogits {
			// Every row needs the head (PrefillLastN, speculative verify, the multi-sequence step): batch it, ONE norm+quant+GEMV
			// over all M rows, bit-identical to the per-row GEMV by construction (bGemvB). batchedHeadFull says what stays per row.
			var e error
			if outs, ids, e = r.batchedHeadFull(xB, aqB, aScB, M, rows); e != nil {
				return e
			}
			return r.launchErr
		}
		// tailLastLogits / tailHiddenLast: only the last row's output is used, so batching the head would compute M-1 rows
		// nobody reads.
		for m := M - 1; m < M; m++ {
			// The last row goes xB -> r.x on the device, the same bytes the host round trip (Download all, Upload one row) produced. A launch on r.stream
			// like every other op here, so it is ordered after the layer stack with no extra sync.
			if e := r.copyF32(xB.At(m*hidden*4), r.x, hidden); e != nil {
				return e
			}
			if e := r.norm(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
				return e
			}
			if tail == tailHiddenLast {
				// No head dispatch: r.aq/r.aSc (just written by the norm above) already hold what the head would read, the quantized
				// post-final-norm hidden state. Download and dequantize it instead.
				if e := r.stream.Sync(); e != nil {
					return e
				}
				q := make([]int8, hidden)
				if e := gpu.Download(r.aq, q); e != nil {
					return e
				}
				sc := make([]float32, 1)
				if e := gpu.Download(r.aSc, sc); e != nil {
					return e
				}
				row := make([]float32, hidden)
				linalg.DequantizeRowInt8(q, sc[0], row)
				outs[m] = row
				continue
			}
			if e := r.doG(r.lmW, r.aq, r.aSc, ArgNull(), r.logits, 0); e != nil {
				return e
			}
			if rows != nil && rows[m].draw != nil {
				// A step row whose token is drawn on-device: ForwardSample's own pick over this row's logits.
				if ids == nil {
					ids = make([]int, M)
				}
				id, e := r.stepDraw(rows[m].draw)
				if e != nil {
					return e
				}
				ids[m] = id
				continue
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
				return e
			}
			outs[m] = append([]float32(nil), r.logitsHost...)
			if rows != nil {
				// Forward's own host tail (step()): a step row must equal Forward's return, not only the device buffer.
				applySoftcap(outs[m], r.finalSoftcap)
				applyLogitScale(outs[m], r.logitScale)
			}
		}
		return r.launchErr
	})
	if err != nil {
		// An OOM inside the job arrives as a recovered panic (runJob) carrying aikit MustBuf's "device allocation failed"
		// message. For prefill that is a decline, not a request failure: the sequential path needs no M-sized scratch. Match
		// that OOM sentinel, not any "panicked": a programming-bug panic in the batched path must surface as an error, not be
		// absorbed into the slower path. Errors that are already declines keep their own wrapping.
		if strings.Contains(err.Error(), "device allocation failed") && !errors.Is(err, errPrefillDeclined) {
			return nil, nil, fmt.Errorf("cuda prefill: out of device memory for M=%d scratch (%w; %w): %v", M, errPrefillDeclined, errPrefillOOM, err)
		}
		return nil, nil, err
	}
	// Final-logit softcap and logit scale (Gemma), host-side, as step() does. Skipped for tailHiddenLast and
	// tailResidualAll: outs holds hidden states there, not logits, and a logit-only transform would silently corrupt them.
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
// other row is downloaded to the pinned host buffer (r.logitsPinned/r.logitsHost) one at a time. Softcap and logit scale
// apply per row and only for step rows (rows != nil): a step row must equal Forward's return, not only the device
// buffer.
func (r *cudaResident) batchedHeadFull(xB, aqB, aScB Buffer, M int, rows []stepRow) (outs [][]float32, ids []int, err error) {
	if M > r.logitsBCap {
		if r.logitsBCap > 0 {
			r.dev.ReleaseBuf(r.logitsB)
			r.dev.ReleaseBuf(r.logitsBIdx)
		}
		r.logitsB = r.af(M * r.vocab)
		r.logitsBIdx = r.ai(M)
		r.logitsBCap = M
	}
	if e := r.bNormB(xB, r.finalNorm, r.hidden, aqB, aScB, M); e != nil {
		return nil, nil, e
	}
	if e := r.bGemvB(r.lmW, aqB, aScB, ArgNull(), r.logitsB, M, 0); e != nil {
		return nil, nil, e
	}
	outs = make([][]float32, M)
	for m := range M {
		row := r.logitsB.At(m * r.vocab * 4)
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
		if rows != nil {
			applySoftcap(outs[m], r.finalSoftcap)
			applyLogitScale(outs[m], r.logitScale)
		}
	}
	return outs, ids, r.launchErr
}

// attnTempRows returns rope_kv_batched's per-row query-scale table for rows at positions [startPos, startPos+M): each
// entry is launchToken's float64 expression rounded once to float32, so a batched row's Q matches decode bit for bit. A
// family without an attention temperature gets a NULL argument, which the kernel reads as scale 1. The table is rebuilt
// only when (startPos, M) changes, so every layer of one prefill shares one upload.
func (r *cudaResident) attnTempRows(startPos, M int) (gpu.KernelArg, error) {
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

// attnFusedBM / attnFusedThreads / attnFusedKPAD MUST equal BM / (WARPS*32) / KPAD in
// attn_fused.cu. BN equals BM there, so one constant serves both the query tile and the key tile.
const (
	attnFusedBM      = 64
	attnFusedThreads = 128
	attnFusedKPAD    = 8
	// attnFusedMinRows: below a full warp-tile of query rows the fused kernel is doing 64 rows'
	// worth of K/V staging for a handful of real ones, and attn_batched — which is ALSO the exact
	// path — is both faster and bit-identical. 16 is one mma tile (m16n8k8's M).
	attnFusedMinRows = 16
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

// FastPrefillFloor satisfies decoder.ResidentFastPrefill: the prompt length from which this resident's batched prefill runs the fast (non-exact) kernels, 0 when it never does (both levers
// off, or neither kernel module loaded, in which case every selection site falls back to the exact path). GOINFER_CUDA_FAST_PREFILL_FLOOR=0 means "fast at any length", reported as 1: 0 is
// the interface's "never". The decoder uses it to decline reuse of a short prefix whose rows an exact-kernel request computed (decoder.declineShortLeadReuse).
func (r *cudaResident) FastPrefillFloor() int {
	attn := r.fastAttn && (r.bAttnFused64 != (Pipeline{}) || r.bAttnFused128 != (Pipeline{}))
	gemm := r.fastGemm && r.bGemmMMA != (Pipeline{})
	if !attn && !gemm {
		return 0
	}
	return max(1, fastPrefillFloorFor(r.knobValue("GOINFER_CUDA_FAST_PREFILL_FLOOR")))
}

// attnFusedShmem is the dynamic shared memory attn_fused needs: Ksh[BN][hd+KPAD] plus Vtsh[hd][BN+KPAD], in halves. It
// is constant in K, unlike the exact path's (maxNWin+128)*4 scratch, which makes checkPrefillShmem decline any layer
// attending more than 12,160 keys (singleBlockAttnShmemLimit).
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
	if r.forceExactKernels || !r.fastAttn || M < attnFusedMinRows || !r.aboveFastPrefillFloor() {
		return 0, false
	}
	p, sh := r.attnFusedFor(hd)
	if p == (Pipeline{}) {
		return 0, false
	}
	return sh, true
}

// gemmMMA* MUST match GBM / GBN / GWARPS*32 / GKSTEP / GAPAD in gemm_w4a8_mma.cu.
const (
	gemmMMABM      = 64
	gemmMMABN      = 64
	gemmMMAThreads = 128
	gemmMMAKStep   = 128
	gemmMMAAPad    = 4
	// gemmMMAMinRows: tensor cores lose below a warp's worth of rows, so M<16 takes gemv_w4a8_rn, the exact path. 16 is two
	// m8n8k16 M-tiles.
	gemmMMAMinRows = 16
)

// gemmMMAShmem is the activation panel: [GBM][GKSTEP/4 + GAPAD] int words. Constant, 9 KB.
func gemmMMAShmem() uint32 {
	return uint32(gemmMMABM * (gemmMMAKStep/4 + gemmMMAAPad) * 4)
}

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

// rnBlockRows must equal RN in gemv_w4a8_rn.cu: each warp computes this many output rows, so the grid covers
// ceil(N/rnBlockRows) warps. Results are bit-identical for any RN; 2 is the profiled optimum at the register/occupancy
// limit.
const rnBlockRows = 2

// bGemvB launches the batched GEMV for a projection. int4 uses gemm_w4a8_mma when useGemmMMA selects it, else the
// register-blocked gemv_w4a8_rn (RN rows per warp), bit-identical to the M=1 GEMV because each row keeps its own
// accumulator across all K. int8 uses gemv_w8a8_batched, bit-identical to gemv_w8a8_fwd by construction (exact int32
// accumulation, so tiling M changes no element).
func (r *cudaResident) bGemvB(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	switch wt.kind {
	case "int4":
		if r.useGemmMMA(wt.kind, wt.K, M) {
			// L3: block tile 64(M) x 64(N), 4 warps, activation panel staged in shared.
			cfg := LaunchConfig{
				GridX: uint32((wt.N + gemmMMABN - 1) / gemmMMABN),
				GridY: uint32((M + gemmMMABM - 1) / gemmMMABM), GridZ: 1,
				BlockX: gemmMMAThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: gemmMMAShmem()}
			r.fastGemmLaunches++
			return r.launch(r.bGemmMMA, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
				gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
				gpu.ArgValue(int32(M)), Arg(dst), gpu.ArgValue(accum))
		}
		warps := (wt.N + rnBlockRows - 1) / rnBlockRows
		cfg := LaunchConfig{GridX: uint32((warps + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
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
