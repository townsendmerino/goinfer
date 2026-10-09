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

// prefillImageDefaultChunk is PrefillImageLast's own row budget — separate from
// prefillDefaultChunk because it prices a BOUNDED, KNOWN quantity (a real image's token budget —
// Gemma 3: 256, Gemma 4: up to 1120, docs/multimodal.md — plus whatever chat text accompanies it)
// rather than an open-ended prompt, and because PrefillImageLast never chunks at all (a
// bidirectional image block split across a chunk boundary is unverified — its own doc comment),
// so M > chunk is an outright decline, not a retry at a smaller width. 2048 is not a fresh guess:
// prefillDefaultChunk's own measurement table above already measured M=2048 batched on THIS box
// (RTX 2070 SUPER) at Qwen2.5-7B — inter=18944, the worst-case width in that table — at 3.543
// ms/token with no OOM (the table's own OOM point is M=8012, four times higher). A card where
// 2048 IS too wide still declines cleanly: prefillCore's own OOM handling wraps a real device OOM
// as errPrefillDeclined/errPrefillOOM regardless of caller, and GenerateVL's fallback reuses the
// already-computed vision features rather than re-running the tower — an overly optimistic
// default costs one wasted allocation attempt, never a crash or a wrong answer.
// GOINFER_PREFILL_IMAGE_CHUNK overrides it (0 or unset = this default).
const prefillImageDefaultChunk = 2048

// prefillImageChunkRows is PrefillImageLast's row budget — prefillImageDefaultChunk unless
// GOINFER_PREFILL_IMAGE_CHUNK says otherwise. Same "an unparseable/non-positive override is
// ignored, not fatal" reasoning as prefillChunkRows.
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
func (r *cudaResident) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return r.prefillChunked(ctx, embeddings, startPos, tailLastLogits, nil)
}

// PrefillImageLast satisfies decoder.ResidentImagePrefill: PrefillLast's Gemma-3 twin for a turn
// whose prompt carries a bidirectional image block [imgStart,imgEnd) (docs/multimodal.md's image
// path — decoder/kvcache.go's SetImageBlocks/attendHi is the CPU reference this must match
// bit-for-bit). embeddings already carry the spliced vision features at that range — the same
// "embed by vector" convention PrefillLast already uses; there is no GPU-side embedding table to
// bypass. startPos is always 0 today (GenerateVL's image-prefill branch only ever runs on a fresh
// turn) but is not hardcoded, for symmetry with PrefillLast and to not foreclose a future
// prefix-reuse combination.
//
// v1 REQUIRES the whole prompt (the image block included) to fit in ONE weight-stationary pass —
// prefillChunked's positional-KV chunking is unverified (likely unsafe) for a bidirectional block
// split across a chunk boundary — so this never chunks, unlike PrefillLast. The row budget for
// that one pass is prefillImageChunkRows (2048 by default), NOT prefillChunkRows's 512: real image
// token budgets are bounded and known (see prefillImageDefaultChunk's own doc comment), so this
// prices "one image plus its surrounding chat text" rather than an open-ended text prompt. Any
// decline (kernel unavailable, invalid range, or M past the chunk width) wraps errPrefillDeclined;
// the caller (decoder.GenerateVL) treats that as "fall through to the CPU-prefill+UploadKV bridge,
// unchanged".
func (r *cudaResident) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return r.PrefillImageBlocksLast(ctx, embeddings, startPos, [][2]int{{imgStart, imgEnd}})
}

// PrefillImageBlocksLast (decoder.ResidentImageBlocksPrefill; S11, several images) is PrefillImageLast for several blocks,
// each [start, end), in order and disjoint, each attending bidirectionally within itself.
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
		// N-41 (docs/audit-2026-09-10.md): unlike prefillChunked, this call cannot retry at a
		// smaller width — a bidirectional image block has to land in one pass, and errPrefillOOM
		// here is a function of THIS M, not of chunk, so a smaller chunk would not change M or
		// rescue this attempt. But leaving prefillChunkCap unlearned means the pre-check above
		// keeps admitting up to prefillImageChunkRows() on every future image turn too, so the
		// SAME OOM repeats on every one — and per prefillChunkCap's own doc comment
		// (cuda/resident.go), repeatedly driving the context to CUDA_ERROR_OUT_OF_MEMORY risks the
		// context afterward launching kernels that "return SUCCESS and execute NOTHING", not just
		// wasted retries. Halve the budget for the NEXT image call, same floor prefillChunked
		// already uses, so a smaller image is caught by the cheap pre-check instead of repeating
		// the same real OOM.
		if errors.Is(err, errPrefillOOM) && chunk > prefillMinChunk {
			r.prefillChunkCap.Store(int64(max(chunk/2, prefillMinChunk)))
		}
		return nil, err
	}
	return outs[len(outs)-1], nil
}

// PrefillMRoPELast satisfies decoder.ResidentMRoPEPrefill: PrefillLast's Qwen2.5-VL twin under
// m-RoPE 3D rotary positions (docs/multimodal.md's image path — decoder/rope.go's
// applyMRoPE/mropePositions is the CPU reference this must match bit-for-bit). embeddings already
// carry the spliced merged-vision features — the same "embed by vector" convention PrefillLast
// already uses. mropePos is decoder/rope.go's mropePositions output, one (t,h,w) triple per
// ABSOLUTE sequence position, covering the WHOLE prompt (never pre-sliced by the caller — see
// mropePosWindow's own doc comment for why the slicing lives inside prefillCore instead).
//
// UNLIKE PrefillImageLast, this DOES chunk: Qwen's image tokens attend causally (no bidirectional
// mask, so attention geometry is completely unaffected by an image block), and each row's rotation
// is independent of every other row's — there is no cross-row coupling for a chunk boundary to
// break. Reuses prefillChunked's existing OOM-retry/cancellation loop rather than duplicating it.
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

// cudaDeepstackPrefillOn is the production switch for the resident DeepStack prefill, ON BY THE OWNER'S DECISION of 2026-10-09 ("just turn it on"), over a registered FAIL. The record (docs/tasks/task-multimodal-support-2026-10.md):
// G-S10g passed on all four images against a text control; G-S10j (one prompt per image, bars set without a noise floor) read FAIL and left it off; the first pass of the 896-pixel investigation found the chunk boundary exact, no
// localized fault in the per-layer residuals, and that one equivalent CPU kernel moves the same statistic as much as G-S10j's "defect"; G-S10k (64 units, the margin taken from an A/A in the same run) read FAIL on its per-image
// guard alone: pooled the resident prefill is within the margin (+0.0007 [-0.0044, +0.0060] against 0.0050), and on the 896-pixel image it is farther in logits cosine by about 0.011 while closer in KL and argmax agreement.
// So the one known residual is a small cosine difference on that image, not a defect located anywhere, and what it buys is a prefill about 13x faster than the CPU prefill and upload. Off, a Qwen3-VL image turn takes the
// CPU prefill and the upload; setting this false restores that path exactly. The gates set it explicitly and put it back.
var cudaDeepstackPrefillOn = true

// deepDefectForTest is G-S10g's planted-defect seam (S16's list): 0 none, 1 the sets not added, 2 each set one layer late, 3 the sets added to the text rows too.
var deepDefectForTest int

const (
	deepDefectNotAdded = iota + 1
	deepDefectOneLayerLate
	deepDefectTextRows
)

// PrefillMRoPEDeepstackLast satisfies decoder.ResidentMRoPEDeepstackPrefill (S10 on CUDA, G-S10g): PrefillMRoPELast, which also adds DeepStack set l to the image rows [imgStart, imgStart+imgLen) after layer l,
// as the CPU prefill does. Only plain dense layers are claimed: a layer with an MoE, DeltaNet or Gemma 4 branch declines, and the turn keeps the CPU prefill and upload, as before.
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

// HybridMRoPEPrefill satisfies decoder.ResidentHybridMRoPEPrefill: for a Gated-DeltaNet hybrid (Qwen3.5+), PrefillMRoPELast builds
// the recurrent state itself, because prefillCore runs the DeltaNet layers' recurrence over the same spliced rows
// (prefillDeltaNetRows) and rotates the full-attention layers by the m-RoPE positions in the layout the model uses
// (rope_kv_mrope_batched mode 1 for Qwen3.5's interleaved one). That is exercised on the dense tiny hybrid VL fixture
// (TestGenerateQwenVL_hybridResidentPrefillMatchesCPU) and, on the real 0.8B, against the CPU path. A hybrid with MoE layers is
// NOT claimed: no MoE hybrid image gate exists, so it stays on the CPU prefill until one does.
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

// prefillChunkRows is the row budget for one batched pass — prefillDefaultChunk unless
// GOINFER_PREFILL_CHUNK says otherwise. An unparseable or non-positive value is ignored rather than
// failing the request: this is a tuning knob on a path that has a correct fallback, so a typo in it
// must not be the thing that takes a model off the fast path.
func prefillChunkRows(v string) int {
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return n
	}
	return prefillDefaultChunk
}

// HiddenLast (decoder.ResidentHiddenLast) is prefillChunked's twin for G4
// (docs/tasks/task-gpu-paths-2026-09.md, embedding requests): the resident batched pass ingests the
// whole sequence exactly as PrefillLast does — same chunking, same K/V it writes, bit-identical
// per-row math — but the final chunk's tail computes the last row's post-final-norm hidden state
// instead of logits, and never dispatches the LM head at all: an embedder never needs it
// (decoder/embed.go's HiddenLast doc comment), and the head is the single most expensive matmul
// in a forward. startPos is always 0 for every caller today (HiddenLast has no prefix reuse), but
// this takes it anyway so it can share prefillChunked's chunk-boundary bookkeeping unchanged.
func (r *cudaResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return r.prefillChunked(ctx, embeddings, startPos, tailHiddenLast, nil)
}

// ResidualAll (decoder.ResidentResidualAll) returns every row's residual stream after the last layer and before the final norm: HiddenLast's twin for a head that reads
// every position (D11's follow-up; the Clef joint head). The passes are the same chunked ones prefillChunked runs, except that every chunk keeps its rows instead of
// discarding them, and all of them take the exact kernels (every tail but an ordinary single-row prefill does; a DeltaNet model always does). The rows are the f32 residual
// the pass already downloads, so nothing here is quantized and the decoder's own f32 final norm finishes the job.
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

// prefillChunked runs the prompt through the batched path in passes of at most
// prefillChunkRows() rows, returning the LAST row's tail output — logits (tailLastLogits,
// PrefillLast/PrefillMRoPELast) or the post-final-norm hidden state (tailHiddenLast,
// HiddenLast/G4). mropePos is nil for every caller except PrefillMRoPELast (Qwen2.5-VL's m-RoPE
// decode past an image block); every other caller's rotation position equals its KV position, the
// same no-op-widening convention prefillCore/rope_kv already use. Every pass writes its own K/V at
// absolute positions startPos+i…, and attention reads the cache — so pass k attends the keys
// passes 0…k-1 wrote exactly as one M=len pass would have, and the result is bit-identical to the
// unchunked path (TestPrefillChunked_bitIdentical covers the tailLastLogits case this refactor
// must not have changed). What changes is only the peak scratch, which is what the unchunked path
// ran out of. Every chunk but the last always runs tailKVOnly regardless of finalTail — a
// non-final chunk's logits/hidden-state output is never read by any caller.
//
// A chunk that OOMs is retried at half the width from the SAME position: the passes already done are
// committed to the positional KV and stay valid, so a retry re-enters at the boundary rather than
// restarting. Only errPrefillOOM is retried; a static decline is M-independent and would spin, so it
// is checked once up front and returned.
//
// NOT CHUNKED when a batched hidden-state capture is armed (r.capBTaps): those taps record the
// residual for ALL M rows of one pass, and a chunked run would leave a block drafter holding the last
// chunk's rows only. That combination is not reachable today — the capture is armed by the verify
// entry points, not by PrefillLast or HiddenLast — but a partial capture would be a silent wrong
// answer rather than a slow one, so it declines to one pass instead of being merely documented as
// unreachable.
func (r *cudaResident) prefillChunked(ctx context.Context, embeddings [][]float32, startPos int, finalTail int, mropePos [][3]int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	// AT ENTRY, before the single-pass branch below. A prompt that fits in one chunk skips the loop
	// entirely, so without this check the whole M<=chunk case is uncancellable — on a dense model
	// there is no per-row loop to catch it either, which is exactly the hole the chunk-boundary
	// check alone leaves open.
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	chunk := prefillChunkRows(r.knobValue("GOINFER_PREFILL_CHUNK"))
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned // a previous prompt already found the default too wide for this card
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
		// Between passes: the coarsest of the two checks, and the one that bounds a cancelled
		// request to a single chunk's work instead of the whole prompt.
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

// PrefillLastN is the D1 (speculative-decode) verify primitive: the SAME batched pass, but returns
// the logits at ALL M positions (row m = the target's prediction for position startPos+m+1). The
// batched layer stack is bit-identical to sequential per position (TestPrefillLast_e2e), and the
// final norm + LM head is applied per row exactly as PrefillLast applies it to the last — so each
// row's logits equal a sequential Forward's, which is what makes greedy accept lossless.
func (r *cudaResident) PrefillLastN(embeddings [][]float32, startPos int) ([][]float32, error) {
	outs, _, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllLogits, nil, nil, nil)
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
	_, ids, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllArgmax, nil, nil, nil)
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
	// context.Background(), and it is a KNOWN GAP of the same class PrefillLast just closed:
	// decoder.ResidentSeedArgmax carries no context, and this one DOES ingest a whole prompt, so a
	// cancelled block-spec seed runs to completion. It is not fixed here because the fix is another
	// interface change on a different seam, and doing it silently as a side effect of this one is
	// how a surface changes without anyone deciding to. Filed with the P20 cancellation item.
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
	if r.actG32 {
		// The batched quantizers and GEMMs read one activation scale per row; per-32 prompts take
		// the sequential decode path, which runs the per-32 kernels.
		return fmt.Errorf("cuda prefill: per-32 activation quantization has no batched kernels yet: %w", errPrefillDeclined)
	}
	// MoE is no longer a categorical refusal: a MoE layer's FFN runs ROW BY ROW off the batched
	// residual (prefillCore), so the attention half batches and the routed experts keep the exact
	// per-token sequence decode uses. What must still decline are the PER-TOKEN DEBUG SEAMS, which
	// that row loop would fire M times per layer instead of once: hidCapTaps records one residual
	// per tap per TOKEN for a block drafter, and layerCap appends one snapshot per layer for the
	// divergence probe. Both would silently return M× the rows their consumers expect. Declining
	// sends those runs down the sequential path, where their semantics are the ones they were
	// written against.
	// SCOPED TO MoE, because that is the only branch that can reach them. layerTail is called from
	// prefill at exactly ONE site — inside the per-row MoE FFN loop — so on a dense model these
	// seams are untouched by the batched pass and refusing it gains nothing.
	//
	// The first version of this guard was NOT scoped, and it broke a real flow: DFlash's block
	// drafter arms hidCapTaps and verifies through the batched path on a DENSE model, so
	// TestDFlashRoundComposition and TestDFlashCompositionResidual both failed with "per-token
	// hidden-state taps are armed". Caught only by the full heavy suite — the targeted prefill
	// subset does not run the drafter composition tests, which is the second time in this change
	// that a guard written for one arch refused another.
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
	// RECURRENT STATE (Gated-DeltaNet: qwen3_5 / qwen3_5_moe / qwen3_next). A DeltaNet layer's conv ring and matrix
	// state must advance strictly one token at a time and in order. The batched path now honours that INSIDE the
	// pass: prefillCore branches on Ly.isDeltaNet, runs the layer's projections over the M rows, and runs the
	// recurrence through row-batched twins of decode's kernels that walk the rows in order (prefillDeltaNetRows,
	// docs/tasks/task-cuda-deltanet-prefill-2026-09.md). It dispatches on the layer's KIND, not on which weights
	// happen to be absent, so the LFM2 bug class (audit-2026-09-02 C-01: a dense attention stack run over conv
	// layers that load no q/k/v/o) cannot recur through this path; nonBatchableKind checks the projections each
	// kind actually binds. ForwardN (spec verify) keeps its sequential path for this family: verify rewinds, and
	// recurrent state does not.
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
		if Ly.vNorm && r.vNormUnit == (Buffer{}) {
			return fmt.Errorf("cuda prefill: v_norm layer at %d but no v_norm unit weight: %w", l, errPrefillDeclined)
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

// imgBlockMaxNWin widens a layer's plain-causal shared-memory row count (causalMaxNWin, computed
// exactly as checkPrefillShmem/prefillCore's launch site already do) for a bidirectional image
// block [imgStart,imgEnd) — the SAME formula both checkPrefillShmemImg (the decline check) and
// prefillCore's attn_img_batched launch site (the actual allocation) call, so the two can never
// drift apart. imgEnd<=imgStart (no block) returns causalMaxNWin unchanged.
//
// Worked out from decoder/kvcache.go's attendHi/WindowStart split (see cuda/attn_img_prefill.cu's
// header for the full derivation): a query row INSIDE the block sees keys
// [max(pos-window+1,0), imgEnd) when windowed, or [0, imgEnd) when not — so its own window width
// is imgEnd-max(pos-window+1,0), maximized over pos in [imgStart,imgEnd). If the block starts at
// or before window-1, some in-block row still has winStart=0 and the max is simply imgEnd; past
// that, the width is DECREASING in pos, so the max is at pos=imgStart: imgEnd-(imgStart-window+1)
// = imgLen+window-1. Unwindowed (window<=0): the block never exceeds causalMaxNWin=startPos+M,
// since imgEnd<=startPos+M is a v1 invariant (the whole prompt, block included, fits one chunk).
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

// imgBlocksMaxNWin is imgBlockMaxNWin over several blocks (S11): each row sees at most its own block, so the widest
// window is the widest over the blocks.
func imgBlocksMaxNWin(causalMaxNWin, window int, blocks [][2]int) int {
	n := causalMaxNWin
	for _, b := range blocks {
		n = max(n, imgBlockMaxNWin(causalMaxNWin, window, b[0], b[1]))
	}
	return n
}

// checkPrefillShmemImg is checkPrefillShmem's image-aware twin (decoder.ResidentImagePrefill):
// widens the per-layer shared-memory sizing check for a bidirectional [imgStart,imgEnd) image
// block using imgBlockMaxNWin — the SAME formula the attn_img_batched launch site in prefillCore
// uses for the actual allocation, so this can never decline a shape the launch would run, or miss
// one it would fail. A SEPARATE call from checkPrefillShmem, not a replacement — see prefillCore's
// call site comment for why.
func (r *cudaResident) checkPrefillShmemImg(startPos, M, imgStart, imgEnd int) error {
	return r.checkPrefillShmemImgBlocks(startPos, M, [][2]int{{imgStart, imgEnd}})
}

// checkPrefillShmemImgBlocks is checkPrefillShmemImg for several image blocks (S11), sized as prefillCore's launch is
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

// fusedAttnNote describes the attention kernel PrefillPath is reporting on. It deliberately states
// the CONDITION rather than a verdict: which kernel runs depends on the layer's head dim and on M,
// and M is a call-time property PrefillPath never sees. Claiming "fused" flat out here would be the
// same shape of untestable promise as the "one pass" claim that let an OOM decline hide behind a
// green startup line for every prompt long enough to matter (prefill-chunking-d7-2026-09-04.md).
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

// fastPrefillEnabled reports which fast prefill kernels are selected, PER LEVER.
//
// DEFAULT ON above fastPrefillFloor (512 prompt tokens) as of 2026-09-05, and `=0` is a complete
// undo. It became a default only after §3's fidelity gate passed at every cell at or above that
// floor, on BOTH bench models, against a CPU reference with f32 weights and f32 activations:
//
//	S  K=512 SHIPS (agree 93.91% vs exact 93.75%, flips 5 v 5, KL 0.03208 v 0.03321)
//	S  K=1024, K=3900 SHIP — fast is CLOSER to the reference than exact on all three criteria
//	D7 K=512 SHIPS (agree 86.41% v 86.56%, flips 16 v 16), D7 K=1024 SHIPS
//
// It FAILS at K=256, which is why the floor exists and why it is 512 and not lower; that cell
// stands on the record and is not withdrawn (docs/completed/task-prefill-gap.md §3, and
// measurements/prefill-l2l3-phase3-2026-09-05.md).
//
// The exact path — attn_batched and gemv_w4a8_rn — remains selectable, remains bit-identical to
// the M=1 decode kernels, remains what spec-decode verify and the parity gates run, and still
// serves every shape the fast kernels decline (hd not in {64,128}, M below the mma row floors,
// int8 bundles, K%32 != 0, and every prompt below the floor).
//
// THE LEVERS ARE SEPARATELY SELECTABLE ON PURPOSE. L2 (fused attention) and L3 (tensor-core GEMM)
// touch DIFFERENT categories of prefill — attention and the weight term — and §5 requires each to
// be measured against the exact path alone before the two are measured together, "so the end-to-end
// number has an attribution". §3.1 needs the same split for a different reason: if the combined
// fast path ever scores worse than exact under the fidelity gate, the first question is WHICH
// kernel, and an all-or-nothing flag cannot answer it.
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
	// unset, "1", "true", or anything else: both levers. Defaulting an UNRECOGNISED value to the
	// default rather than to off is deliberate — a typo'd knob should not silently change which
	// kernels serve production, and PrefillPath() reports what is actually in use either way.
	return true, true
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
	tailHiddenLast // head the LAST row only, but with the norm instead of the head (HiddenLast,
	// G4, docs/tasks/task-gpu-paths-2026-09.md): runs the SAME per-row final-norm+quant
	// the head reads from, then dequantizes r.aq/r.aSc into a float32 hidden
	// vector instead of running the LM head GEMV at all — an embedder never
	// needs logits, and the head is the single most expensive matmul in a
	// forward. Never batched across rows, like tailLastLogits.
	tailResidualAll // return EVERY row's residual stream after the last layer, BEFORE the final norm (ResidualAll, D11's follow-up): the f32 [M, hidden] block the
	// batched pass already downloads for its tails. No norm, no quantization, no head: the decoder applies the final norm on the host in f32.
)

// mropePosWindow returns the [startPos, startPos+M) slice of the WHOLE-PROMPT, ABSOLUTE-indexed
// mropePos array. Named and extracted specifically because this is the one place a chunk-relative
// vs prompt-absolute mismatch could silently ship: prefillChunked passes embeddings CHUNK-RELATIVE
// (embeddings[i:i+n]) but mropePos stays WHOLE on every call — sliced here by ABSOLUTE startPos,
// never by the chunk loop's own relative index i. Getting this backwards (e.g. mropePos[i:i+n]
// instead of mropePos[startPos+i:startPos+i+n]) silently reintroduces the exact wrong-rotation
// bug this whole feature exists to fix, and — unlike an image-block mask bug — it is invisible to
// a real-checkpoint gate whose prompt is too short to ever invoke prefillChunked's multi-pass
// loop. See cuda/mropepos_window_test.go's dedicated coverage of exactly this.
func mropePosWindow(mropePos [][3]int, startPos, M int) [][3]int {
	return mropePos[startPos : startPos+M]
}

// rows (MC3 on CUDA, batchstep.go) is the multi-sequence mode: non-nil means row m is ONE decode token of its own
// sequence — KV slot rows[m].slot at position rows[m].pos — rather than prompt row startPos+m of one sequence. The
// batched layer stack is unchanged; rope, the KV store and attention run per row through decode's own gap
// (decodeAttnGap) with that row's slot bound; the tail must be tailAllLogits, whose per-row head is decode's, and a row
// with a draw ends in ForwardSample's pick. nil is every other caller: behaviour unchanged.
func (r *cudaResident) prefillCore(ctx context.Context, embeddings [][]float32, startPos int, tail int, blocks [][2]int, mropePos [][3]int, rows []stepRow) ([][]float32, []int, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, nil, e
	}
	if r.eModel {
		// Gemma 4 E-model (S9 on CUDA, part A): text prompts batch. What stays sequential, by name: a step (MC3 rows are per-sequence decode tokens whose PLE
		// tails and KV slots the step path does not stage), an image block (part B), and m-RoPE (no E-model has it).
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
	}
	if rows == nil {
		if e := r.checkPrefillShmem(startPos, M); e != nil {
			return nil, nil, e
		}
	}
	// Gemma 3 image-block prefill (decoder.ResidentImagePrefill). imgEnd<=imgStart is the "no
	// block" sentinel every other caller passes (0,0) — see cuda/attn_img_prefill.cu's header
	// for why checkPrefillShmemImg's widened check is a SEPARATE call from checkPrefillShmem
	// above rather than a replacement: TestPrefillCoreAndDraftBlockCallTheShmemGuards statically
	// pins prefillCore's call to checkPrefillShmem by name, and this must not remove it.
	if len(blocks) > 0 {
		if !r.imgPrefillReady {
			return nil, nil, fmt.Errorf("cuda prefill: image-block batched kernel unavailable: %w", errPrefillDeclined)
		}
		if e := r.checkPrefillShmemImgBlocks(startPos, M, blocks); e != nil {
			return nil, nil, e
		}
	}
	// Qwen2.5-VL m-RoPE prefill (decoder.ResidentMRoPEPrefill). mropePos == nil is the "not
	// m-RoPE" sentinel every other caller passes — matches decoder/rope.go's ropeAt's own
	// mropePos==nil convention exactly. No shmem-widening check needed here, unlike the image
	// block above: m-RoPE never widens attention geometry, only the rotation angle.
	if mropePos != nil && !r.mropePrefillReady {
		return nil, nil, fmt.Errorf("cuda prefill: m-RoPE batched kernel unavailable: %w", errPrefillDeclined)
	}
	// The fast-prefill floor is judged on the WHOLE prompt, so record it once here rather than
	// letting each selector see only this chunk's M (prefillChunked passes <=512 rows at a time).
	r.passPromptLen = max(r.chunkPromptLen,
		// a chunk of a longer prompt: the floor is the prompt's, not this pass's
		startPos+M)
	// M-09/M-10/M-11 (docs/audit-2026-09-10.md): every tail EXCEPT tailLastLogits (ordinary
	// single-row prefill) needs decode-identical numerics — HiddenLast's bit-identity contract,
	// speculative verify's "verify == sequential greedy" invariant — so the fast L2/L3 levers,
	// which are cosine-close but not proven bit-identical, must not engage for those tails. See
	// forceExactKernels's own doc comment (cuda/resident.go) for why this is a field rather than a
	// parameter threaded through bGemvB's ~20 call sites.
	r.forceExactKernels = tail != tailLastLogits && !(tail == tailKVOnly && r.chunkOrdinary) ||
		// Gated-DeltaNet: the fast levers' projection error feeds the recurrent state and compounds token after
		// token — measured on Qwen3.5-9B, 561 tokens: cosine 0.994 and different greedy continuations with them,
		// bit-identical to decode without (docs/tasks/task-cuda-deltanet-prefill-2026-09.md).
		r.dnet != nil ||
		// Gemma 4 E-model (S9 on CUDA part A): the fast levers have no fidelity evidence on this family, and the first served read showed it. On the real E2B, a
		// 2,170-token prompt above the 512-row floor took the fast kernels and first differed from the sequential path at generated token 10 (' and' 0.19 against
		// ' issues' 0.19, the latter outside the sequential top 3: not a near-tie under the registered rule), while the same prompt on the exact kernels matched
		// the sequential path in every logprob (docs/tasks/task-multimodal-support-2026-10.md, G3p). So an E-model batches on the exact kernels, bit-identical to
		// decode at every length, until the levers pass a fidelity gate of their own.
		r.eModel ||
		// Qwen3-VL's DeepStack prefill (S10 on CUDA, G-S10g): the first real reading failed on table.png (986 rows, past the 512-row floor): the last-row logits read 0.9125 against the CPU prefill with the fast levers
		// on, 0.9741 with them off (and 0.9711 with them on but no sets added), under a text control minimum of 0.9539. The levers have no fidelity evidence on image rows carrying DeepStack sets, so such a pass runs
		// the exact kernels, as the E-model does, until they have.
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
		// S10 on CUDA (G-S10g): Qwen3-VL's DeepStack sets, for the rows of THIS pass that fall in the image run, uploaded once and added to the residual after each layer (the loop below). A pass that
		// holds none of the image run (a chunk of text before or after it) uploads nothing.
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
					if deepDefectForTest == deepDefectTextRows { // G-S10g planted defect: the set added to the text rows too (the image rows' values cycled over the whole pass)
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

		// Qwen2.5-VL m-RoPE: build and upload this pass's per-row (t,h,w) rotation triples ONCE,
		// reused unchanged across every layer's rope_kv_mrope_batched launch below.
		// mropePosWindow's own doc comment explains why the absolute-vs-chunk-relative slicing
		// lives in a named, separately-tested function rather than an inline expression here.
		// S11: every image block's [start, end), uploaded once for every layer's attn_img_batched launch.
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
					// K=V (Gemma-4 global layers): this layer has NO v_proj. V is v_norm(the RAW pre-RoPE k_proj output), so copy the k projection into the V buffer here and normalize it
					// below, before rope_kv_batched rotates k. Mirrors segA's decode path op for op. It used to project k a SECOND time "because decode does": decode now copies too
					// (R-23, docs/tasks/task-recompute-audit.md), so the two still agree, and neither reads the k weight twice. kBb and vBb are [M, kvDim] row-major, so one contiguous
					// copy of M*kvDim floats moves every row.
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
					// MC3 on CUDA: rope, the KV store and attention per sequence, through decode's own gap, each row on its own
					// slot at its own position; the batched stack resumes at the ctx-quant below.
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
						Arg(qBb), Arg(kBb), Arg(vBb), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.vc[l]),
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
						// qTempRows (Ministral 3, FeatAttnTemp): this launch covers M rows at different
						// positions (startPos+m), so each row's post-RoPE query scale comes from a table
						// built on the host with launchToken's own float64 expression. That keeps a batched
						// row bit-identical to decode (audit-2026-09-10 G-11). Not carried into the m-RoPE
						// branch above: rope_kv_mrope_batched has no attention temperature, and no family
						// needs both today (Ministral 3 has no m-RoPE, Qwen2.5-VL no temperature).
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
						// Widened per checkPrefillShmemImgBlocks's SAME formula (imgBlocksMaxNWin) — the two must
						// never drift apart, or this allocation under-sizes the launch it is meant to cover.
						maxNWin = imgBlocksMaxNWin(maxNWin, int(Ly.window), blocks)
					}
					t = r.profTic()
					// L2: the fused kernel when it serves this (hd, M), else attn_batched. Identical
					// argument list by construction, so the two launches differ only in pipeline, grid and
					// shared memory — see useAttnFused for the selection rule.
					// The argument list is IDENTICAL for all three kernels by construction, so it is built
					// once; the launches differ only in pipeline, grid and shared memory. Each names its
					// pipeline field directly — see useAttnFused for why a local variable will not do.
					attnArgs := []gpu.KernelArg{
						Arg(qBb), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)),
						gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(r.attnScale),
						// N-10: r.sinkArg(l), not ArgNull(). The decode launches thread the gpt-oss
						// learned sink through and this one hard-coded null — unreachable today only
						// because every gpt-oss model is MoE and MoE declines batched prefill, which
						// is a property of a DIFFERENT check and not something this call site should
						// depend on.
						gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB), r.sinkArg(l),
					}
					var attnErr error
					// FLASH-DECODE LANE FOR SPECULATIVE VERIFY (attn-decode-fa-verify-PREREGISTERED.md): rows [laneFrom, M) of an all-rows
					// verify batch are served by the multi-row lane, whose output is bit-identical to the M=1 lane at each row's position
					// (TestFlashDecodeRowsBitIdentical), so a verified position scores exactly as plain lane decode scores it. Rows below
					// laneFrom are under the attended-span floor and take the exact path below, as their M=1 decode would. laneFrom == M is
					// "no lane rows" (the common case, and everything when the lane is off). laneFrom == 0 skips the exact launch entirely.
					laneFrom := r.verifyLaneFrom(l, startPos, M, tail, len(blocks) > 0)
					// IMAGE BLOCK FIRST, unconditionally, before useAttnFused is even consulted — attn_fused's
					// tile-level aggregates assume monotonic per-row nKeys across a 64-row tile, which an image
					// block breaks (not supported; attn_batched's exact-path twin, attn_img_batched, is used
					// instead). Checking imgEnd>imgStart after useAttnFused would risk silently routing a >=512
					// -token image prompt through the incompatible fused kernel instead of declining to it.
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
						// Default tile (attn-fused-tile128-default-PREREGISTERED.md): hd128 layers with no sliding window run the
						// 128-row-tile kernel (bit-identical to the 64x64 one for window == 0, 2.4x faster at K=3900). hd64 stays 64x64
						// (0.5B measured ~5% slower with it), and so do windowed layers (their key-tile grouping starts at the block's
						// first row, so a taller block is not bit-identical there). GOINFER_CUDA_ATTN_FUSED_TILE=64x64 forces 64x64.
						tile := r.attnTile
						if tile == 0 && hd == 128 && Ly.window <= 0 && r.bAttnBM128hd128 != (Pipeline{}) {
							tile = 3
						}
						if tile > 0 {
							// R5 phase-1/diagnostic arms (attn-fused-tile-PREREGISTERED.md): query tile 32 / 32 / 128 rows.
							// Each pipeline is named at its launch (TestPipelineLint_boundKernelsAreLaunched keys on that).
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
				// R11/P20 (cuda/moe_expert_major.go + moe_expert_major_gemma4.go,
				// docs/measurements/p20-expert-locality-2026-09-21.md): route+bucket+admit-once-per-distinct-
				// expert instead of once per (row, rank), when eligible. Checked once per layer, not per
				// row, so ineligible layers pay nothing beyond one field-and-map read.
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
						// Between ROWS: this loop is the one that made cancellation coarse. A MoE
						// chunk is M sequential per-token FFNs, so without this a cancelled 512-row
						// chunk still runs every one of them — measured ~22 s on M26, against the
						// ~46 ms the per-token fallback it replaced would have taken to notice.
						// Checked per row, so the granularity is back to roughly one token.
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
					// Gemma 4 dense per-layer output scalar — segB's sequential-decode twin (the
					// decode path's own fix, cuda/resident.go). fScaleVec is a pure elementwise
					// dst[i]*=s with no per-row structure, so it batches over the flattened
					// M*hidden buffer exactly like bRes above, in one launch. Missing this here
					// (while segB had it) is exactly what TestPrefillNonUniform_bitIdentical exists
					// to catch: batched prefill and sequential decode diverging on a real per-layer
					// value, not agreeing on a shared no-op.
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
			// DeepStack (S10, G-S10g): after decoder layer l, set l is added to the image rows, as the CPU prefill does (addDeepstack; HF's hidden_states[visual_pos_masks] += deepstack_visual_embeds[l]).
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
		if tail == tailResidualAll {
			// The ONLY tail that reads the whole residual on the host (audit R-24): the argmax and all-logits heads below read xB on the device, and
			// the last-row tails need one row, copied device-to-device. Downloading all M rows for them was M*hidden*4 bytes (23 MB at M=2048 on a
			// 2816-wide model) of pageable D2H after every pass, for nothing.
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
			// Every row needs the head (PrefillLastN, speculative verify, MC3's step) — batch it:
			// ONE norm+quant+GEMV over all M rows instead of M single-row calls, the same batched
			// primitives batchedHeadArgmax already uses and bGemvB's own doc comment documents as
			// bit-identical to the per-row GEMV by construction. See batchedHeadFull's own doc
			// comment for what stays per-row (draw, softcap, logit scale, host download) and why.
			var e error
			if outs, ids, e = r.batchedHeadFull(xB, aqB, aScB, M, rows); e != nil {
				return e
			}
			return r.launchErr
		}
		// tailLastLogits / tailHiddenLast: only the LAST row's output is ever used, so batching the
		// head here would compute M-1 rows nobody reads — the exact regression the tailAllLogits
		// branch above exists to avoid. One row, unchanged from before this lever.
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
				// No head dispatch at all — r.aq/r.aSc (just written by rms above) already hold
				// exactly what the head would have read: the quantized, post-final-norm hidden
				// state. Download and dequantize instead of projecting to vocab.
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
	// SKIPPED for tailHiddenLast: outs holds a HIDDEN STATE there, not logits, and Gemma's softcap
	// is a logit-only transform — applying it here would silently corrupt every G4 embedding on a
	// softcapped family.
	if tail == tailHiddenLast || tail == tailResidualAll {
		return outs, ids, nil
	}
	for _, out := range outs {
		applySoftcap(out, r.finalSoftcap)
		applyLogitScale(out, r.logitScale) // step() applies both; this tail used to apply only the softcap (audit C-04)
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
	// The argmax used to be taken on the HOST after downloading all M×vocab logits ("a batched
	// argmax kernel is a later, separate ~1 ms"). R14 measured that tail at 15-16% of a spec round
	// (docs/measurements/r14-drafter-argmax-2026-09-22.md); argmax_rows now reduces on the device
	// and M ints come back.
	ids := make([]int, M)
	if e := r.argmaxRows(r.logitsB, r.logitsBIdx, M, ids); e != nil {
		return e
	}
	*out = ids
	return r.launchErr
}

// batchedHeadFull is prefillCore's batched head for tailAllLogits, where every row's FULL logits
// are needed (PrefillLastN, speculative verify, MC3's StepBatch) — unlike batchedHeadArgmax, which
// only ever returns an id per row. It replaces the M single-row upload+norm+GEMV calls the tail
// used to make with ONE batched norm+quant (bNormB) and ONE batched GEMV (bGemvB) over all M rows
// — the same primitives batchedHeadArgmax already uses, and bGemvB's own doc comment documents its
// int8 kernel as "bit-identical to gemv_w8a8_fwd by construction" (exact int32 accumulation; tiling
// M changes no element) — the S0 measurement (concurrency-mc3-cuda-s0-2026-09-27.md) found the same
// holds for the int4 kernels used elsewhere in this same batched pass. The GEMV was the only
// redundant part of the old loop: it read the SAME lm_head weights M times over.
//
// Everything downstream of the GEMV stays exactly as the loop it replaces did it, per row:
//   - a step row with an on-device draw (rows[m].draw != nil) still goes through the unmodified
//     stepDraw/gumbelPick kernels, which read the single-row r.logits field, not a batched buffer —
//     rather than touching those kernels, this copies that row's slice of the batched logits into
//     r.logits first (gpu.CopyDevice, a existing, documented, synchronous device-to-device verb),
//     which is cheap for one vocab-sized row and leaves the sampling kernels themselves untouched;
//   - every other row is downloaded to the SAME pinned host buffer the per-row loop already used
//     (r.logitsPinned/r.logitsHost), one row at a time — the download itself was never the
//     redundant part, since M rows always need M separate host slices regardless of how the GEMV
//     that produced them was dispatched;
//   - softcap and logit scale are applied per row exactly as before (rows != nil only — the same
//     condition the loop already used, for the same reason: a step row must equal Forward's own
//     return, not only the device buffer's raw contents).
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

// bRmsB launches rmsnorm_quant_batched over M rows (shared = [blockDim]+[hidden]).
// attnTempRows returns rope_kv_batched's per-row query-scale table for rows at positions
// [startPos, startPos+M). Each entry is launchToken's float64 expression, rounded to float32 once,
// so a batched row's Q matches decode bit for bit (audit-2026-09-10 G-11). A family without an
// attention temperature gets a NULL argument, which the kernel reads as scale 1. The table is
// rebuilt only when (startPos, M) changes, so every layer of one prefill shares one upload.
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

// fastPrefillFloor is the PROMPT-LENGTH floor below which neither fast lever engages.
//
// It is set from measurement, not chosen. Two independent lines of evidence put it here and they
// were taken in that order:
//
//   - PERFORMANCE (Phase 1/2, measured before any fidelity run): L2 is 0.94x — SLOWER — at K=128,
//     where attention is 5.0% of prefill and a 64-row query tile still stages 64 keys per block.
//     The win begins by K=512 (1.13x) and grows. L3 wins at every depth measured (1.85x at K=128).
//   - FIDELITY (Phase 3, docs/measurements/prefill-l2l3-phase3-2026-09-05.md): the §3 gate PASSES
//     at K>=512 and FAILS at K=256 for the two levers combined, on both S and D7.
//
// THE FLOOR IS AT A DEPTH THAT WAS ACTUALLY MEASURED. The §3 decision cells were 256 and 1024; a
// floor placed between them would have been interpolating a fidelity result nobody took, so a
// K=512 cell was generated and run specifically to justify this number. Moving it DOWN requires a
// passing gate cell at the new depth — not an argument that the curve looks smooth.
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

// attnFusedShmem is the dynamic shared memory attn_fused needs: Ksh[BN][hd+KPAD] plus
// Vtsh[hd][BN+KPAD], in halves.
//
// IT IS CONSTANT IN K, and that is a property worth stating rather than a detail. The exact path
// sizes its attention scratch (maxNWin+128)*4, so checkPrefillShmem must decline any layer
// attending more than 12,160 keys (resident.go:143) — past that a prompt falls back to the
// sequential per-token path. This kernel's footprint does not grow with the attended span at all.
func attnFusedShmem(hd int) uint32 {
	return uint32(2 * (attnFusedBM*(hd+attnFusedKPAD) + hd*(attnFusedBM+attnFusedKPAD)))
}

// attnFusedFor returns the fused pipeline serving this head dim, or the zero Pipeline if none does.
// hd 64 and 128 are the two instantiations attn_fused.cu emits; anything else — and any hd on a
// build where the module did not load — is served by attn_batched instead. Declining is the whole
// answer for an unsupported shape: there is no special case to write, because the exact path
// already handles every shape correctly.
func (r *cudaResident) attnFusedFor(hd int) (Pipeline, uint32) {
	switch hd {
	case 64:
		return r.bAttnFused64, attnFusedShmem(64)
	case 128:
		return r.bAttnFused128, attnFusedShmem(128)
	}
	return Pipeline{}, 0
}

// useAttnFused is the ONE place the L2 kernel is chosen, so the fallback cannot drift between call
// sites. Every "no" means attn_batched, which is the exact path, is bit-identical to decode, and is
// what spec-decode verify and the parity gates run.
//
// THAT LAST CLAIM WAS ASPIRATIONAL UNTIL M-11 (docs/audit-2026-09-10.md): before
// forceExactKernels existed, the only gates here were M/floor-based (attnFusedMinRows,
// aboveFastPrefillFloor), which verify's own M (often >=16 past the floor) could satisfy by
// coincidence of constants, not by construction — "is what spec-decode verify runs" was true of
// serve's DEFAULT widths, not guaranteed for every caller. forceExactKernels (set by prefillCore,
// cuda/resident.go's own field comment) now makes it a real guarantee: unconditional "no" for any
// tail but tailLastLogits, independent of M/K/position.
// It returns only the shared-memory size, NOT the pipeline: the launch site names
// r.bAttnFused64 / r.bAttnFused128 explicitly. Handing back a Pipeline in a local variable would
// hide WHICH kernel runs from every static reader, including
// TestPipelineLint_boundKernelsAreLaunched, which flags a field bound at every model load and
// launched by nothing — the exact state gemv_w4a8_batched sat in while a benchmark quoted its
// throughput as the shipping kernel's.
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
	// gemmMMAMinRows: tensor cores lose below a warp's worth of rows, and §4 L3 pre-registers
	// gemv_w4a8_rn — the EXACT path — as the M<16 path for that reason. 16 is two m8n8k16 M-tiles.
	gemmMMAMinRows = 16
)

// gemmMMAShmem is the activation panel: [GBM][GKSTEP/4 + GAPAD] int words. Constant, 9 KB.
func gemmMMAShmem() uint32 {
	return uint32(gemmMMABM * (gemmMMAKStep/4 + gemmMMAAPad) * 4)
}

// useGemmMMA is the ONE place the L3 kernel is chosen. Every "no" means gemv_w4a8_rn, which is the
// exact path, is bit-identical to the M=1 decode GEMV, and is what the parity gates run.
//
// M-09/M-11 (docs/audit-2026-09-10.md): before forceExactKernels (see its own doc comment,
// cuda/resident.go), the only gate here was shape-based (gemmMMAMinRows, aboveFastPrefillFloor) —
// speculative verify's own M (often >=16) could cross it by coincidence, not by construction, the
// same M-dependent-kernel gap WebGPU's staged int4 path has independently (M-09). forceExactKernels
// now makes exactness unconditional for any prefillCore tail but tailLastLogits.
//
// The K constraints are not defensive padding: the kernel contracts 32 elements per group scale and
// 8 per packed weight word, so a K that is not a multiple of 32 would misalign the group-scale fold.
// Every production shape here satisfies it (1536, 3584, 8960, 18944 are all multiples of 32), and a
// shape that does not is served correctly by the exact path rather than by a special case.
func (r *cudaResident) useGemmMMA(kind string, K, M int) bool {
	return !r.forceExactKernels && r.fastGemm && kind == "int4" && M >= gemmMMAMinRows &&
		r.aboveFastPrefillFloor() && r.bGemmMMA != (Pipeline{}) && K%32 == 0
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
