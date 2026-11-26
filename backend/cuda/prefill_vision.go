//go:build cuda

package cuda

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
)

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

// mropePosWindow returns the [startPos, startPos+M) slice of the WHOLE-prompt, absolute-indexed mropePos. prefillChunked
// passes embeddings chunk-relative (embeddings[i:i+n]) but mropePos whole on every call, so slicing by the chunk loop's
// own i instead of the absolute startPos silently reintroduces the wrong rotation, and no real-checkpoint gate whose
// prompt is too short to chunk would see it (cuda/mropepos_window_test.go covers exactly this).
func mropePosWindow(mropePos [][3]int, startPos, M int) [][3]int {
	return mropePos[startPos : startPos+M]
}
