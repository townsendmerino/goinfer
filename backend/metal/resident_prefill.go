//go:build darwin

package metal

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// exactAttnNote is PrefillPath's account of the exact attention kernel's key bound, "" when every attention layer has a
// fused kernel: past the bound the batched pass declines, and fallback names the path that then runs the prompt.
func (a *metalResident) exactAttnNote(fallback string) string {
	if !a.r.prefillExactAttn() {
		return ""
	}
	if hd := a.r.exactAttnHeadDim(); hd > exactAttnTiledMaxHD {
		return fmt.Sprintf("; past %d keys the exact attention kernel (head dim %d has no fused kernel, or fused attention is off, and is wider than its tiled form's %d) declines it to %s",
			prefillExactAttnMaxKeys, hd, exactAttnTiledMaxHD, fallback)
	}
	return fmt.Sprintf("; past %d keys the exact attention kernel (head dim %d has no fused kernel, or fused attention is off) tiles its scores",
		prefillExactAttnMaxKeys, a.r.exactAttnHeadDim())
}

// prefillDeclineForTest, when non-nil, is the error batchedPrefillImg declines every pass with, so a test reaches a fallback
// without a model that trips a real one.
var prefillDeclineForTest error

// emodelFallbackOut is where an E-model prefill fallback is announced; a variable so a test reads it.
var emodelFallbackOut io.Writer = os.Stderr

// emodelFallbackSeen holds each E-model prefill decline reason already reported, with its digits masked.
var emodelFallbackSeen sync.Map

// warnEModelFallback reports, once per distinct reason, that the batched pass declined a Gemma 4 E-model's prompt and the
// layer-major pass ran it instead. The decoder reports its own declines the same way (warnPrefillDeclined); this one
// never reaches the decoder, because PrefillLast takes the fallback itself.
func warnEModelFallback(err error) {
	key := strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return '#'
		}
		return r
	}, err.Error())
	if _, seen := emodelFallbackSeen.LoadOrStore(key, true); seen {
		return
	}
	fmt.Fprintf(emodelFallbackOut, "metal: the batched prefill declined a Gemma 4 E-model prompt (%v); the layer-major pass runs it on decode's kernels, "+
		"bit-identical to sequential (each distinct reason is reported once)\n", err)
}

// fastPrefill reports whether this resident's prefill takes the batched fast path.
func (a *metalResident) fastPrefill() bool {
	a.fastOnce.Do(func() {
		a.fast = !a.exact && metalFastPrefillEnabled(a.r.knobValue("GOINFER_METAL_FAST_PREFILL"), a.r.knobValue("GOINFER_METAL_BATCHED_PREFILL"))
	})
	return a.fast
}

// PrefillKernelFloor (decoder.ResidentCaps.KernelFloor) is the whole-prompt length from which the batched
// prefill runs the f16 MMA kernels, below which it runs decode rows or the sequential loop: two kernel classes
// that are not bit-identical. 0 when one class runs at every length (a layer-major path, int8 KV, an arch the
// MMA kernels do not cover, fast prefill off, or GOINFER_METAL_FAST_PREFILL_FLOOR=0). PrefillPath reports the same
// number; both read prefillKernelFloor.
func (a *metalResident) PrefillKernelFloor() int { return a.prefillKernelFloor() }

func (a *metalResident) prefillKernelFloor() int {
	if a.r == nil || a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() || a.r.kvF32 || a.r.attnSink || !a.r.prefillOK || !a.fastPrefill() {
		return 0
	}
	floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if floor > 0 && a.stepPrefill() && !a.r.promptStepOff {
		return a.stepPrefillBound(floor)
	}
	return floor
}

// PrefillPath (decoder.PrefillPathReporter) reports at load time whether this resident's prefill takes the batched f16-MMA
// path, and why or why not. The floor applies per call (PrefillKernelFloor); it reports true iff the enabled state and the arch
// both allow batching.
func (a *metalResident) PrefillPath() (bool, string) {
	if a.g4LayerMajor() || a.moeLayerMajor() {
		return true, "layer-major on decode's kernels (bit-identical to sequential; a paged MoE)"
	}
	if a.emodelLayerMajor() {
		if a.emodelBatched() && a.fastPrefill() {
			floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
			return true, fmt.Sprintf("batched f16-MMA from %d prompt tokens (a Gemma 4 E-model, S9 step 2); layer-major on decode's kernels below it (bit-identical to sequential)", floor) + a.exactAttnNote("the layer-major pass")
		}
		return true, "layer-major on decode's kernels (bit-identical to sequential; a Gemma 4 E-model, S9)"
	}
	if a.r.kvF32 {
		if a.stepPrefill() && !a.r.promptStepOff { // promptStepOK takes every prompt
			return true, "decode rows on the batched step kernels at every length (bit-identical to sequential; the f16 MMA pass does not take this model's f32 (-kv f32) KV cache)"
		}
		return false, "sequential — the f16 MMA prefill kernels write half-precision K/V, and this model's KV cache is f32 (-kv f32)"
	}
	if a.r.attnSink {
		return false, "sequential — the batched prefill kernels implement neither gpt-oss's attention sink nor its clamped SwiGLU with biases"
	}
	if !a.r.prefillOK {
		if a.fastPrefill() && a.stepPrefill() && !a.r.promptStepOff { // promptStepOK takes every prompt
			return true, "decode rows on the batched step kernels at every length (bit-identical to sequential; the f16 MMA pass does not take this model)"
		}
		return false, "sequential — arch/geometry not supported by f16 MMA prefill kernel"
	}
	if !a.fastPrefill() {
		return false, "sequential — fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill)"
	}
	floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if floor > 0 && a.stepPrefill() && !a.r.promptStepOff {
		return true, fmt.Sprintf("batched f16-MMA from %d prompt tokens; below it, decode rows on the batched step kernels (bit-identical to sequential, E-P01)", a.prefillKernelFloor()) + a.exactAttnNote("the per-token path") + a.int8PassNote()
	}
	if floor > 0 {
		return true, fmt.Sprintf("batched f16-MMA above %d prompt tokens; sequential below (§3 floor)", a.prefillKernelFloor()) + a.exactAttnNote("the per-token path") + a.int8PassNote()
	}
	return true, "batched f16-MMA (GOINFER_METAL_FAST_PREFILL_FLOOR=0; §3.2 gate passed 2026-09-09)" + a.exactAttnNote("the per-token path") + a.int8PassNote()
}

// int8PassNote is PrefillPath's account of an int8 cache on the MMA pass, "" for any other cache.
func (a *metalResident) int8PassNote() string {
	if !a.r.kvI8 {
		return ""
	}
	return "; the int8 KV cache is stored by the pass and dequantized per layer into f16 scratch for its attention"
}

// g4LayerMajor reports whether this resident's prompts take prefillG4Paged: a paged Gemma 4 MoE with no learned
// positions (the layer-major rows add none), with g4LayerMajorOn.
func (a *metalResident) g4LayerMajor() bool {
	return g4LayerMajorOn && a.r.g4moe != nil && a.r.g4moe.paged && !a.r.learnedPos
}

// moeLayerMajor is g4LayerMajor for the generic pager (prefillMoEPaged): a paged generic MoE, not gpt-oss (its sink and
// biased experts), with moeLayerMajorOn.
func (a *metalResident) moeLayerMajor() bool {
	return moeLayerMajorOn && a.r.moe != nil && a.r.moe.paged && !a.r.moe.isGptOss && !a.r.learnedPos
}

// PrefillLast (decoder.Prefiller) ingests the whole prompt in one batched f16-MMA pass and
// returns the last token's logits, populating the resident KV. Falls back (declines) for prompts
// shorter than the fast-prefill floor or longer than the resident KV/attention cap.
func (a *metalResident) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return a.prefillLastSrc(ctx, sliceRows(embeddings), startPos)
}

// prefillLastSrc is PrefillLast over a row source (rows.go).
func (a *metalResident) prefillLastSrc(ctx context.Context, src rowSource, startPos int) ([]float32, error) {
	// One check at entry: the pass is a single command buffer with no inner loop to interrupt. The decoder calls it once a
	// block (PrefillBlockRows) or once a chunk while another conversation decodes, so a long prompt is cancelled between
	// those calls; a resident that reports no block runs a lone prompt as one uninterruptible pass.
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	// A paged Gemma 4 MoE, a paged generic MoE and a Gemma 4 E-model run their prompts layer by layer on decode's own kernels,
	// bit-identical to the sequential loop, so neither the floor nor --exact-prefill applies to them (each behind its own switch;
	// ). An E-model prompt at or above the floor takes the batched pass when it is on; below the
	// floor, or on any decline, the layer-major pass below runs it.
	if a.emodelBatched() && a.fastPrefill() && startPos+src.n >= metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR")) {
		if src.width == a.embLen() {
			lg, err := a.batchedPrefillSrc(src, startPos, 0, nil, nil, nil)
			if err == nil {
				return lg, nil
			}
			// Once per request where the request carries a prefill report (serve attaches one to every generation), else once
			// per distinct reason: the pass may be called several times in one generation (a block or a chunk at a time).
			if first, tracked := decoder.NotePrefillFallback(ctx, "layer-major", err.Error()); first {
				fmt.Fprintf(emodelFallbackOut, "metal: this request's Gemma 4 E-model prompt (%d tokens) left the batched prefill for the layer-major pass on decode's kernels, bit-identical to sequential: %v\n",
					startPos+src.n, err)
			} else if !tracked {
				warnEModelFallback(err)
			}
		}
	}
	if a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() {
		if e := a.checkPass(startPos, src.n); e != nil {
			return nil, e
		}
		if startPos == 0 {
			a.Reset() // a fresh sequence, as Forward at position 0 does (it zeroes a DeltaNet's state)
		}
		var lg []float32
		if a.emodelLayerMajor() { // a Gemma 4 E-model, layer by layer on decode's kernels (prefill_emodel.go)
			if src.width != a.embLen() {
				return nil, fmt.Errorf("metal: an E-model prefill row of %d values, want %d ([h | L*P])", src.width, a.embLen())
			}
			lg = a.r.prefillEModelSrc(src, startPos, true)
		} else if a.g4LayerMajor() {
			lg = a.r.prefillG4PagedSrc(src, startPos, true)
		} else {
			lg = a.r.prefillMoEPagedSrc(src, startPos, true)
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err
		}
		return lg, nil
	}
	// The batched pass is on by default (metalFastPrefillEnabled); GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill opts out.
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill / GOINFER_METAL_BATCHED_PREFILL=0); using sequential path")
	}
	// Below metalFastPrefillFloor the batched pass declines. A resident with the step kernels runs a prompt that ends below
	// stepPrefillBound as decode rows on them: the sequential loop's bits, faster than the pass there
	// (promptStepOK).
	promptLen := startPos + src.n
	floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if a.promptStepOK(src.n, startPos, floor) {
		if e := a.checkPass(startPos, src.n); e != nil {
			return nil, e
		}
		if startPos == 0 {
			a.Reset() // a fresh sequence: the bound slot's DeltaNet state, as Forward at position 0 zeroes it
		}
		return a.prefillByStepSrc(ctx, src, startPos)
	}
	// One token whose whole prompt ends in the step's range (a repeated prompt's 1-token suffix, PrefillTailExact): the
	// step needs two rows, and the pass would not compute the step's bits, which the cold prompt of this length ran on.
	// The sequential path's decode does.
	if src.n < 2 && a.inStepRange(promptLen, floor) {
		return nil, fmt.Errorf("metal: a 1-token suffix in the batched step's range runs on the sequential path (the step's bits)")
	}
	return a.batchedPrefillSrc(src, startPos, floor, nil, nil, nil)
}

// prefillUnaligned names the first reduction length the batched pass's GEMMs cannot take, or "" when every one is a multiple of
// 32: gemm_w4f16_tile and its int8 twin stage K in slabs of 32 with no partial-slab handling (prefill.go). An int4 resident
// cannot violate it (int4Buf refuses K%32 != 0 at build), but a native int8 one can. Real checkpoints are 32-aligned; this keeps an unaligned model on the sequential path instead of computing garbage.
func (r *resident) prefillUnaligned() string {
	if r.H%32 != 0 {
		return fmt.Sprintf("the hidden size is %d", r.H)
	}
	for l := range r.layers {
		L := &r.layers[l]
		if L.geom != nil && (r.nH*L.geom.hd)%32 != 0 {
			return fmt.Sprintf("layer %d's attention output width is %d", l, r.nH*L.geom.hd)
		}
		if fi := L.ffnI; L.moe == nil && fi == 0 && r.I%32 != 0 {
			return fmt.Sprintf("the FFN width is %d", r.I)
		} else if fi%32 != 0 {
			return fmt.Sprintf("layer %d's FFN width is %d", l, fi)
		}
	}
	return ""
}

// batchedPrefill is the f16 batched pass behind PrefillLast and PrefillMRoPELast: the same declines (the floor, an
// int8 KV cache, a family the pass does not implement, the resident's cap, the exact attention kernel's key limit), the
// same recovery of a request-time panic, and the same non-finite-logit check. mrope is nil for a text prompt.
func (a *metalResident) batchedPrefill(embeddings [][]float32, startPos, floor int, mrope [][3]int, deep *prefillDeep) ([]float32, error) {
	return a.batchedPrefillImg(embeddings, startPos, floor, mrope, deep, nil)
}

// PrefillImageLast (decoder.ResidentPrefillSubsystem) is the batched f16 pass over an image turn's spliced embeddings, with the image
// block [imgStart, imgEnd) attending bidirectionally (attention_prefill_img), as CUDA's PrefillImageLast and the CPU's
// prefillLogitsVL do. The whole prompt is one pass, never chunked across the block. It takes the plain uniform pass only: a
// Gemma 4 E-model (its rows carry per-layer inputs this route does not build), a layer-major family or a recurrent hybrid
// declines, and the decoder takes its CPU-prefill and upload bridge.
func (a *metalResident) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return a.PrefillImageBlocksLast(ctx, embeddings, startPos, [][2]int{{imgStart, imgEnd}})
}

// PrefillImageBlocksLast (decoder.ResidentPrefillSubsystem) is PrefillImageLast for several images: each block
// [start, end), in order and disjoint, attends bidirectionally within itself.
func (a *metalResident) PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) ([]float32, error) {
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	prev := startPos
	for _, b := range blocks {
		if b[0] < prev || b[1] <= b[0] || b[1] > startPos+len(embeddings) {
			return nil, fmt.Errorf("metal: image blocks %v invalid for %d rows at %d", blocks, len(embeddings), startPos)
		}
		prev = b[1]
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("metal: no image block")
	}
	if !a.r.prefillOK || a.r.pleP > 0 || a.r.recurrentModel() || a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() {
		return nil, fmt.Errorf("metal: the resident image prefill takes the plain uniform pass; this model's image turn prefills on the CPU")
	}
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill); the image prefill runs on the CPU")
	}
	// attention_prefill_img is the exact kernel: its score buffer holds prefillExactAttnMaxKeys keys.
	if startPos+len(embeddings) > prefillExactAttnMaxKeys {
		return nil, fmt.Errorf("metal: an image turn of %d positions is past the image prefill's %d-key limit; the image prefill runs on the CPU",
			startPos+len(embeddings), prefillExactAttnMaxKeys)
	}
	return a.batchedPrefillImg(embeddings, startPos, 0, nil, nil, blocks)
}

var ()

// batchedPrefillImg is batchedPrefill with optional bidirectional image blocks (nil: none).
func (a *metalResident) batchedPrefillImg(embeddings [][]float32, startPos, floor int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) ([]float32, error) {
	return a.batchedPrefillSrc(sliceRows(embeddings), startPos, floor, mrope, deep, blocks)
}

// batchedPrefillSrc is batchedPrefillImg over a row source (rows.go).
func (a *metalResident) batchedPrefillSrc(src rowSource, startPos, floor int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) ([]float32, error) {
	if prefillDeclineForTest != nil {
		return nil, prefillDeclineForTest
	}
	promptLen := startPos + src.n
	if floor > 0 && promptLen < floor {
		return nil, fmt.Errorf("metal: prompt too short (%d tokens) for fast prefill (floor=%d; §3 floor); using sequential path", promptLen, floor)
	}
	// The f16 MMA prefill kernels implement a dense gated FFN (SiLU or GeGLU) and a generically shaped gated-SwiGLU MoE FFN
	// (expert-major by default, the per-token decode MoE chain row by row with GOINFER_MOE_EXPERT_MAJOR=0), with per-layer
	// rope/window, per-head QK-norm and Gemma's sandwich norms. They do NOT implement per-layer-varying attention geometry (dense
	// Gemma 4's local/global head_dim split; prefillOK) or Gemma 4's enable_moe_block variant (HasGemma4MoE, a third FFN
	// shape). Any of these declines here, and the caller re-runs the prompt through the sequential Forward loop.
	if a.r.kvF32 {
		return nil, fmt.Errorf("metal: prefill writes half-precision K/V and this model's KV cache is f32 (-kv f32); using sequential path")
	}
	if a.r.attnSoftcap != 0 && len(blocks) > 0 {
		return nil, fmt.Errorf("metal: the image prefill's attention kernel is not gated with an attention softcap; using sequential path")
	}
	if !a.r.prefillOK && !a.emodelBatched() {
		return nil, fmt.Errorf("metal: prefill not implemented for this arch's FFN shape (use the sequential path)")
	}
	if a.r.layerNorm && (a.r.layerNormBias || a.r.qkNorm) {
		return nil, fmt.Errorf("metal: the prefill's LayerNorm is bias-free with no QK-norm, and this model has a LayerNorm bias or QK-norm; using sequential path")
	}
	if why := a.r.prefillUnaligned(); why != "" {
		return nil, fmt.Errorf("metal: the prefill GEMMs need every reduction length a multiple of 32 and %s; using sequential path", why)
	}
	// startPos < 0 would wrap to a huge uint32 and make kv_store_f16 write far out of bounds, silently corrupting adjacent buffers on
	// unified memory. The decoder never passes one, but the guard is cheap.
	if startPos < 0 || src.n == 0 || startPos+src.n > a.ctxCap() {
		return nil, fmt.Errorf("metal: prompt len %d at startPos %d out of resident cap %d", src.n, startPos, a.ctxCap())
	}
	// The exact attention kernel (no fused kernel for this head dim, or fused attention off) holds prefillExactAttnMaxKeys
	// scores; past that the pass takes its tiled twin (exactAttnTile), which keeps at most exactAttnTiledMaxHD output dims.
	if hd := a.r.exactAttnHeadDim(); a.r.prefillExactAttn() && startPos+src.n > prefillExactAttnMaxKeys && hd > exactAttnTiledMaxHD {
		return nil, fmt.Errorf("metal: prompt reaches %d keys, past the exact prefill attention kernel's %d, and head dim %d is wider than its tiled form's %d",
			startPos+src.n, prefillExactAttnMaxKeys, hd, exactAttnTiledMaxHD)
	}
	// A fresh sequence starts a Gated-DeltaNet hybrid's recurrent state from zero, as Forward(pos 0) does; a continuation keeps the
	// conv window and state the resident holds. No-op for every other family.
	if startPos == 0 {
		a.Reset()
	}
	// ensurePrefill's compile panic and the per-call MustBuf OOM panics fire here, at request time, outside buildResident's recover.
	// Recover into an error so a transient OOM fails the request (the caller falls back to sequential decode) rather than killing
	// the server.
	var logits []float32
	if err := func() (err error) {
		defer func() {
			if p := recover(); p != nil {
				mem.RepanicBug(p)
				err = fmt.Errorf("metal: batched prefill aborted: %v", p)
			}
		}()
		logits = a.r.prefillLastImgSrc(src, startPos, mrope, deep, blocks)
		return nil
	}(); err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err
	}
	// The f16 residual overflowed: the head reads the NaN row as finite zeros, so the logit check below cannot see it.
	if a.r.prefillResidNonFinite {
		return nil, fmt.Errorf("metal: batched prefill's f16 residual overflowed (its last row is non-finite after the trunk); using sequential path")
	}
	if a.poisonPrefillLogitsForTest {
		logits[0] = float32(math.NaN())
	}
	// The batched pass carries f16 activations, which a checkpoint the graded models never exercised could overflow; the sequential
	// kernels carry f32. A non-finite logit declines, and the decoder re-runs the prompt sequentially, which rewrites every K/V row
	// this pass wrote. The scan costs about 0.05 ms at a 152k vocabulary.
	if i := firstNonFinite(logits); i >= 0 {
		return nil, fmt.Errorf("metal: batched prefill produced a non-finite logit (%v at %d); using sequential path", logits[i], i)
	}
	return logits, nil
}

// PrefillMRoPELast (decoder.ResidentPrefillSubsystem) is PrefillLast for a
// Qwen image turn: the batched f16 pass with every row rotated by its own (temporal, height, width) m-RoPE triple, so the
// image turn prefills here instead of on the CPU with its K/V uploaded after. mropePos covers the whole prompt from 0. It
// declines (the decoder then takes the CPU-prefill+UploadKV bridge) when the model has no m-RoPE axis table, when a
// layer-major route owns this family, or when the batched pass itself would; the batched step never takes it (its kernels
// rotate by one scalar position).
func (a *metalResident) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	return a.prefillMRoPE(ctx, embeddings, startPos, mropePos, nil)
}

// PrefillMRoPEDeepstackLast (decoder.ResidentPrefillSubsystem) is PrefillMRoPELast for Qwen3-VL: the same pass,
// which also adds DeepStack set l to the image rows after decoder layer l. Each set must be imgLen rows of the hidden size.
func (a *metalResident) PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) ([]float32, error) {
	if imgStart < 0 || imgLen <= 0 || len(deep) > a.r.nL {
		return nil, fmt.Errorf("metal: DeepStack prefill of %d sets over [%d,%d) for %d layers", len(deep), imgStart, imgStart+imgLen, a.r.nL)
	}
	for l, set := range deep {
		if len(set) != imgLen*a.r.H {
			return nil, fmt.Errorf("metal: DeepStack set %d has %d values, want %d x %d", l, len(set), imgLen, a.r.H)
		}
	}
	return a.prefillMRoPE(ctx, embeddings, startPos, mropePos, &prefillDeep{start: imgStart, n: imgLen, sets: deep})
}

// metalMRoPEPrefillOn and metalDeepstackPrefillOn turn the resident m-RoPE prefill on, for a turn without DeepStack sets
// (Qwen2.5-VL; Qwen3.5+ is not claimed) and with them (Qwen3-VL). Off, an image turn takes the CPU prefill and the upload. Both
// are on by owner decision after the real-checkpoint gates. Tests set them.
var metalMRoPEPrefillOn, metalDeepstackPrefillOn = true, true

func (a *metalResident) prefillMRoPE(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep *prefillDeep) ([]float32, error) {
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if on := metalMRoPEPrefillOn; deep != nil && !metalDeepstackPrefillOn || deep == nil && !on {
		return nil, fmt.Errorf("metal: the resident m-RoPE prefill is off for this model until its fidelity gates are graded; the image prefill runs on the CPU")
	}
	if a.r.mropeAxis == (Buffer{}) {
		return nil, fmt.Errorf("metal: no m-RoPE axis table for this model; the image prefill runs on the CPU")
	}
	if len(embeddings) == 0 || startPos < 0 || len(mropePos) != startPos+len(embeddings) {
		return nil, fmt.Errorf("metal: m-RoPE prefill of %d rows at %d with %d positions (want startPos+rows)", len(embeddings), startPos, len(mropePos))
	}
	if a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() {
		return nil, fmt.Errorf("metal: a layer-major prefill family has no m-RoPE pass")
	}
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill); the image prefill runs on the CPU")
	}
	return a.batchedPrefill(embeddings, startPos, metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR")), mropePos, deep)
}

// firstNonFinite is the index of the first NaN or ±Inf in v, or -1.
func firstNonFinite(v []float32) int {
	for i, x := range v {
		if math.Float32bits(x)&0x7f800000 == 0x7f800000 {
			return i
		}
	}
	return -1
}

// HiddenLast (decoder.ResidentPrefillSubsystem) ingests a whole sequence at startPos and returns the last position's hidden state after
// the model's final norm, for embedding requests: the resident twin of PrefillLast that never runs the LM head. It runs the same
// per-token sequential kernels as decode (one forwardHiddenNoHead and one command-buffer submit per position), so it does not
// match the CPU reference bit for bit: the decode kernels round differently and forwardHiddenNoHead returns the
// int8-dequantized activation (the bar is in hiddenlast_resident_parity_test.go). A batched version, PrefillLast's dispatch
// graph minus the LM head and softcap, gated against this function as its oracle, is not built
// (docs/code-notes/metal.md#metalResident.HiddenLast).
func (a *metalResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: HiddenLast called with no embeddings")
	}
	// forwardHiddenNoHead's trunk encoder has no paged branch, so on a paged MoE it would bind the zero-value stacked expert buffers
	// and return a finite garbage hidden state with no error. Decline instead: decoder.Model.HiddenLast treats a resident decline as
	// "no resident backend for this request" and falls through to the CPU, as it does for an OOM or cap decline.
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return nil, fmt.Errorf("metal: HiddenLast not implemented for paged MoE (no headless paged forward); use the CPU path")
	}
	if e := a.checkPass(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if startPos == 0 {
		a.Reset() // fresh sequence — same DeltaNet-state reset Forward(pos==0) does
	}
	var out []float32
	for i, emb := range embeddings {
		if len(emb) != a.embLen() {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != %d (hidden %d + PLE inputs)", i, len(emb), a.embLen(), a.hidden)
		}
		// An abandoned client must not leave the sequence streaming through the device (residentPrefillSeed's loop checks the same way).
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		want := i == len(embeddings)-1
		h, err := a.r.forwardHiddenNoHead(emb, startPos+i, want)
		if err != nil {
			return nil, err
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err // a command buffer aborted: never return a stale vector
		}
		if want {
			out = h
		}
	}
	return out, nil
}

// ResidualAll (decoder.ResidentPrefillSubsystem) ingests a whole sequence at startPos and returns every position's residual stream
// after the last layer and before the final norm, each row a fresh f32 slice. The decoder applies the final norm on the host in
// f32 (decoder.Model.residualAllResident), so the rows never pass through the int8 the head path reads: Metal fuses the final
// norm into the trunk (encodeTrunkWith: layers, then encodeNorm into r.aq/r.aSc), and that int8 vector is what this avoids.
// Every layer writes its residual into r.x and the norm reads r.x through `device const float*`, so once the trunk has run r.x
// holds the last layer's output untouched; this copies it out per token.
//
// It uses the same per-token sequential kernels as HiddenLast and costs what HiddenLast costs plus an H-float copy per row; a
// batched version is not built. Guards mirror HiddenLast's; rows are returned only once every token has succeeded.
func (a *metalResident) ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) ([][]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: ResidualAll called with no embeddings")
	}
	// The trunk encoder has no paged branch (see HiddenLast): on a paged MoE it would read zero-value expert buffers
	// and return finite garbage. Decline, and the decoder falls back to the CPU.
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return nil, fmt.Errorf("metal: ResidualAll not implemented for paged MoE (no headless paged forward); use the CPU path")
	}
	if e := a.checkPass(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if startPos == 0 {
		a.Reset() // fresh sequence: the same DeltaNet-state reset Forward(pos==0) does
	}
	out := make([][]float32, 0, len(embeddings))
	for i, emb := range embeddings {
		if len(emb) != a.embLen() {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != %d (hidden %d + PLE inputs)", i, len(emb), a.embLen(), a.hidden)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, err := a.r.forwardHiddenNoHead(emb, startPos+i, false); err != nil {
			return nil, err
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err // a command buffer aborted: never return a stale residual
		}
		out = append(out, slices.Clone(a.r.x.Floats()[:a.r.H]))
	}
	return out, nil
}

// prefillBlockScratch is the most row-sized scratch one pass of a blocked prefill may allocate (prefillBlockBudget may
// give less): PrefillBlockRows is the number of rows that fit the budget.
const prefillBlockScratch = 512 << 20

// prefillBlockRowsMin is the shortest block PrefillBlockRows reports, for a model whose rows are so wide that fewer
// would fit prefillBlockScratch.
const prefillBlockRowsMin = 256

// prefillBlockRowsForTest replaces PrefillBlockRows' figure when positive, for a test that needs several blocks in a
// short prompt. It does not make a resident block that otherwise would not.
var prefillBlockRowsForTest int

// PrefillBlockRows (decoder.ResidentPrefillSubsystem) is the most rows the decoder hands one batched pass: as many as fit
// prefillBlockBudget, a multiple of 64, never under prefillBlockRowsMin. The pass allocates its scratch per row, so
// without a block a prompt's prefill takes memory in proportion to its length. 0, meaning the prefill is not split, for a
// resident whose prompts do not take the batched pass at every length: a paged layer-major route, an E-model without its
// batched pass, an f32 KV cache, fast prefill off, an arch or norm the pass does not implement. An exact-attention layer
// blocks only at head dim ≤ 256, where the exact and tiled kernels a block may choose are bit-identical; at 512 the choice
// would have to key on the whole prompt's length.
func (a *metalResident) PrefillBlockRows() int {
	r := a.r
	emodel := a.emodelLayerMajor() && a.emodelBatched()
	if a.g4LayerMajor() || a.moeLayerMajor() || (a.emodelLayerMajor() && !emodel) || r.kvF32 || r.attnSink || (!r.prefillOK && !emodel) ||
		!a.fastPrefill() || (r.layerNorm && (r.layerNormBias || r.qkNorm)) || r.prefillUnaligned() != "" || r.exactAttnHeadDim() > 256 {
		return 0
	}
	if prefillBlockRowsForTest > 0 {
		return prefillBlockRowsForTest
	}
	// A row's price is not the same at every pass length: a long pass fuses the gate|up GEMM with SwiGLU and needs no guF
	// (swigluFused), a short one may not. So the estimate is taken on a long pass, then stepped to the most multiple-of-64 rows
	// that fit.
	const probe = 4096
	per := r.prefillScratchBytes(probe) / probe
	if per <= 0 {
		return 0
	}
	budget := a.prefillBlockBudget()
	B := max(int(budget/per)/64*64, prefillBlockRowsMin)
	for r.prefillScratchBytes(B+64) <= budget {
		B += 64
	}
	for B > prefillBlockRowsMin && r.prefillScratchBytes(B) > budget {
		B -= 64
	}
	if r.kvWin != nil && r.kvWin.bounded { // a bounded window group takes a pass of at most decoder.MetalKVWindowPass rows
		B = min(B, decoder.MetalKVWindowPass)
	}
	return B
}

// prefillBlockBudget is the scratch one blocked pass may take: prefillBlockScratch, or less when the memory guard's budget
// (metalMemoryCeiling, the "metal" probe the fit plan reads) leaves less after this model's need, so the need and one pass
// stay inside that budget together. An unknown need or budget leaves prefillBlockScratch.
func (a *metalResident) prefillBlockBudget() int64 {
	budget := int64(prefillBlockScratch)
	if a.need == nil {
		return budget
	}
	need := a.need()
	ram, err := physicalRAM()
	if need <= 0 || err != nil || ram == 0 {
		return budget
	}
	if ceiling := metalMemoryCeiling(ram); ceiling > 0 {
		budget = min(budget, max(ceiling-need, 0))
	}
	return budget
}

// PrefillScratchBytes (decoder.ResidentPrefillSubsystem) is what one batched pass over rows rows allocates.
func (a *metalResident) PrefillScratchBytes(rows int) int64 { return a.r.prefillScratchBytes(rows) }

// PrefillTailExact (decoder.PrefillTailExact): a short PrefillLast continuing a prefix the pass prefilled reproduces the cold pass
// bit for bit (TestPrefillLast_tailContinuationMatchesCold), and the routing (floor, step range) is on the whole prompt's length.
// Proven for the dense and Gated-DeltaNet passes (TestMC5_prefillChunkInvariance, TestDB01_chunkedPrefillMatchesWhole) and for the
// generic resident MoE's expert-major pass (TestPrefillLast_tailContinuationMatchesCold_MoE). A paged Gemma 4 MoE's layer-major
// path is decode's kernels, so exact by construction; every other MoE shape (paged generic, resident Gemma 4, gpt-oss) declines
// the pass, and its suffix runs the sequential loop.
func (a *metalResident) PrefillTailExact() bool { return true }

// inStepRange: a prompt of promptLen tokens is in the range the batched step takes on this resident (promptStepOK's
// length test, without its two-row minimum).
func (a *metalResident) inStepRange(promptLen, floor int) bool {
	return !a.r.promptStepOff && a.stepPrefill() && floor > 0 && promptLen < max(floor, metalStepPrefillCeiling)
}
