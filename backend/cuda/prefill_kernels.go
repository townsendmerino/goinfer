//go:build cuda

package cuda

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/internal/runtime/batchplan"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

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
		outs[m] = slices.Clone(r.logitsHost)
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
		batchplan.FillAttnTempRows(rows, startPos, r.attnTempBeta, r.attnTempOrigMaxPos)
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
	if r.actG32 {
		return 0 // choosePassKernels keeps every per-32 pass on the exact kernels
	}
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
		return r.bGemvRows(wt, a.At(c.rows*wt.K), as.At(c.rows*r.actScaleLen(wt.K)*4), bias, dst.At(c.rows*wt.N*4), M-c.rows, accum)
	}
	return r.bGemvRows(wt, a, as, bias, dst, M, accum)
}

// bGemvRows is bGemvB's launch over M rows in the class forceExactKernels and the pass allow.
func (r *cudaResident) bGemvRows(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	if r.actG32 {
		return r.bGemvRowsG32(wt, a, as, bias, dst, M, accum)
	}
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

// bGemvRowsG32 is bGemvRows under per-32 activations: prefill_g32.cu's rows GEMV for the weight's kind, each output bit for bit
// its decode kernel's (doG). a and as hold M activation rows of K/4 ints and actScaleLen(K) floats.
func (r *cudaResident) bGemvRowsG32(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	cfg := LaunchConfig{GridX: uint32(launchgeom.PrefillG32RowsGridX(wt.N)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	nSB, n, m := gpu.ArgValue(int32(wt.K/256)), gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(M))
	switch wt.kind {
	case "q4k":
		return r.launch(r.bQ4KRows, cfg, Arg(wt.W), Arg(a), Arg(as), bias, n, nSB, m, Arg(dst), gpu.ArgValue(accum))
	case "q5k":
		return r.launch(r.bQ5KRows, cfg, Arg(wt.W), Arg(a), Arg(as), bias, n, nSB, m, Arg(dst), gpu.ArgValue(accum))
	case "q6k":
		return r.launch(r.bQ6KRows, cfg, Arg(wt.W), Arg(a), Arg(as), bias, n, nSB, m, Arg(dst), gpu.ArgValue(accum))
	case "int8":
		return r.launch(r.bW8G32Rows, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
			n, gpu.ArgValue(int32(wt.K/4)), m, Arg(dst), gpu.ArgValue(accum))
	}
	return fmt.Errorf("cuda prefill: the per-32 batched GEMV takes q4k, q5k, q6k or int8, got %q", wt.kind)
}
