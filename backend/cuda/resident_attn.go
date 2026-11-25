//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
)

// splitkvNever disables the split-KV decode attention for a geometry (no depth within the resident's
// capacity pays for it). Larger than any reachable nWin, so the gate comparison stays a plain >=.
const splitkvNever = 1 << 30

// singleBlockAttnShmemLimit is the dynamic shared memory a single-block attention launch may request. The
// glue/batched attention kernels size their scratch (nWin+128)*4 with no ceiling. 48 KB is the device's default
// per-block limit (the 64 KB opt-in needs a per-kernel cuFuncSetAttribute, which nothing here calls), so past
// 12,160 attended keys the launch is refused: decode fails outright at that position (split-KV takes over where it
// exists), and a batched prefill pass on the exact attention declines (checkPrefillShmem) and the prompt falls back to the
// much slower sequential path. A pass the fused attention serves has no such bound. If per-kernel SetAttribute support
// lands, this becomes a device query (docs/code-notes/cuda.md#singleBlockAttnShmemLimit).
const singleBlockAttnShmemLimit = 48 * 1024

// attnShmemBytes is the dynamic shared memory a single-block attention launch needs for an
// attended span of nWin keys — the sizing every such launch site uses.
func attnShmemBytes(nWin int) int { return (nWin + 128) * 4 }

// splitKVRequired reports whether the single-block attention kernel CANNOT run at this span, so split-KV is the
// only option rather than the faster one. splitkvThreshold decides when split-KV is preferred; this decides when
// it is mandatory.
func splitKVRequired(nWin int) bool { return attnShmemBytes(nWin) > singleBlockAttnShmemLimit }

// splitkvThreshold returns the EFFECTIVE attended-key count (nWin: window-clamped, not the raw position) at or above
// which split-KV decode attention beats the single-block attn_batched(M=1) for this geometry, or splitkvNever to
// disable it.
//
// It is a lookup, not a formula. Split-KV buys SM occupancy (attn_batched launches only nH blocks) and pays for it
// in DRAM: it materializes an nH×nWin f32 score array in global memory and touches it three times. Whether that
// wins depends on the occupancy deficit against the score traffic, and no one-parameter law reproduced the four
// measured geometries (qwen2.5-0.5b, qwen2.5-1.5b, windowed gemma3-1b, phi3-mini's "never").
//
// Do not loosen it from the obvious end. Every threshold is rounded UP, and an unmeasured geometry gets
// splitkvConservative rather than an extrapolation: firing early costs up to 18-25%, firing late a few percent.
// The table is NOT device-portable: the occupancy term scales with SM count and every cell was measured on one
// 40-SM Turing part, so re-measure per device class and do not scale by SM count on paper. Re-measure with an
// end-to-end serving decode: TestSplitKVCrossover's tight in-process loop with best-of-min flatters split-KV and
// must not set a threshold. Measured table and method: docs/benchmarks.md §B6 and
// docs/code-notes/cuda.md#splitkvThreshold.
func splitkvThreshold(nH, nKV, hd int) int {
	switch {
	case nKV*hd >= splitkvNeverKVFloats:
		// The never class is keyed on KV traffic per key, not on query-head count; see splitkvNeverKVFloats.
		return splitkvNever
	case nH == 12 && hd == 128:
		return 1024 // qwen2.5-1.5b class: measured crossover in (512, 1024]
	default:
		return splitkvConservative
	}
}

// splitkvNeverKVFloats: at or above this many KV floats per key (nKV*hd, one of K or V) the single-block kernel is
// close enough to the DRAM roof that extra blocks cannot help, and split-KV is pure cost.
//
// Do not key this on query-head count: models with the same nH measured OPPOSITE signs (phi3-mini, MHA, loses;
// mistral-7b, GQA, wins), and "already fills the device" was measured false. The sign follows how close the
// kernel already runs to the DRAM roof, which nKV*hd sets. 3072 is the lowest MEASURED loss (phi3-mini), not a
// midpoint: everything at or above its traffic is excluded, everything below keeps whatever depth threshold it
// already had, and the unmeasured gap below lands in the conservative default. Evidence:
// docs/measurements/splitkv-aa-floor-2026-09-12.md, splitkv-d7-fthreshold-2026-09-13.md; the refuted rule
// (splitkvMaxHeads = 24) and the traffic table: docs/code-notes/cuda.md#splitkvNeverKVFloats.
const splitkvNeverKVFloats = 3072

// splitkvConservative is qwen2.5-0.5b's measured first-clear-win depth, and the default for any geometry not in
// the table. Most measured geometries do not cross over until past 2048, so an unmeasured one is assumed not to
// either: that forfeits a few percent at depth rather than risk a 18-25% shallow regression.
const splitkvConservative = 3072

// splitkvMin is splitkvThreshold with the runtime override applied: GOINFER_SPLITKV_MIN_KEYS=<n> re-gates a stock
// binary (0 means always take the split path, the force-on A/B arm) and GOINFER_SPLITKV_ATTN=0 disables it
// entirely. nKV is taken PER LAYER, not from layers[0]: gemma4's layers differ in KV width, and a global nKV
// would misclassify its narrow layers against its wide ones.
func (r *cudaResident) splitkvMin(nKV, hd int) int {
	if r.skMinKeys >= 0 {
		return r.skMinKeys
	}
	return splitkvThreshold(r.nH, nKV, hd)
}

// splitKVAttnDecode runs the high-occupancy, bit-identical decode attention for layer l at position pos (M=1):
// three launches replacing the single attn_batched(M=1). scores tile over keys (nH·⌈nWin/128⌉ blocks), softmax
// keeps the exact 128-wide partition+tree (byte-identical max + denominator), vsum tiles over output dims
// (nH·⌈hd/32⌉ blocks, each thread the whole per-d fold). Writes r.cctx exactly as attn_batched would.
func (r *cudaResident) splitKVAttnDecode(l, pos int) error {
	r.skLaunches++
	Ly := &r.layers[l]
	nKeys := pos + 1
	winStart := 0
	if Ly.window > 0 && nKeys > int(Ly.window) {
		winStart = nKeys - int(Ly.window)
	}
	nWin := nKeys - winStart
	const dTile = 32 // 1 warp/block; keeps the coalesced V-read, maximizes blocks without sub-warp waste
	// 1. scores → r.skScoreBuf[h*nWin + i] (raw, ·scale). One thread per key; no reduction. SharedMemBytes: hd floats
	// for the staged q: splitkv_scores stages the block-invariant q vector once instead of every thread re-loading it
	// from global (docs/measurements/splitkv-stall-profile-2026-09-13.md).
	if e := r.launch(r.skScores, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((nWin + 127) / 128), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(Ly.hd * 4)},
		r.kvScaled(l, r.withCap(Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), Arg(r.skScoreBuf), gpu.ArgValue(int32(nWin)))...)...); e != nil {
		return e
	}
	// 2. softmax in place (block 128 — MUST match attn_batched for byte-identical max/denominator).
	if e := r.launch(r.skSoftmax, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		// gpt-oss is the only family with an attention sink, and it is resident-eligible, so r.sinkArg(l) below returns
		// the real per-layer sink for it and ArgNull() for every other family: not an always-null argument.
		Arg(r.skScoreBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nWin)), Arg(r.skInvBuf), r.sinkArg(l)); e != nil {
		return e
	}
	// 3. V-sum → r.cctx (each thread the whole ascending-s fold for one output dim). The branch below is the opt-in
	// SPIKE path: it splits the V fold over S key chunks and combines in fixed order, so it is NOT bit-identical to
	// attn_batched and is unreachable unless a test sets vsumSplitForTest.
	if r.skVsumSplit >= 1 && r.skVsumPartial != (Pipeline{}) && r.skVsumCombine != (Pipeline{}) {
		nSplit := r.skVsumSplit
		dy := uint32((Ly.hd + dTile - 1) / dTile)
		if e := r.launch(r.skVsumPartial, LaunchConfig{GridX: uint32(r.nH), GridY: dy, GridZ: uint32(nSplit), BlockX: dTile, BlockY: 1, BlockZ: 1},
			r.kvScaled(l, Arg(r.skScoreBuf), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
				gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(int32(nWin)),
				gpu.ArgValue(int32(nSplit)), Arg(r.skPartialBuf))...); e != nil {
			return e
		}
		return r.launch(r.skVsumCombine, LaunchConfig{GridX: uint32(r.nH), GridY: dy, GridZ: 1, BlockX: dTile, BlockY: 1, BlockZ: 1},
			Arg(r.skPartialBuf), Arg(r.skInvBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(nSplit)), Arg(r.cctx))
	}
	return r.launch(r.skVsum, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((Ly.hd + dTile - 1) / dTile), GridZ: 1, BlockX: dTile, BlockY: 1, BlockZ: 1},
		r.kvScaled(l, Arg(r.skScoreBuf), Arg(r.kvV(l)), Arg(r.kvT(l)), Arg(r.skInvBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(int32(nWin)), Arg(r.cctx))...)
}

// decodeAttnGap is one layer's dynamic gap of a single-token decode: rope_kv at (pos, ropePos) into the BOUND KV
// slot, then the attention launch decode chooses for this position (the flash-decode lane, split-KV, the coalesced
// attn_batched(M=1), or the glue kernel), reading r.qB/r.kB/r.vB and writing r.cctx. The batched decode step (batchstep.go)
// calls it per sequence inside a step, with those four buffers pointed at the sequence's row and its slot bound,
// so every row gets exactly its own decode's kernels.
func (r *cudaResident) decodeAttnGap(Ly *cudaLayer, l, pos, ropePos int, qTempScale float32) error {
	if err := r.decodeRopeStore(Ly, l, pos, ropePos, qTempScale); err != nil {
		return err
	}
	return r.decodeAttn(Ly, l, pos)
}

// attnArm is one of decode's attention launches (decodeAttnArm).
type attnArm uint8

const (
	armFlash   attnArm = iota // the flash-decode lane
	armSplitKV                // split-KV
	armBatched                // attn_batched at M=1
	armGlue                   // the glue attention
	armRefused                // none: the span needs split-KV, which is off or not loaded (decodeAttn's error)
)

// decodeAttnArm is the attention launch decode makes at layer l over nWin attended keys: the one choice decodeAttn and a ragged
// step's rows (stepAttnRows) share. A span past the single-block shared-memory limit with split-KV off or not loaded is refused
// before any lane is considered; then the flash-decode lane, because it replaces the exact path outright.
func (r *cudaResident) decodeAttnArm(Ly *cudaLayer, l, nWin int) attnArm {
	switch {
	case splitKVRequired(nWin) && (!r.splitkvAttn || r.skScores == (Pipeline{})):
		return armRefused
	case r.faSplit > 0 && r.faExactScope.Load() == 0 && nWin >= r.faMinKeys && r.faEligible(l):
		return armFlash
	case r.splitkvAttn && r.skScores != (Pipeline{}) && (splitKVRequired(nWin) || nWin >= r.splitkvMin(Ly.nKV, Ly.hd)):
		return armSplitKV
	case r.prefillReady:
		return armBatched
	}
	return armGlue
}

// decodeNWin is the keys decode attends at pos on layer Ly: pos+1, or the layer's sliding window.
func decodeNWin(Ly *cudaLayer, pos int) int {
	if nKeys := pos + 1; Ly.window <= 0 || nKeys <= int(Ly.window) {
		return nKeys
	}
	return int(Ly.window)
}

// decodeRopeStore is decodeAttnGap's first half: rope_kv at (pos, ropePos) into the bound KV slot and, for an int8 cache, its store.
func (r *cudaResident) decodeRopeStore(Ly *cudaLayer, l, pos, ropePos int, qTempScale float32) error {
	// fused rope(q)+rope(k)+kv_store(k)+kv_store(v): rhalf == hd/2 for full rotary, rotaryDim/2 for partial.
	// A KV-shared layer (Gemma 4 E-model) rotates Q only: nKV=0 leaves rope_kv's K and V work empty, so nothing is stored
	// through the alias into its source's cache. The attention launches below keep the real Ly.nKV, which is how they read it.
	ropeNKV := Ly.nKV
	if Ly.kvShared && !g4KeepSharedKVStoreForTest {
		ropeNKV = 0
	}
	if err := r.launch(r.ropeKV, g1cfg(r.nH*Ly.rhalf+ropeNKV*Ly.rhalf+ropeNKV*(Ly.hd-2*Ly.rhalf), 256),
		Arg(r.qB), Arg(r.kB), Arg(r.vB), Arg(Ly.invF), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(ropeNKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(pos)), gpu.ArgValue(int32(ropePos)), gpu.ArgValue(int32(Ly.rhalf)),
		gpu.ArgValue(Ly.mscale), gpu.ArgValue(qTempScale)); err != nil {
		return err
	}
	return r.kvStore8(l, r.kB, r.vB, ropeNKV, Ly.hd, pos, 1)
}

// decodeAttn is decodeAttnGap's second half: the attention launch decodeAttnArm chooses at pos, reading r.qB and the bound slot's
// cache and writing r.cctx.
func (r *cudaResident) decodeAttn(Ly *cudaLayer, l, pos int) error {
	nKeys := pos + 1
	// Sliding window (per layer: Mistral all-local, Mellum interleaves); shared sized to the attended span.
	nWin := nKeys
	if Ly.window > 0 && nKeys > int(Ly.window) {
		nWin = int(Ly.window)
	}
	// Split-KV is REQUIRED, not merely preferred, once the single-block launch would exceed the device's shared-memory
	// limit. The `r.splitkvAttn` gate and the per-geometry perf threshold both describe when split-KV is FASTER;
	// neither knows when the alternative cannot run at all. Without this, -ctx 16384 on a geometry where split-KV is
	// otherwise off (splitkvNever, or GOINFER_SPLITKV_ATTN=0) fails at position 12,160, or silently drops to the
	// sequential prefill.
	arm := r.decodeAttnArm(Ly, l, nWin)
	if arm == armRefused {
		return fmt.Errorf("cuda: attention at %d attended keys needs %d B of shared memory, "+
			"past this device's %d B limit, and split-KV is unavailable (%s) — lower -ctx or "+
			"re-enable GOINFER_SPLITKV_ATTN", nWin, attnShmemBytes(nWin),
			singleBlockAttnShmemLimit,
			map[bool]string{true: "kernel not loaded", false: "disabled by GOINFER_SPLITKV_ATTN"}[r.skScores == (Pipeline{})])
	}
	switch arm {
	case armFlash:
		// Flash-decode lane (default ON, GOINFER_CUDA_FLASH_DECODE=0 to disable): NOT bit-identical, fidelity-gated.
		// Checked first because it replaces the exact path's three launches outright when eligible.
		if err := r.flashDecodeAttn(l, pos); err != nil {
			return err
		}
	case armSplitKV:
		// Split-KV: high-occupancy and BIT-IDENTICAL to attn_batched(M=1) (TestSplitKV_bitIdentical); fills the SMs the
		// single-block kernel leaves idle at long ctx. Gated PER LAYER on nWin (the EFFECTIVE attended span) against a
		// per-geometry threshold, so shallow decode keeps the cheaper single-block path. nWin, not nKeys: a sliding-window
		// layer never attends more than `window` keys, so its cost is set by the window, not by position (gating on
		// position made gemma3's windowed layers take the split path, their loss regime, at every depth past the window).
		// Both arms are byte-identical, so a layer flipping arms mid-request as nWin grows is safe by construction.
		if err := r.splitKVAttnDecode(l, pos); err != nil {
			return err
		}
	case armBatched:
		// Coalesced M=1 decode attention: attn_batched with M=1 is BIT-IDENTICAL to the glue `attention`
		// (TestAttnBatched_bitIdentical) but reads K via float4, which recovers the L1TEX-latency-bound glue decode
		// attention at long context. startPos=pos, M=1 gives nKeys = pos+1 and the same GridX/block/shared/ctx-layout as
		// the glue launch, so decode stays byte-identical. The glue `attention` (audited) is UNTOUCHED and is the fallback
		// below.
		if err := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
			r.kvScaled(l, r.withCap(Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1)), Arg(r.cctx), r.sinkArg(l))...)...); err != nil {
			return err
		}
	default: // armGlue
		if err := r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
			r.kvScaled(l, r.withCap(Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx))...)...); err != nil {
			return err
		}
	}
	return nil
}
