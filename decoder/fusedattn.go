package decoder

import (
	"github.com/townsendmerino/aikit/linalg"
)

// Fused (FlashAttention-style) prefill attention. The materialized schedule writes a kt x nKeys score block, reads and
// rewrites it in the softmax, and reads it again for scores*V. This blocks over keys instead, keeping the score block
// small enough to stay in cache and folding it into the output accumulator with a running max and a running sum, so the
// kt x nKeys matrix never exists.
//
// It is not bit-identical to the materialized schedule, and that is structural: the running-max rescale re-associates
// the softmax denominator and the AV fold. It is the same category as --cpu-fast-attention and is taken only on that
// non-f64-accumulating path (attendBatchedHeads: !useAcc64 and no tree mask), so it adds no second user-facing flag.
// GOINFER_FUSED_ATTENTION is a developer A/B handle, not a user setting: it keeps fusion's win attributable separately
// from A3's and lets it roll back without losing A3's. It is on by default (an operator decision, because the
// end-to-end gain is far smaller than the kernel gain); GOINFER_FUSED_ATTENTION=0 restores the materialized schedule.
// The measurements, and the model-level divergence figures behind "same order, small increment", are in
// docs/code-notes/decoder.md#fusedattn.

// fusedKeyBlock is the key-block width. 512 keeps the per-tile score block at kt*512 floats (512 KiB at kt=256), small
// enough to stay resident, which is the entire mechanism; 256 measured within noise of it and 1024 is worse.
const fusedKeyBlock = linalg.FusedAttnKeyBlock

// attendTileFused computes one query tile's attention into ch, keeping the score block resident. It returns false when
// it declines, and the caller must then run the materialized path. It declines for tree attention: treeMask is a
// per-(row, column) predicate over the batch columns, and folding it into a key-blocked running softmax is different
// work from the contiguous [lo, hi] bound the causal and sliding-window cases share.
//
//	qh    gathered Q for this tile, [kt, hd]
//	kh    gathered K, [nKeys, hd]
//	vBlk  gathered V in BLOCK-MAJOR layout (see gatherKVFused)
//	ch    output, [kt, hd]
//	lo/hi per-row inclusive key bounds, already mapped to physical columns
func attendTileFused(
	mm func(a, b, dst []float32, M, K, N int),
	qh, kh, vBlk, ch []float32,
	sBlk, tmp, acc, mRun, lRun []float32,
	kt, hd, nKeys int, scale float64,
	lo, hi []int,
) bool {
	return linalg.AttendTileFused(mm, qh, kh, vBlk, ch,
		&linalg.FusedAttnScratch{SBlk: sBlk, Tmp: tmp, Acc: acc, MRun: mRun, LRun: lRun},
		kt, hd, nKeys, scale, lo, hi)
}

// gatherKVFused writes this kv head's V in block-major layout: block b occupies [b*fusedKeyBlock*hd, ...) holding
// [hd, n] contiguously, which is what MatmulBT's b operand needs for the per-block AV fold. The materialized path's
// [hd, nKeys] layout cannot be key-range sliced contiguously, so the layout is chosen at gather time rather than
// re-transposed per block.
func gatherKVFused(kh, vBlk, keys, vals []float32, kvh, hd, kvDim, nKeys int) {
	linalg.GatherVBlockMajor(kh, vBlk, keys, vals, kvh, hd, kvDim, nKeys)
}

// fusedScratch holds the per-worker buffers the fused schedule needs, allocated only when fusion is enabled. sBlk is
// kt*fusedKeyBlock floats where the materialized scores is kt*nKeys, so it is smaller than what it replaces; but both
// exist while fusion is a knob, and prefillAttnWorkers' budget is computed from the materialized shape, which stays the
// binding one.
type fusedScratch struct {
	sBlk, tmp, acc, mRun, lRun, vBlk []float32
	lo, hi                           []int // per-row key bounds for the current tile
}

// fits reports whether this scratch is already large enough for the requested shape, so a pool slot can be reused
// across calls instead of reallocated per token. Every buffer is used as a prefix slice, so larger is fine.
func (f *fusedScratch) fits(kt, hd, nKeys int) bool {
	return f != nil &&
		len(f.sBlk) >= kt*fusedKeyBlock &&
		len(f.tmp) >= kt*hd && len(f.acc) >= kt*hd &&
		len(f.mRun) >= kt && len(f.lRun) >= kt &&
		len(f.vBlk) >= hd*nKeys &&
		len(f.lo) >= kt && len(f.hi) >= kt
}

func newFusedScratch(kt, hd, nKeys int) *fusedScratch {
	return &fusedScratch{
		sBlk: make([]float32, kt*fusedKeyBlock),
		tmp:  make([]float32, kt*hd),
		acc:  make([]float32, kt*hd),
		mRun: make([]float32, kt),
		lRun: make([]float32, kt),
		vBlk: make([]float32, hd*nKeys),
		lo:   make([]int, kt),
		hi:   make([]int, kt),
	}
}

// fusedIfEnabled returns a fresh fusedScratch when the schedule is on (the model's knobs; knobs.go), else nil.
func fusedIfEnabled(k *knobSet, kt, hd, nKeys int) *fusedScratch {
	if !k.fusedAttention() {
		return nil
	}
	return newFusedScratch(kt, hd, nKeys)
}
