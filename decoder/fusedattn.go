package decoder

import (
	"os"

	"github.com/townsendmerino/aikit/linalg"
)

// Fused (FlashAttention-style) prefill attention — P19.
//
// The materialized schedule writes a kt x nKeys score block, reads and rewrites
// it in the softmax, and reads it again for scores*V: three trips through memory
// for a block that is 8 MiB at the production tile budget. This blocks over KEYS
// instead, keeping the score block small enough to stay in cache and folding it
// into the output accumulator with a running max and a running sum, so the
// kt x nKeys matrix never exists.
//
// MEASURED before it was written into the forward
// (docs/measurements/p19-fused-attention-2026-09-01.md). Causal, row-parallel,
// summed over all 32 tiles of an 8192-token prefill: 1.69-1.73x, cosine
// 1.000000000. Four configurations were measured and three of them imply the
// WRONG verdict -- column-parallel says 0.70x (close it), a single last tile says
// 1.2x (park it) -- so the number above is specifically the one production runs.
//
// IT IS NOT BIT-IDENTICAL, and that is structural rather than incidental: the
// running-max rescale re-associates the softmax denominator and the AV fold. This
// is the same category as --cpu-fast-attention, and it rides that flag rather
// than adding a second user-facing one -- but NOT for the reason first written
// here. That claimed the added divergence was "~5 orders of magnitude smaller",
// comparing a KERNEL number (max|diff| 9.3e-9) to a MODEL-LEVEL one. Measured on
// one checkpoint at one depth, both at model level: acc64 vs f32-materialized is
// cosine 0.998283, acc64 vs f32-fused is 0.998262. Same order; what is small is
// fusion's INCREMENT (~2e-5 of cosine), not its magnitude. The conclusion holds,
// the arithmetic behind it did not. GOINFER_FUSED_ATTENTION is a
// developer A/B handle, not a user setting -- it exists so fusion's win stays
// attributable separately from A3's, and so it can be rolled back without losing
// A3's.
// DEFAULT ON since 2026-09-01, by operator decision. The measurement does not
// make the case on its own and that is recorded rather than smoothed: the fused
// schedule is 1.69-1.73x at the KERNEL over a whole prefill's tiles, but only
// 1.080x END-TO-END (dense 1.5B, K=4096, paired) -- because A3's head fan-out
// already took most of what attention had to give, leaving it ~18% of this
// prefill by Amdahl. Eight percent, bought with a user-visible output change.
//
// GOINFER_FUSED_ATTENTION=0 restores the materialized schedule.
func fusedAttention() bool { return os.Getenv("GOINFER_FUSED_ATTENTION") != "0" }

// fusedKeyBlock is the key-block width. 256 and 512 measured within noise of each
// other (1.731x / 1.687x) and both beat 1024; 512 keeps the per-tile score block
// at kt*512 floats (512 KiB at kt=256), which is the point -- small enough to stay
// resident, which is the entire mechanism.
const fusedKeyBlock = linalg.FusedAttnKeyBlock

// attendTileFused computes one query tile's attention into ch, keeping the score
// block resident. Returns false when it declines, in which case the caller must
// run the materialized path instead.
//
// It declines for TREE attention: treeMask is a per-(row, column) predicate over
// the batch columns, and folding it into a key-blocked running softmax is a
// different piece of work than the contiguous [lo, hi] bound the causal and
// sliding-window cases share. Declining keeps this change to the path it was
// measured on rather than guessing at the one it was not.
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

// gatherKVFused writes this kv head's V in BLOCK-MAJOR layout: block b occupies
// [b*fusedKeyBlock*hd, ...) holding [hd, n] contiguously, which is what
// MatmulBT's b operand needs for the per-block AV fold.
//
// The materialized path wants [hd, nKeys] and a key-range slice of THAT is not
// contiguous, which is why the layout is chosen at gather time rather than
// re-transposed per block. Same work, different order — the gather is not made
// more expensive by this, only differently indexed.
func gatherKVFused(kh, vBlk, keys, vals []float32, kvh, hd, kvDim, nKeys int) {
	linalg.GatherVBlockMajor(kh, vBlk, keys, vals, kvh, hd, kvDim, nKeys)
}

// fusedScratch holds the per-worker buffers the fused schedule needs. Allocated
// only when fusion is enabled: its `sBlk` is kt*fusedKeyBlock floats where the
// materialized `scores` is kt*nKeys, so this is SMALLER than what it replaces —
// but both exist while fusion is a flag, and prefillAttnWorkers' budget is
// computed from the materialized shape, which stays the binding one.
type fusedScratch struct {
	sBlk, tmp, acc, mRun, lRun, vBlk []float32
	lo, hi                           []int // per-row key bounds for the current tile
}

// fits reports whether this scratch is already large enough for the requested shape, so a pool slot
// can be REUSED across calls instead of reallocated per token (M-03). Every buffer is used as a
// prefix slice, so larger is fine.
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

// fusedIfEnabled returns a fresh fusedScratch when the schedule is on, else nil.
func fusedIfEnabled(kt, hd, nKeys int) *fusedScratch {
	if !fusedAttention() {
		return nil
	}
	return newFusedScratch(kt, hd, nKeys)
}
