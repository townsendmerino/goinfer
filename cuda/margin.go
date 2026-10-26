//go:build cuda

package cuda

// The context margin computed from its parts (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build"): what the build allocates
// after the KV pools, one 512-row prefill pass, the allocation floor and a decode allowance, beside the KV priced as the card
// charges it. slotMarginBytes is not this.

// allocFloorBytes is the free VRAM cuMemAlloc will not hand out at any request size: the top of TestAllocFloor's recorded band
// (148–155 MiB, docs/code-notes/cuda.md, "TestAllocFloor").
const allocFloorBytes = 155 << 20

// decodeAllowanceBytes is decode's transient allocations beyond the prefill pass: H6-R measured none on a dense decode
// (docs/tasks/task-kernel-survey-2026-10.md, "H6-R read").
const decodeAllowanceBytes = 16 << 20

// grantReserveBytes is the reserve the runtime grants (kvGrantRollback, allocRecurrentCheckpoint) keep beside two prefill passes.
// It is the flat figure the context margin used to be; the owner's KS-H6 decision did not change these grants.
const grantReserveBytes = 384 << 20

// chargedBytes is what the device charges for one allocation of n bytes: the request, rounded up to the allocation quantum from a
// quantum upward (allocRoundSlack).
func chargedBytes(n int64) int64 { return n + allocRoundSlack(n) }

// kvChargedBytes is the device bytes kvAllocPools takes for slots slots at capacity cap: every owning layer's K and V pool (one
// latent pool for MLA), each holding its group's blocks for every slot plus the null block and charged as the device rounds it,
// and the slots' block tables. It mirrors kvAllocPools and kvSlotBlocks; TestKSH6C_chargedKVMatchesThePools holds it to them.
func kvChargedBytes(cap, slots int, layers []cudaLayer) int64 {
	full := int64(slots*kvBlocksFor(cap) + 1)
	var n int64
	for i := range layers {
		Ly := &layers[i]
		if Ly.kvShared || Ly.noKV() || Ly.kvDim == 0 {
			continue
		}
		elem := int64(4)
		if Ly.kvHalf {
			elem = 2
		}
		blocks := full
		if Ly.kvWin {
			blocks = int64(slots*kvWindowBlocks(cap, int(Ly.window)+kvWindowSlack) + 1)
		}
		buf := chargedBytes(blocks * kvBlock * int64(Ly.kvDim) * elem)
		if Ly.isMLA() {
			n += buf
			continue
		}
		n += 2 * buf
	}
	return n + chargedBytes(int64(slots*kvGroups*kvBlocksFor(cap))*4)
}

// kvChargedCapForBytes is the largest capacity in [lo, hi] whose slots slots of pools charge at most budget (ok=false when even lo
// does not). kvChargedBytes is non-decreasing in the capacity, so a search finds the edge.
func kvChargedCapForBytes(budget int64, slots, lo, hi int, layers []cudaLayer) (c int, ok bool) {
	if hi < lo || kvChargedBytes(lo, slots, layers) > budget {
		return lo, false
	}
	for hi > lo {
		mid := lo + (hi-lo+1)/2
		if kvChargedBytes(mid, slots, layers) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo, true
}

// kvChargedSlotsFit is the largest n in [1, want] whose n slots of pools charge at most free less reserve; never below 1 (the first
// slot is checkKVFits' own and is refused there).
func kvChargedSlotsFit(want int, free int64, cap int, layers []cudaLayer, reserve int64) int {
	n := 1
	for k := 2; k <= want; k++ {
		if kvChargedBytes(cap, k, layers)+reserve > free {
			break
		}
		n = k
	}
	return n
}

// kvRoundingBound is the most the device's rounding can add to the linear KV price the plan uses (decoder's per-position KV):
// one allocation quantum per pool buffer, and one for the tables. owners and mla count the layers that own a K/V pool pair and
// those that own one latent pool.
func kvRoundingBound(owners, mla int) int64 {
	return int64(2*owners+mla+1) * allocQuantumBytes
}
