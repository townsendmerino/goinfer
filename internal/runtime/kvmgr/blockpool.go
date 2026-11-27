package kvmgr

import (
	"fmt"
	"slices"
)

// BlocksFor returns the number of BlockRows-position blocks needed to cover positions (0 when positions <= 0).
func BlocksFor(positions int) int {
	if positions <= 0 {
		return 0
	}
	return (positions + BlockRows - 1) / BlockRows
}

// WithinShare reports whether a slot holding held blocks is within its guaranteed share of shareBlocks blocks when
// requesting one block: strictly below shareBlocks for a new allocation, or up to shareBlocks for a copy-on-write fork
// (which drops its shared reference and holds no net additional block).
func WithinShare(held, shareBlocks int, cow bool) bool {
	return held < shareBlocks || (cow && held <= shareBlocks)
}

// BlockScanOrder returns the 1-based physical block scan order for usableBlocks blocks (excluding null block 0):
// ascending 1..usableBlocks when scramble is false, or alternating from the two ends when scramble is true (starting
// from the high end unless reverse is true, so two layer groups assign distinct physical blocks to one position).
func BlockScanOrder(usableBlocks int, scramble, reverse bool) []uint32 {
	out := make([]uint32, 0, max(0, usableBlocks))
	if !scramble {
		for i := 1; i <= usableBlocks; i++ {
			out = append(out, uint32(i))
		}
		return out
	}
	for i, j := 1, usableBlocks; i <= j; i, j = i+1, j-1 {
		first, second := uint32(j), uint32(i)
		if reverse {
			first, second = second, first
		}
		out = append(out, first)
		if i != j {
			out = append(out, second)
		}
	}
	return out
}

// BlockPool is one KV layer-group's reference-counted free-list block allocator. Block 0 is the zero-filled null block
// and is never handed out or freed.
type BlockPool struct {
	blocks int
	free   []int32
	refs   []int32
}

// NewBlockPool returns a pool of blocks total blocks (including null block 0). When order is non-nil and blocks > 1,
// blocks are popped in order; otherwise ascending 1..blocks-1.
func NewBlockPool(blocks int, order []int32) BlockPool {
	p := BlockPool{
		blocks: blocks,
		free:   make([]int32, 0, max(0, blocks-1)),
		refs:   make([]int32, max(0, blocks)),
	}
	if order != nil && blocks > 1 {
		for _, o := range slices.Backward(order) {
			p.free = append(p.free, o)
		}
		return p
	}
	for b := blocks - 1; b >= 1; b-- {
		p.free = append(p.free, int32(b))
	}
	return p
}

// Blocks returns the total number of blocks in the pool, including null block 0.
func (p *BlockPool) Blocks() int { return p.blocks }

// Free returns the number of free usable blocks currently available to Take.
func (p *BlockPool) Free() int { return len(p.free) }

// InUse returns how many usable blocks are currently referenced by at least one slot table (null block 0 excluded).
func (p *BlockPool) InUse() int { return p.blocks - 1 - len(p.free) }

// Refs returns the current reference count of block b.
func (p *BlockPool) Refs(b int32) int32 { return p.refs[b] }

// Take pops one free block from the pool with reference count 1, or returns (0, false) when none is free.
func (p *BlockPool) Take() (int32, bool) {
	if len(p.free) == 0 {
		return 0, false
	}
	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.refs[b] = 1
	return b, true
}

// Ref increments block b's reference count (ignoring null block 0).
func (p *BlockPool) Ref(b int32) {
	if b > 0 {
		p.refs[b]++
	}
}

// Unref decrements block b's reference count, returning b to the free list when the last reference drops. Null block 0
// is ignored.
func (p *BlockPool) Unref(b int32) {
	if b <= 0 {
		return
	}
	if p.refs[b]--; p.refs[b] <= 0 {
		p.refs[b] = 0
		p.free = append(p.free, b)
	}
}

// PickIdleVictim returns the eligible idle slot != skip with the oldest (smallest) last-write tick use(slot), or -1
// when none is eligible. Setting reverseOrder picks the most recently written idle slot instead (for pressure-order
// defect plants).
func PickIdleVictim(slots, skip int, canEvict func(slot int) bool, use func(slot int) uint64, reverseOrder bool) int {
	best := -1
	for t := range slots {
		if t == skip || !canEvict(t) {
			continue
		}
		if best < 0 || (use(t) < use(best)) != reverseOrder {
			best = t
		}
	}
	return best
}

// PickBorrowerVictim returns the eligible slot != skip holding the largest positive excess(slot) blocks past its share,
// together with that excess count, or (-1, 0) when no eligible slot holds excess blocks.
func PickBorrowerVictim(slots, skip int, canReclaim func(slot int) bool, excess func(slot int) int) (victim, most int) {
	victim = -1
	for t := range slots {
		if t == skip || !canReclaim(t) {
			continue
		}
		if ex := excess(t); ex > most {
			victim, most = t, ex
		}
	}
	return victim, most
}

// RowRuns calls fn(pos, count, physRow) for each maximal run of positions in [lo, hi) whose mapped blocks in ids are
// physically consecutive in the pool (ids[b] == ids[b-1]+1). If any position in [lo, hi) is unmapped (ids[b] == 0),
// RowRuns stops and returns that position as unmappedPos >= 0 with err == nil; on success unmappedPos is -1.
func RowRuns(ids []int32, lo, hi int, fn func(pos, count, physRow int) error) (unmappedPos int, err error) {
	for p := lo; p < hi; {
		b := p / BlockRows
		if b < 0 || b >= len(ids) || ids[b] == 0 {
			return p, nil
		}
		row := int(ids[b])*BlockRows + p%BlockRows
		end := min(hi, (b+1)*BlockRows)
		for end < hi && end/BlockRows < len(ids) && ids[end/BlockRows] == ids[end/BlockRows-1]+1 {
			end = min(hi, end+BlockRows)
		}
		if err := fn(p, end-p, row); err != nil {
			return -1, err
		}
		p = end
	}
	return -1, nil
}

// UploadPass tracks one multi-layer KV upload transaction between BeginKVUpload and EndKVUpload on a paged resident KV
// pool, verifying that every owning layer lands once to the same sequence end and covers its sliding window.
type UploadPass struct {
	slot    int
	end     int
	floor   int
	wrote   bool
	reached map[int]int
}

// NewUploadPass starts a pass writing bound slot.
func NewUploadPass(slot int) *UploadPass {
	return &UploadPass{slot: slot, reached: map[int]int{}}
}

// Slot is the KV slot the pass was started on.
func (p *UploadPass) Slot() int { return p.slot }

// End is the sequence end recorded by the layers uploaded so far (0 before any layer lands).
func (p *UploadPass) End() int { return p.end }

// Floor is the shortest reusable prefix floor imposed by wrapped sliding-window layers uploaded in this pass.
func (p *UploadPass) Floor() int { return p.floor }

// Wrote reports whether the pass has mutated its target slot's mapping.
func (p *UploadPass) Wrote() bool { return p.wrote }

// MarkWrote records that the first layer mapped blocks on the target slot with target end.
func (p *UploadPass) MarkWrote(end int) {
	p.end = end
	p.wrote = true
}

// CheckLayer validates an UploadKV call for layer covering rows [base, end) with sliding-window width window (0 for
// full attention). When checkSameEnd is true and an earlier layer has already set End, end must match it immediately.
func (p *UploadPass) CheckLayer(prefix string, boundSlot, layer, base, end, window int, checkSameEnd bool) error {
	if boundSlot != p.slot {
		return fmt.Errorf("%s: upload pass: slot %d is bound, the pass writes slot %d", prefix, boundSlot, p.slot)
	}
	if _, again := p.reached[layer]; again {
		return fmt.Errorf("%s: upload pass: layer %d uploaded twice", prefix, layer)
	}
	if base > 0 && window <= 0 {
		return fmt.Errorf("%s: upload pass: full-attention layer %d starts at %d, not position 0", prefix, layer, base)
	}
	if base > 0 && !WindowReusable(base, end, window) {
		return fmt.Errorf("%s: upload pass: layer %d's rows [%d, %d) do not cover its %d-row window", prefix, layer, base, end, window)
	}
	if checkSameEnd && p.end != 0 && end != p.end {
		return fmt.Errorf("%s: upload pass: layer %d ends at %d, the pass's first layer at %d", prefix, layer, end, p.end)
	}
	return nil
}

// RecordLayer records that layer successfully uploaded rows [base, end) with sliding-window width window.
func (p *UploadPass) RecordLayer(layer, base, end, window int) {
	p.end = max(p.end, end)
	p.reached[layer] = end
	if base > 0 && window > 0 {
		p.floor = max(p.floor, base+window-1)
	}
}

// Finish validates the completed pass across 0..numLayers-1 where ownsKV reports layers that hold their own KV cache.
// When failed is true, Finish returns (0, 0, nil) so the caller drops any partial writes without synthesizing an error.
func (p *UploadPass) Finish(prefix string, failed bool, numLayers int, ownsKV func(layer int) bool) (end, floor int, err error) {
	if failed {
		return 0, 0, nil
	}
	if p.end == 0 {
		return 0, 0, fmt.Errorf("%s: upload pass: no layer uploaded any rows", prefix)
	}
	for l := range numLayers {
		if ownsKV(l) && p.reached[l] != p.end {
			return 0, 0, fmt.Errorf("%s: upload pass: layer %d reached %d of the pass's %d rows", prefix, l, p.reached[l], p.end)
		}
	}
	return p.end, p.floor, nil
}
