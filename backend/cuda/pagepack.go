//go:build cuda

package cuda

import "sort"

// pagePacker models how the driver places the device buffers under one allocation quantum: each lands in the partly used page
// with the least room that still holds it, and opens a new page when none does. Buffers of a quantum or more take their own
// rounded span (allocRoundSlack) and leave no tail a later buffer can use. A page returns to the driver only when every buffer in
// it is freed, so a build that frees none (the weight upload) is modeled exactly; docs/measurements/kv3-s1-2026-11/packing-model.md
// holds the evidence and the models this replaced.
type pagePacker struct {
	room  []int64 // free bytes of each partly used page, ascending
	small int64   // bytes requested by the sub-quantum buffers placed
	pages int64
}

// add places one buffer of n bytes. Buffers of a quantum or more are the caller's rounding (allocRoundSlack), not a page's.
func (p *pagePacker) add(n int64) {
	if n <= 0 || n >= allocQuantumBytes {
		return
	}
	p.small += n
	i := sort.Search(len(p.room), func(i int) bool { return p.room[i] >= n })
	var left int64
	if i < len(p.room) {
		left = p.room[i] - n
		p.room = append(p.room[:i], p.room[i+1:]...)
	} else {
		p.pages++
		left = allocQuantumBytes - n
	}
	if left == 0 {
		return
	}
	j := sort.Search(len(p.room), func(i int) bool { return p.room[i] >= left })
	p.room = append(p.room, 0)
	copy(p.room[j+1:], p.room[j:])
	p.room[j] = left
}

// waste is what the pages hold beyond the sub-quantum bytes requested of them.
func (p *pagePacker) waste() int64 { return p.pages*allocQuantumBytes - p.small }
