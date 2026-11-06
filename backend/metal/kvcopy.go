//go:build darwin

package metal

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidentKVCopier = (*metalResident)(nil)

// CopyKV (decoder.ResidentKVCopier) makes rows [start, end) of slot dst those of slot src, by reference (KV4a-2's rule): each of dst's
// table entries in the range names src's block for it, counted in kvRefs, and the block dst held there is released. No row moves; a
// slot that later writes into a block another slot also names first takes its own (kvWrite). It declines a range src holds no block for (a slot
// maps none past its sequence), a src whose blocks went to make room, and a start past dst's sequence. Tables are read when a command buffer runs, so nothing may be in flight: the executor and any open chain are stopped first, as useKVSlot does.
func (a *metalResident) CopyKV(src, dst, start, end int) error {
	r := a.r
	n := r.kvSlotCount()
	switch {
	case src < 0 || src >= n || dst < 0 || dst >= n || src == dst:
		return fmt.Errorf("metal: CopyKV slots %d -> %d of %d", src, dst, n)
	case start < 0 || start > end || start%64 != 0 || end%64 != 0:
		return fmt.Errorf("metal: CopyKV rows [%d, %d) are not whole 64-row blocks", start, end)
	case end > r.ctxCap:
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("rows up to %d are past the %d-position cache", end, r.ctxCap)}
	case len(r.kvSlotBufs) < 2:
		return &decoder.KVCopyDecline{Reason: "one KV slot"}
	case r.recurrentModel():
		return &decoder.KVCopyDecline{Reason: "recurrent state is not part of a KV slot"}
	case r.mla != nil && !mlaKVCopyDeclineOffForTest:
		return &decoder.KVCopyDecline{Reason: "an MLA layer's latent rows are not copied (this copy moves K and V rows)"}
	}
	if start == end {
		return nil
	}
	if st := &r.kvSlotSt[src]; st.released || st.stopped {
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d's blocks were released or reclaimed to make room", src)}
	}
	if st := &r.kvSlotSt[dst]; start > 0 && (st.released || st.stopped || st.valid < start) {
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d holds no rows up to %d, where the copy starts", dst, start)}
	}
	for b := start / decoder.KVBlockRows; b < end/decoder.KVBlockRows; b++ {
		if *r.kvRow(src, b) == 0 {
			return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d holds no block for rows from %d", src, b*decoder.KVBlockRows)}
		}
	}
	r.stopExec()
	for b := start / decoder.KVBlockRows; b < end/decoder.KVBlockRows; b++ {
		s, d := *r.kvRow(src, b), r.kvRow(dst, b)
		if s == *d {
			continue
		}
		r.kvRefs[s]++
		r.kvUnref(*d)
		*d = s
	}
	// dst now holds [start, end) as src does and below start what it held: its sequence ends where src's does, within the range, and the
	// blocks of a longer one go back.
	st := &r.kvSlotSt[dst]
	if start == 0 {
		st.released, st.stopped = false, false
	}
	st.valid = min(end, max(r.kvSlotSt[src].valid, start))
	r.kvTrim(dst, st.valid)
	r.kvUseTick++
	st.use = r.kvUseTick
	return nil
}

// kvOwnOffForTest leaves a shared entry shared when its slot writes, so the write lands in a block another slot reads: the plant
// TestKVPages_shareByReference must catch.
var kvOwnOffForTest bool

// kvKeepShortForTest makes a copy on write keep one row too few: the plant TestKVPages_shareByReferenceLayerMajor must catch.
var kvKeepShortForTest bool

// kvBlockCopies is the copies of block from's first rows rows into block to, in every layer that owns K/V, its int8 scales with it.
// Buffer.At sets an absolute offset, so each block is addressed from its pool's start, where the null block is.
func (r *resident) kvBlockCopies(from, to uint32, rows int) []gpu.DeviceCopy {
	var copies []gpu.DeviceCopy
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || r.kvSlotBytes[l] == 0 {
			continue // a shared layer reads its source's pool; a layer with no attention has none
		}
		block, row := r.kvNullBytes[l], r.kvNullBytes[l]/decoder.KVBlockRows
		pools, rowBytes, blockBytes := []Buffer{r.kc[l], r.vc[l]}, []int{row, row}, []int{block, block}
		if r.kvI8 && L.geom != nil { // one float32 scale per KV head per position, block for block (an MLA latent has none)
			sb := L.geom.nKV * 4
			pools, rowBytes, blockBytes = append(pools, r.ks[l], r.vs[l]), append(rowBytes, sb, sb), append(blockBytes, decoder.KVBlockRows*sb, decoder.KVBlockRows*sb)
		}
		for i, b := range pools {
			copies = append(copies, gpu.DeviceCopy{Dst: b.At(int(to) * blockBytes[i]), Src: b.At(int(from) * blockBytes[i]), Bytes: rows * rowBytes[i]})
		}
	}
	return copies
}
