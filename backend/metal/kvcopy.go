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
// slot that later writes into a block another slot also names first takes its own (kvOwn). Tables are read when a command buffer runs,
// so nothing may be in flight: the executor and any open chain are stopped first, as useKVSlot does.
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
	r.stopExec()
	tab := r.kvTab.U32s()
	for b := start / decoder.KVBlockRows; b < end/decoder.KVBlockRows; b++ {
		s, d := tab[src*r.kvTabBlocks+b], &tab[dst*r.kvTabBlocks+b]
		if s == *d {
			continue
		}
		r.kvRefs[s]++
		r.kvUnref(*d)
		*d = s
	}
	return nil
}

// kvUnref drops one table entry's reference to block id; the last one frees it.
func (r *resident) kvUnref(id uint32) {
	if r.kvRefs[id]--; r.kvRefs[id] == 0 {
		r.kvFree = append(r.kvFree, int32(id))
	}
}

// kvOwnOffForTest makes kvOwn a no-op, so a slot writes into a block it shares: the plant TestKVPages_shareByReference must catch.
var kvOwnOffForTest bool

// kvBlocksInUse is how many blocks some table names, the null block not counted: the pool's own count of what its slots hold, a
// block several slots share counted once.
func (r *resident) kvBlocksInUse() int { return len(r.kvRefs) - 1 - len(r.kvFree) }

// kvOwn gives slot s a block of its own for every block overlapping positions [lo, hi) that another table also names, which the caller
// is about to write: a block from kvFree holding a copy of the shared one's rows below lo (copy on write); rows at or past lo are the
// write's. Nothing may be in flight on the GPU, which every caller ensures: the synchronous entry points and the executor's commit run
// after the previous forward completed, and a chain maps its whole range before it opens. kvFree cannot run dry: a slot's table has
// kvTabBlocks entries and the pools hold every slot's, so a block named twice leaves one free.
func (r *resident) kvOwn(s, lo, hi int) {
	if len(r.kvRefs) == 0 || hi <= lo || s >= max(1, len(r.kvSlotBufs)) || kvOwnOffForTest {
		return
	}
	tab := r.kvTab.U32s()
	for b := lo / decoder.KVBlockRows; b < min(kvBlocksFor(hi), r.kvTabBlocks); b++ {
		e := &tab[s*r.kvTabBlocks+b]
		if r.kvRefs[*e] <= 1 {
			continue
		}
		if len(r.kvFree) == 0 {
			panic("metal: no free KV block for a copy on write: the pool sizing rules this out")
		}
		nb := uint32(r.kvFree[len(r.kvFree)-1])
		r.kvFree = r.kvFree[:len(r.kvFree)-1]
		r.kvRefs[nb] = 1
		if keep := lo - b*decoder.KVBlockRows; keep > 0 {
			if err := gpu.CopyDeviceBatch(r.kvBlockCopies(*e, nb, keep)); err != nil {
				panic(fmt.Sprintf("metal: copy on write of KV block %d: %v", *e, err))
			}
		}
		r.kvUnref(*e)
		*e = nb
	}
}

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
