//go:build darwin

package metal

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidentKVCopier = (*metalResident)(nil)

// CopyKV (decoder.ResidentKVCopier) copies rows [start, end) of every layer's K and V, and the int8 scales, from slot
// src to slot dst, block by block: block b of src's table onto block b of dst's (kvBlockRuns), whatever blocks the tables name.
// On unified memory the copy is a memcpy of mapped pages (gpu.CopyDeviceBatch), so nothing may be in flight on the GPU: the
// executor and any open chain are stopped first, as useKVSlot does.
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
	runs := r.kvBlockRuns(src, dst, start/decoder.KVBlockRows, end/decoder.KVBlockRows)
	var copies []gpu.DeviceCopy
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || L.geom == nil || r.kvSlotBytes[l] == 0 {
			continue // a shared layer aliases its source's cache, copied with it; a layer with no attention has none
		}
		// Buffer.At sets an absolute offset, so every block is addressed from its pool's start, where the null block is.
		pools := []Buffer{r.kvSlotBufs[0].kc[l], r.kvSlotBufs[0].vc[l]}
		block := []int{r.kvNullBytes[l], r.kvNullBytes[l]}
		if r.kvI8 { // one float32 scale per KV head per position, block for block
			pools = append(pools, r.kvSlotBufs[0].ks[l], r.kvSlotBufs[0].vs[l])
			block = append(block, decoder.KVBlockRows*L.geom.nKV*4, decoder.KVBlockRows*L.geom.nKV*4)
		}
		for i, b := range pools {
			for _, run := range runs {
				copies = append(copies, gpu.DeviceCopy{Dst: b.At(run.dst * block[i]), Src: b.At(run.src * block[i]), Bytes: run.n * block[i]})
			}
		}
	}
	r.stopExec()
	return gpu.CopyDeviceBatch(copies)
}

// kvBlockRun is n blocks consecutive in both pools: src's from block src, dst's from block dst.
type kvBlockRun struct{ src, dst, n int }

// kvBlockRuns pairs logical blocks [from, to) of slot src's table with the same blocks of slot dst's, merging neighbours that are
// consecutive in both, so in-order tables give one run.
func (r *resident) kvBlockRuns(src, dst, from, to int) []kvBlockRun {
	tab := r.kvTab.U32s()
	var runs []kvBlockRun
	for b := from; b < to; b++ {
		s, d := int(tab[src*r.kvTabBlocks+b]), int(tab[dst*r.kvTabBlocks+b])
		if k := len(runs) - 1; k >= 0 && runs[k].src+runs[k].n == s && runs[k].dst+runs[k].n == d {
			runs[k].n++
			continue
		}
		runs = append(runs, kvBlockRun{s, d, 1})
	}
	return runs
}
