//go:build darwin

package metal

import (
	"fmt"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidentKVCopier = (*metalResident)(nil)

// CopyKV (decoder.ResidentKVCopier) copies rows [start, end) of every layer's K and V, and the int8 scales, from slot
// src to slot dst. A slot's cache is position-major ([pos][kvDim]) and every layer holds the whole context, so each
// buffer's rows are one byte range. On unified memory the copy is a memcpy of mapped pages (gpu.CopyDeviceBatch), so
// nothing may be in flight on the GPU: the executor and any open chain are stopped first, as useKVSlot does.
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
	case r.dnet != nil:
		return &decoder.KVCopyDecline{Reason: "recurrent state is not part of a KV slot"}
	}
	if start == end {
		return nil
	}
	elem := 2
	if r.kvI8 {
		elem = 1
	} else if r.kvF32 {
		elem = 4
	}
	var copies []gpu.DeviceCopy
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || L.geom == nil || r.kvSlotBytes[l] == 0 {
			continue // a shared layer aliases its source's cache, copied with it; a layer with no attention has none
		}
		row := L.geom.kvDim * elem
		copies = append(copies,
			gpu.DeviceCopy{Dst: r.kvRows(dst, l, false, start*row), Src: r.kvRows(src, l, false, start*row), Bytes: (end - start) * row},
			gpu.DeviceCopy{Dst: r.kvRows(dst, l, true, start*row), Src: r.kvRows(src, l, true, start*row), Bytes: (end - start) * row})
		if r.kvI8 {
			sb := L.geom.nKV * 4 // one float32 scale per KV head per position
			copies = append(copies,
				gpu.DeviceCopy{Dst: r.kvSlotBufs[dst].ks[l].At(start * sb), Src: r.kvSlotBufs[src].ks[l].At(start * sb), Bytes: (end - start) * sb},
				gpu.DeviceCopy{Dst: r.kvSlotBufs[dst].vs[l].At(start * sb), Src: r.kvSlotBufs[src].vs[l].At(start * sb), Bytes: (end - start) * sb})
		}
	}
	r.stopExec()
	return gpu.CopyDeviceBatch(copies)
}

// kvRows is slot s's K (v false) or V cache of layer l at byte off within the slot. Buffer.At sets an absolute offset,
// so a contiguous layer is addressed from its allocation (slot 0's view, at offset 0).
func (r *resident) kvRows(s, l int, v bool, off int) Buffer {
	pick := func(b kvSlotBuf) Buffer {
		if v {
			return b.vc[l]
		}
		return b.kc[l]
	}
	if r.kvContig {
		return pick(r.kvSlotBufs[0]).At(s*r.kvSlotBytes[l] + off)
	}
	return pick(r.kvSlotBufs[s]).At(off)
}
