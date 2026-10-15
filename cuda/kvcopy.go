//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/goinfer/decoder"
)

var (
	_ decoder.ResidentKVCopier       = (*cudaResident)(nil)
	_ decoder.ResidentReusableKVSlot = (*cudaResident)(nil)
)

// kvCopyF16 is whether CopyKV serves a load whose KV cache is f16. False declines every such copy, so an f16 load shares
// nothing across conversations and an f32 load is unaffected: the one switch for holding cross-conversation sharing to
// f32 (docs/measurements/kv2-cuda-2026-10-15.md, G0-C).
var kvCopyF16 = true

// CopyKV implements decoder.ResidentKVCopier: rows [start, end) of every layer's KV, from slot src to slot dst, so dst
// continues at position end. It is one executor job, so it is ordered on the one stream after everything posted before
// it (a busy source's prefill included) and runs between MC3 steps, never inside one. A busy source's decode writes only
// rows past its prompt, which the decoder never asks for. The job synchronises the stream before it returns, so the
// caller's timing covers the copy.
//
//   - A full-attention or MLA latent layer copies rows [start, end) of each buffer it has.
//   - A layer that reads another layer's KV copies nothing: it aliases its source.
//   - A windowed layer copies what dst attends to from end, src's rows [end-(window-1), end), through src's shifted view,
//     to dst's physical slot 0, and dst's base becomes that first row. Rows below start that dst already held are rewritten
//     with src's bytes of the same published blocks.
//
// A *decoder.KVCopyDecline, before anything is written, refuses a range the copy cannot serve; its check of src's base runs
// inside the job, so a compaction of a running source cannot fall between the check and the read.
func (r *cudaResident) CopyKV(src, dst, start, end int) error {
	n := r.KVSlots()
	switch {
	case src < 0 || src >= n || dst < 0 || dst >= n || src == dst:
		return fmt.Errorf("cuda: CopyKV slots %d -> %d of %d", src, dst, n)
	case start < 0 || start > end || start%decoder.KVBlockRows != 0 || end%decoder.KVBlockRows != 0:
		return fmt.Errorf("cuda: CopyKV rows [%d, %d) are not whole %d-row blocks", start, end, decoder.KVBlockRows)
	case end > r.ctxCap:
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("rows up to %d are past the %d-position cache", end, r.ctxCap)}
	case r.dnet != nil:
		return &decoder.KVCopyDecline{Reason: "recurrent state is not part of a KV slot"}
	case r.kvF16 && !kvCopyF16:
		return &decoder.KVCopyDecline{Reason: "copies of an f16 KV cache are off (kvCopyF16)"}
	}
	if start == end {
		return nil
	}
	return r.do(func() error { return r.copyKVSlots(src, dst, start, end) })
}

// copyKVSlots is CopyKV's job. Executor thread only.
func (r *cudaResident) copyKVSlots(src, dst, start, end int) error {
	lo := 0 // the first row a windowed layer of dst holds afterwards
	if r.kvWin {
		lo = max(0, end-(r.kvWindow-1))
		if !kvBaseHolds(r.kvBases[src], end, r.kvWindow) {
			return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d's windowed layers hold rows from %d; continuing at %d needs them from %d", src, r.kvBases[src], end, lo)}
		}
		for l := range r.layers {
			if Ly := &r.layers[l]; Ly.kvWin && Ly.kvHalf {
				return &decoder.KVCopyDecline{Reason: fmt.Sprintf("layer %d is windowed and f16, which the shifted view does not address", l)}
			}
		}
	}
	from, to := r.kvSlotBufs[src], r.kvSlotBufs[dst]
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.kvShared {
			continue
		}
		elem := 4
		if Ly.kvHalf {
			elem = 2
		}
		row := Ly.kvDim * elem // bytes
		srcOff, dstOff, rows := start*row, start*row, end-start
		if Ly.kvWin {
			srcOff, dstOff, rows = (lo-r.kvBases[src])*row, 0, end-lo
		}
		for _, p := range [2][2]Buffer{{from.kc[l], to.kc[l]}, {from.vc[l], to.vc[l]}} {
			if p[0].Len() == 0 {
				continue // an MLA latent layer has no V
			}
			if err := r.copyF32(p[0].At(srcOff), p[1].At(dstOff), rows*row/4); err != nil {
				return err
			}
		}
	}
	if r.kvWin {
		r.kvBases[dst] = lo
		if dst == r.kvSlot {
			r.kvBase = lo
		}
	}
	return r.stream.Sync()
}

// ReusableKVSlot implements decoder.ResidentReusableKVSlot: ReusableKV's answer for any slot, bound or not.
func (r *cudaResident) ReusableKVSlot(slot, prefix int) int {
	if !r.kvWin || prefix <= 0 {
		return prefix
	}
	if slot < 0 || slot >= r.KVSlots() {
		return 0
	}
	out := prefix
	// the bases are executor state: read them where they are written
	if err := r.do(func() error {
		if !kvBaseHolds(r.kvBases[slot], prefix, r.kvWindow) {
			out = 0
		}
		return nil
	}); err != nil {
		return 0
	}
	return out
}

// kvBaseHolds reports whether a windowed layer whose physical slot 0 holds position base still holds the history a forward
// continuing at position prefix attends to.
func kvBaseHolds(base, prefix, window int) bool { return base == 0 || prefix-(window-1) >= base }
