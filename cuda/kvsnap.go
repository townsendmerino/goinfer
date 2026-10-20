//go:build cuda

package cuda

import (
	"fmt"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

var (
	_ decoder.ResidentKVSnapshotter      = (*cudaResident)(nil)
	_ decoder.ResidentRecurrentSnapshots = (*cudaResident)(nil)
)

// Prefix snapshots (docs/tasks/task-kv7-prefix-snapshot-2026-10.md): a slot's KV rows in pinned host memory, and back. It is CopyKV with
// a host buffer for the other slot, over the same slot storage:
//
//   - A full-attention or MLA latent layer holds rows [0, end) of each buffer it has.
//   - A layer that reads another layer's KV holds nothing: it aliases its source.
//   - A windowed layer holds the rows its window and the rollback need behind end: [lo, end) with
//     lo = max(base, end-(window-1)-rollback), through the slot's shifted view, as CopyKV copies them. A restore to position q <= end
//     writes [max(lo, q-(window-1)-rollback), q) to physical slot 0 and sets the slot's ring base to that first row.
//
//   - A Gated-DeltaNet model's snapshot also holds every DeltaNet layer's conv ring and recurrent state as they stand after row end-1, so it
//     restores to exactly end (docs/queue-engineering.md, B35). It is taken between two passes of a prefill (ResidentPrefillMarked), when
//     the state belongs to that position. A Mamba-2 model's state is not held.
//
// The host buffer is one pinned allocation per snapshot, made when the snapshot is taken and freed when it is dropped: nothing is
// held while no snapshot is. Every device copy is an executor job that synchronises the stream, so a snapshot read after the prefill that
// wrote the rows and before the decode that would move the ring, and a restore is complete before the prefill that continues from it.

// cudaKVSnapshot is one prefix's KV in pinned host memory.
type cudaKVSnapshot struct {
	r      *cudaResident
	rows   int // positions [0, rows) are held; for a windowed layer only [lo, rows)
	lo     int // the first row a windowed layer holds, 0 with none
	host   *gpu.HostBuffer[float32]
	layers []snapLayer
	dn     []snapDelta // a DeltaNet model's per-layer state, empty otherwise
	bytes  int64
	// the geometry the rows were laid out for: a restore into another one is refused
	ctxCap, window, slack int
	win, half             bool
	once                  sync.Once
}

// snapLayer is where one layer's K and V rows sit in the host buffer, in float32 words (a layer that holds f16 is copied as raw words).
type snapLayer struct {
	held       bool // false for a layer that aliases another's KV
	win        bool
	kOff, vOff int
	hasV       bool
	rowWords   int // words a position takes
}

// snapDelta is where one DeltaNet layer's conv ring and recurrent state sit in the host buffer, in float32 words.
type snapDelta struct {
	layer                        int
	winOff, winWords, stOff, stW int
}

func (s *cudaKVSnapshot) Rows() int    { return s.rows }
func (s *cudaKVSnapshot) Bytes() int64 { return s.bytes }

// Serves implements decoder.ResidentKVSnapshot.
func (s *cudaKVSnapshot) Serves(end int) bool {
	r := s.r
	if end <= 0 || end > s.rows || end%decoder.KVBlockRows != 0 || s.host == nil {
		return false
	}
	if len(s.dn) > 0 && end != s.rows {
		return false // the state is the one after the last row
	}
	if s.ctxCap != r.ctxCap || s.win != r.kvWin || s.window != r.kvWindow || s.slack != r.kvSlack {
		return false
	}
	return !s.win || end-(s.window-1) >= s.lo
}

// Release implements decoder.ResidentKVSnapshot.
func (s *cudaKVSnapshot) Release() {
	s.once.Do(func() {
		if s.host != nil {
			_ = s.host.Close() // idempotent, and safe once the device has released it
		}
	})
}

// slotKV is slot's K and V buffers: the build's own with one slot.
func (r *cudaResident) slotKV(slot int) (kc, vc []Buffer) {
	if len(r.kvSlotBufs) == 0 {
		return r.kc, r.vc
	}
	return r.kvSlotBufs[slot].kc, r.kvSlotBufs[slot].vc
}

// snapshotDecline is the refusal that needs no slot: the same ones CopyKV makes.
func (r *cudaResident) snapshotDecline(end int) *decoder.KVCopyDecline {
	switch {
	case end > r.ctxCap:
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("rows up to %d are past the %d-position cache", end, r.ctxCap)}
	case r.mamba != nil:
		return &decoder.KVCopyDecline{Reason: "Mamba-2 state is not part of a snapshot"}
	}
	if r.kvWin {
		for l := range r.layers {
			if Ly := &r.layers[l]; Ly.kvWin && Ly.kvHalf {
				return &decoder.KVCopyDecline{Reason: fmt.Sprintf("layer %d is windowed and f16, which the shifted view does not address", l)}
			}
		}
	}
	return nil
}

// KVSnapshotRecurrent implements decoder.ResidentRecurrentSnapshots: a Gated-DeltaNet model's snapshots carry its state; a Mamba-2 model's
// are declined (snapshotDecline).
func (r *cudaResident) KVSnapshotRecurrent() bool { return r.dnet != nil && r.mamba == nil }

// snapshotDeltaLayout places every DeltaNet layer's conv ring and state after words, and returns them and the words in all.
func (r *cudaResident) snapshotDeltaLayout(words int) ([]snapDelta, int) {
	if r.dnet == nil {
		return nil, words
	}
	var out []snapDelta
	for l := range r.layers {
		Ly := &r.layers[l]
		if !Ly.isDeltaNet() {
			continue
		}
		d := snapDelta{layer: l, winOff: words, winWords: Ly.dnWin.Len()}
		words += d.winWords
		d.stOff, d.stW = words, Ly.dnState.Len()
		words += d.stW
		out = append(out, d)
	}
	return out, words
}

// snapshotLayout lays the held rows out in the host buffer for a snapshot to end whose windowed layers start at lo, and returns the
// layers and the words in all.
func (r *cudaResident) snapshotLayout(end, lo int) ([]snapLayer, int) {
	_, vc := r.slotKV(r.kvSlot)
	out := make([]snapLayer, len(r.layers))
	words := 0
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.kvShared {
			continue
		}
		elem := 4
		if Ly.kvHalf {
			elem = 2
		}
		sl := snapLayer{held: true, win: Ly.kvWin, rowWords: Ly.kvDim * elem / 4, hasV: vc[l].Len() != 0}
		rows := end
		if Ly.kvWin {
			rows = end - lo
		}
		sl.kOff = words
		words += rows * sl.rowWords
		if sl.hasV {
			sl.vOff = words
			words += rows * sl.rowWords
		}
		out[l] = sl
	}
	return out, words
}

// KVSnapshotBytes implements decoder.ResidentKVSnapshotter. A windowed layer's start is taken at its earliest, so it is an upper bound.
func (r *cudaResident) KVSnapshotBytes(end int) int64 {
	if end <= 0 || end%decoder.KVBlockRows != 0 || r.snapshotDecline(end) != nil {
		return 0
	}
	lo := 0
	if r.kvWin {
		lo = max(0, end-(r.kvWindow-1)-r.kvRollback())
	}
	_, words := r.snapshotLayout(end, lo)
	_, words = r.snapshotDeltaLayout(words)
	return int64(words) * 4
}

// KVSnapshotRollback implements decoder.ResidentKVSnapshotter.
func (r *cudaResident) KVSnapshotRollback() int {
	if !r.kvWin {
		return 0
	}
	return r.kvRollback()
}

// SnapshotKV implements decoder.ResidentKVSnapshotter.
func (r *cudaResident) SnapshotKV(slot, end int) (decoder.ResidentKVSnapshot, error) {
	switch {
	case slot < 0 || slot >= r.KVSlots():
		return nil, fmt.Errorf("cuda: SnapshotKV slot %d of %d", slot, r.KVSlots())
	case end <= 0 || end%decoder.KVBlockRows != 0:
		return nil, fmt.Errorf("cuda: SnapshotKV rows [0, %d) are not whole %d-row blocks", end, decoder.KVBlockRows)
	}
	if d := r.snapshotDecline(end); d != nil {
		return nil, d
	}
	var s *cudaKVSnapshot
	err := r.do(func() error {
		var e error
		s, e = r.snapshotKV(slot, end)
		return e
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}

// snapshotKV is SnapshotKV's job. Executor thread only.
func (r *cudaResident) snapshotKV(slot, end int) (*cudaKVSnapshot, error) {
	lo := 0
	if r.kvWin {
		if !kvWindowReusable(r.kvBases[slot], end, r.kvWindow) {
			return nil, &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d's windowed layers hold rows from %d; position %d needs them from %d; %s",
				slot, r.kvBases[slot], end, end-(r.kvWindow-1), r.kvBudgetText())}
		}
		lo = max(0, r.kvBases[slot], end-(r.kvWindow-1)-r.kvRollback())
	}
	layers, words := r.snapshotLayout(end, lo)
	dn, words := r.snapshotDeltaLayout(words)
	if err := r.stream.Sync(); err != nil { // the prefill that wrote the rows has finished
		return nil, err
	}
	host, err := gpu.NewHostBuffer[float32](r.dev, words)
	if err != nil {
		return nil, &decoder.KVCopyDecline{Reason: fmt.Sprintf("no pinned host memory for %d MiB: %v", int64(words)*4>>20, err)}
	}
	s := &cudaKVSnapshot{r: r, rows: end, lo: lo, host: host, layers: layers, dn: dn, bytes: int64(words) * 4,
		ctxCap: r.ctxCap, window: r.kvWindow, slack: r.kvSlack, win: r.kvWin, half: r.kvF16}
	kc, vc := r.slotKV(slot)
	buf := host.Slice()
	for l := range r.layers {
		sl := layers[l]
		if !sl.held {
			continue
		}
		rows, srcRow := end, 0
		if sl.win {
			rows, srcRow = end-lo, lo-r.kvBases[slot]
		}
		n := rows * sl.rowWords
		off := srcRow * sl.rowWords * 4 // bytes into the device buffer
		if err := gpu.Download(kc[l].At(off), buf[sl.kOff:sl.kOff+n]); err != nil {
			s.Release()
			return nil, err
		}
		if sl.hasV {
			if err := gpu.Download(vc[l].At(off), buf[sl.vOff:sl.vOff+n]); err != nil {
				s.Release()
				return nil, err
			}
		}
	}
	for _, d := range dn {
		Ly := &r.layers[d.layer]
		if err := gpu.Download(Ly.dnWin, buf[d.winOff:d.winOff+d.winWords]); err != nil {
			s.Release()
			return nil, err
		}
		if err := gpu.Download(Ly.dnState, buf[d.stOff:d.stOff+d.stW]); err != nil {
			s.Release()
			return nil, err
		}
	}
	return s, nil
}

// RestoreKV implements decoder.ResidentKVSnapshotter.
func (r *cudaResident) RestoreKV(h decoder.ResidentKVSnapshot, slot, end int) error {
	s, ok := h.(*cudaKVSnapshot)
	switch {
	case !ok || s == nil || s.r != r:
		return fmt.Errorf("cuda: RestoreKV of a snapshot another resident took")
	case slot < 0 || slot >= r.KVSlots():
		return fmt.Errorf("cuda: RestoreKV slot %d of %d", slot, r.KVSlots())
	case end <= 0 || end%decoder.KVBlockRows != 0:
		return fmt.Errorf("cuda: RestoreKV rows [0, %d) are not whole %d-row blocks", end, decoder.KVBlockRows)
	case !s.Serves(end):
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("the snapshot's %d rows (windowed layers from %d) cannot serve a restore to %d, or the geometry changed", s.rows, s.lo, end)}
	}
	return r.do(func() error { return r.restoreKV(s, slot, end) })
}

// restoreKV is RestoreKV's job. Executor thread only.
func (r *cudaResident) restoreKV(s *cudaKVSnapshot, slot, end int) error {
	lo := 0
	if r.kvWin {
		lo = max(s.lo, end-(r.kvWindow-1)-r.kvRollback())
	}
	if err := r.stream.Sync(); err != nil {
		return err
	}
	kc, vc := r.slotKV(slot)
	buf := s.host.Slice()
	for l := range r.layers {
		sl := s.layers[l]
		if !sl.held {
			continue
		}
		rows, from := end, 0
		if sl.win {
			rows, from = end-lo, lo-s.lo
		}
		n := rows * sl.rowWords
		hostOff := from * sl.rowWords
		if err := gpu.Upload(kc[l].At(0), buf[sl.kOff+hostOff:sl.kOff+hostOff+n]); err != nil {
			return err
		}
		if sl.hasV {
			if err := gpu.Upload(vc[l].At(0), buf[sl.vOff+hostOff:sl.vOff+hostOff+n]); err != nil {
				return err
			}
		}
	}
	for _, d := range s.dn {
		Ly := &r.layers[d.layer]
		if Ly.dnWin.Len() != d.winWords || Ly.dnState.Len() != d.stW {
			return fmt.Errorf("cuda: RestoreKV: layer %d's DeltaNet state is %d+%d words, the snapshot's %d+%d", d.layer, Ly.dnWin.Len(), Ly.dnState.Len(), d.winWords, d.stW)
		}
		if err := gpu.Upload(Ly.dnWin, buf[d.winOff:d.winOff+d.winWords]); err != nil {
			return err
		}
		if err := gpu.Upload(Ly.dnState, buf[d.stOff:d.stOff+d.stW]); err != nil {
			return err
		}
	}
	if r.kvWin {
		r.kvBases[slot] = lo
		if slot == r.kvSlot {
			r.kvBase = lo
		}
	}
	return r.stream.Sync()
}
