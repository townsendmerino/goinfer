//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// Every path that writes a slot's rows (a prefill or decode step, an upload, a copy, a restore or promotion) goes through kvWriteBegin and ends in
// kvWriteDone or kvWriteFailed, so a slot's kvValid, kvStopped and released mark move one way. TestKV7_slotWriteTable and TestKV7_uploadPass hold them to it.

// kvSlotWrite is one write in progress on slot s.
type kvSlotWrite struct {
	s, from int
	fresh   bool // released or stopped: it answers for no rows of its own, so the write takes every row from position 0
}

// kvUploadPass is the upload in progress between BeginKVUpload and EndKVUpload: one write of the bound slot, up to the end its layers have landed.
type kvUploadPass struct {
	w    kvSlotWrite
	pass *kvmgr.UploadPass
}

// BeginKVUpload implements decoder.ResidentKVSubsystem.
func (r *cudaResident) BeginKVUpload() error {
	return r.do(func() error {
		w := r.kvWriteBegin(r.kvSlot, 0)
		r.kvWriteForget(w)
		r.kvUpload = &kvUploadPass{w: w, pass: kvmgr.NewUploadPass(r.kvSlot)}
		return nil
	})
}

// EndKVUpload implements decoder.ResidentKVSubsystem. The slot holds the rows only if some landed and every layer with a KV cache reached the same end; UploadKV
// has already refused a layer that uploaded twice, a full-attention layer that did not start at position 0, and a sliding layer whose rows do not cover its window.
// Anything else fails the pass, whatever order the layers came in.
func (r *cudaResident) EndKVUpload(failed bool) error {
	return r.do(func() error {
		p := r.kvUpload
		r.kvUpload = nil
		if p == nil {
			return nil
		}
		end, floor, err := p.pass.Finish("cuda", failed, len(r.layers), func(l int) bool {
			Ly := &r.layers[l]
			return !Ly.kvShared && !Ly.noKV() && Ly.kvDim > 0
		})
		if failed || err != nil {
			r.kvWriteFailed(p.w, 0)
			return err
		}
		r.kvWriteDone(p.w, end)
		r.kvFloor[p.w.s] = floor
		return nil
	})
}

// kvWriteRefused is the error for a write at p0 that a released or stopped slot cannot take: its next write must start from position 0. nil
// otherwise. Executor thread only.
func (r *cudaResident) kvWriteRefused(s, p0 int) error {
	if p0 <= 0 {
		return nil
	}
	if r.kvReleased(s) {
		return fmt.Errorf("cuda: KV slot %d's blocks were released to make room; a write at %d must start from position 0 (ReusableKV answers 0)", s, p0)
	}
	if r.kvStopped[s] {
		return fmt.Errorf("cuda: KV slot %d was stopped: its blocks past its share were taken back for a conversation within its share: %w", s, decoder.ErrKVPoolFull)
	}
	return nil
}

// kvWriteBegin starts a write of slot s's rows from from. A released or stopped slot takes them from position 0. Executor thread only.
func (r *cudaResident) kvWriteBegin(s, from int) kvSlotWrite {
	w := kvSlotWrite{s: s, from: from, fresh: r.kvReleased(s) || r.kvStopped[s]}
	if w.fresh {
		w.from = 0
	}
	return w
}

// kvWriteForget is for a write from position 0, whose caller has already forgotten the slot's rows: a released or stopped slot answers for none, and any
// host copy goes now, so its bytes are there for the releases this write may cause. Executor thread only.
func (r *cudaResident) kvWriteForget(w kvSlotWrite) {
	if w.fresh {
		r.kvDropDemoted(w.s)
		r.kvValid[w.s] = 0
	}
}

// kvWriteDone ends a write that reached end: the slot is live, its host copy of any older rows is gone, and it answers for the rows up to end.
// Executor thread only.
func (r *cudaResident) kvWriteDone(w kvSlotWrite, end int) {
	r.kvClearReleased(w.s)
	r.kvValid[w.s], r.kvStopped[w.s] = end, false
	if w.from == 0 {
		r.kvFloor[w.s] = 0
	}
}

// kvWriteFailed ends a write that did not finish. A released or stopped slot stays so and holds none of the blocks mapped for it, and with no host copy it
// records no rows. A live slot answers for its rows below lost, and holds no block past them. Executor thread only.
func (r *cudaResident) kvWriteFailed(w kvSlotWrite, lost int) {
	if w.fresh {
		r.kvReleaseAll(w.s, kvGroupFull)
		r.kvReleaseAll(w.s, kvGroupWin)
		if r.kvDemoted[w.s] == nil {
			r.kvValid[w.s] = 0
		}
		return
	}
	r.kvValid[w.s] = min(r.kvValid[w.s], lost)
	t := &r.kvTabs[w.s]
	for g := range kvGroups {
		ids := t.ids[g]
		for b := kvBlocksFor(r.kvValid[w.s]); b < len(ids); b++ {
			if ids[b] != 0 {
				r.kvPools[g].Unref(ids[b])
				ids[b] = 0
				t.dirty[g] = true
			}
		}
	}
}
