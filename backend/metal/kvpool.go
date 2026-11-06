//go:build darwin

package metal

import (
	"errors"
	"fmt"
	"sync/atomic"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

// The KV block pool's capacity (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan, "Capacity", KV4a-3's rule). Each layer's pool
// holds slots × kvShareBlocks blocks past the null block: the bytes the plan gives KV at the same flags. A slot's table starts with
// every entry unmapped (block 0) and takes blocks on demand as it writes (kvWrite). Each slot is guaranteed kvShareBlocks, the
// planned per-slot context; one conversation may borrow past it, up to its cap (r.ctxCap, kvRaise). When no block is free, idle slots
// release theirs, least recently written first; a slot within its share then takes back what a borrower holds past its own, which
// stops the borrower; a slot past its share that still finds none gets decoder.ErrKVPoolFull, which ends its turn as the context cap
// does. Every mutation runs on the thread that encodes the write, with nothing of the resident in flight.

var (
	_ decoder.ResidentKVPoolInfo     = (*metalResident)(nil)
	_ decoder.ResidentKVSlotActivity = (*metalResident)(nil)
	_ decoder.ResidentReusableKV     = (*metalResident)(nil)
	_ decoder.ResidentReusableKVSlot = (*metalResident)(nil)
)

// kvSlotState is one slot's standing in the pool. busy is set from the decoder's goroutines (SetKVSlotBusy); the rest only where
// writes are encoded.
type kvSlotState struct {
	busy     atomic.Bool
	use      uint64 // the pool's tick at the slot's last write: least recently written releases first
	released bool   // its blocks went to make room: it answers no reuse, and its next write starts a sequence at 0
	stopped  bool   // a borrower stopped to give its blocks past the share back: its next write past 0 is ErrKVPoolFull
	valid    int    // positions [0, valid) hold the slot's sequence
}

// kvRaise is the cap one conversation may reach: the pool's tokens, within the model's window and metalCtxCapMax, unless a context
// was requested (an operator's -ctx or the fit guard's auto-pin, as CUDA's kvRaise) or the load prices KV per position beside the
// pool (a drafter's). share is the planned per-slot context. One slot borrows from nothing, so it keeps its share.
func kvRaise(m *decoder.Model, slots, shareBlocks, share int) int {
	if slots < 2 || m.ResidentContextRequest() > 0 || m.ExtraResidentKVPerPosition() != 0 {
		return share
	}
	c := min(slots*shareBlocks*decoder.KVBlockRows, metalCtxCapMax)
	if w := m.Config().MaxPositions; w > 0 {
		c = min(c, w)
	}
	return max(c, share)
}

// kvBlockOrder is the order kvFreeBlock scans the pool's blocks: ascending, or with kvScrambleForTest, alternately from the two ends,
// so a sequence's consecutive blocks are physically far apart and out of order.
func kvBlockOrder(blocks int) []uint32 {
	out := make([]uint32, 0, blocks)
	if !kvScrambleForTest {
		for i := 1; i <= blocks; i++ {
			out = append(out, uint32(i))
		}
		return out
	}
	for i, j := 1, blocks; i <= j; i, j = i+1, j-1 {
		out = append(out, uint32(j))
		if i != j {
			out = append(out, uint32(i))
		}
	}
	return out
}

// kvClaim gives block id its first reference.
func (r *resident) kvClaim(id uint32) uint32 {
	r.kvRefs[id] = 1
	r.kvInUse++
	return id
}

// kvUnref drops one table entry's reference to block id; the last one frees it. The null block is never counted.
func (r *resident) kvUnref(id uint32) {
	if id == 0 {
		return
	}
	if r.kvRefs[id]--; r.kvRefs[id] == 0 {
		r.kvInUse--
	}
}

// kvBlocksInUse is how many blocks some table names, the null block not counted: the pool's own count of what its slots hold, a
// block several slots share counted once.
func (r *resident) kvBlocksInUse() int { return r.kvInUse }

// kvRow is slot s's table entry for its block b.
func (r *resident) kvRow(s, b int) *uint32 { return &r.kvTab.U32s()[s*r.kvTabBlocks+b] }

// kvHeld is how many entries slot s's table maps, shared blocks included.
func (r *resident) kvHeld(s int) int {
	n := 0
	for b := range r.kvTabBlocks {
		if *r.kvRow(s, b) != 0 {
			n++
		}
	}
	return n
}

// kvFreeBlock is a block no entry names, in kvOrder.
func (r *resident) kvFreeBlock() (uint32, bool) {
	for _, id := range r.kvOrder {
		if r.kvRefs[id] == 0 {
			return id, true
		}
	}
	return 0, false
}

// kvTake hands slot s a block for its block b: its own home block within the share when that is free (in order, so a slot's rows lie
// as they did before the pool; kvScrambleForTest skips it), else any free one, else one made free (the package comment's order). cow
// marks a copy on write, which holds no more blocks after it, so a slot at its share still counts as within it.
func (r *resident) kvTake(s, b int, cow bool) (uint32, error) {
	if !kvScrambleForTest && !kvTakeFailForTest && b < r.kvShareBlocks {
		if h := uint32(1 + s*r.kvShareBlocks + b); r.kvRefs[h] == 0 {
			return r.kvClaim(h), nil
		}
	}
	for {
		if id, ok := r.kvFreeBlock(); ok && !kvTakeFailForTest {
			return r.kvClaim(id), nil
		}
		if r.kvEvictIdle(s) {
			continue
		}
		if held := r.kvHeld(s); (held < r.kvShareBlocks || cow && held <= r.kvShareBlocks) && r.kvReclaim(s) {
			continue
		}
		return 0, fmt.Errorf("metal: KV slot %d holds %d blocks and the pool has none free: %w", s, r.kvHeld(s), decoder.ErrKVPoolFull)
	}
}

// kvTakeFailForTest makes a take find no free block, its home block included, so it goes straight to the release order.
var kvTakeFailForTest bool

// kvCopyFailForTest makes a copy on write's copy fail.
var kvCopyFailForTest bool

// kvEvictOrderPlantForTest releases the MOST recently written idle slot first: the plant TestKVPages_pressureFollowsD5 must catch.
var kvEvictOrderPlantForTest bool

// kvTrim unmaps every entry of slot s's table wholly at or past position end. Every write and copy keeps a slot to its sequence this way
// (kvSlotState.valid): no block past it is held, counted in kvHeld and kvInUse, or offered to CopyKV.
func (r *resident) kvTrim(s, end int) {
	for b := kvBlocksFor(end); b < r.kvTabBlocks; b++ {
		if e := r.kvRow(s, b); *e != 0 {
			r.kvUnref(*e)
			*e = 0
		}
	}
}

// kvQuiesce stops the executor and any open chain when a write of slot s ending at hi would give a block past hi back (a rewind): a
// pipelined forward's command buffer, still in flight after its ack, may write a row there. Every entry point calls it before kvWrite,
// on the caller's thread; the executor's and the chain's own writes never rewind past a forward still in flight.
func (r *resident) kvQuiesce(s, hi int) {
	for b := kvBlocksFor(hi); b < r.kvTabBlocks; b++ {
		if *r.kvRow(s, b) != 0 {
			r.stopExec()
			return
		}
	}
}

// kvReleaseAll unmaps every entry of slot s's table.
func (r *resident) kvReleaseAll(s int) {
	for b := range r.kvTabBlocks {
		e := r.kvRow(s, b)
		r.kvUnref(*e)
		*e = 0
	}
}

// kvEvictIdle releases every block of the least recently written slot that is not s, not busy and holds some, and reports whether there
// was one. The bound slot is no exception: every writer but the batched step writes the bound slot (so it is s), and the step names its
// rows' slots, which the decoder marks busy. The released slot answers no reuse until it starts a sequence again (kvReusable).
func (r *resident) kvEvictIdle(s int) bool {
	t := -1
	for c := range r.kvSlotSt {
		st := &r.kvSlotSt[c]
		if c == s || st.busy.Load() || r.kvHeld(c) == 0 {
			continue
		}
		if t < 0 || (st.use < r.kvSlotSt[t].use) != kvEvictOrderPlantForTest {
			t = c
		}
	}
	if t < 0 {
		return false
	}
	r.kvReleaseAll(t)
	st := &r.kvSlotSt[t]
	st.released, st.stopped, st.valid = true, false, 0
	r.kvEvictions++
	return true
}

// kvReclaim stops the slot other than s that holds the most blocks past its share, busy or not, giving those blocks back, and reports
// whether there was one. Its next write past position 0 is decoder.ErrKVPoolFull (kvWrite), and a batched step in progress is told to
// launch nothing (kvStepAbort), since one of its rows may be that borrower.
func (r *resident) kvReclaim(s int) bool {
	t, most := -1, 0
	for c := range r.kvSlotSt {
		if c == s || r.kvSlotSt[c].stopped {
			continue
		}
		past := 0
		for b := r.kvShareBlocks; b < r.kvTabBlocks; b++ {
			if *r.kvRow(c, b) != 0 {
				past++
			}
		}
		if past > most {
			t, most = c, past
		}
	}
	if t < 0 {
		return false
	}
	for b := r.kvShareBlocks; b < r.kvTabBlocks; b++ {
		e := r.kvRow(t, b)
		r.kvUnref(*e)
		*e = 0
	}
	st := &r.kvSlotSt[t]
	st.stopped, st.valid = true, min(st.valid, r.kvShareBlocks*decoder.KVBlockRows)
	r.kvStepAbort = true
	return true
}

// kvWrite maps the blocks slot s needs to write positions [lo, hi): an unmapped entry takes a block (kvTake), and an entry another
// table also names takes a copy of its rows below lo (copy on write; rows at or past lo are the write's). The write ends the slot's
// sequence at hi, so the blocks wholly past hi go back first, where the takes can have them; a write at 0 also clears released and
// stopped. A write past 0 on a stopped slot is decoder.ErrKVPoolFull, and on a released one an error (a caller that reused rows
// ReusableKV did not offer). A failure leaves the slot its sequence below lo (none for a write at 0) and no block past it, and the
// caller must not encode the write. Nothing of the resident may be in flight that reads a block this copies into, which every caller
// ensures.
func (r *resident) kvWrite(s, lo, hi int) error {
	if len(r.kvRefs) == 0 || hi <= lo || s >= len(r.kvSlotSt) {
		return nil
	}
	st := &r.kvSlotSt[s]
	switch {
	case lo > 0 && st.stopped:
		return fmt.Errorf("metal: KV slot %d was stopped to give its blocks past its share back: %w", s, decoder.ErrKVPoolFull)
	case lo > 0 && st.released:
		return fmt.Errorf("metal: KV slot %d's blocks were released to make room; its sequence starts again at position 0, not %d", s, lo)
	}
	if lo == 0 {
		st.released, st.stopped = false, false
	}
	r.kvTrim(s, hi)
	fail := func(err error) error {
		if lo == 0 {
			r.kvReleaseAll(s)
			st.valid = 0
		} else {
			st.valid = min(st.valid, lo)
			r.kvTrim(s, st.valid)
		}
		return err
	}
	r.kvUseTick++
	st.use = r.kvUseTick
	for b := lo / decoder.KVBlockRows; b < min(kvBlocksFor(hi), r.kvTabBlocks); b++ {
		old := *r.kvRow(s, b)
		if old != 0 && (r.kvRefs[old] <= 1 || kvOwnOffForTest) {
			continue
		}
		nb, err := r.kvTake(s, b, old != 0)
		if err != nil {
			return fail(err)
		}
		if old != 0 {
			if keep := lo - b*decoder.KVBlockRows; keep > 0 {
				if kvKeepShortForTest {
					keep--
				}
				err := gpu.CopyDeviceBatch(r.kvBlockCopies(old, nb, keep))
				if kvCopyFailForTest {
					err = errors.New("planted copy failure")
				}
				if err != nil {
					r.kvUnref(nb) // the entry still names old, which holds the rows below lo
					return fail(fmt.Errorf("metal: copy on write of KV block %d: %w", old, err))
				}
			}
			r.kvUnref(old)
		}
		*r.kvRow(s, b) = nb
	}
	st.valid = hi
	return nil
}

// kvWriteMust is kvWrite where no caller can take an error back (the executor's commit, which the entry points map ahead of): a failure
// there is a missed entry point, named rather than written through an unmapped block.
func (r *resident) kvWriteMust(s, lo, hi int) {
	if err := r.kvWrite(s, lo, hi); err != nil {
		panic(fmt.Sprintf("metal: an unmapped KV write reached the encoder: %v", err))
	}
}

// kvReusable is how much of prefix slot s can serve from its own rows: prefix, or 0 when its blocks were released or reclaimed or it
// holds fewer rows (ResidentReusableKV's contract: prefix or 0, never another positive value).
func (r *resident) kvReusable(s, prefix int) int {
	if len(r.kvRefs) == 0 || s >= len(r.kvSlotSt) {
		return prefix
	}
	if st := &r.kvSlotSt[s]; st.released || st.stopped || prefix > st.valid {
		return 0
	}
	return prefix
}

// ReusableKV implements decoder.ResidentReusableKV for the bound slot.
func (a *metalResident) ReusableKV(prefix int) int { return a.r.kvReusable(a.r.kvSlot, prefix) }

// ReusableKVSlot implements decoder.ResidentReusableKVSlot.
func (a *metalResident) ReusableKVSlot(slot, prefix int) int { return a.r.kvReusable(slot, prefix) }

// SetKVSlotBusy implements decoder.ResidentKVSlotActivity: a busy slot's blocks are never released to make room for another.
func (a *metalResident) SetKVSlotBusy(slot int, busy bool) {
	if slot >= 0 && slot < len(a.r.kvSlotSt) {
		a.r.kvSlotSt[slot].busy.Store(busy)
	}
}

// KVPool implements decoder.ResidentKVPoolInfo: the positions every slot's blocks hold together, each slot's guaranteed share, and the
// bytes the pools and tables take.
func (a *metalResident) KVPool() decoder.KVPoolDetail {
	r := a.r
	if len(r.kvRefs) == 0 {
		return decoder.KVPoolDetail{}
	}
	var bytes int64 // kc/vc are byte buffers, ks/vs f32 and kvTab u32 ones (Len counts elements)
	for l := range r.layers {
		if r.layers[l].kvShared || r.kc[l] == (Buffer{}) {
			continue
		}
		bytes += int64(r.kc[l].Len())
		if r.vc[l] != r.kc[l] { // an MLA layer's latent rows are both
			bytes += int64(r.vc[l].Len())
		}
		if r.kvI8 && r.ks[l] != (Buffer{}) {
			bytes += 4 * int64(r.ks[l].Len()+r.vs[l].Len())
		}
	}
	bytes += 4 * int64(r.kvTab.Len())
	prec := "f16"
	switch {
	case r.kvI8:
		prec = "i8"
	case r.kvF32 || r.mla != nil:
		prec = "f32"
	}
	return decoder.KVPoolDetail{Positions: (len(r.kvRefs) - 1) * decoder.KVBlockRows, Guaranteed: r.kvShareCtx, BlockRows: decoder.KVBlockRows,
		Groups: []string{"full"}, Precision: prec, Bytes: bytes}
}
