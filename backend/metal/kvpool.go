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
	_ decoder.ResidentKVUploadPass   = (*metalResident)(nil)
)

// kvUploadPass is an upload in progress (decoder.ResidentKVUploadPass): one write of slot s from position 0, which its first layer maps
// through end and every layer that owns K/V must land.
type kvUploadPass struct {
	s, end  int
	reached map[int]int // the end each layer has landed, by layer
}

// BeginKVUpload (decoder.ResidentKVUploadPass) starts one write of the bound slot from position 0: the UploadKV calls until EndKVUpload
// are its layers.
func (a *metalResident) BeginKVUpload() error {
	if a.r == nil {
		return fmt.Errorf("metal: resident is nil")
	}
	a.r.stopExec()
	a.r.kvUpload = &kvUploadPass{s: a.r.kvSlot, reached: map[int]int{}}
	return nil
}

// EndKVUpload (decoder.ResidentKVUploadPass) ends the pass. The slot holds the rows only if some layer landed some and every layer that
// owns K/V landed the same end; otherwise, and when failed, it holds none, as a failed write from position 0 leaves it.
func (a *metalResident) EndKVUpload(failed bool) error {
	r := a.r
	if r == nil || r.kvUpload == nil {
		return nil
	}
	p := r.kvUpload
	r.kvUpload = nil
	drop := func(err error) error {
		if len(r.kvRefs) > 0 && p.s < len(r.kvSlotSt) {
			r.kvReleaseAll(p.s)
			r.kvSlotSt[p.s].valid = 0
		}
		return err
	}
	switch {
	case failed:
		return drop(nil)
	case p.end == 0:
		return drop(fmt.Errorf("metal: upload pass: no layer uploaded any rows"))
	}
	for l := range r.layers {
		if L := &r.layers[l]; L.geom != nil && !L.kvShared && r.kc[l] != (Buffer{}) && p.reached[l] != p.end {
			return drop(fmt.Errorf("metal: upload pass: layer %d reached %d of the pass's %d rows", l, p.reached[l], p.end))
		}
	}
	return nil
}

// kvUploadMap maps the bound slot's blocks for an upload of layer's rows [base, end). Outside a pass the upload is a write from base, which
// may not start past the slot's sequence (kvWrite). In a pass the slot is written from position 0: its first layer maps it through end, and
// each layer lands once, ends where the first did, and starts at 0, or, for a windowed layer, where its rows still cover its window
// (a CPU ring that has wrapped). The rows below that base are other layers' to write, and the decoder's reuse floor keeps a prefix off them.
func (r *resident) kvUploadMap(layer, base, end int) error {
	p := r.kvUpload
	if p == nil {
		return r.kvWrite(r.kvSlot, base, end)
	}
	win := int(r.layers[layer].window)
	switch _, again := p.reached[layer]; {
	case r.kvSlot != p.s:
		return fmt.Errorf("metal: upload pass: slot %d is bound, the pass writes slot %d", r.kvSlot, p.s)
	case again:
		return fmt.Errorf("metal: upload pass: layer %d uploaded twice", layer)
	case base > 0 && win == 0:
		return fmt.Errorf("metal: upload pass: full-attention layer %d starts at %d, not position 0", layer, base)
	case base > 0 && end-(win-1) < base:
		return fmt.Errorf("metal: upload pass: layer %d's rows [%d, %d) do not cover its %d-row window", layer, base, end, win)
	case p.end != 0 && end != p.end:
		return fmt.Errorf("metal: upload pass: layer %d ends at %d, the pass's first layer at %d", layer, end, p.end)
	}
	if p.end == 0 {
		if err := r.kvWrite(p.s, 0, end); err != nil {
			return err
		}
		p.end = end
	}
	p.reached[layer] = end
	return nil
}

// kvSlotState is one slot's standing in the pool. busy is set from the decoder's goroutines (SetKVSlotBusy); the rest only where
// writes are encoded.
type kvSlotState struct {
	busy     atomic.Bool
	use      uint64 // the pool's tick at the slot's last write: least recently written releases first
	released bool   // its blocks went to make room: it answers no reuse, and its next write starts a sequence at 0
	stopped  bool   // a borrower stopped to give its blocks past the share back: its next write past 0 is ErrKVPoolFull
	valid    int    // positions [0, valid) hold the slot's sequence
}

// kvRaise is the cap one conversation may reach on a load of slots slots whose planned per-slot context is share
// (decoder.Model.MetalKVCap, the rule the plan prices).
func kvRaise(m *decoder.Model, slots, share int) int { return m.MetalKVCap(slots, share) }

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
func (r *resident) kvRow(s, b int) *uint32 { return &r.kvTabU32[s*r.kvTabBlocks+b] }

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
	refuse := kvTakeFailForTest
	if kvTakeFailAfterForTest == 0 {
		refuse = true
	} else if kvTakeFailAfterForTest > 0 {
		kvTakeFailAfterForTest--
	}
	if !kvScrambleForTest && !refuse && b < r.kvShareBlocks {
		if h := uint32(1 + s*r.kvShareBlocks + b); r.kvRefs[h] == 0 {
			return r.kvClaim(h), nil
		}
	}
	for {
		if id, ok := r.kvFreeBlock(); ok && !refuse {
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

// kvTakeFailAfterForTest, when not negative, lets that many takes find blocks and makes every later one behave as kvTakeFailForTest.
var kvTakeFailAfterForTest = -1

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

// kvQuiesce readies slot s for a decode write ending at hi. While a greedy chain is open only the chain touches a slot's KV state, so
// it first ends the chain, which leaves the slot at its caller's sequence (stopChain). Then it stops the executor when the write would
// give a block past hi back (a rewind): a pipelined forward's command buffer, still in flight after its ack, may write a row there. The
// decode entry points call it before kvWrite, on the caller's thread; a decode that does not rewind runs on the executor, whose forwards
// in flight read only rows and table entries below it (a block they write is never shared). Every other writer stops the executor
// outright, which ends the chain too.
func (r *resident) kvQuiesce(s, hi int) {
	r.stopChain()
	for b := kvBlocksFor(hi); b < r.kvTabBlocks; b++ {
		if *r.kvRow(s, b) != 0 {
			r.stopExec()
			return
		}
	}
}

// kvEndAt ends slot s's sequence at end when it runs past it, giving the blocks past end back. Nothing of the resident may be in flight.
func (r *resident) kvEndAt(s, end int) {
	if len(r.kvRefs) == 0 || s >= len(r.kvSlotSt) {
		return
	}
	if st := &r.kvSlotSt[s]; st.valid > end {
		st.valid = end
		r.kvTrim(s, end)
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
// whether there was one. Its next write past position 0 is decoder.ErrKVPoolFull (kvWrite), and it joins kvReclaimed, which a batched
// step mapping its rows reads (kvMapStep).
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
	r.kvReclaimed = append(r.kvReclaimed, t)
	return true
}

// kvWrite maps the blocks slot s needs to write positions [lo, hi): an unmapped entry takes a block (kvTake), and an entry another
// table also names takes a copy of its rows below lo (copy on write; rows at or past lo are the write's). The write ends the slot's
// sequence at hi, so the blocks wholly past hi go back first, where the takes can have them; a write at 0 also clears released and
// stopped. A write past 0 on a stopped slot is decoder.ErrKVPoolFull, and on a released one, or one starting past the slot's sequence,
// an error (a caller that reused rows ReusableKV did not offer). It is one mapping (kvTx): a refusal leaves the slot as it was, and the
// caller must not encode the write. Nothing of the resident may be in flight that reads a block this copies into, which every caller
// ensures.
func (r *resident) kvWrite(s, lo, hi int) error {
	if len(r.kvRefs) == 0 || hi <= lo || s >= len(r.kvSlotSt) {
		return nil
	}
	tx := r.kvBegin()
	if err := tx.write(s, lo, hi); err != nil {
		tx.abort()
		return err
	}
	return tx.commit()
}

// kvTx is one mapping of slots' table entries in progress (kvWrite, kvMapStep). It records every entry it changes and each slot's standing
// before it, and changes only tables, counts and standings until it commits, when the copies on write run: a mapping that cannot finish
// is undone (abort) and leaves every slot it touched as it was, having written no row. A release or reclaim of another slot that one of
// its takes caused stands, as it would for a mapping that finished, and a slot of the mapping that a reclaim stopped stays stopped.
type kvTx struct {
	r       *resident
	log     []kvTxEdit
	saved   map[int]kvTxStanding // each slot's standing before the mapping
	pending map[int]kvTxStanding // and as the mapping leaves it
	cows    []kvTxCOW
	freed   map[uint32]bool // blocks the mapping's own trims gave back, which a take may reuse
}

// kvTxEdit is one entry change: slot s's entry b named prev and now names next; take marks next as a block a take handed the entry.
type kvTxEdit struct {
	s, b       int
	prev, next uint32
	take       bool
}

type kvTxStanding struct {
	valid             int
	released, stopped bool
}

// kvTxCOW is a copy on write to run at commit: the first keep rows of block from into block to, which slot s's entry b now names. The
// entry's reference to from goes at commit.
type kvTxCOW struct {
	s, b     int
	from, to uint32
	keep     int
}

// kvBegin starts a mapping.
func (r *resident) kvBegin() *kvTx {
	r.kvReclaimed = r.kvReclaimed[:0]
	return &kvTx{r: r, saved: map[int]kvTxStanding{}, pending: map[int]kvTxStanding{}, freed: map[uint32]bool{}}
}

// standing is slot s's standing as the mapping leaves it so far.
func (tx *kvTx) standing(s int) kvTxStanding {
	if p, ok := tx.pending[s]; ok {
		return p
	}
	st := &tx.r.kvSlotSt[s]
	p := kvTxStanding{valid: st.valid, released: st.released, stopped: st.stopped}
	tx.saved[s], tx.pending[s] = p, p
	return p
}

// trim unmaps every entry of slot s wholly at or past end, and ends its sequence there.
func (tx *kvTx) trim(s, end int) {
	r := tx.r
	p := tx.standing(s)
	for b := kvBlocksFor(end); b < r.kvTabBlocks; b++ {
		e := r.kvRow(s, b)
		if *e == 0 {
			continue
		}
		id := *e
		r.kvUnref(id)
		if r.kvRefs[id] == 0 {
			tx.freed[id] = true
		}
		tx.log = append(tx.log, kvTxEdit{s: s, b: b, prev: id})
		*e = 0
	}
	p.valid = min(p.valid, end)
	tx.pending[s] = p
}

// write maps slot s's blocks for positions [lo, hi), as kvWrite describes.
func (tx *kvTx) write(s, lo, hi int) error {
	r := tx.r
	p := tx.standing(s)
	switch {
	case lo > 0 && p.stopped:
		return fmt.Errorf("metal: KV slot %d was stopped to give its blocks past its share back: %w", s, decoder.ErrKVPoolFull)
	case lo > 0 && p.released:
		return fmt.Errorf("metal: KV slot %d's blocks were released to make room; its sequence starts again at position 0, not %d", s, lo)
	case lo > p.valid:
		return fmt.Errorf("metal: KV slot %d holds rows to %d; a write from %d would leave the rows between unmapped", s, p.valid, lo)
	}
	tx.trim(s, hi)
	for b := lo / decoder.KVBlockRows; b < min(kvBlocksFor(hi), r.kvTabBlocks); b++ {
		old := *r.kvRow(s, b)
		if old != 0 && (r.kvRefs[old] <= 1 || kvOwnOffForTest) {
			continue
		}
		nb, err := r.kvTake(s, b, old != 0)
		if err != nil {
			return err
		}
		tx.log = append(tx.log, kvTxEdit{s: s, b: b, prev: old, next: nb, take: true})
		*r.kvRow(s, b) = nb
		if old != 0 {
			keep := max(lo-b*decoder.KVBlockRows, 0)
			if kvKeepShortForTest && keep > 0 {
				keep--
			}
			tx.cows = append(tx.cows, kvTxCOW{s: s, b: b, from: old, to: nb, keep: keep})
		}
	}
	p = tx.pending[s]
	if lo == 0 {
		p.released, p.stopped = false, false
	}
	p.valid = hi
	tx.pending[s] = p
	return nil
}

// abort undoes the mapping, newest change first, and puts each slot's standing back, except a slot a take released or stopped meanwhile
// (a release or reclaim stands): released, it holds nothing; stopped, nothing past its share.
func (tx *kvTx) abort() {
	r := tx.r
	for i := len(tx.log) - 1; i >= 0; i-- {
		e := tx.log[i]
		if e.take {
			r.kvUnref(e.next) // a copy on write's entry kept its reference to prev throughout
		} else if r.kvRefs[e.prev] == 0 {
			r.kvClaim(e.prev) // a trim gave it back and nothing kept it
		} else {
			r.kvRefs[e.prev]++
		}
		*r.kvRow(e.s, e.b) = e.prev
	}
	share := r.kvShareBlocks * decoder.KVBlockRows
	for s, p := range tx.saved {
		st := &r.kvSlotSt[s]
		switch evicted, reclaimed := st.released && !p.released, st.stopped && !p.stopped; {
		case evicted: // a take released it meanwhile: it holds nothing
			r.kvReleaseAll(s)
			st.valid = 0
		case reclaimed: // a take stopped it meanwhile: nothing past its share
			st.valid = min(p.valid, share)
			r.kvTrim(s, share)
		default:
			st.valid, st.released, st.stopped = p.valid, p.released, p.stopped
		}
	}
}

// commit runs the copies on write and lets each slot's new standing stand. A copy that fails on the device undoes the mapping; a block a
// trim gave back and a failed copy wrote into no longer holds its old rows, so its slot's sequence then ends before it.
func (tx *kvTx) commit() error {
	r := tx.r
	for s, p := range tx.saved {
		if st := &r.kvSlotSt[s]; st.released != p.released || st.stopped != p.stopped {
			tx.abort()
			return fmt.Errorf("metal: KV slot %d was released or reclaimed while its blocks were mapped: %w", s, decoder.ErrKVPoolFull)
		}
	}
	for _, c := range tx.cows {
		if c.keep == 0 {
			continue
		}
		err := gpu.CopyDeviceBatch(r.kvBlockCopies(c.from, c.to, c.keep))
		if kvCopyFailForTest {
			err = errors.New("planted copy failure")
		}
		if err != nil {
			tx.abort()
			for _, d := range tx.cows {
				if tx.freed[d.to] {
					for _, e := range tx.log {
						if !e.take && e.prev == d.to {
							r.kvEndAt(e.s, e.b*decoder.KVBlockRows)
						}
					}
				}
			}
			return fmt.Errorf("metal: copy on write of KV block %d: %w", c.from, err)
		}
	}
	for _, c := range tx.cows {
		r.kvUnref(c.from)
	}
	for s, p := range tx.pending {
		st := &r.kvSlotSt[s]
		st.valid, st.released, st.stopped = p.valid, p.released, p.stopped
		r.kvUseTick++
		st.use = r.kvUseTick
	}
	return nil
}

// kvMapStep maps the K/V block of every row of a batched step before anything is encoded, as running the rows alone would, as one mapping
// (kvTx): each slot's sequence first ends past its last row (a rewinding borrower gives its tail back before another row looks for a
// block), then the rows that write within their slot's share map, then the ones that borrow. When a row finds none, or a take reclaimed
// one of the step's own slots (whose rows would then lack blocks), the mapping is undone and the step must launch nothing; a refusal is
// decoder.ErrKVPoolFull, which the decoder answers by running each row alone. A reclaim of a slot outside the step does not stop it.
func (r *resident) kvMapStep(seqs []batchSeq) error {
	tx := r.kvBegin()
	share := r.kvShareBlocks * decoder.KVBlockRows
	last := map[int]int{} // one past each slot's last row in the step
	for _, q := range seqs {
		last[q.slot] = max(last[q.slot], q.pos+1)
	}
	for s, hi := range last {
		if s < len(r.kvSlotSt) {
			tx.trim(s, hi)
		}
	}
	for _, within := range []bool{true, false} {
		for _, q := range seqs {
			if (q.pos < share) != within || q.slot >= len(r.kvSlotSt) {
				continue
			}
			if err := tx.write(q.slot, q.pos, q.pos+1); err != nil {
				tx.abort()
				return err
			}
		}
	}
	for _, t := range r.kvReclaimed {
		if _, ok := last[t]; ok {
			tx.abort()
			return fmt.Errorf("metal: batched step: slot %d's blocks were reclaimed while the step mapped its rows: %w", t, decoder.ErrKVPoolFull)
		}
	}
	return tx.commit()
}

// kvWriteMust is kvWrite where no caller can take an error back: setPos, reached from the executor's commit and the synchronous forwards,
// which the entry points map ahead of. A failure, a write past the slot's sequence included, is a missed entry point, named rather than
// written through an unmapped block or over stale rows. A test that picks the depth maps first (setPosAheadForTest).
func (r *resident) kvWriteMust(s, lo, hi int) {
	if err := r.kvWrite(s, lo, hi); err != nil {
		panic(fmt.Sprintf("metal: an unmapped KV write reached the encoder: %v", err))
	}
}

// setPosAheadForTest is setPos for a test that picks the depth: it first maps the bound slot's blocks from its sequence's end through pos,
// since setPos's own write does not start past the sequence.
func (r *resident) setPosAheadForTest(pos int, ropePos ...int) {
	if s := r.kvSlot; s < len(r.kvSlotSt) && pos > r.kvSlotSt[s].valid {
		if err := r.kvWrite(s, r.kvSlotSt[s].valid, pos+1); err != nil {
			panic(fmt.Sprintf("metal: setPosAheadForTest(%d): %v", pos, err))
		}
	}
	r.setPos(pos, ropePos...)
}

// kvReusable is how much of prefix slot s can serve from its own rows: prefix, or 0 when its blocks were released or reclaimed or it
// holds fewer rows (ResidentReusableKV's contract: prefix or 0, never another positive value). A slot with a greedy chain open holds its
// caller's sequence, not the look-ahead's (chainEnd).
func (r *resident) kvReusable(s, prefix int) int {
	if len(r.kvRefs) == 0 || s >= len(r.kvSlotSt) {
		return prefix
	}
	st := &r.kvSlotSt[s]
	valid := st.valid
	if r.chainReq != nil && s == r.chainSlot {
		valid = min(valid, r.chainEnd)
	}
	if st.released || st.stopped || prefix > valid {
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
