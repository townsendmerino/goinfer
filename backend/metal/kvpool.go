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

var ()

// kvUploadPass is an upload in progress (decoder.ResidentKVSubsystem): one write of slot s from position 0, which its first layer maps
// through end and every layer that owns K/V must land.
type kvUploadPass struct {
	s, end  int
	reached map[int]int // the end each layer has landed, by layer
	wrote   bool        // the first layer mapped the slot and wrote its rows: a failed pass has changed the slot
}

// BeginKVUpload (decoder.ResidentKVSubsystem) starts one write of the bound slot from position 0: the UploadKV calls until EndKVUpload
// are its layers.
func (a *metalResident) BeginKVUpload() error {
	if a.r == nil {
		return fmt.Errorf("metal: resident is nil")
	}
	a.r.stopExec()
	a.r.kvUpload = &kvUploadPass{s: a.r.kvSlot, reached: map[int]int{}}
	return nil
}

// EndKVUpload (decoder.ResidentKVSubsystem) ends the pass. The slot holds the rows only if some layer landed some and every layer that
// owns K/V landed the same end. Otherwise, and when failed, a pass that wrote leaves it holding none, since its rows are part one
// sequence's and part another's; one that failed before its first layer wrote leaves it as it was.
func (a *metalResident) EndKVUpload(failed bool) error {
	r := a.r
	if r == nil || r.kvUpload == nil {
		return nil
	}
	p := r.kvUpload
	r.kvUpload = nil
	drop := func(err error) error {
		if p.wrote && len(r.kvRefs) > 0 && p.s < len(r.kvSlotSt) {
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

// kvUploadMap maps the bound slot's blocks for an upload of layer's rows [base, end). On a paged pool an upload outside a pass is refused:
// the slot's table is every layer's, so one layer's mapping would claim rows the other layers have not landed. In a pass the slot is written
// from position 0: its first layer maps it through end, and each layer lands once, ends where the first did, and starts at 0, or, for a
// windowed layer, where its rows still cover its window (a CPU ring that has wrapped). The rows below that base are other layers' to
// write, and the decoder's reuse floor keeps a prefix off them.
func (r *resident) kvUploadMap(layer, base, end int) error {
	p := r.kvUpload
	if p == nil {
		if len(r.kvRefs) == 0 {
			return nil
		}
		return fmt.Errorf("metal: UploadKV of layer %d outside an upload pass (BeginKVUpload) on a paged KV pool", layer)
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
		p.end, p.wrote = end, true
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
	lost     int    // how many times a release or a reclaim has taken its blocks: a mapping compares it, not the flags, which may not change
}

// kvRaise is the cap one conversation may reach on a load of slots slots whose planned per-slot context is share
// (decoder.Model.MetalKVCap, the rule the plan prices).
func kvRaise(m *decoder.Model, slots, share int) int { return m.MetalKVCap(slots, share) }

// kvBlockOrder is the order kvFreeBlock scans the pool's blocks: ascending, or with kvScrambleForTest, alternately from the two ends,
// so a sequence's consecutive blocks are physically far apart and out of order.
func kvBlockOrder(blocks int, window bool) []uint32 {
	out := make([]uint32, 0, blocks)
	if !kvScrambleForTest {
		for i := 1; i <= blocks; i++ {
			out = append(out, uint32(i))
		}
		return out
	}
	for i, j := 1, blocks; i <= j; i, j = i+1, j-1 {
		first, second := uint32(j), uint32(i)
		if window { // the window group starts from the other end, so the two groups' tables name different blocks for one position
			first, second = second, first
		}
		out = append(out, first)
		if i != j {
			out = append(out, second)
		}
	}
	return out
}

// kvGroup is the table and the counts over one layer group's pools (KV5-7, docs/tasks/task-kv-architecture-2026-10.md §6): the layers of
// a group share one block id space, so a block id names the same block in each. kvTab holds each slot's table in one allocation,
// kvTabBlocks entries a slot, and kt is the bound slot's view of it. An entry is the block a sequence's 64 positions are in, so a kernel
// bound to a table reads the pool from its start (Buffer.At(0)) (kv_row, KVRowMSL). kvTabU32 is kvTab's shared contents, taken once:
// Buffer.U32s asks Metal for them on every call, and the pool reads entries per row. kvRefs counts the entries that name each block
// (above one, slots share it, CopyKV), kvInUse the blocks some entry names, kvOrder the order a take scans for a free one, and
// kvShareBlocks is each slot's guaranteed share.
//
// A window group (win > 0, KV5-7c) keeps only the recent positions of each sequence: kvBase[s] is the first position slot s still holds,
// a multiple of 64, and the entries below it name block 0, whose zero rows a window kernel's tile over-read lands in. A block that falls
// behind the base is retired (kvRetired), not freed: its entry stays until every command buffer committed before it left has completed.
type kvGroup struct {
	kvTab, kt     Buffer
	kvTabU32      []uint32
	kvTabBlocks   int
	kvRefs        []int32
	kvInUse       int
	kvOrder       []uint32
	kvShareBlocks int
	win           int         // the window length in positions, or 0 for a group that keeps every position
	kvBase        []int       // each slot's first held position (a window group)
	retired       []kvRetired // blocks behind a base whose entries are not yet repointed
}

// kvRetired is slot s's table entry b, naming block id, which fell behind its group's base when tag command buffers had been committed:
// it is repointed to block 0 and the block freed once that many have completed (kvReap).
type kvRetired struct {
	s, b int
	id   uint32
	tag  int64
}

// kvClaim gives block id its first reference.
func (g *kvGroup) kvClaim(id uint32) uint32 {
	g.kvRefs[id] = 1
	g.kvInUse++
	return id
}

// kvUnref drops one table entry's reference to block id; the last one frees it. The null block is never counted.
func (g *kvGroup) kvUnref(id uint32) {
	if id == 0 {
		return
	}
	if g.kvRefs[id]--; g.kvRefs[id] == 0 {
		g.kvInUse--
	}
}

// kvBlocksInUse is how many blocks some table names, the null block not counted: the pool's own count of what its slots hold, a
// block several slots share counted once.
func (g *kvGroup) kvBlocksInUse() int { return g.kvInUse }

// kvRow is slot s's table entry for its block b.
func (g *kvGroup) kvRow(s, b int) *uint32 { return &g.kvTabU32[s*g.kvTabBlocks+b] }

// kvHeld is how many entries slot s's table maps, shared blocks included.
func (g *kvGroup) kvHeld(s int) int {
	n := 0
	for b := range g.kvTabBlocks {
		if *g.kvRow(s, b) != 0 {
			n++
		}
	}
	return n
}

// kvFreeBlock is a block no entry names, in kvOrder.
func (g *kvGroup) kvFreeBlock() (uint32, bool) {
	for _, id := range g.kvOrder {
		if g.kvRefs[id] == 0 {
			return id, true
		}
	}
	return 0, false
}

// kvWindowLayers is the set of layers that read the window group: the layers KVGeometry puts in a window-retention group, when the model
// also has layers in another group. A model whose every layer slides has one group, the primary, and the set is empty. Never nil.
func kvWindowLayers(g decoder.KVGeometry) map[int]bool {
	set := map[int]bool{}
	other := false
	for _, grp := range g.Groups {
		if grp.Retention == decoder.KVRetainWindow {
			for _, l := range grp.Layers {
				set[l] = true
			}
		} else {
			other = true
		}
	}
	if !other {
		clear(set)
	}
	return set
}

// kvWindowSpan is the window length KVGeometry gives the window-retention layers, 0 when they differ (a group keeps one length); allWindow
// is whether every K/V layer of the model slides, so the primary group is the window group.
func kvWindowSpan(g decoder.KVGeometry) (span int, allWindow bool) {
	win, other := false, false
	for _, grp := range g.Groups {
		switch {
		case grp.Retention != decoder.KVRetainWindow:
			other = true
		case !win:
			win, span = true, grp.Span
		case span != grp.Span:
			span = 0
		}
	}
	return span, win && !other
}

// kvGroupOf is the group layer l's attention reads: the window group for a sliding layer of a model that has both kinds, else the
// primary group.
func (r *resident) kvGroupOf(l int) *kvGroup {
	if r.layers[l].kvg == 1 && r.kvWin != nil {
		return r.kvWin
	}
	return &r.kvGroup
}

// kvBindTables points every group's bound table view at slot s's.
func (r *resident) kvBindTables(s int) {
	b := r.kvSlotBufs[s]
	r.kt = b.kt
	if r.kvWin != nil {
		r.kvWin.kt = b.ktw
	}
}

// kvBoundTables is the views the groups have bound now (kvBindTables), the primary group's first, for kvRestoreTables.
func (r *resident) kvBoundTables() [2]Buffer {
	v := [2]Buffer{r.kt, {}}
	if r.kvWin != nil {
		v[1] = r.kvWin.kt
	}
	return v
}

// kvRestoreTables puts back the views kvBoundTables read.
func (r *resident) kvRestoreTables(v [2]Buffer) {
	r.kt = v[0]
	if r.kvWin != nil {
		r.kvWin.kt = v[1]
	}
}

// kvKT is the bound slot's table for layer l, the buffer l's kernels bind.
func (r *resident) kvKT(l int) Buffer { return r.kvGroupOf(l).kt }

// kvKTRead is kvKT for a decode attention read. kvWindowReadPlantForTest binds the primary group's table there for a window layer, whose
// stores still go through its own: the plant TestKVPages_windowGroup must catch.
func (r *resident) kvKTRead(l int) Buffer {
	if kvWindowReadPlantForTest && r.kvWin != nil && r.layers[l].kvg == 1 {
		return r.kt
	}
	return r.kvKT(l)
}

// kvWindowReadPlantForTest makes a window layer's decode attention read the primary group's table (kvKTRead).
var kvWindowReadPlantForTest bool

// kvWindowKeep is how many positions a window group keeps behind the start of the window a write reads: the rollback budget, the floor
// kvmgr.WindowNextBase's callers use (docs/tasks/task-kv-architecture-2026-10.md §5.4), so an edit late in a prompt still finds its
// window held.
const kvWindowKeep = 448

// kvRetainOffForTest keeps every position in the window group, as KV5-7b did: the arm KV5-7c's identity readings compare against.
var kvRetainOffForTest bool

// kvActivityForTest, when set, replaces the device's own count of command buffers committed and waited on, so a test can say when a
// buffer has completed.
var kvActivityForTest func() (submits, waits int64)

// kvActivity is how many command buffers this device has committed and how many the host has waited on. Buffers complete in the order
// they are committed, so a block retired when n had been committed is free to reuse once n have been waited on.
func (r *resident) kvActivity() (submits, waits int64) {
	if kvActivityForTest != nil {
		return kvActivityForTest()
	}
	_, submits, waits = r.d.Activity()
	return submits, waits
}

// kvWindowFloor is the first position a window group keeps after a write of [lo, hi) when it held from base: the lower of what the
// write's own reads reach (lo - win + 1) and the end's window less kvWindowKeep (kvmgr.WindowNextBase), in whole blocks, never below base.
func kvWindowFloor(win, base, lo, hi int) int {
	f := min(lo-win+1, hi-(win-1)-kvWindowKeep)
	if kvRetainEarlyPlantForTest {
		f += kvWindowKeep + win + decoder.KVBlockRows // keeps nothing behind the window start, and drops the block being written
	}
	return max(base, f/decoder.KVBlockRows*decoder.KVBlockRows)
}

// kvRetainEarlyPlantForTest releases the window group's blocks up to the end of the write, the window's own rows included: the plant
// TestKVWindow_retentionIsBitIdentical must catch.
var kvRetainEarlyPlantForTest bool

// kvRetain moves every window group's base for slot s after a write of [lo, hi): the blocks wholly behind it are retired, each tagged
// with the command buffers committed so far. A write at 0 starts a sequence: the base goes back to 0 and the blocks it had retired,
// which the write is about to use again, are not retired any more.
func (r *resident) kvRetain(s, lo, hi int) {
	if kvRetainOffForTest {
		return
	}
	for _, g := range r.kvGs {
		if g.win == 0 {
			continue
		}
		if lo == 0 {
			g.kvBase[s] = 0
		}
		g.dropRetired(s, lo/decoder.KVBlockRows, g.kvTabBlocks)
		from := g.kvBase[s]
		to := kvWindowFloor(g.win, from, lo, hi)
		if to <= from {
			continue
		}
		submits, _ := r.kvActivity()
		for b := from / decoder.KVBlockRows; b < to/decoder.KVBlockRows && b < g.kvTabBlocks; b++ {
			if e := *g.kvRow(s, b); e != 0 {
				g.retired = append(g.retired, kvRetired{s: s, b: b, id: e, tag: submits})
			}
		}
		g.kvBase[s] = to
	}
}

// kvReap repoints each retired entry to block 0 and frees its block once the command buffers committed before it was retired have
// completed; with all set, which only a caller with nothing in flight may do, it does so for every one.
func (r *resident) kvReap(all bool) {
	_, waits := r.kvActivity()
	for _, g := range r.kvGs {
		keep := g.retired[:0]
		for _, x := range g.retired {
			if !all && x.tag > waits && !kvReapEarlyPlantForTest {
				keep = append(keep, x)
				continue
			}
			if e := g.kvRow(x.s, x.b); *e == x.id {
				if !kvNoRepointPlantForTest {
					*e = 0
				}
				g.kvUnref(x.id)
				if kvRetainEarlyPlantForTest {
					r.kvClearBlock(g, x.id)
				}
			}
		}
		g.retired = keep
	}
}

// kvClearBlock zeroes block id's rows in every layer of group g, K and V and int8's scales: what a freed block's next owner would find
// there. Only the early plant does it, so that releasing rows a window needs shows as lost rows.
func (r *resident) kvClearBlock(g *kvGroup, id uint32) {
	for l := range r.layers {
		if L := &r.layers[l]; L.kvShared || r.kvSlotBytes[l] == 0 || r.kvGroupOf(l) != g {
			continue
		}
		n := r.kvNullBytes[l]
		clear(r.kc[l].Int8s()[int(id)*n : (int(id)+1)*n])
		if r.vc[l] != r.kc[l] {
			clear(r.vc[l].Int8s()[int(id)*n : (int(id)+1)*n])
		}
		if r.kvI8 && r.ks[l] != (Buffer{}) {
			s := r.layers[l].geom.nKV * decoder.KVBlockRows
			clear(r.ks[l].Floats()[int(id)*s : (int(id)+1)*s])
			clear(r.vs[l].Floats()[int(id)*s : (int(id)+1)*s])
		}
	}
}

// kvReapEarlyPlantForTest frees a retired block without waiting for the buffers committed before it: the plant
// TestKVWindow_retiredBlocksWaitForTheirBuffers must catch.
var kvReapEarlyPlantForTest bool

// kvNoRepointPlantForTest frees a retired block and leaves its entry naming it: the plant TestKVWindow_retentionIsBitIdentical must catch.
var kvNoRepointPlantForTest bool

// dropRetired forgets slot s's retired entries for blocks [from, to): whoever put an entry there since owns it.
func (g *kvGroup) dropRetired(s, from, to int) {
	keep := g.retired[:0]
	for _, x := range g.retired {
		if x.s != s || x.b < from || x.b >= to {
			keep = append(keep, x)
		}
	}
	g.retired = keep
}

// kvTableBytes is what every group's tables hold.
func (r *resident) kvTableBytes() int64 {
	n := 0
	for _, g := range r.kvGs {
		n += g.kvTab.Len()
	}
	return 4 * int64(n)
}

// kvHeldAny is how many entries slot s's tables map in every group together.
func (r *resident) kvHeldAny(s int) int {
	n := 0
	for _, g := range r.kvGs {
		n += g.kvHeld(s)
	}
	return n
}

// kvSlotKT is slot s's table for layer l, whichever slot is bound.
func (r *resident) kvSlotKT(l, s int) Buffer {
	if r.kvWin != nil && r.kvGroupOf(l) == r.kvWin {
		return r.kvSlotBufs[s].ktw
	}
	return r.kvSlotBufs[s].kt
}

// kvTabOf is layer l's whole table buffer (every slot's), which a batched step's rows index with their own offsets.
func (r *resident) kvTabOf(l int) Buffer { return r.kvGroupOf(l).kvTab }

// kvTake hands slot s a block for its block b: its own home block within the share when that is free (in order, so a slot's rows lie
// as they did before the pool; kvScrambleForTest skips it), else any free one, else one made free (the package comment's order). cow
// marks a copy on write, which holds no more blocks after it, so a slot at its share still counts as within it.
func (r *resident) kvTake(g *kvGroup, s, b int, cow bool) (uint32, error) {
	refuse := kvTakeFailForTest
	if kvTakeFailAfterForTest == 0 {
		refuse = true
	} else if kvTakeFailAfterForTest > 0 {
		kvTakeFailAfterForTest--
	}
	if !kvScrambleForTest && !refuse && b < g.kvShareBlocks {
		if h := uint32(1 + s*g.kvShareBlocks + b); g.kvRefs[h] == 0 {
			return g.kvClaim(h), nil
		}
	}
	for {
		if id, ok := g.kvFreeBlock(); ok && !refuse {
			return g.kvClaim(id), nil
		}
		if r.kvEvictIdle(s) {
			continue
		}
		if held := g.kvHeld(s); (held < g.kvShareBlocks || cow && held <= g.kvShareBlocks) && r.kvReclaim(g, s) {
			continue
		}
		return 0, fmt.Errorf("metal: KV slot %d holds %d blocks and the pool has none free: %w", s, g.kvHeld(s), decoder.ErrKVPoolFull)
	}
}

// kvTakeFailForTest makes a take find no free block, its home block included, so it goes straight to the release order.
var kvTakeFailForTest bool

// kvTakeFailAfterForTest, when not negative, lets that many takes find blocks and makes every later one behave as kvTakeFailForTest.
var kvTakeFailAfterForTest = -1

// kvCopyFailForTest makes a copy on write's copy fail, after it has run.
var kvCopyFailForTest bool

// kvCopyFailAfterForTest, when not negative, lets that many copies on write succeed and makes every later one behave as kvCopyFailForTest.
var kvCopyFailAfterForTest = -1

// kvEvictOrderPlantForTest releases the MOST recently written idle slot first: the plant TestKVPages_pressureFollowsD5 must catch.
var kvEvictOrderPlantForTest bool

// kvTrim unmaps every entry of slot s's table wholly at or past position end. Every write and copy keeps a slot to its sequence this way
// (kvSlotState.valid): no block past it is held, counted in kvHeld and kvInUse, or offered to CopyKV.
func (r *resident) kvTrim(s, end int) {
	for _, g := range r.kvGs {
		g.dropRetired(s, kvBlocksFor(end), g.kvTabBlocks)
		for b := kvBlocksFor(end); b < g.kvTabBlocks; b++ {
			if e := g.kvRow(s, b); *e != 0 {
				g.kvUnref(*e)
				*e = 0
			}
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
	for _, g := range r.kvGs {
		for b := kvBlocksFor(hi); b < g.kvTabBlocks; b++ {
			if *g.kvRow(s, b) != 0 {
				r.stopExec()
				return
			}
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
	for _, g := range r.kvGs {
		for b := range g.kvTabBlocks {
			e := g.kvRow(s, b)
			g.kvUnref(*e)
			*e = 0
		}
		g.kvBase[s] = 0
		g.dropRetired(s, 0, g.kvTabBlocks)
	}
}

// kvEvictIdle releases every block of the least recently written slot that is not s, not busy and holds some, and reports whether there
// was one. The bound slot is no exception: every writer but the batched step writes the bound slot (so it is s), and the step names its
// rows' slots, which the decoder marks busy. The released slot answers no reuse until it starts a sequence again (kvReusable).
func (r *resident) kvEvictIdle(s int) bool {
	t := -1
	for c := range r.kvSlotSt {
		st := &r.kvSlotSt[c]
		if c == s || st.busy.Load() || r.kvHeldAny(c) == 0 {
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
	st.lost++
	r.kvEvictions++
	return true
}

// kvReclaim stops the slot other than s that holds the most blocks past its share, busy or not, giving those blocks back, and reports
// whether there was one. Its next write past position 0 is decoder.ErrKVPoolFull (kvWrite), and it joins kvReclaimed, which a batched
// step mapping its rows reads (kvMapStep).
func (r *resident) kvReclaim(g *kvGroup, s int) bool {
	t, most := -1, 0
	for c := range r.kvSlotSt {
		if c == s || r.kvSlotSt[c].stopped {
			continue
		}
		past := 0
		for b := g.kvShareBlocks; b < g.kvTabBlocks; b++ {
			if *g.kvRow(c, b) != 0 {
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
	for _, h := range r.kvGs {
		h.dropRetired(t, h.kvShareBlocks, h.kvTabBlocks)
		for b := h.kvShareBlocks; b < h.kvTabBlocks; b++ {
			e := h.kvRow(t, b)
			h.kvUnref(*e)
			*e = 0
		}
	}
	st := &r.kvSlotSt[t]
	st.stopped, st.valid = true, min(st.valid, r.kvShareBlocks*decoder.KVBlockRows)
	st.lost++
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
	r.kvReap(false)
	if err := r.kvWindowReads(s, lo); err != nil {
		return err
	}
	if r.kvWriteInPlace(s, lo, hi) {
		r.kvRetain(s, lo, hi)
		return nil
	}
	tx := r.kvBegin()
	if err := tx.write(s, lo, hi); err != nil {
		tx.abort()
		return err
	}
	if err := tx.commit(); err != nil {
		return err
	}
	r.kvRetain(s, lo, hi)
	return nil
}

// kvWindowReads refuses a write from lo that a window group no longer holds the rows for: its first query reads from lo - win + 1, and
// the group released everything below its base. ReusableKV answers 0 for such a prefix, so only a caller that ignored it gets here.
func (r *resident) kvWindowReads(s, lo int) error {
	if kvRetainOffForTest || kvRetainEarlyPlantForTest || lo == 0 {
		return nil
	}
	for _, g := range r.kvGs {
		if g.win > 0 && max(lo-g.win+1, 0) < g.kvBase[s] {
			return fmt.Errorf("metal: KV slot %d's window group holds rows from %d; a write from %d reads rows below it", s, g.kvBase[s], lo)
		}
	}
	return nil
}

// kvWriteInPlace is kvWrite with nothing to map, most decode tokens: every entry the write reaches holds a block of the slot's alone,
// none lies past hi, and the slot is neither released nor stopped, so the write only moves the sequence's end. It reports whether it
// applied; anything else, a refusal included, is the transaction's.
func (r *resident) kvWriteInPlace(s, lo, hi int) bool {
	st := &r.kvSlotSt[s]
	if lo > st.valid || st.stopped || st.released {
		return false
	}
	for _, g := range r.kvGs {
		for b := lo / decoder.KVBlockRows; b < g.kvTabBlocks; b++ {
			e := *g.kvRow(s, b)
			if past := b >= kvBlocksFor(hi); (past && e != 0) || (!past && (e == 0 || (g.kvRefs[e] > 1 && !kvOwnOffForTest))) {
				return false
			}
		}
	}
	st.valid = hi
	r.kvUseTick++
	st.use = r.kvUseTick
	return true
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
}

// kvTxEdit is one entry change: slot s's entry b named prev and now names next; take marks next as a block a take handed the entry, and
// cow a take whose entry gave its reference to prev back (a copy on write).
type kvTxEdit struct {
	g, s, b    int // g indexes resident.kvGs
	prev, next uint32
	take, cow  bool
}

type kvTxStanding struct {
	valid             int
	released, stopped bool
	lost              int // kvSlotState.lost when the mapping first touched the slot
}

// kvTxCOW is a copy on write to run at commit: the first keep rows of block from into block to, which slot s's entry b now names.
type kvTxCOW struct {
	g, s, b  int // g indexes resident.kvGs
	from, to uint32
	keep     int
}

// kvBegin starts a mapping.
func (r *resident) kvBegin() *kvTx {
	r.kvReclaimed = r.kvReclaimed[:0]
	return &kvTx{r: r, saved: map[int]kvTxStanding{}, pending: map[int]kvTxStanding{}}
}

// standing is slot s's standing as the mapping leaves it so far.
func (tx *kvTx) standing(s int) kvTxStanding {
	if p, ok := tx.pending[s]; ok {
		return p
	}
	st := &tx.r.kvSlotSt[s]
	p := kvTxStanding{valid: st.valid, released: st.released, stopped: st.stopped, lost: st.lost}
	tx.saved[s], tx.pending[s] = p, p
	return p
}

// trim unmaps every entry of slot s wholly at or past end, and ends its sequence there.
func (tx *kvTx) trim(s, end int) {
	r := tx.r
	p := tx.standing(s)
	for gi, g := range r.kvGs {
		for b := kvBlocksFor(end); b < g.kvTabBlocks; b++ {
			e := g.kvRow(s, b)
			if *e == 0 {
				continue
			}
			id := *e
			g.kvUnref(id)
			tx.log = append(tx.log, kvTxEdit{g: gi, s: s, b: b, prev: id})
			*e = 0
		}
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
	for b := lo / decoder.KVBlockRows; b < kvBlocksFor(hi); b++ {
		for gi, g := range r.kvGs {
			if b >= g.kvTabBlocks {
				continue
			}
			old := *g.kvRow(s, b)
			if old != 0 && (g.kvRefs[old] <= 1 || kvOwnOffForTest) {
				continue
			}
			nb, err := r.kvTake(g, s, b, old != 0)
			if err != nil {
				return err
			}
			tx.log = append(tx.log, kvTxEdit{g: gi, s: s, b: b, prev: old, next: nb, take: true, cow: old != 0})
			*g.kvRow(s, b) = nb
			if old != 0 {
				// The entry's reference goes now, so a later row of the mapping finds the block as running the rows alone would: another slot's
				// alone, which that slot writes in place. A shared block keeps a reference, so its rows stay as they are until the copy at commit.
				g.kvUnref(old)
				keep := max(lo-b*decoder.KVBlockRows, 0)
				if kvKeepShortForTest && keep > 0 {
					keep--
				}
				tx.cows = append(tx.cows, kvTxCOW{g: gi, s: s, b: b, from: old, to: nb, keep: keep})
			}
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
		g := r.kvGs[e.g]
		if e.take && *g.kvRow(e.s, e.b) == e.next { // a release or reclaim meanwhile gave next back with the entry
			g.kvUnref(e.next)
		}
		if !e.take || e.cow { // a trim or a copy on write gave its reference to prev back: take it again
			if g.kvRefs[e.prev] == 0 {
				g.kvClaim(e.prev)
			} else {
				g.kvRefs[e.prev]++
			}
		}
		*g.kvRow(e.s, e.b) = e.prev
	}
	share := r.kvShareBlocks * decoder.KVBlockRows
	for s, p := range tx.saved {
		st := &r.kvSlotSt[s]
		switch lost := st.lost != p.lost; {
		case lost && st.released: // a take released it meanwhile, or took what it mapped after an earlier release: it holds nothing
			r.kvReleaseAll(s)
			st.valid = 0
		case lost: // a take stopped it meanwhile: nothing past its share
			st.valid = min(p.valid, share)
			r.kvTrim(s, share)
		default:
			st.valid, st.released, st.stopped = p.valid, p.released, p.stopped
		}
	}
}

// commit runs the copies on write and lets each slot's new standing stand. A copy that fails on the device undoes the mapping. A copy
// that ran, the failed one included, wrote a block that was free when taken, so an entry the undo maps to it again (a trim's, or a copy
// on write's old entry) gave it back during the mapping, whoever held it then: that block no longer holds the entry's rows, and the
// entry's slot ends before it.
func (tx *kvTx) commit() error {
	r := tx.r
	for s, p := range tx.saved {
		if r.kvSlotSt[s].lost != p.lost {
			tx.abort()
			return fmt.Errorf("metal: KV slot %d was released or reclaimed while its blocks were mapped: %w", s, decoder.ErrKVPoolFull)
		}
	}
	for i, c := range tx.cows {
		if c.keep == 0 {
			continue
		}
		err := gpu.CopyDeviceBatch(r.kvBlockCopies(r.kvGs[c.g], c.from, c.to, c.keep))
		if kvCopyFailForTest || kvCopyFailAfterForTest == 0 {
			err = errors.New("planted copy failure")
		} else if kvCopyFailAfterForTest > 0 {
			kvCopyFailAfterForTest--
		}
		if err != nil {
			tx.abort()
			for _, d := range tx.cows[:i+1] {
				for _, e := range tx.log {
					if d.keep > 0 && (!e.take || e.cow) && e.prev == d.to && e.g == d.g {
						r.kvEndAt(e.s, e.b*decoder.KVBlockRows)
					}
				}
			}
			return fmt.Errorf("metal: copy on write of KV block %d: %w", c.from, err)
		}
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
	r.kvReap(false)
	for _, q := range seqs {
		if q.slot < len(r.kvSlotSt) {
			if err := r.kvWindowReads(q.slot, q.pos); err != nil {
				return err
			}
		}
	}
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
	if err := tx.commit(); err != nil {
		return err
	}
	for _, q := range seqs {
		if q.slot < len(r.kvSlotSt) {
			r.kvRetain(q.slot, q.pos, q.pos+1)
		}
	}
	return nil
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
// holds fewer rows (ResidentKVSubsystem's contract: prefix or 0, never another positive value). A slot with a greedy chain open holds its
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
	for _, g := range r.kvGs { // a window group answers for the rows it still holds
		if g.win > 0 && !kvRetainOffForTest && !decoder.KVWindowReusable(g.kvBase[s]-kvReusableBasePlantForTest, prefix, g.win) {
			return 0
		}
	}
	return prefix
}

// kvReusableBasePlantForTest lowers the base ReusableKV reads by that many positions: the plant TestKVWindow_reusableKVIsTheWindowOracle
// must catch with 64, a block.
var kvReusableBasePlantForTest int

// ReusableKV implements decoder.ResidentKVSubsystem for the bound slot.
func (a *metalResident) ReusableKV(prefix int) int { return a.r.kvReusable(a.r.kvSlot, prefix) }

// ReusableKVSlot implements decoder.ResidentKVSubsystem.
func (a *metalResident) ReusableKVSlot(slot, prefix int) int { return a.r.kvReusable(slot, prefix) }

// SetKVSlotBusy implements decoder.ResidentKVSubsystem: a busy slot's blocks are never released to make room for another.
func (a *metalResident) SetKVSlotBusy(slot int, busy bool) {
	if slot >= 0 && slot < len(a.r.kvSlotSt) {
		a.r.kvSlotSt[slot].busy.Store(busy)
	}
}

// KVPool implements decoder.ResidentKVSubsystem: the positions every slot's blocks hold together, each slot's guaranteed share, and the
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
	bytes += r.kvTableBytes()
	groups := []string{"full"}
	if r.kvWin != nil {
		groups = append(groups, "window")
	}
	prec := "f16"
	switch {
	case r.kvI8:
		prec = "i8"
	case r.kvF32 || r.mla != nil:
		prec = "f32"
	}
	return decoder.KVPoolDetail{Positions: (len(r.kvRefs) - 1) * decoder.KVBlockRows, Guaranteed: r.kvShareCtx, BlockRows: decoder.KVBlockRows,
		Groups: groups, Precision: prec, Bytes: bytes}
}
