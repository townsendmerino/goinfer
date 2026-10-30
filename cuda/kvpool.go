//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strings"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// Paged KV storage (docs/tasks/task-kv-architecture-2026-10.md §5.3, KV4). Every layer that keeps K/V holds them in a pool of
// kvBlock-row blocks, [blocks][kvBlock][kvDim], one allocation per layer for K and one for V. A layer belongs to one group: the full
// group keeps every position (full attention, MLA latent rows), the window group keeps what a sliding window and the rollback need
// (kvwindow.go). A block id names the same block in every layer of its group. Block 0 of every pool is zero-filled and never handed
// out: an entry a sequence has not mapped, or has released, points at it.
//
// A resident KV slot (MC1) is a sequence's tables, one per group: the block id of each kvBlock positions of the sequence. They live on
// the host and are mirrored to one device buffer, which every KV kernel reads through kvpage.cuh's KV_ROW. A block counts the tables
// that name it: a full-group block another slot's prefix shares (CopyKV, kvShare) has more than one, and a slot that is about to write
// into such a block first takes a block of its own and copies the rows it keeps (kvMap), so a slot only ever writes blocks no other
// slot reads. Each slot can always map its whole context, because each pool is sized for every slot's largest need (kvPoolBlocks) and
// sharing only ever frees blocks. All of it runs on the executor thread.

// kvBlock is the rows of one block: decoder.KVBlockRows, which the kernels' KV_ROW hard-codes as 64.
const kvBlock = decoder.KVBlockRows

// The groups a layer's K/V can belong to.
const (
	kvGroupFull = iota // every position is kept
	kvGroupWin         // windowed layers: [base, end), kvwindow.go
	kvGroups
)

// kvPool is one group's block allocator. Block 0, the null block, is never free.
type kvPool struct {
	blocks int     // blocks in the pool, the null block included
	free   []int32 // free block ids; take pops the last
	refs   []int32 // the tables that name each block
}

// kvPoolOrderForTest, when set, gives the order a pool of n blocks hands its blocks out in (ids 1..n-1, each once). G1a's scrambled
// tables (docs/measurements/kv4-cuda-2026-10-21.md) use it; nil hands out ascending ids.
var kvPoolOrderForTest func(n int) []int32

func newKVPool(blocks int) kvPool {
	p := kvPool{blocks: blocks, free: make([]int32, 0, max(0, blocks-1)), refs: make([]int32, max(0, blocks))}
	if kvPoolOrderForTest != nil && blocks > 1 {
		order := kvPoolOrderForTest(blocks)
		for i := len(order) - 1; i >= 0; i-- {
			p.free = append(p.free, order[i])
		}
		return p
	}
	for b := blocks - 1; b >= 1; b-- {
		p.free = append(p.free, int32(b))
	}
	return p
}

// take hands out a free block with one reference.
func (p *kvPool) take() (int32, bool) {
	if len(p.free) == 0 {
		return 0, false
	}
	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.refs[b] = 1
	return b, true
}

func (p *kvPool) ref(b int32) { p.refs[b]++ }

// unref drops one reference to b; the last one frees it.
func (p *kvPool) unref(b int32) {
	if p.refs[b]--; p.refs[b] <= 0 {
		p.refs[b] = 0
		p.free = append(p.free, b)
	}
}

// inUse is how many blocks some table names, the null block not counted.
func (p *kvPool) inUse() int { return p.blocks - 1 - len(p.free) }

// kvSlotTabs is one slot's tables: ids[g][i] is the block holding positions [i*kvBlock, (i+1)*kvBlock) in group g, 0 for none.
type kvSlotTabs struct {
	ids   [kvGroups][]int32
	dirty [kvGroups]bool // changed since the device copy was written
}

// kvMaxBlocks is the table length: the blocks of a ctxCap-position sequence.
func (r *cudaResident) kvMaxBlocks() int { return kvBlocksFor(r.ctxCap) }

func kvBlocksFor(positions int) int { return (positions + kvBlock - 1) / kvBlock }

// kvGroupOf is layer l's group (meaningless for a layer with no K/V). A KV-shared layer takes its source's, which its kvWin copies.
func (r *cudaResident) kvGroupOf(l int) int {
	if r.layers[l].kvWin {
		return kvGroupWin
	}
	return kvGroupFull
}

// kvSlotBlocks is what group g's pool holds for each slot: in the full group the slot's guaranteed share, the blocks of the context the
// plan chose for one slot (one conversation may borrow past it, kvTake); in the window group kvWindowBlocks, the most a slot ever maps.
func (r *cudaResident) kvSlotBlocks(g int) int {
	if g == kvGroupWin {
		if !r.kvWin {
			return 0
		}
		return kvWindowBlocks(r.ctxCap, r.kvWindow+r.kvSlack)
	}
	return r.kvShareBlocks
}

// kvCapFor is one conversation's position cap with slots slots: the planned per-slot context, or, where kvRaise allows it, everything
// the full pool holds, up to the model's own window.
func (r *cudaResident) kvCapFor(slots int) int {
	if !r.kvRaise || slots < 2 {
		return r.kvShareCtx
	}
	c := slots * r.kvShareBlocks * kvBlock
	if r.modelCtx > 0 {
		c = min(c, r.modelCtx)
	}
	return max(c, r.kvShareCtx)
}

// kvHeld is how many full-group blocks slot s's table names, shared ones included.
func (r *cudaResident) kvHeld(s int) int {
	n := 0
	for _, id := range r.kvTabs[s].ids[kvGroupFull] {
		if id != 0 {
			n++
		}
	}
	return n
}

// kvWithinShare reports whether slot s is within its guaranteed share of the full group for one more block: below it, or at it for a copy
// on write (which holds no more blocks after it).
func (r *cudaResident) kvWithinShare(s int, cow bool) bool {
	held := r.kvHeld(s)
	return held < r.kvShareBlocks || (cow && held <= r.kvShareBlocks)
}

// kvTake hands slot s a block of group g. When none is free it releases idle slots' blocks, least recently used first (D5's step 1); a
// slot within its share then takes back what a running borrower holds past its share, which stops that borrower (kvReclaim). A slot past
// its share that still finds no block gets decoder.ErrKVPoolFull: only a borrower is ever stopped (D-KV4-2), and it ends as at the context
// cap. The window group's pool holds every slot's most, so it never runs short.
func (r *cudaResident) kvTake(s, g int, cow bool) (int32, error) {
	for {
		if id, ok := r.kvPools[g].take(); ok {
			return id, nil
		}
		if r.kvEvictIdle(s) {
			continue
		}
		if g == kvGroupFull && r.kvWithinShare(s, cow) && r.kvReclaim(s) {
			continue
		}
		return 0, fmt.Errorf("cuda: KV pool: slot %d holds %d blocks, its share is %d, and no block is free for it (%d slots, %d blocks): %w",
			s, r.kvHeld(s), r.kvShareBlocks, len(r.kvTabs), r.kvPools[g].blocks-1, decoder.ErrKVPoolFull)
	}
}

// kvReclaim stops the running slot other than s that holds the most blocks past its share: its blocks past the share go back to the pool,
// it is marked stopped, so its next write returns decoder.ErrKVPoolFull, and a batched step in progress is told to abort before it
// launches anything (kvStepAbort), since a row of it may be that borrower. False when no other slot holds more than its share.
func (r *cudaResident) kvReclaim(s int) bool {
	best, most := -1, 0
	for t := range r.kvTabs {
		if t == s || r.kvStopped[t] {
			continue
		}
		if ex := r.kvHeld(t) - r.kvShareBlocks; ex > most {
			best, most = t, ex
		}
	}
	if best < 0 {
		return false
	}
	ids := r.kvTabs[best].ids[kvGroupFull]
	for b := r.kvShareBlocks; b < len(ids); b++ {
		if ids[b] != 0 {
			r.kvPools[kvGroupFull].unref(ids[b])
			ids[b] = 0
		}
	}
	r.kvTabs[best].dirty[kvGroupFull] = true
	r.kvStopped[best], r.kvStepAbort = true, true
	fmt.Fprintf(os.Stderr, "cuda: KV pool: stopped slot %d, %d blocks past its share of %d, for slot %d within its share\n", best, most, r.kvShareBlocks, s)
	return true
}

// kvEvictIdle releases every block of the least recently written slot that is not s, not bound, not held by a running generation
// (kvSlotBusy) and holds any, and marks it released, so it answers 0 to every reuse question until it starts a new sequence. False
// when there is none.
func (r *cudaResident) kvEvictIdle(s int) bool {
	best := -1
	for t := range r.kvTabs {
		if t == s || t == r.kvSlot || r.kvSlotBusy[t] || (r.kvHeld(t) == 0 && !r.kvHoldsWindow(t)) {
			continue
		}
		if best < 0 || r.kvSlotUse[t] < r.kvSlotUse[best] {
			best = t
		}
	}
	if best < 0 {
		return false
	}
	n := r.kvHeld(best)
	r.kvReleaseAll(best, kvGroupFull)
	r.kvReleaseAll(best, kvGroupWin)
	r.kvEvicted[best] = true
	fmt.Fprintf(os.Stderr, "cuda: KV pool: released slot %d's %d blocks (idle, the least recently used) for slot %d, which is past its share of %d\n",
		best, n, s, r.kvShareBlocks)
	return true
}

// kvHoldsWindow reports whether slot s maps any window-group block.
func (r *cudaResident) kvHoldsWindow(s int) bool {
	for _, id := range r.kvTabs[s].ids[kvGroupWin] {
		if id != 0 {
			return true
		}
	}
	return false
}

// SetKVSlotBusy implements decoder.ResidentKVSlotActivity.
func (r *cudaResident) SetKVSlotBusy(slot int, busy bool) {
	_ = r.do(func() error {
		if slot >= 0 && slot < len(r.kvSlotBusy) {
			r.kvSlotBusy[slot] = busy
		}
		return nil
	})
}

// kvReleased reports whether slot s's blocks were released to make room (kvEvictIdle) since it last started a sequence.
func (r *cudaResident) kvReleased(s int) bool {
	return s >= 0 && s < len(r.kvEvicted) && r.kvEvicted[s]
}

// kvWindowBlocks is the most blocks a slot's window group maps at context ctxCap when a windowed layer holds held positions: the held
// positions plus one block, since a held range need not start on a block edge, unless they are the whole context.
// decoder.cudaKVBytes prices the same.
func kvWindowBlocks(ctxCap, held int) int {
	if ctxCap <= held {
		return kvBlocksFor(ctxCap)
	}
	return kvBlocksFor(held) + 1
}

// kvPoolBlocks is group g's pool: every slot's largest need, plus the null block.
func (r *cudaResident) kvPoolBlocks(g, slots int) int { return slots*r.kvSlotBlocks(g) + 1 }

// kvT is layer l's table in the slot whose K/V the kernels are addressing (kvTabSlot: the bound slot, or an MC3 row's).
func (r *cudaResident) kvT(l int) Buffer { return r.kvTabBuf(r.kvTabSlot, r.kvGroupOf(l)) }

// kvTabBuf is slot s's group-g table on the device.
func (r *cudaResident) kvTabBuf(s, g int) Buffer {
	return r.kvTabDev.At((s*kvGroups + g) * r.kvMaxBlocks() * 4)
}

// kvAllocPools allocates every owning layer's pools and every slot's tables for slots slots, and zeroes the null blocks. It replaces the
// per-slot buffers MC1 used to allocate; the caller has run checkKVFits and kvWindowSetup.
func (r *cudaResident) kvAllocPools(slots int) error {
	defer r.onLine(mem.LineKV)()
	r.kvPools = [kvGroups]kvPool{newKVPool(r.kvPoolBlocks(kvGroupFull, slots)), newKVPool(r.kvPoolBlocks(kvGroupWin, slots))}
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.noKV() || Ly.kvShared {
			continue
		}
		rows := r.kvPools[r.kvGroupOf(l)].blocks * kvBlock
		if Ly.isMLA() {
			r.kc[l] = r.akv(l, rows*Ly.kvDim) // one latent row per position, no V
			continue
		}
		r.kc[l], r.vc[l] = r.akv(l, rows*Ly.kvDim), r.akv(l, rows*Ly.kvDim)
	}
	for l := range r.layers {
		if r.layers[l].kvShared {
			r.kc[l], r.vc[l] = r.kc[r.layers[l].kvSrc], r.vc[r.layers[l].kvSrc]
		}
	}
	r.kvTabs = make([]kvSlotTabs, slots)
	for s := range r.kvTabs {
		for g := range kvGroups {
			r.kvTabs[s].ids[g] = make([]int32, r.kvMaxBlocks())
		}
	}
	r.kvTabDev = r.ai(slots * kvGroups * r.kvMaxBlocks())
	if r.kvTabStage == nil {
		stage, e := gpu.NewHostBufferOn[uint8](r.dev, mem.LineKV, kvTabRing*r.kvMaxBlocks()*4)
		if e != nil {
			return e
		}
		r.kvTabStage = stage
		r.kvTabEv = make([]gpu.Event, kvTabRing)
		for k := range r.kvTabEv {
			if r.kvTabEv[k], e = r.dev.NewEvent(); e != nil {
				return e
			}
		}
	}
	if e := r.stream.ZeroAsync(r.kvTabDev, slots*kvGroups*r.kvMaxBlocks()*4); e != nil {
		return e
	}
	if e := r.kvZeroNullBlocks(); e != nil {
		return e
	}
	return r.stream.Sync()
}

// kvZeroNullBlocks writes zeros over block 0 of every owning layer's buffers.
func (r *cudaResident) kvZeroNullBlocks() error {
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.noKV() || Ly.kvShared {
			continue
		}
		n := kvBlock * Ly.kvDim * r.kvElem(l)
		for _, b := range []Buffer{r.kc[l], r.vc[l]} {
			if b.Len() == 0 {
				continue
			}
			if e := r.stream.ZeroAsync(b, n); e != nil {
				return e
			}
		}
	}
	return nil
}

// kvElem is the bytes of one cached element of layer l: 4, or 2 for an f16 layer. An MLA latent row is f32.
func (r *cudaResident) kvElem(l int) int {
	if r.layers[l].kvHalf {
		return 2
	}
	return 4
}

// kvMap gives slot s's group-g table a block of its own for every kvBlock positions overlapping [lo, hi), which the caller is about to
// write. In the window group it first releases every block wholly outside [lo, hi), which holds the slot to kvSlotBlocks. A block in the
// range that another table also names is replaced by a new one holding a copy of its rows below lo (copy on write); rows at or past lo are
// the write's. An error means the pool is exhausted, which the sizing rules out; nothing is left half mapped that a kernel would read.
func (r *cudaResident) kvMap(s, g, lo, hi int) error {
	if hi <= lo {
		return nil
	}
	t := &r.kvTabs[s]
	ids := t.ids[g]
	b0, b1 := lo/kvBlock, kvBlocksFor(hi)
	if b1 > len(ids) {
		return fmt.Errorf("cuda: KV positions [%d, %d) are past the %d-position table", lo, hi, len(ids)*kvBlock)
	}
	if g == kvGroupWin {
		for b, id := range ids {
			if id != 0 && (b < b0 || b >= b1) {
				r.kvPools[g].unref(id)
				ids[b] = 0
				t.dirty[g] = true
			}
		}
	}
	for b := b0; b < b1; b++ {
		old := ids[b]
		if old != 0 && r.kvPools[g].refs[old] <= 1 {
			continue
		}
		id, e := r.kvTake(s, g, old != 0)
		if e != nil {
			return e
		}
		if old != 0 {
			if keep := lo - b*kvBlock; keep > 0 {
				if e := r.kvCopyBlockRows(g, old, id, keep); e != nil {
					r.kvPools[g].unref(id)
					return e
				}
			}
			r.kvPools[g].unref(old)
		}
		ids[b] = id
		t.dirty[g] = true
	}
	return nil
}

// kvCopyBlockRows copies the first rows rows of block from into block to, in every layer of group g that owns its K/V. Stream-ordered
// kv_store launches: exact, no arithmetic.
func (r *cudaResident) kvCopyBlockRows(g int, from, to int32, rows int) error {
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.noKV() || Ly.kvShared || r.kvGroupOf(l) != g {
			continue
		}
		row := Ly.kvDim * r.kvElem(l) // bytes
		for _, b := range []Buffer{r.kc[l], r.vc[l]} {
			if b.Len() == 0 {
				continue // an MLA latent layer has no V
			}
			if e := r.copyF32(b.At(int(from)*kvBlock*row), b.At(int(to)*kvBlock*row), rows*row/4); e != nil {
				return e
			}
		}
	}
	return nil
}

// kvShare makes slot dst's full-group table name slot src's blocks for positions [start, end), whole blocks, each taking a reference,
// in place of whatever dst named there. Nothing is copied: the rows are stored once. A block src never mapped gives dst a block of its
// own holding the null block's zeros, never a reference to the null block, which dst would then write.
func (r *cudaResident) kvShare(src, dst, start, end int) error {
	sIDs, dIDs := r.kvTabs[src].ids[kvGroupFull], r.kvTabs[dst].ids[kvGroupFull]
	p := &r.kvPools[kvGroupFull]
	for b := start / kvBlock; b < end/kvBlock; b++ {
		id := sIDs[b]
		if id == 0 {
			if e := r.kvMap(dst, kvGroupFull, b*kvBlock, (b+1)*kvBlock); e != nil {
				return e
			}
			if e := r.kvCopyBlockRows(kvGroupFull, 0, dIDs[b], kvBlock); e != nil {
				return e
			}
			continue
		}
		if dIDs[b] == id {
			continue
		}
		if dIDs[b] != 0 {
			p.unref(dIDs[b])
		}
		p.ref(id)
		dIDs[b] = id
		r.kvTabs[dst].dirty[kvGroupFull] = true
	}
	return nil
}

// kvReleaseAll returns every block of slot s's group g to its pool.
func (r *cudaResident) kvReleaseAll(s, g int) {
	t := &r.kvTabs[s]
	for b, id := range t.ids[g] {
		if id != 0 {
			r.kvPools[g].unref(id)
			t.ids[g][b] = 0
			t.dirty[g] = true
		}
	}
}

// kvTabRing is how many staged table copies can be in flight on the stream before kvFlushTabs waits for the oldest.
const kvTabRing = 8

// kvFlushTabs writes slot s's changed tables to the device: each through the next pinned staging entry, as a copy on the KV kernels' own
// stream, so every launch queued before it reads the old table and every launch after it the new one, and the host never waits for the
// device. The staging entry is reused only once its previous copy has executed (its event).
func (r *cudaResident) kvFlushTabs(s int) error {
	t := &r.kvTabs[s]
	n := r.kvMaxBlocks() * 4 // bytes of one table
	for g := range kvGroups {
		if !t.dirty[g] {
			continue
		}
		k := r.kvTabNext
		r.kvTabNext = (k + 1) % kvTabRing
		if e := r.kvTabEv[k].Sync(); e != nil {
			return e
		}
		stage := r.kvTabStage.Slice()[k*n : (k+1)*n]
		for i, id := range t.ids[g] {
			stage[4*i], stage[4*i+1], stage[4*i+2], stage[4*i+3] = byte(id), byte(id>>8), byte(id>>16), byte(id>>24)
		}
		if e := r.stream.UploadAsyncAt(r.kvTabBuf(s, g), r.kvTabStage, k*n, n); e != nil {
			return e
		}
		if e := r.stream.Record(r.kvTabEv[k]); e != nil {
			return e
		}
		t.dirty[g] = false
	}
	return nil
}

// kvRowRuns calls fn for each run of consecutive positions [p, p+n) of [lo, hi) whose rows are consecutive in the pool, with the
// physical row of p: blocks of slot s's group g that follow each other in the pool too are one run, so a slot whose blocks were handed out
// in order is copied in one piece. Every block in the range must be mapped.
func (r *cudaResident) kvRowRuns(s, g, lo, hi int, fn func(p, n, row int) error) error {
	ids := r.kvTabs[s].ids[g]
	for p := lo; p < hi; {
		b := p / kvBlock
		if ids[b] == 0 {
			return fmt.Errorf("cuda: slot %d holds no block for KV position %d", s, p)
		}
		row := int(ids[b])*kvBlock + p%kvBlock
		end := min(hi, (b+1)*kvBlock)
		for end < hi && ids[end/kvBlock] == ids[end/kvBlock-1]+1 {
			end = min(hi, end+kvBlock)
		}
		if e := fn(p, end-p, row); e != nil {
			return e
		}
		p = end
	}
	return nil
}

// kvMapWrite maps what a write of n rows at p0 into slot s needs in every group, given the window group's base after the write (kvwindow.go),
// and writes the changed tables to the device.
func (r *cudaResident) kvMapWrite(s, p0, n, winBase int) error {
	if e := r.kvMap(s, kvGroupFull, p0, p0+n); e != nil {
		return e
	}
	if r.kvWin {
		if e := r.kvMap(s, kvGroupWin, winBase, p0+n); e != nil {
			return e
		}
	}
	return r.kvFlushTabs(s)
}

// kvPoolNullBytes is the bytes of the pools' null blocks: one block of every owning layer.
func kvPoolNullBytes(layers []cudaLayer) int64 {
	var n int64
	for i := range layers {
		Ly := &layers[i]
		if Ly.kvShared || Ly.kvDim == 0 || Ly.noKV() {
			continue
		}
		elem := int64(4)
		if Ly.kvHalf {
			elem = 2
		}
		if Ly.isMLA() {
			n += kvBlock * int64(Ly.kvDim) * elem
			continue
		}
		n += 2 * kvBlock * int64(Ly.kvDim) * elem
	}
	return n
}

// kvAllocPoolsFit is kvAllocPools for r.kvSlotsN slots, and on a device allocation failure for one fewer, down to one, which is
// returned as the failure. A panic other than an allocation failure is re-raised.
func (r *cudaResident) kvAllocPoolsFit() error {
	r.kvShareCtx, r.kvShareBlocks = r.ctxCap, kvBlocksFor(r.ctxCap)
	want := max(1, r.kvSlotsN)
	for n := want; n >= 1; n-- {
		ok, err := r.kvTryAllocPools(n)
		if err != nil {
			return err
		}
		if ok {
			if n < want {
				fmt.Fprintf(os.Stderr, "cuda: %d of %d resident KV slots granted: the device ran out of memory allocating the KV pools beside the build's own scratch\n", n, want)
			}
			r.kvSlotsN = n
			return r.kvAfterRaise()
		}
	}
	return fmt.Errorf("cuda: the device ran out of memory allocating one KV slot's pools")
}

// kvTryAllocPools is one kvAllocPools attempt; ok=false, with everything it allocated released, on a device allocation failure.
func (r *cudaResident) kvTryAllocPools(slots int) (ok bool, err error) {
	defer func() {
		if v := recover(); v != nil {
			if !strings.Contains(fmt.Sprint(v), "device allocation failed") {
				panic(v)
			}
			r.kvReleasePools()
			ok, err = false, nil
		}
	}()
	if kvSlotAllocFailAtForTest > 0 && slots > kvSlotAllocFailAtForTest {
		panic("cuda: device allocation failed (test seam): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")
	}
	if r.kvWin {
		r.kvBases = make([]int, slots)
	}
	r.kvSlotBusy, r.kvEvicted, r.kvStopped, r.kvSlotUse = make([]bool, slots), make([]bool, slots), make([]bool, slots), make([]uint64, slots)
	r.ctxCap = r.kvCapFor(slots)
	return true, r.kvAllocPools(slots)
}

// kvReleasePools releases every pool and table buffer kvAllocPools made, leaving none bound.
func (r *cudaResident) kvReleasePools() {
	for l := range r.layers {
		if r.layers[l].kvShared {
			r.kc[l], r.vc[l] = Buffer{}, Buffer{}
		}
	}
	for _, set := range [][]Buffer{r.kc, r.vc} {
		for l := range set {
			if set[l].Len() > 0 {
				r.dev.ReleaseBuf(set[l])
				set[l] = Buffer{}
			}
		}
	}
	if r.kvTabDev.Len() > 0 {
		r.dev.ReleaseBuf(r.kvTabDev)
		r.kvTabDev = Buffer{}
	}
	r.kvTabs = nil
}

// kvAfterRaise follows a per-conversation cap past the planned per-slot context: the split-KV score scratch, sized by the cap, grows with
// it, and the load says what the pool holds (the task doc's §5.16 banner).
func (r *cudaResident) kvAfterRaise() error {
	defer r.onLine(mem.LineKV)()
	if r.ctxCap == r.kvShareCtx {
		return nil
	}
	if r.skScoreBuf.Len() > 0 {
		r.dev.ReleaseBuf(r.skScoreBuf)
		r.skScoreBuf = r.af(r.nH * r.ctxCap)
	}
	fmt.Fprint(os.Stderr, r.kvPoolLine())
	return nil
}

// kvPoolLine is the line the build logs when one conversation's cap is raised past its guaranteed share.
func (r *cudaResident) kvPoolLine() string {
	return fmt.Sprintf("cuda: KV pool: %d positions (%d blocks of %d) for %d conversations; one conversation can use up to %d positions, and each is guaranteed %d\n",
		r.kvSlotsN*r.kvShareBlocks*kvBlock, r.kvSlotsN*r.kvShareBlocks, kvBlock, r.kvSlotsN, r.ctxCap, r.kvShareCtx)
}

// KVPool implements decoder.ResidentKVPoolInfo: the full group's positions across every slot, and each slot's guaranteed share.
func (r *cudaResident) KVPool() (positions, guaranteed int) {
	return r.KVSlots() * r.kvShareBlocks * kvBlock, r.kvShareCtx
}
