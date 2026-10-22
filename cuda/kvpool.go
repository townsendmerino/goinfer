//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strings"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// Paged KV storage (docs/tasks/task-kv-architecture-2026-10.md §5.3, KV4). Every layer that keeps K/V holds them in a pool of
// kvBlock-row blocks, [blocks][kvBlock][kvDim], one allocation per layer for K and one for V. A layer belongs to one group: the full
// group keeps every position (full attention, MLA latent rows), the window group keeps what a sliding window and the rollback need
// (kvwindow.go). A block id names the same block in every layer of its group. Block 0 of every pool is zero-filled and never handed
// out: an entry a sequence has not mapped, or has released, points at it.
//
// A resident KV slot (MC1) is a sequence's tables, one per group: the block id of each kvBlock positions of the sequence. They live on
// the host and are mirrored to one device buffer, which every KV kernel reads through kvpage.cuh's KV_ROW. A slot owns the blocks its
// tables name; nothing is shared between slots yet, and each slot can always map its whole context, because each pool is sized for
// every slot's largest need (kvPoolBlocks). All of it runs on the executor thread.

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
}

// kvPoolOrderForTest, when set, gives the order a pool of n blocks hands its blocks out in (ids 1..n-1, each once). G1a's scrambled
// tables (docs/measurements/kv4-cuda-2026-10-21.md) use it; nil hands out ascending ids.
var kvPoolOrderForTest func(n int) []int32

func newKVPool(blocks int) kvPool {
	p := kvPool{blocks: blocks, free: make([]int32, 0, max(0, blocks-1))}
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

func (p *kvPool) take() (int32, bool) {
	if len(p.free) == 0 {
		return 0, false
	}
	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	return b, true
}

func (p *kvPool) give(b int32) { p.free = append(p.free, b) }

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

// kvSlotBlocks is the most blocks one slot ever maps in group g: its whole context in the full group, and in the window group
// kvWindowBlocks.
func (r *cudaResident) kvSlotBlocks(g int) int {
	if g == kvGroupWin {
		if !r.kvWin {
			return 0
		}
		return kvWindowBlocks(r.ctxCap, r.kvWindow+r.kvSlack)
	}
	return r.kvMaxBlocks()
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
	r.kvPools = [kvGroups]kvPool{newKVPool(r.kvPoolBlocks(kvGroupFull, slots)), newKVPool(r.kvPoolBlocks(kvGroupWin, slots))}
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.noKV() || Ly.kvShared {
			continue
		}
		rows := r.kvPools[r.kvGroupOf(l)].blocks * kvBlock
		if Ly.isMLA() {
			r.kc[l] = r.af(rows * Ly.kvDim) // one latent row per position, no V
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
		stage, e := gpu.NewHostBuffer[uint8](r.dev, kvTabRing*r.kvMaxBlocks()*4)
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

// kvMap gives slot s's group-g table a block for every kvBlock positions overlapping [lo, hi). In the window group it first releases
// every block wholly outside [lo, hi), which holds the slot to kvSlotBlocks. An error means the pool is exhausted, which the sizing
// rules out; nothing is left half mapped that a kernel would read.
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
				r.kvPools[g].give(id)
				ids[b] = 0
				t.dirty[g] = true
			}
		}
	}
	for b := b0; b < b1; b++ {
		if ids[b] != 0 {
			continue
		}
		id, ok := r.kvPools[g].take()
		if !ok {
			return fmt.Errorf("cuda: KV pool (group %d, %d blocks) exhausted mapping slot %d's positions [%d, %d)", g, r.kvPools[g].blocks, s, lo, hi)
		}
		ids[b] = id
		t.dirty[g] = true
	}
	return nil
}

// kvReleaseAll returns every block of slot s's group g to its pool.
func (r *cudaResident) kvReleaseAll(s, g int) {
	t := &r.kvTabs[s]
	for b, id := range t.ids[g] {
		if id != 0 {
			r.kvPools[g].give(id)
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

// kvRowRuns calls fn for each run of consecutive positions [p, p+n) of [lo, hi) that one block of slot s's group g holds, with the
// physical row of p. Every block in the range must be mapped.
func (r *cudaResident) kvRowRuns(s, g, lo, hi int, fn func(p, n, row int) error) error {
	ids := r.kvTabs[s].ids[g]
	for p := lo; p < hi; {
		b := p / kvBlock
		id := ids[b]
		if id == 0 {
			return fmt.Errorf("cuda: slot %d holds no block for KV position %d", s, p)
		}
		n := min(hi, (b+1)*kvBlock) - p
		if e := fn(p, n, int(id)*kvBlock+p%kvBlock); e != nil {
			return e
		}
		p += n
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
			n += kvBlock * int64(Ly.kvDim) * 4
			continue
		}
		n += 2 * kvBlock * int64(Ly.kvDim) * elem
	}
	return n
}

// kvAllocPoolsFit is kvAllocPools for r.kvSlotsN slots, and on a device allocation failure for one fewer, down to one, which is
// returned as the failure. A panic other than an allocation failure is re-raised.
func (r *cudaResident) kvAllocPoolsFit() error {
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
			return nil
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
