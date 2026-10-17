//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// Windowed KV (docs/tasks/task-cuda-windowed-kv-2026-10.md). A sliding-window layer reads only the last `window` positions, so with
// Options.ResidentWindowedKV its K and V buffers hold window+slack positions instead of the whole context. Every kernel still addresses
// K and V by ABSOLUTE position (kc[pos*kvDim + ...]); a windowed layer is handed a view shifted back by base*kvDim*4 bytes, where base is
// the absolute position stored at physical slot 0, so no kernel changes: an address the kernel computes for any position in
// [base, base+cap) lands inside the buffer, and no position outside that range is read (the window floor never drops below base: kvEnsure
// proves it) or written. When the next write would pass base+cap the live tail moves to physical slot 0 (kvCompact) and base advances by
// only as much as keeps the slot's rollback behind the end of the write (kvWindowNextBase), so a compaction comes every kvWindowStep
// positions once the window is full.
//
// The rollback is how far before the newest position a follow-up may diverge and still reuse its prefix (ReusableKV): slack-step positions,
// 448 at the planned 512 slack, at every point of the cycle. kvGrantRollback raises the slack, and so the rollback, from VRAM left over once
// the build is done. Why and what each costs: docs/measurements/prefix-reuse-windowed-kv-2026-10-14.md.
//
// All of it runs on the executor thread. State per slot: kvBases[slot] (kvBase caches the bound slot's).

// kvWindowSlack is the headroom a windowed layer is planned with beyond its window: decoder.WindowedKVSlack, the figure decoder's pricing
// uses, so priced bytes equal what the build allocates before kvGrantRollback. Tests lower it to force compaction on a tiny fixture.
var kvWindowSlack = decoder.WindowedKVSlack

// kvWindowStep is how many positions a full window writes between compactions; the rollback is the slack less this. Each compaction copies
// window-1+rollback positions of every windowed layer, so a smaller step costs more copying per decoded token. 1 <= step < slack.
var kvWindowStep = 64

// kvWindowRollbackMax caps the rollback kvGrantRollback can raise a load to.
var kvWindowRollbackMax = 1024

// kvWindowSetup decides, once the layers are built and before the caches are sized, whether windowed KV engages, and marks the layers
// that use it. It engages only when the option is on, some layer is a windowed attention layer, and the context is longer than a
// window plus its slack (below that a layer would hold the whole context anyway). Every windowed layer shares one window
// (SlidingWindowResident), so one base per slot serves them all.
func (r *cudaResident) kvWindowSetup(m *decoder.Model) {
	r.kvWin, r.kvWindow = false, 0
	if !m.ResidentWindowedKV() {
		return
	}
	window := m.SlidingWindowResident()
	if window <= 0 || r.ctxCap <= window+kvWindowSlack {
		return
	}
	r.kvSlack = kvWindowSlack
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.isMLA() || Ly.isDeltaNet() || Ly.window <= 0 || int(Ly.window) != window || !m.WindowedKVLayer(l) {
			continue
		}
		Ly.kvWin = true
		r.kvWin, r.kvWindow = true, window
	}
	for l := range r.layers {
		if Ly := &r.layers[l]; Ly.kvShared {
			Ly.kvWin = r.layers[Ly.kvSrc].kvWin // aliases its source's buffers, so it reads them through the same view
		}
	}
	if r.kvWin {
		fmt.Fprintf(os.Stderr, "cuda: windowed KV: sliding-window layers hold %d positions (window %d + %d slack; a follow-up may diverge %d before the newest and reuse its prefix) instead of the context's %d\n",
			r.kvWindow+r.kvSlack, r.kvWindow, r.kvSlack, r.kvRollback(), r.ctxCap)
	}
}

// kvRollback is how many positions before the newest a follow-up may diverge and still find its window held.
func (r *cudaResident) kvRollback() int { return r.kvSlack - kvWindowStep }

// kvWidestWindowed is the widest kvDim of any windowed layer.
func (r *cudaResident) kvWidestWindowed() int {
	widest := 0
	for l := range r.layers {
		if r.layers[l].kvWin {
			widest = max(widest, r.layers[l].kvDim)
		}
	}
	return widest
}

// kvAllocScratch sizes the compaction staging buffer to the widest live tail a compaction moves (window-1 plus the rollback, in positions) at
// the widest windowed layer, and starts every slot at base 0.
func (r *cudaResident) kvAllocScratch() {
	r.kvScratch = r.af(max(1, r.kvWindow-1+r.kvRollback()) * r.kvWidestWindowed())
	r.kvBases = make([]int, max(1, r.kvSlotsN))
}

// kvPositions is how many positions layer l's K/V buffers hold.
func (r *cudaResident) kvPositions(l int) int {
	if r.layers[l].kvWin {
		return min(r.ctxCap, int(r.layers[l].window)+r.kvSlack)
	}
	return r.ctxCap
}

// kvWindowGrant is how many positions to add to the windowed slack: the VRAM left over (free less reserve) divided by what one more
// position costs (perPos, across every slot's windowed layers and the compaction scratch), rounded down to a multiple of step and capped at
// maxExtra. 0 when nothing is left over.
func kvWindowGrant(free, reserve, perPos int64, step, maxExtra int) int {
	if perPos <= 0 || maxExtra < step || free <= reserve {
		return 0
	}
	return min(int((free-reserve)/perPos), maxExtra) / step * step
}

// kvGrantRollback raises the windowed slack, and with it the rollback, from VRAM nothing else will use. The build calls it last, once the
// context, the KV slots and the expert cache are fixed, so it never costs any of them, and an expert-cache load gets nothing: its cache
// was sized to what was free. What is held back first: the companion and vision-tower reserve (extraBytes), the decode margin, and two
// default prefill passes' scratch, which prefillCore allocates per request. The windowed buffers are reallocated (nothing is written yet); a
// failed allocation keeps the planned ones.
func (r *cudaResident) kvGrantRollback() {
	if !r.kvWin || r.cacheExperts {
		return
	}
	free, err := kvGrantFreeVRAM(r)
	if err != nil {
		return
	}
	slots := max(1, r.kvSlotsN)
	var perPos int64
	for l := range r.layers {
		if Ly := &r.layers[l]; Ly.kvWin && !Ly.kvShared {
			elem := int64(4)
			if Ly.kvHalf {
				elem = 2
			}
			perPos += int64(slots) * 2 * int64(Ly.kvDim) * elem
		}
	}
	perPos += int64(r.kvWidestWindowed()) * 4
	reserve := r.extraBytes + ctxCapMarginBytes + 2*r.prefillPassBytes(prefillDefaultChunk)
	maxExtra := min(kvWindowRollbackMax-r.kvRollback(), r.ctxCap-1-r.kvWindow-r.kvSlack)
	extra := kvWindowGrant(int64(free), reserve, perPos, kvWindowStep, maxExtra)
	if extra == 0 || !r.kvReallocWindowed(r.kvSlack+extra) {
		return
	}
	r.kvSlack += extra
	r.dev.ReleaseBuf(r.kvScratch)
	r.kvAllocScratch()
	fmt.Fprintf(os.Stderr, "cuda: windowed KV: %d more positions from free VRAM (%.0f MB of %.0f MB free); a follow-up may now diverge %d before the newest\n",
		extra, float64(int64(extra)*perPos)/(1<<20), float64(free)/(1<<20), r.kvRollback())
}

// kvGrantFreeVRAM is kvGrantRollback's free-VRAM probe, a variable so a test can stand in a card with no headroom.
var kvGrantFreeVRAM = func(r *cudaResident) (uint64, error) { return cudaFreeVRAM(r) }

// kvReallocWindowed gives every slot's windowed layers buffers of window+slack positions, all or none: on an allocation failure the new
// buffers are released and the old ones kept. KV-shared layers re-alias their sources.
func (r *cudaResident) kvReallocWindowed(slack int) (ok bool) {
	sets := [][]Buffer{r.kc, r.vc}
	for s := 1; s < len(r.kvSlotBufs); s++ {
		sets = append(sets, r.kvSlotBufs[s].kc, r.kvSlotBufs[s].vc)
	}
	type swap struct {
		set []Buffer
		l   int
		b   Buffer
	}
	var made []swap
	defer func() {
		if v := recover(); v != nil {
			if !strings.Contains(fmt.Sprint(v), "device allocation failed") {
				panic(v)
			}
			for _, m := range made {
				r.dev.ReleaseBuf(m.b)
			}
			ok = false
		}
	}()
	for _, set := range sets {
		for l := range r.layers {
			if Ly := &r.layers[l]; Ly.kvWin && !Ly.kvShared {
				made = append(made, swap{set, l, r.akv(l, min(r.ctxCap, int(Ly.window)+slack)*Ly.kvDim)})
			}
		}
	}
	for _, m := range made {
		r.dev.ReleaseBuf(m.set[m.l])
		m.set[m.l] = m.b
	}
	for l := range r.layers {
		if Ly := &r.layers[l]; Ly.kvShared && Ly.kvWin {
			for _, set := range sets {
				set[l] = set[Ly.kvSrc]
			}
		}
	}
	return true
}

// prefillPassBytes is the device scratch an M-row prefillCore pass allocates for its M-sized activation buffers. It is the dominant part,
// not all of it (attention and MoE scratch are not counted), which is why kvGrantRollback holds back two passes.
func (r *cudaResident) prefillPassBytes(M int) int64 {
	q, kv := r.prefillMaxGeom()
	h, in := r.hidden, r.inter
	for l := range r.layers {
		in = max(in, r.layers[l].ffnI)
	}
	floats := int64(M) * int64(2*h+2*q+2*kv+3*in+5)
	int32s := int64(M) * int64(2*h+q+in) / 4
	return 4 * (floats + int32s)
}

// kview is b (layer l's K or V buffer) as a kernel must address it: a windowed layer's view shifted back by the bound slot's base, so that
// kc[pos*kvDim] means the buffer's physical slot pos-base.
func (r *cudaResident) kview(b Buffer, l int) Buffer {
	if !r.layers[l].kvWin || r.kvBase == 0 {
		return b
	}
	return b.At(-r.kvBase * r.layers[l].kvDim * 4)
}

// kvK / kvV are the K and V buffers of layer l as a kernel must address them (see kview).
func (r *cudaResident) kvK(l int) Buffer { return r.kview(r.kc[l], l) }
func (r *cudaResident) kvV(l int) Buffer { return r.kview(r.vc[l], l) }

// kvEnsure makes room to write n rows starting at absolute position p0 in the bound slot, and proves the history the rows attend to is
// still held. A fresh sequence (p0 == 0) restarts the window. It returns an error rather than ever reading a position that was
// compacted away: such a call is a caller bug (a reuse past ReusableKV, or a rollback further than the slack), and a wrong answer must
// not be the symptom. Executor thread only.
func (r *cudaResident) kvEnsure(p0, n int) error {
	if !r.kvWin {
		return nil
	}
	return r.kvEnsureSlot(r.kvSlot, p0, n)
}

// kvEnsureSlot is kvEnsure for slot s, whether or not it is the bound one (an MC3 step prepares every row's slot before its layer loops).
func (r *cudaResident) kvEnsureSlot(s, p0, n int) error {
	if !r.kvWin {
		return nil
	}
	if len(r.kvBases) == 0 {
		r.kvBases = make([]int, max(1, len(r.kvSlotBufs)))
	}
	base, newBase, err := kvWindowNextBase(r.kvBases[s], p0, n, r.kvWindow, min(r.ctxCap, r.kvWindow+r.kvSlack), r.kvRollback())
	if err != nil {
		return err
	}
	if newBase != base {
		if err := r.kvCompactSlot(s, base, newBase, p0); err != nil {
			return err
		}
		base = newBase
	}
	r.kvBases[s] = base
	if s == r.kvSlot {
		r.kvBase = base
	}
	return nil
}

// kvWindowNextBase is kvEnsureSlot's arithmetic, kept free of device state so it can be tested on recorded traffic. Given the slot's base, it returns
// the base the write of n rows at p0 starts from (0 for a fresh sequence) and the base after it; they differ when the write needs a compaction.
// An error means the write reads history below the base or does not fit capN positions.
//
// A compaction moves the base to rollback+window-1 positions behind the end of the write, not to the first row's window, so a later request
// that diverges up to rollback positions before the furthest position written can still reuse its prefix (ReusableKV), at whatever point of the
// cycle the last decode stopped. capN must exceed window+rollback; the positions beyond that are how far a slot writes between compactions.
func kvWindowNextBase(base, p0, n, window, capN, rollback int) (from, to int, err error) {
	if p0 == 0 {
		base = 0 // a fresh sequence: nothing before it is read
	}
	low := max(0, p0-(window-1)) // the oldest position the first row's attention reads
	if low < base {
		return base, base, fmt.Errorf("cuda: windowed KV: position %d needs history from position %d but this slot holds only from %d (a reuse or rollback past the window slack; the caller must prefill from the reusable prefix, ReusableKV)", p0, low, base)
	}
	if p0+n-base <= capN {
		return base, base, nil
	}
	newBase := max(base, min(low, p0+n-(window-1)-rollback))
	if p0+n-newBase > capN {
		return base, base, fmt.Errorf("cuda: windowed KV: %d rows at position %d need %d positions, past the %d the layers hold (window %d + slack %d): write fewer rows per pass", n, p0, p0+n-newBase, capN, window, capN-window)
	}
	return base, newBase, nil
}

// kvWindowReusable reports whether a forward continuing at position prefix finds its window held by a slot whose base is base.
func kvWindowReusable(base, prefix, window int) bool { return base == 0 || prefix-(window-1) >= base }

// kvRoomFor reports whether n rows at position p0 can ever be written: the window floor of the first row, then the rows themselves, must fit the
// positions a windowed layer holds. False only for a pass wider than the slack allows (a long image block, a wide verify); no state is read, so
// it can run off the executor, and a caller declines to a narrower pass.
func (r *cudaResident) kvRoomFor(p0, n int) bool {
	if !r.kvWin {
		return true
	}
	return p0+n-max(0, p0-(r.kvWindow-1)) <= min(r.ctxCap, r.kvWindow+r.kvSlack)
}

// kvCompactSlot moves slot s's live positions [newBase, p0) to physical slot 0 of every windowed layer (K and V), through a scratch
// buffer because the source and destination ranges overlap whenever the live tail is longer than the slack. Stream-ordered kv_store
// launches: exact, no arithmetic.
func (r *cudaResident) kvCompactSlot(s, oldBase, newBase, p0 int) error {
	kcs, vcs := r.kc, r.vc
	if len(r.kvSlotBufs) > 0 {
		kcs, vcs = r.kvSlotBufs[s].kc, r.kvSlotBufs[s].vc
	}
	live := p0 - newBase
	if live <= 0 {
		return nil
	}
	for l := range r.layers {
		Ly := &r.layers[l]
		if !Ly.kvWin || Ly.kvShared {
			continue // a shared layer aliases its source's buffers, compacted with the source
		}
		n := live * Ly.kvDim
		off := (newBase - oldBase) * Ly.kvDim * 4
		for _, buf := range []Buffer{kcs[l], vcs[l]} {
			if e := r.copyF32(buf.At(off), r.kvScratch, n); e != nil {
				return e
			}
			if e := r.copyF32(r.kvScratch, buf, n); e != nil {
				return e
			}
		}
	}
	return nil
}

// ReusableKV implements decoder.ResidentReusableKV: of the first prefix positions of a sequence this resident's bound slot holds, how many
// it can serve to a forward that continues at position prefix. A windowed layer has dropped everything before its base, so a prefix shorter
// than base+window-1 has lost the history the next row attends to: 0 (prefill cold). Everything else is valid: the full layers hold
// every position, and a windowed layer holds [base, base+cap).
func (r *cudaResident) ReusableKV(prefix int) int {
	if !r.kvWin || prefix <= 0 {
		return prefix
	}
	out := prefix
	// the base is executor state: read it where it is written
	if err := r.do(func() error {
		if !kvWindowReusable(r.kvBase, prefix, r.kvWindow) {
			out = 0
		}
		return nil
	}); err != nil {
		return 0
	}
	return out
}

// uploadKVWindowed is UploadKV's trim for a windowed layer: the CPU-prefill bridge hands every layer the same live range [base, base+n)
// (a wrapped CPU ring starts later than 0), of which only the last window-1 positions can be attended to by the next row. It returns that
// tail, which lands at physical slot 0, and sets the slot's base to its first position; every windowed layer is handed the same range, so each
// call computes the same base.
func (r *cudaResident) uploadKVWindowed(layer, base int, keys, vals []float32) (k, v []float32) {
	Ly := &r.layers[layer]
	n := len(keys) / Ly.kvDim
	start := max(base, base+n-(r.kvWindow-1))
	if skip := (start - base) * Ly.kvDim; skip > 0 {
		keys, vals = keys[skip:], vals[skip:]
	}
	r.kvBases[r.kvSlot], r.kvBase = start, start
	return keys, vals
}
