//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// Windowed KV (docs/tasks/task-cuda-windowed-kv-2026-10.md). A sliding-window layer reads only the last `window` positions, so with
// Options.ResidentWindowedKV its K and V live in the window group's pool (kvpool.go), and a slot maps only the blocks that cover the
// positions it still holds, [base, end): window+slack positions at most, against the context's worth a full layer maps. Every kernel
// addresses K and V by absolute position through the slot's table, so nothing moves: when the next write would pass base+window+slack,
// base advances by only as much as keeps the slot's rollback behind the end of the write (kvWindowNextBase) and the blocks wholly below
// it go back to the pool, so a base move comes every kvWindowStep positions once the window is full. No position below base is read
// (the window floor never drops below it: kvEnsure proves it).
//
// The rollback is how far before the newest position a follow-up may diverge and still reuse its prefix (ReusableKV): slack-step positions,
// 448 at the planned 512 slack, at every point of the cycle. kvGrantRollback raises the slack, and so the rollback, from VRAM left over once
// the build is done. Why and what each costs: docs/measurements/prefix-reuse-windowed-kv-2026-10-14.md.
//
// All of it runs on the executor thread. State per slot: kvBases[slot].

// kvWindowSlack is the headroom a windowed layer is planned with beyond its window: decoder.WindowedKVSlack, the figure decoder's pricing
// uses, so priced bytes equal what the build allocates before kvGrantRollback. Tests lower it to force compaction on a tiny fixture.
var kvWindowSlack = decoder.WindowedKVSlack

// kvWindowStep is how many positions a full window writes between base moves; the rollback is the slack less this. 1 <= step < slack.
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

// kvWindowGrant is how many positions to add to the windowed slack: the VRAM left over (free less reserve) divided by what one more
// position costs (perPos, across every slot's windowed layers), rounded down to a multiple of step and capped at
// maxExtra. 0 when nothing is left over.
func kvWindowGrant(free, reserve, perPos int64, step, maxExtra int) int {
	if perPos <= 0 || maxExtra < step || free <= reserve {
		return 0
	}
	return min(int((free-reserve)/perPos), maxExtra) / step * step
}

// kvGrantRollback raises the windowed slack, and with it the rollback, from VRAM nothing else will use, once the context and the KV slots
// are fixed, so it never costs either. A load without an expert cache calls it last. An expert-cache load calls it just before the cache is
// sized, with holdBack the cache's minimum (expertCacheFloorBytes), so the grant comes out of the elastic cache and never makes the load
// decline: an expert-cache load of a windowed model would otherwise keep the floor rollback, which refuses a follow-up that rewrites more
// than about 448 tokens of the end (docs/queue-performance.md, P34, cause 2). Also held back: the companion and vision-tower reserve
// (extraBytes), the decode margin, and two default prefill passes' scratch, which prefillCore allocates per request. The windowed buffers are
// reallocated (nothing is written yet); a failed allocation keeps the planned ones.
func (r *cudaResident) kvGrantRollback(holdBack int64) {
	if !r.kvWin {
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
	reserve := r.extraBytes + grantReserveBytes + 2*r.prefillPassBytes(prefillDefaultChunk) + holdBack
	maxExtra := min(kvWindowRollbackMax-r.kvRollback(), r.ctxCap-1-r.kvWindow-r.kvSlack)
	extra := kvWindowGrant(int64(free), reserve, perPos, kvWindowStep, maxExtra)
	if extra == 0 || !r.kvReallocWindowed(r.kvSlack+extra) {
		return
	}
	r.kvSlack += extra
	fmt.Fprintf(os.Stderr, "cuda: windowed KV: %d more positions from free VRAM (%.0f MB of %.0f MB free); a follow-up may now diverge %d before the newest\n",
		extra, float64(int64(extra)*perPos)/(1<<20), float64(free)/(1<<20), r.kvRollback())
}

// kvGrantFreeVRAM is kvGrantRollback's free-VRAM probe, a variable so a test can stand in a card with no headroom.
var kvGrantFreeVRAM = func(r *cudaResident) (uint64, error) { return cudaFreeVRAM(r) }

// kvReallocWindowed resizes the window group's pool for every slot to hold window+slack positions, all or none: on an allocation failure
// the new buffers are released and the old ones kept. Nothing may be mapped in the window group yet (kvGrantRollback runs once the build
// is done, before any write). KV-shared layers re-alias their sources.
func (r *cudaResident) kvReallocWindowed(slack int) (ok bool) {
	defer r.onLine(mem.LineKV)()
	for s := range r.kvTabs {
		for _, id := range r.kvTabs[s].ids[kvGroupWin] {
			if id != 0 {
				return false
			}
		}
	}
	old := r.kvSlack
	r.kvSlack = slack
	blocks := r.kvPoolBlocks(kvGroupWin, len(r.kvTabs))
	r.kvSlack = old
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
	for _, set := range [][]Buffer{r.kc, r.vc} {
		for l := range r.layers {
			if Ly := &r.layers[l]; Ly.kvWin && !Ly.kvShared && set[l].Len() > 0 {
				made = append(made, swap{set, l, r.akv(l, blocks*kvBlock*Ly.kvDim)})
			}
		}
	}
	for _, m := range made {
		r.dev.ReleaseBuf(m.set[m.l])
		m.set[m.l] = m.b
	}
	for l := range r.layers {
		if Ly := &r.layers[l]; Ly.kvShared && Ly.kvWin {
			r.kc[l], r.vc[l] = r.kc[Ly.kvSrc], r.vc[Ly.kvSrc]
		}
	}
	r.kvPools[kvGroupWin] = newKVPool(blocks)
	if e := r.kvZeroNullBlocks(); e != nil {
		panic(fmt.Sprintf("cuda: windowed KV: zeroing the null blocks: %v", e))
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

// kvK / kvV are layer l's K and V pools; kvT is the table the kernels address them through (kvpool.go).
func (r *cudaResident) kvK(l int) Buffer { return r.kc[l] }
func (r *cudaResident) kvV(l int) Buffer { return r.vc[l] }

// kvEnsure maps the blocks a write of n rows at absolute position p0 in the bound slot needs, and proves the history the rows attend to
// is still held. A fresh sequence (p0 == 0) restarts the window. It returns an error rather than ever reading a position whose block was
// released: such a call is a caller bug (a reuse past ReusableKV, or a rollback further than the slack), and a wrong answer must not be the
// symptom. Executor thread only.
func (r *cudaResident) kvEnsure(p0, n int) error { return r.kvEnsureSlot(r.kvSlot, p0, n) }

// kvEnsureSlot is kvEnsure for slot s, whether or not it is the bound one (an MC3 step prepares every row's slot before its layer loops).
func (r *cudaResident) kvEnsureSlot(s, p0, n int) error {
	if r.kvReleased(s) {
		if p0 > 0 {
			return fmt.Errorf("cuda: KV slot %d's blocks were released to make room; a write at %d must start from position 0 (ReusableKV answers 0)", s, p0)
		}
		r.kvEvicted[s] = false
	}
	if s < len(r.kvStopped) && r.kvStopped[s] {
		if p0 > 0 {
			return fmt.Errorf("cuda: KV slot %d was stopped: its blocks past its share were taken back for a conversation within its share: %w", s, decoder.ErrKVPoolFull)
		}
		r.kvStopped[s] = false
	}
	if s < len(r.kvSlotUse) {
		r.kvUseTick++
		r.kvSlotUse[s] = r.kvUseTick
	}
	base := 0
	if r.kvWin {
		var err error
		if _, base, err = kvWindowNextBase(r.kvBases[s], p0, n, r.kvWindow, min(r.ctxCap, r.kvWindow+r.kvSlack), r.kvRollback()); err != nil {
			return err
		}
		r.kvBases[s] = base
	}
	return r.kvMapWrite(s, p0, n, base)
}

// kvWindowNextBase is kvEnsureSlot's arithmetic, kept free of device state so it can be tested on recorded traffic. Given the slot's base, it returns
// the base the write of n rows at p0 starts from (0 for a fresh sequence) and the base after it; they differ when the write moves the base.
// An error means the write reads history below the base or does not fit capN positions.
//
// A move takes the base to rollback+window-1 positions behind the end of the write, not to the first row's window, so a later request
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

// ReusableKV implements decoder.ResidentReusableKV: of the first prefix positions of a sequence this resident's bound slot holds, how many
// it can serve to a forward that continues at position prefix. A slot whose blocks were released to make room holds nothing: 0. A
// windowed layer has dropped everything before its base, so a prefix shorter than base+window-1 has lost the history the next row attends
// to: 0 (prefill cold). Everything else is valid: the full layers hold every position, and a windowed layer holds [base, end).
func (r *cudaResident) ReusableKV(prefix int) int {
	if prefix <= 0 {
		return prefix
	}
	out := prefix
	// the base and the released marks are executor state: read them where they are written
	if err := r.do(func() error {
		if r.kvReleased(r.kvSlot) || (r.kvWin && !kvWindowReusable(r.kvBases[r.kvSlot], prefix, r.kvWindow)) {
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
// tail and sets the bound slot's base to its first position; every windowed layer is handed the same range, so each call computes the same
// base. Executor thread only.
func (r *cudaResident) uploadKVWindowed(layer, base int, keys, vals []float32) (k, v []float32) {
	Ly := &r.layers[layer]
	n := len(keys) / Ly.kvDim
	start := max(base, base+n-(r.kvWindow-1))
	if skip := (start - base) * Ly.kvDim; skip > 0 {
		keys, vals = keys[skip:], vals[skip:]
	}
	r.kvBases[r.kvSlot] = start
	return keys, vals
}
