//go:build cuda

package cuda

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
)

// Windowed KV (docs/tasks/task-cuda-windowed-kv-2026-10.md). A sliding-window layer reads only the last `window` positions, so with
// Options.ResidentWindowedKV its K and V buffers hold window+slack positions instead of the whole context. Every kernel still addresses
// K and V by ABSOLUTE position (kc[pos*kvDim + ...]); a windowed layer is handed a view shifted back by base*kvDim*4 bytes, where base is
// the absolute position stored at physical slot 0, so no kernel changes: an address the kernel computes for any position in
// [base, base+cap) lands inside the buffer, and no position outside that range is read (the window floor never drops below base: kvEnsure
// proves it) or written. When the next write would pass base+cap the live tail moves to physical slot 0 (kvCompact) and base advances.
//
// All of it runs on the executor thread. State per slot: kvBases[slot] (kvBase caches the bound slot's).

// kvWindowSlack is the headroom a windowed layer holds beyond its window; tests lower it to force compaction on a tiny fixture. In
// production it is decoder.WindowedKVSlack, the figure decoder's pricing uses, so priced bytes equal allocated bytes.
var kvWindowSlack = decoder.WindowedKVSlack

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
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.isMLA || Ly.isDeltaNet || Ly.window <= 0 || int(Ly.window) != window || !m.WindowedKVLayer(l) {
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
		fmt.Fprintf(os.Stderr, "cuda: windowed KV: sliding-window layers hold %d positions (window %d + %d slack) instead of the context's %d\n",
			r.kvWindow+kvWindowSlack, r.kvWindow, kvWindowSlack, r.ctxCap)
	}
}

// kvAllocScratch sizes the compaction staging buffer to the widest live tail (window-1 positions) any windowed layer can hold, and starts every
// slot at base 0.
func (r *cudaResident) kvAllocScratch() {
	widest := 0
	for l := range r.layers {
		if r.layers[l].kvWin {
			widest = max(widest, r.layers[l].kvDim)
		}
	}
	r.kvScratch = r.af(max(1, r.kvWindow-1) * widest)
	r.kvBases = make([]int, max(1, r.kvSlotsN))
}

// kvPositions is how many positions layer l's K/V buffers hold.
func (r *cudaResident) kvPositions(l int) int {
	if r.layers[l].kvWin {
		return min(r.ctxCap, int(r.layers[l].window)+kvWindowSlack)
	}
	return r.ctxCap
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
	base := r.kvBases[s]
	if p0 == 0 {
		base = 0 // a fresh sequence: nothing before it is read
	}
	window, capN := r.kvWindow, min(r.ctxCap, r.kvWindow+kvWindowSlack)
	low := max(0, p0-(window-1)) // the oldest position the first row's attention reads
	if low < base {
		return fmt.Errorf("cuda: windowed KV: position %d needs history from position %d but this slot holds only from %d (a reuse or rollback past the window slack; the caller must prefill from the reusable prefix, ReusableKV)", p0, low, base)
	}
	if p0+n-base > capN {
		newBase := low
		if p0+n-newBase > capN {
			return fmt.Errorf("cuda: windowed KV: %d rows at position %d need %d positions, past the %d the layers hold (window %d + slack %d): write fewer rows per pass", n, p0, p0+n-newBase, capN, window, kvWindowSlack)
		}
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

// kvRoomFor reports whether n rows at position p0 can ever be written: the window floor of the first row, then the rows themselves, must fit the
// positions a windowed layer holds. False only for a pass wider than the slack allows (a long image block, a wide verify); no state is read, so
// it can run off the executor, and a caller declines to a narrower pass.
func (r *cudaResident) kvRoomFor(p0, n int) bool {
	if !r.kvWin {
		return true
	}
	return p0+n-max(0, p0-(r.kvWindow-1)) <= min(r.ctxCap, r.kvWindow+kvWindowSlack)
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
		if r.kvBase > 0 && prefix-(r.kvWindow-1) < r.kvBase {
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
