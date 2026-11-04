//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"os"
	"strings"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// KV8a's recurrent checkpoint on CUDA (decoder.ResidentRecurrentCheckpoint; docs/measurements/kv8a-2026-10/README.md, "Where the
// checkpoint lives on CUDA"): device copies of every recurrent layer's state and conv window, DeltaNet (dnWin, dnState) and
// Mamba-2 (mbWin, mbSSM), one per KV slot when they fit. They live in device memory because the save runs on every request, and a
// device-to-device copy keeps that cheap; there is no host fallback. A slot saves into its own copy; with fewer copies than slots,
// a save takes a free one and then the least recently saved, and a slot whose copy was taken has its restore refused. A load whose free
// VRAM cannot hold one keeps none, and the slot behaves as before KV8a.

// ckptCopy is one checkpoint: a buffer per live one, in recurrentPair order by layer, and the slot whose state it holds, -1 for none.
type ckptCopy struct {
	bufs  []Buffer
	slot  int
	saved uint64 // the save count when it was last saved: the least recently saved goes first
}

var _ decoder.ResidentRecurrentCheckpoint = (*cudaResident)(nil)

// ckptFreeVRAM is allocRecurrentCheckpoint's free-VRAM probe, a variable so a test can stand in a card with no room.
var ckptFreeVRAM = func(r *cudaResident) (uint64, error) { return cudaFreeVRAM(r) }

// allocRecurrentCheckpoint allocates the checkpoints once the core, the KV and the scratch are up, and before the expert cache and the
// windowed grant are sized from what is left, so both are sized around them: one per KV slot, as many as free VRAM less the reserve
// those two and every request's prefill need can hold. It keeps none, and says why on stderr, when it cannot hold one. Called inside
// the setup job.
func (r *cudaResident) allocRecurrentCheckpoint() {
	defer r.onLine(mem.LineState)()
	if !r.recurrent() {
		return
	}
	if r.recurrentLiveBytes() == 0 {
		return
	}
	reserve := r.ckptReserve()
	free, err := ckptFreeVRAM(r)
	if memReadSeenForTest != nil {
		memReadSeenForTest("ckpt", int64(free), err == nil)
	}
	charged := r.recurrentStateCharged()
	n, d := r.ckptCopiesBeside(r.KVSlots(), reserve, free, err)
	if d != "" {
		r.ckptDeclined = d
		fmt.Fprint(os.Stderr, ckptDeclinedLine(d))
		return
	}
	for range n {
		bufs, ok := r.allocCheckpointCopy()
		if !ok {
			break
		}
		r.ckptCopies = append(r.ckptCopies, ckptCopy{bufs: bufs, slot: -1})
	}
	if len(r.ckptCopies) == 0 {
		r.ckptDeclined = "the device allocation failed"
		fmt.Fprintf(os.Stderr, "cuda: recurrent checkpoint: none kept (%s)\n", r.ckptDeclined)
		return
	}
	r.ckptBytes = int64(len(r.ckptCopies)) * charged
	r.bindCheckpoint(0)
	fmt.Fprint(os.Stderr, ckptKeptLine(r.ckptBytes, len(r.ckptCopies)*len(r.ckptSave)))
}

// recurrentLiveBytes is every recurrent layer's live state and conv window in bytes: the buffers the checkpoint copies, sized as the
// setup job allocates them.
func (r *cudaResident) recurrentLiveBytes() int64 {
	var n int64
	for l := range r.layers {
		switch Ly := &r.layers[l]; {
		case Ly.isDeltaNet():
			n += int64(r.dnet.winFloats() + r.dnet.stateElems)
		case Ly.mixer == decoder.MixerMamba:
			n += int64(r.mamba.winFloats() + r.mamba.ssmFloats())
		}
	}
	return 4 * n
}

// recurrentStateCharged is the recurrent layers' live state and conv windows as the card charges them, one buffer each: what the
// setup job allocates for them, and what a kept checkpoint allocates again.
func (r *cudaResident) recurrentStateCharged() int64 {
	var n int64
	for l := range r.layers {
		switch Ly := &r.layers[l]; {
		case Ly.isDeltaNet():
			n += chargedBytes(int64(r.dnet.winFloats())*4) + chargedBytes(int64(r.dnet.stateElems)*4)
		case Ly.mixer == decoder.MixerMamba:
			n += chargedBytes(int64(r.mamba.winFloats())*4) + chargedBytes(int64(r.mamba.ssmFloats())*4)
		}
	}
	return n
}

// recurrentLayers is the number of recurrent layers, each with two live buffers the checkpoint copies.
func (r *cudaResident) recurrentLayers() int {
	n := 0
	for l := range r.layers {
		if Ly := &r.layers[l]; Ly.isDeltaNet() || Ly.mixer == decoder.MixerMamba {
			n++
		}
	}
	return n
}

// ckptReserve is what the checkpoint must leave free: the companion reserve, the grant reserve, two default prefill passes' scratch,
// and the expert cache's minimum when there is one.
func (r *cudaResident) ckptReserve() int64 {
	reserve := r.extraBytes + grantReserveBytes + 2*r.prefillPassBytes(prefillDefaultChunk)
	if r.cacheExperts {
		reserve += r.expertCacheFloorBytes()
	}
	return reserve
}

// ckptDeclineFor is allocRecurrentCheckpoint's decision over the free VRAM it read: "" keeps a checkpoint of bytes beside reserve,
// else the reason none is kept.
func ckptDeclineFor(bytes, reserve int64, free uint64, err error) string {
	if err != nil {
		return "free VRAM unreadable: " + err.Error()
	}
	if int64(free) < bytes+reserve {
		return fmt.Sprintf("%.1f MiB needed beside a %.0f MiB reserve, %.0f MiB free", mib(bytes), mib(reserve), mib(int64(free)))
	}
	return ""
}

// ckptCopiesBeside is ckptCopiesFor for this build's checkpoint, as the card charges it: the one figure the build and the memory plan decide on.
func (r *cudaResident) ckptCopiesBeside(slots int, reserve int64, free uint64, err error) (int, string) {
	return ckptCopiesFor(slots, r.recurrentStateCharged(), reserve, free, err)
}

// ckptCopiesFor is how many checkpoints of bytes each a build of slots KV slots keeps beside reserve, at most one a slot, and why none
// when it keeps none.
func ckptCopiesFor(slots int, bytes, reserve int64, free uint64, err error) (int, string) {
	if d := ckptDeclineFor(bytes, reserve, free, err); d != "" {
		return 0, d
	}
	return min(max(slots, 1), int((int64(free)-reserve)/bytes)), ""
}

// ckptDeclinedLine and ckptKeptLine are the lines the build logs for its checkpoint decision.
func ckptDeclinedLine(why string) string {
	return fmt.Sprintf("cuda: recurrent checkpoint: none kept (%s); a reply rendered back differently re-prefills the whole prompt\n", why)
}

func ckptKeptLine(bytes int64, buffers int) string {
	return fmt.Sprintf("cuda: recurrent checkpoint: %.1f MiB on the device (%d buffers, KV8a)\n", mib(bytes), buffers)
}

// ckptAllocFailAfterForTest, when above 0, fails allocCheckpointCopy part way once that many copies are kept, as a fragmented device would. Only a test sets it.
var ckptAllocFailAfterForTest int

// allocCheckpointCopy allocates a device buffer the size of each live one, all or none: a failed allocation releases what it made.
func (r *cudaResident) allocCheckpointCopy() (bufs []Buffer, ok bool) {
	defer r.onLine(mem.LineState)()
	defer func() {
		if v := recover(); v != nil {
			if !strings.Contains(fmt.Sprint(v), "device allocation failed") {
				panic(v)
			}
			for _, b := range bufs {
				r.dev.ReleaseBuf(b)
			}
			bufs, ok = nil, false
		}
	}()
	for l := range r.layers {
		if L := &r.layers[l]; L.isRecurrent() {
			p := L.recurrentLive()
			for _, b := range []Buffer{p.win, p.state} {
				bufs = append(bufs, r.af(b.Len()))
				if ckptAllocFailAfterForTest > 0 && len(r.ckptCopies) >= ckptAllocFailAfterForTest {
					panic("cuda: device allocation failed (test seam): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")
				}
			}
		}
	}
	return bufs, true
}

func mib(b int64) float64 { return float64(b) / (1 << 20) }

// RecurrentCheckpointBytes (decoder.ResidentRecurrentCheckpoint) is every checkpoint's device bytes together, 0 when the load kept none.
func (r *cudaResident) RecurrentCheckpointBytes() int64 { return r.ckptBytes }

// bindCheckpoint points the save and restore copies at checkpoint k and the bound slot's live state. Runs on the executor.
func (r *cudaResident) bindCheckpoint(k int) {
	r.ckptSave, r.ckptRestore = r.ckptSave[:0], r.ckptRestore[:0]
	i := 0
	for l := range r.layers {
		L := &r.layers[l]
		if !L.isRecurrent() {
			continue
		}
		p := L.recurrentLive()
		for _, live := range []Buffer{p.win, p.state} {
			c := r.ckptCopies[k].bufs[i]
			r.ckptSave = append(r.ckptSave, gpu.DeviceCopy{Dst: c, Src: live, Bytes: live.Len() * 4})
			r.ckptRestore = append(r.ckptRestore, gpu.DeviceCopy{Dst: live, Src: c, Bytes: live.Len() * 4})
			i++
		}
	}
}

// ckptCopyOf is the checkpoint slot s holds, or -1.
func (r *cudaResident) ckptCopyOf(s int) int {
	for k := range r.ckptCopies {
		if r.ckptCopies[k].slot == s {
			return k
		}
	}
	return -1
}

// RecurrentCheckpointHeld implements decoder.ResidentRecurrentCheckpointOwner.
func (r *cudaResident) RecurrentCheckpointHeld(slot int) bool {
	held := false
	_ = r.do(func() error {
		held = slot >= 0 && r.ckptCopyOf(slot) >= 0
		return nil
	})
	return held
}

// ckptCopyFor is the checkpoint a save on slot s writes: its own, then a free one, then the least recently saved.
func (r *cudaResident) ckptCopyFor(s int) int {
	if k := r.ckptCopyOf(s); k >= 0 {
		return k
	}
	if k := r.ckptCopyOf(-1); k >= 0 {
		return k
	}
	oldest := 0
	for k := range r.ckptCopies {
		if r.ckptCopies[k].saved < r.ckptCopies[oldest].saved {
			oldest = k
		}
	}
	return oldest
}

// SaveRecurrentCheckpoint copies every recurrent layer's live state and conv window, the bound slot's, into that slot's checkpoint, after
// the stream's queued work, so it holds what the last prefill left.
func (r *cudaResident) SaveRecurrentCheckpoint() error {
	if len(r.ckptCopies) == 0 {
		return errNoCheckpoint
	}
	return r.do(func() error {
		k := r.ckptCopyFor(r.kvSlot)
		r.ckptCopies[k].slot = -1 // a partial copy holds no slot's state
		r.bindCheckpoint(k)
		if e := r.copyCheckpointNow(r.ckptSave); e != nil {
			return e
		}
		r.ckptSaves++
		r.ckptCopies[k].slot, r.ckptCopies[k].saved = r.kvSlot, r.ckptSaves
		return nil
	})
}

// RestoreRecurrentCheckpoint copies the bound slot's checkpoint back into the live state; the next prefill continues from it. It refuses
// when the slot holds none, because it never saved or its copy was taken; the decoder then prefills from the start.
func (r *cudaResident) RestoreRecurrentCheckpoint() error {
	if len(r.ckptCopies) == 0 {
		return errNoCheckpoint
	}
	return r.do(func() error {
		k := r.ckptCopyOf(r.kvSlot)
		if k < 0 {
			return fmt.Errorf("cuda: KV slot %d holds no recurrent checkpoint", r.kvSlot)
		}
		r.bindCheckpoint(k)
		return r.copyCheckpointNow(r.ckptRestore)
	})
}

var errNoCheckpoint = errors.New("cuda: no recurrent checkpoint kept")

func (r *cudaResident) copyCheckpoint(copies []gpu.DeviceCopy) error {
	if len(copies) == 0 {
		return errNoCheckpoint
	}
	return r.do(func() error { return r.copyCheckpointNow(copies) })
}

// copyCheckpointNow is copyCheckpoint on the executor.
func (r *cudaResident) copyCheckpointNow(copies []gpu.DeviceCopy) error {
	if e := r.stream.Sync(); e != nil {
		return e
	}
	return gpu.CopyDeviceBatch(copies)
}
