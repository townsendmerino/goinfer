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

// KV8a's recurrent checkpoint on CUDA (decoder.ResidentRecurrentCheckpoint; docs/measurements/kv8a-2026-10/README.md, "Where the
// checkpoint lives on CUDA"): one device copy of every recurrent layer's state and conv window, DeltaNet (dnWin, dnState) and
// Mamba-2 (mbWin, mbSSM). It lives in device memory because the save runs on every request, and a device-to-device copy keeps
// that cheap; there is no host fallback. A load whose free VRAM cannot hold it keeps none, and the slot behaves as before KV8a.

var _ decoder.ResidentRecurrentCheckpoint = (*cudaResident)(nil)

// ckptFreeVRAM is allocRecurrentCheckpoint's free-VRAM probe, a variable so a test can stand in a card with no room.
var ckptFreeVRAM = func(r *cudaResident) (uint64, error) { return cudaFreeVRAM(r) }

// allocRecurrentCheckpoint allocates the checkpoint once the core, the KV and the scratch are up, and before the expert cache and the
// windowed grant are sized from what is left, so both are sized around it. It keeps none, and says why on stderr, when free VRAM less
// the reserve those two and every request's prefill need cannot hold it. Called inside the setup job.
func (r *cudaResident) allocRecurrentCheckpoint() {
	defer r.onLine(mem.LineState)()
	if !r.recurrent() {
		return
	}
	var live []Buffer
	for l := range r.layers {
		Ly := &r.layers[l]
		switch {
		case Ly.isDeltaNet():
			live = append(live, Ly.dnWin, Ly.dnState)
		case Ly.mixer == decoder.MixerMamba:
			live = append(live, Ly.mbWin, Ly.mbSSM)
		}
	}
	bytes := r.recurrentLiveBytes()
	if bytes == 0 {
		return
	}
	reserve := r.ckptReserve()
	free, err := ckptFreeVRAM(r)
	if memReadSeenForTest != nil {
		memReadSeenForTest("ckpt", int64(free), err == nil)
	}
	if d := ckptDeclineFor(bytes, reserve, free, err); d != "" {
		r.ckptDeclined = d
		fmt.Fprint(os.Stderr, ckptDeclinedLine(d))
		return
	}
	copies, ok := r.allocCheckpointCopies(live)
	if !ok {
		r.ckptDeclined = "the device allocation failed"
		fmt.Fprintf(os.Stderr, "cuda: recurrent checkpoint: none kept (%s)\n", r.ckptDeclined)
		return
	}
	r.ckptSave, r.ckptRestore, r.ckptBytes = copies, nil, bytes
	for _, c := range copies {
		r.ckptRestore = append(r.ckptRestore, gpu.DeviceCopy{Dst: c.Src, Src: c.Dst, Bytes: c.Bytes})
	}
	fmt.Fprint(os.Stderr, ckptKeptLine(bytes, len(live)))
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

// ckptDeclinedLine and ckptKeptLine are the lines the build logs for its checkpoint decision.
func ckptDeclinedLine(why string) string {
	return fmt.Sprintf("cuda: recurrent checkpoint: none kept (%s); a reply rendered back differently re-prefills the whole prompt\n", why)
}

func ckptKeptLine(bytes int64, buffers int) string {
	return fmt.Sprintf("cuda: recurrent checkpoint: %.1f MiB on the device (%d buffers, KV8a)\n", mib(bytes), buffers)
}

// allocCheckpointCopies allocates a device buffer the size of each live one and returns the save copies (live into checkpoint), all or
// none: a failed allocation releases what it made.
func (r *cudaResident) allocCheckpointCopies(live []Buffer) (copies []gpu.DeviceCopy, ok bool) {
	defer r.onLine(mem.LineState)()
	var made []Buffer
	defer func() {
		if v := recover(); v != nil {
			if !strings.Contains(fmt.Sprint(v), "device allocation failed") {
				panic(v)
			}
			for _, b := range made {
				r.dev.ReleaseBuf(b)
			}
			copies, ok = nil, false
		}
	}()
	for _, b := range live {
		c := r.af(b.Len())
		made = append(made, c)
		copies = append(copies, gpu.DeviceCopy{Dst: c, Src: b, Bytes: b.Len() * 4})
	}
	return copies, true
}

func mib(b int64) float64 { return float64(b) / (1 << 20) }

// RecurrentCheckpointBytes (decoder.ResidentRecurrentCheckpoint) is the checkpoint's device bytes, 0 when the load kept none.
func (r *cudaResident) RecurrentCheckpointBytes() int64 { return r.ckptBytes }

// SaveRecurrentCheckpoint copies every recurrent layer's live state and conv window into the checkpoint, after the stream's queued
// work, so it holds what the last prefill left.
func (r *cudaResident) SaveRecurrentCheckpoint() error { return r.copyCheckpoint(r.ckptSave) }

// RestoreRecurrentCheckpoint copies the checkpoint back into the live state; the next prefill continues from it.
func (r *cudaResident) RestoreRecurrentCheckpoint() error { return r.copyCheckpoint(r.ckptRestore) }

func (r *cudaResident) copyCheckpoint(copies []gpu.DeviceCopy) error {
	if len(copies) == 0 {
		return fmt.Errorf("cuda: no recurrent checkpoint kept")
	}
	return r.do(func() error {
		if e := r.stream.Sync(); e != nil {
			return e
		}
		return gpu.CopyDeviceBatch(copies)
	})
}
