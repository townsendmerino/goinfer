package metal

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// KV8a's recurrent checkpoint on Metal (decoder.ResidentRecurrentCheckpoint; docs/measurements/kv8a-metal-2026-10/README.md): one copy
// of every recurrent layer's conv window and state, Gated-DeltaNet (win, state) and Mamba-2 (win, ssm), in buffers beside the live
// ones, copied on the device after the executor has drained. A load the memory guard cannot also fit it into keeps none, and the
// slot behaves as before KV8a.

var _ decoder.ResidentRecurrentCheckpoint = (*metalResident)(nil)

// ckptFits reports whether the guard's budget holds the load's need plus the checkpoint's bytes (residentMemoryDecline's quantities,
// not the OS's account of free memory), and why not. A variable so a test can stand in a machine with no room.
var ckptFits = func(m *decoder.Model, bytes int64) (bool, string) {
	need := residentNeedBytes(m)
	budget, ok := residentBudget()
	if need <= 0 || !ok {
		return true, "" // nothing to compare against, as residentMemoryDecline
	}
	if need+bytes > budget {
		return false, fmt.Sprintf("%.1f MiB needed beside the load's %.2f GB, over the %.2f GB budget", mib(bytes), float64(need)/(1<<30), float64(budget)/(1<<30))
	}
	return true, ""
}

// allocRecurrentCheckpoint allocates the checkpoint once the resident is built. It keeps none, and says why on stderr, when the
// memory guard's budget cannot also hold it.
func (r *resident) allocRecurrentCheckpoint(m *decoder.Model) {
	if !r.recurrentModel() {
		return
	}
	bytes := metalRecurrentSlotBytes(m)
	if bytes == 0 {
		return
	}
	if ok, why := ckptFits(m, bytes); !ok {
		r.ckptDeclined = why
		logs.Warnf("metal: recurrent checkpoint: none kept (%s); a reply rendered back differently re-prefills the whole prompt", why)
		return
	}
	for l := range r.layers { // in recurrentLive's order, which bindCheckpoint pairs them by
		if r.layers[l].delta == nil && r.layers[l].mamba == nil {
			continue
		}
		win, state, err := stateBuffersOn(r.d, m, l)
		if err != nil {
			for _, b := range r.ckptBufs {
				r.d.ReleaseBuf(b)
			}
			r.ckptBufs, r.ckptDeclined = nil, err.Error()
			logs.Warnf("metal: recurrent checkpoint: none kept (%v); a reply rendered back differently re-prefills the whole prompt", err)
			return
		}
		r.ckptBufs = append(r.ckptBufs, win, state)
	}
	r.bindCheckpoint()
	r.ckptBytes = bytes
	logs.Infof("metal: recurrent checkpoint: %.1f MiB (%d buffers, KV8a)", mib(bytes), len(r.ckptBufs))
}

func mib(b int64) float64 { return float64(b) / (1 << 20) }

// RecurrentCheckpointBytes (decoder.ResidentRecurrentCheckpoint) is the checkpoint's bytes, 0 when the load kept none.
func (a *metalResident) RecurrentCheckpointBytes() int64 { return a.r.ckptBytes }

// recurrentLive is every recurrent layer's live conv window and state, in layer order: the bound KV slot's (useKVSlot).
func (r *resident) recurrentLive() []Buffer {
	var live []Buffer
	for i := range r.layers {
		L := &r.layers[i]
		switch {
		case L.delta != nil:
			live = append(live, L.delta.win, L.delta.state)
		case L.mamba != nil:
			live = append(live, L.mamba.win, L.mamba.ssm)
		}
	}
	return live
}

// bindCheckpoint points the save and restore copies at the bound slot's live state.
func (r *resident) bindCheckpoint() {
	live := r.recurrentLive()
	r.ckptSave, r.ckptRestore = r.ckptSave[:0], r.ckptRestore[:0]
	for i, c := range r.ckptBufs {
		r.ckptSave = append(r.ckptSave, gpu.DeviceCopy{Dst: c, Src: live[i], Bytes: c.Len() * 4})
		r.ckptRestore = append(r.ckptRestore, gpu.DeviceCopy{Dst: live[i], Src: c, Bytes: c.Len() * 4})
	}
}

// SaveRecurrentCheckpoint copies every recurrent layer's live conv window and state, the bound slot's, into the checkpoint,
// after the work in flight, so it holds what the last prefill left. The checkpoint is one copy: it then belongs to that slot.
func (a *metalResident) SaveRecurrentCheckpoint() error {
	a.r.bindCheckpoint()
	if err := a.r.copyCheckpoint(a.r.ckptSave); err != nil {
		a.r.ckptSlot = -1 // a partial copy holds no slot's state
		return err
	}
	a.r.ckptSlot = a.r.kvSlot
	return nil
}

// RestoreRecurrentCheckpoint copies the checkpoint back into the live state; the next prefill continues from it. It refuses before any save,
// and when another slot is bound than the one the checkpoint was saved on; the decoder then prefills from the start.
func (a *metalResident) RestoreRecurrentCheckpoint() error {
	if a.r.ckptSlot < 0 {
		return fmt.Errorf("metal: no recurrent checkpoint saved")
	}
	if a.r.kvSlot != a.r.ckptSlot {
		return fmt.Errorf("metal: the recurrent checkpoint holds KV slot %d's state, and slot %d is bound", a.r.ckptSlot, a.r.kvSlot)
	}
	a.r.bindCheckpoint()
	return a.r.copyCheckpoint(a.r.ckptRestore)
}

func (r *resident) copyCheckpoint(copies []gpu.DeviceCopy) error {
	if len(copies) == 0 {
		return fmt.Errorf("metal: no recurrent checkpoint kept")
	}
	r.stopExec() // the executor and an open chain have command buffers in flight that read or write the live state
	return gpu.CopyDeviceBatch(copies)
}
