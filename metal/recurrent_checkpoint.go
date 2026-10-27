package metal

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
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
	var bytes int64
	for _, b := range live {
		bytes += int64(b.Len()) * 4
	}
	if bytes == 0 {
		return
	}
	if ok, why := ckptFits(m, bytes); !ok {
		r.ckptDeclined = why
		fmt.Fprintf(os.Stderr, "metal: recurrent checkpoint: none kept (%s); a reply rendered back differently re-prefills the whole prompt\n", why)
		return
	}
	for _, b := range live {
		c := NewBufferFloatsOn(mem.LineState, r.d, make([]float32, b.Len()))
		r.ckptSave = append(r.ckptSave, gpu.DeviceCopy{Dst: c, Src: b, Bytes: b.Len() * 4})
		r.ckptRestore = append(r.ckptRestore, gpu.DeviceCopy{Dst: b, Src: c, Bytes: b.Len() * 4})
	}
	r.ckptBytes = bytes
	fmt.Fprintf(os.Stderr, "metal: recurrent checkpoint: %.1f MiB (%d buffers, KV8a)\n", mib(bytes), len(live))
}

func mib(b int64) float64 { return float64(b) / (1 << 20) }

// RecurrentCheckpointBytes (decoder.ResidentRecurrentCheckpoint) is the checkpoint's bytes, 0 when the load kept none.
func (a *metalResident) RecurrentCheckpointBytes() int64 { return a.r.ckptBytes }

// SaveRecurrentCheckpoint copies every recurrent layer's live conv window and state into the checkpoint, after the work in flight, so
// it holds what the last prefill left.
func (a *metalResident) SaveRecurrentCheckpoint() error { return a.r.copyCheckpoint(a.r.ckptSave) }

// RestoreRecurrentCheckpoint copies the checkpoint back into the live state; the next prefill continues from it.
func (a *metalResident) RestoreRecurrentCheckpoint() error {
	return a.r.copyCheckpoint(a.r.ckptRestore)
}

func (r *resident) copyCheckpoint(copies []gpu.DeviceCopy) error {
	if len(copies) == 0 {
		return fmt.Errorf("metal: no recurrent checkpoint kept")
	}
	r.stopExec() // the executor and an open chain have command buffers in flight that read or write the live state
	return gpu.CopyDeviceBatch(copies)
}
