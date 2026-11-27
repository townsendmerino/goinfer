package residency

import (
	"slices"
	"time"

	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// One recurrent checkpoint for the one resident slot a hybrid model has.
// The generation saves the recurrent state once its prefill has processed the whole prompt, before the first decode step, at
// c = len(prompt). A next request that matches the slot's sequence through c but does not extend all of it restores the state
// and prefills from c (kvmgr's scoreRecurrentTokens), where today it prefills from 0. The attention KV is positional and stays.

// checkpointHeld reports whether slot holds a checkpoint a request on a load of several slots may score.
func (h *Host) CheckpointHeld(slot int) bool {
	rc := h.RecurrentCheckpointer()
	return rc != nil && rc.RecurrentCheckpointHeld(slot)
}

// recurrentCheckpointer is the resident's checkpoint, or nil when the model has no recurrent state or the resident keeps none.
func (h *Host) RecurrentCheckpointer() resident.KVSubsystem {
	if h.Resident == nil || !h.hasRecurrentState() {
		return nil
	}
	kv := h.resKV()
	if kv.RecurrentCheckpointBytes() <= 0 {
		return nil
	}
	return kv
}

// residentCheckpointRestore makes the live recurrent state the checkpoint's when the acquire scored a restart from it: a recurrent
// reuse shorter than the slot's sequence. It falls back to a cold prefill (reuse 0) when the suffix is shorter than the resident's
// final-pass minimum, where the suffix would run other kernels than a cold prefill runs over those rows, and when the restore fails.
func (h *Host) CheckpointRestore(prompt []int, acq *KVAcquired) {
	rc := h.RecurrentCheckpointer()
	nextReuse, restore, declined := kvmgr.DecideRecurrentCheckpoint(
		acq.Reuse, acq.SlotLen, len(prompt), h.PrefillTailMin, h.hasRecurrentState(), rc != nil,
	)
	if declined != "" {
		acq.Reuse, acq.CkptDeclined = nextReuse, declined
		return
	}
	if !restore {
		return
	}
	t0 := time.Now()
	if err := rc.RestoreRecurrentCheckpoint(); err != nil {
		acq.Reuse, acq.CkptDeclined = 0, "restore failed: "+err.Error()
		return
	}
	acq.CkptRestored, acq.CkptDur = acq.Reuse, time.Since(t0)
}

// residentCheckpointSave saves the live recurrent state as the checkpoint for prompt, which the generation has just prefilled, and
// returns how long the copy took. A model or resident without one, or a failed copy, records nothing: the next request cannot
// restore from it.
func (h *Host) CheckpointSave(prompt []int) (time.Duration, bool) {
	(*h.CkptPrompt) = nil
	rc := h.RecurrentCheckpointer()
	if rc == nil {
		return 0, false
	}
	t0 := time.Now()
	if err := rc.SaveRecurrentCheckpoint(); err != nil {
		return 0, false
	}
	(*h.CkptPrompt) = slices.Clone(prompt)
	return time.Since(t0), true
}
