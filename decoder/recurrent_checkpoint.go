package decoder

import (
	"slices"
	"time"
)

// KV8a (docs/tasks/task-kv-architecture-2026-10.md, KV8a): one recurrent checkpoint for the one resident slot a hybrid model has.
// The generation saves the recurrent state once its prefill has processed the whole prompt, before the first decode step, at
// c = len(prompt). A next request that matches the slot's sequence through c but does not extend all of it restores the state
// and prefills from c (kvmgr's scoreRecurrent), where today it prefills from 0. The attention KV is positional and stays.

// ResidentRecurrentCheckpoint is the optional capability of a resident with recurrent state to keep one copy of it, every
// recurrent layer's state and conv window, beside the live state. RecurrentCheckpointBytes is the copy's size, 0 when the resident
// keeps none (its load did not admit it). SaveRecurrentCheckpoint copies the live state into it and RestoreRecurrentCheckpoint
// copies it back; neither touches the attention KV. An error leaves the live state unknown to the caller.
type ResidentRecurrentCheckpoint interface {
	RecurrentCheckpointBytes() int64
	SaveRecurrentCheckpoint() error
	RestoreRecurrentCheckpoint() error
}

// recurrentCheckpointer is the resident's checkpoint, or nil when the model has no recurrent state or the resident keeps none.
func (m *Model) recurrentCheckpointer() ResidentRecurrentCheckpoint {
	if m.resident == nil || !m.hasRecurrentState() {
		return nil
	}
	rc, ok := m.resident.(ResidentRecurrentCheckpoint)
	if !ok || rc.RecurrentCheckpointBytes() <= 0 {
		return nil
	}
	return rc
}

// residentCheckpointRestore makes the live recurrent state the checkpoint's when the acquire scored a restart from it: a recurrent
// reuse shorter than the slot's sequence. It falls back to a cold prefill (reuse 0) when the suffix is shorter than the resident's
// final-pass minimum, where the suffix would run other kernels than a cold prefill runs over those rows, and when the restore fails.
func (m *Model) residentCheckpointRestore(prompt []int, acq *kvAcquired) {
	if acq.reuse <= 0 || acq.reuse >= acq.slotLen || !m.hasRecurrentState() {
		return
	}
	rc := m.recurrentCheckpointer()
	switch {
	case rc == nil:
		acq.reuse, acq.ckptDeclined = 0, "no checkpoint kept"
	case len(prompt)-acq.reuse < m.prefillTailMinRows():
		acq.reuse, acq.ckptDeclined = 0, "suffix under the final-pass minimum"
	default:
		t0 := time.Now()
		if err := rc.RestoreRecurrentCheckpoint(); err != nil {
			acq.reuse, acq.ckptDeclined = 0, "restore failed: "+err.Error()
			return
		}
		acq.ckptRestored, acq.ckptDur = acq.reuse, time.Since(t0)
	}
}

// residentCheckpointSave saves the live recurrent state as the checkpoint for prompt, which the generation has just prefilled, and
// returns how long the copy took. A model or resident without one, or a failed copy, records nothing: the next request cannot
// restore from it.
func (m *Model) residentCheckpointSave(prompt []int) (time.Duration, bool) {
	m.resCkptPrompt = nil
	rc := m.recurrentCheckpointer()
	if rc == nil {
		return 0, false
	}
	t0 := time.Now()
	if err := rc.SaveRecurrentCheckpoint(); err != nil {
		return 0, false
	}
	m.resCkptPrompt = slices.Clone(prompt)
	return time.Since(t0), true
}
