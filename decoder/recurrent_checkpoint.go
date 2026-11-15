package decoder

import (
	"slices"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// KV8a (docs/tasks/task-kv-architecture-2026-10.md, KV8a): one recurrent checkpoint for the one resident slot a hybrid model has.
// The generation saves the recurrent state once its prefill has processed the whole prompt, before the first decode step, at
// c = len(prompt). A next request that matches the slot's sequence through c but does not extend all of it restores the state
// and prefills from c (kvmgr's scoreRecurrentTokens), where today it prefills from 0. The attention KV is positional and stays.

// checkpointHeld reports whether slot holds a checkpoint a request on a load of several slots may score.
func (m *Model) checkpointHeld(slot int) bool {
	rc := m.recurrentCheckpointer()
	return rc != nil && rc.RecurrentCheckpointHeld(slot)
}

// recurrentCheckpointer is the resident's checkpoint, or nil when the model has no recurrent state or the resident keeps none.
func (m *Model) recurrentCheckpointer() ResidentKVSubsystem {
	if m.resident == nil || !m.hasRecurrentState() {
		return nil
	}
	kv := m.resKV()
	if kv.RecurrentCheckpointBytes() <= 0 {
		return nil
	}
	return kv
}

// residentCheckpointRestore makes the live recurrent state the checkpoint's when the acquire scored a restart from it: a recurrent
// reuse shorter than the slot's sequence. It falls back to a cold prefill (reuse 0) when the suffix is shorter than the resident's
// final-pass minimum, where the suffix would run other kernels than a cold prefill runs over those rows, and when the restore fails.
func (m *Model) residentCheckpointRestore(prompt []int, acq *kvAcquired) {
	rc := m.recurrentCheckpointer()
	nextReuse, restore, declined := kvmgr.DecideRecurrentCheckpoint(
		acq.reuse, acq.slotLen, len(prompt), m.prefillTailMinRows(), m.hasRecurrentState(), rc != nil,
	)
	if declined != "" {
		acq.reuse, acq.ckptDeclined = nextReuse, declined
		return
	}
	if !restore {
		return
	}
	t0 := time.Now()
	if err := rc.RestoreRecurrentCheckpoint(); err != nil {
		acq.reuse, acq.ckptDeclined = 0, "restore failed: "+err.Error()
		return
	}
	acq.ckptRestored, acq.ckptDur = acq.reuse, time.Since(t0)
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
