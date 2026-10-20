package decoder

import (
	"errors"
	"slices"
	"testing"
)

// ckptResident is a one-slot resident with recurrent state that keeps a KV8a checkpoint: live is the state, saved the copy.
type ckptResident struct {
	*slotResident
	bytes             int64
	live, saved       []float32
	saves, restores   int
	failSave, failRes bool
}

func newCkptResident() *ckptResident {
	return &ckptResident{slotResident: &slotResident{n: 1}, bytes: 4096, live: []float32{1, 2, 3}}
}

func (r *ckptResident) RecurrentCheckpointBytes() int64 { return r.bytes }
func (r *ckptResident) SaveRecurrentCheckpoint() error {
	if r.failSave {
		return errors.New("ckptResident: forced save failure")
	}
	r.saves++
	r.saved = slices.Clone(r.live)
	return nil
}
func (r *ckptResident) RestoreRecurrentCheckpoint() error {
	if r.failRes {
		return errors.New("ckptResident: forced restore failure")
	}
	r.restores++
	r.live = slices.Clone(r.saved)
	return nil
}

// ckptTurn plays one generation's bookkeeping on m as generateInto does on the resident path: acquire under a checkpoint scope,
// forget, (prefill), save, commit. It returns the acquire.
func ckptTurn(m *Model, prompt, reply []int, lora *loraRuntime) kvAcquired {
	acq := m.residentAcquireIn(prompt, nil, lora, nil, KVScope{checkpoint: m.recurrentCheckpointer() != nil})
	m.residentForgetIDs()
	m.residentCheckpointSave(prompt)
	m.residentCommitIDs(prompt, reply, nil, lora)
	return acq
}

// TestCacheStateGrid_checkpointCells checks KV8a's two paths, save and restore, against the code they declare, through the functions a
// generation calls: every kind's cell on both paths must name a handling this test has a check for.
func TestCacheStateGrid_checkpointCells(t *testing.T) {
	prompt := kvSeq(200, 1)
	reply := kvSeq(30, 2)
	// Turn 2 as KV0-H's B requests are: the previous prompt, the reply rendered differently 15 tokens in, and a new message.
	rerender := append(append(slices.Clone(prompt), reply[:15]...), kvSeq(40, 3)...)
	recurrent := func(r ResidentForward) *Model {
		m := recurrentTestModel(nil)
		m.resident = r
		return m
	}
	checks := map[lifecyclePath]map[stateHandling]func(t *testing.T, s cacheState){
		lcCkptSave: {
			hExact: func(t *testing.T, s cacheState) {
				r := newCkptResident()
				m := recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				if r.saves != 1 || !slices.Equal(r.saved, r.live) {
					t.Fatalf("%d saves, saved %v of live %v: want one copy of the live state", r.saves, r.saved, r.live)
				}
				if c, _ := m.kv.Seq(m.kv.Bound()).Checkpoint(); c != len(prompt) {
					t.Errorf("the committed sequence records a checkpoint at %d, want the prompt's end %d", c, len(prompt))
				}
				// A failed save records nothing, and a resident that keeps no checkpoint saves nothing.
				r = newCkptResident()
				r.failSave = true
				m = recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				if c, _ := m.kv.Seq(m.kv.Bound()).Checkpoint(); c != 0 {
					t.Errorf("after a failed save the sequence records a checkpoint at %d", c)
				}
				r = newCkptResident()
				r.bytes = 0
				m = recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				if r.saves != 0 {
					t.Errorf("a resident that keeps no checkpoint was asked to save %d times", r.saves)
				}
			},
			hKept: func(t *testing.T, s cacheState) {
				switch s {
				case stateAdapter:
					lora := &loraRuntime{name: "a"}
					m := recurrent(newCkptResident())
					ckptTurn(m, prompt, reply, lora)
					if got := m.kv.Seq(m.kv.Bound()).Root().Adapter; got != kvAdapterID(lora) {
						t.Errorf("the checkpoint's sequence is keyed under adapter %d, the generation's is %d", got, kvAdapterID(lora))
					}
				default: // a positional kind: nothing is copied, the rows stay, and a positional model takes its usual prefix
					m := &Model{resident: &slotResident{n: 1}}
					ckptTurn(m, prompt, reply, nil)
					if got := m.residentAcquire(rerender, nil, nil); got != len(prompt)+15 {
						t.Errorf("a positional model reuses %d rows of the re-rendered turn, want the common prefix %d", got, len(prompt)+15)
					}
				}
			},
		},
		lcCkptRestore: {
			hExact: func(t *testing.T, s cacheState) {
				r := newCkptResident()
				m := recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				saved := slices.Clone(r.live)
				r.live = []float32{9, 9, 9} // the decode moved the live state past the prompt
				acq := ckptTurn(m, rerender, reply, nil)
				if acq.reuse != len(prompt) || acq.ckptRestored != len(prompt) || r.restores != 1 {
					t.Fatalf("reuse %d, restored at %d, %d restores: want the checkpoint at %d restored once", acq.reuse, acq.ckptRestored, r.restores, len(prompt))
				}
				if !slices.Equal(r.live, saved) {
					t.Errorf("after the restore the live state is %v, want turn 1's %v", r.live, saved)
				}
				// A strict extension continues the live state with no restore.
				r = newCkptResident()
				m = recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				ext := append(append(slices.Clone(prompt), reply...), kvSeq(20, 4)...)
				if acq := ckptTurn(m, ext, nil, nil); acq.reuse != len(prompt)+len(reply) || r.restores != 0 {
					t.Errorf("a strict extension: reuse %d with %d restores, want %d with none", acq.reuse, r.restores, len(prompt)+len(reply))
				}
				// A divergence before c, a resent prompt, and a suffix under the final-pass minimum stay cold.
				for name, p := range map[string][]int{
					"a divergence before c": append(append(slices.Clone(prompt[:150]), 7), kvSeq(80, 5)...),
					"the prompt resent":     prompt,
					"a 3-token suffix":      append(slices.Clone(prompt), reply[0], 1, 2),
				} {
					r := newCkptResident()
					m := recurrent(r)
					ckptTurn(m, prompt, reply, nil)
					if acq := ckptTurn(m, p, nil, nil); acq.reuse != 0 || r.restores != 0 {
						t.Errorf("%s: reuse %d with %d restores, want a cold prefill", name, acq.reuse, r.restores)
					}
				}
				// A failed restore falls back to a cold prefill.
				r = newCkptResident()
				m = recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				r.failRes = true
				if acq := ckptTurn(m, rerender, nil, nil); acq.reuse != 0 || acq.ckptDeclined == "" {
					t.Errorf("a failed restore: reuse %d, declined %q, want 0 and a reason", acq.reuse, acq.ckptDeclined)
				}
			},
			hExactUnlessWrapped: func(t *testing.T, s cacheState) {
				gone := &ckptWindowResident{ckptResident: newCkptResident(), dropped: true}
				m := recurrent(gone)
				ckptTurn(m, prompt, reply, nil)
				if acq := ckptTurn(m, rerender, nil, nil); acq.reuse != 0 || gone.restores != 0 {
					t.Errorf("with the window dropped: reuse %d with %d restores, want a cold prefill and no restore", acq.reuse, gone.restores)
				}
			},
			hColdOnMismatch: func(t *testing.T, s cacheState) {
				if s != stateAdapter {
					t.Fatalf("%q is declared cold-on-mismatch on restore and this test has no mismatch to try for it", s)
				}
				r := newCkptResident()
				m := recurrent(r)
				ckptTurn(m, prompt, reply, &loraRuntime{name: "a"})
				if acq := ckptTurn(m, rerender, nil, &loraRuntime{name: "a"}); acq.reuse != 0 || r.restores != 0 {
					t.Errorf("another adapter runtime: reuse %d with %d restores, want 0", acq.reuse, r.restores)
				}
			},
			hRefused: func(t *testing.T, s cacheState) {
				// An image span on either side refuses the checkpoint (kvmgr's scoreRecurrent); checked here through the manager, since
				// the restoring path takes text prompts only.
				r := newCkptResident()
				m := recurrent(r)
				ckptTurn(m, prompt, reply, nil)
				claims := []residentImageClaim{{Start: len(prompt) + 2, Len: 4, Hash: 5}}
				acq := m.residentAcquireIn(rerender, claims, nil, nil, KVScope{checkpoint: true})
				if acq.reuse != 0 || r.restores != 0 {
					t.Errorf("a request with an image span: reuse %d with %d restores, want 0", acq.reuse, r.restores)
				}
			},
		},
	}
	for _, p := range []lifecyclePath{lcCkptSave, lcCkptRestore} {
		for _, s := range cacheStates {
			h := cacheStateGrid[s][p].h
			if h == hTransient {
				continue // nothing of the kind is on this path; the cell's reason says why
			}
			check, ok := checks[p][h]
			if !ok {
				t.Errorf("%q x %q is %q, and this test has no check for that handling", s, p, h)
				continue
			}
			t.Run(string(p)+"/"+string(s), func(t *testing.T) { check(t, s) })
		}
	}
}

// ckptWindowResident is a ckptResident whose windowed layers have dropped the rows a restart at c would attend to.
type ckptWindowResident struct {
	*ckptResident
	dropped bool
}

func (w *ckptWindowResident) ReusableKV(prefix int) int {
	if w.dropped {
		return 0
	}
	return prefix
}
