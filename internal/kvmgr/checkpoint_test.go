package kvmgr

import "testing"

// KV8a (docs/tasks/task-kv-architecture-2026-10.md, KV8a): a recurrent sequence with a checkpoint at c, the end of its last prompt,
// scores c for a request that matches it through c, carries a token past c and does not extend the whole sequence, when the caller
// restores the checkpoint. Every other request scores as before.
func TestScoreRecurrent_checkpoint(t *testing.T) {
	root := Root{Model: "m"}
	prompt := []int{1, 2, 3, 4, 5, 6}
	held := append(append([]int(nil), prompt...), 70, 71, 72) // the prompt, then the reply as generated
	c := len(prompt)
	newSlots := func(ckpt int, imgs []ImageSpan) *Slots {
		a := NewSlots(1)
		a.Commit(0, root, held, imgs, c, ClassMap{})
		a.SetCheckpoint(0, ckpt)
		return a
	}
	img := []ImageSpan{{Start: 1, Len: 2, Hash: 9}}
	for _, tc := range []struct {
		name string
		ckpt int
		imgs []ImageSpan
		req  []int
		opt  bool
		want int
	}{
		{"the reply re-rendered differently (B): restore at c", c, nil, []int{1, 2, 3, 4, 5, 6, 70, 99, 100}, true, c},
		{"one token past c", c, nil, []int{1, 2, 3, 4, 5, 6, 99}, true, c},
		{"a strict extension continues the live state", c, nil, append(append([]int(nil), held...), 80), true, len(held)},
		{"the prompt sent again (identical re-request)", c, nil, prompt, true, 0},
		{"a divergence before c", c, nil, []int{1, 2, 3, 4, 9, 6, 70, 71}, true, 0},
		{"no checkpoint held", 0, nil, []int{1, 2, 3, 4, 5, 6, 70, 99}, true, 0},
		{"the caller does not restore", c, nil, []int{1, 2, 3, 4, 5, 6, 70, 99}, false, 0},
		{"an image span in the sequence", c, img, []int{1, 2, 3, 4, 5, 6, 70, 99}, true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newSlots(tc.ckpt, tc.imgs)
			if got := a.Score(0, Request{Root: root, Tokens: tc.req, Checkpoint: tc.opt}, true); got != tc.want {
				t.Errorf("Score = %d, want %d", got, tc.want)
			}
		})
	}
	t.Run("an image claim in the request", func(t *testing.T) {
		a := newSlots(c, nil)
		req := Request{Root: root, Tokens: []int{1, 2, 3, 4, 5, 6, 70, 99}, Images: []ImageSpan{{Start: 7, Len: 1, Hash: 3}}, Checkpoint: true}
		if got := a.Score(0, req, true); got != 0 {
			t.Errorf("Score = %d, want 0", got)
		}
	})
	t.Run("another root", func(t *testing.T) {
		a := newSlots(c, nil)
		if got := a.Score(0, Request{Root: Root{Model: "other"}, Tokens: []int{1, 2, 3, 4, 5, 6, 99}, Checkpoint: true}, true); got != 0 {
			t.Errorf("Score = %d, want 0", got)
		}
	})
	t.Run("a positional model ignores it", func(t *testing.T) {
		a := newSlots(c, nil)
		req := []int{1, 2, 3, 4, 5, 6, 70, 99}
		if got := a.Score(0, Request{Root: root, Tokens: req, Checkpoint: true}, false); got != 7 {
			t.Errorf("Score = %d, want the common prefix 7", got)
		}
	})
}

// The checkpoint belongs to one committed sequence: a commit or a forget of the slot drops it, and its key is (chain hash at c, c).
func TestSetCheckpoint_lifetimeAndKey(t *testing.T) {
	root := Root{Model: "m"}
	a := NewSlots(1)
	a.Commit(0, root, []int{1, 2, 3, 4}, nil, 2, ClassMap{})
	a.SetCheckpoint(0, 2)
	if c, k := a.Seq(0).Checkpoint(); c != 2 || k != checkpointKey(a.Seq(0).rootHash, []int{1, 2}) || k == (Hash{}) {
		t.Fatalf("checkpoint %d %x", c, k)
	}
	b := NewSlots(1)
	b.Commit(0, root, []int{1, 9, 3, 4}, nil, 2, ClassMap{})
	b.SetCheckpoint(0, 2)
	if _, kb := b.Seq(0).Checkpoint(); kb == func() Hash { _, k := a.Seq(0).Checkpoint(); return k }() {
		t.Error("two different prefixes share a key")
	}
	for _, bad := range []int{0, -1, 5} {
		a.SetCheckpoint(0, bad)
		if c, _ := a.Seq(0).Checkpoint(); c != 0 {
			t.Errorf("SetCheckpoint(%d) recorded %d", bad, c)
		}
	}
	a.SetCheckpoint(0, 2)
	a.Commit(0, root, []int{1, 2, 3, 4, 5, 6}, nil, 5, ClassMap{})
	if c, _ := a.Seq(0).Checkpoint(); c != 0 {
		t.Errorf("a commit kept the earlier sequence's checkpoint at %d", c)
	}
	a.SetCheckpoint(0, 5)
	a.Forget(0)
	if s := a.Seq(0); s != nil {
		if c, _ := s.Checkpoint(); c != 0 {
			t.Errorf("a forget kept the checkpoint at %d", c)
		}
	}
}
