package decoder

import "testing"

// TestSpecRollbackSafe_refusesWindowedAndRecurrent is the C-04 regression gate. specRollbackSafe used
// to exempt sliding-window models only when m.resident == nil, but three of four speculative loops
// (EAGLE, grammar always; n-gram under a Session) drive the STAGED ring even when a resident exists,
// and a resident CAS loss also falls back to staged — so a windowed ring could wrap and a >1-position
// rollback would read stale history, breaking losslessness. The fix refuses windowed models
// unconditionally (and recurrent-state families, which have no per-position history to rewind). This
// pins that: a change reopening windowed/recurrent speculation without consuming TruncateTo's exact
// bool at every rollback site fails here.
func TestSpecRollbackSafe_refusesWindowedAndRecurrent(t *testing.T) {
	mk := func(a *Architecture) *Model { return &Model{w: &Weights{arch: a}} }
	for _, c := range []struct {
		name string
		arch *Architecture
		want bool
	}{
		{"plain (no window, no recurrent)", &Architecture{}, true},
		{"sliding-window (Gemma-3 local / Mistral / Phi-3)", &Architecture{SlidingWindow: 512}, false},
		{"sliding-window even with a resident-eligible dense arch", &Architecture{SlidingWindow: 4096}, false},
		{"recurrent: granite (Mamba-2)", &Architecture{granite: &graniteParams{}}, false},
		{"recurrent: nemotron-h", &Architecture{nemotron: &nemotronParams{}}, false},
		{"recurrent: qwen3_5_moe (Gated DeltaNet)", &Architecture{qwen35: &qwen35Params{}}, false},
	} {
		if got := mk(c.arch).specRollbackSafe(); got != c.want {
			t.Errorf("%s: specRollbackSafe = %v, want %v", c.name, got, c.want)
		}
	}
}
