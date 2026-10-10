package decoder

import "testing"

// TestFastAttnPromptFloor pins the prompt-length floor, which keeps flipping the default from changing the output of
// every SHORT request for no speed: a short prompt diverges from acc64 within a few tokens while the win at that
// length is nil.
//
// Deliberately NOT here: a MoE exclusion. It was removed after it was measured (a3_moe_*_test.go), and re-adding one
// without a new measurement would reverse a decision made with evidence.
func TestFastAttnPromptFloor(t *testing.T) {
	if fastAttnMinPrompt <= 1 {
		t.Fatalf("floor %d does not floor anything — every prompt takes the f32 path", fastAttnMinPrompt)
	}
	// A floor set absurdly high would silently retire the flag instead of gating it.
	if fastAttnMinPrompt > 8192 {
		t.Errorf("floor %d is above the length the flag's own 2.28x win was measured at", fastAttnMinPrompt)
	}
}

// TestCPUFastAttentionDefaultOn pins the decoder-side default, which is the one that decides
// behaviour for library callers who never touch the server flags.
func TestCPUFastAttentionDefaultOn(t *testing.T) {
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "")
	if !cpuFastAttention() {
		t.Error("unset must mean ON (default flipped 2026-08-31)")
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	if cpuFastAttention() {
		t.Error(`"0" must turn it off — that is what --cpu-exact-prefill sets`)
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "1")
	if !cpuFastAttention() {
		t.Error(`an explicit "1" must still mean ON — the sense of the old opt-in must not invert`)
	}
}
