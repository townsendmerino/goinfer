package decoder

import "testing"

// C-11: THE BLOCK-SPEC LOOP'S STOP SET MUST EQUAL THE ONE PLAIN DECODING USES.
//
// Every other speculative loop calls target.isStop(tok, sp). This one rebuilt the set from
// Cfg.EOSIDs() alone — config.json's eos_token_id, missing generation_config.json's additions
// (which resolveEOSIDs merges into m.eosIDs) and the caller's StopIDs (the chat template's stops on
// a served request). On the shipped pairing that is {151645} against {151645, 151643}: an
// <|endoftext|> came out as content, generation continued to <|im_end|> or max_tokens, and
// streamTokens decoded the stop token into the response.
//
// The invariant is AGREEMENT, so the test compares the two predicates directly rather than
// re-listing what the set should contain — a list would be a second copy of the same belief.
func TestBlockSpec_stopSetAgreesWithPlainDecoding(t *testing.T) {
	m := &Model{eosIDs: []int{151645, 151643}} // config.json + generation_config.json, merged
	sp := SamplingParams{StopIDs: []int{151668}}
	opt := BlockSpecOptions{StopIDs: sp.StopIDs}

	got := blockSpecStopSet(m, opt)
	for id := range 151700 {
		if got[id] != m.isStop(id, sp) {
			t.Fatalf("token %d: block-spec stop=%v, plain decoding stop=%v — the two paths would "+
				"end the turn at different tokens, which is what makes 'lossless by construction' "+
				"false", id, got[id], m.isStop(id, sp))
		}
	}

	// The specific regression: the generation_config-only id must stop. Cfg.EOSIDs() omits it.
	if !got[151643] {
		t.Error("151643 (generation_config.json's eos) is not a stop: it would be emitted as " +
			"content and generation would run on")
	}
	// And the caller's own stop, which no EOS list carries.
	if !got[151668] {
		t.Error("the caller's SamplingParams.StopIDs are not honoured")
	}
}

// M-13: a round commits up to `width` tokens, and the budget was checked once per ROUND.
func TestBlockSpec_roundWidthRespectsBothBudgets(t *testing.T) {
	const w = 8
	// max_tokens=2 with 8-wide rounds: the loop condition passes at len(out)=1 and the round then
	// commits 8 more, so the request's own cap is exceeded and usage.completion_tokens with it.
	if got := blockSpecRoundWidth(w, 2, 1, 100, 0); got != 1 {
		t.Errorf("width with 1 of 2 tokens emitted = %d, want 1 — a round commits its whole burst, "+
			"so max_tokens=2 could return 9", got)
	}
	if got := blockSpecRoundWidth(w, 0, 100, 100, 0); got != w {
		t.Errorf("unlimited max_tokens clamped to %d, want %d", got, w)
	}
	// The context cap: verifying width rows at pos with no clamp makes the backend refuse the WHOLE
	// round, so a nearly complete response ends in an error instead of a clean "length".
	if got := blockSpecRoundWidth(w, 0, 0, 4090, 4096); got != 6 {
		t.Errorf("width at pos 4090 of a 4096 cap = %d, want 6", got)
	}
	// Exhausted: < 1 tells the caller to finish cleanly rather than ask for a refused round.
	if got := blockSpecRoundWidth(w, 4, 4, 100, 0); got >= 1 {
		t.Errorf("width with the budget spent = %d, want < 1", got)
	}
	if got := blockSpecRoundWidth(w, 0, 0, 4096, 4096); got >= 1 {
		t.Errorf("width at the context cap = %d, want < 1", got)
	}
	// The tighter of the two binds.
	if got := blockSpecRoundWidth(w, 100, 97, 4094, 4096); got != 2 {
		t.Errorf("width = %d, want 2 (the context cap is tighter than the 3 tokens left)", got)
	}
}
