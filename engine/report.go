package engine

import (
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// Report is a generation's account as the shell reads it: the budget its finish reason is judged against, its
// logprobs, how much of the prompt it reused, copied or restored, and its error.
type Report struct {
	Budget        int  // the budget the engine clamped the turn to; meaningful only when BudgetClamped
	BudgetClamped bool // a clamped budget can be a genuine 0 (the prompt fills the context)
	Logprobs      []decoder.SampleInfo

	PrefillReused, PrefillSlot, PrefillMatch int
	PrefillShared                            int
	KVCopy                                   time.Duration
	KVCopyDeclined                           string
	KVSnapRestored                           int
	CheckpointRestored                       int
	CheckpointRestore                        time.Duration
	CheckpointDeclined                       string
	CheckpointSaved                          bool
	CheckpointSave                           time.Duration

	Err error
}

// BudgetFor is the token budget a finish reason is judged against: the clamped budget when the engine clamped the
// turn (a resident context cap), else the requested max_tokens. A generation truncated at the cap emits fewer tokens
// than requested, so judging against the request would report "stop" instead of "length" and the client would never
// continue. A nil report (an engine or path that does not clamp) falls back to the request.
func (r *Report) BudgetFor(requested int) int {
	if r != nil && r.BudgetClamped {
		return r.Budget
	}
	return requested
}

// reportOf is gen's account now; nil for a nil gen.
func reportOf(gen *decoder.Generation) *Report {
	if gen == nil {
		return nil
	}
	return &Report{
		Budget: gen.Budget, BudgetClamped: gen.BudgetClamped, Logprobs: gen.Logprobs,
		PrefillReused: gen.PrefillReused, PrefillSlot: gen.PrefillSlot, PrefillMatch: gen.PrefillMatch,
		PrefillShared: gen.PrefillShared, KVCopy: gen.KVCopy, KVCopyDeclined: gen.KVCopyDeclined,
		KVSnapRestored:     gen.KVSnapRestored,
		CheckpointRestored: gen.CheckpointRestored, CheckpointRestore: gen.CheckpointRestore,
		CheckpointDeclined: gen.CheckpointDeclined, CheckpointSaved: gen.CheckpointSaved, CheckpointSave: gen.CheckpointSave,
		Err: gen.Err(),
	}
}
