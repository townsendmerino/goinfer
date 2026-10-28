package engine

import (
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TokenLogprob is a token id and its log-probability.
type TokenLogprob struct {
	ID      int
	Logprob float64
}

// SampleInfo is one generated token's logprob and, when asked for, its top alternatives.
type SampleInfo struct {
	ID      int
	Logprob float64
	Top     []TokenLogprob
}

// Report is a generation's account as the shell reads it: the budget its finish reason is judged against, its
// logprobs, how much of the prompt it reused, copied or restored, and its error.
type Report struct {
	Budget        int  // the budget the engine clamped the turn to; meaningful only when BudgetClamped
	BudgetClamped bool // a clamped budget can be a genuine 0 (the prompt fills the context)
	Logprobs      []SampleInfo

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

// ReportOf is gen's account now; nil for a nil gen.
func ReportOf(gen *decoder.Generation) *Report {
	if gen == nil {
		return nil
	}
	return &Report{
		Budget: gen.Budget, BudgetClamped: gen.BudgetClamped, Logprobs: LogprobsOf(gen.Logprobs),
		PrefillReused: gen.PrefillReused, PrefillSlot: gen.PrefillSlot, PrefillMatch: gen.PrefillMatch,
		PrefillShared: gen.PrefillShared, KVCopy: gen.KVCopy, KVCopyDeclined: gen.KVCopyDeclined,
		KVSnapRestored:     gen.KVSnapRestored,
		CheckpointRestored: gen.CheckpointRestored, CheckpointRestore: gen.CheckpointRestore,
		CheckpointDeclined: gen.CheckpointDeclined, CheckpointSaved: gen.CheckpointSaved, CheckpointSave: gen.CheckpointSave,
		Err: gen.Err(),
	}
}

// Reporter is gen's report function, read afresh at each call as gen's fields were.
func Reporter(gen *decoder.Generation) func() *Report { return func() *Report { return ReportOf(gen) } }

// ReportNow is report's account now, or nil when there is no report function.
func ReportNow(report func() *Report) *Report {
	if report == nil {
		return nil
	}
	return report()
}

// LogprobsOf is the decoder's per-token logprobs as SampleInfo, nil for nil and empty for empty at both levels.
func LogprobsOf(in []decoder.SampleInfo) []SampleInfo {
	if in == nil {
		return nil
	}
	out := make([]SampleInfo, len(in))
	for i, s := range in {
		out[i] = SampleInfo{ID: s.ID, Logprob: s.Logprob}
		if s.Top != nil {
			out[i].Top = make([]TokenLogprob, len(s.Top))
			for j, t := range s.Top {
				out[i].Top[j] = TokenLogprob(t)
			}
		}
	}
	return out
}
