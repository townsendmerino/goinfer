package decoder

import (
	"io"

	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

type (
	SpecTrace          = specdecode.SpecTrace
	TraceCollector     = specdecode.TraceCollector
	SpecStats          = specdecode.SpecStats
	AdaptiveDepth      = specdecode.AdaptiveDepth
	VerifyCostReporter = specdecode.VerifyCostReporter
	OptFwdStats        = specdecode.OptFwdStats
	ConfidentDrafter   = specdecode.ConfidentDrafter
	OutcomeRecorder    = specdecode.OutcomeRecorder
	RouterDrafter      = specdecode.RouterDrafter
	GrammarDrafter     = specdecode.GrammarDrafter
	Drafter            = specdecode.Drafter
	NgramDrafter       = specdecode.NgramDrafter
	DraftInfo          = specdecode.DraftInfo
	TracingDrafter     = specdecode.TracingDrafter

	specTracer   = specdecode.Tracer
	optFwdGate   = specdecode.OptFwdGate
	optFwdResult = specdecode.OptFwdResult[SampleInfo]
	ngramTarget  = specdecode.NgramTarget
)

const (
	defaultSpecDepth        = specdecode.DefaultSpecDepth
	defaultGrammarSpecDepth = specdecode.DefaultGrammarSpecDepth
	optFwdMaxTemp           = specdecode.OptFwdMaxTemp
)

// NewTraceCollector returns a collector that also writes each row as JSON to w if w is non-nil.
func NewTraceCollector(w io.Writer) *TraceCollector { return specdecode.NewCollector(w) }

type ngramSamplerAdapter struct{ s *Sampler }

func (a ngramSamplerAdapter) DrawTarget(logits []float32, hist []int) int {
	return a.s.drawTarget(logits, hist)
}
func (a ngramSamplerAdapter) DistVector(logits []float32) []float64 {
	return a.s.distVector(logits)
}
func (a ngramSamplerAdapter) DistVectorHist(logits []float32, hist []int) []float64 {
	return a.s.distVectorHist(logits, hist)
}
func (a ngramSamplerAdapter) SpecStep(p []float64, draftTok int) (int, bool) {
	return a.s.specStep(p, draftTok)
}
