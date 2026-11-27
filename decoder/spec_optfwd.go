package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/specdecode"

// optFwdEligible reports whether sampled decode may attempt the optimistic-forward overlap.
func (m *Model) optFwdEligible(sp SamplingParams) bool {
	return m.resident != nil && sp.Temperature > 0 && sp.Temperature <= m.knobs.optFwdTempCap() &&
		sp.LogitProcessor == nil && m.specRollbackSafe()
}

// optFwdStep runs one sampled-decode position's optimistic-forward overlap.
func (m *Model) optFwdStep(sampler *Sampler, logits []float32, gpuPos int, gate *optFwdGate, stats *OptFwdStats) (optFwdResult, error) {
	fwd := func(tok, pos int) ([]float32, error) {
		return m.resident.Forward(m.embedResident(tok), pos)
	}
	return specdecode.OptFwdStep(fwd, sampler.SampleWithInfo, func(info SampleInfo) int { return info.ID }, logits, gpuPos, gate, stats)
}
