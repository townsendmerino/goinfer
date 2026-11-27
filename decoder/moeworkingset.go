package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/paging"
)

// moeSlowTokPerSecThreshold is the refusal floor for paged and mapped CPU MoE loads; tests may override it via
// testgate.SetForTest when verifying wiring on tiny fixtures whose real byte counts always predict a fast rate.
var moeSlowTokPerSecThreshold = mem.MoESlowTokPerSecFloor

// moeWorkingSetPrediction is predictMoEWorkingSet for a just-loaded Model and its just-built expert pager.
func moeWorkingSetPrediction(m *Model) (tokPerSec float64, ok bool) {
	if m == nil || m.pager == nil {
		return 0, false
	}
	return predictMoEWorkingSet(m, m.pager.TotalMappedBytes(), m.pager.ExpertCount(), m.pager.Budget())
}

// mappedMoEWorkingSetPrediction is moeWorkingSetPrediction for a MoE load that decodes from its mapping with no pager: the
// page cache holds the experts, priced at the budget a pager would take for requested on this machine (pagerBudget over
// mappedExpertBytes), so the same arithmetic and floor apply to both.
func mappedMoEWorkingSetPrediction(m *Model, requested int64) (tokPerSec float64, ok bool) {
	if m == nil || m.w == nil {
		return 0, false
	}
	total, maxExpert, n := paging.MappedExpertBytes(m.w, m.mmap)
	if total <= 0 {
		return 0, false
	}
	return predictMoEWorkingSet(m, total, n, paging.PagerBudget(requested, total, maxExpert))
}

func predictMoEWorkingSet(m *Model, total int64, nExperts int, budget int64) (tokPerSec float64, ok bool) {
	if m == nil || m.w == nil || m.w.Arch == nil || m.w.Arch.MoE == nil {
		return 0, false
	}
	return mem.PredictMoEWorkingSet(total, nExperts, len(m.w.Layers), m.w.Arch.MoE.TopK, m.ResidentDenseWeightBytes(), budget)
}

func moeWorkingSetRefusal(modelName string, predicted float64, acceptSlow bool) error {
	return mem.MoEWorkingSetRefusal(modelName, predicted, moeSlowTokPerSecThreshold, acceptSlow)
}
