//go:build darwin

package metal

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S8c (S8, docs/tasks/task-multimodal-support-2026-10.md): this build's live tower registrations equal Metal's row of
// multimodal's declaration, and Metal implements the hybrid image prefill exactly where decoder declares it, so the
// support table's Metal column is true. Planted: a declaration with one tower dropped must be reported.
func TestSupportTable_metalDeclarations(t *testing.T) {
	var extra []string
	if slices.Contains(embeddinggemma2.AudioAccelerators(), "metal") {
		extra = append(extra, multimodal.TowerGemma4Audio)
	}
	if slices.Contains(embeddinggemma2.Accelerators(), "metal") {
		extra = append(extra, multimodal.TowerEmbeddingGemma2)
	}
	live, declared := multimodal.LiveTowers("metal", extra...), multimodal.DeclaredTowers("metal")
	if missing, undeclared := multimodal.TowerDrift(live, declared); len(missing)+len(undeclared) > 0 {
		t.Errorf("Metal towers: declared but not registered %v, registered but not declared %v (live %v)", missing, undeclared, live)
	}
	_, hybrid := any((*metalResident)(nil)).(decoder.ResidentHybridMRoPEPrefill)
	if hybrid != decoder.HybridImagePrefillDeclared("metal", false) {
		t.Errorf("Metal implements ResidentHybridMRoPEPrefill: %v; decoder declares it for Metal: %v", hybrid, decoder.HybridImagePrefillDeclared("metal", false))
	}
	if _, undeclared := multimodal.TowerDrift(live, declared[1:]); len(undeclared) != 1 || undeclared[0] != declared[0] {
		t.Errorf("planted: a tower dropped from the declaration (%s) was not reported as registered but undeclared: %v", declared[0], undeclared)
	}
}
