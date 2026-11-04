//go:build cuda

package cuda

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S8c (S8, docs/tasks/task-multimodal-support-2026-10.md): this build's live tower registrations equal CUDA's row of
// multimodal's declaration, and CUDA implements the hybrid image prefill exactly where decoder declares it (dense hybrids).
func TestSupportTable_cudaDeclarations(t *testing.T) {
	var extra []string
	if slices.Contains(embeddinggemma2.AudioAccelerators(), "cuda") {
		extra = append(extra, multimodal.TowerGemma4Audio)
	}
	if slices.Contains(embeddinggemma2.Accelerators(), "cuda") {
		extra = append(extra, multimodal.TowerEmbeddingGemma2)
	}
	live, declared := multimodal.LiveTowers("cuda", extra...), multimodal.DeclaredTowers("cuda")
	if missing, undeclared := multimodal.TowerDrift(live, declared); len(missing)+len(undeclared) > 0 {
		t.Errorf("CUDA towers: declared but not registered %v, registered but not declared %v (live %v)", missing, undeclared, live)
	}
	_, hybrid := any((*cudaResident)(nil)).(decoder.ResidentHybridMRoPEPrefill)
	if hybrid != decoder.HybridImagePrefillDeclared("cuda", false) {
		t.Errorf("CUDA implements ResidentHybridMRoPEPrefill: %v; decoder declares it for CUDA's dense hybrids: %v", hybrid, decoder.HybridImagePrefillDeclared("cuda", false))
	}
}
