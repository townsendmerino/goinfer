//go:build gpu

package gpu

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S8c (S8, docs/tasks/task-multimodal-support-2026-10.md): this build's live tower registrations equal WebGPU's row of
// multimodal's declaration (SigLIP only), and WebGPU implements no hybrid image prefill, as decoder declares.
func TestSupportTable_webgpuDeclarations(t *testing.T) {
	live, declared := multimodal.LiveTowers("webgpu"), multimodal.DeclaredTowers("webgpu")
	if missing, undeclared := multimodal.TowerDrift(live, declared); len(missing)+len(undeclared) > 0 {
		t.Errorf("WebGPU towers: declared but not registered %v, registered but not declared %v (live %v)", missing, undeclared, live)
	}
	_, hybrid := any((*residentDecoder)(nil)).(decoder.ResidentHybridMRoPEPrefill)
	if hybrid != decoder.HybridImagePrefillDeclared("webgpu", false) {
		t.Errorf("WebGPU implements ResidentHybridMRoPEPrefill: %v; decoder declares it: %v", hybrid, decoder.HybridImagePrefillDeclared("webgpu", false))
	}
}
