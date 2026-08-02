package decoder

import "testing"

// TestGemma4Admission_envGated is the 9a-P2 admission gate. Dense Gemma 4 is bridged to the
// resident runner ONLY behind GOINFER_GEMMA4_RESIDENT (Split-A bring-up, granite precedent);
// the enable_moe_block variant is declined until Split B. This asserts all four corners so a
// regenerated capability/hardware matrix cannot quietly claim residency users cannot reach —
// the matrix's GPUResident column is arch.decodeRunnerEligible() (capability_matrix_test.go),
// exactly what this pins.
func TestGemma4Admission_envGated(t *testing.T) {
	denseArch := func() *Architecture {
		a, _, err := resolveArchitecture(representativeConfig("gemma4"))
		if err != nil {
			t.Fatalf("resolve gemma4 (dense): %v", err)
		}
		if a.gemma4 == nil || a.MoE != nil {
			t.Fatalf("representativeConfig(gemma4) is not dense Gemma 4 (gemma4=%v MoE=%v)", a.gemma4 != nil, a.MoE != nil)
		}
		return a
	}

	// env OFF (default): declined everywhere — the arch predicate AND every backend. This is the
	// invariant that keeps the generated matrix honest with the gate off (like granite behind
	// GOINFER_SSM_RESIDENT: hardware-matrix.md shows CPU across the board).
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	a := denseArch()
	if a.decodeRunnerEligible() {
		t.Error("dense Gemma 4 admitted with GOINFER_GEMMA4_RESIDENT off — must stay CPU (unreachable-by-default)")
	}
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if ResidentEligible(a, be) {
			t.Errorf("%s: ResidentEligible(dense Gemma 4)=true with env off — the matrix must not claim residency users can't reach", be)
		}
	}

	// env ON: the arch predicate admits dense Gemma 4; CUDA ships every feature it needs
	// (including the new FeatFinalLogitSoftcap), so CUDA admits; WebGPU still lacks the four
	// Gemma kernels, so the feature gate refuses it — no overclaim.
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	a = denseArch()
	if !a.decodeRunnerEligible() {
		t.Error("dense Gemma 4 NOT admitted with env on — the bring-up gate should open")
	}
	if !ResidentEligible(a, "cuda") {
		t.Error("cuda declines dense Gemma 4 with env on despite shipping every required feature (incl. FeatFinalLogitSoftcap)")
	}
	if ResidentEligible(a, "webgpu") {
		t.Error("webgpu admits dense Gemma 4 but lacks its Gemma kernels — the feature gate must refuse it")
	}

	// enable_moe_block declined even with env on — Gemma 4's parallel dense‖MoE is not the
	// generic FeatMoE shape; deferred to Split B.
	moe, _, err := resolveArchitecture(representativeConfig("gemma4_text"))
	if err != nil {
		t.Fatalf("resolve gemma4_text (MoE): %v", err)
	}
	if moe.MoE == nil {
		t.Fatal("representativeConfig(gemma4_text) is not the enable_moe_block variant")
	}
	if moe.decodeRunnerEligible() {
		t.Error("gemma4_text (enable_moe_block) admitted with env on — MoE is deferred to Split B, must decline")
	}
}
