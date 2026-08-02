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

	// enable_moe_block now admits with env on — Split B (splitB.2c) landed the parallel dense‖MoE
	// FFN on its own cuda path (gemma4MoeMLP, routed around the generic MoE checks via
	// HasGemma4MoEResident), so it falls through the arch predicate like the dense variant. CUDA
	// ships every required feature; WebGPU still lacks the Gemma kernels, so the feature gate
	// refuses it — same no-overclaim shape as dense.
	moe, _, err := resolveArchitecture(representativeConfig("gemma4_text"))
	if err != nil {
		t.Fatalf("resolve gemma4_text (MoE): %v", err)
	}
	if moe.MoE == nil {
		t.Fatal("representativeConfig(gemma4_text) is not the enable_moe_block variant")
	}
	if !moe.decodeRunnerEligible() {
		t.Error("gemma4_text (enable_moe_block) NOT admitted with env on — Split B landed the resident MoE path, the gate should open")
	}
	if !ResidentEligible(moe, "cuda") {
		t.Error("cuda declines gemma4_text (enable_moe_block) with env on despite shipping the gemma4MoeMLP resident path")
	}
	if ResidentEligible(moe, "webgpu") {
		t.Error("webgpu admits gemma4_text (enable_moe_block) but lacks its Gemma kernels — the feature gate must refuse it")
	}

	// env OFF: the MoE variant must decline everywhere too — the env gate is the arch predicate's,
	// so a regenerated matrix cannot claim gemma4_text residency users can't reach.
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	moeOff, _, err := resolveArchitecture(representativeConfig("gemma4_text"))
	if err != nil {
		t.Fatalf("resolve gemma4_text (MoE, env off): %v", err)
	}
	if moeOff.decodeRunnerEligible() {
		t.Error("gemma4_text (enable_moe_block) admitted with GOINFER_GEMMA4_RESIDENT off — must stay CPU (unreachable-by-default)")
	}
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if ResidentEligible(moeOff, be) {
			t.Errorf("%s: ResidentEligible(gemma4_text MoE)=true with env off — the matrix must not claim residency users can't reach", be)
		}
	}
}
