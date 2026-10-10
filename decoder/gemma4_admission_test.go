package decoder

import "testing"

// TestGemma4Admission_unconditional pins that Gemma-4 admission does not read GOINFER_GEMMA4_RESIDENT. Dense Gemma 4 is
// admitted on cuda, metal and webgpu, enable_moe_block on cuda and not webgpu, and an E-model on metal and cuda and not
// webgpu; the per-backend answer stays with the feature gate.
//
// It asserts the variable is INERT rather than dropping the coverage: a reintroduced read would silently gate a family on an
// env var with no gate behind it, so this fails if setting or clearing it changes any answer. The matrix's GPUResident
// column is arch.decodeRunnerEligible() (capability_matrix_test.go), so this also keeps docs/hardware-matrix.md honest.
func TestGemma4Admission_unconditional(t *testing.T) {
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

	// The variable is INERT in both states.
	for _, v := range []string{"", "1"} {
		t.Setenv("GOINFER_GEMMA4_RESIDENT", v)
		a := denseArch()
		if !a.decodeRunnerEligible() {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: dense Gemma 4 NOT admitted — the predicate must no longer read this variable", v)
		}
		if !ResidentEligible(a, "cuda") || !ResidentEligible(a, "metal") || !ResidentEligible(a, "webgpu") {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: cuda/metal/webgpu decline dense Gemma 4 despite shipping every required feature", v)
		}
	}

	// enable_moe_block is admitted like the dense variant: the parallel dense||MoE FFN has its own cuda path (gemma4MoeMLP,
	// routed around the generic MoE checks via HasGemma4MoEResident). CUDA ships every required feature; WebGPU lacks the Gemma
	// kernels, so the feature gate refuses it (the same no-overclaim shape as dense). It is the PLE-free 26B-A4B shape, so no
	// FeatGemma4EModel.
	moe, _, err := resolveArchitecture(representativeConfig("gemma4_text"))
	if err != nil {
		t.Fatalf("resolve gemma4_text (MoE): %v", err)
	}
	if moe.MoE == nil {
		t.Fatal("representativeConfig(gemma4_text) is not the enable_moe_block variant")
	}
	if !moe.decodeRunnerEligible() {
		t.Error("gemma4_text (enable_moe_block) NOT admitted — Split B landed the resident MoE path, the gate should open")
	}
	if !ResidentEligible(moe, "cuda") {
		t.Error("cuda declines gemma4_text (enable_moe_block) despite shipping the gemma4MoeMLP resident path")
	}
	if ResidentEligible(moe, "webgpu") {
		t.Error("webgpu admits gemma4_text (enable_moe_block) but lacks its Gemma kernels — the feature gate must refuse it")
	}

	// A Gemma-4 E-MODEL (E2B/E4B: per-layer embeddings / shared-KV / variable FFN) is admitted by Metal (S1) and CUDA (S1 on CUDA,
	// docs/tasks/task-multimodal-support-2026-10.md), which implement all three. WebGPU has none of it, so admitting the model there
	// would silently skip PLE and mis-run: its feature gate must refuse it.
	e := denseArch()
	e.gemma4.HiddenSizePerLayerInput = 256 // turn the dense arch into an E-model shape
	for _, be := range []string{"metal", "cuda"} {
		if !ResidentEligible(e, be) {
			t.Errorf("%s declines a Gemma-4 E-model although it declares FeatGemma4EModel (missing %v)", be, missingFeatures(e.residentFeatures(), residentBackendFeatures[be]))
		}
	}
	if ResidentEligible(e, "webgpu") {
		t.Error("webgpu: admits a Gemma-4 E-model (PLE hidden_size_per_layer_input>0) — the bridge skips the PLE branch and would mis-run")
	}

	// The MoE variant is inert to the variable too.
	for _, v := range []string{"", "1"} {
		t.Setenv("GOINFER_GEMMA4_RESIDENT", v)
		m, _, err := resolveArchitecture(representativeConfig("gemma4_text"))
		if err != nil {
			t.Fatalf("resolve gemma4_text (MoE, env=%q): %v", v, err)
		}
		if !m.decodeRunnerEligible() {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: gemma4_text (enable_moe_block) NOT admitted — "+
				"the predicate must no longer read this variable", v)
		}
		if !ResidentEligible(m, "cuda") {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: cuda declines gemma4_text MoE", v)
		}
	}
}
