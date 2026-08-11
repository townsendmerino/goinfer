package decoder

import "testing"

// TestGemma4Admission_unconditional pins Gemma-4 admission after the Check-A backfill removed
// GOINFER_GEMMA4_RESIDENT from the arch predicate. Dense and enable_moe_block are admitted
// unconditionally; the per-backend answer stays with the feature gate, so WebGPU still declines
// (no Gemma kernels) and E-models still decline everywhere (PLE).
//
// It asserts the variable is INERT rather than dropping the coverage: a reintroduced read would
// silently reopen the hole the flag used to hide — a family gated on an env var with no gate
// behind it — so this fails if setting or clearing it changes any answer. The matrix's
// GPUResident column is arch.decodeRunnerEligible() (capability_matrix_test.go), so this is also
// what keeps docs/hardware-matrix.md honest.
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

	// The variable is INERT in both states. Previously env-off meant "declined everywhere", which
	// is what kept the generated matrix reporting CPU for a path that worked — see the flip commit.
	for _, v := range []string{"", "1"} {
		t.Setenv("GOINFER_GEMMA4_RESIDENT", v)
		a := denseArch()
		if !a.decodeRunnerEligible() {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: dense Gemma 4 NOT admitted — the predicate must no longer read this variable", v)
		}
		if !ResidentEligible(a, "cuda") || !ResidentEligible(a, "metal") {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: cuda/metal decline dense Gemma 4 despite shipping every required feature", v)
		}
		if ResidentEligible(a, "webgpu") {
			t.Errorf("GOINFER_GEMMA4_RESIDENT=%q: webgpu admits dense Gemma 4 but lacks its Gemma kernels", v)
		}
	}

	// enable_moe_block now admits — Split B (splitB.2c) landed the parallel dense‖MoE
	// FFN on its own cuda path (gemma4MoeMLP, routed around the generic MoE checks via
	// HasGemma4MoEResident), so it falls through the arch predicate like the dense variant. CUDA
	// ships every required feature; WebGPU still lacks the Gemma kernels, so the feature gate
	// refuses it — same no-overclaim shape as dense. It is the PLE-free 26B-A4B shape, so no
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

	// A Gemma-4 E-MODEL (E2B/E4B: per-layer embeddings / shared-KV / variable FFN) is DECLINED on
	// every resident backend EVEN with the bring-up gate ON — none implements the PLE branch, and
	// admitting it would silently skip PLE and mis-run (FeatGemma4EModel, declared by no backend).
	// The arch predicate can pass (a gemma4); the per-backend feature gate is what refuses.
	e := denseArch()
	e.gemma4.HiddenSizePerLayerInput = 256 // turn the dense arch into an E-model shape
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if ResidentEligible(e, be) {
			t.Errorf("%s: admits a Gemma-4 E-model (PLE hidden_size_per_layer_input>0) — the bridge skips the PLE branch and would mis-run", be)
		}
	}

	// The MoE variant is inert to the variable too. This used to assert the OPPOSITE — that env-off
	// declined everywhere, "so a regenerated matrix cannot claim gemma4_text residency users can't
	// reach". That reasoning was sound and the conclusion still holds; what changed is which way it
	// resolves. The matrix is honest either way, but only one of the two is honest AND useful.
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
