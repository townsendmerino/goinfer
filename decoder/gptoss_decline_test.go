package decoder

import (
	"slices"
	"testing"
)

// TestGptOss_cudaWebgpuDecline asserts the load-bearing guarantee (docs/task-mxfp4-gptoss.md
// §3/§6.4) for the two backends that do NOT implement gpt-oss's novel ops: CUDA and WebGPU must
// DECLINE gpt-oss and fall back to CPU, never mis-run it. Metal is the exception — it declares
// FeatAttnSink (2026-08-18: kernels.go's attention sink term, moe.go's route_gptoss/
// swiglu_quant_gptoss/gemv_w4a8_moe_wacc_bias, TestGptOssResidentParity) and is asserted
// admitted below, not declined; CUDA has the same kernels loaded but not yet dispatched
// (cuda/backend.go's gptOssSw/gptOssSinks/gptOssExpBias scaffolding — dead code until wired),
// and WebGPU has none of it, so both still refuse via the shared feature-taxonomy check.
func TestGptOss_cudaWebgpuDecline(t *testing.T) {
	cfg := representativeConfig("gpt_oss")
	if cfg == nil {
		t.Fatal("no representativeConfig for gpt_oss")
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture(gpt_oss): %v", err)
	}

	// The novel op is declared as a required feature.
	req := arch.residentFeatures()
	if !slices.Contains(req, FeatAttnSink) {
		t.Errorf("gpt-oss required features %v missing FeatAttnSink", req)
	}

	for _, be := range []string{"cuda", "webgpu"} {
		if impl, ok := residentBackendFeatures[be]; ok && impl[FeatAttnSink] {
			t.Errorf("backend %q claims FeatAttnSink but has no dispatched sink/MoE kernels — must not implement it", be)
		}
		if ResidentEligible(arch, be) {
			t.Errorf("backend %q must DECLINE gpt-oss (ResidentEligible=true; want false → CPU fallback)", be)
		}
	}

	// Metal is the one backend that DOES implement FeatAttnSink now — the positive half of the
	// same guarantee: a backend that ships the kernels must actually admit, not stay declined by
	// a stale check. TestGptOssResidentParity is the end-to-end proof (8/8 argmax-exact, min
	// cosine 0.9989 on the tiny fixture); this only re-asserts the admission wiring, cheaply,
	// without a GPU on the box.
	if !residentBackendFeatures["metal"][FeatAttnSink] {
		t.Error(`backend "metal" no longer declares FeatAttnSink — TestGptOssResidentParity should have started skipping; update this test if the kernels were intentionally reverted`)
	}
	if !ResidentEligible(arch, "metal") {
		t.Error(`backend "metal" declares FeatAttnSink but does not admit gpt-oss (ResidentEligible=false) — check decodeRunnerEligible's gptoss case and residentMoECapacityOK`)
	}

	// The arch-shape gate falls through for gpt-oss (2026-08-18, mirroring gemma4 and the
	// GPT-2 NonGatedMLP/LearnedPosEmbed/OutBias precedent): the decline lives at the feature gate
	// (FeatAttnSink, asserted above), not here — a backend that implements the sink/clamped-SwiGLU
	// kernels admits through ResidentEligible without this arch predicate needing a second edit.
	if !arch.decodeRunnerEligible() {
		t.Errorf("gpt-oss decodeRunnerEligible=false; want true (the decline/admit split lives at FeatAttnSink, not the arch shape)")
	}
}
