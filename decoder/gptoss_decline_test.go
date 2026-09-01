package decoder

import (
	"slices"
	"testing"
)

// TestGptOss_webgpuDecline asserts the load-bearing guarantee (docs/task-mxfp4-gptoss.md §3/§6.4)
// for the backend that does NOT implement gpt-oss's novel ops: WebGPU must DECLINE gpt-oss and
// fall back to CPU, never mis-run it. Metal and CUDA both implement it now and are asserted
// ADMITTED below — the positive half of the same guarantee.
//
// CUDA MOVED FROM THE DECLINE SIDE TO THE ADMIT SIDE on 2026-08-31 (G7), and the condition this
// test used to encode is exactly what changed: it said CUDA "has the same kernels loaded but not
// yet dispatched — dead code until wired". They are wired now, and wiring them found three silent
// defects no kernel test could see, because each was a term the WIRING dropped rather than a
// kernel computing it wrongly:
//
//	d9829ce  the gate‖up bias table indexed by SLOT id under expert caching
//	610ce7f  the per-expert down bias never applied (needed gemv_w4a8_moe_wacc_bias)
//	6cfb15c  route_gptoss never LOADED, so the router fell back to moe_route, which takes the
//	         mixing weight from the UNBIASED score — same experts, different weights
//
// The declaration rests on a real 20B forward, resident on an 8 GB card via --moe-cache-experts:
// 7/8 argmax-exact, min cosine 0.996392 (cuda.TestGptOssResidentParityCUDA). 2224441 declared
// FeatAttnSink once on kernel-level evidence and was correctly reverted; this time the whole model
// ran. WebGPU still has none of it and still refuses via the shared feature-taxonomy check.
func TestGptOss_webgpuDecline(t *testing.T) {
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

	for _, be := range []string{"webgpu"} {
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
	// CUDA, the 2026-08-31 addition. Asserted the same way as metal and for the same reason: a
	// backend that ships AND DISPATCHES the kernels must actually admit, not stay declined by a
	// stale check — which is the failure this test was written to catch in the other direction.
	for _, be := range []string{"metal", "cuda"} {
		if !residentBackendFeatures[be][FeatAttnSink] {
			t.Errorf("backend %q no longer declares FeatAttnSink — its gpt-oss parity gate should have started skipping; update this test if the kernels were intentionally reverted", be)
		}
		if !ResidentEligible(arch, be) {
			t.Errorf("backend %q declares FeatAttnSink but does not admit gpt-oss (ResidentEligible=false) — check decodeRunnerEligible's gptoss case and residentMoECapacityOK", be)
		}
	}
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
