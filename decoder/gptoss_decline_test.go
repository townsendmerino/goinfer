package decoder

import (
	"slices"
	"testing"
)

// TestGptOss_webgpuDecline asserts the load-bearing guarantee (docs/task-mxfp4-gptoss.md §3/§6.4):
// a backend that does NOT implement gpt-oss's novel ops must DECLINE it and fall back to CPU,
// never mis-run it — and a backend that DOES ship and dispatch them must actually ADMIT, not stay
// declined by a stale check. All three resident backends are now on the admit side; the name is
// historical (kept so `git log -p` on it still tells the right story).
//
// CUDA MOVED FROM THE DECLINE SIDE TO THE ADMIT SIDE on 2026-08-31 (G7): wiring the already-loaded
// kernels found three silent defects no kernel test could see, because each was a term the WIRING
// dropped rather than a kernel computing it wrongly:
//
//	d9829ce  the gate‖up bias table indexed by SLOT id under expert caching
//	610ce7f  the per-expert down bias never applied (needed gemv_w4a8_moe_wacc_bias)
//	6cfb15c  route_gptoss never LOADED, so the router fell back to moe_route, which takes the
//	         mixing weight from the UNBIASED score — same experts, different weights
//
// The declaration rests on a real 20B forward, resident on an 8 GB card via --moe-cache-experts:
// 7/8 argmax-exact, min cosine 0.996392 (cuda.TestGptOssResidentParityCUDA). 2224441 declared
// FeatAttnSink once on kernel-level evidence and was correctly reverted; this time the whole model
// ran.
//
// WEBGPU MOVED FROM THE DECLINE SIDE TO THE ADMIT SIDE on 2026-09-08 (G6,
// docs/task-gpu-paths-2026-09.md): the sink threaded through every attention kernel (attn,
// attn-keys, attn-f16, attn-i8, and all three wide variants — 7 pipelines), plus three brand-new
// MoE kernels (gpt-oss disagrees with the generic MoE path on what the router bias means and what
// the activation clamps — routeGptOssWGSL/gptossGluQuantWGSL/moeExpertGptOssDownGEMVWGSL,
// gpu/moe.go). Verified against the real (non-seeded) decoder/testdata/gptoss_tiny.gguf, resident
// vs CPU: min cosine 0.9943 across 8 positions (gpu.TestGptOssResidentParityWebGPU) — the same
// 0.95 floor Metal's own gate uses on this fixture.
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

	// A backend that ships AND DISPATCHES the kernels must actually admit, not stay declined by a
	// stale check — which is the failure this test was written to catch in the other direction
	// (and did, twice: CUDA in 2026-08-31, WebGPU in 2026-09-08).
	for _, be := range []string{"metal", "cuda", "webgpu"} {
		if !residentBackendFeatures[be][FeatAttnSink] {
			t.Errorf("backend %q no longer declares FeatAttnSink — its gpt-oss parity gate should have started skipping; update this test if the kernels were intentionally reverted", be)
		}
		if !ResidentEligible(arch, be) {
			t.Errorf("backend %q declares FeatAttnSink but does not admit gpt-oss (ResidentEligible=false) — check decodeRunnerEligible's gptoss case and residentMoECapacityOK", be)
		}
	}

	// The arch-shape gate falls through for gpt-oss (2026-08-18, mirroring gemma4 and the
	// GPT-2 NonGatedMLP/LearnedPosEmbed/OutBias precedent): the decline lives at the feature gate
	// (FeatAttnSink, asserted above), not here — a backend that implements the sink/clamped-SwiGLU
	// kernels admits through ResidentEligible without this arch predicate needing a second edit.
	if !arch.decodeRunnerEligible() {
		t.Errorf("gpt-oss decodeRunnerEligible=false; want true (the decline/admit split lives at FeatAttnSink, not the arch shape)")
	}
}
