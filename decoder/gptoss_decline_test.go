package decoder

import (
	"slices"
	"testing"
)

// TestGptOss_webgpuDecline asserts the load-bearing guarantee (docs/completed/task-mxfp4-gptoss.md §3/§6.4): a backend that
// does NOT implement gpt-oss's novel ops must DECLINE it and fall back to CPU, never mis-run it, and a backend that DOES ship
// and dispatch them must actually ADMIT, not stay declined by a stale check. All three resident backends are on the admit
// side; the name is historical.
//
// Declare FeatAttnSink on the evidence of a whole-model forward, never on kernel tests alone: the defects found wiring the
// kernels were terms the WIRING dropped, which no kernel test can see (the gate||up bias table indexed by slot id under expert
// caching, the per-expert down bias never applied, route_gptoss never loaded so the router took its mixing weight from the
// UNBIASED score). The admissions rest on cuda.TestGptOssResidentParityCUDA (a real 20B forward, resident on an 8 GB card via
// --moe-cache-experts), gpu.TestGptOssResidentParityWebGPU and Metal's own gate (decoder/testdata/gptoss_tiny.gguf, resident
// vs CPU, 0.95 cosine floor); history in docs/code-notes/decoder.md#TestGptOss_webgpuDecline.
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

	// A backend that ships AND DISPATCHES the kernels must actually admit, not stay declined by a stale check, which is the
	// failure this test was written to catch in the other direction.
	for _, be := range []string{"metal", "cuda", "webgpu"} {
		if !residentBackendFeatures[be][FeatAttnSink] {
			t.Errorf("backend %q no longer declares FeatAttnSink — its gpt-oss parity gate should have started skipping; update this test if the kernels were intentionally reverted", be)
		}
		if !ResidentEligible(arch, be) {
			t.Errorf("backend %q declares FeatAttnSink but does not admit gpt-oss (ResidentEligible=false) — check decodeRunnerEligible's gptoss case and residentMoECapacityOK", be)
		}
	}

	// The arch-shape gate falls through for gpt-oss (as for gemma4 and the GPT-2 NonGatedMLP/LearnedPosEmbed/OutBias
	// precedent): the decline lives at the feature gate (FeatAttnSink, asserted above), not here, so a backend that
	// implements the sink/clamped-SwiGLU kernels admits through ResidentEligible without this arch predicate needing a
	// second edit.
	if !arch.decodeRunnerEligible() {
		t.Errorf("gpt-oss decodeRunnerEligible=false; want true (the decline/admit split lives at FeatAttnSink, not the arch shape)")
	}
}
