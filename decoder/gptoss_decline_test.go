package decoder

import (
	"slices"
	"testing"
)

// TestGptOss_backendsDecline asserts the load-bearing guarantee (docs/task-mxfp4-gptoss.md
// §3/§6.4): the resident GPU backends must DECLINE gpt-oss and fall back to CPU, never
// mis-run it. gpt-oss requires FeatAttnSink (the per-head softmax sink + clamped-SwiGLU
// experts), which no resident backend implements, so admission refuses all three. This is
// the same feature-taxonomy check the CUDA (cuda/backend.go) and Metal load paths apply, so
// verifying it here proves the decline without a GPU on the box (Metal has none here at all).
func TestGptOss_backendsDecline(t *testing.T) {
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

	// No resident backend implements FeatAttnSink → every one must decline.
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if impl, ok := ResidentBackendFeatures[be]; ok && impl[FeatAttnSink] {
			t.Errorf("backend %q claims FeatAttnSink but gpt-oss is CPU-only — must not implement it", be)
		}
		if ResidentEligible(arch, be) {
			t.Errorf("backend %q must DECLINE gpt-oss (ResidentEligible=true; want false → CPU fallback)", be)
		}
	}

	// And it must NOT reach the resident decode runner regardless of backend.
	if arch.decodeRunnerEligible() {
		t.Errorf("gpt-oss decodeRunnerEligible=true; want false (own CPU forward)")
	}
}
