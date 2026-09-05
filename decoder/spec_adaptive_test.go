package decoder

import (
	"context"
	"testing"
)

// M-14: verifyTheta keyed on the BACKEND NAME, so every CUDA model got 0.251 — a value measured
// on dense 0.5B/1.5B, where the batched prefill pass actually runs. But cuda's ForwardN falls
// back to one `step` per row for every MoE / K=V / non-uniform / non-int4-or-int8 model, and a
// loop of single-token forwards has Theta ≈ 1 by construction.
//
// So on a resident MoE the controller was told a verify node costs a quarter of a decode step
// and drafted 8, when each node costs a FULL step: nine sequential steps per round for ~6.7
// committed tokens at high acceptance, worse below it. Metal hit the same shape, MEASURED it
// (1.006–1.048, linear to n=16) and ships 1.02 to disable speculation. This makes CUDA reach the
// same conclusion the same way — by asking the resident instead of inferring from its name.
func TestVerifyTheta_asksWhetherForwardNIsBatched(t *testing.T) {
	for name, tc := range map[string]struct {
		batched bool
		want    float64
	}{
		"batched pass runs → the backend's own constant": {true, defaultTheta},
		"batched pass declines → sequential, ~1":         {false, sequentialVerifyTheta},
	} {
		t.Run(name, func(t *testing.T) {
			// cpuBackend is a real Backend; thetaFor("cpu") is defaultTheta, so the two arms
			// below differ ONLY by what the resident reports about its prefill path — which is
			// exactly the thing M-14 added and the thing this test is about.
			m := &Model{resident: &fakeThetaResident{batched: tc.batched}, be: &cpuBackend{}}
			got := m.verifyTheta()
			if tc.batched && got != defaultTheta {
				t.Errorf("batched resident on a cpu backend: verifyTheta = %v, want the backend's "+
					"own constant %v", got, defaultTheta)
			}
			if !tc.batched && got != sequentialVerifyTheta {
				t.Errorf("resident whose ForwardN is SEQUENTIAL: verifyTheta = %v, want %v — the "+
					"backend name must not override what the resident reports (M-14)",
					got, sequentialVerifyTheta)
			}
		})
	}

	// The sequential value must DISABLE speculation, which is the point: a Theta below 1 says a
	// verify node is cheaper than a decode step, and for a loop of decode steps it is not.
	if sequentialVerifyTheta < 1 {
		t.Errorf("sequentialVerifyTheta = %v is below 1: it would still authorise drafting on a "+
			"path where each verify node costs a full step (M-14)", sequentialVerifyTheta)
	}
	// A staged model must still take the CPU constant, not either GPU one.
	if got := (&Model{}).verifyTheta(); got != defaultTheta {
		t.Errorf("staged/CPU model got %v, want the CPU default %v", got, defaultTheta)
	}
}

// fakeThetaResident reports a prefill path; everything else is unused by verifyTheta.
type fakeThetaResident struct {
	fakeResident
	batched bool
}

func (f *fakeThetaResident) PrefillLast(_ context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return nil, nil
}

func (f *fakeThetaResident) PrefillPath() (bool, string) {
	if f.batched {
		return true, "batched"
	}
	return false, "sequential — one forward per row"
}

// The CUDA constant itself is unchanged and still measured — M-14 is about WHEN it applies, not
// what it is. Pinned so a change to one is not mistaken for the other.
func TestThetaFor_cudaConstantUnchanged(t *testing.T) {
	if got := thetaFor("cuda"); got != 0.251 {
		t.Errorf("thetaFor(cuda) = %v, want 0.251 (cuda/theta_probe_test.go, dense 0.5B/1.5B)", got)
	}
	if got := thetaFor("metal"); got < 1 {
		t.Errorf("thetaFor(metal) = %v; it is >= 1 BY MEASUREMENT and disables speculation", got)
	}
}
