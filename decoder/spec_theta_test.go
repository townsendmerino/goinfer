package decoder

import "testing"

// TestVerifyTheta_stagedGetsCPUValue is the subtle one. Theta describes the path
// the VERIFY runs on. genNgramInto uses resident.ForwardN only when residency
// built and otherwise falls back to the CPU batched forwardN — so a model whose
// residency DECLINED must get the CPU constant even though its backend is a GPU.
// Keying off the requested backend alone would hand a declined model the GPU
// value, mis-tuning exactly the case where the decline already costs the user
// the whole forward.
func TestVerifyTheta_stagedGetsCPUValue(t *testing.T) {
	staged := &Model{be: cpuBackendForTest(t), resident: nil}
	if got := staged.verifyTheta(); got != defaultTheta {
		t.Fatalf("staged model got Theta %v, want the CPU %v — it verifies on the CPU path", got, defaultTheta)
	}
	var nilModel *Model
	if got := nilModel.verifyTheta(); got != defaultTheta {
		t.Fatalf("nil model got %v, want %v", got, defaultTheta)
	}
}

func cpuBackendForTest(t *testing.T) Backend {
	t.Helper()
	be, err := NewBackend("cpu")
	if err != nil {
		t.Fatalf("NewBackend(cpu): %v", err)
	}
	return be
}
