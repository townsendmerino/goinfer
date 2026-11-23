//go:build gpu

package kernels

import "testing"

// TestRunModelToPrefill_declinesAttnTemp: a plain dense RunModel takes the batched pass, and the
// same model with RunModel.AttnTemp set does not, since the pass applies no temperature. No device: RunModelToPrefill reads
// the plan's fields only.
func TestRunModelToPrefill_declinesAttnTemp(t *testing.T) {
	base := RunModel{LmHead: &ResidentW8A8{}}
	if _, _, ok := RunModelToPrefill(&base, 64); !ok {
		t.Fatal("the plain dense baseline declined: the test cannot isolate the attention temperature")
	}
	temp := base
	temp.AttnTemp = func(int) float32 { return 1.25 }
	if _, _, ok := RunModelToPrefill(&temp, 64); ok {
		t.Error("RunModelToPrefill admitted a model with an attention temperature: the batched pass would drop it")
	}
}
