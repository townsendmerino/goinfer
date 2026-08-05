//go:build darwin

package metal

import (
	"os"
	"runtime"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4_26B_posSweep is Step-6 divergence Step 1: the position-signature sweep. It measures the
// FINAL-hidden divergence (paged-Metal vs CPU, after all 64 layers) at positions 0,1,2,3,8,16 over one
// prompt, to discriminate the bug CLASS without reading a kernel:
//   - grows smoothly with position   → wrong rope base/fraction (angle error scales with pos);
//   - ~flat, and pos 1 ≈ pos 0        → off-by-one position index;
//   - tracks number-of-keys-attended  → KV indexing / multi-key attention.
//
// (The rope table is already proven byte-identical CPU↔Metal by code inspection — same gemma4InvFreq
// call — so "wrong base/fraction" is a priori unlikely; the sweep separates off-by-one from KV.)
func TestGemma4_26B_posSweep(t *testing.T) {
	requireHeavyModel(t)
	const giw = "/Users/francistownsend-merino/models/gemma4-26b-int4.giw"
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no .giw")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(32))
	m, err := decoder.Load(giw, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()

	// a 17-token prompt so we can probe up to pos 16.
	prompt := make([]int, 17)
	for i := range prompt {
		prompt[i] = 100 + i*211 // arbitrary distinct valid ids
	}
	nL1 := r.nL + 1

	decoder.SetGemma4HiddenCaptureForTest(true)
	defer decoder.SetGemma4HiddenCaptureForTest(false)
	cpuCache := m.NewCache(len(prompt))
	for _, tk := range prompt {
		if _, err := m.ForwardForTest(tk, cpuCache); err != nil {
			t.Fatalf("cpu: %v", err)
		}
	}
	cpuAll := decoder.Gemma4HiddenCaptureForTest()

	metalFinal := make([][]float32, len(prompt))
	func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for pos, tk := range prompt {
			copy(r.x.Floats(), m.EmbedResidentForTest(tk))
			cap := r.forwardPagedCaptureForTest(pos)
			metalFinal[pos] = cap[r.nL-1] // hidden after the last layer (pre final-norm)
		}
	}()

	t.Logf("=== position-signature sweep: final-hidden cosine (paged-Metal vs CPU) ===")
	for _, pos := range []int{0, 1, 2, 3, 8, 16} {
		cpuFinal := cpuAll[pos*nL1+r.nL] // this position's after-L63 hidden
		c, _ := cosMaxAbs(cpuFinal, metalFinal[pos])
		t.Logf("  pos %2d (%2d keys attended): final-hidden cosine %.4f", pos, pos+1, c)
	}
}
