//go:build gpu

package gpu

import (
	"os"
	"testing"
)

// TestDeviceExhaustion_repro finds how many Context create/destroy cycles this driver
// tolerates in ONE process. The ./gpu/ suite loses its GPU partway through a full run —
// every later test skips with "failed to request device", which silently turns
// TestWebGPU_forwardParity and TestResidentForwardN_parity from gates into no-ops.
//
// Diagnostic, not a gate: it reports the cycle count rather than asserting one, because the
// limit is a driver property and pinning it would make this fail on someone else's machine
// for a reason that is not a goinfer defect.
func TestDeviceExhaustion_repro(t *testing.T) {
	// ~45s and it deliberately exhausts the GPU, so it is opt-in rather than part of every run.
	if os.Getenv("GOINFER_DEVICE_PROBE") == "" {
		t.Skip("driver-limit probe: set GOINFER_DEVICE_PROBE=1 (takes ~45s and exhausts the GPU)")
	}
	const max = 200
	for i := range max {
		c, err := New()
		if err != nil {
			t.Logf("FAILED to create Context #%d: %v", i+1, err)
			t.Logf("⇒ this driver tolerates %d create/destroy cycles per process", i)
			return
		}
		c.Close()
		if (i+1)%50 == 0 {
			t.Logf("  churn: %d contexts created AND CLOSED, still healthy", i+1)
		}
	}
	t.Logf("⇒ %d create/destroy cycles all succeeded — Context CHURN is not the cause", max)

	// The other axis: contexts that are never closed. A test that loads a model and does not
	// Close it holds its device for the rest of the process, and those accumulate.
	var held []*Context
	defer func() {
		for _, c := range held {
			c.Close()
		}
	}()
	for i := range max {
		c, err := New()
		if err != nil {
			t.Logf("FAILED to create SIMULTANEOUS Context #%d: %v", i+1, err)
			t.Logf("⇒ this driver allows %d LIVE contexts at once — that is the real limit", i)
			return
		}
		held = append(held, c)
		if (i+1)%10 == 0 {
			t.Logf("  live: %d contexts open simultaneously", i+1)
		}
	}
	t.Logf("⇒ %d simultaneous contexts all succeeded", max)
}
