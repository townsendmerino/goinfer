package towerid

import "testing"

// TestTowerIdentity_cpu is AX2a-T for the CPU: every tower family's record, twice, agreeing.
func TestTowerIdentity_cpu(t *testing.T) {
	r := Check(t, "../..", "cpu")
	families := map[string]bool{}
	for _, c := range r.Cases {
		families[c.Family] = true
	}
	if len(families) != 12 {
		t.Errorf("%d families ran on the CPU, want 12 (§7.8's list): %v", len(families), families)
	}
}

// TestRun_unregisteredDeviceIsAnError holds the rule a registry split depends on: a declared family whose device
// registration is missing (none is linked into this package's test binary) is an error, never a CPU result.
func TestRun_unregisteredDeviceIsAnError(t *testing.T) {
	for _, backend := range []string{"metal", "cuda", "webgpu"} {
		if _, err := Run("../..", backend); err == nil {
			t.Errorf("%s: Run succeeded with no %s registration in the binary", backend, backend)
		} else {
			t.Logf("%s: %v", backend, err)
		}
	}
}
