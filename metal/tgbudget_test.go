//go:build darwin

package metal

import "testing"

// TestMaxThreadgroupStageBytes is the M-11 arithmetic gate: the resident's threadgroup staging is
// 2 bytes/element over the widest STAGED contraction width — hidden, the q-width (nH·hd), or the MoE
// expert intermediate — and BuildResident declines when 2·K exceeds the device tile-memory limit
// (~32 KiB). The dense down-proj is non-staging, so the dense intermediate must NOT enter the max.
func TestMaxThreadgroupStageBytes(t *testing.T) {
	const appleLimit = 32768 // MaxThreadgroupMemoryLength on Apple GPUs (device-checked separately)
	for _, tc := range []struct {
		name                               string
		hidden, qWidth, moeInter, g4moeInt int
		want                               int
		overApple                          bool
	}{
		{"dense small", 2048, 2048, 0, 0, 4096, false},
		{"hidden dominates", 8192, 4096, 0, 0, 16384, false},
		{"q-width dominates (GQA-ish)", 2048, 6144, 0, 0, 12288, false},
		{"mixtral experts (inter 14336)", 4096, 4096, 14336, 0, 28672, false}, // 87% of 32 KiB, fits
		{"inter exactly at limit", 1024, 1024, 16384, 0, 32768, false},        // 2·16384 == limit, allowed
		{"inter over limit", 1024, 1024, 16385, 0, 32770, true},               // one past → declines
		{"gemma4 moe inter over", 2048, 2048, 0, 20000, 40000, true},
		{"huge hidden over", 20000, 2048, 0, 0, 40000, true},
	} {
		got := maxThreadgroupStageBytes(tc.hidden, tc.qWidth, tc.moeInter, tc.g4moeInt)
		if got != tc.want {
			t.Errorf("%s: maxThreadgroupStageBytes = %d, want %d", tc.name, got, tc.want)
		}
		if over := got > appleLimit; over != tc.overApple {
			t.Errorf("%s: over-32KiB = %v, want %v (got %d B)", tc.name, over, tc.overApple, got)
		}
	}
	// The dense intermediate is non-staging: a model with a huge dense inter but small hidden/q-width
	// must NOT be flagged (the down-proj uses plain pGemv). moeInter=0 for a dense model.
	if b := maxThreadgroupStageBytes(4096, 4096, 0, 0); b > appleLimit {
		t.Errorf("dense model with a large intermediate wrongly flagged over budget (%d B) — dense down-proj is non-staging", b)
	}
}
