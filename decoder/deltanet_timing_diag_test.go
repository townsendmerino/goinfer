package decoder

import (
	"os"
	"testing"
)

// TestDeltaNetTiming_gate0Split is Gate D0's sixth-outing split
// (docs/prompts/deltanet-cpu-recurrence.md): env-gated
// (GOINFER_DELTANET_TIMING=1) on the real 35B-A3B checkpoint, this diagnostic
// runs a real decode and reports the projections/recurrence/other breakdown
// via PrintDeltaNetTiming, plus the recurrence's own ns/state-elem rate against
// the ~1.4 ns/MAC serial-chain signature (docs/task-attention-decode-cost.md).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DELTANET_TIMING=1 \
//	  go test ./decoder -run TestDeltaNetTiming_gate0Split -v -timeout 30m
func TestDeltaNetTiming_gate0Split(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_DELTANET_TIMING") == "" {
		t.Skip("set GOINFER_DELTANET_TIMING=1 too (this test reports the split, doesn't force it)")
	}
	path := expandHome(t, envOr("GOINFER_QWEN35_GIW", "~/models/Qwen3.5-35B-A3B-Q4_K_M.int4.giw"))
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no qwen3.5-35b-a3b .giw at %s: %v", path, err)
	}

	m, err := Load(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	toks := greedyN(t, m, prompt, 40)
	if len(toks) == 0 {
		t.Fatal("no tokens generated")
	}
	PrintDeltaNetTiming()
}
