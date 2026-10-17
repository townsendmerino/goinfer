//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"os"
	"testing"
)

// TestProgram_S_walkerSpeed is OP3's gate S (docs/tasks/task-forward-architecture-2026-10.md §4.8): per-token decode through the
// program walker against the loop it replaced (programWalkOffForTest), interleaved ABBA in one process on the 0.5B at depths 128
// and 4,096. The quantity is the paired ratio old ms/token ÷ walker ms/token, the decode-rate ratio walker ÷ old; the bar is a
// median of at least 0.99 in both cells. The harness fails on any difference in the greedy streams.
//
//	GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label op3-S -- \
//	  go test -tags goinfer_testhooks ./decoder/ -run TestProgram_S_walkerSpeed -v -count=1 -timeout 30m
func TestProgram_S_walkerSpeed(t *testing.T) {
	for _, mc := range []struct {
		name  string
		depth int
		pairs int
	}{
		{"0.5B-d128", 128, 6},
		{"0.5B-d4096", 4096, 4},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", mc.depth)
			fmt.Fprintf(os.Stderr, "[S %s] depth %d, %d pairs\n", mc.name, len(h.ids), mc.pairs)
			t.Cleanup(func() { programWalkOffForTest = false })
			med := h.run("walker ("+mc.name+")", mc.pairs, func(walker bool) { programWalkOffForTest = !walker })
			fmt.Fprintf(os.Stderr, "[S %s] paired median rate walker/old %.3f (bar 0.99)\n", mc.name, med)
		})
	}
}
