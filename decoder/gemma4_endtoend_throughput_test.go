//go:build arm64

package decoder

import (
	"os"
	"strconv"
	"testing"
	"time"
)

// TestGemma4EndToEndThroughput re-measures the END-TO-END paged-decode gap
// (docs/task-zeno-compare.md's "Quiet-machine re-measure": gemma4-26b kind-4
// vs kind-3, -47.0%/-49.0% at 4GB/8GB, budget-invariant) — the number the
// whole cold-touch investigation was chasing an explanation for, now that
// the kernel-level 69%-slower finding has failed to reproduce 3/3 on a
// corrected methodology. This is NOT the isolated kernel microbenchmark;
// it's the real production path (Load with StreamWeights+WeightCacheBytes,
// then Model.Generate), same steady-state total_tokens/wall_time metric the
// original measurement used, adapted from cmd/serve+HTTP to a direct
// in-process call (no HTTP round-trip noise; the model is loaded ONCE and
// generation run 3 times against the same resident/paged state, matching
// the original "3 runs against one running server" shape without paying a
// multi-minute reload cost 3 times over).
//
// Env vars: GOINFER_EE_GIW (path), GOINFER_EE_BUDGET_GB (float, WeightCacheBytes
// in GB), GOINFER_EE_LABEL (free-text label for the log line).
func TestGemma4EndToEndThroughput(t *testing.T) {
	requireHeavyModel(t)
	path := expandHome(t, envOr("GOINFER_EE_GIW", "~/models/gemma4-26b-int4.giw"))
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no .giw at %s: %v", path, err)
	}
	budgetGB, _ := strconv.ParseFloat(envOr("GOINFER_EE_BUDGET_GB", "4"), 64)
	label := envOr("GOINFER_EE_LABEL", path)
	budget := int64(budgetGB * 1e9)

	m, err := Load(path, Options{StreamWeights: true, WeightCacheBytes: budget})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	const genTokens = 60

	var tokPerSec [3]float64
	for run := range 3 {
		start := time.Now()
		toks := greedyN(t, m, prompt, genTokens)
		elapsed := time.Since(start)
		total := len(prompt) + len(toks)
		tokPerSec[run] = float64(total) / elapsed.Seconds()
		t.Logf("%s run %d: %d total tokens (prompt %d + completion %d) in %v = %.3f tok/s",
			label, run, total, len(prompt), len(toks), elapsed, tokPerSec[run])
	}
	avg := (tokPerSec[0] + tokPerSec[1] + tokPerSec[2]) / 3
	t.Logf("%s (budget=%.0fGB): 3-run avg = %.3f tok/s", label, budgetGB, avg)
}
