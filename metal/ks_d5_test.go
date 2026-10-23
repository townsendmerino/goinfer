//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD5_guFShare is KS-D5's first measurement (docs/tasks/task-kernel-survey-2026-10.md): the batched prefill's scratch
// on Qwen2.5-Coder-1.5B (the Metal .giw in ~/models) for one 3,900-token pass, read from the scratch counter
// (prefillScratchSeen) while the prefill runs as one block, against the planned bytes (prefillScratchBytes) and guF's share
// of them (Mpad × 2 × maxI halves). It changes no product code. Not timed; a real checkpoint (GOINFER_HEAVY_TESTS=1).
func TestKSD5_guFShare(t *testing.T) {
	requireHeavyModel(t)
	path, opts := ksd5Checkpoint()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	const n = 3900
	prev := prefillBlockRowsForTest
	prefillBlockRowsForTest = n + 64 // one block: the whole prompt in one pass
	defer func() { prefillBlockRowsForTest = prev }()
	_, _, _, _, _, _, vocab := m.Dims()
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = (i*7919 + 13) % vocab
	}
	pass := func(twoStep bool) int64 { // one pass's counted scratch; the two-step arm is KS-D5c's baseline
		prevOff := swigluFusedOffForTest
		swigluFusedOffForTest = twoStep
		defer func() { swigluFusedOffForTest = prevOff }()
		before := prefillScratchSeen.Load()
		mgGen(t, m, prompt, 1)
		return prefillScratchSeen.Load() - before
	}
	seenTwo := pass(true)
	seen := pass(false)
	_, _, maxI := a.r.prefillWidths()
	mpad := (n + 63) / 64 * 64
	guF := int64(mpad * 2 * maxI * 2)
	planned := a.PrefillScratchBytes(n)
	const MB = 1e6
	fmt.Fprintf(os.Stderr, "[KS-D5] %s (%q, w8 %v, fused %v), one %d-row pass: scratch counted %.1f MB, planned %.1f MB; guF %.1f MB = %.1f%% of the counted (maxI %d, Mpad %d); without it %.1f MB\n",
		path, opts.Quant, a.r.w8, a.r.swigluFused(mpad), n, float64(seen)/MB, float64(planned)/MB, float64(guF)/MB, 100*float64(guF)/float64(seen), maxI, mpad, float64(seen-guF)/MB)
	fmt.Fprintf(os.Stderr, "[KS-D5] the two-step pass counted %.1f MB; less guF %.1f MB; this pass %.1f MB\n",
		float64(seenTwo)/MB, float64(seenTwo-guF)/MB, float64(seen)/MB)
}
