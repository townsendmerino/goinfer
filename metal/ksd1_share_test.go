//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD1_attnShare is KS-D1's first measurement (docs/tasks/task-kernel-survey-2026-10.md, "KS-D1 first measurement"):
// Gemma 4 E2B (its default Metal sidecar), PrefillLast on 4,096 resident rows through the batched E-model pass, the full pass
// against the same pass with the attention dispatch skipped (prefillAttnSkipForTest), A B B A over 3 pairs after one warm run
// each; attention's share is 1 - skip/full. A timed run: hold the timing lock (GOINFER_KSD1_TIMING=1).
func TestKSD1_attnShare(t *testing.T) {
	if os.Getenv("GOINFER_KSD1_TIMING") == "" {
		t.Skip("set GOINFER_KSD1_TIMING=1 (a timed run under the timing lock)")
	}
	home, _ := os.UserHomeDir()
	gguf := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf")
	path := decoder.SidecarPath(gguf, "int4", "metal", false)
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: 4608})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[KS-D1] FIT-REFUSED: %v\n", err)
		}
		t.Fatal(err)
	}
	defer m.Close()
	decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if !a.emodelBatched() {
		t.Fatal("the batched pass does not claim E2B")
	}
	const n = 4096
	_, _, _, _, _, _, vocab := m.Dims()
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
	}
	fmt.Fprintf(os.Stderr, "[KS-D1] %s, %d rows, exact attention head dim %d, PrefillBlockRows %d\n", filepath.Base(path), n, a.r.exactAttnHeadDim(), a.PrefillBlockRows())
	run := func(skip bool) float64 {
		prefillAttnSkipForTest = skip
		defer func() { prefillAttnSkipForTest = false }()
		a.Reset()
		t0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), rows, 0); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds()
	}
	run(false)
	run(true)
	var shares []float64
	for i := range 3 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var full, skip float64
		if i%2 == 0 {
			full, skip = run(false), run(true)
		} else {
			skip, full = run(true), run(false)
		}
		shares = append(shares, 1-skip/full)
		fmt.Fprintf(os.Stderr, "[KS-D1] pair %d (load %s): full %.2f s, without attention %.2f s, attention share %.1f%%\n", i, la[:len(la)-1], full, skip, 100*(1-skip/full))
	}
	slices.Sort(shares)
	fmt.Fprintf(os.Stderr, "[KS-D1] median attention share %.1f%%\n", 100*shares[1])
}
