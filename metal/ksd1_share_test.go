//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
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

// ksd1E2B loads Gemma 4 E2B's default Metal sidecar with the batched E-model pass and its tokenizer.
func ksd1E2B(t *testing.T, ctx int) (*decoder.Model, *metalResident, *tokenizer.Tokenizer) {
	t.Helper()
	home, _ := os.UserHomeDir()
	gguf := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf")
	m, err := decoder.Load(decoder.SidecarPath(gguf, "int4", "metal", false), decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: ctx})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[KS-D1] FIT-REFUSED: %v\n", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if !a.emodelBatched() {
		t.Fatal("the batched pass does not claim E2B")
	}
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatal(err)
	}
	return m, a, tk
}

// TestKSD1_longCell is D1-G2 (b): E2B, a 2,048-token prompt (this checkout's CLAUDE.md, its SHA-256 printed), the CPU int4
// reference's logits at the last prompt row and 32 teacher-forced positions after it (the CPU's own greedy continuation);
// argmax agreement of the batched pass with the split kernel against the same pass with the exact kernel
// (prefillSteelAttnOff), each followed by Metal decode over the CPU's sequence. PASS iff split >= exact - 2.0 points.
func TestKSD1_longCell(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	const np, gen = 2048, 32
	m, a, tk := ksd1E2B(t, np+gen+64)
	text, err := os.ReadFile("../CLAUDE.md")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode(string(text), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < np {
		t.Fatalf("CLAUDE.md is %d tokens, under %d", len(ids), np)
	}
	ids = ids[:np]
	fmt.Fprintf(os.Stderr, "[D1-G2b] CLAUDE.md sha256 %x, %d prompt tokens\n", sha256.Sum256(text), np)
	t0 := time.Now()
	cc := m.NewCache(np + gen)
	seq := slices.Clone(ids)
	cpuL := map[int][]float32{}
	for i := 0; i < np+gen-1; i++ {
		l, err := m.ForwardForTest(seq[i], cc)
		if err != nil {
			t.Fatal(err)
		}
		if i >= np-1 {
			cpuL[i] = slices.Clone(l)
			if len(seq) == i+1 {
				seq = append(seq, argmaxF(l))
			}
		}
		if i%256 == 0 {
			fmt.Fprintf(os.Stderr, "[D1-G2b] CPU reference %d/%d (%.0f s)\n", i, np+gen-1, time.Since(t0).Seconds())
		}
	}
	rows := make([][]float32, np)
	for i, id := range ids {
		rows[i] = m.EmbedResidentForTest(id)
	}
	agree := func(exact bool) (int, int) {
		prefillSteelAttnOff = exact
		defer func() { prefillSteelAttnOff = false }()
		a.Reset()
		lg, err := a.batchedPrefill(rows, 0, 0, nil, nil)
		if err != nil {
			t.Fatalf("batched pass declined: %v", err)
		}
		hit, n := 0, 1
		if argmaxF(lg) == argmaxF(cpuL[np-1]) {
			hit++
		}
		for i := np; i < np+gen-1; i++ {
			l, err := a.Forward(m.EmbedResidentForTest(seq[i]), i)
			if err != nil {
				t.Fatal(err)
			}
			if argmaxF(l) == argmaxF(cpuL[i]) {
				hit++
			}
			n++
		}
		return hit, n
	}
	eh, en := agree(true)
	sh, sn := agree(false)
	pe, ps := 100*float64(eh)/float64(en), 100*float64(sh)/float64(sn)
	verdict := "PASS"
	if ps < pe-2.0 {
		verdict = "FAIL"
	}
	fmt.Fprintf(os.Stderr, "[D1-G2b] exact %d/%d (%.2f%%), split %d/%d (%.2f%%): %s\n", eh, en, pe, sh, sn, ps, verdict)
	if verdict != "PASS" {
		t.Errorf("D1-G2b: split %.2f%% against exact %.2f%%", ps, pe)
	}
}

// TestKSD1_speed is D1-G3: E2B, PrefillLast on 4,096 rows through the batched pass, the split kernel against the exact one
// (prefillSteelAttnOff), A B B A over 3 pairs after one warm run each. A timed run: hold the timing lock (GOINFER_KSD1_TIMING=1).
func TestKSD1_speed(t *testing.T) {
	if os.Getenv("GOINFER_KSD1_TIMING") == "" {
		t.Skip("set GOINFER_KSD1_TIMING=1 (a timed run under the timing lock)")
	}
	m, a, _ := ksd1E2B(t, 4608)
	const n = 4096
	_, _, _, _, _, _, vocab := m.Dims()
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
	}
	run := func(exact bool) float64 {
		prefillSteelAttnOff = exact
		defer func() { prefillSteelAttnOff = false }()
		a.Reset()
		t0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), rows, 0); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds()
	}
	run(true)
	run(false)
	var ratios []float64
	for i := range 3 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var o, nw float64
		if i%2 == 0 {
			o, nw = run(true), run(false)
		} else {
			nw, o = run(false), run(true)
		}
		ratios = append(ratios, nw/o)
		fmt.Fprintf(os.Stderr, "[D1-G3] pair %d (load %s): exact %.2f s, split %.2f s, new/old %.4f\n", i, la[:len(la)-1], o, nw, nw/o)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[D1-G3] median new/old %.4f\n", ratios[1])
}
