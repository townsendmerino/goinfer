package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// op3tLoadReal loads a real checkpoint for OP3t's tier 1 cells at int4, with the KV priced at 4,096 positions (the cells use at
// most 2,500), or skips: they need GOINFER_HEAVY_TESTS=1 and the model under ~/models (env overrides the path). A fit-guard
// refusal is reported as the skip reason.
func op3tLoadReal(t *testing.T, env, def string) *Model {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("GOINFER_HEAVY_TESTS=1 runs OP3t's tier 1 cells")
	}
	path := os.Getenv(env)
	if path == "" {
		path = filepath.Join(os.Getenv("HOME"), "models", def)
	}
	m, err := Load(path, Options{Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Skipf("NOT RUN: %s: %v", path, err)
	}
	return m
}

// op3tIDs are n deterministic in-vocabulary token ids: identity of two forwards does not depend on what the tokens say.
func op3tIDs(n, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*7919 + 13) % vocab
	}
	return ids
}

// op3tSame counts the values of got that differ from want in any bit.
func op3tSame(want, got []float32) (diff int) {
	for j := range want {
		if math.Float32bits(want[j]) != math.Float32bits(got[j]) {
			diff++
		}
	}
	return diff
}

// op3tABBA is TW5's instrument: arm(twin) runs one timed call; pairs alternate which arm goes first. The quantity is twin time ÷
// walker time per pair, the walker's rate relative to the twin's, and the bar is a median of at least 0.99. The cell is the whole
// run: it is void, and fails as such, when the 1-minute load average at its start is above half the core count (CLAUDE.md, "Run
// budget"). Each pair logs its starting load for the record; both arms of a pair share it, which the pairing is for.
func op3tABBA(t *testing.T, label string, pairs int, arm func(twin bool) time.Duration) {
	t.Helper()
	arm(false) // warm-up, both arms
	arm(true)
	if la, half := loadAvg1(), float64(runtime.NumCPU())/2; la > half {
		t.Fatalf("VOID: the cell started at load %.2f, above %.1f; re-run it when the box is quieter", la, half)
	}
	var ratios []float64
	start := time.Now()
	for p := range pairs {
		la := loadAvg1()
		order := []bool{false, true} // walker, twin
		if p%2 == 1 {
			order = []bool{true, false}
		}
		var dWalk, dTwin time.Duration
		for _, twin := range order {
			if d := arm(twin); twin {
				dTwin = d
			} else {
				dWalk = d
			}
		}
		r := float64(dTwin) / float64(dWalk)
		fmt.Fprintf(os.Stderr, "[TW5 %s] pair %d of %d: load %.2f, walker %v, twin %v, ratio %.4f (elapsed %v)\n",
			label, p+1, pairs, la, dWalk.Round(time.Millisecond), dTwin.Round(time.Millisecond), r, time.Since(start).Round(time.Second))
		ratios = append(ratios, r)
	}
	sorted := append([]float64(nil), ratios...)
	sort.Float64s(sorted)
	med := (sorted[pairs/2-1] + sorted[pairs/2]) / 2
	slower := 0
	for _, r := range ratios {
		if r < 1 {
			slower++
		}
	}
	verdict := "PASS"
	switch {
	case med >= 0.99:
	case slower == pairs:
		verdict = "FAIL"
	default:
		verdict = "PARKED"
	}
	fmt.Fprintf(os.Stderr, "[TW5 %s] ratios %v, median %.4f, %d of %d pairs slower: %s (bar 0.99)\n", label, ratios, med, slower, pairs, verdict)
	if verdict != "PASS" {
		t.Errorf("TW5 %s %s: median %.4f", label, verdict, med)
	}
}

// op3tPromptHidden runs PromptHidden through the walker (twin false) or the twin and returns the hidden state and the call's
// wall time.
func op3tPromptHidden(t *testing.T, m *Model, ids []int, twin bool) ([]float32, time.Duration) {
	t.Helper()
	qwen35TwinForTest = twin
	defer func() { qwen35TwinForTest = false }()
	t0 := time.Now()
	h, err := m.PromptHidden(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	return h, time.Since(t0)
}

// TestOP3t_qwen35RealT1 is OP3t's TW4 for qwen3.5 (docs/tasks/task-forward-architecture-2026-10.md §4.8): PromptHidden on the
// real Qwen3.5-0.8B over prompts of 145, 621 and 2,500 tokens, through the walker and the twin in one process, bit for bit.
//
//	GOINFER_HEAVY_TESTS=1 go test ./decoder/ -run TestOP3t_qwen35RealT1 -v -count=1 -timeout 20m
func TestOP3t_qwen35RealT1(t *testing.T) {
	m := op3tLoadReal(t, "GOINFER_OP3T_QWEN35", "qwen3.5-0.8b")
	defer m.Close()
	start := time.Now()
	for i, n := range []int{145, 621, 2500} {
		ids := op3tIDs(n, m.w.arch.VocabSize)
		if !m.qwen35BatchN(n, m.NewCache(n)) {
			t.Fatalf("%d tokens: the batched path does not apply", n)
		}
		want, dTwin := op3tPromptHidden(t, m, ids, true)
		got, dWalk := op3tPromptHidden(t, m, ids, false)
		diff := op3tSame(want, got)
		fmt.Fprintf(os.Stderr, "[TW4] prompt %d of 3: %d tokens, twin %v, walker %v, %d of %d values differ (elapsed %v)\n",
			i+1, n, dTwin.Round(time.Millisecond), dWalk.Round(time.Millisecond), diff, len(want), time.Since(start).Round(time.Second))
		if diff != 0 {
			t.Errorf("%d tokens: DIFFERENT: %d of %d values", n, diff, len(want))
		}
	}
}

// TestOP3t_qwen35RealSpeed is OP3t's TW5 for qwen3.5: PromptHidden on the real 0.8B at 621 tokens, walker against twin, 6 ABBA
// pairs in one process (op3tABBA). Run it under the exclusive timing lock:
//
//	GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label op3t-tw5 -- \
//	  go test ./decoder/ -run TestOP3t_qwen35RealSpeed -v -count=1 -timeout 20m
func TestOP3t_qwen35RealSpeed(t *testing.T) {
	m := op3tLoadReal(t, "GOINFER_OP3T_QWEN35", "qwen3.5-0.8b")
	defer m.Close()
	ids := op3tIDs(621, m.w.arch.VocabSize)
	op3tABBA(t, "qwen3.5", 6, func(twin bool) time.Duration {
		_, d := op3tPromptHidden(t, m, ids, twin)
		return d
	})
}

// op3tGemma4RealSpan is the real cells' synthetic image block: 256 rows from position 16.
var op3tGemma4RealSpan = []ImageSpan{{Pos: 16, Len: 256}}

// op3tGemma4Real runs one prefill of n tokens through runLayersGemma4FromEmbedN (walker, or twin) over a fresh cache, with
// deterministic stand-ins for projected features in the span rows, and returns the last row after the head's norm.
func op3tGemma4Real(t *testing.T, m *Model, n int, spans []ImageSpan, twin bool) ([]float32, time.Duration) {
	t.Helper()
	gemma4TwinForTest = twin
	defer func() { gemma4TwinForTest = false }()
	ids := op3tIDs(n, m.w.arch.VocabSize)
	cache := m.NewCache(n)
	t0 := time.Now()
	out := op3tG4Pass(t, m, ids, 0, spans, cache)
	return out, time.Since(t0)
}

// TestOP3t_gemma4RealT1 is OP3t's TW4 for Gemma 4: the real E2B, text only at 145 and 621 tokens, and 621 tokens with one
// 256-row span, through the walker and the twin in one process, bit for bit.
//
//	GOINFER_HEAVY_TESTS=1 go test ./decoder/ -run TestOP3t_gemma4RealT1 -v -count=1 -timeout 20m
func TestOP3t_gemma4RealT1(t *testing.T) {
	m := op3tLoadReal(t, "GOINFER_OP3T_GEMMA4", "gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf")
	defer m.Close()
	start := time.Now()
	for i, c := range []struct {
		n     int
		spans []ImageSpan
	}{{145, nil}, {621, nil}, {621, op3tGemma4RealSpan}} {
		want, dTwin := op3tGemma4Real(t, m, c.n, c.spans, true)
		got, dWalk := op3tGemma4Real(t, m, c.n, c.spans, false)
		diff := op3tSame(want, got)
		fmt.Fprintf(os.Stderr, "[TW4 gemma4] cell %d of 3: %d tokens, %d span(s), twin %v, walker %v, %d of %d values differ (elapsed %v)\n",
			i+1, c.n, len(c.spans), dTwin.Round(time.Millisecond), dWalk.Round(time.Millisecond), diff, len(want), time.Since(start).Round(time.Second))
		if diff != 0 {
			t.Errorf("%d tokens, %d span(s): DIFFERENT: %d of %d values", c.n, len(c.spans), diff, len(want))
		}
	}
}

// TestOP3t_gemma4RealSpeed is OP3t's TW5 for Gemma 4: the E2B at 621 tokens with one span, walker against twin, 6 ABBA pairs in
// one process (op3tABBA), under the exclusive timing lock.
func TestOP3t_gemma4RealSpeed(t *testing.T) {
	m := op3tLoadReal(t, "GOINFER_OP3T_GEMMA4", "gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf")
	defer m.Close()
	op3tABBA(t, "gemma4", 6, func(twin bool) time.Duration {
		_, d := op3tGemma4Real(t, m, 621, op3tGemma4RealSpan, twin)
		return d
	})
}

// loadAvg1 is the 1-minute load average, or -1 when it cannot be read.
func loadAvg1() float64 {
	var s string
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		s = string(b)
	} else if out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output(); err == nil {
		s = strings.Trim(strings.TrimSpace(string(out)), "{} ")
	}
	var la float64
	if _, err := fmt.Sscan(s, &la); err != nil {
		return -1
	}
	return la
}
