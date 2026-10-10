package decoder

import (
	"fmt"
	"os"
	"slices"
	"sort"
	"testing"
	"time"
)

// A3 fan-out, end to end on a real checkpoint. The kernel ratio (TestA3FanoutUtilization) does not
// project to a whole model, so the shipping claim comes from here.
//
// Method: paired and interleaved (CLAUDE.md measurement discipline: difference matched observations).
// Each pair runs the same prompt through the same model twice, once with the fan-out and once with
// GOINFER_PREFILL_ATTN_WORKERS=1, which forces prefillAttnWorkers to one slot and so takes the serial
// head loop (the pre-A3 shape, MatmulBT still column-parallel inside it).
//
// It also asserts the two arms' logits are bit-identical at every depth: that is what makes the timing a
// like-for-like comparison, and it drives the fan-out through its real caller (forwardLayersN).
// Why a kernel ratio is not quoted for a whole model: docs/code-notes/decoder.md#TestA3FanoutEndToEnd.
func TestA3FanoutEndToEnd(t *testing.T) {
	if os.Getenv("GOINFER_A3_FANOUT") == "" {
		t.Skip("set GOINFER_A3_FANOUT=1 to run the A3 end-to-end prefill A/B")
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	ctx := deadlineCtx(t)
	const pairs = 3
	// 8192 is the depth the f32 flag's own headline uses; the trend over the lower three depths does not
	// answer whether it holds there (docs/queue-performance.md).
	depths := []int{1024, 2048, 4096, 8192}

	setKnob(t, m, knobCPUFastAttention, "1") // this item is about the f32 path only
	start := time.Now()
	fmt.Fprintf(os.Stderr, "A3 e2e: start %s, %d depths x %d pairs x 2 arms\n",
		start.Format("15:04:05"), len(depths), pairs)

	type row struct {
		K            int
		ratios       []float64
		medOn, medOf time.Duration
	}
	var rows []row

	for _, K := range depths {
		if !m.canBatchN(K) {
			t.Skipf("model has no batched prefill at K=%d", K)
		}
		ids := make([]int, K)
		for i := range ids {
			ids[i] = 700 + i%64
		}
		run := func(workers string) ([]float32, time.Duration) {
			t.Helper()
			setKnob(t, m, knobPrefillWorkers, workers)
			t0 := time.Now()
			out, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), true)
			d := time.Since(t0)
			if err != nil {
				t.Fatalf("K=%d workers=%s: %v", K, workers, err)
			}
			return out, d
		}
		// Warm both arms once so the first pair does not carry page-in cost into whichever arm runs first.
		refOff, _ := run("1")
		refOn, _ := run("")
		for i := range refOn {
			if refOn[i] != refOff[i] {
				t.Fatalf("K=%d: fan-out changed the logits at index %d (%v vs %v) — the arms are not computing the same thing, so any timing below would be meaningless",
					K, i, refOn[i], refOff[i])
			}
		}

		var onD, offD []time.Duration
		var ratios []float64
		for p := range pairs {
			// Alternate which arm leads, so a monotone drift (thermal, page cache) cannot favour one of them.
			var dOn, dOff time.Duration
			if p%2 == 0 {
				_, dOff = run("1")
				_, dOn = run("")
			} else {
				_, dOn = run("")
				_, dOff = run("1")
			}
			onD, offD = append(onD, dOn), append(offD, dOff)
			ratios = append(ratios, float64(dOff)/float64(dOn))
			fmt.Fprintf(os.Stderr, "  K=%-5d pair %d/%d  pre-A3 %7.2fs  A3 %7.2fs  %.2fx   [elapsed %s]\n",
				K, p+1, pairs, dOff.Seconds(), dOn.Seconds(), float64(dOff)/float64(dOn),
				time.Since(start).Round(time.Second))
		}
		rows = append(rows, row{K: K, ratios: ratios, medOn: medianDur(onD), medOf: medianDur(offD)})
	}

	fmt.Fprintf(os.Stderr, "\n  END-TO-END PREFILL (%s, paired+interleaved, n=%d per depth)\n",
		"qwen2.5-coder-0.5b q4_k_m", pairs)
	fmt.Fprintf(os.Stderr, "  %-8s %12s %12s %8s   %s\n", "K", "pre-A3", "A3", "median", "pairs")
	for _, r := range rows {
		sort.Float64s(r.ratios)
		fmt.Fprintf(os.Stderr, "  %-8d %11.2fs %11.2fs %7.2fx   %v\n",
			r.K, r.medOf.Seconds(), r.medOn.Seconds(), float64(r.medOf)/float64(r.medOn), fmtRatios(r.ratios))
	}
	fmt.Fprintf(os.Stderr, "  total %s\n", time.Since(start).Round(time.Second))
}

// medianDur is the time.Duration sibling of theta_probe_test.go's median
// (which takes float64s); named apart rather than made generic so neither
// test's helper moves under the other.
func medianDur(d []time.Duration) time.Duration {
	if len(d) == 0 {
		return 0
	}
	c := append([]time.Duration(nil), d...)
	slices.Sort(c)
	return c[len(c)/2]
}

func fmtRatios(r []float64) string {
	s := ""
	for i, v := range r {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%.2f", v)
	}
	return s
}
