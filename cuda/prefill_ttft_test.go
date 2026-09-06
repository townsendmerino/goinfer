//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillTTFT measures the batched PrefillLast vs the sequential ForwardNoLogits loop on a real
// dense model, at the prompt lengths that bracket the Ollama crossover (128/512/2048). It is the
// milestone-2 speedup number: goinfer's sequential prefill reads every weight once per prompt token
// (weight-bandwidth-bound), so its TTFT grows ~linearly; the batched path reads each weight once for
// all M tokens. Heavy (loads a 1.5B model); gated on GOINFER_HEAVY_TESTS + a GPU.
//
// WHAT THE "batched" COLUMN MEANS CHANGED ON 2026-09-05, without this file changing. CUDA fast
// prefill (attn_fused + gemm_w4a8_mma) became the DEFAULT above a 512-token floor, so at K >= 512
// the batched column now times the FAST path, not the exact one. That is the right thing for a
// standing test — it measures what ships — but it means a number from this test taken before that
// date and one taken after are not the same quantity. Set GOINFER_CUDA_FAST_PREFILL=0 to time the
// exact batched path, which is what every pre-2026-09-05 row in benchmarks.md holds.
//
// THE DEFAULTS ARE THE REGRESSION TEST AND DO NOT MOVE. The four env knobs below only widen what a
// deliberate measurement run can ask for; with none of them set this test runs exactly the model,
// quants and depths it always has, so its role as a standing check is unchanged. They exist because
// docs/task-prefill-gap.md §4 L2 sets its band on an END-TO-END cell this test could not reach —
// S at K=3900 — and prices L2/L3 on D7 as a second model, while the fixed list stops at K=2048 on a
// 1.5B (which is exactly the blind spot prefill-chunking-d7-2026-09-04.md records: "TestPrefillTTFT,
// the harness built for exactly this question, stops at M=2048 on a 1.5B model — a shape that fits,
// on a model that fits").
//
//	GOINFER_TTFT_MODEL   checkpoint name under the models dir, or an absolute path (default: S)
//	GOINFER_TTFT_QUANTS  comma-separated quants          (default: "int4,int8int8")
//	GOINFER_TTFT_K       comma-separated prompt lengths  (default: "128,512,2048")
//	GOINFER_TTFT_NOSEQ=1 skip the sequential arm — it is the "before" for a DIFFERENT question
//	                     (batched-vs-sequential), and at K=3900 it costs minutes per rep while
//	                     contributing nothing to a fast-vs-exact prefill comparison.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillTTFT -v
func TestPrefillTTFT(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if v := os.Getenv("GOINFER_TTFT_MODEL"); v != "" {
		if filepath.IsAbs(v) {
			path = v
		} else {
			path = modelPath(v)
		}
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	// int8's "before" is the sequential column (what int8int8 fell back to pre-§C6); "after" is the
	// batched column. int4 is measured at the same lengths so the remaining int8-vs-int4 gap is visible.
	for _, quant := range ttftCSV("GOINFER_TTFT_QUANTS", []string{"int4", "int8int8"}) {
		t.Run(quant, func(t *testing.T) { ttftMeasure(t, path, quant) })
	}
}

func ttftMeasure(t *testing.T, path, quant string) {
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if !rf.prefillReady {
		t.Fatal("batched prefill kernels did not load")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	build := func(n int) [][]float32 {
		embs := make([][]float32, n)
		var s uint32 = 12345
		for i := range embs {
			s = s*1664525 + 1013904223
			embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
		}
		return embs
	}
	// Confirm the batched path accepts this dense model rather than declining.
	if _, e := rf.PrefillLast(context.Background(), build(8), 0); e != nil {
		t.Fatalf("PrefillLast declined a dense qwen2.5 model: %v", e)
	}

	median := func(f func(), reps int) time.Duration {
		ds := make([]time.Duration, reps)
		for i := range ds {
			t0 := time.Now()
			f()
			ds[i] = time.Since(t0)
		}
		// simple min (best-of) — prefill is compute/bandwidth bound, min is the cleanest signal
		best := ds[0]
		for _, d := range ds {
			if d < best {
				best = d
			}
		}
		return best
	}

	noSeq := os.Getenv("GOINFER_TTFT_NOSEQ") != ""
	depths := make([]int, 0, 4)
	for _, f := range ttftCSV("GOINFER_TTFT_K", []string{"128", "512", "2048"}) {
		k, err := strconv.Atoi(f)
		if err != nil || k <= 0 {
			t.Fatalf("GOINFER_TTFT_K: bad depth %q", f)
		}
		depths = append(depths, k)
	}
	t.Logf("%-6s %12s %12s %8s", "N", "sequential", "batched", "speedup")
	for _, n := range depths {
		embs := build(n)
		seq := time.Duration(0)
		if !noSeq {
			seq = median(func() {
				for i := 0; i < n-1; i++ {
					if e := rf.ForwardNoLogits(embs[i], i); e != nil {
						t.Fatalf("seq pos %d: %v", i, e)
					}
				}
				if _, e := rf.Forward(embs[n-1], n-1); e != nil {
					t.Fatalf("seq last: %v", e)
				}
			}, 3)
		}
		bat := median(func() {
			if _, e := rf.PrefillLast(context.Background(), embs, 0); e != nil {
				t.Fatalf("batched n=%d: %v", n, e)
			}
		}, 3)
		// PREFILL_TTFT_ROW is a grep target: these runs are read back out of a detached log, and a
		// t.Logf table with a per-quant subtest prefix is awkward to machine-read across models.
		if noSeq {
			t.Logf("%-6d %12s %12v %8s", n, "(skipped)", bat, "-")
			t.Logf("PREFILL_TTFT_ROW model=%s quant=%s K=%d seq=- batched=%v speedup=-",
				filepath.Base(path), quant, n, bat)
			continue
		}
		t.Logf("%-6d %12v %12v %7.2fx", n, seq, bat, float64(seq)/float64(bat))
		t.Logf("PREFILL_TTFT_ROW model=%s quant=%s K=%d seq=%v batched=%v speedup=%.2f",
			filepath.Base(path), quant, n, seq, bat, float64(seq)/float64(bat))
	}
}

// ttftCSV reads a comma-separated env override, trimming blanks, and falls back to def when unset
// or empty after trimming. Shared by the quant and depth knobs so both behave the same way.
func ttftCSV(env string, def []string) []string {
	v := os.Getenv(env)
	if strings.TrimSpace(v) == "" {
		return def
	}
	out := make([]string, 0, 4)
	for _, f := range strings.Split(v, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		return def
	}
	return out
}
