package decoder

import (
	"os"
	"strconv"
	"sync"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// benchModel loads the 0.5B once and caches it — the go test harness calls a
// benchmark function several times (growing b.N), and reloading (a full GGUF
// dequant+requant) each time would swamp the CPU profile with load cost instead
// of decode. Returns "" path → skip.
var (
	benchOnce  sync.Once
	benchModel *Model
	benchErr   error
)

func loadBenchModel() (*Model, error) {
	benchOnce.Do(func() {
		// lookupAsset, not assetPath: this runs inside a sync.Once with no testing.TB to skip
		// on. The error carries the predicate's reason, so a caller reporting benchErr says
		// WHICH candidate failed and why, instead of a bare stat error.
		path, err := lookupAsset("GOINFER_PREQUANT_GGUF")
		if err != nil {
			benchErr = err
			return
		}
		// int8int8 by DEFAULT, so every existing benchmark and the recorded decode
		// calibration are unchanged. The override exists because a quantization choice
		// here decides which kernel is measured at all:
		//
		//	int8int8 -> matmulInto -> linalg.MatmulBTW8A8Into   (never enters blockedFill)
		//	""/f32   -> matmulInto -> matmul -> MatmulBT -> blockedFill
		//
		// P10 measures aikit's f32 blocked-matmul rework, which lives in blockedFill, so
		// an int8int8 prefill run would exercise the changed code NOT AT ALL and return a
		// confident flat result measuring nothing. Set GOINFER_BENCH_QUANT="" for f32.
		quant := "int8int8"
		if q, ok := os.LookupEnv("GOINFER_BENCH_QUANT"); ok {
			quant = q
		}
		benchModel, benchErr = Load(path, Options{Quant: quant})
	})
	return benchModel, benchErr
}

// BenchmarkDecode times steady-state single-token decode (the chat case: batch=1, greedy). It loads the 0.5B at
// int8int8, prefills a short fixed prompt, then runs b.N forward+sample steps in the timed loop: the same
// forward/runLayers + LM head + sampler path Generate drives. Reports tok/s.
//
// DO NOT COMPARE TWO RUNS OF THIS TAKEN AT DIFFERENT TIMES. Session-level drift on a box (about 5% between a morning
// and an afternoon session) is larger than the effects this benchmark is used to settle, so interleave the arms in ONE
// session, alternating, and discard the first sample of each process. Pre-register the noise floor and the warm-up
// discard before running and apply them identically to both arms. Worked examples with raw samples:
// docs/measurements/aikit-v1.17.0-decode-ab.md and -v1.17.1-decode-ab.md.
//
// It is the perf campaign's regression guard; run with profiles:
//
//	go test ./decoder -run '^$' -bench BenchmarkDecode -benchmem \
//	  -cpuprofile cpu.out -memprofile mem.out -benchtime 5s
//
// Skips cleanly without the model asset (set GOINFER_PREQUANT_GGUF or drop the gguf in testdata), like the other
// model-dependent tests.
func BenchmarkDecode(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	// Measure the shipping config by default (the demo's decode threshold), so
	// this is a faithful regression guard. GOINFER_PAR_THRESHOLD overrides it (in
	// MACs; 0 = parallelize everything, huge = serial) for sweeps.
	thr := DefaultDecodeParallelThreshold
	if t := os.Getenv("GOINFER_PAR_THRESHOLD"); t != "" {
		if v, err := strconv.Atoi(t); err == nil {
			thr = v
		}
	}
	linalg.SetParallelThreshold(thr)
	// GOINFER_PAR_WIDTH caps the matmul fan-out (0 = GOMAXPROCS) — the P-core
	// straggler sweep.
	if w, err := strconv.Atoi(os.Getenv("GOINFER_PAR_WIDTH")); err == nil {
		linalg.SetParallelWidth(w)
	}
	// GOINFER_BENCH_ACT_GROUP sets aikit's process-wide activation group (0 = per-row, 32 =
	// per-32): the A/B arm of docs/tasks/task-actquant-pergroup-2026-09.md's end-to-end speed gate.
	if g, err := strconv.Atoi(os.Getenv("GOINFER_BENCH_ACT_GROUP")); err == nil {
		linalg.SetActQuantGroup(g)
		defer linalg.SetActQuantGroup(0)
	}

	// A short fixed prompt; greedy so the decode is deterministic.
	prompt := []int{785, 264, 6573, 311, 1438, 279, 2038, 25}
	cache := m.NewCache(len(prompt) + b.N + 8)
	for _, id := range prompt[:len(prompt)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			b.Fatalf("prefill: %v", err)
		}
	}
	sampler := NewSampler(SamplingParams{Temperature: 0})
	logits, err := m.forward(prompt[len(prompt)-1], cache)
	if err != nil {
		b.Fatalf("seed forward: %v", err)
	}
	next, err := sampler.Sample(logits)
	if err != nil {
		b.Fatalf("seed sample: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := m.forward(next, cache)
		if err != nil {
			b.Fatalf("forward: %v", err)
		}
		if next, err = sampler.Sample(logits); err != nil {
			b.Fatalf("sample: %v", err)
		}
	}
	b.StopTimer()

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}
