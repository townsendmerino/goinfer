package decoder

import (
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkDecodeSerialVsParallel — S-09.1, the corrected A/B.
//
// THE 2026-08-11 MEASUREMENT COMPARED TWO PARALLEL ARMS. BenchmarkDecode sets only the PROCESS
// GLOBAL (linalg.SetParallelThreshold), but since 2026-08-01 decode runs on a PER-WORKSPACE
// threshold that newDecodeScratch installs on every scratch it builds (scratch.go: `ws :=
// &linalg.Workspace{}; ws.SetThreshold(DefaultDecodeParallelThreshold)`). The global is
// therefore overridden before the first token, so GOINFER_PAR_THRESHOLD=<huge> did not make the
// "serial" arm serial — and "serial 54.77 vs parallel 54.34 tok/s" was two parallel runs
// differing by 0.8%, which is inside this box's noise.
//
// S-02's whole premise ("serial ties parallel, so fork/join is net-neutral") rests on that
// number, so it has to be re-taken against a genuinely serial arm.
//
// SERIAL IS PROVEN, NOT ASSUMED. Setting the threshold is the same kind of act that failed last
// time, so the arm also counts goroutine spawns: parallelSpawnCols starts one goroutine per
// shard per matmul, so a real serial arm shows a flat goroutine count while a parallel one
// spikes. The counter is sampled rather than instrumented because parallelSpawnCols is
// unexported in aikit — but a flat maximum across thousands of matmuls is unambiguous.
//
//	go test -tags realckpt -run '^$' -bench BenchmarkDecodeSerialVsParallel -benchtime 300x ./decoder/
func BenchmarkDecodeSerialVsParallel(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	width := 0
	if w, err := strconv.Atoi(os.Getenv("GOINFER_PAR_WIDTH")); err == nil {
		width = w
	}
	linalg.SetParallelWidth(width)

	for _, arm := range []struct {
		name string
		thr  int
	}{
		// 1<<62 MACs: no matmul this decode performs can reach it, so every span runs on the
		// calling goroutine. NOT GOINFER_PAR_THRESHOLD, which sets the global the workspace
		// overrides — that is the bug this benchmark exists to correct.
		{"serial", 1 << 62},
		{"parallel", DefaultDecodeParallelThreshold},
	} {
		b.Run(arm.name, func(b *testing.B) {
			prompt := []int{785, 264, 6573, 311, 1438, 279, 2038, 25}
			cache := m.NewCache(len(prompt) + b.N + 8)
			// THE FIX: the PER-WORKSPACE threshold, which is what decode actually consults.
			cache.scr.ws.SetThreshold(arm.thr)

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

			// Goroutine-count sampler: the spawn observable. Started after warm-up so the
			// baseline is the steady state, and stopped before the report.
			var maxG, baseG int64
			atomic.StoreInt64(&baseG, int64(runtime.NumGoroutine()))
			stop := make(chan struct{})
			done := make(chan struct{})
			go func() {
				defer close(done)
				t := time.NewTicker(200 * time.Microsecond)
				defer t.Stop()
				for {
					select {
					case <-stop:
						return
					case <-t.C:
						if g := int64(runtime.NumGoroutine()); g > atomic.LoadInt64(&maxG) {
							atomic.StoreInt64(&maxG, g)
						}
					}
				}
			}()

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
			close(stop)
			<-done

			// base+1 is this sampler goroutine itself; anything above that is fan-out.
			excess := max(atomic.LoadInt64(&maxG)-atomic.LoadInt64(&baseG)-1, 0)
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
			b.ReportMetric(float64(excess), "excess-goroutines")
			b.Logf("%s: threshold=%d peak goroutines %d (baseline %d, excess %d)",
				arm.name, arm.thr, atomic.LoadInt64(&maxG), atomic.LoadInt64(&baseG), excess)
		})
	}
}
