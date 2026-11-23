//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"math/rand/v2"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestCPUAttnKernelProbe splits CPU decode attention's time between its two acc64 kernels at one layer's shape, and sets each
// against this host's parallel read bandwidth over the same bytes (docs/measurements/cpu-kv-f16-2026-11-23/README.md § "Kernel
// split"). It answers whether halving the cache's bytes can speed a kernel: one that runs far under the bandwidth ceiling is bound
// by its arithmetic, not its reads. A measurement tool: it skips unless CPU_ATTN_KERNEL_PROBE=1. The defaults are Phi-3 mini's
// layer at depth 3,900: CPU_ATTN_PROBE_KEYS, _HEADS, _HD override them.
func TestCPUAttnKernelProbe(t *testing.T) {
	if os.Getenv("CPU_ATTN_KERNEL_PROBE") != "1" {
		t.Skip("set CPU_ATTN_KERNEL_PROBE=1")
	}
	envInt := func(k string, d int) int {
		var v int
		if _, err := fmt.Sscan(os.Getenv(k), &v); err == nil && v > 0 {
			return v
		}
		return d
	}
	nKeys, nH, hd := envInt("CPU_ATTN_PROBE_KEYS", 3900), envInt("CPU_ATTN_PROBE_HEADS", 32), envInt("CPU_ATTN_PROBE_HD", 96)
	kvDim := nH * hd
	rng := rand.New(rand.NewPCG(1, 2))
	keys, vals := make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
	for i := range keys {
		keys[i], vals[i] = rng.Float32()-0.5, rng.Float32()-0.5
	}
	q := make([]float32, kvDim)
	for i := range q {
		q[i] = rng.Float32() - 0.5
	}
	layerBytes := float64(4 * len(keys)) // one of K or V
	const reps = 20
	run := func(workers int, body func(h int, sc []float32, dst []float32, acc []float64)) time.Duration {
		best := time.Duration(1 << 62)
		for range reps {
			t0 := time.Now()
			var wg sync.WaitGroup
			per := (nH + workers - 1) / workers
			for w := range workers {
				h0, h1 := w*per, min((w+1)*per, nH)
				wg.Add(1)
				go func() {
					defer wg.Done()
					sc, dst, acc := make([]float32, nKeys), make([]float32, hd), make([]float64, hd)
					for i := range sc {
						sc[i] = 1 / float32(nKeys)
					}
					for h := h0; h < h1; h++ {
						body(h, sc, dst, acc)
					}
				}()
			}
			wg.Wait()
			best = min(best, time.Since(t0))
		}
		return best
	}
	var sink float32
	var mu sync.Mutex
	qk := func(h int, sc, _ []float32, _ []float64) {
		linalg.MatmulQKAcc64(q[h*hd:h*hd+hd], keys, sc, 1, hd, nKeys, h*hd, kvDim)
	}
	av := func(h int, sc, dst []float32, acc []float64) {
		linalg.MatmulAVAcc64(sc, vals, dst, acc, 1, nKeys, hd, h*hd, kvDim)
	}
	// The ceiling: read the same layer's K bytes in the same head-sliced pattern, one add a float, eight chains so the adds are
	// not the bound.
	readK := func(h int, _, _ []float32, _ []float64) {
		var s0, s1, s2, s3, s4, s5, s6, s7 float32
		for s := range nKeys {
			r := keys[s*kvDim+h*hd : s*kvDim+h*hd+hd]
			for d := 0; d+8 <= hd; d += 8 {
				s0 += r[d]
				s1 += r[d+1]
				s2 += r[d+2]
				s3 += r[d+3]
				s4 += r[d+4]
				s5 += r[d+5]
				s6 += r[d+6]
				s7 += r[d+7]
			}
		}
		mu.Lock()
		sink += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
		mu.Unlock()
	}
	fmt.Fprintf(os.Stderr, "[attn-kernel] layer shape: %d keys, %d heads x %d = kvDim %d; K (or V) %.1f MB f32; best of %d\n", nKeys, nH, hd, kvDim, layerBytes/1e6, reps)
	for _, w := range []int{1, 16} {
		tq, ta, tr := run(w, qk), run(w, av), run(w, readK)
		gb := func(d time.Duration) float64 { return layerBytes / d.Seconds() / 1e9 }
		fmt.Fprintf(os.Stderr, "=== ATTN KERNELS workers=%d: QK %.2f ms (%.1f GB/s), AV %.2f ms (%.1f GB/s), read-only ceiling %.2f ms (%.1f GB/s); per token at 32 layers: QK %.0f ms, AV %.0f ms ===\n",
			w, tq.Seconds()*1e3, gb(tq), ta.Seconds()*1e3, gb(ta), tr.Seconds()*1e3, gb(tr), 32*tq.Seconds()*1e3, 32*ta.Seconds()*1e3)
	}
	_ = sink
}
