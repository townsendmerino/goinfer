package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"
)

// Re-measure the ABSOLUTE CPU prefill table of docs/benchmarks.md §A: dense 1.5B `int8int8`, prefill + 1 token, at prompt
// lengths 170, 620, 1520 and 3020.
//
// CONFIGURATION IS MATCHED TO THE ORIGINAL ON PURPOSE: same model class, same quant, same prompt lengths, same "prefill + 1
// token" quantity. A re-measurement that quietly changes the cell definition is not a re-measurement. Best-of-3 rather than
// the original's single shot is a method improvement, stated so the two are not read as identically obtained.
func TestPrefillAbsoluteTable(t *testing.T) {
	if os.Getenv("GOINFER_PREFILL_ABS") == "" {
		t.Skip("set GOINFER_PREFILL_ABS=1 (loads a 1.5B model, ~15 min)")
	}
	path := os.Getenv("GOINFER_PREFILL_ABS_MODEL")
	if path == "" {
		path = os.Getenv("HOME") + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s: %v", path, err)
	}
	quant := os.Getenv("GOINFER_PREFILL_ABS_QUANT")
	if quant == "" {
		quant = "int8int8" // the original table's quant
	}
	m, err := Load(path, Options{Quant: quant})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	start := time.Now()
	fmt.Fprintf(os.Stderr, "§A prefill re-measure: start %s  quant=%s GOMAXPROCS=%d\n",
		start.Format("15:04:05"), quant, runtime.GOMAXPROCS(0))
	fmt.Fprintf(os.Stderr, "%-8s %12s %12s %12s %10s\n", "tokens", "recorded", "measured", "tok/s", "speedup")

	recorded := map[int]float64{170: 3.3, 620: 19.7, 1520: 93.2, 3020: 334.9}
	vocab := m.w.arch.VocabSize
	for _, L := range []int{170, 620, 1520, 3020} {
		prompt := make([]int, L)
		for i := range prompt {
			prompt[i] = (i*131 + 7) % vocab
		}
		best := time.Duration(1<<62 - 1)
		for r := range 3 {
			cache := m.NewCache(L + 4)
			t0 := time.Now()
			if _, err := m.prefillLogits(context.Background(), prompt, cache); err != nil {
				t.Fatalf("prefill %d: %v", L, err)
			}
			d := time.Since(t0)
			if d < best {
				best = d
			}
			_ = r
		}
		sec := best.Seconds()
		fmt.Fprintf(os.Stderr, "%-8d %11.1fs %11.1fs %11.1f %9.2fx  [elapsed %s]\n",
			L, recorded[L], sec, float64(L)/sec, recorded[L]/sec, time.Since(start).Round(time.Second))
	}
	fmt.Fprintf(os.Stderr, "total %s\n", time.Since(start).Round(time.Second))
}
