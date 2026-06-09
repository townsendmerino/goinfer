package decoder

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestCPUInt4Bench re-measures the docs/gpu-assessment.md §1 decision-matrix CPU
// rows — int4 (.giw) vs int8 (gguf) — through decoder.Generate, replicating
// TestDecisionMatrix's CPU-row methodology exactly (warm 4 tokens discarded, then
// 12 greedy/temp-0 tokens timed; tok/s = (n-1)/decode-seconds) but WITHOUT the
// GPU-HW gate so the CPU rows run on their own. Env-gated out of the default
// suite. Run, once per size:
//
//	GOINFER_CPUINT4_GGUF=<gguf> GOINFER_CPUINT4_INT4=<int4.giw> \
//	GOINFER_CPUINT4_LABEL=1.5B go test ./decoder/ -run TestCPUInt4Bench -v
func TestCPUInt4Bench(t *testing.T) {
	gguf := os.Getenv("GOINFER_CPUINT4_GGUF")
	int4 := os.Getenv("GOINFER_CPUINT4_INT4")
	label := os.Getenv("GOINFER_CPUINT4_LABEL")
	if gguf == "" || int4 == "" {
		t.Skip("set GOINFER_CPUINT4_GGUF, GOINFER_CPUINT4_INT4, GOINFER_CPUINT4_LABEL")
	}

	// Tokenizer from the int4 .giw's metadata GGUF (same prompt for both rows).
	data, err := os.ReadFile(int4)
	if err != nil {
		t.Fatal(err)
	}
	_, tokGGUF, err := giw.Read(data)
	if err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.LoadGGUFBytes(tokGGUF)
	if err != nil {
		t.Fatal(err)
	}
	promptIDs, _ := tk.Encode("Write a short note about recursion.\n", true)

	const greedyN = 12
	greedy := SamplingParams{Temperature: 0}
	consume := func(ch <-chan int) (n int, ttft, total time.Duration) {
		t0 := time.Now()
		for range ch {
			if n == 0 {
				ttft = time.Since(t0)
			}
			n++
		}
		return n, ttft, time.Since(t0)
	}
	row := func(name, path string, opts Options) {
		m, err := Load(path, opts)
		if err != nil {
			t.Fatalf("%s load(%s): %v", name, path, err)
		}
		defer m.Close()
		if ch, _ := m.Generate(context.Background(), promptIDs, 4, greedy); ch != nil { // warm
			consume(ch)
		}
		ch, _ := m.Generate(context.Background(), promptIDs, greedyN, greedy)
		n, ttft, total := consume(ch)
		tps := float64(n-1) / (total - ttft).Seconds()
		t.Logf("MATRIX-CPU %-6s | %-9s | %7.2f tok/s | decode=%v tokens=%d", label, name, tps, (total - ttft).Round(time.Millisecond), n)
	}
	row("CPU int8", gguf, Options{Backend: "cpu", Quant: "int8int8"})
	row("CPU int4", int4, Options{Backend: "cpu"})
}
