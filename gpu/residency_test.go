//go:build gpu

package gpu

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestResidency_matchesCPU is the end-to-end gate for the GPU full-residency
// decode path: a real int4 .giw loaded through decoder.Generate on the webgpu
// backend must produce greedy tokens matching the CPU int4 decode (the oracle).
// Needs a staged int4 .giw via GOINFER_W4A8_GIW (skips otherwise) and real HW.
// Reports the real-weights GPU decode tok/s and the prefill TTFT (option-(b)
// CPU-prefill + K/V upload).
func TestResidency_matchesCPU(t *testing.T) {
	path := os.Getenv("GOINFER_W4A8_GIW")
	if path == "" {
		t.Skip("set GOINFER_W4A8_GIW=<path to an int4 .giw>")
	}
	newOrSkipHW(t).Close() // real-HW gate (webgpu Load builds its own context)

	data, err := os.ReadFile(path)
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
	ids, err := tk.Encode("<|im_start|>user\nWrite a one-sentence definition of recursion.<|im_end|>\n<|im_start|>assistant\n", false)
	if err != nil {
		t.Fatal(err)
	}

	const N = 16
	greedy := decoder.SamplingParams{Temperature: 0}

	gen := func(backend string) (toks []int, ttft, total time.Duration, resident bool) {
		m, err := decoder.Load(path, decoder.Options{Backend: backend})
		if err != nil {
			t.Fatalf("Load(%s): %v", backend, err)
		}
		defer m.Close()
		resident = m.ResidentActive()
		t0 := time.Now()
		ch, _ := m.Generate(context.Background(), ids, N, greedy)
		for id := range ch {
			if len(toks) == 0 {
				ttft = time.Since(t0)
			}
			toks = append(toks, id)
		}
		total = time.Since(t0)
		return
	}

	gpuTok, ttft, gpuTotal, resident := gen("webgpu")
	if !resident {
		t.Skip("residency path not active (software adapter or ineligible arch) — nothing to gate")
	}
	cpuTok, _, _, _ := gen("cpu")

	matched := 0
	for i := 0; i < len(gpuTok) && i < len(cpuTok); i++ {
		if gpuTok[i] != cpuTok[i] {
			break
		}
		matched++
	}
	gpuTxt, _ := tk.Decode(gpuTok)
	decodeTps := float64(len(gpuTok)-1) / (gpuTotal - ttft).Seconds()
	t.Logf("GPU residency: %d tok | TTFT %.0f ms (CPU prefill + K/V upload) | decode %.1f tok/s",
		len(gpuTok), float64(ttft.Microseconds())/1000, decodeTps)
	t.Logf("matched %d/%d greedy tokens vs CPU int4 oracle", matched, N)
	t.Logf("GPU output: %q", gpuTxt)
	if matched < N {
		cpuTxt, _ := tk.Decode(cpuTok)
		t.Errorf("GPU/CPU greedy diverged at token %d/%d\n  gpu=%v\n  cpu=%v\n  cpu output: %q", matched, N, gpuTok, cpuTok, cpuTxt)
	}
}
