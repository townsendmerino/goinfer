//go:build gpu

package gpu

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDecisionMatrix measures one model's decode paths end-to-end via
// decoder.Generate (warm — the first run is discarded so first-token pipeline
// compilation doesn't poison the rate). Env-driven, one model per invocation:
//
//	GOINFER_MATRIX_GGUF=<source.gguf> GOINFER_MATRIX_INT4=<int4.giw> \
//	GOINFER_MATRIX_LABEL=1.5B go test -tags gpu -run TestDecisionMatrix -v
//
// Prints a markdown-ish row per path for docs/gpu-assessment.md §1.
func TestDecisionMatrix(t *testing.T) {
	gguf := os.Getenv("GOINFER_MATRIX_GGUF")
	int4 := os.Getenv("GOINFER_MATRIX_INT4")
	label := os.Getenv("GOINFER_MATRIX_LABEL")
	if gguf == "" || int4 == "" {
		t.Skip("set GOINFER_MATRIX_GGUF, GOINFER_MATRIX_INT4, GOINFER_MATRIX_LABEL")
	}
	newOrSkipHW(t).Close() // real-HW gate
	// Cold TTFT is the quantity this table reports, so resident prefix reuse is disabled
	// here. Without this the warm-up Generate leaves the prompt in the resident KV and the
	// TIMED Generate reuses it, collapsing TTFT to near zero and publishing a warm number
	// under a cold heading (decoder/resident_reuse.go).
	os.Setenv("GOINFER_NO_RESIDENT_REUSE", "1")
	defer os.Unsetenv("GOINFER_NO_RESIDENT_REUSE")

	// tokenizer (from the int4 .giw's metadata GGUF) + a short prompt and a
	// ~256-token prompt for TTFT.
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
	shortIDs, _ := tk.Encode("Write a short note about recursion.\n", true)
	longIDs, _ := tk.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40), true)
	if len(longIDs) > 256 {
		longIDs = longIDs[:256]
	}

	const greedyN = 12
	greedy := decoder.SamplingParams{Temperature: 0}
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

	t.Logf("=== %s ===", label)
	t.Logf("%-22s | %-22s | %-9s | %-8s | %s", "path", "fits / resident", "tok/s", "TTFT256", "note")

	row := func(name, path string, opts decoder.Options, noResidency bool) {
		if noResidency {
			os.Setenv("GOINFER_NO_RESIDENCY", "1")
		} else {
			os.Unsetenv("GOINFER_NO_RESIDENCY")
		}
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Logf("%-22s | LOAD FAILED: %v", name, err)
			return
		}
		defer m.Close()
		gpu := opts.Backend == "webgpu"
		resident := m.ResidentActive()
		vram := ""
		fits := "yes"
		if gpu {
			vram = nvidiaSmiUsedMiB() // MiB used right after load (incl ~860 desktop)
			if name == "GPU residency int8" && !resident {
				fits = "no → staged" // BuildResident refused (OOM / too big)
			}
		}
		// warm (discard — compiles GPU pipelines via the prefill), then timed.
		if ch, _ := m.Generate(context.Background(), shortIDs, 4, greedy); ch != nil {
			consume(ch)
		}
		ch, _ := m.Generate(context.Background(), shortIDs, greedyN, greedy)
		n, ttftShort, total := consume(ch)
		tps := float64(n-1) / (total - ttftShort).Seconds()
		// TTFT on the ~256-token prompt (option-(a) prefill is O(prompt-len)).
		ch2, _ := m.Generate(context.Background(), longIDs, 1, greedy)
		_, ttft256, _ := consume(ch2)

		fitsCol := fits
		if gpu {
			fitsCol = fits + " / " + vram + " MiB"
		} else {
			fitsCol = "yes (cpu)"
		}
		t.Logf("%-22s | %-22s | %7.1f   | %6.0fms | tokens=%d", name, fitsCol, tps, float64(ttft256.Milliseconds()), n)
	}

	row("CPU int8", gguf, decoder.Options{Backend: "cpu", Quant: "int8int8"}, false)
	row("CPU int4", int4, decoder.Options{Backend: "cpu"}, false)
	row("GPU staged (int8)", gguf, decoder.Options{Backend: "webgpu", Quant: "int8int8"}, true)
	row("GPU residency int8", gguf, decoder.Options{Backend: "webgpu", Quant: "int8int8"}, false)
	row("GPU residency int4", int4, decoder.Options{Backend: "webgpu"}, false)
}

func nvidiaSmiUsedMiB() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return "?"
	}
	s := strings.TrimSpace(string(out))
	if _, e := strconv.Atoi(s); e != nil {
		return "?"
	}
	return s
}
