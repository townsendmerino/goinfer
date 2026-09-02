//go:build gpu

package gpu

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGraniteResidentSpeedup is the P7 deliverable: resident granite-4.0-h-tiny generates
// coherently on the resident SSM DecodeRunner, and the measured decode rate beats the CPU
// (300 ms/tok) and staged-webgpu (498 ms/tok) baselines. The resident path runs W8A8 (int8);
// its logits diverge from the f32 CPU reference (cosine ~0.37 — granite's 64-expert MoE
// selection + SSM are int8-sensitive, see docs/ssm-residency-build.md), but the engine is
// computationally correct (matches an int8 reference at cosine ~0.99) and the int8 output is
// coherent + factual. A tight f32 gate would need f16 weights (follow-up).
func TestGraniteResidentSpeedup(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("granite resident speedup")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	os.Setenv("GOINFER_SSM_RESIDENT", "1") // P5b/P6 guarded eligibility
	// Cold TTFT is the quantity this table reports, so resident prefix reuse is disabled
	// here. Without this the warm-up Generate leaves the prompt in the resident KV and the
	// TIMED Generate reuses it, collapsing TTFT to near zero and publishing a warm number
	// under a cold heading (decoder/resident_reuse.go).
	os.Setenv("GOINFER_NO_RESIDENT_REUSE", "1")
	defer os.Unsetenv("GOINFER_NO_RESIDENT_REUSE")
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Fatal("granite did not go resident — SSM engine not active")
	}
	prompt, _ := tk.Encode("The capital of France is", true)
	greedy := decoder.SamplingParams{Temperature: 0}

	// coherence: greedy continuation must contain the factual answer.
	ch, _ := m.Generate(context.Background(), prompt, 24, greedy)
	var out []int
	for tok := range ch {
		out = append(out, tok)
	}
	txt, _ := tk.Decode(out)
	t.Logf("resident granite generated: %q", txt)
	if !strings.Contains(txt, "Paris") {
		t.Errorf("incoherent generation (no 'Paris'): %q", txt)
	}

	// speedup: best steady-state inter-token rate over rounds (prefill excluded).
	best := 1e9
	for range 4 {
		ch, _ := m.Generate(context.Background(), prompt, 32, greedy)
		var n int
		var ttft time.Duration
		t0 := time.Now()
		for range ch {
			if n == 0 {
				ttft = time.Since(t0)
			}
			n++
		}
		if n > 1 {
			if ms := float64((time.Since(t0) - ttft).Microseconds()) / float64(n-1) / 1000; ms < best {
				best = ms
			}
		}
	}
	t.Logf("RESIDENT granite-4.0-h-tiny: %.1f ms/token = %.1f tok/s", best, 1000/best)
	t.Logf("  vs CPU 300 ms (3.3 tok/s): %.1fx | vs staged 498 ms (2.0 tok/s): %.1fx", 300/best, 498/best)
	if best > 150 { // generous bound; expected ~30-50 ms (6-15x). Fails only if residency broke.
		t.Errorf("resident decode %.1f ms/tok — no meaningful speedup over CPU", best)
	}
}
