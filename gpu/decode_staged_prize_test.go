//go:build gpu

package gpu

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDecodeStaged_prize sizes the §2 (widen resident-runner eligibility) lever: the
// per-token cost of the STAGED fallback path (recreates ~330 bind groups + uniforms +
// scratch buffers per token) vs the resident DecodeRunner. It measures the SAME model
// both ways (resident, then GOINFER_NO_RESIDENCY=1 → staged) so the delta is purely the
// path, no size-normalization. A genuinely-ineligible family (Gemma-4 softcap) is also
// measured to confirm the staged cost on weights that REQUIRE the fallback.
//
// Pure measurement; reports best (throttle-free) inter-token rate via decoder.Generate.
// G-10: a Benchmark, not a Test. It reports numbers and asserts nothing, so as a Test*
// its green said only that the harness ran — not that the effect it maps is there.
// Go runs a benchmark this slow exactly once (N=1 already exceeds benchtime).
func BenchmarkDecodeStaged_prize(b *testing.B) {
	requireHeavyModel(b)
	if testing.Short() {
		b.Skip("staged prize")
	}
	home, _ := os.UserHomeDir()
	// best inter-token ms/token over `rounds` Generate calls (prefill/TTFT excluded).
	bestMsPerTok := func(path string, staged bool) (float64, bool, error) {
		if _, err := os.Stat(path); err != nil {
			return 0, false, nil // missing → skip
		}
		if staged {
			os.Setenv("GOINFER_NO_RESIDENCY", "1")
			defer os.Unsetenv("GOINFER_NO_RESIDENCY")
		} else {
			os.Unsetenv("GOINFER_NO_RESIDENCY")
		}
		tk, err := tokenizer.LoadGGUF(path)
		if err != nil {
			return 0, false, err
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
		if err != nil {
			return 0, false, err
		}
		defer m.Close()
		resident := m.ResidentActive()
		prompt, _ := tk.Encode("Write a short note about recursion in programming.\n", true)
		greedy := decoder.SamplingParams{Temperature: 0}
		rate := func(n int) float64 {
			ch, _ := m.Generate(context.Background(), prompt, n, greedy)
			if ch == nil {
				return 0
			}
			var ttft, total time.Duration
			cnt := 0
			t0 := time.Now()
			for range ch {
				if cnt == 0 {
					ttft = time.Since(t0)
				}
				cnt++
			}
			total = time.Since(t0)
			if cnt < 2 {
				return 0
			}
			return float64((total - ttft).Microseconds()) / float64(cnt-1) / 1000 // ms/token
		}
		rate(8) // warm
		best := 1e9
		for range 6 {
			if v := rate(48); v > 0 && v < best {
				best = v
			}
		}
		return best, resident, nil
	}

	type row struct {
		label, path string
		staged      bool
	}
	qwen := home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	rows := []row{
		{"Qwen2.5-1.5B resident", qwen, false},
		{"Qwen2.5-1.5B staged (NO_RESIDENCY)", qwen, true},
		{"Gemma-4-E2B (softcap → forced staged)", home + "/models/gemma-4-E2B_q4_0-it.gguf", false},
	}
	b.Logf("=== §2 staged-vs-resident per-token (best inter-token, resident int8) ===")
	var residentMs, stagedMs float64
	for _, r := range rows {
		ms, resident, err := bestMsPerTok(r.path, r.staged)
		if err != nil {
			b.Logf("  %-38s LOAD FAILED: %v", r.label, err)
			continue
		}
		if ms == 0 {
			b.Logf("  %-38s skipped (model not on box)", r.label)
			continue
		}
		b.Logf("  %-38s %6.2f ms/token = %5.1f tok/s   [resident=%v]", r.label, ms, 1000/ms, resident)
		if r.label == "Qwen2.5-1.5B resident" {
			residentMs = ms
		}
		if r.staged {
			stagedMs = ms
		}
	}
	if residentMs > 0 && stagedMs > 0 {
		b.Logf("  → §2 prize (same model, staged − resident): %.2f ms/token recoverable (%.0f%% of staged)",
			stagedMs-residentMs, (stagedMs-residentMs)/stagedMs*100)
	}
}
