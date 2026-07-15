//go:build cuda

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestBackendResidentWired exercises the PRODUCTION path end to end: decoder.Load with
// Backend:"cuda" must, via withResidency → BuildResident, engage the real cudaResident, and
// its Forward (the wired code that cmd/serve and demo/chat call — not the inline test
// harness) must match the CPU decode under goinfer's 3%-near-tie rule. This is the
// backend-equivalence gate for the shipped path; it catches wiring regressions the
// inline-forward parity test can't.
func TestBackendResidentWired(t *testing.T) {
	gguf := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model")
	}

	mc, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("BuildResident did not engage — resident is nil; the wired --backend cuda path fell back to staged (wiring/driver regression)")
	}

	mcpu, err := decoder.Load(gguf, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	prompt := []int{785, 3840, 315, 24231, 6137, 448, 264, 4285}
	cache := mcpu.NewCache(len(prompt) + 2)
	exact, hardFail, worst := 0, 0, 0.0
	for i, tok := range prompt {
		cpuL, _ := mcpu.ForwardForTest(tok, cache)
		cudaL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("pos %d: resident Forward: %v", i, err)
		}
		ca, ga := argmaxF(cpuL), argmaxF(cudaL)
		if ca == ga {
			exact++
			continue
		}
		lo, hi := cpuL[0], cpuL[0]
		for _, v := range cpuL {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
		if gap > worst {
			worst = gap
		}
		if gap > 0.03 {
			hardFail++
			t.Errorf("pos %d: CPU=%d CUDA=%d gap=%.3f%% > 3%% — wired backend diverges", i, ca, ga, gap*100)
		}
	}
	t.Logf("wired --backend cuda path: %d/%d exact | worst near-tie %.3f%% | hard fails %d", exact, len(prompt), worst*100, hardFail)
	if hardFail == 0 {
		t.Logf("WIRED GATE GREEN: production decoder.Load(cuda)→BuildResident→Forward matches CPU")
	}
}
