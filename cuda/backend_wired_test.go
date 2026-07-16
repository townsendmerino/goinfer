//go:build cuda

package cuda

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestBackendResidentWired exercises the PRODUCTION path end to end: decoder.Load with
// Backend:"cuda" must, via withResidency → BuildResident, engage the real cudaResident, and
// its Forward (the wired code that cmd/serve and demo/chat call — not the inline test
// harness) must match the CPU decode under goinfer's 3%-near-tie rule. This is the
// backend-equivalence gate for the shipped path; it catches wiring regressions the
// inline-forward parity test can't.
func TestBackendResidentWired(t *testing.T) {
	gguf := os.Getenv("GOINFER_CUDA_MODEL")
	if gguf == "" {
		gguf = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
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

// TestProdThroughput measures the SHIPPED path's tok/s — decoder.Load(cuda) → resident
// Forward, which returns full logits (a 151936×4 B = 608 KB D2H + sync every token) because
// the ResidentForward contract must feed the sampler / constrained decode / logprobs. The
// e2e harness's headline instead uses on-device argmax + a 4 B readback. The delta between
// them IS the cost the audit's "wire argmax_reduce into the greedy path" lever would recover
// — measured here rather than assumed, since recovering it needs an interface addition
// (a greedy-only fast path), not a drop-in.
func TestProdThroughput(t *testing.T) {
	gguf := os.Getenv("GOINFER_CUDA_MODEL")
	if gguf == "" {
		gguf = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model")
	}
	m, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("resident nil — cuda path declined")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	emb := m.EmbedResidentForTest(785)
	for i := 0; i < 8; i++ { // warm
		if _, err := rf.Forward(emb, i); err != nil {
			t.Fatalf("warm: %v", err)
		}
	}
	best := 1e18
	for b := 0; b < 6; b++ {
		const N = 16
		t0 := time.Now()
		for i := 0; i < N; i++ {
			if _, err := rf.Forward(emb, 8+i); err != nil {
				t.Fatalf("fwd: %v", err)
			}
		}
		if dt := time.Since(t0).Seconds() / N; dt < best {
			best = dt
		}
	}
	// greedy fast path: same chain, 4 B readback instead of the logits vector
	rg, ok := rf.(decoder.ResidentGreedy)
	bestG := 1e18
	if ok {
		for i := 0; i < 8; i++ {
			_, _ = rg.ForwardArgmax(emb, i)
		}
		for b := 0; b < 6; b++ {
			const N = 16
			t0 := time.Now()
			for i := 0; i < N; i++ {
				if _, err := rg.ForwardArgmax(emb, 8+i); err != nil {
					t.Fatalf("argmax fwd: %v", err)
				}
			}
			if dt := time.Since(t0).Seconds() / N; dt < bestG {
				bestG = dt
			}
		}
	}
	t.Logf("logits path (full %.0f KB D2H/token): %.2f ms | %.1f tok/s", float64(vocab*4)/1024, best*1000, 1/best)
	t.Logf("greedy fast path (on-device argmax, 4 B D2H): %.2f ms | %.1f tok/s  → +%.1f%%",
		bestG*1000, 1/bestG, (best/bestG-1)*100)
}

// TestGreedyFastPathIdentical is the gate for the on-device-argmax greedy fast path: the
// tokens Generate emits with the fast path MUST be identical to the full-logits path. This
// exercises the decode-loop restructure (fastNext), not just the kernel — the real risk. If
// the two ever diverge, the fast path is silently changing output and must not ship.
func TestGreedyFastPathIdentical(t *testing.T) {
	gguf := os.Getenv("GOINFER_CUDA_MODEL")
	if gguf == "" {
		gguf = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model")
	}
	m, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.ResidentForwardForTest() == nil {
		t.Skip("cuda resident declined")
	}
	prompt := []int{785, 3840, 315, 24231, 6137, 448, 264, 4285}
	gen := func() []int {
		ch, g := m.Generate(context.Background(), prompt, 24, decoder.SamplingParams{}) // greedy
		var got []int
		for tok := range ch {
			got = append(got, tok)
		}
		if g.Err() != nil {
			t.Fatalf("generate: %v", g.Err())
		}
		return got
	}
	t.Setenv("GOINFER_NO_GREEDY_FASTPATH", "1")
	slow := gen()
	t.Setenv("GOINFER_NO_GREEDY_FASTPATH", "")
	fast := gen()

	if len(slow) == 0 {
		t.Fatal("no tokens generated")
	}
	if !slices.Equal(slow, fast) {
		t.Fatalf("GREEDY FAST PATH DIVERGES — must not ship.\n  logits path: %v\n  fast path:   %v", slow, fast)
	}
	t.Logf("greedy fast path token-identical to the logits path over %d tokens: %v", len(fast), fast)
}
