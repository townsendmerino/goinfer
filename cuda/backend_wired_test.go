//go:build cuda && goinfer_testhooks

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
	requireHeavyModel(t)
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
	// A model needing features this backend does not implement SHOULD decline (that is the
	// admission gate working, not a regression) — skip rather than fail. For anything CUDA
	// does implement, a nil resident IS a regression.
	if missing := mc.MissingResidentFeatures(decoder.ResidentBackendFeatures("cuda")); len(missing) > 0 {
		t.Skipf("model needs unimplemented feature(s) %v — CUDA correctly declines to the staged path", missing)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("BuildResident did not engage — resident is nil; the wired --backend cuda path fell back to staged (wiring/driver regression)")
	}
	// Sliding-window WIRING check. The kernel semantics are proven bit-exact separately
	// (TestSlidingWindowAttention), but a window only engages past `window` keys — so if the
	// per-layer value were silently 0, the short-prompt parity below could not tell. Assert the
	// resident carries the arch's window on exactly the layers the decoder marks local.
	if w := mc.SlidingWindowResident(); w > 0 {
		cr, ok := rf.(*cudaResident)
		if !ok {
			t.Fatalf("resident is %T, want *cudaResident", rf)
		}
		for l := range cr.layers {
			want := int32(0)
			if mc.LayerIsLocalResident(l) {
				want = int32(w)
			}
			if got := cr.layers[l].window; got != want {
				t.Fatalf("layer %d: resident window = %d, want %d — the window is not reaching the kernel", l, got, want)
			}
		}
		t.Logf("sliding window %d wired on all %d layers (local per LayerIsLocalResident)", w, len(cr.layers))
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
	requireHeavyModel(t)
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
	// AUTOREGRESSIVE with real advancing positions (growing KV) — feed each step's argmax back
	// in, as decode actually runs. A fixed embedding at fixed positions would flatter the
	// number by keeping the KV tiny and the embedding hot.
	tok := 785
	pos := 0
	step := func() {
		l, err := rf.Forward(m.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatalf("fwd pos %d: %v", pos, err)
		}
		tok, pos = argmaxF(l), pos+1
	}
	for range 8 { // warm
		step()
	}
	best := 1e18
	for range 6 {
		const N = 16
		t0 := time.Now()
		for range N {
			step()
		}
		if dt := time.Since(t0).Seconds() / N; dt < best {
			best = dt
		}
	}
	// greedy fast path: same chain, 4 B readback instead of the logits vector
	rg, ok := rf.(decoder.ResidentGreedy)
	bestG := 1e18
	if ok {
		stepG := func() {
			id, err := rg.ForwardArgmax(m.EmbedResidentForTest(tok), pos)
			if err != nil {
				t.Fatalf("argmax fwd pos %d: %v", pos, err)
			}
			tok, pos = id, pos+1
		}
		for range 8 {
			stepG()
		}
		for range 6 {
			const N = 16
			t0 := time.Now()
			for range N {
				stepG()
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
	requireHeavyModel(t)
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
	// TAUTOLOGY GUARD (v1.0 gate §2 census). Both arms below are one build with one env var
	// flipped; the fast arm only DIFFERS if the resident implements ResidentGreedy, since that
	// interface assertion — not the env var — is what routes decode to the on-device argmax. If
	// cudaResident stopped satisfying it, both arms would take the logits path and this gate would
	// report "token-identical" having compared the logits path to itself. The kv-only sibling
	// (kvonly_prefill_test.go) already asserts its ResidentPrefillKV precondition this way.
	if _, ok := m.ResidentForwardForTest().(decoder.ResidentGreedy); !ok {
		t.Fatal("cuda resident does not implement ResidentGreedy — the fast arm would take the " +
			"logits path and this gate would compare it to itself")
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
