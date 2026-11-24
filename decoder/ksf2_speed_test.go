package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"
)

// TestKSF2_cpuDecodeSpeed is KS-F2's G2 instrument (docs/measurements/ks-f2-2026-10/README.md): one CPU run of a model, a prompt of
// KSF2_DEPTH pseudo-random token ids (attention's cost depends on the depth, not on the text), then KSF2_TOKENS greedy decode tokens
// timed from the first token to the last, so the prefill is not in the figure. It writes one JSON line to KSF2_OUT. It uses only
// exported API so the same file builds at the revision before the change, which is the A arm. A measurement tool, not a check: it
// skips unless KSF2_MODEL is set.
//
//	KSF2_MODEL=<model> KSF2_DEPTH=2048 KSF2_TOKENS=32 KSF2_OUT=<file> KSF2_ARM=A go test -run '^TestKSF2_cpuDecodeSpeed$' -v ./decoder/
func TestKSF2_cpuDecodeSpeed(t *testing.T) {
	model := os.Getenv("KSF2_MODEL")
	if model == "" {
		t.Skip("set KSF2_MODEL, KSF2_DEPTH, KSF2_TOKENS and KSF2_OUT")
	}
	envInt := func(k string, def int) int {
		if v := os.Getenv(k); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				t.Fatalf("%s=%q: %v", k, v, err)
			}
			return n
		}
		return def
	}
	depth, n := envInt("KSF2_DEPTH", 128), envInt("KSF2_TOKENS", 32)
	t0 := time.Now()
	m, err := Load(model, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", model, err)
	}
	defer m.Close()
	load := time.Since(t0)
	_, _, _, _, _, _, vocab := m.Dims()
	r := rand.New(rand.NewSource(int64(depth)))
	lo := min(1000, vocab/4) // past the special ids, which sit at the low end of most vocabularies
	prompt := make([]int, depth)
	for i := range prompt {
		prompt[i] = lo + r.Intn(min(vocab-lo, 30000))
	}
	t1 := time.Now()
	ch, g := m.Generate(context.Background(), prompt, n+1, SamplingParams{})
	var first, last time.Time
	got := 0
	for range ch {
		if got == 0 {
			first = time.Now()
		}
		last = time.Now()
		got++
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got != n+1 {
		t.Fatalf("generated %d tokens, want %d (a stop token cut the run short)", got, n+1)
	}
	rec := map[string]any{"arm": os.Getenv("KSF2_ARM"), "model": model, "depth": depth, "tokens": n,
		"load_s": load.Seconds(), "prefill_s": first.Sub(t1).Seconds(), "decode_ms_per_token": float64(last.Sub(first).Microseconds()) / 1000 / float64(n)}
	raw, _ := json.Marshal(rec)
	if out := os.Getenv("KSF2_OUT"); out != "" {
		f, err := os.OpenFile(out, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		fmt.Fprintln(f, string(raw))
	}
	fmt.Fprintln(os.Stderr, "ksf2:", string(raw))
}
