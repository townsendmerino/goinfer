//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestRealModel_parityAndThroughput — the spike's GO/NO-GO: the full real qwen2.5-coder-
// 1.5b decode running on Metal (cgo-free), (a) argmax-parity vs the decoder's own CPU
// forward token-by-token (same int8 weights), and (b) device-timed decode tok/s vs the
// ~71 GO bar. Skips without the checkpoint.
func TestRealModel_parityAndThroughput(t *testing.T) {
	requireHeavyModel(t)
	// Default qwen2.5-coder; override to validate other families (Qwen3/Mistral/Phi-3):
	//   GOINFER_METAL_MODEL=/path/to/qwen3-1.7b-q4_k_m.gguf go test -run TestRealModel...
	path := os.Getenv("GOINFER_METAL_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	_, nL, _, nKV, hd, _, _ := m.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)

	// ---- (a) argmax parity vs CPU decode, in lockstep ----
	ids := []int{785, 12095, 8948, 264, 6236, 1140, 13, 358, 3003, 264} // arbitrary valid ids
	steps := 24
	tok, pos, mism, amaxMism := ids[0], 0, 0, 0
	var lastCos float64
	for i := range steps {
		cpu, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		gpu := r.Forward(tok, pos)
		ca, ga := argmaxF(cpu), argmaxF(gpu)
		lastCos = cosF(cpu, gpu)
		if fa := r.ForwardArgmax(tok, pos); fa != uint32(ga) { // fused argmax must equal argmax(full logits)
			amaxMism++
			t.Logf("pos %d: fused-argmax=%d != argmax(logits)=%d", pos, fa, ga)
		}
		if ca != ga {
			mism++
			t.Logf("pos %d: CPU argmax=%d GPU=%d (logit cos %.5f)", pos, ca, ga, lastCos)
		}
		if i+1 < len(ids) {
			tok = ids[i+1]
		} else {
			tok = ca // greedy continuation, driven by CPU so both stay in lockstep
		}
		pos++
	}
	t.Logf("argmax parity (Metal vs CPU decode): %d/%d match, last logit cosine=%.5f", steps-mism, steps, lastCos)
	if mism > steps/5 {
		t.Fatalf("argmax parity FAIL: %d/%d mismatches", mism, steps)
	}
	if amaxMism > 0 {
		t.Fatalf("fused-argmax lm head disagrees with argmax(logits) in %d/%d steps", amaxMism, steps)
	}
	t.Logf("fused-argmax lm head: %d/%d exact vs argmax(full logits)", steps, steps)

	// ---- (b) decode throughput (warm, best-of-N, wall = real per-token latency incl commit/wait) ----
	// Production decode path: ForwardArgmax (fused block-argmax lm head, no 608KB readback).
	for w := range 6 {
		r.ForwardArgmax(tok, pos+w)
	}
	base := pos + 6
	best := time.Hour
	for it := range 40 {
		t0 := time.Now()
		r.ForwardArgmax(tok, base+it)
		if dt := time.Since(t0); dt < best {
			best = dt
		}
	}
	tps := 1.0 / best.Seconds()
	t.Logf("METAL DECODE (cgo-free, int8, qwen2.5-coder-1.5b, ctx~%d): %.1f tok/s = %.2f ms/token (best-of-40 warm)",
		base, tps, best.Seconds()*1000)
	t.Logf("  vs WebGPU-on-Metal 31.9 (int8) = %.2fx | vs GO bar ~71 (int4 Ollama-anchored) = %.2fx | Ollama-Metal 83.3 (int4)",
		tps/31.9, tps/71.0)
}
