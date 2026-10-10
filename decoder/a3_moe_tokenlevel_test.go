//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

// Does the A3 MoE divergence change the tokens a user sees? A cosine on hidden states does not say: dense
// divergence is smooth numeric drift, a MoE's is largely discontinuous routing flips, and the same cosine can
// behave differently in generated text.
//
// The flag perturbs PREFILL only (forwardLayersN); decode is the same code on both sides, so any divergence in
// the continuation is downstream of the prefill's KV cache and first logits. Greedy decode is a sequence of
// argmaxes, so one near-tie flip makes every later token differ; a mismatch count would mislead. This reports
// the FIRST divergence and the baseline's top1-vs-top2 margin there, normalized by the logit range (the
// repo's 3%-of-range near-tie rule, gpu/kv_f16_test.go and gpu/kv_i8_parity_test.go). Context and figures:
// docs/code-notes/decoder.md#TestA3MoETokenLevel.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DIAG=1 GOINFER_MELLUM_CKPT=~/models/mellum2-unq \
//	GOINFER_MELLUM_K=2048 GOINFER_MELLUM_N=48 \
//	go test -count=1 -tags goinfer_testhooks ./decoder/ -run TestA3MoETokenLevel -v -timeout 120m
func TestA3MoETokenLevel(t *testing.T) {
	if os.Getenv("GOINFER_DIAG") == "" {
		t.Skip("DIAGNOSTIC (set GOINFER_DIAG=1): reports evidence for a judgement.")
	}
	path := assetPath(t, "GOINFER_MELLUM_CKPT")
	requireHeavyModel(t)
	K, N := 2048, 48
	if v := os.Getenv("GOINFER_MELLUM_K"); v != "" {
		fmt.Sscanf(v, "%d", &K)
	}
	if v := os.Getenv("GOINFER_MELLUM_N"); v != "" {
		fmt.Sscanf(v, "%d", &N)
	}
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.MoE == nil {
		t.Fatalf("%s is not a MoE", path)
	}
	if !m.canBatchN(K) {
		t.Fatalf("canBatchN(%d) = false", K)
	}
	vocab := m.w.arch.VocabSize
	prompt := make([]int, K)
	for i := range prompt {
		prompt[i] = (i*131 + 7) % vocab
	}

	// margin returns (top1-top2) as a fraction of the logit range — the near-tie statistic the
	// f16/int8 KV parity tests already key on, so a number here is comparable to those.
	margin := func(lg []float32) float64 {
		b1, b2 := math.Inf(-1), math.Inf(-1)
		lo := math.Inf(1)
		for _, v := range lg {
			f := float64(v)
			if f > b1 {
				b1, b2 = f, b1
			} else if f > b2 {
				b2 = f
			}
			if f < lo {
				lo = f
			}
		}
		if b1 == lo {
			return 0
		}
		return (b1 - b2) / (b1 - lo)
	}

	run := func(probe bool) ([]int, []float64) {
		t.Helper()
		setKnob(t, m, knobCPUFastAttention, map[bool]string{true: "1", false: "0"}[probe])
		ctx := deadlineCtx(t)
		cache := m.NewCache(K + N + 8)
		start := time.Now()
		lg, err := m.prefillLogits(ctx, prompt, cache)
		if err != nil {
			t.Fatalf("prefill probe=%v: %v", probe, err)
		}
		fmt.Fprintf(os.Stderr, "A3-token: probe=%v prefill %.1fs; decoding %d\n", probe, time.Since(start).Seconds(), N)
		toks, margins := make([]int, 0, N), make([]float64, 0, N)
		for step := 0; step < N; step++ {
			tok := argmax(lg)
			toks = append(toks, tok)
			margins = append(margins, margin(lg))
			if lg, err = m.forward(tok, cache); err != nil {
				t.Fatalf("decode step %d probe=%v: %v", step, probe, err)
			}
		}
		return toks, margins
	}

	base, baseMargins := run(false)
	fast, _ := run(true)

	first, diff := -1, 0
	for i := range base {
		if base[i] != fast[i] {
			if first < 0 {
				first = i
			}
			diff++
		}
	}
	if first < 0 {
		fmt.Fprintf(os.Stderr,
			"A3-MoE-tokenlevel: K=%d N=%d — continuations are IDENTICAL (%d/%d tokens agree)\n",
			K, N, N, N)
		return
	}
	// Everything from `first` on is downstream of one flip, so the honest headline is WHERE it
	// diverged and at what margin, not how many tokens differ after it.
	fmt.Fprintf(os.Stderr,
		"A3-MoE-tokenlevel: K=%d N=%d\n"+
			"  first divergence at token %d/%d (%d differ after it — all downstream of that one flip)\n"+
			"  baseline margin there: %.4f of logit range  (>0.03 = a REAL preference change, not a near-tie)\n"+
			"  baseline tok %d -> f32 tok %d\n"+
			"  agreement before divergence: %d/%d\n",
		K, N, first, N, diff, baseMargins[first], base[first], fast[first], first, N)
}
