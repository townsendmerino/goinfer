//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestSpecDecode is D1's end-to-end gate: n-gram-drafted, batched-verified speculative decode must
// (1) produce the EXACT plain-greedy token stream (lossless, by construction) and (2) be faster on a
// repetitive/self-similar workload. It greedy-generates a ground-truth stream, then reproduces it via
// the speculative loop (ngramDraft → PrefillLastN verify → accept longest greedy-matching prefix +
// bonus), asserting byte-identity and reporting the speedup + accept stats. Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestSpecDecode -v
func TestSpecDecode(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := func(id int) []float32 { return mc.EmbedResidentForTest(((id % vocab) + vocab) % vocab) }

	// Seed: a short prompt to prime the KV. Greedy on a coder model tends to self-similar output, which
	// is exactly where n-gram drafting pays — an honest measure of the mechanism (real-prompt accept
	// rates are the decoder-level follow-up).
	const S = 32
	seed := make([]int, S)
	var s uint32 = 20260804
	for i := range seed {
		s = s*1664525 + 1013904223
		seed[i] = int(s>>9) % vocab
	}
	seedEmb := make([][]float32, S)
	for i, tk := range seed {
		seedEmb[i] = emb(tk)
	}
	prime := func() []float32 {
		lg, e := rf.PrefillLast(seedEmb, 0)
		if e != nil {
			t.Fatalf("prime: %v", e)
		}
		return lg
	}

	const N = 200 // tokens to generate

	// --- ground truth: plain greedy ---
	lg := prime()
	gt := make([]int, 0, N)
	pos := S
	t0 := time.Now()
	for i := 0; i < N; i++ {
		tk := argmaxF(lg)
		gt = append(gt, tk)
		// Baseline uses the batched path per-token (PrefillLast M=1) — the SAME forward the verify uses,
		// so losslessness is w.r.t. the batched forward (the decode-step Forward differs at the last ULP
		// for pos>0, a batched-vs-decode rope/attention numerics gap under investigation).
		outs, e := rf.PrefillLast([][]float32{emb(tk)}, pos)
		if e != nil {
			t.Fatalf("plain decode %d: %v", i, e)
		}
		lg = outs
		pos++
	}
	plainDur := time.Since(t0)

	// --- speculative decode (re-prime KV first) ---
	prime()
	const k, ctxLen = 8, 2
	hist := append([]int(nil), seed...)
	var st SpecStats
	t1 := time.Now()
	for len(hist)-S < N {
		p := len(hist) - 1
		draft := ngramDraft(hist, k, ctxLen)
		feed := append([]int{hist[p]}, draft...)
		embs := make([][]float32, len(feed))
		for i, tk := range feed {
			embs[i] = emb(tk)
		}
		Ls, e := rf.PrefillLastN(embs, p) // Ls[i] = logits at position p+i (prediction for p+i+1)
		if e != nil {
			t.Fatalf("verify: %v", e)
		}
		st.Rounds++
		st.VerifyToks += len(feed)
		st.Drafted += len(draft)
		if len(draft) == 0 {
			st.PlainRounds++
		}
		accepted := 0
		for i := 0; i < len(draft); i++ {
			if argmaxF(Ls[i]) == draft[i] {
				accepted++
			} else {
				break
			}
		}
		st.Accepted += accepted
		for i := 0; i < accepted; i++ {
			hist = append(hist, draft[i])
		}
		hist = append(hist, argmaxF(Ls[accepted])) // bonus: the target's correct token after the accepted prefix
	}
	specDur := time.Since(t1)
	specStream := hist[S:]

	// --- gate 1: LOSSLESS (spec stream == plain greedy stream) ---
	if len(specStream) < N {
		t.Fatalf("spec produced %d < %d tokens", len(specStream), N)
	}
	for i := 0; i < N; i++ {
		if specStream[i] != gt[i] {
			t.Fatalf("LOSSLESS VIOLATION at token %d: spec %d vs greedy %d", i, specStream[i], gt[i])
		}
	}
	t.Logf("LOSSLESS: %d spec tokens == plain greedy, byte-for-byte", N)

	// --- report ---
	acceptRate := 0.0
	if st.Drafted > 0 {
		acceptRate = float64(st.Accepted) / float64(st.Drafted)
	}
	tokPerRound := float64(N) / float64(st.Rounds)
	t.Logf("plain:  %d tok in %.1f ms = %.1f tok/s", N, float64(plainDur.Microseconds())/1000, float64(N)/plainDur.Seconds())
	t.Logf("spec:   %d tok in %.1f ms = %.1f tok/s  →  %.2f×", N, float64(specDur.Microseconds())/1000, float64(N)/specDur.Seconds(), float64(plainDur)/float64(specDur))
	t.Logf("rounds=%d (plain-fallback=%d)  tokens/round=%.2f  drafted=%d accepted=%d (accept-rate=%.2f)  verify-toks=%d",
		st.Rounds, st.PlainRounds, tokPerRound, st.Drafted, st.Accepted, acceptRate, st.VerifyToks)
}
