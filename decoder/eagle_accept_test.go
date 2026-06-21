package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestEagleAcceptedLength measures the 05 head's autoregressive accepted-prefix length
// against the base's greedy continuation (inc 4a): at each position the head drafts K
// tokens (step 0 from the fused target hidden, later steps from its own hidden), and
// we count the leading run that matches what the base actually decodes. Mean accepted
// ≈ (tokens committed per verify) − 1 — the EAGLE acceptance signal. (Greedy proxy;
// the lossless verify integration is inc 4b.) Run -v.
func TestEagleAcceptedLength(t *testing.T) {
	home, _ := os.UserHomeDir()
	headDir := filepath.Join(home, "models", "qwen3-1.7b-eagle3")
	basePath := filepath.Join(home, "models", "qwen3-1.7b-q8_0.gguf")
	for _, p := range []string{filepath.Join(headDir, "model.safetensors"), basePath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s: %v", p, err)
		}
	}
	head, err := LoadEagleHead(headDir)
	if err != nil {
		t.Fatalf("LoadEagleHead: %v", err)
	}
	defer head.Close()
	base, err := Load(basePath, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load base: %v", err)
	}
	defer base.Close()
	tk, _ := tokenizer.LoadGGUF(basePath)
	L := base.w.arch.NumLayers
	capLayers := []int{2, L / 2, L - 3}
	embedOf := func(tok int, dst []float32) { base.embedToken(tok, dst) }

	prompt, _ := tk.Encode("Write a short paragraph about the history of computing and its impact on society.", true)
	const M, K = 48, 6

	// Capture the target's fused feature at EVERY position (whole prompt + M greedy
	// continuation), so the head can prefill its KV over the full context.
	cache := base.NewCache(len(prompt) + M + 4)
	var toks []int        // token at each absolute position (from 0)
	var feats [][]float32 // fused target feature at each position
	feed := func(tok int) int {
		logits, hid, err := base.ForwardCapture(tok, cache, capLayers)
		if err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
		h3 := make([]float32, 0, 3*head.Hidden())
		for _, h := range hid {
			h3 = append(h3, h...)
		}
		toks = append(toks, tok)
		feats = append(feats, head.Fuse(base.be, h3))
		return argmax(logits)
	}
	var next int
	for _, id := range prompt {
		next = feed(id) // teacher-forced over the prompt
	}
	for s := 0; s < M; s++ {
		next = feed(next) // greedy continuation
	}

	// At each position i (past the prompt), prefill the head's KV over toks[:i] then
	// draft K and compare to the realized continuation toks[i+1..].
	var sumAcc, n int
	hist := make([]int, K+1)
	for i := len(prompt); i+K < len(toks); i++ {
		st := head.Prefill(base.be, embedOf, toks[:i], feats[:i], 0)
		draft := head.DraftFrom(base.be, st, embedOf, toks[i], feats[i], i, K)
		acc := 0
		for j := 0; j < K; j++ {
			if draft[j] == toks[i+1+j] {
				acc++
			} else {
				break
			}
		}
		sumAcc += acc
		hist[acc]++
		n++
	}
	mean := float64(sumAcc) / float64(n)
	t.Logf("EAGLE autoregressive accept (K=%d, %d positions): mean accepted %.2f → ~%.2f tok/verify | histogram %v",
		K, n, mean, mean+1, hist)
	// INC-4a FINDING: step-0 matches ~26-41% (the validated forward), but steps 1+ are
	// ~0 — the head's attention KV starts EMPTY here, so it has no prompt context. A
	// real EAGLE head keeps its own KV over the prompt (context-prefill), which needs
	// per-position target-hidden capture during prefill (inc 4b). Not asserted — a
	// diagnostic of where the multi-token draft stands.
}
