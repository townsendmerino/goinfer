package decoder

import (
	"os"
	"path/filepath"
	"strings"
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
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	headDir := filepath.Join(home, "models", "qwen3-1.7b-eagle3")
	basePath := filepath.Join(home, "models", "qwen3-1.7b-q8_0.gguf")
	for _, p := range []string{filepath.Join(headDir, "model.safetensors"), basePath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s: %v", p, err)
		}
	}
	head := sharedEagleHead(t, headDir)
	// GOINFER_EAGLE_BASE overrides the base (e.g. a bf16 safetensors dir — the head was
	// trained on bf16 hidden states, so q8 likely depresses acceptance). f32 for a dir.
	loadPath, quant := basePath, "int8int8"
	if b := os.Getenv("GOINFER_EAGLE_BASE"); b != "" {
		loadPath = b
		if !strings.HasSuffix(b, ".gguf") {
			quant = "" // safetensors dir → f32 (max fidelity to the head's training)
		}
	}
	base := sharedEagleBase(t, loadPath, quant)
	tk, _ := tokenizer.LoadGGUF(basePath) // tokenizer (same vocab) from the local gguf
	L := base.w.arch.NumLayers
	capLayers := []int{2, L / 2, L - 3}
	embedOf := func(tok int, dst []float32) { base.embedToken(tok, dst) }

	prompt, _ := tk.Encode("<|im_start|>user\nWrite a short paragraph about the history of computing.<|im_end|>\n<|im_start|>assistant\n", true)
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
	for range M {
		next = feed(next) // greedy continuation
	}

	// At each position i (past the prompt), prefill the head's KV over toks[:i] then
	// draft K and compare to the realized continuation toks[i+1..].
	var sumAcc, sumOracle, sumTree, n int
	const treeB, treeD = 2, 5 // full binary tree to depth 5 = 62 nodes
	var cosPre, cosPost float64
	hist := make([]int, K+1)
	emb := make([]float32, head.Hidden())
	for i := len(prompt); i+K < len(toks); i++ {
		st := head.Prefill(base.be, embedOf, toks[:i], feats[:i], 0)
		draft := head.DraftFrom(base.be, st, embedOf, toks[i], feats[i], i, K)
		acc := 0
		for j := range K {
			if draft[j] == toks[i+1+j] {
				acc++
			} else {
				break
			}
		}
		sumAcc += acc
		hist[acc]++

		// ORACLE recurrence: feed the TARGET's actual feature at each step (instead of the
		// head's own hidden output). Isolates whether the self-recurrence feature is the
		// multi-step weak link (oracle≫self) vs attention/position (oracle≈self).
		sto := head.Prefill(base.be, embedOf, toks[:i], feats[:i], 0)
		oacc, tok := 0, toks[i]
		for j := range K {
			base.embedToken(tok, emb)
			lg, _ := head.Step(base.be, emb, feats[i+j], i+j, sto)
			if head.TargetID(argmax(lg)) != toks[i+1+j] {
				break
			}
			oacc++
			tok = toks[i+1+j]
		}
		sumOracle += oacc

		// TREE accept: root-branch B, depth K. The tree recovers positions where the
		// correct token toks[i+1] is in the head's top-B (not just top-1).
		str := head.Prefill(base.be, embedOf, toks[:i], feats[:i], 0)
		td := head.DraftTree(base.be, str, embedOf, toks[i], feats[i], i, treeB, treeD)
		tacc, parent := 0, -1
		for tacc < len(toks)-(i+1) { // walk the tree following the realized continuation
			hit := -1
			for _, ch := range td.Children(parent) {
				if td.Tokens[ch] == toks[i+1+tacc] {
					hit = ch
					break
				}
			}
			if hit < 0 {
				break
			}
			tacc++
			parent = hit
		}
		sumTree += tacc

		// How well does each candidate recurrence quantity predict the NEXT fused feature
		// feats[i+1] (what step-0's output must become)? cos(pre-norm resid) vs
		// cos(post-finalNorm). The oracle feeds feats[i+1] exactly.
		stp := head.Prefill(base.be, embedOf, toks[:i], feats[:i], 0)
		base.embedToken(toks[i], emb)
		_, hOut := head.Step(base.be, emb, feats[i], i, stp)
		post := append([]float32(nil), hOut...)
		rmsNorm(post, head.finalNorm, 1, head.Hidden(), head.normEps, false)
		cosPre += cosine(hOut, feats[i+1])
		cosPost += cosine(post, feats[i+1])
		n++
	}
	mean := float64(sumAcc) / float64(n)
	t.Logf("EAGLE autoregressive accept (K=%d, %d positions): mean accepted %.2f → ~%.2f tok/verify | histogram %v | ORACLE-feature mean %.2f",
		K, n, mean, mean+1, hist, float64(sumOracle)/float64(n))
	t.Logf("TREE accept (B=%d, D=%d): mean %.2f → ~%.2f tok/verify  [linear %.2f, oracle %.2f]", treeB, treeD, float64(sumTree)/float64(n), float64(sumTree)/float64(n)+1, mean, float64(sumOracle)/float64(n))
	t.Logf("recurrence-feature vs next fused feature: cos(pre-norm resid)=%.3f cos(post-finalNorm)=%.3f",
		cosPre/float64(n), cosPost/float64(n))
	// INC-4a FINDING: step-0 matches ~26-41% (the validated forward), but steps 1+ are
	// ~0 — the head's attention KV starts EMPTY here, so it has no prompt context. A
	// real EAGLE head keeps its own KV over the prompt (context-prefill), which needs
	// per-position target-hidden capture during prefill (inc 4b). Not asserted — a
	// diagnostic of where the multi-token draft stands.
}
