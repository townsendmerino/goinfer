package decoder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestEagleAlpha measures the EAGLE head's true per-position acceptance under
// rejection sampling — α = 1 − TV(p, q) = Σ_x min(p(x), q(x)), where p is the target's
// softmax and q is the head's full draft distribution (softmax over the draft vocab,
// scattered to the target vocab via d2t). This is the §00-core acceptance identity and
// the metric EAGLE's published ~0.8 is reported in — far more informative than the
// greedy top-1 match (~0.4). If α is high, SAMPLED rejection (not greedy exact-match)
// is the lever; if it's also ~0.4, the head/protocol is mis-calibrated. Run -v.
func TestEagleAlpha(t *testing.T) {
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
	loadPath, quant := basePath, "int8int8"
	if b := os.Getenv("GINFER_EAGLE_BASE"); b != "" {
		loadPath = b
		if !strings.HasSuffix(b, ".gguf") {
			quant = ""
		}
	}
	base, err := Load(loadPath, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load %s: %v", loadPath, err)
	}
	defer base.Close()
	tk, _ := tokenizer.LoadGGUF(basePath)
	L := base.w.arch.NumLayers
	// CHAT-formatted prompt: the head is trained on the target's hidden states over
	// chat data, so the in-distribution regime (chatml + the model's own generation)
	// is where its acceptance is real — raw completion text is out-of-distribution.
	prompt, _ := tk.Encode("<|im_start|>user\nExplain how a transformer neural network works, in a few sentences.<|im_end|>\n<|im_start|>assistant\n", true)
	const M = 32

	// Decode the base greedily once, recording each position's token + all-layer hidden
	// (capture every layer so the sweep is free), so we can score α for several
	// capture-layer triples without re-running the base per config.
	allLayers := make([]int, L)
	for i := range allLayers {
		allLayers[i] = i
	}
	cache := base.NewCache(len(prompt) + M + 4)
	for i := 0; i < len(prompt)-1; i++ {
		base.forward(prompt[i], cache)
	}
	cur := prompt[len(prompt)-1]
	pos := len(prompt) - 1
	type step struct {
		tok, next int
		hidden    [][]float32 // per layer [hidden]
		p         []float64   // target softmax
	}
	var steps []step
	for range M {
		logits, hid, err := base.ForwardCapture(cur, cache, allLayers)
		if err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
		hc := make([][]float32, L)
		for l := range hid {
			hc[l] = append([]float32(nil), hid[l]...)
		}
		next := argmax(logits)
		steps = append(steps, step{tok: cur, next: next, hidden: hc, p: softmaxStable(logits, 1)})
		cur = next
		pos++
	}

	score := func(cap []int) (alpha, greedy float64) {
		emb := make([]float32, head.Hidden())
		for i, s := range steps {
			h3 := make([]float32, 0, 3*head.Hidden())
			for _, cl := range cap {
				h3 = append(h3, s.hidden[cl]...)
			}
			feat := head.Fuse(base.be, h3)
			base.embedToken(s.tok, emb)
			st := head.NewState()
			draftLogits, _ := head.Step(base.be, emb, feat, len(prompt)-1+i, st)
			q := softmaxStable(draftLogits, 1)
			for j := range q {
				if pi := s.p[head.TargetID(j)]; q[j] < pi {
					alpha += q[j]
				} else {
					alpha += pi
				}
			}
			if head.TargetID(argmax(draftLogits)) == s.next {
				greedy++
			}
		}
		nn := float64(len(steps))
		return alpha / nn, greedy / nn
	}

	for _, cap := range [][]int{
		{2, L / 2, L - 3}, {0, L / 2, L - 1}, {0, 1, 2}, {L - 3, L - 2, L - 1},
		{1, L/2 - 2, L - 2}, {4, L / 2, L - 5}, {L / 4, L / 2, 3 * L / 4},
	} {
		a, gr := score(cap)
		t.Logf("capture %v: α=%.3f greedy=%.3f", cap, a, gr)
	}
}
