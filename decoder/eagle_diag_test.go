package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestEagleAgreementSweep is an 05 diagnostic (not a gate): it greedily decodes the
// base, and at each step drafts the next token with the head from the captured target
// hidden + the current token's embedding, counting how often the head's top draft ==
// the base's own next-token argmax. That agreement IS the head's job, so a config
// with high agreement confirms the forward is wired right (and finds the fused-layer
// indices); ~0 across all configs means a structural bug (concat order / d2t / capture
// point), not just layer tuning. Run -v.
func TestEagleAgreementSweep(t *testing.T) {
	requireHeavyModel(t)
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
	hid := base.w.arch.HiddenDim

	prompt, _ := tk.Encode("Once upon a time, in a small village, there lived a clever fox who", true)
	const steps = 32

	configs := [][]int{
		{2, 14, 25}, {2, 14, 26}, {3, 14, 25}, {2, 15, 25},
		{1, 14, 26}, {0, 14, 25}, {2, 9, 25}, {2, 18, 25},
	}
	for _, cap := range configs {
		// fresh greedy decode, drafting at each step.
		cache := base.NewCache(len(prompt) + steps + 4)
		var logits []float32
		var hs [][]float32
		// prefill all but last
		for i := 0; i < len(prompt)-1; i++ {
			base.forward(prompt[i], cache)
		}
		cur := prompt[len(prompt)-1]
		agree, total := 0, 0
		for s := range steps {
			logits, hs, err = base.ForwardCapture(cur, cache, cap)
			if err != nil {
				t.Fatalf("ForwardCapture: %v", err)
			}
			baseNext := argmax(logits)
			// head draft from (embed(cur), fuse(captured))
			h3 := make([]float32, 0, 3*hid)
			for _, h := range hs {
				h3 = append(h3, h...)
			}
			feat := head.Fuse(base.be, h3)
			emb := make([]float32, hid)
			base.embedToken(cur, emb)
			st := head.NewState()
			draft, _ := head.Step(base.be, emb, feat, len(prompt)+s-1, st)
			if head.TargetID(argmax(draft)) == baseNext {
				agree++
			}
			total++
			cur = baseNext // follow the base's greedy path
		}
		t.Logf("capture %v: head/base top-1 agreement %d/%d = %.0f%%", cap, agree, total, 100*float64(agree)/float64(total))
	}
}
