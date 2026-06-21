package decoder

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestEagleSpecThroughput is the wall-clock decision gate for EAGLE speculation: it
// times plain greedy decode vs the linear-chain spec vs the tree spec on the same
// prompt/base, reporting tokens/sec. tok/verify being up means nothing if the extra
// draft+verify work costs more wall-clock than it saves — this measures the real win
// (or loss). Run with -v; uses the q8 base (the realistic serving config).
func TestEagleSpecThroughput(t *testing.T) {
	home, _ := os.UserHomeDir()
	headDir := filepath.Join(home, "models", "qwen3-1.7b-eagle3")
	basePath := filepath.Join(home, "models", "qwen3-1.7b-q8_0.gguf")
	for _, p := range []string{filepath.Join(headDir, "model.safetensors"), basePath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s", p)
		}
	}
	head, err := LoadEagleHead(headDir)
	if err != nil {
		t.Fatalf("LoadEagleHead: %v", err)
	}
	defer head.Close()
	base, err := Load(basePath, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer base.Close()
	tk, _ := tokenizer.LoadGGUF(basePath)
	L := base.w.arch.NumLayers
	capLayers := []int{2, L / 2, L - 3}
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const n = 64
	prompt, _ := tk.Encode("<|im_start|>user\nExplain how a transformer neural network works, with examples.<|im_end|>\n<|im_start|>assistant\n", true)

	timeit := func(run func() int) (float64, int) {
		t0 := time.Now()
		emitted := run()
		return time.Since(t0).Seconds(), emitted
	}

	// baseline greedy
	dBase, nBase := timeit(func() int {
		ch, _ := first(base.Generate(ctx, prompt, n, greedy)), 0
		c := 0
		for range ch {
			c++
		}
		return c
	})

	// linear spec
	var linRounds, linEmit int
	dLin, _ := timeit(func() int {
		out, g, err := base.GenerateEagleSpeculative(ctx, prompt, n, head, capLayers, 6, greedy)
		if err != nil {
			t.Fatalf("linear: %v", err)
		}
		c := 0
		for range out {
			c++
		}
		linRounds, linEmit = g.Spec.Rounds, g.Spec.Emitted
		return c
	})

	// tree spec (B=2, D=4)
	var treeRounds, treeEmit int
	dTree, _ := timeit(func() int {
		out, g, err := base.GenerateEagleSpeculativeTree(ctx, prompt, n, head, capLayers, 2, 4, greedy)
		if err != nil {
			t.Fatalf("tree: %v", err)
		}
		c := 0
		for range out {
			c++
		}
		treeRounds, treeEmit = g.Spec.Rounds, g.Spec.Emitted
		return c
	})

	tps := func(d float64, em int) float64 { return float64(em) / d }
	t.Logf("greedy:      %5.2f tok/s  (%d tok, %.2fs)", tps(dBase, nBase), nBase, dBase)
	t.Logf("linear spec: %5.2f tok/s  (%.2fx greedy, %d tok/%d rounds = %.2f tok/verify)",
		tps(dLin, linEmit), tps(dLin, linEmit)/tps(dBase, nBase), linEmit, linRounds, float64(linEmit)/float64(linRounds))
	t.Logf("tree spec:   %5.2f tok/s  (%.2fx greedy, %d tok/%d rounds = %.2f tok/verify)",
		tps(dTree, treeEmit), tps(dTree, treeEmit)/tps(dBase, nBase), treeEmit, treeRounds, float64(treeEmit)/float64(treeRounds))
}
