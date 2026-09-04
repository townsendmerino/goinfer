package decoder

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestEagleSpecParity is the 05 inc-5 gate: end-to-end EAGLE speculative decode must
// be token-identical to plain greedy on the same base (lossless) — the head only
// proposes, the target's argmax decides. Also reports the realized tokens/verify.
func TestEagleSpecParity(t *testing.T) {
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
	base := sharedEagleBase(t, basePath, "int8int8")
	tk, _ := tokenizer.LoadGGUF(basePath)
	L := base.w.arch.NumLayers
	capLayers := []int{2, L / 2, L - 3}
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const n = 24

	prompt, _ := tk.Encode("<|im_start|>user\nWrite one sentence about the history of computing.<|im_end|>\n<|im_start|>assistant\n", true)

	ref := collectTokens(first(base.Generate(ctx, prompt, n, greedy)))

	// V-11 (docs/review-2026-09-04.md): GenerateEagleSpeculative never touches base.resident
	// (it uses embedToken + a local NewCache, confirmed when R-03 investigated this file), so
	// a pre-existing resIDs describing an unrelated resident generation must survive this call
	// untouched — it used to get unconditionally forgotten.
	base.resIDs = []int{9, 9, 9}
	ch, g, err := base.GenerateEagleSpeculative(ctx, prompt, n, head, capLayers, 5, greedy)
	if err != nil {
		t.Fatalf("GenerateEagleSpeculative: %v", err)
	}
	got := collectTokens(ch)
	if g.Err() != nil {
		t.Fatalf("spec stream err: %v", g.Err())
	}
	if !slices.Equal(got, ref) {
		t.Fatalf("EAGLE spec != greedy (lossless broken)\n got %v\n ref %v", got, ref)
	}
	t.Logf("EAGLE spec parity OK: %d tok, %.3f acceptance, %.2f tok/verify", len(got), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound())
	if !equalIntSlices(base.resIDs, []int{9, 9, 9}) {
		t.Errorf("resIDs = %v after GenerateEagleSpeculative, want unchanged [9 9 9] (V-11)", base.resIDs)
	}

	// Tree variant: must also be lossless, and should commit more tokens/verify than the
	// linear chain (root branching recovers top-2..B first tokens the chain misses).
	base.resIDs = []int{9, 9, 9} // V-11: re-set; the call above already proved unchanged
	tch, tg, err := base.GenerateEagleSpeculativeTree(ctx, prompt, n, head, capLayers, 2, 4, greedy)
	if err != nil {
		t.Fatalf("GenerateEagleSpeculativeTree: %v", err)
	}
	tgot := collectTokens(tch)
	if tg.Err() != nil {
		t.Fatalf("tree spec stream err: %v", tg.Err())
	}
	if !slices.Equal(tgot, ref) {
		t.Fatalf("EAGLE tree spec != greedy (lossless broken)\n got %v\n ref %v", tgot, ref)
	}
	t.Logf("EAGLE TREE spec parity OK: %d tok, %.2f tok/verify (B=2,D=4) — vs linear %.2f", len(tgot), tg.Spec.TokensPerRound(), g.Spec.TokensPerRound())
	if !equalIntSlices(base.resIDs, []int{9, 9, 9}) {
		t.Errorf("resIDs = %v after GenerateEagleSpeculativeTree, want unchanged [9 9 9] (V-11)", base.resIDs)
	}
}
