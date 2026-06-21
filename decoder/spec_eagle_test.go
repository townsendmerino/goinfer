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
		t.Fatalf("Load: %v", err)
	}
	defer base.Close()
	tk, _ := tokenizer.LoadGGUF(basePath)
	L := base.w.arch.NumLayers
	capLayers := []int{2, L / 2, L - 3}
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const n = 24

	prompt, _ := tk.Encode("The history of computing began with mechanical calculators, and", true)

	ref := collectTokens(first(base.Generate(ctx, prompt, n, greedy)))
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
}
