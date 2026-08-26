package decoder

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestTreeMaskLinearParity gates the tree-attention primitive: a DEGENERATE tree that
// is actually a linear chain (row i attends to all rows j≤i, RoPE position startPos+i)
// must reproduce the ordinary causal forwardN bit-for-bit. This proves the per-row
// position + ancestor-mask plumbing (cache.treeRowPos/treeMask) is correct before it's
// used for real EAGLE tree drafting; the nil path stays byte-identical by construction.
func TestTreeMaskLinearParity(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	basePath := filepath.Join(home, "models", "qwen3-1.7b-q8_0.gguf")
	if _, err := os.Stat(basePath); err != nil {
		t.Skipf("missing %s", basePath)
	}
	base, err := Load(basePath, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer base.Close()
	tk, _ := tokenizer.LoadGGUF(basePath)
	prompt, _ := tk.Encode("The history of computing began with", true)
	const K = 5
	// the K "draft" tokens — any token ids work for a numerics-parity check
	draft := []int{prompt[1], prompt[2], prompt[0], prompt[3], prompt[1]}

	cache := base.NewCache(len(prompt) + K + 4)
	if _, err := base.forwardN(context.Background(), prompt, cache); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	startPos := len(prompt)

	// A: ordinary causal forwardN over the draft.
	la, err := base.forwardN(context.Background(), draft, cache)
	if err != nil {
		t.Fatalf("forwardN A: %v", err)
	}
	ref := make([][]float32, K)
	for i := range la {
		ref[i] = append([]float32(nil), la[i]...)
	}

	// B: the same draft as a LINEAR tree — row i attends to {0..i}, position startPos+i.
	cache.TruncateTo(startPos)
	pos := make([]int, K)
	mask := make([][]bool, K)
	for i := range K {
		pos[i] = startPos + i
		mask[i] = make([]bool, K)
		for j := 0; j <= i; j++ {
			mask[i][j] = true
		}
	}
	cache.treeRowPos, cache.treeMask = pos, mask
	lb, err := base.forwardN(context.Background(), draft, cache)
	cache.treeRowPos, cache.treeMask = nil, nil
	if err != nil {
		t.Fatalf("forwardN B: %v", err)
	}

	for i := range K {
		for j := range ref[i] {
			if ref[i][j] != lb[i][j] {
				t.Fatalf("row %d col %d: causal %g != linear-tree %g", i, j, ref[i][j], lb[i][j])
			}
		}
	}
	t.Logf("tree-mask linear parity OK: %d rows bit-identical to causal forwardN", K)
}
