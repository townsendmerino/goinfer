package decoder

import (
	"os"
	"testing"
)

// The decoder-as-embedder seam (docs/completed/task-decoder-as-embedder.md). These gates are MECHANICAL —
// shape, determinism, pooling position, guards — and run against a base Qwen3 decoder, which shares
// the architecture (and here the exact 1024×28 geometry) of Qwen3-Embedding-0.6B. The SEMANTIC gates
// (cosine vs the HF sentence-transformers reference, retrieval top-k) need the embedding-tuned
// checkpoint plus a pinned golden and live in cmd/serve; see scripts/pin_qwen3_embedding.py.

func loadQwen3ForEmbed(t *testing.T) *Model {
	t.Helper()
	p := os.ExpandEnv("$HOME/models/Qwen3-0.6B-Q8_0.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no checkpoint at %s", p)
	}
	m, err := Load(p, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return m
}

// TestHiddenLast_shapeDeterminismAndPooling pins the three properties the pooling head depends on:
// the vector is HiddenDim wide, the same ids give a bit-identical vector (an embedder that drifts
// run-to-run is useless for a vector index), and it is genuinely a LAST-token pool — changing the
// final token, or extending the sequence so the pooled position moves, must change the vector.
func TestHiddenLast_shapeDeterminismAndPooling(t *testing.T) {
	requireHeavyModel(t)
	m := loadQwen3ForEmbed(t)
	hidden := m.Config().HiddenDim

	base := []int{9707, 1879, 374, 264, 1273} // arbitrary in-vocab ids
	v1, err := m.HiddenLast(base)
	if err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	if len(v1) != hidden {
		t.Fatalf("len = %d, want HiddenDim %d", len(v1), hidden)
	}
	if allZero(v1) {
		t.Fatal("hidden state is all zeros — the stack did not run")
	}

	// Determinism: identical input → bit-identical output.
	v2, err := m.HiddenLast(base)
	if err != nil {
		t.Fatalf("HiddenLast (repeat): %v", err)
	}
	for i := range v1 {
		if v1[i] != v2[i] {
			t.Fatalf("non-deterministic at %d: %v vs %v (a fresh KV cache must give identical results)", i, v1[i], v2[i])
		}
	}

	// Last-token pooling: swapping only the FINAL token must move the vector.
	swapped := append(append([]int(nil), base[:len(base)-1]...), 8112) // different FINAL token
	vSwap, err := m.HiddenLast(swapped)
	if err != nil {
		t.Fatalf("HiddenLast (swapped last): %v", err)
	}
	if cosine(v1, vSwap) > 0.9999 {
		t.Errorf("changing the LAST token left the vector unchanged (cos %.6f) — this is not last-token pooling",
			cosine(v1, vSwap))
	}

	// Extending the sequence moves the pooled position, so the vector must change too. (A mean pool
	// over a 1-token extension could stay nearly identical; a last-token pool cannot.)
	longer := append(append([]int(nil), base...), 30)
	vLong, err := m.HiddenLast(longer)
	if err != nil {
		t.Fatalf("HiddenLast (extended): %v", err)
	}
	if cosine(v1, vLong) > 0.9999 {
		t.Errorf("extending the sequence left the vector unchanged (cos %.6f) — the pooled position did not move",
			cosine(v1, vLong))
	}

	// Causality sanity: the pooled vector depends on the preceding context, not just the last token.
	ctxDiff := []int{40, 1079, 264, 2155, 1273}
	ctxDiff[len(ctxDiff)-1] = base[len(base)-1] // same final token, different context
	vCtx, err := m.HiddenLast(ctxDiff)
	if err != nil {
		t.Fatalf("HiddenLast (ctx): %v", err)
	}
	if cosine(v1, vCtx) > 0.9999 {
		t.Errorf("different context with the same final token gave the same vector (cos %.6f) — context is being ignored",
			cosine(v1, vCtx))
	}
}

// TestHiddenLast_guards: the seam refuses inputs it cannot pool rather than returning a
// plausible-looking vector — the silent-wrong class this whole task is fenced against.
func TestHiddenLast_guards(t *testing.T) {
	requireHeavyModel(t)
	m := loadQwen3ForEmbed(t)
	vocab := m.Config().VocabSize

	if _, err := m.HiddenLast(nil); err == nil {
		t.Error("empty token sequence must error (there is no position to pool)")
	}
	if _, err := m.HiddenLast([]int{}); err == nil {
		t.Error("empty token slice must error")
	}
	if _, err := m.HiddenLast([]int{vocab}); err == nil {
		t.Errorf("out-of-vocab id %d must error, not index past the embedding table", vocab)
	}
	if _, err := m.HiddenLast([]int{-1}); err == nil {
		t.Error("negative id must error")
	}
	// A valid id adjacent to the bound still works (the guard is not off-by-one).
	if _, err := m.HiddenLast([]int{vocab - 1}); err != nil {
		t.Errorf("last valid id %d must be accepted, got %v", vocab-1, err)
	}
}

// TestHiddenLast_batchedMatchesSequential is the P-17 gate: hiddenLastBatched
// (one runLayersFromEmbedN call over the whole sequence) must agree with
// hiddenLastSequential (the original one-runLayers-call-per-token loop) on a
// real checkpoint — same model, same ids, only the batching differs. Checked at
// several lengths, including one long enough that canBatchN would clearly
// select the batched path in HiddenLast itself.
func TestHiddenLast_batchedMatchesSequential(t *testing.T) {
	requireHeavyModel(t)
	m := loadQwen3ForEmbed(t)

	base := []int{9707, 1879, 374, 264, 1273, 40, 1079, 264, 2155, 1273, 30, 8112}
	for _, n := range []int{2, 3, 5, len(base)} {
		ids := base[:n]
		if !m.canBatchN(len(ids)) {
			t.Fatalf("n=%d: canBatchN false — test assumes the batched path applies here", n)
		}
		seq, err := m.hiddenLastSequential(ids)
		if err != nil {
			t.Fatalf("n=%d: hiddenLastSequential: %v", n, err)
		}
		batched, err := m.hiddenLastBatched(ids)
		if err != nil {
			t.Fatalf("n=%d: hiddenLastBatched: %v", n, err)
		}
		if len(seq) != len(batched) {
			t.Fatalf("n=%d: len(sequential)=%d, len(batched)=%d", n, len(seq), len(batched))
		}
		if c := cosine(seq, batched); c < 0.999999 {
			t.Errorf("n=%d: batched vs sequential cosine = %.8f, want ~1.0", n, c)
		}
		var maxDiff float32
		for i := range seq {
			if d := seq[i] - batched[i]; d > maxDiff || -d > maxDiff {
				maxDiff = max(d, -d)
			}
		}
		if maxDiff > 1e-3 {
			t.Errorf("n=%d: max|sequential-batched| = %v, want ~0", n, maxDiff)
		}
	}
}

func allZero(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}
