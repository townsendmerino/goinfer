//go:build realckpt

// CPU block-drafting gate, on the ALREADY-VALIDATED Qwen3-4B + DFlash pairing.
//
// WHY THIS PAIRING AND NOT LAGUNA. The property under test — that speculative output is
// TOKEN-IDENTICAL to plain greedy decoding — is model-independent, and this target loads
// in seconds where Laguna-XS.2 takes ~5 minutes at int4. Debugging a new orchestration
// path against a 33B model would mean a 5-minute penalty on every iteration, so the
// mechanics are proven here and the expensive model is used only to measure acceptance.
//
// WHAT LOSSLESSNESS ACTUALLY PROVES HERE. Block drafting is lossless BY CONSTRUCTION —
// every emitted token is one the target's own argmax produced — so it cannot catch a bad
// drafter. What it CAN catch is the thing this file introduces: wrong positions or a
// missed cache rollback. If PrefillLastNArgmax verified without truncating to startPos,
// the rejected tail of the previous block would stay in the KV cache and later positions
// would attend to tokens the model never emitted. That does not crash and does not break
// the losslessness invariant on its own terms; it just silently conditions on garbage,
// and the ONLY way it shows is the output diverging from plain greedy.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestCPUBlockSpec -v -timeout 60m
package decoder

import (
	"context"
	"testing"
	"time"
)

func TestCPUBlockSpec_lossless(t *testing.T) {
	requireHeavyModel(t)
	target := assetPath(t, "GOINFER_QWEN3_4B")
	ddir := assetPath(t, "GOINFER_DFLASH_F32")

	m, err := Load(target, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load target: %v", err)
	}
	defer m.Close()
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()

	prompt := []int{9707, 11, 358, 1079, 264, 8720, 315}
	const maxNew = 40

	// Plain greedy, the reference.
	base := make([]int, 0, maxNew)
	tPlain := time.Now()
	{
		out, _ := m.Generate(context.Background(), prompt, maxNew, SamplingParams{})
		for id := range out {
			base = append(base, id)
		}
	}
	plainDur := time.Since(tPlain)
	if len(base) == 0 {
		t.Fatal("plain generation produced nothing")
	}

	spec, err := m.NewCPUBlockSpec(d, len(prompt)+maxNew+d.BlockSize()+2)
	if err != nil {
		t.Fatalf("NewCPUBlockSpec: %v", err)
	}
	tSpec := time.Now()
	got, rounds, err := spec.Generate(prompt, BlockSpecOptions{MaxTokens: maxNew})
	if err != nil {
		t.Fatalf("BlockSpec.Generate: %v", err)
	}
	specDur := time.Since(tSpec)
	perRound := 0.0
	if rounds > 0 {
		perRound = float64(len(got)) / float64(rounds)
	}
	// THE COUNTERFACTUAL TO LAGUNA. Block drafting measured 0.82x on Laguna-XS.2, a SPARSE
	// MoE: each verified row routes to its own top-8 of 256 experts, so an 8-row batched
	// verify touches ~8x the expert weight instead of amortizing. Qwen3-4B is DENSE, where
	// every row reads the same weights — the case where batched verify should actually pay.
	// Reporting the ratio per token, since the two runs may emit different counts.
	speedup := float64(plainDur) / float64(specDur) * float64(len(base)) / float64(max(len(got), 1))
	t.Logf("cpu block spec (DENSE target): %d tokens in %d rounds (%.2f tok/round); plain produced %d",
		len(got), rounds, perRound, len(base))
	t.Logf("  wall clock: spec %v vs plain %v -> ~%.2fx",
		specDur.Round(time.Millisecond), plainDur.Round(time.Millisecond), speedup)

	n := min(len(base), len(got))
	if n == 0 {
		t.Fatal("speculative generation produced nothing")
	}
	for i := range n {
		if got[i] != base[i] {
			t.Fatalf("DIVERGED at token %d: spec=%d plain=%d\n  spec=%v\n  plain=%v\n"+
				"Block drafting is lossless by construction, so a divergence is not a bad draft — "+
				"it is wrong positions or a missed cache rollback in the new CPU verify path.",
				i, got[i], base[i], got[:min(len(got), i+3)], base[:min(len(base), i+3)])
		}
	}
	// Acceptance above 1.0 means drafts were actually taken. At exactly 1.0 the path is
	// lossless but useless — every draft rejected — which no correctness check can see.
	if rounds > 0 && perRound <= 1.0 {
		t.Errorf("%.2f tok/round — no draft accepted on a pairing P10 measured well above "+
			"break-even, so this is the new CPU path drafting wrongly, not a weak drafter", perRound)
	}
}
