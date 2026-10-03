//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPromptStepAboveFloor_reuseVsCold measures what E-P01's above-floor half (promptStepAboveFloor) costs in
// cache-independence: a reuse turn — a prefix prefilled earlier, then a short suffix — against a cold prefill of the
// same whole prompt. With the half off, the suffix runs on the batched pass, which is chunk-invariant, so reuse equals
// cold bit for bit (the control arm, asserted). With it on, the suffix's K/V come from decode's int8 activations
// instead of the pass's f16 ones. Per prompt and suffix length it reports the seed row's KL(cold ‖ reuse), whether its
// argmax agrees, and where 32 greedy tokens decoded from each first diverge. It decides nothing by itself; the owner
// decides with these numbers (docs/tasks/task-metal-audit-2026-10.md).
//
//	GOINFER_METAL_MC3=1 GOINFER_PREFILL_GATE_PROMPTS=b go test -tags goinfer_testhooks -count=1 -run '^TestPromptStepAboveFloor_reuseVsCold$' -v ./metal/
func TestPromptStepAboveFloor_reuseVsCold(t *testing.T) {
	path, tokPath := mc3Checkpoint(t)
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	tk, err := tokenizer.LoadGGUF(tokPath)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	const prefix, greedyN = 512, 32
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: 1024, ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || a.VerifyCost() == nil {
		t.Fatalf("no metal resident with step kernels")
	}
	r := a.r
	defer func() { promptStepAboveFloor = false }()
	label, files := decoder.PrefillGatePromptSet()
	ctx := context.Background()
	emb := func(id int) []float32 { return m.EmbedResidentForTest(id) }
	// run prefills ids on slot (whole, or prefix then suffix), then decodes greedyN greedy tokens; it returns the seed
	// logits and the tokens.
	run := func(slot int, ids []int, split int) ([]float32, []int) {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		embs := make([][]float32, len(ids))
		for i, id := range ids {
			embs[i] = emb(id)
		}
		var seed []float32
		var err error
		if split == 0 {
			seed, err = a.PrefillLast(ctx, embs, 0)
		} else {
			if _, err = a.PrefillLast(ctx, embs[:split], 0); err == nil {
				seed, err = a.PrefillLast(ctx, embs[split:], split)
			}
		}
		if err != nil {
			t.Fatalf("prefill: %v", err)
		}
		seed = append([]float32(nil), seed...)
		toks, lg := []int{}, seed
		for i := range greedyN {
			id := argmaxF32(lg)
			toks = append(toks, id)
			if lg, err = a.Forward(emb(id), len(ids)+i); err != nil {
				t.Fatalf("decode: %v", err)
			}
		}
		return seed, toks
	}
	firstDiff := func(x, y []int) int {
		for i := range x {
			if x[i] != y[i] {
				return i
			}
		}
		return -1
	}
	fmt.Fprintf(os.Stderr, "[step-reuse] prompt set %q, %d-token prefix, then the suffix; %d greedy tokens\n", label, prefix, greedyN)
	for _, s := range []int{8, 16, 32} {
		agree, diverged, maxKL, sumKL := 0, 0, 0.0, 0.0
		for pi, f := range files {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, f, prefix+s)[:prefix+s]
			coldSeed, coldToks := run(0, ids, 0)
			promptStepAboveFloor = false
			ctlSeed, ctlToks := run(1, ids, prefix)
			for j := range coldSeed {
				if coldSeed[j] != ctlSeed[j] {
					t.Fatalf("control: prompt %d, suffix %d: reuse with the half off differs from cold at logit %d — the probe's premise fails", pi+1, s, j)
				}
			}
			if firstDiff(coldToks, ctlToks) >= 0 {
				t.Fatalf("control: prompt %d, suffix %d: greedy tokens differ with the half off", pi+1, s)
			}
			promptStepAboveFloor = true
			if !a.promptStepOK(s, prefix, 64) {
				t.Fatalf("the half is on but promptStepOK(%d, %d) is false", s, prefix)
			}
			reuseSeed, reuseToks := run(1, ids, prefix)
			promptStepAboveFloor = false
			kl := decoder.KLDivergenceForTest(coldSeed, reuseSeed)
			sumKL += kl
			maxKL = max(maxKL, kl)
			am := argmaxF32(coldSeed) == argmaxF32(reuseSeed)
			if am {
				agree++
			}
			d := firstDiff(coldToks, reuseToks)
			if d >= 0 {
				diverged++
			}
			fmt.Fprintf(os.Stderr, "[step-reuse] suffix %2d prompt %2d: seed KL(cold || reuse) %.5f, argmax agrees %v, greedy first differs at %d (-1 = never)\n",
				s, pi+1, kl, am, d)
		}
		fmt.Fprintf(os.Stderr, "[step-reuse] RESULT suffix %d: seed argmax agrees %d of %d, mean KL %.5f, max %.5f; greedy %d tokens diverge on %d of %d prompts\n",
			s, agree, len(files), sumKL/float64(len(files)), maxKL, greedyN, diverged, len(files))
	}
}
