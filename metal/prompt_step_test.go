//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestPrefillLast_stepRouteBitIdentical is E-P01's gate (docs/audit-metal-2026-09-30.md, Phase 3 item 1 of
// docs/tasks/task-metal-audit-2026-10.md), run through PrefillLast, the entry point the decoder calls, rather than the
// step it routes to. On a resident with the batched step (the default-run fixture; the 1.5B under GOINFER_METAL_MC3=1),
// a prompt ending below the step bound (the larger of the fast-prefill floor and metalStepPrefillCeiling, A-P02) runs as decode rows on the step kernels. Each case prefills on slot 1 and the sequential loop runs on slot 0 from the same
// embeddings; every K/V element of every position and every logit of the last row must match bit for bit. The batched
// f16-MMA pass would not: it is not bit-identical to decode (a mutation forcing it on prompts ending at 72, 81 and 132 keys
// differed in every logit), so a match also shows the route was taken. Prompts that reach the bound take the pass,
// whatever their length: a short suffix there is also a chunked prefill's tail (promptStepOK).
func TestPrefillLast_stepRouteBitIdentical(t *testing.T) {
	a := mc3PrefillResident(t, 2, 1024)
	r := a.r
	if a.VerifyCost() == nil {
		t.Fatalf("the resident has no step kernels (batchIneligible: %q)", r.batchIneligible())
	}
	floor := metalFastPrefillFloorFor(r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if floor != 16 || metalStepPrefillCeiling != 32 {
		t.Fatalf("fast-prefill floor %d, step bound %d; the cases below are laid out for 16 and 32 (A-P02)", floor, metalStepPrefillCeiling)
	}
	seed := uint32(24680)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	kv := func(slot, n int) [][]uint16 { // per layer: K rows then V rows, positions 0..n-1
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
		}
		return out
	}
	cases := []struct {
		start, n int
		step     bool
	}{
		{0, 8, true}, {0, 9, true}, {0, 15, true}, {0, 20, true}, {0, 31, true}, {10, 21, true}, // ends below the step bound
		{0, 32, false}, {0, 33, false}, {0, 63, false}, {24, 8, false}, {56, 8, false}, {100, 32, false}, // reaches it: the pass
	}
	for _, c := range cases {
		if got := a.promptStepOK(c.n, c.start, floor); got != c.step {
			t.Errorf("start %d, %d tokens: promptStepOK = %v, want %v", c.start, c.n, got, c.step)
		}
		if !c.step {
			continue
		}
		total := c.start + c.n
		embs := make([][]float32, total)
		for i := range embs {
			embs[i] = mc3Emb(r, rnd())
		}
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		var ref []float32
		for i, e := range embs {
			ref = append(ref[:0], r.ForwardEmb(e, i)...)
		}
		refKV := kv(0, total)
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		for i := range c.start {
			r.ForwardEmb(embs[i], i)
		}
		got, err := a.PrefillLast(context.Background(), embs[c.start:], c.start)
		if err != nil {
			t.Fatalf("start %d, %d tokens: PrefillLast: %v", c.start, c.n, err)
		}
		gotKV := kv(1, total)
		lg, kd := 0, 0
		for j := range ref {
			if math.Float32bits(got[j]) != math.Float32bits(ref[j]) {
				lg++
			}
		}
		for i := range refKV {
			for j := range refKV[i] {
				if refKV[i][j] != gotKV[i][j] {
					kd++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[prompt-step] start %d, %d tokens: %d of %d last-row logits and %d K/V elements differ from the sequential loop\n",
			c.start, c.n, lg, len(ref), kd)
		if len(got) != len(ref) || lg != 0 || kd != 0 {
			t.Errorf("start %d, %d tokens: %d logits and %d K/V elements differ from the sequential loop", c.start, c.n, lg, kd)
		}
	}
	r.promptStepOff = true
	if a.promptStepOK(8, 0, floor) {
		t.Error("promptStepOff did not keep PrefillLast off the step route")
	}
	r.promptStepOff = false
}

// TestPrefillLast_stepRouteCancels: the step route checks the context between pieces, so a cancelled request stops
// rather than running the whole prompt (the batched pass can only check at entry; A-D02).
func TestPrefillLast_stepRouteCancels(t *testing.T) {
	a := mc3PrefillResident(t, 2, 1024)
	if a.VerifyCost() == nil {
		t.Fatal("the resident has no step kernels")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	embs := make([][]float32, 20)
	for i := range embs {
		embs[i] = mc3Emb(a.r, i+1)
	}
	if _, err := a.PrefillLast(ctx, embs, 0); err == nil {
		t.Fatal("PrefillLast on a cancelled context returned no error")
	}
}
