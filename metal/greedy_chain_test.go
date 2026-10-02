//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// chainFixtureResident is the MC3 fixture with a tied LM head (writeMC3FixtureTied), loaded at int4 with one KV slot and
// built as a Metal resident: the greedy chain (C-B01) needs the tie.
func chainFixtureResident(t *testing.T, ctx int) *resident {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(writeMC3FixtureTied(t, max(ctx, 4096), true), decoder.Options{Quant: "int4", ResidentContext: ctx, ResidentKVSlots: 1})
	if err != nil {
		t.Fatalf("load the tied fixture: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build the tied fixture's resident: %v", err)
	}
	t.Cleanup(func() { r.Close(); m.Close(); debug.FreeOSMemory() })
	if why := r.greedyChainWhyNot(); why != "" {
		t.Fatalf("the tied fixture cannot run the greedy chain: %s", why)
	}
	return r
}

// TestGreedyChain_bitIdentical is C-B01's identity gate (docs/audit-metal-2026-09-30.md): greedy decode through the
// chain (chainStart / chainNext: on-device argmax, on-device embedding gather, the next buffer committed before the host
// waits) against production's path (ForwardEmbPipe, host argmax, host embedding), from the same prompt. 80 tokens, so
// the chain passes its pool drain at 64, with attention_fa's floor moved into the run so the attention plan changes
// between two buffers in flight. Every token and every K/V element of every decoded position must match; after
// stopChain the executor must carry on from the chain's last position exactly as the reference does.
func TestGreedyChain_bitIdentical(t *testing.T) {
	r := chainFixtureResident(t, 1024)
	const prompt, n = 40, 80
	r.attnFAFloorOverride = prompt + 30 // the plan changes at 70 keys, mid-chain
	t.Cleanup(func() { r.attnFAFloorOverride = 0 })
	if r.attnPlanFor(prompt+1).fa || !r.attnPlanFor(prompt+n).fa {
		t.Fatalf("attention_fa should be off at %d keys and on at %d: the run would not cross the plan change", prompt+1, prompt+n)
	}
	ids := make([]int, prompt)
	for i := range ids {
		ids[i] = (i*977 + 31) % r.V
	}
	kv := func(n int) [][]uint16 {
		var out [][]uint16
		for l := range r.layers {
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
		}
		return out
	}
	prefill := func() int {
		var lg []float32
		for i, id := range ids {
			lg = r.ForwardEmbPipe(mc3Emb(r, id), i)
		}
		return argmaxF32(lg)
	}

	first := prefill()
	ref := []int{}
	id := first
	for i := range n {
		id = argmaxF32(r.ForwardEmbPipe(mc3Emb(r, id), prompt+i))
		ref = append(ref, id)
	}
	refKV := kv(prompt + n)
	refAfter := argmaxF32(r.ForwardEmbPipe(mc3Emb(r, id), prompt+n))

	if got := prefill(); got != first {
		t.Fatalf("the prompt's argmax changed between runs: %d then %d", first, got)
	}
	if err := r.chainStart(first, prompt); err != nil {
		t.Fatalf("chainStart: %v", err)
	}
	var got []int
	for range n {
		id, err := r.chainNext()
		if err != nil {
			t.Fatalf("chainNext after %d tokens: %v", len(got), err)
		}
		got = append(got, id)
	}
	r.stopChain()
	gotKV := kv(prompt + n)
	after := argmaxF32(r.ForwardEmbPipe(mc3Emb(r, got[n-1]), prompt+n))

	tok, kd := 0, 0
	for i := range ref {
		if got[i] != ref[i] {
			tok++
		}
	}
	for i := range refKV {
		for j := range refKV[i] {
			if refKV[i][j] != gotKV[i][j] {
				kd++
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[greedy-chain] %d tokens from position %d, floor at %d keys: %d tokens and %d K/V elements differ; next token after the chain %d vs %d\n",
		n, prompt, prompt+30, tok, kd, after, refAfter)
	if tok != 0 || kd != 0 || after != refAfter {
		t.Fatalf("the greedy chain differs from production's greedy path: %d of %d tokens, %d K/V elements; after it %d vs %d", tok, n, kd, after, refAfter)
	}
	if math.IsNaN(float64(r.gpuEnd - r.gpuStart)) {
		t.Fatal("no GPU timestamps recorded")
	}
}
