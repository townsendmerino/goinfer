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

// chainFixtureResident is the MC3 fixture with a tied or an untied LM head (writeMC3FixtureTied), loaded at int4 with one
// KV slot and built as a Metal resident. The greedy chain (C-B01) gathers from the LM-head table when the head is tied
// and from a device copy of the embedding table when it is not.
func chainFixtureResident(t *testing.T, ctx int, tied bool) *resident {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(writeMC3FixtureTied(t, max(ctx, 4096), tied), decoder.Options{Quant: "int4", ResidentContext: ctx, ResidentKVSlots: 1})
	if err != nil {
		t.Fatalf("load the fixture: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build the fixture's resident: %v", err)
	}
	t.Cleanup(func() { r.Close(); m.Close(); debug.FreeOSMemory() })
	if why := r.greedyChainWhyNot(); why != "" {
		t.Fatalf("the fixture (tied %v) cannot run the greedy chain: %s", tied, why)
	}
	if r.lmTied != tied {
		t.Fatalf("the fixture's head: lmTied %v, want %v", r.lmTied, tied)
	}
	return r
}

// TestGreedyChain_bitIdentical is C-B01's identity gate (docs/audit-metal-2026-09-30.md): greedy decode through the
// chain (chainStart / chainNext: on-device argmax, on-device embedding gather, the next buffer committed before the host
// waits) against production's path (ForwardEmbPipe, host argmax, host embedding), from the same prompt. 80 tokens, so
// the chain passes its pool drain at 64, with attention_fa's floor moved into the run so the attention plan changes
// between two buffers in flight. Every token and every K/V element of every decoded position must match; after
// stopChain the executor must carry on from the chain's last position exactly as the reference does. Tied and untied
// heads both: the untied one gathers from the device embedding table chainEmbedTable makes.
func TestGreedyChain_bitIdentical(t *testing.T) {
	for _, tied := range []bool{true, false} {
		t.Run(map[bool]string{true: "tied", false: "untied"}[tied], func(t *testing.T) { greedyChainBitIdentical(t, tied) })
	}
}

func greedyChainBitIdentical(t *testing.T, tied bool) {
	r := chainFixtureResident(t, 1024, tied)
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
	fmt.Fprintf(os.Stderr, "[greedy-chain] tied %v: %d tokens from position %d, floor at %d keys: %d tokens and %d K/V elements differ; next token after the chain %d vs %d\n",
		tied, n, prompt, prompt+30, tok, kd, after, refAfter)
	if !tied && r.chainEmbW == (Buffer{}) {
		t.Fatal("the untied chain ran without its device embedding table")
	}
	if tok != 0 || kd != 0 || after != refAfter {
		t.Fatalf("the greedy chain differs from production's greedy path: %d of %d tokens, %d K/V elements; after it %d vs %d", tok, n, kd, after, refAfter)
	}
	if math.IsNaN(float64(r.gpuEnd - r.gpuStart)) {
		t.Fatal("no GPU timestamps recorded")
	}
}

// TestHeadID_mapsNoIDToZero is G-MA2 (docs/tasks/task-subsystems-2026-10.md §8, "The device argmax on Metal: NaN rows"): the
// fused head's id passes through when it is a token, and an id at or past V, which the head writes only for a row that is NaN
// throughout, becomes argmaxF32's 0 for that row.
func TestHeadID_mapsNoIDToZero(t *testing.T) {
	for _, c := range []struct {
		id      uint32
		v, want int
	}{{5, 100, 5}, {99, 100, 99}, {0, 100, 0}, {100, 100, 0}, {0xFFFFFFFF, 100, 0}} {
		if got := headID(c.id, c.v); got != c.want {
			t.Errorf("headID(%#x, %d) = %d, want %d", c.id, c.v, got, c.want)
		}
	}
	nan := float32(math.NaN())
	if got := argmaxF32([]float32{nan, nan, nan}); got != 0 {
		t.Fatalf("argmaxF32 of an all-NaN row = %d; headID's 0 is meant to match it", got)
	}
}

// TestGreedyChain_nanHeadRecovers is G-MA3 (§8, "The device argmax on Metal: NaN rows"): with chainNaNAtForTest at one
// position, that buffer's head sees only NaN logits and writes 0xFFFFFFFF. The chain must yield 0 there, never -1, and carry on
// exactly as production's greedy path does with that token forced to 0: every token and every K/V element of every decoded
// position equal.
func TestGreedyChain_nanHeadRecovers(t *testing.T) {
	r := chainFixtureResident(t, 1024, true)
	const prompt, n, p = 40, 24, 9
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
	var ref []int
	id := first
	for i := range n {
		id = argmaxF32(r.ForwardEmbPipe(mc3Emb(r, id), prompt+i))
		if i == p {
			id = 0 // what the chain's head gives on its all-NaN row
		}
		ref = append(ref, id)
	}
	refKV := kv(prompt + n)

	prefill()
	chainNaNAtForTest = prompt + p
	t.Cleanup(func() { chainNaNAtForTest = -1 })
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
	chainNaNAtForTest = -1
	gotKV := kv(prompt + n)
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
	fmt.Fprintf(os.Stderr, "[greedy-chain NaN head] position %d: chain %d, reference %d; %d tokens and %d K/V elements differ\n",
		prompt+p, got[p], ref[p], tok, kd)
	if got[p] != 0 || tok != 0 || kd != 0 {
		t.Fatalf("the chain did not recover from its NaN head: token at %d is %d (want 0); %d of %d tokens and %d K/V elements differ",
			prompt+p, got[p], tok, n, kd)
	}
}

// TestSampleChain_bitIdentical is C-P02's identity gate at the resident: temperature-only sampling through the sampled
// chain (chainStartSampled / chainNextSampled: the device draw, the next buffer committed before the host waits)
// against production's device-sampled path (ForwardSample with host embedding, the draws Sampler.NextDraw would hand
// it), from the same prompt, as TestGreedyChain_bitIdentical does for the argmax. Then the -1 recovery: a draw with
// nothing comparable (forced by chainFallbackAtForTest) must give ForwardSample's own fallback, the argmax of that
// row, with every later token and K/V element unchanged; and a draw that is not the chain's must be refused.
func TestSampleChain_bitIdentical(t *testing.T) {
	for _, tied := range []bool{true, false} {
		t.Run(map[bool]string{true: "tied", false: "untied"}[tied], func(t *testing.T) { sampleChainBitIdentical(t, tied, -1) })
	}
	t.Run("fallback", func(t *testing.T) { sampleChainBitIdentical(t, false, 40+17) })
}

func sampleChainBitIdentical(t *testing.T, tied bool, fallbackAt int) {
	r := chainFixtureResident(t, 1024, tied)
	const prompt, n, temp = 40, 80, 0.9
	const seed, draw0 = uint64(0x9e3779b97f4a7c15), uint64(5)
	r.attnFAFloorOverride = prompt + 30
	t.Cleanup(func() { r.attnFAFloorOverride = 0 })
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
	var ref []int
	id := first
	for i := range n {
		pos := prompt + i
		var err error
		if pos == fallbackAt { // ForwardSample's fallback for a row with nothing comparable: the argmax of the row
			id = argmaxF32(r.ForwardEmbPipe(mc3Emb(r, id), pos))
		} else if id, err = r.ForwardSample(mc3Emb(r, id), pos, temp, seed, draw0+uint64(i)); err != nil {
			t.Fatalf("ForwardSample at %d: %v", pos, err)
		}
		ref = append(ref, id)
	}
	refKV := kv(prompt + n)

	if got := prefill(); got != first {
		t.Fatalf("the prompt's argmax changed between runs: %d then %d", first, got)
	}
	chainFallbackAtForTest = fallbackAt
	t.Cleanup(func() { chainFallbackAtForTest = -1 })
	if err := r.chainStartSampled(first, prompt, temp, seed, draw0); err != nil {
		t.Fatalf("chainStartSampled: %v", err)
	}
	if _, err := r.chainNextSampled(seed, draw0+1); err == nil {
		t.Fatal("chainNextSampled accepted a draw that is not the chain's")
	}
	var got []int
	for i := range n {
		id, err := r.chainNextSampled(seed, draw0+uint64(i))
		if err != nil {
			t.Fatalf("chainNextSampled after %d tokens: %v", len(got), err)
		}
		got = append(got, id)
	}
	r.stopChain()
	gotKV := kv(prompt + n)
	tok, kd, distinct := 0, 0, map[int]bool{}
	for i := range ref {
		if got[i] != ref[i] {
			tok++
		}
		distinct[ref[i]] = true
	}
	for i := range refKV {
		for j := range refKV[i] {
			if refKV[i][j] != gotKV[i][j] {
				kd++
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[sample-chain] tied %v, fallback at %d: %d tokens at T=%.1f (%d distinct): %d tokens and %d K/V elements differ\n",
		tied, fallbackAt, n, temp, len(distinct), tok, kd)
	if tok != 0 || kd != 0 {
		t.Fatalf("the sampled chain differs from ForwardSample's path: %d of %d tokens, %d K/V elements", tok, n, kd)
	}
	if len(distinct) < n/4 {
		t.Fatalf("only %d distinct tokens in %d: the draw is not sampling, so the comparison proves little", len(distinct), n)
	}
}
