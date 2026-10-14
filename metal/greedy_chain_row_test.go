//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// TestGreedyChainRow_bitIdentical is HL-G1 (docs/tasks/task-forward-architecture-2026-10.md §6.1): the greedy chain
// with rows and redos (chainNextRow, chainRedo) against production's full-logits path (ForwardEmbPipe), from the same
// prompt, on the MC3 fixture with tied and untied heads. Most steps take their row, the steps around the pool drain
// all do, and on some of them the host picks the row's second-best token, as a processor's mask would, so the chain
// must redo the forward it queued. Every row must equal the reference's bit for bit; every token and every K/V element
// of every decoded position must match; and after stopChain the executor must carry on from the chain's last position
// exactly as the reference does.
func TestGreedyChainRow_bitIdentical(t *testing.T) {
	for _, tied := range []bool{true, false} {
		t.Run(map[bool]string{true: "tied", false: "untied"}[tied], func(t *testing.T) { greedyChainRowBitIdentical(t, tied) })
	}
}

func greedyChainRowBitIdentical(t *testing.T, tied bool) {
	r := chainFixtureResident(t, 1024, tied)
	const prompt, n = 40, 80
	r.attnFAFloorOverride = prompt + 30 // the attention plan changes mid-chain, as in TestGreedyChain_bitIdentical
	t.Cleanup(func() { r.attnFAFloorOverride = 0 })
	rowStep := func(i int) bool { return i%3 != 0 || (i >= 58 && i < 72) }
	ban := func(i int) bool { return rowStep(i) && i%5 == 1 }
	secondBest := func(row []float32) int {
		top := argmaxF32(row)
		best := -1
		for i, v := range row {
			if i != top && (best < 0 || v > row[best]) {
				best = i
			}
		}
		return best
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
	var ref []int
	refRows := map[int][]float32{}
	id := first
	for i := range n {
		row := r.ForwardEmbPipe(mc3Emb(r, id), prompt+i)
		if rowStep(i) {
			refRows[i] = append([]float32(nil), row...)
		}
		if id = argmaxF32(row); ban(i) {
			id = secondBest(row)
		}
		ref = append(ref, id)
	}
	refKV := kv(prompt + n)
	refAfter := argmaxF32(r.ForwardEmbPipe(mc3Emb(r, id), prompt+n))

	if got := prefill(); got != first {
		t.Fatalf("the prompt's argmax changed between runs: %d then %d", first, got)
	}
	rows0, redos0 := r.chainRows, r.chainRedos
	if err := r.chainStart(first, prompt); err != nil {
		t.Fatalf("chainStart: %v", err)
	}
	var got []int
	rowDiff, redos := 0, 0
	devPick := first
	for i := range n {
		if i > 0 && got[i-1] != devPick {
			if err := r.chainRedo(got[i-1], prompt+i); err != nil {
				t.Fatalf("chainRedo at token %d: %v", i, err)
			}
			redos++
		}
		var err error
		if rowStep(i) {
			var row []float32
			devPick, row, err = r.chainNextRow()
			if err == nil {
				if argmaxF32(row) != devPick {
					t.Fatalf("token %d: the row's argmax %d is not the chain's pick %d", i, argmaxF32(row), devPick)
				}
				for j, v := range row {
					if math.Float32bits(v) != math.Float32bits(refRows[i][j]) {
						rowDiff++
					}
				}
				id = devPick
				if ban(i) {
					id = secondBest(row)
				}
			}
		} else {
			devPick, err = r.chainNext()
			id = devPick
		}
		if err != nil {
			t.Fatalf("token %d: %v", i, err)
		}
		got = append(got, id)
	}
	// The last token's forward ran with the device's pick; production forwards the host's (the after-token below).
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
	fmt.Fprintf(os.Stderr, "[greedy-chain-row] tied %v: %d tokens from position %d, %d with rows, %d redos: %d row elements, %d tokens and %d K/V elements differ; next token after the chain %d vs %d\n",
		tied, n, prompt, r.chainRows-rows0, r.chainRedos-redos0, rowDiff, tok, kd, after, refAfter)
	if redos == 0 || r.chainRedos-redos0 != redos {
		t.Fatalf("%d redos asked, %d counted: the test exercised no redo, or the count is wrong", redos, r.chainRedos-redos0)
	}
	if rowDiff != 0 || tok != 0 || kd != 0 || after != refAfter {
		t.Fatalf("the chain with rows differs from production's path: %d row elements, %d of %d tokens, %d K/V elements; after it %d vs %d",
			rowDiff, tok, n, kd, after, refAfter)
	}
}
