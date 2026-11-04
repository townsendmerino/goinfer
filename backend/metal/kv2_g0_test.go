//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"
)

// TestKV2_G0splitInvariance is G0 on Metal (docs/measurements/kv2-2026-10-14.md), KV0's arms: a 1,000-token prompt
// prefilled in two calls split at 512, 768 and 984, and in one call at chunk widths 64 and 448 (successive calls), each
// against the prompt in one call: every K and V element and the last logits, bit for bit. KV0's control (a token changed
// at position 300 shows from 300 on, nowhere before) runs first. The decode arm (the prompt decoded token by token) is
// measured and reported: it decides whether decode rows are the batched class, which the class map must state.
func TestKV2_G0splitInvariance(t *testing.T) {
	a := mc3PrefillResident(t, 2, 2048)
	r := a.r
	const N = 1000
	embs := kv2Embs(r, N, 4242)
	rb := kv2RowBytes(r)
	fill := func(slot int, cuts []int, e [][]float32) ([]float32, [][]byte) {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var lg []float32
		var err error
		prev := 0
		for _, c := range append(slices.Clone(cuts), N) {
			if lg, err = a.PrefillLast(context.Background(), e[prev:c], prev); err != nil {
				t.Fatal(err)
			}
			prev = c
		}
		return slices.Clone(lg), kv2Slot(t, r, slot, N)
	}
	wholeLg, whole := fill(0, nil, embs)
	pert := slices.Clone(embs)
	pert[300] = mc3Emb(r, 999)
	_, ctl := fill(1, nil, pert)
	if b, p := kv2RowsEqual(ctl, whole, rb, 0, N); b < 0 || p != 300 {
		t.Fatalf("control: first difference at buffer %d row %d, want row 300 — the comparison cannot see a change", b, p)
	}
	chunks := func(w int) []int {
		var out []int
		for c := w; c < N; c += w {
			out = append(out, c)
		}
		return out
	}
	for _, arm := range []struct {
		name string
		cuts []int
	}{{"split@512", []int{512}}, {"split@768", []int{768}}, {"split@984", []int{984}}, {"chunk=64", chunks(64)}, {"chunk=448", chunks(448)}} {
		lg, kv := fill(1, arm.cuts, embs)
		b, p := kv2RowsEqual(kv, whole, rb, 0, N)
		lgDiff := 0
		for i := range lg {
			if math.Float32bits(lg[i]) != math.Float32bits(wholeLg[i]) {
				lgDiff++
			}
		}
		fmt.Fprintf(os.Stderr, "[kv2-g0 metal] %s: first KV difference buffer %d row %d (-1: none); last logits %d of %d differ\n", arm.name, b, p, lgDiff, len(lg))
		if b >= 0 || lgDiff != 0 {
			t.Errorf("%s is not invariant: KV differs at buffer %d row %d, %d logits differ", arm.name, b, p, lgDiff)
		}
	}
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	for p, e := range embs {
		if _, err := a.Forward(e, p); err != nil {
			t.Fatal(err)
		}
	}
	b, p := kv2RowsEqual(kv2Slot(t, r, 1, N), whole, rb, 0, N)
	fmt.Fprintf(os.Stderr, "[kv2-g0 metal] decode token by token against the batched pass: first KV difference buffer %d row %d (-1: none)\n", b, p)
}
