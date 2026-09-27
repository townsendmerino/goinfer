//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSpecVerify_forwardNMatchesForward asks whether n-gram speculation can be lossless on a Metal resident. With
// `--spec ngram`, every round is verified with ForwardN (ForwardBatch; Metal has neither ForwardArgmax nor an
// argmax-only batched verify). Plain decode runs Forward. Speculation's output equals plain decode's only if the two
// give the same bits: the logits a round is accepted on, and the K/V that later tokens and the next turn attend to.
//
// The same 300-token prefix is prefilled into two KV slots. Then 8 tokens run as 8 sequential Forward calls in slot 0,
// and as one ForwardN of 8 rows in slot 1, and again as ForwardN of 1 row. Logits and K/V at positions 300..307 are
// compared bit for bit, per layer.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestSpecVerify_forwardNMatchesForward$' -v ./metal/
func TestSpecVerify_forwardNMatchesForward(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_MC3_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || len(a.r.kvSlotBufs) < 2 || !a.r.kvContig || a.r.kvF32 {
		t.Skip("needs a Metal resident with 2 contiguous f16 KV slots")
	}
	r := a.r
	const P, N = 300, 8
	seed := uint32(24680)
	emb := func() []float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return mc3Emb(r, int(seed%20000))
	}
	prefix := make([][]float32, P)
	for i := range prefix {
		prefix[i] = emb()
	}
	toks := make([][]float32, N)
	for i := range toks {
		toks[i] = emb()
	}
	fill := func(slot int) {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		if _, err := a.PrefillLast(context.Background(), prefix, 0); err != nil {
			t.Fatalf("prefill slot %d: %v", slot, err)
		}
	}
	kvRows := func(slot, from, n int) [][]uint16 { // per layer: K rows then V rows, positions from..from+n-1
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			kvDim := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2) + from*kvDim
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*kvDim]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*kvDim]...))
		}
		return out
	}
	diffKV := func(x, y [][]uint16, kvDimOf func(int) int) (n int, firstPos int) {
		firstPos = -1
		for i := range x {
			kvDim := kvDimOf(i / 2)
			for j := range x[i] {
				if x[i][j] != y[i][j] {
					n++
					if p := j / kvDim; firstPos < 0 || p < firstPos {
						firstPos = p
					}
				}
			}
		}
		return n, firstPos
	}
	kvDimOf := func(l int) int { return r.layers[l].geom.kvDim }
	diffLogits := func(x, y []float32) (n int, worst float64, argEq bool) {
		ax, ay := 0, 0
		for i := range x {
			if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
				n++
				worst = math.Max(worst, math.Abs(float64(x[i]-y[i])))
			}
			if x[i] > x[ax] {
				ax = i
			}
			if y[i] > y[ay] {
				ay = i
			}
		}
		return n, worst, ax == ay
	}

	// slot 0: plain decode, one Forward per token
	fill(0)
	var seqLogits [][]float32
	for i, e := range toks {
		lg, err := a.Forward(e, P+i)
		if err != nil {
			t.Fatal(err)
		}
		seqLogits = append(seqLogits, append([]float32(nil), lg...))
	}
	seqKV := kvRows(0, P, N)

	// slot 1: the same tokens as one ForwardN of N rows (a verify round)
	fill(1)
	nLogits, err := a.ForwardN(toks, P)
	if err != nil {
		t.Fatal(err)
	}
	nKV := kvRows(1, P, N)
	failed := false
	for i := range N {
		d, worst, argEq := diffLogits(seqLogits[i], nLogits[i])
		fmt.Fprintf(os.Stderr, "[spec-verify] M=%d row %d: logits %d of %d differ (max |diff| %.3g), argmax equal %v\n", N, i, d, len(seqLogits[i]), worst, argEq)
		failed = failed || d != 0
	}
	kd, kp := diffKV(seqKV, nKV, kvDimOf)
	where := "none"
	if kd > 0 {
		where = fmt.Sprintf("first at position %d", P+kp)
	}
	fmt.Fprintf(os.Stderr, "[spec-verify] M=%d: K/V at positions %d..%d: %d elements differ (%s)\n", N, P, P+N-1, kd, where)
	failed = failed || kd != 0

	// slot 1 again: ForwardN of one row, rewriting position P (a one-row round, the most common on chat traffic)
	one, err := a.ForwardN(toks[:1], P)
	if err != nil {
		t.Fatal(err)
	}
	d, worst, argEq := diffLogits(seqLogits[0], one[0])
	oneKV := kvRows(1, P, 1)
	kd1, _ := diffKV(kvRows(0, P, 1), oneKV, kvDimOf)
	fmt.Fprintf(os.Stderr, "[spec-verify] M=1: logits %d of %d differ (max |diff| %.3g), argmax equal %v; K/V at position %d: %d elements differ\n",
		d, len(seqLogits[0]), worst, argEq, P, kd1)
	failed = failed || d != 0 || kd1 != 0

	if failed {
		t.Errorf("ForwardN is not bit-identical to sequential Forward on this Metal resident: a verify round's logits or K/V " +
			"differ from plain decode's, so n-gram speculation cannot be token-identical to plain greedy")
	}
}
