//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestMC5_prefillChunkInvariance asks the question chunked prefill's design turns on (docs/tasks/task-concurrency-
// 2026-09.md, chunked prefill): does Metal's fast prefill give the SAME bits when a prompt is ingested in chunks —
// PrefillLast(chunk, startPos) one after another — as when it is ingested whole? The same 1000-token prompt is
// prefilled whole into one resident KV slot and in chunks of C tokens into another; every K and V element of every
// layer, and the last token's logits, are compared bit for bit. It runs by default on the generated fixture
// (mc3_fixture_test.go; E-G01 and F-G02, docs/audit-metal-2026-09-30.md), and on the real checkpoint with:
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC5_prefillChunkInvariance$' -v ./metal/
func TestMC5_prefillChunkInvariance(t *testing.T) {
	a := mc3PrefillResident(t, 2, 2048)
	r := a.r
	const N = 1000
	seed := uint32(13579)
	embs := make([][]float32, N)
	for i := range embs {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		embs[i] = mc3Emb(r, int(seed%20000))
	}
	snapshot := func(slot int) [][]uint16 {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			kvDim := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+N*kvDim]...), append([]uint16(nil), r.vc[l].U16s()[o:o+N*kvDim]...))
		}
		return out
	}
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	whole, err := a.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("whole prefill: %v", err)
	}
	whole = append([]float32(nil), whole...)
	wantKV := snapshot(0)
	// 100 and 77 start chunks off the steel kernel's 32-row tiles; a prefix-reuse turn's startPos is arbitrary too. The
	// aligned sizes cannot see a bug at a tile edge, which shows the same way whole and chunked: with the kernel's causal
	// limit moved one key, every aligned size still matched bit for bit, and 100 and 77 differed from the first chunk
	// boundary on (2026-10-01, F-G02). 512 is serve's default chunk.
	for _, C := range []int{64, 128, 256, 384, 512, 100, 77} {
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		var lg []float32
		for c := 0; c < N; c += C {
			e := min(c+C, N)
			if lg, err = a.PrefillLast(context.Background(), embs[c:e], c); err != nil {
				t.Fatalf("chunk %d..%d: %v", c, e, err)
			}
		}
		lg = append([]float32(nil), lg...)
		gotKV := snapshot(1)
		kvDiff, kvTotal := 0, 0
		firstBad := -1
		for i := range wantKV {
			for j := range wantKV[i] {
				kvTotal++
				if wantKV[i][j] != gotKV[i][j] {
					kvDiff++
					if firstBad < 0 {
						firstBad = j / r.layers[0].geom.kvDim // position
					}
				}
			}
		}
		lgDiff, worst := 0, 0.0
		for i := range whole {
			if math.Float32bits(whole[i]) != math.Float32bits(lg[i]) {
				lgDiff++
				worst = math.Max(worst, math.Abs(float64(whole[i]-lg[i])))
			}
		}
		fmt.Fprintf(os.Stderr, "[mc5-chunk] C=%d: KV %d of %d elements differ (first at position %d); last logits %d of %d differ (max |diff| %.3g)\n",
			C, kvDiff, kvTotal, firstBad, lgDiff, len(whole), worst)
		if kvDiff != 0 || lgDiff != 0 {
			t.Errorf("C=%d: chunked prefill is not bit-identical to whole prefill (%d KV elements, %d logits differ)", C, kvDiff, lgDiff)
		}
	}
	// Control: the comparison must be able to see a difference. The same chunked prefill with ONE token changed (at
	// position 700) must differ, and only from that position on.
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	pert := append([][]float32(nil), embs...)
	pert[700] = mc3Emb(r, 777)
	for c := 0; c < N; c += 256 {
		if _, err := a.PrefillLast(context.Background(), pert[c:min(c+256, N)], c); err != nil {
			t.Fatal(err)
		}
	}
	gotKV := snapshot(1)
	kvDim := r.layers[0].geom.kvDim
	before, after := 0, 0
	for i := range wantKV {
		for j := range wantKV[i] {
			if wantKV[i][j] != gotKV[i][j] {
				if j/kvDim < 700 {
					before++
				} else {
					after++
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[mc5-chunk] control (token 700 changed): %d KV elements differ before position 700, %d from it on\n", before, after)
	if before != 0 || after == 0 {
		t.Errorf("control: %d differences before the changed token, %d after — the comparison is not seeing what it should", before, after)
	}
}
