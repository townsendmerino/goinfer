//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestB35Snapshot_restoreEqualsCold is B35's CUDA gate (docs/queue-engineering.md, B35): a Gated-DeltaNet model's prefix snapshot, taken
// between two passes of turn 1's cold prefill at the mark, then restored after another prompt has overwritten the slot, continues turn 2
// exactly as a cold prefill of turn 2 does: the last row's logits, every DeltaNet layer's conv ring and state, and the full-attention K/V
// rows, bit for bit. The pass width is pinned to 64 so a short prompt has several passes. A plant (the restore without the state) must differ.
func TestB35Snapshot_restoreEqualsCold(t *testing.T) {
	requireCUDADevice(t)
	path := filepath.Join("..", "testdata", "qwen3next-tiny")
	const (
		nA, mark, n2 = 640, 576, 720 // turn 1, the mark (a pass boundary), turn 2
		keepShared   = 600           // turn 2 repeats turn 1's first 600 ids
	)
	type out struct {
		logits     []float32
		win, state [][]float32
		k          [][]float32
	}
	load := func() (*decoder.Model, *cudaResident) {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { m.Close() })
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
		}
		if !r.KVSnapshotRecurrent() {
			t.Fatal("qwen3next-tiny's snapshots do not carry its DeltaNet state")
		}
		r.prefillChunkCap.Store(64)
		if r.PrefillMarkAlign() != 64 {
			t.Fatalf("pass width %d, want the pinned 64", r.PrefillMarkAlign())
		}
		return m, r
	}
	ids := func(seed, n, vocab int) []int {
		o := make([]int, n)
		for i := range o {
			o[i] = (i*131 + seed*977 + (i/7)*13 + 5) % (vocab - 1)
		}
		return o
	}
	emb := func(m *decoder.Model, xs []int) [][]float32 {
		rows := make([][]float32, len(xs))
		for i, x := range xs {
			rows[i] = m.EmbedResidentForTest(x)
		}
		return rows
	}
	read := func(r *cudaResident, lg []float32, n int) out {
		o := out{logits: append([]float32(nil), lg...)}
		dl := func(b gpu.Buffer, words int) []float32 {
			v := make([]float32, words)
			if err := gpu.Download(b.At(0), v); err != nil {
				t.Fatal(err)
			}
			return v
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			if Ly.isDeltaNet() {
				o.win = append(o.win, dl(Ly.dnWin, Ly.dnWin.Len()))
				o.state = append(o.state, dl(Ly.dnState, Ly.dnState.Len()))
			} else if !Ly.kvShared && !Ly.kvHalf && Ly.kvDim > 0 {
				k, _ := r.readKVForTest(l, n)
				o.k = append(o.k, k)
			}
		}
		return o
	}
	same := func(a, b [][]float32) bool {
		for i := range a {
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					return false
				}
			}
		}
		return true
	}
	ctx := context.Background()

	// Turn 2 cold, on a fresh resident.
	mc, rc := load()
	_, _, _, _, _, _, vocab := mc.Dims()
	a := ids(1, nA, vocab)
	b := append(append([]int(nil), a[:keepShared]...), ids(2, n2-keepShared, vocab)...)
	lc, err := rc.PrefillLast(ctx, emb(mc, b), 0)
	if err != nil {
		t.Fatal(err)
	}
	cold := read(rc, lc, n2)
	if len(cold.state) == 0 || len(cold.k) == 0 {
		t.Fatalf("%d DeltaNet and %d f32 attention layers compared: the fixture must have both", len(cold.state), len(cold.k))
	}

	for _, plant := range []bool{false, true} {
		m, r := load()
		var snap decoder.ResidentKVSnapshot
		if _, err := r.PrefillLastMarked(ctx, emb(m, a), 0, mark, func(pos int) {
			s, e := r.SnapshotKV(0, pos)
			if e != nil {
				t.Fatalf("SnapshotKV(0, %d) at the mark: %v", pos, e)
			}
			snap = s
		}); err != nil {
			t.Fatal(err)
		}
		if snap == nil || snap.Rows() != mark {
			t.Fatalf("no snapshot at the mark %d", mark)
		}
		if snap.Serves(mark-64) || !snap.Serves(mark) {
			t.Fatalf("a recurrent snapshot must serve exactly its end %d: Serves(%d) %v, Serves(%d) %v", mark, mark-64, snap.Serves(mark-64), mark, snap.Serves(mark))
		}
		if _, err := r.PrefillLast(ctx, emb(m, ids(3, 300, vocab)), 0); err != nil { // another conversation takes the slot
			t.Fatal(err)
		}
		if plant {
			snap.(*cudaKVSnapshot).dn = nil // PLANT: the rows without the state
		}
		if err := r.RestoreKV(snap, 0, mark); err != nil {
			t.Fatalf("RestoreKV(%d): %v", mark, err)
		}
		lr, err := r.PrefillLast(ctx, emb(m, b[mark:]), mark)
		if err != nil {
			t.Fatal(err)
		}
		got := read(r, lr, n2)
		ok := same([][]float32{cold.logits}, [][]float32{got.logits}) && same(cold.win, got.win) && same(cold.state, got.state) && same(cold.k, got.k)
		t.Logf("plant=%v: restore at %d then turn 2's suffix bit-identical to a cold turn 2: %v", plant, mark, ok)
		switch {
		case !plant && !ok:
			t.Errorf("the restored turn 2 differs from a cold turn 2")
		case plant && ok:
			t.Errorf("restoring without the state changed nothing: the comparison cannot see the state")
		}
		snap.Release()
	}
}
