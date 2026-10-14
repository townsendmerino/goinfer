//go:build darwin

package metal

import (
	"context"
	"errors"
	"math"
	"runtime/debug"
	"slices"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// kv2Embs is n deterministic prompt embeddings, distinct per seed.
func kv2Embs(r *resident, n int, seed uint32) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		out[i] = mc3Emb(r, int(seed%20000))
	}
	return out
}

// kv2Slot is every byte of slot s's K, V and (int8) scale rows [0, rows), layer by layer.
func kv2Slot(t *testing.T, r *resident, s, rows int) [][]byte {
	t.Helper()
	var out [][]byte
	read := func(b Buffer, n int) []byte {
		h := make([]byte, n)
		if err := gpu.Download(b, h); err != nil {
			t.Fatal(err)
		}
		return h
	}
	elem := 2
	if r.kvI8 {
		elem = 1
	}
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || L.geom == nil || r.kvSlotBytes[l] == 0 {
			continue
		}
		n := rows * L.geom.kvDim * elem
		out = append(out, read(r.kvRows(s, l, false, 0), n), read(r.kvRows(s, l, true, 0), n))
		if r.kvI8 {
			out = append(out, read(r.kvSlotBufs[s].ks[l], rows*L.geom.nKV*4), read(r.kvSlotBufs[s].vs[l], rows*L.geom.nKV*4))
		}
	}
	return out
}

// kv2RowsEqual reports the first buffer and row where a and b differ inside [lo, hi) (rowBytes per buffer), or -1.
func kv2RowsEqual(a, b [][]byte, rowBytes []int, lo, hi int) (buf, row int) {
	for i := range a {
		rb := rowBytes[i]
		for p := lo; p < hi; p++ {
			if !slices.Equal(a[i][p*rb:(p+1)*rb], b[i][p*rb:(p+1)*rb]) {
				return i, p
			}
		}
	}
	return -1, -1
}

func kv2RowBytes(r *resident) []int {
	elem := 2
	if r.kvI8 {
		elem = 1
	}
	var out []int
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || L.geom == nil || r.kvSlotBytes[l] == 0 {
			continue
		}
		out = append(out, L.geom.kvDim*elem, L.geom.kvDim*elem)
		if r.kvI8 {
			out = append(out, L.geom.nKV*4, L.geom.nKV*4)
		}
	}
	return out
}

// kv2Resident is the MC3 fixture's resident with three KV slots, at f16 KV or int8 KV.
func kv2Resident(t *testing.T, i8 bool) (*metalResident, *resident) {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	o := decoder.Options{Quant: mc3TestQuant, ResidentContext: 2048, ResidentKVSlots: 3}
	if i8 {
		o.KVPrecision = "i8"
	}
	m, err := decoder.Load(mc3FixtureWriter(t, 4096), o)
	if err != nil {
		t.Fatalf("load the MC3 fixture: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build the resident: %v", err)
	}
	t.Cleanup(func() { r.Close(); m.Close(); debug.FreeOSMemory() })
	if len(r.kvSlotBufs) != 3 || r.kvI8 != i8 {
		t.Skipf("the fixture built %d slots, int8 KV %v; the test needs 3, int8 %v", len(r.kvSlotBufs), r.kvI8, i8)
	}
	return &metalResident{r: r, hidden: r.H}, r
}

// TestKV2_copyKVExact is G2 on Metal (docs/measurements/kv2-2026-10-14.md): after CopyKV(src, dst, start, end) dst's
// rows in the range equal src's byte for byte, dst's other rows and every other slot are unchanged, and a range past the
// cache is declined with nothing written.
func TestKV2_copyKVExact(t *testing.T) {
	for _, i8 := range []bool{false, true} {
		name := map[bool]string{false: "f16", true: "i8"}[i8]
		t.Run(name, func(t *testing.T) {
			a, r := kv2Resident(t, i8)
			const rows = 1024
			for s, n := range []int{1000, 1000, 300} {
				if err := r.useKVSlot(s); err != nil {
					t.Fatal(err)
				}
				embs := kv2Embs(r, n, uint32(100+s))
				if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
					// int8 KV has no batched prefill (it writes f16 K/V): fill the slot as decode does
					for p, e := range embs {
						if _, err := a.Forward(e, p); err != nil {
							t.Fatalf("slot %d position %d: %v", s, p, err)
						}
					}
				}
			}
			rb := kv2RowBytes(r)
			for _, tc := range []struct{ src, dst, start, end int }{{0, 1, 0, 64}, {0, 1, 0, 1024}, {1, 2, 256, 1024}} {
				before := [][][]byte{kv2Slot(t, r, 0, rows), kv2Slot(t, r, 1, rows), kv2Slot(t, r, 2, rows)}
				if err := a.CopyKV(tc.src, tc.dst, tc.start, tc.end); err != nil {
					t.Fatalf("CopyKV%v: %v", tc, err)
				}
				after := [][][]byte{kv2Slot(t, r, 0, rows), kv2Slot(t, r, 1, rows), kv2Slot(t, r, 2, rows)}
				if b, p := kv2RowsEqual(after[tc.dst], before[tc.src], rb, tc.start, tc.end); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d is not src's", tc, b, p)
				}
				if b, p := kv2RowsEqual(after[tc.dst], before[tc.dst], rb, 0, tc.start); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d below the range changed", tc, b, p)
				}
				if b, p := kv2RowsEqual(after[tc.dst], before[tc.dst], rb, tc.end, rows); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d above the range changed", tc, b, p)
				}
				for s := range 3 {
					if s != tc.dst {
						if b, p := kv2RowsEqual(after[s], before[s], rb, 0, rows); b >= 0 {
							t.Errorf("CopyKV%v: slot %d buffer %d row %d changed", tc, s, b, p)
						}
					}
				}
			}
			before := kv2Slot(t, r, 2, rows)
			var d *decoder.KVCopyDecline
			if err := a.CopyKV(0, 2, 0, 4096); !errors.As(err, &d) {
				t.Errorf("a range past the cache: %v, want a KVCopyDecline", err)
			}
			if b, p := kv2RowsEqual(kv2Slot(t, r, 2, rows), before, rb, 0, rows); b >= 0 {
				t.Errorf("a declined copy wrote buffer %d row %d", b, p)
			}
		})
	}
}

// TestKV2_copyThenSuffixEqualsWhole is G0's split arm and G3's device check on Metal: a prompt's first 960 rows copied
// from a slot that prefilled a longer prompt sharing them, then the rest prefilled from 960, gives the KV and the last
// logits of the prompt prefilled whole, bit for bit, both for a 40-row suffix and for a 5-row one (Metal's short
// continuation is exact, PrefillTailExact). The control changes one token before the split and must differ.
func TestKV2_copyThenSuffixEqualsWhole(t *testing.T) {
	a := mc3PrefillResident(t, 2, 2048)
	r := a.r
	if len(r.kvSlotBufs) < 2 {
		t.Skip("needs two slots")
	}
	shared := kv2Embs(r, 990, 7)
	src := append(slices.Clone(shared), kv2Embs(r, 10, 99)...) // a longer prompt that holds the shared rows
	for _, n := range []int{1000, 965} {
		prompt := append(slices.Clone(shared[:960]), kv2Embs(r, n-960, 31)...)
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		whole, err := a.PrefillLast(context.Background(), prompt, 0)
		if err != nil {
			t.Fatal(err)
		}
		whole = slices.Clone(whole)
		want := kv2Slot(t, r, 1, n)
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		if _, err := a.PrefillLast(context.Background(), src, 0); err != nil {
			t.Fatal(err)
		}
		if err := a.CopyKV(0, 1, 0, 960); err != nil {
			t.Fatal(err)
		}
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		got, err := a.PrefillLast(context.Background(), prompt[960:], 960)
		if err != nil {
			t.Fatal(err)
		}
		if b, p := kv2RowsEqual(kv2Slot(t, r, 1, n), want, kv2RowBytes(r), 0, n); b >= 0 {
			t.Errorf("%d tokens: copied + suffix KV differs from whole at buffer %d row %d", n, b, p)
		}
		for i := range whole {
			if math.Float32bits(whole[i]) != math.Float32bits(got[i]) {
				t.Errorf("%d tokens: last logits differ at %d (%g vs %g)", n, i, got[i], whole[i])
				break
			}
		}
	}
	// Control: the source holds a different token at row 500, so the copied rows are not the prompt's, and the result
	// must differ from that row on.
	pert := slices.Clone(src)
	pert[500] = mc3Emb(r, 4321)
	prompt := append(slices.Clone(shared[:960]), kv2Embs(r, 40, 31)...)
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), prompt, 0); err != nil {
		t.Fatal(err)
	}
	want := kv2Slot(t, r, 1, 1000)
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), pert, 0); err != nil {
		t.Fatal(err)
	}
	if err := a.CopyKV(0, 1, 0, 960); err != nil {
		t.Fatal(err)
	}
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), prompt[960:], 960); err != nil {
		t.Fatal(err)
	}
	if b, p := kv2RowsEqual(kv2Slot(t, r, 1, 1000), want, kv2RowBytes(r), 0, 1000); b < 0 || p != 500 {
		t.Errorf("control: first difference at buffer %d row %d, want row 500", b, p)
	}
}
