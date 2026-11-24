package decoder

import (
	"path/filepath"
	"testing"
)

// rowPadArms runs fn with the row-pad prototype on at alignment align (bytes) and returns to the defaults after.
func rowPadArms[T any](t *testing.T, on bool, align int, fn func() T) T {
	t.Helper()
	kvRowPadForTest, kvRowPadAlign = on, align
	defer func() { kvRowPadForTest, kvRowPadAlign = false, kvRowPadAlignDefault }()
	return fn()
}

// TestKVRowPad_padOnlyAtAnAlignedRow pins the pad rule: a row of elems elements of elemBytes bytes is padded by one 64 B line when its
// bytes are a multiple of the alignment (4 KiB by default), and not otherwise.
func TestKVRowPad_padOnlyAtAnAlignedRow(t *testing.T) {
	for _, c := range []struct{ elems, elemBytes, want int }{
		{3072, 4, 16}, // Phi-3 mini f32: 12 KiB
		{3072, 2, 0},  // Phi-3 mini f16: 6 KiB
		{1024, 4, 16}, // 4 KiB
		{2048, 2, 32}, // f16, 4 KiB
		{4096, 2, 32}, // f16, 8 KiB
		{128, 4, 0},   // Qwen2.5-0.5B f32: 512 B
		{1000, 4, 0},
	} {
		if got := kvRowPadElems(c.elems, c.elemBytes, kvRowPadAlignDefault); got != c.want {
			t.Errorf("kvRowPadElems(%d, %d) = %d, want %d", c.elems, c.elemBytes, got, c.want)
		}
	}
}

// TestKVRowPad_decodeBitIdentical holds single-token decode over a padded cache bit-identical to the unpadded one, through
// Model.forward, at f32 and f16, at an alignment that pads both precisions' rows (64 B: llama-tiny's 128 B f32 and 64 B f16 rows) and
// at one that pads neither (4 KiB), and checks the padded store is the one decode read. 40 tokens run the 16-key QK kernel over the
// first 32 keys and the 4-key kernel over the rest.
func TestKVRowPad_decodeBitIdentical(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	f16, f32 := f1Arms(m)
	_, _, _, _, _, _, vocab := m.Dims()
	ids := f1IDs(40, vocab)
	for _, arm := range []struct {
		name string
		m    *Model
	}{{"f32", f32}, {"f16", f16}} {
		for _, align := range []int{64, kvRowPadAlignDefault} {
			ref := f1Decode(t, arm.m, ids)
			var padded *KVCache
			got := rowPadArms(t, true, align, func() [][]float32 {
				c := arm.m.NewCache(len(ids))
				padded = c
				var out [][]float32
				for _, id := range ids {
					lg, err := arm.m.forward(id, c)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out
			})
			wantPad := align == 64
			if (padded.rowPad > 0) != wantPad {
				t.Fatalf("%s align %d: rowPad %d, want padded=%v", arm.name, align, padded.rowPad, wantPad)
			}
			if wantPad && padded.storedRows(0, m.w.arch.NumKVHeads*m.w.arch.HeadDim) != len(ids) {
				t.Fatalf("%s align %d: the padded store holds %d rows, want %d", arm.name, align, padded.storedRows(0, m.w.arch.NumKVHeads*m.w.arch.HeadDim), len(ids))
			}
			if d := f1Diff(got, ref); d != 0 {
				t.Errorf("%s align %d: padded decode differs from unpadded in %d logits", arm.name, align, d)
			}
		}
	}
}

// TestKVRowPad_planPricesThePad holds the CPU KV price to what a padded cache holds: kvBytesForCtx with the prototype on equals the
// bytes a padded cache stores after ctx positions, at both precisions.
func TestKVRowPad_planPricesThePad(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	f16, f32 := f1Arms(m)
	_, _, _, _, _, _, vocab := m.Dims()
	const ctx = 24
	ids := f1IDs(ctx, vocab)
	for _, arm := range []struct {
		name string
		m    *Model
		f16  bool
	}{{"f32", f32, false}, {"f16", f16, true}} {
		rowPadArms(t, true, 64, func() int {
			c := arm.m.NewCache(ctx)
			for _, id := range ids {
				if _, err := arm.m.forward(id, c); err != nil {
					t.Fatal(err)
				}
			}
			held := c.rowPadBytesHeld()
			priced := kvBytesForCtx(arm.m.w.arch, ctx, arm.f16, false)
			if held != priced || held == 0 {
				t.Errorf("%s: the padded cache holds %d bytes at %d positions; the plan prices %d", arm.name, held, ctx, priced)
			}
			if unpadded := rowPadArms(t, false, 64, func() int64 { return kvBytesForCtx(arm.m.w.arch, ctx, arm.f16, false) }); priced <= unpadded {
				t.Errorf("%s: padded price %d is not above the unpadded %d", arm.name, priced, unpadded)
			}
			return 0
		})
	}
}

// TestKVRowPad_otherReadersRefuse checks that the readers the prototype does not cover refuse a padded cache instead of reading its
// empty unpadded store: batched prefill, Keys/Vals, LayerKV and TruncateTo.
func TestKVRowPad_otherReadersRefuse(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	_, f32 := f1Arms(m)
	_, _, _, _, _, _, vocab := m.Dims()
	ids := f1IDs(4, vocab)
	refuses := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s accepted a padded cache", name)
			}
		}()
		fn()
	}
	rowPadArms(t, true, 64, func() int {
		c := f32.NewCache(16)
		if _, err := f32.forward(ids[0], c); err != nil {
			t.Fatal(err)
		}
		refuses("forwardN", func() { _, _ = f32.forwardN(t.Context(), ids, c) })
		refuses("Keys", func() { c.Keys(0) })
		refuses("Vals", func() { c.Vals(0) })
		refuses("LayerKV", func() { c.LayerKV(0) })
		refuses("TruncateTo", func() { c.TruncateTo(0) })
		return 0
	})
}
