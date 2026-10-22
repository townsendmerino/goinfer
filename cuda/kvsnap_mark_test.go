//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// KV7 step 2 on CUDA (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8, M1 and M2): a snapshot taken between two passes of a cold prefill is the
// prefix computed, a prefill that carries a mark is the prefill without one, and a restore of that snapshot followed by the rest of the prompt reaches
// the cold prompt's rows and logits. The windowed fixtures run with a slack of 63 so a pass is 64 rows (a block, as the snapshot needs) and a mark can
// fall on a pass boundary.
//
//	go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKV7Mark_' -v ./cuda/

const kv7MarkSlack, kv7MarkStep = 63, 8

func (g *kv2Rig) embeds(ids []int) [][]float32 {
	e := make([][]float32, len(ids))
	for i, id := range ids {
		e[i] = append([]float32(nil), g.m.EmbedResidentForTest(id)...)
	}
	return e
}

// kv7SameRows compares two slots' buffers: a full layer's rows [0, end), a windowed layer's from lo (the first row both hold).
func kv7SameRows(t *testing.T, g *kv2Rig, what string, a, b int, end int) {
	t.Helper()
	ra, baseA := g.raw(a)
	rb, baseB := g.raw(b)
	lo := max(baseA, baseB)
	for i, bf := range g.bufs() {
		if bf.win {
			n := (end - lo) * bf.row
			if !bytes.Equal(ra[i][(lo-baseA)*bf.row:(lo-baseA)*bf.row+n], rb[i][(lo-baseB)*bf.row:(lo-baseB)*bf.row+n]) {
				t.Errorf("%s: layer %d %s (windowed): rows [%d, %d) of slots %d and %d differ", what, bf.layer, kv2KV(bf.v), lo, end, a, b)
			}
			continue
		}
		n := end * bf.row
		if !bytes.Equal(ra[i][:n], rb[i][:n]) {
			t.Errorf("%s: layer %d %s: rows [0, %d) of slots %d and %d differ", what, bf.layer, kv2KV(bf.v), end, a, b)
		}
	}
}

func kv7MarkRun(t *testing.T, fx string) {
	requireCUDADevice(t)
	withWindowedKV(t, kv7MarkSlack, kv7MarkStep, kv7MarkSlack-kv7MarkStep)
	g := kv2Load(t, filepath.Join("..", "testdata", fx), decoder.Options{ResidentKVSlots: 4, ResidentWindowedKV: true})
	if !g.r.kvWin {
		t.Fatalf("windowed KV did not engage (ctxCap %d, window %d)", g.r.ctxCap, g.r.kvWindow)
	}
	if g.r.PrefillMarkAlign() != 64 {
		t.Fatalf("VOID: a pass is %d rows, the mark needs a block-wide pass", g.r.PrefillMarkAlign())
	}
	const n, mark = 400, 256
	p := g.toks(n, 71)
	other := g.toks(300, 72)

	// the cold references: slot 3 the whole prompt in one unmarked prefill, slot 2 its first mark ids.
	use := func(slot int) {
		t.Helper()
		if err := g.r.UseKVSlot(slot); err != nil {
			t.Fatal(err)
		}
	}
	use(3)
	wantLogits, err := g.r.PrefillLast(g.ctx, g.embeds(p), 0)
	if err != nil {
		t.Fatalf("unmarked prefill: %v", err)
	}
	use(2)
	if _, err := g.r.PrefillLast(g.ctx, g.embeds(p[:mark]), 0); err != nil {
		t.Fatalf("reference prefill of the first %d ids: %v", mark, err)
	}

	// M1/M2: slot 0 prefills the prompt with a mark; the snapshot is taken from inside the callback, between two passes.
	use(0)
	var snap decoder.ResidentKVSnapshot
	var fired []int
	gotLogits, err := g.r.PrefillLastMarked(g.ctx, g.embeds(p), 0, mark, func(pos int) {
		fired = append(fired, pos)
		if _, base := g.raw(0); base == 0 {
			t.Errorf("VOID: slot 0's ring base is 0 at the mark, so no windowed layer has released a block below it and the moved ring is not exercised")
		}
		s, e := g.r.SnapshotKV(0, pos)
		if e != nil {
			t.Errorf("SnapshotKV(0, %d) between the passes: %v", pos, e)
			return
		}
		snap = s
	})
	if err != nil {
		t.Fatalf("marked prefill: %v", err)
	}
	if !slices.Equal(fired, []int{mark}) {
		t.Fatalf("the mark fired at %v, want once at %d", fired, mark)
	}
	if snap == nil {
		t.Fatal("no snapshot was taken")
	}
	t.Cleanup(snap.Release)
	if snap.Rows() != mark {
		t.Errorf("the snapshot holds %d rows, want %d", snap.Rows(), mark)
	}
	// M2: taking it did not change the prefill in progress.
	if !slices.Equal(gotLogits, wantLogits) {
		t.Error("M2: the marked prefill's final logits differ from the unmarked prefill's")
	}
	kv7SameRows(t, g, "M2: the marked prefill against the unmarked one", 0, 3, n)

	// M1: the restored rows are the prefix computed, and the rest of the prompt reaches the cold prompt.
	use(1)
	if _, err := g.r.PrefillLast(g.ctx, g.embeds(other), 0); err != nil { // slot 1 holds another conversation
		t.Fatal(err)
	}
	if err := g.r.RestoreKV(snap, 1, mark); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	kv7SameRows(t, g, "M1: the restored prefix against the computed one", 1, 2, mark)
	use(1)
	cont, err := g.r.PrefillLast(g.ctx, g.embeds(p[mark:]), mark)
	if err != nil {
		t.Fatalf("prefill after the restore: %v", err)
	}
	if !slices.Equal(cont, wantLogits) {
		t.Error("M1: the logits after restore-and-continue differ from the cold prompt's")
	}
	kv7SameRows(t, g, "M1: restore-and-continue against the cold prompt", 1, 3, n)

	// a mark that is not at a pass boundary calls nothing
	use(0)
	called := false
	if _, err := g.r.PrefillLastMarked(g.ctx, g.embeds(p), 0, mark+8, func(int) { called = true }); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Errorf("a mark at %d, which is not a pass boundary, fired", mark+8)
	}
}

func TestKV7Mark_midPrefillSnapshot(t *testing.T) {
	eachWindowFixture(t, func(t *testing.T, fx string) { kv7MarkRun(t, fx) })
}
