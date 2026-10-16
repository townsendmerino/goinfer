//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The prefix snapshot's gates on CUDA (docs/tasks/task-kv7-prefix-snapshot-2026-10.md): a restored prefix is the prefix computed, bit
// for bit, in every K and V row the next position reads and in the logits that follow; a snapshot declines what its ring cannot serve
// and nothing is written when it does.
//
//	go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKV7_' -v ./cuda/

// kv7Restore is G-snap(a): slot 0 computes a prompt of end+16 ids and a snapshot of its first end rows is taken; another conversation
// then overwrites slot 0 (the one-slot card's next session) and the rows are restored into slot 0 and into slot 1, each of which held
// other rows. Slot 2 computes the prefix cold. The destination's rows must be slot 2's, from the first row its window reads, and a
// continuation must give slot 2's logits bit for bit.
func kv7Restore(t *testing.T, g *kv2Rig, end int) {
	t.Helper()
	pre := g.toks(end+16, 31) // the prompt goes on 16 ids past the snapshot's end, the lowered rollback of these fixtures
	g.fill(0, pre, 0, false)
	snapLo := 0
	if g.r.kvWin {
		_, base := g.raw(0)
		snapLo = max(0, base, end-(g.r.kvWindow-1)-g.r.kvRollback())
		if base == 0 {
			t.Fatal("VOID: slot 0's base is 0 at the snapshot, so a windowed layer's rows are not read through a shifted view")
		}
	}
	snap, err := g.r.SnapshotKV(0, end)
	if err != nil {
		t.Fatalf("SnapshotKV(0, %d): %v", end, err)
	}
	t.Cleanup(snap.Release)
	if snap.Rows() != end || snap.Bytes() <= 0 || snap.Bytes() > g.r.KVSnapshotBytes(end) {
		t.Errorf("snapshot: %d rows, %d bytes (the estimate is %d)", snap.Rows(), snap.Bytes(), g.r.KVSnapshotBytes(end))
	}
	for _, dst := range []int{0, 1} {
		g.fill(dst, g.toks(300, 32+uint32(dst)), 0, false) // the slot holds another conversation's rows
		g.fill(2, pre[:end], 0, false)                     // the reference computes the prefix cold
		ref, refBase := g.raw(2)
		before, _ := g.raw(dst)
		if err := g.r.RestoreKV(snap, dst, end); err != nil {
			t.Fatalf("slot %d: RestoreKV(%d): %v", dst, end, err)
		}
		after, dstBase := g.raw(dst)
		differed := false
		if g.r.kvWin {
			if dstBase != snapLo {
				t.Errorf("slot %d: ring base %d after the restore, want %d", dst, dstBase, snapLo)
			}
			if snapLo < refBase {
				t.Fatalf("VOID: the reference holds rows from %d, the restore from %d", refBase, snapLo)
			}
		}
		for i, b := range g.bufs() {
			if b.win {
				n := (end - snapLo) * b.row
				want := ref[i][(snapLo-refBase)*b.row : (snapLo-refBase)*b.row+n]
				differed = differed || !bytes.Equal(before[i][:n], want)
				if !bytes.Equal(after[i][:n], want) {
					t.Errorf("slot %d layer %d %s (windowed): rows [%d, %d) are not the computed ones", dst, b.layer, kv2KV(b.v), snapLo, end)
				}
				continue
			}
			n := end * b.row
			differed = differed || !bytes.Equal(before[i][:n], ref[i][:n])
			if !bytes.Equal(after[i][:n], ref[i][:n]) {
				t.Errorf("slot %d layer %d %s: rows [0, %d) are not the computed ones", dst, b.layer, kv2KV(b.v), end)
			}
		}
		if !differed {
			t.Errorf("slot %d: it already held the computed rows before the restore, so the check could not see a restore that did nothing", dst)
		}
		tail := g.toks(200, 34)
		got := g.fill(dst, tail, end, false)
		want := g.fill(2, tail, end, false)
		for i := range got {
			if j, ok := sameBits(got[i], want[i]); !ok {
				t.Fatalf("slot %d: step %d (position %d): logits differ from the computed prefix's (first at %d)", dst, i, end+i, j)
			}
		}
	}
}

// TestKV7_restoreWindowed runs it on the three windowed fixtures, the slack lowered so every slot's base is past 0.
func TestKV7_restoreWindowed(t *testing.T) {
	eachWindowFixture(t, func(t *testing.T, fx string) {
		requireCUDADevice(t)
		withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, windowedKVTestSlack-windowedKVTestStep)
		g := kv2Load(t, filepath.Join("..", "testdata", fx), decoder.Options{ResidentKVSlots: 3, ResidentWindowedKV: true})
		if !g.r.kvWin {
			t.Fatalf("windowed KV did not engage (ctxCap %d, window %d)", g.r.ctxCap, g.r.kvWindow)
		}
		kv7Restore(t, g, 256)
	})
}

// TestKV7_restoreFull runs it on a whole-context fixture, f32 and f16 (the raw words of an f16 cache are copied as they are).
func TestKV7_restoreFull(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv2Load(t, dir, decoder.Options{ResidentKVSlots: 3, ResidentContext: 1024, KVPrecision: kv})
			kv7Restore(t, g, 256)
		})
	}
}

// TestKV7_declines: a snapshot of a slot whose ring has moved past the rows its window needs is refused before anything is
// allocated, and a restore its ring cannot serve, or into another geometry, is refused with the slot untouched.
func TestKV7_declines(t *testing.T) {
	eachWindowFixture(t, func(t *testing.T, fx string) {
		requireCUDADevice(t)
		withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, windowedKVTestSlack-windowedKVTestStep)
		g := kv2Load(t, filepath.Join("..", "testdata", fx), decoder.Options{ResidentKVSlots: 3, ResidentWindowedKV: true})
		const end = 256
		pre := g.toks(end+16, 31)

		// a snapshot taken after the slot decoded on: its ring has moved past the rows position end reads
		g.fill(0, pre, 0, false)
		g.fill(0, g.toks(200, 35), end+16, false)
		if _, base := g.raw(0); base <= end-(g.r.kvWindow-1) {
			t.Fatalf("VOID: slot 0's base %d still holds the rows from %d", base, end-(g.r.kvWindow-1))
		}
		if s, err := g.r.SnapshotKV(0, end); !kv2IsDecline(err) || s != nil || !strings.Contains(err.Error(), "windowed layers hold rows from") {
			t.Fatalf("SnapshotKV after the ring moved on: got %v, want a KVCopyDecline naming the ring", err)
		}

		// a snapshot at the end of a prefill restores to its end and to nothing earlier than the rollback allows
		g.fill(0, pre, 0, false)
		snap, err := g.r.SnapshotKV(0, end)
		if err != nil {
			t.Fatal(err)
		}
		defer snap.Release()
		if !snap.Serves(end) || snap.Serves(end+64) || snap.Serves(0) || snap.Serves(end-1) {
			t.Errorf("Serves: end %v (want true), end+64 %v, 0 %v, end-1 %v (want false)", snap.Serves(end), snap.Serves(end+64), snap.Serves(0), snap.Serves(end-1))
		}
		g.fill(1, g.toks(300, 36), 0, false)
		before, bb := g.raw(1)
		for _, e := range []int{64, 128, 192} {
			if snap.Serves(e) {
				t.Errorf("Serves(%d) with a rollback of %d from %d: the ring behind it is not held", e, g.r.kvRollback(), end)
			}
			if err := g.r.RestoreKV(snap, 1, e); !kv2IsDecline(err) {
				t.Errorf("RestoreKV to %d: got %v, want a KVCopyDecline", e, err)
			}
		}
		// another geometry: the same bytes laid out for a different ring
		g.r.kvSlack++
		if snap.Serves(end) {
			t.Error("a snapshot served a restore after the ring's slack changed")
		}
		if err := g.r.RestoreKV(snap, 1, end); !kv2IsDecline(err) {
			t.Errorf("RestoreKV into another geometry: got %v, want a KVCopyDecline", err)
		}
		g.r.kvSlack--
		after, ab := g.raw(1)
		if !kv2SameRaw(before, after) || ab != bb {
			t.Errorf("a refused restore changed the slot (base %d -> %d)", bb, ab)
		}
		for _, bad := range [][2]int{{3, end}, {-1, end}, {1, 0}, {1, 63}} {
			if err := g.r.RestoreKV(snap, bad[0], bad[1]); err == nil || kv2IsDecline(err) {
				t.Errorf("RestoreKV(%d, %d): got %v, want a plain error", bad[0], bad[1], err)
			}
		}
	})
}

// TestKV7_optionKVHostBytes_resident is the option's end-to-end gate on the real resident: three conversations on a one-slot load,
// the second unrelated and taking the slot, the third starting with the first's prefix. With a budget the third restores the
// prefix from the snapshot and its reply is the reply of the same load with no budget, which computed it; with 0 nothing is held.
func TestKV7_optionKVHostBytes_resident(t *testing.T) {
	type fx struct {
		name, dir string
		opts      decoder.Options
		windowed  bool
	}
	cases := []fx{
		{"full/f32", "../testdata/llama-attnfa-tiny", decoder.Options{ResidentKVSlots: 1, ResidentContext: 4096, KVPrecision: "f32"}, false},
		{"full/f16", "../testdata/llama-attnfa-tiny", decoder.Options{ResidentKVSlots: 1, ResidentContext: 4096, KVPrecision: "f16"}, false},
	}
	for _, name := range windowFixtures {
		cases = append(cases, fx{"windowed/" + name, filepath.Join("..", "testdata", name), decoder.Options{ResidentKVSlots: 1, ResidentWindowedKV: true}, true})
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			requireDeviceAndFixture(t, c.dir)
			if c.windowed {
				withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, windowedKVTestSlack-windowedKVTestStep)
			}
			dir := kv2WideFixture(t, c.dir, 4096)
			run := func(budget int64) (reply []int, gen *decoder.Generation, stats decoder.KVSnapshotStats, kept bool) {
				o := c.opts
				o.KVHostBytes = budget
				if c.windowed {
					o.ResidentContext = 4096
				}
				g := kv2Load(t, dir, o)
				pre := g.toks(2432, 41)
				a := append(slices.Clone(pre), g.toks(16, 42)...) // 16 past its last block: within the lowered rollback of the windowed fixtures
				b := append(slices.Clone(pre), g.toks(40, 43)...) // 40 left to compute: over the 16-row fast-pass minimum
				kv2Generate(t, g.m, a, 6)
				kv2Generate(t, g.m, g.toks(2300, 44), 6) // takes the one slot
				reply, gen = kv2Generate(t, g.m, b, 6)
				stats, kept = g.m.KVSnapshotStats()
				return
			}
			want, gc, _, keptC := run(0)
			if gc.KVSnapRestored != 0 || keptC {
				t.Fatalf("with no budget: %d rows restored, stats kept %v", gc.KVSnapRestored, keptC)
			}
			got, g, st, kept := run(1 << 28)
			if !kept || st.Taken < 1 || st.Restored != 1 || g.KVSnapRestored != 2432 {
				t.Fatalf("VOID: the restore case is not set up: kept %v, %+v, restored %d rows (want 2432)", kept, st, g.KVSnapRestored)
			}
			if !slices.Equal(got, want) {
				t.Errorf("the reply with a restored prefix %v differs from the computed one %v", got, want)
			}
			if st.Used > st.Budget || st.Held < 1 {
				t.Errorf("budget %d, used %d, held %d", st.Budget, st.Used, st.Held)
			}
		})
	}
}
