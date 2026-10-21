//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// A prompt that restores a snapshot of its first `end` rows and prefills the rest must leave the cache and give the logits a cold prefill
// of the whole prompt gives, when the restore point is a pass boundary of the cold prefill. This is the production shape of a session
// start (decoder.kvsnap: restore, then prefill the tail), not the rows-only comparison of TestKV7_restoreFull. It runs at a pass width of
// 64 so a few hundred ids are several passes. The two planted restores, one row short and one row long, must DIFFER from the cold prefill:
// they are the check that this comparison can go red.
//
//	go test -count=1 -timeout 10m -tags 'cuda goinfer_testhooks' -run '^TestKV7_restoreMatchesCold' -v ./cuda/
func TestKV7_restoreMatchesColdAtPassBoundaries(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv2Load(t, filepath.Join(dir), decoder.Options{ResidentKVSlots: 3, ResidentContext: 1024, KVPrecision: kv})
			const pass = 64
			decoder.SetKnobEnvForTest(t, g.m, "GOINFER_PREFILL_CHUNK", strconv.Itoa(pass))
			if got := g.r.prefillPassRows(); got != pass {
				t.Fatalf("VOID: pass width %d, the test needs %d", got, pass)
			}
			// Both prefills must run the fast kernels: the kernel class follows the length of the prompt a pass belongs to, so the snapshot's own
			// prompt is at least the fast-prefill floor long, as production's are (a snapshot is taken after a cold prefill of 2048 ids or more).
			floor := g.r.FastPrefillFloor()
			end := (floor/pass + 2) * pass
			n := end + 3*pass
			if n > 960 {
				t.Fatalf("VOID: the fast-prefill floor %d leaves no room for the prompt in a context of 1024", floor)
			}
			ids := g.toks(n, 41)

			// The cold prefill: slot 2, the whole prompt.
			cold := g.fill(2, ids, 0, true)[0]
			coldKV, _ := g.raw(2)

			// snapshotOf computes `prompt` in slot 0 and snapshots its first `rows` rows.
			snapshotOf := func(prompt []int, rows int) decoder.ResidentKVSnapshot {
				t.Helper()
				g.fill(0, prompt, 0, true)
				s, err := g.r.SnapshotKV(0, rows)
				if err != nil {
					t.Fatalf("SnapshotKV(0, %d): %v", rows, err)
				}
				t.Cleanup(s.Release)
				return s
			}

			// restored runs the production sequence into slot 1, which holds another conversation: restore `rows` rows of snap, prefill ids[from:].
			restored := func(snap decoder.ResidentKVSnapshot, rows, from int) ([]float32, [][]byte) {
				t.Helper()
				g.fill(1, g.toks(n, 42), 0, true)
				if err := g.r.RestoreKV(snap, 1, rows); err != nil {
					t.Fatalf("RestoreKV(%d): %v", rows, err)
				}
				lg := g.fill(1, ids[from:], from, true)[0]
				kv, _ := g.raw(1)
				return lg, kv
			}
			same := func(lg []float32, kv [][]byte) (logits, rows bool) {
				_, logits = sameBits(lg, cold)
				rows = true
				for i, b := range g.bufs() {
					if !bytes.Equal(kv[i][:n*b.row], coldKV[i][:n*b.row]) {
						rows = false
					}
				}
				return logits, rows
			}

			snap := snapshotOf(ids[:end], end)
			lg, kvRows := restored(snap, end, end)
			if l, r := same(lg, kvRows); !l || !r {
				t.Errorf("restore at %d (a pass boundary of the cold prefill, fast floor %d), %d ids: logits bit-equal=%v, K/V rows [0,%d) bit-equal=%v; want both", end, floor, n, l, n, r)
			}

			// The plants: each is a restore that is wrong by a position, and each must differ from the cold prefill.
			shifted := snapshotOf(ids[1:end+1], end) // the rows are in place and hold the K/V of the prompt shifted by one token
			short := snapshotOf(ids[:end-pass], end-pass)
			for _, p := range []struct {
				name       string
				snap       decoder.ResidentKVSnapshot
				rows, from int
			}{
				{"resumed one row late", snap, end, end + 1},
				{"rows of the prompt shifted by one token", shifted, end, end},
				{"restored one block short", short, end - pass, end},
			} {
				lg, kvRows := restored(p.snap, p.rows, p.from)
				if l, r := same(lg, kvRows); l && r {
					t.Errorf("VOID: the plant %q matched the cold prefill, so this test cannot see a restore that is wrong by a position", p.name)
				}
			}
		})
	}
}
