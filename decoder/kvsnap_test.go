package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

// The prefix snapshot's policy (kvsnap.go) on a hash fake: mc3Fake's rows are a hash of each position's input and its logits a hash
// of the slot's whole history, so a restore that puts back the wrong rows changes every token after it. One slot, as on the 8 GB
// card, so a second conversation overwrites the first and only a snapshot can bring its prefix back.

// snapFake is mc3Fake with a host snapshot (ResidentKVSnapshotter): a snapshot is a copy of a slot's rows, a restore writes them back.
// serveFloor stands in for a windowed ring's rollback: Serves(end) needs end >= rows-serveFloor (0: any end). plant makes the
// restore wrong in one way, to show the gate goes red: "shift" restores every row one position off (a ring based wrong), "noop" reports
// success and writes nothing (the slot keeps the conversation that overwrote it), "short" restores a block less than it claims.
type snapFake struct {
	*mc3Fake
	plant                           string
	serveFloor                      int
	taken, restored, released, rbyt int
	failTake                        error
}

type snapFakeSnap struct {
	f        *snapFake
	rows     int
	hist     []uint64
	released bool
}

func (s *snapFakeSnap) Rows() int    { return s.rows }
func (s *snapFakeSnap) Bytes() int64 { return int64(s.rows) * 1024 }
func (s *snapFakeSnap) Serves(end int) bool {
	return end > 0 && end <= s.rows && end%kvmgr.BlockRows == 0 && (s.f.serveFloor == 0 || end >= s.rows-s.f.serveFloor)
}
func (s *snapFakeSnap) Release() {
	if !s.released {
		s.released = true
		s.f.released++
	}
}

func (f *snapFake) KVSnapshotBytes(end int) int64 { return int64(end) * 1024 }
func (f *snapFake) KVSnapshotRollback() int       { return f.serveFloor }

func (f *snapFake) SnapshotKV(slot, end int) (ResidentKVSnapshot, error) {
	f.enter()
	defer f.leave()
	if f.failTake != nil {
		return nil, f.failTake
	}
	if len(f.hist[slot]) < end {
		return nil, &KVCopyDecline{Reason: "snapFake: the rows are not there"}
	}
	f.taken++
	return &snapFakeSnap{f: f, rows: end, hist: slices.Clone(f.hist[slot][:end])}, nil
}

func (f *snapFake) RestoreKV(h ResidentKVSnapshot, slot, end int) error {
	f.enter()
	defer f.leave()
	s := h.(*snapFakeSnap)
	if !s.Serves(end) {
		return &KVCopyDecline{Reason: "snapFake: cannot serve"}
	}
	f.restored++
	switch f.plant {
	case "noop":
		return nil
	case "short":
		f.hist[slot] = slices.Clone(s.hist[:end-kvmgr.BlockRows])
		return nil
	case "shift":
		rows := slices.Clone(s.hist[:end])
		for i := range rows {
			rows[i] ^= 0x9e3779b97f4a7c15
		}
		f.hist[slot] = rows
		return nil
	}
	f.hist[slot] = slices.Clone(s.hist[:end])
	return nil
}

type snapFakeBackend struct {
	Backend
	rf         *snapFake
	plant      string
	serveFloor int
}

func (b *snapFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &snapFake{mc3Fake: newMC3Fake(vocab, 1), plant: b.plant, serveFloor: b.serveFloor}
	return b.rf, true, nil
}
func (b *snapFakeBackend) Close() error { return nil }

func loadWithSnapFake(t *testing.T, budget int64, plant string, serveFloor int) (*Model, *snapFake) {
	t.Helper()
	be := &snapFakeBackend{plant: plant, serveFloor: serveFloor}
	name := fmt.Sprintf("fake-snap-%d", mc3FakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name, KVHostBytes: budget})
	if err != nil {
		t.Fatalf("Load with the snapshot fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	old, oldW := kvSnapMinPrompt, stderrKV
	kvSnapMinPrompt, stderrKV = 256, io.Discard
	t.Cleanup(func() { kvSnapMinPrompt, stderrKV = old, oldW })
	return m, be.rf
}

func snapToks(n, seed, vocab int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = (i*131 + seed*977 + (i/7)*13 + 5) % (vocab - 1)
	}
	return out
}

// snapRun generates prompt on m under ns and returns the reply and the generation.
func snapRun(t *testing.T, m *Model, ns string, prompt []int) ([]int, *Generation) {
	t.Helper()
	ctx := WithKVScope(context.Background(), KVScope{Namespace: ns})
	ch, gen := m.Generate(ctx, prompt, 6, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids, gen
}

// snapSession is the card's life: conversation A, an unrelated one J that takes the one slot, then B, which starts with A's prefix.
type snapSession struct{ a, j, b []int }

func newSnapSession(vocab int) snapSession {
	pre := snapToks(1000, 1, vocab)
	return snapSession{
		a: append(slices.Clone(pre), snapToks(40, 2, vocab)...),
		j: snapToks(900, 3, vocab),
		b: append(slices.Clone(pre), snapToks(40, 4, vocab)...),
	}
}

// TestKVSnapshot_restoreIsTheComputedPrefix is the tier-0 identity gate at the decoder: B's reply with a snapshot restore is B's reply
// computed, and a restore that is wrong in any of three ways changes it.
func TestKVSnapshot_restoreIsTheComputedPrefix(t *testing.T) {
	run := func(budget int64, plant string) (reply []int, gen *Generation, f *snapFake, m *Model) {
		m, f = loadWithSnapFake(t, budget, plant, 0)
		_, _, _, _, _, _, vocab := m.Dims()
		s := newSnapSession(vocab)
		snapRun(t, m, "", s.a)
		snapRun(t, m, "", s.j)
		reply, gen = snapRun(t, m, "", s.b)
		return
	}
	want, gc, fc, _ := run(0, "")
	if gc.KVSnapRestored != 0 || fc.taken != 0 {
		t.Fatalf("the control restored %d rows and took %d snapshots with the feature off", gc.KVSnapRestored, fc.taken)
	}
	got, g, f, m := run(1<<30, "")
	if g.KVSnapRestored == 0 || f.restored != 1 {
		t.Fatalf("no restore: KVSnapRestored %d, restores %d (taken %d); VOID: the restore case is not set up", g.KVSnapRestored, f.restored, f.taken)
	}
	if g.KVSnapRestored%kvmgr.BlockRows != 0 || g.PrefillReused != g.KVSnapRestored {
		t.Errorf("restored %d rows, reused %d: want whole blocks and the restore as the reuse", g.KVSnapRestored, g.PrefillReused)
	}
	if !slices.Equal(got, want) {
		t.Errorf("B's reply with a restore %v differs from computed %v", got, want)
	}
	if got, want := g.KVSnapRestored, 960; got != want { // 1,000 shared ids, whole blocks
		t.Errorf("restored %d rows, want %d", got, want)
	}
	for _, plant := range []string{"noop", "short", "shift"} {
		bad, gb, _, _ := run(1<<30, plant)
		if gb.KVSnapRestored == 0 {
			t.Fatalf("plant %s: nothing restored", plant)
		}
		if slices.Equal(bad, want) {
			t.Errorf("plant %s: a wrong restore gave the computed reply: the gate cannot see it", plant)
		}
	}
	// every snapshot is released with the model
	_ = m.Close()
	if f.released != f.taken {
		t.Errorf("%d snapshots taken, %d released at Close", f.taken, f.released)
	}
}

// A snapshot belongs to its namespace: B in another one does not restore A's rows; in A's it does.
func TestKVSnapshot_namespaceIsolation(t *testing.T) {
	for _, tc := range []struct {
		nsB  string
		want bool
	}{{"a", true}, {"b", false}, {"", false}} {
		m, f := loadWithSnapFake(t, 1<<30, "", 0)
		_, _, _, _, _, _, vocab := m.Dims()
		s := newSnapSession(vocab)
		snapRun(t, m, "a", s.a)
		snapRun(t, m, "a", s.j)
		_, g := snapRun(t, m, tc.nsB, s.b)
		if (g.KVSnapRestored > 0) != tc.want || (f.restored > 0) != tc.want {
			t.Errorf("B in namespace %q: restored %d rows, want a restore = %v", tc.nsB, g.KVSnapRestored, tc.want)
		}
	}
	// KVScope.Off (-kv-share off) turns snapshots off, taken and restored
	m, f := loadWithSnapFake(t, 1<<30, "", 0)
	_, _, _, _, _, _, vocab := m.Dims()
	s := newSnapSession(vocab)
	off := WithKVScope(context.Background(), KVScope{Off: true})
	for _, p := range [][]int{s.a, s.j, s.b} {
		ch, gen := m.Generate(off, p, 4, SamplingParams{})
		for range ch {
		}
		if gen.Err() != nil || gen.KVSnapRestored != 0 {
			t.Fatalf("under an Off scope: err %v, restored %d", gen.Err(), gen.KVSnapRestored)
		}
	}
	if f.taken != 0 {
		t.Errorf("a snapshot was taken under an Off scope")
	}
}

// The budget holds, the least recently used goes first, and a longer prompt of the same prefix replaces a shorter snapshot.
func TestKVSnapshot_budgetEvictionAndReplace(t *testing.T) {
	m, f := loadWithSnapFake(t, 2_300_000, "", 0) // 1,024-row snapshots are 1.05 MB: two fit
	_, _, _, _, _, _, vocab := m.Dims()
	fam := func(seed, n int) []int { return append(snapToks(n, seed, vocab), snapToks(40, 99, vocab)...) }
	snapRun(t, m, "", fam(10, 1024)) // family 10
	snapRun(t, m, "", fam(11, 1024)) // family 11
	st, _ := m.KVSnapshotStats()
	if st.Held != 2 || st.Taken != 2 || st.Used > st.Budget {
		t.Fatalf("after two families: %+v", st)
	}
	snapRun(t, m, "", fam(12, 1024)) // a third: the least recently used (10) goes
	st, _ = m.KVSnapshotStats()
	if st.Held != 2 || st.Evicted != 1 || st.EvictedUnused != 1 || st.Used > st.Budget {
		t.Errorf("after a third family: %+v", st)
	}
	if x, lcp := m.kvSnaps.best(m.kvRootIn(nil, ""), fam(10, 1024)); x != nil && lcp >= 1024 {
		t.Errorf("the evicted family still holds a snapshot")
	}
	// the same prefix again, longer: it replaces
	snapRun(t, m, "", append(snapToks(1536, 11, vocab), snapToks(40, 98, vocab)...)) // family 11 grown from 1,024 to 1,536 rows
	st, _ = m.KVSnapshotStats()
	rows := 0
	for _, x := range st.Snapshots {
		rows = max(rows, x.Rows)
	}
	if st.Held > 2 || rows != 1536 {
		t.Errorf("a longer prompt of a held prefix: %+v", st)
	}
	// the same length again is not held twice
	before := st.Taken
	snapRun(t, m, "", snapToks(200, 5, vocab)) // a short prompt takes the one slot and is never snapshotted
	snapRun(t, m, "", append(snapToks(1536, 11, vocab), snapToks(40, 97, vocab)...))
	st, _ = m.KVSnapshotStats()
	if st.Taken != before || st.Declined["held"] == 0 {
		t.Errorf("a covered prefix was taken again: taken %d -> %d, declined %v", before, st.Taken, st.Declined)
	}
	// a snapshot bigger than the whole budget is declined, nothing is dropped for it
	big, _ := loadWithSnapFake(t, 500_000, "", 0)
	snapRun(t, big, "", fam(20, 1024))
	if st, _ := big.KVSnapshotStats(); st.Taken != 0 || st.Declined["budget"] == 0 || st.Used != 0 {
		t.Errorf("a snapshot over the budget: %+v", st)
	}
	_ = f
}

// A first prompt whose shared prefix the window cannot serve is counted, and nothing is restored.
func TestKVSnapshot_unqualifiedIsCounted(t *testing.T) {
	m, f := loadWithSnapFake(t, 1<<30, "", 64) // the ring serves a restore only within one block of the snapshot's end
	_, _, _, _, _, _, vocab := m.Dims()
	pre := snapToks(1000, 1, vocab)
	a := append(slices.Clone(pre), snapToks(400, 2, vocab)...) // 1,400 ids: its snapshot is 1,344 rows; the shared 1,000 is far from its end
	b := append(slices.Clone(pre), snapToks(400, 4, vocab)...)
	snapRun(t, m, "", a)
	snapRun(t, m, "", snapToks(900, 3, vocab))
	_, g := snapRun(t, m, "", b)
	st, _ := m.KVSnapshotStats()
	if g.KVSnapRestored != 0 || f.restored != 0 || st.Unqualified != 1 || st.Declined["window"] != 1 {
		t.Errorf("restored %d, unqualified %d, declined %v", g.KVSnapRestored, st.Unqualified, st.Declined)
	}
}

// Prompts under the minimum, with an adapter or off the resident, take nothing; a failing backend is counted, never fatal.
func TestKVSnapshot_admission(t *testing.T) {
	m, f := loadWithSnapFake(t, 1<<30, "", 0)
	_, _, _, _, _, _, vocab := m.Dims()
	snapRun(t, m, "", snapToks(200, 1, vocab))
	if st, _ := m.KVSnapshotStats(); st.Taken != 0 || st.Declined["short"] != 1 {
		t.Errorf("a short prompt: %+v", st)
	}
	f.failTake = &KVCopyDecline{Reason: "no pinned memory"}
	reply, g := snapRun(t, m, "", snapToks(600, 2, vocab))
	if st, _ := m.KVSnapshotStats(); st.Taken != 0 || st.Declined["backend"] != 1 || !strings.Contains(st.LastDecline, "pinned") || len(reply) == 0 || g.Err() != nil {
		t.Errorf("a backend that refuses: %+v", st)
	}
	f.failTake = fmt.Errorf("boom")
	snapRun(t, m, "", snapToks(700, 3, vocab))
	if st, _ := m.KVSnapshotStats(); st.Declined["error"] != 1 {
		t.Errorf("a backend that fails: %+v", st)
	}
	off, fo := loadWithSnapFake(t, 0, "", 0)
	snapRun(t, off, "", snapToks(600, 4, vocab))
	if _, ok := off.KVSnapshotStats(); ok || fo.taken != 0 || off.KVHostBytes() != 0 {
		t.Errorf("a budget of 0 kept snapshots")
	}
}

// The stats /admin/status serves carry the keys its readers look for.
func TestKVSnapshot_statsJSON(t *testing.T) {
	m, _ := loadWithSnapFake(t, 1<<30, "", 0)
	_, _, _, _, _, _, vocab := m.Dims()
	snapRun(t, m, "", snapToks(600, 1, vocab))
	st, _ := m.KVSnapshotStats()
	b, err := json.Marshal(st)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{`"budget_bytes":1073741824`, `"used_bytes":`, `"held":1`, `"taken":1`, `"restored":0`, `"unqualified":0`, `"declined":{`, `"snapshots":[{"rows":576`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("stats JSON lacks %s: %s", k, b)
		}
	}
	if !m.KVSnapshotsSupported() {
		t.Error("a model with a snapshotting resident and a budget reports none supported")
	}
}
