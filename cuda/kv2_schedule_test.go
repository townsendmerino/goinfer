//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G3-C, schedule independence on a CUDA resident through Generate (docs/measurements/kv2-cuda-2026-10-15.md): three
// conversations sharing a 1,000-token preamble on two slots, in the registered schedules. Every turn emits what its
// reference emits, and the K/V rows its slot holds afterwards (prompt and decode rows) are byte-identical to the
// reference's, so every step's logits are too. The reference is the task doc's §5.6: a turn that continued its own slot
// is the conversation run alone, continuing; any other turn is its whole prompt prefilled cold on a fresh load.
//
//	go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKV2_schedule' -v ./cuda/
//
// GOINFER_HEAVY_TESTS=1 adds the 1.5B.

type kv2Turn struct {
	prompt, ids    []int
	reused, shared int
	declined       string
	own            bool     // it continued its own slot: it reused at least the previous turn's prompt
	kv             [][]byte // the bound slot's rows [0, len(prompt)+len(ids)-1) after the turn, per buffer
}

type kv2Sched struct {
	t      *testing.T
	path   string
	opts   decoder.Options
	sp     decoder.SamplingParams
	maxTok int
	pre    []int // token ids, reduced modulo the vocabulary when used
	own    int
	vocab  int
	alone  map[int][]kv2Turn
	colds  map[string]kv2Turn
}

func (s *kv2Sched) load(slots int) *kv2Rig {
	o := s.opts
	o.ResidentKVSlots = slots
	g := kv2Load(s.t, s.path, o)
	s.vocab = g.vocab
	return g
}

func (s *kv2Sched) tok(x int) int { return x % s.vocab }

func (s *kv2Sched) first(c int) []int {
	p := make([]int, 0, len(s.pre)+s.own)
	for _, x := range s.pre {
		p = append(p, s.tok(x))
	}
	for i := range s.own {
		p = append(p, s.tok(3000+c*97+i*7))
	}
	return p
}

func (s *kv2Sched) next(x kv2Turn, c int) []int {
	return append(append(slices.Clone(x.prompt), x.ids...), s.tok(3900+c), s.tok(3901+c*5))
}

// rows is what slot holds of positions [0, n) per buffer, where n-1 is the last position written: every row of a
// whole-context layer, and of a windowed layer the rows the next position attends to, [n-(window-1), n), read through the
// slot's base.
func (g *kv2Rig) rows(slot, n int) [][]byte {
	raw, base := g.raw(slot)
	out := make([][]byte, len(raw))
	for i, b := range g.bufs() {
		if b.win {
			lo := max(0, n-(g.r.kvWindow-1))
			out[i] = raw[i][(lo-base)*b.row : (n-base)*b.row]
			continue
		}
		out[i] = raw[i][:n*b.row]
	}
	return out
}

func (s *kv2Sched) gen(g *kv2Rig, prompt []int) kv2Turn {
	s.t.Helper()
	ch, gn := g.m.Generate(context.Background(), prompt, s.maxTok, s.sp)
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gn.Err(); err != nil {
		s.t.Fatalf("generate: %v", err)
	}
	x := kv2Turn{prompt: prompt, ids: ids, reused: gn.PrefillReused, shared: gn.PrefillShared, declined: gn.KVCopyDeclined}
	slot, _ := g.m.KVBoundClassesForTest()
	x.kv = g.rows(slot, len(prompt)+len(ids)-1)
	return x
}

func (s *kv2Sched) cold(prompt []int) kv2Turn {
	k := fmt.Sprint(prompt)
	if x, ok := s.colds[k]; ok {
		return x
	}
	g := s.load(1)
	x := s.gen(g, prompt)
	_ = g.m.Close() // loads are kept to one or two at a time: an 8 GB card holds few copies of the 1.5B
	s.colds[k] = x
	return x
}

func (s *kv2Sched) references() {
	s.alone, s.colds = map[int][]kv2Turn{}, map[string]kv2Turn{}
	for c := range 3 {
		g := s.load(1)
		t1 := s.gen(g, s.first(c))
		s.alone[c] = []kv2Turn{t1, s.gen(g, s.next(t1, c))}
		_ = g.m.Close()
	}
}

// check compares every turn with its reference and returns how many rows were copied in all.
func (s *kv2Sched) check(t *testing.T, got map[int][]kv2Turn) (shared int) {
	for c, turns := range got {
		for tn, x := range turns {
			want := s.alone[c][tn]
			if tn > 0 && !x.own {
				want = s.cold(x.prompt)
			}
			if !slices.Equal(x.ids, want.ids) {
				t.Errorf("conversation %d turn %d (reused %d, copied %d, own %v): %v, reference %v", c, tn, x.reused, x.shared, x.own, x.ids, want.ids)
			} else if !kv2SameRaw(x.kv, want.kv) {
				t.Errorf("conversation %d turn %d (reused %d, copied %d, own %v): the slot's K/V rows differ from the reference's", c, tn, x.reused, x.shared, x.own)
			}
			shared += x.shared
		}
	}
	return shared
}

func (s *kv2Sched) sequential(slots int, order []int) map[int][]kv2Turn {
	g := s.load(slots)
	out := map[int][]kv2Turn{}
	prompts := map[int][]int{0: s.first(0), 1: s.first(1), 2: s.first(2)}
	for _, c := range order {
		x := s.gen(g, prompts[c])
		if n := len(out[c]); n > 0 {
			x.own = x.reused >= len(out[c][n-1].prompt)
		}
		out[c] = append(out[c], x)
		prompts[c] = s.next(x, c)
	}
	_ = g.m.Close()
	return out
}

func kv2Schedules(t *testing.T, s *kv2Sched) {
	s.references()
	for _, sc := range []struct {
		name  string
		order []int
	}{
		{"(a) A1 B1 C1 then A2 B2 C2", []int{0, 1, 2, 0, 1, 2}},
		{"(b) C1 B1 A1 then C B A", []int{2, 1, 0, 2, 1, 0}},
		{"(c) A1 B1 A2 C1 B2 C2", []int{0, 1, 0, 2, 1, 2}},
	} {
		t.Run(sc.name, func(t *testing.T) {
			if n := s.check(t, s.sequential(2, sc.order)); n == 0 {
				t.Error("no turn copied: the schedule checks nothing of KV2")
			}
		})
	}
	// (f) an edit: A's first prompt, its first reply and a different second turn, after A's second turn. It keeps A's whole
	// first prompt, so it is A's own rollback: it continues A's slot in place, copies nothing, and replies as A does
	// continuing alone into the edited turn.
	t.Run("(f) A edits its second turn", func(t *testing.T) {
		g := s.load(2)
		a1 := s.gen(g, s.first(0))
		a2p := append(slices.Clone(a1.prompt), a1.ids...)
		for i := range s.own {
			a2p = append(a2p, s.tok(3500+i*3))
		}
		s.gen(g, a2p)
		edit := append(append(slices.Clone(a1.prompt), a1.ids...), s.tok(3800), s.tok(3811))
		b := s.gen(g, edit)
		_ = g.m.Close()
		ref := s.load(1)
		s.gen(ref, s.first(0))
		want := s.gen(ref, edit)
		_ = ref.m.Close()
		if b.shared != 0 || b.reused < len(a1.prompt)+len(a1.ids)-1 {
			t.Errorf("the edit reused %d and copied %d: it should continue A's slot in place", b.reused, b.shared)
		}
		if !slices.Equal(b.ids, want.ids) || !kv2SameRaw(b.kv, want.kv) {
			t.Errorf("the edit: %v, A continuing alone %v (K/V equal %v)", b.ids, want.ids, kv2SameRaw(b.kv, want.kv))
		}
	})
	// (d) B arrives while A decodes; (e) A and B start together. Both under MC3. Afterwards each conversation's rows are
	// found in one of the two slots.
	for _, together := range []bool{false, true} {
		name := map[bool]string{false: "(d) B during A's decode", true: "(e) A and B together"}[together]
		t.Run(name, func(t *testing.T) {
			g := s.load(2)
			if got := g.m.EnableResidentConcurrency(2); got != 2 {
				t.Fatalf("EnableResidentConcurrency(2) = %d", got)
			}
			chA, gA := g.m.Generate(context.Background(), s.first(0), s.maxTok, s.sp)
			var idsA []int
			if !together {
				idsA = append(idsA, <-chA)
			}
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				for id := range chA {
					idsA = append(idsA, id)
				}
			}()
			chB, gB := g.m.Generate(context.Background(), s.first(1), s.maxTok, s.sp)
			var idsB []int
			for id := range chB {
				idsB = append(idsB, id)
			}
			wg.Wait()
			if err := gA.Err(); err != nil {
				t.Fatal(err)
			}
			if err := gB.Err(); err != nil {
				t.Fatal(err)
			}
			wa, wb := s.alone[0][0], s.alone[1][0]
			if !slices.Equal(idsA, wa.ids) || !slices.Equal(idsB, wb.ids) {
				t.Errorf("A %v (alone %v); B %v (alone %v), B copied %d", idsA, wa.ids, idsB, wb.ids, gB.PrefillShared)
			}
			if !together && gB.PrefillShared == 0 {
				t.Error("B copied nothing from A's busy slot")
			}
			na, nb := len(wa.prompt)+len(wa.ids)-1, len(wb.prompt)+len(wb.ids)-1
			foundA, foundB := false, false
			for slot := range 2 {
				foundA = foundA || kv2SameRaw(g.rows(slot, na), wa.kv)
				foundB = foundB || kv2SameRaw(g.rows(slot, nb), wb.kv)
			}
			if !foundA || !foundB {
				t.Errorf("K/V rows equal to the reference's: A in a slot %v, B in a slot %v", foundA, foundB)
			}
			t.Logf("B reused %d, copied %d", gB.PrefillReused, gB.PrefillShared)
			_ = g.m.Close()
		})
	}
}

func TestKV2_scheduleIndependence_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, arm := range []struct {
		name string
		kv   string
		sp   decoder.SamplingParams
	}{
		{"greedy-f32", "f32", decoder.SamplingParams{}},
		{"greedy-f16", "f16", decoder.SamplingParams{}},
		{"sampled-f16", "f16", decoder.SamplingParams{Temperature: 0.8, TopK: 40, Seed: 7}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			kv2Schedules(t, &kv2Sched{t: t, path: dir, opts: decoder.Options{ResidentContext: 2048, KVPrecision: arm.kv},
				sp: arm.sp, maxTok: 16, pre: kv2Preamble(1000), own: 70})
		})
	}
}

func TestKV2_scheduleIndependence_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	for _, arm := range []struct {
		name string
		sp   decoder.SamplingParams
	}{
		{"greedy", decoder.SamplingParams{}},
		{"sampled", decoder.SamplingParams{Temperature: 0.8, TopK: 40, Seed: 7}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			kv2Schedules(t, &kv2Sched{t: t, path: path, opts: decoder.Options{ResidentContext: 2048, KVPrecision: "f16"},
				sp: arm.sp, maxTok: 16, pre: kv2Preamble(1000), own: 70})
		})
	}
}

func kv2Preamble(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = 100 + (i*37)%2500
	}
	return p
}

// TestKV2_scheduleWindowedDecline is G3-C's windowed schedule. On a fresh load: A1 then B1, which copies A's preamble; then,
// on another: A1, A2 (long enough to move A's window past the preamble) and C1, whose preamble only A's slot holds, so
// the copy is declined and C prefills cold. Every turn equals its cold reference. A third arm, A1 B1 A2 C1 on three slots,
// reports which source C is offered when one slot has moved on and another still holds the rows; only its correctness is
// graded.
func TestKV2_scheduleWindowedDecline(t *testing.T) { eachWindowFixture(t, kv2ScheduleWindowedDecline) }

func kv2ScheduleWindowedDecline(t *testing.T, fx string) {
	requireCUDADevice(t)
	// A's own part is longer than the 448-row allowance a windowed model gives up in place, so B copies rather than taking
	// A's slot; the fixture's context is widened to hold it. The slack holds A's first turn without a compaction and not
	// its second, so A's base moves past the preamble then.
	withWindowedKVSlack(t, 800)
	s := &kv2Sched{t: t, path: kv2WideFixture(t, filepath.Join("..", "testdata", fx), 4096),
		opts: decoder.Options{ResidentWindowedKV: true, ResidentContext: 4096}, maxTok: 16, pre: kv2Preamble(256), own: 460}
	s.colds = map[string]kv2Turn{}
	coldOK := func(name string, x kv2Turn) {
		t.Helper()
		if want := s.cold(x.prompt); !slices.Equal(x.ids, want.ids) || !kv2SameRaw(x.kv, want.kv) {
			t.Errorf("%s (reused %d, copied %d, declined %q): %v, cold %v (K/V equal %v)", name, x.reused, x.shared, x.declined, x.ids, want.ids, kv2SameRaw(x.kv, want.kv))
		}
	}
	moveA := func(g *kv2Rig, a1 kv2Turn) {
		t.Helper()
		a2p := s.next(a1, 0)
		for i := range 120 {
			a2p = append(a2p, s.tok(3600+i*3))
		}
		s.gen(g, a2p)
		if g.r.kvBases[0] <= len(s.pre)-(g.r.kvWindow-1) {
			t.Fatalf("slot bases %v: A's window has not moved past the preamble, so nothing can decline", g.r.kvBases)
		}
	}

	g := s.load(2)
	if !g.r.kvWin {
		t.Fatalf("windowed KV did not engage (ctxCap %d, slack %d)", g.r.ctxCap, kvWindowSlack)
	}
	s.gen(g, s.first(0))
	b1 := s.gen(g, s.first(1))
	if b1.shared == 0 {
		t.Errorf("B copied nothing from A's slot (declined %q)", b1.declined)
	}
	coldOK("B1", b1)

	g = s.load(2)
	moveA(g, s.gen(g, s.first(0)))
	c1 := s.gen(g, s.first(2))
	t.Logf("A1 A2 C1: C reused %d, copied %d, declined %q; slot bases %v", c1.reused, c1.shared, c1.declined, g.r.kvBases)
	if c1.declined == "" || c1.shared != 0 {
		t.Errorf("C copied %d rows, declined %q: the only source's window had moved on, so the copy must be declined", c1.shared, c1.declined)
	}
	coldOK("C1", c1)

	g = s.load(3)
	a1 := s.gen(g, s.first(0))
	coldOK("B1 (3 slots)", s.gen(g, s.first(1)))
	moveA(g, a1)
	c1 = s.gen(g, s.first(2))
	t.Logf("A1 B1 A2 C1 on 3 slots: C reused %d, copied %d, declined %q; slot bases %v", c1.reused, c1.shared, c1.declined, g.r.kvBases)
	coldOK("C1 (3 slots)", c1)
}
