//go:build darwin

package metal

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// kv2Turn is one served turn: its prompt, what it emitted, and where its reuse came from.
type kv2Turn struct {
	prompt, ids     []int
	reused, shared  int
	ownHistoryReuse bool // it continued its own slot: it reused at least the previous turn's prompt
}

// TestKV2_scheduleIndependenceMetal is G3 on a Metal resident through the production Generate path
// (docs/measurements/kv2-2026-10-14.md): three conversations sharing a 1,000-token preamble on two slots, in the registered
// schedules, each turn emitting exactly what its reference emits. The reference follows the task doc's §5.6: a turn that
// continued its own slot is compared with the conversation run alone (continuing); any other turn with its whole prompt
// prefilled cold on a fresh model. Schedules (d) and (e) run under MC3.
func TestKV2_scheduleIndependenceMetal(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const pre, own, maxTok = 1000, 70, 16
	path := mc3FixtureWriter(t, 4096)
	load := func(slots int) *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: mc3TestQuant, ResidentContext: 2048, ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { m.Close() })
		if !m.ResidentActive() {
			t.Skipf("not resident: %s", m.ResidentDecline())
		}
		return m
	}
	preamble := make([]int, pre)
	for i := range preamble {
		preamble[i] = 100 + (i*37)%15000
	}
	first := func(c int) []int {
		p := slices.Clone(preamble)
		for i := range own {
			p = append(p, 16000+c*500+i)
		}
		return p
	}
	gen := func(m *decoder.Model, prompt []int) kv2Turn {
		ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return kv2Turn{prompt: prompt, ids: ids, reused: g.PrefillReused, shared: g.PrefillShared}
	}
	next := func(tn kv2Turn, c int) []int {
		return append(append(slices.Clone(tn.prompt), tn.ids...), 17000+c, 17001+c)
	}
	// References: each conversation alone, continuing, for two turns; and any prompt cold on a fresh one-slot model.
	alone := map[int][]kv2Turn{}
	for c := range 3 {
		m := load(1)
		t1 := gen(m, first(c))
		alone[c] = []kv2Turn{t1, gen(m, next(t1, c))}
	}
	coldCache := map[string][]int{}
	cold := func(prompt []int) []int {
		k := fmt.Sprint(prompt)
		if ids, ok := coldCache[k]; ok {
			return ids
		}
		ids := gen(load(1), prompt).ids
		coldCache[k] = ids
		return ids
	}
	check := func(t *testing.T, got map[int][]kv2Turn) (shared int) {
		for c, turns := range got {
			for tn, x := range turns {
				want := alone[c][tn].ids
				if tn > 0 && !x.ownHistoryReuse {
					want = cold(x.prompt)
				}
				if !slices.Equal(x.ids, want) {
					t.Errorf("conversation %d turn %d (reused %d, copied %d, own history %v): %v, reference %v", c, tn, x.reused, x.shared, x.ownHistoryReuse, x.ids, want)
				}
				shared += x.shared
			}
		}
		return shared
	}
	sequential := func(order []int) map[int][]kv2Turn {
		m := load(2)
		out := map[int][]kv2Turn{}
		prompts := map[int][]int{0: first(0), 1: first(1), 2: first(2)}
		for _, c := range order {
			x := gen(m, prompts[c])
			if n := len(out[c]); n > 0 {
				x.ownHistoryReuse = x.reused >= len(out[c][n-1].prompt)
			}
			out[c] = append(out[c], x)
			prompts[c] = next(x, c)
		}
		return out
	}
	for _, s := range []struct {
		name  string
		order []int
	}{
		{"(a) A1 B1 C1 then A2 B2 C2", []int{0, 1, 2, 0, 1, 2}},
		{"(b) C1 B1 A1 then C B A", []int{2, 1, 0, 2, 1, 0}},
		{"(c) A1 B1 A2 C1 B2 C2", []int{0, 1, 0, 2, 1, 2}},
	} {
		t.Run(s.name, func(t *testing.T) {
			if n := check(t, sequential(s.order)); n == 0 {
				t.Error("no turn copied: the schedule checks nothing of KV2")
			}
		})
	}
	// (f) B is an edit of A: after A's second turn (70 tokens of its own), B is A1, A's first reply and a different
	// second turn, so it gives up more than a block of A's slot and copies. Metal's decode rows are not the cold
	// class, so the copy must stop before A's reply.
	t.Run("(f) B edits A", func(t *testing.T) {
		m := load(2)
		a1 := gen(m, first(0))
		a2p := append(slices.Clone(a1.prompt), a1.ids...)
		for i := range own {
			a2p = append(a2p, 17500+i)
		}
		gen(m, a2p)
		b := gen(m, append(append(slices.Clone(a1.prompt), a1.ids...), 18000, 18001))
		if b.shared > len(a1.prompt) {
			t.Errorf("the edit copied %d rows, past A's prompt (%d): it took rows A's decode wrote", b.shared, len(a1.prompt))
		}
		if !slices.Equal(b.ids, cold(b.prompt)) {
			t.Errorf("the edit: %v, cold %v (copied %d)", b.ids, cold(b.prompt), b.shared)
		}
		if b.shared == 0 {
			t.Error("the edit copied nothing")
		}
	})
	// (d) B arrives while A decodes; (e) A and B start together. Both under MC3.
	for _, together := range []bool{false, true} {
		name := map[bool]string{false: "(d) B during A's decode", true: "(e) A and B together"}[together]
		t.Run(name, func(t *testing.T) {
			m := load(2)
			if got := m.EnableResidentConcurrency(2); got != 2 {
				t.Skipf("EnableResidentConcurrency(2) = %d", got)
			}
			var a, b kv2Turn
			var wg sync.WaitGroup
			chA, gA := m.Generate(context.Background(), first(0), maxTok, decoder.SamplingParams{})
			var idsA []int
			if !together {
				idsA = append(idsA, <-chA)
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				for id := range chA {
					idsA = append(idsA, id)
				}
			}()
			b = gen(m, first(1))
			wg.Wait()
			if err := gA.Err(); err != nil {
				t.Fatal(err)
			}
			a = kv2Turn{ids: idsA}
			if !slices.Equal(a.ids, alone[0][0].ids) || !slices.Equal(b.ids, alone[1][0].ids) {
				t.Errorf("A %v (alone %v); B %v (alone %v), B copied %d", a.ids, alone[0][0].ids, b.ids, alone[1][0].ids, b.shared)
			}
			if !together && b.shared == 0 {
				t.Error("B copied nothing from A's busy slot")
			}
		})
	}
}
