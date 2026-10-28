//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// kv2SchedTurn is one served turn: its prompt, what it emitted, and where its reuse came from.
type kv2SchedTurn struct {
	prompt, ids     []int
	reused, shared  int
	ownHistoryReuse bool // it continued its own slot: it reused at least the previous turn's prompt
}

// TestKV2_scheduleIndependenceWebGPU is G3 on a WebGPU resident through the production Generate path
// (docs/measurements/kv2-2026-10-14.md) on llama-attnfa-tiny (llama-tiny's 128 positions cannot hold a copy: a 64-row
// hit plus the block a request must give up is more): three conversations sharing a 200-token preamble on two slots, in the registered
// schedules, each turn emitting exactly what its reference emits. The reference follows the task doc's §5.6: a turn that
// continued its own slot is compared with the conversation run alone (continuing); any other turn with its whole prompt
// prefilled cold on a fresh model. Schedules (d) and (e) run under MC3.
func TestKV2_scheduleIndependenceWebGPU(t *testing.T) {
	const pre, own, maxTok = 200, 70, 8
	dir := filepath.Join("..", "testdata", "llama-attnfa-tiny")
	load := func(slots int) *decoder.Model {
		m, _ := loadWebGPUResident(t, dir, decoder.Options{Quant: "int4"}, slots)
		t.Cleanup(func() { m.Close() })
		return m
	}
	preamble := make([]int, pre)
	for i := range preamble {
		preamble[i] = 3 + (i*37)%200
	}
	first := func(c int) []int {
		p := slices.Clone(preamble)
		for i := range own {
			p = append(p, (c*61+i*7+5)%256)
		}
		return p
	}
	gen := func(m *decoder.Model, prompt []int) kv2SchedTurn {
		ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return kv2SchedTurn{prompt: prompt, ids: ids, reused: g.PrefillReused, shared: g.PrefillShared}
	}
	next := func(tn kv2SchedTurn, c int) []int {
		return append(append(slices.Clone(tn.prompt), tn.ids...), 230+c, 240+c)
	}
	// References: each conversation alone, continuing, for two turns; and any prompt cold on a fresh one-slot model.
	alone := map[int][]kv2SchedTurn{}
	for c := range 3 {
		m := load(1)
		t1 := gen(m, first(c))
		alone[c] = []kv2SchedTurn{t1, gen(m, next(t1, c))}
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
	check := func(t *testing.T, got map[int][]kv2SchedTurn) (shared int) {
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
	sequential := func(order []int) map[int][]kv2SchedTurn {
		m := load(2)
		out := map[int][]kv2SchedTurn{}
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
	// (f) B is an edit of A: after A's second turn, B is A1, A's first reply and a different second turn. WebGPU has one
	// class (G0: its decode rows equal its batched rows), so the copy may include rows A's decode wrote.
	// (f) An edit: after A's second turn, A's first prompt, its first reply and a different second turn. It keeps A's
	// whole first prompt, so it is A's own rollback (§1, choice 3 as amended): it continues A's slot in place, copies
	// nothing, and replies as A does continuing alone into the edited turn.
	t.Run("(f) A edits its second turn", func(t *testing.T) {
		m := load(2)
		a1 := gen(m, first(0))
		a2p := append(slices.Clone(a1.prompt), a1.ids...)
		for i := range own {
			a2p = append(a2p, 250-i)
		}
		gen(m, a2p)
		edit := append(append(slices.Clone(a1.prompt), a1.ids...), 251, 252)
		b := gen(m, edit)
		ref := load(1)
		gen(ref, first(0))
		want := gen(ref, edit)
		if b.shared != 0 || b.reused < len(a1.prompt)+len(a1.ids)-1 {
			t.Errorf("the edit reused %d and copied %d: it should continue A's slot in place", b.reused, b.shared)
		}
		if !slices.Equal(b.ids, want.ids) {
			t.Errorf("the edit: %v, A continuing alone %v", b.ids, want.ids)
		}
	})
	// (d) B arrives while A decodes; (e) A and B start together. Both under MC3.
	for _, together := range []bool{false, true} {
		name := map[bool]string{false: "(d) B during A's decode", true: "(e) A and B together"}[together]
		t.Run(name, func(t *testing.T) {
			m := load(2)
			if m.EnableResidentConcurrency(2) < 2 {
				t.Skip("WebGPU serves one generation at a time: a concurrent Generate that loses the resident claim runs on the CPU path, so this would compare backends, not schedules (serve queues them; schedules (a) to (c) cover the order)")
			}
			var a, b kv2SchedTurn
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
			a = kv2SchedTurn{ids: idsA}
			if !slices.Equal(a.ids, alone[0][0].ids) || !slices.Equal(b.ids, alone[1][0].ids) {
				t.Errorf("A %v (alone %v); B %v (alone %v), B copied %d", a.ids, alone[0][0].ids, b.ids, alone[1][0].ids, b.shared)
			}
		})
	}
}
