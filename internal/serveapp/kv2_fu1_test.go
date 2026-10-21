package serveapp

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPickSessionFor_ownRollback: with room in the LRU, a request that keeps its candidate's whole first prompt but its
// last token is that conversation rolling back, and reuses the session however much it gives up; one that diverges
// inside the first prompt, or whose first prompt is not known, still goes fresh.
func TestPickSessionFor_ownRollback(t *testing.T) {
	first := make([]int, 600)
	for i := range first {
		first[i] = i
	}
	stored := append(slices.Clone(first), 7000, 7001) // the first prompt and its reply
	for i := range 700 {
		stored = append(stored, 8000+i) // a long second turn
	}
	rollback := append(slices.Clone(first), 7000, 7001, 9000, 9001) // keeps 602, gives up 700
	other := append(slices.Clone(first[:500]), 9500, 9501)          // diverges inside the first prompt
	for _, tc := range []struct {
		name   string
		firsts []int
		prompt []int
		spare  bool
		want   int
	}{
		{"own rollback, room", []int{600}, rollback, true, 0},
		{"a regenerate of the first prompt, room", []int{600}, first, true, 0},
		{"own rollback, first prompt not known", []int{0}, rollback, true, -1},
		{"own rollback, no lengths at all", nil, rollback, true, -1},
		{"diverges inside the first prompt, room", []int{600}, other, true, -1},
		{"diverges inside the first prompt, no room", []int{600}, other, false, 0},
	} {
		if got := pickSessionFor([][]int{stored}, tc.firsts, tc.prompt, tc.spare); got != tc.want {
			t.Errorf("%s: pickSessionFor = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// fu1CPUTurn is one turn served through the LRU: its prompt, its reply with each step's log-probabilities, the session
// it ran on, and what it reused.
type fu1CPUTurn struct {
	prompt, ids    []int
	logprobs       []uint64 // each step's chosen and top log-probabilities, as float64 bits
	s              *decoder.Session
	reused, shared int
}

// TestKV2FU1_cpuOwnRollback is FU1-G3's CPU arm (docs/tasks/task-kv-architecture-2026-10.md §6, KV2 follow-up 1),
// through serve's session LRU on the CPU forward: a conversation that rolls back to its first prompt and first reply,
// giving up more than it keeps, is handed its own session again, reuses what it kept, copies nothing, and replies, to
// the bit of every step's log-probabilities, as it does alone in an LRU of one, with room in the LRU and without.
func TestKV2FU1_cpuOwnRollback(t *testing.T) {
	dir := filepath.Join("..", "..", "testdata", "llama-attnfa-tiny")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", dir)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	vocab := m.Config().VocabSize
	// next is conversation c's next prompt: a 600-token first prompt (300 shared, 300 its own); its reply and 700
	// tokens; then a rollback to the first prompt and its reply plus 2 tokens that diverge right there.
	next := func(c int, ts []fu1CPUTurn) []int {
		switch len(ts) {
		case 0:
			p := make([]int, 0, 600)
			for i := range 300 {
				p = append(p, (i*37+11)%vocab)
			}
			for i := range 300 {
				p = append(p, (c*61+i*7+5)%vocab)
			}
			return p
		case 1:
			p := append(slices.Clone(ts[0].prompt), ts[0].ids...)
			for i := range 700 {
				p = append(p, (c*53+i*3+17)%vocab)
			}
			return p
		default:
			base := append(slices.Clone(ts[0].prompt), ts[0].ids...)
			e := (ts[1].prompt[len(base)] + 1) % vocab
			return append(base, e, (e+97)%vocab)
		}
	}
	sp := decoder.SamplingParams{Logprobs: true, TopLogprobs: 4}
	run := func(size int, order []int) map[int][]fu1CPUTurn {
		l := newSessionLRU(m, size, 0, "fp")
		out := map[int][]fu1CPUTurn{}
		for _, c := range order {
			p := next(c, out[c])
			s := l.acquire(p)
			ch, g := s.Generate(context.Background(), p, 6, sp)
			x := fu1CPUTurn{prompt: p, s: s}
			for id := range ch {
				x.ids = append(x.ids, id)
			}
			if g.Err() != nil {
				t.Fatal(g.Err())
			}
			l.checkin(s)
			for _, in := range g.Logprobs {
				x.logprobs = append(x.logprobs, math.Float64bits(in.Logprob))
				for _, tl := range in.Top {
					x.logprobs = append(x.logprobs, math.Float64bits(tl.Logprob))
				}
			}
			x.reused, x.shared = g.PrefillReused, g.PrefillShared
			out[c] = append(out[c], x)
		}
		return out
	}
	alone := map[int][]fu1CPUTurn{0: run(1, []int{0, 0, 0})[0], 1: run(1, []int{1, 1, 1})[1]}
	for _, tc := range []struct {
		name  string
		size  int
		order []int
	}{
		{"room, A alone", 2, []int{0, 0, 0}},
		{"room, A and B interleaved", 3, []int{0, 1, 0, 1, 0, 1}},
		{"no room, A and B interleaved (control)", 2, []int{0, 1, 0, 1, 0, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := run(tc.size, tc.order)
			deep := 0
			for c, ts := range got {
				for tn, x := range ts {
					ref := alone[c][tn]
					if !slices.Equal(x.prompt, ref.prompt) || !slices.Equal(x.ids, ref.ids) || !slices.Equal(x.logprobs, ref.logprobs) {
						t.Errorf("conversation %d turn %d (reused %d, copied %d): %v, alone %v; log-probabilities equal: %v",
							c, tn, x.reused, x.shared, x.ids, ref.ids, slices.Equal(x.logprobs, ref.logprobs))
					}
					if tn == 0 {
						continue
					}
					prev := ts[tn-1]
					full := append(slices.Clone(prev.prompt), prev.ids...)
					keep := min(len(x.prompt)-1, commonPrefix(full, x.prompt))
					if x.s != prev.s || x.reused != keep || x.shared != 0 {
						t.Errorf("conversation %d turn %d: same session %v, reused %d, copied %d; want its own session, %d reused, nothing copied",
							c, tn, x.s == prev.s, x.reused, x.shared, keep)
					} else if len(full)-keep > keep {
						deep++
					}
				}
			}
			if deep == 0 {
				t.Error("no turn gave up more than it kept: the run checks nothing of FU1")
			}
			t.Logf("rollbacks that gave up more than they kept, served in their own session: %d", deep)
		})
	}
}
