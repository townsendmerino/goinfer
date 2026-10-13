package kvmgr

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// naiveScore is Score written the long way round, from the rules and not from the implementation: walk the two token
// lists, and treat each held image span as one unit that either matches a claim whole or stops the walk at its start.
func naiveScore(held, req []int, heldImgs, claims []ImageSpan, sameRoot, recurrent bool) int {
	if len(held) == 0 || len(req) == 0 || !sameRoot {
		return 0
	}
	claim := func(pos int) (ImageSpan, bool) {
		for _, c := range claims {
			if c.Start == pos {
				return c, true
			}
		}
		return ImageSpan{}, false
	}
	if recurrent {
		if len(req) <= len(held) || !slices.Equal(held, req[:len(held)]) {
			return 0
		}
		for _, sp := range heldImgs {
			if c, ok := claim(sp.Start); !ok || sp.Hash == 0 || c.Hash != sp.Hash || c.Len != sp.Len {
				return 0
			}
		}
		for _, c := range claims {
			if c.Start >= len(held) {
				continue
			}
			if c.Start+c.Len > len(held) || !slices.ContainsFunc(heldImgs, func(sp ImageSpan) bool { return sp.Start == c.Start && sp.Len == c.Len }) {
				return 0
			}
		}
		return len(held)
	}
	limit := min(len(req)-1, len(held))
	for i := 0; i < limit; {
		at := slices.IndexFunc(heldImgs, func(sp ImageSpan) bool { return sp.Start == i })
		if at >= 0 {
			sp := heldImgs[at]
			c, ok := claim(i)
			if !ok || c.Len != sp.Len || sp.Hash == 0 || c.Hash != sp.Hash || sp.Start+sp.Len > limit {
				return i
			}
			i = sp.Start + sp.Len
			continue
		}
		if held[i] != req[i] {
			return i
		}
		i++
	}
	return limit
}

type slotModel struct {
	root    Root
	tokens  []int
	images  []ImageSpan
	classes []Class // per row
}

// TestOracle_slots walks the slot adapter at random against a plain per-slot record and checks, after every
// operation: each slot's length and score, where each block lives ((slot, i*BlockRows)), which blocks are published,
// their reference counts, the index and the cache, and the producer class of every row.
func TestOracle_slots(t *testing.T) {
	roots := []Root{{Model: "m", Precision: "f16"}, {Model: "m", Precision: "f16", Adapter: 3}}
	cm := ClassMap{Floor: 96, MinRows: 16}
	for seed := int64(1); seed <= 8; seed++ {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			recurrent := seed%4 == 0
			a := NewSlots(1 + rng.Intn(4))
			model := make([]slotModel, a.Count())
			var history [][]int
			randTokens := func(n int) []int {
				out := make([]int, n)
				for i := range out {
					out[i] = rng.Intn(4)
				}
				return out
			}
			request := func() Request {
				r := Request{Root: roots[rng.Intn(len(roots))]}
				if len(history) > 0 && rng.Intn(4) != 0 {
					h := history[rng.Intn(len(history))]
					r.Tokens = append(r.Tokens, h[:rng.Intn(len(h)+1)]...)
				}
				r.Tokens = append(r.Tokens, randTokens(rng.Intn(3*BlockRows))...)
				if n := len(r.Tokens); n > 4 && rng.Intn(3) == 0 {
					at := rng.Intn(n - 3)
					r.Images = []ImageSpan{{Start: at, Len: 1 + rng.Intn(min(n-at, 40)), Hash: uint64(rng.Intn(3))}}
				}
				return r
			}
			check := func(op int) {
				t.Helper()
				m := a.Manager()
				published := 0
				for sl := 0; sl < a.Count(); sl++ {
					mo := model[sl]
					if a.Len(sl) != len(mo.tokens) {
						t.Fatalf("seed %d op %d: slot %d holds %d rows, the model %d", seed, op, sl, a.Len(sl), len(mo.tokens))
					}
					s := a.Seq(sl)
					if s == nil {
						continue
					}
					if want := len(mo.tokens) / BlockRows; s.Published() != want {
						t.Fatalf("seed %d op %d: slot %d has %d published blocks, %d are full", seed, op, sl, s.Published(), want)
					}
					published += s.Published()
					for i, id := range s.Table() {
						b := m.blocks[id]
						if gs, gr := SlotRow(sl, i); int(b.slot) != gs || int(b.at)*BlockRows != gr {
							t.Fatalf("seed %d op %d: block %d of slot %d is recorded at (%d, %d), want (%d, %d)", seed, op, i, sl, b.slot, int(b.at)*BlockRows, gs, gr)
						}
						if pub := i < s.Published(); b.published != pub || b.cached != pub || (pub && b.refs != 0) || (!pub && b.refs != 1) {
							t.Fatalf("seed %d op %d: slot %d block %d: published %v cached %v refs %d, want published %v", seed, op, sl, i, b.published, b.cached, b.refs, pub)
						}
					}
					for p, want := range mo.classes {
						if got := classAt(s.Classes(), p); got != want {
							t.Fatalf("seed %d op %d: slot %d row %d has class %d, the model %d", seed, op, sl, p, got, want)
						}
					}
				}
				inIndex := 0
				for _, ids := range m.index {
					inIndex += len(ids)
				}
				if inIndex != published || m.Cached() != published || m.Blocks() != published+tails(a) {
					t.Fatalf("seed %d op %d: %d blocks published; the index holds %d, the cache %d, %d blocks in use (%d tails)", seed, op, published, inIndex, m.Cached(), m.Blocks(), tails(a))
				}
			}
			for op := 0; op < 4000; op++ {
				switch k := rng.Intn(10); {
				case k < 6: // a generation: score every slot, pick, bind, forget, and usually commit
					req := request()
					scores, lens, uses := make([]int, a.Count()), make([]int, a.Count()), make([]uint64, a.Count())
					for sl := range scores {
						scores[sl], lens[sl], uses[sl] = a.Score(sl, req, recurrent), a.Len(sl), a.Use(sl)
						mo := model[sl]
						if want := naiveScore(mo.tokens, req.Tokens, mo.images, req.Images, mo.root == req.Root, recurrent); scores[sl] != want {
							t.Fatalf("seed %d op %d: slot %d scored %d, the rules say %d (held %d, request %d, images %v / %v)", seed, op, sl, scores[sl], want, len(mo.tokens), len(req.Tokens), mo.images, req.Images)
						}
					}
					pick := PickSlot(scores, lens, uses, nil)
					a.SetBound(pick)
					a.Touch(pick)
					reuse := cm.DeclineShortLead(len(req.Tokens), scores[pick])
					announced := rng.Intn(5) != 0
					if announced {
						a.Acquired(pick, reuse, req.Tokens)
					} else {
						a.Unrecorded(pick)
					}
					prev := model[pick]
					a.Forget(pick)
					model[pick] = slotModel{}
					if rng.Intn(6) == 0 || len(req.Tokens) == 0 {
						break // a cancelled generation: the slot stays forgotten
					}
					gen := randTokens(rng.Intn(BlockRows))
					all := append(slices.Clone(req.Tokens), gen...)
					imgs := slices.Clone(req.Images) // a span with hash 0 is held too: it claims nothing and must never match
					a.Commit(pick, req.Root, all, imgs, len(req.Tokens), cm)
					mo := slotModel{root: req.Root, tokens: all, images: imgs, classes: make([]Class, len(all))}
					if announced && reuse < len(req.Tokens) {
						copy(mo.classes, prev.classes[:reuse])
						for p := reuse; p < len(req.Tokens); p++ {
							mo.classes[p] = cm.Pass(len(req.Tokens)-reuse, len(req.Tokens))
						}
						for p := len(req.Tokens); p < len(all); p++ {
							mo.classes[p] = ClassExact
						}
					}
					model[pick] = mo
					history = append(history, all)
					if len(history) > 16 {
						history = history[1:]
					}
					// What was just published is findable by a request with the same key, up to the last full block.
					probe := append(slices.Clone(all), 0)
					var runs []ClassRun
					for p, c := range mo.classes {
						runs = appendClass(runs, ClassRun{Start: p, Len: 1, Class: c})
					}
					want := len(all) / BlockRows * BlockRows
					for moved := true; moved; {
						moved = false
						for _, sp := range imgs {
							if sp.Start < want && sp.Start+sp.Len > want {
								want, moved = sp.Start/BlockRows*BlockRows, true
							}
						}
					}
					if got := a.Manager().Lookup(req.Root, probe, imgs, runs); got != want {
						t.Fatalf("seed %d op %d: the index returns %d rows of the %d just committed, want %d", seed, op, got, len(all), want)
					}
				case k < 8: // forget the bound slot
					a.Forget(a.Bound())
					model[a.Bound()] = slotModel{}
				default: // the slot count changes: the bound slot becomes slot 0, the rest are emptied
					n := 1 + rng.Intn(4)
					if n != a.Count() {
						keep := model[a.Bound()]
						a.Resize(n)
						model = make([]slotModel, n)
						model[0] = keep
					}
				}
				check(op)
			}
		})
	}
}

// tails counts the private tail blocks the slots hold.
func tails(a *Slots) int {
	n := 0
	for sl := 0; sl < a.Count(); sl++ {
		if s := a.Seq(sl); s != nil {
			n += len(s.Table()) - s.Published()
		}
	}
	return n
}

func TestPickSlot(t *testing.T) {
	for _, tc := range []struct {
		name         string
		scores, lens []int
		uses         []uint64
		busy         []bool
		want         int
	}{
		{"a continuation wins", []int{3, 170}, []int{170, 172}, []uint64{1, 2}, nil, 1},
		{"a shared lead takes the empty slot", []int{7, 0}, []int{170, 0}, []uint64{1, 0}, nil, 1},
		{"full, a shared lead: least recently used", []int{7, 7, 7}, []int{200, 180, 190}, []uint64{5, 2, 9}, nil, 1},
		{"keep equal to discard reuses", []int{5, 0}, []int{10, 0}, []uint64{1, 0}, nil, 0},
		{"a busy continuation is not taken", []int{3, 170}, []int{170, 172}, []uint64{1, 2}, []bool{false, true}, 0},
		{"all busy", []int{0, 0}, []int{1, 1}, []uint64{1, 2}, []bool{true, true}, -1},
	} {
		if got := PickSlot(tc.scores, tc.lens, tc.uses, tc.busy); got != tc.want {
			t.Errorf("%s: PickSlot = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// A request for 20,000 tokens (Claude Code's first request) and its completion: what the manager adds to a request.
func benchRequest(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(1))
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = rng.Intn(150000)
	}
	gen := prompt[:min(500, n)]
	a := NewSlots(2)
	root := Root{Model: "qwen2.5-7b/int4/metal", Precision: "f16"}
	cm := ClassMap{Floor: 512, MinRows: 16}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := Request{Root: root, Tokens: prompt}
		scores, lens, uses := []int{a.Score(0, req, false), a.Score(1, req, false)}, []int{a.Len(0), a.Len(1)}, []uint64{a.Use(0), a.Use(1)}
		pick := PickSlot(scores, lens, uses, nil)
		a.SetBound(pick)
		a.Touch(pick)
		a.Acquired(pick, cm.DeclineShortLead(n, scores[pick]), prompt)
		a.Forget(pick)
		a.Commit(pick, root, append(slices.Clone(prompt), gen...), nil, n, cm)
	}
}

func BenchmarkRequest_20000(b *testing.B) { benchRequest(b, 20000) }
func BenchmarkRequest_512(b *testing.B)   { benchRequest(b, 512) }
