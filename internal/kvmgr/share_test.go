package kvmgr

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

var (
	cudaClasses  = ClassMap{Floor: 512, MinRows: 16}
	metalClasses = ClassMap{Floor: 64, MinRows: 1}
)

// TestCapHit is G4(b) of docs/measurements/kv2-2026-10-14.md: a hit from another sequence leaves the requester a suffix
// its cold prefill's class would run.
func TestCapHit(t *testing.T) {
	for _, tc := range []struct {
		name         string
		cm           ClassMap
		n, hit, want int
	}{
		{"CUDA: 15 rows left would be exact class; one block back leaves 79", cudaClasses, 975, 960, 896},
		{"CUDA: 40 rows left are fast", cudaClasses, 1000, 960, 960},
		{"CUDA: exactly 16 left", cudaClasses, 976, 960, 960},
		{"CUDA: a prompt under the floor is exact whatever the hit", cudaClasses, 300, 256, 256},
		{"CUDA: 8 rows left; one block back leaves 72", cudaClasses, 520, 512, 448},
		{"Metal: no row minimum", metalClasses, 975, 960, 960},
		{"one class at every length", ClassMap{}, 975, 960, 960},
		{"no hit", cudaClasses, 975, 0, 0},
	} {
		if got := tc.cm.CapHit(tc.n, tc.hit); got != tc.want {
			t.Errorf("%s: CapHit(%d, %d) = %d, want %d", tc.name, tc.n, tc.hit, got, tc.want)
		}
	}
}

// commitPrompt runs one generation in slot the way the decoder does: announce it, forget the slot, commit prompt
// plus gen.
func commitPrompt(a *Slots, slot int, root Root, prompt, gen []int, cm ClassMap) {
	a.Acquired(slot, 0, prompt)
	a.Forget(slot)
	a.Commit(slot, root, append(slices.Clone(prompt), gen...), nil, len(prompt), cm)
}

func seqOf(n, seed int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = (i*7 + seed*13) % 101
	}
	return out
}

// TestHits_preambleAcrossSlots: B shares A's 200-token preamble. Slot 0 holds A; Hits gives B the three full blocks
// of it, and nothing under another root, another class or past an unclaimed image.
func TestHits_preambleAcrossSlots(t *testing.T) {
	root := Root{Model: "m", Precision: "f16"}
	pre := seqOf(200, 1)
	a := NewSlots(2)
	commitPrompt(a, 0, root, append(slices.Clone(pre), seqOf(30, 2)...), seqOf(20, 3), metalClasses)
	b := append(slices.Clone(pre), seqOf(40, 4)...)
	cold := metalClasses.ColdClasses(len(b))
	if got := a.Hits(Request{Root: root, Tokens: b}, cold); !slices.Equal(got, []int{192, 0}) {
		t.Errorf("Hits = %v, want [192 0]", got)
	}
	other := root
	other.Namespace = "key-2"
	if got := a.Hits(Request{Root: other, Tokens: b}, metalClasses.ColdClasses(len(b))); got[0] != 0 {
		t.Errorf("another namespace: Hits = %v, want nothing", got)
	}
	if got := a.Hits(Request{Root: root, Tokens: b}, []ClassRun{{0, len(b), ClassExact}}); got[0] != 0 {
		t.Errorf("a requester whose cold rows are exact class: Hits = %v, want nothing (A's rows are fast)", got)
	}
	img := []ImageSpan{{Start: 70, Len: 10}} // no content hash: claims nothing
	if got := a.Hits(Request{Root: root, Tokens: b, Images: img}, cold); got[0] != 64 {
		t.Errorf("an unclaimed image in block 1: Hits = %v, want block 0 only", got)
	}
	if got := a.Hits(Request{Root: root, Tokens: pre[:65]}, metalClasses.ColdClasses(65)); got[0] != 64 {
		t.Errorf("a 65-token prompt: Hits = %v, want 64 (one token left to compute)", got)
	}
	if got := a.Hits(Request{Root: root, Tokens: pre[:64]}, metalClasses.ColdClasses(64)); got[0] != 0 {
		t.Errorf("a 64-token prompt: Hits = %v, want 0 (its last token is always computed)", got)
	}
}

// TestHits_imageSpanBoundary: a hit never ends inside an image span; one that would backs off to the span's block.
func TestHits_imageSpanBoundary(t *testing.T) {
	root := Root{Model: "m", Precision: "f16"}
	imgs := []ImageSpan{{Start: 120, Len: 20, Hash: 9}} // rows 120..139 straddle the boundary at 128
	prompt := seqOf(150, 5)
	a := NewSlots(2)
	a.Acquired(0, 0, prompt)
	a.Commit(0, root, append(slices.Clone(prompt), seqOf(70, 6)...), imgs, len(prompt), metalClasses)
	b := append(slices.Clone(prompt), seqOf(70, 6)...)
	b = append(b, 1, 2, 3)
	if got := a.Hits(Request{Root: root, Tokens: b, Images: imgs}, metalClasses.ColdClasses(len(b))); got[0] != 64 {
		t.Errorf("Hits = %v, want 64: block 1 ends inside the image, so the hit backs off to before it", got)
	}
}

// TestCopied: after a copy, the destination holds the prefix as its own published sequence, scores it as a
// continuation, and the source counts as used.
func TestCopied(t *testing.T) {
	root := Root{Model: "m", Precision: "f16"}
	pre := seqOf(200, 1)
	a := NewSlots(3)
	commitPrompt(a, 0, root, append(slices.Clone(pre), seqOf(30, 2)...), seqOf(20, 3), metalClasses)
	use0 := a.Use(0)
	a.Copied(0, 2, 192)
	s := a.Seq(2)
	if s == nil || s.Len() != 192 || s.Published() != 3 || !slices.Equal(s.Tokens(), pre[:192]) {
		t.Fatalf("slot 2 after the copy: %+v", s)
	}
	if !slices.Equal(s.Classes(), []ClassRun{{Start: 0, Len: 192, Class: ClassFast}}) {
		t.Errorf("slot 2's classes %v, want the source's first 192 rows", s.Classes())
	}
	b := append(slices.Clone(pre), seqOf(40, 4)...)
	if got := a.Score(2, Request{Root: root, Tokens: b}, false); got != 192 {
		t.Errorf("Score(2) = %d, want 192", got)
	}
	if a.Use(0) <= use0 {
		t.Errorf("the source's use tick did not move (%d -> %d)", use0, a.Use(0))
	}
	if got := a.Hits(Request{Root: root, Tokens: b}, metalClasses.ColdClasses(len(b))); !slices.Equal(got, []int{192, 0, 192}) {
		t.Errorf("Hits = %v, want both copies of the preamble", got)
	}
	// Forgetting the source leaves the copy: it is the destination's own.
	a.Forget(0)
	if got := a.Hits(Request{Root: root, Tokens: b}, metalClasses.ColdClasses(len(b))); !slices.Equal(got, []int{0, 0, 192}) {
		t.Errorf("Hits after the source is forgotten = %v, want only the copy", got)
	}
}

// TestPrefilled: a prompt's blocks are published when its prefill ends, so a request arriving during the decode finds
// them; the slot is live until its commit; the commit keeps the announced classes.
func TestPrefilled(t *testing.T) {
	root := Root{Model: "m", Precision: "f16"}
	pre := seqOf(200, 1)
	promptA := append(slices.Clone(pre), seqOf(30, 2)...)
	a := NewSlots(2)
	a.Acquired(0, 0, promptA)
	a.Forget(0)
	a.Prefilled(0, root, promptA, nil, metalClasses)
	if !a.Live(0) || a.Seq(0).Published() != 3 {
		t.Fatalf("after Prefilled: live %v, %d published blocks, want live and 3", a.Live(0), a.Seq(0).Published())
	}
	b := append(slices.Clone(pre), seqOf(40, 4)...)
	if got := a.Hits(Request{Root: root, Tokens: b}, metalClasses.ColdClasses(len(b))); got[0] != 192 {
		t.Errorf("during A's decode: Hits = %v, want 192", got)
	}
	a.Commit(0, root, append(slices.Clone(promptA), seqOf(20, 3)...), nil, len(promptA), metalClasses)
	if a.Live(0) {
		t.Error("still live after the commit")
	}
	if c := a.Seq(0).Classes(); len(c) != 2 || c[0].Class != ClassFast || c[0].Len != len(promptA) || c[1].Class != ClassExact {
		t.Errorf("classes after the commit %v, want the prompt fast and the decode exact", c)
	}
	a.Acquired(1, 0, promptA)
	a.Forget(1)
	a.Prefilled(1, root, promptA, nil, metalClasses)
	a.Forget(1)
	if a.Live(1) || a.Seq(1) != nil {
		t.Error("a forgotten live slot keeps its record")
	}
}

// TestPickSlot_oneBlock: a request takes another sequence's slot in place only when it gives up less than one block
// of it; otherwise it goes to an empty slot, else the least recently used one.
func TestPickSlot_oneBlock(t *testing.T) {
	for _, tc := range []struct {
		name         string
		scores, lens []int
		uses         []uint64
		want         int
	}{
		{"a continuation giving up 63 rows stays", []int{1000, 0}, []int{1063, 0}, []uint64{1, 0}, 0},
		{"giving up 64 rows goes to the empty slot", []int{1000, 0}, []int{1064, 0}, []uint64{1, 0}, 1},
		{"a shared preamble: 18,213 of 20,300 goes to the empty slot", []int{18213, 0}, []int{20300, 0}, []uint64{1, 0}, 1},
		{"no empty slot: the least recently used, here the other one", []int{18213, 5}, []int{20300, 900}, []uint64{2, 1}, 1},
		{"no empty slot, and the least recently used holds the hit", []int{18213, 5}, []int{20300, 900}, []uint64{1, 2}, 0},
	} {
		if got := PickSlot(tc.scores, tc.lens, tc.uses, nil); got != tc.want {
			t.Errorf("%s: PickSlot = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestOracle_share walks commits, prefill-end publishes, copies and forgets at random over a few slots, with no images,
// and checks after each: every slot's rows against a plain record, the published counts, the index and the cache,
// and Hits against hits computed from the plain record.
func TestOracle_share(t *testing.T) {
	roots := []Root{{Model: "m", Precision: "f16"}, {Model: "m", Precision: "f16", Namespace: "k2"}}
	for _, cm := range []ClassMap{metalClasses, {Floor: 96, MinRows: 16}} {
		for seed := int64(1); seed <= 6; seed++ {
			t.Run(fmt.Sprintf("floor%d/seed%d", cm.Floor, seed), func(t *testing.T) {
				rng := rand.New(rand.NewSource(seed))
				a := NewSlots(2 + rng.Intn(3))
				type rec struct {
					root    Root
					tokens  []int
					classes []Class
				}
				model := make([]rec, a.Count())
				pre := seqOf(3*BlockRows+5, int(seed))
				prompt := func() []int {
					p := slices.Clone(pre[:rng.Intn(len(pre)+1)])
					for range rng.Intn(2 * BlockRows) {
						p = append(p, rng.Intn(3))
					}
					if len(p) == 0 {
						p = append(p, 1)
					}
					return p
				}
				fill := func(cls Class, from, to int, out []Class) {
					for p := from; p < to; p++ {
						out[p] = cls
					}
				}
				for op := 0; op < 3000; op++ {
					sl := rng.Intn(a.Count())
					switch k := rng.Intn(10); {
					case k < 4: // a cold generation, completed
						p, root := prompt(), roots[rng.Intn(2)]
						gen := seqOf(rng.Intn(BlockRows), op)
						commitPrompt(a, sl, root, p, gen, cm)
						all := append(slices.Clone(p), gen...)
						r := rec{root: root, tokens: all, classes: make([]Class, len(all))}
						fill(cm.Pass(len(p), len(p)), 0, len(p), r.classes)
						fill(ClassExact, len(p), len(all), r.classes)
						model[sl] = r
					case k < 6: // a prompt prefilled, its generation still running
						p, root := prompt(), roots[rng.Intn(2)]
						a.Acquired(sl, 0, p)
						a.Forget(sl)
						a.Prefilled(sl, root, p, nil, cm)
						r := rec{root: root, tokens: slices.Clone(p), classes: make([]Class, len(p))}
						fill(cm.Pass(len(p), len(p)), 0, len(p), r.classes)
						model[sl] = r
					case k < 9: // a copy of another slot's published prefix
						src := rng.Intn(a.Count())
						if src == sl || a.Seq(src) == nil || a.Seq(src).Published() == 0 {
							continue
						}
						rows := (1 + rng.Intn(a.Seq(src).Published())) * BlockRows
						a.Copied(src, sl, rows)
						model[sl] = rec{root: model[src].root, tokens: slices.Clone(model[src].tokens[:rows]), classes: slices.Clone(model[src].classes[:rows])}
					default:
						a.Forget(sl)
						model[sl] = rec{}
					}
					// The checks.
					published := 0
					for s := 0; s < a.Count(); s++ {
						mo := model[s]
						if a.Len(s) != len(mo.tokens) || (a.Seq(s) != nil && !slices.Equal(a.Seq(s).Tokens(), mo.tokens)) {
							t.Fatalf("op %d: slot %d holds %d rows, the record %d", op, s, a.Len(s), len(mo.tokens))
						}
						if q := a.Seq(s); q != nil {
							if q.Published() != len(mo.tokens)/BlockRows {
								t.Fatalf("op %d: slot %d has %d published blocks, %d are full", op, s, q.Published(), len(mo.tokens)/BlockRows)
							}
							published += q.Published()
							for p, c := range mo.classes {
								if got := classAt(q.Classes(), p); got != c {
									t.Fatalf("op %d: slot %d row %d class %d, the record %d", op, s, p, got, c)
								}
							}
						}
					}
					inIndex := 0
					for _, ids := range a.Manager().index {
						inIndex += len(ids)
					}
					if inIndex != published || a.Manager().Cached() != published {
						t.Fatalf("op %d: %d published; the index holds %d, the cache %d", op, published, inIndex, a.Manager().Cached())
					}
					req, root := prompt(), roots[rng.Intn(2)]
					cold := cm.ColdClasses(len(req))
					got := a.Hits(Request{Root: root, Tokens: req}, cold)
					for s := 0; s < a.Count(); s++ {
						mo, want := model[s], 0
						for (want+BlockRows) <= len(mo.tokens) && want+BlockRows <= len(req)-1 && mo.root == root {
							blk := want + BlockRows
							if !slices.Equal(mo.tokens[want:blk], req[want:blk]) {
								break
							}
							ok := true
							for p := want; p < blk; p++ {
								ok = ok && mo.classes[p] == classAt(cold, p) && mo.classes[p] != ClassUnknown
							}
							if !ok {
								break
							}
							want = blk
						}
						if got[s] != want {
							t.Fatalf("op %d: Hits[%d] = %d, the record says %d", op, s, got[s], want)
						}
					}
				}
			})
		}
	}
}
