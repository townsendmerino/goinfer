package decoder

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

// What the KV manager's three paths can do with a kind of state at all (cachestate_test.go's allowedHandling, which
// TestCacheStateGrid_everyCellDeclared reads). An evict cannot "share" a kind; a publish cannot refuse one.
func init() {
	allowedHandling[lcPublish] = map[stateHandling]bool{hExact: true, hExactUnlessWrapped: true, hKept: true, hCallerBound: true, hTransient: true}
	allowedHandling[lcShare] = map[stateHandling]bool{hExact: true, hExactUnlessWrapped: true, hInexact: true, hColdOnMismatch: true, hCallerBound: true, hTransient: true}
	allowedHandling[lcEvict] = map[stateHandling]bool{hCleared: true, hKept: true, hTransient: true}
	allowedHandling[lcCkptSave] = map[stateHandling]bool{hExact: true, hKept: true, hTransient: true}
	allowedHandling[lcCkptRestore] = map[stateHandling]bool{hExact: true, hExactUnlessWrapped: true, hColdOnMismatch: true, hRefused: true, hTransient: true}
}

// kvSeq is n token ids no two of which repeat a pattern, so a prefix match is never an accident.
func kvSeq(n, salt int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = (i*7919 + salt*104729) % 50021
	}
	return out
}

// windowResident is a slotResident whose windowed layers have dropped the rows a prefix would need.
type windowResident struct {
	*slotResident
	dropped bool
}

func (w *windowResident) ReusableKV(prefix int) int {
	if w.dropped {
		return 0
	}
	return prefix
}

// TestCacheStateGrid_managerCells checks the publish, share and evict cells against the code they declare, through
// the functions a generation calls. Every kind's cell on every one of the three paths must name a handling this test
// knows how to check for that kind: a new kind, or a cell changed to a handling with no check here, fails.
func TestCacheStateGrid_managerCells(t *testing.T) {
	const n = 3*kvmgr.BlockRows + 17
	hist := kvSeq(n, 1)
	next := append(append([]int(nil), hist...), kvSeq(40, 2)...)
	committed := func(res ResidentForward, lora *loraRuntime, blocks []residentImageBlock) *Model {
		m := &Model{resident: res}
		m.residentAcquire(hist, nil, lora)
		m.residentForgetIDs()
		m.residentCommitIDs(hist, nil, blocks, lora)
		return m
	}
	img := []residentImageBlock{{start: 70, end: 90, hash: 0xA11CE}}
	claim := func(hash uint64) []residentImageClaim { return []residentImageClaim{{Start: 70, Len: 20, Hash: hash}} }

	checks := map[lifecyclePath]map[stateHandling]func(t *testing.T, s cacheState){
		lcPublish: {
			hExact: func(t *testing.T, s cacheState) {
				m := committed(&slotResident{n: 2}, nil, img)
				seq := m.kv.Seq(m.kv.Bound())
				if seq.Published() != n/kvmgr.BlockRows || len(seq.Table()) != seq.Published()+1 {
					t.Fatalf("%d rows committed: %d blocks published of a table of %d, want every full block and one private tail", n, seq.Published(), len(seq.Table()))
				}
				spans, classes := kvSpansOfBlocks(img), seq.Classes()
				if got := m.kv.Manager().Lookup(m.kvRoot(nil), next, spans, classes); got != n/kvmgr.BlockRows*kvmgr.BlockRows {
					t.Errorf("the index returns %d of the %d rows just published", got, n)
				}
				if s == stateImageBlocks {
					other := []kvmgr.ImageSpan{{Start: 70, Len: 20, Hash: 0xB0B}}
					if got := m.kv.Manager().Lookup(m.kvRoot(nil), next, other, classes); got != kvmgr.BlockRows {
						t.Errorf("with another image at the same place the index returns %d rows, want only the block before it (%d)", got, kvmgr.BlockRows)
					}
				}
			},
			hExactUnlessWrapped: func(t *testing.T, s cacheState) {
				// The block is published whether or not the resident still holds the window; what a request gets of it
				// is the share cell's business.
				m := committed(&windowResident{slotResident: &slotResident{n: 2}, dropped: true}, nil, nil)
				if got := m.kv.Seq(m.kv.Bound()).Published(); got != n/kvmgr.BlockRows {
					t.Errorf("%d blocks published, want %d", got, n/kvmgr.BlockRows)
				}
			},
			hKept: func(t *testing.T, s cacheState) {
				switch s {
				case stateAdapter:
					lora := &loraRuntime{name: "a"}
					m := committed(&slotResident{n: 2}, lora, nil)
					if got := m.kv.Seq(m.kv.Bound()).Root().Adapter; got == 0 || got != kvAdapterID(lora) {
						t.Errorf("the published sequence's root names adapter %d, the generation's is %d", got, kvAdapterID(lora))
					}
				default: // a recurrent kind: its state is in no block, and the model keeps one slot
					m := recurrentTestModel(nil)
					m.resident = &slotResident{n: 4}
					if got := m.residentSlotCount(); got != 1 {
						t.Errorf("a recurrent model chooses among %d slots, want 1", got)
					}
				}
			},
		},
		lcShare: {
			hExact: func(t *testing.T, s cacheState) {
				m := committed(&slotResident{n: 2}, nil, nil)
				if got := m.residentAcquire(next, nil, nil); got != n {
					t.Errorf("a continuation reuses %d rows, want all %d", got, n)
				}
				m = committed(&slotResident{n: 2}, nil, nil)
				if got := m.residentAcquire(hist, nil, nil); got != n-1 {
					t.Errorf("the same prompt again reuses %d rows, want %d: the last token is always recomputed", got, n-1)
				}
			},
			hExactUnlessWrapped: func(t *testing.T, s cacheState) {
				kept := &windowResident{slotResident: &slotResident{n: 2}}
				if got := committed(kept, nil, nil).residentAcquire(next, nil, nil); got != n {
					t.Errorf("with the window still held a continuation reuses %d rows, want %d", got, n)
				}
				gone := &windowResident{slotResident: &slotResident{n: 2}, dropped: true}
				if got := committed(gone, nil, nil).residentAcquire(next, nil, nil); got != 0 {
					t.Errorf("with the window dropped a continuation reuses %d rows, want 0", got)
				}
			},
			hInexact: func(t *testing.T, s cacheState) {
				m := recurrentTestModel(nil)
				m.residentCommitIDs(hist, nil, nil, nil)
				if got := m.residentReuseLen(next, nil, nil); got != n {
					t.Errorf("a strict extension reuses %d rows, want %d", got, n)
				}
				edited := append([]int(nil), next...)
				edited[n-1]++
				if got := m.residentReuseLen(edited, nil, nil); got != 0 {
					t.Errorf("an edit one token before the end reuses %d rows of a recurrent sequence, want 0", got)
				}
			},
			hColdOnMismatch: func(t *testing.T, s cacheState) {
				switch s {
				case stateAdapter:
					m := committed(&slotResident{n: 2}, &loraRuntime{name: "a"}, nil)
					if got := m.residentAcquire(next, nil, &loraRuntime{name: "a"}); got != 0 {
						t.Errorf("another adapter runtime reuses %d rows, want 0", got)
					}
				case stateImageBlocks:
					if got := committed(&slotResident{n: 2}, nil, img).residentAcquire(next, claim(0xA11CE), nil); got != n {
						t.Errorf("the same image reuses %d rows, want %d", got, n)
					}
					// One slot, so that the 70 rows are a reuse and not a shared lead that sends the prompt to an empty slot.
					if got := committed(&slotResident{n: 1}, nil, img).residentAcquire(next, claim(0xB0B), nil); got != 70 {
						t.Errorf("another image at the same place reuses %d rows, want the 70 before it", got)
					}
				case stateInt8KV:
					m := committed(&slotResident{n: 2}, nil, nil)
					root := m.kvRoot(nil)
					if root.Precision != m.ResidentKVPrecision() {
						t.Fatalf("the root's precision is %q, the model's is %q", root.Precision, m.ResidentKVPrecision())
					}
					req := kvmgr.Request{Root: root, Tokens: next}
					if got := m.kv.Score(m.kv.Bound(), req, false); got != n {
						t.Fatalf("under its own root the sequence scores %d, want %d", got, n)
					}
					req.Root.Precision = "i8"
					if got := m.kv.Score(m.kv.Bound(), req, false); got != 0 {
						t.Errorf("a request at another KV precision scores %d on these rows, want 0", got)
					}
				default:
					t.Errorf("%q is declared cold-on-mismatch on share and this test has no mismatch to try for it", s)
				}
			},
		},
		lcEvict: {
			hCleared: func(t *testing.T, s cacheState) {
				m := committed(&slotResident{n: 2}, nil, img)
				spans, classes := kvSpansOfBlocks(img), m.kv.Seq(m.kv.Bound()).Classes()
				m.residentForgetIDs()
				if m.resIDs != nil || m.resImgBlocks != nil || m.kv.Seq(m.kv.Bound()) != nil {
					t.Errorf("after a forget the bound slot still has a record")
				}
				if got := m.kv.Manager().Lookup(m.kvRoot(nil), next, spans, classes); got != 0 || m.kv.Manager().Blocks() != 0 {
					t.Errorf("after a forget the index returns %d rows and %d blocks are in use", got, m.kv.Manager().Blocks())
				}
				// A slot given to another prompt: three conversations through two slots, the first is overwritten.
				m = &Model{resident: &slotResident{n: 2}}
				for salt := 1; salt <= 3; salt++ {
					p := kvSeq(n, salt)
					m.residentAcquire(p, nil, nil)
					m.residentForgetIDs()
					m.residentCommitIDs(p, nil, nil, nil)
				}
				// Asked with the classes the first conversation's rows were committed with, so that a zero is the index
				// having dropped them and not a request that names no producer.
				if got := m.kv.Manager().Lookup(m.kvRoot(nil), next, nil, classes); got != 0 {
					t.Errorf("the first conversation's slot was reused and the index still returns %d of its rows", got)
				}
				last := kvSeq(n, 3)
				if got := m.kv.Manager().Lookup(m.kvRoot(nil), append(last, 1), nil, classes); got != n/kvmgr.BlockRows*kvmgr.BlockRows {
					t.Errorf("the index returns %d rows of the last conversation committed, want its full blocks", got)
				}
			},
			hKept: func(t *testing.T, s cacheState) {
				lora := &loraRuntime{name: "a"}
				m := committed(&slotResident{n: 2}, lora, nil)
				id := kvAdapterID(lora)
				m.residentForgetIDs()
				if kvAdapterID(lora) != id {
					t.Errorf("a forget changed the adapter's identity")
				}
				m.residentCommitIDs(hist, nil, nil, lora)
				if got := m.residentAcquire(next, nil, lora); got != n {
					t.Errorf("the same adapter after a forget and a new commit reuses %d rows, want %d", got, n)
				}
			},
		},
	}
	for _, p := range []lifecyclePath{lcPublish, lcShare, lcEvict} {
		for _, s := range cacheStates {
			h := cacheStateGrid[s][p].h
			if h == hTransient || h == hCallerBound {
				continue // nothing of the kind is held by the manager on this path; the cell's reason says where it lives
			}
			check, ok := checks[p][h]
			if !ok {
				t.Errorf("%q x %q is %q, and this test has no check for that handling", s, p, h)
				continue
			}
			t.Run(string(p)+"/"+string(s), func(t *testing.T) { check(t, s) })
		}
	}
}
