package kvmgr

import (
	"os"
	"testing"
)

// The class map is the one docs/measurements/kv0-invariance-2026-10-13.md registered and measured on CUDA: each arm
// of that record is a pass here, with the class its launch counts showed.
func TestClassMap_kv0Arms(t *testing.T) {
	cuda := ClassMap{Floor: 512, MinRows: 16}
	for _, tc := range []struct {
		name         string
		rows, judged int
		want         Class
	}{
		{"the whole 1,000-token prompt", 1000, 1000, ClassFast},
		{"a chunk inside a chunked prefill", 64, 1000, ClassFast},
		{"split@256, first call: judged on its own 256", 256, 256, ClassExact},
		{"split@256, second call: 744 rows of a 1,000-token prompt", 744, 1000, ClassFast},
		{"split@511, first call", 511, 511, ClassExact},
		{"split@512, first call", 512, 512, ClassFast},
		{"split@984, second call: 16 rows", 16, 1000, ClassFast},
		{"split@985, second call: 15 rows", 15, 1000, ClassExact},
		{"chunk=496, the 8-row tail", 8, 1000, ClassExact},
	} {
		if got := cuda.Pass(tc.rows, tc.judged); got != tc.want {
			t.Errorf("%s: class %d, want %d", tc.name, got, tc.want)
		}
	}
	if got := (ClassMap{}).Pass(4000, 4000); got != ClassExact {
		t.Errorf("a backend with no fast kernels wrote class %d", got)
	}
	if got := (ClassMap{Floor: 16}).Pass(1, 40); got != ClassFast { // Metal: the pass's end decides, any row count
		t.Errorf("a one-row pass ending past a floor of 16, with no row minimum, wrote class %d", got)
	}
}

func TestDeclineShortLead(t *testing.T) {
	cuda := ClassMap{Floor: 512, MinRows: 16}
	for _, tc := range []struct {
		name                string
		cm                  ClassMap
		prompt, reuse, want int
	}{
		{"a short lead under a fast prompt is declined", cuda, 941, 13, 0},
		{"one under the margin is declined", cuda, 941, MinLeadReuse - 1, 0},
		{"the margin is kept", cuda, 941, MinLeadReuse, MinLeadReuse},
		{"a continuation is kept", cuda, 941, 900, 900},
		{"a prompt under the floor keeps its lead: its cold prefill is exact too", cuda, 511, 5, 5},
		{"a prompt at the floor is fast", cuda, 512, 5, 0},
		{"no fast kernels: kept", ClassMap{}, 941, 13, 13},
		// The 16-row rule: a pass of 15 rows or fewer is exact whatever the floor says, so a prompt that short keeps
		// its lead even when the floor is under it.
		{"15 rows over a floor of 8: exact, kept", ClassMap{Floor: 8, MinRows: 16}, 15, 5, 5},
		{"16 rows over a floor of 8: fast, declined", ClassMap{Floor: 8, MinRows: 16}, 16, 5, 0},
		{"nothing reused", cuda, 941, 0, 0},
	} {
		if got := tc.cm.DeclineShortLead(tc.prompt, tc.reuse); got != tc.want {
			t.Errorf("%s: %d, want %d", tc.name, got, tc.want)
		}
	}
}

// The same classes cut into different runs are the same key.
func TestBlockHash_runsDoNotMatter(t *testing.T) {
	tokens := make([]int, 2*BlockRows)
	for i := range tokens {
		tokens[i] = i * 7
	}
	rh := rootHash(Root{Model: "m"}, allParts)
	var h hasher
	one := blockHash(&h, rh, 1, tokens, nil, []ClassRun{{0, 100, ClassFast}, {100, 28, ClassExact}}, allParts)
	two := blockHash(&h, rh, 1, tokens, nil, []ClassRun{{0, 64, ClassFast}, {64, 36, ClassFast}, {100, 10, ClassExact}, {110, 18, ClassExact}}, allParts)
	if one != two {
		t.Error("the same row classes in different runs hashed differently")
	}
	if other := blockHash(&h, rh, 1, tokens, nil, []ClassRun{{0, 101, ClassFast}, {101, 27, ClassExact}}, allParts); other == one {
		t.Error("a block whose classes meet one row later hashed the same")
	}
}

// Wrong-hit mutations (docs/measurements/kv1-2026-10-13.md, G4). For each part of the key, two requests that differ
// only in that part: the second must take no rows from what the first published. KVMGR_DROP_KEY_PART=<name> drops
// that part from the key for the run, which is how each of these is shown going red.
func TestWrongHit(t *testing.T) {
	const n = 4 * BlockRows
	tokens := make([]int, n+1)
	for i := range tokens {
		tokens[i] = 100 + i%9
	}
	base := Root{Model: "m", Namespace: "key-a", Adapter: 1, Precision: "f16"}
	img := []ImageSpan{{Start: 70, Len: 20, Hash: 0xA11CE}}
	fast := []ClassRun{{0, n + 1, ClassFast}}
	for _, tc := range []struct {
		part   string
		bit    keyPart
		root   Root
		images []ImageSpan
		cold   []ClassRun
		rows   int // the rows the second request may still take: what lies before the first difference
	}{
		{"adapter", partAdapter, Root{Model: "m", Namespace: "key-a", Adapter: 2, Precision: "f16"}, img, fast, 0},
		{"image", partImage, base, []ImageSpan{{Start: 70, Len: 20, Hash: 0xB0B}}, fast, BlockRows},
		{"class", partClass, base, img, []ClassRun{{0, 200, ClassFast}, {200, n + 1 - 200, ClassExact}}, 3 * BlockRows},
		{"precision", partPrecision, Root{Model: "m", Namespace: "key-a", Adapter: 1, Precision: "f32"}, img, fast, 0},
		{"namespace", partNamespace, Root{Model: "m", Namespace: "key-b", Adapter: 1, Precision: "f16"}, img, fast, 0},
	} {
		t.Run(tc.part, func(t *testing.T) {
			m := New(0)
			if os.Getenv("KVMGR_DROP_KEY_PART") == tc.part {
				m.dropKeyPart(tc.bit)
			}
			first, hit := m.Admit(base, tokens, img, fast)
			if hit != 0 {
				t.Fatalf("an empty manager gave a hit of %d", hit)
			}
			if err := first.Append(tokens, ClassFast, img...); err != nil {
				t.Fatal(err)
			}
			first.Publish()
			// The control: the same request again takes everything it may, so a zero below is not an index that never hits.
			if _, again := m.Admit(base, tokens, img, fast); again != n {
				t.Fatalf("the same request took %d rows of the %d published", again, n)
			}
			if _, got := m.Admit(tc.root, tokens, tc.images, tc.cold); got != tc.rows {
				t.Errorf("WRONG HIT: a request that differs in its %s took %d rows from the other's blocks, want %d", tc.part, got, tc.rows)
			}
		})
	}
}

// An image span with no content hash claims nothing. Two requests with the same tokens and such a span at the same
// place may hold different images, so the second takes only the blocks before it, however alike the keys would be.
func TestWrongHit_unclaimedImage(t *testing.T) {
	const n = 4 * BlockRows
	tokens := make([]int, n+1)
	for i := range tokens {
		tokens[i] = 100 + i%9
	}
	root := Root{Model: "m"}
	fast := []ClassRun{{0, n + 1, ClassFast}}
	for _, tc := range []struct {
		name string
		img  []ImageSpan
		want int
	}{
		{"a claimed image is shared whole", []ImageSpan{{Start: 70, Len: 20, Hash: 7}}, n},
		{"an unclaimed image: only the block before it", []ImageSpan{{Start: 70, Len: 20}}, BlockRows},
		{"an unclaimed image in the first block: nothing", []ImageSpan{{Start: 3, Len: 5}}, 0},
		{"an unclaimed image across two blocks: nothing from the first of them on", []ImageSpan{{Start: 120, Len: 20}}, BlockRows},
	} {
		m := New(0)
		first, _ := m.Admit(root, tokens, tc.img, fast)
		if err := first.Append(tokens, ClassFast, tc.img...); err != nil {
			t.Fatal(err)
		}
		first.Publish()
		if _, got := m.Admit(root, tokens, tc.img, fast); got != tc.want {
			t.Errorf("%s: the same request again took %d rows, want %d", tc.name, got, tc.want)
		}
	}
}

// A row whose producer is not named matches nothing: not a stored row of a named class, and not another unnamed one.
func TestWrongHit_unknownClass(t *testing.T) {
	const n = 4 * BlockRows
	tokens := make([]int, n+1)
	for i := range tokens {
		tokens[i] = 100 + i%9
	}
	root := Root{Model: "m"}
	for _, tc := range []struct {
		name   string
		stored []ClassRun
		cold   []ClassRun
		want   int
	}{
		{"both fast: shared", []ClassRun{{0, n + 1, ClassFast}}, []ClassRun{{0, n + 1, ClassFast}}, n},
		{"both unnamed: nothing", nil, nil, 0},
		{"stored unnamed from row 100: only the block before", []ClassRun{{0, 100, ClassFast}}, []ClassRun{{0, n + 1, ClassFast}}, BlockRows},
		{"the request's rows unnamed from row 200: three blocks", []ClassRun{{0, n + 1, ClassFast}}, []ClassRun{{0, 200, ClassFast}}, 3 * BlockRows},
		{"named unknown is unnamed", []ClassRun{{0, n + 1, ClassUnknown}}, []ClassRun{{0, n + 1, ClassUnknown}}, 0},
	} {
		m := New(0)
		first, _ := m.Admit(root, nil, nil, nil)
		first.tokens = append([]int(nil), tokens...)
		first.classes = tc.stored
		for i := 0; i*BlockRows < len(tokens); i++ {
			id, _ := m.alloc(-1, int32(i))
			first.table = append(first.table, id)
		}
		first.Publish()
		if _, got := m.Admit(root, tokens, nil, tc.cold); got != tc.want {
			t.Errorf("%s: took %d rows, want %d", tc.name, got, tc.want)
		}
	}
}

// On a model with sliding-window layers, a fast pass that starts off a block boundary writes rows no other request
// would compute (G2b), so the commit records them with no producer and the index never returns them.
func TestCommit_unalignedFastPassOnAWindowedModel(t *testing.T) {
	const n = 600
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = 7 + i%11
	}
	root := Root{Model: "m"}
	for _, tc := range []struct {
		name     string
		cm       ClassMap
		reuse    int
		wantTail Class
	}{
		{"full attention, a reuse of 100: fast", ClassMap{Floor: 512, MinRows: 16}, 100, ClassFast},
		{"windowed, a reuse of 128 (a block boundary): fast", ClassMap{Floor: 512, MinRows: 16, Windowed: true}, 128, ClassFast},
		{"windowed, a reuse of 100: no producer named", ClassMap{Floor: 512, MinRows: 16, Windowed: true}, 100, ClassUnknown},
		{"windowed, a cold prefill: fast", ClassMap{Floor: 512, MinRows: 16, Windowed: true}, 0, ClassFast},
		{"windowed, an exact-class prompt at an odd start stays exact", ClassMap{Windowed: true}, 100, ClassExact},
	} {
		a := NewSlots(1)
		a.Commit(0, root, append([]int(nil), prompt[:tc.reuse]...), nil, tc.reuse, ClassMap{}) // an earlier turn, no producer named
		a.Acquired(0, tc.reuse, prompt)
		a.Forget(0)
		a.Commit(0, root, append([]int(nil), prompt...), nil, n, tc.cm)
		if got := classAt(a.Seq(0).Classes(), n-1); got != tc.wantTail {
			t.Errorf("%s: the prefilled rows are class %d, want %d", tc.name, got, tc.wantTail)
		}
	}
}
