package kvmgr

import (
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"testing"
)

// The oracle property test (docs/measurements/kv1-2026-10-13.md, G3): a seeded random walk of admit, append,
// publish, truncate, fork, release and evict against a plain model that holds each sequence as an array of rows,
// with a fake device that holds what each block's rows are. After every operation the manager's tables, reference
// counts, index and cache are checked against the model.

// orow is one position as the model sees it: everything that decides its KV bytes. The device holds these, so a block
// handed to a request that would have computed different rows is caught by comparing them.
type orow struct {
	tok   int
	class Class
	root  Root
	img   uint64 // the content hash of the image span covering the row, 0 for none
	off   int    // the row's offset inside that span
}

// stamp fills in the identity of rows [from, len) of q from its root and image spans.
func (q *oseq) stamp(from int) {
	for p := from; p < len(q.rows); p++ {
		r := &q.rows[p]
		r.root, r.img, r.off = q.root, 0, 0
		for _, sp := range q.images {
			if p >= sp.Start && p < sp.end() {
				r.img, r.off = sp.Hash, p-sp.Start
			}
		}
	}
}

// oseq is the model's sequence: a plain array of rows, its image spans, and the manager's sequence beside it.
type oseq struct {
	root   Root
	rows   []orow
	images []ImageSpan
	s      *Seq
}

func (o *oseq) tokens() []int {
	t := make([]int, len(o.rows))
	for i, r := range o.rows {
		t[i] = r.tok
	}
	return t
}

func (o *oseq) classes() []ClassRun {
	var runs []ClassRun
	for i, r := range o.rows {
		runs = appendClass(runs, ClassRun{Start: i, Len: 1, Class: r.class})
	}
	return runs
}

// opub is what the model recorded when a block was published: the whole prefix that its chain hash stands for.
type opub struct {
	root   Root
	rows   []orow // rows [0, (i+1)*BlockRows)
	images []ImageSpan
}

type oracle struct {
	t     *testing.T
	rng   *rand.Rand
	m     *Manager
	seqs  []*oseq
	dev   map[int32][]orow // the device: each block's rows
	pub   map[int32]opub
	hist  []*oseq // snapshots of earlier sequences, to build requests that share a prefix
	roots []Root
	op    int
	seed  int64
}

func (o *oracle) failf(format string, args ...any) {
	o.t.Helper()
	o.t.Fatalf("seed %d, operation %d: %s", o.seed, o.op, fmt.Sprintf(format, args...))
}

// write puts rows [from, len) of q on the device through its table, as a forward pass would.
func (o *oracle) write(q *oseq, from int) {
	for p := from; p < len(q.rows); p++ {
		id := q.s.Table()[p/BlockRows]
		blk := o.dev[id]
		for len(blk) <= p%BlockRows {
			blk = append(blk, orow{})
		}
		blk[p%BlockRows] = q.rows[p]
		o.dev[id] = blk
	}
}

func (o *oracle) applyCopies(copies []Copy) {
	for _, c := range copies {
		o.dev[c.Dst] = append([]orow(nil), o.dev[c.Src][:c.Rows]...)
	}
}

func (o *oracle) randRows(n int) []orow {
	rows := make([]orow, n)
	for i := range rows {
		rows[i] = orow{tok: o.rng.Intn(5), class: ClassExact}
	}
	return rows
}

// request builds rows for a new sequence: usually a prefix of something seen before (so hits happen) plus a random
// tail, sometimes with one class changed or under another root (so near-misses happen).
func (o *oracle) request() *oseq {
	q := &oseq{root: o.roots[o.rng.Intn(len(o.roots))]}
	if len(o.hist) > 0 && o.rng.Intn(5) != 0 {
		h := o.hist[o.rng.Intn(len(o.hist))]
		k := o.rng.Intn(len(h.rows) + 1)
		q.rows = append(q.rows, h.rows[:k]...)
		for _, sp := range h.images {
			if sp.end() <= k {
				q.images = append(q.images, sp)
			}
		}
		if o.rng.Intn(3) != 0 {
			q.root = h.root
		}
	}
	tail := o.rng.Intn(3 * BlockRows)
	start := len(q.rows)
	q.rows = append(q.rows, o.randRows(tail)...)
	cls := []Class{ClassExact, ClassFast, ClassUnknown}[o.rng.Intn(3)]
	for i := start; i < len(q.rows); i++ {
		q.rows[i].class = cls
	}
	if tail > 8 && o.rng.Intn(4) == 0 { // an image span inside the new rows
		at := start + o.rng.Intn(tail-8)
		q.images = append(q.images, ImageSpan{Start: at, Len: 1 + o.rng.Intn(min(8+BlockRows, len(q.rows)-at)), Hash: uint64(1 + o.rng.Intn(3))})
	}
	q.stamp(0)
	return q
}

func (o *oracle) remember(q *oseq) {
	o.hist = append(o.hist, &oseq{root: q.root, rows: slices.Clone(q.rows), images: slices.Clone(q.images)})
	if len(o.hist) > 24 {
		o.hist = o.hist[1:]
	}
}

func (o *oracle) step() {
	live := len(o.seqs)
	switch k := o.rng.Intn(100); {
	case k < 22 || live == 0: // admit
		want := o.request()
		s, hit := o.m.Admit(want.root, want.tokens(), want.images, want.classes())
		if hit%BlockRows != 0 || (len(want.rows) > 0 && hit > len(want.rows)-1) || hit > len(want.rows) {
			o.failf("admit of %d rows returned a hit of %d", len(want.rows), hit)
		}
		for _, sp := range want.images {
			if sp.Start < hit && sp.end() > hit {
				o.failf("a hit of %d rows ends inside the image span %+v", hit, sp)
			}
		}
		q := &oseq{root: want.root, rows: slices.Clone(want.rows[:hit]), s: s}
		for _, sp := range want.images {
			if sp.end() <= hit {
				q.images = append(q.images, sp)
			}
		}
		// The rows it was given must be the rows this request would have computed: the wrong-hit check.
		for p := 0; p < hit; p++ {
			if got := o.dev[s.Table()[p/BlockRows]][p%BlockRows]; got != want.rows[p] {
				o.failf("WRONG HIT: row %d of the shared prefix is %+v, the request's own is %+v", p, got, want.rows[p])
			}
		}
		o.seqs = append(o.seqs, q)
		// Sometimes compute the rest at once, as a prefill does.
		if o.rng.Intn(2) == 0 {
			o.appendRows(q, want.rows[hit:], want.images)
		}
	case k < 50: // append
		q := o.seqs[o.rng.Intn(live)]
		n := 1 + o.rng.Intn(BlockRows+BlockRows/2)
		rows := o.randRows(n)
		cls := []Class{ClassExact, ClassFast}[o.rng.Intn(2)]
		for i := range rows {
			rows[i].class = cls
		}
		o.appendRows(q, rows, nil)
	case k < 68: // publish
		q := o.seqs[o.rng.Intn(live)]
		before := q.s.Published()
		q.s.Publish()
		for i := before; i < q.s.Published(); i++ {
			o.pub[q.s.Table()[i]] = opub{root: q.root, rows: slices.Clone(q.rows[:(i+1)*BlockRows]), images: slices.Clone(q.images)}
		}
		o.remember(q)
	case k < 78: // truncate
		q := o.seqs[o.rng.Intn(live)]
		p := o.rng.Intn(len(q.rows) + 1)
		want := p
		for _, sp := range q.images {
			if sp.Start < want && sp.end() > want {
				want = sp.Start
			}
		}
		at, copies, err := q.s.Truncate(p)
		if err != nil {
			if err != ErrNoBlocks || !o.full() {
				o.failf("truncate to %d: %v with %d blocks of %d in use and %d cached", p, err, o.m.Blocks(), o.m.limit, o.m.Cached())
			}
			return
		}
		if at != want {
			o.failf("truncate to %d cut at %d, the model says %d", p, at, want)
		}
		o.applyCopies(copies)
		q.rows = q.rows[:at]
		imgs := q.images[:0]
		for _, sp := range q.images {
			if sp.end() <= at {
				imgs = append(imgs, sp)
			}
		}
		q.images = imgs
	case k < 84: // fork
		q := o.seqs[o.rng.Intn(live)]
		// A fork needs a block for each of q's private blocks, and takes none when it cannot have them all.
		need := len(q.s.Table()) - q.s.Published()
		avail := o.m.limit - o.m.Blocks() + o.m.Cached()
		blocksBefore := o.m.Blocks()
		t, copies, err := q.s.Fork()
		if (err != nil) != (need > avail) {
			o.failf("fork needing %d blocks with %d available: err = %v", need, avail, err)
		}
		if err != nil {
			if err != ErrNoBlocks || o.m.Blocks() > blocksBefore {
				o.failf("a failed fork returned %v and left %d blocks in use, %d before", err, o.m.Blocks(), blocksBefore)
			}
			return
		}
		o.applyCopies(copies)
		o.seqs = append(o.seqs, &oseq{root: q.root, rows: slices.Clone(q.rows), images: slices.Clone(q.images), s: t})
	case k < 96: // release
		i := o.rng.Intn(live)
		o.seqs[i].s.Release()
		o.seqs = append(o.seqs[:i], o.seqs[i+1:]...)
	default: // evict
		had := o.m.Cached() > 0
		if o.m.EvictOldest() != had {
			o.failf("EvictOldest disagreed with Cached() = %v", had)
		}
	}
}

// full reports whether the pool really has nothing to give: every block in use and none cached.
func (o *oracle) full() bool {
	return o.m.limit > 0 && o.m.Blocks() >= o.m.limit && o.m.Cached() == 0
}

func (o *oracle) appendRows(q *oseq, rows []orow, images []ImageSpan) {
	// Append takes one class per call, so feed it run by run.
	for len(rows) > 0 {
		n := 1
		for n < len(rows) && rows[n].class == rows[0].class {
			n++
		}
		start := len(q.rows)
		toks := make([]int, n)
		for i := range toks {
			toks[i] = rows[i].tok
		}
		var spans []ImageSpan
		for _, sp := range images {
			if sp.Start >= start && sp.Start < start+n {
				spans = append(spans, sp)
			}
		}
		err := q.s.Append(toks, rows[0].class, spans...)
		got := q.s.Len() - start
		if err != nil && (err != ErrNoBlocks || !o.full() || got >= n) {
			o.failf("append of %d rows: %v after %d, with %d blocks of %d in use and %d cached", n, err, got, o.m.Blocks(), o.m.limit, o.m.Cached())
		}
		q.rows = append(q.rows, rows[:got]...)
		if err == nil {
			q.images = append(q.images, spans...)
		}
		q.stamp(start)
		o.write(q, start)
		if err != nil {
			return
		}
		rows = rows[n:]
	}
}

// check compares everything the manager holds with the model.
func (o *oracle) check() {
	m := o.m
	wantRefs := map[int32]int32{}
	owner := map[int32]int{}
	for qi, q := range o.seqs {
		if q.s.Len() != len(q.rows) {
			o.failf("sequence %d holds %d rows, the model %d", qi, q.s.Len(), len(q.rows))
		}
		if !slices.Equal(q.s.Tokens(), q.tokens()) {
			o.failf("sequence %d: tokens differ from the model's", qi)
		}
		tab := q.s.Table()
		if want := (len(q.rows) + BlockRows - 1) / BlockRows; len(tab) != want {
			o.failf("sequence %d: %d rows in a table of %d blocks, want %d", qi, len(q.rows), len(tab), want)
		}
		if q.s.Published() > len(q.rows)/BlockRows {
			o.failf("sequence %d: %d blocks published, only %d are full", qi, q.s.Published(), len(q.rows)/BlockRows)
		}
		for p, r := range q.rows {
			blk := o.dev[tab[p/BlockRows]]
			if p%BlockRows >= len(blk) || blk[p%BlockRows] != r {
				o.failf("sequence %d: row %d read through its table is not the row it wrote", qi, p)
			}
		}
		for i, id := range tab {
			wantRefs[id]++
			if i >= q.s.Published() {
				if prev, dup := owner[id]; dup {
					o.failf("private block %d is in the tables of sequences %d and %d", id, prev, qi)
				}
				owner[id] = qi
				if m.blocks[id].published {
					o.failf("sequence %d: block %d at table index %d is published but the sequence has published only %d", qi, id, i, q.s.Published())
				}
			} else if !m.blocks[id].published {
				o.failf("sequence %d: block %d at table index %d should be published", qi, id, i)
			}
		}
	}
	inFree := map[int32]bool{}
	for _, id := range m.free {
		if inFree[id] {
			o.failf("block %d is on the free list twice", id)
		}
		inFree[id] = true
	}
	indexed := map[int32]Hash{}
	for h, ids := range m.index {
		if len(ids) == 0 {
			o.failf("the index holds an empty entry")
		}
		for _, id := range ids {
			if _, dup := indexed[id]; dup {
				o.failf("block %d is indexed twice", id)
			}
			indexed[id] = h
		}
	}
	cached := map[int32]bool{}
	prev := int32(-1)
	for id := m.lruHead; id >= 0; id = m.blocks[id].next {
		if cached[id] {
			o.failf("the LRU list loops at block %d", id)
		}
		cached[id] = true
		if m.blocks[id].prev != prev {
			o.failf("LRU back link of block %d is %d, want %d", id, m.blocks[id].prev, prev)
		}
		prev = id
	}
	if prev != m.lruTail || len(cached) != m.cached {
		o.failf("LRU tail %d (walked to %d), %d blocks walked, Cached() says %d", m.lruTail, prev, len(cached), m.cached)
	}
	if m.limit > 0 && m.Blocks() > m.limit {
		o.failf("%d blocks in use, the pool has %d", m.Blocks(), m.limit)
	}
	hs := hasher{}
	for i := range m.blocks {
		id := int32(i)
		b := &m.blocks[id]
		if inFree[id] {
			if b.inUse || wantRefs[id] != 0 || b.published {
				o.failf("block %d is free and also in use (refs wanted %d, published %v)", id, wantRefs[id], b.published)
			}
			if _, ok := indexed[id]; ok {
				o.failf("free block %d is still in the index", id)
			}
			continue
		}
		if !b.inUse {
			o.failf("block %d is neither in use nor free", id)
		}
		if b.refs != wantRefs[id] {
			o.failf("block %d has %d references, %d sequences hold it", id, b.refs, wantRefs[id])
		}
		// The index holds exactly the published blocks; the cache exactly the published blocks nobody references.
		if _, ok := indexed[id]; ok != b.published {
			o.failf("block %d: published %v, in the index %v", id, b.published, ok)
		}
		if want := b.published && b.refs == 0; cached[id] != want || b.cached != want {
			o.failf("block %d: published %v with %d references, in the LRU %v (flag %v)", id, b.published, b.refs, cached[id], b.cached)
		}
		if !b.published {
			if b.refs != 1 {
				o.failf("private block %d has %d references", id, b.refs)
			}
			continue
		}
		// A published block: immutable, full, and its hash is the chain recomputed from block 0.
		rec, ok := o.pub[id]
		if !ok {
			o.failf("block %d is published and the model never published it: an unpublished row is reachable", id)
		}
		n := len(rec.rows)
		if n%BlockRows != 0 || len(o.dev[id]) != BlockRows || !slices.Equal(o.dev[id], rec.rows[n-BlockRows:]) {
			o.failf("published block %d no longer holds the rows it was published with", id)
		}
		toks := make([]int, n)
		var runs []ClassRun
		for p, r := range rec.rows {
			toks[p] = r.tok
			runs = appendClass(runs, ClassRun{Start: p, Len: 1, Class: r.class})
		}
		h := rootHash(rec.root, m.parts)
		for bi := 0; bi < n/BlockRows; bi++ {
			h = blockHash(&hs, h, bi, toks, rec.images, runs, m.parts)
			if bi == n/BlockRows-2 {
				if _, ok := m.index[h]; !ok {
					o.failf("published block %d is indexed and its parent's hash is not: an unreachable block", id)
				}
			}
		}
		if h != b.hash || indexed[id] != h {
			o.failf("published block %d: its hash is not the chain recomputed from its rows", id)
		}
	}
	for id := range o.pub {
		if inFree[id] || !m.blocks[id].published {
			delete(o.pub, id)
		}
	}
}

func oracleOps() int {
	if v, err := strconv.Atoi(os.Getenv("KVMGR_ORACLE_OPS")); err == nil && v > 0 {
		return v
	}
	return 20000
}

func TestOracle_pool(t *testing.T) {
	for seed := int64(1); seed <= 8; seed++ {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			o := &oracle{t: t, rng: rand.New(rand.NewSource(seed)), seed: seed, dev: map[int32][]orow{}, pub: map[int32]opub{},
				m: New([]int{12, 24, 48, 96}[seed%4])}
			o.roots = []Root{
				{Model: "m", Precision: "f16"},
				{Model: "m", Precision: "f16", Adapter: 7},
				{Model: "m", Precision: "f16", Namespace: "key-b"},
			}
			hits := 0
			for o.op = 0; o.op < oracleOps(); o.op++ {
				before := len(o.seqs)
				o.step()
				if len(o.seqs) > before && o.seqs[len(o.seqs)-1].s.Published() > 0 {
					hits++
				}
				o.check()
			}
			if hits == 0 {
				t.Fatalf("seed %d: no admit or fork ever shared a published block; the walk is not exercising sharing", seed)
			}
		})
	}
}
