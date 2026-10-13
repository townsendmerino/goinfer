package kvmgr

import "errors"

// ErrNoBlocks is returned when a sequence needs a block and none is free or evictable.
var ErrNoBlocks = errors.New("kvmgr: no free block, and none cached to evict")

// Manager owns the blocks of one model's KV, their index and their reference counts.
//
// A block is BlockRows consecutive positions of one sequence. It is private while its sequence is still writing it.
// Publish makes a full block immutable and puts it in the index under its chain hash; only published blocks are ever
// held by more than one sequence. A published block that no sequence references stays cached, in least-recently-used
// order, until a new block needs its place.
type Manager struct {
	parts  keyPart
	limit  int // the most blocks that may exist; 0: no limit
	blocks []block
	free   []int32
	index  map[Hash][]int32 // published blocks by chain hash
	// The cached blocks (published, referenced by nobody), oldest first.
	lruHead, lruTail int32
	cached           int
	h                hasher
}

type block struct {
	hash       Hash
	refs       int32
	published  bool
	cached     bool
	inUse      bool
	slot, at   int32 // slot placement: the slot it lives in and its logical index there; -1, -1 in a pool
	prev, next int32
}

// Copy is rows the caller must copy on the device before it uses the destination block: the first Rows positions of
// block Src into block Dst.
type Copy struct {
	Src, Dst int32
	Rows     int
}

// New returns a manager for at most limit blocks (0: unlimited, as under slot placement, where the slots bound it).
func New(limit int) *Manager {
	return &Manager{parts: allParts, limit: limit, index: map[Hash][]int32{}, lruHead: -1, lruTail: -1}
}

// Seq is one conversation's KV: its tokens, the image spans and producer classes that are part of their identity,
// and the table of blocks that hold its rows.
type Seq struct {
	m        *Manager
	root     Root
	rootHash Hash
	tokens   []int
	images   []ImageSpan
	classes  []ClassRun
	table    []int32
	hashes   []Hash // the chain hash of each published block, in order
	// pinned: the sequence holds a reference on each of its published blocks. A pool sequence is pinned from Admit
	// to Release. A slot's idle sequence is not: its blocks stay where they are until the slot is given away.
	pinned   bool
	slot     int32
	released bool
}

// Len is the number of positions the sequence holds.
func (s *Seq) Len() int { return len(s.tokens) }

// Tokens is the sequence's token ids. The caller must not modify them.
func (s *Seq) Tokens() []int { return s.tokens }

// Images is the sequence's image spans, ascending. The caller must not modify them.
func (s *Seq) Images() []ImageSpan { return s.images }

// Classes is which class wrote each row, as runs. The caller must not modify them.
func (s *Seq) Classes() []ClassRun { return s.classes }

// Root is what the sequence was built under.
func (s *Seq) Root() Root { return s.root }

// Table is the sequence's blocks in logical order. The caller must not modify it.
func (s *Seq) Table() []int32 { return s.table }

// Published is how many of the sequence's leading blocks are published.
func (s *Seq) Published() int { return len(s.hashes) }

func (m *Manager) alloc(slot, at int32) (int32, error) {
	var id int32
	switch {
	case len(m.free) > 0:
		id = m.free[len(m.free)-1]
		m.free = m.free[:len(m.free)-1]
	case m.limit == 0 || len(m.blocks) < m.limit:
		id = int32(len(m.blocks))
		m.blocks = append(m.blocks, block{})
	case m.lruHead >= 0:
		id = m.lruHead
		m.evict(id)
		m.free = m.free[:len(m.free)-1] // evict returned it to the free list; take it straight back
	default:
		return -1, ErrNoBlocks
	}
	m.blocks[id] = block{refs: 1, inUse: true, slot: slot, at: at, prev: -1, next: -1}
	return id, nil
}

func (m *Manager) release(id int32) {
	m.blocks[id] = block{prev: -1, next: -1}
	m.free = append(m.free, id)
}

// evict drops a cached block: out of the LRU, out of the index, back to the free list.
func (m *Manager) evict(id int32) {
	b := &m.blocks[id]
	m.lruRemove(id)
	m.unindex(id, b.hash)
	m.release(id)
}

func (m *Manager) unindex(id int32, h Hash) {
	ids := m.index[h]
	for i, v := range ids {
		if v == id {
			ids = append(ids[:i], ids[i+1:]...)
			break
		}
	}
	if len(ids) == 0 {
		delete(m.index, h)
	} else {
		m.index[h] = ids
	}
}

func (m *Manager) lruPush(id int32) {
	b := &m.blocks[id]
	b.cached, b.prev, b.next = true, m.lruTail, -1
	if m.lruTail >= 0 {
		m.blocks[m.lruTail].next = id
	} else {
		m.lruHead = id
	}
	m.lruTail = id
	m.cached++
}

func (m *Manager) lruRemove(id int32) {
	b := &m.blocks[id]
	if !b.cached {
		return
	}
	if b.prev >= 0 {
		m.blocks[b.prev].next = b.next
	} else {
		m.lruHead = b.next
	}
	if b.next >= 0 {
		m.blocks[b.next].prev = b.prev
	} else {
		m.lruTail = b.prev
	}
	b.cached, b.prev, b.next = false, -1, -1
	m.cached--
}

// ref takes a reference on a published block, pulling it out of the cache if it was idle.
func (m *Manager) ref(id int32) {
	m.lruRemove(id)
	m.blocks[id].refs++
}

// unref drops a reference. A published block nobody references becomes cached; a private one is freed.
func (m *Manager) unref(id int32) {
	b := &m.blocks[id]
	b.refs--
	if b.refs > 0 {
		return
	}
	if b.published {
		m.lruPush(id)
		return
	}
	m.release(id)
}

// Cached is the number of published blocks no sequence references.
func (m *Manager) Cached() int { return m.cached }

// Blocks is the number of blocks in use, private, referenced or cached.
func (m *Manager) Blocks() int { return len(m.blocks) - len(m.free) }

// hitBlocks is how many leading blocks of a request are published under root, and their ids: the longest chain of
// index hits from block 0, capped so that the last token is always recomputed (its logits seed decode) and so that
// the boundary never falls inside an image span. cold is the class the requester's own cold prefill would write each
// row with, so a block matches only when the rows it holds are the ones the requester would have computed.
func (m *Manager) hitBlocks(rh Hash, tokens []int, images []ImageSpan, cold []ClassRun) ([]int32, []Hash) {
	limit := 0
	if len(tokens) > 0 {
		limit = (len(tokens) - 1) / BlockRows
	}
	var ids []int32
	var hashes []Hash
	prev := rh
	for i := 0; i < limit; i++ {
		h := blockHash(&m.h, prev, i, tokens, images, cold, m.parts)
		c := m.index[h]
		if len(c) == 0 {
			break
		}
		ids, hashes, prev = append(ids, c[0]), append(hashes, h), h
	}
	// Back the boundary off any image span it cuts. Each step moves it down, so this ends.
	for moved := true; moved; {
		moved = false
		edge := len(ids) * BlockRows
		for _, s := range images {
			if s.Start < edge && s.end() > edge {
				ids, hashes = ids[:s.Start/BlockRows], hashes[:s.Start/BlockRows]
				moved = true
				break
			}
		}
	}
	return ids, hashes
}

// Lookup is how many leading rows of a request are published under root, without taking them.
func (m *Manager) Lookup(root Root, tokens []int, images []ImageSpan, cold []ClassRun) int {
	ids, _ := m.hitBlocks(rootHash(root, m.parts), tokens, images, cold)
	return len(ids) * BlockRows
}

// Admit starts a sequence for a request. It shares the longest published prefix of the request (Lookup's rule) and
// returns how many rows that is; the caller computes the rest and reports it with Append. cold is the class the
// requester's cold prefill would write each row with.
func (m *Manager) Admit(root Root, tokens []int, images []ImageSpan, cold []ClassRun) (*Seq, int) {
	s := &Seq{m: m, root: root, rootHash: rootHash(root, m.parts), pinned: true, slot: -1}
	ids, hashes := m.hitBlocks(s.rootHash, tokens, images, cold)
	for _, id := range ids {
		m.ref(id)
	}
	hit := len(ids) * BlockRows
	s.table, s.hashes = ids, hashes
	s.tokens = append([]int(nil), tokens[:hit]...)
	for _, sp := range images {
		if sp.end() <= hit {
			s.images = append(s.images, sp)
		}
	}
	s.classes = clipClasses(cold, hit)
	return s, hit
}

// Append records rows the sequence has written: tokens after its current end, all by class, with any image spans
// that lie inside them (absolute positions). It takes a new private block each time the tail fills.
func (s *Seq) Append(tokens []int, class Class, images ...ImageSpan) error {
	start := len(s.tokens)
	for i := range tokens {
		if (start+i)%BlockRows == 0 {
			id, err := s.m.alloc(s.slot, int32(len(s.table)))
			if err != nil {
				s.tokens = append(s.tokens, tokens[:i]...)
				s.classes = appendClass(s.classes, ClassRun{Start: start, Len: i, Class: class})
				return err
			}
			s.table = append(s.table, id)
		}
	}
	s.tokens = append(s.tokens, tokens...)
	s.classes = appendClass(s.classes, ClassRun{Start: start, Len: len(tokens), Class: class})
	s.images = append(s.images, images...)
	return nil
}

// Publish makes every full block the sequence has finished immutable and indexes it. Call it only when the device
// work that wrote those rows has completed: a block published early could be read by another sequence before its
// rows exist.
func (s *Seq) Publish() {
	m := s.m
	prev := s.rootHash
	if n := len(s.hashes); n > 0 {
		prev = s.hashes[n-1]
	}
	for i := len(s.hashes); (i+1)*BlockRows <= len(s.tokens); i++ {
		h := blockHash(&m.h, prev, i, s.tokens, s.images, s.classes, m.parts)
		b := &m.blocks[s.table[i]]
		b.published, b.hash = true, h
		m.index[h] = append(m.index[h], s.table[i])
		s.hashes = append(s.hashes, h)
		prev = h
	}
}

// Truncate cuts the sequence back to p positions (a speculative rollback, an edit) and returns where it actually
// cut and the copies that requires. A cut inside an image span moves down to the span's start, because half an
// image's rows are not a prefix of anything. Blocks wholly past the cut are released. A cut inside a published block
// gives the sequence a private copy of the rows it keeps, since the published block is not its to overwrite.
func (s *Seq) Truncate(p int) (int, []Copy, error) {
	m := s.m
	p = max(0, min(p, len(s.tokens)))
	for _, sp := range s.images {
		if sp.Start < p && sp.end() > p {
			p = sp.Start
		}
	}
	keep := (p + BlockRows - 1) / BlockRows
	full := p / BlockRows
	var copies []Copy
	if full < len(s.hashes) && p%BlockRows != 0 {
		// The tail block is published: take a private copy of the rows kept, before anything is given up, so a
		// failure leaves the sequence as it was.
		id, err := m.alloc(s.slot, int32(full))
		if err != nil {
			return len(s.tokens), nil, err
		}
		copies = append(copies, Copy{Src: s.table[full], Dst: id, Rows: p % BlockRows})
		for i := len(s.table) - 1; i >= full; i-- {
			s.dropBlock(i)
		}
		s.table = append(s.table[:full], id)
	} else {
		for i := len(s.table) - 1; i >= keep; i-- {
			s.dropBlock(i)
		}
		s.table = s.table[:keep]
	}
	if full < len(s.hashes) {
		s.hashes = s.hashes[:full]
	}
	s.tokens = s.tokens[:p]
	imgs := s.images[:0]
	for _, sp := range s.images {
		if sp.end() <= p {
			imgs = append(imgs, sp)
		}
	}
	s.images = imgs
	s.classes = clipClasses(s.classes, p)
	return p, copies, nil
}

// dropBlock gives up the sequence's hold on table entry i: a reference when it is pinned and the block is
// published, ownership when the block is private.
func (s *Seq) dropBlock(i int) {
	id := s.table[i]
	if i < len(s.hashes) {
		if s.pinned {
			s.m.unref(id)
		}
		return
	}
	s.m.unref(id) // private: the one reference is the owner's
}

// Fork returns a second sequence with the same content. Published blocks are shared; every block still private
// (the tail, and full blocks not yet published) is copied, and the copies are returned for the caller to perform.
func (s *Seq) Fork() (*Seq, []Copy, error) {
	m := s.m
	t := &Seq{m: m, root: s.root, rootHash: s.rootHash, pinned: true, slot: -1,
		tokens: append([]int(nil), s.tokens...), images: append([]ImageSpan(nil), s.images...),
		classes: append([]ClassRun(nil), s.classes...), hashes: append([]Hash(nil), s.hashes...)}
	var copies []Copy
	for i, id := range s.table {
		if i < len(s.hashes) {
			m.ref(id)
			t.table = append(t.table, id)
			continue
		}
		nb, err := m.alloc(-1, int32(i))
		if err != nil {
			t.Release()
			return nil, nil, err
		}
		rows := min(len(s.tokens)-i*BlockRows, BlockRows)
		copies = append(copies, Copy{Src: id, Dst: nb, Rows: rows})
		t.table = append(t.table, nb)
	}
	return t, copies, nil
}

// Release ends the sequence. Its published blocks stay cached once nobody else references them, the last block
// first into the cache so that a chain is evicted from its end; its private blocks are freed.
func (s *Seq) Release() {
	if s.released {
		return
	}
	for i := len(s.table) - 1; i >= 0; i-- {
		s.dropBlock(i)
	}
	s.table, s.hashes, s.released = nil, nil, true
}

// EvictOldest drops the least recently used cached block and reports whether there was one.
func (m *Manager) EvictOldest() bool {
	if m.lruHead < 0 {
		return false
	}
	m.evict(m.lruHead)
	return true
}
