package kvmgr

import "slices"

// The slot adapter: the manager over a backend that still stores KV as one fixed reservation per slot. Logical block
// i of the sequence in slot s lives at rows [i*BlockRows, (i+1)*BlockRows) of slot s, so nothing in a kernel or an
// allocation changes. What the adapter cannot do is let two sequences hold one block: a slot's rows are overwritten
// by whoever uses the slot next. So a sequence is reused only by continuing it in its own slot, at token
// granularity, and its published blocks are cached (referenced by nobody) for as long as the slot keeps them.

// Slots is the manager's view of a backend with n KV slots.
type Slots struct {
	m    *Manager
	seqs []*Seq
	uses []uint64
	pend []pending
	live []bool
	cur  int
	tick uint64
}

// pending is what Acquired recorded for a slot, for the Commit that ends the same generation.
type pending struct {
	ok        bool
	reuse     int
	promptLen int
	sum       uint64     // a fingerprint of the prompt, so a Commit of another prompt does not take these classes
	prefix    []ClassRun // the classes of the reused rows, read before the slot was forgotten
	first     int        // the reused sequence's first prompt length (Seq.first), 0 when none was reused
}

// NewSlots returns the adapter for n slots (at least one).
func NewSlots(n int) *Slots {
	n = max(n, 1)
	return &Slots{m: New(0), seqs: make([]*Seq, n), uses: make([]uint64, n), pend: make([]pending, n), live: make([]bool, n)}
}

// Manager is the block manager under the adapter.
func (a *Slots) Manager() *Manager { return a.m }

// Count is the number of slots.
func (a *Slots) Count() int { return len(a.seqs) }

// Resize sets the number of slots. When it changes, the bound slot's sequence becomes slot 0 and every other slot is
// emptied: their rows belong to a layout that no longer exists.
func (a *Slots) Resize(n int) {
	n = max(n, 1)
	if n == len(a.seqs) {
		return
	}
	keep, kept, live := a.seqs[a.cur], a.pend[a.cur], a.live[a.cur]
	for i := range a.seqs {
		if i != a.cur {
			a.drop(i)
		}
	}
	a.seqs, a.uses, a.pend, a.live, a.cur = make([]*Seq, n), make([]uint64, n), make([]pending, n), make([]bool, n), 0
	a.seqs[0], a.pend[0], a.live[0] = keep, kept, live
	if keep != nil {
		keep.slot = 0
		for i, id := range keep.table {
			a.m.blocks[id].slot, a.m.blocks[id].at = 0, int32(i)
		}
	}
}

// Bound is the slot the backend currently reads and writes.
func (a *Slots) Bound() int { return a.cur }

// SetBound records that the backend bound slot.
func (a *Slots) SetBound(slot int) { a.cur = slot }

// Touch marks slot as the most recently used, and its cached blocks with it.
func (a *Slots) Touch(slot int) {
	a.tick++
	a.uses[slot] = a.tick
	if s := a.seqs[slot]; s != nil {
		for i := len(s.hashes) - 1; i >= 0; i-- {
			if a.m.blocks[s.table[i]].cached {
				a.m.lruRemove(s.table[i])
				a.m.lruPush(s.table[i])
			}
		}
	}
}

// Use is slot's last-use tick (0: never used).
func (a *Slots) Use(slot int) uint64 { return a.uses[slot] }

// Seq is the sequence slot holds, or nil when it holds nothing known.
func (a *Slots) Seq(slot int) *Seq { return a.seqs[slot] }

// Len is how many positions slot holds.
func (a *Slots) Len(slot int) int {
	if s := a.seqs[slot]; s != nil {
		return len(s.tokens)
	}
	return 0
}

// SlotRow is where the adapter places logical block i of the sequence in slot: the slot itself, at row i*BlockRows.
func SlotRow(slot, block int) (s, row int) { return slot, block * BlockRows }

// Request is a prompt asking what it may reuse.
type Request struct {
	Root   Root
	Tokens []int
	Images []ImageSpan
	// Checkpoint: the caller restores a recurrent sequence's checkpoint before it prefills (KV8a). A recurrent request
	// that is not a strict extension may then score the checkpoint's position (scoreRecurrent).
	Checkpoint bool
}

// Score is how many leading tokens of req the sequence in slot already holds and req may skip. The rules:
//
//   - the match is on token ids, never text;
//   - the roots must be equal: the same model, namespace, adapter and KV precision;
//   - at least one token is always left to compute, because its logits seed decode;
//   - an image span is taken whole or not at all, and only when the request claims the same image at the same
//     place: image placeholder ids are the same for every image, so ids alone prove nothing about one;
//   - a recurrent model continues a strict extension of the whole sequence from where its state stopped. A request that
//     is not one may restart from the sequence's checkpoint instead, when the caller restores it (Request.Checkpoint):
//     scoreRecurrent.
//
// The producer class is not compared here: a sequence continuing in its own slot keeps the rows it wrote, as
// resident reuse always has (the task doc's §5.6, property 3). DeclineShortLead is the one class rule on this path.
func (a *Slots) Score(slot int, req Request, recurrent bool) int {
	s := a.seqs[slot]
	if s == nil || len(s.tokens) == 0 || len(req.Tokens) == 0 {
		return 0
	}
	if rootHash(req.Root, a.m.parts) != s.rootHash {
		return 0
	}
	if recurrent {
		return scoreRecurrent(s, req)
	}
	n := min(len(req.Tokens)-1, len(s.tokens))
	i, bi := 0, 0 // bi: the next image span of s that could start at or after i
	for i < n {
		for bi < len(s.images) && s.images[bi].end() <= i {
			bi++
		}
		if bi < len(s.images) && s.images[bi].Start == i {
			sp := s.images[bi]
			if c, ok := claimAt(req.Images, sp.Start); ok && c.Len == sp.Len && sp.Hash != 0 && c.Hash == sp.Hash && sp.end() <= n {
				i = sp.end() // the whole span verified: past it in one step, never a stop inside it
				bi++
				continue
			}
			return i // another image, no claim, or a span that does not fit under the cap: stop exactly at its start
		}
		if s.tokens[i] != req.Tokens[i] {
			return i
		}
		i++
	}
	return i
}

// scoreRecurrent is Score for a recurrent model. It returns the sequence's length for a strict extension of the whole
// sequence, which continues from the live state. Otherwise, under Request.Checkpoint, it returns the checkpoint's
// position c when the request matches the sequence through c and carries a token past it, and neither side has an
// image span; else 0. A divergence before c, a re-send of the prompt that ended at c, and any image stay cold.
func scoreRecurrent(s *Seq, req Request) int {
	n := len(s.tokens)
	if len(req.Tokens) <= n || !slices.Equal(s.tokens, req.Tokens[:n]) {
		c := s.ckpt
		if !req.Checkpoint || c <= 0 || len(req.Tokens) <= c || len(s.images) > 0 || len(req.Images) > 0 ||
			!slices.Equal(s.tokens[:c], req.Tokens[:c]) {
			return 0
		}
		return c
	}
	// The spans held and the request's claims must be the same set over the reused rows: the state cannot be
	// rewound to an image boundary to repair a wrong guess.
	for _, sp := range s.images {
		c, ok := claimAt(req.Images, sp.Start)
		if !ok || sp.Hash == 0 || c.Hash != sp.Hash || c.Len != sp.Len {
			return 0
		}
	}
	for _, c := range req.Images {
		if c.Start >= n {
			continue // a new image past the reused rows is ordinary new work
		}
		if c.end() > n {
			return 0 // it straddles the boundary: half of it would be reused
		}
		held := false
		for _, sp := range s.images {
			if sp.Start == c.Start && sp.Len == c.Len {
				held = true
				break
			}
		}
		if !held {
			return 0
		}
	}
	return n
}

func claimAt(images []ImageSpan, pos int) (ImageSpan, bool) {
	for _, c := range images {
		if c.Start == pos {
			return c, true
		}
	}
	return ImageSpan{}, false
}

// PickSlot is PickSlotWithin with an allowance of one block, for a model without window groups.
func PickSlot(scores, lens []int, uses []uint64, busy []bool) int {
	return PickSlotWithin(scores, lens, uses, busy, BlockRows)
}

// FirstPrompt is the length of the first prompt of the conversation slot holds (0: none known). A request that keeps
// all of it but its last token is that conversation rolling back its own history (PickSlotFor).
func (a *Slots) FirstPrompt(slot int) int {
	if s := a.seqs[slot]; s != nil {
		return s.first
	}
	return 0
}

// PickSlotFor is PickSlotWithin that tells a conversation's own rollback from another conversation's request:
// firsts[i] is slot i's FirstPrompt. A request whose score reaches firsts[i] - 1 keeps that conversation's whole first
// prompt, so it is the same conversation rolling back, and when slot i is the best-scoring one it takes it in place
// however much it gives up: the rows it discards are its own abandoned history. Only a request that diverges inside the
// first prompt, such as a new conversation sharing a preamble, is held to PickSlotWithin's rule and allowance.
func PickSlotFor(scores, lens, firsts []int, uses []uint64, busy []bool, allowance int) int {
	return pickSlot(scores, lens, uses, busy, allowance, func(i int) bool {
		return i < len(firsts) && firsts[i] > 0 && scores[i] >= firsts[i]-1
	})
}

// RollbackFloor is the least a window group's rollback budget can be: its base compacts in 64-token steps, so it keeps
// at least this much behind the window (docs/tasks/task-kv-architecture-2026-10.md §5.12).
const RollbackFloor = 448

// PickSlotWithin chooses a slot for a prompt from each slot's score (tokens of the prompt it holds), its length and
// its last-use tick. The best-scoring slot wins when reusing it keeps at least as much of it as the new suffix will
// overwrite and gives up fewer than allowance rows of it: a continuation of that slot's conversation, rolled back no
// further than the slot can serve (one block; for window groups the rollback budget, RollbackFloor or more). Anything
// else would cut another conversation's history (a shared lead such as a chat template's preamble, or a shared
// preamble the prompt can take by copy instead, Hits), so the prompt takes an empty slot, else the least recently used
// one. busy (nil: none) marks slots another running generation holds; they are never picked. -1 when every slot is
// busy.
func PickSlotWithin(scores, lens []int, uses []uint64, busy []bool, allowance int) int {
	return pickSlot(scores, lens, uses, busy, allowance, nil)
}

// pickSlot is PickSlotWithin, except that a best-scoring slot for which own (nil: none) is true is taken in place
// whatever the request gives up of it.
func pickSlot(scores, lens []int, uses []uint64, busy []bool, allowance int, own func(int) bool) int {
	free := func(i int) bool { return i >= len(busy) || !busy[i] }
	best := -1
	for i, s := range scores {
		if free(i) && s > 0 && (best < 0 || s > scores[best]) {
			best = i
		}
	}
	if best >= 0 {
		keep, cut := scores[best], lens[best]-scores[best]
		if (own != nil && own(best)) || (keep >= cut && cut < allowance) {
			return best
		}
	}
	for i, l := range lens {
		if free(i) && l == 0 {
			return i
		}
	}
	lru := -1
	for i := range uses {
		if free(i) && (lru < 0 || uses[i] < uses[lru]) {
			lru = i
		}
	}
	return lru
}

// Acquired records what a generation is about to do in slot: skip reuse leading rows of a prompt of promptLen
// tokens and prefill the rest. The Commit that ends the generation reads it to say which class wrote each row. Call
// it before Forget: the classes of the reused rows are read from the sequence the slot holds now.
func (a *Slots) Acquired(slot, reuse int, prompt []int) {
	p := pending{ok: true, reuse: reuse, promptLen: len(prompt), sum: fingerprint(prompt)}
	if s := a.seqs[slot]; s != nil && reuse > 0 {
		p.prefix = clipClasses(s.classes, reuse)
		p.first = s.first
	}
	a.pend[slot] = p
}

// Unrecorded says that whatever writes slot next was not announced with Acquired, so the Commit that follows has no
// producer to name and its rows are ClassUnknown.
func (a *Slots) Unrecorded(slot int) { a.pend[slot] = pending{} }

func fingerprint(tokens []int) uint64 {
	h := uint64(14695981039346656037)
	for _, t := range tokens {
		h = (h ^ uint64(t)) * 1099511628211
	}
	return h
}

// Forget empties slot: its rows are no longer known to be any sequence's. Its blocks leave the index at once, since
// a block that may have been overwritten must never be found.
func (a *Slots) Forget(slot int) { a.drop(slot) }

// SetCheckpoint records that the resident holds a recurrent checkpoint for slot's sequence at position c, the end of the
// prompt the sequence's last generation prefilled. c outside (0, Len] records none. A later Commit or Forget drops it.
func (a *Slots) SetCheckpoint(slot, c int) {
	s := a.seqs[slot]
	if s == nil {
		return
	}
	if c <= 0 || c > len(s.tokens) {
		s.ckpt, s.ckptKey = 0, Hash{}
		return
	}
	s.ckpt, s.ckptKey = c, checkpointKey(s.rootHash, s.tokens[:c])
}

func (a *Slots) drop(slot int) {
	a.live[slot] = false
	s := a.seqs[slot]
	if s == nil {
		return
	}
	for i := len(s.table) - 1; i >= 0; i-- {
		if i < len(s.hashes) {
			a.m.evict(s.table[i])
		} else {
			a.m.unref(s.table[i])
		}
	}
	s.table, s.hashes, s.released = nil, nil, true
	a.seqs[slot] = nil
}

// Commit records that slot now holds exactly tokens, written by a generation that ran to completion: a prompt of
// promptLen tokens and what decode emitted after it. Nothing an incomplete pass wrote is ever committed, which is
// why the full blocks can be published here. cm names the class of the prefilled rows; the reused rows keep the
// classes Acquired read, and decode's rows are ClassExact. Without a matching Acquired the rows are ClassUnknown.
// tokens and images become the sequence's and must not be modified afterwards.
func (a *Slots) Commit(slot int, root Root, tokens []int, images []ImageSpan, promptLen int, cm ClassMap) {
	a.commit(slot, root, tokens, images, promptLen, cm)
}

func (a *Slots) commit(slot int, root Root, tokens []int, images []ImageSpan, promptLen int, cm ClassMap) {
	p := a.pend[slot]
	a.pend[slot] = pending{}
	a.drop(slot)
	if len(tokens) == 0 {
		return
	}
	s := &Seq{m: a.m, root: root, rootHash: rootHash(root, a.m.parts), slot: int32(slot), tokens: tokens, images: images, first: promptLen}
	if p.ok && p.first > 0 && p.reuse >= p.first-1 {
		s.first = p.first // a continuation of the same conversation: it kept that conversation's first prompt
	}
	if p.ok && p.promptLen == promptLen && promptLen <= len(tokens) && p.reuse < promptLen && p.sum == fingerprint(tokens[:promptLen]) {
		for _, r := range p.prefix {
			s.classes = appendClass(s.classes, r)
		}
		s.classes = appendClass(s.classes, ClassRun{Start: p.reuse, Len: promptLen - p.reuse, Class: cm.PassAt(p.reuse, promptLen-p.reuse, promptLen)})
		s.classes = appendClass(s.classes, ClassRun{Start: promptLen, Len: len(tokens) - promptLen, Class: ClassExact})
	}
	for i := 0; i*BlockRows < len(tokens); i++ {
		id, _ := a.m.alloc(int32(slot), int32(i)) // an unlimited manager always has a block
		s.table = append(s.table, id)
	}
	s.Publish()
	// The sequence is idle: nobody references its published blocks. They are cached, the last first, until the
	// slot is given to something else.
	for i := len(s.hashes) - 1; i >= 0; i-- {
		a.m.unref(s.table[i])
	}
	a.seqs[slot] = s
}
