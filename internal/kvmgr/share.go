package kvmgr

// Sharing between slots: a request takes the published blocks another slot holds by copying them into its own slot
// (docs/tasks/task-kv-architecture-2026-10.md §6, KV2). The rows are the same bytes only when the chain hashes agree,
// and the chain carries the root, the tokens, the image spans and the class of every row, so a hash match is the whole
// test.

// ColdClasses is the class the requester's own cold prefill of a prompt of n tokens writes each row with: one pass
// from row 0.
func (c ClassMap) ColdClasses(n int) []ClassRun {
	if n <= 0 {
		return nil
	}
	return []ClassRun{{Start: 0, Len: n, Class: c.PassAt(0, n, n)}}
}

// CapHit shrinks a hit of hit rows on a prompt of n tokens, a block at a time, until the pass that prefills the rest
// writes the class the prompt's cold prefill writes. A hit that leaves the fast kernels too few rows would make the
// suffix exact class, and the reply would differ from the cold one (docs/measurements/kv0-invariance-2026-10-13.md).
func (c ClassMap) CapHit(n, hit int) int {
	cold := c.Pass(n, n)
	for hit > 0 && c.PassAt(hit, n-hit, n) != cold {
		hit -= BlockRows
	}
	return max(hit, 0)
}

// CapInPlace is CapHit for a request that takes a slot's own rows in place: reuse of the n-token prompt's leading tokens
// match the sequence whose rows were written with the classes held and whose image spans are images. When every reused row
// is the class the prompt's cold prefill writes and the pass that prefills the rest would write another, it backs reuse off
// to a block boundary, caps it as a hit and backs it off any image span it then cuts, so the reply equals the cold one. A
// reuse that already holds rows of another class (a conversation's own decoded history) cannot equal the cold prefill
// whatever its tail does, and is returned unchanged.
func (c ClassMap) CapInPlace(held []ClassRun, images []ImageSpan, n, reuse int) int {
	cold := c.Pass(n, n)
	if reuse <= 0 || reuse >= n || c.PassAt(reuse, n-reuse, n) == cold {
		return reuse
	}
	if runs := clipClasses(held, reuse); len(runs) != 1 || runs[0].Start != 0 || runs[0].Len != reuse || runs[0].Class != cold {
		return reuse
	}
	return offImages(images, c.CapHit(n, reuse/BlockRows*BlockRows)/BlockRows) * BlockRows
}

// requestChain is the chain hash of each leading block of a request, up to the last block that leaves a token to
// compute. It stops at a block an unclaimed image span touches or a row whose producer is not named, as hitBlocks does.
func (m *Manager) requestChain(root Root, tokens []int, images []ImageSpan, cold []ClassRun) []Hash {
	limit := 0
	if len(tokens) > 0 {
		limit = (len(tokens) - 1) / BlockRows
	}
	var out []Hash
	prev := rootHash(root, m.parts)
	for i := 0; i < limit; i++ {
		if unclaimed(images, i) || unknownClass(cold, i) {
			break
		}
		prev = blockHash(&m.h, prev, i, tokens, images, cold, m.parts)
		out = append(out, prev)
	}
	return out
}

// offImages backs a boundary of blocks blocks off any image span it cuts.
func offImages(images []ImageSpan, blocks int) int {
	for moved := true; moved; {
		moved = false
		edge := blocks * BlockRows
		for _, s := range images {
			if s.Start < edge && s.end() > edge {
				blocks, moved = s.Start/BlockRows, true
				break
			}
		}
	}
	return blocks
}

// Hits is, for each slot, how many leading rows of req its published blocks hold: whole blocks, at least one token
// left to compute, never ending inside an image span. cold is the class the requester's cold prefill writes each row
// with (ColdClasses). A slot under another root holds none.
func (a *Slots) Hits(req Request, cold []ClassRun) []int {
	out := make([]int, len(a.seqs))
	chain := a.m.requestChain(req.Root, req.Tokens, req.Images, cold)
	if len(chain) == 0 {
		return out
	}
	for sl, s := range a.seqs {
		if s == nil {
			continue
		}
		k := 0
		for k < len(chain) && k < len(s.hashes) && s.hashes[k] == chain[k] {
			k++
		}
		out[sl] = offImages(req.Images, k) * BlockRows
	}
	return out
}

// Copied records that the backend copied the first rows rows of src's sequence into dst: dst now holds that prefix,
// under the same root, with the same images and classes, and its full blocks are published. rows is a multiple of
// BlockRows no longer than src's published rows. Using src as a source counts as a use of it.
func (a *Slots) Copied(src, dst, rows int) {
	s := a.seqs[src]
	a.drop(dst)
	a.pend[dst] = pending{}
	if s == nil || rows <= 0 {
		return
	}
	rows = min(rows, len(s.hashes)*BlockRows)
	t := &Seq{m: a.m, root: s.root, rootHash: s.rootHash, slot: int32(dst), tokens: append([]int(nil), s.tokens[:rows]...),
		classes: clipClasses(s.classes, rows)}
	for _, sp := range s.images {
		if sp.end() <= rows {
			t.images = append(t.images, sp)
		}
	}
	for i := 0; i*BlockRows < rows; i++ {
		id, _ := a.m.alloc(int32(dst), int32(i)) // an unlimited manager always has a block
		t.table = append(t.table, id)
	}
	t.Publish()
	for i := len(t.hashes) - 1; i >= 0; i-- {
		a.m.unref(t.table[i])
	}
	a.seqs[dst] = t
	a.Touch(src)
}

// Restored records that the backend restored the first len(tokens) rows (a whole number of blocks; any tail is ignored) of a
// sequence into dst from a host snapshot: dst now holds that prefix under root, with the given classes, and its full blocks
// are published, as after Copied. The tokens and classes are the snapshot's own, so a later hit on dst is a hit on exactly the
// rows that were restored.
func (a *Slots) Restored(dst int, root Root, tokens []int, classes []ClassRun) {
	a.drop(dst)
	a.pend[dst] = pending{}
	rows := len(tokens) / BlockRows * BlockRows
	if rows <= 0 {
		return
	}
	t := &Seq{m: a.m, root: root, rootHash: rootHash(root, a.m.parts), slot: int32(dst), tokens: append([]int(nil), tokens[:rows]...),
		classes: clipClasses(classes, rows)}
	for i := 0; i*BlockRows < rows; i++ {
		id, _ := a.m.alloc(int32(dst), int32(i)) // an unlimited manager always has a block
		t.table = append(t.table, id)
	}
	t.Publish()
	for i := len(t.hashes) - 1; i >= 0; i-- {
		a.m.unref(t.table[i])
	}
	a.seqs[dst] = t
}

// Prefilled records that the generation announced in slot (Acquired) has finished prefilling prompt, and publishes the
// prompt's full blocks, so a request that arrives while the generation decodes can copy them. Decode writes only past
// the prompt, so those rows are final. The slot is live until Commit or Forget: its record belongs to the running
// generation.
func (a *Slots) Prefilled(slot int, root Root, prompt []int, images []ImageSpan, cm ClassMap) {
	p := a.pend[slot]
	a.commit(slot, root, append([]int(nil), prompt...), append([]ImageSpan(nil), images...), len(prompt), cm)
	a.pend[slot] = p // the generation's Commit reads it again
	if a.seqs[slot] != nil {
		a.live[slot] = true
	}
}

// Live reports whether slot's record belongs to a running generation (Prefilled, not yet committed or forgotten).
func (a *Slots) Live(slot int) bool { return a.live[slot] }
