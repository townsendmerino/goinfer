package decoder

// Prefix reuse on the resident positional KV.
//
// A resident model decodes statelessly by default: a session's prefix cache is CPU-side while the resident KV lives on
// the GPU, and both cannot be the source of truth, so every turn would re-prefill its whole prompt. Reuse done natively
// on the GPU cache avoids the conflict, and the resident cache suits it: it is positional (the token at position p
// lives at slot p), so "truncate to P" costs nothing (residentDecoder.TruncateTo is a no-op for that reason, and
// attention reads only nKeys = pos+1), and an agent turn is a strict prefix extension of the last (turn N+1 is turn N
// plus the assistant's tool call plus the tool result). So the mechanism is bookkeeping: remember which ids are
// committed to the cache and prefill only the divergent suffix. The measured gain is in docs/integrations/claude-code.md.
//
// Correctness is the entire risk: a wrong prefix match produces confidently wrong output with no error anywhere. Four
// rules keep it honest:
//
//  1. Match on token ids, never text. A client that edits its last message shifts tokenisation, and the longest common
//     prefix is what absorbs that.
//  2. The recorded ids are cleared to nil ("unknown, cold-prefill next time") on any path that is not a fully
//     completed generation: an error, a cancellation, a resident claim lost to a concurrent generation. Forgetting
//     costs a slow turn; remembering wrongly gives a wrong answer.
//  3. At least one token is always prefilled, so the seed logits that start decode are freshly computed.
//  4. The record names the weights as well as the ids. A LoRA adapter changes every targeted projection and so every
//     later layer's K/V, so an identical id prefix built under a different adapter (or none) is someone else's
//     context. residentCommitIDs records the bound *loraRuntime and residentReuseLen refuses on any mismatch, which
//     keeps same-adapter reuse and closes every crossing. Pointer identity is sound: LoadAdapter always builds a fresh
//     runtime and registerAdapter retires the one it displaces rather than freeing it, so a reload under one name
//     never compares equal.

// residentReuseDisabled is the escape hatch / A-B switch, same convention as
// GOINFER_NO_GREEDY_FASTPATH and GOINFER_NO_KVONLY_PREFILL.
func (m *Model) residentReuseDisabled() bool { return m.knobs.get(knobNoResidentReuse) != "" }

// residentImageBlock records one image block committed to the resident KV: its absolute position span within resIDs and
// a content hash of the raw image bytes that produced it (docs/multimodal.md). Every position inside [start,end)
// attended every other position in the block under a bidirectional mask during its CPU prefill, so a reuse match must
// treat the block as one atomic unit: either the whole span verifies and is skipped together, or none of it is.
//
// hash == 0 means "no claim": a caller with no reuse story of its own may pass 0, and residentReuseLen's block-match
// guard requires blk.hash != 0 before comparing, so two callers who both pass 0 cannot satisfy claim.Hash == blk.hash
// by coincidence. The block is still recorded with hash 0 rather than omitted: an unrecorded region falls through to
// the per-position id comparison, which would treat two different images sharing a slot as identical, because image
// placeholder ids are content-independent. Recording keeps the atomicity check in play; the guard ensures it never
// emits a false positive.
type residentImageBlock struct {
	start, end int
	hash       uint64
}

// residentImageClaim describes one image block in the PROMPT being checked for reuse.
// GenerateVL/GenerateQwenVL each produce exactly one claim per call (today's one-image-per-turn
// shape), but residentReuseLen itself doesn't assume a count of one.
type residentImageClaim struct {
	Start, Len int
	Hash       uint64
}

// residentReuseLen returns how many leading tokens of prompt are already committed to the
// resident KV and may be skipped. imgs describes any image blocks in prompt (nil for plain
// text — the common case, byte-for-byte the original LCP scan with zero added cost).
//
// Capped at len(prompt)-1 on purpose: generateInto's contract is that prefill covers at least
// one token, whose logits seed decode. Returning len(prompt) would leave the caller with no
// seed logits and nothing to recompute them from.
//
// lora is the adapter bound for THIS generation (nil = base weights); rule 4 refuses any reuse
// of KV that was built under a different one.
func (m *Model) residentReuseLen(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	return m.residentReuseFloor(m.reuseLenOf(m.resIDs, m.resIDsLora, m.resImgBlocks, prompt, imgs, lora))
}

// ResidentReusableKV is implemented by a resident whose KV for some layers holds only a recent window of positions (Options.ResidentWindowedKV).
// ReusableKV(prefix) is how many of the first prefix positions of the bound slot's sequence it can serve to a forward that continues at
// position prefix: prefix itself, or 0 when the window that forward attends to has already been dropped, in which case the caller prefills
// the prompt from the start. A positive answer other than prefix is never returned.
type ResidentReusableKV interface {
	ReusableKV(prefix int) int
}

// residentReuseFloor caps a reuse length of the BOUND slot at what its resident can serve (ResidentReusableKV); a resident that keeps every
// position is untouched. A cap only ever lengthens the prefill, never changes what it computes.
func (m *Model) residentReuseFloor(reuse int) int {
	if reuse <= 0 {
		return reuse
	}
	if rk, ok := m.resident.(ResidentReusableKV); ok {
		return rk.ReusableKV(reuse)
	}
	return reuse
}

// reuseLenOf is residentReuseLen over one slot's bookkeeping (ids, the adapter that built it, its image blocks) — the
// bound slot's (residentReuseLen) or a parked one's (residentAcquire, MC1).
func (m *Model) reuseLenOf(resIDs []int, resLora *loraRuntime, resImgBlocks []residentImageBlock,
	prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	if m.residentReuseDisabled() || len(resIDs) == 0 || len(prompt) == 0 {
		return 0
	}
	if lora != resLora {
		return 0 // rule 4: same ids, different weights — the KV is not this generation's
	}
	// Recurrent families can only reuse an exact, strict extension. The attention-only matching below (longest common
	// prefix, capped at len(prompt)-1) cannot help a recurrent family, whose state cannot be rewound to an arbitrary
	// earlier position: a Gated DeltaNet's conv ring and matrix state (and Mamba-2's, and LFM2's conv window) are mutated
	// in place per token with no per-position history, and the resident path re-zeroes them only at pos == 0. The only
	// safe continuation point is exactly len(resIDs), where the live recurrent state already equals the state after
	// resIDs (residentCommitIDs's invariant, held by every other writer calling residentForgetIDs). A prompt that is
	// resIDs plus at least one new token can decode forward from there, which is what an agent turn is.
	//
	// Anything else (an edited earlier message, a shorter resend, an identical resend) has no safe continuation point,
	// because the state would have to run backward, and returns 0. An identical resend has len(prompt) == n and takes that
	// branch: reusing there diverged from token 0 with no error. TestPagerDeterminism is the gate.
	if m.hasRecurrentState() {
		n := len(resIDs)
		if len(prompt) <= n {
			return 0
		}
		for i := range n {
			if resIDs[i] != prompt[i] {
				return 0
			}
		}
		// Image blocks. The placeholder id is the same for every image, so an id match over an image block proves nothing about
		// which image the recurrent state was built from, and the state cannot be rewound to an image boundary to repair a
		// wrong guess. So the extension is honoured only if the committed blocks and this prompt's claims are the same set
		// over the reused span: every committed block has a claim with the same start, length and nonzero hash, and no claim
		// reaches into [0, n) without a committed block behind it. Anything else is cold.
		for _, blk := range resImgBlocks {
			c, ok := findImageClaim(imgs, blk.start)
			if !ok || blk.hash == 0 || c.Hash != blk.hash || c.Len != blk.end-blk.start {
				return 0
			}
		}
		for _, c := range imgs {
			if c.Start >= n {
				continue // a new image past the reused span is ordinary new work
			}
			if c.Start+c.Len > n {
				return 0 // straddles the boundary: half of it would be reused
			}
			ok := false
			for _, blk := range resImgBlocks {
				if blk.start == c.Start && blk.end-blk.start == c.Len {
					ok = true
					break
				}
			}
			if !ok {
				return 0
			}
		}
		return n
	}
	n := min(len(prompt)-1, len(resIDs))
	i, bi := 0, 0 // bi: next candidate index into m.resImgBlocks (stored ascending by start)
	for i < n {
		for bi < len(resImgBlocks) && resImgBlocks[bi].end <= i {
			bi++ // fully behind us
		}
		if bi < len(resImgBlocks) && resImgBlocks[bi].start == i {
			blk := resImgBlocks[bi]
			if claim, ok := findImageClaim(imgs, blk.start); ok &&
				claim.Len == blk.end-blk.start && blk.hash != 0 && claim.Hash == blk.hash && blk.end <= n {
				// Whole block verified byte-identical: jump straight past it, atomically —
				// never a partial in-block stop (see residentImageBlock's doc comment).
				i = blk.end
				bi++
				continue
			}
			// Different image, no claim at all, or the block doesn't fully fit under the
			// seed cap n: stop EXACTLY at the block's start — not one position later (that
			// would trust an unverified row) or earlier (that would discard a verified
			// match for no reason).
			return i
		}
		if resIDs[i] != prompt[i] {
			return i
		}
		i++
	}
	return i
}

// residentSlot is one resident KV slot's reuse bookkeeping (MC1, docs/tasks/task-concurrency-2026-09.md). The BOUND
// slot's ids / adapter / image blocks live in the Model's own resIDs, resIDsLora and resImgBlocks, so every commit and
// forget site in this file and its callers is unchanged; the other slots' are parked in Model.resSlots and swapped in
// by residentAcquire. lastUse is kept for every slot, the bound one included.
type residentSlot struct {
	ids       []int
	lora      *loraRuntime
	imgBlocks []residentImageBlock
	lastUse   uint64
}

// residentSlotCount is how many resident KV slots a generation may choose among: what the resident allocated
// (ResidentKVSlotter), else 1. A family with recurrent state keeps one — its state is mutated in place and is not
// part of a KV slot, so two conversations could not both keep theirs (the conservative choice MC1 registers).
func (m *Model) residentSlotCount() int {
	sl, ok := m.resident.(ResidentKVSlotter)
	if !ok || m.hasRecurrentState() {
		return 1
	}
	return max(1, sl.KVSlots())
}

// residentAcquire binds the resident KV slot a generation over prompt will use and returns how many leading tokens of
// prompt that slot already holds (residentReuseLen's contract, for the slot it bound). With one slot it is
// residentReuseLen. With several it scores every slot and binds pickResidentSlot's choice, parking the previously bound
// slot's bookkeeping (residentBind). Call it where residentReuseLen was called: under the resident claim, before the
// forget.
func (m *Model) residentAcquire(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	reuse, _ := m.residentAcquireSlot(prompt, imgs, lora, nil)
	return reuse
}

// residentAcquireSlot is residentAcquire that also returns the slot it bound, and never picks a slot busy marks (MC3:
// another running generation's). With every slot busy it binds nothing and returns slot -1.
func (m *Model) residentAcquireSlot(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	reuse, slot = m.residentAcquireSlotAll(prompt, imgs, lora, busy)
	// the slot is bound now, so its window is the one to ask (a parked slot's score was not floored when it was counted)
	return m.residentReuseFloor(m.declineShortLeadReuse(prompt, reuse)), slot
}

// minLeadReuseFast is the shortest prefix worth reusing under a resident with non-exact prefill kernels
// (ResidentFastPrefill), for a prompt that will run them. Below it the reused rows are a shared lead (a chat
// template's preamble, a few tokens left by an unrelated short request): re-prefilling them costs next to nothing, and
// keeping them can change the reply, because they were computed by the exact kernels while the rest of the prompt runs
// fast. A longer reuse is a continuation and is kept. Not an Options field: it is a correctness margin, not an
// operator choice.
const minLeadReuseFast = 64

// declineShortLeadReuse zeroes a short reuse when the resident prefills this prompt on non-exact kernels, so the prompt is prefilled whole by one kernel class. Every caller
// forgets the slot's ids and prefills from the returned index, so 0 simply means a cold prefill into the slot it already bound.
func (m *Model) declineShortLeadReuse(prompt []int, reuse int) int {
	if reuse <= 0 || reuse >= minLeadReuseFast {
		return reuse
	}
	fp, ok := m.resident.(ResidentFastPrefill)
	if !ok {
		return reuse
	}
	if floor := fp.FastPrefillFloor(); floor <= 0 || len(prompt) < floor {
		return reuse // fast kernels off, or a prompt that stays on the exact ones: the reused rows are the same class
	}
	return 0
}

func (m *Model) residentAcquireSlotAll(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	n := m.residentSlotCount()
	if n <= 1 {
		return m.residentReuseLen(prompt, imgs, lora), 0
	}
	if len(m.resSlots) != n {
		m.resSlots, m.resCur = make([]residentSlot, n), 0
	}
	scores, lens, uses := make([]int, n), make([]int, n), make([]uint64, n)
	for i := range n {
		if i == m.resCur {
			scores[i], lens[i] = m.residentReuseLen(prompt, imgs, lora), len(m.resIDs)
		} else {
			st := &m.resSlots[i]
			scores[i], lens[i] = m.reuseLenOf(st.ids, st.lora, st.imgBlocks, prompt, imgs, lora), len(st.ids)
		}
		uses[i] = m.resSlots[i].lastUse
	}
	pick := pickResidentSlot(scores, lens, uses, busy)
	if pick < 0 {
		return 0, -1
	}
	if err := m.residentBind(pick); err != nil {
		// The resident keeps its previous binding; with its bookkeeping unknown, go cold on it. Slot -1: an MC3 caller
		// must not decode on a binding it did not choose (it may be another generation's slot); residentAcquire's
		// callers carry on cold on the previous binding, as they always have.
		m.residentForgetIDs()
		return 0, -1
	}
	m.resTick++
	m.resSlots[pick].lastUse = m.resTick
	return scores[pick], pick
}

// residentBind binds resident KV slot `slot` and swaps its reuse bookkeeping into the Model's bound-slot fields,
// parking the previously bound slot's. A no-op when it is already bound. Switching slots also clears
// resDrafterSynced: a block drafter's own context follows ONE conversation, the one the previous slot was serving.
func (m *Model) residentBind(slot int) error {
	if slot == m.resCur || m.residentSlotCount() <= 1 {
		return nil
	}
	sl := m.resident.(ResidentKVSlotter) // residentSlotCount > 1 implies it
	if err := sl.UseKVSlot(slot); err != nil {
		return err
	}
	if len(m.resSlots) != m.residentSlotCount() {
		m.resSlots = make([]residentSlot, m.residentSlotCount())
	}
	cur := &m.resSlots[m.resCur]
	cur.ids, cur.lora, cur.imgBlocks = m.resIDs, m.resIDsLora, m.resImgBlocks
	in := &m.resSlots[slot]
	m.resIDs, m.resIDsLora, m.resImgBlocks = in.ids, in.lora, in.imgBlocks
	in.ids, in.lora, in.imgBlocks = nil, nil, nil // live in the Model's fields while bound
	m.resDrafterSynced = nil
	m.resCur = slot
	return nil
}

// pickResidentSlot chooses a slot for a prompt from each slot's reuse score (tokens of the prompt it holds), its
// committed length, and its last-use tick. The best-scoring slot wins when reusing it keeps at least as much of it as
// the new suffix will overwrite: a continuation of that slot's own conversation. A match shorter than what it would
// discard is a shared lead (a chat template's preamble), not a continuation: taking it would truncate another
// conversation's history to save re-prefilling a few tokens, the trap serve's session LRU also had to avoid
// (internal/serveapp pickSession). Such a prompt takes an empty slot, else the least recently used one.
//
// busy (nil: none) marks slots another running generation holds (MC3); they are never picked. -1 when all are busy.
func pickResidentSlot(scores, lens []int, uses []uint64, busy []bool) int {
	free := func(i int) bool { return i >= len(busy) || !busy[i] }
	best := -1
	for i, s := range scores {
		if free(i) && s > 0 && (best < 0 || s > scores[best]) {
			best = i
		}
	}
	if best >= 0 && scores[best] >= lens[best]-scores[best] {
		return best
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

// findImageClaim returns the claim starting at pos, if any.
func findImageClaim(imgs []residentImageClaim, pos int) (residentImageClaim, bool) {
	for _, c := range imgs {
		if c.Start == pos {
			return c, true
		}
	}
	return residentImageClaim{}, false
}

// residentCommitIDs records the exact token sequence now committed to the resident KV: the prompt followed by
// everything decode emitted, since decode writes its own K/V at each position as it goes. newBlocks records the image
// blocks this turn added or re-verified, in prompt order (nil for plain text; several for a multi-image turn).
//
// Called only on the fully completed path. Everything else leaves resIDs (and resImgBlocks) nil.
func (m *Model) residentCommitIDs(prompt, generated []int, newBlocks []residentImageBlock, lora *loraRuntime) {
	ids := make([]int, 0, len(prompt)+len(generated))
	ids = append(ids, prompt...)
	ids = append(ids, generated...)
	m.resIDs = ids
	m.resIDsLora = lora
	if len(newBlocks) > 0 {
		kept := m.resImgBlocks[:0:0]
		for _, b := range m.resImgBlocks {
			if b.end <= newBlocks[0].start { // still valid: append-only, earlier positions unchanged
				kept = append(kept, b)
			}
		}
		m.resImgBlocks = append(kept, newBlocks...)
	}
}

// residentForgetIDs marks the resident KV's contents unknown. Anything that writes the cache
// outside a completed generateInto — a failed prefill, a cancelled decode, a batched verify —
// must call this, because a stale id list is the one way this feature can be WRONG rather than
// merely slow. Clears resImgBlocks together with resIDs, always — an image block's identity is
// meaningless once the token sequence backing its position is no longer trusted.
func (m *Model) residentForgetIDs() {
	m.resIDsLora = nil
	m.resIDs = nil
	m.resImgBlocks = nil
	m.resDrafterSynced = nil // any resident write invalidates a block drafter's own reuse too
}
