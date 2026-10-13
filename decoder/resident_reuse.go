package decoder

import (
	"fmt"
	"slices"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

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
//     context. The adapter's identity is in the root every sequence is keyed under (kvRoot), so a mismatch matches
//     nothing. The identity is one id per loaded runtime (kvAdapterID): LoadAdapter always builds a fresh runtime and
//     registerAdapter retires the one it displaces rather than freeing it, so a reload under one name never compares
//     equal.
//
// The bookkeeping is the KV manager's (internal/kvmgr, docs/tasks/task-kv-architecture-2026-10.md §5): one sequence per
// KV slot, its tokens, image spans, root and the kernel class that wrote each row; which slot a prompt takes; what it
// may skip; and the index of published blocks. This file is the decoder's side of it: it asks the resident what the
// manager cannot know (the slot count, the window a slot still holds, the prefill floor), binds slots, and keeps
// resIDs, resIDsLora and resImgBlocks as the bound sequence's record for the generation code that reads them.

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

// kvSlots returns the model's KV manager, built on first use: a Model made by hand in a test has none until it asks.
func (m *Model) kvSlots() *kvmgr.Slots {
	if m.kv == nil {
		m.kv = kvmgr.NewSlots(m.residentSlotCount())
	}
	return m.kv
}

// kvAdapterIDs hands out adapter identities; 0 is the base weights.
var kvAdapterIDs atomic.Uint64

// kvAdapterID is the identity of a loaded adapter runtime in a KV key: assigned once, on first use, never reused.
func kvAdapterID(rt *loraRuntime) uint64 {
	if rt == nil {
		return 0
	}
	if id := rt.kvID.Load(); id != 0 {
		return id
	}
	rt.kvID.CompareAndSwap(0, kvAdapterIDs.Add(1))
	return rt.kvID.Load()
}

// kvRoot is the root a sequence built under lora is keyed by: the model's numerics identity, the sharing namespace,
// the adapter and the KV precision. The namespace is the single local one: serve passes no per-key scope yet (the
// task doc's D4, built with KV2's sharing).
func (m *Model) kvRoot(lora *loraRuntime) kvmgr.Root {
	r := kvmgr.Root{Adapter: kvAdapterID(lora), Precision: m.ResidentKVPrecision()}
	if m.w != nil && m.w.arch != nil {
		a := m.w.arch
		r.Model = fmt.Sprintf("%s/%dx%d/%d", a.Name, a.NumLayers, a.HiddenDim, a.VocabSize)
	}
	if m.be != nil {
		r.Model += "@" + m.be.Name()
	}
	return r
}

// fastPrefillMinRows is the fewest rows a pass needs to run a ResidentFastPrefill resident's fast kernels: a pass of
// 15 rows or fewer is exact whatever the prompt's length (docs/measurements/kv0-invariance-2026-10-13.md).
const fastPrefillMinRows = 16

// kvLeadClassMap is the class rule the short-prefix guard reads: a ResidentFastPrefill resident's.
func (m *Model) kvLeadClassMap() kvmgr.ClassMap {
	if fp, ok := m.resident.(ResidentFastPrefill); ok {
		return kvmgr.ClassMap{Floor: fp.FastPrefillFloor(), MinRows: fastPrefillMinRows}
	}
	return kvmgr.ClassMap{}
}

// kvClassMap is the rule for which kernel class a prefill pass of this resident writes, recorded with each committed
// row. A ResidentPrefillKernelFloor resident judges a pass on where it ends and has no row minimum.
func (m *Model) kvClassMap() kvmgr.ClassMap {
	if _, ok := m.resident.(ResidentFastPrefill); ok {
		return m.kvLeadClassMap()
	}
	if kf, ok := m.resident.(ResidentPrefillKernelFloor); ok {
		return kvmgr.ClassMap{Floor: kf.PrefillKernelFloor(), MinRows: 1}
	}
	return kvmgr.ClassMap{}
}

func kvSpansOfBlocks(blocks []residentImageBlock) []kvmgr.ImageSpan {
	if len(blocks) == 0 {
		return nil
	}
	out := make([]kvmgr.ImageSpan, len(blocks))
	for i, b := range blocks {
		out[i] = kvmgr.ImageSpan{Start: b.start, Len: b.end - b.start, Hash: b.hash}
	}
	return out
}

func kvSpansOfClaims(claims []residentImageClaim) []kvmgr.ImageSpan {
	if len(claims) == 0 {
		return nil
	}
	out := make([]kvmgr.ImageSpan, len(claims))
	for i, c := range claims {
		out[i] = kvmgr.ImageSpan{Start: c.Start, Len: c.Len, Hash: c.Hash}
	}
	return out
}

// kvAdopt makes the manager's record of the bound slot agree with resIDs, resIDsLora and resImgBlocks. Every writer
// in this file keeps the two together, so this does nothing in a generation. It is for a caller that assigns the
// fields directly (tests do, to stand for an earlier commit): the manager takes what it finds as the slot's content,
// with no producer named.
func (m *Model) kvAdopt(a *kvmgr.Slots) {
	slot := a.Bound()
	s := a.Seq(slot)
	if len(m.resIDs) == 0 {
		if s != nil {
			a.Forget(slot)
		}
		return
	}
	if s != nil && len(s.Tokens()) == len(m.resIDs) && &s.Tokens()[0] == &m.resIDs[0] &&
		s.Root().Adapter == kvAdapterID(m.resIDsLora) && slices.Equal(s.Images(), kvSpansOfBlocks(m.resImgBlocks)) {
		return
	}
	a.Unrecorded(slot)
	a.Commit(slot, m.kvRoot(m.resIDsLora), m.resIDs, kvSpansOfBlocks(m.resImgBlocks), len(m.resIDs), kvmgr.ClassMap{})
	m.kvSetLora(slot, m.resIDsLora)
}

// kvSetLora records which adapter runtime slot's sequence was built under. The manager keys on the adapter's
// identity; the runtime itself is the decoder's, and resIDsLora is read back from here when a slot is bound.
func (m *Model) kvSetLora(slot int, lora *loraRuntime) {
	for len(m.kvLora) <= slot {
		m.kvLora = append(m.kvLora, nil)
	}
	m.kvLora[slot] = lora
}

// residentReuseLen returns how many leading tokens of prompt the BOUND slot already holds and may skip. imgs
// describes any image blocks in prompt (nil for plain text). The rules are kvmgr.Slots.Score's. lora is the adapter
// bound for this generation (nil = base weights). A caller that takes this length without going through
// residentAcquireSlot has not told the manager what it will write, so the commit that follows names no producer.
func (m *Model) residentReuseLen(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	a := m.kvSlots()
	m.kvAdopt(a)
	a.Unrecorded(a.Bound())
	return m.residentReuseFloor(m.reuseLenOf(a.Bound(), prompt, imgs, lora))
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

// reuseLenOf is how many leading tokens of prompt the sequence in slot holds, the bound slot's or a parked one's,
// before the resident's own floor.
func (m *Model) reuseLenOf(slot int, prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	if m.residentReuseDisabled() {
		return 0
	}
	return m.kvSlots().Score(slot, kvmgr.Request{Root: m.kvRoot(lora), Tokens: prompt, Images: kvSpansOfClaims(imgs)}, m.hasRecurrentState())
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
// residentReuseLen. With several it scores every slot and binds pickResidentSlot's choice. Call it where
// residentReuseLen was called: under the resident claim, before the forget.
func (m *Model) residentAcquire(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	reuse, _ := m.residentAcquireSlot(prompt, imgs, lora, nil)
	return reuse
}

// residentAcquireSlot is residentAcquire that also returns the slot it bound, and never picks a slot busy marks (MC3:
// another running generation's). With every slot busy it binds nothing and returns slot -1. It tells the manager
// what the generation will do in the slot (skip reuse rows, prefill the rest of prompt), which the commit reads.
func (m *Model) residentAcquireSlot(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	reuse, slot = m.residentAcquireSlotAll(prompt, imgs, lora, busy)
	// the slot is bound now, so its window is the one to ask (a parked slot's score was not floored when it was counted)
	reuse = m.residentReuseFloor(m.declineShortLeadReuse(prompt, reuse))
	if slot >= 0 {
		a := m.kvSlots()
		a.Acquired(a.Bound(), reuse, prompt)
	}
	return reuse, slot
}

// minLeadReuseFast is the shortest prefix worth reusing under a resident with non-exact prefill kernels
// (ResidentFastPrefill), for a prompt that will run them (kvmgr.MinLeadReuse). Not an Options field: it is a
// correctness margin, not an operator choice.
const minLeadReuseFast = kvmgr.MinLeadReuse

// declineShortLeadReuse zeroes a short reuse when the resident prefills this prompt on non-exact kernels, so the prompt is prefilled whole by one kernel class
// (kvmgr.ClassMap.DeclineShortLead). Every caller forgets the slot's ids and prefills from the returned index, so 0 simply means a cold prefill into the slot it
// already bound.
func (m *Model) declineShortLeadReuse(prompt []int, reuse int) int {
	return m.kvLeadClassMap().DeclineShortLead(len(prompt), reuse)
}

func (m *Model) residentAcquireSlotAll(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	n := m.residentSlotCount()
	a := m.kvSlots()
	a.Resize(n)
	if n <= 1 {
		return m.residentReuseLen(prompt, imgs, lora), 0
	}
	scores, lens, uses := make([]int, n), make([]int, n), make([]uint64, n)
	bound := a.Bound()
	scores[bound] = m.residentReuseLen(prompt, imgs, lora) // adopts the bound record first, and asks its window
	for i := range n {
		if i != bound {
			scores[i] = m.reuseLenOf(i, prompt, imgs, lora)
		}
		lens[i], uses[i] = a.Len(i), a.Use(i)
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
	a.Touch(pick)
	return scores[pick], pick
}

// residentBind binds resident KV slot `slot` and makes resIDs, resIDsLora and resImgBlocks that slot's record. A
// no-op when it is already bound. Switching slots also clears resDrafterSynced: a block drafter's own context
// follows ONE conversation, the one the previous slot was serving.
func (m *Model) residentBind(slot int) error {
	a := m.kvSlots()
	if slot == a.Bound() || m.residentSlotCount() <= 1 {
		return nil
	}
	sl := m.resident.(ResidentKVSlotter) // residentSlotCount > 1 implies it
	if err := sl.UseKVSlot(slot); err != nil {
		return err
	}
	a.Resize(m.residentSlotCount())
	m.kvAdopt(a) // the outgoing slot's record, in case its fields were written directly
	a.SetBound(slot)
	m.resIDs, m.resIDsLora, m.resImgBlocks = nil, nil, nil
	if s := a.Seq(slot); s != nil {
		m.resIDs = s.Tokens()
		if slot < len(m.kvLora) {
			m.resIDsLora = m.kvLora[slot]
		}
		for _, sp := range s.Images() {
			m.resImgBlocks = append(m.resImgBlocks, residentImageBlock{start: sp.Start, end: sp.Start + sp.Len, hash: sp.Hash})
		}
	}
	m.resDrafterSynced = nil
	return nil
}

// pickResidentSlot chooses a slot for a prompt from each slot's reuse score, its committed length and its last-use
// tick (kvmgr.PickSlot): a continuation keeps its slot; a shared lead takes an empty slot, else the least recently
// used. busy (nil: none) marks slots another running generation holds (MC3); -1 when all are busy.
func pickResidentSlot(scores, lens []int, uses []uint64, busy []bool) int {
	return kvmgr.PickSlot(scores, lens, uses, busy)
}

// residentCommitIDs records the exact token sequence now committed to the resident KV: the prompt followed by
// everything decode emitted, since decode writes its own K/V at each position as it goes. newBlocks records the image
// blocks this turn added or re-verified, in prompt order (nil for plain text; several for a multi-image turn).
//
// Called only on the fully completed path, which is what lets the manager publish the sequence's full blocks here.
// Everything else leaves resIDs (and resImgBlocks) nil.
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
	a := m.kvSlots()
	a.Commit(a.Bound(), m.kvRoot(lora), ids, kvSpansOfBlocks(m.resImgBlocks), len(prompt), m.kvClassMap())
	m.kvSetLora(a.Bound(), lora)
}

// residentForgetIDs marks the resident KV's contents unknown. Anything that writes the cache
// outside a completed generateInto — a failed prefill, a cancelled decode, a batched verify —
// must call this, because a stale id list is the one way this feature can be WRONG rather than
// merely slow. Clears resImgBlocks together with resIDs, always — an image block's identity is
// meaningless once the token sequence backing its position is no longer trusted. The bound slot's
// blocks leave the manager's index with it.
func (m *Model) residentForgetIDs() {
	m.resIDsLora = nil
	m.resIDs = nil
	m.resImgBlocks = nil
	m.resDrafterSynced = nil // any resident write invalidates a block drafter's own reuse too
	if m.kv != nil {
		m.kv.Forget(m.kv.Bound())
	}
}
