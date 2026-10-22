package decoder

import (
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"time"

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

// kvRoot is kvRootIn in the single local namespace.
func (m *Model) kvRoot(lora *loraRuntime) kvmgr.Root { return m.kvRootIn(lora, "") }

// kvRootIn is the root a sequence built under lora in namespace ns is keyed by: the model's numerics identity, the
// sharing namespace (KVScope), the adapter and the KV precision.
func (m *Model) kvRootIn(lora *loraRuntime, ns string) kvmgr.Root {
	r := kvmgr.Root{Namespace: ns, Adapter: kvAdapterID(lora), Precision: m.ResidentKVPrecision()}
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
// row. A ResidentPrefillKernelFloor resident judges a pass on where it ends and has no row minimum. On a model with
// sliding-window layers a fast pass that starts off a block boundary is recorded with no producer
// (kvmgr.ClassMap.Windowed; measured on CUDA, applied to every backend with a fast class).
func (m *Model) kvClassMap() kvmgr.ClassMap {
	cm := kvmgr.ClassMap{}
	if _, ok := m.resident.(ResidentFastPrefill); ok {
		cm = m.kvLeadClassMap()
	} else if kf, ok := m.resident.(ResidentPrefillKernelFloor); ok {
		cm = kvmgr.ClassMap{Floor: kf.PrefillKernelFloor(), MinRows: 1}
	}
	cm.Windowed = m.w != nil && m.w.arch != nil && m.w.arch.SlidingWindow > 0
	return cm
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
// with no producer named. A live slot's record is a running generation's (kvmgr.Slots.Live), not the bound fields',
// and is left alone.
func (m *Model) kvAdopt(a *kvmgr.Slots) {
	slot := a.Bound()
	if a.Live(slot) {
		return
	}
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
	a.Commit(slot, m.kvRootIn(m.resIDsLora, m.kvNSOf(slot)), m.resIDs, kvSpansOfBlocks(m.resImgBlocks), len(m.resIDs), kvmgr.ClassMap{})
	m.kvSetLora(slot, m.resIDsLora)
}

// kvSetNS records the sharing namespace of the request that took slot; the commit that ends it keys the slot's
// sequence under it.
func (m *Model) kvSetNS(slot int, ns string) {
	for len(m.kvNS) <= slot {
		m.kvNS = append(m.kvNS, "")
	}
	m.kvNS[slot] = ns
}

// kvNSOf is the namespace recorded for slot ("" when none was).
func (m *Model) kvNSOf(slot int) string {
	if slot < len(m.kvNS) {
		return m.kvNS[slot]
	}
	return ""
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
	return m.residentReuseLenIn(prompt, imgs, lora, KVScope{})
}

// residentReuseLenIn is residentReuseLen for a request under scope sc that takes the bound slot as it is: the caller
// prefills into it and commits there.
func (m *Model) residentReuseLenIn(prompt []int, imgs []residentImageClaim, lora *loraRuntime, sc KVScope) int {
	n := m.boundReuseLen(prompt, imgs, lora, sc.Namespace, false)
	m.kvSetNS(m.kvSlots().Bound(), sc.Namespace)
	return n
}

// boundReuseLen scores the bound slot for prompt; ckpt lets a recurrent sequence's checkpoint score (kvmgr.Request.Checkpoint).
// A live slot's announcement is a running generation's, so it is not marked unrecorded.
func (m *Model) boundReuseLen(prompt []int, imgs []residentImageClaim, lora *loraRuntime, ns string, ckpt bool) int {
	a := m.kvSlots()
	m.kvAdopt(a)
	if !a.Live(a.Bound()) {
		a.Unrecorded(a.Bound())
	}
	return m.residentReuseFloor(m.reuseLenOf(a.Bound(), prompt, imgs, lora, ns, ckpt))
}

// ResidentKVCopier is the optional capability of a resident with several KV slots (ResidentKVSlotter) to copy rows
// between them. CopyKV copies rows [start, end) of every layer from slot src to slot dst, src != dst, start and end
// multiples of kvmgr.BlockRows; dst then continues at position end, and src is unchanged. A range the backend cannot
// serve (a window that no longer holds rows dst will attend to, a layout it does not copy) returns a *KVCopyDecline
// before anything is written. The caller treats any error as having written nothing it may trust from start on.
type ResidentKVCopier interface {
	CopyKV(src, dst, start, end int) error
}

// KVCopyDecline is CopyKV's refusal of a range, with the reason; dst is untouched.
type KVCopyDecline struct{ Reason string }

func (d *KVCopyDecline) Error() string { return "kv copy declined: " + d.Reason }

// ResidentReusableKVSlot is ResidentReusableKV's question for any slot, bound or not: of the first prefix positions
// of slot's sequence, how many it can serve to a forward that continues at position prefix (prefix or 0).
type ResidentReusableKVSlot interface {
	ReusableKVSlot(slot, prefix int) int
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
func (m *Model) reuseLenOf(slot int, prompt []int, imgs []residentImageClaim, lora *loraRuntime, ns string, ckpt bool) int {
	if m.residentReuseDisabled() {
		return 0
	}
	req := kvmgr.Request{Root: m.kvRootIn(lora, ns), Tokens: prompt, Images: kvSpansOfClaims(imgs), Checkpoint: ckpt}
	return m.kvSlots().Score(slot, req, m.hasRecurrentState())
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
	return m.residentAcquireIn(prompt, imgs, lora, nil, KVScope{}).reuse
}

// residentAcquireSlot is residentAcquire that also returns the slot it bound, and never picks a slot busy marks (MC3:
// another running generation's). With every slot busy it binds nothing and returns slot -1. It tells the manager
// what the generation will do in the slot (skip reuse rows, prefill the rest of prompt), which the commit reads.
func (m *Model) residentAcquireSlot(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	acq := m.residentAcquireIn(prompt, imgs, lora, busy, KVScope{})
	return acq.reuse, acq.slot
}

// kvAcquired is what residentAcquireIn did: the rows the generation skips, the slot it bound (-1: none), and how many
// of the skipped rows were copied from another slot, from which, at what cost, or why a copy was declined.
type kvAcquired struct {
	reuse, slot int
	shared, src int
	copyDur     time.Duration
	declined    string
	// snapRows is how many rows a host snapshot restored into the slot (residentRestoreSnapshot), snapDur that restore's wall time.
	snapRows int
	snapDur  time.Duration
	snapFrom *kvSnap // the snapshot those rows came from
	// markFired is set when the prefill reached its snapshot mark and offered a boundary snapshot (taken or declined): the end of the prompt takes none.
	markFired bool
	// slotLen and match are the bound slot's record length before this request and the prompt's common prefix with it
	// (Generation.PrefillSlot and PrefillMatch).
	slotLen, match int
	// ckptRestored is the position a recurrent checkpoint was restored at (0: none), ckptDur the copy's time, and ckptDeclined why a
	// checkpoint hit fell back to a cold prefill (residentCheckpointRestore).
	ckptRestored int
	ckptDur      time.Duration
	ckptDeclined string
}

// residentAcquireIn is residentAcquireSlot for a request under scope sc. When another slot's published blocks hold
// more of the prompt than the bound slot does, it copies them into the bound slot (ResidentKVCopier) and skips them.
func (m *Model) residentAcquireIn(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool, sc KVScope) kvAcquired {
	acq := m.residentAcquireSlotAll(prompt, imgs, lora, busy, sc)
	if imgs == nil {
		m.residentRestoreSnapshot(prompt, lora, sc, &acq)
	}
	// the slot is bound now, so its window is the one to ask (a parked slot's score was not floored when it was counted)
	acq.reuse = m.residentReuseFloor(m.declineShortLeadReuse(prompt, acq.reuse))
	if acq.reuse < acq.shared { // the bound slot's window cannot serve the copied rows: none of them count
		acq.shared = 0
	}
	if acq.slot >= 0 {
		a := m.kvSlots()
		if sq := a.Seq(a.Bound()); sq != nil {
			acq.slotLen, acq.match = sq.Len(), commonPrefixLen(sq.Tokens(), prompt)
		}
		if sc.checkpoint {
			m.residentCheckpointRestore(prompt, &acq) // before the manager records what the request skips
		}
		a.Acquired(a.Bound(), acq.reuse, prompt)
		m.kvSetNS(a.Bound(), sc.Namespace)
	}
	return acq
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

func (m *Model) residentAcquireSlotAll(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool, sc KVScope) kvAcquired {
	n := m.residentSlotCount()
	a := m.kvSlots()
	a.Resize(n)
	if n <= 1 {
		return kvAcquired{reuse: m.boundReuseLen(prompt, imgs, lora, sc.Namespace, sc.checkpoint), src: -1}
	}
	scores, lens, firsts, uses := make([]int, n), make([]int, n), make([]int, n), make([]uint64, n)
	bound := a.Bound()
	if bound >= len(busy) || !busy[bound] { // a busy slot is never picked, and its record is another generation's
		scores[bound] = m.boundReuseLen(prompt, imgs, lora, sc.Namespace, false) // adopts the bound record first, and asks its window
	}
	for i := range n {
		if i != bound {
			scores[i] = m.reuseLenOf(i, prompt, imgs, lora, sc.Namespace, false)
		}
		lens[i], uses[i] = a.Len(i), a.Use(i)
		if a.KeepsFirst(i, prompt) {
			firsts[i] = a.FirstPrompt(i)
		}
	}
	if kvAllowanceOverride > 0 {
		firsts = nil
	}
	pick := kvmgr.PickSlotFor(scores, lens, firsts, uses, busy, m.kvRollbackAllowance())
	if pick < 0 {
		return kvAcquired{slot: -1, src: -1}
	}
	if err := m.residentBind(pick); err != nil {
		// The resident keeps its previous binding; with its bookkeeping unknown, go cold on it. Slot -1: an MC3 caller
		// must not decode on a binding it did not choose (it may be another generation's slot); residentAcquire's
		// callers carry on cold on the previous binding, as they always have.
		m.residentForgetIDs()
		return kvAcquired{slot: -1, src: -1}
	}
	a.Touch(pick)
	acq := kvAcquired{reuse: scores[pick], slot: pick, src: -1}
	m.residentShare(prompt, imgs, lora, sc, &acq)
	return acq
}

// kvChooseDonor picks the slot to copy from: of the slots other than dst whose capped hit beats reuse, the longest hit
// (the lower slot on a tie) that the resident says still holds the rows (ResidentReusableKVSlot; a resident without it
// holds every row it has). src is -1 when none beats reuse, and when every candidate's window has moved past its rows,
// in which case declined says so.
func kvChooseDonor(res ResidentForward, hits []int, cm kvmgr.ClassMap, n, dst, reuse int) (src, hit int, declined string) {
	type cand struct{ slot, hit int }
	var cands []cand
	for s, h := range hits {
		if h = cm.CapHit(n, h); s != dst && h > reuse {
			cands = append(cands, cand{s, h})
		}
	}
	slices.SortStableFunc(cands, func(a, b cand) int { return b.hit - a.hit })
	rk, windowed := res.(ResidentReusableKVSlot)
	for _, c := range cands {
		if !windowed || rk.ReusableKVSlot(c.slot, c.hit) >= c.hit {
			return c.slot, c.hit, ""
		}
	}
	if len(cands) > 0 {
		return -1, 0, fmt.Sprintf("no source slot's window still holds the rows (%d candidates)", len(cands))
	}
	return -1, 0, ""
}

// kvOtherDonors states, for a declined copy, how many slots other than src and dst hold a hit longer than reuse that their
// windows still hold. It changes nothing; it is the G5-C diagnostic of how much of a decline is the choice of donor
// (docs/measurements/kv2-cuda-2026-10-15.md, §6).
func kvOtherDonors(rk ResidentReusableKVSlot, hits []int, cm kvmgr.ClassMap, n, src, dst, reuse int) string {
	k := 0
	for s, h := range hits {
		if s == src || s == dst {
			continue
		}
		if h = cm.CapHit(n, h); h > reuse && rk.ReusableKVSlot(s, h) >= h {
			k++
		}
	}
	return fmt.Sprintf("; other slots that could serve: %d", k)
}

// residentShare copies into the bound slot acq.slot the longest run of prompt another slot's published blocks hold
// (kvmgr.Slots.Hits, capped by kvmgr.ClassMap.CapHit), when that is more than acq.reuse, the rows the slot itself
// holds. It does nothing under an Off scope or Options.KVShareOff, for a recurrent family, without ResidentKVCopier, or when the source's
// window no longer holds the rows. A decline or an error leaves acq.reuse at what the slot can still trust.
func (m *Model) residentShare(prompt []int, imgs []residentImageClaim, lora *loraRuntime, sc KVScope, acq *kvAcquired) {
	cp, ok := m.resident.(ResidentKVCopier)
	if !ok || sc.Off || m.kvShareOff || m.residentReuseDisabled() || m.hasRecurrentState() {
		return
	}
	a, dst, cm := m.kvSlots(), acq.slot, m.kvClassMap()
	hits := a.Hits(kvmgr.Request{Root: m.kvRootIn(lora, sc.Namespace), Tokens: prompt, Images: kvSpansOfClaims(imgs)}, cm.ColdClasses(len(prompt)))
	src, hit, declined := kvChooseDonor(m.resident, hits, cm, len(prompt), dst, acq.reuse)
	if src < 0 {
		acq.declined = declined
		return
	}
	start := min(hits[dst], hit) // dst already holds these as the same published blocks
	t0 := time.Now()
	err := cp.CopyKV(src, dst, start, hit)
	acq.copyDur = time.Since(t0)
	if err != nil {
		var d *KVCopyDecline
		if errors.As(err, &d) {
			acq.declined = d.Reason
			if rk, ok := m.resident.(ResidentReusableKVSlot); ok {
				acq.declined += kvOtherDonors(rk, hits, cm, len(prompt), src, dst, acq.reuse)
			}
		} else {
			acq.declined = err.Error()
			acq.reuse = min(acq.reuse, start)
		}
		return
	}
	a.Copied(src, dst, hit)
	s := a.Seq(dst)
	m.resIDs, m.resIDsLora, m.resImgBlocks = s.Tokens(), lora, nil
	for _, sp := range s.Images() {
		m.resImgBlocks = append(m.resImgBlocks, residentImageBlock{start: sp.Start, end: sp.Start + sp.Len, hash: sp.Hash})
	}
	m.kvSetLora(dst, lora)
	acq.reuse, acq.shared, acq.src = hit, hit, src
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

// kvAllowanceOverride, when positive, replaces kvRollbackAllowance and leaves the slot choice no first-prompt lengths,
// so an own rollback is held to the keep-at-least-as-much-as-it-overwrites rule like any request. Set only by in-package
// tests: a large value switches both off, so a test can compare the rest of the slot choice with the bookkeeping KV1
// replaced.
var kvAllowanceOverride int

// kvRollbackAllowance is how many rows a request may give up of a slot it takes in place (kvmgr.PickSlotWithin): one
// block, or for a model with sliding-window layers the window group's rollback floor (the task doc's §5.12).
func (m *Model) kvRollbackAllowance() int {
	if kvAllowanceOverride > 0 {
		return kvAllowanceOverride
	}
	if m.w != nil && m.w.arch != nil && m.w.arch.SlidingWindow > 0 {
		return kvmgr.RollbackFloor
	}
	return kvmgr.BlockRows
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
	a.Commit(a.Bound(), m.kvRootIn(lora, m.kvNSOf(a.Bound())), ids, kvSpansOfBlocks(m.resImgBlocks), len(prompt), m.kvClassMap())
	m.kvSetLora(a.Bound(), lora)
	if m.resCkptPrompt != nil && slices.Equal(m.resCkptPrompt, prompt) {
		a.SetCheckpoint(a.Bound(), len(prompt)) // this generation saved the state at the end of this prompt
	}
	m.resCkptPrompt = nil
}

// residentPublishPrompt publishes the full blocks of prompt, which the generation in slot has just finished
// prefilling, so a request arriving while it decodes can copy them (kvmgr.Slots.Prefilled). Only a generation that
// shares the resident with others while it runs (MC3) needs it; the slot stays live until its commit or forget.
func (m *Model) residentPublishPrompt(slot int, prompt []int, lora *loraRuntime) {
	m.kvSlots().Prefilled(slot, m.kvRootIn(lora, m.kvNSOf(slot)), prompt, nil, m.kvClassMap())
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
	m.resCkptPrompt = nil
	if m.kv != nil {
		m.kv.Forget(m.kv.Bound())
	}
}

// residentForgetIDsIn is residentForgetIDs for a path that is about to write the bound slot for a request under scope
// sc, without residentAcquireIn: the commit that follows keys the slot under sc's namespace.
func (m *Model) residentForgetIDsIn(sc KVScope) {
	m.residentForgetIDs()
	m.kvSetNS(m.kvSlots().Bound(), sc.Namespace)
}
