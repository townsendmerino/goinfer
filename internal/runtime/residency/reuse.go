package residency

import (
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
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
// The bookkeeping is the KV manager's: one sequence per
// KV slot, its tokens, image spans, root and the kernel class that wrote each row; which slot a prompt takes; what it
// may skip; and the index of published blocks. This file is the decoder's side of it: it asks the resident what the
// manager cannot know (the slot count, the window a slot still holds, the prefill floor) and binds slots. The manager's
// sequence for the bound slot is the one record of what the resident KV holds there.

// ImageBlock records one image block committed to the resident KV: its absolute position span within the sequence and
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
type ImageBlock struct {
	Start, End int
	Hash       uint64
}

// ImageClaim describes one image block in the PROMPT being checked for reuse.
// GenerateVL/GenerateQwenVL each produce exactly one claim per call (today's one-image-per-turn
// shape), but residentReuseLen itself doesn't assume a count of one.
type ImageClaim struct {
	Start, Len int
	Hash       uint64
}

// kvSlots returns the model's KV manager, built on first use: a Model made by hand in a test has none until it asks.
func (h *Host) KVSlots() *kvmgr.Slots {
	if *h.KV == nil {
		*h.KV = kvmgr.NewSlots(h.SlotCount())
	}
	return *h.KV
}

// kvRoot is kvRootIn in the single local namespace.
func (h *Host) KVRoot(adapterID uint64) kvmgr.Root { return h.KVRootIn(adapterID, "") }

// kvRootIn is the root a sequence built under lora in namespace ns is keyed by: the model's numerics identity, the
// sharing namespace (KVScope), the adapter and the KV precision.
func (h *Host) KVRootIn(adapterID uint64, ns string) kvmgr.Root {
	r := kvmgr.Root{Namespace: ns, Adapter: adapterID, Precision: h.KVPrecision}
	if h.Arch != nil {
		a := h.Arch
		r.Model = fmt.Sprintf("%s/%dx%d/%d", a.Name, a.NumLayers, a.HiddenDim, a.VocabSize)
	}
	if h.BackendName != "" {
		r.Model += "@" + h.BackendName
	}
	return r
}

// FastPrefillMinRows is the fewest rows a pass needs to run a fast-prefill resident's fast kernels (ResidentCaps.HasFastPrefill): a pass of
// 15 rows or fewer is exact whatever the prompt's length (docs/measurements/kv0-invariance-2026-10-13.md).
const FastPrefillMinRows = 16

// kvLeadClassMap is the class rule the short-prefix guard reads: a fast-prefill resident's (ResidentCaps.HasFastPrefill).
func (h *Host) KVLeadClassMap() kvmgr.ClassMap {
	if fast, ok, _, _ := h.residentFloors(); ok {
		return FastClassMap(fast)
	}
	return kvmgr.ClassMap{}
}

// fastClassMap is the class rule of a resident with fast prefill kernels from floor prompt tokens.
func FastClassMap(floor int) kvmgr.ClassMap {
	return kvmgr.ClassMap{Floor: floor, MinRows: FastPrefillMinRows}
}

// kvClassMap is the rule for which kernel class a prefill pass of this resident writes, recorded with each committed
// row. A kernel-floor resident (ResidentCaps.HasKernelFloor) judges a pass on where it ends and has no row minimum. On a model with
// sliding-window layers a fast pass that starts off a block boundary is recorded with no producer
// (kvmgr.ClassMap.Windowed; measured on CUDA, applied to every backend with a fast class).
func (h *Host) KVClassMap() kvmgr.ClassMap {
	cm := kvmgr.ClassMap{}
	if fast, hasFast, kernel, hasKernel := h.residentFloors(); hasFast {
		cm = FastClassMap(fast)
	} else if hasKernel {
		cm = kvmgr.ClassMap{Floor: kernel, MinRows: 1}
	}
	cm.Windowed = h.Arch != nil && h.Arch.SlidingWindow > 0
	return cm
}

func KVSpansOfBlocks(blocks []ImageBlock) []kvmgr.ImageSpan {
	if len(blocks) == 0 {
		return nil
	}
	out := make([]kvmgr.ImageSpan, len(blocks))
	for i, b := range blocks {
		out[i] = kvmgr.ImageSpan{Start: b.Start, Len: b.End - b.Start, Hash: b.Hash}
	}
	return out
}

func KVSpansOfClaims(claims []ImageClaim) []kvmgr.ImageSpan {
	if len(claims) == 0 {
		return nil
	}
	out := make([]kvmgr.ImageSpan, len(claims))
	for i, c := range claims {
		out[i] = kvmgr.ImageSpan{Start: c.Start, Len: c.Len, Hash: c.Hash}
	}
	return out
}

// kvSetNS records the sharing namespace of the request that took slot; the commit that ends it keys the slot's
// sequence under it.
func (h *Host) KVSetNS(slot int, ns string) {
	for len((*h.KVNS)) <= slot {
		(*h.KVNS) = append((*h.KVNS), "")
	}
	(*h.KVNS)[slot] = ns
}

// kvNSOf is the namespace recorded for slot ("" when none was).
func (h *Host) KVNSOf(slot int) string {
	if slot < len((*h.KVNS)) {
		return (*h.KVNS)[slot]
	}
	return ""
}

// residentReuseLen returns how many leading tokens of prompt the BOUND slot already holds and may skip. imgs
// describes any image blocks in prompt (nil for plain text). The rules are kvmgr.Slots.Score's. lora is the adapter
// bound for this generation (nil = base weights). A caller that takes this length without going through
// residentAcquireSlot has not told the manager what it will write, so the commit that follows names no producer.
func (h *Host) ReuseLen(prompt []int, imgs []ImageClaim, adapterID uint64) int {
	return h.ReuseLenIn(prompt, imgs, adapterID, KVScope{})
}

// residentReuseLenIn is residentReuseLen for a request under scope sc that takes the bound slot as it is: the caller
// prefills into it and commits there.
func (h *Host) ReuseLenIn(prompt []int, imgs []ImageClaim, adapterID uint64, sc KVScope) int {
	n := h.BoundReuseLen(prompt, imgs, adapterID, sc.Namespace, false)
	h.KVSetNS(h.KVSlots().Bound(), sc.Namespace)
	return n
}

// boundReuseLen scores the bound slot for prompt; ckpt lets a recurrent sequence's checkpoint score (kvmgr.Request.Checkpoint).
// A live slot's announcement is a running generation's, so it is not marked unrecorded.
func (h *Host) BoundReuseLen(prompt []int, imgs []ImageClaim, adapterID uint64, ns string, ckpt bool) int {
	a := h.KVSlots()
	if !a.Live(a.Bound()) {
		a.Unrecorded(a.Bound())
	}
	return h.ReuseFloorFor(h.ReuseLenOf(a.Bound(), prompt, imgs, adapterID, ns, ckpt))
}

// ResidentKVPool reports the resident's KV pool (KVPool), the zero value when it has none.
func (h *Host) KVPool() resident.KVPoolDetail {
	return h.resKV().KVPool()
}

// KVShareState says whether a request on this resident can take KV rows another conversation computed, by residentShare's own tests:
// "on"; "off" when the operator turned it off (--kv-share off, GOINFER_NO_RESIDENT_REUSE), with why; or "none" when this load cannot
// share at all (off the resident path, one KV slot, a backend that copies no rows, recurrent state), with why.
func (h *Host) KVShareState() (state, why string) {
	switch {
	case h.Resident == nil:
		return "none", "not resident"
	case h.SlotCount() <= 1:
		return "none", "one KV slot"
	case h.hasRecurrentState():
		return "none", "recurrent state cannot be copied"
	}
	if !h.resKV().KVCopierAvailable() {
		return "none", "this backend copies no KV rows"
	}
	switch {
	case h.ShareOff:
		return "off", "--kv-share off"
	case h.ReuseDisabled:
		return "off", "GOINFER_NO_RESIDENT_REUSE"
	}
	return "on", ""
}

// KVCopyDecline is CopyKV's refusal of a range, with the reason; dst is untouched.
type KVCopyDecline struct{ Reason string }

func (d *KVCopyDecline) Error() string { return "kv copy declined: " + d.Reason }

// residentPromote brings the bound slot's demoted rows back when this request reuses them. A failure leaves nothing of the slot's to
// trust, so the request prefills cold.
func (h *Host) Promote(acq *KVAcquired) {
	kv := h.resKV()
	if !kv.KVDemoterAvailable() || acq.Slot < 0 || acq.Reuse <= 0 {
		return
	}
	blocks, err := kv.PromoteKV(acq.Slot, acq.Reuse)
	if err != nil {
		acq.Declined, acq.Reuse, acq.Shared = err.Error(), 0, 0
		h.ForgetIDs()
		return
	}
	if blocks > 0 {
		acq.PromotedBlocks, acq.PromotedRows = acq.PromotedBlocks+blocks, acq.Reuse
		(*h.KVSnaps).Promoted(blocks)
		fmt.Fprintf(h.stderr(), "kv host: promoted %d blocks into slot %d; the request reuses %d rows\n", blocks, acq.Slot, acq.Reuse)
	}
}

// residentReuseFloor caps a reuse length of the BOUND slot at what its resident can serve (ReusableKV); a resident that keeps every
// position is untouched. A cap only ever lengthens the prefill, never changes what it computes.
func (h *Host) ReuseFloorFor(reuse int) int {
	if reuse <= 0 {
		return reuse
	}
	return kvmgr.SafeReuseLen(reuse, reuse+1, h.KVSlots().State(h.KVSlots().Bound()).Floor(), h.resKV().ReusableKV)
}

// reuseLenOf is how many leading tokens of prompt the sequence in slot holds, the bound slot's or a parked one's,
// before the resident's own floor.
func (h *Host) ReuseLenOf(slot int, prompt []int, imgs []ImageClaim, adapterID uint64, ns string, ckpt bool) int {
	if h.ReuseDisabled {
		return 0
	}
	req := kvmgr.Request{Root: h.KVRootIn(adapterID, ns), Tokens: prompt, Images: KVSpansOfClaims(imgs), Checkpoint: ckpt}
	return h.KVSlots().Score(slot, req, h.hasRecurrentState())
}

// residentSlotCount is how many resident KV slots a generation may choose among: what the resident allocated
// (KVSlotterAvailable), else 1. A family with recurrent state keeps one unless the resident's backend keeps that state
// once per slot (recurrentSlotsOn): otherwise its state is mutated in place outside any slot, so two conversations could
// not both keep theirs.
func (h *Host) SlotCount() int {
	kv := h.resKV()
	if !kv.KVSlotterAvailable() || (h.hasRecurrentState() && !h.RecurrentSlotsOK) {
		return 1
	}
	return max(1, kv.KVSlots())
}

// residentAcquire binds the resident KV slot a generation over prompt will use and returns how many leading tokens of
// prompt that slot already holds (residentReuseLen's contract, for the slot it bound). With one slot it is
// residentReuseLen. With several it scores every slot and binds PickResidentSlot's choice. Call it where
// residentReuseLen was called: under the resident claim, before the forget.
func (h *Host) Acquire(prompt []int, imgs []ImageClaim, adapterID uint64) int {
	return h.AcquireIn(prompt, imgs, adapterID, nil, KVScope{}).Reuse
}

// residentAcquireSlot is residentAcquire that also returns the slot it bound, and never picks a slot busy marks (under the
// batched decode, another running generation's). With every slot busy it binds nothing and returns slot -1. It tells the manager
// what the generation will do in the slot (skip reuse rows, prefill the rest of prompt), which the commit reads.
func (h *Host) AcquireSlot(prompt []int, imgs []ImageClaim, adapterID uint64, busy []bool) (reuse, slot int) {
	acq := h.AcquireIn(prompt, imgs, adapterID, busy, KVScope{})
	return acq.Reuse, acq.Slot
}

// KVAcquired is what residentAcquireIn did: the rows the generation skips, the slot it bound (-1: none), and how many
// of the skipped rows were copied from another slot, from which, at what cost, or why a copy was declined.
type KVAcquired struct {
	Reuse, Slot                  int
	Shared, Src                  int
	CopyDur                      time.Duration
	Declined                     string
	PromotedBlocks, PromotedRows int
	SnapRows                     int
	SnapDur                      time.Duration
	snapFrom                     *kvSnap
	MarkFired                    bool
	MsgRows                      int
	PreambleRows                 int
	SlotLen, Match               int
	CkptRestored                 int
	CkptDur                      time.Duration
	CkptDeclined                 string
}

// residentAcquireIn is residentAcquireSlot for a request under scope sc. When another slot's published blocks hold
// more of the prompt than the bound slot does, it copies them into the bound slot (KVCopierAvailable) and skips them.
func (h *Host) AcquireIn(prompt []int, imgs []ImageClaim, adapterID uint64, busy []bool, sc KVScope) KVAcquired {
	acq := h.AcquireSlotAll(prompt, imgs, adapterID, busy, sc)
	if imgs == nil {
		h.RestoreSnapshot(prompt, adapterID, sc, &acq)
	}
	// the slot is bound now, so its window is the one to ask (a parked slot's score was not floored when it was counted)
	acq.Reuse = h.ReuseFloorFor(h.CapInPlaceReuse(h.DeclineShortLeadReuse(prompt, acq.Reuse), prompt, acq))
	if acq.Reuse < acq.Shared { // the bound slot's window cannot serve the copied rows: none of them count
		acq.Shared = 0
	}
	h.Promote(&acq)
	if acq.Slot >= 0 {
		a := h.KVSlots()
		if sq := a.Seq(a.Bound()); sq != nil {
			acq.SlotLen, acq.Match = sq.Len(), kvmgr.CommonPrefixLen(sq.Tokens(), prompt)
		}
		if CheckpointEnabled(sc) {
			h.CheckpointRestore(prompt, &acq) // before the manager records what the request skips
		}
		a.Acquired(a.Bound(), acq.Reuse, prompt)
		h.KVSetNS(a.Bound(), sc.Namespace)
	}
	return acq
}

// MinLeadReuseFast is the shortest prefix worth reusing under a resident with non-exact prefill kernels
// (ResidentCaps.HasFastPrefill), for a prompt that will run them (kvmgr.MinLeadReuse). Not an Options field: it is a
// correctness margin, not an operator choice.
const MinLeadReuseFast = kvmgr.MinLeadReuse

// declineShortLeadReuse zeroes a short reuse when the resident prefills this prompt on non-exact kernels, so the prompt is prefilled whole by one kernel class
// (kvmgr.ClassMap.DeclineShortLead). Every caller forgets the slot's ids and prefills from the returned index, so 0 simply means a cold prefill into the slot it
// already bound.
func (h *Host) DeclineShortLeadReuse(prompt []int, reuse int) int {
	return h.KVLeadClassMap().DeclineShortLead(len(prompt), reuse)
}

// capInPlaceReuse caps a reuse of the bound slot's own rows so the prompt's remaining pass runs the kernel class its cold
// prefill runs (kvmgr.ClassMap.CapInPlace). Copied and restored rows were capped where they were chosen and pass through. A
// recurrent model's state cannot be rewound to an earlier row, so its reuse is never capped.
func (h *Host) CapInPlaceReuse(reuse int, prompt []int, acq KVAcquired) int {
	if h.InPlaceCapOff || acq.Shared > 0 || acq.SnapRows > 0 || acq.Slot < 0 || h.hasRecurrentState() {
		return reuse
	}
	a := h.KVSlots()
	sq := a.Seq(a.Bound())
	if sq == nil {
		return reuse
	}
	return h.KVClassMap().CapInPlace(sq.Classes(), sq.Images(), len(prompt), reuse)
}

func (h *Host) AcquireSlotAll(prompt []int, imgs []ImageClaim, adapterID uint64, busy []bool, sc KVScope) KVAcquired {
	n := h.SlotCount()
	a := h.KVSlots()
	a.Resize(n)
	if n <= 1 {
		return KVAcquired{Reuse: h.BoundReuseLen(prompt, imgs, adapterID, sc.Namespace, CheckpointEnabled(sc)), Src: -1}
	}
	scores, lens, firsts, uses := make([]int, n), make([]int, n), make([]int, n), make([]uint64, n)
	bound := a.Bound()
	if bound >= len(busy) || !busy[bound] { // a busy slot is never picked, and its record is another generation's
		scores[bound] = h.BoundReuseLen(prompt, imgs, adapterID, sc.Namespace, CheckpointEnabled(sc) && h.CheckpointHeld(bound)) // adopts the bound record first, and asks its window
	}
	for i := range n {
		if i != bound {
			scores[i] = h.ReuseLenOf(i, prompt, imgs, adapterID, sc.Namespace, CheckpointEnabled(sc) && h.CheckpointHeld(i))
		}
		lens[i], uses[i] = a.Len(i), a.Use(i)
		if a.KeepsFirst(i, prompt) {
			firsts[i] = a.FirstPrompt(i)
		}
	}
	if h.AllowanceOverride > 0 {
		firsts = nil
	}
	pick := kvmgr.PickSlotFor(scores, lens, firsts, uses, busy, h.KVRollbackAllowance())
	if pick < 0 {
		return KVAcquired{Slot: -1, Src: -1}
	}
	if err := h.Bind(pick); err != nil {
		// The resident keeps its previous binding; with its bookkeeping unknown, go cold on it. Slot -1: a batched caller
		// must not decode on a binding it did not choose (it may be another generation's slot); residentAcquire's
		// callers carry on cold on the previous binding, as they always have.
		h.ForgetIDs()
		return KVAcquired{Slot: -1, Src: -1}
	}
	a.Touch(pick)
	acq := KVAcquired{Reuse: scores[pick], Slot: pick, Src: -1}
	h.Share(prompt, imgs, adapterID, sc, &acq)
	return acq
}

// KVChooseDonor picks the slot to copy from: of the slots other than dst whose capped hit beats reuse, the longest hit
// (the lower slot on a tie) that the resident says still holds the rows (ReusableKVSlotAvailable; a resident without it
// holds every row it has). src is -1 when none beats reuse, and when every candidate's window has moved past its rows,
// in which case declined says so.
func KVChooseDonor(res resident.Forward, hits []int, cm kvmgr.ClassMap, n, dst, reuse int) (src, hit int, declined string) {
	type cand struct{ slot, hit int }
	var cands []cand
	for s, h := range hits {
		if h = cm.CapHit(n, h); s != dst && h > reuse {
			cands = append(cands, cand{s, h})
		}
	}
	slices.SortStableFunc(cands, func(a, b cand) int { return b.hit - a.hit })
	kv := ResolveSubsystems(res).KV
	windowed := kv.ReusableKVSlotAvailable()
	for _, c := range cands {
		if !windowed || kv.ReusableKVSlot(c.slot, c.hit) >= c.hit {
			return c.slot, c.hit, ""
		}
	}
	if len(cands) > 0 {
		return -1, 0, fmt.Sprintf("no source slot's window still holds the rows (%d candidates)", len(cands))
	}
	return -1, 0, ""
}

// KVOtherDonors states, for a declined copy, how many slots other than src and dst hold a hit longer than reuse that their
// windows still hold. It changes nothing; it is the diagnostic of how much of a decline is the choice of donor
// (docs/measurements/kv2-cuda-2026-10-15.md, §6).
func KVOtherDonors(rk resident.KVSubsystem, hits []int, cm kvmgr.ClassMap, n, src, dst, reuse int) string {
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

// residentShare copies into the bound slot acq.Slot the longest run of prompt another slot's published blocks hold
// (kvmgr.Slots.Hits, capped by kvmgr.ClassMap.CapHit), when that is more than acq.Reuse, the rows the slot itself
// holds. It does nothing under an Off scope or Options.KVShareOff, for a recurrent family, without KVCopierAvailable, or when the source's
// window no longer holds the rows. A decline or an error leaves acq.Reuse at what the slot can still trust.
func (h *Host) Share(prompt []int, imgs []ImageClaim, adapterID uint64, sc KVScope, acq *KVAcquired) {
	kv := h.resKV()
	if !kv.KVCopierAvailable() || sc.Off || h.ShareOff || h.ReuseDisabled || h.hasRecurrentState() {
		return
	}
	a, dst, cm := h.KVSlots(), acq.Slot, h.KVClassMap()
	hits := a.Hits(kvmgr.Request{Root: h.KVRootIn(adapterID, sc.Namespace), Tokens: prompt, Images: KVSpansOfClaims(imgs)}, cm.ColdClasses(len(prompt)))
	src, hit, declined := KVChooseDonor(h.Resident, hits, cm, len(prompt), dst, acq.Reuse)
	if src < 0 {
		acq.Declined = declined
		return
	}
	start := min(hits[dst], hit) // dst already holds these as the same published blocks
	t0 := time.Now()
	held := false
	var err error
	if kv.KVDemoterAvailable() {
		var blocks int
		if blocks, held, err = kv.CopyDemotedKV(src, dst, start, hit); blocks > 0 {
			acq.PromotedBlocks += blocks
			(*h.KVSnaps).Promoted(blocks)
			fmt.Fprintf(h.stderr(), "kv host: wrote %d blocks of slot %d's copy into slot %d\n", blocks, src, dst)
		}
	}
	if !held && err == nil {
		err = kv.CopyKV(src, dst, start, hit)
	}
	acq.CopyDur = time.Since(t0)
	if err != nil {
		if d, ok := errors.AsType[*KVCopyDecline](err); ok {
			acq.Declined = d.Reason
			if kv.ReusableKVSlotAvailable() {
				acq.Declined += KVOtherDonors(kv, hits, cm, len(prompt), src, dst, acq.Reuse)
			}
		} else {
			acq.Declined = err.Error()
			acq.Reuse = min(acq.Reuse, start)
		}
		return
	}
	a.Copied(src, dst, start, hit)
	acq.Reuse, acq.Shared, acq.Src = hit, hit, src
}

// residentBind binds resident KV slot `slot`, whose sequence in the manager becomes the bound record. A no-op when it is
// already bound. Switching slots also clears resDrafterSynced: a block drafter's own context
// follows ONE conversation, the one the previous slot was serving.
func (h *Host) Bind(slot int) error {
	a := h.KVSlots()
	if slot == a.Bound() || h.SlotCount() <= 1 {
		return nil
	}
	if err := h.resKV().UseKVSlot(slot); err != nil {
		return err
	}
	a.Resize(h.SlotCount())
	a.SetBound(slot)
	if h.DrafterSynced != nil {
		*h.DrafterSynced = nil
	}
	return nil
}

// kvRollbackAllowance is how many rows a request may give up of a slot it takes in place (kvmgr.PickSlotWithin): one
// block, or for a model with sliding-window layers the window group's rollback floor (the task doc's §5.12).
func (h *Host) KVRollbackAllowance() int {
	if h.AllowanceOverride > 0 {
		return h.AllowanceOverride
	}
	if h.Arch != nil && h.Arch.SlidingWindow > 0 {
		return kvmgr.RollbackFloor
	}
	return kvmgr.BlockRows
}

// PickResidentSlot chooses a slot for a prompt from each slot's reuse score, its committed length and its last-use
// tick (kvmgr.PickSlot): a continuation keeps its slot; a shared lead takes an empty slot, else the least recently
// used. busy (nil: none) marks slots another running generation holds (the batched decode); -1 when all are busy.
func PickResidentSlot(scores, lens []int, uses []uint64, busy []bool) int {
	return kvmgr.PickSlot(scores, lens, uses, busy)
}

// residentCommitIDs records the exact token sequence now committed to the resident KV: the prompt followed by
// everything decode emitted, since decode writes its own K/V at each position as it goes. newBlocks records the image
// blocks this turn added or re-verified, in prompt order (nil for plain text; several for a multi-image turn).
//
// Called only on the fully completed path, which is what lets the manager publish the sequence's full blocks here.
// Everything else leaves the bound slot forgotten.
func (h *Host) CommitIDs(prompt, generated []int, newBlocks []ImageBlock, adapterID uint64) {
	ids := make([]int, 0, len(prompt)+len(generated))
	ids = append(ids, prompt...)
	ids = append(ids, generated...)
	a := h.KVSlots()
	var images []kvmgr.ImageSpan
	if s := a.Seq(a.Bound()); s != nil {
		images = s.Images()
	}
	if len(newBlocks) > 0 {
		var kept []kvmgr.ImageSpan
		for _, sp := range images {
			if sp.Start+sp.Len <= newBlocks[0].Start { // still valid: append-only, earlier positions unchanged
				kept = append(kept, sp)
			}
		}
		images = append(kept, KVSpansOfBlocks(newBlocks)...)
	}
	a.Commit(a.Bound(), h.KVRootIn(adapterID, h.KVNSOf(a.Bound())), ids, images, len(prompt), h.KVClassMap())
	if (*h.CkptPrompt) != nil && slices.Equal((*h.CkptPrompt), prompt) {
		a.SetCheckpoint(a.Bound(), len(prompt)) // this generation saved the state at the end of this prompt
	}
	(*h.CkptPrompt) = nil
}

// residentPublishPrompt publishes the full blocks of prompt, which the generation in slot has just finished
// prefilling, so a request arriving while it decodes can copy them (kvmgr.Slots.Prefilled). Only a generation that
// shares the resident with others while it runs (the batched decode) needs it; the slot stays live until its commit or forget.
func (h *Host) PublishPrompt(slot int, prompt []int, adapterID uint64) {
	h.KVSlots().Prefilled(slot, h.KVRootIn(adapterID, h.KVNSOf(slot)), prompt, nil, h.KVClassMap())
}

// residentForgetIDs marks the resident KV's contents unknown. Anything that writes the cache
// outside a completed generateInto — a failed prefill, a cancelled decode, a batched verify —
// must call this, because a stale id list is the one way this feature can be WRONG rather than
// merely slow. The sequence goes with its image blocks, always — an image block's identity is
// meaningless once the token sequence backing its position is no longer trusted — and its
// blocks leave the manager's index with it.
func (h *Host) ForgetIDs() {
	if h.DrafterSynced != nil {
		*h.DrafterSynced = nil
	} // any resident write invalidates a block drafter's own reuse too
	(*h.CkptPrompt) = nil
	if *h.KV != nil {
		(*h.KV).Forget((*h.KV).Bound())
	}
}

// residentForgetScratch prepares the resident for a call that writes from position 0 and keeps nothing (HiddenLast,
// PromptHidden, PromptHiddenAll): it binds an empty slot, else the least recently used one, and forgets that slot's
// record, so a chat parked in another slot keeps its rows for its next turn. With one slot the chat's rows are lost to the
// call; its next turn prefills cold. A failed bind leaves the previous binding, forgotten. Call it under the exclusive
// resident claim (tryClaimResident).
func (h *Host) ForgetScratch() {
	if n := h.SlotCount(); n > 1 {
		a := h.KVSlots()
		a.Resize(n)
		lens, uses := make([]int, n), make([]uint64, n)
		for i := range n {
			lens[i], uses[i] = a.Len(i), a.Use(i)
		}
		if pick := PickResidentSlot(make([]int, n), lens, uses, nil); pick >= 0 && !h.ScratchSlotPlant {
			_ = h.Bind(pick) // on an error the resident keeps its binding, which the forget below makes cold
		}
	}
	h.ForgetIDs()
}

// residentWritesFromZero says the bound slot's next write starts at position 0, so no rows stay missing below a wrapped upload's floor
// (kvmgr.Slots.SetFloor). Its caller has forgotten the slot's record, so a write that fails part way grants nothing. A path that
// writes from 0 and does not say so only keeps a floor it could have dropped: the slot re-prefills a prefix it could have reused.
func (h *Host) WritesFromZero() {
	a := h.KVSlots()
	a.SetFloor(a.Bound(), 0)
}

// residentForgetIDsIn is residentForgetIDs for a path that is about to write the bound slot for a request under scope
// sc, without residentAcquireIn: the commit that follows keys the slot under sc's namespace.
func (h *Host) ForgetIDsIn(sc KVScope) {
	h.ForgetIDs()
	h.KVSetNS(h.KVSlots().Bound(), sc.Namespace)
}
