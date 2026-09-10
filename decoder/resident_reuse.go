package decoder

import "os"

// Prefix reuse on the RESIDENT positional KV.
//
// A resident model decodes statelessly: decoder.Generate engages the resident runner only
// when there is no session commit and no prefix reuse, because a session's prefix cache is
// CPU-side while the resident KV lives on the GPU and both cannot be the source of truth. The
// consequence is that every turn re-prefills its whole prompt — measured 8.85 s for a
// 2,293-token Claude Code agent turn on a 7B int4 (docs/integrations/claude-code.md), on every
// turn, growing with the conversation.
//
// That trade is only forced for the CPU-side session cache. Doing the reuse NATIVELY on the
// GPU cache has no such conflict, and the resident cache is unusually well suited to it:
//
//   - It is POSITIONAL — token at position p lives at slot p — so "truncate to P" costs
//     nothing. residentDecoder.TruncateTo is already a no-op for exactly this reason, and
//     attention reads only nKeys = pos+1, so entries past the new length are never consulted.
//   - An agent turn is a strict PREFIX EXTENSION: turn N+1 is turn N plus the assistant's tool
//     call plus the tool result. Measured deltas in a real /v1/messages loop were 45 and 51
//     tokens against prompts of 255 and 306.
//
// So the whole mechanism is bookkeeping: remember which ids are committed to the cache, and
// prefill only the divergent suffix.
//
// CORRECTNESS IS THE ENTIRE RISK. A wrong prefix match produces confidently wrong output with
// no error anywhere — no exception, no NaN, just a reply conditioned on someone else's
// context. Four rules keep it honest:
//
//  1. Match on TOKEN IDS, never text. A client that edits its last message shifts
//     tokenisation, and the longest-common-prefix is exactly what absorbs that.
//  2. The recorded ids are cleared to nil (meaning "unknown, cold-prefill next time") on ANY
//     path that is not a fully completed generation — an error, a cancellation, a resident
//     claim lost to a concurrent generation. Conservative by construction: the failure mode of
//     forgetting is a slow turn, and the failure mode of remembering wrongly is a wrong answer.
//  3. At least one token is always prefilled, so the seed logits that start decode are always
//     freshly computed rather than assumed.
//  4. The record names the WEIGHTS as well as the ids (audit C-02, 2026-09-10). A LoRA adapter
//     changes every targeted projection, hence the residual stream, hence every later layer's
//     K/V — so an identical id prefix built under a different adapter (or none) is someone
//     else's context exactly as surely as a different prompt is. residentCommitIDs records the
//     bound *loraRuntime and residentReuseLen refuses on any mismatch, which keeps
//     adapter->same-adapter reuse (the agent-loop win) while closing every crossing. Pointer
//     identity is sound: LoadAdapter always builds a fresh runtime and registerAdapter RETIRES
//     the one it displaces rather than freeing it, so a reload under one name never compares equal.

// residentReuseDisabled is the escape hatch / A-B switch, same convention as
// GOINFER_NO_GREEDY_FASTPATH and GOINFER_NO_KVONLY_PREFILL.
func residentReuseDisabled() bool { return os.Getenv("GOINFER_NO_RESIDENT_REUSE") != "" }

// residentImageBlock records one image block committed to the resident KV: its absolute
// position span within resIDs and a content hash of the raw image bytes that produced it (P9a,
// docs/multimodal.md). Every position inside [start,end) attended every OTHER position in the
// same block under a bidirectional mask during its CPU prefill (prefillLogitsVL/
// prefillLogitsQwenVL) — there is no such thing as "half the block's KV, causally consistent
// with a differently-completed other half," so a reuse match must treat the block as one atomic
// unit: either the whole span verifies and gets skipped together, or none of it does.
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
	if residentReuseDisabled() || len(m.resIDs) == 0 || len(prompt) == 0 {
		return 0
	}
	if lora != m.resIDsLora {
		return 0 // rule 4: same ids, different weights — the KV is not this generation's
	}
	// RECURRENT FAMILIES CAN ONLY REUSE AN EXACT, STRICT EXTENSION. The three rules below (LCP
	// matching, capping at len(prompt)-1) police WHICH PREFIX of the resident KV is matched for an
	// attention-only family; none of them can help a recurrent one, because its state cannot be
	// rewound to an arbitrary earlier position at all. A Gated DeltaNet's conv ring and matrix
	// state (and Mamba-2's, and LFM2's conv window) are mutated in place per token with no
	// per-position history, and the resident path re-zeroes them only at pos == 0. So the ONLY
	// safe continuation point is exactly len(m.resIDs): the live recurrent state right now already
	// equals the state after resIDs (residentCommitIDs's invariant, held by R-00 forgetting on
	// every other writer), so a prompt that is resIDs plus at least one new token can decode
	// forward from there with no rewind needed at all — which is exactly what an agent turn is
	// (previous prompt + reply + tool result, a strict prefix extension).
	//
	// Anything else — an edited earlier message, a shorter resend, an identical resend — has no
	// safe continuation point (the state would have to run BACKWARD) and falls to 0, cold. An
	// identical resend is the qwen3.6-35B-A3B repro that motivated the original blanket refusal:
	// measured 2026-09-02, repeated identical greedy prompts diverged at token 0, differently on
	// every repeat, decaying to a one-token reply, with no error anywhere. len(prompt) <= n below
	// is exactly that case (no new token to extend with) and keeps falling to 0.
	// TestPagerDeterminism is the gate (reuse-on red before this guard, green after; still green
	// with this narrower rule since an identical resend has len(prompt) == n).
	if m.hasRecurrentState() {
		// No recurrent-family VL arch exists today; refuse rather than silently mis-serve a
		// claim the rewind-free recurrent rule below can't honour (its state has no per-position
		// history to selectively keep, so an image block's atomicity has no meaning here at all).
		if len(imgs) > 0 {
			return 0
		}
		n := len(m.resIDs)
		if len(prompt) <= n {
			return 0
		}
		for i := range n {
			if m.resIDs[i] != prompt[i] {
				return 0
			}
		}
		return n
	}
	n := min(len(prompt)-1, len(m.resIDs))
	i, bi := 0, 0 // bi: next candidate index into m.resImgBlocks (stored ascending by start)
	for i < n {
		for bi < len(m.resImgBlocks) && m.resImgBlocks[bi].end <= i {
			bi++ // fully behind us
		}
		if bi < len(m.resImgBlocks) && m.resImgBlocks[bi].start == i {
			blk := m.resImgBlocks[bi]
			if claim, ok := findImageClaim(imgs, blk.start); ok &&
				claim.Len == blk.end-blk.start && claim.Hash == blk.hash && blk.end <= n {
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
		if m.resIDs[i] != prompt[i] {
			return i
		}
		i++
	}
	return i
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

// residentCommitIDs records the exact token sequence now committed to the resident KV: the
// prompt followed by everything decode emitted, since decode writes its own K/V at each
// position as it goes. newBlock records an image block this turn added or re-verified (nil for
// plain text — the common case).
//
// Called ONLY on the fully-completed path. Everything else leaves resIDs (and resImgBlocks) nil.
func (m *Model) residentCommitIDs(prompt, generated []int, newBlock *residentImageBlock, lora *loraRuntime) {
	ids := make([]int, 0, len(prompt)+len(generated))
	ids = append(ids, prompt...)
	ids = append(ids, generated...)
	m.resIDs = ids
	m.resIDsLora = lora
	if newBlock != nil {
		kept := m.resImgBlocks[:0:0]
		for _, b := range m.resImgBlocks {
			if b.end <= newBlock.start { // still valid: append-only, earlier positions unchanged
				kept = append(kept, b)
			}
		}
		m.resImgBlocks = append(kept, *newBlock)
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
}
