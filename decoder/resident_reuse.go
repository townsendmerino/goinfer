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
// context. Three rules keep it honest:
//
//  1. Match on TOKEN IDS, never text. A client that edits its last message shifts
//     tokenisation, and the longest-common-prefix is exactly what absorbs that.
//  2. The recorded ids are cleared to nil (meaning "unknown, cold-prefill next time") on ANY
//     path that is not a fully completed generation — an error, a cancellation, a resident
//     claim lost to a concurrent generation. Conservative by construction: the failure mode of
//     forgetting is a slow turn, and the failure mode of remembering wrongly is a wrong answer.
//  3. At least one token is always prefilled, so the seed logits that start decode are always
//     freshly computed rather than assumed.

// residentReuseDisabled is the escape hatch / A-B switch, same convention as
// GOINFER_NO_GREEDY_FASTPATH and GOINFER_NO_KVONLY_PREFILL.
func residentReuseDisabled() bool { return os.Getenv("GOINFER_NO_RESIDENT_REUSE") != "" }

// residentReuseLen returns how many leading tokens of prompt are already committed to the
// resident KV and may be skipped.
//
// Capped at len(prompt)-1 on purpose: generateInto's contract is that prefill covers at least
// one token, whose logits seed decode. Returning len(prompt) would leave the caller with no
// seed logits and nothing to recompute them from.
func (m *Model) residentReuseLen(prompt []int) int {
	if residentReuseDisabled() || len(m.resIDs) == 0 || len(prompt) == 0 {
		return 0
	}
	// RECURRENT FAMILIES REUSE NOTHING. The three rules below all police WHICH PREFIX is matched;
	// none of them can help a model whose state cannot be rewound to that prefix at all. A Gated
	// DeltaNet's conv ring and matrix state (and Mamba-2's, and LFM2's conv window) are mutated in
	// place per token with no per-position history, and the resident path re-zeroes them only at
	// pos == 0 — which a reused prefix never reaches. So the second generation would decode from
	// the first one's TAIL state.
	//
	// That is not hypothetical: measured 2026-09-02 on qwen3.6-35B-A3B, repeated identical greedy
	// prompts diverged at token 0, differently on every repeat, decaying to a one-token reply, with
	// no error anywhere — the failure this file's own header calls out as the whole risk.
	// TestPagerDeterminism is the gate (reuse-on red before this guard, green after).
	//
	// This costs nothing for attention-only families, which is most of what an agent loop runs.
	// Getting reuse BACK for recurrent families needs a state checkpoint at the reuse point — L-05's
	// job, not a patch here. The predicate is the shared one so a fourth state kind is added in the
	// dispatch table once rather than missed here for a ninth time.
	if m.hasRecurrentState() {
		return 0
	}
	n := len(m.resIDs)
	if len(prompt)-1 < n {
		n = len(prompt) - 1
	}
	i := 0
	for i < n && m.resIDs[i] == prompt[i] {
		i++
	}
	return i
}

// residentCommitIDs records the exact token sequence now committed to the resident KV: the
// prompt followed by everything decode emitted, since decode writes its own K/V at each
// position as it goes.
//
// Called ONLY on the fully-completed path. Everything else leaves resIDs nil.
func (m *Model) residentCommitIDs(prompt, generated []int) {
	ids := make([]int, 0, len(prompt)+len(generated))
	ids = append(ids, prompt...)
	ids = append(ids, generated...)
	m.resIDs = ids
}

// residentForgetIDs marks the resident KV's contents unknown. Anything that writes the cache
// outside a completed generateInto — a failed prefill, a cancelled decode, a batched verify —
// must call this, because a stale id list is the one way this feature can be WRONG rather than
// merely slow.
func (m *Model) residentForgetIDs() { m.resIDs = nil }
