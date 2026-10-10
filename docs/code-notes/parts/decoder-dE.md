# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## NgramDrafter

Moved from `decoder/spec_ngram.go` (the comment above `NgramDrafter`) on 2026-10-10.

```text
NgramDrafter is a zero-model "prompt-lookup" drafter (02-cache-ngram): it finds
the most recent earlier occurrence of the current token suffix within the
running context (prompt + everything generated so far) and proposes the tokens
that followed it. It costs no model compute and works on a single model — no
separate draft model is needed. It earns its keep when the output echoes the
input (code edits, RAG, agent loops with a fixed system prompt); on novel prose
it simply misses, which is free.

This is the deliberate "n-gram hash" baseline from the design doc, not yet the
suffix automaton — start simple, measure the gap. Matching is exact-suffix and
greedy-longest: try the longest pattern first, take the most recent earlier hit.
```

## ngramAlphaAnchors

Moved from `decoder/spec_ngram.go` (the comment above `ngramAlphaAnchors`) on 2026-10-10.

```text
ngramAlphaAnchors is α̂_ngram(match_len) — the calibrated acceptance probability by
suffix match length, fit (monotone non-decreasing, noise-smoothed) from the §06
trace measurement (TestNgramAlphaPredictor: AUC 0.818 for match_len → accept; copy-
heavy workloads, qwen2.5-coder-0.5b). The measured buckets were len2→0.70, len3→0.86,
len4→0.83 (noise), len11→0.92, len16(capped)→0.97. This is the tiny "coefficient
table" the §06 boundary permits in the pure-Go runtime — fit offline, evaluated as a
cheap interpolation on the hot path. Re-fit when a new family/workload lands (§06 §9);
an online per-source rate is the documented drift fallback.
```

## specRoundDraftWidth

Moved from `decoder/spec_ngram.go` (the comment above `specRoundDraftWidth`) on 2026-10-10.

```text
specRoundDraftWidth clamps a speculative round's draft width (k proposed tokens, verified
together with `cur` at absolute position pos as a [1+k]-row batch) to what's left under the
resident context cap — the shared M-03 (docs/audit-2026-09-10.md) shape used by both
genNgramInto (this file) and GenerateSpeculative (speculative.go), mirroring
blockSpecRoundWidth's (blockspec.go) identical M-13 fix for block speculation. Returns k
unchanged when ctxCap doesn't bind (<=0) or there's already room. Returns -1 when there is no
room even for `cur` alone (pos >= ctxCap) — the caller's signal to stop cleanly rather than
attempt a verify round, exactly like blockSpecRoundWidth's own width<1 case.
```

## Model.genNgramInto.claim

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.claim`) on 2026-10-10.

```text
The resident path drives the model's ONE shared positional KV; two concurrent generations would
interleave writes at overlapping positions and corrupt it. Claim it non-blockingly (mirroring
generateInto's M9 guard) — a loser falls back to the staged CPU path with this call's own cache,
so both still complete correctly, only the loser loses resident speed. Released on return.

Through the batcher's claimExclusive when one exists, not a bare CAS on resBusy: a plain
concurrent generation becomes an MC3 holder via bt.claim, which only READS resBusy (it never
sets it) — a bare CAS here would succeed at the same time a holder is mid-step, corrupting the
resident. claimExclusive additionally requires holders == 0, which bt.claim's own bookkeeping
makes correct. Production never exercises this combination (-spec without -spec-adaptive forces
concurrency to 1, so a spec generation and an MC3 holder never coexist), but the guard should not
depend on that — found by TestSpecAdaptiveSwitch_offByDefaultUnaffected, which deliberately does.
```

## Model.genNgramInto.argmax

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.argmax`) on 2026-10-10.

```text
ARGMAX-ONLY VERIFY (greedy, resident, no tracer): the accept decision compares each draft token with the target's argmax, so the full logits row
per verified row — 608 KB of device-to-host at a 152k vocab, plus a host argmax over it — is pure overhead. A one-row round (no draft, i.e. a plain decode
step) takes the same device-argmax fast path Model.Generate uses (ResidentGreedy: a 4-byte readback); a multi-row round takes the batched argmax pass block
speculation already verifies with (PrefillLastNArgmax). The ids equal the logits path's argmax (TestPrefillLastNArgmax_matchesPerRow; the M=1 path is the one
plain decode takes), so output is unchanged. Any error falls back to the full-logits ForwardN once and turns this path off, so a backend whose batched pass
declines (MoE) keeps working. nil when not applicable.
```

## Model.genNgramInto.prefill

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.prefill`) on 2026-10-10.

```text
Shared with generateInto (model.go). This used to be its own per-token
Forward loop, which is how it missed batched prefill entirely when that
landed in c36698a — 6.3x per prompt token, and `off` beating speculation
3-4.5x on realistic prompts. See residentPrefillSeed's comment.

P-05 (audit-2026-09-10): R-03 already made this function COMMIT the accepted
sequence correctly on exit (target.residentCommitIDs below), but until now nothing
on ENTRY ever consulted what that commit left behind — every round cold-prefilled
the whole prompt from 0 regardless, so a --spec/--drafter agent loop got no prefix
reuse at all despite committing one every round. residentReuseLen answers the exact
same question generateInto already asks before its own forget (model.go) — it is
safe here for the same reason: resIDs is accurate BECAUSE the commit below keeps it
so, and this is the entry moment, not the "verify writes positions this function
owns" moment the forget below is about.

Speculative verify (this round) writes the resident KV at positions this function
owns, not generateInto's, so the recorded id list stops being an accurate
description of PAST-THIS-POINT state as soon as verify runs. Forget it AFTER
computing reuseFrom, not before: the next turn's commit is what makes it true
again, and forgetting first would only lose the reuse this round could have had —
the same ordering generateInto already uses (resident_reuse.go).
```

## Model.genNgramInto.slot

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.slot`) on 2026-10-10.

```text
Our own slot, reserved for the whole generation (not re-acquired per round): MC3's
busySlots() keeps this pick from colliding with another generation's. residentAcquireSlot
itself calls residentBind (picking a slot rebinds the resident to it immediately), so it
needs the resident to itself exactly like generateInto's own MC3 prefill does (model.go)
— the slot pick and the forget below are NOT safe bare, only prefillExclusive's own
residentPrefillSeed call was protected before this fix.
```

## Model.genNgramInto.rebind

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.rebind`) on 2026-10-10.

```text
Re-bind: the slot-acquisition exclusive() section above already ended, so another
generation's own exclusive section or batched run could have rebound the resident to
a DIFFERENT slot in between (mc3Prefill, model.go, does this same rebind for exactly
this reason — it never assumes a slot stays bound across two separate exclusive
sections). Skipping this was the bug TestSpecAdaptiveSwitch_matchesPlainDecode caught
intermittently (~1 run in 10-20): the prefill silently wrote another generation's slot.
```

## Model.genNgramInto.commitResident

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.commitResident`) on 2026-10-10.

```text
audit R-03: every return below that follows a successful (or stop/cancelled) emit —
as opposed to a targetVerify/prefill error, which can leave a partial write and must
stay forgotten (resIDs is already nil from the forget above) — happens at a point
where hist is exactly what the resident cache holds, NEVER ahead of it: targetVerify's
ForwardN writes a round's [cur, accepted draft…] K/V before this loop streams any of
them, and hist is appended in the same step. The one asymmetry is the round's OWN
trailing token: `cur = nextTok` is streamed here but not forwarded until the START of
the NEXT round (as that round's targetVerify seq[0]), so a return right after that
specific emit leaves hist one token BEHIND what was just streamed — safe (the cache
is never claimed to hold more than it does), just one token short of the reuse this
round could otherwise offer, at exactly the exit that also ends the whole generation.
Recurrent families never reach here at all (validateNgramSpec's specRollbackSafe check
rejects them before this goroutine starts), so there is no forget-otherwise branch to
add, unlike R-01/R-03's general shape.
```

## Model.genNgramInto.finishTrailing

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.finishTrailing`) on 2026-10-10.

```text
finishTrailing forwards the round's trailing token when the generation ended by reaching maxTokens just
after streaming it. Plain decode forwards every token it emits, the last included, so its cache holds
prompt + every emitted token. Here a round's trailing token is otherwise forwarded only as the NEXT
round's seq[0], so the cache ended one token short of plain's.

Left short, the next turn re-prefills that one position. On a backend whose batched prefill is not
bit-identical to decode, that changes the next turn's output: Metal's f16-MMA prefill runs whenever the
whole prompt is past its floor, so every W7 turn after the first diverged from plain decode
(docs/measurements/spec-vs-batching-metal-2026-09-27.md §4). The forward is the one a one-row round would
make, and a stop, a cancel or a full context takes no forward, as in plain decode.
```

## Model.genNgramInto.depth0

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.depth0`) on 2026-10-10.

```text
When the depth controller is already at 0 for any proposal (its acceptance estimate says even one node is not worth it, and no probe is due),
the round is a plain decode step, so the drafter's scan of the whole context (88 us at 4.9k tokens, growing with it) is skipped.
```

## Model.genNgramInto.verifyBusy

Moved from `decoder/spec_ngram.go` (the comment above `Model.genNgramInto.verifyBusy`) on 2026-10-10.

```text
resBusy (claimExclusive above) keeps MC3's HOLDERS off the resident, but it is not the flag another
generation's own exclusive sections wait on: the slot pick, the prefill and the commit all run under
the batcher's busy flag (exclusive / prefillExclusive) and never read resBusy. Verifying here without
busy let that generation's residentBind-then-prefill interleave with this round's residentBind-then-
ForwardN, and each wrote the other's slot (first token already wrong, then garbage, from two
generations on; MC4 graded run 2026-10-01, docs/measurements/mc4-candidate-cuda-2026-10-01.md). The
round therefore takes busy too, as the design said it would ("each round runs in exclusive").
```

## SamplingParams.LogitProcessorGate

Moved from `decoder/sampler.go` (the comment above `SamplingParams.LogitProcessorGate`) on 2026-10-10.

```text
LogitProcessorGate, when set together with LogitProcessor, makes the processor LAZY: it is
asked after every emitted token (with the ids generated so far) whether the NEXT step needs
the processor. While it answers false the processor is not called and the step keeps every
on-device fast path (greedy argmax, device sampling, device top-K) exactly as if there were
no processor; a step it answers true for reads the full logits and runs the processor. It
exists for a constraint that only switches on partway through a turn (a tool call under
tool_choice auto, constrain.LazyMasker): an ungated processor costs every token of the turn
its fast path, prose included — measured 0.81x decode at T=0.7 on a 1.5B
(docs/measurements/tool-union-2026-09-24.md). The gate must be a pure function of generated
ids. It is consulted only on the decode loop in model.go; the speculative paths still refuse
any LogitProcessor, gated or not, and callers fall back to Generate as before.
```

## NewSampler.minp

Moved from `decoder/sampler.go` (the comment above `NewSampler.minp`) on 2026-10-10.

```text
M-08: MinP > 1 asks for "keep tokens with probability ≥ 1.2× the maximum", which is
nothing — the threshold maxL + T·ln(minP) sits ABOVE maxL, so the candidate set comes
back empty and the always-keep-the-top-token clamps then index ips[:1] on an empty
slice or ips[-1]. Measured before the fix: `index out of range [-1]`, panicking inside
the Generate goroutine. min_p is not on the HTTP surface, so this reached users through
the library and `goinfer-chat --min-p`.

Clamped rather than rejected: SamplingParams has no error return here, every other
degenerate value in this struct is clamped or ignored, and 1.0 is the identity for
min-p (keep only what ties the max) — the nearest meaningful reading of "more than
everything". The docstring's "min_p at any value are safe" is now true rather than
aspirational.
```

## Sampler.applyPenalties

Moved from `decoder/sampler.go` (the comment above `Sampler.applyPenalties`) on 2026-10-10.

```text
applyPenalties applies repeat/presence/frequency penalties to the logits of
tokens in the sampler's own penalty window. Repeat scales (llama.cpp);
presence/frequency subtract (OpenAI). They compose.

P-15: the default (RepeatLastN ≤ 0) window is the WHOLE history, which grows
by one token per step — applyPenaltiesOver used to rebuild a fresh counts map
by rescanning all of it every call (O(n) per token, O(n²) over a generation).
That window is exactly what histCounts already tracks incrementally (kept in
sync by recordHistory), so this case skips the rescan entirely. The windowed
case (RepeatLastN > 0) keeps the rebuild — its window is bounded and small, not
the O(n²) shape this fixes.
```

## computeLogprobs

Moved from `decoder/sampler.go` (the comment above `computeLogprobs`) on 2026-10-10.

```text
computeLogprobs returns log P(chosen) and the topN highest-prob (id, logprob)
pairs, over the full-vocab softmax at the sampling temperature (1 when greedy
— temperature 0 would be a degenerate point mass). dst is the softmax scratch (P-07,
audit-2026-09-10): pass s.distBufN(len(logits)) from a *Sampler's own per-token loop to reuse
its buffer instead of paying a fresh full-vocab make() on every logprobs:true request; nil
(from the free-standing tests below) falls back to softmaxStableInto's own fresh allocation.
```

## lastWithMass

Moved from `decoder/sampler.go` (the comment above `lastWithMass`) on 2026-10-10.

```text
lastWithMass is the float-rounding fall-through for a cumulative draw: the last index in
[lo,hi) whose probability is non-zero, or hi-1 when every one of them is zero.

N-02. The draws returned `hi-1` outright. The vector is MASKED — top-k and top-p zero the
excluded tail — so hi-1 is very often a token the filter deliberately removed, and returning it
emits something the caller configured to be impossible. spec_sample.go's drawTree already walked
back like this; drawFull and drawChunked did not. ~1e-16 per draw, so a contract nick rather
than a live bug, but it is the contract top-k and top-p exist to provide.
```

## topFilterLogits

Moved from `decoder/sampler.go` (the comment above `topFilterLogits`) on 2026-10-10.

```text
topFilterLogits applies top-k, then min-p, then top-p to a LOGIT vector,
returning the surviving (id, renormalized-prob) pairs in descending order.
It replaces the old topFilter, which softmaxed all V then full-sorted all V.

TIE-BREAK CONTRACT (amendment 1): entries with equal probability are ordered by
ASCENDING token id. The old path used sort.Slice, which is not stable, so the order
of tied entries was arbitrary — and since that order feeds the cumulative-CDF draw,
it was an unspecified part of the sampling result. It is now specified. The test-only
reference (refTopFilter, sampler_selection_test.go) carries the identical tie-break
and this path is gated bit-for-bit against it. (Same defect class as the CUDA
argmax-reduce index tie-break, which is FIXED — audit C-14, c6600fc: argmax_reduce returns the
lowest index on an exact tie, matching this contract. Gate: cuda.TestArgmaxTieBreak.)

SUMMATION-ORDER CONTRACT (amendment 2): every probability sum below — the top-p
cumulative and the final renormalization — runs in DESCENDING probability order.
The denominator is load-bearing: summing in a different order moves it by ULPs and
can flip which side of the cumulative-p boundary a token lands on, changing the draw.
Do not reorder these loops. (Same treatment as the Metal reduction widths.)

LOGIT-SPACE SELECTION (amendment 3): temperature scaling and exp are monotone, so
top-k and min-p are decided on the raw logits with no softmax over V; exp is applied
only to the (small) retained set. top-p is the one filter whose cutoff is defined on
the NORMALIZED mass, so it needs the full-vocab softmax denominator Z — computed as a
single O(V) exp-sum, with no full probability array and no O(V·log V) sort. That Z
pass is irreducible for an exact nucleus; the sort it replaces is not.

candScratch/ipsScratch (P-07, audit-2026-09-10): caller-owned scratch for the no-filter
default case's candidate list and for the indexed-probability pairs every case builds — the
audit's own "2.4 MB" figure, previously a fresh make() on every call regardless of branch. Both
are consumed synchronously within this call (by the sort/cut logic below and by the caller
immediately after return) and never retained, the same lifetime vocabScratch already has.
```

## Sampler.scratch

Moved from `decoder/sampler.go` (the comment on the scratch fields of `Sampler`) on 2026-10-10.

```text
[vocab] scratch shared by sampleChunked's e and chunkedZ's tmp — one full-vocab
allocation reused every token instead of a fresh make() each draw; the two are never live
simultaneously (Sample's branches are mutually exclusive per call, so one buffer covers both).
[vocab] scratch for distVectorFrom's returned distribution (P-14). Each
speculative verify position calls dist(...), which is consumed synchronously by specStep /
drawDist / traceFromDist before the loop moves to the next position — never two positions'
vectors live at once (spec_ngram.go's verify loop) — so one buffer covers the whole round.
[vocab] scratch for distVectorHist's history-dependent copy (P-14):
same one-position-at-a-time lifetime as distBuf.
[vocab] scratch for SampleWithInfo's bias/penalty-mutated copy (P-07,
audit-2026-09-10). Replaces a fresh slices.Clone(logits) every token a penalty or bias is
active: `work` is read by argmax/drawFiltered/softmaxStable/sampleChunked/computeLogprobs
within that same SampleWithInfo call and never stored or returned, so one buffer reused
every token is safe — the same reasoning distBuf/specLogitsBuf already document above.
[vocab] scratch for topFilterLogits' candidate-id list in its no-filter
default case (P-07). Consumed synchronously by the SAME call's indexedProb pass
immediately below it, never retained past topFilterLogits' own return.
[vocab] scratch for topFilterLogits' indexed-probability pairs (P-07)
— the audit's own "2.4 MB" figure, the largest single allocation topFilterLogits made, on
EVERY call regardless of which candidate branch ran. Consumed synchronously by
drawFiltered/distVectorFrom immediately after topFilterLogits returns, never retained.
per-id occurrence count over the WHOLE of history, maintained
incrementally by recordHistory (P-15) so applyPenalties' unbounded-window case (the
default: RepeatLastN ≤ 0) never rebuilds a map by rescanning all of history — every
history mutation MUST go through recordHistory, or this desyncs from s.history.
```

## DFlashDrafter

Moved from `decoder/dflash.go` (the comment above `DFlashDrafter`) on 2026-10-10.

```text
DFlashDrafter is an imported z-lab DFlash block drafter (P10 / docs/spec/08): a small
non-causal transformer trunk that reads the target's hidden states (the ForwardCapture
seam) and proposes a whole BLOCK of tokens in one pass, instead of one token per head
forward the way 05's EAGLE head did (removed 2026-09-24). That is the draft-side economics the spec
program's scorecard said was the lever.

It is deliberately smaller than it looks. The checkpoint ships ONLY the trunk —
5 decoder layers + `fc` + two norms, 58 tensors — with **no embedding, no LM head, no
Markov head and no confidence head** (verified against the published file byte-for-byte,
see docs/spec/08). Both ends are the TARGET's: the block is embedded with the target's
embed_tokens and the draft logits come out of the target's lm_head. So this type holds
no vocab-sized weight at all, and the caller supplies both ends.

One round, given the target's captured hidden states for the committed context:

	fused    = hiddenNorm(fc · concat(h[l] for l in targetLayerIDs))   per context position
	blockIn  = targetEmbed([anchor, MASK, MASK, ...])                  [blockSize, hidden]
	trunk    = layers(blockIn, attending over concat(fused, block))    bidirectional
	logits   = targetLMHead(norm(trunk)[1:])                           blockSize-1 drafts

The attention is CROSS-attention and NON-causal: queries come only from the block,
keys/values from the fused context concatenated with the block, and every block
position sees every other. That is what makes the block one pass — and it is also why
the block width is not truncatable (each position's hidden depends on all of them, so
a narrower block is off the trained distribution).
```

## DFlashDrafter.ownHead

Moved from `decoder/dflash.go` (the comment above `DFlashDrafter.ownHead`) on 2026-10-10.

```text
OWN head + embedding, present only on drafters that ship them (poolside's
Laguna speculators do; z-lab's do not and borrow the target's — see
DrafterHeadLogits). When lmHead is set the drafter emits ids in its OWN
REDUCED vocabulary, which d2t maps back to target ids:

	target_id = i + d2t[i]   for draft index i

That is the same scheme the EAGLE head used (removed 2026-09-24, be9aeea8); the
arithmetic was kept identical to it rather than re-derived. embed is the drafter's own token embedding over the TARGET vocab
(it is fed target ids and produces the trunk's block input).
```

## DFlashDrafter.DraftTokenID

Moved from `decoder/dflash.go` (the comment above `DFlashDrafter.DraftTokenID`) on 2026-10-10.

```text
DraftTokenID turns one already-normed trunk hidden row into a TARGET token id
using the drafter's own reduced-vocab head, mapping the argmax back through d2t.

The reduced vocab is the whole point of the design — a 32000-row head over a
100352-token target vocab is ~3x less work per drafted position — but it means an
unmapped argmax is a VALID-LOOKING id in the wrong space. Drafting is lossless by
construction (the target re-verifies every token), so a missed mapping would not
corrupt output; it would silently destroy acceptance and read as "this pairing is
just bad", which is exactly the failure P10 spent a day chasing on a wrong mask
token. Hence the mapping lives here, next to the argmax, rather than in the caller.
```

## blockTrunk

Moved from `decoder/dflash.go` (the comment above `blockTrunk`) on 2026-10-10.

```text
blockTrunk is the non-causal block trunk shared by DFlash and DSpark.

Shared because they compute it IDENTICALLY, which was established from first-party source
rather than assumed: DeepSpec's `_forward_backbone` is `hidden_norm(fc(ctx))` → rotary →
N layers → `norm`, its attention sets `is_causal=False` and takes K/V from
concat(raw fused context, block), and its `apply_rotary_pos_emb` — q taking
`cos[..., -q_len:, :]` while k takes the full `cos` — is byte-identical to z-lab's. Two
separately-developed drafters converged on the same trunk; carrying two copies of it here
would be the third copy this repo's own rule warns about.

What differs between them lives in the enclosing type: DFlash borrows the target's embedding
and LM head and predicts from slot 1; DSpark ships its own, predicts from slot 0, and adds a
rank-256 Markov chain plus a confidence head.
```

## dflashConfig.rope

Moved from `decoder/dflash.go` (the comment above `dflashConfig.rope`) on 2026-10-10.

```text
RoPE in both spellings too — flat on Qwen3-4B-DFlash-b16, nested rope_parameters on
Qwen3.6-35B-A3B-DFlash (and on every DSpark checkpoint). Three checkpoints, three
config dialects; reading only the first one's is how a supported pairing looks broken.
```

## dflashConfig.blockSize

Moved from `decoder/dflash.go` (the comment above `dflashConfig.blockSize`) on 2026-10-10.

```text
block_size appears in BOTH places across z-lab's own checkpoints: top-level on
Qwen3-4B-DFlash-b16, nested in dflash_config on Qwen3.6-35B-A3B-DFlash. One publisher,
two spellings — so read either rather than assume the one the first checkpoint used.
```

## LoadDFlashDrafter.v2

Moved from `decoder/dflash.go` (the comment above `LoadDFlashDrafter.v2`) on 2026-10-10.

```text
REFUSE DFlash 2. Measured on the real checkpoint (incoai/Qwen3.8-27B-DFlash2, 2026-08-20):
v2's config carries every field this loader reads, and 76.2% of its tensors are v1-shaped,
so it loads WITHOUT ERROR and silently drops 914,309,120 bytes — 23.8% of the file — of
two-tap dynamic convolutions (`layers.N.{attention,mlp}_conv.*`) and the candidate selector
(`candidate_selector.*`).

The convs are inserted before AND after every attention and FFN sublayer, so dropping them
changes every layer's output. The failure is not wrong tokens — DFlash verify is lossless,
so the target still gates everything — it is a drafter that drafts badly: LOWER acceptance,
slower than v1, and no diagnostic anywhere. Exactly the silent-degradation shape this
program keeps paying for.

Detected from config rather than tensor names because it is cheaper and it is what the
publisher controls; a v2 checkpoint that omitted these keys would still be caught by the
missing-tensor path, since v1 requires nothing v2 lacks. Revisit when P15 implements v2.
```

## DFlashContext

Moved from `decoder/dflash.go` (the comment above `DFlashContext`) on 2026-10-10.

```text
DFlashContext caches the committed context's PROJECTED K/V per layer — K already
RoPE'd at its absolute position, V neither roped nor normed.

It exists because the context is projected once per position and then read by every
subsequent round: without it, each round re-projects the whole context in every layer,
which is O(ctx x layers) of pure repeat work per round and made the CPU trunk scale
1.6 s/block at ctx 64 to 12.9 s at ctx 2048 (measured, BenchmarkDFlashTrunk). Both
reference implementations cache it — mlx-dspark's CtxCache and dflash.py's DynamicCache
— so this matches them rather than inventing a shortcut.

Append-only, plus TruncateTo for the speculative rollback: the drafter's context only
ever grows with COMMITTED tokens, and a rejected draft's positions must come back off.
```

## AdaptiveDepth.Theta

Moved from `decoder/spec_adaptive.go` (the comment above `AdaptiveDepth.Theta`) on 2026-10-10.

```text
Theta is the marginal cost of one extra verify node, in units of one
single-token target step. Domain is (0, +inf): >= 1 is LEGAL and means
"an extra node costs at least a whole step", i.e. never draft. Zero or
negative means unset and takes the backend default (see thetaFor).

It used to be documented and enforced as [0,1), which silently rejected
every value Metal actually measures -- 1.006 to 1.048 across two models
and two depths, 2026-09-01 -- and substituted 0.5, the most over-drafting
setting available. A parameter whose domain excludes the measurement is
not a default, it is a wrong answer that cannot be corrected.
```

## AdaptiveDepth.Cost

Moved from `decoder/spec_adaptive.go` (the comment above `AdaptiveDepth.Cost`) on 2026-10-10.

```text
Cost, when set, replaces the linear Theta rule with a measured cost curve: Cost[m] is the time to verify m rows
(m >= 1) in units of one single-token target step, so Cost[1] is 1 (a round that drafts nothing is a plain
decode step). A verify whose cost is NOT linear in m needs it: Metal's step-kernel verify costs ~1.6-1.8 tokens
at 2 rows but only ~1.8-2.7 at 8 (a fixed overhead, then a nearly flat slope), so the linear rule, which drafts
while alpha^k > Theta, would draft single tokens at an acceptance that cannot pay for the overhead. With Cost,
Depth picks the depth that maximises expected emitted tokens per unit cost, and 0 when nothing beats plain decode.
```

## AdaptiveDepth.ensure.probe

Moved from `decoder/spec_adaptive.go` (the comment above `AdaptiveDepth.ensure.probe`) on 2026-10-10.

```text
A probe verifies one draft at Cost[2], ~0.6-0.8 of a token over a plain step on a step-kernel verify;
every 64 idle rounds that is ~1% on a stream that never drafts, against ~5% at 16.
```

## AdaptiveDepth.Depth.theta1

Moved from `decoder/spec_adaptive.go` (the comment above `AdaptiveDepth.Depth.theta1`) on 2026-10-10.

```text
Theta >= 1 means one extra verify node costs at least a whole target step,
so no acceptance rate can pay for it: alpha < 1 always, and the test below
would return 0 for every alpha. Skip the periodic probe too -- it exists to
refresh a STALE ALPHA, and here the decision does not depend on alpha, so a
probe cannot change the answer and is pure wasted draft work. This was the
live case on Metal while its ForwardN was a loop of single-token Forwards
(measured Theta 1.006-1.048); its batched ForwardBatch (2026-09-16) re-measured
0.96 (thetaFor).
```

## GenerateNgramSpeculativeAdaptive.theta1

Moved from `decoder/spec_adaptive.go` (the comment above `GenerateNgramSpeculativeAdaptive.theta1`) on 2026-10-10.

```text
P-16: Theta >= 1 means Depth() always returns 0 (see its own comment) --
no acceptance rate can pay for even one verify node, on ANY round,
regardless of the stream. Every round would still pay for a history
clone, an n-gram scan, and a full-logits ForwardN to draft and verify
nothing (measured 1.01x of plain Generate on Metal, where this is the
live case). Validate first so an invalid drafter/sp still errors exactly
as genNgram would, then decline straight to the cheap path.
```

## defaultTheta

Moved from `decoder/spec_adaptive.go` (the comment above `defaultTheta`) on 2026-10-10.

```text
defaultTheta is the fallback when a caller sets no Theta and the backend is
not in the table below. It is the CPU value, which is where the constant came
from originally -- re-measured 0.506 (depth 128) and 0.532 (depth 512) on
2026-09-01, so 0.5 remains right for the path it was named after.
```

## thetaFor

Moved from `decoder/spec_adaptive.go` (the comment above `thetaFor`) on 2026-10-10.

```text
thetaFor returns the measured marginal verify-node cost for a backend.

Every value here is MEASURED by the probes that share one definition --
Theta = (least-squares slope of T(n)) / T(1) -- so the three are directly
comparable: decoder/theta_probe_test.go (CPU control),
cuda/theta_probe_test.go, metal/theta_probe_test.go.

Before this table every backend ran the CPU constant 0.5, which
spec_adaptive.go's own doc comment had asked someone to measure since it was
written. The consequences ran in opposite directions: CUDA drafts far too
shallow (its real cost is ~1/3 of the constant, so deeper chains would pay
and were never tried), while Metal drafts when it should not draft at all.
```

## thetaFor.metal

Moved from `decoder/spec_adaptive.go` (the comment above `thetaFor.metal`) on 2026-10-10.

```text
Re-measured 2026-09-17 after layer-major ForwardBatch restructuring, on the
M1 Pro. The shipped value at the time (0.76, "the conservative end of the
measured range 0.315-0.760") was checked against a wider sweep: {0.5B, 1.5B}
qwen2.5-coder int4 AND {0.6B, 1.7B} Qwen3 q8_0, depth in {128, 256, 512, 1024,
2048}, 20 cells, 9-sample median per cell, isolated (no concurrent load; an
earlier contaminated attempt read as high as 7.35). It did not hold: 19 of 20
cells measured above 0.76 (min 0.710, median 0.919, max 0.962). Every cell stayed
under 1.0, so speculation is still correctly enabled here, but 0.76 was
understating the true marginal cost across most of the range — same risky
direction the CUDA comment below warns about. 0.96 covers the observed max with
no further margin needed.
```

## thetaFor.cuda

Moved from `decoder/spec_adaptive.go` (the comment above `thetaFor.cuda`) on 2026-10-10.

```text
0.155-0.251 measured (cuda/theta_probe_test.go). The CONSERVATIVE
end of the measured range is used deliberately: Theta appears
inside floor(ln(Theta)/ln(alpha)), so understating it drafts
deeper, and a too-deep draft on a low-acceptance stream is the
exact failure the adaptive controller exists to prevent. Taking
the shallow end means the win is under-claimed rather than the
regression risked.
```

## Model.verifyTheta

Moved from `decoder/spec_adaptive.go` (the comment above `Model.verifyTheta`) on 2026-10-10.

```text
verifyTheta returns the Theta for the path this model's speculative VERIFY
actually runs on.

The distinction that matters is resident vs staged, not which backend was
requested. genNgramInto verifies through target.resident.ForwardN when
residency built, and falls back to the CPU batched forwardN when it did not
(spec_ngram.go) -- so a "webgpu-staged" or "metal-staged" model verifies on
CPU and must get the CPU constant, not its GPU one. Keying this off
Options.Backend alone would hand a declined-residency model the GPU value and
silently mis-tune the one case where the decline is already costing the user
the whole forward.

N-49 (docs/audit-2026-09-10.md, resolved 2026-09-17): VerifyPathReporter explicitly
asks whether the VERIFY forward (ForwardN) is batched, rather than guessing from
PrefillPath (prompt ingestion) which prices a separate code path. Both metalResident
and cudaResident implement VerifyPathReporter.
```

## sequentialVerifyTheta

Moved from `decoder/spec_adaptive.go` (the comment above `sequentialVerifyTheta`) on 2026-10-10.

```text
sequentialVerifyTheta is Theta for a resident whose ForwardN is a loop of single-token
forwards. Not measured per backend because it does not need to be: T(n) is n·T(1) by
construction, so the ratio is 1 plus whatever per-row overhead the loop adds. Metal MEASURED
exactly that shape (1.006-1.048 across four configurations, linear to n=16) and ships 1.02;
the same number is used here for the same reason.
```

## optFwdMaxTemp

Moved from `decoder/spec_optfwd.go` (the comment above `optFwdMaxTemp`) on 2026-10-10.

```text
optFwdMaxTemp is the temperature at or below which the optimistic-forward overlap is allowed to
run. ABOVE IT THE FEATURE IS A MEASURED LOSS, and it used to run unconditionally.

WHY A FIXED THRESHOLD, AND WHY 0.2. The overlap can only pay back the sampler cost it hides, and
its hit rate falls with temperature. Both halves are model-dependent, so the break-even
temperature is too: MEASURED 2026-08-27 at T ~ 0.26 on phi3-mini (vocab 32064) and T ~ 0.95 on
qwen2.5-coder-1.5B (151936). 0.2 sits below the LOWER of the two, which is the only safe place
for a single constant: at the higher crossover phi3-mini pays 2.8-6.8%.

WHAT THIS COSTS, RECORDED SO IT IS NOT REDISCOVERED AS A BUG. On large-vocab models the overlap
still wins between 0.2 and their own crossover — 6.0% at T=0.4 and 5.1% at T=0.6 on the 1.5B —
and this threshold forfeits that. An adaptive per-model gate was designed to recover it
(docs/spec/10-optfwd-gate.md) and DELIBERATELY NOT BUILT: its whole value was those two cells,
resting on a model-dependence generalised from two models, which is the same shape of error that
shipped this feature unconditionally in the first place. Revisit if a third model's crossover
lands somewhere this gate gets badly wrong; 10's pre-registered bar stands.

TRUNCATED SAMPLING IS UNMEASURED. The ladders were temperature-only with no truncation. top_k /
top_p cut the candidate set, which should RAISE the hit rate and push the crossover up, so this
gate is probably conservative there — forfeiting a possible win rather than taking a measured
loss, which is the correct direction to be wrong in until it is measured.

GOINFER_OPTFWD_MAX_TEMP overrides it, for MEASUREMENT rather than tuning: moving this number
without a ladder behind it is how the original default happened.
```

## optFwdGate

Moved from `decoder/spec_optfwd.go` (the comment above `optFwdGate`) on 2026-10-10.

```text
optFwdGate is a binary (speculate / don't) trailing-hit-rate switch, the same EMA shape as
AdaptiveDepth (spec_adaptive.go) but for an on/off decision rather than a continuous depth.

THESE THRESHOLDS ARE FITTED ON ONE MODEL AND ARE WRONG FOR SMALL-VOCAB ONES (G27). Break-even
hit rate is not a constant: it rises as the sampler share falls, because the overlap can only pay
back the sampler it hides. The 90.9% below was measured on qwen2.5-coder-0.5b — 152k vocab, a
LARGE sampler share. On phi3-mini (32k, 5.4% share against the 1.5B's 18.2%) the true break-even
sits ABOVE 0.90, so the entire 0.90/0.75 dead band lies below it and this gate cannot turn off in
the regime where the feature loses 2.8-6.8%. That is why the loss needed a temperature cap
(optFwdMaxTemp) rather than being caught here.

The enable threshold is pinned to the WORST measured break-even across depth (90.9% at a
shallow qwen2.5-coder-0.5b context, rounding down to 0.90) rather than a depth-aware curve:
break-even only rises with depth (toward ~97% at 2048), so a fixed threshold at the shallow
floor is never a net loss at any depth — it just leaves some upside on the table deep in
context, where a depth-aware policy could re-enable more aggressively. Not worth the extra free
parameter for v1 without a measured gpuPos->cost mapping on every backend.
```

## optFwdGate.ensure

Moved from `decoder/spec_optfwd.go` (the comment above `optFwdGate.ensure`) on 2026-10-10.

```text
The dead band is wide (0.90/0.75), not the tight one a first cut assumed (0.90/0.85): an EMA
close to a real, sustained 93% hit rate (the measured T=0.7 number, comfortably above the 90.9%
worst-case break-even) still has enough sampling variance at Lambda=0.9's ~10-sample effective
window to dip below a threshold just 3 points below the true mean, causing real flapping with no
underlying regime change (caught by TestOptFwdGate_marginalRateHoldsSteady). Lambda=0.95 (~20
samples) and a wide dead band down to 0.75 both push in the same direction: tolerate the natural
noise around a genuinely-profitable rate, while still reliably catching a sustained drop toward
the T=1.0-like 73% regime, where staying on would be a real loss.
```

## optFwdGate.Should

Moved from `decoder/spec_optfwd.go` (the comment above `optFwdGate.Should`) on 2026-10-10.

```text
Should reports whether this step should attempt the optimistic guess.

THE HYSTERESIS ABOVE IS A ONE-WAY LATCH IN PRACTICE, AND THE DEAD BAND'S UPPER HALF IS DEAD CODE
(G27). Observe is called ONLY from optFwdStep, and the caller in model.go invokes optFwdStep only
when Should() is true. So the moment alpha falls below DisableAt and this returns false, no
further outcomes are ever observed: alpha freezes, and the `alpha >= EnableAt` re-enable branch in
Observe cannot be reached again for the rest of that Generate. It reads as a two-way band and
behaves as a latch.

TestOptFwdGate_hysteresis DOES NOT CATCH THIS and actively vouches for the two-way reading,
because it drives Observe in an unconditional loop — a calling convention production never uses.
It is correct about the component and blind to the composition.

Left as-is deliberately: since T > 0.2 no longer reaches the overlap at all (optFwdMaxTemp), the
gate barely runs and the latch is unreachable in the regime that mattered. Fix it together with
the thresholds, not before — see G27 in docs/QUEUE.md.
```

## Model.optFwdStep.copy

Moved from `decoder/spec_optfwd.go` (the comment above `Model.optFwdStep.copy`) on 2026-10-10.

```text
COPY THE LOGITS BEFORE STARTING THE OVERLAP. This is not defensive tidiness; without it the
feature is silently wrong on CUDA.

A resident backend may legitimately return a slice that ALIASES its own reusable host buffer,
valid only until the next Forward — cuda/resident.go returns r.logitsHost, "a zero-copy view
of logitsPinned". The overlap then has the speculative Forward's device->host DMA writing that
exact buffer while SampleWithInfo is reading it, so the sampler sees a torn mixture of this
position's logits and the next one's.

MEASURED on the CUDA box before this copy existed: same token id, logprob -2.6463 vs -2.6266
(feature on vs GOINFER_NO_OPTFWD=1) — and under `-race`, where the timing shifts, the emitted
TOKEN STREAM diverged outright. Severity tracking timing is what identified it as a race
rather than an arithmetic difference.

`go test -race` CANNOT SEE THIS and reported nothing: the write is a driver DMA into pinned
memory, not a Go memory access, so the detector is structurally blind to it. That is the
reason this comment is long — the next person to touch the overlap will not get a warning.

Metal was unaffected because its Forward hands back a per-call slice, which is why the feature
verified clean there. The copy fixes every backend, including ones that do not exist yet, and
costs one vocab-sized memcpy against a full forward.
```

## GenerateSpeculative.rollback

Moved from `decoder/speculative.go` (the comment above `GenerateSpeculative.rollback`) on 2026-10-10.

```text
Rollback safety (audit C-02): verify advances the target's KV by K per round and
rolls back the rejected tail. A recurrent (Mamba-2 / Gated DeltaNet) or staged
sliding-window cache cannot losslessly restore that — the other three speculative
entry points guard this; GenerateSpeculative did not.
```

## GenerateSpeculative.draftResident

Moved from `decoder/speculative.go` (the comment above `GenerateSpeculative.draftResident`) on 2026-10-10.

```text
The draft runs on the GPU too when it was loaded --backend webgpu and is
eligible. This is the decisive lever: a CPU draft is slower PER TOKEN than the
GPU target it feeds (measured ~4.9× on the 2070S), so K CPU draft tokens/round
cost far more than they save and speculation is a net loss; a resident draft is
~0.5× the target's per-token cost, which is the only regime where it can pay.
Draft + target are separate Contexts (one device/queue each), driven
sequentially here. Output stays token-identical to plain target greedy — the
draft is only a proposer; the target decides (so its backend can't change it).
```

## GenerateSpeculative.claim

Moved from `decoder/speculative.go` (the comment above `GenerateSpeculative.claim`) on 2026-10-10.

```text
Claim the single shared resident KV before any device write (audit C-03): both
generateInto and the n-gram path CAS this, GenerateSpeculative did not — so a
second concurrent Generate on the same *Model would prefill into the same
positional device KV, interleaving writes and corrupting both streams against a
Model doc that promises concurrent distinct sequences. On loss, fall back to the
staged CPU cache. Draft is a separate Model with its own claim.
tryClaimResident, not a bare CAS on resBusy: MC3's batched holders never set resBusy, so only the claim
that also requires no holder keeps this off a resident mid-batch (docs/completed/task-audit-followups-2026-10-06.md, A1).
```

## GenerateSpeculative.draftClaim

Moved from `decoder/speculative.go` (the comment above `GenerateSpeculative.draftClaim`) on 2026-10-10.

```text
R-00's shape on the second Model (V-09, docs/review-2026-09-04.md): the
target's own claim just above forgets first for exactly this reason — from
here until this generation completes, draft's resident KV is mid-write, so a
later unrelated generation on this SAME draft model must not trust whatever
resIDs happened to be set from before. Missed when R-00 fixed the target half.
```

## GenerateSpeculative.prefill

Moved from `decoder/speculative.go` (the comment above `GenerateSpeculative.prefill`) on 2026-10-10.

```text
Prefill the prompt. The target's last-token logits seed cur; the draft's are
discarded (it just needs its KV filled). Resident: residentPrefillSeed (P-06,
audit-2026-09-10) — the batched/KV-only-prefill helper generateInto and genNgramInto
already share, in place of this function's own third copy of the per-token Forward
loop the helper was unified to prevent (+2.66 ms/token vs 0.42 batched, per the audit's
own measurement). CPU: batched prefillLogits, unchanged.
```

## grammarConf

Moved from `decoder/spec_grammar.go` (the comment above `grammarConf`) on 2026-10-10.

```text
grammarConf is α̂_grammar: the calibrated acceptance probability of a forced grammar
proposal, on the SAME accept-prob scale as α̂_ngram (ngramAlpha) so the router (03)
compares sources principally (§06). TRACE-FIT (TestGrammarAlphaPredictor, qwen2.5-
coder-0.5b, JSON-schema workloads): forced tokens accept only ~0.20 — far below the
"forced ⇒ ≈1" intuition — because the drafter's CANONICAL retokenization of the
forced bytes usually differs from how the model tokenizes the same bytes under the
mask, and the mismatch compounds within a run (depth-0 ~0.24 → depth-1 ~0.08). Read
0.20 precisely: it indicts THIS drafter, not grammar speculation in principle — free
grammar drafting must GUESS the tokenization of the forced bytes (canonical
retokenization), and getting it right fundamentally needs the model; a tokenizer-
aligned forced drafter (unbuilt, §01/§06) is the headroom. So grammar ranks LAST *as
currently built*: any n-gram copy (ngramAlpha ≥ 0.70) outranks it and the router
treats it as the floor (it drafts only when n-gram has no copy — a 20% free-token
shot still beats nothing, and a miss costs ~nothing). Cross-model STABLE: 0.205 on
llama-3.2-1b (a different tokenizer), so the fragility is a property of canonical-
bytes drafting, not one model — grammarConf holds. Re-fit per model/tokenizer (§06 §9).
```

## Model.genGrammarInto.finishTrailing

Moved from `decoder/spec_grammar.go` (the comment above `Model.genGrammarInto.finishTrailing`) on 2026-10-10.

```text
finishTrailing forwards cur when the generation ended by reaching maxTokens just after streaming it, so the cache
holds prompt + every emitted token, as plain constrained decode's does. A round's trailing token is otherwise
forwarded only as the next round's seq[0], leaving the cache one token short for the next turn to re-prefill
(the n-gram loop's same defect, which on Metal changed every later turn:
docs/measurements/spec-vs-batching-metal-2026-09-27.md §4). A stop or a cancel takes no forward.
```

## BlockDrafterWeights

Moved from `decoder/blockdrafter.go` (the comment above `BlockDrafterWeights`) on 2026-10-10.

```text
The read-only view a backend needs to make a block drafter GPU-resident.

WHY AN INTERFACE RATHER THAN EXPORTED FIELDS. The alternative was exporting `blockTrunk`'s
fields so `cuda` could reach them, which publishes the drafter's LAYOUT as API — and
`blockTrunk` is deliberately shared between DFlash and DSpark precisely so it can be
refactored as more families land. This exports the CAPABILITY instead: a backend learns the
geometry and gets the weights, and the struct behind them stays free to change.

It also inverts the dependency the way `ResidentForward` already does — `decoder` declares
what a backend may read, and `cuda` (or `metal`, or `gpu`) consumes it, rather than a backend
importing a concrete drafter type and hard-coupling to one family.

BOTH FAMILIES SATISFY IT FOR FREE. The methods are implemented once on `*blockTrunk`, which
DFlashDrafter and DSparkDrafter both embed — so a resident path written against this interface
serves either without a type switch. That is the property that made an interface worth writing
now: there are two implementations to generalize over, not one to guess from.

Everything returned is READ-ONLY. The WeightMat pointers alias the drafter's own storage
(they are hundreds of MB; copying them to hand out would defeat the purpose), so a backend
packs and uploads from them and must not write through them.
```

## BlockDrafterWeights.MaskTokenID

Moved from `decoder/blockdrafter.go` (the comment above `BlockDrafterWeights.MaskTokenID`) on 2026-10-10.

```text
MaskTokenID is the TRAINED token the drafter expects at unfilled block positions. It is
on the interface because getting it wrong is silent and expensive: the drafter still
runs, still produces lossless output, and simply drafts badly. Measured cost of passing
any other id — 1.77 tok/round against a known 4.97, turning a 1.60x speedup into 0.66x.
```

## DrafterResidentBytesEstimate

Moved from `decoder/blockdrafter.go` (the comment above `DrafterResidentBytesEstimate`) on 2026-10-10.

```text
DrafterResidentBytesEstimate approximates the VRAM a resident backend will claim uploading dw
— tasks/task-fit-to-hardware.md §2's "the drafter's weights (~500 MB for the 4B pairing, uploaded to
the target's device at attach)" term, computed instead of quoted, so a caller (a fit guard
pricing a --drafter attach BEFORE the target's own residency is built) has a real number rather
than a fixed constant that drifts from whatever pairing is actually loaded.

EVERY DRAFTER MATRIX IS f32 ON HOST (loaded straight from safetensors via WrapF32 — dflash.go's
loadMat, never QuantizeInt8/QuantizeInt4) and the only backend that hosts one today, CUDA's
AttachDrafter (cuda/drafter.go), packs every f32 matrix through packWeight's f32 branch, which
is ALWAYS int8 (linalg.QuantizeRowsInt8) regardless of the target model's own quant — a drafter
is never packed to int4. So this is not a generic multi-quant estimate the way
decoder/fitguard.go's quantBytesPerElem is for the main model: it prices int8 specifically,
because that is the one encoding an attach can actually produce today. Per-row scale (one f32
per row) is included; the norm vectors and RoPE table are f32 already and round to nothing
beside the matrices, the same exemption ResidentWeightBytes' own doc comment gives the main
model's norms/biases.

NOT included: the verify/capture buffers NewBlockSpec allocates (SetBatchedCapture's batched
logits, FuseContext's per-call context scratch) — a few MB at the default verify width against
hundreds of MB of weights, and already the kind of small residual the existing 384 MiB
ctxCapMarginBytes/slotMarginBytes margins in cuda/resident.go are there to absorb. Naming this
rather than silently folding it into "close enough": the weights are the term this function
computes for real, not the whole attach.

ALSO NOT included: the drafter's own K/V (M-22, docs/audit-2026-09-10.md) — see
DrafterKVBytesPerPosition below for why that term needs a ctx to multiply by that this function,
called before the target's own residency is built, does not yet have.
```

## DrafterKVBytesPerPosition

Moved from `decoder/blockdrafter.go` (the comment above `DrafterKVBytesPerPosition`) on 2026-10-10.

```text
DrafterKVBytesPerPosition is M-22's own fix (docs/audit-2026-09-10.md): the drafter's device K/V
scales with the TARGET's own resident context, not a fixed size of its own —
cuda/drafter.go's ExtendContext sizes d.kc/d.vc at capRows = max(need+512, r.ctxCap) — so a
5-layer trunk at a large target context is hundreds of MB, not the "few MB" the scratch-buffer
exemption on DrafterResidentBytesEstimate is about, and DrafterResidentBytesEstimate itself
cannot price it: it runs BEFORE the target's own residency (and so its chosen ctx) exists.

Returns bytes PER POSITION (layers × kvDim × 2 for K+V × 4 bytes, matching cuda/drafter.go's
own d.kc[l]/d.vc[l] = af(capRows*kvDim) sizing exactly) so the caller — resolveCtxCapFit, which
already knows the candidate ctx it is about to ask Plan to fit, and the real build, which knows
the FINAL ctxCap once chosen — can each multiply by the ctx value they actually have in hand,
rather than this function guessing one. Both multiplications land on a real, non-circular bound:
Plan's chooseCtx only ever SHRINKS from the candidate it is asked with, never grows past it, so
pricing the planning-time call against the candidate can only over-estimate (safe) or be exact.
```

## mc3_batch.overview

Moved from `decoder/mc3_batch.go` (the comment at the top of the file) on 2026-10-10.

```text
MC3 (docs/tasks/task-concurrency-2026-09.md): several generations decoding on ONE resident at once, each on its own
resident KV slot, their decode tokens joined into shared steps.

A resident is one command queue with one set of scratch buffers, so everything that touches it — a prefill, a
single-sequence forward, a batched step, the slot bookkeeping — runs one at a time, inside the batcher's exclusive
section. What MC3 adds is WHAT runs there at a decode token: the batcher coalesces the tokens every decoding
generation submits into one run. A run of at least lo batch-eligible tokens goes to ResidentBatchStepper.StepBatch
(every row bit-identical to that sequence's own Forward); anything else — a lone generation above all — runs
production's own per-sequence call, unchanged, so a request served alone takes exactly the path it takes today.

Generations join at token boundaries: a run starts once every decoding generation has submitted, or once the oldest
submission has waited batchStragglerWait (a generation whose consumer is slow to read its stream must not stall the
rest). A newcomer's prefill runs between runs, whole, and takes precedence over the next run — chunked prefill
interleaved with decode is MC5, not this.
```

## batchStragglerWait

Moved from `decoder/mc3_batch.go` (the comment above `batchStragglerWait`) on 2026-10-10.

```text
batchStragglerWait bounds how long a submitted token waits for the other decoding generations' tokens before its run
starts without them. Host work between two tokens is well under a millisecond per generation; a batched step costs
20+ ms (MC3 S1), so this is at most a few percent of a step when it fires, and it fires only for a straggler.
```

## tokenCoalescer.forward.window

Moved from `decoder/mc3_batch.go` (the comment above `tokenCoalescer.forward.window`) on 2026-10-10.

```text
The straggler window opens when a run COULD start: at the oldest submission or when the resident came
free, whichever is later. Timed from submission alone, a token submitted during a run has "waited" the
whole run when it ends, starts one at once without the others, and the generations phase-lock into
split runs every token (measured 2026-09-26: 4 generations ran as 3 batched + 1 solo on all 128 tokens,
255 of 256 runs straggler-started).
```

## resident_reuse.overview

Moved from `decoder/resident_reuse.go` (the comment at the top of the file) on 2026-10-10.

```text
Prefix reuse on the RESIDENT positional KV.

A resident model decodes statelessly: decoder.Generate engages the resident runner only
when there is no session commit and no prefix reuse, because a session's prefix cache is
CPU-side while the resident KV lives on the GPU and both cannot be the source of truth. The
consequence is that every turn re-prefills its whole prompt — measured 8.85 s for a
2,293-token Claude Code agent turn on a 7B int4 (docs/integrations/claude-code.md), on every
turn, growing with the conversation.

That trade is only forced for the CPU-side session cache. Doing the reuse NATIVELY on the
GPU cache has no such conflict, and the resident cache is unusually well suited to it:

  - It is POSITIONAL — token at position p lives at slot p — so "truncate to P" costs
    nothing. residentDecoder.TruncateTo is already a no-op for exactly this reason, and
    attention reads only nKeys = pos+1, so entries past the new length are never consulted.
  - An agent turn is a strict PREFIX EXTENSION: turn N+1 is turn N plus the assistant's tool
    call plus the tool result. Measured deltas in a real /v1/messages loop were 45 and 51
    tokens against prompts of 255 and 306.

So the whole mechanism is bookkeeping: remember which ids are committed to the cache, and
prefill only the divergent suffix.

CORRECTNESS IS THE ENTIRE RISK. A wrong prefix match produces confidently wrong output with
no error anywhere — no exception, no NaN, just a reply conditioned on someone else's
context. Four rules keep it honest:

 1. Match on TOKEN IDS, never text. A client that edits its last message shifts
    tokenisation, and the longest-common-prefix is exactly what absorbs that.
 2. The recorded ids are cleared to nil (meaning "unknown, cold-prefill next time") on ANY
    path that is not a fully completed generation — an error, a cancellation, a resident
    claim lost to a concurrent generation. Conservative by construction: the failure mode of
    forgetting is a slow turn, and the failure mode of remembering wrongly is a wrong answer.
 3. At least one token is always prefilled, so the seed logits that start decode are always
    freshly computed rather than assumed.
 4. The record names the WEIGHTS as well as the ids (audit C-02, 2026-09-10). A LoRA adapter
    changes every targeted projection, hence the residual stream, hence every later layer's
    K/V — so an identical id prefix built under a different adapter (or none) is someone
    else's context exactly as surely as a different prompt is. residentCommitIDs records the
    bound *loraRuntime and residentReuseLen refuses on any mismatch, which keeps
    adapter->same-adapter reuse (the agent-loop win) while closing every crossing. Pointer
    identity is sound: LoadAdapter always builds a fresh runtime and registerAdapter RETIRES
    the one it displaces rather than freeing it, so a reload under one name never compares equal.
```

## residentImageBlock

Moved from `decoder/resident_reuse.go` (the comment above `residentImageBlock`) on 2026-10-10.

```text
residentImageBlock records one image block committed to the resident KV: its absolute
position span within resIDs and a content hash of the raw image bytes that produced it (P9a,
docs/multimodal.md). Every position inside [start,end) attended every OTHER position in the
same block under a bidirectional mask during its CPU prefill (prefillLogitsVL/
prefillLogitsQwenVL) — there is no such thing as "half the block's KV, causally consistent
with a differently-completed other half," so a reuse match must treat the block as one atomic
unit: either the whole span verifies and gets skipped together, or none of it does.
```

## residentImageBlock.hash

Moved from `decoder/resident_reuse.go` (the comment between `residentImageBlock` and `residentImageClaim`) on 2026-10-10.

```text
hash == 0 means "no claim" (M-06, audit-2026-09-10.md): a caller with no reuse story of its
own may pass 0 for imgHash, and residentReuseLen's block-match guard requires blk.hash != 0
before comparing against a claim's hash — otherwise two callers who both pass 0 satisfy
`claim.Hash == blk.hash` by coincidence (0 == 0) and the block is treated as verified when
nothing was actually checked. The block itself is still RECORDED with hash 0 (residentCommitIDs
makes no exception for it) rather than omitted: an unrecorded region falls through to the
plain per-position id comparison, which is exactly M-07's hazard — image placeholder ids are
content-independent, so that comparison would silently treat two different images sharing a
slot as identical. Recording the block (even hash-less) keeps the atomicity check in play; the
guard here just ensures it can never emit a false positive.
```

## Model.reuseLenOf.recurrent

Moved from `decoder/resident_reuse.go` (the comment above `Model.reuseLenOf.recurrent`) on 2026-10-10.

```text
RECURRENT FAMILIES CAN ONLY REUSE AN EXACT, STRICT EXTENSION. The three rules below (LCP
matching, capping at len(prompt)-1) police WHICH PREFIX of the resident KV is matched for an
attention-only family; none of them can help a recurrent one, because its state cannot be
rewound to an arbitrary earlier position at all. A Gated DeltaNet's conv ring and matrix
state (and Mamba-2's, and LFM2's conv window) are mutated in place per token with no
per-position history, and the resident path re-zeroes them only at pos == 0. So the ONLY
safe continuation point is exactly len(m.resIDs): the live recurrent state right now already
equals the state after resIDs (residentCommitIDs's invariant, held by R-00 forgetting on
every other writer), so a prompt that is resIDs plus at least one new token can decode
forward from there with no rewind needed at all — which is exactly what an agent turn is
(previous prompt + reply + tool result, a strict prefix extension).

Anything else — an edited earlier message, a shorter resend, an identical resend — has no
safe continuation point (the state would have to run BACKWARD) and falls to 0, cold. An
identical resend is the qwen3.6-35B-A3B repro that motivated the original blanket refusal:
measured 2026-09-02, repeated identical greedy prompts diverged at token 0, differently on
every repeat, decaying to a one-token reply, with no error anywhere. len(prompt) <= n below
is exactly that case (no new token to extend with) and keeps falling to 0.
TestPagerDeterminism is the gate (reuse-on red before this guard, green after; still green
with this narrower rule since an identical resend has len(prompt) == n).
```

## minLeadReuseFast

Moved from `decoder/resident_reuse.go` (the comment above `minLeadReuseFast`) on 2026-10-10.

```text
minLeadReuseFast is the shortest prefix worth reusing under a resident with non-exact prefill kernels (ResidentFastPrefill), for a prompt that will run them. Below it
the reused rows are a SHARED LEAD (a chat template's preamble, a few tokens left by an unrelated short request), re-prefilling them costs next to nothing, and keeping them
can change the reply: they were computed by the exact kernels while the rest of the prompt runs fast. Observed lead lengths were 3 to 16 tokens (the chat header under the
default templates, ~20 to 25 with a default system prompt), so 64 leaves a margin; a longer reuse is a continuation and is kept. Not an Options field: it is a correctness
margin, not an operator choice.
```

## GenerateVL

Moved from `decoder/generate_vl.go` (the comment above `GenerateVL`) on 2026-10-10.

```text
GenerateVL streams a continuation for a multimodal (vision-language) prompt.
`ids` are the text token ids with a run of `imgLen` image-placeholder ids
starting at `imgPos`; `imgHash` is a content hash of the raw image bytes behind
that run (P9a, docs/multimodal.md — used for resident prefix-reuse; a caller
with no reuse story of its own may pass 0, which residentReuseLen treats as "no
claim" and never matches, M-06). `features`
is invoked AT MOST ONCE, and only when the image cannot be fully reused from the
resident KV — a lazy closure specifically so an unchanged, resent screenshot
never re-runs the vision tower at all.

Prefill runs through the bidirectional image-block mask (prefillLogitsVL) — image
tokens attend mutually, text attends causally, and tokens decoded after the image
attend causally too — then decodes up to maxTokens exactly like Generate (same
Sampler, LogitProcessor, and stop rule).

Resident GPU decode (gap 0, docs/multimodal.md): claims the shared resident KV
when available and not busy with a concurrent generation; falls back to the CPU
decode loop otherwise. Image-aware prefix reuse (P9a): before paying for the tower
or the CPU prefill at all, checks whether this turn's image (and everything
before it) is already sitting in the resident KV from a prior turn — if so,
decode resumes directly from there with no re-tower, no re-prefill, no re-upload.
Any other outcome (no resident, a different image, busy) falls through to the
unconditional full-prefill path, unchanged from before P9a. The caller owns the
returned channel: range over it to consume tokens, then check Generation.Err for
a terminal error.
```

## GenerateQwenVLDeepstackSpans.recurrent

Moved from `decoder/generate_vl.go` (the comment above `GenerateQwenVLDeepstackSpans.recurrent`) on 2026-10-10.

```text
A recurrent family (the Gated-DeltaNet hybrids) takes NO resident REUSE branch and NO CPU-prefill-then-UploadKV bridge here, and
must be refused those rather than merely not engaged: every resident executor implements ResidentMRoPE, so the type assertions
below succeed on a CUDA-resident qwen3.5, and the bridge copies only layers that have KV, a DeltaNet layer has none, so it
is skipped and resident decode would start from a ZEROED recurrent state, with no error and wrong tokens (the failure class
62309847 fixed for reuse). Its one resident route is the resident m-RoPE prefill, taken only when the resident says it
builds the recurrent state itself (ResidentHybridMRoPEPrefill; the CUDA resident, dense hybrids, since 2026-10-06, P26b), and
otherwise an image turn on a recurrent family is CPU prefill + CPU decode. docs/multimodal.md P8 record, item 5.
```

## Model.checkImageBlockFitsWindow

Moved from `decoder/generate_vl.go` (the comment above `Model.checkImageBlockFitsWindow`) on 2026-10-10.

```text
checkImageBlockFitsWindow refuses an image block longer than the sliding window (docs/multimodal.md,
"Finishing this doc", F3). The bidirectional image-block mask this path uses (KVCache.attendHi, and the resident
image prefill that mirrors it) bounds a sliding layer's keys by the query's own window and never extends it back
to the block's start, so a block longer than the window would under-attend its own image. No shipped checkpoint
reaches it (Gemma 3's block is 256 tokens against a 1024 window; Qwen2.5-VL has no window, and its image tokens are
causal anyway); Gemma 4's own path applies the correction and does not come here. A refusal by name instead of a
quietly wrong answer.
```

## expertPager

Moved from `decoder/moepaging.go` (the comment above `expertPager`) on 2026-10-10.

```text
expertPager bounds the resident RAM of a MoE model's expert weights by paging
them on demand out of the read-only .giw mapping (idea #2,
docs/ideas-weight-memory.md). A 35B-A3B holds ~32 GB of experts but activates
only K·L per token; the router's top-k selection is the demand signal. The
generic span-residency pager (touch → fault-in WILLNEED, evict over budget
DONTNEED) now lives in aikit/mmap.SpanCache; this pager runs it with the
frequency-aware EvictLeastRecent policy (newExpertPager). This type holds only the
MoE-specific half: which experts alias the mapping and the touch hook the
router calls: moeMLP (mlp.go) for [NumExperts]expertWeights families, and gemma4MoEFFN
(forward_gemma4_moe.go) for gemma4's fused gate‖up + down experts. Releasing is lossless — the mapping is read-only
and file-backed, so an evicted expert merely re-faults from disk (output stays
bit-identical; the only cost is the cold-miss fault, ~+24 ms/token at a 16 GB
budget on the measured 35B-A3B — moepaging_spike_test.go).

Only experts whose quantized weights actually alias the mapping are managed;
heap-backed weights (a GGUF load) and the always-on shared expert are left alone.

Two backing modes, chosen once at build time (newExpertPager):
  - mmap+madvise (the default except on darwin): cache aliases the read-only mapping directly, WILLNEED faults it
    in, DONTNEED releases it. Zero-copy, but on darwin DONTNEED is a no-op (madvise_darwin.go)
    -- there is no real RAM cap on macOS with this mode.
  - owned-buffer pread (the darwin default since S5, 2026-09-24; Options.MoEPager / --moe-pager
    elsewhere; Lever 1b, task-moe-streaming.md):
    pool holds a fixed set of owned buffers it fills via pread, giving a firm cap on every
    platform at the cost of a memcpy per miss and losing .giw zero-copy aliasing.

Guarded by two internal mutexes (audit C-30 plus Lever 1b's own cross-call requirement — see
their doc comments below): the pager lives on *Model and StreamWeights supports concurrent
decode streams, so its shared LRU state is locked (SpanCache is not internally locked, and
pool needs its own protection for a different reason).
```

## expertPager.faultBase

Moved from `decoder/moepaging.go` (the comment above `expertPager.faultBase`) on 2026-10-10.

```text
S5 (task-never-swap-2026-09.md): process-wide getrusage(RUSAGE_SELF) minor/major
page-fault counts at pager creation — faultDelta() reports how many faults have
happened SINCE, the comparison this brief's own A/B (mmap mode's real WILLNEED faults
vs pool mode's pread, which should show near-zero additional faults) needs. Process-wide
on purpose, not pager-scoped: getrusage has no way to attribute a fault to a specific
mapping, so this is contaminated by whatever else the process does between the baseline
and the read — acceptable for an A/B run with nothing else happening, stated rather than
hidden as a per-fault-attributed number it is not. faultsOK false means the platform has
no probe (faultcount_other.go) — every unknown proceeds; pagerSummary omits the term.
```

## MoEPagerDefault

Moved from `decoder/moepaging.go` (the comment above `MoEPagerDefault`) on 2026-10-10.

```text
newExpertPager builds a pager over the experts of an mmap-backed MoE model, or
returns nil when paging doesn't apply (not MoE, not mmap-backed, or no expert
weights alias the mapping). budget ≤ 0 selects an automatic budget (~half of
available RAM); it is clamped to [one expert, total expert bytes]. giwPath is the
.giw file the mapping was built from (Model.GiwPath) -- only consulted when the
owned-buffer pread mode is requested (GOINFER_MOE_PREAD_CPU=1), to open an
independent fd for pread (the mmap's own fd is closed right after mapping).
MoEPagerDefault is S5's registered default (task-never-swap-2026-09.md): darwin, where
MADV_DONTNEED is a no-op and mmap mode therefore cannot enforce its budget, gets the owned-buffer
pool; every other platform keeps mmap mode, where DONTNEED works and the alias is free. The ONE
source for it: serve's --moe-pager default and a library Load with Options.MoEPager unset both
resolve here (they used to disagree on darwin — serve set an env var, a library caller got mmap).
```

## newExpertPager.addExpert

Moved from `decoder/moepaging.go` (the comment above `newExpertPager.addExpert`) on 2026-10-10.

```text
addExpert registers one expert under a stable identity (the address of its primary
weight struct — the same value the forward touches), collecting only the projections
that actually alias the mapping. MappedSpan returns nil for heap-backed (GGUF)
weights and the always-on shared expert, so those are silently skipped.

A kind-4 tensor carries TWO on-disk representations (canonical + row4,
docs/completed/task-w4a8-neon-bandwidth.md's "Format follow-on"), but the M==1 decode kernel
(MatmulBTW4A8Into) reads ONLY row4 whenever it's present — this arch's forward is
always M==1, decode and prefill alike (confirmed by "prefill path: sequential" on
every load). Registering both spans under one cache key was a real, measured bug
(docs/completed/task-zeno-compare.md's "At-scale acceptance run"): SpanCache.Touch issues
MADV_WILLNEED on EVERY span under a key, unconditionally, so a cold kind-4 touch
prefetched both copies from disk though only one was ever read — a fixed ~2x I/O
tax per miss that produced a ~25-30% throughput regression instead of the row4
kernel's proven gain. Fix: register only the span that will actually be read —
row4 when present, canonical otherwise. Never both.
```

## newExpertPager.policy

Moved from `decoder/moepaging.go` (the comment above `newExpertPager.policy`) on 2026-10-10.

```text
Frequency-aware (classic LRU tail) eviction: the router's demand signal is
skewed FREQUENCY, not a scan — the hottest ~10% of experts absorb ~72% of the
top-k picks — so the hot set must stay resident. SpanCache's default is
scan-resistant (evict-most-recent), which is right for the ANN cyclic scan but
evicts exactly the hot experts here (measured −51 pp hit rate at a 4 GB budget on
a real 35B-A3B). EvictLeastRecent restores it. See aikit mmap.EvictPolicy.
```

## expertPager.advisedBytes

Moved from `decoder/moepaging.go` (the comment above `expertPager.advisedBytes`) on 2026-10-10.

```text
advisedBytes returns cumulative bytes fetched from disk over every miss — WILLNEED-hinted
bytes in mmap mode, pread'd bytes in pool mode — independent of whatever else the machine's
disk is doing. A durable, contamination-proof I/O check: a member registering redundant spans
(the kind-4 double-WILLNEED bug this exists to catch, docs/completed/task-zeno-compare.md's
"At-scale acceptance run") shows up here directly as bytes-per-miss exceeding the expected
per-expert working set, immune to whatever an external tool like iostat would also be
counting on a shared machine.
```

## GenerateGemma4VL

Moved from `decoder/generate_gemma4_vl.go` (the comment above `GenerateGemma4VL`) on 2026-10-10.

```text
GenerateGemma4VL streams a continuation for a Gemma 4 multimodal prompt. Like
GenerateVL (Gemma 3) / GenerateQwenVL in shape.

Prefill dispatches between two forwards depending on the checkpoint:
sequential (prefillLogitsGemma4VL, E2B/E4B-class, causal) or batched
(prefillLogitsGemma4VLBidirectional, 26B-A4B/31B-class,
use_bidirectional_attention: "vision"). Decode is always the unchanged
per-token path either way — a decode token is never "inside" the image
block again, so no masking distinction applies post-prefill.

Resident GPU decode (gap 0, docs/multimodal.md): when the model has a resident, the CPU prefill's KV is
uploaded through the generic residentUploadPrefill bridge (the one Gemma 3/Qwen's GenerateVL use) and decode
runs resident. That was gated on UseBidirectionalAttention (the 26B-A4B/31B class) until no backend could run an
E2B/E4B; since S1 (docs/tasks/task-multimodal-support-2026-10.md) Metal does — the bridge skips KV-shared layers,
which own no KV, and embedResident carries the PLE inputs — so the E-model class is admitted too. CUDA runs the
E-model since S1 on CUDA and reaches this too (its prefill declines E-models, so an image turn prefills here on the
CPU and uploads); WebGPU still declines the E-model shape at load. A causal, non-E Gemma 4 keeps
decoding on the CPU after an image, as before: widening to it is untested on CUDA.

`imgHash`, `imgPos` and `imgLen` are also what the resident commit below records as this
turn's residentImageBlock, so a later turn's prefix scan (residentReuseLen) can tell this
image's span apart from ordinary text — Gemma4's soft-token placeholder ids are
content-independent (same id/count per patch grid), so two different images of the same size
produce IDENTICAL ids, and without a block record the scan cannot tell them apart (M-07,
docs/audit-2026-09-10.md).
```

## moepool.overview

Moved from `decoder/moepool.go` (the comment at the top of the file) on 2026-10-10.

```text
Lever 1b (docs/completed/task-moe-streaming.md): an owned-buffer pread cache for MoE expert
weights, replacing expertPager's mmap+madvise mode. Darwin's MADV_DONTNEED is a documented
no-op (madvise_darwin.go) -- an mmap-aliased cache can WILLNEED bytes in but can never
actually release them, so it gives NO real RAM cap on macOS. Owned buffers the pool itself
allocates and reuses give a firm cap on every platform: eviction here means "this slot's
bytes may now be overwritten", genuinely freeing that RAM for the next miss.

Mirrors metal/expertpool.go's shape (measured 1.26x faster darwin cold-stage vs mmap
byte-copy, zero page faults -- gemma4_moe.go) translated from GPU Buffer writes to plain
[]byte + WeightMat repointing via aikit/linalg's exported Wrap*/accessor API. No aikit API
change is needed: scales/rows/cols/group are already heap copies untouched by paging (only
the packed nibble/code payload bytes ever alias the mmap -- decoder/serialize.go's
giwReader.weightMat), and a WeightMat can be rebuilt from those plus a freshly pread payload.

The default on darwin since S5 (2026-09-24), where mmap mode cannot enforce its budget;
measured at 1.02x the mmap decode rate (docs/measurements/moe-pager-mode-darwin-2026-09-23.md).
Elsewhere --moe-pager=pool (Options.MoEPager) selects it.
```

## moePreadRateBytesPerSec

Moved from `decoder/moeworkingset.go` (the comment above `moePreadRateBytesPerSec`) on 2026-10-10.

```text
moePreadRateBytesPerSec is S4 item 5's own default pread-rate prior
(task-never-swap-2026-09.md): docs/completed/task-w4a8-neon-bandwidth.md measured ~3.7 GB/s at
concurrency 1 on this Mac's SSD. A DEFAULT, not a live measurement — the brief's own "Read
first" note names the eventual improvement ("until the guard measures its own, one 64 MB pread
at load"), not built here; this constant is what stands in for it until that lands.
```

## moeHitRatePrior

Moved from `decoder/moeworkingset.go` (the comment above `moeHitRatePrior`) on 2026-10-10.

```text
moeHitRatePrior maps a residency fraction (pager budget / total mapped expert bytes) to a
predicted LRU cache hit rate, interpolated from the only measured hit-rate curve in this tree —
CUDA's C′ expert cache, Gemma 4 26B-A4B int4, benchmarks.md §B4.1 (12.5% residency -> 57.3% hit
rate, 23.4% -> 76.1%, 31.25% -> 82.2%).

BORROWED, NOT MEASURED FOR THIS PAGER — stated as a PRIOR, per the brief's own wording, not a
claim about this pager's real behavior: a different architecture (CUDA host<->VRAM streaming vs
this pager's mmap/pread), though the SAME eviction discipline (LRU over routed-expert demand),
which is why the SHAPE is expected to transfer even though the exact numbers will not.

Piecewise-linear between the calibration points; clamped to [0, 0.999] so a zero budget predicts
a zero hit rate (no free lunch) and a budget covering every expert predicts NEAR-certainty
rather than exactly 1.0 — an exact 1.0 would make predictedMoETokPerSec's miss-bytes term
vanish and report +Inf, which is a worse failure mode (an unusable number) than a merely
optimistic finite one.
```

## predictedMoETokPerSec

Moved from `decoder/moeworkingset.go` (the comment above `predictedMoETokPerSec`) on 2026-10-10.

```text
predictedMoETokPerSec is S4 item 5's own arithmetic: predicted tok/s ~= 1 / (missBytes /
preadRate). Compute time is deliberately OMITTED — this repo has no measured CPU-paged-MoE
compute-only (I/O-free) per-token cost to anchor it on, and guessing one wrong would be worse
than leaving it out (a confidently wrong estimate is worse than none, per this repo's own
measurement discipline). Omitting it makes this an UPPER BOUND on tok/s (an optimistic
prediction), not a point estimate: I/O dominates a CPU-paged MoE's per-token cost in every real
measurement this tree has (R-05, task-recompute-audit.md: "MoE is ~70% of a CPU-paged 35B
token"), so the omitted compute term can only make the real rate SMALLER, never larger — the
same "can only get stricter" direction S4 items 1/3 already use for their own margins, applied
here to a prediction rather than a refusal threshold.

activeBytesPerToken is topK x expertBytes + denseCoreBytes — the caller's job to resolve
(moeWorkingSetPrediction, below); this function is pure arithmetic over numbers already known.
hitRate is clamped to [0, 0.999] for the same reason moeHitRatePrior clamps its own output.
Returns 0 (not a rate) for any input this function cannot turn into a positive, finite
prediction — "don't know" is what the caller should treat that as, not "zero tok/s".
```

## ExactAttentionScoper

Moved from `decoder/spec_verify_guard.go` (the comment above `ExactAttentionScoper`) on 2026-10-10.

```text
ExactAttentionScoper is implemented by a resident whose M=1 decode attention has an OPT-IN lane that is not
bit-identical to the tree its batched verify uses (cuda's flash-decode lane, GOINFER_CUDA_FLASH_DECODE). While a
scope is held the resident runs its EXACT decode attention, so a speculative generation decodes and verifies with
one tree and stays token-identical to plain greedy on that tree — instead of refusing to run (the V-sum spike's
treatment, which has no scope to enter and so still refuses via DecodeVerifyDiverger).

A scope is per GENERATION and must be held from before the first decode step of a speculative run until its last;
nesting is allowed (a counter, not a flag). The resident's KV is single-tenant, so no plain generation can be
interleaved inside a speculative one on the same Model. Exported so an implementation can pin itself to it at
compile time, for the same fail-open reason as DecodeVerifyDiverger.
```

## Model.SpecDecodeConflict

Moved from `decoder/spec_verify_guard.go` (the comment above `Model.SpecDecodeConflict`) on 2026-10-10.

```text
SpecDecodeConflict reports why speculative decoding on this model would not be lossless, or nil
when it would be. Two independent sources, checked in order:

 1. M-09 (docs/audit-2026-09-10.md): the STAGED webgpu backend's own MatmulW4A8
    (gpu/backend.go) declines any M != 1 and falls through to the CPU int4 kernel — so a
    staged-int4 model on webgpu decodes each M=1 token on the device (a WGSL f32 GEMV with f16
    group scales) but verifies M>1 batches on the CPU's integer kernel: two different kernel
    implementations of the same logical matmul, with no numeric tolerance between them ever
    measured or pinned. This is a STAGED-path risk specifically — it exists because m.resident
    is nil, not despite it — so it is checked before, and independent of, the resident check
    below.
 2. The resident's own verify-vs-decode divergence (DecodeVerifyDiverger), for backends whose
    resident M=1 decode and batched verify can disagree (cuda's flash-decode V-sum spike is the
    only such case today).

Every speculative loop that verifies on the resident — the n-gram path, the two-model draft path,
and block drafting — is sold as token-identical to plain greedy. That holds only while a verified
position scores exactly as a decoded one does. When either source above says the two can diverge,
those entry points refuse instead of running, the same shape specRollbackSafe uses for state a
rollback cannot restore: refuse at the source, never degrade silently.

Exported for callers that must refuse at STARTUP rather than per request. cmd/serve's --spec
ngram treats a per-request spec error as "fall back to plain decode", which would leave an
operator who asked for speculation serving at 1x with no signal; it checks this once at load.

The RESIDENT half is NOT consulted by the staged speculative paths (grammar-fused, and any
Session-driven n-gram run): those verify on a CPU cache, never touch the resident, and so
cannot see a resident divergence. The STAGED (M-09) half above applies to them too in principle
— a staged webgpu-int4 model's decode/verify split is the same regardless of caller — but neither
of those entry points calls this function today, matching this function's own
"exported for callers that must refuse at startup" scope; they are unaffected either way.
```

## residentneed.overview

Moved from `decoder/residentneed.go` (the comment at the top of the file) on 2026-10-10.

```text
One memory-accounting path per backend (docs/tasks/task-memory-accounting-2026-09.md).

Before this, Plan("metal") — what `fit` reports — and Metal's own pre-build guard priced the same
load differently, measured on an M1 Pro (2026-09-25, qwen2.5-coder-1.5b at ctx 4096):

  - a direct .gguf: the guard 4.33 GB, Plan 2.34 GB. Plan had no term for Metal's second copy of
    the dense weights (unified memory holds the host WeightMat AND the re-packed device buffer —
    ResidentHostCopyBytes), 2.1 GB here, so `fit` could say "resident" for a load the guard refuses;
  - a v14 .giw (weights aliased, host copy 0): the guard 1.40 GB, Plan 1.52 GB. Plan priced Metal's
    KV at f32; Metal only allocates f16 (the f32 KV kernels are compiled out, metal/model.go's
    r.kvF32), so Plan's KV was twice the real one.

ResidentNeedBytes is now the single definition both use. The KV half is ResidentKVBytes, which
prices a backend's KV the way that backend ALLOCATES it — the layout differs per backend, so "one
formula" has to be one function with a per-backend layout, not one flat formula (Metal allocates
the full context for every attention layer, sliding-window ones included, padded to 8 positions;
kvBytesForCtx's sliding-window cap models the CPU ring buffer and would under-count Metal).

CUDA got its own layout on 2026-09-25 (docs/measurements/memory-accounting-cuda-2026-09-25.md), after
measuring real CUDA residents against Plan's per-position formula: CUDA allocates f32 K/V whatever
KV precision was asked for (it reads no KV-precision option), so a requested f16 / i8 halved or
quartered Plan's figure below what CUDA builds; and an MLA layer holds ONE latent buffer, not K and
V, so Plan doubled a DeepSeek-class model's KV. Dense, sliding-window (full context, no ring buffer),
per-layer-geometry and DeltaNet-hybrid models already agreed. Other backends (webgpu, cpu) keep Plan's
existing per-position formula unchanged.
```

## prefillQwen35VLBatched

Moved from `decoder/generate_vl_qwen35.go` (the comment above `prefillQwen35VLBatched`) on 2026-10-10.

```text
prefillQwen35VLBatched is P26: every projection one M=len(ids) matmul, so each weight is read once for the prompt instead of
once per token (a 662-token image turn on Qwen3.5-0.8B took ~34 s one token at a time, which is decode speed). The DeltaNet
recurrence stays sequential inside runLayersQwen35N; its full-attention layers rotate q and k by cache.mropePos exactly as the
per-token loop does. NOT bit-identical to that loop (a batched matmul can reduce in a different order):
TestQwen35VL_batchedPrefillMatchesPerToken bounds the difference, and the HF-golden tests hold either path to the same bars.
The caller has checked qwen35BatchNAnyPos and set cache.mropePos / mropeDelta.
```

## prefillLogitsGemma4VL

Moved from `decoder/generate_gemma4_vl.go` (the comment above `prefillLogitsGemma4VL`) on 2026-10-10.

```text
prefillLogitsGemma4VL sequentially prefills a Gemma 4 multimodal prompt — ids
with a run of imgLen image-placeholder ids at [imgPos, imgPos+imgLen) — and
returns the last-position logits.

UNLIKE prefillLogitsVL/prefillLogitsQwenVL, this walks ids ONE TOKEN AT A TIME
via runLayersGemma4/runLayersGemma4FromEmbed (decoder/forward_gemma4.go) —
Gemma 4's own-forward layer loop has no batched (multi-token-at-once) variant
(gemma4 is dispatched via arch.ownForward(), not the generic
runLayersFromEmbedN path every other multimodal family's CPU prefill uses).
This is NOT an approximation for the checkpoints this reaches: E2B/E4B ship
use_bidirectional_attention="" (docs/multimodal.md's P7 entry, confirmed
against a real checkpoint), so the real HF forward attends to the image block
strictly causally too — a sequential per-position walk IS that forward, not a
stand-in for a bidirectional one. A checkpoint with
use_bidirectional_attention="vision" (26B-A4B/31B) needs a genuinely batched,
blockwise-masked forward — see prefillLogitsGemma4VLBidirectional /
runLayersGemma4FromEmbedN (decoder/forward_gemma4_batched.go) below.
GenerateGemma4VL dispatches to the right one; this function itself is never
called for a "vision"-mode checkpoint.
```

## prefillLogitsQwen35VL

Moved from `decoder/generate_vl_qwen35.go` (the comment above `prefillLogitsQwen35VL`) on 2026-10-10.

```text
prefillLogitsQwen35VL is prefillLogitsQwenVL for the Gated-DeltaNet hybrid (qwen3_5 /
qwen3_5_moe, P8a): the same splice — the merged tower rows replace the placeholder run
[imgPos, imgPos+imgLen) raw, no embed scale, exactly HF's inputs_embeds.masked_scatter — and the
same m-RoPE positions on the full-attention layers, but through the per-token
runLayersQwen35FromEmbed loop, because the recurrent state must see every token in order and
there is no batched hybrid prefill (canBatchN is false for this family).

It leaves cache.mropePos / cache.mropeDelta set so decode past the prompt rotates at
seqPos+delta, exactly as prefillLogitsQwenVL does.
```

## topFilterLogits.Z

Moved from `decoder/sampler.go` (the comment in the body of `topFilterLogits`) on 2026-10-10.

```text
P2b step 3: the same fixed-chunk fold as the temperature-only path. This Z feeds the
nucleus cut, so regrouping it also moves boundary draws — which is exactly why it lands in
the SAME release rather than dribbling out later as a second seed change.
```

## AdaptiveDepth

Moved from `decoder/spec_adaptive.go` (the comment above `AdaptiveDepth`) on 2026-10-10.

```text
AdaptiveDepth chooses the per-round draft depth from a running estimate of
per-position acceptance (an EMA of realized accepts), replacing a fixed K. It is
the 04-adaptive-depth controller: depth grows on copy-heavy streams (α→1) and
collapses toward plain decode on novel text (α low), which kills the fixed-K
over-draft that makes low-acceptance workloads SLOWER than plain (a verify pass
over draft tokens that mostly get rejected isn't paid back).

The rule (00-core §2): extend depth while the expected marginal committed token
— probability α^d that the chain reaches depth d — still beats the marginal cost
of one more verify node, Theta:

	D = floor( ln(Theta) / ln(α) ),  clamped to [0, min(MaxDraft, proposed)]

Theta is the relative cost of one extra verify node on *this backend* — measure
it. It is ~0.5 on the batched-CPU ForwardN path here (a node is half a target
step), and →0 on a fully memory-bound GPU verify (extra nodes are ~free, so deep
drafts always pay). D=0 means "don't speculate this round" (plain decode); a
periodic probe forces D≥1 occasionally so a stream that becomes copyable again
can climb back out of D=0.
```

## Sampler.distVectorFrom.default

Moved from `decoder/spec_sample.go` (the comment in the body of `distVectorFrom`) on 2026-10-10.

```text
P-04 (audit-2026-09-10): P-14 (09-02) reused distBufN on the greedy and filtered
branches above/below but left this one — the server's DEFAULT sampling shape
(temperature 1, no top_k/top_p/min_p) — calling the always-allocating softmaxStable
directly. softmaxStableInto shares the same scratch the other two branches already use.
```

## DecodeVerifyDiverger

Moved from `decoder/spec_verify_guard.go` (the comment above `DecodeVerifyDiverger`) on 2026-10-10.

```text
DecodeVerifyDiverger is implemented by a resident runner whose M=1 decode and batched verify can
be made to disagree (cuda's flash-decode V-sum spike is the only such case today). It is an
OPTIONAL interface, asserted rather than added to ResidentForward, so backends that cannot diverge
— every one of them except an explicitly opted-in CUDA resident — carry no method for it.

EXPORTED SO AN IMPLEMENTATION CAN PIN ITSELF TO IT AT COMPILE TIME. An optional interface fails
OPEN: rename the method on either side and the type assertion in SpecDecodeConflict simply stops
matching, the guard reports "no conflict" forever, and nothing errors. The cuda package holds a
`var _ decoder.DecodeVerifyDiverger = (*cudaResident)(nil)`, which turns that silent disarm into a
build failure.
```

## Model.genNgramInto.adaptive

Moved from `decoder/spec_ngram.go` (the comment in the body of `genNgramInto`) on 2026-10-10.

```text
MC4 candidate (docs/tasks/task-concurrency-2026-09.md): "speculate when alone, batch under load".
adaptive claims the resident PER ROUND (below) instead of once for the whole generation, so MC3's
batcher can admit other generations between rounds. Scoped to greedy: a sampled generation takes
today's always-exclusive path unchanged — the batch-yield round needs a device-side greedy verify
via the batcher, which does not exist for a sampled draw here.
```

## Model.genNgramInto.neverYield

Moved from `decoder/spec_ngram.go` (the comment in the body of `genNgramInto`) on 2026-10-10.

```text
MC4 "spec inside a batch" premise measurement (docs/tasks/task-concurrency-2026-09.md): forces
this generation to wait for exclusive access rather than ever yielding, reproducing today's plain
-spec ngram behavior (one at a time) even with -spec-adaptive on. Not a shipped policy — see the
knob's own comment (decoder/knobs.go).
```
