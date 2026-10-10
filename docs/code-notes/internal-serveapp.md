# internal/serveapp: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/serveapp`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## maxTopLogprobs

Moved from `internal/serveapp/openai.go` (the comment above `maxTopLogprobs`) on 2026-10-09.

```text
maxTopLogprobs is the largest `top_logprobs` a request may ask for — OpenAI's own documented
ceiling, so no compatible client is refused by it. Unbounded, one request retains
max_tokens x top_logprobs entries and materializes a map per entry before writing a byte
(audit-2026-09-02 C-08).
```

## maxOutputTokensCeiling

Moved from `internal/serveapp/openai.go` (the comment above `maxOutputTokensCeiling`) on 2026-10-09.

```text
maxOutputTokensCeiling is a hard upper bound on a request's max_tokens. KV is
preallocated as len(prompt)+max_tokens per layer, so an unbounded value (e.g.
{"max_tokens": 2000000000}) triggers a fatal, unrecoverable Go "out of memory"
throw that kills the server for every client (audit C-18). A request above this
is rejected 400 rather than clamped, so the caller learns its request was too big.
```

## loadedModel.blockSpec

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.blockSpec`) on 2026-10-09.

```text
blockSpec is an attached pretrained block drafter (--drafter), nil when unused. Attached
ONCE at load: the weight upload is a per-process cost, and doing it per request measured
0.17x — a 6x loss — with the loop itself perfectly healthy (docs/spec/08).
```

## loadedModel.turns

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.turns`) on 2026-10-09.

```text
decode worker — replaces the OLD lm.mu's admission role. A mutex has no notion of context,
so a waiting request could not notice its own client disconnecting (or a K2 halt) without
first being granted the lock. Zero value ready to use, same as the mutex it replaces.

sessMu guards the sessionLRU, which is not goroutine-safe (sessions.go): drive's acquire and checkin, and the
background goroutines (admin.go's load-time restore, liveness.go's idle-close save, main.go's graceful-shutdown
save and idle-demote ticker). It is held around those operations only, NOT across a generation (MC3c,
docs/tasks/task-concurrency-2026-09.md): the session a generation is using is CHECKED OUT instead
(sessionLRU.busy), and the LRU never hands out, evicts, demotes, saves or even reads a busy session. That is
what lets several generations of one CPU model run at once (-max-concurrent), and it keeps the old safety
property — nothing frees or moves memory out from under a running generation — for one generation too.
```

## loadedModel.maxTokBytes

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.maxTokBytes`) on 2026-10-09.

```text
maxTokBytes is the byte length of the longest token in the vocab, computed once. It
bounds tokenization cost (G1): a servable prompt is ≤ ctx tokens, so its text is ≤
ctx·maxTokBytes bytes; any longer input needs > ctx tokens and cannot fit — reject it
before the O(n) BPE runs, instead of tokenizing a multi-MiB body to completion (~27 s
on a 32 MiB body) and only then comparing against the context window.
```

## chatInputBytes

Moved from `internal/serveapp/openai.go` (the comment above `chatInputBytes`) on 2026-10-09.

```text
chatInputBytes sums the tokenizable text across chat messages — the input the BPE runs
over (JSON structure and image data are not tokenized), so it is what the G1c guard bounds.

M-15 (audit-2026-09-10): also sums each message's REPLAYED tool_calls[].function.arguments.
messagesToTurns converts an assistant message's ToolCalls into chat.Turn.ToolCalls
unconditionally — a chat template renders that replay into the prompt whenever it appears in
history, regardless of whether tools are active THIS turn — so a large replayed arguments blob
used to pass this guard in constant time and then run the full BPE over it anyway.
```

## toolSchemaBytes

Moved from `internal/serveapp/openai.go` (the comment above `toolSchemaBytes`) on 2026-10-09.

```text
toolSchemaBytes sums an OpenAI-shaped tool declaration list's rendered bytes (M-15,
audit-2026-09-10): RenderToolsSegments renders every tool's name/description/parameters into
the prompt whenever tools are active, so a large schema list must be priced at the same guard
that already prices the messages, not left to run the full BPE unpriced.
```

## server.lookupLocked

Moved from `internal/serveapp/openai.go` (the comment above `server.lookupLocked`) on 2026-10-09.

```text
pick resolves the OpenAI `model` field to a loaded generative model: an exact
served-name match, else (for single-model OpenAI compatibility, where clients
send an arbitrary name) the sole model when only one is loaded. nil otherwise —
the handler returns an OpenAI-shaped 404.
lookupLocked resolves a request's model name to a loaded entry. The CALLER MUST HOLD regMu
(read or write). It is deliberately unexported and lock-requiring so it cannot be the route a
handler uses — withModel (liveness.go) is the ONLY way a request reaches a *loadedModel, because
withModel also takes the liveness read-lock that keeps the model alive for the request's duration.
Adding a handler that calls this directly would skip that lock; there is no exported pick to call.
```

## server.modelByName

Moved from `internal/serveapp/openai.go` (the comment above `server.modelByName`) on 2026-10-09.

```text
servedNames lists the loaded generative + embedding model ids (sorted).
modelByName is an EXACT registry lookup under the read lock — unlike pick, which falls back to
"the only loaded model" for any name (right for request routing, wrong for listing: it would
attach a decoder's paths to the embedding-model entry).
```

## chatReq.N

Moved from `internal/serveapp/openai.go` (the comment above `chatReq.N`) on 2026-10-09.

```text
N is OpenAI's "how many choices" field (N-30, docs/audit-2026-09-10.md — this repo
generates exactly one). Was entirely unparsed: an unrecognized JSON key is silently
dropped, so `n: 3` used to get one choice back under a 200 rather than an error naming
what was ignored. *int (not int) so n:0 and "omitted" are distinguishable from n:1.
```

## completionReq.Logprobs

Moved from `internal/serveapp/openai.go` (the comment above `completionReq.Logprobs`) on 2026-10-09.

```text
Logprobs shadows sampling.Logprobs (embedded below) for the /v1/completions surface.
The legacy Completions API types logprobs as an INTEGER (# of top alternatives), not the
chat API's bool — the standard SDK sends `logprobs: 5`, which failed to decode into a bool
and returned a 400 leaking Go struct/field names (audit M-06). The outer (shallower) field
wins during JSON decode, so req.sampling.Logprobs stays false and the expensive per-token
logprobs path never engages; the handler 400s explicitly when it is set (unimplemented here).
```

## serveChatText.guard

Moved from `internal/serveapp/openai.go` (the comment above `serveChatText.guard`) on 2026-10-09.

```text
Reject an over-context prompt before tokenizing it (G1c): turns a multi-MiB body from
~27 s of BPE + gigabytes of ids into a byte-length comparison.
```

## serveChatText.heartbeat

Moved from `internal/serveapp/openai.go` (the comment above `serveChatText.heartbeat`) on 2026-10-09.

```text
N-24 (docs/audit-2026-09-10.md): nothing else is sent between here and the first
token — on CPU that gap is the whole prefill (an 8k agent prompt ~270s), against a
300s harness idle timeout. Streaming per token once generation starts does not cover
the prefill window itself, same shape as the buffer-then-stream sites M-19 fixed.
```

## validateN

Moved from `internal/serveapp/openai.go` (the comment above `validateN`) on 2026-10-09.

```text
validateN rejects an explicit "n" other than 1 (N-30, docs/audit-2026-09-10.md): this server
always generates exactly one choice, and "n" used to be an unrecognized JSON key that decoded
silently — a request asking for n:3 got ONE choice back under a 200, with nothing in the
response naming what was ignored. n omitted (nil) is the default (1) and passes; n:1 passes;
anything else is a clean 400 rather than a silently wrong choice count.
```

## loadedModel.contextWindow

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.contextWindow`) on 2026-10-09.

```text
contextWindow is the context cap prepare enforces for one request: the model's MaxPositions,
lowered to the resident KV cap when the request runs the stateless resident path. ONE function for
both the enforcement and what /v1/models publishes as context_window (W8), so the number a client
plans against is the number that rejects it. 0 = unknown.

Why the resident cap: on a resident backend the stateless path prefills the fixed-size resident KV,
which is often smaller than MaxPositions. A prompt in (residentCap, MaxPositions) passes the
MaxPositions check, then dies mid-prefill with a 500 whose body leaks the internal "use the staged
path" hint (there is no staged fallback on the stateless resident path) — so it is rejected as a
clean context_length_exceeded 400 instead (audit R-10). Only when the request actually runs
stateless-resident — never for vision (CPU VL) or adapter (session path; N-39: not purely staged
since G3, but this specific check is about the STATELESS path's own fixed KV cap, which adapter
requests never enter) requests, which are bounded by MaxPositions, not the resident cap (audit F-01).
```

## loadedModel.residentPath

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.residentPath`) on 2026-10-09.

```text
residentPath reports whether a text-completion request against lm will actually run the
stateless GPU-resident decode path, for contextWindow/prepare's residentPath argument (M-01,
docs/audit-2026-09-10.md). NOT `lm.adapter == ""`: an adapter model's FIRST turn
(prefillFrom==0) still runs resident GPU decode when one is active
(decoder/model.go's generateInto useGPU condition doesn't exclude adapters), so keying this on
"no adapter" left an adapter+resident request enforced against the uncapped MaxPositions while
generateInto actually bound it to the smaller ResidentContextCap() — dying mid-prefill with a
leaking 500 instead of a clean 400 (the R-10 regression this fixes). ResidentActive() is safe on
every other combination too: not resident (adapter or not) makes ResidentContextCap() report 0,
so contextWindow's own `rc > 0` guard no-ops regardless of what residentPath says — this only
tightens the one case that was actually broken. Guarded against lm.model == nil (an embedding-only
entry) the same way contextWindow itself is, even though no text-completion caller should reach
prepare with one.
```

## prepare.temperature

Moved from `internal/serveapp/openai.go` (the comment above `prepare.temperature`) on 2026-10-09.

```text
G4: temperature has the same lower bound as top_p (rejected < 0), for consistency —
previously top_p=-1 was a 400 but temperature=-1 was accepted. It is not silently-wrong
output: SampleWithInfo short-circuits Temperature <= 0 to greedy argmax before any logit
scaling, so a negative temperature decoded greedily rather than inverting the ordering.
0 is the documented greedy/deterministic setting and stays valid; only negatives are rejected.
```

## prepare.topP

Moved from `internal/serveapp/openai.go` (the comment above `prepare.topP`) on 2026-10-09.

```text
M-02: top_p == 0 is the tightest nucleus (the single most-likely token), which is greedy — the old
`< 1` path stored 0, and the sampler treats TopP == 0 as DISABLED, so the request asking for the
tightest filter got a full-vocab draw. Reject outside [0,1]; map explicit 0 to greedy.
```

## prepare.topLogprobs

Moved from `internal/serveapp/openai.go` (the comment above `prepare.topLogprobs`) on 2026-10-09.

```text
C-08: top_logprobs was the one sampling field `prepare` passed through unvalidated, and it is
the most expensive one to get wrong. Each retained entry is a TokenLogprob, and the response
builder then materializes a map[string]any per entry BEFORE writing a byte, so
{logprobs:true, top_logprobs:150000, max_tokens:4096} on a 152k-vocab model retains
4096 x 150k of them (~9.8 GB) and OOM-kills the process — a fatal Go allocation failure, not
a 500. top_logprobs:20000 with max_tokens:2000 already reaches ~6 GB. One request does it.

[0,20] is OpenAI's own documented range, so this rejects nothing a compatible client sends.
```

## prepare.maxTokens

Moved from `internal/serveapp/openai.go` (the comment above `prepare.maxTokens`) on 2026-10-09.

```text
A negative/zero value reaches NewCache(len(prompt)+maxTokens) → makeslice with a
negative cap → an unrecovered panic (fatal on the VL path's bare goroutine — audit
C-19); a huge value OOM-kills the server (C-18). Reject both here so every endpoint
that calls prepare (OpenAI/responses/tools/vision) gets a clean 400. The Anthropic
endpoint already rejects <= 0 upstream.
```

## prepare.clamp

Moved from `internal/serveapp/openai.go` (the comment above `prepare.clamp`) on 2026-10-09.

```text
C-18: bound max_tokens by the model's context window. The KV cache is preallocated as
NewCache(len(prompt)+max_tokens); the server ceiling above (131072) is far larger than most
models' context, so a request at the ceiling against a small-context model preallocates tens of
GiB per layer and OOM-kills the server. The model cannot attend past MaxPositions, so tokens
beyond it are wasted anyway — clamp to what fits, matching the resident path's ContextCap clamp
(decoder/model.go). Only shrinks; a request already within context is untouched. (A prompt that
itself exceeds the context is C-20's concern; here we only bound the max_tokens contribution.)
```

## prepare.admit

Moved from `internal/serveapp/openai.go` (the comment above `prepare.admit`) on 2026-10-09.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the load-time fit guard
prices the worst case a request COULD reach, once, at load — it cannot see the request
that actually arrives. Runs LAST, after gr.promptIDs/gr.maxTokens are fully resolved
(including the C-18 clamp above), so the numbers in a refusal are the real ones, and every
prepare() caller gets this for free rather than needing its own copy of the check.
```

## prepare.share

Moved from `internal/serveapp/openai.go` (the comment above `prepare.share`) on 2026-10-09.

```text
MC3c: with several generations allowed at once, this request's prefill shares the margin with the ones running
or queued ahead of it — counted now, capped at the model's concurrency. A lone request keeps the whole margin.
So does a request that prefills on a GPU resident: its prefill passes run one at a time (MC3's exclusive
section), and the margin is live memory, which already excludes what the other generations hold. Split there,
it refused ~1000-token 7B prompts under MC3 (docs/measurements/spec-vs-batching-metal-2026-09-27.md §5).
```

## contextLengthError

Moved from `internal/serveapp/openai.go` (the comment above `contextLengthError`) on 2026-10-09.

```text
contextLengthError rejects a prompt that alone fills or exceeds the model's context window (C-20).
MaxPositions is loaded but was never compared to len(prompt): a multi-MiB body tokenizes to ~1M ids
and preallocates tens of GiB of KV (NewCache is sized len(prompt)+max_tokens) → OOM-kill; and even
within memory, positions past the trained context drive out-of-range RoPE and return plausible
garbage under HTTP 200 instead of a 400. ctx ≤ 0 (unknown) never rejects.
```

## contextLengthErrorFor

Moved from `internal/serveapp/openai.go` (the comment above `contextLengthErrorFor`) on 2026-10-09.

```text
contextLengthErrorFor is contextLengthError plus the remedy, for the case where the window that rejected the prompt is the
server's resident KV capacity and not the model's own (ctx < modelWindow): a coding agent's first request is ~11k tokens
(R19, docs/tasks/task-first-hour.md), the default resident capacity is 8192 on CUDA and 4096 on Metal, and the 400 used to
say only "context window is 8192" — which reads as the model's limit, so the client (opencode) compacted 34 times instead of
the operator raising -ctx. When the model's own window is the limit, -ctx cannot help and the message is unchanged.
```

## seedOrRandom

Moved from `internal/serveapp/openai.go` (the comment above `seedOrRandom`) on 2026-10-09.

```text
seedOrRandom returns the request's seed, or a fresh random seed when absent (M-03). OpenAI's contract
is that an OMITTED seed varies output run to run — best-of-N, "regenerate", and agent retry-for-
diversity all depend on it — but deref(sm.Seed, 0) pinned every seedless request to the deterministic
seed-0 stream. A supplied seed (including 0) is still honored verbatim for reproducibility.
```

## loadedModel.drive

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.drive`) on 2026-10-09.

```text
drive runs the generation, applying stop strings and UTF-8 holdback, calling
onText with each newly-completed text fragment. Returns the finish reason
("stop" | "length" | "cancelled" — K1, docs/tasks/task-halt-2026-09.md), the completion
token count, (non-stream) per-token logprobs, the stop string that was hit
(empty unless a stop sequence ended the turn), how many leading prompt tokens
this generation's prefill skipped via resident/session reuse
(decoder.Generation.PrefillReused — 0 when nothing was reused or the path
doesn't track it), the K1 admin-cancel reason (empty unless an admin cancel —
not a client disconnect or shutdown — ended the turn), and any terminal
generation error (nil on a clean end — see genErr). The context is cancelled
on a stop-string hit to end generation.

gens is the server's K1 registry; nil (or gr.id == "") skips registration — every
generation funnels through here or driveVL, so this is the one place that bookkeeping
lives, not each of the ~15 call sites (see generations.go's own doc comment).

jobs is J2's store (task-work-queue-2026-09.md) — every job store instance always exists (see
newJobStore), so this is nil only in a unit test calling drive directly. Gated on the SAME
gr.id != "" condition as K1's registration, sharing the same id, so the two records can be
joined by id later (J3) without a retrofit.
```

## drive.adminCancel

Moved from `internal/serveapp/openai.go` (the comment above `drive.adminCancel`) on 2026-10-09.

```text
K1's cancel must land on `parent` itself, not on `ctx` below (drive's own
stop-string-derived child context) — streamTokens tells an admin cancel apart from a
stop-string hit by checking parent.Err(), specifically BECAUSE the stop-string cancel
only ever touches ctx (see streamTokens' own doc comment, M-23). Registering ctx's
cancel here would make an admin cancel indistinguishable from a natural stop:
discovered by the K1 gate test failing to see "cancelled" even though the generation
genuinely stopped — cancelling ctx silences the loop but parent.Err() stays nil.
```

## loadedModel.drive.resident

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.drive.resident`) on 2026-10-09.

```text
GPU-resident models take the STATELESS path. decoder.Generate only engages the resident
DecodeRunner when there is no session commit and no prefix reuse (model.go:
useGPU = resident != nil && prefillFrom == 0 && commit == nil) — the resident's KV lives
on the GPU, while a session's prefix-reuse cache is CPU-side, so the two cannot both be
the source of truth. Going through a session therefore silently dropped every request to
the staged/CPU path: measured 13 tok/s vs ~460 resident on a 0.5B (RTX 2070 SUPER).

Trading prefix reuse for resident decode is a large net win and semantically safe: the
OpenAI API is stateless (the client resends the whole conversation), so sessions here are
purely a TTFT optimisation, not correctness.

N-gram spec-decode DOES mix with residency: the drafter is pure-Go (prompt-lookup, no GPU
memory, unlike the two-model draft that measured 0.11x against a GPU target), and the verify
is the resident batched ForwardN (one weight stream for the whole [cur, draft…] run — the D1
win, ~1.8x mid-context on copy-heavy traffic). genNgramInto claims the shared resident KV
(resBusy) like Generate. Constrained/tool requests (grammar masker) keep plain resident
Generate for now; a validation error (sampler not yet on the spec path) falls back to plain
Generate before the KV is touched, so the fallback is exact.
Adapter (compute-time LoRA) models MUST NOT take the stateless resident path: the LoRA is
applied only through the session binding (sessionLRU.bindAdapter → Session.UseAdapter → the
cache's lora), and the stateless Generate/GenerateNgram… run on a fresh cache with lora == nil,
so they'd silently return BASE-model output. Route adapter requests down the session path below
instead; base models keep the resident fast path here (audit R-01). Since G3
(docs/tasks/task-gpu-paths-2026-09.md), the session path itself is NOT always staged/CPU for
an adapter: generateInto's useGPU (decoder/model.go) admits prefillFrom==0 with a bound
resident adapter onto the resident GPU path — a session's FIRST turn. A later turn on the
same session (prefillFrom>0, continuing off the reused warm prefix) still drops to CPU;
nothing wires compute-time LoRA into the resident prefix-reuse path yet (N-39).
```

## streamTokens.sb

Moved from `internal/serveapp/openai.go` (the comment above `streamTokens.sb`) on 2026-10-09.

```text
sb accumulates the decoded text INCREMENTALLY instead of re-decoding the whole `ids`
sequence every token (audit R-08: was O(n^2) in output length). decode()'s per-token loop
(tokenizer/sentencepiece.go) has no state that depends on chunk boundaries — a byte-fallback
token's raw byte is written out whenever a later flush happens, and concatenation is
associative regardless of WHEN that flush lands — so DecodePiece(id) appended one token at a
time is byte-identical to DecodeContinuation(ids) computed fresh each time.
TestDecodeContinuation_isIncrementallyAssociative (tokenizer/) proves this directly across a
multi-byte-emoji byte-fallback run, the case this used to be cautious about. strings.Builder,
not `text += piece`: Go strings are immutable, so naive concatenation is itself O(n) per
append and would silently reintroduce the O(n^2) this exists to remove.
```

## streamTokens.tail

Moved from `internal/serveapp/openai.go` (the comment above `streamTokens.tail`) on 2026-10-09.

```text
tail is text[printed:] — the not-yet-emitted suffix, bounded (does not grow with total
output length). Scanning it instead of the whole text is the other half of R-08's fix:
text[:printed] provably never contains a stop match, complete or in progress. Any
complete match either lies entirely in tail (found here), or would have to start before
printed and extend into it — impossible, because stopTailHold's own invariant is exactly
"hold back every suffix of the CURRENT text that could be a stop's prefix", so printed
never advances past the start of a still-possible match; if one completes, it completes
within what stopTailHold already held back, i.e. within tail. firstStop returns an
offset into whatever string it searches, so cut/end below are translated back to
absolute offsets into text (+= printed) before use.
```

## logprobImpossible

Moved from `internal/serveapp/openai.go` (the comment above `logprobImpossible`) on 2026-10-09.

```text
logprobImpossible is the logprob reported for a token whose probability is exactly zero: OpenAI's own convention (-9999.0), because -Inf, the true value, cannot be written in JSON.
A zero-probability candidate is normal, not an error: when the thinking budget forces the end-of-thinking token, that token has probability 1 and every other candidate, including the
filler entries that pad the top-k list, has none (S6's 35B image check, 2026-10-09: step 24 of a 32-token answer). NaN is NOT mapped: it would be a bug, and writeJSON reports it.
```

## admissionRecord

Moved from `internal/serveapp/admission.go` (the comment above `admissionRecord`) on 2026-10-09.

```text
admissionRecord is what a future scheduler reads to decide ordering — J1 keeps strict FIFO and
never inspects this itself (task-work-queue-2026-09.md: "J1 itself keeps strict FIFO — it
establishes the structure and changes no order"). promptIDs is this repo's own "session/prefix
key": sessionLRU (sessions.go's bestExtend) has no client-visible session id at all — it matches
purely by longest-common-prefix over a request's actual prompt token ids against the sessions
currently resident — so the ids themselves are what a later prefix-aware scheduler (J6) would
need to run that same match against the LRU at pick time.
```

## admission

Moved from `internal/serveapp/admission.go` (the comment above `admission`) on 2026-10-09.

```text
admission is a size-1, FIFO, context-aware turn-granter (J1, task-work-queue-2026-09.md),
replacing loadedModel's plain sync.Mutex. A mutex has no notion of context: a waiter blocked on
Lock() cannot notice its own client disconnecting, or a K2 halt landing, until it is actually
GRANTED the lock — at which point it has already occupied its place in line for nothing, and
everything behind it waits that much longer. admission.enter drops out the instant ctx ends,
whether or not the turn has been granted yet, so a dead request never delays a live one.

FIFO order only; it does not reorder waiters on anything in admissionRecord — that is left for
J6 (prefix-aware scheduling) and J7 (classes/priorities) to build on top of, not decided here.

Modeled on golang.org/x/sync/semaphore.Weighted's Acquire, specialized to weight-1 (one holder
at a time — ground rule: "one decode worker per model stays," task-work-queue-2026-09.md's own
"Ground rules" §1). The MECHANISM is reused, not the package: pulling in a dependency for one
specialized case would be the wrong trade for a three-method primitive this repo can own and
test directly (task-work-queue-2026-09.md's own ground rule 4: "no new root module dependency").

Zero value is ready to use — no constructor needed — and admits ONE holder. setCap widens it (MC3c,
docs/tasks/task-concurrency-2026-09.md: -max-concurrent on a CPU model): up to cap holders at once, still strict FIFO
for everyone waiting.
```

## admission.position

Moved from `internal/serveapp/admission.go` (the comment above `admission.position`) on 2026-10-09.

```text
position reports where the waiter with this id stands: its place among the waiters (1 = next to be
granted the turn), how many are waiting in all, and whether one is running now. ok is false when no
waiter has this id — it was never queued, has already been granted the turn, or dropped out. Order is
exactly arrival order: J1 is strict FIFO, and J6's reordering was measured and not shipped.
```

## streamMessages.heartbeat

Moved from `internal/serveapp/anthropic_stream.go` (the comment above `streamMessages.heartbeat`) on 2026-10-09.

```text
N-24 (docs/audit-2026-09-10.md): the ping above is a one-shot liveness check, not a
keep-alive — nothing else is sent until the first token, which on CPU is after the whole
prefill (minutes for an image, ~270s for an 8k agent prompt) against a 300s idle timeout.
```

## streamMessagesTools

Moved from `internal/serveapp/anthropic_stream.go` (the comment above `streamMessagesTools`) on 2026-10-09.

```text
streamMessagesTools runs a tool-bearing turn on /v1/messages: prose streams as a text content block
while the model writes it (G21, where the family can stream prose safely — the shared tool turn,
tool_turn.go), then one tool_use block per parsed call. With no call it is one text block.

Before this, the route Claude Code uses buffered the WHOLE generation and sent only heartbeats
(audit-2026-09-02 M-19 added those: the single `ping` at message_start was the only byte for the
entire generation, measured elsewhere at 1682.6s against a 300s idle timeout) — only the OpenAI
route streamed prose during a tool call.
```

## bannerFacts.fit

Moved from `internal/serveapp/banner.go` (the comment above `bannerFacts.fit`) on 2026-10-09.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the context cap, KV at that
cap, and what remains of the RAM budget — printed at every load, not only when something is
tight, because "79% of budget" at load time and "14 GB RSS, swapping" on the first real
request were the same load with no line connecting them.
```

## modelBannerFrom.context

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.context`) on 2026-10-09.

```text
How much context, and at what KV precision — the two numbers that decide whether a
harness's turn fits at all.

THE RESOLVED NUMBER, NOT THE REQUEST. This line used to print "backend default" when -ctx was
unset, and the -ctx value when it was set — neither of which is the limit. The limit is
min(model maximum, resident KV cap), and an invisible default found out by degradation is the
exact complaint users make about other local servers. It now prints ctxWindow (what prepare
enforces and /v1/models publishes), and says what set it. cfg.load.Ctx is the REQUESTED -ctx for
this model (the caller passes the per-model ctx= override when there is one).
```

## modelBannerFrom.cpuNote

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.cpuNote`) on 2026-10-09.

```text
R20 (docs/tasks/task-first-hour.md): on the CPU path the window is the model's whole maximum — -ctx is the GPU-resident KV capacity and caps nothing
here — and KV is allocated per request, so the figure (and the "fit:" KV cost below) is a ceiling a request reaches only by filling the window. A
cold-user run read "262144 tokens, 10.4 GB KV" after a CUDA decline beside a `fit` that had priced the GPU plan at 8192, and took them for the same
question. Said here, beside the figure, so it is not left to be worked out.
```

## modelBannerFrom.fit

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.fit`) on 2026-10-09.

```text
R13: the memory this context cap actually costs, and what is left — the line a harness
user needs to judge whether their own turn size fits, before finding out from swap.

R13-follow-on: fitBudgetBytes is now priced against CURRENTLY AVAILABLE memory
(decoder.FitBudgetSummary), read after the model is already resident — so weightBytes is
ALREADY excluded from it by the OS's own accounting. Subtracting it again here would
double-count a footprint that is not there to subtract (the same bug AdmitPrefillMemory
had at request time). weights are still shown, for the reader's own arithmetic, not this
function's.
```

## modelBannerFrom.reuse

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.reuse`) on 2026-10-09.

```text
Prefix reuse, and WHY when it is off or narrower than it sounds. This is the line that makes
an agent loop's per-turn re-prefill visible before it is paid for. A resident model does not
use the CPU-side session LRU (its KV lives on the device; see loadedModel.drive), but its own
cache reuses the prefix committed by the last generation (decoder/resident_reuse.go) — so a
continuing conversation prefills only its new suffix, and a DIFFERENT conversation re-prefills
in full. This line used to say every resident turn re-prefilled everything, which stopped being
true when resident reuse shipped (2026-09-03).
```

## modelBannerFrom.reuseNotMetal

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.reuseNotMetal`) on 2026-10-09.

```text
Not Metal: this used to print "Metal keeps N by default" for the count a CUDA card had granted.
```

## concurrencyLine

Moved from `internal/serveapp/banner.go` (the comment above `concurrencyLine`) on 2026-10-09.

```text
concurrencyLine is how many generations of a model run at once (MC3c / MC3, -max-concurrent) and why fewer than asked
when that happens; "" when nothing needs saying. It is NOT part of the load-time banner: concurrency is decided by
setConcurrency after every model, adapter and vision tower has loaded, so a line printed with the banner would
report the undecided value (it did, 2026-09-26: every model's banner said "one generation at a time").
```

## warnThinkingTemplate

Moved from `internal/serveapp/blockdrafter.go` (the comment above `warnThinkingTemplate`) on 2026-10-09.

```text
warnThinkingTemplate says so when the served template will put the target in THINKING mode.

This is the one deployment footgun that survives every other safeguard, because it is
INVISIBLE: block drafting is lossless, so a thinking-mode target returns correct responses at
reduced speed and nothing in any log says why. Measured on Qwen3-4B, same model, same
hardware, only the template differing:

	non-thinking   5.76 accepted/round   1.57x
	THINKING       3.00 accepted/round   0.82x   (0.96x once the acceptance guard trips)

The runtime guard stops the bleeding, so this is a warning and not a refusal — a thinking
deployment is merely not getting the win, rather than being harmed. But an operator who chose
--drafter deserves to be told that this configuration is why it is not paying off.
```

## qwen3EmbedEOD

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `qwen3EmbedEOD`) on 2026-10-09.

```text
Qwen3-Embedding conventions. Read from the model's own config files AND — the part config-reading
alone gets wrong — from what sentence-transformers actually feeds the model:
  - config_sentence_transformers.json prompts: query carries the Instruct preamble, document "".
  - 1_Pooling/config.json: pooling_mode_lasttoken=true, include_prompt=true — so the query prompt
    is part of the pooled input and is NOT stripped before pooling.
  - tokenizer_config.json: add_bos_token=false, add_eos_token ABSENT -> no BOS, and the plain
    tokenizer call appends nothing.
  - BUT sentence-transformers appends <|endoftext|> (151643) to EVERY input. Nothing in the
    configs or the model card says so — `model.tokenize()` had to be inspected to see it. This is
    not cosmetic: last-token pooling pools THAT token, so omitting it pools the final content
    token instead and yields a plausible, semantically-ordered, but WRONG vector (cosine ~0.4-0.8
    vs the reference, while retrieval still looks fine). Note it is <|endoftext|>, NOT the
    configured eos_token <|im_end|> (151645) — reading eos_token would also have been wrong.
```

## newDecoderEmbedder.maxTokens

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `newDecoderEmbedder.maxTokens`) on 2026-10-09.

```text
BOUNDED BY THE MODEL'S CONTEXT, always. The reference imposes nothing shorter than the
tokenizer's model_max_length (Qwen3-Embedding ships no sentence_bert_config.json), but
"nothing shorter" is not "unbounded": HiddenLast preallocates KV for len(ids) positions
and runs a sequential per-token forward with no context, so 0 here meant the ONLY bound
was C-21's 1 MiB byte cap — about 500k tokens of short words. For a 28-layer, kvDim-1024
embedder that is ~114 GB of KV and attention over up to 500k keys per token: hours of one
pinned core, every other /v1/embeddings request blocked behind the embed mutex, and then
an OOM kill. Uncancellable, and un-queued.

Even a legitimate 200 KB document (~45k tokens) silently exceeded Qwen3-Embedding's 40k
window and returned a vector pooled from out-of-range RoPE positions — plausible, and
wrong. MaxPositions is HF's truncation=True semantics, which is what the aikit encoder
path already does; only this one was unbounded (audit-2026-09-02 C-07, an incomplete
closure of the 2026-08-05 audit's C-21).
```

## server.loadDecoderEmbedder

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `server.loadDecoderEmbedder`) on 2026-10-09.

```text
M-17 (docs/audit-2026-09-10.md): Backend/Quant were left zero-valued here, so this embedder
always loaded CPU-only regardless of -backend, and -embed-quant was silently never read for
a .gguf embed model — the resident HiddenLast path (decoder/embed.go) this model may
implement was shipped but never reachable from serve. decoder.Load builds m.resident off
Options.Backend alone (decoder/model.go's .withResidency()), so this is sufficient for
reachability — no separate resident-build step, matching how chat models get residency
(modelSpec.options, main.go). A resident embedder gets its own resBusy CAS
(decoder/model.go), structurally identical to a second resident chat model — nothing shared
needs a new guard.

cfg.embedQuant is NOT passed through as decoder.Options.Quant directly: the two use
different vocabularies (-embed-quant's "f32"|"q8", the aikit-encoder path's own precision
names, vs decoder.Options.Quant's ""|"int8"|"int8int8"|"int4", decoder/model.go's
parseQuant) — parseQuant hard-errors on any string it doesn't recognize, so passing "q8"
straight through would make -embed-quant q8 FAIL TO LOAD instead of silently ignoring it, a
worse regression than the bug this fixes. "q8" maps to Quant's own "int8" (weight-only
per-row), the same precision class the aikit encoder's LoadQ8 path uses.
```

## decoderEmbedder.EncodeBatchCounted

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `decoderEmbedder.EncodeBatchCounted`) on 2026-10-09.

```text
EncodeBatchCounted is EncodeBatch plus each input's token count, read off the SAME tokenize
call encodeLocked already makes rather than a second pass over the text (audit R-07: this
embedder's EncodeBatch's ids were tokenized once and thrown away, then countEmbedTokens
(embeddings.go) tokenized every input again from scratch purely to report prompt_tokens).
embedBatchCounter in embeddings.go is the optional capability the handler prefers this
through; encoders that don't implement it (the aikit-embed.Tokenizer path) are unaffected.
```

## truncateForContext

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `truncateForContext`) on 2026-10-09.

```text
truncateForContext is tokenize's C-07 truncation arithmetic, pulled out so a test can call the
SAME code the request path runs instead of re-deriving it beside it (V-21,
docs/review-2026-09-04.md: the old test re-implemented room--/ids[:room], so a bug in tokenize's
own arithmetic — not this function's, since it didn't exist yet — would have passed unnoticed).
Truncates BEFORE the caller appends appendID, reserving its slot, so the appended token is
never the thing truncation drops — it must stay last, because it is the pooled position.
```

## embedTask

Moved from `internal/serveapp/embeddings.go` (the comment above `embedTask`) on 2026-10-09.

```text
embedTask picks the prompt a taskEmbedder applies (docs/tasks/task-embeddinggemma2.md, Gate 3, owner decision
2026-10-06): an explicit task names one of the model's prompts ("none" for none); else input_type maps onto the
model's own "query" and "document" prompts; else no prompt, which is what sentence-transformers applies by default,
so a client that knows nothing about prompts gets the reference's own output.
```

## maxEmbedInputs

Moved from `internal/serveapp/embeddings.go` (the comment above `maxEmbedInputs`) on 2026-10-09.

```text
Embedding request bounds (audit C-21). /v1/embeddings is deliberately un-queued (the encoder is
goroutine-safe and parallelizes internally), so without a per-request cap a single body drives an
unbounded allocation and N concurrent requests multiply it: a 4 MiB body of empty strings is ~2M
inputs, and the response builder materializes 2M maps + 2M []float32 of HiddenDim (~6 GB at dim 768)
before writing a byte; with a decoder-as-embedder (maxTokens 0) one multi-MiB string prefills ~1M
positions. maxEmbedInputs matches OpenAI's per-request batch cap; maxEmbedInputBytes bounds a single
input to text sizes (an embedding input is a query/passage, not a document dump).
```

## server.resolveDimensions

Moved from `internal/serveapp/embeddings.go` (the comment above `server.resolveDimensions`) on 2026-10-09.

```text
resolveDimensions validates the optional output-dimension override against the
model's native width. nil/0 means full width.

Truncation is only legitimate for models trained with Matryoshka Representation Learning.
Slicing any other embedder returns a unit-length, entirely plausible vector that simply
RETRIEVES WORSE — a silent-wrong, and measured rather than theoretical: aikit's
TestEmbedderCoverage_matryoshka shows multilingual-e5-base sliced to a quarter width dropping
paraphrase-pair recall 1.00 → 0.80, while genuine MRL models hold their documented floor. So a
dimensions request below the model's floor, or ANY dimensions for a non-MRL model, is a 400
rather than a quietly degraded vector. s.embedMRLMin carries that floor (0 = not truncatable),
resolved at load from aikit's exported registry — the same source of truth that generates its
published Truncatable column.
```

## generation

Moved from `internal/serveapp/generations.go` (the comment above `generation`) on 2026-10-09.

```text
K1 (docs/tasks/task-halt-2026-09.md): a process-wide registry of in-flight generations, keyed by the
id each handler already mints (reqID(), via chatcmpl-/msg_/resp_ prefixes). It exists so an
operator can cancel one generation, or every generation of one model, from outside the
process that started it — the handle a cancel-by-id needs already existed (helpers.go's
reqID); it was just never registered anywhere.

Registration happens inside drive/driveVL (the two places EVERY generation path funnels
through — openai.go's own doc comment on drive), not in each of the ~15 call sites, so the
bookkeeping lives in one place. A handler that wants cancellation opts in by setting
genRequest.id before calling drive/driveVL; gr.id == "" (a caller that hasn't been wired, or a
test) skips registration entirely — nil-safe, not an error.

NOTE ON "SESSION" (found while implementing, not assumed): docs/tasks/task-halt-2026-09.md's K1 also
asks for `POST /admin/sessions/{id}/cancel` ("every generation of that session, since an agent
loop is a session"). That endpoint is NOT implemented here. goinfer's own "session"
(sessionLRU/decoder.Session, sessions.go) is a content-addressed KV-reuse cache selected by
longest-common-prefix match (bestExtend) — it has no client-visible, stable identifier an
operator could type into a cancel request. None of the four request surfaces (chat
completions, completions, responses, messages) carry a session/user/thread id either
(embeddings.go's `User` field is the only "user" field in the tree, and it is explicitly
"accepted, ignored"). Inventing one here would be exactly the kind of undocumented design
decision the task asked to be flagged instead of worked around — see the report accompanying
this commit.
```

## jobEventLog.markDone

Moved from `internal/serveapp/jobeventlog.go` (the comment above `jobEventLog.markDone`) on 2026-10-09.

```text
markDone marks the log closed — no further events will ever be appended — and wakes every
waiter so a blocked reader notices the job ended even if it produced no final event of its own.
Idempotent-safe to call once, by runJob's own defer chain; calling it twice would double-close
an already-replaced channel and panic, so it must not be called more than once per job.
```

## maxEmbedBodyBytes

Moved from `internal/serveapp/helpers.go` (the comment above `maxEmbedBodyBytes`) on 2026-10-09.

```text
maxEmbedBodyBytes is /v1/embeddings' own floor, deliberately NOT derived from any decoder's
context window. A batch embeddings body scales with (batch count × input length) — quantities
the route already bounds itself (maxEmbedInputs=2048, maxEmbedInputBytes=1 MiB) — and has
nothing to do with a chat model's MaxPositions. Deriving it from the text cap made the limit
arbitrary in both directions: on an embed-only server (no decoder loaded) it collapsed to the
4 MiB text floor, rejecting a perfectly legal 2048×4 KiB batch at 8 MiB before
checkEmbedInputBounds could accept it; alongside a 128k-context chat model it ballooned for no
reason. 64 MiB covers a realistic maximal RAG batch (2048 inputs × ~32 KiB) while still
bounding the read. -max-body-bytes overrides it like the others.
```

## maxBytes

Moved from `internal/serveapp/helpers.go` (the comment above `maxBytes`) on 2026-10-09.

```text
maxBytes wraps a handler so its request body is bounded to n bytes (n <= 0 disables).
Two layers (G1/G2/G3):
  - Content-Length pre-check: a declared body over the cap is rejected 413 BEFORE a byte
    is read — the case that matters, and it costs nothing (no allocation, no upload wait).
  - http.MaxBytesReader backstop for chunked encoding or a lying Content-Length; it fails
    the read with *http.MaxBytesError, which the decode helpers render as 413.

The 413 names the limit (and the received size when the client declared one) so a client
sees why it was rejected rather than a bare close. A client still uploading when the
pre-check fires may see EPIPE regardless — but today it gets no HTTP response at all (G3).
note, when non-empty, is appended to the 413. A route whose own validator declares limits in
DIFFERENT units than the body cap needs it: /v1/embeddings advertises "up to 2048 inputs of up to
1 MiB each", which multiplies out to 2 GiB and can therefore never all be satisfied at once. Those
are per-DIMENSION bounds; the body cap bounds the TOTAL. A request can respect both per-dimension
limits and still exceed the total, and a 413 naming only the total leaves the client unable to
tell which of the three numbers it actually violated.
```

## jsonDecodeMessage

Moved from `internal/serveapp/helpers.go` (the comment above `jsonDecodeMessage`) on 2026-10-09.

```text
decodeJSON reads the (size-bounded, see maxBytes) request body into v, writing
an OpenAI-shaped error on failure: 413 when the body exceeded the limit, else
400. Returns false iff it wrote an error. M3.
jsonDecodeMessage shapes a json decode error into a message safe to return to a client: it never
echoes the raw error, whose UnmarshalTypeError rendering leaks the Go struct name (e.g. "…Go
struct field completionReq.logprobs of type bool"). tooLarge reports the body-cap case, which the
callers answer with a different status.

Extracted so BOTH surfaces share it. The OpenAI decoder was hardened by M-06 and R-11; the
Anthropic one still appended err.Error() verbatim and leaked exactly what those removed
(audit-2026-09-02 N-16).
```

## sseWriter

Moved from `internal/serveapp/helpers.go` (the comment above `sseWriter`) on 2026-10-09.

```text
sseWriter owns one SSE response: every frame goes through it, one at a time, under a bounded
write deadline. Both the OpenAI and the Anthropic stream paths write through it.

IT EXISTS BECAUSE TWO GOROUTINES WERE WRITING THE SAME ResponseWriter. G19 added a heartbeat
goroutine that owns w while the handler is silent; G21 then made the incremental tool paths write
prose deltas to that same w during the same window, deliberately. net/http's response/bufio.Writer
is not safe for concurrent Write/Flush, so outcomes ran from ": ping" spliced into a data: line
(the client drops or mis-parses the chunk) to a bufio slice-bounds panic — and a panic in the
HEARTBEAT goroutine is outside net/http's per-request recover, so it takes the PROCESS
(audit-2026-09-02 C-06). sseHeartbeat's own doc promised "no risk of interleaving two writers";
that was true only on the paths whose callback appends to a builder.

The deadline is a second defect on the same write path. Flush blocks in net.Conn.Write with no
deadline, so a client that stops READING without closing pins the handler inside onText holding
the model's queue slot — r.Context() cancels on close, not on a stalled read — and every other
request queues then 429s for as long as that socket stays open (M-17). This is NOT the
server-wide WriteTimeout the M3 comment conflated it with: that would truncate a legitimately
long stream; this bounds one frame.

The first write error is sticky and readable: a stream whose client vanished should stop
generating rather than keep pushing frames into a dead socket.
```

## sseHeartbeat

Moved from `internal/serveapp/helpers.go` (the comment above `sseHeartbeat`) on 2026-10-09.

```text
sseHeartbeat keeps a stream alive while the handler is producing nothing to
send (G19). The tool paths must buffer the whole generation before they can
parse a tool call, so they would otherwise send zero bytes for the entire
generation — measured at 1682.6s against a client whose idle timeout was 300s.

A COMMENT frame (":" + text) is the right instrument: it is protocol-legal,
carries no data, and every SSE parser drops it, so nothing downstream can
mistake a keep-alive for content. The buffering guarantee is untouched.

The returned stop JOINS the goroutine before returning. That join was once the ONLY thing keeping
two writers apart, and it was not enough: it orders the heartbeat against the handler's writes
AFTER drive returns, not against the prose deltas the incremental paths emit WHILE it runs. The
serialization now comes from sseWriter's lock; the join remains so nothing ticks after the
caller's final frame (audit-2026-09-02 C-06).
```

## sendUsage

Moved from `internal/serveapp/helpers.go` (the comment above `sendUsage`) on 2026-10-09.

```text
sendUsage emits the include_usage chunk, if the client asked for one, immediately before
sseDone. THE ONE PLACE that decision is made.

M-26: include_usage was honoured on the plain chat stream only. The tool and vision streams
silently omitted it and /v1/completions did not even parse the field — and agent harnesses,
which declare tools on every turn, are exactly the clients that depend on it for context
accounting. Routing all four through one helper is what lets a test COUNT the sites, so a
fifth streaming surface cannot be added quietly without one.
```

## writeJSON

Moved from `internal/serveapp/helpers.go` (the comment above `writeJSON`) on 2026-10-09.

```text
Encode BEFORE the status line goes out. The body used to be streamed straight from the encoder after WriteHeader(code) with its error dropped, so a value JSON cannot
carry (a NaN or an infinity in a float) reached the client as HTTP 200 with Content-Length 0: S6's 35B image check read "empty answer" for a whole night and
the server said nothing (2026-10-09). A body that cannot be encoded is a server error and says so.
```

## reqID

Moved from `internal/serveapp/helpers.go` (the comment above `reqID`) on 2026-10-09.

```text
reqID is a short, monotonically increasing id suffix (unique per process run).
reqID returns an opaque, UNGUESSABLE id for a response/message/tool-call.

N-17: this was a sequential counter (seeded from UnixNano, then +1 per id). Seeding hid the
problem without fixing it — an id is still exactly one more than the id before it, so a
client holding its own `resp_<hex>` can walk ±1 and land on other clients' ids. That matters
because `previous_response_id` continues a stored conversation: with `-addr 0.0.0.0` and one
shared API key, guessing an id reads back someone else's turns.

crypto/rand, not math/rand: guessability is the whole property. 16 bytes because these ids go
in URLs and logs, and 128 bits is beyond enumeration.
```

## serveResponsesWith.guard

Moved from `internal/serveapp/responses.go` (the comment above `serveResponsesWith.guard`) on 2026-10-09.

```text
G1c, extended (audit-2026-09-02 M-21). Placed BEFORE the tools/plain branch so one guard
covers both — /v1/responses was two of the five routes that tokenized an arbitrary body
before rejecting it, and the assembled `messages` here include anything a stored
previous_response_id dragged in, which is the input the BPE would actually run over.
M-15 (audit-2026-09-10): req.Tools' schema bytes are added unconditionally too, matching
this guard's own "covers both branches" design — if tools end up active below,
RenderToolsSegments renders every one of them into the prompt.
```

## respondTools.stream

Moved from `internal/serveapp/responses.go` (the comment above `respondTools.stream`) on 2026-10-09.

```text
The shared tool turn (tool_turn.go). Streaming: response.created goes out first, as on the
plain-text path, and prose streams as output_text.delta while the model writes it where the
family allows (G21) — before, this path sent nothing but heartbeats until the generation ended.
```

## server.maybeStore

Moved from `internal/serveapp/responses.go` (the comment above `server.maybeStore`) on 2026-10-09.

```text
maybeStore records this turn's assistant output for a later previous_response_id
continuation — serveResponsesWith appends prior.messages VERBATIM onto the next request's
input, so whatever is missing here is missing from every stateful continuation.

V-18 (docs/review-2026-09-04.md): toolCalls used to be dropped — only the lead text (often
empty, when the model went straight into a tool call) was stored. M-18's fix taught the DECODE
side to turn a resent function_call/function_call_output pair back into ToolCalls/a tool turn,
but that only helps a STATELESS caller that resends the whole conversation itself. The SDK
DEFAULT is previous_response_id (stateful): the client sends only the new
function_call_output, and the server is expected to have kept the matching function_call from
its own prior turn. Without ToolCalls here, that turn reconstructs as user → assistant("") →
tool(result) — a tool result answering a call that, as far as the stored conversation shows,
was never made.
```

## responseInputToMessages

Moved from `internal/serveapp/responses.go` (the comment above `responseInputToMessages`) on 2026-10-09.

```text
`type` and the function-call fields are decoded, not just {role, content}. A Responses tool
loop feeds the model's own `function_call` back with a `function_call_output`, and NEITHER
carries a role or a content field — so both used to fall through to the default below and
become `{Role:"user", Content:""}`: two empty user turns. The model never saw the tool
result, so it either answered without it or called the same tool again, forever, under HTTP
200. docs/server.md and this file's own comment both claim the round-trip works;
TestServe_responses step 4 never feeds a result back, which is why nothing caught it
(audit-2026-09-02 M-18).
```

## handleCreateJob.scope

Moved from `internal/serveapp/jobs_http.go` (the comment above `handleCreateJob.scope`) on 2026-10-09.

```text
J3 (task-work-queue-2026-09.md): POST /v1/jobs submits a generation and returns its id
immediately; GET /v1/jobs/{id} polls state+result; GET /v1/jobs/{id}/events re-attaches to its
output (replay-then-live); DELETE /v1/jobs/{id} cancels it. Text-only chat body for this pass —
see jobs_run.go's own doc comment and the task doc's closure note for what's scoped out
(vision, tools, the K4 user/user_id key that doesn't exist yet).
```

## qwen3MaxImageTokens

Moved from `internal/serveapp/qwen35_vision.go` (the comment above `qwen3MaxImageTokens`) on 2026-10-09.

```text
qwen3MaxImageTokens is the serve-side ceiling on merged image tokens per image. The checkpoint's own
preprocessor_config allows up to 16384 (longest_edge 16777216 px), which is not a usable request on
the CPU decode path an image turn takes today: a Gated-DeltaNet hybrid prefills one token at a time,
and the resident GPU paths cannot yet carry an image turn (docs/multimodal.md P8 record, item 5).
1024 tokens is a 1024x1024 image. Larger images are smart-resized down to fit, exactly as the
processor would for its own max_pixels; a build-time constant, not an environment read.
```

## isQwen35VisionDir

Moved from `internal/serveapp/qwen35_vision.go` (the comment above `isQwen35VisionDir`) on 2026-10-09.

```text
isQwen35VisionDir reports whether dir is a Qwen3.5+ checkpoint that carries a usable vision tower:
model_type qwen3_5/qwen3_5_moe, a non-empty vision_config with no DeepStack, and a preprocessor
config LoadQwen3PreprocessConfig accepts. Used for AUTO-discovery only: a stripped text-only copy
(no vision_config or no preprocessor_config.json) is simply not a vision model, not an error.
```

## sessionLRU

Moved from `internal/serveapp/sessions.go` (the comment above `sessionLRU`) on 2026-10-09.

```text
sessionLRU keeps up to size prefilled KV sessions and hands each request the
one that already holds its prompt as a prefix — so a continuing chat (or an
agent loop with a fixed system prompt + tool specs) prefills only the new
suffix instead of the whole history. It is the cmd/serve layer over
decoder.Session; the decoder does the exact prefix reuse, the LRU just decides
which session to reuse and which to evict.

Not goroutine-safe. The server holds s.mu across every generation, which
serializes all access — the same lock that already serializes the shared model.
The optional tiered-KV demotion (see enableTiering) runs under that same lock,
so the background demoter and the request path never race.
```

## sessionDirPerm

Moved from `internal/serveapp/sessions.go` (the comment above `sessionDirPerm`) on 2026-10-09.

```text
KV snapshots and cold blobs are the CONVERSATION, not a cache of public data: a .giw-kv blob
replays what the user said and what the model answered. They were written 0o644 inside a 0o755
directory, so every local account could read them (audit-2026-09-02 N-21). Owner-only, both
levels — the directory matters as much as the files, since a readable directory lists the
session ids.
```

## sessionLRU.acquire

Moved from `internal/serveapp/sessions.go` (the comment above `sessionLRU.acquire`) on 2026-10-09.

```text
acquire returns the session to generate prompt against and marks it most-
recently-used. It reuses whichever session's stored tokens share the longest
prefix with prompt (L-15), not only an exact continuation — the decoder's own
rewindForReuse truncates that session's cache to the shared length and
prefills just the divergent suffix. This also covers a stop-string hit, a
max_tokens cut mid-word, or an edited last message (P-18): each commits a few
tokens the client's next prompt can't reproduce, which the old whole-
containment rule rejected outright, cold-prefilling and evicting a session
that was almost entirely reusable. A prompt that merely shares a system-prompt
preamble with some other conversation's session still does NOT hijack it —
bestExtend requires beating what that session shares with every OTHER known
session, not just any nonzero match — so distinct conversations keep their own
slots. Anything else gets a fresh slot, evicting the coldest session when full.

With size 0 reuse is disabled: every call gets a throwaway session (the old
re-prefill-everything behavior).
```

## bestExtend

Moved from `internal/serveapp/sessions.go` (the comment above `bestExtend`) on 2026-10-09.

```text
bestExtend returns the index of the session whose common prefix with prompt is
longest (L-15), preferring reuse whenever a candidate's match beats what that
same candidate merely shares with every OTHER known session — a shared
system-prompt preamble alone never qualifies, however long, or any two
conversations that open the same way would evict each other's own turns.
This subsumes the old whole-containment rule (an exact continuation's match is
its own full length, which always clears the floor set by other sessions) and
additionally catches P-18's cases: a stop-string hit, a max_tokens cut
mid-word, or an edited last message each commit a few tokens the client's next
prompt can't reproduce, so the session's own tokens are no longer FULLY
contained in prompt even though almost all of them still are — the old rule
rejected these outright and cold-prefilled instead. The decoder's own
rewindForReuse (decoder/session.go) truncates the chosen session's cache to
the matched length; bestExtend only picks which session gets that treatment.
-1 if nothing qualifies.
```

## sessionLRU.save

Moved from `internal/serveapp/sessions.go` (the comment above `sessionLRU.save`) on 2026-10-09.

```text
Snapshot refuses RECURRENT state (Mamba-2 / DeltaNet / LFM2 conv / MLA latent),
which cannot be restored from a KV blob. It used to refuse sliding-window rings
too, and this line still said so — rings have been persistable since
kvSnapVersion 2 (N-34).
```

## pickSession

Moved from `internal/serveapp/sessions.go` (the comment above `pickSession`) on 2026-10-09.

```text
pickSession is acquire's choice: bestExtend's candidate, unless the LRU has room for a fresh session (spare) and
reusing the candidate would throw away more of it than it keeps. bestExtend's floor learns the shared preamble only
from OTHER resident sessions, so with a single one it cannot tell a chat-template preamble from a continuation: two
interleaved conversations then took each other's session on every turn, each truncating the other to the preamble,
and the LRU never grew past one session (MC0, docs/tasks/task-concurrency-2026-09.md, 2026-09-26: 7 tokens reused per
turn at 2 clients, 0.69x the 1-client aggregate on CPU). A fresh session costs only re-prefilling the shared lead;
the truncation costs the other conversation its history. P-18's partial matches (a stop-string tail, an edited last
message) keep most of their session and still reuse. With no room, reusing the candidate IS the eviction, so it
stands. -1 means take a fresh session.
```

## handleSystemOne

Moved from `internal/serveapp/systemone.go` (the comment above `handleSystemOne`) on 2026-10-09.

```text
POST /v1/systemone — TypeSafe's decisions wire shape (D5 of docs/tasks/task-constrained-confidence.md; the request
and response JSON are recorded verbatim in docs/measurements/decisions-d0-prior-art-2026-09-27.md §3), answered by
label scoring on the served model (Route A, internal/decide). It exists so jevx and TypeSafe's SDKs (JS, Python,
Vercel, LangChain), which all take a base-URL override, work against goinfer unchanged.

What it is not: TypeSafe's hosted model or a trained decision head. Its distributions are the served model's own
probabilities over the options, calibrated only for a kind whose temperature -decisions-calibration supplies (the
response's goinfer.calibrated says which).
```

## toolTurn

Moved from `internal/serveapp/tool_turn.go` (the comment above `toolTurn`) on 2026-10-09.

```text
toolTurn is the outcome of one tool-bearing generation. /v1/chat/completions, /v1/responses and
/v1/messages each used to run this turn their own way — drive, buffer, parse, reconcile — and only
the OpenAI route streamed prose while the model wrote it (G21); /v1/messages, the route Claude
Code uses, sent nothing but heartbeats until the whole generation was done. runToolTurn is the one
implementation; each front end keeps only its wire format.
```

## reconcileProse

Moved from `internal/serveapp/tool_turn.go` (the comment above `reconcileProse`) on 2026-10-09.

```text
reconcileProse settles a finished turn's prose against what already streamed. lead is the turn's
prose for the final message: the parser's lead when there are calls, the raw output when there are
none (as every route has always returned it). rest is what the front end still has to send so the
streamed prose adds up to the turn's prose.

What streamed is held against the PARSER's lead, never the raw output: the prose streamer trims the
leading whitespace and withholds trailing whitespace exactly as every family's parser trims its
lead (chat.ProseStreamer), so a no-call answer that opens with a newline streams "Hello" where the
raw text is "\nHello". Checking that against the raw text — what the OpenAI route did before this
was shared — reported a false divergence and ended the stream in an error. When nothing streamed
(a family that cannot stream prose, or a non-streaming request) rest is lead itself, unchanged.
```

## versionReport.history

Moved from `internal/serveapp/version.go` (the comment above `versionReport.history`) on 2026-10-09.

```text
versionReport is what `serve --version` prints. Its load-bearing line is `backends:` — the
list of backends COMPILED INTO this binary, which is not the same thing as the list
--backend accepts.

R2 (docs/measurements/cold-user-2026-09-06.md, finding #3): the v0.16.0 darwin release asset
was built from the root cmd/serve, which links no backend at all. `--backend metal` on it
therefore ran on CPU — 37.9 tok/s against the 82.3 the same box does with Metal — and the
only signal was one warning line that scrolled past before the banner. Nothing on the binary
could be asked. Now it can be, without loading a model, which is also what the release
workflow greps to prove each asset carries the backend for its platform.
injectedVersion is set via `-ldflags -X` on a release-built binary. R6 (docs/measurements/
cold-user-2026-09-06-nobara-pc.md): the v0.17.0 linux-amd64 release asset's `--version`
printed "v0.0.0-20260907045005-f36b095ac9a1+dirty", not "v0.17.0" — R2-follow-on's
`go mod edit -replace` on the ephemeral submodule checkout is an uncommitted go.mod edit, and
that alone is enough for the VCS stamp to read "modified". A binary built the ordinary way
(`go install .../cmd/serve@v0.17.0`, no replace, no ephemeral checkout) is unaffected and
keeps reporting its real tag from cliutil.BuildIdent with no injection at all — this only overrides
the release workflow's own path.
```

## countFlags

Moved from `internal/serveapp/version.go` (the comment above `countFlags`) on 2026-10-09.

```text
countFlags reports how many flags are registered, so the help header's "all N flags" line cannot
drift from reality the way a hand-typed count would. That drift is the same defect class as the
parity manifest's hand-typed aikit_version, which sat seventeen versions stale.
```

## prepImage.qwenBlock

Moved from `internal/serveapp/vision_multi.go` (the comment above `prepImage.qwenBlock`) on 2026-10-09.

```text
M-38 (audit-2026-09-10): Qwen2.5-VL's real chat_template.json (verified live against
Qwen/Qwen2.5-VL-7B-Instruct) splices <|vision_start|><|image_pad|><|vision_end|> inline
with NO adjacent newline on either side — the trailing "\n" this used to append doesn't
exist in the real template.
```

## prepImage.gemma4Block

Moved from `internal/serveapp/vision_multi.go` (the comment above `prepImage.gemma4Block`) on 2026-10-09.

```text
M-38 (audit-2026-09-10): Gemma 4's own processor (processing_gemma4.py, verified against the
real transformers source) does f"{boi_token}{image_tokens}{eoi_token}" — no adjacent
newline at all, unlike the trailing "\n" this used to append.
```

## Main.usage

Moved from `internal/serveapp/main.go` (the comment above `flag.Usage` in `Main`) on 2026-10-09.

```text
A SKIMMABLE HELP HEADER, printed before the 39-flag dump.

Cold-user run 2026-09-06, scenario B: "--help is 13,583 bytes / 39 flags / 100 lines, with
paragraph-length prose per flag containing commit SHAs and self-critique. Unusable as a quick
reference; I could not skim it for the flag I needed." That is not a style complaint — the
SAME tester then drove a 16 GB machine +7.8 GB into swap because they did not find
-stream-weights, whose help text names their exact model and their exact RAM. The flag was
there and the document was too long to find it in.

The long text stays: every paragraph in it is a disclosure some measurement earned, and
deleting disclosures to shorten a page is how a trade-off stops being disclosed. This adds a
map ABOVE it rather than trimming it, so skimming and reading are both possible.
```

## Main.routes

Moved from `internal/serveapp/main.go` (the comment above `reqLogOut` in `Main`) on 2026-10-09.

```text
K2 (docs/tasks/task-halt-2026-09.md): halt is checked AFTER auth (a bad key is still rejected
during a halt) and BEFORE inf (a halt must not wait for an inflight slot — "a halt that
has to wait for a slot is not a halt", the doc's own words). /admin/* and /health are
deliberately NOT wrapped in this — an operator must always be able to resume/check status.
Registered whether or not a model is loaded at startup. A server started with only --web,
--allow-admin or --admin-socket loads its model later (the web UI's Models tab, /admin/models/load);
the mux is built once, so these routes registered only when a model existed at startup left that
server with no /v1/chat/completions and no /v1/jobs for its whole life, and the web UI's own chat
got a 404. Every handler resolves its model through resolveAndLock, which answers an unknown or
absent model with the OpenAI-shaped 404 "model not found (served: …)".
-log-requests wraps the four generation routes OUTERMOST, so a request the auth, halt or queue gates turned away is logged with its status too.
```

## Main.webroutes

Moved from `internal/serveapp/main.go` (the comment above the `GET /{$}` route in `Main`) on 2026-10-09.

```text
"GET /{$}" matches the root path EXACTLY. A bare "GET /" would be a catch-all and
would turn every unknown GET into the UI page instead of a 404, which is worse than
unhelpful for an API server — a typo'd route would render HTML to an SDK.

UNAUTHENTICATED on purpose (V-02, docs/review-2026-09-04.md): the page embeds no
secrets (handleWebUI's own comment), but a browser's plain navigation sends no
Authorization header, and the page is the ONLY place a user could type the key in —
its own JS holds it for the fetch() calls to /web/models/*. Wrapping this route in
auth() made that impossible whenever -api-key was set (required off loopback): the
page needed the key to load, and there was nowhere to enter the key without the page.
auth stays on the two routes below, which actually act (list a repo, pull a model).
```

## Main.shutdown

Moved from `internal/serveapp/main.go` (the comment above `srvCtx` in `Main`) on 2026-10-09.

```text
ReadHeaderTimeout + ReadTimeout + IdleTimeout bound slow-header (slowloris), slow-body
dribble, and idle keep-alive connections. ReadTimeout is the whole-request read deadline
(60s: generous for a 32 MiB vision body on a slow link) — before it, ReadHeaderTimeout
bounded only the headers, so a client sending the body one byte per minute pinned a
goroutine indefinitely (audit M-01). It only bounds the request READ; the SSE response is a
write, so a long stream is unaffected. WriteTimeout stays 0: SSE responses are long-lived
and a write deadline would truncate a legitimate stream (M3).
srvCtx is the server-lifetime context. BaseContext makes every request's r.Context() a child of
it, so cancelling srvCtx at shutdown cancels every in-flight generation (drive derives its
context from r.Context()). Without this, httpSrv.Shutdown waits for a long streaming generation
but never cancels it, so it runs past the 30s timeout — the checkpoint loop below no longer
deadlocks on that specific generation's lm.sessMu (J1, task-work-queue-2026-09.md, split
sessMu out from the admission turn it used to share: sessMu is now held only briefly, around
sessions.acquire, not for the whole generation), but Shutdown itself still waits for the
handler to return, so cancelling the generation is still what bounds the overall shutdown
(audit C-22); tryLockUntil's own deadline is the remaining belt-and-braces bound on sessMu
specifically, for whatever brief window a generation is actually inside sessions.acquire.
```

## loadVisionTower.gemma3

Moved from `internal/serveapp/main.go` (the comment above `attachGemma3Tower` in `server.loadVisionTower`) on 2026-10-09.

```text
On CUDA the unset default is the float32 tower (towerInt8); if it does not attach, attachGemma3Tower releases it and attaches the int8 device
tower instead of leaving the CPU one (an explicit -vision-quant never falls back). The unset default was already settled before the plan was made:
resolveGemma3VisionQuant makes it int8 on a card too tight for the decoder, one KV slot and the float32 tower. Both are in gemma3_tower.go.
The default flipped on the S13-lite float32 arm (2.40 s against 4.59 s a new image) and on G-S3b/d (the int8 tower changes the reply); see
docs/tasks/task-multimodal-support-2026-10.md, "Gemma 3's CUDA default is now float32". The float32 tower holds about 1.7 GiB against 0.56, which on
the 8 GB card is 2 resident KV slots at the 4096 floor instead of 3.
M-18 (docs/audit-2026-09-10.md): cuda joins webgpu here now that the resident CUDA vision tower's own leak/threading bugs are fixed
(cuda/vision_encoder.go); cuda/vision_register.go registered its factory with vision.RegisterResident via cuda/cmd/serve's blank import.
```

## towerInt8

Moved from `internal/serveapp/main.go` (the comment above `towerInt8`) on 2026-10-09.

```text
towerInt8 says whether a vision tower loads with int8 matmul weights. Gemma 3's SigLIP: WebGPU's device tower needs int8 (W8A8), so --backend webgpu
implies it without --vision-quant; CUDA's default is float32 since 2026-10-08 (owner; gemma3_tower.go) and Metal's (S3) is float32. The int8 tower
is lossy at real size (relative L2 0.16-0.52 against float32: docs/measurements/siglip-int8-fidelity-2026-10-07.md) and slower on CUDA (4.6 s against
2.4 s a new image). Every other tower (Qwen2.5-VL, Qwen3.5+, Qwen3-VL, Gemma 4, GLM-OCR) gets int8 only when asked for, whatever
the backend: its device towers are float32, and its gates ran float32. (Qwen3-VL was missing from this list until 2026-10-07, so under
cuda/webgpu it got an int8 CPU tower nobody had validated, under a banner naming a -vision-quant the user never passed.) The old rule forced int8
on three of them under cuda/webgpu, which bought no speed (the CPU int8 tower is not faster) and cost fidelity: measured 2026-10-02
against each tower's own f32 on the same image, relative L2 0.21 (Qwen2.5-VL), 0.14 (Qwen3.5-0.8B), 0.31 (Gemma 4), per-token
cosine mean 0.975 / 0.992 / 0.950 (docs/measurements/vision-tower-int8-fidelity-2026-10-02.md). The gates for all of them ran f32.
```

## towerInt8.f32

Moved from `internal/serveapp/main.go` (the comment inside `towerInt8` (case "f32")) on 2026-10-09.

```text
Explicit, and the only way to ask for Gemma 3's float32 device tower on CUDA (S4 addendum); unset keeps the old rule below. On webgpu there is no float32
device tower, so the attach declines by name and the CPU tower runs.
```

## enableResidentTower

Moved from `internal/serveapp/main.go` (the comment above `enableResidentTower`) on 2026-10-09.

```text
enableResidentTower attaches the device-resident vision tower when the backend is webgpu, cuda or metal and reports whether it
is attached. A failed attach (no VRAM left for the tower, a build without the backend) is a warning, not an error: it used to
abort serve startup and throw away the model already loaded on the GPU, but EnableResident leaves the CPU path intact, so the
tower runs there (slower) and the banner does not claim "-resident". Metal's towers (SigLIP and Qwen2.5-VL, S3) are float32
and decline an int8 tower, which lands here as that warning.
```

## loadQwenVisionTower

Moved from `internal/serveapp/main.go` (the comment inside `server.loadQwenVisionTower`) on 2026-10-09.

```text
Metal's tower (S3) attaches through attachResidentTower; CUDA's (aikit's gpu/qwencuda, S4: G-S4q correct at real size and 1.6-2.7x the CPU tower)
through qwenTowerPlacement, which names every CPU fallback; every other backend runs the CPU tower.
```

## server.loadGemma4VisionTower

Moved from `internal/serveapp/main.go` (the comment above `server.loadGemma4VisionTower`) on 2026-10-09.

```text
loadGemma4VisionTower attaches the Gemma 4 vision tower (aikit) to the single
loaded model. No separate projector — Gemma4Encoder.Forward bakes the
embed_vision projection in. decoder.GenerateGemma4VL dispatches between two
forwards depending on the checkpoint: the E2B/E4B-class sequential/causal
path (use_bidirectional_attention unset) and the 26B-A4B/31B-class batched
path (use_bidirectional_attention: "vision", decoder/forward_gemma4_batched.go).
Any OTHER value is refused at load time — rather than silently serving it
with the wrong mask — since only those two are implemented. No GPU-resident
vision path either way: aikit's Gemma4Encoder has no EnableResident method
(unlike vision.Encoder), so --backend webgpu has no effect on this tower
beyond the optional int8 CPU weight format.
```

## tryLockUntil

Moved from `internal/serveapp/main.go` (the comment above `tryLockUntil` (with the `demoteLoop` and `modelList` comments, which were glued above it)) on 2026-10-09.

```text
demoteLoop periodically demotes idle KV sessions across all models to disk
(tiered KV). It polls at a fraction of the idle threshold (clamped to [5s, 1m])
and takes each model's lock per sweep, so it stalls no in-flight generation and
skips a busy model until its lock is free. Returns when stop is closed.
modelList snapshots the registry under regMu. Background sweeps (demote, shutdown
checkpoint) must iterate this, not range s.models directly: admin load/unload mutate the
map under regMu (admin.go), and a concurrent map iteration+write is a runtime-fatal panic,
not just a race (M4). The returned slice is a copy of the pointers; each loadedModel is
still locked via its own lm.mu by the caller.
tryLockUntil acquires mu, giving up at deadline instead of blocking forever, so the shutdown
checkpoint can never deadlock on a generation that outlived the drain (audit C-22). Returns false
if the lock was not taken by the deadline.
```

## visionPathError

Moved from `internal/serveapp/main.go` (the comment above `visionPathError`) on 2026-10-09.

```text
visionPathError is -vision's refusal for a path that is a file, not a vision-tower directory (R22, docs/tasks/task-first-hour.md).
A GGUF mmproj handed to it used to fail inside the encoder loader as ".../mmproj-....gguf/config.json: not a directory". A path that
does not exist, or a directory, is the loaders' to judge, with their own messages.
```

## qwenTowerPlacement

Moved from `internal/serveapp/main.go` (the comment above `qwenTowerPlacement`) on 2026-10-09.

```text
qwenTowerPlacement decides where Qwen2.5-VL's vision tower runs and attaches it (S4, docs/tasks/task-multimodal-support-2026-10.md): aikit's gpu/qwencuda
tower under --backend cuda when the binary registers one (cuda/vision_towers.go imports it) and the tower is float32 (G-S4q: correct at real size, 1.6-2.7x the
CPU tower); the CPU everywhere else, with the reason named. -require-backend turns each CPU fallback under cuda into a refusal. -vision-device cpu arrives
here as backend "cpu". Metal has no Qwen2.5-VL device tower yet (the owner's rebuild on the Metal base is the Mac's), so only cuda asks for one.
```

## server.publishLoaded

Moved from `internal/serveapp/admin.go` (the comment above `lm.model.Close()` in `server.publishLoaded`) on 2026-10-09.

```text
M-24: the LOSER of the race holds a fully loaded model — resident device
memory, the .giw mmap, an uploaded block drafter — and refusing to publish it
used to just drop the pointer. purego installs no finalizers, so nothing ever
reclaims those; that is the whole reason the drain design exists. Close here
rather than in a defer: it must NOT run on the success path, where the registry
now owns the model.

Safe to Close unconditionally: loadDecoder always builds a fresh decoder.Load,
so this entry shares its weights with no other registry entry, and retainLocked
has not run for it — it was never published.
```

## server.handleAdminUnload

Moved from `internal/serveapp/admin.go` (the comment above `server.handleAdminUnload`) on 2026-10-09.

```text
handleAdminUnload drops a model from the registry and DRAINS before freeing its native memory.

The naive fix — lm.model.Close() straight after the registry delete — is a use-after-free: a
request past pick() but not yet at enter() holds the *lm pointer and touches lm.model in its
preamble (tokenize/prepare) with no lock, so a Close there frees weights mid-request (on CUDA, a
driver SIGSEGV). The safe fix is a DRAIN: every in-flight holder takes a per-model liveness RLock
via withModel (spanning the preamble and the generation), and unload waits that lock out before
closing. See docs/completed/task-admin-unload-drain.md and the reciprocal note at resident.Close.

Two phases, in unloadByName below (shared with the web route, W32 — task-web-ui-2026-09.md).
Phase 1 (under regMu): unpublish the entry and decide last-ownership — delete-before-decide, so
two concurrent sibling unloads cannot both decline (releaseLocked). Phase 2 (startDrain, detached):
drain in-flight holders, checkpoint the settled KV, close the entry's private natives, close the
shared model iff last owner. The response is a bounded wait: 200 (freed) if the drain completes
within -unload-drain-wait, else 202 with the drain continuing detached — the model is unroutable
immediately either way, and /health lists what is still draining. ?wait=false skips straight to
202. (This replaces the old 409-busy, which was only ever safe because it never freed anything.)
```

## anthropicInputBytes

Moved from `internal/serveapp/anthropic.go` (the comment above `anthropicInputBytes`) on 2026-10-09.

```text
anthropicInputBytes sums the TOKENIZABLE text across an Anthropic request — the system prompt
plus every message's text blocks, its replayed tool_use calls, and any declared tool schemas.
It is the /v1/messages analogue of chatInputBytes, and it matters that the text half uses
anthropicText: image blocks (and cache_control metadata) are excluded, so a base64 image is
never charged against a context window it does not consume. A vision request is a few hundred
tokens of image regardless of its megabytes on the wire.

M-15 (audit-2026-09-10): tool_use blocks and req.Tools were excluded entirely before this —
anthropicText's text-only sum skips a tool_use block's own type ("tool_use", not "text"), and
nothing summed req.Tools at all. anthropicTurns renders an assistant's replayed tool_use calls
into the prompt regardless of whether tools are active THIS turn, and RenderToolsSegments
renders every declared tool's schema whenever they are — both callers of this function guard
BOTH branches with one call (their own documented design), so both are priced unconditionally
here too, matching that intent rather than duplicating the guard per branch.
```

## anthropicTurns.roles

Moved from `internal/serveapp/anthropic.go` (the comment above the role check in `anthropicTurns`) on 2026-10-09.

```text
G13: the Anthropic Messages API accepts exactly two roles in this array.
Reject anything else instead of folding it into the conversation.

Before this check, anthropicRole mapped everything that is not "assistant"
to a USER turn and nothing validated, so a typo'd, invented, or wrong-API
role ("developer", "Assistant", "sytem") did not fail — it silently
restructured what the model saw. That is a worse outcome than a 400 for
every caller: a real Anthropic-shape client only ever sends legal roles, so
rejection costs it nothing, while anything else gets a loud failure instead
of a quiet mangling. It is also what upstream does, which is the
compatibility bar this surface is held to.
```

## batchRecord

Moved from `internal/serveapp/batches.go` (the comment above `batchRecord`) on 2026-10-09.

```text
batchRecord is J4's unit of work over J2's job store (task-work-queue-2026-09.md): N lines,
submitted as N ordinary jobs (see batches_run.go), tracked here only for what a job alone
cannot answer — aggregate status, original input order, and the assembled output.

wg is Add(n) once at creation (batchStore.create) and Done() once per line as its own last act
(batches_run.go) — not a job-store concept, this record's own completion signal, so
finalizeBatch (batches_finalize.go) needs no polling: wg.Wait() IS "every line is terminal."

mu guards Status/Results/OutputFileID/ErrorFileID/CompletedAt: many line goroutines write
Results[i] concurrently (disjoint indices, so the writes themselves never race each other) but
Status and the two file-id fields are read by every GET and written by exactly one finalizer,
so they need the same discipline job.go's own mu gained in J3 after -race caught a real bug
there (internal/serveapp/job.go:jobStore's doc comment) — written locked from the start here instead of finding the
same class of defect twice.
```

## batchRecord.requestCancel

Moved from `internal/serveapp/batches.go` (the comment above `batchRecord.requestCancel`) on 2026-10-09.

```text
requestCancel is the batch-level analogue of jobStore.cancel: mark cancelling, then cancel
every constituent job not yet terminal (checked live against jobStore, not against Results,
since Results is only written by a line's OWN goroutine as it finishes — a line still running
has no Results entry yet either way).

JobIDs is copied under b.mu before ranging over it — found by -race, not by inspection: each
line's goroutine writes its own slot via setJobID (batches_run.go) under the same lock, and a
plain read of the slice here raced those writes even though every write lands at a disjoint
index (internal/serveapp/job.go:jobStore's doc comment names the same class of bug in J3's job store).
```

## setJobID

Moved from `internal/serveapp/batches_run.go` (the comment above `setJobID`) on 2026-10-09.

```text
setJobID records line i's job id under the batch's own lock. Found by -race, not by
inspection, the same way job.go's own mu gained coverage in J3 (internal/serveapp/job.go:jobStore's doc comment):
requestCancel (batches.go) reads the whole JobIDs slice from a different goroutine while lines
are still writing their own slot — disjoint indices don't save a plain slice write from racing a
concurrent read of the slice's backing array with no synchronization between them.
```

## featureCache

Moved from `internal/serveapp/feature_cache.go` (the comment above `featureCacheBudget`) on 2026-10-09.

```text
A vision tower is the slow part of an image turn on the families whose tower runs on the CPU: the cold-user run
(docs/measurements/cold-user-2026-10-05-nobara-pc.md, F) measured 37.5 to 42 s to first token for a 1024x640
screenshot on Qwen3.5-0.8B, the same for a byte-identical resend, with `-backend cuda` changing nothing because the
tower stays on the CPU. The decoder's own image reuse (P9a) keeps the KV of a resent image, but only on a resident
non-recurrent model, so a hybrid like Qwen3.5 re-encoded every time.

This caches the tower's OUTPUT per image, so a chat client that resends the conversation, which every one does
each turn, pays for the encode once. The features are a pure function of the image bytes and of the loaded
model's fixed vision settings, and the cache is per loaded model, so a hit is bit-identical to recomputing.
Keyed by SHA-256 of the raw bytes rather than the 64-bit imgHash the KV reuse uses: this one hands back numbers
instead of skipping a prefix, and a collision there would be a wrong picture, not a missed optimisation.
```

## attachGemma3Tower

Moved from `internal/serveapp/gemma3_tower.go` (the comment above `attachGemma3Tower`) on 2026-10-09.

```text
attachGemma3Tower loads Gemma 3's SigLIP encoder (load(true) is the int8 one) and attaches its device-resident form.

Float32 is the CUDA default since 2026-10-08 (owner): it reproduces the CPU float32 reference's reply and serves a new image in 2.4 s against the int8 tower's 4.6 s, at the price of about 1.7 GiB of VRAM
against 0.56 (docs/tasks/task-multimodal-support-2026-10.md, "Night 2026-10-08"). A card with less room might not hold it, and a failed attach would drop the request to the CPU tower at about 21 s, which is
worse than the int8 device tower it replaced. So when fallback is set (the float32 tower was the DEFAULT, not asked for) a float32 attach that does not take is released and the int8 tower is attached in its
place, with a line saying why. An explicit `-vision-quant f32` sets fallback false and keeps the old behaviour: attach, or warn, or fail under -require-backend.

It returns the encoder that is attached (or the CPU one), whether it is int8, and whether its device form is attached.
```

## resolveGemma3VisionQuant

Moved from `internal/serveapp/gemma3_tower.go` (the comment above `resolveGemma3VisionQuant`) on 2026-10-09.

```text
resolveGemma3VisionQuant settles Gemma 3's UNSET -vision-quant on CUDA before anything is planned. Float32 is the default there (towerInt8), and the plan subtracts its reserve (about 2.4 GB with the margin) from the
KV budget up front. On a card where one KV slot at the context floor no longer fits beside the decoder and that reserve, the resident build would decline and the decoder would run on the CPU, which is far worse than
the int8 tower it replaced, and the post-attach fallback (attachGemma3Tower) cannot undo a plan already made. So the default is float32 only when the card can hold the decoder, one floor-context KV slot and the float32
tower; otherwise it becomes int8 here, with a note, and the plan prices the small reserve. An explicit -vision-quant is never touched.

The decoder and KV sizes are estimates from the checkpoint, because the model is not loaded yet: resident weights were 2.0 GB from 8.6 GB of safetensors on the 4B (0.233, taken as 0.27 for cushion; a GGUF or .giw is
already quantized, taken at its size), and one 4096-position KV slot was 1088 MB (0.54 of the weights, taken as 0.55). The estimate leans toward int8 on a card that would just have fit float32; it never
leans toward declining. Anything it cannot read (an hf: reference, no free-VRAM probe, several models) leaves the default alone and the post-attach fallback as the guard.
```

## resolveGemma3VisionQuantMetal

Moved from `internal/serveapp/gemma3_tower.go` (the comment above `resolveGemma3VisionQuantMetal`) on 2026-10-09.

```text
resolveGemma3VisionQuantMetal is resolveGemma3VisionQuant on Metal (S18 on the Mac, docs/tasks/task-multimodal-support-2026-10.md): the unset default is
the f16 tower when the budget holds the decoder, one KV slot at Metal's 2048-position floor and that tower, and the int8 tower (tower_gemm_w8, groups of
32) otherwise, with a note. The decoder's bytes are an estimate from the checkpoint: about 0.34 of the safetensors on the device (Gemma 3 4B's 5.15 GB
guard figure, less its 0.53 GB of KV, is two copies of about 2.3-2.9 GB), counted twice unless a fresh sidecar .giw will be loaded instead (a heap
load keeps the host copy beside the device one; part 1 of S18). It leans toward int8: a wrong f16 choice can push the decoder itself to the CPU, which
is far worse than the int8 tower.
```

## glm_ocr_vision.header

Moved from `internal/serveapp/glm_ocr_vision.go` (the comment at the top of the file) on 2026-10-09.

```text
GLM-OCR image serving (O3, docs/tasks/task-glm-ocr-2026-10.md). The route is the Qwen2.5-VL / Qwen3.5+ one —
preprocess -> tower -> GenerateQwenVL with m-RoPE, the merged tower rows replacing the <|image|> run — with these
differences, all of which live here or in the loadedModel.glm branches of vision_serve.go:
  - the tower is aikit's GlmOcrVisionEncoder, loaded on first use (like qwen3Tower);
  - the preprocessing config halves the file's pixel bounds (multimodal.LoadGlmOcrPreprocessConfig);
  - the image block and the template are GLM's, image first (multimodal.GlmOcrImageBlock, chat.GlmOCR);
  - NO per-image token cap here, unlike Qwen3.5's qwen3MaxImageTokens: the processor's own ceiling is 6,144 tokens
    (4.82 MP) and the serve-side default is O4's decision (the owner picks it), so serve accepts what the processor
    does and an image that does not fit the context is refused by name (imageFitsContext), never truncated.

The tower stays f32 unless -vision-quant int8 is given: serve's usual "a GPU backend implies an int8 tower" rule is for
the resident GPU encoders. Under --backend metal the f32 tower runs on Metal (S2, planGridTower); its int8 form is not
gated on the real checkpoint.
```

## jobStore.markRunning

Moved from `internal/serveapp/job.go` (the comment above `jobStore.markRunning`) on 2026-10-09.

```text
markRunning records the pending -> running transition for an already-created job.

J3 (task-work-queue-2026-09.md) added a genuine concurrent reader of *job's fields:
handleGetJob, running on the HTTP handler's own goroutine while runJob's goroutine calls this
and finish (below). J2 never needed s.mu held across a field mutation — nothing read a job's
fields from outside the single goroutine that owned it. Every mutation here is under s.mu now;
see snapshot's own comment for the matching read-side fix (found by -race, not by inspection).
```

## loadedModel.closeEntryNatives

Moved from `internal/serveapp/liveness.go` (the comment inside `loadedModel.closeEntryNatives`) on 2026-10-09.

```text
N-32 (docs/audit-2026-09-10.md): gemma4Enc (*vision.Gemma4Encoder) has no Close method
either — same shape as vproj, plain weights — but unlike vproj it was never dropped here
at all, retaining a served Gemma 4 vision tower's weight memory past unload. The audit
flagged this as a latent risk "if a native Close appears"; it is simpler than that: the
field is already live (main.go's loadGemma4VisionTower, vision_serve.go's Forward path)
and just needed the same nil-out vproj already gets.
```

## reqTrace

Moved from `internal/serveapp/reqlog.go` (the comment at the top of the file) on 2026-10-09.

```text
-log-requests (R25, docs/tasks/task-first-hour.md): one line per generation request on stderr — route, model, status, prompt and completion tokens,
time to first token, total time. A cold-user run on goinfer-serve learned its prompt sizes only from error bodies: nothing said what a client had sent.
Off by default, so a quiet server stays quiet.

The middleware owns the status and the clock; the token counts and the first-token time can only be known where tokens are read, which is
streamTokens — the shared tail of every generation (drive and driveVL), so a request that makes several generations (a tool loop) is counted once
per request: prompt tokens of its first generation, completion tokens summed, first token of the first.
```

## swapguard.header

Moved from `internal/serveapp/swapguard.go` (the comment at the top of the file) on 2026-10-09.

```text
S3 (docs/tasks/task-never-swap-2026-09.md): both halves of the swap tripwire.

The SERVING half (armSwapGuard, armed once for the process's life): "the callback flips the
admission gate to refuse new requests..., lets in-flight generations finish, logs once, and
re-opens when swap-used returns to within threshold of the baseline for 30 s (hysteresis, so it
does not flap)." Reuses haltGate's exact chokepoint (right before inf, on every generation
route) rather than adding a tenth wrapper alongside the 9 existing srv.haltGate(...) call
sites: swapGuardTripped is a second condition haltGate checks, next to K2's halted pointer —
see halt.go. Unlike halt(), which cancels every in-flight generation (K1's cancelAll), tripping
the swap guard does NOT cancel anything already running, only refuses new admissions — the
brief's own "lets in-flight generations finish".

The LOAD-TIME half (internal/swapguard's ArmLoad, shared with goinfer-chat; armed and torn down around exactly one decoder.Load
call): "the callback cancels the load's context... the direct build needs a check between
layers in parallelLayers" — that plumbing lives in decoder (Options.LoadAbort, checked by
parallelLayers), this file only arms/tears down the watch around loadDecoder's decoder.Load
call sites and turns a plain ErrLoadAborted into the priced message the brief asks for ("swap
grew 0.9 GB during load — resident weights 12.6 GB + mapped source 12.1 GB on a 16 GB machine;
use the sidecar path / a smaller quant"). Scoped to the GGUF direct-build resident path only —
see Options.LoadAbort's own doc comment for why the .giw/streaming path is unaffected (the
abort channel is a genuine no-op there today, not specially handled here).

Both halves share swapguard.ThresholdBytes so GOINFER_SWAP_GUARD can't quote two different
numbers for the same env var.
```

## systemone_clef.header

Moved from `internal/serveapp/systemone_clef.go` (the comment at the top of the file) on 2026-10-09.

```text
Route C of POST /v1/systemone (D13 of docs/tasks/task-constrained-confidence.md): a model directory that carries joint_head.safetensors is Cloudflare's Clef
decision model, and one backbone pass over the whole record answers every question (internal/clef). The request is the same TypeSafe shape; what differs is
that it is read as the reference's encoder reads it, from the raw body, and answered in the reference's own shape (systemone_answer in the checkpoint's
joint_schema_model.py, read in full for docs/measurements/decisions-d10-clef-2026-10-02.md):

  - noul: round(P(true), 4);
  - choice: the argmax over the REQUEST's criteria order (the first maximum wins), the probabilities by option in that order, and confidence = the top probability;
  - score: the expected level, the legend, the probabilities by level, and confidence = the top probability.

confidence is the top probability on this route (owner decision 2026-10-03), not the label route's margin over uniform, and the response says so. Probabilities
are rounded to four decimals as the reference rounds them.
```

## clefQuantRefusal

Moved from `internal/serveapp/systemone_clef.go` (the comment above `clefQuantRefusal`) on 2026-10-09.

```text
clefQuantRefusal is the error a Clef model gets for a quant it is not offered at, or nil. int4 is refused (owner decision 2026-10-03, after D13): on 150
records it read mean KL 0.051 and top-1 0.840 against the f32 reference on the CPU, and 0.044 and 0.840 on CUDA, where the default int8int8 reads 0.0165
and 0.907 (decisions-d13-clef-fidelity-2026-10-03.md section 7). Other quants are not refused here: f32 and int8int8 are graded, int4mix is simply not measured.
```

## serveChatToolsWith.guard

Moved from `internal/serveapp/tools.go` (the comment above the context-window guard in `serveChatToolsWith`) on 2026-10-09.

```text
G1c, extended (audit-2026-09-02 M-21). The guard reached three routes; this was one of the
five that still ran a full O(n) tokenize over an arbitrary body before rejecting it — the
G1c comment prices what it removes at "~27 s of BPE + gigabytes of ids" on a multi-MiB body.
M-15 (audit-2026-09-10): tools are ACTIVE on this path (that's why serveChatToolsWith was
reached at all), so req.Tools' schema bytes are added here — RenderToolsSegments below
renders every one of them into the prompt.
```

## serveChatToolsWith.buffer

Moved from `internal/serveapp/tools.go` (the comment above `ss` in `serveChatToolsWith`) on 2026-10-09.

```text
Tool decisions need the whole output, so buffer (even when streaming).

G19: when streaming, SSE now starts BEFORE the generation and a keep-alive
comment frame ticks while the buffer fills. Without it this path sent zero
bytes for the whole generation — measured at 1682.6s to first byte against a
harness whose idle timeout was 300s, which no output can survive however
correct it is. The buffering itself is unchanged, and comment frames carry no
data, so tool-call parsing sees exactly what it saw before.

The cost of starting SSE early: a generation error can no longer be a 500 on
the streaming path, because the headers are already flushed. That is the M1
convention sseErr exists for and what the non-tool streaming paths already do.
The non-streaming path below keeps its 500 unchanged.
```

## constrainForcedTool.named

Moved from `internal/serveapp/tools.go` (the comment above the `namedForce` check in `constrainForcedTool`) on 2026-10-09.

```text
N-18: a NAMED tool_choice that matched no tool used to land here and return nil —
the request then generated completely unconstrained, having asked for one specific
function. The 2026-08-05 audit made "named but unconstrainable" a 400 and left
"named but nonexistent" falling through, which is the louder of the two errors: the
caller has a typo or a stale tool list, and a prose answer looks like the model
simply chose not to call anything.
```

## toolUnionEnabled

Moved from `internal/serveapp/tools.go` (the comment above `toolUnionEnabled`) on 2026-10-09.

```text
toolUnionEnabled / toolUnionAuto are GOINFER_TOOL_UNION: on by default for both required /
Anthropic any (from token 1) and auto (lazily, on the opener); "0" turns both off (decoding
exactly as before T1). docs/measurements/tool-union-2026-09-24.md:

  - auto first shipped default-OFF: an UNGATED processor disabled the decoder's on-device
    greedy/sampling fast paths for the whole turn, and prose auto turns decoded 0.81-0.99x (gate B).
  - the gated build (LazyMasker.Gate → SamplingParams.LogitProcessorGate) keeps those paths until
    the opener: prose turns 0.999-1.001x (gate B'), and a turn that never arms is byte-identical to
    "=0" at greedy AND T=0.7 (630/630, gate A'). The 7B's T0 turns then produce 0 invented or
    broken calls and 0 invalid arguments (gate C).
```

## forcedTool

Moved from `internal/serveapp/tools.go` (the comment above `forcedTool`) on 2026-10-09.

```text
forcedTool returns the single tool the call must be (a forced function, or the
lone tool) — the "tight when unambiguous" case. nil means don't constrain.

The lone-tool convenience is for the model's FIRST call. afterToolResult (the conversation already ends in a tool
result, endsWithToolResult) lifts it under auto: forced on every turn it left a client with exactly one tool unable to
ever get an answer, the agent-livelock `serve check` reported as "turn two asked for the tool again instead of
answering" on Qwen2.5-Coder-7B and Qwen2.5-7B-Instruct alike (G39). An explicit "required" or a named function is the
client's own request and is forced on every turn, as before; "none" is never forced.
```

## towerParts.glm_ocr

Moved from `internal/serveapp/tower_reserve.go` (the comment inside the `glm_ocr` case of `towerParts`) on 2026-10-09.

```text
The ceiling's scratch alone is ~2.5 GB and, reserved, took GLM-OCR's CUDA context from 16384 to 4931 positions on the 8 GB card (measured 2026-10-07). Reserve for
about 1.5 MP and let a larger image fall back to the CPU tower (deviceFallback) instead.
```

## metalTowerEstimate

Moved from `internal/serveapp/tower_reserve.go` (the comment above `metalTowerEstimate`) on 2026-10-09.

```text
metalTowerEstimate is what a Metal device tower holds of the Mac's memory (S18 on the Mac): the projections at the form Metal uploads, the rest in float32, plus
the scratch. The grid towers (SigLIP, Qwen2.5-VL, Qwen3.5+, GLM-OCR) upload f16 projections since S17's lever B, or int8 in groups of 32 for Gemma 3's
int8 tower (S18, tower_gemm_w8: a byte plus an f32 scale per 32); Gemma 4's Metal tower is float32. Calibrated against Gemma 3's measured footprint
(G-S18h, metal TestS18TowerHostMemory, 2026-10-08): 1172 MB f16 and 843 MB int8 against 1292 and 932 from this, about 10% over, the safe direction (the
scratch term is towerParts', which over-counts Metal's).
```

## towerInt8VRAMEstimate

Moved from `internal/serveapp/tower_reserve.go` (the comment above `towerInt8VRAMEstimate`) on 2026-10-09.

```text
towerInt8VRAMEstimate is the Gemma 3 W8A8 SigLIP tower's footprint on the device: the block weights at one byte, the patch embed and position table in float32, and the fixed scratch (about nine
hidden-wide float32 buffers per patch). Calibrated against a measurement, not derived: the tower held 558 MiB of the 8 GB card after attaching and after a forward (cuda TestSiglipCUDA_int8VRAM,
2026-10-07), against 575 MiB from this formula.
```

## qwen25VLTowerEstimate

Moved from `internal/serveapp/tower_reserve.go` (the comment above `qwen25VLTowerEstimate`) on 2026-10-09.

```text
qwen25VLTowerEstimate prices goinfer's float32 Qwen2.5-VL tower on CUDA (cuda/qwen25_vision.go), calibrated to a measurement and not derived (S7 on CUDA's fix, 2026-10-08). On the 8 GB card the tower's weights took
2758 MiB, 337 MiB over the arithmetic size (about 570 separate buffers, each rounded up to the allocator's 2 MiB quantum); the first image's scratch is np * (7 hidden + 2 padded intermediate + patch dim + 2 head dim) * 4
bytes, 270 MiB at 4096 patches. The ceiling stays 8192 patches (about 1.6 MP); a larger image falls back to the CPU tower by name (deviceFallback).

There used to be a further 256 MiB of "slack for the resident build's scratch beyond the plan's reading, measured 535 MiB against the 384 MiB margin". That was a misattribution: the text build's scratch after the plan's
probe is 0 MiB, and the shortfall it was patching was the allocator rounding of the TEXT model's weights, which the CUDA plan now prices itself (cuda.packedAllocSlack, 2026-10-08). Removed after a served probe on the
card: serve's defaults on the 3B, a ~8,000-patch image (2,014 prompt tokens) then a small one, the tower on CUDA, no failure, 1152 MiB of the 8192 still free after all three requests.
```

## encodeVisionSegments

Moved from `internal/serveapp/vision_serve.go` (the comment above `encodeVisionSegments`) on 2026-10-09.

```text
encodeVisionSegments encodes a vision prompt the way the TEXT path already does: the template's
structural markers and the image block as Special segments, the user's own words as ordinary
content the added-token trie never sees.

M-22. The vision path called lm.encode(lm.tmpl.Render(...)) — Tokenizer.Encode, whose own doc
says "do NOT use this on untrusted content" — while the text path had used EncodeSegments since
M25. So a user message in an IMAGE request containing "<end_of_turn>\n<start_of_turn>model\n"
(or "<|im_end|>…<|im_start|>system") became real control tokens and forged a turn boundary: the
hardening reached one route and not the other, which is audit §0 theme 2 exactly.

The image block has to stay SPECIAL — its sentinels and soft-token run are what FindImageRun
locates and what the embed-by-vector seam replaces — so it cannot simply be prepended to the
content segment, which is untrusted by construction. It is spliced back in as its own Special
segment instead, and a block that fails to splice is an error rather than a prompt that silently
tokenizes the sentinels as text (the imgLen check downstream would catch it, but late and with a
misleading message about a template mismatch).
```

## recoverDeviceTower

Moved from `internal/serveapp/vision_serve.go` (the comment above `recoverDeviceTower`) on 2026-10-09.

```text
recoverDeviceTower runs a tower's forward and turns a panic into an error that names the likely cause. A device tower grows its scratch on the first image, after the
resident decoder has taken the device's memory, and aikit's Qwen2.5-VL tower panicked on an allocation failure (`CUDA_ERROR_OUT_OF_MEMORY`) instead of returning it:
unrecovered, one big image killed the whole server (found by the S4 default-plan run, 2026-10-07). The request fails with the reason; the server and the other requests
live. Every tower passes through it: the grid towers inside deviceFallback.run, Qwen2.5-VL inside qwenDeviceForward, and every family's features (Gemma 3's SigLIP,
Gemma 4's image and audio towers included) at withFeatureCache.
```

## webUIFS

Moved from `internal/serveapp/webui.go` (the comment above `webUIFS`) on 2026-10-09.

```text
ONE DIRECTORY, NOT ONE FILE (docs/tasks/task-web-ui-2026-09.md §6.1). The page was a single
1,828-line HTML file; the web-UI plan roughly doubles it, so it is split into index.html plus
webui/ui/app.css and webui/ui/app.js. What is deliberately NOT given up: still no build step, no
bundler, no toolchain — plain files, embedded verbatim, one binary, fully offline. The assets
live under ui/ and are referenced relatively ("ui/app.js"), so the same page also loads from
file:// with no server, which is how a headless browser can check it on a box whose sandbox
blocks loopback HTTP.
```

## searchLimit

Moved from `internal/serveapp/webui.go` (the comment above `searchLimit`) on 2026-10-09.

```text
searchLimit bounds how many suggestions a search returns — a dropdown, not a full listing;
pull.Search sends no limit= to HuggingFace at all when given 0, which is a request shape this
route should never produce. Raised from 8 to 50 live during testing (2026-09-17): 8 was too
narrow to surface a less-trending-but-still-relevant repo past HF's own trendingScore ordering
(pull.Search's own doc comment — not downloads or likes) for anything but the most obvious query.
```

## webPullRef

Moved from `internal/serveapp/webui.go` (the comment above `webPullRef`) on 2026-10-09.

```text
webPullRef resolves a pull request's repo box and the clicked selector (if any) into one Ref,
WITHOUT string concatenation. req.Repo alone decides ref.Repo (`pull.ParseRef` cuts at the
FIRST colon, so blindly appending a second selector after a box that already carries one —
"owner/repo:q4_k_m" typed in, then a file clicked — produced "owner/repo:q4_k_m:file.gguf",
re-cut into repo="owner/repo", selector="q4_k_m:file.gguf": a ".gguf"-suffixed string that
LOOKS like a filename and is looked up as one, verbatim, in a repo that publishes no such name.
Listing never showed this, because handleWebList parses req.Repo alone and only ever reads
ref.Repo back out of it — so every Pull button failed while List worked, which read like a bad
repo rather than a bad concatenation.

A clicked file or quant REPLACES whatever selector the box already carried, rather than
appending to it — the click is the more specific, more recent choice. With neither clicked, the
box's own parse is returned as-is: re-parsing a resolved `demo:` tier would still yield the same
Repo/File, but would drop Pin/Bytes (ParseRef only ever sets those for a literal "demo:tier"
input, not for the repo/file pair a tier resolves to) — the digest a demo: pull is supposed to
verify against.
```

# Test files

Moved from the comments of the package's `_test.go` files (CC5); same conventions as above.

## TestPrepare_rejectsUnboundedTopLogprobs

Moved from `internal/serveapp/limits_test.go` (the comment above `TestPrepare_rejectsUnboundedTopLogprobs`) on 2026-10-09.

```text
C-08: top_logprobs was the one sampling field `prepare` passed through without a range check.

Each retained entry is a TokenLogprob and the response builder then materializes a
map[string]any per entry BEFORE writing a byte, so {logprobs:true, top_logprobs:150000,
max_tokens:4096} on a 152k-vocab model retains ~9.8 GB of them and OOM-kills the process — a
fatal Go allocation failure, not a 500 anyone can catch. One request does it, and every other
sampling field on the same struct was already validated.
```

## TestStreamTokens_externallyCancelledIsTruncatedNotClean

Moved from `internal/serveapp/limits_test.go` (the comment above `TestStreamTokens_externallyCancelledIsTruncatedNotClean`) on 2026-10-09.

```text
M-23: graceful shutdown cancels every in-flight generation, and `genErr` treats every
context.Canceled as a clean end — so the truncated text came back as a 200 with
finish_reason:"stop". The client cannot tell it from a model that finished.

srvCancel() runs BEFORE Shutdown, so the client is still connected and reads a partial answer as
complete. A client disconnect looks identical here (both arrive as a cancelled request context),
which is why the answer is "length" rather than an error: truthful in both cases, and the signal
a client already knows how to act on.
```

## TestServe_everyTokenizingRouteGuardsItsInputSize

Moved from `internal/serveapp/limits_test.go` (the comment above `TestServe_everyTokenizingRouteGuardsItsInputSize`) on 2026-10-09.

```text
M-21: EVERY ROUTE THAT TOKENIZES MUST REJECT AN OVERSIZED BODY FIRST.

G1c added `promptTooLargeForContext` to three routes and left five running a full O(n) BPE over
an arbitrary body before rejecting it — the G1c comment prices that at "~27 s of BPE + gigabytes
of ids" on a multi-MiB body. anthropic.go's own comment called its half "the same defect this
release claims to fix, left half-covered on one surface", and count_tokens was the worst of them:
it never enters the per-model queue, so up to -max-inflight (128) such tokenizations run at once.

Three routes guarded and five not is a COUNTING failure, so it is counted rather than remembered.
A route here is a handler that tokenizes: it takes a ResponseWriter and calls one of the
tokenizing entry points. Each must call the guard, or name itself as guarded by its caller.
```

## TestServe_everyTokenizingRouteGuardsItsInputSize.transitive

Moved from `internal/serveapp/limits_test.go` (the comment above `TestServe_everyTokenizingRouteGuardsItsInputSize.transitive`) on 2026-10-09.

```text
Direct calls, plus the prompt builders that tokenize TRANSITIVELY. The vision routes were the
hole in the first cut of this check: serveVisionChatWith contains no tokenizer call of its
own — visionPrompt does — so the gate passed over the very route whose guard it was written
to protect, and dropping that guard produced no failure. Caught by mutation, not by reading.
```

## TestDecoderEmbedder_truncatesToTheContextWindow

Moved from `internal/serveapp/limits_test.go` (the comment above `TestDecoderEmbedder_truncatesToTheContextWindow`) on 2026-10-09.

```text
C-07, the serving half: the decoder-as-embedder must TRUNCATE to the model's context window.

C-21 capped one input at 1 MiB of BYTES and left the token count unbounded, so ~1 MiB of short
words is ~500k tokens: HiddenLast preallocates KV for every one of them and runs a sequential
per-token forward with no context, under the embed mutex, until the process is OOM-killed. The
decoder-side guard (TestHiddenLast_refusesMoreTokensThanTheContextWindow) makes that an error
rather than an OOM — but an ERROR is not the right answer for the embedder, which should do what
HF's truncation=True does and what the aikit encoder path already did. So both halves are pinned:
the decoder refuses, and this one never sends it more than it can take.
```

## TestDecoderEmbedder_truncatesToTheContextWindow.exact

Moved from `internal/serveapp/limits_test.go` (the comment above `TestDecoderEmbedder_truncatesToTheContextWindow.exact`) on 2026-10-09.

```text
truncateForContext is the EXACT function tokenize() calls (V-21, docs/review-2026-09-04.md):
this used to re-derive the room--/ids[:room] arithmetic beside it instead of calling it, so
a bug in the real code — not a paraphrase of it — would have passed this test unnoticed.
```

## TestJSONDecodeMessage_neverLeaksGoTypeNames

Moved from `internal/serveapp/limits_test.go` (the comment above `TestJSONDecodeMessage_neverLeaksGoTypeNames`) on 2026-10-09.

```text
N-16: BOTH decoders must shape a JSON error the same way, and neither may echo the raw one.

M-06 and R-11 removed the Go struct/field/type leak from decodeJSON; decodeAnthropicJSON still
appended err.Error() verbatim, so "invalid request body: json: cannot unmarshal string into Go
struct field anthropicReq.max_tokens of type int" went straight to the client. §0 theme 2 again:
one route hardened, its twin not.
```

## TestAnthropic_decodeUsesTheSharedShaping

Moved from `internal/serveapp/limits_test.go` (the comment above `TestAnthropic_decodeUsesTheSharedShaping`) on 2026-10-09.

```text
The Anthropic decoder must USE it. The test above proves the shaper is clean and says nothing
about the call site — the same gap that let M-21's vision route and M-22's call site pass.
```

## TestSessions_persistedStateIsOwnerOnly

Moved from `internal/serveapp/limits_test.go` (the comment above `TestSessions_persistedStateIsOwnerOnly`) on 2026-10-09.

```text
N-21: session persistence is the CONVERSATION. A .giw-kv blob replays what the user said and what
the model answered, and it was written 0o644 inside a 0o755 directory — readable by every local
account. The directory matters as much as the files: a readable one lists the session ids.
```

## TestSessions_persistedStateIsOwnerOnly.scan

Moved from `internal/serveapp/limits_test.go` (the comment above `TestSessions_persistedStateIsOwnerOnly.scan`) on 2026-10-09.

```text
Code lines only. The comment above the constants NAMES the old modes, and a substring scan
over the whole file matches its own explanation of the defect — the second time a check in
this batch did that, so it is worth doing deliberately rather than rediscovering.
```

## TestAdminLoad_racedDuplicateIsClosed

Moved from `internal/serveapp/limits_test.go` (the comment above `TestAdminLoad_racedDuplicateIsClosed`) on 2026-10-09.

```text
M-24: two concurrent admin loads of the same name leaked the loser's model.

loadDecoder runs OUTSIDE the registry lock (deliberately — it takes seconds), so two
requests for the same name both load, and the post-check refuses to publish the second.
The refused *loadedModel holds resident device memory, the .giw mmap and an uploaded block
drafter, and dropping the pointer released none of it: purego installs no finalizers, which
is the whole reason the drain design exists.

Asserted on the SOURCE because the leak has no observable behaviour to test — nothing
errors, nothing is slower, the memory is simply never returned. A test that loaded two real
models to watch RSS would be measuring Darwin's UBC rather than the defect (the audit's own
note about RSS-based guards inverting under pressure).

Since W5 the post-check lives in publishLoaded (admin.go), shared by the admin load and the web
UI's load (webui.go). So this checks two things: the refusal branch in publishLoaded closes what
it refuses, and neither handler publishes into s.models on its own, around that branch.
```

## TestReqID_isNotGuessable

Moved from `internal/serveapp/limits_test.go` (the comment above `TestReqID_isNotGuessable`) on 2026-10-09.

```text
N-17: response/message/tool-call ids were a SEQUENTIAL counter. Seeding it from UnixNano hid
that without fixing it — each id is exactly one more than the previous, so a client holding
its own `resp_<hex>` can walk ±1 onto other clients' ids. `previous_response_id` continues a
stored conversation from an id, so with `-addr 0.0.0.0` and one shared key that reads back
someone else's turns.
```

## TestAdminLoad_requestQuantIsTheExplicitOne

Moved from `internal/serveapp/limits_test.go` (the comment above `TestAdminLoad_requestQuantIsTheExplicitOne`) on 2026-10-09.

```text
N-19: admin-load set c.quant from the request but inherited c.quantSet from the CLI, and
explicitQuant() — which drives the .giw baked-quant mismatch check — reads quantSet. Both
directions were wrong, and they are opposite failures, so a fix has to be checked both ways:
```

## TestConstrainForcedTool_namedButNonexistentIs400

Moved from `internal/serveapp/limits_test.go` (the comment above `TestConstrainForcedTool_namedButNonexistentIs400`) on 2026-10-09.

```text
N-18: a NAMED tool_choice whose function is not in tools decoded completely unconstrained,
having asked for one specific function. The 2026-08-05 audit made "named but unconstrainable"
a 400 and left "named but nonexistent" falling through — the louder of the two, since it
means a typo or a stale tool list and the prose answer looks like a free choice.
```

## TestChatInputBytes_countsReplayedToolCallArguments

Moved from `internal/serveapp/limits_test.go` (the comment above `TestChatInputBytes_countsReplayedToolCallArguments`) on 2026-10-09.

```text
M-15 (audit-2026-09-10): the G1c guard's argument counted message TEXT only — tool schemas and
replayed tool_calls[].arguments are rendered into the prompt too (via RenderToolsSegments and
messagesToTurns respectively) but were never priced, so a small message with a huge schema or
huge replayed arguments passed the guard in constant time and then ran the full BPE anyway. The
existing TestServe_everyTokenizingRouteGuardsItsInputSize only checks the guard is PRESENT, not
what it measures — these test what it measures, directly and end to end.
```

## TestGuardComposition_hugeToolSchemaTripsTheBudgetAloneOnATinyMessage

Moved from `internal/serveapp/limits_test.go` (the comment above `TestGuardComposition_hugeToolSchemaTripsTheBudgetAloneOnATinyMessage`) on 2026-10-09.

```text
TestGuardComposition_hugeToolSchemaTripsTheBudgetAloneOnATinyMessage matches the audit's own
worst-case example (a tiny message, a huge schema): composed the exact way
serveChatToolsWith's real guard call does (chatInputBytes(msgs) + toolSchemaBytes(tools)), the
combined byte count must trip a budget the message text alone would pass. This is NOT a call
through the real HTTP handler — promptTooLargeForContext short-circuits to nil without a real
*decoder.Model, which a unit test here does not load — so it instead proves the composition
this session's fix sites now share is correct, leaving the direct per-function unit tests above
to prove each addend's own arithmetic.
```

## TestWebUI_pullRefReplacesSelector

Moved from `internal/serveapp/webui_test.go` (the comment above `TestWebUI_pullRefReplacesSelector`) on 2026-10-09.

```text
TestWebUI_pullRefReplacesSelector pins the fix for a real bug hit from the page: typing
"owner/repo:q4_k_m" into the repo box (List works fine — handleWebList parses req.Repo alone)
and then clicking a file button used to send {repo: "owner/repo:q4_k_m", file: "x.gguf"},
concatenated into "owner/repo:q4_k_m:x.gguf" and re-parsed. pull.ParseRef cuts at the FIRST
colon, so that became repo="owner/repo", selector="q4_k_m:x.gguf" — a string that still ends in
".gguf" and is therefore looked up as a literal filename no repo publishes, in a repo that
actually has "x.gguf" under a normal name. Every Pull button failed while List kept working,
which reads like a bad repo rather than a bad concatenation — exactly what the user hit.
```

## TestWebUI_rootRouteIsUnauthenticated

Moved from `internal/serveapp/webui_test.go` (the comment above `TestWebUI_rootRouteIsUnauthenticated`) on 2026-10-09.

```text
TestWebUI_rootRouteIsUnauthenticated guards V-02 (docs/review-2026-09-04.md): GET /{$} used to
be wrapped in auth(...), so a browser's plain navigation -- which sends no Authorization header
-- got the 401 JSON instead of the page, whenever -api-key was set (required off loopback). The
page is the ONLY place a user could type the key in, so this was a deadlock: loading the page
needed the key, and there was nowhere to enter the key without the page. auth stays on
/web/models/list and /web/models/pull, which actually act.

main()'s mux-building is inline, not a separately testable function (this is exactly why the
bug went unguarded -- webui_test.go could exercise handleWebUI directly but never through the
auth-wrapped mux registration), so this is asserted structurally: mux.HandleFunc("GET /{$}", ...)
must NOT wrap its handler in the auth closure, while the /web/models/* registrations must.
```

## TestWebUI_rootRouteIsUnauthenticated.chain

Moved from `internal/serveapp/webui_test.go` (the comment above `TestWebUI_rootRouteIsUnauthenticated.chain`) on 2026-10-09.

```text
Searches the WHOLE wrapper chain, not just the outermost call — V-20
(docs/review-2026-09-04.md) nested list/pull one layer deeper as
sameOrigin(auth(maxBytes(...))), and a check anchored on the outermost call alone
would have silently stopped seeing auth(...) the moment that landed.
```

## TestWebUI_listAndPullAreWrappedInSameOrigin

Moved from `internal/serveapp/webui_test.go` (the comment above `TestWebUI_listAndPullAreWrappedInSameOrigin`) on 2026-10-09.

```text
TestWebUI_listAndPullAreWrappedInSameOrigin is the wiring guard: the unit test above proves
sameOrigin works in isolation, but that says nothing about whether the actual routes call it —
the exact shape of gap this session's audit keeps finding (a helper with a test, and a call
site nobody checked).
```

## TestSSEWriter_heartbeatAndHandlerDoNotRace

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `TestSSEWriter_heartbeatAndHandlerDoNotRace`) on 2026-10-09.

```text
C-06: THE HEARTBEAT GOROUTINE AND THE HANDLER WRITE THE SAME ResponseWriter.

G19 started a ticker that owns w while the handler is silent; G21 then made the incremental tool
paths emit prose deltas to that same w during the same window. net/http's response/bufio.Writer
is not safe for concurrent Write/Flush. The existing gate could not see it:
heartbeat_test.go's model "never emits prose", so the overlap never happens there and -race has
nothing to observe.

This drives the overlap directly. Run with -race, which is what turns it from "probably fine" to
a verdict — the failure mode is a torn frame or a bufio panic, and the panic would be in the
TICKER goroutine, outside net/http's per-request recover, i.e. the process.
```

## TestSSEWriter_heartbeatAndHandlerDoNotRace.until

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `TestSSEWriter_heartbeatAndHandlerDoNotRace.until`) on 2026-10-09.

```text
Send until the ticker has written at least one heartbeat into the stream, not for a fixed time: 800 small
writes, or then 20ms of them, can finish before the first tick on a runner with a coarse timer. It failed
that way with "no heartbeat frame at all" on the macOS runner and three times on the Windows ones (about
15.6ms timer granularity; 2026-10-08, -09, -09). The watcher reads the recorder under the writer's own lock,
so the check is race-free; the 5s cap turns a heartbeat that never comes into the failure below, not a hang.
```

## TestSSEWriter_stalledClientFailsTheWriteInsteadOfBlocking

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `TestSSEWriter_stalledClientFailsTheWriteInsteadOfBlocking`) on 2026-10-09.

```text
M-17: a client that stops READING must not pin the handler forever.

sseSend's Flush blocked in net.Conn.Write with no deadline, so a stalled reader held the model's
queue slot — r.Context() cancels when the connection CLOSES, which a stalled client never does —
and every other request queued then 429'd. The fix is a per-write deadline, which is not the
server-wide WriteTimeout the M3 comment conflated it with.
```

## TestSSE_everyStreamingDriveSiteHeartbeats

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `TestSSE_everyStreamingDriveSiteHeartbeats`) on 2026-10-09.

```text
N-24 (docs/audit-2026-09-10.md): EVERY lm.drive/driveVL STREAMING SITE MUST START A HEARTBEAT,
NOT ONLY THE BUFFER-THEN-STREAM ONES.

TestSSE_everyBufferedStreamSiteHeartbeats below (M-19) only classifies buffer-then-stream
callbacks, and only ones that literally call lm.drive( — a site that streams per token as it
generates is deliberately exempted there (unconditionalStreamCall), and a site that calls
lm.driveVL( instead of lm.drive( is invisible to driveCallback's own substring search. Neither
exemption holds for the gap N-24 is about: the PREFILL window, which is silent no matter how a
site streams once tokens start, and applies identically to drive and driveVL. Six sites had it —
serveChatText and serveCompletion (openai.go, drive), streamMessages's plain-text branch
(anthropic_stream.go, drive), serveVisionChatWith and serveVisionMessages (vision_serve.go,
driveVL), and serveResponsesWith's plain-text branch (responses.go, drive) — found by reading
every lm.drive(/lm.driveVL( call site directly rather than trusting either classifier's
coverage. This pins them explicitly rather than trying to widen the delicate M-19 classifier
(driveCallback/unconditionalStreamCall) to a shape it was not designed for.
```

## TestSSE_everyBufferedStreamSiteHeartbeats

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `TestSSE_everyBufferedStreamSiteHeartbeats`) on 2026-10-09.

```text
M-19: EVERY BUFFER-THEN-STREAM SITE MUST START A HEARTBEAT.

G19 fixed two of three. The third — streamMessagesTools, the Anthropic tool path — emitted
nothing after message_start's single `ping` until the whole generation finished, on the surface
docs/server.md markets for Claude Code, where tool-bearing requests are the norm. The existing
gates (heartbeat_test.go) cannot catch a missing site: they need GOINFER_SERVE_MODEL and skip
without it, and they test the sites that already had one.

A buffer-then-stream site is definable, so it is checked rather than remembered: a handler that
drives a generation whose callback only appends to a builder — writing nothing to the client —
must start a heartbeat, because it is silent for the whole generation by construction.
```

## TestSSE_everyBufferedStreamSiteHeartbeats.callback

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `TestSSE_everyBufferedStreamSiteHeartbeats.callback`) on 2026-10-09.

```text
THE CALLBACK, NOT THE BODY. streamMessagesTools writes plenty — tool_use blocks,
message_end — but all of it AFTER drive returns. What makes a site silent for the
whole generation is that its drive CALLBACK sends nothing, and an earlier cut of
this check looked at the whole function and so excluded the one site it exists for.
```

## unconditionalStreamCall

Moved from `internal/serveapp/sse_writer_test.go` (the comment above `unconditionalStreamCall`) on 2026-10-09.

```text
unconditionalStreamCall reports whether cb — a driveCallback result — calls one of the
per-token streaming sends AT THE CALLBACK'S OWN TOP LEVEL, i.e. on every invocation, not
merely somewhere inside it behind a conditional.

V-21 (docs/review-2026-09-04.md): a plain strings.Contains(cb, "sseSend(") used to exempt a
call site the moment the literal substring appeared anywhere in the callback — including
inside an `if` that is frequently false. serveChatToolsWith's callback is exactly that shape:
sb.WriteString(t); if prose == nil { return }; if out := prose.Push(t); out != "" {
sseSend(...) } — sseSend is reachable only for incremental families, and even then only once
prose.Push has enough content to flush, so most tokens (and the whole non-incremental family
case) hit NEITHER branch. The old check still saw the substring and skipped the site entirely,
so it never even looked for sseHeartbeat(...) in the enclosing function — dropping that
heartbeat would have failed nothing. This walks brace depth relative to the callback's own `{`
(depth 1 = the callback's own top level) and only counts a marker call found there, exactly as
driveCallback already walks depth to find the callback's closing brace.
```

## TestConstrainForcedTool_M05.named

Moved from `internal/serveapp/openai_test.go` (the comment above `TestConstrainForcedTool_M05.named`) on 2026-10-09.

```text
N-18: forced == nil now depends on WHY. With namedForce, the caller named a function that
is not in tools — a typo or a stale tool list — and generating unconstrained answers a
different question than the one asked, so it is a 400. This case previously asserted the
opposite ("nil forced tool should not error"), which is the defect N-18 describes.
```

## TestLoadDecoderEmbedder_requestedBackendReachesTheLoad

Moved from `internal/serveapp/decoder_embedder_test.go` (the comment above `TestLoadDecoderEmbedder_requestedBackendReachesTheLoad`) on 2026-10-09.

```text
TestLoadDecoderEmbedder_requestedBackendReachesTheLoad is M-17's own gate
(docs/audit-2026-09-10.md): loadDecoderEmbedder must actually pass -backend through to the
embedder's own decoder.Load, or the resident HiddenLast path (decoder/embed.go) it may
implement is shipped but unreachable — every /v1/embeddings call silently takes the CPU
sequential path regardless of -backend.

Asserts against decoder.Model.RequestedBackend(), not ResidentActive(): this test binary
(plain `go test ./internal/serveapp/...`, no -tags gpu, no metal/cmd/serve submodule build)
cannot make ANY GPU backend actually activate — decoder.NewBackend falls back to CPU with an
explanatory error for every non-cpu name here, by design (M-19: the real backends live in
submodule entrypoints, not a build tag on the root binary). RequestedBackend() is exactly the
signal that survives that fallback: it records what Options.Backend WAS at load time,
independent of whether the backend actually built — which is exactly M-17's own claim ("the
decoder-embedder is loaded with decoder.Options{} (Backend ”)"), so this is a real,
hardware-independent test of the fix, not a proxy for one.
```

## TestLoadDecoderEmbedder_embedQuantQ8MapsToInt8

Moved from `internal/serveapp/decoder_embedder_test.go` (the comment above `TestLoadDecoderEmbedder_embedQuantQ8MapsToInt8`) on 2026-10-09.

```text
TestLoadDecoderEmbedder_embedQuantQ8MapsToInt8 is the other half of M-17: -embed-quant's own
vocabulary ("f32"|"q8") is NOT decoder.Options.Quant's ("" |"int8"|"int8int8"|"int4") — passing
"q8" straight through would make decoder.Load's parseQuant reject it outright as unknown
(worse than the silent-ignore this fixes: a hard load failure), so loadDecoderEmbedder must
translate "q8" to Quant's own "int8" instead.
```

## TestVision_userTextIsNotSpecialButTheImageBlockIs

Moved from `internal/serveapp/vision_hardening_test.go` (the comment above `TestVision_userTextIsNotSpecialButTheImageBlockIs`) on 2026-10-09.

```text
M-22: THE VISION PATH LET USER TEXT FORGE A TURN BOUNDARY.

The text path has used EncodeSegments since M25, so a "<end_of_turn>" typed into a message stays
literal. The vision path called lm.encode(lm.tmpl.Render(...)) — Tokenizer.Encode, whose own doc
says "do NOT use this on untrusted content" — so the same string in an IMAGE request became real
control tokens. §0 theme 2: the hardening reached one route and not the other.

This asserts the segment SHAPE rather than token ids, so it needs no tokenizer or model: the
image block must come out Special (FindImageRun depends on it) and the user's words must not.
```

## TestVision_userTextIsNotSpecialButTheImageBlockIs.real

Moved from `internal/serveapp/vision_hardening_test.go` (the comment above `TestVision_userTextIsNotSpecialButTheImageBlockIs.real`) on 2026-10-09.

```text
The REAL function, not a copy of it beside the test: an earlier cut of spliceImageBlock
looked for the block as a segment PREFIX and would have refused every vision request, and a
test that re-implemented the splice would have agreed with itself about that.
```

## TestVision_spliceUsesTheLastOccurrenceNotTheFirst

Moved from `internal/serveapp/vision_hardening_test.go` (the comment above `TestVision_spliceUsesTheLastOccurrenceNotTheFirst`) on 2026-10-09.

```text
TestVision_spliceUsesTheLastOccurrenceNotTheFirst pins V-19 (docs/review-2026-09-04.md):
spliceImageBlock used to splice the FIRST non-Special segment containing the block, anywhere in
the rendered history. An earlier turn that happens to contain the literal block text as ordinary
words — a user asking what the sentinel means, say — would get spliced instead of the real
current image turn, reopening the special-token-forging class M-22 closed (for this sentinel
instead of a role marker): the earlier turn's unrelated text gets tagged Special and parsed as
sentinels, while the real image tokens stay unspliced plain text.
```

## TestServe_noRouteEncodesARenderedPrompt

Moved from `internal/serveapp/vision_hardening_test.go` (the comment above `TestServe_noRouteEncodesARenderedPrompt`) on 2026-10-09.

```text
NO ROUTE MAY TOKENIZE A RENDERED CHAT PROMPT WITH Encode.

The two tests above prove spliceImageBlock segments correctly — they do NOT prove the vision
routes CALL it, and reverting visionPrompt to `lm.encode(lm.tmpl.Render(...))` leaves both of
them green. That is the same shape as M-21's vision hole one commit earlier: a helper with a test,
and a call site nobody checked.

So the unsafe PATTERN is banned instead of the safe one being asserted. Tokenizer.Encode consults
the added-token trie and its own doc says "do NOT use this on untrusted content"; a rendered chat
prompt always contains user content. EncodeSegments is the only correct way to tokenize one.
```

## TestServe_noRouteEncodesARenderedPrompt.comments

Moved from `internal/serveapp/vision_hardening_test.go` (the comment above `TestServe_noRouteEncodesARenderedPrompt.comments`) on 2026-10-09.

```text
Comments describing the old code are not the old code — this check fired on its own
explanation of the defect the first time it ran.
```

## TestVision_imageBlockNewlinesMatchTheRealProcessors

Moved from `internal/serveapp/vision_hardening_test.go` (the comment above `TestVision_imageBlockNewlinesMatchTheRealProcessors`) on 2026-10-09.

```text
TestVision_imageBlockNewlinesMatchTheRealProcessors is M-38's regression guard
(docs/audit-2026-09-10.md): the real HF processors (verified live 2026-09-16 against
transformers' processing_gemma3.py/processing_gemma4.py and Qwen2.5-VL-7B-Instruct's own
chat_template.json) wrap the image sequence in "\n\n" on both sides for Gemma 3, and with NO
adjacent newline at all for Gemma 4 and Qwen2.5-VL — the opposite of what this file used to
splice (a bare trailing "\n" on all three, or none on Gemma 3). This can't be driven through
the real prompt builders without a real vision tower, so it is asserted structurally on the
source: each family's block assignment must be exactly the multimodal block call, with nothing
concatenated onto it — in prepImage (S11, vision_multi.go), case by case, and in
glmOcrVisionPrompt — in each function's own AST, not a repo-wide grep, which could not tell one
family's assignment from another's.
```

## TestBanner_contextIsTheEnforcedWindow.gpu

Moved from `internal/serveapp/banner_test.go` (the comment above `TestBanner_contextIsTheEnforcedWindow.gpu`) on 2026-10-09.

```text
On cpu the resident cap does not exist, so window == MaxPositions and a factsOf that read
MaxPositions directly would pass the check above. The GPU case — where they differ, 8192 against
40960 on the 2070 — is held by the source: factsOf must take the window from contextWindow, and
the load path must hand the banner the -ctx THIS model resolved (a per-model ctx= override wins).
```

## TestFactsOf_aPlainCPULoadDeclinedNothing

Moved from `internal/serveapp/banner_test.go` (the comment above `TestFactsOf_aPlainCPULoadDeclinedNothing`) on 2026-10-09.

```text
A plain CPU load is not a decline: the CPU was the choice. decoder sets ResidentDecline for it too ("backend does not implement residency (… the CPU backend)"), so
the banner must not report that as a GPU load that fell through — found when a real gpt-oss CPU run printed "this model declined one".
```

## TestBanner_headTable

Moved from `internal/serveapp/banner_test.go` (the comment above `TestBanner_headTable`) on 2026-10-09.

```text
The head table's precision is on the banner (D of the --embed-int4 decision, 2026-10-08): the same checkpoint is a different model under a different table, and the default differs by backend.
Two halves: the line itself from the facts, and the facts from a real load with the flag both ways.
```

## TestHandleEmbeddings_dimensionsRequiresMatryoshka

Moved from `internal/serveapp/embeddings_test.go` (the comment above `TestHandleEmbeddings_dimensionsRequiresMatryoshka`) on 2026-10-09.

```text
TestHandleEmbeddings_dimensionsRequiresMatryoshka is the guard against a silent-wrong: honoring
`dimensions` for a model not trained with Matryoshka Representation Learning returns a
unit-length, entirely plausible vector that simply RETRIEVES WORSE. That is measured, not
theoretical — aikit's TestEmbedderCoverage_matryoshka shows multilingual-e5-base sliced to a
quarter width dropping paraphrase-pair recall 1.00 → 0.80, while genuine MRL models hold their
documented floor. Only two of aikit's eight certified embedders qualify.

Both directions are asserted, because a guard that only rejects is as broken as one that only
accepts: it would refuse the legitimate MRL truncation the parameter exists for.
```

## TestServe_anthropic_streamAbort

Moved from `internal/serveapp/anthropic_test.go` (the comment above `TestServe_anthropic_streamAbort`) on 2026-10-09.

```text
sessMu (held for the duration of a generation, same span the old lm.mu covered — J1,
task-work-queue-2026-09.md) must free promptly once the request context is cancelled.
```

## TestServe_anthropic_tools.replay

Moved from `internal/serveapp/anthropic_test.go` (the comment above `TestServe_anthropic_tools.replay`) on 2026-10-09.

```text
Replay the call + a result → expect a grounded text answer (end_turn). tool_choice "none" on the
replay: what this checks is the SERVER — the tool_use/tool_result history renders and the model can
answer from it. Under "auto" the next move is the model's own choice, and the 0.5B fixture re-calls
get_weather (its prompt still ends in "Use the tool."; the rendering matches Qwen's template, checked
2026-09-25), which failed this test for a reason no server change could fix.
```

## developerrole_test.go.header

Moved from `internal/serveapp/developerrole_test.go` (the comment above `developerrole_test.go.header`) on 2026-10-09.

```text
Gates for the `role: "developer"` → `role: "system"` alias (queue G12).

The before-state these replace, verified at dc8355e and pinned by a test shown
failing against the fix: "developer" matched no arm of messagesToTurns' switch
and fell through `default:` to a USER turn — not a 400, not a drop. A harness
that sends its system prompt as "developer" (OpenAI's newer APIs, and the
agent harnesses following them) had its entire scaffold delivered as the
user's first message, and the resulting behavior read as a bad model rather
than a mangled request. The equality gates below are what keep that silent
failure from returning: they assert the aliased form renders BYTE-IDENTICALLY
to the equivalent system message, so a future edit that half-recognizes the
role fails here rather than in someone's agent loop.
```

## TestAnthropicRejectsIllegalRoles

Moved from `internal/serveapp/developerrole_test.go` (the comment above `TestAnthropicRejectsIllegalRoles`) on 2026-10-09.

```text
TestAnthropicRejectsIllegalRoles is the G12 pin, FLIPPED by G13.

It used to assert that /v1/messages silently demoted a developer-role message
to a user turn — pinned deliberately, so the behavior was visible rather than
silent while the decision was pending. The decision came out the other way:
the Anthropic Messages API accepts only "user" and "assistant" and rejects
anything else, so demote-vs-alias was the wrong menu for this surface and
rejection is the faithful answer.

The class matters more than the instance. "developer" was never special here —
ANY typo'd or invented role was folded into the conversation, restructuring
what the model saw. Each case below is a shape that used to be swallowed.
```

## TestStreamOptions_includeUsageShape

Moved from `internal/serveapp/openai_stream_usage_test.go` (the comment above `TestStreamOptions_includeUsageShape`) on 2026-10-09.

```text
A streaming client has no token count unless the server sends one, and counting SSE chunks is not
a substitute: streamTokens emits a chunk only when `end > printed`, so a token held back for an
incomplete UTF-8 rune or a partial stop-string match produces NO chunk, and the token that
resolves the holdback produces one chunk carrying several tokens' bytes. Chunks <= tokens, always
in the same direction. bench_peer.py counted chunks and called them tokens.
```

## TestStreamSurfaces_allSendUsageBeforeDone

Moved from `internal/serveapp/openai_stream_usage_test.go` (the comment above `TestStreamSurfaces_allSendUsageBeforeDone`) on 2026-10-09.

```text
M-26: include_usage was honoured on the plain chat stream ONLY. The tool and vision streams
silently omitted the usage chunk, and /v1/completions did not even parse stream_options.
Agent harnesses declare tools on every turn and rely on that chunk for context accounting,
so the one surface that worked was the one they least often use.

THE ANTI-DRIFT GATE, in the shape this audit's "N of M sites" findings keep calling for:
count the sites rather than list them. Every streaming surface reaches sseDone at its normal
completion; each such path must send the usage chunk first, and they all now go through the
one sendUsage helper. Reading the AST means a fifth surface added later cannot slip past by
being formatted differently.
```

## TestStreamTokens_decodesAsAContinuation

Moved from `internal/serveapp/openai_stream_usage_test.go` (the comment above `TestStreamTokens_decodesAsAContinuation`) on 2026-10-09.

```text
M-25, AT THE CALL SITE. The tokenizer tests prove DecodePiece/DecodeContinuation keep the
leading space; they cannot prove the streaming loop calls one of them. Measured: reverting
streamTokens to Decode broke NO test — the component was correct and vouched for behaviour
the system did not produce, which is the trap CLAUDE.md names. So the assertion belongs here.

streamTokens is shared by chat, /v1/completions and vision, so all three carried the defect
the audit scoped to /v1/completions. The generated ids must never go through Decode: those
ids continue the prompt, and Decode applies SentencePiece's sequence-level dummy-prefix strip
to them — eating the response's leading space on Llama-2/Mistral.

R-08 (audit-2026-09-02 / docs/tasks/task-recompute-audit.md) replaced the per-token DecodeContinuation(ids)
re-decode of the WHOLE generated sequence (O(n^2) in output length) with DecodePiece(id)
appended incrementally. DecodePiece is the same non-stripping contract DecodeContinuation was
relied on for here — its own doc comment states it explicitly ("does NOT apply the
whole-sequence dummy-prefix strip... a caller printing piece-by-piece emitted
'Theanswerisfour'" if it did) — so this guard now requires DecodePiece instead of
DecodeContinuation, and forbids Decode exactly as before.
```

## bodycaps_routes_test.go.header

Moved from `internal/serveapp/bodycaps_routes_test.go` (the comment above `bodycaps_routes_test.go.header`) on 2026-10-09.

```text
Route-specific body-cap gates (v0.10.3 pre-tag review of the G1 default).

The derived body cap is a NEW DEFAULT that can reject requests which previously worked, so each
route's cap has to be justified by what that route actually carries. Two of the four checks had
real answers:

  - Multimodal is SAFE, and this pins why: the pre-tokenization guard measures only `type:"text"`
    content parts, so a multi-megabyte base64 image never counts against a context window it does
    not consume (an image is a few hundred tokens). The byte cap on the chat/messages routes is
    visionCap = textCap + 32 MiB, so the payload itself fits too. Both halves must hold — if
    either regresses, VL requests start failing on small-context models.

  - Embeddings was WRONG: /v1/embeddings is served by the encoder, which is not in s.models, so a
    cap derived from decoder context windows measured the wrong thing. On an embed-only server it
    collapsed to the 4 MiB text floor and rejected a batch the route's own bounds accept.
```

## TestAnthropicInputBytes_excludesImages

Moved from `internal/serveapp/bodycaps_routes_test.go` (the comment above `TestAnthropicInputBytes_excludesImages`) on 2026-10-09.

```text
TestAnthropicInputBytes_excludesImages is the /v1/messages half of the pre-tokenization guard.
The guard was added to the Anthropic route in v0.10.3 (it previously ran only on the OpenAI
routes, so a body under the cap still tokenized in full before rejection). It must bound the same
thing the OpenAI side bounds — tokenizable TEXT — or it becomes a new way to reject valid vision
requests on small-context models, trading one defect for a worse one.
```

## TestBestExtend_stopStringTokensReuse

Moved from `internal/serveapp/sessions_test.go` (the comment above `TestBestExtend_stopStringTokensReuse`) on 2026-10-09.

```text
TestBestExtend_stopStringTokensReuse is L-15's fix for the P-18 gate: a
session that ends with tokens generated AFTER a stop-string hit is committed
to the session's Tokens() (openai.go's streamTokens appends every generated id
to ids regardless of the stop cut, and Session.Generate's commit records the
whole thing), even though only the text up to the cut point ever reached the
client. The client's NEXT prompt is built from what it actually saw, so it
never contains those invisible tokens — the old whole-containment rule missed
on every such turn, forcing a cold prefill plus an eviction of a session that
was almost entirely reusable. bestExtend now picks it anyway (the decoder's
own rewindForReuse truncates to the shared prefix, discarding just the
invisible tail).
```

## TestPickSession_sparePreambleGoesFresh

Moved from `internal/serveapp/sessions_test.go` (the comment above `TestPickSession_sparePreambleGoesFresh`) on 2026-10-09.

```text
TestPickSession_sparePreambleGoesFresh is the bug MC0 found (docs/tasks/task-concurrency-2026-09.md, 2026-09-26):
with one resident session, bestExtend's floor (what a candidate shares with every OTHER session) is 0, so the shared
chat-template preamble alone qualified. Two interleaved conversations then took each other's single session, each
truncating the other to the 7-token preamble, and the LRU never grew past one session despite -kv-sessions 4. On the
CPU W7 workload that read prefill_reused_tokens = 7 on every turn at 2 clients, and 0.69x the 1-client aggregate.
```

## tinyServed

Moved from `internal/serveapp/prefillpath_test.go` (the comment above `tinyServed`) on 2026-10-09.

```text
The serve half of the prefill-path report.

WHY THIS EXISTS. Both fast paths a GPU backend advertises — the resident decode runner and the
batched prefill — fall back SILENTLY when a model doesn't qualify. `--backend cuda --quant
int8int8` loads clean, reports a GPU, decodes resident, and then takes one forward per prompt
token because the batched GEMV is int4-only: 9× TTFT on a 300-token prompt, discovered only by
timing it. serve therefore states the resolved paths at startup, publishes them on /v1/models, and
-require-backend turns a decline into a startup failure for clients that can't absorb a 9×.

The registry is built directly from decoder.Load rather than newServer: the committed tiny GGUF
carries no embedded tokenizer (serve's loadDecoder needs one), and neither the /v1/models handler
nor requireFastPaths touches the tokenizer. That keeps this gate in CI instead of behind a
GOINFER_SERVE_MODEL download.
```

## TestVersionReport_backendsLineIsDerived

Moved from `internal/serveapp/version_test.go` (the comment above `TestVersionReport_backendsLineIsDerived`) on 2026-10-09.

```text
R2 gate (docs/measurements/cold-user-2026-09-06.md, finding #3). On v0.16.0 there was no way
to ask a binary which backends it contained: `serve --version` printed
"flag provided but not defined: -version" and exited 2. The darwin asset linked no Metal
backend, and the only signal was a warning line that scrolled past the load banner.

Two halves. This one pins that the `backends:` line is DERIVED from the registry rather than
a hand-written string — a hard-coded "cpu cuda metal" would satisfy a text check while being
exactly the lie the finding is about.
```

## TestServeUnknownPositional_namesTheSubcommands

Moved from `internal/serveapp/version_test.go` (the comment above `TestServeUnknownPositional_namesTheSubcommands`) on 2026-10-09.

```text
R6's other half (docs/measurements/cold-user-2026-09-06-nobara-pc.md): an unrecognized
subcommand/positional fell through silently on v0.17.0 (flag.Args() went unchecked). The
mutation this guards against: removing the flag.Args() check makes this go red — the process
exits 0 with no error, instead of naming the argument as unrecognized.
```

## TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered

Moved from `internal/serveapp/version_test.go` (the comment above `TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered`) on 2026-10-09.

```text
TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered is R4's gate for the scenario-B friction
entry "--help is 13,583 bytes / 39 flags / 100 lines … I could not skim it for the flag I
needed" (docs/measurements/cold-user-2026-09-06.md).

That is not a style complaint. The same tester then drove a 16 GB machine +7.8 GB into swap
because they never found `-stream-weights`, whose own help text names their exact model and
their exact RAM. The flag was present and the page was too long to find it in.

So the assertion is not "the help is short" — the long text is deliberate and every paragraph in
it is a disclosure some measurement earned. It is that a MAP exists above the dump, that the map
names the flag whose absence did the damage, and that it comes FIRST.
```

## TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered.count

Moved from `internal/serveapp/version_test.go` (the comment above `TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered.count`) on 2026-10-09.

```text
The flag count must be DERIVED. A hand-typed count is the same defect class as the parity
manifest's hand-typed aikit_version, which sat seventeen versions stale.
Count only the DUMP's entries. The header's own "  --model" lines start the same way, so
counting the whole page conflates the map with the territory — which this test did on its
first run, reporting 47 against a real 40.
```

## askAboutImage

Moved from `internal/serveapp/gemma4_vl_real_test.go` (the comment above `askAboutImage`) on 2026-10-09.

```text
askAboutImage drives the REAL serving-side vision wiring end to end on real weights:
lm.imagesPrompt (real tower forward, real preprocessing) -> lm.prepare -> lm.driveVL
(real GenerateGemma4VL). This is the first-ever real run of this exact seam — every prior
gemma4 VL gate this session called decoder-level functions directly with precomputed
image_features, never imagesPrompt's own real-tower invocation (docs/multimodal.md's
Phase B "HTTP-level integration smoke test" gap).
```

## TestGemma4VLReal_26BA4B

Moved from `internal/serveapp/gemma4_vl_real_test.go` (the comment above `TestGemma4VLReal_26BA4B`) on 2026-10-09.

```text
TestGemma4VLReal_26BA4B is the real-checkpoint end-to-end gate for the 26B-A4B/31B-class
(use_bidirectional_attention: "vision", batched blockwise prefill, Phase C) path — the
first real-checkpoint proof of the batched forward, not just the tiny/scaled synthetic
fixtures. 26B-A4B carries NO PLE (hidden_size_per_layer_input=0, confirmed directly against
the real config — corrects docs/multimodal.md's earlier, over-generalized claim that every
real vision-capable checkpoint has PLE), so this loads straight off safetensors, no GGUF
needed.
```

## newToolsTestServer

Moved from `internal/serveapp/heartbeat_test.go` (the comment above `newToolsTestServer`) on 2026-10-09.

```text
Gates for G19 — SSE heartbeats while the tool path buffers.

The before-state: `stream: true` with tools declared sent ZERO bytes until the
whole generation finished (measured 1682.6s to first byte against a client
whose idle timeout was 300s). The buffering is correct — a tool call can only
be parsed from the complete output — so the fix keeps the buffer and adds
content-free comment frames, which every SSE parser drops.
```

## realGptOssReply

Moved from `internal/serveapp/harmony_test.go` (the comment above `realGptOssReply`) on 2026-10-09.

```text
A real gpt-oss-20b reply (goinfer-chat, temperature 0, 2026-09-30; the same capture chat/harmony_parse_test.go pins) through
serve's own reasoning router — newThinkOut is the object streamTokens holds for every route — cut into token-sized pieces the
way a decode loop delivers it. Before the Harmony parser every marker reached `content`; here the client-facing streams must
carry the analysis as reasoning and the final channel as the answer, and nothing else.
```

## TestServe_haltUnderLoad

Moved from `internal/serveapp/halt_test.go` (the comment above `TestServe_haltUnderLoad`) on 2026-10-09.

```text
K2 gate (docs/tasks/task-halt-2026-09.md): 32 concurrent generations at -max-inflight 32
(saturated), halt, assert every one of them stopped because of the halt (not a natural
completion), and measure time-to-quiescence. Then resume and assert a request succeeds.

"Every stream ended cancelled" (the doc's own words) needs one honest caveat, found while
building this test, not assumed going in: goinfer serializes ALL generations for one model
behind a single mutex (loadedModel.mu, "the single decode worker" — openai.go's own doc
comment). With -max-inflight 32 admitting all 32 requests, only ONE of them is ever actually
registered in K1's cancel registry and streaming at a time; the other 31 are blocked
acquiring that mutex, which is not context-aware and would otherwise run to natural
completion one at a time regardless of a halt (a real bug this task's own K2 work found and
fixed in tryEnter/enter — see halt.go and openai.go's doc comments on both). The fixed
behavior: a request already streaming when halt lands gets finish_reason "cancelled"; a
request still queued behind the mutex gets refused with a 503 {"error":"halted"} the moment
it would otherwise have started, WITHOUT ever running a real generation. Both are "the halt
switch worked" outcomes; neither is "ran to completion after being told to stop". This test
asserts exactly that split, not a literal 32/32 "cancelled".

Gated on GOINFER_SERVE_MODEL like this package's other real-model tests.
```

## TestServe_haltUnderLoad.admitted

Moved from `internal/serveapp/halt_test.go` (the comment above `TestServe_haltUnderLoad.admitted`) on 2026-10-09.

```text
admitted: Do() blocks until the server sends response headers, which for a request
still queued behind the model's single decode mutex does not happen until its turn
comes up. Waiting on Do() to return here (as an earlier version of this test did)
self-deadlocks — nothing frees that mutex until the halt this goroutine is itself
blocking the issuing of.
```

## TestServe_qwen35Image_G4

Moved from `internal/serveapp/qwen35_vision_real_test.go` (the comment above `TestServe_qwen35Image_G4`) on 2026-10-09.

```text
	    HF's 32 greedy tokens over those ids, decoded, for each of the 3 images. Before 2026-09-30 serve
	    rendered nothing there (goinfer's generic ChatML renderer) and this test pinned a named 4-token
	    delta; `-thinking asis` still renders those old bytes, and (a1) keeps checking that (serve_{A,B,C}.json);
```

## fuzz_test.go.header

Moved from `internal/serveapp/fuzz_test.go` (the comment above `fuzz_test.go.header`) on 2026-10-09.

```text
Track 2.6 (testing campaign): the request JSON for /v1/chat/completions and
/v1/responses is attacker-supplied. These targets fuzz the model-free request
shaping — decode + sampling/stop/content/message translation + the
response_format → grammar dispatch — which must never panic. (The grammar
COMPILE itself is fuzzed harder in constrain; the end-to-end handler path with
a loaded model is the maintainer-only soak in Track 5.)
```

## TestSetupGlmOcrVision_pixelCap

Moved from `internal/serveapp/glm_ocr_vision_test.go` (the comment above `TestSetupGlmOcrVision_pixelCap`) on 2026-10-09.

```text
TestSetupGlmOcrVision_pixelCap: -vision-max-pixels (O4, owner decision 2026-10-04: the default stays the model's own 4.82 MP ceiling, the flag only lowers it).
It is tested through setupGlmOcrVision, the one function serve's loader calls: a cap below the ceiling lowers MaxPixels and nothing else; 0 leaves the ceiling; a cap
ABOVE the ceiling does not raise it (the tower and the context were gated for 6,144 image tokens, no more); a cap below the model's own floor is refused by name.
```

## TestTowerReserve_everyDeviceTower

Moved from `internal/serveapp/siglip_f32_default_test.go` (the comment above `TestTowerReserve_everyDeviceTower`) on 2026-10-09.

```text
TestTowerReserve_everyDeviceTower is S18's G-S18c (docs/tasks/task-multimodal-support-2026-10.md): towerReserve is nonzero for every family that has a DEVICE tower under its backend, int8 included, and
zero where the tower runs on the CPU. A table over family x -vision-quant x backend. Red on the code before S18 for Gemma 3 under cuda, where the shipped default (the int8 device tower) was priced at zero.
```

## TestResolveGemma3VisionQuant.sidecar

Moved from `internal/serveapp/gemma3_tower_test.go` (the comment above `TestResolveGemma3VisionQuant.sidecar`) on 2026-10-09.

```text
The sidecar it asks about is the one a Metal load reads: int4 with the plain head (embed-int4 resolves off on Metal
unless asked for). It asked for the ".e4h" one before, which a Metal load never builds or reads (found 2026-10-08
by G-S18a: the default arm transcoded "gemma-3-4b-it.int4.metal.giw" beside a fresh ".e4h" one).
```

## TestPrepare_prefillShare

Moved from `internal/serveapp/prefill_share_testhooks_test.go` (the comment above `TestPrepare_prefillShare`) on 2026-10-09.

```text
Before the fix the resident request was split four ways too: on the 7B, MC3 cells lost ~1000-token prompts to 413
(docs/measurements/spec-vs-batching-metal-2026-09-27.md §5).
```

## TestServe_admin

Moved from `internal/serveapp/admin_test.go` (the comment above `TestServe_admin`) on 2026-10-09.

```text
TestServe_admin is the Inc3 gate: --allow-admin off → 403; with it on, a
load → generate → unload → 404 → reload cycle, and unload-while-busy → 409.
Gated on GOINFER_SERVE_MODEL (a .gguf).
```

## TestServe_admin.unload

Moved from `internal/serveapp/admin_test.go` (the comment above `TestServe_admin.unload`) on 2026-10-09.

```text
unload an idle model → 200 freed (the drain completes at once; native memory released). The
old contract was "busy → 409"; unload now DRAINS instead of refusing — the in-flight-vs-unload
race is the drain regression test's job (TestUnloadDrain_*), not this cycle test.
```

## TestPlanGridTower

Moved from `internal/serveapp/grid_tower_plan_test.go` (the comment above `TestPlanGridTower`) on 2026-10-09.

```text
TestPlanGridTower pins where a Qwen3.5+ or GLM-OCR tower runs (S2 of docs/tasks/task-multimodal-support-2026-10.md):
the device tower only under --backend metal or cuda, float32 and registered for THAT backend; the CPU otherwise with the reason; and each CPU
fallback a refusal under -require-backend. (S4 made the rules backend-generic; until then CUDA was pinned to the CPU.)
```

## requireHeavyModel

Moved from `internal/serveapp/heavytest_test.go` (the comment above `requireHeavyModel`) on 2026-10-09.

```text
requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an explicit
env opt-in — the package-local twin of decoder/heavytest_test.go's helper, same GOINFER_HEAVY_TESTS
key, so `GOINFER_HEAVY_TESTS=1 go test ./...` runs every package's heavy tests uniformly.

The bug this closes: the serve integration tests decided whether to run by PATH EXISTENCE alone
(os.ExpandEnv("$HOME/models/...") → skip if absent, else boot a server on it). On a box with the
model zoo present, `go test ./cmd/serve/` fired them all opportunistically. The asset happening to
be on disk is not a request to run a multi-GB test. The per-test os.Stat skip stays as a second
guard, so opting in on a bare box is still harmless.
```

## TestPrefillFlags_reachTheDecoderThroughOptions

Moved from `internal/serveapp/exactprefill_test.go` (the comment above `TestPrefillFlags_reachTheDecoderThroughOptions`) on 2026-10-09.

```text
The prompt-ingestion flags reach each model through its decoder.Options, never the process environment
(phase 5, docs/tasks/task-env-config-2026-09.md — they used to be applied by applyExactPrefillEnv's
os.Setenv, process-wide). M-48 still holds: --exact-prefill covers all three backends, now because
Options.ExactPrefill is consulted by each (CPU, CUDA resident build, Metal resident) per model.
```

## TestMoEPagerFlag_reachesTheDecoderThroughOptions

Moved from `internal/serveapp/exactprefill_test.go` (the comment above `TestMoEPagerFlag_reachesTheDecoderThroughOptions`) on 2026-10-09.

```text
TestMoEPagerFlag_reachesTheDecoderThroughOptions: --moe-pager is carried to decoder.Load in
Options.MoEPager (it used to be applied by setting GOINFER_MOE_PREAD_CPU, process-wide).
```

## TestLoadVisionTower_siglipHonoursVisionDevice

Moved from `internal/serveapp/vision_device_siglip_test.go` (the comment above `TestLoadVisionTower_siglipHonoursVisionDevice`) on 2026-10-09.

```text
TestLoadVisionTower_siglipHonoursVisionDevice: -vision-device cpu keeps Gemma 3's SigLIP tower on the CPU, f32, under a
device backend. It used to follow --backend alone: under cuda (and webgpu) the tower loaded int8 and the device tower
attached whatever the flag said, so G-S3c's "tower on the CPU in every arm" did not hold on CUDA.
```

## TestDeviceFallback_requireBackendRefuses

Moved from `internal/serveapp/device_fallback_test.go` (the comment above `TestDeviceFallback_requireBackendRefuses`) on 2026-10-09.

```text
Under -require-backend a device tower's memory failure fails the request: no CPU run, the accelerator kept (S4 audit, 2026-10-07: the fallback used to run the
CPU tower even when the user had asked never to leave the backend).
```

## TestForcedTool_loneToolConvenienceEndsAtTheToolResult

Moved from `internal/serveapp/lone_tool_livelock_test.go` (the comment above `TestForcedTool_loneToolConvenienceEndsAtTheToolResult`) on 2026-10-09.

```text
PRE-REGISTERED (docs/queue-correctness.md G39). `serve check`'s `tools, OpenAI` row failed with "turn two asked for the tool
again instead of answering — the agent-livelock shape" on BOTH Qwen2.5-Coder-7B and Qwen2.5-7B-Instruct, and the same turn
with a second tool in the request, or with the identical prompt sent as a raw completion, got a normal answer. Cause: with
exactly ONE tool and tool_choice auto, forcedTool constrained the reply to that tool's call from its first token on EVERY turn,
including the one after the tool's result, so a client with a single tool could never get an answer.

The rule: the lone-tool convenience is for the model's FIRST call. It is lifted when the conversation already ends in a tool
result and tool_choice is auto (or absent). An explicit "required", a named function, and "none" mean what they always meant.
```

## TestServe_generationRoutesExistWithoutAStartupModel

Moved from `internal/serveapp/routes_nomodel_test.go` (the comment above `TestServe_generationRoutesExistWithoutAStartupModel`) on 2026-10-09.

```text
TestServe_generationRoutesExistWithoutAStartupModel: a server started with only --web (or
--allow-admin / --admin-socket) loads its model later, but the mux is built once, and the
generation routes used to be registered only when a model existed at startup. That server had no
/v1/chat/completions and no /v1/jobs for its whole life — the web UI's own chat got the mux's bare
"404 page not found" even after loading a model through the UI. Driven through the real binary,
since the mux is built in Main: with no model loaded, a chat request must reach the handler and get
its JSON "model not found", not the mux's plain-text 404.
```

## TestDriveUsesRequestContext

Moved from `internal/serveapp/reqctx_test.go` (the comment above `TestDriveUsesRequestContext`) on 2026-10-09.

```text
TestDriveUsesRequestContext gates M6 (and the whole class it belongs to): every
lm.drive / lm.driveVL call in a request path must pass a request-derived context,
never context.Background(). A detached context means a disconnected client can't
cancel the decode — the model runs to max_output_tokens holding its turn, lm.sessMu,
and a queue slot, and retries amplify into a DoS. responses.go's tool path was the
one caller that got this wrong; this scans the whole package so a future one can't regress.
```

## TestWriteJSON_unencodableBodyIsAServerError

Moved from `internal/serveapp/writejson_test.go` (the comment above `TestWriteJSON_unencodableBodyIsAServerError`) on 2026-10-09.

```text
writeJSON used to send the status line and then stream the encoder's output with its error dropped, so a body JSON cannot carry (NaN, an infinity) reached the client as
HTTP 200 with an EMPTY body: S6's served check on the 35B read "no answer" for a night (2026-10-09). An unencodable body must be a 500 that names the problem, and an ordinary body
must be byte-for-byte what it was (200, the JSON and its trailing newline).
```

## TestEnableResidentTower

Moved from `internal/serveapp/vision_resident_fallback_test.go` (the comment above `TestEnableResidentTower`) on 2026-10-09.

```text
A resident GPU vision tower that cannot be attached (no VRAM left for it, a build without the backend) used to abort serve
startup: loadVisionTower returned the EnableResident error and the model already loaded on the GPU was thrown away. The attach
is now a warning, and the tower runs on the CPU path EnableResident leaves intact.
```

## TestTowerInt8

Moved from `internal/serveapp/vision_tower_int8_test.go` (the comment above `TestTowerInt8`) on 2026-10-09.

```text
Which towers load int8, for every vision family serve dispatches on (loadVisionTower: "" and "gemma3" reach the SigLIP path)
on every backend, unset and with -vision-quant f32 or int8. Only Gemma 3's SigLIP tower gets int8 unasked, and only under webgpu, whose
device tower needs it: on CUDA it is float32 since 2026-10-08 (owner). Every other family stays float32 unless asked: its gates ran
float32, and int8 measured lossy and not faster (see towerInt8). qwen3_vl is the case this table was written for: it was missing from the float32 list.
```

## TestWithFeatureCache_recoversAnyTower

Moved from `internal/serveapp/recover_device_tower_test.go` (the comment above `TestWithFeatureCache_recoversAnyTower`) on 2026-10-09.

```text
Every family's tower runs through withFeatureCache, so a tower that panics there (Gemma 3's SigLIP, Gemma 4's image or audio tower, on any device) fails the
request with an error, not the server. Before 2026-10-07 only Qwen2.5-VL's forward was recovered.
```

## TestSetConcurrency_reportsTheDecidedValue.prior

Moved from `internal/serveapp/concurrent_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestSetConcurrency_reportsTheDecidedValue: the concurrency line serve prints comes from setConcurrency itself, after
it has decided — not from the load banner, which runs before (every model's banner once said "one generation at a
time" whatever -max-concurrent was, because it read the value before setConcurrency set it).
```

## TestBanner_sessionReuseNamesMetalOnlyOnMetal.prior

Moved from `internal/serveapp/banner_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
The session-reuse line used to say "Metal keeps N by default" for a CUDA card's granted count. It names Metal only for a Metal decode path.
```

## TestServe_prepareEnforcesResidentCapForAdapter.prior

Moved from `internal/serveapp/residentpath_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestServe_prepareEnforcesResidentCapForAdapter reproduces M-01's exact failure scenario end to
end: an adapter model whose resident backend has a smaller context cap than MaxPositions must
have prepare enforce the SMALLER (resident) cap, matching what generateInto's useGPU condition
actually admits for an adapter's first turn — not the old behavior, which enforced the uncapped
MaxPositions and let a resident prefill past the cap die mid-request instead of being rejected
cleanly up front.
```

## TestServe_prepareEnforcesResidentCapForAdapter.reject.prior

Moved from `internal/serveapp/residentpath_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
A prompt past the resident cap, but still under MaxPositions, must now be rejected cleanly —
the old `lm.adapter == ""` predicate let this through to die mid-prefill instead (M-01).
```

## TestMaybeStore_preservesToolCallsForPreviousResponseID.prior

Moved from `internal/serveapp/responses_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestMaybeStore_preservesToolCallsForPreviousResponseID pins V-18 (docs/review-2026-09-04.md):
maybeStore used to store only the lead text (often empty, when the model went straight into a
tool call), dropping the tool calls entirely. serveResponsesWith appends a stored entry's
messages VERBATIM onto the next request's input (see the previous_response_id branch), so a
later request's own function_call_output — correctly decoded into a {Role:"tool"} turn by
M-18's fix — would answer a call that, as far as the reconstructed conversation shows, was
never made: user → assistant("") → tool(result). No model or HTTP server needed — maybeStore
and the store it writes to are plain Go values.
```

## TestRespondTools_callsMaybeStoreWithTheToolCalls.prior

Moved from `internal/serveapp/responses_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestRespondTools_callsMaybeStoreWithTheToolCalls is the wiring guard: the unit test above
proves maybeStore preserves tool calls when GIVEN them, not that respondTools actually passes
them — the exact shape of gap this session's audit keeps finding (a helper with a test, and a
call site nobody checked).
```

## TestResponses_toolLoopItemsSurviveDecoding.prior

Moved from `internal/serveapp/responses_tools_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
M-18: A RESPONSES TOOL LOOP COULD NOT COMPLETE.

`function_call` and `function_call_output` carry no `role` and no `content`, so decoding items as
{role, content} turned both into `{Role:"user", Content:""}` — two empty user turns. The model
never saw the tool result, so it answered without it or re-called the same tool forever, under
HTTP 200. docs/server.md and responses.go both claim the round-trip; TestServe_responses step 4
never feeds a result back, which is exactly why nothing caught it.

This is the step-4 that was missing, at the decode layer where the loss happened.
```

## TestStreamTokens_thinkStopStrings.prior

Moved from `internal/serveapp/think_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
Stop strings are matched against the ANSWER only. A stop string is a request about what the model says to the client; the
reasoning is scratch work, and one that appears in it used to end the reply there with no answer at all. (Not so in
deepseek-legacy, where `content` is the raw text, tags included — what the client sees is what the stop logic watches.)
```

## TestStreamTokens_thinkReasoningNeverSplitsARune.prior

Moved from `internal/serveapp/think_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
Reasoning no longer rides through the stop logic, which used to hold a partial UTF-8 rune back for it. A multi-byte character
split across two tokens inside the reasoning must still reach the client whole — every reported fragment valid UTF-8.
```

## TestHandleChat_imageStreamLogprobsRejected.prior

Moved from `internal/serveapp/vision_logprobs_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
An image request used to accept logprobs:true and answer 200 with no logprobs at all: driveVL threw them away. They are now
returned on the buffered reply (TestGemma4VLReal_E2B_logprobs is the through-the-model proof), and the streamed vision reply,
which has no logprobs field, refuses the combination the way the text route does rather than dropping them again.
```

## TestTowerInt8_gemma3ExplicitF32.prior

Moved from `internal/serveapp/siglip_f32_default_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
The S4 addendum (docs/tasks/task-multimodal-support-2026-10.md): -vision-quant is unset by default so an explicit f32 is distinguishable from the default (which matters for the
fallback, attachGemma3Tower). Unset is float32 on CUDA since 2026-10-08 (owner) and int8 on WebGPU, which has no float32 device tower; an explicit f32 selects the float32 tower;
int8 stays int8 everywhere.
```

## TestTowerReserve_gemma3.prior

Moved from `internal/serveapp/siglip_f32_default_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
Gemma 3's SigLIP tower is priced into the resident plan on CUDA: the default (float32 since 2026-10-08) and an explicit f32 reserve about 2 GB (27 layers of hidden 1152 / intermediate
4304 in float32 plus the 4096-patch scratch; measured at about 1.7 GiB, so the figure is generous), an explicit int8 about 558 MiB.
```

## TestTowerReserve_everyDeviceTower.metal.prior

Moved from `internal/serveapp/siglip_f32_default_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
G-S18c-Mac: Gemma 3's Metal figures within 25% of the phys_footprint its towers measured (metal TestS18TowerHostMemory, 2026-10-08)
```

## TestWebUI_pageIsSelfContained.prior

Moved from `internal/serveapp/webui_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestWebUI_pageIsSelfContained guards the offline property: the UI of an engine that runs
offline must not need the network to RENDER. A CDN <script>/<link> would break that
silently — the page would still look fine on the machine that added it.

This does NOT forbid an <a href="https://…"> — an out-bound link the user may click (the
AmbientCSS restyle, docs/completed/task-web-ui-ambient.md, added one to the published book)
does not cost the page anything at render time; only an asset the page's own load depends on
does.
A blanket "no http(s):// substring anywhere" check would have banned that link too, which is
```

## TestWebUI_pageIsSelfContained.embedded.prior

Moved from `internal/serveapp/webui_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
EVERY EMBEDDED FILE, not just index.html. The page is split across webui/ (§6.1 of
docs/tasks/task-web-ui-2026-09.md); a check that still read only index.html would pass while
a CDN import sat in ui/app.js — narrowing silently exactly when the page grew.
```

## TestWebUI_pageIsSelfContained.localcss.prior

Moved from `internal/serveapp/webui_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
A LOCAL stylesheet link (href="ui/...") is now expected; an external one is still banned.
```

## TestWebUI_pageIsSelfContained.external.prior

Moved from `internal/serveapp/webui_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
The two allowed external references, neither a render-time asset: an out-bound link to
the book, and a plain-text attribution comment naming the vendored CSS's source (never
fetched — browsers strip CSS comments). Both pinned exactly rather than left as
"anything goes" — a DIFFERENT http(s) reference slipping in later (a tracking pixel, a
font @import, a fetch to an analytics host) is still exactly the kind of silent
offline-break this test exists to catch.
```

## TestWebUI_appGateInBrowser.prior

Moved from `internal/serveapp/webui_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestWebUI_appGateInBrowser runs scripts/webui_app_gate.mjs: the SHIPPED index.html driven through its own
send() with a fake SSE stream — W1's streamed rendering end to end, and W2's Copy on both clipboard
paths (async API, and the textarea fallback an insecure http://<lan-ip> page needs), on a stopped
answer, and after a re-render rebuilds every code-block button.
```

## TestSameOrigin_refusesForeignOriginAllowsMatchingOrNone.prior

Moved from `internal/serveapp/webui_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestSameOrigin_refusesForeignOriginAllowsMatchingOrNone pins V-20 (docs/review-2026-09-04.md):
on the key-free loopback default, auth() alone is a no-op (requireAuth returns h unchanged
when key==""), so list/pull had NO protection against a cross-origin POST — any page open in
the same browser could drive a caller-named multi-GB download onto the user's disk. Mirrors
N-26's identical sameOrigin in demo/agent/cmd/agent-web/main.go.
```

## TestLoadDecoder_clefRefusesInt4.prior

Moved from `internal/serveapp/systemone_clef_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
A Clef model is not served at int4 (owner decision 2026-10-03, D13). The refusal is tested through loadDecoder, the one caller that matters: a refusal function
that nothing calls would pass its own unit test. It must fire for an explicit quant=int4, name the measured figures, and not fire for the decision-model default
(int8int8) or for f32. The tiny directory carries only a head, so a load that gets past the quant check fails later for a different reason, which is what is asserted.
```

## TestModelSpec_headDefaultsToTheDecisionQuant.prior

Moved from `internal/serveapp/modelspec_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
A model with a decision head (head=) loads at decoder.DecisionHeadQuant unless a quant was chosen, through the same
options() path serve uses for every --model entry (D6b, owner decision 2026-10-01).
```

## TestEmbeddings_taskPromptChoice.prior

Moved from `internal/serveapp/embeddinggemma2_embedder_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestEmbeddings_taskPromptChoice is Gate 3's prompt rule (docs/tasks/task-embeddinggemma2.md, owner decision
2026-10-06): with no task and no input_type the model gets no prompt (sentence-transformers' own default); input_type
maps onto the model's query and document prompts; task names any of its prompts, and "none" means none. The prompt
applied is echoed in goinfer_task and the X-Goinfer-Embedding-Task header.
```

## TestSSEJSONSender_cancelsOnWriteFailure.prior

Moved from `internal/serveapp/sse_writer_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestSSEJSONSender_cancelsOnWriteFailure is N-23's other half (docs/audit-2026-09-10.md):
sseWriter.frame's write deadline (proven above) stops a stalled write from blocking forever,
but handleWebPull (webui.go) still needs THAT failure to actually stop the pull — otherwise
the goroutine returns from send but pull.Download keeps running against a dead client, still
pinning the single-flight pullState until the download finishes or errors on its own. This
pins the wiring sseJSONSender exists for: a failed frame must call the CancelFunc, the same
way the r.Context() cancellation already stops Download for an outright closed tab.
```

## TestReqID_isNotGuessable.property.prior

Moved from `internal/serveapp/limits_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
THE PROPERTY THAT WAS BROKEN: consecutive ids must not be consecutive integers. Parsed as
big-endian hex, a counter gives a difference of exactly 1 every time.
```

## TestStartDrain_savesSessionsUnderTheLRULock.prior

Moved from `internal/serveapp/limits_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
N-20: demoteLoop mutates the session LRU on its own ticker while the unload drain saved it
without holding lm.sessMu. The drain waits out in-flight REQUESTS, which is a different thing —
the idle-demote goroutine is not a request and holds no ml.rw. sessions.go documents the LRU
as not goroutine-safe, so this is a concurrently-mutated map: a process crash, not a wrong
answer.

Asserted on the source. Reproducing it needs -kv-idle-demote and -session-dir and an unload
landing inside a tick, and a test that raced for it would be flaky in the direction that
reports success.
```

## TestAdminLoad_requestQuantIsTheExplicitOne.callsite.prior

Moved from `internal/serveapp/limits_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
And the handler must actually do that resolution — the table above is the RULE, this is
the call site. Same gap that made M-25's component test vouch for nothing.
```

## TestGuardComposition_hugeToolSchemaTripsTheBudgetAloneOnATinyMessage.prior

Moved from `internal/serveapp/limits_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestGuardComposition_hugeToolSchemaTripsTheBudgetAloneOnATinyMessage matches the audit's own
worst-case example (a tiny message, a huge schema): composed the exact way
serveChatToolsWith's real guard call does (chatInputBytes(msgs) + toolSchemaBytes(tools)), the
combined byte count must trip a budget the message text alone would pass. This is NOT a call
through the real HTTP handler — promptTooLargeForContext short-circuits to nil without a real
*decoder.Model, which a unit test here does not load — so it instead proves the composition
this session's fix sites now share is correct, leaving the direct per-function unit tests above
to prove each addend's own arithmetic.
```

## TestCloseEntryNatives_dropsGemma4Enc.prior

Moved from `internal/serveapp/liveness_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestCloseEntryNatives_dropsGemma4Enc is N-32 (docs/audit-2026-09-10.md): closeEntryNatives
nils venc/qwenEnc/vproj but, until this fix, never touched gemma4Enc at all — so unloading a
served Gemma 4 vision model left its encoder (weights, no native Close of its own — same
shape as vproj) referenced from the closed entry, retaining that memory past unload.
gemma4Enc has no Close method (unlike venc/qwenEnc), so this only needs to observe the
reference is dropped, not that anything was called on it.
```

## TestFitFlag_realBinaryAcceptsOff.prior

Moved from `internal/serveapp/fitflag_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestFitFlag_realBinaryAcceptsOff proves the fix through the FULL registered flag.CommandLine,
not just cliutil.OnOff.Set in isolation — --version exits before touching a model, so this is a
cheap way to prove "--fit=off" (the spelling tasks/task-fit-to-hardware.md and this flag's own help
promise) parses cleanly end to end, where the original flag.BoolVar exited 2 with "invalid
boolean value \"off\"".
```

## TestGemma4VLReal_E2B_logprobs.prior

Moved from `internal/serveapp/gemma4_vl_real_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestGemma4VLReal_E2B_logprobs: an image turn's driveVL returns the per-token logprobs the request asked for (they were
discarded before, so an image request with logprobs:true answered 200 with none). One entry per generated token, each with
the asked-for number of alternatives, the chosen token's logprob <= 0 and (greedy) no better than the top alternative,
and the OpenAI-shaped rendering carries the same count.
```

## TestPrepare_temperatureValidation.prior

Moved from `internal/serveapp/validation_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestPrepare_temperatureValidation gates G4: temperature has the same lower bound as top_p.
A negative temperature is rejected (previously it was accepted and decoded greedily, since
SampleWithInfo treats Temperature <= 0 as argmax — a validation inconsistency, not inverted
output). 0 (greedy) and positive values stay valid.
```

## TestPick_modelValidation.prior

Moved from `internal/serveapp/validation_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestPick_modelValidation gates G6: an unknown NON-EMPTY model name is rejected on both a
single-model and a multi-model server, while an omitted name still routes on a single-model
server. Previously a single-model server served any name (confident wrong-model output).
```

## pickTest.prior

Moved from `internal/serveapp/validation_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
pickTest is a test-only wrapper for the request-path lookup (lookupLocked requires regMu). It
exists because pick was removed in favor of withModel; the resolution logic under test (exact
match + single-model fallback, G6) is unchanged.
```

## TestConstrainToolUnion_wiring.prior

Moved from `internal/serveapp/openai_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestConstrainToolUnion_wiring pins T1-T3's routing decisions (docs/tasks/task-tool-grammar-union-2026-09.md):
2+ tools under auto arm the union LAZILY (a LogitProcessor, never the fused-spec masker, which
assumes a grammar live from token 1); required forces it from token 1 (masker set, so fused
spec applies); a family with no opener (llama3) stays unconstrained under auto but is still
constrained under required; GOINFER_TOOL_UNION=0 restores the pre-T1 behaviour exactly.
```

## TestConstrainToolUnion_wiring.spec.prior

Moved from `internal/serveapp/openai_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
a speculative server keeps its pre-T1 auto behaviour (its drafter), but still gets required
```

## TestSplitter_gemmaToolResponsePromptIsForcedOpen.prior

Moved from `internal/serveapp/history_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
A reply to a Gemma 4 thinking-on prompt that ends after a tool response starts INSIDE an open thought channel, so the
splitter must start forced-open or the reasoning leaks into content (found by the history goldens).
```

## chaos_test.go.header.prior

Moved from `internal/serveapp/chaos_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
Track 5 (testing campaign): the untested interaction is multi-model + admin +
sessions under concurrency. These tests drive real small GGUFs (env-gated, so
CI skips) and run under -race. Bars: zero races, zero goroutine leaks, every
response a valid OpenAI shape or the correct status, and a --session-dir
restart restores warm KV without corrupting the continuation.
```

## TestServe_tieredKVDemoteFaultBack.prior

Moved from `internal/serveapp/chaos_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestServe_tieredKVDemoteFaultBack exercises idea #8 (tiered KV): a warm session
that goes idle is demoted to disk (RAM freed), and the next continuation faults
it back transparently — producing a continuation byte-identical to a cold
prefill. The LRU's clock is injected so idle demotion is deterministic.
```

## TestBanner_sessionReuseMatchesTheDecodePath.prior

Moved from `internal/serveapp/banner_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestBanner_sessionReuseMatchesTheDecodePath is G6's core assertion, and the one worth
having: the banner may claim prefix reuse ONLY when the server would actually use it.

decoder.Generate engages the resident DecodeRunner only when there is no session commit and
no prefix reuse — the resident KV lives on the GPU while a session's prefix cache is
CPU-side, and the two cannot both be the source of truth (loadedModel.drive). So a resident
model is stateless and re-prefills every turn, whatever --kv-sessions says. A banner that
reported "session reuse: on" there would be telling an agent-loop author the opposite of
what their latency will do.
```

## TestServe_multiAdapter.prior

Moved from `internal/serveapp/adapter_test.go` (the comment at the start of this block, before the diet) on 2026-10-09.

```text
TestServe_multiAdapter is the end-to-end #7 proof: one synthetic safetensors base
serves two compute-time LoRA adapters. It asserts the adapters SHARE the base's
resident decoder.Model (the RAM win), route by served name, and produce distinct
greedy outputs (base vs each adapter, and the two adapters vs each other) — i.e.
the per-session adapter binding is live and isolated.
```
