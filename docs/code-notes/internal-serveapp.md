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
