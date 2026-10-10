# internal/serveapp: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/serveapp`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## maxTopLogprobs

Moved from `internal/serveapp/openai.go` (the comment above `maxTopLogprobs`) on 2026-10-10.

```text
maxTopLogprobs is the largest `top_logprobs` a request may ask for — OpenAI's own documented
ceiling, so no compatible client is refused by it. Unbounded, one request retains
max_tokens x top_logprobs entries and materializes a map per entry before writing a byte
(audit-2026-09-02 C-08).
```

## maxOutputTokensCeiling

Moved from `internal/serveapp/openai.go` (the comment above `maxOutputTokensCeiling`) on 2026-10-10.

```text
maxOutputTokensCeiling is a hard upper bound on a request's max_tokens. KV is
preallocated as len(prompt)+max_tokens per layer, so an unbounded value (e.g.
{"max_tokens": 2000000000}) triggers a fatal, unrecoverable Go "out of memory"
throw that kills the server for every client (audit C-18). A request above this
is rejected 400 rather than clamped, so the caller learns its request was too big.
```

## loadedModel.blockSpec

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.blockSpec`) on 2026-10-10.

```text
blockSpec is an attached pretrained block drafter (--drafter), nil when unused. Attached
ONCE at load: the weight upload is a per-process cost, and doing it per request measured
0.17x — a 6x loss — with the loop itself perfectly healthy (docs/spec/08).
```

## loadedModel.turns

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.turns`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.maxTokBytes`) on 2026-10-10.

```text
maxTokBytes is the byte length of the longest token in the vocab, computed once. It
bounds tokenization cost (G1): a servable prompt is ≤ ctx tokens, so its text is ≤
ctx·maxTokBytes bytes; any longer input needs > ctx tokens and cannot fit — reject it
before the O(n) BPE runs, instead of tokenizing a multi-MiB body to completion (~27 s
on a 32 MiB body) and only then comparing against the context window.
```

## chatInputBytes

Moved from `internal/serveapp/openai.go` (the comment above `chatInputBytes`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `toolSchemaBytes`) on 2026-10-10.

```text
toolSchemaBytes sums an OpenAI-shaped tool declaration list's rendered bytes (M-15,
audit-2026-09-10): RenderToolsSegments renders every tool's name/description/parameters into
the prompt whenever tools are active, so a large schema list must be priced at the same guard
that already prices the messages, not left to run the full BPE unpriced.
```

## server.lookupLocked

Moved from `internal/serveapp/openai.go` (the comment above `server.lookupLocked`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `server.modelByName`) on 2026-10-10.

```text
servedNames lists the loaded generative + embedding model ids (sorted).
modelByName is an EXACT registry lookup under the read lock — unlike pick, which falls back to
"the only loaded model" for any name (right for request routing, wrong for listing: it would
attach a decoder's paths to the embedding-model entry).
```

## chatReq.N

Moved from `internal/serveapp/openai.go` (the comment above `chatReq.N`) on 2026-10-10.

```text
N is OpenAI's "how many choices" field (N-30, docs/audit-2026-09-10.md — this repo
generates exactly one). Was entirely unparsed: an unrecognized JSON key is silently
dropped, so `n: 3` used to get one choice back under a 200 rather than an error naming
what was ignored. *int (not int) so n:0 and "omitted" are distinguishable from n:1.
```

## completionReq.Logprobs

Moved from `internal/serveapp/openai.go` (the comment above `completionReq.Logprobs`) on 2026-10-10.

```text
Logprobs shadows sampling.Logprobs (embedded below) for the /v1/completions surface.
The legacy Completions API types logprobs as an INTEGER (# of top alternatives), not the
chat API's bool — the standard SDK sends `logprobs: 5`, which failed to decode into a bool
and returned a 400 leaking Go struct/field names (audit M-06). The outer (shallower) field
wins during JSON decode, so req.sampling.Logprobs stays false and the expensive per-token
logprobs path never engages; the handler 400s explicitly when it is set (unimplemented here).
```

## serveChatText.guard

Moved from `internal/serveapp/openai.go` (the comment above `serveChatText.guard`) on 2026-10-10.

```text
Reject an over-context prompt before tokenizing it (G1c): turns a multi-MiB body from
~27 s of BPE + gigabytes of ids into a byte-length comparison.
```

## serveChatText.heartbeat

Moved from `internal/serveapp/openai.go` (the comment above `serveChatText.heartbeat`) on 2026-10-10.

```text
N-24 (docs/audit-2026-09-10.md): nothing else is sent between here and the first
token — on CPU that gap is the whole prefill (an 8k agent prompt ~270s), against a
300s harness idle timeout. Streaming per token once generation starts does not cover
the prefill window itself, same shape as the buffer-then-stream sites M-19 fixed.
```

## validateN

Moved from `internal/serveapp/openai.go` (the comment above `validateN`) on 2026-10-10.

```text
validateN rejects an explicit "n" other than 1 (N-30, docs/audit-2026-09-10.md): this server
always generates exactly one choice, and "n" used to be an unrecognized JSON key that decoded
silently — a request asking for n:3 got ONE choice back under a 200, with nothing in the
response naming what was ignored. n omitted (nil) is the default (1) and passes; n:1 passes;
anything else is a clean 400 rather than a silently wrong choice count.
```

## loadedModel.contextWindow

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.contextWindow`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.residentPath`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `prepare.temperature`) on 2026-10-10.

```text
G4: temperature has the same lower bound as top_p (rejected < 0), for consistency —
previously top_p=-1 was a 400 but temperature=-1 was accepted. It is not silently-wrong
output: SampleWithInfo short-circuits Temperature <= 0 to greedy argmax before any logit
scaling, so a negative temperature decoded greedily rather than inverting the ordering.
0 is the documented greedy/deterministic setting and stays valid; only negatives are rejected.
```

## prepare.topP

Moved from `internal/serveapp/openai.go` (the comment above `prepare.topP`) on 2026-10-10.

```text
M-02: top_p == 0 is the tightest nucleus (the single most-likely token), which is greedy — the old
`< 1` path stored 0, and the sampler treats TopP == 0 as DISABLED, so the request asking for the
tightest filter got a full-vocab draw. Reject outside [0,1]; map explicit 0 to greedy.
```

## prepare.topLogprobs

Moved from `internal/serveapp/openai.go` (the comment above `prepare.topLogprobs`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `prepare.maxTokens`) on 2026-10-10.

```text
A negative/zero value reaches NewCache(len(prompt)+maxTokens) → makeslice with a
negative cap → an unrecovered panic (fatal on the VL path's bare goroutine — audit
C-19); a huge value OOM-kills the server (C-18). Reject both here so every endpoint
that calls prepare (OpenAI/responses/tools/vision) gets a clean 400. The Anthropic
endpoint already rejects <= 0 upstream.
```

## prepare.clamp

Moved from `internal/serveapp/openai.go` (the comment above `prepare.clamp`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `prepare.admit`) on 2026-10-10.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the load-time fit guard
prices the worst case a request COULD reach, once, at load — it cannot see the request
that actually arrives. Runs LAST, after gr.promptIDs/gr.maxTokens are fully resolved
(including the C-18 clamp above), so the numbers in a refusal are the real ones, and every
prepare() caller gets this for free rather than needing its own copy of the check.
```

## prepare.share

Moved from `internal/serveapp/openai.go` (the comment above `prepare.share`) on 2026-10-10.

```text
MC3c: with several generations allowed at once, this request's prefill shares the margin with the ones running
or queued ahead of it — counted now, capped at the model's concurrency. A lone request keeps the whole margin.
So does a request that prefills on a GPU resident: its prefill passes run one at a time (MC3's exclusive
section), and the margin is live memory, which already excludes what the other generations hold. Split there,
it refused ~1000-token 7B prompts under MC3 (docs/measurements/spec-vs-batching-metal-2026-09-27.md §5).
```

## contextLengthError

Moved from `internal/serveapp/openai.go` (the comment above `contextLengthError`) on 2026-10-10.

```text
contextLengthError rejects a prompt that alone fills or exceeds the model's context window (C-20).
MaxPositions is loaded but was never compared to len(prompt): a multi-MiB body tokenizes to ~1M ids
and preallocates tens of GiB of KV (NewCache is sized len(prompt)+max_tokens) → OOM-kill; and even
within memory, positions past the trained context drive out-of-range RoPE and return plausible
garbage under HTTP 200 instead of a 400. ctx ≤ 0 (unknown) never rejects.
```

## contextLengthErrorFor

Moved from `internal/serveapp/openai.go` (the comment above `contextLengthErrorFor`) on 2026-10-10.

```text
contextLengthErrorFor is contextLengthError plus the remedy, for the case where the window that rejected the prompt is the
server's resident KV capacity and not the model's own (ctx < modelWindow): a coding agent's first request is ~11k tokens
(R19, docs/tasks/task-first-hour.md), the default resident capacity is 8192 on CUDA and 4096 on Metal, and the 400 used to
say only "context window is 8192" — which reads as the model's limit, so the client (opencode) compacted 34 times instead of
the operator raising -ctx. When the model's own window is the limit, -ctx cannot help and the message is unchanged.
```

## seedOrRandom

Moved from `internal/serveapp/openai.go` (the comment above `seedOrRandom`) on 2026-10-10.

```text
seedOrRandom returns the request's seed, or a fresh random seed when absent (M-03). OpenAI's contract
is that an OMITTED seed varies output run to run — best-of-N, "regenerate", and agent retry-for-
diversity all depend on it — but deref(sm.Seed, 0) pinned every seedless request to the deterministic
seed-0 stream. A supplied seed (including 0) is still honored verbatim for reproducibility.
```

## loadedModel.drive

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.drive`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `drive.adminCancel`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `loadedModel.drive.resident`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `streamTokens.sb`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `streamTokens.tail`) on 2026-10-10.

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

Moved from `internal/serveapp/openai.go` (the comment above `logprobImpossible`) on 2026-10-10.

```text
logprobImpossible is the logprob reported for a token whose probability is exactly zero: OpenAI's own convention (-9999.0), because -Inf, the true value, cannot be written in JSON.
A zero-probability candidate is normal, not an error: when the thinking budget forces the end-of-thinking token, that token has probability 1 and every other candidate, including the
filler entries that pad the top-k list, has none (S6's 35B image check, 2026-10-09: step 24 of a 32-token answer). NaN is NOT mapped: it would be a bug, and writeJSON reports it.
```

## admissionRecord

Moved from `internal/serveapp/admission.go` (the comment above `admissionRecord`) on 2026-10-10.

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

Moved from `internal/serveapp/admission.go` (the comment above `admission`) on 2026-10-10.

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

Moved from `internal/serveapp/admission.go` (the comment above `admission.position`) on 2026-10-10.

```text
position reports where the waiter with this id stands: its place among the waiters (1 = next to be
granted the turn), how many are waiting in all, and whether one is running now. ok is false when no
waiter has this id — it was never queued, has already been granted the turn, or dropped out. Order is
exactly arrival order: J1 is strict FIFO, and J6's reordering was measured and not shipped.
```

## streamMessages.heartbeat

Moved from `internal/serveapp/anthropic_stream.go` (the comment above `streamMessages.heartbeat`) on 2026-10-10.

```text
N-24 (docs/audit-2026-09-10.md): the ping above is a one-shot liveness check, not a
keep-alive — nothing else is sent until the first token, which on CPU is after the whole
prefill (minutes for an image, ~270s for an 8k agent prompt) against a 300s idle timeout.
```

## streamMessagesTools

Moved from `internal/serveapp/anthropic_stream.go` (the comment above `streamMessagesTools`) on 2026-10-10.

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

Moved from `internal/serveapp/banner.go` (the comment above `bannerFacts.fit`) on 2026-10-10.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the context cap, KV at that
cap, and what remains of the RAM budget — printed at every load, not only when something is
tight, because "79% of budget" at load time and "14 GB RSS, swapping" on the first real
request were the same load with no line connecting them.
```

## modelBannerFrom.context

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.context`) on 2026-10-10.

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

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.cpuNote`) on 2026-10-10.

```text
R20 (docs/tasks/task-first-hour.md): on the CPU path the window is the model's whole maximum — -ctx is the GPU-resident KV capacity and caps nothing
here — and KV is allocated per request, so the figure (and the "fit:" KV cost below) is a ceiling a request reaches only by filling the window. A
cold-user run read "262144 tokens, 10.4 GB KV" after a CUDA decline beside a `fit` that had priced the GPU plan at 8192, and took them for the same
question. Said here, beside the figure, so it is not left to be worked out.
```

## modelBannerFrom.fit

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.fit`) on 2026-10-10.

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

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.reuse`) on 2026-10-10.

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

Moved from `internal/serveapp/banner.go` (the comment above `modelBannerFrom.reuseNotMetal`) on 2026-10-10.

```text
Not Metal: this used to print "Metal keeps N by default" for the count a CUDA card had granted.
```

## concurrencyLine

Moved from `internal/serveapp/banner.go` (the comment above `concurrencyLine`) on 2026-10-10.

```text
concurrencyLine is how many generations of a model run at once (MC3c / MC3, -max-concurrent) and why fewer than asked
when that happens; "" when nothing needs saying. It is NOT part of the load-time banner: concurrency is decided by
setConcurrency after every model, adapter and vision tower has loaded, so a line printed with the banner would
report the undecided value (it did, 2026-09-26: every model's banner said "one generation at a time").
```

## warnThinkingTemplate

Moved from `internal/serveapp/blockdrafter.go` (the comment above `warnThinkingTemplate`) on 2026-10-10.

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

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `qwen3EmbedEOD`) on 2026-10-10.

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

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `newDecoderEmbedder.maxTokens`) on 2026-10-10.

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

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `server.loadDecoderEmbedder`) on 2026-10-10.

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

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `decoderEmbedder.EncodeBatchCounted`) on 2026-10-10.

```text
EncodeBatchCounted is EncodeBatch plus each input's token count, read off the SAME tokenize
call encodeLocked already makes rather than a second pass over the text (audit R-07: this
embedder's EncodeBatch's ids were tokenized once and thrown away, then countEmbedTokens
(embeddings.go) tokenized every input again from scratch purely to report prompt_tokens).
embedBatchCounter in embeddings.go is the optional capability the handler prefers this
through; encoders that don't implement it (the aikit-embed.Tokenizer path) are unaffected.
```

## truncateForContext

Moved from `internal/serveapp/decoder_embedder.go` (the comment above `truncateForContext`) on 2026-10-10.

```text
truncateForContext is tokenize's C-07 truncation arithmetic, pulled out so a test can call the
SAME code the request path runs instead of re-deriving it beside it (V-21,
docs/review-2026-09-04.md: the old test re-implemented room--/ids[:room], so a bug in tokenize's
own arithmetic — not this function's, since it didn't exist yet — would have passed unnoticed).
Truncates BEFORE the caller appends appendID, reserving its slot, so the appended token is
never the thing truncation drops — it must stay last, because it is the pooled position.
```

## embedTask

Moved from `internal/serveapp/embeddings.go` (the comment above `embedTask`) on 2026-10-10.

```text
embedTask picks the prompt a taskEmbedder applies (docs/tasks/task-embeddinggemma2.md, Gate 3, owner decision
2026-10-06): an explicit task names one of the model's prompts ("none" for none); else input_type maps onto the
model's own "query" and "document" prompts; else no prompt, which is what sentence-transformers applies by default,
so a client that knows nothing about prompts gets the reference's own output.
```

## maxEmbedInputs

Moved from `internal/serveapp/embeddings.go` (the comment above `maxEmbedInputs`) on 2026-10-10.

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

Moved from `internal/serveapp/embeddings.go` (the comment above `server.resolveDimensions`) on 2026-10-10.

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

Moved from `internal/serveapp/generations.go` (the comment above `generation`) on 2026-10-10.

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

Moved from `internal/serveapp/jobeventlog.go` (the comment above `jobEventLog.markDone`) on 2026-10-10.

```text
markDone marks the log closed — no further events will ever be appended — and wakes every
waiter so a blocked reader notices the job ended even if it produced no final event of its own.
Idempotent-safe to call once, by runJob's own defer chain; calling it twice would double-close
an already-replaced channel and panic, so it must not be called more than once per job.
```

## maxEmbedBodyBytes

Moved from `internal/serveapp/helpers.go` (the comment above `maxEmbedBodyBytes`) on 2026-10-10.

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

Moved from `internal/serveapp/helpers.go` (the comment above `maxBytes`) on 2026-10-10.

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

Moved from `internal/serveapp/helpers.go` (the comment above `jsonDecodeMessage`) on 2026-10-10.

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

Moved from `internal/serveapp/helpers.go` (the comment above `sseWriter`) on 2026-10-10.

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

Moved from `internal/serveapp/helpers.go` (the comment above `sseHeartbeat`) on 2026-10-10.

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

Moved from `internal/serveapp/helpers.go` (the comment above `sendUsage`) on 2026-10-10.

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

Moved from `internal/serveapp/helpers.go` (the comment above `writeJSON`) on 2026-10-10.

```text
Encode BEFORE the status line goes out. The body used to be streamed straight from the encoder after WriteHeader(code) with its error dropped, so a value JSON cannot
carry (a NaN or an infinity in a float) reached the client as HTTP 200 with Content-Length 0: S6's 35B image check read "empty answer" for a whole night and
the server said nothing (2026-10-09). A body that cannot be encoded is a server error and says so.
```

## reqID

Moved from `internal/serveapp/helpers.go` (the comment above `reqID`) on 2026-10-10.

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

Moved from `internal/serveapp/responses.go` (the comment above `serveResponsesWith.guard`) on 2026-10-10.

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

Moved from `internal/serveapp/responses.go` (the comment above `respondTools.stream`) on 2026-10-10.

```text
The shared tool turn (tool_turn.go). Streaming: response.created goes out first, as on the
plain-text path, and prose streams as output_text.delta while the model writes it where the
family allows (G21) — before, this path sent nothing but heartbeats until the generation ended.
```

## server.maybeStore

Moved from `internal/serveapp/responses.go` (the comment above `server.maybeStore`) on 2026-10-10.

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

Moved from `internal/serveapp/responses.go` (the comment above `responseInputToMessages`) on 2026-10-10.

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

Moved from `internal/serveapp/jobs_http.go` (the comment above `handleCreateJob.scope`) on 2026-10-10.

```text
J3 (task-work-queue-2026-09.md): POST /v1/jobs submits a generation and returns its id
immediately; GET /v1/jobs/{id} polls state+result; GET /v1/jobs/{id}/events re-attaches to its
output (replay-then-live); DELETE /v1/jobs/{id} cancels it. Text-only chat body for this pass —
see jobs_run.go's own doc comment and the task doc's closure note for what's scoped out
(vision, tools, the K4 user/user_id key that doesn't exist yet).
```

## qwen3MaxImageTokens

Moved from `internal/serveapp/qwen35_vision.go` (the comment above `qwen3MaxImageTokens`) on 2026-10-10.

```text
qwen3MaxImageTokens is the serve-side ceiling on merged image tokens per image. The checkpoint's own
preprocessor_config allows up to 16384 (longest_edge 16777216 px), which is not a usable request on
the CPU decode path an image turn takes today: a Gated-DeltaNet hybrid prefills one token at a time,
and the resident GPU paths cannot yet carry an image turn (docs/multimodal.md P8 record, item 5).
1024 tokens is a 1024x1024 image. Larger images are smart-resized down to fit, exactly as the
processor would for its own max_pixels; a build-time constant, not an environment read.
```

## isQwen35VisionDir

Moved from `internal/serveapp/qwen35_vision.go` (the comment above `isQwen35VisionDir`) on 2026-10-10.

```text
isQwen35VisionDir reports whether dir is a Qwen3.5+ checkpoint that carries a usable vision tower:
model_type qwen3_5/qwen3_5_moe, a non-empty vision_config with no DeepStack, and a preprocessor
config LoadQwen3PreprocessConfig accepts. Used for AUTO-discovery only: a stripped text-only copy
(no vision_config or no preprocessor_config.json) is simply not a vision model, not an error.
```

## sessionLRU

Moved from `internal/serveapp/sessions.go` (the comment above `sessionLRU`) on 2026-10-10.

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

Moved from `internal/serveapp/sessions.go` (the comment above `sessionDirPerm`) on 2026-10-10.

```text
KV snapshots and cold blobs are the CONVERSATION, not a cache of public data: a .giw-kv blob
replays what the user said and what the model answered. They were written 0o644 inside a 0o755
directory, so every local account could read them (audit-2026-09-02 N-21). Owner-only, both
levels — the directory matters as much as the files, since a readable directory lists the
session ids.
```

## sessionLRU.acquire

Moved from `internal/serveapp/sessions.go` (the comment above `sessionLRU.acquire`) on 2026-10-10.

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

Moved from `internal/serveapp/sessions.go` (the comment above `bestExtend`) on 2026-10-10.

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

Moved from `internal/serveapp/sessions.go` (the comment above `sessionLRU.save`) on 2026-10-10.

```text
Snapshot refuses RECURRENT state (Mamba-2 / DeltaNet / LFM2 conv / MLA latent),
which cannot be restored from a KV blob. It used to refuse sliding-window rings
too, and this line still said so — rings have been persistable since
kvSnapVersion 2 (N-34).
```

## pickSession

Moved from `internal/serveapp/sessions.go` (the comment above `pickSession`) on 2026-10-10.

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

Moved from `internal/serveapp/systemone.go` (the comment above `handleSystemOne`) on 2026-10-10.

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

Moved from `internal/serveapp/tool_turn.go` (the comment above `toolTurn`) on 2026-10-10.

```text
toolTurn is the outcome of one tool-bearing generation. /v1/chat/completions, /v1/responses and
/v1/messages each used to run this turn their own way — drive, buffer, parse, reconcile — and only
the OpenAI route streamed prose while the model wrote it (G21); /v1/messages, the route Claude
Code uses, sent nothing but heartbeats until the whole generation was done. runToolTurn is the one
implementation; each front end keeps only its wire format.
```

## reconcileProse

Moved from `internal/serveapp/tool_turn.go` (the comment above `reconcileProse`) on 2026-10-10.

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

Moved from `internal/serveapp/version.go` (the comment above `versionReport.history`) on 2026-10-10.

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

Moved from `internal/serveapp/version.go` (the comment above `countFlags`) on 2026-10-10.

```text
countFlags reports how many flags are registered, so the help header's "all N flags" line cannot
drift from reality the way a hand-typed count would. That drift is the same defect class as the
parity manifest's hand-typed aikit_version, which sat seventeen versions stale.
```

## prepImage.qwenBlock

Moved from `internal/serveapp/vision_multi.go` (the comment above `prepImage.qwenBlock`) on 2026-10-10.

```text
M-38 (audit-2026-09-10): Qwen2.5-VL's real chat_template.json (verified live against
Qwen/Qwen2.5-VL-7B-Instruct) splices <|vision_start|><|image_pad|><|vision_end|> inline
with NO adjacent newline on either side — the trailing "\n" this used to append doesn't
exist in the real template.
```

## prepImage.gemma4Block

Moved from `internal/serveapp/vision_multi.go` (the comment above `prepImage.gemma4Block`) on 2026-10-10.

```text
M-38 (audit-2026-09-10): Gemma 4's own processor (processing_gemma4.py, verified against the
real transformers source) does f"{boi_token}{image_tokens}{eoi_token}" — no adjacent
newline at all, unlike the trailing "\n" this used to append.
```
