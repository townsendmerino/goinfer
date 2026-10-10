# internal/serveapp: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/serveapp`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

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
