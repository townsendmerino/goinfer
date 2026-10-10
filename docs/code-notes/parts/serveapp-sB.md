# internal/serveapp: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/serveapp`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## Main.usage

Moved from `internal/serveapp/main.go` (the comment above `flag.Usage` in `Main`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above `reqLogOut` in `Main`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above the `GET /{$}` route in `Main`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above `srvCtx` in `Main`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above `attachGemma3Tower` in `server.loadVisionTower`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above `towerInt8`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment inside `towerInt8` (case "f32")) on 2026-10-10.

```text
Explicit, and the only way to ask for Gemma 3's float32 device tower on CUDA (S4 addendum); unset keeps the old rule below. On webgpu there is no float32
device tower, so the attach declines by name and the CPU tower runs.
```

## enableResidentTower

Moved from `internal/serveapp/main.go` (the comment above `enableResidentTower`) on 2026-10-10.

```text
enableResidentTower attaches the device-resident vision tower when the backend is webgpu, cuda or metal and reports whether it
is attached. A failed attach (no VRAM left for the tower, a build without the backend) is a warning, not an error: it used to
abort serve startup and throw away the model already loaded on the GPU, but EnableResident leaves the CPU path intact, so the
tower runs there (slower) and the banner does not claim "-resident". Metal's towers (SigLIP and Qwen2.5-VL, S3) are float32
and decline an int8 tower, which lands here as that warning.
```

## loadQwenVisionTower

Moved from `internal/serveapp/main.go` (the comment inside `server.loadQwenVisionTower`) on 2026-10-10.

```text
Metal's tower (S3) attaches through attachResidentTower; CUDA's (aikit's gpu/qwencuda, S4: G-S4q correct at real size and 1.6-2.7x the CPU tower)
through qwenTowerPlacement, which names every CPU fallback; every other backend runs the CPU tower.
```

## server.loadGemma4VisionTower

Moved from `internal/serveapp/main.go` (the comment above `server.loadGemma4VisionTower`) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above `tryLockUntil` (with the `demoteLoop` and `modelList` comments, which were glued above it)) on 2026-10-10.

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

Moved from `internal/serveapp/main.go` (the comment above `visionPathError`) on 2026-10-10.

```text
visionPathError is -vision's refusal for a path that is a file, not a vision-tower directory (R22, docs/tasks/task-first-hour.md).
A GGUF mmproj handed to it used to fail inside the encoder loader as ".../mmproj-....gguf/config.json: not a directory". A path that
does not exist, or a directory, is the loaders' to judge, with their own messages.
```

## qwenTowerPlacement

Moved from `internal/serveapp/main.go` (the comment above `qwenTowerPlacement`) on 2026-10-10.

```text
qwenTowerPlacement decides where Qwen2.5-VL's vision tower runs and attaches it (S4, docs/tasks/task-multimodal-support-2026-10.md): aikit's gpu/qwencuda
tower under --backend cuda when the binary registers one (cuda/vision_towers.go imports it) and the tower is float32 (G-S4q: correct at real size, 1.6-2.7x the
CPU tower); the CPU everywhere else, with the reason named. -require-backend turns each CPU fallback under cuda into a refusal. -vision-device cpu arrives
here as backend "cpu". Metal has no Qwen2.5-VL device tower yet (the owner's rebuild on the Metal base is the Mac's), so only cuda asks for one.
```

## server.publishLoaded

Moved from `internal/serveapp/admin.go` (the comment above `lm.model.Close()` in `server.publishLoaded`) on 2026-10-10.

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

Moved from `internal/serveapp/admin.go` (the comment above `server.handleAdminUnload`) on 2026-10-10.

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

Moved from `internal/serveapp/anthropic.go` (the comment above `anthropicInputBytes`) on 2026-10-10.

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

Moved from `internal/serveapp/anthropic.go` (the comment above the role check in `anthropicTurns`) on 2026-10-10.

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

Moved from `internal/serveapp/batches.go` (the comment above `batchRecord`) on 2026-10-10.

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

Moved from `internal/serveapp/batches.go` (the comment above `batchRecord.requestCancel`) on 2026-10-10.

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

Moved from `internal/serveapp/batches_run.go` (the comment above `setJobID`) on 2026-10-10.

```text
setJobID records line i's job id under the batch's own lock. Found by -race, not by
inspection, the same way job.go's own mu gained coverage in J3 (internal/serveapp/job.go:jobStore's doc comment):
requestCancel (batches.go) reads the whole JobIDs slice from a different goroutine while lines
are still writing their own slot — disjoint indices don't save a plain slice write from racing a
concurrent read of the slice's backing array with no synchronization between them.
```

## featureCache

Moved from `internal/serveapp/feature_cache.go` (the comment above `featureCacheBudget`) on 2026-10-10.

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

Moved from `internal/serveapp/gemma3_tower.go` (the comment above `attachGemma3Tower`) on 2026-10-10.

```text
attachGemma3Tower loads Gemma 3's SigLIP encoder (load(true) is the int8 one) and attaches its device-resident form.

Float32 is the CUDA default since 2026-10-08 (owner): it reproduces the CPU float32 reference's reply and serves a new image in 2.4 s against the int8 tower's 4.6 s, at the price of about 1.7 GiB of VRAM
against 0.56 (docs/tasks/task-multimodal-support-2026-10.md, "Night 2026-10-08"). A card with less room might not hold it, and a failed attach would drop the request to the CPU tower at about 21 s, which is
worse than the int8 device tower it replaced. So when fallback is set (the float32 tower was the DEFAULT, not asked for) a float32 attach that does not take is released and the int8 tower is attached in its
place, with a line saying why. An explicit `-vision-quant f32` sets fallback false and keeps the old behaviour: attach, or warn, or fail under -require-backend.

It returns the encoder that is attached (or the CPU one), whether it is int8, and whether its device form is attached.
```

## resolveGemma3VisionQuant

Moved from `internal/serveapp/gemma3_tower.go` (the comment above `resolveGemma3VisionQuant`) on 2026-10-10.

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

Moved from `internal/serveapp/gemma3_tower.go` (the comment above `resolveGemma3VisionQuantMetal`) on 2026-10-10.

```text
resolveGemma3VisionQuantMetal is resolveGemma3VisionQuant on Metal (S18 on the Mac, docs/tasks/task-multimodal-support-2026-10.md): the unset default is
the f16 tower when the budget holds the decoder, one KV slot at Metal's 2048-position floor and that tower, and the int8 tower (tower_gemm_w8, groups of
32) otherwise, with a note. The decoder's bytes are an estimate from the checkpoint: about 0.34 of the safetensors on the device (Gemma 3 4B's 5.15 GB
guard figure, less its 0.53 GB of KV, is two copies of about 2.3-2.9 GB), counted twice unless a fresh sidecar .giw will be loaded instead (a heap
load keeps the host copy beside the device one; part 1 of S18). It leans toward int8: a wrong f16 choice can push the decoder itself to the CPU, which
is far worse than the int8 tower.
```

## glm_ocr_vision.header

Moved from `internal/serveapp/glm_ocr_vision.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `internal/serveapp/job.go` (the comment above `jobStore.markRunning`) on 2026-10-10.

```text
markRunning records the pending -> running transition for an already-created job.

J3 (task-work-queue-2026-09.md) added a genuine concurrent reader of *job's fields:
handleGetJob, running on the HTTP handler's own goroutine while runJob's goroutine calls this
and finish (below). J2 never needed s.mu held across a field mutation — nothing read a job's
fields from outside the single goroutine that owned it. Every mutation here is under s.mu now;
see snapshot's own comment for the matching read-side fix (found by -race, not by inspection).
```

## loadedModel.closeEntryNatives

Moved from `internal/serveapp/liveness.go` (the comment inside `loadedModel.closeEntryNatives`) on 2026-10-10.

```text
N-32 (docs/audit-2026-09-10.md): gemma4Enc (*vision.Gemma4Encoder) has no Close method
either — same shape as vproj, plain weights — but unlike vproj it was never dropped here
at all, retaining a served Gemma 4 vision tower's weight memory past unload. The audit
flagged this as a latent risk "if a native Close appears"; it is simpler than that: the
field is already live (main.go's loadGemma4VisionTower, vision_serve.go's Forward path)
and just needed the same nil-out vproj already gets.
```

## reqTrace

Moved from `internal/serveapp/reqlog.go` (the comment at the top of the file) on 2026-10-10.

```text
-log-requests (R25, docs/tasks/task-first-hour.md): one line per generation request on stderr — route, model, status, prompt and completion tokens,
time to first token, total time. A cold-user run on goinfer-serve learned its prompt sizes only from error bodies: nothing said what a client had sent.
Off by default, so a quiet server stays quiet.

The middleware owns the status and the clock; the token counts and the first-token time can only be known where tokens are read, which is
streamTokens — the shared tail of every generation (drive and driveVL), so a request that makes several generations (a tool loop) is counted once
per request: prompt tokens of its first generation, completion tokens summed, first token of the first.
```

## swapguard.header

Moved from `internal/serveapp/swapguard.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `internal/serveapp/systemone_clef.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `internal/serveapp/systemone_clef.go` (the comment above `clefQuantRefusal`) on 2026-10-10.

```text
clefQuantRefusal is the error a Clef model gets for a quant it is not offered at, or nil. int4 is refused (owner decision 2026-10-03, after D13): on 150
records it read mean KL 0.051 and top-1 0.840 against the f32 reference on the CPU, and 0.044 and 0.840 on CUDA, where the default int8int8 reads 0.0165
and 0.907 (decisions-d13-clef-fidelity-2026-10-03.md section 7). Other quants are not refused here: f32 and int8int8 are graded, int4mix is simply not measured.
```

## serveChatToolsWith.guard

Moved from `internal/serveapp/tools.go` (the comment above the context-window guard in `serveChatToolsWith`) on 2026-10-10.

```text
G1c, extended (audit-2026-09-02 M-21). The guard reached three routes; this was one of the
five that still ran a full O(n) tokenize over an arbitrary body before rejecting it — the
G1c comment prices what it removes at "~27 s of BPE + gigabytes of ids" on a multi-MiB body.
M-15 (audit-2026-09-10): tools are ACTIVE on this path (that's why serveChatToolsWith was
reached at all), so req.Tools' schema bytes are added here — RenderToolsSegments below
renders every one of them into the prompt.
```

## serveChatToolsWith.buffer

Moved from `internal/serveapp/tools.go` (the comment above `ss` in `serveChatToolsWith`) on 2026-10-10.

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

Moved from `internal/serveapp/tools.go` (the comment above the `namedForce` check in `constrainForcedTool`) on 2026-10-10.

```text
N-18: a NAMED tool_choice that matched no tool used to land here and return nil —
the request then generated completely unconstrained, having asked for one specific
function. The 2026-08-05 audit made "named but unconstrainable" a 400 and left
"named but nonexistent" falling through, which is the louder of the two errors: the
caller has a typo or a stale tool list, and a prose answer looks like the model
simply chose not to call anything.
```

## toolUnionEnabled

Moved from `internal/serveapp/tools.go` (the comment above `toolUnionEnabled`) on 2026-10-10.

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

Moved from `internal/serveapp/tools.go` (the comment above `forcedTool`) on 2026-10-10.

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

Moved from `internal/serveapp/tower_reserve.go` (the comment inside the `glm_ocr` case of `towerParts`) on 2026-10-10.

```text
The ceiling's scratch alone is ~2.5 GB and, reserved, took GLM-OCR's CUDA context from 16384 to 4931 positions on the 8 GB card (measured 2026-10-07). Reserve for
about 1.5 MP and let a larger image fall back to the CPU tower (deviceFallback) instead.
```

## metalTowerEstimate

Moved from `internal/serveapp/tower_reserve.go` (the comment above `metalTowerEstimate`) on 2026-10-10.

```text
metalTowerEstimate is what a Metal device tower holds of the Mac's memory (S18 on the Mac): the projections at the form Metal uploads, the rest in float32, plus
the scratch. The grid towers (SigLIP, Qwen2.5-VL, Qwen3.5+, GLM-OCR) upload f16 projections since S17's lever B, or int8 in groups of 32 for Gemma 3's
int8 tower (S18, tower_gemm_w8: a byte plus an f32 scale per 32); Gemma 4's Metal tower is float32. Calibrated against Gemma 3's measured footprint
(G-S18h, metal TestS18TowerHostMemory, 2026-10-08): 1172 MB f16 and 843 MB int8 against 1292 and 932 from this, about 10% over, the safe direction (the
scratch term is towerParts', which over-counts Metal's).
```

## towerInt8VRAMEstimate

Moved from `internal/serveapp/tower_reserve.go` (the comment above `towerInt8VRAMEstimate`) on 2026-10-10.

```text
towerInt8VRAMEstimate is the Gemma 3 W8A8 SigLIP tower's footprint on the device: the block weights at one byte, the patch embed and position table in float32, and the fixed scratch (about nine
hidden-wide float32 buffers per patch). Calibrated against a measurement, not derived: the tower held 558 MiB of the 8 GB card after attaching and after a forward (cuda TestSiglipCUDA_int8VRAM,
2026-10-07), against 575 MiB from this formula.
```

## qwen25VLTowerEstimate

Moved from `internal/serveapp/tower_reserve.go` (the comment above `qwen25VLTowerEstimate`) on 2026-10-10.

```text
qwen25VLTowerEstimate prices goinfer's float32 Qwen2.5-VL tower on CUDA (cuda/qwen25_vision.go), calibrated to a measurement and not derived (S7 on CUDA's fix, 2026-10-08). On the 8 GB card the tower's weights took
2758 MiB, 337 MiB over the arithmetic size (about 570 separate buffers, each rounded up to the allocator's 2 MiB quantum); the first image's scratch is np * (7 hidden + 2 padded intermediate + patch dim + 2 head dim) * 4
bytes, 270 MiB at 4096 patches. The ceiling stays 8192 patches (about 1.6 MP); a larger image falls back to the CPU tower by name (deviceFallback).

There used to be a further 256 MiB of "slack for the resident build's scratch beyond the plan's reading, measured 535 MiB against the 384 MiB margin". That was a misattribution: the text build's scratch after the plan's
probe is 0 MiB, and the shortfall it was patching was the allocator rounding of the TEXT model's weights, which the CUDA plan now prices itself (cuda.packedAllocSlack, 2026-10-08). Removed after a served probe on the
card: serve's defaults on the 3B, a ~8,000-patch image (2,014 prompt tokens) then a small one, the tower on CUDA, no failure, 1152 MiB of the 8192 still free after all three requests.
```

## encodeVisionSegments

Moved from `internal/serveapp/vision_serve.go` (the comment above `encodeVisionSegments`) on 2026-10-10.

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

Moved from `internal/serveapp/vision_serve.go` (the comment above `recoverDeviceTower`) on 2026-10-10.

```text
recoverDeviceTower runs a tower's forward and turns a panic into an error that names the likely cause. A device tower grows its scratch on the first image, after the
resident decoder has taken the device's memory, and aikit's Qwen2.5-VL tower panicked on an allocation failure (`CUDA_ERROR_OUT_OF_MEMORY`) instead of returning it:
unrecovered, one big image killed the whole server (found by the S4 default-plan run, 2026-10-07). The request fails with the reason; the server and the other requests
live. Every tower passes through it: the grid towers inside deviceFallback.run, Qwen2.5-VL inside qwenDeviceForward, and every family's features (Gemma 3's SigLIP,
Gemma 4's image and audio towers included) at withFeatureCache.
```

## webUIFS

Moved from `internal/serveapp/webui.go` (the comment above `webUIFS`) on 2026-10-10.

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

Moved from `internal/serveapp/webui.go` (the comment above `searchLimit`) on 2026-10-10.

```text
searchLimit bounds how many suggestions a search returns — a dropdown, not a full listing;
pull.Search sends no limit= to HuggingFace at all when given 0, which is a request shape this
route should never produce. Raised from 8 to 50 live during testing (2026-09-17): 8 was too
narrow to surface a less-trending-but-still-relevant repo past HF's own trendingScore ordering
(pull.Search's own doc comment — not downloads or likes) for anything but the most obvious query.
```

## webPullRef

Moved from `internal/serveapp/webui.go` (the comment above `webPullRef`) on 2026-10-10.

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
