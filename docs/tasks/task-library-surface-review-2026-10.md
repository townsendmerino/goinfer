# Task: review the Go library as the embedding developer's product, before v1.0 binds it (2026-10)

> **Status: RUN 2026-10-09** against v0.22.0 (the latest tag; the plan said v0.20.0), by a fresh agent with no
> access to this repository, on the M1 Pro. Report verbatim:
> [`library-surface-2026-10-09.md`](../measurements/library-surface-2026-10-09.md). **9 must-fix, 19 should-fix,
> 4 notes.** Under §3's outcome rule v1.0 waits on the nine; §5 maps each to its tier and a change.

## 1. Why now

The owner decided on 2026-09-30 that goinfer's primary user is a Go engineer embedding a model in their own program.
The README was rewritten to lead with that, and `docs/use-from-go.md` now promises the experience. Nobody has reviewed
the Go API against that promise. The v1.0 tiers (`docs/api-tiers.md`) were signed off on 2026-08-18, before the
decision, and once v1.0 binds the Hard tier, an awkward call shape costs a major version to fix. Documentation
cannot fix an awkward API.

The cold-user runs (`docs/tasks/task-first-hour.md` §1) tested the library only as scenario C: a 40-line program
printing one completion. That found the bare-module `go get` failure; it never exercised constrained output, tool
calls, embeddings, concurrency or a GPU backend.

**Friction already seen, while writing `docs/use-from-go.md` and `examples/structured` (2026-09-30).** These are
leads for the run, not findings of it:

- **Defaults differ by surface.** The zero `decoder.Options{}` loads f32, while `goinfer-chat` and `goinfer-serve`
  default to int4; the first run of `examples/structured` hit the fit guard's "tight" warning for that reason.
  `--embed-int4` is on by default in the CLIs and off in the zero `Options`.
- **Constrained output needs boilerplate.** `examples/structured` and `examples/confidence` each repeat the same
  dozen lines collecting stop ids (`Special().EOS`, `Special().EndOfTurn`, every `Template.Stops()` string) before
  they can build a `constrain.Masker`. A user has to know to do it.
- **The tiers do not cover what the README leads with.** `constrain.GrammarFromStruct`, `constrain.JSONSchema`, the
  confidence API (`Masker.CaptureConfidence`, `FieldConfidence`), `Tokenizer.PreTokenizerDecline` and
  `Template.Stops` are in neither list of `api-tiers.md`, whose Experimental list says it is explicit, "never
  whatever is not listed above".

## 2. What the run does

Five programs, each written from the published docs only (the README, pkg.go.dev, `goinfer.dev/docs`), each a
separate `go mod init` module outside the repository with `GOWORK=off`, against `@v0.20.0`:

1. **Chat:** a multi-turn conversation through the checkpoint's own chat template.
2. **Structured output:** fill a Go struct, then a JSON Schema the program did not derive from a struct.
3. **Tool calls:** declare two or more tools, take a call, run it, return its result, take the answer
   (`chat.Template.ParseToolCalls` and the tool grammar).
4. **Embeddings:** embed a handful of strings with an embedding checkpoint and rank them by similarity. Finding the
   library's embedding entry point is part of the test.
5. **An agent loop:** tools plus a loop that runs until the model stops calling them, with a deadline and a
   cancellation from another goroutine.

One of the five also imports a GPU backend (`metal` on the Mac or `cuda` on nobara-pc), since that is where the
modules' separate aikit pins reach a user. A CPU-only program never sees them.

**What each program records:** every place the writer guessed, wanted something absent, or hit an error, with a
timestamp, tagged as in the cold-user protocol (`Guessed` / `Wanted and absent` / `Error`), and the answer to each
of these:

- **Cancellation:** does a cancelled `context` stop `Generate` promptly, and does the error say it was cancelled?
- **Errors:** can a caller tell the kinds apart (model refused by the fit guard, unknown chat template, unsupported
  family, a resident decline) without matching strings? Are they `errors.Is`/`errors.As`-able?
- **Concurrency:** is a `decoder.Model` safe to use from several goroutines? What do `NewSession`, `NewCache` and
  `Generate` promise, and where is that written? (The `serve` queue is not the question; the library's own model is.)
- **Ownership:** what must the caller `Close`, in what order, and what happens on a second `Close` or a use after
  it?
- **Module pins:** how many `go get` lines, and do the root and a GPU submodule ever disagree on aikit?

**Measured, with provenance** (machine, Go version, cold or warm module cache, network, date; spread over 3 runs):
`go get` to first token for program 1, the binary size of each program (`CGO_ENABLED=0`), and a clean `go build`
time. These are not graded speed claims, so they may be taken by day; nothing here times the engine against a peer.

## 3. Pre-registration

**A finding is a must-fix before v1.0** when it is any of:

- **a. A silent wrong result:** the program runs and produces a wrong or degraded output with no error or warning
  (for example a default that quietly changes quality).
- **b. A trap:** the obvious code, written from the docs, fails or misbehaves, and the fix is not findable from the
  error text or the docs.
- **c. Required boilerplate:** the same non-trivial sequence has to be written by every caller of a Hard-tier task
  (load, tokenize, render, generate, constrain), which means the surface is missing a piece.
- **d. An unstated contract** on a Hard-tier surface: concurrency, cancellation, error kinds or ownership that a
  caller needs and cannot find written down.

**Otherwise it is a should-fix** (friction a better doc or a small addition removes) or a **note**.

Every finding is mapped to its surface's tier in `api-tiers.md` (Hard, Experimental, or unlisted). An unlisted
surface that one of the five programs needs is itself a finding of kind d: it has to be placed before v1.0.

**Outcome rule:**

- **No must-fix findings:** the Hard tier stands as signed off; the should-fix list becomes ordinary work.
- **One or more must-fix findings:** each becomes its own scoped item with the program that showed it. v1.0 waits
  for them, and `api-tiers.md` is amended in the same change as each fix.
- A finding that is a doc gap only, with the code behaving as documented somewhere, is a should-fix, never a
  must-fix: the bar is the API, and the docs are fixed separately.

## 4. Who runs it, and what comes back

A fresh session with no access to this repository, as the cold-user protocol requires (`task-first-hour.md` §1): an
empty directory, the published tag only, declared prior contact with the project. The deliverable is one file,
`docs/measurements/library-surface-<date>.md`, verbatim: the five programs, the friction log, the measurements with
provenance, and the findings graded by §3. This session then maps each finding to a change.

Not in scope: the HTTP server's API (`docs/server.md`), the CLIs, and performance. Each has its own instruments.

## 5. Outcome (2026-10-09)

Each must-fix below is graded by the run, not re-argued here. "Checked" is this session reading the source at
`origin/main` 5802692a to confirm the mechanism; a row not checked rests on the run's evidence alone.

| id | kind | surface | tier (`api-tiers.md`) | checked | change it becomes |
|---|---|---|---|---|---|
| M1 | a | `constrain.JSONSchema` `format` | **unlisted** | yes: `constrain/schema.go:120` lists `format` among the ignored annotation keywords beside `title` and `description` | owner's call: refusing `format` breaks callers whose schemas carry it (serve's `response_format` included); enforcing the common formats keeps them. Either keeps the doc promise |
| M2 | c | `chat` + `tokenizer` + `decoder` render, encode, stop ids | Hard (`Detect`, `RenderSegments`, `EncodeSegments`, `Generate`, `chat.Stops`) | no | a `Template.StopIDs(tok)` (or a chat-level prompt builder) so the 12-line loop leaves every caller |
| M3 | c | `tokenizer.Decode` / `DecodePiece` for streaming | Hard | partly: `docs/use-from-go.md:94` streams with per-token `Decode` | a streaming detokenizer (UTF-8 and leading-space state); fix the doc's smallest program |
| M4 | c | tool-call wiring: `ToolCallWrapper`, `ToolCallOpener`, `ToolCallsGrammar`, `NewLazyMasker`, `ParseToolCallsFor`, `SplitThink` | `ParseToolCalls` Hard; the rest **unlisted** | no | a `Template.ToolGrammar(tools)`-style helper returning the lazy masker; state the empty `ToolCall.ID` and the `<think>` lead on `ParseToolCalls` |
| M5 | b | `decoder.Load` of a cpu-arm64 `.giw` with the zero `Options.Backend` | `Load` Hard; `Backend` Experimental | yes: `wantsCanonicalInt4` (`decoder/weightmat.go:497`) treats `""` as a non-CPU backend, so the zero value fails where `"cpu"` loads | **done** (branch `libsurface-harness-2026-10`): the `.giw` check treats `""` as `"cpu"`; the kind-5 test loads under `Options{}` |
| M6 | d | `Model.Close` during a generation | Hard | yes, structurally: `metal.(*resident).Close` (`metal/model.go:2631`) releases the device with no wait for an open chain (`chainOpen`, `metal/greedy_chain.go:167`) | Close waits for, or cancels and drains, in-flight generations; write the contract |
| M7 | d | GPU-resident `Model.Generate` concurrency | Hard (`Generate`); `CPUConcurrentSafe` unlisted | no | one written statement on `Model`: concurrent calls are safe; extras decode on the CPU while the resident KV is claimed |
| M8 | d | `Generate` / `Generation.Err` cancellation | Hard | no | document that `Err` wraps `ctx.Err()` and the prefill check granularity |
| M9 | d | `decoder.Load` error kinds | `Load` Hard; `ResidentDeclineError` unlisted | no | typed errors for unsupported family, quant and backend; an `Options` way to make a backend decline an error |

**Unlisted surfaces the five programs needed** (each a kind-d finding by §3's own rule, to be placed before v1.0):
`constrain.JSONSchema`, `constrain.GrammarFromStruct`, the tool-grammar set in M4, `decoder.FitDeclineError` and
`ErrWontFitResident`, `Model.EffectiveBackend` / `BackendReport` / `CPUConcurrentSafe`, and the `embeddinggemma2`
package. The run's S12 says the same of the README's headline API.

**Should-fix worth naming now:** S1 (a long prompt silently leaves the Metal resident path: 40 s against 4.9 s to
first token) and S5 (use after `Close` on a `.giw` is a SIGSEGV) are the two a user would hit first.
