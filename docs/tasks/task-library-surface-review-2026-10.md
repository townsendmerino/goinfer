# Task: review the Go library as the embedding developer's product, before v1.0 binds it (2026-10)

> **Status: SCOPED 2026-10-01, not started.** To run after v0.20.0 is tagged, in a session with no access to this
> repository, against the published tag. The pre-registration below (§3) is written before any program is, and a
> finding is graded against it rather than argued about afterwards.

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
