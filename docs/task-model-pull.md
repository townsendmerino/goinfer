# Model pull — a download helper, scoped against what already exists

> **Framing.** This isn't "should goinfer download models" so much as "goinfer already does
> almost everything downloading would need — what's actually missing is one small piece."
> `--model` already loads a `.gguf` OR a `.giw` (`internal/serveapp/main.go:330`). A plain
> `.gguf` is already transparently transcoded to a sidecar `.giw` cache on first use
> (`internal/serveapp/main.go:838`). `cmd/prequant` already converts a `.gguf` to a `.giw`
> bundle. `demo/chat/build-embed.sh` already builds a model-in-binary release from a local
> `.gguf` path, prequant by default. The only step that does not exist anywhere in the repo
> is getting the `.gguf` onto disk in the first place — right now that's "go find it on
> HuggingFace yourself." Confirmed no HTTP client/download code exists anywhere outside the
> release CI (`grep -rl "http.Get\|http.Client{" --include="*.go" .` → nothing).

## 0. What must not change

The README's headline pitch — "no model download" (`README.md:10`) — describes the
pre-baked `goinfer-chat-0.5b`/`goinfer-chat-1.5b` releases specifically, and this work
doesn't touch them. What it adds is a convenience for the OTHER existing tier: the ~5 MB
bare runtime that already says "bring your own GGUF" (`README.md:25`). That tier has real,
unaddressed friction today — a user has to already know which HuggingFace repo and file to
grab — and this closes that gap. It is additive, not a repositioning.

Anything downloaded goes through the exact same path as a manually-supplied file: the same
`--model` flag, the same transparent `.giw` transcode, the same HF-logit-parity gate. No new
loading code, no new quantization code, no shortcut around the correctness gates that
already exist. The new surface is entirely "get bytes onto disk," full stop.

## 1. What already exists (the inventory, so the new-code surface is honest)

- **Loading:** `--model` accepts `.gguf`, `.giw`, or an HF dir (`internal/serveapp/main.go:330`).
- **Auto-transcode:** a bare `.gguf` becomes a sidecar `.giw` cache on first use, one-time
  (`internal/serveapp/main.go:838`, `:642`) — this is *already* most of what a naive "pull and
  cache" command would build from scratch.
- **Conversion:** `cmd/prequant/main.go` — `prequant -o out.giw -quant {int8int8|int8|int4}
  [-embed-int4] [-row4] input.gguf`. Takes a local path only (`flag.Arg(0)`); no fetch step.
- **Embedding into a release binary:** `demo/chat/build-embed.sh [--gguf] [--name NAME]
  <model.gguf> [os/arch ...]` — stages the `.giw` (or raw `.gguf`) next to the `//go:embed`
  directive in `internal/chatapp` (`internal/chatapp/embed.go:41` for `.gguf` mode, `internal/chatapp/prequant.go:19` for the
  default prequant mode) and cross-compiles. This already does everything `--embed` below
  would need; the new command just needs to call it (or replicate its ~15 lines) after a
  fetch, not reinvent it.
- **Precedent for the fetch step itself:** `.github/workflows/release-assets.yml`'s own model
  step is `curl -fsSL --retry 3 --retry-delay 5 -o model.gguf "$URL"` against a direct HF
  `resolve/main/` URL — no auth, no fancy resumable logic. That's the existing bar; a Go
  `pull` command should clear it (progress reporting, integrity), not necessarily do dramatically
  more for v1.

## 2. What's actually new: `cmd/pull`

New package, same shape as the existing `cmd/gate` / `cmd/prequant` / `cmd/serve` tools.

### 2.1 The reference format

Primary form is explicit and auditable, not a hidden name→repo table: an HF repo plus a file
or quant selector —

```
goinfer pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m
goinfer pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

The `:quant` short form resolves against the repo's file listing (HF's API lists files per
repo; needs a real check during implementation of exactly what that API returns before
committing to a matching scheme — don't assume). This keeps the tool's whole model of the
world honest: "here is exactly what repo and file this pulled," which matches this project's
own citation discipline better than an opaque Ollama-style tag would.

A **thin, explicitly-curated** convenience layer sits on top for the two models the project
already vets and ships itself: `goinfer pull demo:0.5b` / `demo:1.5b`, resolving to the exact
same URLs already hardcoded in `release-assets.yml`. Read that mapping from one shared place
(a small config both the workflow and the CLI reference, or generate the workflow's matrix
from it) rather than hand-duplicating it — this project already has a demonstrated allergy to
two copies of the same fact drifting apart (`book_link_lint.py`, `queue_citation_lint.py`
elsewhere in the repo). Do not grow this into a large hand-maintained name registry; the
explicit form is the real interface.

### 2.2 The three depths, mapped onto the three tiers that already exist

```
goinfer pull <ref>            fetch the .gguf to the cache dir, nothing else.
                               (equivalent to what a user does by hand today)

goinfer pull <ref> --prequant  fetch, then run cmd/prequant → a .giw next to it.
  (default? see below)         --model can point straight at the .giw. Faster boot,
                               smaller heap, same as the README's own pitch for why
                               prequant is the default embed mode.

goinfer pull <ref> --embed[=NAME] [os/arch...]
                               fetch, prequant, then call build-embed.sh (or its
                               logic inlined) to produce a goinfer-chat-<NAME>-<os>-<arch>
                               binary for the current platform by default, or the
                               listed targets. This is the part that's actually new
                               relative to Ollama/WebLLM: neither of them can hand a
                               user back a single portable file for a model that
                               ISN'T one of a project-curated set. Any goinfer user,
                               for any supported architecture, could produce their
                               own zero-install artifact.
```

Whether plain `pull` should prequant by default (matching `build-embed.sh`'s own default) or
leave a bare `.gguf` (matching `--model`'s existing zero-config bare-GGUF path) is a real
open call — leaning toward defaulting to prequant, since `--stream-weights` already proves
the codebase treats "just transcode it" as the unsurprising default, and a fresh `.giw` is
strictly more useful downstream (both `--model` and `--embed` want one).

### 2.3 Download mechanics

Plain `net/http` streaming GET into the cache dir, `io.Copy` through a byte-counting
`io.Writer` for progress — no new go.mod dependency, matching the project's existing
cgo-free/minimal-deps discipline everywhere else. Show a single-line, carriage-return-updated
progress indicator (bytes, rate, ETA) in the CLI; nothing fancier is needed for v1.

**Integrity.** *(Corrected 2026-09-02, after audit M-33 landed: the premise below was true when
this was drafted and is not any more. `release-assets.yml` now PINS each model file's sha256 and
byte count, verifies with `sha256sum -c`, fails the build on a mismatch, and checksums every
attached asset rather than only its own binaries. That is a precedent to follow, and the pinned
digests are a shared place §2.1's "read that mapping from one place" argument already wants — a
`pull` that re-derived them by hand would be the second copy this repo keeps finding.)*

For v1: always compute and print the sha256 of what was fetched — cheap, and it means
every pull is at least self-documenting and reproducible-by-record, matching the "no number
without a source" ethos in `docs/benchmarks.md`. Support an optional `--expect-sha256` for
repeat/scripted pulls where the user (or a future curated list) already knows the right
value. Whether the HF API can hand back an expected hash ahead of the download to verify
against automatically needs to be checked against the real API during implementation, not
assumed — don't build that on faith.

**Cache layout.** A platform-appropriate user-cache directory (`os.UserCacheDir()`, stdlib —
no new dependency needed for this either), namespaced under `goinfer/models/<repo>/<file>`.
Gitignored/irrelevant to the repo itself, obviously, but worth a line in the design so
`--embed` and repeat `pull` calls agree on where things live.

**Known gap, not solved in v1:** some HF repos are access-gated (login + license
acceptance in the browser before a token can fetch them). An anonymous GET against those
will 401/403. Detect that case and fail with a clear message pointing at manual download +
`--model` (which already works) rather than attempting any auth flow — that's real scope
(device-code login, token storage) this shouldn't take on for a v1 convenience feature.

**Nice-to-haves, explicitly deferred:** resumable downloads via `Range` requests (stdlib can
do this; just not v1), a disk-space precheck before starting a multi-GB fetch.

## 3. Where `pull` lives, and a real open question it raises

`cmd/pull` as a standalone binary matches the existing `cmd/gate` / `cmd/prequant` /
`cmd/serve` convention — but none of those are what an end user actually has in hand. The
person this is *for* already has `goinfer-chat-<os>-<arch>` (the bare runtime) or
`goinfer-chat-0.5b`. Shipping the fetch helper as a fifth separate binary they'd have to
separately go get defeats a good chunk of the point.

That argues for `pull` (and maybe `serve`) becoming *subcommands* of the binaries people
already download — `goinfer-chat pull <ref>` — rather than a sixth thing to distribute. Every
`cmd/*` tool today is a standalone `func main()` with its own flat flag set; making
`demo/chat`'s binary dispatch on `os.Args[1]` is a real, if modest, restructuring, not a
trivial addition, and it touches the release build (`build-embed.sh`, `release-assets.yml`)
too. Flagging this as a decision to make explicitly rather than something to discover
mid-implementation.

## 4. The GUI question

Nothing in the repo has a GUI today — `cmd/` is `gate`/`prequant`/`serve`, all CLI. Three
real options, not a binary yes/no:

**A good CLI/TUI (recommended, build this).** A clean progress bar and status output for
`pull`, matching the polish `ollama pull` has, in this project's own terser register. Zero
new dependencies, zero risk to the cgo-free/single-binary story, and it's most of what "nice
download UX" actually means for the audience this project has always targeted (`docs/book`'s
own tagline: "for someone who knows Go"). This is the actual ask, most likely.

**A local web UI riding on the existing HTTP server (real option, not v1).** `cmd/serve`
already runs an OpenAI-compatible HTTP server. A minimal page — model list, pull-and-watch-
progress, maybe a basic chat box — could be embedded via Go's own `embed` package the same
way `internal/chatapp` already embeds its model, served off a new route on the *same*
process, with zero new external runtime dependency and no change to "single static binary."
This is the version of "GUI" that actually fits what goinfer is. Worth a follow-up design doc
of its own if wanted, not bundled into `pull` v1.

**A native desktop app — not recommended.** Electron/Fyne/Wails-class GUI toolkits are a
different kind of project: new dependencies (several of the realistic options need cgo or a
bundled webview runtime), a different audience than "Go engineers who'd read an 11-chapter
primer on how a language model runs," and a maintenance surface (packaging, code-signing,
auto-update) unrelated to anything else this repo does. It would solve "make goinfer
approachable to non-technical users," which is a real but different product decision than
"make fetching a model less annoying for the people already using goinfer" — worth having as
an explicit, separate conversation if that's ever the actual goal, not something to fold into
this.

## 5. Suggested phasing

1. `cmd/pull`, explicit-reference form only, plain fetch + `--prequant`, CLI progress, sha256
   printed. No `--embed`, no curated shortlist, no subcommand restructuring.
2. `--embed`, wired to `build-embed.sh`'s existing logic.
3. The `demo:0.5b`/`demo:1.5b` curated shortcuts, sourced from one place shared with
   `release-assets.yml`.
4. Decide and execute the subcommand question (§3) once it's clear whether `pull` earns its
   keep as a separate tool or belongs inside `goinfer-chat`/`serve`.
5. The local web UI (§4), as its own scoped design, if wanted at all.

## Sources

`README.md:10,17-33` (the three existing tiers) · `internal/serveapp/main.go:330,642,838`
(`--model`'s .gguf/.giw handling and the transparent transcode) · `cmd/prequant/main.go`
(conversion CLI) · `demo/chat/build-embed.sh` (existing embed pipeline) ·
`.github/workflows/release-assets.yml` (current model-fetch precedent + checksum practice) ·
`docs/positioning.md:60-64` (aikit, and what goinfer explicitly is not) ·
`docs/capability-matrix.md` (confirms no existing checkpoint-level catalog, only
architecture-family coverage).
