# Model pull — a download helper, scoped against what already exists

> **ARCHIVED 2026-09-02 — a record, not instructions.** Every phase in this doc shipped:
> `pull`, `-embed`, the `demo:` shortcuts, `serve pull`, resumable downloads, and the §4
> web UI. Checkboxes and "phase N" language below record the plan as it stood, **not**
> outstanding work. The three items that were still open when this was archived were moved
> to the live queue (`docs/QUEUE.md`, "Model-pull leftovers") rather than left here, per the
> archiving rule in `docs/parity-coverage-policy.md` — nothing in `docs/completed/` should
> tell a future reader what to do.
>
> The design questions this doc refused to answer without checking (the HF digest, the quant
> matching scheme, where `pull` should live) were all measured before implementation; the
> answers are in the STATUS section immediately below.

> **Framing.** This isn't "should goinfer download models" so much as "goinfer already does
> almost everything downloading would need — what's actually missing is one small piece."
> `--model` already loads a `.gguf` OR a `.giw` (`internal/serveapp/main.go:340`). A plain
> `.gguf` is already transparently transcoded to a sidecar `.giw` cache on first use
> (`internal/serveapp/main.go:862`). `cmd/prequant` already converts a `.gguf` to a `.giw`
> bundle. `demo/chat/build-embed.sh` already builds a model-in-binary release from a local
> `.gguf` path, prequant by default. The only step that does not exist anywhere in the repo
> is getting the `.gguf` onto disk in the first place — right now that's "go find it on
> HuggingFace yourself." Confirmed no HTTP client/download code exists anywhere outside the
> release CI (`grep -rl "http.Get\|http.Client{" --include="*.go" .` → nothing).

## STATUS — phase 1 SHIPPED 2026-09-02, and the two open questions are now measured

`goinfer-chat pull <owner/repo>[:quant|:file.gguf]`. Library in `internal/modelpull`, CLI in
`internal/chatapp/pull.go`. Stdlib only — `net/http`, `crypto/sha256`, `encoding/json` — so the
cgo-free single-static-binary property is untouched and no module dependency was added.

**Both things this doc refused to build on faith were checked against the live API first.**

1. **HF DOES hand back the expected digest before the transfer.** `GET /api/models/{repo}/tree/main`
   returns, per file, `path`, `size`, and `lfs.oid` — and **the oid IS the sha256**. Confirmed by
   identity, not by assumption: the API returns
   `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046` for
   `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, which is byte-for-byte the digest
   `.github/workflows/release-assets.yml` already pins for that same file. So §2.3's v1 plan is
   upgraded: the download is **verified**, not merely printed. A hash nobody compares is
   self-documentation, not a check. `--expect-sha256` is unnecessary and was not built.

2. **The `:quant` matching scheme has two traps, both found by looking.** Matching is on the full
   `-<quant>.gguf` suffix, **case-insensitively**:
   - Case differs by publisher — Qwen ships `…-q4_k_m.gguf`, bartowski ships `…-Q4_K_M.gguf`.
   - `Q2_K` and `Q2_K_L` **coexist in one repo**, so a substring match on `q2_k` is ambiguous and
     would silently return whichever sorted first. Both are pinned by `TestSelect_realWorldNaming`.

**Gating is much smaller than §2.3 feared, and is now detected up front.** Measured: upstream
originals are gated (`google/gemma-3-4b-it` and `meta-llama/Llama-3.2-3B-Instruct` both report
`gated: "manual"`), but the community GGUF re-uploads this command actually targets are not
(`bartowski/google_gemma-3-4b-it-GGUF`, `unsloth/Llama-3.2-3B-Instruct-GGUF` → `gated: false`).
Since `pull` targets GGUF repos by construction it mostly sidesteps the problem, and `gated` is a
field on the repo metadata, so a gated repo fails in a second with an actionable message instead of
after a multi-gigabyte 401. Note the field is `false` when open and a **string** when not, so it
cannot be decoded as a bool — doing so fails on exactly the repos the check exists to catch.

**§3 DECIDED: `pull` is a subcommand of `goinfer-chat`, not a fifth binary.** §3 called this out and
§5 phased it fourth; doing it fourth would have meant building the shape §3 already suspected was
wrong and then redoing the flag surface plus `build-embed.sh` and `release-assets.yml` anyway. The
person `pull` is for is holding `goinfer-chat-<os>-<arch>` — the ~5 MB "point it at your own GGUF"
tier — and has no other goinfer binary; a standalone fetcher they must first go and download
reintroduces the friction it exists to remove. Cost was ~10 lines of `os.Args[1]` dispatch.

**Verified end to end on the box:** listing, gated-repo refusal, unknown-quant (lists what exists),
malformed ref, a real 395.9 MiB download in 26 s with `sha256 verified`, a 0.58 s cache-hit re-pull,
and the pulled checkpoint loading and generating correct Go through `--model`.

### §4 the GUI question — option B (local web UI) SHIPPED 2026-09-02

`serve -web` serves a browser UI at `/`, off by default. §4 recommended a good CLI (option A,
built) and put the web UI as "a real option, not v1, worth a follow-up design of its own"; it
turned out small enough to do directly, because it adds no second inference path — the page is a
CLIENT of the same `/v1/models` and `/v1/chat/completions` routes every other client uses, so it
cannot drift from the API.

- **One embedded HTML file, no external stylesheet, font or script.** Non-negotiable rather than
  tidy: a CDN reference would make the UI of an offline-capable engine require the network.
  `TestWebUI_pageIsSelfContained` fails the build on any `http://`, `https://` or `integrity=`
  in the page, and also asserts the page still contains the routes it drives so the check cannot
  pass against an empty file.
- **Chat** (streaming, stop button, tok/s), **model list**, and **pull with live progress** over
  SSE — reusing `internal/modelpull` unchanged, so digest verification and the .part-then-rename
  behaviour are identical to the CLI's. Two front ends, one implementation.
- **Off by default, same two gates as `-allow-admin`.** The page is inert; `POST /web/models/pull`
  is not — it starts a caller-named multi-gigabyte download and writes it to disk. So: explicit
  `-web` opt-in, plus the pre-existing startup rule that a non-loopback bind must carry
  `-api-key`. Loopback stays key-free, so the single-user desktop case has no auth friction.
- **Single-flight pulls.** One at a time; a second returns 409. Without it a handful of clicks
  queue unbounded concurrent multi-gigabyte downloads against one disk. Not wrapped in the
  inflight gate either — that gate bounds INFERENCE, and a minutes-long download must not hold
  one of those slots.
- **`GET /{$}`, not `GET /`.** The bare pattern is a catch-all in Go 1.22+ ServeMux and would
  render HTML to an SDK that typo'd a route; `{$}` matches the root path exactly. Verified: `/`
  is 200 with `-web`, 404 without, and `/nope` stays 404 either way.
- **`-web` satisfies the "need at least one of --model/--embed-model/--allow-admin" startup
  check.** Fetching a model is the point of the Models tab, and requiring a model in order to go
  and get a model is a bootstrap the user cannot satisfy. Verified: `serve -web` with no model
  starts, serves the UI, and lists/pulls.
- **The done state says "Downloaded, not loaded".** Serving a pulled file is a separate,
  deliberately-gated action (`--model` on restart, or `-allow-admin`); a green tick that implied
  the model was live would be the more comfortable lie.

Verified live: UI served, unknown routes still 404, list, a real 412 MiB pull over SSE reporting
15 progress events and `verified: true` in 17 s, a cached re-pull returning start+done instantly,
and a concurrent pull refused with 409.

**A native desktop app stays rejected**, for §4's own reason: every realistic toolkit needs cgo or
a bundled webview runtime, which is the property this project exists to avoid.

### Phases 2–5 SHIPPED 2026-09-02 — the doc's whole plan is now built

**`-embed` (phase 2).** `pull <ref> -embed [os/arch…]` fetches, then bakes the model into a
single static cgo-free binary per target. It DRIVES `demo/chat/build-embed.sh` — the same script
the released `goinfer-chat-0.5b`/`-1.5b` assets are built with — rather than reimplementing it,
because the staging rules, the prequant-vs-raw tag and the cross-compile line already have one
owner. It therefore needs a source checkout and a Go toolchain, reported after the download
succeeds so a multi-gigabyte fetch is never wasted. That constraint is the feature's shape, not a
limitation: the person BUILDING a distributable is a developer; the person RECEIVING it needs
nothing. Verified: a 618 MB `goinfer-chat-qwen2.5-coder-0.5b-instruct-linux-amd64` that `ldd`
calls "not a dynamic executable" and that generates with no `--model` and no network.

**`demo:` shortcuts (phase 3), and the shared-source problem solved without touching the release
pipeline.** `pull demo:0.5b` / `demo:1.5b` resolve to an exact repo + filename + **pinned sha256**
from `internal/modelpull/curated.json`. §2.1 asked for one shared place with `release-assets.yml`;
rewriting that workflow's matrix to consume the JSON would have meant editing the release pipeline,
so instead `TestCurated_matchesReleaseWorkflow` parses the workflow and fails the build if repo,
file, digest or byte count disagree. The copies may exist; they cannot DISAGREE. (Proven red: a
one-character digest change fails the test.) The pin is load-bearing rather than decorative —
`resolve/main` moves, and verifying only against the API-declared digest would cheerfully confirm
a re-uploaded file's own hash, so `checkPin` refuses a mismatch **before** downloading.

**`serve pull` (phase 4 follow-on).** The CLI moved to `internal/pullcmd`, and both
`goinfer-chat pull` and `goinfer-serve pull` dispatch to it — one implementation, two front ends,
with the usage text naming whichever binary invoked it. §3's decision is unchanged; this just
stops serve users needing a second binary for the same job.

**Resumable downloads.** A dropped transfer keeps its `.part` and the next run sends
`Range: bytes=N-`. Three cases are tested against an `httptest` server rather than the network:
a mid-transfer drop then resume (asserting the FINAL sha256, because the running hash must cover
the bytes already on disk — hashing only the tail would make verification theatre); a server that
answers **200 to a Range request**, where appending would silently corrupt the file, so the offset
and hash both reset; and a digest mismatch, which must delete the partial or every later run
resumes from known-bad bytes forever.

### Still deliberately NOT built

- **`--prequant`.** `--model` already transcodes a bare `.gguf` to a sidecar `.giw` on first use, so
  the primary flow needs no conversion step; adding one would duplicate `cmd/prequant`'s job for no
  user-visible gain. §2.2's "should plain pull prequant by default" question is therefore moot for
  now rather than answered.
- **Disk-space precheck** before starting a multi-gigabyte fetch.
- **Split-GGUF checkpoints.** A shard is *detected* and refused by name rather than pulling one
  useless piece; actually assembling a split checkpoint needs loader work beyond this command.
- **Generating the release workflow's matrix from `curated.json`.** The drift test makes this
  optional rather than necessary; it is still the tidier end state.

---

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

- **Loading:** `--model` accepts `.gguf`, `.giw`, or an HF dir (`internal/serveapp/main.go:340`).
- **Auto-transcode:** a bare `.gguf` becomes a sidecar `.giw` cache on first use, one-time
  (`internal/serveapp/main.go:862`, `:642`) — this is *already* most of what a naive "pull and
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

## 5. Phasing as planned — ALL SHIPPED 2026-09-02 (record only)

1. ~~`cmd/pull`, explicit-reference form only, plain fetch + `--prequant`, CLI progress, sha256
   printed.~~ SHIPPED as a `goinfer-chat`/`goinfer-serve` subcommand (§3 was decided first, not
   fourth — see STATUS). `--prequant` was dropped as redundant: `--model` already transcodes.
   The sha256 is VERIFIED, not merely printed.
2. ~~`--embed`, wired to `build-embed.sh`'s existing logic.~~ SHIPPED, exactly that way.
3. ~~The `demo:0.5b`/`demo:1.5b` curated shortcuts, sourced from one place shared with
   `release-assets.yml`.~~ SHIPPED, with the sharing enforced by a drift test rather than by
   rewriting the release workflow.
4. ~~Decide and execute the subcommand question (§3) last.~~ Decided FIRST instead: doing it
   fourth would have built the shape §3 already suspected was wrong. It belongs inside
   `goinfer-chat`, and `goinfer-serve` shares the implementation.
5. ~~The local web UI (§4), as its own scoped design, if wanted at all.~~ SHIPPED as `serve -web`.

## Sources

`README.md:10,17-33` (the three existing tiers) · `internal/serveapp/main.go:340,642,838`
(`--model`'s .gguf/.giw handling and the transparent transcode) · `cmd/prequant/main.go`
(conversion CLI) · `demo/chat/build-embed.sh` (existing embed pipeline) ·
`.github/workflows/release-assets.yml` (current model-fetch precedent + checksum practice) ·
`docs/positioning.md:60-64` (aikit, and what goinfer explicitly is not) ·
`docs/capability-matrix.md` (confirms no existing checkpoint-level catalog, only
architecture-family coverage).
