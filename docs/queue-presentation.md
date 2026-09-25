# Presentation queue

What a user sees, reads, or can find — the web UI, the README, the landing site, published
docs, release notes, model discovery, and outbound communication. Anything whose deliverable is
a measurement or a kernel belongs in [queue-performance.md](queue-performance.md) or
[queue-correctness.md](queue-correctness.md) instead; "would be nice" is not an entry here.

> **One of five queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md) · **presentation** (this page).
> [`QUEUE.md`](QUEUE.md) is the index over all five and holds the cross-cutting sweeps.
>
> **Why a fourth — now fifth — queue.** The other three had no home for work whose deliverable is
> what a user sees or reads. That work had been accumulating as loose task docs and chat
> decisions, and some of it is now higher-leverage than anything in the other queues: the engine
> is well ahead of how it is presented. Filed 2026-09-15.

Numbered `U1`–`U14`, one sequence across both sections. Every citation below was checked against
the tree at filing time, not copied from the brief that proposed these items — several premises
turned out to be stale (a doc already shipped under a different name, a file already 2.5× shorter
than assumed, a claim that no longer exists to correct). Where that happened, this entry says so
instead of restating the stale version.

## UI

**U1 · The web chat UI restyle shipped for `serve`'s own UI — `demo/agent`'s browser UI still has
the cream palette that was the reason for the change** — web UI, **OPEN, filed 2026-09-15**

[`docs/completed/task-web-ui-ambient.md`](completed/task-web-ui-ambient.md) records the restyle as
**SHIPPED 2026-09-04** (`internal/serveapp/webui/index.html`, commit `032d5cd`) — AmbientCSS,
light surface, bright blue accent, vendored rather than fetched from the jsdelivr CDN (a network
fetch would break the offline claim), with contrast checked at archival on the two weakest
elements: the muted mono metadata rows and the inset user message.

That closed record is scoped to `internal/serveapp/webui/index.html` only. A second, separate
browser chat UI — `demo/agent/cmd/agent-web/index.html` — still carries the original warm cream
palette the archived task doc gave as its reason to change (`--bg: #faf9f5`, `--user-bg:
#f0eee6`, `--border: #e3e0d5`, `demo/agent/cmd/agent-web/index.html:11-12`), which still reads as
Claude's own color scheme. No task doc exists for restyling this second UI specifically, but the
completed one is a directly reusable precedent: same library, same non-negotiables (vendor the
CSS, don't fetch it; check contrast on muted mono rows and inset messages before landing).

**U2 · Runtime stats as a first-class UI element** — web UI, **OPEN, filed 2026-09-15**

Model, quantization, backend and tok/s in the header; token count, wall time and context use
under each response. No task doc or brief exists for this yet — it is proposed here, not
designed. Nothing in either web UI (U1) currently surfaces this class of information as a UI
element; it is the truest thing about this project and currently invisible to a browser user.
Depends on **U11** (load time is not measured anywhere yet, so it cannot be displayed anywhere
either).

**U3 · "Will it run on my computer?" as a user-facing feature — the capability already exists,
framed as internal tooling** — web UI, **OPEN, filed 2026-09-15**

[`docs/tasks/task-fit-to-hardware.md`](tasks/task-fit-to-hardware.md) — status header: "Phase 1 is
DONE for its `plan()` core and `goinfer-chat fit` dry run" — names the actual function
(`decoder/fitplan.go`) and the CLI surface (`internal/fitcmd`, `goinfer-chat fit`) that reads a
machine, picks a placement, and can report what it chose before any weights load. Phase 3
(WebGPU) is also done as of 2026-09-10 per the same status header. The capability is real and
already exercises the hardware-fit logic end to end; what does not exist is a *user-facing*
version of it — the "will it run" question is currently answered by a CLI subcommand or code
path, not a front door.

This is a case where the comparison the brief drew to a competing browser's setup flow
understates the position: `plan()` is backed by this repo's own capability matrix
([`docs/capability-matrix.json`](capability-matrix.json)) and parity provenance
(`docs/what-parity-gated-means.md`), not a heuristic — turning it into a plain-words
"here's what I picked and why" surface is a UI/copy task layered on an already-working core, not
a new capability.

**U4 · Browser demo — blocked, and the actual blocker is a withdrawn plan, not unstarted work**
— web UI, **PARKED, filed 2026-09-15**

There is no browser demo — the local-server `demo/gemma-web` was removed 2026-09-25 (never released), and
there is no WASM build: no in-page, install-free demo the way a competing browser's own demo leads with. The blocker is not merely "not started
yet": [`docs/tasks/parked/task-bindings.md`](tasks/parked/task-bindings.md) records that an
earlier WASM-based plan for reaching browsers/mobile was **evaluated and withdrawn** —
`GOARCH=wasm` has no SIMD (falls onto `linalg/dot_other.go`'s `dotGeneric` path), cannot reach the
CUDA or Metal backends, and wasm32 caps usable memory near 2–3 GiB
(`docs/tasks/parked/task-bindings.md:3–5`). Sidecar (desktop) and c-archive FFI (mobile) were
chosen instead for the *bindings* use case because they keep native SIMD, GPU access, and no
model-size ceiling.

That withdrawal was scoped to language/mobile bindings, not specifically to an in-tab demo, but
the identical mechanism (`GOARCH=wasm`) and the identical first blocker (no SIMD) apply to a pure
browser demo too — there is no separate, milder path to "0.5B in a tab, pure Go" that the
bindings withdrawal doesn't already rule out. File this as a known, named gap. It is not
queued work with a plan behind it; it is a limitation to state plainly if the comparison comes up,
which is what a competing browser demo's own prominence invites.

## Messaging

**U5 · README restructure — already substantially shorter than the premise this was filed
against; the front-loading question is still open** — README, **OPEN, filed 2026-09-15**

`README.md` is currently **377 lines**, not the 878 this item was originally filed against — no
restructure commit or brief doc was found in the tree to account for the difference (`git log
--oneline -- README.md` shows no such restructure among its recent history), so the 878 figure
appears to have been stale or mismeasured from the start rather than superseded by work already
done. The gap to a ~150-line front page is real but roughly 2.5×, not 6×.

What is still open: the front page is not yet just the GIF, the one-liner, the download command,
and the JSON-schema hook — `README.md` currently front-loads install instructions, a
quantization guide, and a model-family table before any of the deeper material moves to `docs/`.
No task doc exists for this restructure; it is proposed here, not designed.

**U6 · README's pure-Go-ports comparison — already landed, and does not make the claim this item
was filed to correct** — README / positioning, **CLOSED at filing — recorded for the record**

The premise: a claim that pure-Go llama.cpp ports are unmaintained, contradicted by
`goccy/go-llama` being active and MIT and occupying the same lane. That claim does not exist
anywhere in the current tree — `grep -rn -i "unmaintained\|not maintained\|abandoned" README.md
docs/positioning.md` finds nothing relevant to a competing project. What exists instead, already,
is [`docs/positioning.md:78–86`](positioning.md), linked from `README.md:297` as the "longer
form": `goccy/go-llama` is named directly, credited with reaching pure Go by transpiling
llama.cpp's `wasm64-wasip1` build to Go (`goccy/llamawasm2go`), and the distinction drawn is
exactly the one this item asks for — "goinfer implements the forward pass itself, which is what
lets it decode multi-threaded, reach a GPU, and load safetensors, GPTQ and AWQ alongside GGUF" —
not a maintenance claim about the other project.

Left in the queue at CLOSED rather than deleted, per this project's own convention of keeping a
negative or already-resolved finding on the record rather than silently dropping it.

**U7 · "What parity-gated means" — already written and already linked from the README** —
docs/, **CLOSED at filing — recorded for the record**

[`docs/what-parity-gated-means.md`](what-parity-gated-means.md) exists and is linked from
`README.md:310` at the exact quantization bullet this item asks for ("HuggingFace logit-parity
gate per family"), and from `docs/README.md`'s own reference table. The doc states the boundary
this item calls for explicitly: parity gating is not a quality benchmark.

Also left in the queue at CLOSED rather than deleted, for the same reason as U6.

**U8 · Landing site — a decision, not a doc** — landing site, **DECISION PENDING, filed
2026-09-15**

No landing site exists. The GitHub Pages site this repo does publish
(`https://townsendmerino.github.io/goinfer/`, built by `.github/workflows/book-pages.yml`) is the
book reader (see U9) — a chapter-by-chapter primer, not a hero/download/proof page. Static hosting
on Cloudflare Pages (free tier, binaries staying on GitHub releases) is the proposed structure: a
platform-detected download, a terminal transcript as proof, the JSON-schema hook, measured
numbers including the losses, then a link into the book. A domain registration is part of the
same decision — there are at least four other GitHub projects named `goinfer`, and owning
`goinfer.dev` (unregistered as of filing; not verified against a live registrar from this
session) would make this one canonical. Nothing to build until the structure and the domain are
decided.

**U9 · The book — publish standalone, or keep as GitHub Pages only** — docs/book/, **DECISION
PENDING, filed 2026-09-15**

[`docs/book/`](book/) holds twelve files — eleven numbered chapters
(`01-text-becomes-numbers.md` through `11-knowing-youre-right.md`) plus
[`12-glossary.md`](book/12-glossary.md) — written for Go engineers new to ML, each chapter tied to
a measured number. It is already built and served at
`https://townsendmerino.github.io/goinfer/` via `.github/workflows/book-pages.yml`, and linked
from both `README.md:341,347` and `docs/README.md:16`. The open question is not whether it is
readable — it already is — but whether it should also be pushed somewhere with independent
reach (e.g. a publishing platform, syndication), since it is plausibly the asset most likely to
be shared independently of whether anyone runs the engine.

**U10 · Model registry and short names — no doc written yet; the underlying data already exists**
— pull / docs, **OPEN, filed 2026-09-15**

`goinfer-chat pull` requires an owner/repo and a quant choice today. No `docs/task-model-registry.md`
or equivalent was found in the tree — this item's own premise that "doc written" applies does not
hold; it is proposed here, not designed. What does already exist is the data a registry would be
derived from: [`docs/capability-matrix.json`](capability-matrix.json) (generated, not
hand-maintained — `decoder/registry.go`/`pull/registry.go` are the code paths this would sit on
top of, not duplicate). The shape stays as stated: a curated registry derived from the capability
matrix, not a second hand-maintained list; no CDN — ship checksums and point at HuggingFace; and a
stance on which quants are recommended, which does not exist yet either.

**U11 · Load time — no instrumentation exists anywhere, confirmed** — benchmarks / UI, **OPEN,
filed 2026-09-15**

Confirmed by search: no cold/warm model-load timer exists in `internal/serveapp/` or
`decoder/model.go` (the load-time-adjacent code that does exist — the fit guard's VRAM check —
measures *whether* a model fits, not *how long* loading takes). This is one of the few metrics
where a single static binary with no runtime dependency has a structural advantage over a peer
that shells out or spins up a Python process, and it is currently unmeasured and unsurfaced.
Needs: instrumentation, a banner line (feeds **U2**), and a `benchmarks.md` cell distinguishing
cold and warm load. No task doc exists for this; proposed here.

**U12 · Go Weekly submission — the root cause is fixed; the release itself still has to be
re-cut before the premise holds** — release, **BLOCKED, filed 2026-09-15, root cause fixed
2026-09-15**

This item's premise was that release binaries (six platforms, chat and serve, plus
model-embedded 0.5B/1.5B builds) now exist, unblocking a submission to Go Weekly
(`editor@cooperpress.com`). Checked directly against the live release rather than assumed: the
latest tag, `v0.18.0`, had **zero** release assets at filing (`gh api
repos/townsendmerino/goinfer/releases/tags/v0.18.0 --jq '.assets | length'` → `0`), against 27
assets on each of the three releases before it (`v0.17.0`, `v0.17.1`, `v0.17.2`). Its
`release-assets.yml` run failed at the `cross-compile goinfer-serve per platform` step
(run `34807285691`, job `runtime binaries`): the windows/amd64 cross-compile failed because
`internal/serveapp/main.go` referenced `syscall.SIGUSR1`/`SIGUSR2` unconditionally, and those
signals do not exist on Windows. The chat binaries and both model-embedded builds built
successfully in that same run but were never attached, because the attach step never runs when
the job fails upstream.

This was the exact failure class `RELEASING.md:66-68` already names as its reason the
release-assets gate exists at all ("v0.16.0 shipped a README naming a binary that was not in the
release ... every internal gate was green") — recurring, not hypothetical, and filed separately
as its own engineering item (`docs/queue-engineering.md` H1) since nothing noticed the zero-asset
publish itself, which will recur on the next platform-specific call regardless of this one fix.

**Fixed same day**: `internal/serveapp/haltsignal_unix.go`/`haltsignal_windows.go` move the
SIGUSR1/SIGUSR2 wiring behind a build tag (Windows gets a no-op; `-halt-file` and the admin
socket are its equivalents). Verified by cross-compiling all six release targets directly,
including the one that failed (`GOOS=windows GOARCH=amd64 go build ./cmd/serve` — clean); `go
vet`/staticcheck clean under both the native and Windows `GOOS`.

**Still blocking U12**: the fix lands the *next* release; `v0.18.0` itself was already published
with zero assets and is not retroactively fixed by a later commit. A submission to Go Weekly with
a `README` "Download a binary" instruction that 404s on the current latest tag would be a worse
outcome than no submission — this stays BLOCKED until either `v0.18.0`'s release-assets workflow
is re-run (backfilling the same tag) or a new tag is cut and its own run is confirmed to attach a
full asset set, checked the same way this finding was: by counting, not by trusting the workflow's
own green checkmark.

**U13 · Upstream the internlm2 HuggingFace finding — the finding is documented; the report is
not drafted** — outbound, **OPEN, filed 2026-09-15**

A parity gate caught a real bug in `transformers` itself, not in this project:
`InternLM2RotaryEmbedding.__init__` computes `inv_freq` and registers it as a `persistent=False`
buffer, and the installed version's `from_pretrained` fast-init path never re-runs that formula
for buffers absent from the checkpoint's state dict — so `inv_freq` came back as uninitialized
memory instead of the real frequency table. Documented in
[`docs/parity-coverage-policy.md:1599–1640`](parity-coverage-policy.md) (the decisive test,
the fix applied to this repo's own pin script, and the generalization to any
`trust_remote_code=True` model with a `persistent=False` buffer computed in `__init__`), and
retold in [`docs/book/11-knowing-youre-right.md`](book/11-knowing-youre-right.md). No draft
upstream bug report or PR was found in the tree — the premise that "a separate prompt exists for
the report itself" does not hold at filing; writing that report is the open work, not
transcribing one that already exists.

**U14 · An offline statement, prominently placed — the fact is stated, but scattered, not
singular** — README, **OPEN, filed 2026-09-15**

"Runs offline" already appears twice in `README.md` (line 9, the hero line; line 148, on the web
UI: "no external assets, so it works offline like everything else here") and once in
`docs/positioning.md:18`. What does not exist is one prominent, specific statement of the kind a
competing browser puts before install — naming exactly what does and does not connect to a
network. This project's actual answer is shorter and stronger than a competitor's ("here is what
connects online" implies something does) — nothing connects, ever — and stating that as its own
line, near the top, is the open work; the underlying fact is already true and already said, just
not gathered into one place a skimming reader would find.

**U15 · Surface prefix caching and KV persistence as user-visible claims** — README /
benchmarks, **OPEN, filed 2026-09-19**

We have both and advertise neither. `docs/multimodal.md`'s P9(a) shipped image-block resident
prefix reuse — a cold turn's forced full CPU prefill drops to warm resident reuse, measured
**159.98×** (8.215 s → 0.051 s, `docs/benchmarks.md:759–760`) — and `--session-dir` persists warm
`.giw-kv` sessions to disk **and restores them on restart**
([`docs/server.md:211`](server.md)). Neither appears in the README pitch or the landing-site plan
(U8).

Nirvana Code (github.com/niravlekinwala/nirvana-code, MIT, v0.3.0) leads its README with exactly
these two as headline features, quantified: warm-vs-cold TTFT on an 862-token prompt and a
first-token number for a freshly relaunched process under `--persist-kv` — their numbers, one
M2 Pro 16 GB, self-measured with their own bench tool, DIRECTIONAL ONLY and not ours to adopt.
Its comparison table lists disk-persisted KV cache as a column where Ollama and LM Studio both
read No. Depends on **U11** (load-time instrumentation) — a claim like this needs our own
measured cold-vs-warm figures, not a prose assertion.

**U16 · A comparison table** — README / landing site, **OPEN, filed 2026-09-19**

Nirvana Code (see U15) built one placing itself beside Ollama, LM Studio and llama.cpp CLI, and
it reads as confident rather than defensive. Our equivalent facts appear nowhere a visitor can
see: a single static binary with no daemon to install (`README.md:69`), no native runtime
dependency (no CUDA toolkit, no C++ compiler, no Python — `docs/positioning.md:15–18`), the
model itself compilable into the binary (`docs/roadmap.md:17`), decode gated bit-identical
against its own reference path on every backend (`docs/positioning.md:56–58`), and 36
parity-gated model families (`docs/positioning.md:35–49`) — stronger than several of their rows.

Note the tension to resolve before building this: a comparison table sits awkwardly next to this
project's own hedging convention (a claim ships with its own caveat attached, not smoothed over —
`benchmarks.md`'s Methodology section and CLAUDE.md's measurement-discipline rules are both built
on this), and any row claiming a speed advantage must match what `benchmarks.md` actually says,
including where goinfer loses (its own CUDA-vs-native figure is already flagged as narrower and
older than it reads, `docs/positioning.md:26–27`).
