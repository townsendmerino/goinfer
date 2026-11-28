# The docs, and how they fit together

`docs/` holds the public record. It is not one kind of thing, and reading it as if it were is the
main way people get a wrong answer here: a **current-state page** says what is true of the engine
now, a **measurement** is evidence with a machine and a date on it, and an **archive** is finished
work kept for its reasoning. Only some of them are current claims about the engine.

The queues of open work, the task docs (design records with their item ledgers) and the plans and
scoping notes are the maintainer's working record and are not in this tree; see
[Open work, task docs and plans](#open-work-task-docs-and-plans) below.

This page is the map.

## Start here

| | |
|---|---|
| [**use-from-go.md**](use-from-go.md) | importing goinfer into your own Go program: the working `go get`, the smallest program, output shaped like a Go struct or a JSON Schema, per-field confidence, and what v1.0 will bind; every code block is copied from `apps/examples/` and tested |
| [**book/**](book/) · [read online](https://goinfer.dev/book/) | eleven-chapter inference primer for Go engineers — concepts from zero, each chapter ending in a measured number |
| [bigger-than-memory.md](bigger-than-memory.md) | running a model bigger than your RAM or your GPU: `-stream-weights`, `-moe-cache-experts`, and the swap-tripwire history |
| [small-devices.md](small-devices.md) | a Raspberry Pi, not a microcontroller: cross-compile and copy, what memory floor and CPU features to expect, and why no board figure is published yet |
| [quantization.md](quantization.md) | which quants goinfer stands behind, which it measured and refused, and where there is no evidence — reading a format is not endorsing it |
| [how-inference-works.md](how-inference-works.md) | the same ground in ~2,300 words, anchored to specific source lines. The code map |
| [webgpu-primer.md](webgpu-primer.md) | orientation for anyone touching `backend/webgpu/` |

## Current state — what is true of the engine now

These are the pages to trust, and to update when reality moves.

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the pieces fit: the forward pass, loading and memory, the GPU backends and how residency is admitted, serving, configuration, modules and where cgo is quarantined |
| [capability-matrix.md](capability-matrix.md) | **generated** from the `decoder` registry — 45 model families as of 2026-11-27. The registry is the source of truth; do not hand-edit |
| [option-state-grid.md](option-state-grid.md) | **generated** from `decoder/optiongrid.go` and `decoder/cachestate.go` — every load option × execution path (tested, declined, admitted untested, n/a) and every kind of cache state × lifecycle path. The untested count only falls; do not hand-edit |
| [citation-index.md](citation-index.md) | **generated** by `scripts/queue_citation_lint.py --update` — the commit citations of the queue with the subject each resolved to. The lint's state, not reading material; do not edit or cite |
| [tool-call-coverage.md](tool-call-coverage.md) | per family: tool calls **constrained**, parsed only, or none — from a census of real checkpoints through `chat.Detect` (2026-09-24). What "tools: yes" does and does not promise |
| [benchmarks.md](benchmarks.md) | **current claims only**, provenance-gated: machine, checkpoint, quant, date, thermal note. Section IDs are stable; a *Retired section IDs* index maps the ones that moved |
| [legacy-benchmarks.md](legacy-benchmarks.md) | the retired, superseded and historical rows `benchmarks.md` used to carry — moved verbatim 2026-08-31 and 2026-09-05 (was `benchmarks-archive.md`). **Never a current claim**; kept so a retraction can be audited |
| [server.md](server.md) | the HTTP surface — OpenAI, Anthropic, vision, embeddings, admin |
| [api-tiers.md](api-tiers.md) | which surfaces v1.0 semver-binds, and which are explicitly Experimental |
| [positioning.md](positioning.md) | what goinfer is for and is not — the long form of the README's framing |
| [roadmap.md](roadmap.md) | direction — where we are, the promotion gate, open programs by owner, what is parked and why, the v1.0 checklist; the June 2026 roadmap is archived |
| [next-models.md](next-models.md) | which model families next, and why; what became of the last list (was `post-v1.0-models.md`) |
| [capability-matrix.md](capability-matrix.md), [hardware-matrix.md](hardware-matrix.md), [env-vars.md](env-vars.md), [giw-bundles.md](giw-bundles.md) | generated or reference tables |

## Open work, task docs and plans

What is still open, and why a thing is built the way it is, is kept in a private notes checkout beside this repository
(`../goinfer-notes`), not here: `queues/` (`QUEUE.md` and the per-question queues, with the generated `citation-index.md`),
`tasks/` (`task-*.md` design records, each with an item ledger, and `tasks/parked/`) and `plans/` (`plan-*.md`,
`scoping-*.md`). `scripts/task_docs.py`, `scripts/queue_citation_lint.py`, `scripts/doc_id_collisions.py` and
`scripts/doc_review_staleness.py` read them there when the checkout is present (`--notes PATH` names it) and say what they skipped
when it is not; CI has no checkout. What the engine does and why is on the current-state pages above,
[`ARCHITECTURE.md`](ARCHITECTURE.md), `code-notes/` and `spec/`. A code comment that points at a task doc or a queue points into the notes. A few of them still sit in `docs/` (`QUEUE.md` with its generated
[`citation-index.md`](citation-index.md), `queue-engineering.md`, and six task docs) because the website's writeups cite them as their evidence.

`spec/` (13) holds the design series for speculative decoding, run as numbered documents with
pre-registered kill-gates.

## Evidence — `measurements/` (275)

Raw logs and per-run write-ups. A number in `benchmarks.md` should be traceable to one of these.
They are dated and machine-stamped by convention, and they are **not** updated when the world
moves — a superseded measurement stays as it was and the page that quotes it is what changes.

## Archive — `completed/`

Finished work, kept for the reasoning rather than the outcome — including negative results, which
are archived with the same care as wins. **Nothing under `completed/` is scanned by the citation
lint**, so archiving a document also retires its citations from the live gate. When a document
moves here, a pointer stub is left behind, because other pages link to the old path, and the
standard archival header (`parity-coverage-policy.md`'s "archiving a doc strips its imperatives"
rule) is prepended **in the same move** — 21 files were found missing it 2026-09-12, added in a
separate sweep because this step kept being skipped at move time.

## Other kinds

- `releases/` (12) — per-release records; `RELEASING.md` at the repo root is the authority on ritual.
- [`what-parity-gated-means.md`](what-parity-gated-means.md) — the reader-facing explanation of
  what a parity claim covers and does not; `parity-coverage-policy.md` and `parity-hunt-playbook.md`
  below are the internal detail on how it is established and chased.
- `parity-coverage-policy.md`, `parity-hunt-playbook.md` — how parity is established and chased.
- [`demand-evidence-2026-10-01.md`](demand-evidence-2026-10-01.md) — what Go developers struggle with, read from the issue trackers of the
  Go inference peers (yzma, Kronk, go-llama.cpp and others), Ollama, langchaingo and Charm's tools: 143 user-filed issues coded by hand into
  themes, set against goinfer's current posture. A dated evidence pass with its sampling limits stated, not a plan of record; it is not
  updated when the trackers move.
- `audit-<date>.md` — a whole-repo audit at a named commit; findings are dispositioned in place
  and the file moves to `completed/` when every one is closed. A large audit may split its closed
  findings out incrementally before that, the way the queue docs do — a
  [`completed/audit-<date>.md`](completed/) sibling holding the closed Critical/Gate/Major
  findings, with a summary note in the live doc pointing to it; the live doc is not empty and is
  not archived by this. Current: [audit-2026-09-10.md](audit-2026-09-10.md) (at `c7ef16a`; 187
  closed findings — every Critical, Gate, and Major finding, plus all but seven Minor and all but
  P-15's measurement half — split to
  [completed/audit-2026-09-10.md](completed/audit-2026-09-10.md) across two sweeps, 2026-09-12 and
  2026-09-16) and, focused on one backend, [audit-metal-2026-09-12.md](audit-metal-2026-09-12.md)
  (Metal, performance-led, at `da1e461`; M-/C-/G-/N- numbering is its own) and its redo,
  [audit-metal-2026-09-30.md](audit-metal-2026-09-30.md) (Metal, performance-led with an MLX-technique
  lens, at `844700f8`; area-letter numbering, and every Sep 12 ID is carried forward in its §12, so
  the older file stays as the record those IDs point into); the previous ones are
  [completed/audit-2026-09-02.md](completed/audit-2026-09-02.md) and
  [completed/audit-2026-08-05.md](completed/audit-2026-08-05.md).

## The one rule worth knowing before you cite anything

**A number in this tree carries its regime.** Backend, model size, quant, context length, machine
and date all change what a figure means, and several pages here were wrong at some point precisely
because a figure was quoted outside the conditions it was measured under. `book/11-knowing-youre-right.md`
is the collected version of how that has gone wrong, and `CLAUDE.md` at the repo root holds the
working rules that came out of it.
