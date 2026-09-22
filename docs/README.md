# The docs, and how they fit together

`docs/` holds ~470 files. They are not one kind of thing, and reading them as if they were is the
main way people get a wrong answer here: a **design record** explains why something is built as it
is, a **queue** holds what is still open, a **measurement** is evidence with a machine and a date
on it, and an **archive** is finished work kept for its reasoning. Only some of them are current
claims about the engine.

This page is the map. [`QUEUE.md`](QUEUE.md) is the map of *open work* specifically, and is the
better starting point if you are picking something up.

## Start here

| | |
|---|---|
| [**book/**](book/) · [read online](https://townsendmerino.github.io/goinfer/) | eleven-chapter inference primer for Go engineers — concepts from zero, each chapter ending in a measured number |
| [task-download-and-load.md](tasks/task-download-and-load.md) | which checkpoints to download and what a load actually costs — plus why the load is compute-bound, not storage-bound |
| [quantization.md](quantization.md) | which quants goinfer stands behind, which it measured and refused, and where there is no evidence — reading a format is not endorsing it |
| [how-inference-works.md](how-inference-works.md) | the same ground in ~2,300 words, anchored to specific source lines. The code map |
| [webgpu-primer.md](webgpu-primer.md) | orientation for anyone touching `gpu/` |

## Current state — what is true of the engine now

These are the pages to trust, and to update when reality moves.

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | modules, packages, the forward pass, where cgo is quarantined |
| [capability-matrix.md](capability-matrix.md) | **generated** from the `decoder` registry — 36 model families as of 2026-09-12. The registry is the source of truth; do not hand-edit |
| [benchmarks.md](benchmarks.md) | **current claims only**, provenance-gated: machine, checkpoint, quant, date, thermal note. Section IDs are stable; a *Retired section IDs* index maps the ones that moved |
| [legacy-benchmarks.md](legacy-benchmarks.md) | the retired, superseded and historical rows `benchmarks.md` used to carry — moved verbatim 2026-08-31 and 2026-09-05 (was `benchmarks-archive.md`). **Never a current claim**; kept so a retraction can be audited |
| [server.md](server.md) | the HTTP surface — OpenAI, Anthropic, vision, embeddings, admin |
| [api-tiers.md](api-tiers.md) | which surfaces v1.0 semver-binds, and which are explicitly Experimental |
| [positioning.md](positioning.md) | what goinfer is for and is not — the long form of the README's framing |
| [roadmap.md](roadmap.md) | direction — where we are, the promotion gate, open programs by owner, what is parked and why, the v1.0 checklist; the June 2026 roadmap is archived |
| [next-models.md](next-models.md) | which model families next, and why; what became of the last list (was `post-v1.0-models.md`) |
| [capability-matrix.md](capability-matrix.md), [hardware-matrix.md](hardware-matrix.md), [env-vars.md](env-vars.md), [giw-bundles.md](giw-bundles.md) | generated or reference tables |

## Open work — five queues by success criterion

[`QUEUE.md`](QUEUE.md) indexes them and holds the cross-cutting material. An entry lives in
exactly one queue, keyed by *the question it answers*:

| queue | the question |
|---|---|
| [queue-performance.md](queue-performance.md) | how fast, how much memory — **four open items as of 2026-09-05** (P20–P23; was briefly empty on 2026-08-31); the closed record is [completed/queue-performance.md](completed/queue-performance.md) |
| [queue-correctness.md](queue-correctness.md) | does it compute the right thing — one open item (G12, LFM2 GGUF loader) plus one PARKED item (G8, unvalidatable on hardware here); closed entries in [completed/queue-correctness.md](completed/queue-correctness.md) |
| [queue-engineering.md](queue-engineering.md) | would we find out |
| [queue-release.md](queue-release.md) | can we tag |
| [queue-presentation.md](queue-presentation.md) | what a user sees, reads, or can find — filed 2026-09-15 |

## Design records — `task-*.md` (36: 31 in `tasks/`, 5 in `tasks/parked/`)

Why a thing is built the way it is. **These are cited from 88 code comments**, which is why they
stay put rather than collapsing into queue entries: a queue entry cannot carry a design argument.
A `task-*.md` is not a claim that the work is open — read its status header.

**`tasks/` is the home for these** (`tasks/parked/` for a doc-review PARKED verdict — blocked,
trigger named). Migration finished 2026-09-13: every `task-*.md` that was still live moved out of
the flat `docs/` root during a full `/doc-review` sweep, on top of the 7 moved a few hours earlier;
`docs/README.md`'s own migration note above this line is now historical, not a live caveat — a new
`task-*.md` doc should be filed directly into `tasks/` (or `tasks/parked/`) rather than the root.
One exception, by design: [`task-demo-refresh.md`](task-demo-refresh.md) stays in the root as a
two-line pointer stub (the `docs/plan-still-slow.md` shape) because its real content already lives
at `docs/completed/task-demo-refresh.md` and other pages still link the old path — it is not a
design record itself, so it is not counted above.

One is a program rather than a single design, and is named without the `task-` prefix:
[`red-october.md`](tasks/red-october.md) — the backend × area performance gap matrix (CPU / WebGPU /
CUDA / Metal against the peers, as of 2026-09-18), the mechanism behind each cell, the registered
projection bands, and the twelve briefs R1–R12 that would close them. It is the one place the whole
chase is on a page; `benchmarks.md` stays the source of every ratio it quotes.

[`task-never-swap-2026-09.md`](tasks/task-never-swap-2026-09.md) (S0–S6, filed 2026-09-22) is its
memory-side companion: what is anonymous and what is file-backed on each load path (why a plain
`.gguf` load and a Metal resident build swap on the MacBook and a CPU `.giw` load does not), and the
briefs for file-backed weights by default, a swap tripwire in the binary, the fit guard on the
`.giw` path, Metal aliasing the mapping instead of copying it, and a firm cap for the MoE pager
where darwin allows one.

One is the outside view rather than a design: [`task-first-hour.md`](tasks/task-first-hour.md)
records what a cold user hit against a published tag, what was fixed, and the protocol for running
it again — `RELEASING.md`'s pre-flight now calls for one before each release.

`spec/` (13) is the same kind of thing for speculative decoding specifically, run as a numbered
series with pre-registered kill-gates.

## Evidence — `measurements/` (79)

Raw logs and per-run write-ups. A number in `benchmarks.md` should be traceable to one of these.
They are dated and machine-stamped by convention, and they are **not** updated when the world
moves — a superseded measurement stays as it was and the page that quotes it is what changes.

## Archive — `completed/` (118)

Finished work, kept for the reasoning rather than the outcome — including negative results, which
are archived with the same care as wins. **Nothing under `completed/` is scanned by the citation
lint**, so archiving a document also retires its citations from the live gate. When a document
moves here, a pointer stub is left behind, because other pages link to the old path, and the
standard archival header (`parity-coverage-policy.md`'s "archiving a doc strips its imperatives"
rule) is prepended **in the same move** — 21 files were found missing it 2026-09-12, added in a
separate sweep because this step kept being skipped at move time.

## Other kinds

- `prompts/` (6) — briefs written for another session or the other machine to execute. 21 of the
  original 25 were archived to `completed/` in a 2026-09-13 sweep once verified delivered; the 4
  remaining are each genuinely still open (drafted-by-decision, blocked, or unshipped, per their
  own status). Newest: `prompts/cuda-r6-flash-decode.md` (2026-09-19), the Linux kickoff for
  `red-october.md`'s R6.
- `releases/` (8) — per-release records; `RELEASING.md` at the repo root is the authority on ritual.
- `scoping-*.md`, `plan-*.md` — pre-build scoping, some superseded; check the status header.
- [`what-parity-gated-means.md`](what-parity-gated-means.md) — the reader-facing explanation of
  what a parity claim covers and does not; `parity-coverage-policy.md` and `parity-hunt-playbook.md`
  below are the internal detail on how it is established and chased.
- `parity-coverage-policy.md`, `parity-hunt-playbook.md` — how parity is established and chased.
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
  (Metal, performance-led, at `da1e461`; M-/C-/G-/N- numbering is its own); the previous ones are
  [completed/audit-2026-09-02.md](completed/audit-2026-09-02.md) and
  [completed/audit-2026-08-05.md](completed/audit-2026-08-05.md).

## The one rule worth knowing before you cite anything

**A number in this tree carries its regime.** Backend, model size, quant, context length, machine
and date all change what a figure means, and several pages here were wrong at some point precisely
because a figure was quoted outside the conditions it was measured under. `book/11-knowing-youre-right.md`
is the collected version of how that has gone wrong, and `CLAUDE.md` at the repo root holds the
working rules that came out of it.
