# The docs, and how they fit together

`docs/` holds ~270 files. They are not one kind of thing, and reading them as if they were is the
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
| [how-inference-works.md](how-inference-works.md) | the same ground in ~2,300 words, anchored to specific source lines. The code map |
| [webgpu-primer.md](webgpu-primer.md) | orientation for anyone touching `gpu/` |

## Current state — what is true of the engine now

These are the pages to trust, and to update when reality moves.

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | modules, packages, the forward pass, where cgo is quarantined |
| [capability-matrix.md](capability-matrix.md) | **generated** from the `decoder` registry — 27 model families. The registry is the source of truth; do not hand-edit |
| [benchmarks.md](benchmarks.md) | every measured number, provenance-gated: machine, checkpoint, quant, date, thermal note |
| [server.md](server.md) | the HTTP surface — OpenAI, Anthropic, vision, embeddings, admin |
| [api-tiers.md](api-tiers.md) | which surfaces v1.0 semver-binds, and which are explicitly Experimental |
| [positioning.md](positioning.md) | what goinfer is for and is not — the long form of the README's framing |
| [capability-matrix.md](capability-matrix.md), [hardware-matrix.md](hardware-matrix.md), [env-vars.md](env-vars.md), [giw-bundles.md](giw-bundles.md) | generated or reference tables |

## Open work — four queues by success criterion

[`QUEUE.md`](QUEUE.md) indexes them and holds the cross-cutting material. An entry lives in
exactly one queue, keyed by *the question it answers*:

| queue | the question |
|---|---|
| [queue-performance.md](queue-performance.md) | how fast, how much memory — **empty as of 2026-08-31**; the closed record is [completed/queue-performance.md](completed/queue-performance.md) |
| [queue-correctness.md](queue-correctness.md) | does it compute the right thing |
| [queue-engineering.md](queue-engineering.md) | would we find out |
| [queue-release.md](queue-release.md) | can we tag |

## Design records — `task-*.md` (29)

Why a thing is built the way it is. **These are cited from 88 code comments**, which is why they
stay put rather than collapsing into queue entries: a queue entry cannot carry a design argument.
A `task-*.md` is not a claim that the work is open — read its status header.

`spec/` (13) is the same kind of thing for speculative decoding specifically, run as a numbered
series with pre-registered kill-gates.

## Evidence — `measurements/` (104)

Raw logs and per-run write-ups. A number in `benchmarks.md` should be traceable to one of these.
They are dated and machine-stamped by convention, and they are **not** updated when the world
moves — a superseded measurement stays as it was and the page that quotes it is what changes.

## Archive — `completed/` (62)

Finished work, kept for the reasoning rather than the outcome — including negative results, which
are archived with the same care as wins. **Nothing under `completed/` is scanned by the citation
lint**, so archiving a document also retires its citations from the live gate. When a document
moves here, a pointer stub is left behind, because other pages link to the old path.

## Other kinds

- `prompts/` (24) — briefs written for another session or the other machine to execute.
- `releases/` (8) — per-release records; `RELEASING.md` at the repo root is the authority on ritual.
- `scoping-*.md`, `plan-*.md` — pre-build scoping, some superseded; check the status header.
- `parity-coverage-policy.md`, `parity-hunt-playbook.md` — how parity is established and chased.

## The one rule worth knowing before you cite anything

**A number in this tree carries its regime.** Backend, model size, quant, context length, machine
and date all change what a figure means, and several pages here were wrong at some point precisely
because a figure was quoted outside the conditions it was measured under. `book/11-knowing-youre-right.md`
is the collected version of how that has gone wrong, and `CLAUDE.md` at the repo root holds the
working rules that came out of it.
