# Performance queue

> **EMPTY as of 2026-08-31.** Every item that was here is closed, refuted, withdrawn, or has moved
> to the track that owns it. This file is live — new performance work is filed here — it just has
> nothing in it right now.
>
> **The closed record is [`docs/completed/queue-performance.md`](./completed/queue-performance.md)**
> (2,245 lines, G15–G24 · P1–P16 · A1–A11 and the A9-\* series). It moved rather than being deleted
> because the reasoning is the point, negative results included. Per `docs/README.md`'s archival
> rule, `completed/` is not scanned by the citation lint, so archiving it also retired its citations
> from the live gate — that is deliberate, not an oversight. Pages that linked to
> `docs/queue-performance.md` still resolve here.

## What is open

Nothing.

## Where the last items went

| item | disposition |
|---|---|
| **P16** · re-anchor to Nobara 44 / driver `595.91.07` | **DONE 2026-08-31.** Six legs were already re-anchored on 2026-08-27; the seventh (the v0.11.0 qualification) was **retired** rather than re-measured — its numbers were §B6/§B7 by code-identity, and both are current. |
| **P14** · the CPU gap is kernel arithmetic | **DONE.** Items 1+2 refuted (centering), item 3 built, wired, measured (+2.10% decode) and **parked default-OFF** against its pre-registered 4% bar. |
| **A2** · 26B documentation correction | **DONE 2026-08-31**, published as `docs/benchmarks.md` §B4.2. |
| **A10** · the ~150 MiB allocation floor | **CLOSED 2026-08-12** — fully decomposed, nothing unattributed. |
| **D3b** · the expert-cache default | **SHIPPED 2026-08-20** (`8f3c5e7`). Lives in `docs/queue-release.md`. |
| **P10** · DSpark / DFlash block drafters | **MOVED to the speculation track** — see below. Not finished; not performance-queue debt. |
| **P15** · DFlash 2 | **MOVED to the speculation track** — see below. |

## P10 and P15 moved rather than closed, and the distinction matters

They are **not done**, and nothing here should be read as saying they are. They left this queue
because they were never really performance-queue items: the rest of this queue was *find a
bottleneck, measure it, fix or refute it*, and those two are an ongoing research program with
pre-registered kill-gates.

Their substance already lived in [`docs/spec/08-dspark-dflash.md`](./spec/08-dspark-dflash.md)
(~25.5k words — the kill-gates, the increment log, the licensing correction, the Metal verdict);
the entries here were a second, thinner copy that could drift from it. The spec track owns them,
[`docs/spec/README.md`](./spec/README.md) indexes them, and
[`docs/spec/experiments.md`](./spec/experiments.md) is the dated run log.

**Open state, carried over so it is not lost in the move:**

- **P10 · increment 4.** Kill-gates 1 and 2 cleared 2026-08-15 (6.78 tok/verify against a ≥3.0
  bar). Remaining: gate 3 — end-to-end **≥1.3× vs plain resident decode on ≥1 GPU backend** — and
  gate 4's mixed-workload width router. The **Metal leg is measured not-ready**: ~1.13× ceiling
  even at `draft_ms=0`, and `PrefillLast` is not bit-identical, so the lossless contract cannot be
  met there. `gpt-oss` is blocked on a missing harmony chat template, not on the seam.
- **P15 · DFlash 2.** Filed 2026-08-20, **gates before code**, not started. Sequenced to land
  **before** P10's gate-4 width router — doing gate 4 first would mean redoing it.

## Filing a new item

Read `docs/completed/queue-performance.md` first — it is 2,245 lines of what was already tried,
including the negative results, and several of its entries exist because something was rebuilt that
had already been measured and rejected. Then follow the same discipline the archive does: state the
mechanism, pre-register the decision rule and its ambiguous band, include the do-nothing arm, and
record a negative with the same care as a win.

**One recurring defect that archive documents four times over, worth knowing before you add
anything:** an entry gets its resolution *appended* while the stale conclusion is left standing in
verdict position, so the item reads as open long after it closed. A2 (a pre-registration answered
four days earlier), D3b (shipped eleven days earlier), A10 (resolved, header still said OPEN) and
P16 (a stale-list four items out of date) all failed this way. **When you close something, correct
the sentence a scanner stops at — not only the body.**
