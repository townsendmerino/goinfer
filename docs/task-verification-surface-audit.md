# Audit: the verification surface itself

> **Opened 2026-08-27, after v0.15.0.** Not "are the tests passing" — they were. This asks whether
> the things that are supposed to CATCH problems actually do, because six of them were found not to
> in a single day, and the six are one class rather than six coincidences.

## Why this exists

| what claimed to work | what it actually did | found |
|---|---|---|
| `TestQwen3NextReal_oracle`, a REQUIRED gate | unreachable by the cell's `-run` regex for ~5 weeks | by chasing a release blocker |
| the int4 gpt2 cosine floor | sat *below* its own cross-arch baseline, failing healthy builds | by a sweep going red on a good tree |
| the T3 oracle bar | an int8 bar applied to the repo's first int4 run | by reading the bar's own comment |
| §B6's 48-cell harness | never existed; raw cells in a wiped scratchpad | by trying to re-run it |
| `bench_prompts_calibrate.py` | hardcoded a dead `/tmp` path — could not execute at all | by trying to run it |
| `queue_citation_lint.py` | crashed on a missing `go`, printed as a lint failure | by pushing from an ssh shell |

Every one is a mechanism meant to detect a problem, silently not detecting it. None was found by
the mechanism itself; all six were found by someone doing unrelated work and noticing.

## What is NOT wrong (checked first, so the audit does not invent a crisis)

**The parity checkset design is sound.** It classifies four outcomes — PASS / FAIL / SKIP / MISSING
— and blocks on three, with MISSING called out as "the one a tally would miss". It reported the
qwen3next oracle as a blocker correctly, and continuously.

**So the machinery worked and the LABEL failed.** "DID NOT RUN" reads as "the asset is missing",
which is where every investigation went — including three separate sessions, one of which verified
41/41 shards of a 163 GB checkpoint that was fine the whole time. The gate could not reach the test.

⇒ **A blocker message must name the cause it can actually distinguish.** "DID NOT RUN" cannot tell
an absent asset from an unreachable test, and it defaulted to implying the first.

## Finding 1 — gate reachability (FIXED, `20fc6e6`)

The realckpt cell selected on `-run "Qwen35|Real_gate"`; the test is `...Real_oracle`. Of ten
required gates it was the only one the filter missed. `TestRealckptCellCanReachEveryGate` now
compiles the cell's actual pattern and asserts it matches every gate in the required list;
`TestBaseCellIsUnfiltered` guards the other cell. Mutation-checked.

**Generalisable shape:** two definitions that must agree, in different files, with nothing comparing
them. Worth grepping for others.

## Finding 2 — loose bars without a recorded basis

`scripts/audit_gate_bars.py` inventories numeric tolerances that gate a result:

    loose tolerances (< 0.999): 32 | basis recorded: 9 | bare: 23

A tight bar (≥ 0.999) means "these should agree" and needs no defence. A loose one encodes a
judgement about acceptable error, and that judgement is only checkable if the population it was
calibrated on is written beside it.

**Calibrated honestly, this is a traceability gap, not 23 live defects.** The structural danger is
one bar shared across populations — which was the oracle case and is fixed. The remaining 23 are
per-test, and some files already do the right thing: `decoder/gguf_forward_test.go` gates
`int8_resident` at 0.99 and `int4_resident` at 0.98, differentiating by precision exactly as the
oracle bar failed to. **The repo knew this in one place and not another**, which is the real lesson.

**First pass over-counted (25) because the heuristic only walked contiguous comments directly above
the compare** — and the justification usually sits above the enclosing `if cos := ...`, one
non-comment line further up. The committed script uses a 6-line window. Recorded because a
verification audit that miscounts is the thing it is auditing.

## Finding 3 — numbers travel out of their regime, and the fix is a label

Three instances in one day, which makes it a property of how measurements move through these docs
rather than three slips:

| a number from | got applied to | cost |
|---|---|---|
| an int8 W8A8 bar (`realLogitOracleQuant`) | the repo's first int4 T3 | five weeks of a blocker, resolved as G25 |
| a block drafter's break-even, `decoder/blockspec.go:521` | an MTP round, whose draft term scales with K | caught in review, before it reached a gate |
| a 0.8B `c_head/c_decode` ratio | *would have been* read as a break-even for a 27B trunk | caught pre-emptively in `docs/spec/09-mtp-heads.md` |

The first cost real time. The second and third were caught by someone asking which population a
number came from — which is not a mechanism, it is luck plus attention.

**The convention that generalises: label a number's regime AT THE POINT OF RECORDING, not at the
point of reading.** Every one of these was recoverable from context by a reader who happened to
check; none announced itself. Concretely, a recorded measurement carries the population it was taken
on — precision, geometry, model scale, the pairing — in the same line or table cell as the value, and
a number that cannot be evaluated in a regime says so where it is written rather than where it is
quoted. 09's `c_head/c_decode` is the worked example: measured, recorded, and labelled
**non-transferable** in the same breath.

This is the same defect as Finding 2 seen from the other end. Finding 2 counts bars whose basis was
never written down; this one is what happens next, when a bar outlives the population it was
calibrated on and nothing on the line says so. `scripts/audit_gate_bars.py` inventories the first;
the convention above is what would stop the second.

## Method for continuing

1. **Reachability.** For every checkset, prove each named gate can be selected. Done for parity;
   `census`, `heavy`, `composition`, `selector`, `gpu`, `mutation` are unchecked.
2. **Bars.** Walk the 23. For each: what population, measured when, still that population? Record
   beside the bar. Prefer per-precision/per-geometry tables over one scalar (G25).
3. **Harnesses.** Every measurement in `docs/benchmarks.md` should name a committed, runnable
   harness. §B6's did not exist; §B4's is a Go test and does. §B7's and §B's are unverified.
4. **Diagnostics.** Where a check reports a blocker, does the message distinguish the causes it can
   actually tell apart? "DID NOT RUN" did not. The citation lint's `go`-missing crash did not.

## Not in scope

Making any of this a hard gate. Requiring every bar to be documented is a policy decision, and this
audit is an inventory. `audit_gate_bars.py` reports and never fails.
