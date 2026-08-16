# Release-path restructure — T0–T5

**Status:** filed 2026-08-16, uncommitted when written. Decided 2026-08-12 by Francis
as "the first thing after the release." v0.13.0 has since shipped; this was never
recorded and was reconstructed from the conversation that produced it.

---

## T0 — why this exists

The v0.13.0 tag took two days. The evidence, recorded so it is not re-litigated at
the next red:

**Day one found real defects** — the 26B slot-cap bug, the goldens covering f32 only,
an aikit decode regression fixed upstream the same day, 35 dropped errors on the
production forward path.

**Day two was almost entirely the CUDA tier and found zero production defects:**
three test defects, one test-interaction phenomenon (A13) that no shipped path can
reach, and a large amount of careful measurement establishing exactly that.

The rigor was not the problem. Nothing throttled it, and it was pointed at something
that could not affect what ships.

---

## T2 — triage before diagnosis · DO FIRST, cheapest and highest leverage

Rule, into `RELEASING.md` and `docs/parity-coverage-policy.md`:

> When a gate goes red, the first question is NOT "what causes this". It is
> **"can any shipped path reach this?"** — answered by enumerating the shipped
> paths against the trigger, not by hunting mechanism.
>
> - reachable → mechanism work is justified; it blocks the tag
> - not reachable → file it with the trigger named, do not block, investigate on
>   its own schedule

Evidence: A12/A13's enumeration took about an hour and settled the tag question. It
was done on day two, after the mechanism hunt. Doing it first would have cost an hour
instead of a day.

**Acceptance:** the rule is written, and the next red is handled this way with the
triage recorded *before* any mechanism work begins.

## T1 — instruments are not gates

`TestAllocFloor`, the A10 probes, the reservation and VRAM sweeps deliberately create
device states no shipped path creates. They are diagnostic instruments that happen to
be written as Go tests, and they sit in the same package and the same gate as the
parity tests. **All three original poisoners were instruments.**

- move them to their own package (`cuda/instruments/` or a build tag), excluded from
  the gate and from CI
- split criterion, written down: *does this test deliberately create a device state
  no shipped path creates?* If yes, it is an instrument
- the marked drainers are the seed list; the derived free-VRAM-at-boundaries check
  catches new ones (see T5)
- instrument results go to `docs/measurements/`, not to a gate verdict

**Acceptance:** the gated package contains no test that drives the device to refusal.
Instruments run on demand and cannot block a tag.

## T3 — four gate outcomes · PARTLY DONE

`docs/parity-coverage-policy.md` already defines pass / fail / cannot-evaluate /
first-run. What remains is enforcement: only **fail** blocks a tag, cannot-evaluate is
counted and named with its reason and its env var or missing asset, first-run routes
to triage and never to the tag decision.

Evidence: a test reporting "resident declined, this run says nothing" blocked a
release. A correct decline on an 8 GB card blocked a release.

Applies to the gate runner, the goldens refresh, the parity sweep, the heavy tier.

**Acceptance:** mutation-checked both ways — a missing asset yields cannot-evaluate,
a wrong number yields fail.

## T4 — re-tier by cost · this is B3, promoted

smoke / standard / heavy, with **measured** wall-clock for each printed by the gate,
not estimated. `RELEASING.md` names which tiers gate a tag. Target: smoke carries most
of the signal in about two minutes.

Evidence: every iteration of the A12/A13 investigation cost a full run, 26 minutes to
two hours. B3 has been queued since the beginning.

## T5 — the gate owns its environment

a. **co-tenancy check at start:** enumerate processes holding device memory; if any,
   refuse with cannot-evaluate naming them rather than producing an ambiguous result.
b. **derived unmarked-drainer assertion:** record free VRAM at test boundaries and
   fail if an *unmarked* test leaves free below a floor. Turns the marker's blind spot
   from "requires inspection" into "fails on the run that introduces it."

Evidence: an SSH session on the GPU confounded a gate run and the gate had no way to
say so. An unmarked drainer ran in the main tier as though harmless and was found one
run later, by luck.

---

## Budget and stopping rule

This restructure is capable of becoming its own two-day item, which would be the
failure it exists to fix.

- T2 is an afternoon. T1 and T3 are a day between them. T4 and T5 are a day.
- If any single item exceeds its estimate by more than half, **stop, report, and take
  the decision** rather than continuing.

**Not in scope: splitting the repo.** The coupling that cost the time was between
kinds of test inside one package, not between modules. The five-module structure is
fine; the test taxonomy is what is missing.

## Not a code change — how the reviewing side operates

Recorded because it is half the cause: across day two, nearly every round ended with a
proposed next measurement, each individually justified, collectively a day, with no one
asking "does this change what we ship?" until the question was raised directly.

The same rule applies on that side: triage before diagnosis, and a red that no shipped
path can reach gets filed rather than chased.
