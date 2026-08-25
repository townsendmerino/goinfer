# goinfer task: adaptive speculation — width controller, grammar prior, MTP spike

> **Box:** `linux` (the drafter path is CUDA-resident; all baselines below were measured there).
> **Stimulus:** a llama.cpp fork shipping acceptance-driven draft-length adaptation
> (`LaurentZuijdwijk/llama.cpp`, `--spec-draft-adaptive --spec-draft-n-min/--spec-draft-n-max`,
> reporting up to +50% on structured content on a Strix Halo), and mainline PR
> `ggml-org/llama.cpp#27210` (MTP-only variant). Verify both at pickup — the post is fresh and
> either may have moved. The idea is one goinfer's own sweeps already validated; this task takes
> the improvement without taking the number (an APU result does not transfer to a 2070).
>
> **Read first:** the guard work (`e4cb262`, `420e0fd`, `815a513`), the width sweeps (`11aeed4`,
> `0f7e61a`, `048f8e9`, lever filed at `e09be54`), the shipped drafter surface
> (`decoder/blockspec.go` / `BlockSpec.GenerateStream` `5f0b60d`, `serve --drafter` `0ce7724`),
> and the grammar-aware speculation harness that already exists (`decoder/spec_grammar.go`,
> `spec_grammar_parity_test.go`).

## Baselines to hold (re-measure FIRST, same session, interleaved)

Absolute numbers on this box drift ~3.5% between sessions (`docs/benchmarks.md` §measurement
notes), so every verdict below is against a **same-session interleaved baseline**, never against
these historical figures:

- End-to-end `--drafter`: code **1.44–1.60×**, math **1.50–1.79×**, chat **0.61× unguarded /
  0.92× guarded** (`133619c`, `d2cb052`, `e8ae719`).
- Static-width optima per suite: **math 8, code 7, chat 4** (`11aeed4`); code at width 7
  projected 1.74× vs 1.28× at 16 (`0f7e61a`). Front-loading is **per-pairing** (`048f8e9`) —
  re-derive the optima for whichever pairing you run, don't inherit them.
- Settled negatives that stay settled: CPU **sidecar** drafting loses at every width
  (0.75–0.82×, `88cd374`, `4d07a21`); Metal batched verify is a NO-GO (`f96f889`). Neither is
  reopened by this task except where Phase 3 explicitly says so.

## Phase 1 — the width controller (subsumes the guard)

**What.** Replace the binary acceptance guard with a controller that moves block width within
`[min, max]` between rounds, hitting `min` before turning drafting off — one mechanism, and the
guard's magic threshold (2.5, `e4cb262`) becomes its floor case.

**Design constraints, from measured lessons — re-derive freely, but against these:**

1. **Damped signal.** The 6→3 window experiment backfired (`e4cb262`): a fast loop chases
   acceptance noise and gives back the gains. Start from the guard's cumulative average (or an
   EWMA with a long half-life); no raw per-round reactions.
2. **Shrink fast, grow slow, with hysteresis.** A rejection wastes a whole verify round; an
   over-wide draft wastes draft+verify compute — the losses are asymmetric the same way
   split-KV's were, and the remedy is the same: round toward the cheap error. Grow by +1 after
   N consecutive rounds above the high-water mark; shrink hard on a round below the low-water
   mark; high > low.
3. **Per-pairing initial width** (the `048f8e9` finding), clamped to `[min, max]` from flags on
   `serve --drafter` (flag surface directly, the D3 pattern — no new env vars).
4. **Lossless is non-negotiable and already free.** Verify admits only tokens the target would
   produce, so any width schedule yields byte-identical output. Extend the existing invariance
   test to assert it across a *changing* schedule (force a schedule that exercises min, max, and
   both transitions), so the property is gated, not assumed.

**Gates to ship (default-on only if all pass; else it lands behind a default-off flag):**

- Adaptive ≥ the best **static** width per suite, all three suites, same-session interleaved —
  the D3b rule: a default changes only on the measurement its own comment demands.
- A **mixed-content suite** (new; build it): prompts that force thinking-prose → structured
  transitions mid-generation. This is the regime the whole idea exists for and no current suite
  exercises it; the claimed win lives here. Target: beat both static arms, the §B6.1
  gemma3-3900 shape ("no static value can reach this point").
- Chat ≥ the guarded 0.92× (the floor case must not regress the guard it replaces).
- The schedule-invariance (lossless) gate green.

## Phase 2 — the grammar prior (the part the fork cannot have)

Their loop *infers* "structured now" from trailing acceptance, with lag. goinfer *knows*: when a
`constrain` grammar is active (JSON schema `response_format`, constrained `tool_choice`), output
is structured by construction — the high-acceptance regime, known at request time, per token.

- Feed LogitProcessor state into the controller as a **prior**: grammar active → start at (or
  step immediately to) max; grammar released mid-generation → fall back to the Phase-1 signal.
  The existing grammar-aware spec path (`spec_grammar.go`) is the integration point — the drafter
  side must see the same mask, which that harness already handles; extend, don't fork it.
- Same for the thinking-mode signal `serve` already detects at template level (`815a513`):
  thinking span → prior toward min, instead of only warning at startup.
- **Gates:** constrained-JSON and constrained-tool-call suites, adaptive-with-prior ≥
  adaptive-without (isolates the prior's contribution); plus a correctness gate on tool-call
  traffic under speculation — the thread's warning ("looks like the model chose not to use the
  tool") is exactly the class our constrained tool calls exist to prevent; prove the combination,
  don't assume it.

## Phase 3 — MTP self-drafting spike (separate, kill-gated; do not start before 1 ships)

Qwen3.8 (`qwen3_5`) and the DeepSeek-V3 family ship **MTP head tensors goinfer currently skips
at load**. Drafting from the model's own MTP head means: no sidecar checkpoint, no pairing
dialect table, weights already in the file — and a much cheaper draft than any sidecar.

- **3.0** Read the shipped checkpoints: which MTP tensors exist (Qwen3.8-27B, a DeepSeek-V3-class
  model), shapes, and what HF/the fork's `draft-mtp` actually computes with them. Document before
  estimating (the G6 phase-0 discipline).
- **3.1** Load the head and wire it as a self-drafter through the existing seam
  (`BlockDrafterWeights`, `5045364`, is the interface shape). **Kill-gate 1** (the `ae7ac75`
  pattern): drafted ids match a reference MTP forward end-to-end, and the gate is shown able to
  fail.
- **3.2** **Kill-gate 2:** measured acceptance ≥ break-even on one suite, GPU-resident, before
  any further build. Below it: stop, record the verdict in the queue (the `4d07a21` precedent —
  "works, lossless, 0.82× — DO NOT SHIP" is the house norm for exactly this moment).
- **3.3** The one settled question this legitimately reopens: **CPU** break-even. The sidecar
  loss (`e008138`: CPU drafting loses >10×) priced a full drafter forward; an MTP head is a
  fraction of that. One afternoon, numbers decide, `WhoRoger`'s question answered either way.

## Telemetry (ship regardless of the above)

Expose per-request accept-rate and tok/verify in `serve`'s response metadata/metrics — "accept
rate first, then tok/s" is the debugging norm the thread converged on, and we already compute
both. Cheap, and it makes every later report checkable by users.

## Filed, NOT built: a third shape — the DERIVED default

Opened by the width sweep (`docs/measurements/default-verify-width-2026-08-25.md`), which found
the optimum is **7 at int4 and 6 at int8** on one pairing — and found a *mechanism* for it, not
just a correlation: optimal width is set by the ratio between a plain decode step and a batched
verify, and target quant moves exactly that ratio (int8 decode ran ~half int4's rate here).

If that ratio is what determines width, then the right long-term shape may be **neither a
constant nor Phase 1's feedback controller**, but a width **derived once at startup** from the
measured-or-estimated cost ratio for the active config. No runtime adaptation, none of the
controller's failure modes (it measured worse than both endpoints it moved between), and it
dissolves the "no single constant is right" problem the sweep pre-registered and then hit.

**Scoped, not scheduled.** It becomes interesting the moment the Mac pairing
(`mac-default-verify-width.md`) lands and either confirms the ratio story or breaks it — if a
second pairing's optimum does NOT track its cost ratio, the mechanism is wrong and this idea
dies with it. Cheap first probe if it survives: a startup micro-benchmark of one decode step
against one k-wide batched verify, width chosen from the measured ratio, scored against the best
static width per config on the same harness the ship-gates already use.

## Non-goals

Metal batched verify (NO-GO stands) · CPU **sidecar** drafting (settled; only 3.3's MTP variant
reopens CPU) · tree/multi-branch speculation · any change to sampled-path semantics · chasing the
fork's +50% figure on unlike hardware.

## Budget and stopping rule (T2 discipline)

Phase 1 ≈ two sessions, Phase 2 ≈ one, Phase 3 ≈ two with its kill-gates. Any phase exceeding
its estimate by half: stop, report, take the decision. A red found on the way gets triage-before-
diagnosis — "can any shipped path reach this?" — before any mechanism hunt.

## Deliverable

Controller + prior landed per their gates (default-on only if earned); the schedule-invariance
gate; the mixed-content suite committed to the harness; accept-rate telemetry; `e09be54`'s
verify-width queue lever closed with the measurements; a `docs/measurements/` note recording
per-suite adaptive-vs-static, the mixed-suite result, and Phase 3's kill-gate verdicts (whatever
they are); queue rows filed for anything discovered and deliberately not done.
