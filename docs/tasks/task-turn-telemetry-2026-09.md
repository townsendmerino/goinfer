# Task: what a turn actually cost — per-generation telemetry on the wire (X1–X6) — 2026-09

> **Status: SCOPED 2026-09-17, unstarted. X7/X8 added and the build order reversed the same day.**
>
> goinfer measures itself better than any peer and tells a *caller* almost none of it. Per
> generation the API reports token counts and one vendor extension
> (`usage.prefill_reused_tokens`); it reports no timing, and nothing about the mechanisms the
> repo's own primer spends eleven chapters on. This doc puts the facts the engine already knows
> onto the wire, as ignorable vendor extensions.
>
> **Three consumers, which is why this is a server item and not a UI one:**
> [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) **W34** (the per-turn breakdown, and the only
> one that needs it to be pretty), [`task-peer-benchmarks.md`](task-peer-benchmarks.md) (per-stage
> columns currently reconstructed from outside the process, when the process knows exactly), and
> any harness that wants a TTFT it did not have to time itself.

---

## 0. Build order

**X7 first, then X8, then X1.** X7 and X8 are static or near-static facts the engine has already
decided — no clocks, no per-token work, nothing that can slow decode — and between them they cover
Chapters 1, 2, 3, 4, 5, 6, 10 and 11. X1's timing block covers Chapters 8 and part of 4 and costs
the most care, because it is the only part that instruments the hot path.

This reverses the order these items were first filed in on 2026-09-17, on the reasoning above.

## 1. What exists today

- **`usage`** carries `prompt_tokens`, `completion_tokens`, `total_tokens`, plus
  **`prefill_reused_tokens`** — a documented vendor extension whose own comment states the
  ambiguity this doc should fix: *0 on a cold prefill, on a path that doesn't track reuse, or when
  nothing was reused — those are indistinguishable from the number alone.*
- **`/v1/models`** carries `decode_path` and `prefill_path` (the same vendor-extension convention)
  and `context_window`.
- **`/v1/jobs`** carries queue position and state (J1/J3).
- **No timing at all.** Not TTFT, not the prefill/decode split, not queue wait — though the server
  is the only place each of those is exactly knowable.
- **Mechanism counters exist but stay inside the process.** Speculative-decode acceptance is
  measured in `decoder`'s own tests (`decoder/dflash_accept_test.go`, `decoder/eagle_accept_test.go`)
  and the MoE expert pager tracks its cache, but neither reaches a caller.

## 2. Ground rules

1. **Vendor extensions, absent when not measured.** Same convention as `decode_path` and
   `prefill_reused_tokens`: additive, ignorable by a standard OpenAI or Anthropic client, and
   **omitted rather than zeroed** when the path did not measure them. Zero must mean zero. This is
   the correction `prefill_reused_tokens` needs and the rule every field here inherits.
2. **Experimental until v1.0.** No standard field changes shape or meaning.
3. **Measurement must not cost measurable time.** Stage boundaries only — never a clock per token.
   A telemetry feature that slows decode would be the joke writing itself in a repo whose whole
   subject is decode speed. X6's gate is the one that enforces this.
4. **One shape, every route.** A field that appears on `/v1/chat/completions` but not on
   `/v1/messages` or `/v1/jobs` forces every consumer to special-case, and the UI is a consumer.

## 3. The items

### X1 — the timing block
A `timings` object beside `usage`: `queue_wait_ms`, `prefill_ms`, `decode_ms`, `ttft_ms`.

- Measured at stage boundaries the server already crosses — admission granted, prefill returns,
  first token emitted, generation ends.
- `ttft_ms` is server-side (admission to first token), so it excludes the network and is the honest
  complement to a client's own measurement rather than a duplicate of it.
- This is what lets W34 show a prefill *duration* at all; until it exists the UI is under
  instruction to show a token count and no rate.

### X2 — make the prefill split unambiguous
`prefill_reused_tokens` stays; add `prefill_computed_tokens` and make **both absent** on a path
that does not track reuse. A caller can then tell "nothing was reused" from "nobody counted",
which today it cannot.

### X3 — speculative decoding, on the wire (Chapter 9)
When a drafter is attached: `draft_tokens_proposed`, `draft_tokens_accepted`, the rounds count, and
**the break-even α for the configuration in force**. The acceptance figure currently exists only in
test logs. Ship it *with* its break-even: Chapter 9's own conclusion is that speculative decoding
"is not a speedup, it is a bet on α", so α alone is a number while α against break-even is the
idea — and the difference decides whether a user should keep the drafter attached. Absent when
`-drafter`/`-spec` is off.

### X4 — MoE expert cache, on the wire (Chapter 7)
When the expert pager is active: hits, misses, the slot count in force, and **expert transfer as a
share of decode time**. This is the number that explains a paged MoE's decode rate, and it is the
one a user staring at 1.4 tok/s most needs. The share is what makes it Chapter 7 rather than a
counter — that chapter's measured result is that roughly half of a token is expert transfer on the
constrained GPU, and a share is the only form in which a user can recognise it. It also makes the
chapter's stranger claim checkable from the chair: content determines cost, so two prompts on the
same model give different hit rates. Absent on a dense model or a fully-resident MoE.

### X5 — the same shape everywhere
`/v1/chat/completions` (streamed — in the final usage chunk — and buffered), `/v1/completions`,
`/v1/responses`, `/v1/messages`, and the job object `/v1/jobs/{id}` so W27's re-attach sees the
same facts a live stream would. Where a route's own schema has a natural home (Anthropic's
`usage`), use it rather than inventing a parallel one.

### X6 — the contract, documented and gated
- `docs/server.md` gains the fields; `docs/api-tiers.md` records them as Experimental extensions.
- **Gates:** a standard OpenAI SDK round-trip ignores every new field; absent-not-zero is asserted
  per field on a path that does not measure it; and an **overhead gate** — decode tok/s with
  telemetry on is within noise of the same build with it compiled out, measured the way this repo
  measures anything (paired, same session, on the quiet box). If it is not within noise, the
  design is wrong and gets coarser, not shipped.

### X7 — the model card: what was decided at load (Chapters 1, 2, 5, 6, 10, 11)

**Build this first — see the order note in §0.** Everything here is resolved once, at load, so it
costs nothing per token, needs no clock, and is not subject to X6's overhead gate at all. It also
covers six chapters where X1–X4 together cover four.

**The gap it closes is an asymmetry, not an absence.** `internal/serveapp/banner.go` already prints
most of this at startup — decode path, prefill path, the context cap and KV precision, the features
line, and session reuse with the reason when it is off ("resident decode is stateless, so every turn
re-prefills its whole prompt"). A terminal user is told. A browser user gets a model name and a
decode path. Since the web UI is now the product surface (owner decision 2026-09-14), **the page
shows less than the startup log**, and that is the finding.

Expose on `/v1/models`, per model, as vendor extensions under the existing convention:

- `vocab_size` — Chapter 1's closing cost made concrete: every forward pass produces a score for
  every entry, on every token, forever.
- `n_layers`, `d_model`, `n_params` — Chapter 2's shape.
- `weight_bytes_resident`, and what the same model would cost at the other quant — Chapter 5,
  including its counter-intuitive result.
- `placement` with the arithmetic that chose it — resident / N expert slots / paged / CPU, against
  weights + KV + scratch vs available. Chapter 6. The fit guard already computes this to make its
  decision; it simply does not report it.
- `kernels` — the variant selections this build resolved for this model and machine (fast attention
  on or off, `i8mm` absent, which W4A8 kernel). Chapter 10.
- `parity` — the loaded family's row from `docs/capability-matrix.json`: `full-oracle
  100.0%/1.00000`, `real-oracle 100.0%/0.98988`, `experimental: tiny-oracle`, and so on.
  **Chapter 11, and the most distinctive line available to this product.** No other local engine
  tells a user how its numerics were validated, or admits which families are still experimental.
  It is already in the tree; nothing computes it, nothing reports it.

Rule: the card is read once, not watched. It belongs beside the model, not in the reply stream, and
it must not drift into a benchmark readout.

### X8 — two per-turn facts that need no clock (Chapters 3, 4)

Neither is a timing, so neither carries X1's measurement risk.

- **Reuse eligibility, and the reason when it is no** (Chapter 4). The highest-value single field in
  this doc. On the recurrent and hybrid families the prefix-reuse exclusion means every turn
  re-prefills the whole prompt; the user sees consistently slow turns with nothing to attribute them
  to, and `prefill_reused_tokens: 0` is indistinguishable from a cold first turn (X2). The banner
  already carries the sentence for the whole server — this is the same fact, per turn, for the model
  actually answering.
- **The sampler path taken** (Chapter 3) — greedy, or the sampler with the parameters that survived
  clamping, and whether the T ≤ 0.2 optimistic-forward cap applied. That chapter is about a shipped
  default that was wrong and the measurement that nearly missed it; reporting the path actually
  taken is the chapter's own lesson applied to the product.

**Not here, because it needs no server at all:** characters per token for the user's own prompt
(Chapter 1) is derivable in the page from `prompt_tokens` and the text they typed.
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W34 owns it.

## 4. Not in scope

- **A metrics endpoint / Prometheus.** This is per-generation facts for the caller who asked, not
  a monitoring surface. `/admin/queue` (J9) is the operational view and stays separate.
- **Per-layer or per-kernel breakdown.** That is the profiler's job, and the gate in X6 exists
  partly to stop this doc drifting into one.
- **Anything about how the UI draws it** — W34 owns that, including the rule that it must not show
  what was not measured.

## Sources

`internal/serveapp/openai.go` (the `usage` struct and `prefill_reused_tokens`'s own comment on its
ambiguity; the `decode_path`/`prefill_path` vendor-extension convention) ·
`decoder/dflash_accept_test.go`, `decoder/eagle_accept_test.go` (acceptance, measured but not
exported) · `internal/serveapp/banner.go` (what a terminal user is already told at startup, and
the page is not — X7's premise) · `docs/capability-matrix.json` (the per-family `parity` column
X7 surfaces) · `decoder/fitguard.go` (the placement arithmetic X7 reports rather than recomputes) · [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W34 ·
[`task-peer-benchmarks.md`](task-peer-benchmarks.md) · [`task-work-queue-2026-09.md`](task-work-queue-2026-09.md)
J1/J3/J9 · `docs/book/` chapters 4, 7, 8, 9 (what each field is the observable of) ·
`docs/api-tiers.md` (what may be added without breaking the promise)

<!-- doc-reviewed: 2026-09-17 -->
