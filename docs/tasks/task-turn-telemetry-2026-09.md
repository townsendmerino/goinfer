# Task: what a turn actually cost — per-generation telemetry on the wire (X1–X6) — 2026-09

> **Status: SCOPED 2026-09-17, unstarted.**
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
When a drafter is attached: `draft_tokens_proposed`, `draft_tokens_accepted`, and the rounds count.
The acceptance figure is the single number that says whether speculation is earning its keep on
this traffic, and it currently exists only in test logs. Absent when `-drafter`/`-spec` is off.

### X4 — MoE expert cache, on the wire (Chapter 7)
When the expert pager is active: hits, misses, and the slot count in force. This is the number that
explains a paged MoE's decode rate, and it is the one a user staring at 1.4 tok/s most needs.
Absent on a dense model or a fully-resident MoE.

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
exported) · [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W34 ·
[`task-peer-benchmarks.md`](task-peer-benchmarks.md) · [`task-work-queue-2026-09.md`](task-work-queue-2026-09.md)
J1/J3/J9 · `docs/book/` chapters 4, 7, 8, 9 (what each field is the observable of) ·
`docs/api-tiers.md` (what may be added without breaking the promise)

<!-- doc-reviewed: 2026-09-17 -->
