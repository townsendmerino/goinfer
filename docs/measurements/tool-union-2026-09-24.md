# Multi-tool union grammar (T1–T3): gates and re-measure (2026-09-24)

Task: [`task-tool-grammar-union-2026-09.md`](../tasks/task-tool-grammar-union-2026-09.md); the failure it targets was measured in
[`tool-call-failure-t0-2026-09-23.md`](tool-call-failure-t0-2026-09-23.md) (Qwen2.5-7B: 14/111 attempted calls unusable under `auto`,
10 of them invented tool names; a further 10 parsed calls with invalid arguments).

## What was built (before any measurement)

- `constrain.ToolCallsGrammar` — one call constrained to any of N tools, as the parallel-branch union of the existing single-tool
  grammars (`0cd37118`). Its language is exactly the union of theirs.
- `constrain.LazyMasker` — masks nothing until the family's call opener appears in the output, then constrains that call, then
  disarms, so a second call in the turn is constrained independently (T5 decided: repeated wrapper, each constrained). Fails open
  if the bytes after the opener cannot be a call (it cannot take emitted bytes back).
- Wiring (`internal/serveapp/tools.go`, `constrainToolUnion`), on all three surfaces: **auto** (OpenAI auto, Anthropic auto incl. a
  lone tool) → the lazy union, as a LogitProcessor only; **required** (OpenAI required, Anthropic any) with 2+ tools → the union
  from token 1, with the fused-spec masker (closes N-18). llama3 (no opener) is NOT armed under auto — the task's option (b).
  `GOINFER_TOOL_UNION=0` restores the pre-T1 behaviour exactly.

## Pre-registration (written BEFORE any served run)

**Known consequence, stated up front.** Any LogitProcessor disables the decoder's on-device greedy argmax (`fastGreedy`), its
on-device sampling fast paths, and every speculative path, for the whole turn — including the prose part, where the lazy masker
does nothing. So (a) a prose turn may decode slower, and (b) at T > 0 a prose turn is drawn from the same distribution through the
host sampler, with a different random stream: its bytes can differ from `=0` even though nothing was masked. Greedy argmax is the
same function either way, so at T = 0 the bytes must not change.

**Gate A — non-forcing, greedy (hard).** On the T0 transcript at T = 0, every one of the 10 turns' outputs on Qwen2.5-Coder-1.5B,
Qwen2.5-7B and Llama-3.2-1B must be byte-identical between default and `GOINFER_TOOL_UNION=0`, EXCEPT outputs that contain the
opener (where the union is supposed to act). Any other difference is a defect.

**Gate B — prose-turn cost (decides the default).** CUDA, fresh server per arm, ABBA ×2, 3 decodes of 256 tokens per server, on a
12-tool `auto` request whose answer is prose (the tool list plus "Explain how a token bucket works, in prose." — checked to contain
no opener in both arms). Cells: 1.5B and 7B × greedy and T = 0.7. Statistic: per-block median ratio default / `=0`.
- **≥ 0.98 on all four cells → the lazy union ships default-on** as built.
- **< 0.98 on any cell → the lazy auto union ships default-OFF** (required-union stays on: it only applies when a call is mandatory),
  and a two-phase design (generate to the opener with every fast path intact, then continue constrained) is the next item.

**Gate C — the failure it exists to remove (T6 re-measure).** `scripts/bench_tool_failure.py`, same transcript, seeds, 300 sampled +
10 greedy, Qwen2.5-7B int4, default build:
- `unknown_name` + `unparsed` = **0** of attempted calls. `truncated` (max_tokens ran out mid-call) is reported separately — a
  grammar cannot prevent a length stop.
- `args_invalid` (required key missing / wrong top-level type) = **0**.
- Anything above zero means the grammar has a hole: a defect, not a smaller effect.
Also run on Llama-3.2-1B and the 1.5B to show what does NOT change (llama3 unarmed; the Coder model never writes the opener): their
class counts are reported, not gated (T > 0 draws move with the sampler path, per the consequence above).

**Not covered.** Speculative-decode interaction is covered only for the required path, by construction (it reuses the tested
single-tool fused-spec plumbing with a grammar whose Clone is tested); no live spec run. No MoE. One transcript.

## Gated follow-up — pre-registration (written BEFORE its runs, after gate B failed)

Gate B failed (numbers below). Owner call: build the two-phase design. Built INSIDE the decoder instead of as two generations:
`SamplingParams.LogitProcessorGate` — the lazy masker is asked after every emitted token whether the next step must be masked; while
it says no, the decode loop keeps on-device greedy argmax, device Gumbel sampling and device top-K exactly as with no processor, and
only steps inside a call read full logits. One generation, no re-prefill, no second RNG stream. Unit-gated through the real decode loop
with a fake resident (`decoder/logit_gate_test.go`; each assertion red under a mutation).

Re-run with the gated build, same scripts, same rules:
- **Gate A′ — non-forcing, now also at T = 0.7.** Greedy: as gate A. Sampled: every T = 0.7 output of the transcript (300 per
  model) on Qwen2.5-Coder-1.5B and Llama-3.2-1B — neither ever writes the opener, and llama3 is never armed — must be byte-identical
  default (union on, gated) vs `=0`. The prose part now runs on the identical path, so there is no RNG argument left: any
  difference is a defect.
- **Gate B′ — prose cost, identical rule to B:** per-cell median ≥ 0.98 on all four cells → the auto union ships default-on;
  otherwise it stays opt-in.
- Gate C stands from the ungated run: the grammar, masker and wiring are unchanged; only which forward runs outside a call is.

## Results

_(appended after the runs)_
