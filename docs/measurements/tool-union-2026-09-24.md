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

`nobara-pc`, RTX 2070 SUPER, driver 595.91.07, q4_k_m GGUFs from `~/models` (sidecar `.giw` default), idle-gated. Ungated build
`29e2706a`, gated build `2ede8d77`. Raw per-sample data (full output text kept, `T0_KEEP_RAW=1`), logs and progress timestamps:
[`tool-union-2026-09-24/ungated/`](tool-union-2026-09-24/ungated/), [`tool-union-2026-09-24/gated/`](tool-union-2026-09-24/gated/).
Byte-identity checks: `scripts/tool_union_cmp.py` (call ids stripped — they are random per request).

**Gate A — PASS (ungated build).** Greedy, 10 turns each, default vs `=0`: 1.5B 10/10, 7B 10/10, llama3-1B 10/10 byte-identical —
including the 7B's two armed turns, whose constrained call came out byte-identical to the unconstrained one (T6's parity gate, live).

**Gate B — FAIL (ungated build).** Per-block default / `=0`, 4 ABBA blocks:

| cell | blocks | median |
|---|---|---|
| 1.5B greedy | 0.9730, 0.9796, 0.9767, 0.9792 | 0.978 |
| 1.5B T=0.7 | 0.8028, 0.8066, 0.8079, 0.8063 | 0.807 |
| 7B greedy | 0.9878, 0.9896, 0.9882, 0.9884 | 0.988 |
| 7B T=0.7 | 0.8964, 0.8959, 0.8952, 0.8990 | 0.896 |

As pre-registered, the auto union went default-OFF and the gated design was built.

**Gate C — PASS (ungated build; the grammar is unchanged in the gated one).** Qwen2.5-7B, union on, T0 transcript, same seeds:

| class | T0 (no union) | union on |
|---|---|---|
| parsed | 97 | **110** |
| unknown tool name | 10 | **0** |
| unparsed body | 3 | **0** |
| args invalid (informational in T0, gated here) | 10 | **0** |
| truncated at max_tokens | 1 | 1 (the same turn-5 prose that hit 512 tokens; excluded by the rule) |
| prose | 189 | 189 |
| greedy | 2 parsed / 8 prose | 2 / 8 |

Every call the 7B attempted reached the harness as a well-formed call to a supplied tool with schema-valid arguments. Not gated,
reported: llama3-1B 173 / 9 unknown / 2 unparsed / 1 truncated / 115 prose — unchanged, as designed (not armed under auto); its
failures remain and need the task's option (c). 1.5B 219 parsed / 4 unwrapped / 77 prose — the bare-call parser's numbers; the
Coder model never writes the opener, so the union never arms.

**Gate A′ — PASS (gated build).** Default vs `=0`, byte-identical: 1.5B **310/310** (10 greedy + 300 at T=0.7), llama3-1B
**310/310**, 7B greedy **10/10** (two armed turns included). The prose part of a turn is now the unconstrained decode itself, sampled
turns included.

**Gate B′ — PASS (gated build).** Same harness and rule as B:

| cell | blocks | median | (gate B) |
|---|---|---|---|
| 1.5B greedy | 1.0008, 1.0018, 1.0042, 0.9988 | **1.001** | 0.978 |
| 1.5B T=0.7 | 0.9955, 1.0000, 1.0006, 1.0031 | **1.000** | 0.807 |
| 7B greedy | 0.9988, 0.9991, 0.9995, 0.9993 | **0.999** | 0.988 |
| 7B T=0.7 | 1.0004, 1.0000, 0.9974, 1.0002 | **1.000** | 0.896 |

**Decision (pre-registered): the auto union ships default-ON** (`GOINFER_TOOL_UNION=0` opts out).

**Speculative servers.** Every speculative path refuses a LogitProcessor, gated or not, so on a server started with
`--drafter` / speculation the auto union would cost multi-tool auto turns their drafter. Such servers are therefore left on their
pre-T1 auto behaviour (not armed; unit-tested); required still gets the union, on the grammar-fused speculative path. Making the lazy
union speculation-compatible is open.

**Not established.** No MoE, one transcript, CUDA only. The union's cost INSIDE a call (full-logits steps for the tokens of
the call itself) is not separated out; it is included in gate C's run but was not timed.
