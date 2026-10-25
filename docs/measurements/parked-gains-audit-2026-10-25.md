# Parked speed gains: an audit, 2026-10-25

Francis asked whether a month of gates left gains on the table. This record answers against the rule he set on 2026-10-25: a
lossless opt-in feature merges at a consistent ≥ 1.05× (9 of 10 pairs faster), with its memory cost stated, and an investment bar
is not a merge bar. Records only; nothing here changes code.

**Scope.** 210 records on `origin/main` that changed on or after 2026-09-25 and carry a stop verdict (parked, not advanced, no-go,
below bar, closed, killed, fail, not merged), across `docs/tasks/` (with `parked/`), `docs/measurements/` and `docs/completed/`
(where archived tasks live; there is no `docs/tasks/archive/`). Also SP0's record, which is only on the unmerged branch
`sp0-2026-10` (`docs/measurements/sp0-2026-10/README.md`). The search kept a verdict when the change was lossless (bit-identical or
inside its fidelity gate) and the stopped reading was ≥ 1.05× or ≥ 5% less time.

**Answer: few gains were left behind.** Every *measured*, lossless change that read ≥ 1.05× in this window shipped, was turned on by
Francis, or was overtaken by a better build that shipped. What remains are two SP0 **projections** stopped at an investment bar
(not measurements of a build), one opt-in whose arm64 default waits on a Mac measurement, and one killed idea whose ≥ 1.05× cells
were on a model its gate did not grade.

## Candidates

| ID and record | change | reading | why it stopped | cost | opt-in or default | code | recommendation |
|---|---|---|---|---|---|---|---|
| SP0, Qwen3.6-35B-A3B + DFlash (`sp0-2026-10` branch, `docs/measurements/sp0-2026-10/README.md`, "Verdicts against the registered rule") | speculative decoding with the DFlash block drafter on the 35B (DeltaNet + MoE), experts streamed, nobara CUDA | **projected** 1.22× at L = 512 (w = 3, band 1.10–1.23), 1.05× at L = 4096; from measured acceptance (5.54 tokens a cycle at w = 8) and verify cost. Not a timed end-to-end A/B | the investment band: 1.20–1.30× parks ("the owner decides"); ≥ 1.30× builds SP2 | a drafter reserve (1.70 GB on the 9B; not stated for the 35B) and a verify pass of 8 rows at 5.0× a step, because the CUDA MoE FFN runs row by row | opt-in (`--spec` with a drafter) | no product code for it: SP0 is a harness. SP2 (#232, open) builds hybrid speculation for Qwen3.5-9B | **remeasure once SP2 merges:** if SP2's verify covers the 35B, time it end to end at the merge bar (≥ 1.05×, 9 of 10). The 1.30× is an investment bar and does not apply to that merge |
| SP0, gpt-oss-20b + n-gram (same record) | n-gram speculation on gpt-oss (windowed, MoE) | **projected** 1.07× at L = 512 (K = 1, band 0.83–1.29); 1.28× at 4096 (band 0.74–2.30, "resolves nothing"). The drafter matches on 22–42% of rounds | below 1.20×: not advanced. SP1 (exact rollback for windowed models) not started on its account | n-gram has no drafter cost; verify cost grows with w on the CUDA MoE path | opt-in | none: needs SP1 | **leave closed** unless SP1 is built for another model: the 512 band crosses 1.0, so it is not consistently faster |
| R-06 on arm64 (`docs/env-vars.md`, `GOINFER_W4A8_BATCH`; `docs/tasks/task-recompute-audit.md`; `docs/tasks/task-cpu-decode-peer-gap-2026-09.md`, L2) | a layer's W4A8 q/k/v as one fork/join in CPU decode; bit-identical | Mac 1.5B 1.071× mean (1.063–1.089), September, before this window; Mac 7B null (58.98 vs 59.0 ms a token, 2026-09-20). On nobara it is on by default, at 1.016× / 1.030× / 1.018× on top of the fused gate+up | not stopped by a verdict here: arm64's default "awaits a Mac measurement" on the current tree | none stated (bit-identical; no new buffer) | opt-in on arm64 (`=1`); default on elsewhere | in the tree | **remeasure on the Mac** (0.5B, 1.5B, 7B; an in-process A B B A, minutes). Turning it on for arm64 is a default change, so the result goes to Francis |
| E-P06, a measured verify-cost curve for `--spec ngram` on Metal (`docs/completed/task-metal-audit-2026-10.md`, "E-P06: KILLED and removed") | each model measures its own verify cost at load instead of using the 7B's constant; lossless (every spec reply equals plain) | the graded cell, 1.5B chat: 1.014× against a 1.02 bar. Ungraded: 0.5B 1.237× copy, 1.084× chat; 7B 1.000×, 1.006× | its registered bar on the 1.5B ("killed is good", Francis) | load-time cost "within load-time noise" | inside the opt-in `--spec ngram` | removed; would need rebuilding | **leave closed for now.** The gain appears only where a model's curve departs from the constant (the 0.5B). Revisit if small-model speculation becomes a target, graded on that model |

## Matched, then overtaken by a build that shipped (nothing lost)

- **KV7 step 2** (`docs/measurements/kv7-step2-2026-10-17/README.md`): parked at 0.216 of cold against a 0.2 line; the tier-2 re-run
  read 0.145 / 0.144 and the code is on main.
- **R16 S2 prototypes 1 and 2** (`docs/measurements/metal-prefill-gemm-s2-2026-09-25.md`): 2.78× / 2.80×, bit-identical, parked under a
  2.85× line; prototype 4 (3.22×) shipped the same day.
- **R18 Metal decode GEMV** (`metal-decode-gemv-r18-2026-09-26.md`): 1.171× / 1.317×, parked by band, shipped by Francis the same day.
- **E-P05 multi-row decode attention** (1.057–1.060×): parked, turned on by Francis the same day.
- **R2 `attention_fa`** (1.188×): killed before the window, default on since 2026-09-21 by Francis.
- **MC4 `-spec-adaptive`** (`docs/measurements/mc4-candidate-cuda-2026-10-01.md`): lossless and already opt-in; its deciding cell
  was a 14% loss (0.858 at 4 clients), so it is not a lost gain.

## Excluded: stopped for a non-speed reason (for Francis to see)

- **KS-O2**, the fast GEMM for prompts at 12,160–16,256 keys (1.37× on Gemma-3-4B): cosine drift 0.9865–0.990; Francis, 2026-10-25,
  not without its own fidelity gate.
- **KV3**, Gemma-3 tensor-core GEMM past 12,160 keys: fidelity 0.986525 against 0.99; exact kernels chosen past the bound.
- **KV4b f16 KV for MLA** (DeepSeek-V2-Lite, 1.067× / 1.061×): `auto` stays f32 because its fidelity reference never finished;
  `--kv f16` is available on request.
- **Gemma 4 E-model CUDA batched prefill, fast levers** (about 1.5× time to first token): failed G3p at 2,170 tokens; exact by force.
- **Qwen3-VL DeepStack prefill on CUDA** (13× prefill): failed its fidelity checks, then turned on by Francis on 2026-10-09.
- **M26 lever 2, zero-copy persistent buffers** (0.66 of the time a layer): wired memory +2,016 MB.
- **E-P01, the above-floor half on Metal**: breaks cache independence (6–8 of 10 prompts diverged warm against cold).
- **Chunked prefill at 256-token chunks**: failed its wall-time gate (1.052 against ≤ 1.05); the 512-token version shipped.
- **R17 step 0, `attention_fa` split count 32** (1.20× on attention): changes the merge order and its fidelity gate was never run.

## Unsure (no verdict that fits, listed for completeness)

- **MLA batched CPU prefill** (branch `mla-batched-cpu-prefill-2026-10-24`, bit-identical at tier 0): parked because the reference need
  went away; its speed was never read.
- **`-cpu-batch auto` on the 1.5B** (nobara): batching leads the workers 1.02–1.11× but auto's 2 GiB line keeps the 1.5B on the
  workers; `-cpu-batch on` is the opt-in.
- **Native Q6_K on CUDA** (`task-int4-weight-quality-2026-09.md`, Lever 2): parked on a ~6–8% projection, not a measurement; open as
  KS-A1.
