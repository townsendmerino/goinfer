# MC3 S1 — a batched decode step on Metal, in sequence: bit-identical to production on the 1.5B and 7B, 1.73–1.84× aggregate at B = 4 (2026-09-26)

S1 (exploratory) for MC3 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), Metal batched
decode. S0 ([`concurrency-mc3-s0-2026-09-26.md`](concurrency-mc3-s0-2026-09-26.md)) built bit-identical 8-row
matmul kernels and timed them standalone. S1 puts them in a whole decode step: B sequences, each on its own MC1
resident KV slot at its own position, one command buffer, one token each. That step is checked logit by logit against
production's single-token forward and timed in sequence.

**Result.**
- **Every logit of every sequence equals production's**, on qwen2.5-coder-1.5b and qwen2.5-7b-instruct (4 sequences
  × 12 steps × 152k logits, 0 differ on each).
- A B = 4 step costs 2.1–2.3× one production token, so **4 sequences decode at 1.73–1.84× one stream's aggregate**;
  B = 8 reaches 2.4–3.2×.
- These track S0's projection to within ~5%.
- **B = 2 is a wash** (0.98–1.10×). **B = 1 must stay on production's path** (0.51–0.64×), because an 8-row
  fragment's cost does not shrink with the batch.

## The step

`mc3Step.step` (`metal/mc3_step_test.go`, test-only) encodes, per layer:
- per sequence, production's own kernels for:
  - the norm + int8 quantisation;
  - RoPE, the KV store into that sequence's slot, and attention over that slot;
  - the ctx quantisation;
  - SwiGLU + quantisation;
  - the final norm;
- **once for all sequences**, the five matmuls: qkv (+bias), o (+residual), gate/up, down (+residual) and the int8
  LM head. These are S0's `mc3_bt` / `mc3_btd` / `mc3_lm` kernels with production's epilogues, fed by a pack kernel
  that lays the B rows' int8 activations out as the fragments' right operand.

Every row's data is therefore production's, and the batched matmuls reproduce production's arithmetic (S0). Scope is
the plain dense W4A8 path (qwen2 family), depths below `attnFADepthFloor` (1536 keys, where production's plan is the
shipped attention kernel), no speculation and no LoRA.

## Identity

`TestMC3Step_bitIdentical`:
- 4 sequences filled to depths 5, 23, 40 and 300 through production's `ForwardBatch`, on slots 0–3;
- each history copied to a twin slot (4–7) the same way;
- 12 teacher-forced steps. Each step runs production's `ForwardEmb` per sequence on its twin, then the batched
  step, and every logit is compared bit for bit.

| model | logits compared | differ |
|---|---:|---:|
| qwen2.5-coder-1.5b-instruct q4_k_m (`.gguf`), int4 | 4 × 12 × 151,936 | **0** |
| qwen2.5-7b-instruct q4_k_m (`.int4.metal.giw`), int4 | 4 × 12 × 152,064 | **0** |

The test can go red, and did. The first cut composed a sub-row offset as `row.At(off)`, but aikit's `Buffer.At`
*sets* the offset from the allocation's start. So sequences 1–3 read sequence 0's K/V and up-projection. Every one
of their logits differed (max |diff| 12–41) while sequence 0's matched. Logs:
[`identity-1.5b.log`](concurrency-mc3-s1-2026-09-26/identity-1.5b.log) (after the fix),
[`identity-throughput-7b.log`](concurrency-mc3-s1-2026-09-26/identity-throughput-7b.log).

## In-sequence cost

`TestMC3Step_throughput`:
- M1 Pro 16 GB, macOS 26.6.2, goinfer `67490d4b` plus the MC1 slot fix (committed as `6807ab95`) and the test file,
  1024-token resident context, load1 1.8–2.5.
- Checkpoints are the ones R18b graded, both from `~/models`.
- 9 slots filled to depth D.
- Per rep, five arms, rotating order, 12 tokens each:
  - production's single-token decode on slot 8, rewriting position D so the depth stays D;
  - the batched step at B = 1, 2, 4 and 8.
- GPU time per command buffer, median per arm per rep, 7 reps.
- aggregate = B × production ÷ step(B), paired per rep; the median and range are shown.
- 2026-09-26 21:30–21:37 PDT. Logs: [`throughput-1.5b.log`](concurrency-mc3-s1-2026-09-26/throughput-1.5b.log),
  [`identity-throughput-7b.log`](concurrency-mc3-s1-2026-09-26/identity-throughput-7b.log).

| model | depth | production token | B = 1 | B = 2 | **B = 4** | B = 8 |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B | 128 | 10.66 ms | 0.61× (step 1.59×) | 1.10× (1.77×) | **1.84×** (2.11×) [1.78–1.95] | 2.90× (2.78×) |
| 1.5B | 512 | 11.32 | 0.64× (1.56×) | 1.09× (1.82×) | **1.73×** (2.31×) [1.72–1.74] | 2.42× (3.30×) |
| 7B | 128 | 33.32 | 0.51× (1.95×) | 0.99× (2.04×) | **1.84×** (2.19×) [1.60–1.99] | 3.19× (2.51×) |
| 7B | 512 | 35.77 | 0.53× (1.89×) | 0.98× (2.04×) | **1.76×** (2.27×) [1.69–1.85] | 2.87× (2.79×) |

(aggregate × one stream; in parentheses the step's time ÷ one production token; [paired per-rep range])

Reading:
- **The standalone S0 projection held in sequence.** For the 1.5B at depth 128 it projected B = 2 / 4 / 8 at 1.14 /
  1.94 / 2.98×, and S1 measured 1.10 / 1.84 / 2.90×.
- **Depth erodes it**, as S0 predicted and as on the CPU (MC2). Attention runs per sequence and does not amortise:
  the B = 8 step grows from 2.78× to 3.30× a token between depth 128 and 512 on the 1.5B.
- **B = 1 and B = 2 argue for a threshold.** At B = 1 the fragment computes 8 columns for one, so the step costs
  1.6–1.95× a token. At B = 2 it is even or slightly ahead. A production scheduler runs production's per-sequence path
  below B = 3; a hybrid B = 2 (per-row GEMVs for the int4 projections, the batched LM head) is unmeasured.

## What S1 found outside MC3

**MC1's slot clamp double-counted the weights**, fixed in `6807ab95` (task doc, MC1).
- The first S1 load was granted 2 of 8 slots, 28 MB each, with 5.5 GB available.
- `metalKVSlots` was priced after the build had allocated its weights, against a live-memory figure that no longer
  included them, while its base still did.

## What is left before MC3's registered gates

MC3's gates are end to end (W7, `serve`): the identity gate; 4-client aggregate ≥ 1.2× MC1's; 4-client p99 no worse
than MC1's; and the lone-request guard (≤ 1.05×). S1 says nothing about them directly. Still to build:
- **production kernels and wiring**: the batched kernels in the resident, the `simd_sum`-tree self-check with the
  robust fallback, and attention_fa's per-sequence plan above 1536 keys;
- **a batched step behind the decoder's resident interface**, with production's path at B < 3;
- **the scheduler**: concurrent generations on distinct slots joining a shared step at token boundaries, with a
  newcomer's prefill run between steps;
- **serve**: `-max-concurrent` admitting GPU-resident dense models, and slot check-out so two running generations
  never share a slot.
