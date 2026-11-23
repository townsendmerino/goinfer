# CUDA batched prefill on the q4k lane (Phi-3 mini): registration (2026-11-23)

**Status: REGISTERED 2026-11-23 08:20 PST, before any tier-1 or timed run.** Gates in order, each passing before the next; fixed N,
no early stop. If it ships, the batched pass becomes the default route for Phi-3's default load only with Francis's word.

**Question.** Does Phi-3 mini's default CUDA load (`--quant q4k`: Q5_K q/k/v, Q4_K o/gate/up, Q4_K or Q6_K down, Q6_K head, all
per-32 activations) prefill faster through a batched pass than through the route main takes, one decode forward per prompt
token (`PrefillPath`: "sequential: batched prefill needs int4 or int8 projections"), with output unchanged?

**The change.** Row-batched twins of the per-32 decode kernels (`actgroup.cu`): the three quantizers one block per activation row,
and the Q4_K, Q5_K, Q6_K and int8 GEMVs looping an M tile of activation rows over weights held in registers. Each (output row,
activation row) pair runs decode's arithmetic in decode's lane order, so the pass is designed to be bit-identical to the
sequential path. The fast levers do not apply (attention at head dim 96 and block-quant weights are outside L2 and L3), so the
pass is the exact path at every length. The per-row activation scales grow from 1 float to 2·K/32; `PrefillScratchBytes` prices them.

**Gate 1, tier 0 (defect detector, not a decision).** On small shapes, every rows kernel bit for bit against its decode kernel
(M = 1..33, odd N, K = 256 to 8192), each quantizer against its decode twin, a planted defect red; `TestAttnKernelDigests`
unchanged; the tiny per-32 model admitted to the pass (`PrefillPath` reads batched) and its `PrefillLast` logits bit-equal to
the sequential loop's.

**Gate 2, bit-identity on the real model** (`TestOP6C_q4kPassBitIdentity`, a new real-checkpoint test in the CUDA backend,
`~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`, `q4k`, serve's f16 KV, exclusive lock). Prompt set A's first three
prompts at K = 128 and 3,900: the seed logits after the prompt, then 16 greedy decode steps, each step's full logits, from the
batched arm against the sequential arm. **PASS:** every value bit-equal. **Not bit-equal:** a defect against the design; it is
debugged, not graded by a looser bar. Only if the difference traces to an attention kernel that is exact but ordered differently
from decode's, the record says so, and `fidelity.NonInferior` (AgreeMarginPts 1, KLMargin 0.10, alpha 0.05) against the
sequential arm on set A at K = 128 and 3,900 decides, registered here before any reading.

**Gate 3, speed** (`TestOP6C_q4kPassSpeed`, exclusive lock, after gate 2). A = the sequential loop (main's route, the do-nothing
arm), B = `PrefillLast`, one load, P = 128 and 3,900 from position 0 on set A's first prompt. Per P one untimed call of each arm,
then 5 pairs alternating A B / B A (10 per P in all). Each cell records the load average and available memory at its start; a cell
started under heavy load (load above 8.0, or available memory below the fit need plus 2 GB) is void, re-run in order, at most 4 a
P, and listed. Metric: the median of the per-pair A ÷ B TTFT (wall time to the seed logits, the device synchronised).
- **ADOPT:** median ≥ 3.0 at 3,900 and ≥ 2.0 at 128, each with at least 9 of its 10 pairs above 1.
- **KILL:** median under 1.5 at 3,900.
- **AMBIGUOUS, parked to Francis:** anything between.

The PR states the pass's device scratch at the width the build chooses and the load's fit verdict; the standing device bytes and
the longest context that fits are unchanged, since the kernels add module code and no standing buffer.

**Stopping rule.** One run of each gate decides; a run is repeated only when a P exhausts its void re-runs.

**What each outcome means.** ADOPT: DECISION NEEDED to Francis with the readings (the default route for Phi-3's default load).
AMBIGUOUS: the branch stays open for Francis. KILL, or a gate 2 failure that the fallback does not clear: the record says why and
the lane keeps the sequential route; a tensor-core q4k loader is then its own registration.

**Cost.** Gate 1 about 10 minutes of card tests. Gate 2 about 6 minutes (the sequential arm is about 40 s a 3,900-token prompt).
Gate 3 about 10 minutes. By day; nothing on the card after 21:30 PST.

## Gate 1, 2026-11-23 08:30–08:38 PST (nobara, RTX 2070 SUPER, driver 595.112.07): PASS

- **The kernels:** `prefill_g32.cu` at `bccf9c8e5` (PTX sha256 `b7c962b6…`, NVRTC 12.9.86 through `build_ptx.sh`). `actgroup.cu`
  gains only the `ACTGROUP_DEVICE_ONLY` guards; `actgroup.ptx` rebuilt to its committed bytes.
- **`TestPrefillG32_rowsMatchDecode`:** 1,344 cases (Q4_K, Q5_K, Q6_K, int8; M = 1, 2, 7, 8, 9, 17, 33; N = 1, 15, 1001;
  K = 256, 768, 3072, 8192; bias on and off; write and accumulate), every output bit-equal to the decode kernel's.
- **`TestPrefillG32_quantizersMatchDecode`:** the three batched quantizers equal decode's, codes and both halves of the scale row.
- **`TestPrefillG32_tinyPassMatchesSequential`:** phi3-tiny at int8int8/per-32 is admitted (`PrefillPath` reads batched); seed
  logits, four decode steps and `HiddenLast` bit-equal to the sequential path at 1 to 300 tokens, one pass and 16-row passes.
  **Red on main first** (`26a73a8b0`: `PrefillPath` read "sequential — per-32 activation quantization has no batched kernels yet").
- **Plants:** the Q4_K rows kernel taking the high sub-block's minimum for the low one, and the batched rmsnorm taking `rsqrtf`:
  all 336 Q4_K cases red, the quantizer test red (one ulp in a scale), the tiny pass red (`gate1/plant-red.log.gz`). Restored, the
  PTX rebuilt to the committed bytes.
- `TestAttnKernelDigests` and the FMA lint pass (`touched.log`, `touched.py` PASS at `cc5052218`, 21 checks).

## Gate 2, 2026-11-23 09:45–09:49 PST: PASS

`TestOP6C_q4kPassBitIdentity` at `cc5052218`, exclusive lock: **3,270,528 logits compared (3 prompts × K = 128 and 3,900 × 17
rows), 0 rows differ** ([`gate2/gate2.log`](gate2/gate2.log)). The batched pass is bit-identical to the sequential path,
so the fallback fidelity gate does not apply.

## Gate 3, 2026-11-23 09:49–10:02 PST: AMBIGUOUS by the registered rule, parked to Francis

`TestOP6C_q4kPassSpeed` at `cc5052218`, exclusive lock, one load, set A's first prompt; every cell started at load 2.2–2.3 with
57 GB available; no void cell ([`gate3/gate3.log`](gate3/gate3.log)).

| P | A, sequential (ms) | B, batched (ms) | median A ÷ B | pairs above 1 | bar |
|---|---|---|---|---|---|
| 128 | 953 | 340 | **2.806** | 10 / 10 | ≥ 2.0: met |
| 3,900 | 44,707 | 28,484 | **1.570** | 10 / 10 | ≥ 3.0 ADOPT, < 1.5 KILL: **between** |

The pairs are tight (1.569–1.570 at 3,900). The pass's device scratch is 290.5 MiB at the 2,048-row width the build chose (3,276 MiB
free beyond the margin kept beside a pass); the standing device bytes are unchanged.

**How the 3,900 bar was set.** The registration took ADOPT at 3.0× from the review's projection of about 10× (39 s sequential
against 1–2 s of batched GEMM) and did not price the depth term: this step batches the weights but leaves attention at head dim 96
on the exact kernels, whose cost grows with the prompt, which is why 3,900 reads 1.57× while 128 reads 2.81×. From the 128 cell,
about 2.6 ms a token is weights and glue, so at 3,900 roughly 10 s of the 28.5 s is weights and about 18 s grows with depth
(arithmetic; the cost split below measures it).

## Decision, 2026-11-23 about 12:25 PST

Francis took option 1 through the coordinator: the batched pass becomes the default route for Phi-3's default CUDA load, on the
grounds that its output is bit-identical, it is faster on 10 of 10 pairs at both depths, and it replaces the sequential route.
Gate 3's AMBIGUOUS reading against the registered bars stands as read. The depth term is its own follow-up registration.

## The longest context that fits, main against the branch (2026-11-23 12:30 PST)

Serve's own load banner for Phi-3 mini at its default load (`fit_banner.sh`; [`fit/banner-main.log`](fit/banner-main.log) at
`d32a91e7a`, [`fit/banner-branch.log`](fit/banner-branch.log) at `b4e7f69e0`), exclusive lock:

| | main | branch |
|---|---|---|
| context | 4,096 (model maximum), f16 KV | 4,096 (model maximum), f16 KV |
| KV plan | 2 conversations × 4,096 | 2 conversations × 4,096 |
| prefill width | 2,048 rows, 282 MiB of scratch | 2,048 rows, 290 MiB of scratch |
| free beyond the pass margin | 1,568 MiB | 1,568 MiB |
| prefill path | sequential | batched, exact kernels |

**Unchanged.** The context margin already prices a 512-row pass on every load; the per-32 scale rows add 32–216 KiB to it on the
memory census's per-32 and q4k cells (114 of 765 lines move, `margin` only; no planned context, KV or width moves), and the pass
width is chosen from what is left after the build. Host bytes: none added.

## Cost split (diagnostic, not a registered cell)

`TestPrefillDecomp` with `GOINFER_DECOMP_QUANT=q4k` on Phi-3 mini, one profiled run a depth, at the test suite's f32 KV
([`costsplit-f32kv.log`](costsplit-f32kv.log)): at 128, GEMV 324 ms of 340; at 3,900, GEMV 10.13 s, attention 28.79 s, glue 0.12 s.
The GEMV term does not depend on the KV type, so at serve's f16 KV (gate 3's 28.5 s) attention is about 18.3 s: the depth term is
attention, as the arithmetic above said.
