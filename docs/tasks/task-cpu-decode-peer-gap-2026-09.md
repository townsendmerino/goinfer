# Task: close the CPU decode gap to Ollama — 2026-09

> **Status: OPENED 2026-09-27.** Phase 0 is done
> ([`cpu-decode-peer-gap-2026-09-27.md`](../measurements/cpu-decode-peer-gap-2026-09-27.md)). HEAD reads 0.818× / 0.810× /
> 0.856× Ollama (0.5B / 1.5B / 7B, nobara, CPU). On the 1.5B and 7B the gap is about a third bytes (f32 group scales,
> int8 head) and two-thirds bandwidth; on the 0.5B it is bandwidth only.
>
> **Owner decisions 2026-09-27:**
> 1. Run the f16-scale quality gate (L1 below).
> 2. Flip R-06's fused W4A8 q/k/v on by default (L2 below).
>
> Both are pre-registered here, before any code or run.
>
> **L2 SHIPPED 2026-09-27:** R-06 is on for non-arm64. Every gate passes: logits bit-identical, the suites green, and
> paired 1.016× / 1.030× / 1.018× (0.5B / 1.5B / 7B). **L1 FAILS as pre-registered** (worst drop 0.0168 > 0.015), but
> the drops are symmetric about zero (13 of 24 cells improve, mean +0.0044 in f16's favour). The gate had no noise arm;
> whether to re-gate with one is the owner's call. **Owner decision 2026-09-27: override, build it, for consistency.**
> CPU int4 moves to the f16 group scales CUDA, Metal and WebGPU already store (L1 build, below).

## L1 — f16 group scales for CPU int4: the quality gate (pre-registered 2026-09-27)

**What it decides.** Whether CPU int4 may store its per-32 group scales as f16 (0.5625 B/param) instead of f32
(0.625). The byte saving is 22 / 82 / 408 MB per token (0.5B / 1.5B / 7B). A PASS authorises building the aikit
kernels. It does not ship anything: the kernels get their own speed gate.

**Why a quality gate can run before any kernel.** `GOINFER_INT4_F16_SCALES=1` (`decoder/int4f16scales.go`, a
load-time diagnostic) rounds every int4 scale through IEEE half precision. A CPU load then carries exactly the weights
a kernel reading f16 scales would see, and the ones CUDA, Metal and WebGPU already serve.

**Method.**
- The actquant sweep program ([`actquant-trackb-2026-09-25/actsweep.go.txt`](../measurements/actquant-trackb-2026-09-25/actsweep.go.txt)),
  changed only to add an arm. Per family it runs:
  - f32 (the reference);
  - `int4` at production CPU defaults (per-row activations, f32 scales);
  - the same `int4` with `GOINFER_INT4_F16_SCALES=1` set around its load.
- Same 12 families and two prompts (filler, prose) as the sweep and Track B. The tree is `ab1a8b39` plus the arm;
  weights come from `~/models`; the box is idle.
- **Metric:** p10 per-position logit cosine against f32, per family × prompt, for both int4 arms. Reported, not
  graded: argmax agreement, and the p10 cosine of the f16 arm against the f32-scale arm directly (the single-variable
  reading).

**Bands** (Track B's no-regression allowance, unchanged). For each of the 24 family × prompt cells, the drop is
Δ = p10(int4, f32 scales) − p10(int4, f16 scales).
- **PASS:** every Δ ≤ 0.005.
- **AMBIGUOUS → the owner, with the cells named:** the worst Δ is in (0.005, 0.015].
- **FAIL:** any Δ > 0.015. Lever L1 is then dead as a lossless-enough byte cut, and the next byte lever is the head.

Phi-3 is graded too, though serve routes it off `int4` by default (its activation hazard). A regression there still
counts.

## L2 — R-06 fused W4A8 q/k/v, default on (owner decision 2026-09-27; gates pre-registered the same day)

**What changes.** `w4a8BatchEnabled` (`decoder/weightmat.go`) becomes default-on on non-arm64, and `GOINFER_W4A8_BATCH=0`
opts out. This is the pattern `fusedGateUpDefault` set in `cpu_tuning_{arm64,other}.go`. With the fused gate+up
default-on there, R-06 changes only q/k/v (one fork/join instead of three).
- **arm64 is unchanged** until the Mac measures it on the current tree. R-06's arm64 reading (1.071×) predates the
  fused gate+up.
- No new environment read: the same variable, with its default inverted on non-arm64.

**Gates (hard).**
1. **Bit-identical logits.** A new `TestCPURoofline_w4a8Batch_logitsBitIdentical`, mirroring the fused gate+up's: the
   0.5B, 1.5B and 7B, 48 decode steps, every logit compared with `!=`, on vs off. 0 differ.
2. **Existing suites.** `TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity`, `TestDecodeParityInt4` and
   `TestSession_reuseParity` pass. The parity manifest is refreshed (`scripts/refresh_parity_hashes.sh`) only after
   they do.
3. **No size regresses.** `TestCPURoofline_w4a8Batch` (paired in-process ABBA, 5 pairs, one loaded model, at today's
   defaults, so the fused gate+up is on in both arms). The 7B is added to its model list. The flip stands unless any
   size's paired median speed-up (off ms/token ÷ on ms/token, the harness's own "ON is N× faster") reads < 0.98 (the 09-23 record's "no size regresses > 2%"). The marginal is recorded
   whatever it is.

**Not a gate:** a served re-read against Ollama after both levers. Its ratios are reported, not graded.

### L2 result (2026-09-27): every gate passes, on for non-arm64

The code was measured at `cc1769c7` with the test edits; the flip is the next commit. Logs are in
[`cpu-decode-peer-gap-2026-09-27/`](../measurements/cpu-decode-peer-gap-2026-09-27/) (`r06-*`).

1. **Bit-identical** (`r06-gate1-bitident.log`): 48 decode steps × 151,936 / 151,936 / 152,064 logits on the 0.5B,
   1.5B and 7B, **0 differ**.
2. **Suites** (`r06-gate2-suites.log`, run with the flip in place): `TestForwardN_matchesSequential`,
   `TestSpeculativeGreedyParity`, `TestDecodeParityInt4`, `TestSession_reuseParity`, `TestInt4_forwardParity`,
   `TestEnvVars_docAndCodeAgree` and `TestDispatchCensus` all pass.
   - The parity refresh ran the forward goldens first: 62 passed, 0 skipped, 0 failed (`r06-parity-refresh.log`).
3. **Speed** (`r06-gate3-ab.log`; paired in-process ABBA, 5 pairs, depth 128, the fused gate+up on in both arms):

   | model | off → on ms/token | paired median (min–max) |
   |---|---|---|
   | 0.5B | 23.81 → 23.54 | **1.016×** (0.979–1.040) |
   | 1.5B | 51.33 → 49.83 | **1.030×** (1.026–1.036) |
   | 7B | 195.88 → 192.72 | **1.018×** (1.012–1.020) |

   No size's median is below 0.98, so the flip stands. The gains are small, as R-06's mechanism predicts: it saves a
   roughly fixed per-barrier cost, and with gate+up already fused only q/k/v's two extra barriers remain.

### L1 result (2026-09-27): FAIL by the pre-registered bands; the reading says the bar is below the metric's noise

The sweep ran 20:10–20:56 PDT with the program ([`f16sweep.go.txt`](../measurements/cpu-decode-peer-gap-2026-09-27/f16sweep.go.txt))
built at `cc1769c7`. Raw data: `f16sweep.jsonl`; the table: `f16sweep-graded.txt`, same directory.

**Verdict, unchanged from the bands:** the worst Δ is **+0.0168** (qwen3-1.7b filler) > 0.015, so **FAIL**. Six of 24
cells exceed 0.005.

| family | filler: int4 → f16 p10 (Δ) | prose: int4 → f16 p10 (Δ) |
|---|---|---|
| phi3-mini | 0.2359 → 0.2408 (−0.0049) | 0.9436 → 0.9384 (+0.0052) |
| qwen2.5-7b | 0.1523 → 0.1463 (+0.0061) | 0.9423 → 0.9323 (+0.0100) |
| qwen2.5-coder-1.5b | 0.9940 → 0.9931 (+0.0009) | 0.9911 → 0.9913 (−0.0002) |
| qwen3-1.7b | 0.9692 → 0.9524 (**+0.0168**) | 0.8900 → 0.9068 (−0.0168) |
| qwen3.5-0.8b | 0.9622 → 0.9644 (−0.0023) | 0.9538 → 0.9571 (−0.0033) |
| gemma3-1b | 0.9832 → 0.9834 (−0.0002) | 0.9774 → 0.9770 (+0.0004) |
| llama3.2-1b | 0.6453 → 0.6849 (−0.0396) | 0.7308 → 0.7290 (+0.0018) |
| tinyllama-1.1b | 0.9682 → 0.9572 (+0.0110) | 0.9897 → 0.9898 (−0.0001) |
| mistral-7b | 0.9952 → 0.9951 (+0.0001) | 0.9973 → 0.9972 (+0.0001) |
| granite-4.2-3b | 0.9757 → 0.9705 (+0.0052) | 0.9502 → 0.9572 (−0.0069) |
| smollm3-3b | 0.9653 → 0.9721 (−0.0068) | 0.8466 → 0.8623 (−0.0158) |
| olmo3-7b | 0.7983 → 0.8441 (−0.0457) | 0.9365 → 0.9580 (−0.0216) |

(Δ = int4 − f16; positive means f16 is worse.)

**Reading (not a grade).**
- **No direction.** 11 cells worsen and 13 improve. The mean Δ is −0.0044, meaning f16 is slightly better on average.
- **The largest moves are improvements:** olmo3-7b by 0.046 and 0.022, llama3.2-1b filler by 0.040.
- **The failing family moves both ways.** qwen3-1.7b loses 0.0168 on filler and gains exactly 0.0168 on prose.
- **The cleanest families barely move:** mistral-7b, coder-1.5b and gemma3 stay within ±0.001.
- **The perturbation is tiny.** Rounding a scale to f16 changes it by at most 2^-11 (~0.05%). Where the f16 arm is
  compared against today's int4 directly, it reads p10 0.95–0.999, lowest where the int4 path is already damaged
  (qwen2.5-7b filler 0.72; llama 0.955).
- **So the p10 cosine on the per-row int4 path moves by ±0.02 under a perturbation this small, in either direction.**
  Track B's 0.005 allowance on the worst of 24 cells sits below that. The gate had no do-nothing arm to measure the
  noise, which is this repo's "include the do-nothing arm" rule, and its MoE router-flip lesson (judge the mean, not
  the minimum over N).

**Not done, on purpose:** re-banding, and building the kernel. The bar is the bar; any re-gate is a new
pre-registration and the owner's call.
- **The re-gate this suggests:** the same sweep with control arms that perturb each int4 scale by a seeded random
  relative amount of the same size as f16 rounding (|ε| ≤ 2^-11), 2–3 seeds, through a `goinfer_testhooks` seam (no
  production env read).
- f16 would then pass if its per-cell Δ sits inside the controls' spread, and its mean Δ is no worse than theirs.
- Cost: about 1.5–2 hours of sweep.

### L1 owner override (2026-09-27): build f16 scales for CPU int4, for consistency with the GPU backends

**The owner overrode the FAIL:** "for consistency, let's override."
- **Basis:** the three GPU backends already serve these exact weights, so the CPU joins them.
- **The quality question stays open, not answered.** The gate above could not separate f16's effect from the metric's
  noise. Nothing here claims f16 passed.
- **Precedent for the form:** R8's fused vision-tower kernel, default by owner override after missing both fidelity
  gates.

The build gets its own correctness and speed gates, pre-registered before any timing.

### L1 S0 — does an f16-scale kernel actually buy the bytes? (pre-registered 2026-09-27, before any code)

**Why an S0 first.** The override settles quality, not speed. The 1.06–1.10× is an estimate from bytes, and the
kernel pays a conversion per 32 weights. The full migration is large, so the speed is measured on a prototype first:
- aikit: `Int4()`, `W4A8Op.Scales` and the upload APIs are typed `[]float32`, and the arm64 NEON kernels need f16
  loads;
- goinfer: the `.giw` format and every backend's upload path;
- an aikit release.

**Prototype (local branches in aikit and goinfer; nothing pushed; goinfer builds against the local aikit via
`go.work`):**
- **The hot kernel.** On amd64 decode, q/k/v, gate/up, o/down and MoE experts all end in `dotW4A8FoldAVX2`.
  `dotW4A8FoldF16AVX2` is its copy, with `VBROADCASTSS (BX)` replaced by `VPBROADCASTW (BX)` plus `VCVTPH2PS`, and the
  scale pointer advancing 2 bytes per group instead of 4. It requires F16C.
- **Routing.** A `WeightMat` carries an optional f16 copy of its scales, and `W4A8Op` an optional f16 slice. A
  test-hook switch picks f16 or f32 at run time, so one loaded model runs both arms. Prefill kernels (M ≥ 4) are out
  of S0's scope: the tile kernel stays f32.
- **Weights.** Loaded with `GOINFER_INT4_F16_SCALES=1`, so the f32 arm reads f16-rounded scales and the f16 arm reads
  their exact half-precision encoding. Decoding f16 to f32 is exact, so the two arms compute identical numbers.

**Gates:**
1. **Correctness (hard).** Logits are bit-identical between the arms: 48 decode steps on the 0.5B / 1.5B / 7B, every
   element compared with `!=`. Any difference is a kernel or routing bug, not a result.
2. **Speed.** The same paired in-process ABBA harness (`cpuDecodeAB`), 5 pairs, depth 128, at today's defaults
   (fused gate+up and R-06 on).
   - **BUILD** (proceed to the full migration): the paired median is ≥ 1.04× on both the 1.5B and the 7B.
   - **PARK:** < 1.02× on either; the conversion eats the bytes.
   - **Between: the owner decides.**
   - The 0.5B is reported, not gated.

### L1 S0 result (2026-09-27): BUILD — 1.059× (1.5B), 1.084× (7B), bit-identical; the 0.5B reads 0.957×

**Setup.** The prototype is on the unpushed `f16-scales-s0` branches of aikit (off `4eec1c5`) and goinfer (off
`7e17f520`), wired through `go.work`:
- `dotW4A8FoldF16AVX2`: `VPBROADCASTW` + `VCVTPH2PS` in place of `VBROADCASTSS`;
- `MatmulBTW4A8F16Into`, `W4A8Op.ScalesF16`, and the `WeightMat` entry point;
- goinfer's `wmW4A8Op` and fused gate+up passing the f16 slices.

aikit's `TestW4A8F16S0_matchesF32` checks all three entry points bit-identical at four shapes, and mutation-checks
that a one-ulp scale change moves the output; the full linalg suite passes. Logs: `s0-*` in
[`cpu-decode-peer-gap-2026-09-27/`](../measurements/cpu-decode-peer-gap-2026-09-27/).

1. **Correctness: PASS.** f16 scales attached to every int4 weight (168 / 196 / 196). 48 decode steps × 151,936 /
   151,936 / 152,064 logits on the 0.5B / 1.5B / 7B: **0 differ**.
2. **Speed** (paired in-process ABBA, 5 pairs, depth 128, fused gate+up and R-06 on):

   | model | f32 → f16 ms/token | paired median (min–max) | byte-count estimate |
   |---|---|---|---|
   | 1.5B | 50.03 → 47.15 | **1.059×** (1.058–1.070) | ≤ 1.078× |
   | 7B | 192.74 → 177.71 | **1.084×** (1.082–1.087) | ≤ 1.097× |
   | 0.5B (reported) | 23.03 → 24.10 | **0.957×** (0.899–1.007) | ≤ 1.065× |

**Verdict: BUILD** (both gated sizes ≥ 1.04×). The 1.5B and 7B recover 76% and 87% of the byte estimate's upper
bound. The 0.5B is slower and noisy. Phase 0 found its gap is not bytes, so the added `VCVTPH2PS` per group
(a conversion per 32 weights, on a path limited by compute and dispatch rather than DRAM) is not paid back there.
The full build first tries converting eight groups' scales per instruction (one `VCVTPH2PS` to a Y register per 8
groups, then a lane broadcast per group) to recover it; a 0.5B regression that survives goes to the owner.

### L1 S0b (2026-09-27): a row-widening kernel — the 0.5B improves to 0.968×, the 7B to 1.090×

**Why.** An aikit benchmark (single thread, cache-hot, minimum of 5; a diagnostic of compute cost, not a decode
prediction) put the per-group `VCVTPH2PS` at +6–10% compute over the f32 kernel at the 0.5B / 1.5B shapes. That is
hidden where decode is DRAM-bound and shows where it is not (the 0.5B).

**The change.** `dotW4A8FoldF16RowAVX2` is one asm call per weight row. It widens the row's scales eight per
`VCVTPH2PS` into an L1 buffer (the last block overlapping, so no scalar tail), then runs the f32 kernel's own loop.
Measured on the same benchmark:

| shape | f32 | f16, row-widened | f16, per group |
|---|---|---|---|
| K1536 × N8960 | 894.4 µs | 918.1 (+2.7%) | 961.6 (+7.5%) |
| K4864 × N896 | 280.0 | 287.1 (+2.5%) | 305.1 (+9.0%) |
| K896 × N4864 | 309.0 | 316.5 (+2.4%) | 326.5 (+5.7%) |

**The S0 gates re-run on it** (`s0b-*` logs):
- Bit-identical on the 0.5B / 1.5B / 7B (0 differ).
- Paired medians:

  | model | f32 → f16 ms/token | paired median (min–max) |
  |---|---|---|
  | 1.5B | 50.02 → 47.20 | **1.059×** (1.055–1.066) |
  | 7B | 192.88 → 177.04 | **1.090×** (1.083–1.094) |
  | 0.5B | 22.91 → 23.20 | **0.968×** (0.960–1.027) |

The row kernel is the one the build uses: better or equal at every size. **A ~3% 0.5B regression survives**, and
goes to the owner, as S0 said it would.
