# Per-group activation scales for W4A8 / W8A8 (2026-09)

> **Status 2026-09-25: Track A quality gate NOT PASSED as registered (int8int8 met; int4 limited by weights) → both tracks. Track B: both candidates FAIL. Track A speed gate: 7B SHIP, 1.5B FAIL/AMBIGUOUS (fixed per-call costs). CUDA per-32 decode built (branch `actgroup-wiring`). Next: remove the small-shape overhead, re-gate, release aikit.** Owner decision 2026-09-25: this is the fix for
> `queue-engineering.md` H2. The guard shipped first (`dcbbaa91`); this lifts it.

## Why

Every quantized projection in goinfer (int4 = W4A8, int8int8 = W8A8, int4mix, and every resident GPU
projection) quantizes its f32 input to int8 with **one max/127 scale per activation row**. A family whose
projection inputs carry massive outliers loses nearly the whole row to rounding:

- Phi-3-mini's `down_proj` input has max/rms ≈ 80–90 in most layers at a filler position (layer 4:
  max 563). One per-row scale rounds 99.9% of the row to zero.
- goinfer's own f32 matches HF exactly (cosine 1.000000 at all 129 positions). int4 and int8int8 fall
  to cosine ≈ 0 by position 16.
- Simulated on HF, per-32 activation scales restore cosine 0.9998 (median) / 0.980 (min) with f32
  weights. The full record is H2.

llama.cpp quantizes activations per block (Q8_0 / Q8_K), which is why its Q4 output stays coherent on
the same weights.

**Blast radius beyond Phi-3: activation damage is general, and Phi-3 is the extreme case.** H2 step 1
(2026-09-25, raw data and tool in [`measurements/actquant-sweep-2026-09-25/`](../measurements/actquant-sweep-2026-09-25/)).

**Method.** goinfer on CPU, each quant against goinfer's own f32 on the same checkpoint, per position, on
two chat-templated prompts of ~130 tokens: the essay-v2 filler, and a natural prose paragraph. `int8`
(weight-only) against `int8int8` differs ONLY in activation quantization. Cells are the 10th-percentile
logit cosine over positions, and the first position where cosine drops below 0.9 (−1 = never).

| family | int8 (w-only) p10 | int8int8 p10 filler / prose | int4 p10 filler / prose | first < 0.9 (int4) |
|---|---|---|---|---|
| phi3-mini | ≥ 0.97 | **0.322 / 0.958** | **0.236 / 0.944** | 0 |
| qwen2.5-7b | ≥ 0.99 | **0.833** / 0.972 | **0.152** / 0.942 | 2 |
| olmo3-7b | ≥ 0.98 | 0.976 / 0.968 | **0.798** / 0.936 | 2 |
| llama3.2-1b | ≥ 0.98 | 0.965 / 0.974 | **0.704 / 0.729** | 1 |
| smollm3-3b | ≥ 0.98 | 0.993 / 0.989 | 0.965 / **0.847** | 3–4 |
| qwen3-1.7b | ≥ 0.97 | 0.992 / 0.980 | 0.969 / **0.890** | 0 |
| granite-4.2-3b | ≥ 0.99 | 0.991 / 0.987 | 0.976 / 0.950 | 0 |
| qwen3.5-0.8b | ≥ 0.99 | 0.995 / 0.994 | 0.962 / 0.954 | 0 |
| gemma3-1b | ≥ 0.99 | 0.997 / 0.995 | 0.983 / 0.977 | 1 |
| tinyllama-1.1b | ≥ 0.99 | 0.981 / 0.998 | 0.968 / 0.990 | 0 |
| qwen2.5-coder-1.5b | ≥ 0.99 | 0.999 / 0.998 | 0.994 / 0.991 | 0 |
| mistral-7b | 1.000 | 0.999 / 0.999 | 0.995 / 0.997 | never |

Not measured: lfm2.5 (its own-forward layers have no logit-capture seam). olmo3-7b's int8int8 minimum
(−0.14) sits at position 2; int4 widens it on filler. Three groups:

- **Damage confined to the first token or two** (p10 ≥ 0.95; the low minimum sits at position 0, the
  "attention sink" token whose activations are the classic massive outliers): most families here.
- **Widespread damage from activation quantization** (int8int8 itself falls): **phi3-mini** and
  **qwen2.5-7b** (int8int8 p10 0.83 on filler, and int4 p10 0.15). This is what per-group activation
  scales fix, and qwen2.5-7b is one of the peer-benchmark models.
- **Widespread damage from the int4 WEIGHTS, not the activations** (int8int8 fine, int4 not):
  **llama3.2-1b**, and on prose smollm3-3b and qwen3-1.7b. Per-group activation scales will not fix
  these. goinfer's int4 is symmetric g32, re-quantized from an already-Q4 GGUF (a double quantization),
  so it is a separate lever: int4 weight quality, with mins or k-quant style scales. It is out of scope
  here and should get its own queue entry.

Weight-only int8 holds everywhere, which is why the Phi-3 guard is safe. Mistral-7b is clean at every
precision.

## The change

- **Scale granularity:** one scale per 32 input elements, matching `int4GroupSize`. The scale buffer
  grows from M to M×K/32 floats.
- **W4A8:** every kernel already turns each 32-element group (or each 8-element word) into f32 using the
  weight's group scale. The multiply becomes `wScale[g]·aScale[g]`, and the final `·aScale` goes away.
  This is cheap.
- **W8A8:** every kernel accumulates the whole row in one int32. It needs per-group int32 partials,
  each scaled, which restructures the inner loop.
- **Keep a per-row API** for callers outside goinfer's forward: aikit `ann` FlatI8, anncuda/annmetal,
  and the ViT kernels.

## Where (survey of 2026-09-25; paths as of that date)

| area | quantizers | W8A8 (restructure) | W4A8 (cheap) | difficulty |
|---|---|---|---|---|
| aikit CPU amd64 | Go (`linalg/quant.go`) | dotI8 AVX2/VNNI, Tile4x1 asm | fold AVX2/VNNI, tile, splithalf asm | hard (asm) |
| aikit CPU arm64 | Go + NEON asm | dotI8 NEON/SDOT, Tile4x4 asm | ~12 SDOT kernels + tiles | hard (asm) |
| aikit CUDA (`gpu/gemv_quant.cu`) | none | gemv_w8a8_fwd | gemv_w4a8_fwd | medium (PTX regen) |
| goinfer CUDA | 10 kernels + 4 fused | gemv_w8a8_batched | rn, mma, moe×3, fused×4 | medium–hard (frozen PTX) |
| Metal | 8 kernels | 3 | ~13 | easy–medium |
| WebGPU | host + 6 shaders | 9 | 3 | easy–medium |

## Traps named up front

1. **The zero-scale fast path.** The CPU W8A8/W4A8 spans skip a whole row when `aScales[i] == 0`. A
   zero *group* is not a zero row. The backends also disagree on the all-zero scale today: 0 on CPU and
   CUDA, 1 on Metal and WebGPU (`QuantizeRowsInt8`).
2. **Pairs must stay bit-identical.** Decode and batched kernels are pinned bit-identical in pairs:
   aikit `gemv_w4a8_fwd` ↔ goinfer `gemv_w4a8_rn`, and `gemv_w8a8_fwd` ↔ `gemv_w8a8_batched`. Each
   pair must change together, with every new FMA written explicitly (`TestKernelFMALint`).
3. **Frozen PTX.**
   - `glue.ptx` and `moe.ptx` are audited 12.6.85 artifacts: regenerate them with the pinned NVRTC venv.
   - The rest were built with 12.9.86.
   - Gates: `TestPTX_matchesSourcesAndBindings`, aikit `TestPTXReproducible`.
4. **The 4-row asm tiles** (W4A8, both arches) share one weight-scale broadcast across four activation
   rows. They need four activation-scale broadcasts per group, in the hottest loops.
5. **`gemm_w4a8_mma`** applies the activation scale in its epilogue. It needs a shared-memory scale tile
   of one scale per row per group, for each KSTEP.
6. **Every Go caller that sizes `aSc` / `aScales`** as 1 or M must be resized: CUDA bindings, the Metal
   and WebGPU callers, the LoRA hooks, and the host dequant in `cuda/prefill.go`.
7. **Every precision golden re-baselines:**
   - `testdata/int4_forward_goldens.json` (+ `scripts/refresh_parity_hashes.sh`);
   - `testdata/metal_snapshot_golden.json`;
   - CUDA's CPU-oracle tests;
   - aikit's quant/tile tests.

   Re-baselining is only legitimate with the mechanism written down, which this doc is.

## Pre-registered quality gate for the Go reference (2026-09-25, before the run)

The reference (aikit `linalg/actgroup.go`, `SetActQuantGroup(32)`) re-runs the H2 step-1 sweep: same 12
families, same two prompts, same f32 baseline. Only `int8int8` and `int4` change under it, since
weight-only `int8` and f32 never quantize activations. The metric is the 10th-percentile per-position
logit cosine against f32 (p10), as in the table above.

- **PASS:** phi3-mini and qwen2.5-7b reach grouped `int8int8` p10 ≥ 0.95 on both prompts, and grouped
  `int4` p10 ≥ 0.90 on both. No family's grouped p10 falls more than 0.005 below its per-row p10 at the
  same quant.
- **AMBIGUOUS → parked, investigate before any kernel work:** phi3-mini or qwen2.5-7b grouped `int8int8`
  p10 in [0.90, 0.95), or grouped `int4` p10 in [0.80, 0.90).
- **FAIL:** grouped `int8int8` p10 < 0.90 for either model. Per-32 is then not enough, and the plan
  changes (finer groups, or smoothing) before any SIMD/PTX is written.

`int4` is judged at a lower bar on purpose: its weights carry their own error (llama3.2-1b shows int4
weight damage even where `int8int8` is clean), and this change does not touch weights.

## Gate result (2026-09-25): NOT PASS — activations fixed, int4 weights now the limit

Run with the aikit reference (`a71b204`, `SetActQuantGroup(32)`), same 12 families, prompts and f32
baseline as step 1. Raw data: [`measurements/actquant-sweep-g32-2026-09-25/`](../measurements/actquant-sweep-g32-2026-09-25/).
p10 cosine against f32, per-row → per-32:

| family | int8int8 filler / prose | int4 filler / prose |
|---|---|---|
| phi3-mini | 0.322 → **0.973** / 0.958 → **0.998** | 0.236 → **0.766** / 0.944 → 0.970 |
| qwen2.5-7b | 0.833 → **0.991** / 0.972 → **0.998** | 0.152 → **0.307** / 0.942 → 0.958 |
| qwen2.5-coder-1.5b | 0.999 → 1.000 / 0.998 → 1.000 | 0.994 → 0.995 / 0.991 → 0.993 |
| qwen3-1.7b | 0.992 → 0.999 / 0.980 → 0.997 | 0.969 → 0.985 / 0.890 → 0.915 |
| qwen3.5-0.8b | 0.995 → 0.999 / 0.994 → 0.999 | 0.962 → 0.968 / 0.954 → 0.962 |
| gemma3-1b | 0.997 → 1.000 / 0.995 → 1.000 | 0.983 → 0.985 / 0.977 → 0.980 |
| llama3.2-1b | 0.965 → 0.989 / 0.974 → 0.996 | 0.704 → **0.689** / 0.729 → 0.780 |
| tinyllama-1.1b | 0.981 → 0.998 / 0.998 → 1.000 | 0.968 → 0.983 / 0.990 → 0.992 |
| mistral-7b | 0.999 → 1.000 / 0.999 → 1.000 | 0.995 → 0.999 / 0.997 → 0.998 |
| granite-4.2-3b | 0.991 → 0.999 / 0.987 → 0.999 | 0.976 → 0.971 / 0.950 → 0.962 |
| smollm3-3b | 0.993 → 0.998 / 0.989 → 0.996 | 0.965 → 0.991 / 0.847 → 0.859 |
| olmo3-7b | 0.976 → 0.979 / 0.968 → 0.998 | 0.798 → **0.773** / 0.936 → 0.963 |

Graded against the pre-registration above, unchanged:
- **int8int8 criterion: MET.** phi3-mini 0.973 / 0.998 and qwen2.5-7b 0.991 / 0.998, both ≥ 0.95. Every
  family's int8int8 p10 rises.
- **int4 criterion: NOT MET.** Filler p10 is 0.766 (phi3-mini) and 0.307 (qwen2.5-7b), against 0.90.
  Both are also below the AMBIGUOUS band's 0.80. The pre-registration defined FAIL only for int8int8, so
  these land outside every band it drew; they are recorded as "not a pass", not re-banded after the fact.
- **No-regression criterion: NOT MET at int4.** Filler: llama3.2-1b −0.0145, olmo3-7b −0.0256
  (granite −0.0043, inside the 0.005 allowance). Every int8int8 cell improved.

**Reading, not a pre-registered claim.** Per-group activation scales remove the activation damage they
were built for: int8int8 isolates it, and it is gone. What limits int4 is the int4 **weights**: qwen2.5-7b
sits at 0.991 with int8 weights and 0.307 with int4 weights under the same per-32 activations. That is
the separate lever step 1 named, and it is not only double quantization: olmo3-7b and smollm3-3b
quantize bf16 safetensors to int4 directly and still sit at 0.77–0.86 on some prompts. The two int4
regressions are on the degenerate filler prompt, where near-ties dominate, which fits rounding noise but
has not been shown to be.

### Note (2026-09-25, after the gate): the int8int8 pass is prompt-sensitive on Phi-3

The gate's int8int8 criterion passed on its registered filler prompt (phi3-mini p10 0.973). The same
prompt with a `"\n\n"` before the instruction (141 tokens, as the CUDA resident test builds it) gives
**p10 0.930 for the CPU per-32 path itself** against f32 (min 0.126). That is below the 0.95 bar, on a
prompt the gate did not register. The verdict stands as graded, but this is not a clearance for every
input: lifting the Phi-3 guard to int8int8 + per-32 should cite both numbers. CUDA's per-32 resident
path matches the CPU per-32 path on that prompt (agreement median 0.99983; quality p10 0.934 vs
0.930), so this is the configuration's, not a backend's.

## Amendment 2026-09-25 (owner decision after the gate): both tracks

The gate above did not pass, so this is a dated change of plan with its mechanism, not a re-reading of
the bar. **Mechanism:** the int8int8 criterion, which isolates activation quantization, was met on every
family. The int4 shortfall is int4 *weight* error, which per-group activation scales were never meant to
touch. So:

- **Track A, per-group activation scales, proceeds** to the kernels (Order of work, steps 3–6). The
  Phi-3 guard lifts only to `int8int8` with per-32 activations, which passed; `int4` stays guarded for
  Phi-3 until Track B clears it.
- **Track B, int4 weight quality**, with its own bar, pre-registered below before any run.

### Track B — why int4 weights, and the first candidates

goinfer's int4 quantizer (aikit `QuantizeGroupInt4Row`) sets `scale = max|w|/7` and clamps codes to
[−7, 7]. It uses **15 of the 16 levels** the nibble holds (−8 is never produced) and does no scale
search. llama.cpp's Q4_0 maps the signed extreme to −8 (all 16 levels); its k-quants search the scale.
Both candidates keep goinfer's format exactly (decode `(nibble−8)·scale`, group 32), so neither touches
a kernel:

- **C1 full range:** scale = (signed element of largest magnitude) / −8. That element becomes code −8,
  the rest round into [−8, 7]. The scale may be negative, which the decode handles as-is.
- **C2 MSE scale:** per group, try scales max|w|/d for d over a small grid in [6.5, 8.5], both signs,
  round and clamp to [−8, 7], and keep the lowest squared error.

### Track B pre-registered gate (2026-09-25, before the run)

Same sweep, same 12 families, both prompts, per-32 activations ON (the state Track A ships), `int4`
only, one run per candidate. Metric: p10 logit cosine against f32. Every band is defined this time:

- **PASS:** phi3-mini and qwen2.5-7b int4 p10 ≥ 0.90 on both prompts, AND no family's int4 p10 more
  than 0.005 below its current per-32 int4 p10 (the Gate result table above).
- **AMBIGUOUS → parked:** the lower of the two models' filler p10 in [0.80, 0.90), with no regression
  beyond 0.005.
- **FAIL:** below 0.80, or any family regressing beyond 0.005. A failed candidate is not shipped.

If both candidates pass, the one with the higher minimum over (phi3-mini, qwen2.5-7b) × (filler, prose)
wins. If neither passes, int4 does not serve these families; they default to `int8int8` once Track A lands.

### Track A — decode W4A8 without new assembly (2026-09-25)

The M=1 W4A8 kernels on both arches return Σ_g int32dot_g · scale[g] and apply the per-row activation
scale afterwards, in Go. Fed the combined scale `wS[j,g]·aS[g]` per group, with no final multiply, the
**unchanged** kernel computes the per-group product exactly (float multiply is commutative, so
`aS·wS` = `wS·aS` bit-for-bit). The cost is one f32 multiply per 32 weights. Built in aikit for the
amd64 split-half AVX2 kernel and the arm64 row4 kernels (S-05 fold on and off); both match the Go
reference to accumulation order (tests on amd64, and on arm64 under qemu). What it cannot cover:
- the M>1 tiles (prefill), where four activation rows share one scale stream;
- W8A8, whose kernels accumulate the whole row in one int32.

Those still fall back to the reference and need real kernel work.

### Track A pre-registered speed gate (2026-09-25, before any timing)

Measured when the box is idle (not during a sweep), same process, per-row and per-32 interleaved:

- **Kernel level** (an aikit Go benchmark at goinfer's decode shapes: qwen2.5-coder-1.5b and qwen2.5-7b
  projection K×N, M=1), per-32 time ÷ per-row time: **≤ 1.05 SHIP; (1.05, 1.10] AMBIGUOUS → optimize
  before shipping** (e.g. hoist the combined scales); **> 1.10 FAIL** for that kernel, which then gets a
  real assembly variant.
- **End to end** (goinfer CPU int4 decode, 1.5B and 7B, `bench_compare.sh`, goinfer against goinfer):
  per-32 ÷ per-row tok/s **≥ 0.97 SHIP; [0.93, 0.97) AMBIGUOUS; < 0.93 FAIL.**

## Track B result (2026-09-25): both candidates FAIL

Raw: [`measurements/actquant-trackb-2026-09-25/`](../measurements/actquant-trackb-2026-09-25/). int4 p10 with
per-32 activations. The bar needs ≥ 0.90 for phi3-mini and qwen2.5-7b on both prompts and no family
regressing more than 0.005 against the current per-32 int4.

| candidate | phi3-mini filler / prose | qwen2.5-7b filler / prose | regressions > 0.005 |
|---|---|---|---|
| current (max/7) | 0.766 / 0.970 | 0.307 / 0.958 | — |
| fullrange | 0.762 / 0.963 | **0.863** / 0.962 | 4: phi3 prose −0.007, tinyllama −0.007, granite −0.029, smollm3 −0.028 |
| mse | 0.782 / **0.680** | 0.808 / 0.972 | 5, incl. phi3 prose −0.290 and olmo3 prose −0.093 |

Both FAIL. `fullrange` has large wins elsewhere (olmo3-7b filler 0.773 → 0.962, smollm3 prose 0.859 →
0.956, llama3.2-1b +0.04 to +0.11) and still regresses four families, so the default stays. Lower
weight MSE is not better output: `mse` has the lowest reconstruction error and the worst phi3-mini
prose. The qwen2.5-7b `mse` cell was re-run alone after the first attempt was refused by the memory-fit
guard while a concurrent test held ~15 GB. **Reading:** neither symmetric rule recovers int4 on the
Q4_K GGUF sources, which are asymmetric (a per-block minimum). The next int4 lever is asymmetric int4
or keeping Q4_K as-is, both format changes. Parked until owned.

## Track A speed gate result (2026-09-25): 7B SHIP, 1.5B FAIL / AMBIGUOUS

Raw: [`measurements/actquant-speedgate-2026-09-25/`](../measurements/actquant-speedgate-2026-09-25/). The box
was idle (load < 0.8 before every step), with aikit at `c946e31` plus the goinfer branch.

**End to end** (`BenchmarkDecode`, CPU, per-row and per-32 in separate processes, order alternating,
first sample of each process discarded, 6 paired rounds):

| model | per-row tok/s | per-32 tok/s | ratio (min–max) | verdict |
|---|---|---|---|---|
| 1.5B int4 | 20.07 | 18.30 | 0.912 (0.909–0.921) | **FAIL** |
| 1.5B int8int8 | 14.69 | 14.24 | 0.968 (0.963–0.972) | **AMBIGUOUS** |
| 7B int4 | 5.17 | 5.04 | 0.974 (0.973–0.976) | **SHIP** |
| 7B int8int8 | 3.56 | 3.50 | 0.984 (0.983–0.985) | **SHIP** |

**Kernel** (aikit benchmarks, M=1, count 10, medians): the memory-bound 7B FFN shapes are free (W4A8
1.01–1.04, W8A8 1.00–1.01, SHIP); the small shapes pay 1.10–1.23 (W4A8) and 1.18–1.39 (W8A8), FAIL.
The small shapes' run-to-run spread is 13–34%, so their medians are soft, but they agree with the
end-to-end 1.5B result.

**Mechanism** (from the code, not yet profiled): fixed per-call costs the per-row path does not pay.
- The grouped quantizer calls the per-row core once per 32 elements instead of one vectorized pass.
- Each parallel span allocates its combined-scale scratch per call.
- The W8A8 kernel's single f32 accumulator chain is latency-bound.

These costs vanish against a 7B matmul's weight stream and dominate a 1.5B one.
**Next:** fix those three, re-run the same gate, and cut the aikit release after it passes.

## Order of work

1. **Measure first:** the H2 step-1 sweep sizes the problem across families. A W8A8 perf baseline per
   backend (bench_compare.sh for goinfer-vs-goinfer) is taken before any kernel change.
2. **aikit CPU, Go reference first:** per-group quantizer + Go W4A8/W8A8 spans, behind a flag. Measure
   quality (the sweep, re-run) and CPU speed. Kill threshold: pre-register it before building the asm.
3. **aikit asm, amd64 then arm64:** W4A8 first (cheap), then the W8A8 per-group partial kernels.
4. **CUDA:** the W4A8 family (rn, mma, moe, fused), then the W8A8 batched kernel; PTX regenerated; the
   aikit/goinfer pairs in lockstep.
5. **Metal, then WebGPU.**
6. **Re-gate every family**, re-run the peer sweep, and lift the Phi-3 guard only after a long-prompt
   quality gate passes at int4.

<!-- doc-reviewed: 2026-09-25 -->
