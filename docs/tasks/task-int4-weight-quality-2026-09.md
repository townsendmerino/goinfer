# Task: int4 weight quality: stop re-quantizing Q4_K (2026-09-26)

**Owner decision 2026-09-26:** this goes ahead of R6 phase 3, the f16 resident KV cache, which is parked
(`docs/prompts/cuda-r6-flash-decode.md`). The reasons are in `docs/measurements/f16kv-baseline-2026-09-26.md`:
Phi-3 on CUDA is 0.70× the peers at depth 128 and 0.68× at 2048. That gap comes from its **int8
weights**, the only safe setting after H2, not from the KV cache.

## Why

H2's per-32 activation scales fixed activation quantization
(`docs/tasks/task-actquant-pergroup-2026-09.md`). At int8int8 + per-32, phi3-mini scores p10 0.973 /
0.998 and qwen2.5-7b 0.991 / 0.998 against f32. At int4 + per-32 the same models score 0.766 / 0.970
and 0.307 / 0.958. The difference is int4 **weight** error, and Track B showed that neither symmetric
rule recovers it (both candidates FAIL).

The mechanism is double quantization. `buildWeightsFromGGUF` (`decoder/gguf.go`) dequantizes every GGUF
tensor to f32 and then re-quantizes it to goinfer's symmetric int4, which decodes as
`(nibble−8)·scale` with `scale = max|w|/7`. A Q4_K tensor is already 4-bit, and **asymmetric**: per
32-element sub-block, `w = d·sc·q − dmin·m` with `q ∈ [0, 15]`. Its values sit on a grid that
symmetric int4 cannot represent, so the transcode adds a second rounding on top of the file's own.
The quality reference in every sweep here is goinfer's f32 from the same GGUF, which means the
dequantized Q4_K values themselves. **A layout that stores Q4_K's own per-sub-block scale and
minimum reproduces those weights exactly.**

## What the files contain (layer matmuls, from the GGUF headers)

| model | Q4_K | Q5_K | Q6_K | other |
|---|---|---|---|---|
| phi3-mini | 64% | 25% | 11% | |
| qwen2.5-7b, qwen2.5-coder-1.5b, llama3.2-1b, mistral-7b | ~85% | | ~15% | |
| gemma3-1b | 19% | | 15% | 65% Q5_0, <1% Q8_0 |

Only the Q4_K share can be carried exactly in 4 bits. The rest is the design question: Q5_K/Q6_K kept
near-exact at int8 (8 bits a weight) or re-quantized to 4 bits (lossy).

## Bytes per decoded token, layer matmuls (why the split matters)

At the measured ~330 GB/s, decode is weight-byte-bound at shallow depth, so bytes ≈ speed:

| model | today's int4 (4.5 bpw) | Q4_K exact + rest at int8 | everything 4-bit at 4.5 bpw | today's int8int8 |
|---|---|---|---|---|
| phi3-mini | 2.04 GB | 2.61 GB (+28%) | 2.04 GB | 3.62 GB |
| qwen2.5-7b | 3.67 GB | 4.10 GB (+12%) | 3.67 GB | 6.53 GB |

"Q4_K exact" at 4.5 bpw means keeping Q4_K's own super-block layout (f16 `d`/`dmin`, 6-bit sub-block
scales and minimums), which is the bytes goinfer's int4 already spends. A plain affine layout with an
f16 scale and f16 offset per 32 costs 5.0 bpw.

## Phase 0: the quality experiment (no kernels), PRE-REGISTERED 2026-09-26 before any run

**Instrument.** The H2 sweep tool (`docs/measurements/actquant-sweep-g32-2026-09-25/actsweep.go.txt`),
unchanged in its metric: p10 logit cosine against goinfer's f32 on the filler and prose prompts, with
per-32 activations on. A test-hook seam in the GGUF loader fake-quantizes each layer weight row by its
source GGUF type. Each arm is then loaded at **int8int8** with per-32 activations, so weights ride
the int8 carrier the int8int8 cells already use. The carrier adds the same small int8 error to every
arm, including the reference arm A0. Embeddings and the LM head are untouched in every arm.

**Families:** the six GGUF sources: phi3-mini, qwen2.5-coder-1.5b, qwen2.5-7b, gemma3-1b, llama3.2-1b,
mistral-7b. bf16-sourced families have no Q4_K to preserve and are out of scope for Phase 0.

**Arms, all same session:**
- **C0**: today's int4 with per-32, the regression baseline;
- **A0**: carrier only, no fake-quant. Represents "Q4_K exact, every other tensor near-exact";
- **A2**: Q4_K rows unchanged; Q5_K/Q6_K/Q5_0/Q8_0 rows → affine int4, per-32 min/max
  (`s = (max−min)/15`, codes 0..15);
- **A3**: Q4_K rows unchanged; the other rows → goinfer's symmetric int4 (`max/7`, per 32);
- **A4**: every row → affine int4 per-32 min/max, Q4_K included. Tests whether *exactness* matters or
  a generic asymmetric quantizer is enough. It is also the only candidate for bf16 sources.

**Bar, Track B's, reused unchanged:** PASS = phi3-mini and qwen2.5-7b p10 ≥ 0.90 on both prompts, AND no
family's p10 more than 0.005 below its own C0. AMBIGUOUS → parked = the lower of those two models'
filler p10 in [0.80, 0.90) with no such regression. Everything else FAILs.

**Decision rule, fixed now:**
1. If A3 passes, build native Q4_K plus symmetric int4 for the rest: **4.5 bpw, today's bytes**.
2. Else if A2 passes, build native Q4_K plus affine int4 for the rest (4.5 to 5.0 bpw).
3. Else if A0 passes, build native Q4_K plus int8 for the rest, and state the byte cost from the table
   above in the build doc.
4. If A4 also passes whichever arm decided, and its minimum over (phi3-mini, qwen2.5-7b) × (filler,
   prose) is within 0.01 of that arm's, **prefer a generic affine quantizer over a native Q4_K layout**.
   It is simpler, and it serves bf16 sources too.
5. If nothing passes, kill: int4 stays guarded for Phi-3, as today.

A arm that crashes or errors is reported void, not guessed.

### Amendment 2026-09-26, 1:40 pm PDT, before A5 has run: arm A5 added

**What prompted it:** a partial result. phi3-mini came back with A2/A3/A4 below the bar (0.675, 0.758
and 0.762 on filler) and A0 at 0.973 / 0.998. That shows squeezing the file's 5–6-bit tensors is
damaging. It does not show whether re-quantizing the **Q4_K** tensors to today's symmetric int4 is,
and no arm isolates that. If it is not, the fix needs no new format: `int4mix` already mixes int4 and
int8 per tensor through every existing kernel and residency path.

- **A5**: Q4_K rows → goinfer's symmetric int4 (`max/7`, f16 scale, the same rule as A3); every other
  row unchanged (the int8 carrier). Bytes are identical to A0 built natively: Q4_K tensors at
  4.5 bpw, the rest at int8.
- **Same bar, same families, same session as a re-run of A0 and C0** (A0 and C0 are repeated beside A5
  so it is compared within one session).
- **Rule, added ahead of the original rule's step 3:** if A5 passes, the build is a **loader policy
  only**: int4 (today's format) for tensors the GGUF stores at Q4_K, int8 for everything else. No
  new format, no kernels. A0's native Q4_K layout is built only if A5 fails. The A0−A5 gap is
  recorded either way, as information for a later quality decision, not as a gate.
- The original rule's other steps stand. A2, A3 and A4 are already failed on phi3-mini, a result that
  later families cannot change.

## Phase 0 result (2026-09-26): A0 PASSES, and so does the rule's step 3; A2, A3, A4 and A5 FAIL

Raw data, the tool (`int4q.go.txt`) and the tensor-type maps are in
[`measurements/int4q-phase0-2026-09-26/`](../measurements/int4q-phase0-2026-09-26/).

**Validity checks.** Every arm's filter saw every layer-matmul row and no unknown tensor: for example,
376,832 of 376,832 rows on llama3.2-1b, and 1,015,808 on phi3-mini, which includes the fused QKV and
gate/up via `fusedSplit`. The f16 rounding matches numpy on 5,108 values. In the A5 session, C0 and A0
reproduced the first session to three decimals on every family.

p10 cosine against f32, filler / prose, per-32 activations:

| family | C0 today's int4 | **A0** | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|---|
| llama3.2-1b | 0.705 / 0.769 | **0.987 / 0.996** | 0.815 / 0.841 | 0.850 / 0.898 | 0.799 / 0.823 | 0.669 / 0.875 |
| gemma3-1b | 0.985 / 0.979 | **1.000 / 1.000** | 0.994 / 0.990 | 0.991 / 0.984 | 0.993 / 0.989 | 0.995 / 0.995 |
| qwen2.5-coder-1.5b | 0.995 / 0.993 | **1.000 / 1.000** | 0.999 / 0.999 | 0.999 / 0.998 | 0.998 / 0.998 | 0.995 / 0.995 |
| phi3-mini | 0.734 / 0.969 | **0.973 / 0.998** | 0.675 / 0.980 | 0.758 / 0.982 | 0.762 / 0.981 | 0.796 / 0.987 |
| mistral-7b | 0.999 / 0.998 | **1.000 / 1.000** | 0.992 / 0.999 | 0.999 / 0.999 | 0.992 / 0.999 | 0.999 / 0.999 |
| qwen2.5-7b | 0.306 / 0.958 | **0.992 / 0.998** | 0.986 / 0.992 | 0.892 / 0.988 | 0.917 / 0.985 | 0.726 / 0.958 |

Graded against the bar (Phi-3 and qwen2.5-7b ≥ 0.90 on both prompts; no family more than 0.005 below
its C0):
- **A0: PASS.** Both gate models are at or above 0.973, and every family is above its C0.
- **A2: FAIL.** phi3-mini 0.675 on filler, below the AMBIGUOUS floor and 0.059 below C0; mistral −0.007.
- **A3: FAIL.** phi3-mini 0.758 on filler (below 0.80) and qwen2.5-7b 0.892.
- **A4: FAIL.** phi3-mini 0.762 on filler; mistral −0.007.
- **A5 (amendment): FAIL.** phi3-mini 0.796 and qwen2.5-7b 0.726 on filler; llama3.2-1b −0.036
  against C0.

**Decision, by the rule as registered:** A3 fails, A2 fails and A5 fails (the amendment's step), so
**step 3 applies. Build native Q4_K for tensors the GGUF stores at Q4_K, and int8 for everything
else**, at the byte cost stated above. Rule 4 does not apply, because A4 failed. Nothing was
re-banded.

**Reading (not a pre-registered claim).** Both error sources matter, and they compound. On qwen2.5-7b,
re-quantizing only the Q4_K tensors (A5) scores 0.726 on filler. Squeezing only the 5–6-bit tensors
(A3) scores 0.892. Doing both, which is today's int4 (C0), scores 0.306. Phi-3 does not tolerate *any*
4-bit form of its 36% of Q5_K/Q6_K tensors; qwen2.5-7b tolerates the affine form (A2 0.986). So no
single "squeeze the rest" policy serves both. One finding reaches past this task: **today's default
int4 on qwen2.5-7b, the headline peer model, scores 0.306 on the filler prompt.** The degenerate
filler exaggerates it (prose is 0.958), but it is a quality debt on a benchmark row.

## Phase 1+ (scoping, after the Phase 0 record)

**Format:** Q4_K's own super-block (256 weights: f16 `d` and `dmin`, 12 bytes of packed 6-bit
sub-block scales and minimums, 128 bytes of codes). It is 4.5 bpw and exact. A per-32 f16 scale and
offset (5.0 bpw) is neither: `d·sc` has up to 17 significant bits and would round.

**Kernel math per 32-element group g** (unsigned codes q ∈ [0, 15], per-32 int8 activations):
Σ w·a = aS_g · (s_g · Σ q·a_int − m_g · Σ a_int). `Σ a_int` is computed once per activation row, as
llama.cpp's `bsums` are. The unsigned × signed byte dot is AVX2's `vpmaddubsw`, a natural fit.

**Byte cost of the decision** (layer matmuls, from the table above): phi3-mini **2.61 GB**, against
3.62 GB at today's int8int8 default (−28%) and 2.04 GB at today's int4. qwen2.5-7b **4.10 GB**, against
3.67 GB at int4 (+12%).

**Owner decisions before the build:** where the new mode becomes the default. Candidates: Phi-3,
where it strictly beats today's int8int8 on bytes; every Q4_K GGUF, trading +12% bytes on K_M files
for the quality above; or opt-in only.

Format and kernels per the decision: aikit `WeightMat`, the CPU W4A8 kernels (amd64/arm64), CUDA
`gemv_w4a8`, then Metal/WebGPU. Then the quality gate on the real path, and the speed gate against
peers on the Phi-3 and 7B cells.

<!-- doc-reviewed: 2026-09-26 -->
