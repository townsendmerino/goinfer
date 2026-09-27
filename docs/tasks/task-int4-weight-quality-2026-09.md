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

**Owner decision 2026-09-26: build Phase 1**, opt-in first. Order: aikit format and Go reference, then
the amd64 AVX2 kernel and the goinfer loader mode (step 1a); then the CUDA kernel (1b); then arm64,
Metal and WebGPU (later, each scoped separately). The default scope is decided after the speed gate.

### Phase 1 design (step 1a)

- **Mode `--quant q4k`:** tensors the GGUF stores as Q4_K keep their super-block bytes verbatim in a
  new aikit `WeightMat` kind. Every other layer matmul is int8 W8A8. Activations are per-32 int8
  everywhere; the Q4_K kind is per-32 by construction, because its 32-weight sub-blocks are the
  activation groups. Embeddings and the LM head follow the existing embedding policy.
- **GGUF sources only**, and only from the GGUF directly. `.giw` serialization and the sidecar cache
  decline `q4k` until a later step adds a `.giw` kind, so a `q4k` load never goes through the
  transcode cache.
- **GPU backends decline `q4k`** until their kernels exist (1b: CUDA), with the reason reported.

### Phase 1a gates, PRE-REGISTERED 2026-09-26 before any build or run

**Correctness (unit tests, aikit):** the Go reference and the AVX2 kernel both match an f64
evaluation of `Σ (d·sc·q − dmin·m)·a` over dequantized-then-requantized activations: relative error
≤ 1e-6 on random and saturated blocks, at M = 1 and M > 1, single- and multi-op batch, odd N.

**Quality (real path, CPU amd64):** the Phase 0 tool's metric and prompts, with the new mode loaded
for real (no row filter). **PASS:** every one of the six families' p10 on both prompts is ≥ its Phase
0 A0 value − 0.01. Anything lower is a defect to find, not a result to band.

**Speed (CPU amd64, end to end, goinfer against goinfer):** `BenchmarkDecode`, 6 paired rounds,
alternating order, separate processes, idle box, the speed-gate protocol of
`task-actquant-pergroup-2026-09.md`:
- **qwen2.5-7b, `q4k` ÷ today's `int4`** (+12% bytes projected): **≥ 0.85 SHIP; [0.78, 0.85)
  AMBIGUOUS; < 0.78 FAIL.**
- **phi3-mini, `q4k` ÷ `int8int8` with per-32** (−28% bytes projected): **≥ 1.15 SHIP; [1.05, 1.15)
  AMBIGUOUS; < 1.05 FAIL.**

The CPU W4A8 kernels are not purely bandwidth-bound (the amd64 split-half kernel is shuffle-port
bound), so these bands are set below the byte projections on purpose. The CUDA peer gate is 1b's,
registered before 1b is built.

Format and kernels per the decision: aikit `WeightMat`, the CPU W4A8 kernels (amd64/arm64), CUDA
`gemv_w4a8`, then Metal/WebGPU. Then the quality gate on the real path, and the speed gate against
peers on the Phi-3 and 7B cells.

<!-- doc-reviewed: 2026-09-26 -->

## Phase 1a results (2026-09-26): quality PASS; speed FAIL on the first kernel, SHIP after a one-instruction fix

Raw data, the gate tool (`q4kgate.go.txt`) and the logs are in
[`measurements/q4k-phase1a-2026-09-26/`](../measurements/q4k-phase1a-2026-09-26/). Code lives in aikit
(`683937f` the Q4_K kind and Go reference, `d1b71a5` `embed.GGUFFile.Q4KRaw`, `5910e74` the AVX2
kernel, `366b8fe` its fix) and goinfer branch `q4k-phase1` (`cfda301f`, `--quant q4k`). Both are
unreleased and unmerged; see "Next" below.

**Correctness: PASS.** The Go reference and the AVX2 kernel match a float64 evaluation within
relative error 1e-6 at M = 1 and 3, odd N, random and saturated blocks, serial and fanned out. Four
deliberate breaks each turn tests red: dropping the minimum term, the wrong nibble, a flipped
minimum sign, a corrupted scale-unpack mask. On a real Q4_K_M file, every Q4_K tensor (96) wrapped
natively dequantizes bit-identically to `RowDequantizer`. The Go path also passes on arm64 under qemu.

**Quality: PASS on all six families**, real `--quant q4k` loads, AVX2 kernel. The bar is each
family's Phase 0 A0 − 0.01:

| family | q4k p10, filler / prose | bar | argmax agreement |
|---|---|---|---|
| llama3.2-1b | 0.993 / 0.997 | 0.977 / 0.986 | 0.982 / 0.993 |
| gemma3-1b | 1.000 / 1.000 | 0.990 / 0.990 | 0.993 / 0.948 |
| qwen2.5-coder-1.5b | 1.000 / 1.000 | 0.990 / 0.990 | 0.979 / 0.983 |
| phi3-mini | 0.969 / 0.999 | 0.963 / 0.988 | **0.572** / 0.928 |
| mistral-7b | 1.000 / 1.000 | 0.990 / 0.990 | 0.993 / 0.992 |
| qwen2.5-7b | 0.992 / 0.999 | 0.982 / 0.988 | 0.993 / 0.957 |

Two families were first gated on the Go reference kernel (`quality/sweep-goref-partial.jsonl`). The
run was then restarted so that the kernel which ships is the one graded. **Caveat:** phi3-mini's
filler prompt passes on cosine with argmax agreement 0.572. That prompt is dense with near-ties, and
Phase 0's A0 behaved the same way, so it is the configuration's sensitivity rather than a q4k
defect. Phi-3's quality claim should rest on the prose prompt and on real output, not on filler
cosines.

**Speed, run 1: FAIL** (`speed-run1-legacy-movq/`). qwen2.5-7b 0.532× today's int4; phi3-mini 0.816×
int8int8. The profile put 76% of decode in `dotQ4KAVX2` itself. A bare-routine microbenchmark read
1900 ns per 14-block row, barely ahead of the scalar Go oracle, where the instruction count
predicts about 30 cycles per block. The cause was two `MOVQ reg, X` per block, which the Go
assembler encodes as legacy SSE. Writing an XMM register that way, inside an AVX loop with dirty
upper YMM halves, cost about 450 cycles per block on the Ryzen 7 3700X. With VEX `VMOVQ` the row
takes 115 ns (16×), numerics unchanged. That is a new mechanism, so the gate was re-run with its
registered bands and protocol unchanged. Run 1 stays as the record of the failure.

**Speed, run 2: SHIP both** (`speed-run2/`), 6 paired rounds each, CPU 55–70 °C:

| cell | baseline | q4k | ratio (range) | band | verdict |
|---|---|---|---|---|---|
| qwen2.5-7b, q4k ÷ today's int4 | 5.18 tok/s | 5.06 tok/s | **0.977** (0.975–0.983) | ≥ 0.85 | **SHIP** |
| phi3-mini, q4k ÷ int8int8 + per-32 | 6.52 tok/s | 8.53 tok/s | **1.308** (1.305–1.309) | ≥ 1.15 | **SHIP** |

**Protocol change, before any valid pair, owner's choice:** the idle check before each run changed from
"1-minute load average < 0.8" to "CPU ≥ 95% idle over 3 s from `/proc/stat`", with the CPU
temperature logged per run. The load average mostly waited 2–3 minutes for the previous run's own
trailing average to decay. The new check still refuses to measure while anything else runs, since
one busy thread of 16 reads about 94% idle. The one run taken under the old check is archived and
unused (`run-loadavg-aborted.log`).

**Next:**
- aikit release, then goinfer's bump, the parity refresh and the merge of `q4k-phase1`;
- the owner decides the default scope;
- 1b, the CUDA kernel, registered before it is built.

## Phase 1b: the CUDA resident kernel, PRE-REGISTERED 2026-09-26 before any build or run

**Design.** A Q4_K tensor is uploaded as its raw super-blocks (36 uint32 words per 256 weights). A new
`gemv_q4k_g32` matvec uses one warp per output row: its 32 lanes map onto a block's 32 code words
(coalesced), and each lane runs two `dp4a` (low and high nibbles) against per-32 int8 activations.
The kernel unpacks each block's f16 `d`/`dmin` and 6-bit sub-block scales and minimums.

The minimum term needs `aS_g · Σaq_g` per group. The per-32 quantizers (`actgroup.cu`'s `quantG32`)
write it into the second half of the activation-scale buffer they already fill, whose length goes
from K/32 to 2·K/32. Kernel signatures and the pipeline-field swap are unchanged, and the existing
per-32 matvecs read only the first half. Only `doG` gains a case; fusion, batched prefill and MoE
already decline under per-32. Residency stops declining `q4k` on CUDA, and keeps declining it on
Metal and WebGPU.

**Correctness (unit tests, RTX 2070 SUPER):**
- the quantizers' new sums equal the host twin exactly, and their scales and codes are unchanged;
- `gemv_q4k_g32` matches a host sum over the same codes, scales and minimums within relative error
  1e-5: random and saturated blocks, bias, accumulate.

**Agreement and quality:** Phi-3 on the 141-token filler prompt (the H2 resident test's), CUDA `q4k`
resident against CPU `q4k`:
- per-position logit cosine median ≥ 0.999;
- quality against f32: p10 within 0.01 of CPU `q4k`'s on the same prompt.

**Capacity:** Phi-3 `q4k` builds resident at the default context on the 8 GB card, which int8int8
cannot. Pass or fail.

**Speed.** Resident greedy decode, goinfer against goinfer, a new `BenchmarkResidentDecode` (cuda
module), 6 paired rounds, alternating order, separate processes, the 1a idle check:
- **phi3-mini, `q4k` ÷ int8int8 + per-32, at depths 128 and 2048** (−25% weight bytes projected):
  **≥ 1.15 SHIP; [1.05, 1.15) AMBIGUOUS; < 1.05 FAIL**, per depth; the cell's verdict is the worse
  of the two;
- **qwen2.5-7b, `q4k` ÷ today's int4, at depth 128** (+12% bytes projected): **≥ 0.85 SHIP;
  [0.78, 0.85) AMBIGUOUS; < 0.78 FAIL.**

A peer comparison against Ollama and llama.cpp (`scripts/bench_peer.py`) follows a SHIP as
information, not as a gate. If the Phi-3 cells SHIP, the owner decides the CUDA default.

## Phase 1b results (2026-09-26): all gates PASS on the second kernel; the first failed two ways, on record

Raw data, logs and the two diagnostic tools are in
[`measurements/q4k-phase1b-2026-09-26/`](../measurements/q4k-phase1b-2026-09-26/). The code is on branch
`q4k-cuda`.

**Correctness: PASS.** The quantizers' new per-group sums equal the host twin exactly. `gemv_q4k_g32`
is within relative error 1e-5 of a host float64 sum, random and saturated, with bias and accumulate.
Dropping the minimum term turns the test red, on both kernels.

**Agreement, run 1: FAIL**, median 0.99775 (bar 0.999), while quality against f32 was *better* than
the CPU's (0.961 vs 0.926). Diagnosis:
- **Isolation.** On real Phi-3 Q4_K tensors with one host-quantized activation, both the CPU and the
  CUDA matvec are within relative error ≤ 7.1e-7 of a float64 truth, so the kernel was not the source.
- **Per-layer differencing** (`layer-diff.txt`). The CUDA-vs-CPU divergence starts at **layer 4**,
  Phi-3's known outlier layer, in `q4k` (per-position minimum 0.834) and in the int8int8 control
  (0.977) alike.
- **Mechanism.** The CUDA per-32 quantizers used fast-math `__expf` (SiLU) and `rsqrtf` (RMSNorm). With
  precise `expf` and `1/sqrtf`, layer 4 recovers to 0.998 (`q4k`) and 0.9999 (control).

That was a new mechanism, so the gate was re-run.

**Agreement, run 3 (second kernel, precise math): PASS.** Median **0.99980**; quality p10 **0.940**
against CPU `q4k`'s 0.926. H2's int8int8 per-32 CUDA-vs-CPU agreement also rises, 0.99957 → 0.99984.

**Capacity: PASS.** Phi-3 `q4k` builds resident at the default **4096**-position context on the 8 GB
card, where int8int8 declines to the CPU.

**Speed, run 1 (first kernel): FAIL, stopped after one round** (`speed-run1-localmem/`). Phi-3 at depth
128 read 0.82× int8int8 + per-32. Under the "worse depth decides" rule nothing later could change
that, so the run stopped. Two causes:
- the scale-unpack helper's indexed byte array compiled to a 48-byte `__local_depot`, spilled per
  block (fixed: 73.7 → 84.7 tok/s);
- a lane per code word paid the block-header decode per 8 weights, which is compute-bound. The second
  kernel gives a lane one sub-block pair (64 weights) with 16-byte vector loads.

**Speed, run 2 (second kernel): SHIP on every cell** (`speed-run2/`), 6 paired rounds, CPU 54–77 °C:

| cell | baseline | q4k | ratio (range) | band | verdict |
|---|---|---|---|---|---|
| phi3-mini, depth 128, ÷ int8int8 + per-32 | 89.8 tok/s | 112.9 tok/s | **1.257** (1.255–1.257) | ≥ 1.15 | SHIP |
| phi3-mini, depth 2048, ÷ int8int8 + per-32 | 62.9 tok/s | 73.3 tok/s | **1.166** (1.164–1.166) | ≥ 1.15 | SHIP |
| qwen2.5-7b, depth 128, ÷ today's int4 | 81.6 tok/s | 72.6 tok/s | **0.890** (0.888–0.893) | ≥ 0.85 | SHIP |

**Owner decisions 2026-09-26:**
- Phi-3 from a `.gguf` defaults to `q4k` on CUDA as well as the CPU. Metal and WebGPU keep int8int8 +
  per-32 until they have a Q4_K kernel. Serve check: no flags → `decode path: cuda-resident (q4k)`
  at the full 4096 context, with a coherent answer.
- Every other Q4_K GGUF keeps the int4 default for now. A peer comparison of `q4k` against Ollama and
  llama.cpp decides first: `q4k` is 0.890× today's int4 on CUDA for qwen2.5-7b, which lands on the
  headline peer cells.

## Default for other Q4_K GGUFs on CUDA: the peer comparison, PRE-REGISTERED 2026-09-26 before any run

The owner held a `q4k` default for Q4_K GGUFs other than Phi-3 pending this comparison, because `q4k`
is 0.890× today's int4 on CUDA (qwen2.5-7b).

**Instrument.** `scripts/bench_peer.py`, CUDA, greedy, essay-v2 prompts, `BENCH_RUNS=3`. Engines: goinfer,
Ollama and llama.cpp, same weights as `peer-claim-2026-09-25.md`. Models: qwen2.5-coder-1.5b and
qwen2.5-7b (the headline dense cells). Depths 128, 2048 and 3900. Two sessions, back to back:
- **S1:** goinfer at `q4k` (`BENCH_QUANT_OVERRIDE=1.5B=q4k,7B=q4k`);
- **S2:** goinfer at today's default int4.

Each session carries its own peers, so its ratios are in-session. goinfer is served from `main` at
`210e7307` or later.

**Decision column:** goinfer ÷ llama.cpp, graded with `peer-claim-2026-09-25.md`'s bands and all-pairs
rule (LEVEL [0.97, 1.03], AHEAD > 1.03, the AMBIGUOUS variants, BEHIND < 0.97). Ollama is reported
beside it, because its `usage` sometimes reports no token count and voids its cells.

**Rule:**
- If every cell that is "level or ahead" (AHEAD, LEVEL, AMBIGUOUS-HIGH) against llama.cpp in S2 stays
  "level or ahead" in S1, `q4k` becomes the CUDA default for Q4_K GGUFs.
- If any such cell turns BEHIND in S1, int4 stays the default and `q4k` stays opt-in.
- If any turns only AMBIGUOUS-LOW, it is parked and the owner decides.
- A VOID cell on the llama.cpp side is re-run once, then reported as void.

The CPU default is not decided here.

## Peer comparison result (2026-09-26): int4 stays the default for other Q4_K GGUFs; q4k stays opt-in

Raw data: [`measurements/q4k-peer-2026-09-26/`](../measurements/q4k-peer-2026-09-26/). The two sessions are
comparable: llama.cpp's own tok/s moved at most 0.3% between them. S2's first attempt was refused at
startup (load average 1.03 left over from S1's servers). `bench_peer.py` exits 0 on a refusal, so the
script reported success with no results file; S2 was re-run after an idle wait (`run-s2.sh`,
`sweep-s2.log`).

| cell (CUDA decode, tok/s) | int4 | q4k | llama.cpp | int4 vs llama.cpp | q4k vs llama.cpp | q4k ÷ int4 |
|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b, depth 128 | 253.1 | 193.8 | 228.5 | AHEAD (1.108) | BEHIND (0.851) | 0.77 |
| qwen2.5-coder-1.5b, depth 2048 | 227.7 | 178.2 | 218.5 | AHEAD (1.042) | BEHIND (0.814) | 0.78 |
| qwen2.5-coder-1.5b, depth 3900 | 214.9 | 174.4 | 210.8 | LEVEL (1.019) | BEHIND (0.829) | 0.81 |
| qwen2.5-7b, depth 128 | 81.3 | 72.6 | 78.2 | AHEAD (1.040) | BEHIND (0.928) | 0.89 |
| qwen2.5-7b, depth 2048 | 76.3 | 68.6 | 76.1 | LEVEL (1.003) | BEHIND (0.901) | 0.90 |
| qwen2.5-7b, depth 3900 | 73.2 | 66.0 | 74.3 | LEVEL (0.985) | BEHIND (0.887) | 0.90 |

**Verdict, by the registered rule:** every cell that is level or ahead under int4 turns BEHIND under
q4k, with all pairs agreeing. **int4 stays the CUDA default for Q4_K GGUFs; q4k stays opt-in.**
Against Ollama, q4k reads level or ahead on 4 of 6 cells; that column does not decide anything here.

**New finding: q4k costs the 1.5B more than the 7B on CUDA** (0.77–0.81× int4, against 0.89–0.90×).
Phase 1b's speed gate measured only the 7B. A likely mechanism, not yet measured: the 1.5B's
1536-wide projections are only 6 Q4_K blocks, and `gemv_q4k_g32` covers 8 blocks per warp pass, so a
quarter of the lanes idle on every such row. Narrow-row geometry for `gemv_q4k_g32` is the kernel
follow-up if a q4k default is revisited, re-graded on these same cells.

## Toward a CUDA q4k default: lever 1, narrow-row geometry, PRE-REGISTERED 2026-09-26 before any build

**Why** (byte accounting against the peer session, `measurements/q4k-peer-2026-09-26/`): on qwen2.5-7b
all three engines decode at ~340 GB/s of weight bytes, so the 7B gap is bytes (lever 2, native Q6_K).
On qwen2.5-coder-1.5b the effective bandwidths differ: int4 246 GB/s, llama.cpp 224, **q4k 205**. That
gap is kernel efficiency. The likely cause: `gemv_q4k_g32` gives each row a whole warp (8 blocks × 4
sub-block pairs per pass), and a 1536-wide row is only 6 blocks, so 25% of lanes idle on every
q/k/v/o/gate/up row. **Change:** 8 lanes per row (4 rows per warp), each lane looping over its row's
(block, pair) tasks. Utilization: 1536 → 100%, 8960 → 97%, 3072/3584/18944 → 100%.

**Instrument:** `BenchmarkResidentDecode` (cuda), goinfer against goinfer, depth 128, 6 paired rounds,
alternating order, separate processes, idle check. Effective bandwidth = tok/s × weight bytes per
token (int4 0.970 GB, q4k 1.057 GB for the 1.5B).

**Bar:** 1.5B q4k effective bandwidth ÷ int4's (the same session):
- **≥ 0.90 PASS** (proceed to lever 2);
- [0.85, 0.90) AMBIGUOUS;
- < 0.85 FAIL.

Before this change the ratio is about 205 / 246 = 0.83. **Guard:** the 7B q4k's tok/s must not fall
below 0.98× its current kernel's (a same-session A/B, old kernel against new), or the change is
reworked.

## Lever 1 result (2026-09-26): FAIL — the 1.5B gap is the per-32 activation path, not the Q4_K kernel

Raw data: [`measurements/q4k-lever1-2026-09-26/`](../measurements/q4k-lever1-2026-09-26/). The narrow-row kernel
(8 lanes per row) is branch `q4k-narrow` at `de885035`, pushed and **not merged**.

| | median (6 paired rounds) | range | bar | verdict |
|---|---|---|---|---|
| 1.5B q4k effective bandwidth ÷ int4 | **0.811** | 0.803–0.821 | ≥ 0.90 (< 0.85 FAIL) | **FAIL** |
| 7B guard: new kernel ÷ old kernel | **0.926** | 0.925–0.927 | ≥ 0.98 | **FAIL** |

The hypothesis was wrong. The 1.5B ratio did not move (about 0.83 before), and 8 lanes per row cost
the 7B 7%: each lane now walks its tasks one after another, so fewer loads are in flight per row.
The main kernel stays.

**Diagnosis afterwards** (`fusion-check*.txt`, same binary, same session, two reps each, depth 128):

| CUDA decode, tok/s | int4 (per-row) | int4 + per-32 (same bytes) | q4k |
|---|---|---|---|
| qwen2.5-coder-1.5b | 254.0 / 252.8 | **189.3 / 188.3** | 193.2 / 188.3 |
| qwen2.5-7b | 81.6 / 81.5 | 79.3 / 79.3 | 72.6 / 72.6 |

- **1.5B: the per-32 path costs 25% on its own**, whatever the weight format. Fusion is off, and there
  are separate quantize launches and per-group scale work. At ~4 ms per token that overhead dominates.
  q4k is no slower than int4 + per-32 despite reading 9% more bytes, so the Q4_K kernel is not the
  1.5B's problem.
- **7B: per-32 costs about 3%, and the rest is bytes**: 79.3 × (4.215 / 4.643) = 72.0, against 72.6
  measured.

**Consequences for a CUDA q4k default:**
- **Lever 2** (native Q6_K, llama.cpp's bytes) addresses the 7B.
- **A new lever 3, cheaper per-32 on CUDA** (fused per-32 RMSNorm+QKV and gate/up, fewer launches),
  is what the 1.5B needs. It would also speed up Phi-3's CUDA default.

Neither is started.

**Instrument note:** zsh does not word-split an unquoted `$q`, so an interactive `set -- $q` loop handed
the benchmark a quant of "int4 0". Several ad-hoc runs printed FAIL for that reason (not the VRAM
release first suspected). The gate script ran under bash and was unaffected.

## Lever 3: fused per-32 kernels on CUDA, PRE-REGISTERED 2026-09-26 before any build

**Attribution** (`measurements/q4k-lever1-2026-09-26/attribution-1.5b.txt`, qwen2.5-coder-1.5b, depth 128):
int4 fused 250.6 / 252.0 tok/s; int4 with fusion OFF (`GOINFER_CUDA_NO_FUSE`) 195.6 / 190.2; int4 + per-32
184.1 / 188.3. **Losing fusion is ~23% on the 1.5B; the per-32 kernels themselves are ~3%.** Per-32
turns fusion off because `fused_rms_qkv` and `fused_rms_gu` quantize per vector and read int4 only.

**Change:**
- `fused_rms_qkv_g32` and `fused_rms_gu_g32`: each block recomputes RMSNorm and the per-32 quantize
  (codes, scales, group sums: `quantG32` itself) into shared memory, then runs its rows.
- Each projection carries its weight kind (int8, int4 or Q4_K), and its row code is the matching
  per-32 matvec's body, verbatim.
- They replace "rmsnorm_quant_g32 + 3 (or 2) matvecs" whenever per-32 is on and every Q/K/V and
  gate/up weight has one of those kinds. `GOINFER_CUDA_NO_FUSE` turns them off like the per-row ones.

**Gates:**
- **Correctness:** the fused kernels are **bit-identical** to the unfused per-32 chain: a unit test over
  mixed kinds, and a resident test comparing q4k logits fused against `GOINFER_CUDA_NO_FUSE`.
- **Speed:** `BenchmarkResidentDecode`, depth 128, 6 paired rounds, the same protocol as lever 1:
  - **qwen2.5-coder-1.5b, q4k effective bandwidth ÷ int4's: ≥ 0.90 PASS; [0.85, 0.90) AMBIGUOUS;
    < 0.85 FAIL** (now ~0.81);
  - guard: qwen2.5-7b q4k fused ≥ 0.98× unfused.
- phi3-mini fused ÷ unfused is reported for information.
