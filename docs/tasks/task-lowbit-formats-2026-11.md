# Task: low-bit read formats, NVFP4 first, and a fit planner that prices them (2026-11)

**Status 2026-11-06: planned, nothing built.** The owner decided on 2026-11-06 to read low-bit formats, NVFP4 first, and
to plan and estimate the rest here. It is v0.26.0 work (`docs/releases/v0.25.0-plan.md` §7, §8). This doc replaces item C
(NVFP4 read support) of `docs/tasks/task-fp4-formats.md`, which links here. Items A (a native MXFP4 CPU kernel), B (MXFP4
beyond gpt-oss) and D (Blackwell FP4 tensor cores) stay there and are not this doc's subject.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| LB0 | open | the Bonsai guard (a named decline for a Hadamard-rotated GGUF, keyed on its metadata, with a test); count the real files per format (NVFP4 safetensors and GGUF, the IQ types in Unsloth's dynamic quants); read the NVFP4 GGUF type id and block layout from `ggml-common.h` at a pinned llama.cpp build, and ModelOpt's nibble order from its exporter | §1, §2 |
| LB1 | open | NVFP4 dequant in aikit (`embed`), for safetensors and GGUF; tier 0 against the reference | §1 |
| LB2 | open | NVFP4 loader admission in goinfer (safetensors: the packed `[N, K/2]` weight and its scale tensors; GGUF type), and the fit's pricing of the packed shape; tier 0 for the pricing, tier 1 on one small file | §1 |
| LB3 | moved | into LB0: the Bonsai guard ships first, since a rotated file can load today | §2 |
| LB4 | open | IQ2_XXS, IQ2_XS, IQ3_XXS dequant (aikit), tier 0 each; Bolt's experts | §2 |
| LB5 | open | IQ1_S, IQ1_M, TQ1_0, TQ2_0, and the legacy Q4_1/Q5_1 | §2 |
| LB6 | open | fit F1: price the source bytes per stored type; F2: say what the stored format does and does not change | §3 |
| LB7 | decision | whether to build kept lanes (compute in a low-bit format) at all; until then a low-bit file fits no better resident than a Q4_K one at the same `--quant` | §3 |

---

## 0. What the tree does today

- **goinfer reads, then requantizes.** aikit's `embed` package dequantizes a GGUF tensor to f32 (`dequantRange` in
  `embed/gguf_dequant.go`, one block function per type), and the loader requantizes it to the load's resident precision
  (`docs/quantization.md`, "Reading a format is not standing behind it"). It reads `IQ2_S`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` and
  `MXFP4` already; any other type is a load error naming the type id (aikit's `embed.GGUFFile.RowDequantizer`), not garbage.
- **So the resident backends need nothing for a read format.** CUDA, Metal and WebGPU receive goinfer's resident
  precisions (int8, int4, f32), never the file's blocks. The one exception is the q4k lane, which uploads a GGUF's own
  `Q4_K`/`Q5_K`/`Q6_K` blocks (`decoder.Model.KQuantNative`, `backend/cuda/pack_stream.go:planWeight`). That lane is the
  pattern for a kept low-bit format, and the only place a backend dequantizes in a kernel.
- **Safetensors fp8 is already read:** `decoder/fp8.go` decodes e4m3fn (all 256 patterns gated against PyTorch) and
  composes it with its block scales (`fp8DequantBlocked`). NVFP4's scale is the same e4m3 byte.

## 1. NVFP4 first

**Layout (to verify, LB0).** Blocks of 16 elements: 16 e2m1 nibbles (8 bytes), one e4m3 block scale (1 byte), so
4.5 bits per weight, plus one fp32 scale per tensor: `w = e2m1 · e4m3(scale) · global`. In NVIDIA ModelOpt safetensors:
`weight` (uint8, `[N, K/2]`), `weight_scale` (`F8_E4M3`, `[N, K/16]`), `weight_scale_2` (fp32 scalar), and
`input_scale` (the W4A4 activation scale, unused by a weight-only read). The GGUF form (the Mamba layers of Nemotron 3.5
Bolt's 24 GB build) is llama.cpp's own type. **Read its id, block size and nibble order from `ggml-common.h`, never by
analogy:** MXFP4's nibbles are sequential in safetensors and split in GGML (`decoder.buildGptOssWeights`), and a wrong
order loads finite, plausible, wrong weights.

**How the loader reads the safetensors shape (LB2).** The `weight` tensor's stored shape is not the matrix's: a uint8
`[N, K/2]` holds two e2m1 values per byte, so the loader takes the logical shape `[N, K]` and admits the tensor only when
`weight_scale` is `F8_E4M3` `[N, K/16]` and `weight_scale_2` is a one-element fp32, both naming the same N and K, and when
`[N, K]` matches the shape the architecture expects for that weight. Any other combination is a load error that names
the tensor, never a guess. The nibble order within a byte (which element sits in the low nibble) is ModelOpt's and is
read in LB0. The loader dequantizes each block to f32 through the existing e2m1 table and `fp8E4M3ToF32`, multiplies by
`weight_scale_2`, and requantizes to the load's `--quant` as every read format does. `input_scale` is read and ignored.

**How fit and admission price it (LB2, LB6).** The resident bytes are set by `--quant`, as for every read-only format
(§3). The shape-only estimator (`fit.SafetensorsWeightBytes`, through `decoder.estimateSafetensorsWeightBytes`) counts
elements from the stored shape, so today it would count a packed weight as `N·K/2` elements, half the matrix, and
underprice the load in the accepting direction, and it would price each 2-D `weight_scale` as a matrix at the load's
quant. LB2 changes it as GPTQ's `.qweight` already is: a `.weight` whose sibling `.weight_scale` is present counts
`N·K` elements (pack factor 2), and `.weight_scale`, `.weight_scale_2` and `.input_scale` are skipped as consumed during
the dequant, as `gptqAWQAuxSuffix` skips GPTQ's auxiliaries. A tier 0 test prices a synthetic NVFP4 index against the
same model in bf16 and requires equal resident bytes. The source term (F1, the mapped file and the `.giw` transcode) is
priced at the stored bytes: 8 bytes of nibbles plus one e4m3 scale byte per 16 weights, 0.5625 bytes (4.5 bits) per
weight, plus 4 bytes per tensor for `weight_scale_2` and 4 for `input_scale`, which are negligible.

**What reading it needs, per backend.**

| backend | work |
|---|---|
| CPU | the dequant (aikit) and loader admission; the e2m1 table is MXFP4's, the scale decode is `fp8E4M3ToF32` |
| CUDA (Turing, nobara: no FP4 hardware) | none: it receives int8/int4. A kept NVFP4 lane would dequantize in the kernel, as q4k does (LB7) |
| Metal, WebGPU | none, for the same reason |

**Files that use it.** Nemotron 3.5 Bolt 74B-A9B's 24 GB GGUF (Mamba layers; its experts are IQ2_XXS, LB4), NVIDIA's
`*-NVFP4` safetensors repos (Nemotron 3.5 Super, DeepSeek V4.1 Flash: both out by `docs/next-models.md` rule 2 on size),
and NVIDIA's ModelOpt exports of models goinfer already runs. LB0 counts the last group, which is where the demand is.

**Gates.** Tier 0: bit-exact dequant of every e2m1 × e4m3 pair and of a real tensor's first rows against the reference
(gguf-py's `gguf.quants` for the GGUF type if it has one; the ModelOpt formula in PyTorch for safetensors), seconds, as
`decoder/fp8_test.go` and aikit's MXFP4 tests do. Tier 1: one small real NVFP4 file (under 4 GB) loads and its first
greedy tokens and per-layer hidden states match HF or llama.cpp on the same file; wall time stated first.

**Size.** LB0 half a day; LB1 1 day (table reuse, two containers); LB2 1 to 2 days (scale-tensor plumbing beside
`weight_scale_inv`, the GGUF type, the tier 1 asset).

## 2. The other low-bit formats

All are read-only (dequant in aikit, no backend work) unless LB7 says otherwise. Each needs its codebook transcribed
(`embed/iq_grids.go` already holds IQ2_S's and IQ3_S's) and a tier 0 test against gguf-py's `gguf.quants`.

| format | bpw | files | estimate | order |
|---|---|---|---|---|
| IQ2_XXS | 2.06 | Bolt 24 GB's experts; Unsloth dynamic quants | 1 day (256-entry grid, sign table) | LB4, first |
| IQ2_XS | 2.31 | Unsloth dynamic quants | ½ day after IQ2_XXS (512-entry grid) | LB4 |
| IQ3_XXS | 3.06 | Unsloth dynamic quants | ½ day | LB4 |
| IQ1_S, IQ1_M | 1.56, 1.75 | Unsloth's smallest builds of large MoEs | 1 day both (2048-entry grid, the delta term) | LB5 |
| TQ1_0, TQ2_0 | 1.69, 2.06 | ternary models (BitNet-style) | ½ day both | LB5 (the guard is in LB0) |
| Q4_1, Q5_1 | 5.0, 6.0 | older GGUFs | 2 hours both | LB5, whenever |

**The Bonsai guard comes first, in LB0.** PrismML's Ternary Bonsai 2 folds a blockwise Hadamard rotation into its
weights, so a correct dequant of its tensors still produces garbage without the runtime activation transform
(`docs/next-models.md`, "Next up" item 2, and item 2 of "Changes to items already in Next up and Watching"). Its
architecture is `qwen35`, which goinfer runs, so the tensor type is the only thing standing between a rotated file and a
silently wrong load: a rotated tensor stored in a type goinfer already reads, or a third-party build requantized from the
rotated weights to one (a `Q4_K`, say), loads today and generates garbage. The guard therefore does not wait for a
`Q2_0` or `TQ*` reader. It is a named decline keyed on the GGUF metadata that marks the rotation (the key is read from
PrismML's files and llama.cpp's `b11556` loader in LB0), checked before any tensor is read, with a test on a synthetic
GGUF carrying that key. Supporting the rotation is out of scope until it is upstream.

## 3. Fit

**The planner prices the resident bytes, and those are set by `--quant`, not by the file.**
`decoder.estimateGGUFWeightBreakdown` prices each matrix at `quantBytesPerElem` of the load's quant (or, for q4k, at the
file's Q4_K bytes, `q4kKeptBytes` through `fit.Rates.Kept`), and `fit.GGUFWeightBytes` sums them. A read-only IQ2_XXS
build of a model therefore costs the same resident memory as its Q4_K build at the same `--quant`, so "the IQ3 build of
this model fits" is false for a read-only format. Saying it would send a user to download a file that is declined the
same way. What the format does change:

- **F1, the source term.** The mapped source and the `.giw` transcode are priced from the file
  (`mem.HostCheck.SourceBytes`); add a per-type bytes-per-weight table (the block bytes over the block elements aikit
  already holds per type) so the fit answer can price another build's source before it is downloaded.
- **F2, the advice.** `mem.HostCheck.Decline`'s smaller-load line (`smallerLoad`) names a smaller `--quant`; it must not
  name a smaller file for a read-only format. It may name one once a kept lane exists for that format (LB7), priced through
  `fit.Rates.Kept` as q4k is: `FitAnswer` gains one row per format that has a lane, and the decline names the largest that fits.

**LB7 is the owner's decision**, because it is the only way Bolt's 24 GB build fits a 24 GB machine. Read-only, its experts
requantize to int4: about 74B × 0.56 ≈ 42 GB resident. A kept lane is CPU first (an aikit `WeightMat` kind, as int4 is),
then CUDA (dequant in the kernel on Turing), each with its own parity bar; roughly 1 to 2 weeks per format per backend.

## 4. Risks

- **Fidelity.** A read format is gated for bit-exact dequant, not for quality; `docs/quantization.md` reads Q2_K and declines
  only to recommend it. The new types join "Read but not endorsed", and the page's count of read types changes with them.
  A kept lane needs a registered fidelity bar (top-2 agreement and KL against f32, as int4's).
- **Parity hashes and censuses.** LB1, LB4 and LB5 are aikit changes and an aikit bump; no decoder forward file changes,
  so no parity hash moves. LB2's loader change touches `decoder/` and needs the parity refresh before its push.
  LB7 changes the admission and kernel censuses on every backend that takes a lane.
- **CUDA without FP4 hardware.** Turing and Ampere have no FP4 tensor path. A kept NVFP4 lane there is a dequant-in-kernel
  lane, as q4k is; its speed is unmeasured, and memory-bound decode may not repay it. Native FP4 compute stays item D of
  `docs/tasks/task-fp4-formats.md` (Blackwell, not owned).

## 5. Order and size

LB0 (1 day, the Bonsai guard included) → LB1 + LB2 NVFP4 (2 to 3 days) → LB4 IQ2_XXS/IQ2_XS/IQ3_XXS (2 days; with NVFP4 this
reads Bolt's 24 GB build) → LB6 fit F1 and F2 (1 day) → LB5 (2 days). About two weeks of read work, all at tier 0 and
tier 1. LB7, if chosen, is per format per backend and gets its own registration.
