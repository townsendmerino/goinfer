# Task: low-bit read formats, NVFP4 first, and a fit planner that prices them (2026-11)

**Status 2026-11-08: planned, nothing built.** The owner decided on 2026-11-06 to read low-bit formats, NVFP4 first, and
to plan and estimate the rest here; #731 (2026-11-07) merged this plan and nothing else (no NVFP4, IQ2_XXS or rotation-guard
code is on `main` in goinfer or aikit). It is v0.26.0 work (`docs/releases/v0.25.0-plan.md` §7, §8), except that the plan
pinned for v0.25.0 lists the Bonsai guard (LB0a) as cheap v0.25.0 work: which release takes LB0a is open. This doc replaces item C
(NVFP4 read support) of `docs/tasks/task-fp4-formats.md`, which links here. Items A (a native MXFP4 CPU kernel), B (MXFP4
beyond gpt-oss) and D (Blackwell FP4 tensor cores) stay there; LB7's MXFP4 half (rows MX0 to MX2) is decided here. A likely first NVFP4 asset is
the Bolt E4B DFlash drafter (`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4`, named blocked on NVFP4 dequant in
`docs/measurements/bolt-e4b-g0-2026-10.md`).

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| LB0a | open | the Bonsai guard: a named decline keyed on the GGUF metadata marking the Hadamard rotation, checked before any tensor read, tier 0 on a synthetic GGUF; settle whether Ollama's `Q2_0` Bonsai files are upstream Q2_0 or relabeled PQ2_0; release slot (v0.25.0 or v0.26.0) open | §2; `docs/next-models.md` "Bonsai-2: the format question changed" |
| LB0b | open | count real files per format (NVFP4 safetensors and GGUF, IQ types in Unsloth dynamic quants, ModelOpt exports of models goinfer runs) | §1 |
| LB0c | done | NVFP4 GGUF type id (`GGML_TYPE_NVFP4 = 40`), 64-element super-block (4 × 16-element sub-blocks, `ue4m3` scales + 32 packed nibble bytes = 36 B/block, split `j`/`j+8` nibble order within each 16-element sub-block), plus companion `<name>.scale` `F32 [1]` tensor in `llama.cpp` PR `#21445` | `docs/measurements/flint-f0-2026-10.md` §4.1 |
| LB0d | done | ModelOpt's NVFP4 nibble order from its exporter: sequential (`b & 0x0F` = even index `2j`, `b >> 4` = odd index `2j+1`), 16-element micro-blocks with `F8_E4M3` `weight_scale` `[N, K/16]` and scalar `F32` `weight_scale_2` | `docs/measurements/flint-f0-2026-10.md` §4.1; `decoder/nvfp4.go:nvfp4DequantRow` |
| LB1a | done | NVFP4 safetensors dequant (`decoder/nvfp4.go:nvfp4DequantRow`, `decoder/mlx_safetensors.go:mlxDequantNVFP4Row`); tier 0 bit-exact against all 4,096 `e2m1 × e4m3` pairs and the ModelOpt formula | `decoder/nvfp4_mlx_test.go:TestNVFP4_e2m1TableAndExhaustiveE4M3` |
| LB1b | open | aikit NVFP4 dequant, GGUF type (`GGML_TYPE_NVFP4 = 40`); tier 0 against gguf-py; companion `<name>.scale` tensor multiplication is already wired in `decoder/gguf.go:buildWeightsFromGGUF` | §1 |
| LB2a | done | loader admission for ModelOpt NVFP4 safetensors (`uint8 [N,K/2]` + `F8_E4M3 [N,K/16]` + optional scalar fp32 `weight_scale_2`) and MLX quantized safetensors (`affine` 4/8-bit, `nvfp4`, `mxfp4`, plus 3-D `switch_mlp` experts) | `decoder/nvfp4.go:nvfp4Reconstruct`, `decoder/mlx_safetensors.go:mlxReconstruct2D`, `decoder/nvfp4_mlx_test.go` |
| LB2b | blocked | loader admission for the NVFP4 GGUF type (`type 40` in `aikit`); companion `<name>.scale` tensor multiplication is done in `decoder/gguf.go:buildWeightsFromGGUF`; needs LB1b | §1 |
| LB2c | done | fit: a `.weight` with a sibling `.weight_scale` counts N·K (pack factor 2) and with `.scales` counts N·K (pack factor 8 at 4-bit, 4 at 8-bit); skip `weight_scale`, `weight_scale_2`, `input_scale`, `.scales`, `.biases`; tier 0 on a synthetic index | `decoder/fit/weights.go:SafetensorsWeightBytes`, `decoder/nvfp4_mlx_test.go:TestFit_nvfp4AndMLXSafetensorsWeightBytes` |
| LB2d | blocked | tier 1: one small real NVFP4 file loads, and its first greedy tokens and per-layer hidden states match the reference; needs LB2a and the asset LB0b finds | §1 |
| LB3 | moved | into LB0a: the Bonsai guard ships first, since a rotated file can load today | §2 |
| LB4a | open | IQ2_XXS dequant (aikit), tier 0 | §2 |
| LB4b | open | IQ2_XS dequant, tier 0 | §2 |
| LB4c | open | IQ3_XXS dequant, tier 0 | §2 |
| LB5a | open | IQ1_S, IQ1_M | §2 |
| LB5b | open | TQ1_0, TQ2_0 | §2 |
| LB5c | open | legacy Q4_1, Q5_1 | §2 |
| LB5d | open | upstream `Q2_0` (Ollama serves Bonsai as `Q2_0`); read it correctly or decline it by name, with LB0a | `docs/next-models.md` |
| LB6a | open | F1: a per-type source bytes-per-weight table, so fit prices another build's source before download | §3 |
| LB6b | open | F2: `mem.HostCheck.smallerLoad` never names a smaller file for a read-only format | §3; `internal/runtime/mem/host.go:HostCheck.smallerLoad` |
| LB6c | open | fit prices `int4mix` (and `q4k` in `measureBytesPerElem`) as all-int4, an underestimate an admission guard must not make: price per tensor through `fit.Rates.Kept`, as `q4k`'s GGUF estimator does; tier 0 | `decoder/fitguard.go:measureBytesPerElem`; found by the 2026-11-08 close-out |
| LB7 | decided (MXFP4 only) | Francis, 2026-11-09: build a kept MXFP4 lane for gpt-oss, `q4k`'s lane as its pattern (MX0 to MX2 below). NVFP4's kept lanes still wait on LB0b's demand and LB1-LB4, as before. Why: gpt-oss's tool-call loss is the int4 requantize (#542, #549, #555: int8 with f32 activations is within 0.99 nats of llama.cpp, int4 is 7.7-9.4 nats under); gpt-oss stays a target. MXFP4 moves alongside or ahead of NVFP4; the order against LB1-LB2 and the release are Francis's to confirm | §3; `docs/tasks/task-fp4-formats.md` items A, B |
| MX0 | done | closed with no run (Francis, 2026-11-10, on the coordinator's reading): P8 is the ceiling arm. int8 weights with f32 activations read a step-22 margin of +4.829 against llama.cpp's +5.822, 0.99 nats under, with llama.cpp's top-1; an f32-dequantized arm (about 84 GB) fits no box and would add nothing on weights. MX1's gate takes P88's lesson | `docs/measurements/gptoss-forced-call-2026-10-30/README.md` "P8, read" |
| MX1 | open | a kept MXFP4 lane on CUDA (dequant in the kernel, as q4k). Its scope names what the lane does about activations: P88 (int8 weights, per-row int8 activations) reads -0.097, so per-row int8 activations carry about 4.9 nats of the gap even at int8 weights, and CUDA has no f32-activation GEMV today. The lane must carry f32 or f16 activations, or grouped (per-32) ones (P4g, int4 weights at per-32 activations, read -1.267: grouping recovers part, not all), and says which. Registered gates: fidelity (the step-22 margin, with the activation format stated, plus KL against llama.cpp) and speed against today's int4 (A B B A, one box, `~/models`); fit priced at MXFP4's own bytes through `fit.Rates.Kept`; registered before any kernel | LB7, LB6c; `docs/measurements/gptoss-forced-call-2026-10-30/README.md` "P8, read" |
| MX2 | blocked | the same lane on Metal, same two gates and the same activation question; needs MX1's lane shape | LB7, MX1 |

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

<!-- doc-reviewed: 2026-11-08 -->
