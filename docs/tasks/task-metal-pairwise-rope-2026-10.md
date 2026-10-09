# Task — pairwise RoPE on Metal: GLM-OCR, Command-R7B and Aya resident on the Mac (2026-10)

**Status:** SHIPPED 2026-10-09 (owner).
- G-PR1 to G-PR4 and G-PR6 PASS.
- G-PR5 on Command-R7B missed the registered mean bar by 0.0008 (0.989242 against 0.99), and the owner accepted it
  ("close enough, so it's a go").
- G-PR5 on Aya was not run (disk).
- G-PR7 (speed) is not queued; the exploratory decode timing is below.
Gates were registered before any code (owner: "yes do that next").

## Why

- GLM-OCR, Command-R, Command-R7B and Aya rotate GPT-J **pairwise** (dims 2d, 2d+1). Metal's rope kernels are NeoX
  half-split (d, d+half). Running those is exact at position 0 and wrong after it, so Metal declines these families
  to the CPU (`metal does not implement [pairwise-rope]`, and `[pairwise-mrope pairwise-rope]` for GLM-OCR).
- CUDA has had the port since 2026-10-01 (`docs/measurements/cuda-pairwise-rope-2026-10-01.md`, which also lists the
  Metal steps).
- **What it buys on the Mac.** The 1,656-token test invoice in `goinfer-chat` on the M1 Pro takes 23 s since its tower
  moved to Metal (8f61f000; exploratory, one run). Of that, the CPU decoder is about 15.5 s: prefill 5.7 s, plus 416
  tokens of decode at 42 tok/s. Command-R7B and Aya run on the CPU on Metal today.

## What changes

Pairwise twins of every live Metal rope kernel, with the same arguments and grid, are bound into the **same** pipeline
fields when `Model.PairwiseRoPEResident()`. Every other family binds exactly the kernel it binds today, and no dispatch
site changes.

| kernel (NeoX, today) | twin | where it runs |
|---|---|---|
| `rope2` | `rope2_pw` | decode, Q and K in one dispatch; `ForwardMRoPE` (rotation position != cache position) |
| `rope` | `rope_pw` | decode's split Q/K site |
| `rope_f16` | `rope_f16_pw` | the batched prefill (and chunked prefill, which runs it) |
| `rope_mrope_f16` | `rope_mrope_f16_pw` | S16's m-RoPE image prefill (on since 2026-10-09) |

- Pair d is dims (2d, 2d+1) at `invFreq[d]`; partial rotary rotates the first 2·rhalf dims and passes the tail through.
  YaRN `mscale` and the Ministral Q-temperature scale are kept.
- Pairwise m-RoPE takes `axis[d]` from the same host table (`MRopeAxisResident`, contiguous sections for GLM), as in
  `decoder/rope.go`'s `applyMRoPEPairwise`.
- **Not ported, excluded instead:**
  - MC3's batched decode step (`mc3_rope2_rows`, generated from `rope2`) gets an explicit pairwise exclusion with a
    named reason. Cohere is already excluded there (parallel block), so nothing pairwise reaches it today.
  - The f16 decode lane is off by default. If it rotates through `rope2` it inherits the twin; this is checked, not
    assumed.
- **Admission:** `FeatPairwiseRoPE` and `FeatPairwiseMRoPE` join the `"metal"` set in `decoder/features.go`.
  - Update `admissionGolden`, `TestResidentBackendFeatures_noOverclaim`'s metal row, the capability and hardware
    matrices, and the matrix footnote.
  - `metal/cohere_resident_smoke_test.go` falls through to its original smoke body, as the CUDA session wrote it to.

## Gates, registered before any code

- **G-PR1, kernel level** (hermetic, `metal/rope_pairwise_test.go`).
  - **Reference:** each twin against a float64 reference written from `decoder/rope.go`'s semantics.
  - **Cases:**
    - full and partial rotary;
    - mscale 0.85, a Q-temperature scale of 1.3 on `rope2_pw`, and a rotation position different from the cache
      position;
    - multi-row f16 batches with distinct per-row positions;
    - m-RoPE with distinct (t,h,w) triples over contiguous sections;
    - poisoned tails (unrotated dims must come back untouched).
  - **Tolerances:**
    - f32 kernels: max |diff| <= 2e-5;
    - f16 kernels: max |diff| <= 4e-3, with inputs in [-2, 2].
  - **Degenerate case:** an m-RoPE row with t = h = w must be **bit-identical** to `rope_f16_pw` at that position.
  - **Control:** the NeoX kernel on the same input must miss each tolerance by at least 10x, so no case is vacuous.
- **G-PR2, peaked tiny residents vs the CPU** (`metal/pairwise_rope_resident_parity_test.go`).
  - **Fixtures:** cohere-tiny and cohere2-tiny with every 2-D weight x12.5 (CUDA's `peakedCheckpoint`); a 40-token
    prompt.
  - **Quant:** int4. An int8int8 tiny load goes native int8 on Metal and its prefill declines first (memory note).
  - **Paths:**
    - sequential decode at 40 positions;
    - batched prefill, last token;
    - decode at [32, 40) after a 32-row batched prefill.

    A path the family's batched pass declines is named in the log and not graded; the test fails if that silently
    covers every batched path.
  - **References:**
    - CPU int4 per-row for the decode paths (Metal decode's own arithmetic);
    - CPU int4 per-32 for the batched-prefill last token. The batched pass runs f16 activations, which Step 0 of
      `docs/completed/task-metal-prefill-precision-2026-10.md` measured as closer to that arm than to per-row.
  - **Bars:** cosine >= 0.995 and relL2 <= 0.15 on every graded path (CUDA's final tier).
  - **Control:** the NeoX kernels rebound into the same resident must read below the bar on every graded path, every run.
- **G-PR3, glm-ocr-tiny** (`metal/glm_ocr_resident_test.go`). The same references and bars as G-PR2, and the same
  NeoX control. Paths:
  - text, sequential decode at 48 positions;
  - batched prefill, last token;
  - the 12-token image block of `glm_ocr_tiny_mrope_golden.json` through `PrefillMRoPELast`;
  - decode past the image through `ForwardMRoPE`, teacher-forced on the golden continuation, with `mropeDelta != 0`
    asserted.
- **G-PR3 amendment, 2026-10-09, after its first run (mechanism measured, bar unchanged):** G-PR3 runs glm-ocr-tiny
  **unpeaked**, as CUDA's glm_ocr gate does.
  - **Measured, peaked x12.5:** the resident read 0.9999-1.0000 at almost every position, with isolated dips (position
    16 at 0.989). At those positions the CPU's two quantizations disagree with each other by more: the resident against
    CPU weight-only int8 reads 0.958 at position 4 and 0.906 at position 47.
  - **Why:** at x12.5 the 48-wide model's attention is near-argmax, so an int8 activation rounding difference flips which
    key dominates. A wrong rotation degrades every position from 1 on instead (the NeoX control, 0.59).
  - **Unpeaked:** the pairwise kernels read 0.99776 (text decode) and 0.99920 (decode past the image). The NeoX control
    reads -0.328 and -0.082, already red flat (CUDA measured -0.337).
  - The Cohere fixtures stay peaked: they pass there (0.99872, 0.99932) and are flat-blind.
  - Log: `docs/measurements/metal-pairwise-rope-2026-10/gpr1-3.log`.
- **Found on the first run, recorded rather than graded:** on Metal neither family has a batched prefill. The batched
  pass declines both FFN shapes ("prefill not implemented for this arch's FFN shape"), and GLM-OCR's resident has no
  m-RoPE axis table.
  - So production prefills Cohere sequentially, and a GLM-OCR image turn takes the CPU prefill and the upload, then
    Metal decode.
  - The batched twins (`rope_f16_pw`, `rope_mrope_f16_pw`) are bound but unreached by today's pairwise families; G-PR1
    covers them at kernel level for the day a family reaches them.
  - G-PR2 and G-PR3 grade the paths production takes: decode, and decode past an uploaded image prefill through
    `ForwardMRoPE`.
- **G-PR4, real GLM-OCR on the Mac** (by day, a few minutes).
  - **Invoice:** the 1,656-token test invoice through `metal/cmd/chat`, the Metal resident against `--backend cpu`.
    Both arms run the decoder on their own backend; the tower is not under test.
    - Bar: the JSON replies are byte-identical, or every difference sits behind a teacher-forced near-tie (3% rule).
  - **Per position:** teacher-forced per-position logits over the reply, Metal resident against the CPU int4.
    - Bars (CUDA's real-checkpoint tier): mean cosine >= 0.99, worst >= 0.90.
    - Control: NeoX rebound must read red.
- **G-PR5, real Command-R7B and Aya-expanse-8B.**
  - Bars and procedure: CUDA's noise-referenced tier, with the NeoX control.
  - **Blocked:** both are 15 GB in the archive, and this Mac has 2.3 GB free (2026-10-09).
  - Admission covers Cohere and GLM together (one feature), so the commit that declares `FeatPairwiseRoPE` on Metal
    waits for G-PR5, unless the owner accepts the tiny gates (G-PR2) as Cohere's evidence. CUDA's tiny gates read the
    defect as plainly as its real ones did.
- **G-PR6, no regression.**
  - A structural test: a NeoX model binds exactly today's pipeline names, and a pairwise model binds the twins.
  - `gate quick` green; parity hashes refreshed if a decoder file changes.
- **G-PR7, speed** (night, the served gate). GLM-OCR invoice decode tok/s and TTFT, Metal resident against the CPU
  decoder, same session, interleaved.
  - Ship >= 1.02x; 1.00-1.02 parked; < 1.00 off (a resident slower than the CPU would not be admitted).
  - A by-day exploratory timing comes first and is not quoted.

## Results, 2026-10-09 (by day, the M1 Pro; logs in `docs/measurements/metal-pairwise-rope-2026-10/`)

- **G-PR1 PASS** (`gpr1-3.log`). Each twin against the float64 reference; the NeoX kernel on the same input missed every
  case by 3-5:

  | kernel | max abs diff | tolerance |
  |---|---|---|
  | f32 twins | 5.9e-7 to 8.2e-6 | 2e-5 |
  | f16 twins | about 9.5e-4 | 4e-3 |

  Poisoned V came back untouched, and the t = h = w m-RoPE rows were bit-identical to `rope_f16_pw`.
- **G-PR2 PASS** (decode, the one path production takes for Cohere on Metal):

  | fixture | pairwise worst cos | relL2 | NeoX control |
  |---|---|---|---|
  | cohere-tiny | 0.998722 | 0.0522 | 0.431459 |
  | cohere2-tiny | 0.999315 | 0.0370 | 0.271841 |

  The batched prefill declines both (parallel-block FFN), so those paths were logged and not graded.
- **G-PR3 PASS** (unpeaked, per the amendment above):

  | path | pairwise worst cos | NeoX control |
  |---|---|---|
  | text decode | 0.997760 | -0.328297 |
  | decode past an uploaded image prefill | 0.999195 | -0.081681 |

  The batched and m-RoPE prefills decline (FFN shape; no axis table) and were logged, not graded.
- **G-PR4 PASS:**
  - **Invoice:** the 1,656-token test invoice through `metal/cmd/chat` with the Metal-resident decoder gives a reply
    byte-identical to `--backend cpu`'s.
  - **Per position** (`TestPairwiseRoPERealMetal`, `gpr4-5-real.log`, 64 tokens of README plus a 16-token CPU
    continuation): mean cos 0.996190, worst 0.988864, 0 continuation flips. The NeoX control read mean 0.772141 and
    worst 0.252920.
  - **Exploratory timing, not a G-PR7 result:** decode went from about 42 to 92-95 tok/s (4.4-4.8 s against
    8.8-10.6 s for 416 tokens); the whole invoice took about 20 s, against 23 s with the CPU decoder.
- **G-PR5, Command-R7B** (`gpr5-real-r7b.log`). Run by day on the owner's word ("run g-pr5 now"), with
  `GOINFER_NO_FIT_GUARD=1` for this test and for the two sidecar builds (owner-approved; the CPU int4 reference is
  about 8.3 GB).
  - The Metal arm loads through `command-r7b.int4.metal.giw`. A directory load needs 9.33 GB against Metal's own
    guard; the sidecar aliases the weights, so it needs 4.70 GB.
  - The test returns the CPU model's heap to the OS before the Metal load (`debug.FreeOSMemory`); otherwise Metal's
    live-available budget still counts it.

  | | mean cos | worst | continuation flips |
  |---|---|---|---|
  | pairwise | 0.989242 (bar >= 0.99: **missed by 0.0008**) | 0.945679 at position 13 (>= 0.90) | 0 |
  | NeoX control | 0.543841 | -0.694615 | |

  - **Owner decision, 2026-10-09:** "close enough, so it's a go." The bar was not amended; the result ships as a
    recorded miss.
  - **Not measured:** why Metal sits below CUDA here (CUDA's resident read mean 0.9967 on a 48-token prompt). Metal
    stores its KV cache in f16 where CUDA defaults to f32; that is a hypothesis, not a finding.
- **G-PR5, Aya-expanse-8B: NOT RUN.** Its CPU reference ran (`gpr5-real-aya.log`). Its Metal sidecar build ran out of
  disk, and the directory load is refused by Metal's guard (9.33 GB). Aya rides on Command-R7B's result and the tiny
  Cohere gates.
- The first attempt's log (`gpr5-real.log`, both models refused by Metal's guard on a directory load) is kept.
- **G-PR6 PASS:**
  - `TestPairwiseRoPE_neoxFamiliesUnchanged`: llama-tiny is not pairwise; the prefill kernel names map one to one.
  - The capability and hardware matrices were regenerated: Command-R, Command-R7B and GLM-OCR read Metal resident, and
    WebGPU still declines.

## Cost

- Kernels and binding: an hour or two.
- G-PR1 to G-PR3: about an hour with their tests, seconds to run.
- G-PR4: minutes.
- G-PR5: two pulls (30 GB, needs disk) and a few minutes each.
- G-PR7: one night job of about 20 minutes.
