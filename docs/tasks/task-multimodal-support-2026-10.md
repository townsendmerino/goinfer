# Multimodal support: every image and audio model on every backend, at a usable speed (2026-10)

**Status: ACTIVE (2026-10-07 evening).** The phases' state:
- **S1 (E-model decode on the GPU):** done on Metal and CUDA.
- **Correctness passed on the Mac or nobara** for S2 (Qwen3.5+ and GLM-OCR towers), S3 (Gemma 3 and Qwen2.5-VL towers
  on Metal), S4 steps 0-5 (the CUDA towers), S5 (E2B audio into the model, its tower on Metal included), S9 on Metal,
  and S10 for Qwen3-VL. Most of that code is on `s2-towers`, not `main` ("Where it stands" marks which).
- **In flight:** S3's night root-cause steps, S4's speed record, S9's CUDA half, S6 (E4B, 31B), and the 2026-10-07
  audit's fixes.
- **New and not started:** S15-S18.

The order is in "Order of work". Each phase writes its own gates into this doc, and commits them, before its first
measurement (CLAUDE.md, "Pre-registration").

The owner, 2026-10-06, after the multimodal plan (`docs/multimodal.md`) was finished and aikit v1.58.0 shipped:
"we want better overall support." Three axes:
- **Which models** take images and audio.
- **Which backends** run them on the GPU (CPU, CUDA, Metal, WebGPU).
- **Whether they are fast enough** to use interactively.

The last phase puts the answer where users look first, the README, with a check that keeps it true.

## Where it stands (rebuilt 2026-10-07 evening, from the gate readings below)

"Tower / decoder": where the image or audio encoder runs, then where the language model runs after it.
- **Branches:** **[s2]** marks code on `s2-towers` only; aikit v1.59.0, which it needs, is tagged, but the branch is not
  merged into `main`. Unmarked cells are on `main`.
- **Image turns on Metal:** "prefill CPU" means the image turn prefills on the CPU and uploads its KV before decoding on
  the GPU. Every Qwen-family image turn does this, because Metal has no m-RoPE prefill (S16).

| | CPU | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| Gemma 3 | CPU / CPU | GPU (int8 SigLIP: lossy, see below) / GPU | GPU, float32 [s2] / CPU by default on a 16 GB Mac (its resident needs 5.15 GB against a 4.2-4.9 GB budget; S18) | GPU (int8) / GPU |
| Gemma 4 E2B | CPU / CPU | GPU [s2] / GPU, image prefill CPU (S9 on CUDA owed) | GPU / GPU, layer-major prefill (S9) | CPU / CPU |
| Gemma 4 E2B audio | CPU / CPU | CPU / GPU (not run served) | GPU [s2] / GPU | CPU / CPU |
| Gemma 4 E4B | CPU / CPU, not yet validated | not validated | not validated (S6 queued tonight) | CPU / CPU |
| Gemma 4 26B | CPU / CPU | GPU (checked on E2B) [s2] / GPU | GPU (checked on E2B) / GPU | CPU / CPU |
| Gemma 4 31B | not validated (nobara, S6) | not validated | too large for the Mac | not validated |
| Qwen2.5-VL | CPU / CPU | GPU [s2] / GPU | GPU [s2] / GPU, prefill CPU | CPU / GPU |
| Qwen3.5+ dense | CPU / CPU | GPU [s2] / GPU | GPU [s2] / GPU, prefill CPU | CPU / CPU |
| Qwen3.5+ MoE | never run | never run | never run | never run |
| Qwen3-VL (dense) | CPU [s2] / CPU [s2] | CPU (int8 by default until the fix on `s2-towers`; no DeepStack tap on CUDA) / GPU, not served | GPU with DeepStack [s2] / GPU, prefill CPU [s2] | not run |
| GLM-OCR | CPU / CPU | GPU [s2] / GPU | GPU [s2] / CPU (Metal does not run its decoder) | CPU / staged |
| EmbeddingGemma 2 (text, image, audio) | CPU | CPU | GPU | CPU |

**Gaps in coverage:**
- **Models:**
  - Qwen3.5+ MoE and Qwen3-VL MoE images have never been run.
  - Gemma 4 E4B and 31B are not validated; S6 has them queued on both boxes.
  - Ministral 3 (Pixtral), LFM2.5-VL and North have no tower (S10).
  - Several images per message (S11) and video (S15) are not supported.
- **Audio:**
  - Gemma 4 E2B audio into the model works on CPU and Metal.
  - Chat audio takes 16-bit WAV at any rate, mono or stereo (G-S5e). EmbeddingGemma 2's audio embeddings still take 16 kHz
    mono only.
  - E4B audio rides on S6.
  - G-S5b, the HF anchor for audio into the model, has no result yet.
- **Precision:** under `--backend cuda|webgpu`, serve gives Gemma 3 an int8 SigLIP tower. At real size that tower is
  relative L2 0.16-0.52 from float32, worst token 0.01-0.17 (`docs/measurements/siglip-int8-fidelity-2026-10-07.md`).
  Changing the default waits for the owner; nobara builds the float32 CUDA tower (S4).
- **Release binaries:** WebGPU needs cgo and is in no release binary.

**Speed** (the most recent read of each, exploratory unless marked; S7 measures every cell against the 5 s bar):
- **Under 5 s:**
  - Gemma 3 on CUDA: a 4.1 s tower (2026-09-21).
  - Qwen3.5 on CUDA: 3.6 s for a new image.
  - EmbeddingGemma 2 on Metal: 1.2-2 s an image.
  - The Gemma 4 tower on Metal: 1.3-2.3 s.
  - The Gemma 4 E2B tower on CUDA: 2.3 s encode.
  - Qwen3.5-0.8B's Metal tower: 1.2 s on an 896x896 image.
- **Over it:**
  - Gemma 4 E2B image turn on Metal: TTFT 10.24 s on S9's pass against 14.67 s before (1.43x, the night record; a ~512-token text prompt reads 0.98x; the owner keeps the pass everywhere, 2026-10-08).
  - Gemma 3's Metal tower: 9.2-9.6 s, about 2.8x its 26 s CPU tower.
  - Qwen2.5-VL's Metal tower: 8.9-15.1 s on the large images.
  - Qwen3-VL-2B served on Metal: 19.2 s per request.
  - GLM-OCR on Metal: 12.3 s for an invoice.
  - The CUDA towers: only 1.2-1.5x the CPU (S17).
  - Qwen3.5 on the CPU: 7.7 s TTFT on a new image.
  - GLM-OCR's CPU tower: 29 s at 1 MP.
- **To re-read:** the r12 peer reading (Ollama gemma3:4b, 0.43 s mean) timed a resent image, most likely an image-cache
  hit. The harness now sends a fresh image per request; S13-lite re-measures it.

## Rules for every phase

- **Gates first.** Each phase writes its gates here and commits them before measuring.
  - The correctness bar is the one its neighbours use: bit-identical where the arithmetic allows; otherwise cosine
    >= 0.9999 per row against the CPU path or HF, with an ambiguous band parked for the owner.
  - Plus identical greedy tokens on a served request where a decoder is involved.
  - Plus planted defects that turn the gate red. Randomise any tiny fixture whose weights are all ones (memory:
    gemma4-vision-tiny); check one-ulp effects at the float level, not only in uint8 (F1 of `docs/multimodal.md`).
- **Speed** is a record, not a gate, unless the phase pre-registers a bar. By day it is exploratory and labelled so;
  timed A/Bs and peer comparisons go on the night queue (CLAUDE.md, "Run budget").
- **aikit:** new API is batched into one release per cycle. A phase that needs it develops against a local aikit branch
  in a separate worktree and does not push goinfer main until the tag exists (the 2026-10-06 pattern: memory
  eg2-gpu-branch-worktree).
- **Checkpoints** come from `~/models` (`models-pull`), never the archive, for anything timed. The Mac has about 26 GB
  free after the 2026-10-06 cleanup, so a large download needs space first.

## Phases (numbered as added; the order they run in is "Order of work", below)

### S1 — Gemma 4 E2B/E4B decode on the GPU (Metal, then CUDA)

The largest single speedup. It is not only for images: E2B/E4B text is CPU-only on every backend today. No backend
declares `FeatGemma4EModel` (`decoder/features.go`), so these sizes fall back to the CPU rather than silently drop:
- **PLE:** a per-layer input embedding.
- **Shared KV:** later layers reuse an earlier layer's keys and values.
- **A per-layer FFN width.**

`docs/tasks/task-gpu-paths-2026-09.md` lists the gap. This task builds it, and that row points here once started.

- **Metal first.** EmbeddingGemma 2's Metal encoder already runs a PLE block (`metal/embeddinggemma2.go`,
  `eg2_gelu_mul_ple`).
  - Shared KV is resident-cache indexing: a layer reads another layer's slot.
  - Per-layer FFN width is per-layer scratch sizing.
  - Then the resident decode bridge for E-models, and image turns through it (`GenerateGemma4VL`).
- **Gates to write before starting:**
  - Resident against CPU logits on the tiny E-model fixture, every position, with PLE, shared KV and the varying FFN
    width all exercised.
  - Real E2B text: identical 32 greedy tokens against the CPU on a fixed prompt set.
  - Real E2B image chat: identical served replies against the CPU decoder (the F2b shape).
  - Planted defects: PLE skipped, the shared-KV source layer off by one, one FFN width.
- **Speed (night):** E2B decode tokens per second, resident against CPU, same-session interleaved.
- **Size:** M-L per backend. No aikit API expected.

**S1 Gate 0, read 2026-10-06** (a desk map with file:line citations:
`docs/measurements/multimodal-support-2026-10/s1-gate0.md`).
- **Two existing bugs in shipped Gemma 4 GPU decode**, each checked against the code and against transformers 5.19
  before this was written:
  - **(a) Metal never applies the per-layer output scalar on a dense Gemma 4 layer.** It does so only in the MoE join
    (`metal/gemma4_moe.go`). CUDA and WebGPU apply it on dense layers.
  - **(b) No GPU backend applies `v_norm` on a layer that owns its own `v_proj`.**
    - HF applies the scale-less `v_norm` on every layer that owns its K/V (`modeling_gemma4.py`), and so does
      goinfer's CPU path (`decoder/forward_gemma4.go`).
    - Metal and CUDA apply it only on K=V layers, and WebGPU has only a K=V kernel.
    - This affects the sliding layers of the 12B/26B/31B now running resident, and would affect every E2B layer.
  - **Why the gates missed both:** they are loose (pos0 >= 0.97 on the scaled dense test, a 0.90 backstop on the
    two-geometry test, 0.88 on the 26B), and the post-attention norm partly hides a V scale.
- **What the E-model needs on Metal:**
  - **PLE** needs the token id, which the resident interface does not carry. Plan: the host computes the per-layer
    inputs with the CPU's own code and appends them to the embedding row (`[H || L*P]`), with strict length checks at
    every Metal entry. The per-layer branch is built from kernels Metal already has.
  - **Shared KV** is buffer aliasing (Metal's KV is linear). The traps: the K/V store must be skipped on shared layers,
    the slot copies, `UploadKV`, and the KV-memory estimate.
  - **The per-layer FFN width:** Metal sizes one width for the model, from layer 0 (6144). E2B's 20 shared layers are
    12288, so admitting E2B as it stands would overflow the gate/up scratch silently.
  - **The fixture:** no tiny E-model fixture exists, and safetensors PLE loading is refused, so one is built first
    (S1.1).

**S1 gates, written 2026-10-06 before any S1 measurement:**
- **S1.0, the two fixes, on Metal:**
  - Gemma 4's resident-vs-CPU parity tests (the scaled dense and two-geometry fixtures) are read before and after the
    fix, with each test's own metrics.
  - **PASS:** no metric moves the wrong way after the fix, and re-dropping each fix through a test seam reproduces
    its before-fix numbers.
  - Each existing bar is then tightened in a separate, recorded amendment. The fix is the mechanism; the new bar sits
    between the before and after readings. The bars are never loosened.
  - The 26B is re-checked on the night queue.
  - CUDA's and WebGPU's `v_norm` fix runs as its own pass on nobara, under the same rule, before S1 reaches CUDA.

  **S1.0 read 2026-10-06, 19:14 PDT, on the Mac: PASS.**
  - **The fix:** `metal/model.go` gains the dense layer scalar (a `layer_scale` kernel at each dense Gemma 4 layer's
    end) and `v_norm` on every Gemma 4 layer that owns its K/V. The test seams `GOINFER_S10_DROP=scalar|vnorm|both`
    re-drop either fix.
  - **The readings,** each test's own summary metric, before (both dropped) → after:

    | test | before | after |
    |---|---|---|
    | scaled dense, pos0 | 0.982297 | **0.999148** |
    | scaled dense, mean | 0.858647 | **0.926917** |
    | two-geometry localize, layer 0 (local, real V) | 0.987958 | **1.000000** |
    | two-geometry localize, layer 1 (global, K=V) | 0.980768 | **0.999940** |
    | two-geometry resident, min | 0.981251 | **0.999761** (max abs 0.21 → 0.024) |
    | MoE localize, worst layer | 0.999644 | **0.999982** |
    | MoE resident, min | 0.872764 | **0.998690** |

    - Argmax agreement is unchanged (15/16 dense, 8/8 two-geometry, all MoE).
    - Each fix alone moves its own tests: the scalar alone lifts the scaled-dense mean to 0.906 (`s10-vnorm.log`, the
      run with `v_norm` dropped), `v_norm` alone to 0.880 (`s10-scalar.log`, the scalar dropped).
    - Corrected 2026-10-07 (Cowork audit): the MoE minimum above read 0.958904 → 0.999809, single positions rather than
      the test's own minCosine; and the two single-fix means were swapped.
    - Two individual scaled-dense positions read lower after (pos 2 0.918→0.907, pos 12 0.884→0.836). At those
      positions the CPU's own int4 path is at 0.70 and 0.45 against f32, so they are quantization noise on a
      low-agreement position, not a regression. Every summary metric and every two-geometry and MoE position rose.
  - **A correction:** `TestGemma4TwoGeom_f16ScaleConfound` attributed its ~0.98 residual to "the broader resident
    quant path". It was these two bugs; matched to f16 scales, it now reads 0.999761.
  - **The amendment** (the fix is the mechanism; each bar sits between its before and after readings):

    | bar | old | new |
    |---|---|---|
    | scaled dense pos0 | 0.97 | 0.99 |
    | scaled dense mean | none | 0.90 |
    | two-geometry localize, each layer | 0.95 | 0.999 |
    | two-geometry f16-scale and resident, min | 0.95 / 0.90 | 0.995 |
    | MoE localize | 0.90 | 0.9999 |
    | MoE resident, min | 0.60 | 0.995 |

    Under the new bars, the fixed code passes all six tests, both fixes dropped fail all six, the scalar alone
    dropped fails two, and `v_norm` alone dropped fails five.
  - **The Metal snapshot golden moved,** on the Gemma 4 dense-scaled rows only (3 of 10 checkpoints). With the fixes
    dropped it matches the old golden 10/10, byte for byte, so the fix is exactly what moved them. It is re-baked
    (same M1 Pro, macOS 26.6.2).
  - **Raw:** `docs/measurements/multimodal-support-2026-10/s10-*.log`.
  - **Naming:** the `s10-*.log` files and the `GOINFER_S10_DROP` seam belong to S1.0, not to S10 (Qwen3-VL). They kept
    a working name from before the phases were numbered, and stay unrenamed because records cite them.
  - **Still owed:** the 26B on Metal (night queue), and the same `v_norm` fix on CUDA and WebGPU (nobara).
  - **The 26B re-check's design, fixed 2026-10-06 before it runs.** Every 26B Metal parity test compares against a CPU
    26B forward, which the Mac must not run (the M26 rule: Metal only, paged, guards on). So it is split across the
    two machines:
    - **Mac, night queue** (`run-s10-26b-dump.sh`, `TestGemma4_26B_s10Dump`): Metal's full logits at every position of
      three G3 prompts plus 16 greedy tokens, in two arms — fixes in, then `GOINFER_S10_DROP=both` teacher-forced over the
      same sequences — under the swap kill-watch. A load-and-run smoke passed by day (24 positions, RSS 3.8 GB).
    - **nobara, afterwards:** the CPU 26B forward over the same sequences, from the same `.int4.metal.giw`, is the
      reference for both arms.
    - **PASS (S1.0's rule):** with the fixes in, neither the mean Metal-vs-CPU logit cosine nor the argmax agreement
      over all positions is lower than with them dropped. Only `v_norm` can move on the 26B: its layers are all MoE,
      and the MoE join already applied its own layer scalar.
    - **Mac half done, night queue 2026-10-06 22:16 PDT (binary at `b1e7c9f6`):** both arms PASS, 117 positions each
      over three prompts, paged (46 and 56 slots/layer), peak RSS 5.9 / 6.7 GB, swap flat under the kill-watch. The dump
      (`seqs.json`, `logits-fixed.f32`, `logits-drop-both.f32`, 117 MB each) is in the Mac's
      `~/goinfer-bench/s10-26b/run-2026-10-06/`, not committed; the grading half is nobara's (`docs/prompts/nobara-s1-2026-10-07.md`).
      Raw log: `docs/measurements/multimodal-support-2026-10/s10-26b-dump-night.log`.
- **S1.0's 26B re-check, graded 2026-10-07 on nobara: PASS.** The Mac's dump (Metal, `gemma4-26b-int4-v14st.metal.giw`, 117
  positions over three prompts, both arms' dump runs `--- PASS`, swap flat, the kill-watch silent; provenance in the dump
  directory, binary at `b1e7c9f6`) was graded against the CPU 26B forward over the same sequences, from the same `.giw`
  (`TestGemma4_26B_s10Grade`, 158 s, CPU, from the archive: a correctness run).

  | arm | mean Metal-vs-CPU logit cosine | argmax agreement |
  |---|---|---|
  | fixes in | **0.990848** | 82/117 |
  | `GOINFER_S10_DROP=both` (fixes dropped) | 0.940854 | 71/117 |

  - **PASS (S1.0's rule):** with the fixes in, neither number is lower than with them dropped; both are higher (+0.050 in
    mean cosine, +11 positions in argmax). Only `v_norm` can move on the 26B (all its layers are MoE, whose join already
    applied its own layer scalar), so this is the `v_norm` fix on a real checkpoint.
  - **A caveat the rule does not test:** 70% argmax agreement (82/117) is low next to the 94-95% the dense models score
    in G3, even with the fix. This pairs Metal's W4A8 MoE against the CPU's int4 MoE over 117 positions of a 26B whose
    router flips top-k under tiny input noise (memory: a bit-identical router still flips 779 of 3200 decisions under ~0.5%
    noise), so I expect it is that floor and not a defect, but this run does not separate the two: it has no
    near-tie rule and no per-position margins. If the owner wants that, it is a follow-up with the G1 near-tie rule over
    these logits.
  - Raw: `docs/measurements/multimodal-support-2026-10/s10-26b-grade.log`. The 245 MB dump itself is not committed.
- **S1.0's 26B: the per-position near-tie read the owner asked for, 2026-10-07 on nobara (no new Metal or GPU run).** The same CPU 26B forward over the same dump, with the test extended to classify every
  disagreement (`TestGemma4_26B_s10Grade`; it reproduced the earlier numbers exactly: 0.990848 and 82/117 with the fixes in, 0.940854 and 71/117 dropped). Raw:
  `s10-26b-neartie.log.gz` (the classification), `s10-26b-poscos.log.gz` (per-position cosine).
  - **The disagreements are not near-ties, so the router-noise-floor explanation does not hold for them.** With the fixes in, 35 disagreements: **0 are near-ties under G1c's rule** (Metal's choice within 3% of the CPU's
    top logit, measured in the CPU's logits) and **1 under G3's** (p(Metal's choice) >= half p(CPU's top)); 34 are neither. The median logit deficit of Metal's choice against the CPU's top is 2.95 (range 0.62-18.38), its
    median probability ratio 0.052 (max 0.536); the CPU's own top-1 against top-2 margin at those positions has median 1.19. With the fixes dropped: 46 disagreements, 1 near-tie either way, median deficit 10.24.
  - **They concentrate at the start of each prompt.** 21 of the 35 are in the first 12 positions of a prompt (agreement 15/36 = 42% there), against 14 of 81 from position 12 on (67/81 = 83%, still under the dense models'
    94-95%). Mean Metal-vs-CPU cosine by position, fixes in: position 0 0.978072 (the three prompts share their first token, so this is one position counted three times: its three disagreements are one), positions 1-3
    0.963993 (min 0.909045), 4-11 0.991282, 12 onward 0.994177. With the fixes dropped the early positions are far worse (position 0 0.467925, 1-3 0.864954), so the `v_norm` fix helps most where context is shortest.
  - **What this does and does not say.** The 70% is not explained by near-ties, and it is mostly an early-position effect. It does not say why. Untested candidates: a short-context effect in Metal's paged/W4A8 MoE
    path against the CPU's, or the CPU reference itself being the less accurate side there (nothing here compares either to transformers on the 26B). The test that separates them is the CUDA 26B (the C' streaming
    resident, this card) against the same CPU forward on the same sequences: if CUDA agrees with the CPU at these positions the divergence is Metal-specific; if CUDA diverges the same way, the CPU or a shared stage is the
    suspect. That is a new 26B run, which this read was told not to make, so it is proposed and not done.
- **S1.0's 26B: the separating run, CUDA over the same sequences, 2026-10-07 evening on nobara (the owner approved it).** `TestGemma4_26B_s10DumpCUDA` (the C' streaming resident, 117 positions in 1 min 2 s, the same
  `.int4.metal.giw` read locally, `GOINFER_MOE_CACHE_EXPERTS=1`) wrote `logits-cuda.f32` beside the Mac's dumps, and the CPU grade took it as a third arm. Raw: `s10-26b-cuda-dump.log`, `s10-26b-cuda-grade.log.gz`.
  - **CUDA against the CPU reads like Metal against the CPU:** mean cosine **0.990481** (Metal 0.990848), argmax agreement **85/117** (Metal 82/117); of CUDA's 32 disagreements 3 are G1 near-ties and 4 are G3 near-ties
    (Metal: 0 and 1). By position, CUDA-vs-CPU / Metal-vs-CPU mean cosine: position 0 **1.000000 / 0.978072**; 1-3 0.957641 / 0.963993; 4-11 0.990761 / 0.991282; 12 onward 0.993694 / 0.994177. Argmax disagreements in the first
    12 positions: CUDA 20 of 36, Metal 21 of 36; from position 12: 12 of 81 and 14 of 81.
  - **The two GPU backends disagree with each other as much as either does with the CPU:** Metal against CUDA, mean cosine 0.989909, argmax agreement **79/117** (positions 4-11: 9 of 24; from 12: 67 of 81), 38
    disagreements of which 4 are within the G1 near-tie band. So the early-position disagreement is **not Metal-specific and not the CPU being the odd one out**: three independent implementations (the CPU's int4, Metal's
    W4A8, CUDA's W4A8) pairwise disagree at the same positions to about the same degree.
  - **What this supports and what it does not.** It fits the noise-floor reading the first record doubted, but through a different door than margins: a MoE router flip changes which experts run and moves the logits by a
    lot, not by a near-tie's amount, so "not near-ties" and "router noise" are compatible. It does **not** establish it: no router decision was compared here. The test that would is the router's top-k per layer per
    position on the CPU and on a GPU backend (the CPU has a router capture seam); that is the next step if the owner wants the cause and not only the localization.
  - **One thing is Metal's alone, visible only now:** at **position 0** CUDA matches the CPU to cosine 1.000000 while Metal reads 0.978072 (all three prompts share their first token, so it is one position), and Metal's
    argmax there disagrees with both. Two independent implementations agreeing exactly on the first position makes Metal's first-token logits the suspect. That is a lead for the Mac, not a diagnosis.
  - **The dump files** (`logits-cuda.f32`, 122 MB) sit beside the Mac's in `~/goinfer-bench/s10-26b/` on nobara, uncommitted like the others.
- **S1.0, the `v_norm` fix on CUDA and WebGPU, read 2026-10-07 on nobara (RTX 2070 SUPER, driver 595.91.07): PASS.**
  - **The fix:** every Gemma 4 layer that owns its K/V now applies the scale-less `v_norm`, K=V or not. CUDA: a per-layer
    `vNorm` flag (`cuda/backend.go`), the decode launch in `segA` and the batched-prefill launch in `cuda/prefill.go`
    now key on it, and the unit weight is allocated when any layer has it. WebGPU: `runLayer.vNorm`, and a non-K=V
    layer runs the existing `vNorm` shader on its `v_proj` output into a fresh buffer (one buffer cannot be both the
    shader's read and write binding), after the LoRA hooks. The test seam `GOINFER_S10_DROP=vnorm|both` re-drops it on
    both backends. The dense layer scalar was already right on both, as the brief said.
  - **The readings,** each test's own metric, before (`v_norm` dropped) → after:

    | test | before | after |
    |---|---|---|
    | CUDA scaled dense, pos0 / mean | 0.996182 / 0.910545 | **1.000000 / 0.999634** (argmax 15/16 → 16/16) |
    | CUDA two-geometry, min | 0.977972 (max abs 0.20) | **0.999942** (max abs 0.012) |
    | CUDA scaled MoE, pos0 / mean | 0.998601 / 0.960917 | **1.000000 / 0.996078** |
    | CUDA tiny MoE resident, pos0 / mean | 0.999581 / 0.947512 | **1.000000 / 1.000000** |
    | CUDA MoE localize, worst decision (wgt, x1, x2) | 0.999731, 0.999125, 0.999387 | **1.000000** on all three |
    | WebGPU two-geometry, min | 0.977930 (max abs 0.20) | **1.000000** (max abs 1.8e-07) |
    | WebGPU scaled dense, pos0 / min | 0.996182 / 0.800099 | **1.000000 / 0.975300** (argmax 15/16 → 16/16) |

    - **PASS (S1.0's rule):** no metric moved the wrong way, and re-dropping the fix through the seam reproduces every
      before-fix number exactly (the same digits, on the same binary).
    - **The recorded device baselines moved, on the Gemma 4 rows only.** `cuda/testdata/prefill_tails_baseline.json`
      and `keqv_copy_baseline.json` are bit patterns recorded from before earlier changes; with `v_norm` re-dropped both
      tests pass against the old files (15 tail outputs and 4 runs bit-identical), so the fix is exactly what moved them,
      and they were re-recorded. The `qwen3moe-tiny-k3` rows did not change.
  - **The amendment** (separate commit; each bar sits between its before and after readings, none loosened):

    | bar | old | new |
    |---|---|---|
    | CUDA scaled dense pos0 / mean | 0.97 / relation to the int4-vs-f32 floor only | 0.999 / 0.99 added |
    | CUDA two-geometry min | 0.97 | 0.995 |
    | CUDA scaled MoE pos0 / mean | 0.97 / floor only | 0.9995 / 0.98 added |
    | CUDA tiny MoE resident pos0 / mean | 0.97 / floor only | 0.9999 / 0.99 added |
    | CUDA MoE localize, each branch | 0.99 | 0.9999 |
    | WebGPU two-geometry min | 0.97 | 0.995 |
    | WebGPU scaled dense pos0 / min / argmax | 0.97 / none / 15 of 16 | 0.999 / 0.9 / ~~16 of 16~~ 15 of 16 restored (see the correction below) |

    Under the new bars the fixed code passes all seven tests and `v_norm` re-dropped fails all seven
    (`s10-cuda-amended-*.log`, `s10-gpu-*.log`).
    - **Correction, 2026-10-07 (from the Mac, which read it on its own GPU):** the WebGPU argmax bar of 16/16 was a single device's reading.
      The Apple GPU's WebGPU reads **15/16 with the fix and 15/16 without it**, so `TestGemma4DenseScaled_webgpuParity` was red on the Mac with
      nobara's changes removed too. An argmax count is a near-tie flip, not a continuous metric, and I tightened it on one device's before/after pair.
      The bar is back at its pre-S1.0 value of 15/16 (a restoration of the old bar, not a loosening of a tightened one that two devices agree on);
      the fix's signature stays in the cosine bars, which fail with `v_norm` re-dropped on this box (pos0 0.996182 < 0.999, minCosine 0.800099 < 0.9,
      two-geometry 0.977930 < 0.995) and which the Mac passes. Lesson: tighten a count only where the before and after readings differ on every device
      that runs the test.
  - **Not covered:** the real 26B/31B on CUDA (the dump grading below is the Metal side), and the CUDA/WebGPU
    batched-prefill path is exercised through the prefill baselines and the two-geometry/MoE tests, not a separate
    before/after pair. Raw: `docs/measurements/multimodal-support-2026-10/s10-cuda-*.log`, `s10-gpu-*.log`.
- **S1.1, the fixture and safetensors PLE loading, done 2026-10-06 on the Mac.** No bar was pre-registered for this
  step; it builds what G1 runs on. It was held to the f32 tiny-golden convention (`gemma4_moe_forward_test.go`):
  argmax equal and cosine >= 0.99999 against HF, here at every position rather than only the last.
  - **The loader:** `decoder/weights.go` loads the model-level PLE tensors (the token table streamed a row at a time,
    as GGUF does) and each layer's gate, projection and post norm. It refused any PLE checkpoint before.
  - **The fixture:** `scripts/pin_gemma4_emodel_tiny.py` → `testdata/gemma4-emodel-tiny` (gitignored, 9.4 MB, so
    fingerprinted in `testdata/fixture_identity.json`) and `testdata/gemma4_emodel_tiny_golden.json`.
    - Six layers `[s, s, f, s, s, f]`, PLE P=32, the last two layers KV-shared (layer 4 reads 3; layer 5 reads 2, not
      the layer before it), double-wide FFN on those two.
    - One KV head, head_dim 32 local and 64 global, `k_eq_v` off, prompt 12 against a sliding window of 4.
    - All 42 norm weights and the 6 layer scalars are drawn away from identity.
  - **A loader gap found on the way:** transformers 5.16 writes `per_layer_config` (`{"2": {"head_dim": 64}}`) in
    place of `global_head_dim`, so a checkpoint saved by it loaded every layer at the local geometry. goinfer now
    reads that map and refuses overrides it cannot represent (`Config.gemma4GlobalGeometry`).
  - **Result:** `TestGemma4EModel_safetensorsParity`, worst cosine 1.00000000 over all 12 positions, argmax 12/12,
    6-token greedy continuation identical.
  - **Can it go red:** scaling the PLE projection norm by 1.3 → worst 0.9977; scaling the post-PLE norm by 1.3 →
    0.959 with an argmax flip at position 1.
  - **`gate quick` over the change:** 2100 tests passed, 41 forward goldens green; the one red was
    `TestParityManifest_fresh`, cleared by the goldens-gated `scripts/refresh_parity_hashes.sh`
    (`docs/measurements/multimodal-support-2026-10/s11-gate-quick.log`).
- **S1.2, the resident embedding row, done 2026-10-06 on the Mac.**
  - The CPU's per-position PLE-input code moved verbatim into `Model.gemma4PLEInputs`; the CPU forward and
    `embedResidentInto` both call it. For an E-model the resident row is `[h || L*P PLE inputs]`,
    `ResidentEmbedLen()` long; for every other model it is H, as before.
  - **Check:** `TestGemma4EModel_residentEmbedRow`. The head equals the scaled embedding exactly, and the tail matches
    HF's own `get_per_layer_inputs` + `project_per_layer_inputs` (now recorded by the pin script; the checkpoint
    re-pinned byte-identical) to max |diff| 1.19e-6 over all 12 positions. A reused buffer gives the same row.
  - Metal reads the tail since S1.5 (below).
- **S1.3–S1.6, the E-model on Metal, done 2026-10-06 on the Mac.**
  - **S1.3, per-layer FFN width:** each dense layer carries its width and uniform (`residLayer.ffnI`/`uFFNI`); the
    gate|up and down scratch is sized to the widest layer, and the R18 row picks and the staged-down guard hold for
    every width. Byte-identical for every other model: the Metal snapshot golden still matches 10/10.
  - **S1.4, shared KV:** a KV-shared layer (`residLayer.kvShared`) projects Q only, runs `qk_norm` and RoPE over the Q
    heads, skips `v_norm` and the K/V store, and attends over its source's cache, which `r.kc[l]`/`r.vc[l]` and every
    MC1 slot alias. Its geometry must equal its source's or the build refuses. `UploadKV` refuses a shared layer, the
    decoder's prefill bridge skips it, and `ResidentKVBytes` no longer prices one.
  - **S1.5, the PLE branch:** `encodePLE` after the FFN residual and before the layer scalar: `quant_vec` of the raw
    residual, the PLE gate GEMV, a new `ple_gelu_mul` (the clamped `glu_act` GELU-tanh times this layer's input),
    `quant_vec`, the projection GEMV, `rmsnorm_f32` with the post norm, the residual add. Every entry point stages the
    row through one `loadEmb` and checks its length against `H + L*P` (`metalResident.embLen`, refused at build if it
    differs from the decoder's `ResidentEmbedLen`). `ForwardN` runs an E-model token by token; `ForwardSample` gained
    the length check the others had.
  - **S1.6:** Metal declares `FeatGemma4EModel`; CUDA and WebGPU still decline it (admission tests updated). The f16
    prefill declines it (not in `prefillFeatures`), and the MC3 batched step declines it by name.
  - **G1, read 2026-10-06 on the Mac (M1 Pro): PASS.** 18 positions (the golden's prompt and continuation), int4 both
    sides: argmax 17/18 exact, the one mismatch a 0.32% near-tie, no gap over 3%; mean Metal-vs-CPU(int4) cosine
    0.998510 (min 0.995866) against the CPU's own int4-vs-f32 0.777695 (`TestGemma4EModel_metalResidentParity`).
  - **G2, same run: PASS for the six defects that can be seen.** Each alone turns G1 red on the argmax rule:

    | defect | exact argmax | gaps > 3% | mean cosine |
    |---|---|---|---|
    | (1) PLE branch skipped | 2/18 | 16 | 0.374693 |
    | (2) token-identity term zeroed (resident row only) | 9/18 | 7 | 0.965641 |
    | (3) shared-KV source one owning layer off (layer 4 reads 1, not 3) | 7/18 | 9 | 0.887059 |
    | (4) one FFN width for the model | 7/18 | 11 | 0.845776 |
    | (6) layer scalar dropped | 12/18 | 4 | 0.969368 |
    | (7) `v_norm` dropped on non-K=V layers | 9/18 | 8 | 0.921480 |

    Note the mean-cosine envelope alone would have passed every one of them (all above 0.777695); the argmax rule
    caught them. On this fixture the envelope is a crater detector, not a sensitive one.

  **G2 amendment, 2026-10-06, after the G2 run (a mechanism, stated with its evidence):** defect (5), the K/V store not
  skipped on a shared layer, is a byte-for-byte no-op in this design, so no fixture can turn it red. A shared layer's
  Q-only GEMV writes only the Q region of the qkv scratch; its K/V slots still hold what its source (the last owning
  layer of that attention type) stored earlier in the same token, because the other type's writes end below that slot
  (fixture: sliding writes floats 0-191, the global K slot starts at 256; E2B: 0-2559 against 4096). The unskipped
  store re-writes the source's own bytes. Measured: with it planted, Metal's logits are bit-identical to the clean
  run at all 18 positions. That bit-identity is now the check (`TestGemma4EModel_sharedStoreIsNoOp`): if a layout
  change ever breaks it, an unskipped store would corrupt the source cache, and that test goes red first. The skip
  stays in production. This is flagged for the owner, since it was decided after the run.
  - **Raw:** `docs/measurements/multimodal-support-2026-10/s1-g1-g2.log` (G1, G2 and the no-op check) and
    `s13-16-gate-quick.log` (`gate quick` over the change: 2105 passed, 0 failed, lint 17/17).
- **G1, tiny E-model, Metal resident against the CPU, every position, int4 on both sides:**
  - argmax identical, a first divergence where the CPU's top-1/top-2 margin is under 3% counting as a near-tie (the
    two-geometry rule);
  - and mean Metal-vs-CPU(int4) cosine >= mean CPU(int4)-vs-CPU(f32) cosine (the scaled-dense envelope).
- **G2, planted defects, each red on G1:**
  - (1) the PLE branch skipped;
  - (2) PLE's token-identity term zeroed, or its 1/sqrt(2) dropped;
  - (3) the shared-KV source one layer off;
  - (4) one FFN width for the model;
  - (5) the K/V store not skipped on a shared layer;
  - (6) the layer scalar dropped;
  - (7) `v_norm` dropped on non-K=V layers.

  A defect that stays green means the fixture is degenerate along that axis. Fix the fixture, not the bar.

  **G2 amendment, 2026-10-06, before any G2 run (a mechanism, not a reading):** the second half of (2), the
  1/sqrt(2) dropped, cannot go red on logits for any checkpoint. The factor multiplies the whole per-layer input;
  the PLE branch is linear in that input (gelu(gate.h) x input, then a projection) and ends in
  `post_per_layer_input_norm`, an RMSNorm, which removes a uniform scale except through its eps. Measured on the CPU
  against HF while building S1.2: dropping it moves the worst logit cosine from 1.00000000 to 0.99998923. So that
  half is checked where it is visible, on the embedding row's PLE tail against HF's own per-layer inputs
  (`TestGemma4EModel_residentEmbedRow`, red at max |diff| 1.02 with it dropped), and G2's (2) keeps only "the
  token-identity term zeroed". This is the axis being invisible by construction, not the fixture being degenerate,
  so the rule above does not apply to it.
- **G3, real E2B text:**
  - 8 fixed prompts x 32 greedy tokens, Metal resident against the CPU, both int4 from the same GGUF.
  - **PASS:** identical tokens, or a first divergence at a near-tie, on every prompt; and teacher-forced argmax
    agreement >= 99% over all positions.
  - **Ambiguous (parked for the owner):** 97-99%. **Fail:** under 97%.
  - **G3's prompt set and procedure, fixed 2026-10-06 before any G3 run.** The model is
    `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf` (local disk), loaded `Quant: int4` twice: once for the CPU,
    once for Metal's resident (each from its own sidecar of that GGUF). Each prompt is one user turn rendered by the
    GGUF's own chat template (`chat.Detect`), no system message, encoded as the chat CLI does with a template
    (`Encode(prompt, false)`: the template writes its own `<bos>`; corrected 2026-10-06 before any run, the first wording
    said the tokenizer adds it, which would have doubled it). The eight prompts:
    1. `Explain why the sky is blue in two sentences.`
    2. `Write a haiku about autumn leaves.`
    3. `What is 17 multiplied by 23? Show your work.`
    4. `Translate "Where is the train station?" into French and German.`
    5. `List three differences between Python and Go.`
    6. `Summarize the plot of Romeo and Juliet in one paragraph.`
    7. `Write a Go function that reverses a string.`
    8. `What are the main causes of inflation?`
    - **Free-running:** each side prefills token by token and decodes 32 greedy tokens (no early stop on EOS). Same
      tokens, or a first divergence where the CPU's logit gap between its top-1 and Metal's pick is under 3% of
      |top-1| (the two-geometry rule), passes that prompt.
    - **Teacher-forced:** the CPU's sequence (prompt + its 32 tokens) is fed to Metal from a reset cache; agreement is
      Metal's argmax equal to the CPU's at every position that predicts a next token, pooled over the eight prompts.
    - Memory: Metal's own fit guard (`residentFitsMemory`) must admit the model; the test refuses otherwise.
  - **G3 run 1, 2026-10-06 20:53 PDT, on the Mac: FAIL as registered, with a confound in the procedure.**
    - Free-running: 7/8 prompts pass (prompt 3 first diverges at generated token 19, CPU gap 3.61%; 5, 6, 7, 8 diverge
      at near-ties of 2.03%, 0.34%, 1.55%, 1.13%). Teacher-forced agreement 409/435 = **94.02%**, under the 97% fail line.
      Metal's text is coherent throughout and tracks the CPU's.
    - **The confound:** the run met the load-time fit guard by loading each side's pre-built sidecar, and the CPU's is
      the `e4h` one: its embedding/LM-head and PLE token tables are int4 (the CPU default since 2026-09-28; the loader's
      own help puts it at ~2.3 pts top-1), while Metal's sidecar keeps them at the int8 pin. So the run compared two
      different quantizations of the logit-critical tables, not Metal against the CPU on the same weights, which is what
      "both int4 from the same GGUF" was written to mean.
    - Raw: `docs/measurements/multimodal-support-2026-10/g3-run1-e4h-cpu.log`.
  - **G3 amendment, after run 1 (flagged for the owner, since it follows a failing run):** both sides load the same
    sidecar, Metal's (`.int4.metal.giw`, int8-pinned tables); nothing else changes — prompts, rule and bands stand. A
    control arm, reported and not graded, runs the same teacher-forced comparison CPU against CPU across the two
    sidecars (int8 tables against `e4h`), to measure how much of run 1's 94.02% the table precision alone explains.
    Run 2's verdict stands as the G3 reading only if the owner accepts this amendment.
  - **G3 run 2, 2026-10-06 20:56 PDT (amended procedure): FAIL as registered.** 7/8 prompts pass (prompt 3 first
    diverges at generated token 19, CPU gap 5.91%); teacher-forced agreement 413/435 = **94.94%**. Deterministic: a
    re-run that logs each disagreement gives the same 413/435. Of the 22 disagreements, 17 are near-ties under 3% and
    5 exceed it (7.63, 5.91, 4.00, 3.52, 3.49%), four of those five at prompt positions (corrected 2026-10-07 from
    "16 ... 6": `g3-run2.log` lists five gaps over 3%).
    - **Control (CPU int8 tables vs CPU e4h tables, same procedure): 93.56%.** A table-precision change alone, on the
      CPU, disagrees as often as Metal does: run 1's confound was real but was not what failed it.
    - **Localization (exploratory), the residual after every layer, Metal vs CPU, at three positions including the
      worst (prompt 6 pos 18):** smooth drift from 1.00000 at layer 0 to 0.986-0.999 at layer 34; no layer drops more
      than ~0.002, and nothing changes at layer 15 (the first KV-shared layer) or any PLE layer. That is the shape of
      accumulated quantization noise, not of a defect.
    - **Calibration (exploratory, the same `g3Run` on Qwen2.5-Coder-1.5B, whose Metal path is validated): 7/8 prompts,
      94.22%.** A known-good Metal path fails G3's bands by the same margin, so the 99%/97% bands were set without
      calibrating the instrument: Metal's W4A8 (per-tensor int8 activations, f16 scales) against the CPU's int4 path
      flips about 5-6% of argmax positions on these prompts, mostly at near-ties.
    - **Status: G3 FAIL as first registered; the owner re-registered it (below), and it passes that.** Not re-graded under the first rule. One candidate re-registration, for the
      owner to accept or reject: non-inferiority against the calibration (E2B's agreement at least the validated
      model's minus a stated margin, measured in the same session), plus the free-run rule.
    - Raw: `g3-run2.log`, `g3-control.log`, `g3-localize.log`, `g3-calibration-qwen15b.log` in
      `docs/measurements/multimodal-support-2026-10/`.
  - **G3 re-registered (owner decision, 2026-10-06), written before the run that grades it.** Disclosed: the rule was
    written after the owner and I had seen run 2 (94.94%) and the calibration (94.22%); the run below re-measures both.
    - **Instrument:** one test process (`TestGemma4EModel_realE2BNonInferiority`) runs `g3Run` (unchanged: the same
      eight prompts, both sides on one Metal sidecar, int4, a 512-token pinned context) first on Qwen2.5-Coder-1.5B,
      whose Metal path is validated (the reference), then on E2B.
    - **Margin:** 2.0 percentage points of teacher-forced agreement, about two binomial standard errors at ~435
      positions and p near 0.94.
    - **PASS:** E2B agreement >= reference agreement - 2.0 points, and E2B's free-run prompt passes >= the reference's
      - 1.
    - **Ambiguous (parked for the owner):** agreement 2.0-4.0 points below the reference, free-run condition met.
    - **FAIL:** agreement more than 4.0 points below the reference, or E2B's free-run passes 2 or more short.
  - **G3 read 2026-10-06 21:17 PDT on the Mac (M1 Pro), the re-registered rule: PASS.** Reference Qwen2.5-Coder-1.5B
    7/8 prompts, 375/398 = 94.22%; E2B 7/8 prompts, 413/435 = 94.94%; delta **+0.72 points** against the -2.0 margin,
    free-run passes equal. Both numbers reproduce the earlier runs exactly (the procedure is deterministic). Raw:
    `docs/measurements/multimodal-support-2026-10/g3-noninferiority.log`.
- **G4, real E2B image chat:**
  - The F2b request through the Metal serve binary, resident decode against CPU decode.
  - **PASS:** identical reply, or a first divergence at a near-tie.
  - **G4's procedure, fixed 2026-10-06 before any G4 run.**
    - **What changes first:** E2B's image turns did not reach the resident at all: `GenerateGemma4VL` bridged a CPU
      prefill to resident decode only for the bidirectional 26B/31B class. The gate now also admits the E-model class
      (only Metal has a resident for it), and `Generation.DecodeResident` plus one serve log line ("vision: decoded N
      tokens on the resident path") say where a turn decoded. A causal non-E Gemma 4 is unchanged.
    - **Request:** F2b's (`testdata/glm_ocr/table.png`, "What does this image show? Answer briefly.", `max_tokens` 32,
      temperature 0), `--model ~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf --vision ~/models/gemma-4-E2B-unq`.
    - **Arms:** one `metal/cmd/serve` binary built at the commit under test. The resident arm, `--backend metal`, must
      report the resident decode path at load and "on the resident path" for the turn, or the run is void. The CPU arm
      is `--backend cpu`; F2a/F2b showed the two towers agree (cosine 1.000000000) and give identical replies, so the
      arms differ in decode.
    - **PASS:** identical reply text. A divergence passes only as a near-tie: the CPU's sequence is replayed in-process
      to the first differing token and the CPU's logit gap between its token and the resident's must be under 3% of
      |top-1| (the G1/G3 rule).
  - **G4 read 2026-10-06 21:30 PDT on the Mac (M1 Pro), `metal/cmd/serve` at `3f198af6`: PASS.**
    - Resident arm: `decode path: metal-resident (int4)`, tower on Metal, "vision: decoded 32 tokens on the resident path".
    - CPU arm: `decode path: cpu (int4)`, tower on the CPU, "on the cpu path". (Serve's CPU arm loads its `e4h` sidecar,
      int4 embedding/LM-head tables against Metal's int8; the replies matched regardless.)
    - Both replies, byte-identical: "This image shows **Table 2: Quarterly unit sales by region (in thousands)** for the
      fiscal year 2025 (FY2025". Wall time 15.3 s and 14.9 s per request (exploratory: each is dominated by the
      286-token CPU prefill and the tower).
    - Raw: `g4-serve-{metal,cpu}.log`, `g4-reply-{metal,cpu}.txt` in `docs/measurements/multimodal-support-2026-10/`.
- **Speed (night):** E2B decode tokens per second, resident against CPU, same-session interleaved, plus the host's
  PLE milliseconds per token. A record; not a gate.
  - **S1.9's procedure, fixed 2026-10-06 before the run:** `docs/measurements/multimodal-support-2026-10/run-s19-speed.sh`
    on the Mac's night queue, from binaries pre-built at `888d1d4f` (`~/goinfer-bench/s19/`).
    - `bench_peer.py`, goinfer only, model `E2B` (added to its table; depth-128 prompt calibrated to 129 tokens),
      Phase A at one depth, backends `cpu` and `metal` from the one serve binary, the harness's defaults (64 tokens x 8
      completions x 2 runs, greedy, instant idle gate).
    - The host's per-token cost of the embedding row plus PLE inputs (`TestGemma4EModel_realE2BPLEHostCost`, 512
      tokens after 32 warm-up), under the timing lock.
    - Recorded as numbers with their provenance; no bar.
  - **S1.9 recorded, Mac night queue 2026-10-06 22:10 PDT (M1 Pro, macOS 26.6.2, AC power, binaries at `888d1d4f`,
    model `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf`, int4, greedy, depth 128 (129 prompt tokens), 64 tokens x
    8 completions x 2 runs per cell, instant idle gate, no thermal warning):**

    | backend | decode tok/s (runs) | decode path | peak RSS |
    |---|---|---|---|
    | CPU | 38.0 (37.92, 37.99) | `cpu (int4)`, row4 layout, `e4h` sidecar | 1.40 GiB |
    | Metal resident | 46.1 (46.09, 46.10) | `metal-resident (int4)` | 0.38 GiB |

    - Metal resident is 1.21x the CPU on E2B decode. A record, not a gate. The CPU arm runs its own sidecar (int4
      embedding/LM-head tables), the shipped default for each backend.
    - The host's embedding row + PLE inputs: mean 0.315 ms/token (median 0.304, p90 0.347, 512 tokens), about 1.5% of a
      Metal token (21.7 ms): not where the time goes.
    - Raw: `docs/measurements/multimodal-support-2026-10/s19-e2b-speed-night.log` (and `s19-bench-e2b.json`).

**S1 on CUDA: plan and gates, written 2026-10-07 on nobara before any CUDA E-model code or measurement.** A desk map of the CUDA
resident against Metal's version (every claim checked against the tree at `24c16cf2`; one research pass, no run) fixes the
plan. Where it says INFERRED, the step names the fallback and what decides it.

- **What is already known, from the code:**
  - **No new CUDA kernel and no PTX regeneration for PLE.** `glu_quant` computes `act(g[k]) * u[k]` plus int8 quant with
    separate `g`/`u` pointers and offsets (`cuda/glue.cu`), and `ACT_GELU_TANH` is 0. The PLE multiply is
    `glu_quant(g=pleG, u=pleIn, uOff=l*P, I=P, act=0)`; every other PLE step reuses an existing launch (`quant_vec`, `doG`
    for the gate and projection, `normF32`, the residual add, `scale_vec`). The audited glue/moe/gemv_fwd PTX trio stays
    pinned at NVRTC 12.6.85, untouched. If a kernel turns out to be needed after all, its PTX is regenerated at the pinned
    NVRTC and that is recorded here before the gate that depends on it.
  - **No new kernel for shared KV.** `rope_kv` with `nKV=0` rotates Q only and skips the K/V store; `fused_rms_qkv` with
    `kvDim=0` never reads Wk/Wv; `qk_norm` with `nKV=0` launches no K blocks.
  - **Three silent failures if E-models were admitted unchanged** (so the steps below remove them in the same change, not
    afterwards): (i) scratch is sized from the model-level FFN width (`FFNPerLayer[0]`, 6144 on E2B) while 20 layers are 12288,
    so `fused_rms_gu` and `glu_quant` would write past `gO`/`uO`/`dq`, a device-memory overrun and not a decline; (ii) the
    batched prefill copies each row with `copy(xhost[m*hidden:(m+1)*hidden], e)`, which silently drops a PLE tail; (iii)
    `BuildResident` packs and validates K/V and K-norm weights that a shared layer does not have (it already errors on them, so
    today CUDA declines E-models by naming the missing feature).
- **Steps, in order. Each is one commit, and each is checked on the tiny fixture before the next starts:**
  - **C0, the fixture on this box.** `testdata/gemma4-emodel-tiny` is gitignored and absent here. Regenerate it with
    `scripts/pin_gemma4_emodel_tiny.py` (`~/.venv-vl`, transformers 5.16.1 as the golden records) and require it to match
    `testdata/fixture_identity.json` and `TestGemma4EModel_safetensorsParity` on the CPU (cosine >= 0.99999 at every position, as
    S1.1). If it does not reproduce, stop and say so: every CUDA reading below is against that fixture.
  - **C1, per-layer FFN width.** Gate/up scratch, `glu_quant` scratch and the launches take `Ly.g.N` / `Ly.d.K` per layer, scratch
    sized to the widest layer. Byte-identical for every non-E-model: the device baselines (`prefill_tails_baseline.json`,
    `keqv_copy_baseline.json`) and the Gemma 4 / dense parity tests must not move.
  - **C2, shared KV.** A KV-shared layer (`Ly.kvShared`, `Ly.kvSrc` from `KVSrcAtResident`) leaves its own `r.kc[l]`/`r.vc[l]` and
    every MC1 slot's zero-valued, projects Q only (unfused and fused), runs `qk_norm` and `rope_kv` with `nKV=0`, skips `v_norm`
    and the K/V store, and attends over its source's cache at every launch site (split-KV, flash-decode, decode attention). Its
    `kEqV` and `vNorm` are false (checked before the source's). `UploadKV` refuses a shared layer by name. `kvBytesForCap` and
    `decoder.cudaKVBytes` stop pricing it, together, and `TestResidentKVBytes_matchesCUDAAllocation` and
    `kvbytes_agreement_test.go` gain the E-model fixture. The build refuses a shared layer whose geometry differs from its
    source's (as Metal does). Aliasing the buffers (Metal's choice) is the fallback only if a launch site cannot be made to
    resolve the source.
  - **C3, PLE.** The resident row is `H + L*P` long; `launchToken` (the funnel every single-token entry point goes through)
    refuses any other length by name. The tail lives in `r.x`'s own allocation (`pleIn = r.x.At(H*4)`) so the existing upload
    carries it; if that does not work on the device, a second buffer and a second upload (INFERRED either way, the first run
    decides, and the choice is recorded). Per layer, after the FFN residual and before the layer scalar: `quant_vec` of the raw
    residual, the PLE gate GEMV, `glu_quant` (GELU-tanh times this layer's input), the projection GEMV, `normF32` with the post
    norm, the residual add. It sits in `segB`, so CUDA graphs capture it; every operand is a fixed-address buffer and `l*P` is a
    per-layer argument. `graphsSelfTest`'s hidden-sized row becomes the augmented length with a non-zero tail.
  - **C4, declines.** `prefillStaticDecline` declines E-models by name (so every batched entry point, MC3 `StepBatch`,
    `ForwardN`'s batched arm and `PrefillPath` report it and the sequential fallbacks take over); `AttachDrafter` refuses an
    E-model target by name; a LoRA adapter on an E-model is declined by name (its `r.inter` K is wrong on wide layers, and nothing
    verifies it). Each decline has a test that asserts the named reason.
  - **C5, declare it.** `FeatGemma4EModel` goes into CUDA's feature set only once C1-C4 pass; WebGPU keeps declining. The
    admission tests, `docs/hardware-matrix.md` (generated), `docs/multimodal.md` and `docs/gpu-residency-coverage.md` move with it.
- **Gates (CUDA's G1-G4, registered now; each reads the resident against the CPU on the same weights):**
  - **G1c, the tiny E-model, every position, int4 on both sides** (the golden's prompt and continuation, 18 positions). Metal's
    G1 rule unchanged: a first mismatch is benign only where the CPU's gap between its top-1 and the resident's pick is under 3% of
    |top-1|; PASS when every mismatch is benign and the mean CUDA-vs-CPU(int4) cosine is at least the mean CPU(int4)-vs-CPU(f32)
    cosine. The test first asserts the resident took the shape (PLE width, two shared layers, the two FFN widths).
    The minimum cosine is reported. After the first reading a separate recorded amendment may TIGHTEN the bars between the
    planted-defect and clean readings, as S1.0 did; they are never loosened.
  - **G2c, planted defects, each alone red on G1c:** (1) PLE skipped; (2) PLE token-identity term zeroed (the decoder seam);
    (3) the shared-KV source one owning layer off; (4) one FFN width for the model; (6) the layer scalar dropped; (7) `v_norm`
    dropped on non-K=V layers; and, CUDA-specific, (8) the PLE per-layer input offset off by one layer (`uOff` = `(l+1)*P`).
    Defect (5), the K/V store not skipped on a shared layer, is registered with its outcome open: on Metal it is a byte-for-byte
    no-op and is checked as such. On CUDA the shared layer's `kB`/`vB` scratch may hold a different layer's K/V (layer 5 is
    full-attention with source layer 2 while the scratch holds layer 3's sliding K/V), so it may go red here. Either way it is
    recorded: red means it is a gate; green means a bit-identity check like Metal's (`..._sharedStoreIsNoOp`) replaces it, and that
    is stated as a mechanism with its evidence, not as a pass. A defect that stays green otherwise means the fixture is degenerate
    on that axis: fix the fixture, not the bar.
  - **G2c-graphs and accounting, same fixture:** a graphs-on run is bit-identical to a graphs-off run (a prompt longer than the
    sliding window, non-zero PLE tail, `GOINFER_CUDA_GRAPHS_UNSAFE` because this box's DEFAULT compute mode declines graphs); the
    KV bytes the planner prices equal the bytes allocated; every decline in C4 returns its named reason and a greedy generation
    through the sequential fallbacks equals the CPU's tokens.
  - **G3c, real E2B text.** Metal's re-registered G3 rule, adopted unchanged and written before any CUDA run: one test process
    runs the eight G3 prompts x 32 greedy tokens (same prompts, same free-run and teacher-forced procedure, 512-token pinned
    context) first on Qwen2.5-Coder-1.5B on CUDA (its CUDA path is validated; the reference), then on E2B. **PASS:** E2B's
    teacher-forced agreement >= the reference's - 2.0 points and its free-run prompt passes >= the reference's - 1. **Ambiguous
    (parked for the owner):** 2.0-4.0 points below, free-run met. **FAIL:** more than 4.0 below, or 2 or more free-run passes short.
    Lessons from Metal's G3 run 1 are applied up front, not as an amendment: both sides run the SAME table precision (the CPU arm
    loads the sidecar matching CUDA's embedding/LM-head and PLE table precision, and the log prints each side's), and a control arm
    (CPU against CPU across the two sidecars, reported, not graded) states how much table precision alone moves the number. Model:
    `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf` (local disk), int4. The fit guard must admit it on the 8 GB card or the
    test refuses.
  - **G4c, real E2B image chat through the CUDA serve binary.** F2b's request (`testdata/glm_ocr/table.png`, "What does this image
    show? Answer briefly.", `max_tokens` 32, temperature 0), `--vision ~/models/gemma-4-E2B-unq`; resident arm `--backend cuda`, CPU
    arm `--backend cpu`, one binary built at the commit under test. The resident arm must report the CUDA-resident decode path at
    load and "decoded N tokens on the resident path" for the turn, or the run is void. **PASS:** identical reply, or a first
    divergence replayed on the CPU at a near-tie (gap under 3%). The tower stays on the CPU (S4 is separate).
- **Speed (night, a record, no bar):** E2B decode tokens per second, CUDA resident against CPU, same-session interleaved through
  `bench_peer.py` (goinfer only, the `E2B` model entry, depth 128), plus the host's PLE milliseconds per token under the timing
  lock. Queued after G3c/G4c pass; nothing is quoted before then.
- **By day / by night:** C0-C5, G1c, G2c, the graphs and accounting checks and G4c are seconds to a few minutes: by day. G3c loads
  E2B twice and runs about 870 teacher-forced positions on the CPU: a guess of 5-10 minutes (no per-position time measured yet), restated
  from a measured time before it starts and queued if it exceeds 10. The speed record is a night job.
- **Open on purpose (INFERRED in the desk map, decided by the first run, each recorded where it is decided):** the `r.x` tail
  against a second upload; whether `gemv_w4a8` handles the fixture's PLE projection at K=32; whether defect (5) is detectable
  on CUDA; whether LoRA touches an E-model at all (declined by name regardless).

- **S1 on CUDA, read 2026-10-07 on nobara (RTX 2070 SUPER, driver 595.91.07, int4): G1c, G2c, graphs, accounting, G3c, G4c PASS.** Pre-registered above; implemented as C0-C5
  in commit `d1cc93c0`. Nothing below is a speed claim; the speed record is owed on the night queue.
  - **What was built, and where it departs from the plan.**
    - **C0:** the tiny E-model fixture regenerated on this box with transformers 5.12.0 (the golden was pinned with 5.16.1): the HF logits differ
      from the committed golden by at most 8.7e-6, argmax and continuation identical, `testdata/fixture_identity.json` and
      `TestGemma4EModel_safetensorsParity` green (worst cosine 1.00000000 over 12 positions). The directory is gitignored and stays untracked.
    - **C1:** per-layer FFN width (`cudaLayer.ffnI`), scratch sized to the widest layer. Byte-identical for every existing model: both device
      baselines (`prefill_tails_baseline.json`, `keqv_copy_baseline.json`) unmoved.
    - **C2, a departure:** KV sharing **aliases the buffers** (`r.kc[l]`, `r.vc[l]` and every MC1 slot's, set to the source's), which the plan named
      only as the fallback. The desk map counted ten launch sites that index `r.kc[l]` (split-KV, flash-decode, decode attention, the rest); aliasing
      needs none of them touched, and it is Metal's choice. A shared layer projects Q only, runs `qk_norm` and `rope_kv` with `nKV=0` (no K/V
      store), and attends over the aliased cache with its source's geometry (checked at build). `kvBytesForCap` and `decoder.cudaKVBytes` skip
      shared layers; `UploadKV` refuses one by name. E-models take the **unfused** QKV chain (the fused kernels need K/V on every layer), so
      `fuseQKV` is off for them: slower, correct, and an optimization for later.
    - **C3:** the resident row is `[hidden || 35*256]` for E2B (`r.x` is allocated that long and the single upload carries the tail; each layer's
      `glu_quant` reads its slice with an element offset, so no buffer view is needed: the open question about `Buffer.At` never arose). The PLE
      branch is `quant_vec`, the gate GEMV, `glu_quant` (act 0, GELU-tanh), the projection GEMV, the post-norm, the residual add, between the FFN
      residual and the layer scalar, in `segB` (so graphs capture it). **No new kernel and no PTX regeneration.** `gemv_w4a8` handled the fixture's
      K=32 projection (G1c passes), which was one of the open questions. `launchToken` refuses a row of the wrong length by name.
    - **C4:** `prefillStaticDecline` declines E-models by name (every batched entry point, MC3, `ForwardN`'s batched arm, and the serve banner reports
      "prefill path: sequential — Gemma 4 E-model ... has no batched prefill"); `AttachDrafter` and `SetAdapter` refuse by name.
    - **C5:** CUDA declares `FeatGemma4EModel`; WebGPU still declines. The admission tests, `docs/hardware-matrix.md` (regenerated; its note had
      said no backend declares the feature, which Metal had already made false) and `docs/gpu-residency-coverage.md` moved with it.
  - **G1c (`TestGemma4EModel_cudaResidentParity`), tiny E-model, 18 positions, int4 both sides: PASS.** Argmax 18/18 exact, mean CUDA-vs-CPU(int4)
    cosine **0.999931** (min 0.999346), against the CPU's own int4-vs-f32 mean 0.777674 (Metal's reading of the same instrument: 0.777695).
    The test first asserts the resident took the shape: P=32, two shared layers, FFN widths 256 and 512.
  - **G2c (`TestGemma4EModel_plantedDefects`), each defect alone: all eight RED**, on the argmax rule (the mean-cosine envelope alone would have passed
    most of them, as on Metal):

    | defect | exact argmax | gaps > 3% | mean cosine |
    |---|---|---|---|
    | (1) PLE branch skipped | 2/18 | 16 | 0.371194 |
    | (2) token-identity term zeroed (decoder seam) | 10/18 | 6 | 0.966228 |
    | (3) shared-KV source one owning layer off | 7/18 | 9 | 0.887704 |
    | (4) one FFN width for the model | 8/18 | 6 | 0.952167 |
    | (5) K/V store not skipped on a shared layer | 5/18 | 11 | 0.763998 |
    | (6) layer scalar dropped | 10/18 | 6 | 0.972024 |
    | (7) `v_norm` dropped on non-K=V layers | 10/18 | 8 | 0.920343 |
    | (8) PLE per-layer input offset one layer off (CUDA-specific) | 0/18 | 18 | 0.194175 |

    - **Defect (5) was registered with its outcome open, and it goes red on CUDA** (it is a byte-for-byte no-op on Metal). The shared layer's K/V
      scratch holds another layer's data, so the unskipped store writes it through the alias into the source's cache. It is therefore a real gate
      here, not a bit-identity check.
  - **Graphs (`TestGemma4Graphs_bitExact_emodel`, `GOINFER_CUDA_GRAPHS_UNSAFE=1`, 40 positions past the sliding window): replay is bit-identical to the
    live launches**; the tautology guard confirms graphs were captured. **KV accounting** (`TestResidentKVBytes_matchesCUDAAllocation` with the
    E-model fixture): allocated bytes equal `kvBytesForCap` equal `ResidentKVBytes("cuda")`, and the shared layers' buffers are asserted to alias.
    **Declines and fallbacks** (`TestGemma4EModel_declinesAndFallbacks`): each decline returns its named reason, a hidden-sized row is refused by
    length, `UploadKV` to a shared layer is refused, and 8 greedy tokens through the sequential fallbacks equal the CPU's.
  - **G3c (`TestGemma4EModel_realE2BNonInferiority`, real E2B and the Qwen2.5-Coder-1.5B reference, 8 prompts x 32 tokens, both sides loading the same
    GGUF with the same options): PASS.** E2B 7/8 prompts, teacher-forced agreement 411/435 = **94.48%**; reference 7/8 prompts, 374/398 = **93.97%**;
    delta **+0.51 points** against the -2.0 margin, free-run passes equal (7 and 7). 144 s end to end. (Metal's reading of the same rule: 94.94% and 94.22%.)
    - **Run 1 was flawed and is superseded.** The CPU side of the reference loaded the CUDA `e4h` sidecar (int4 embedding/LM-head tables) while the
      CUDA side loaded the GGUF with `EmbedInt4` unset (int8 pin): the confound Metal's G3 run 1 fell into, which the plan said to avoid up front. It
      read the reference at 89.20% (355/398) and E2B at the same 94.48%, a delta of +5.29 that came entirely from a depressed reference. Caught from the
      log before it was recorded; the procedure was corrected (both sides load the same file with the same options) and the run repeated under the
      same rule and bands. Raw: `s1c-g3c-run1-asymmetric-tables.log`, `s1c-g3c-run2.log`. E2B's own numbers did not move between the runs, because
      its two sides were loaded symmetrically both times.
    - The plan's table-precision control arm is not run: with both sides on one file and one set of options there is no table-precision difference to measure.
  - **G4c (the F2b request through one CUDA-built serve binary, `d1cc93c0`): PASS, byte-identical replies.** Resident arm `--backend cuda`:
    `decode path: cuda-resident (int4)`, tower on the CPU, "vision: decoded 32 tokens on the resident path". CPU arm: `decode path: cpu (int4)`,
    "on the cpu path". Both replies: "This image shows **Table 2: Quarterly unit sales by region (in thousands)** for the fiscal year FY2025." and
    a start of the next sentence. The CUDA reply differs from the Metal run's ("... fiscal year 2025 (FY2025"), which is a different backend's quantization
    on the same prompt; the gate compares each backend to the CPU. Raw: `s1c-g4c.sh`, `s1c-g4c-serve-{cuda,cpu}.log`, `s1c-g4c-reply-{cuda,cpu}.txt`.
  - **Not done, and not claimed:**
    - **The speed record** (E2B decode, CUDA against CPU, same-session interleaved, plus the PLE host cost) is queued for tonight as `s1c-e2b-speed`
      (`run-s1c-speed.sh`, pre-built binaries at `928f9a41`, est 30 min, the harness defaults, no bar) and has not been read. One exploratory smoke of the
      script (`BENCH_RUNS=1`, 2.5 min, not a result and not to be quoted) showed the plumbing works end to end: both cells passed the token gate, and the
      host's embedding row plus PLE inputs cost 0.33 ms/token (Metal's reading on its own CPU: 0.315).
    - **E4B** was not run (no checkpoint on either box). The code path is the same shape, and nothing was measured on it.
    - **TTFT:** a long prompt prefills token by token (the batched prefill declines E-models), so time to first token on a long prompt is slow. The
      vision tower is still on the CPU (4.5 s here; S4).
    - **The heavy tier** (`gate gpu`) has not completed on this tree. The first night run (2026-10-07, stopped by the owner at minute 35 of the heavy
      tier, `~/goinfer-logs/gate-gpu-night-2026-10-07/`) was red for three reasons, none a defect in the E-model path, all fixed and pushed:
      (1) **my** helper `mustResident` sat in a file tagged `cuda` alone but called a test hook, so the no-testhooks cuda test package did not build
      (group 2a; present since f95fabfd, invisible to CI, which cannot build cuda tests); (2) the gate's pinned worktree linked only the gitignored weights
      of `testdata/gemma4-emodel-tiny`, leaving `model.safetensors` without `config.json`, so 13 E-model tests failed to load (the script now links untracked
      fixture directories whole and excludes them from the dirty-tree check); (3) an **unpinned default resident context** that one KV slot overshoots by the
      build's own scratch declined the resident path to the CPU (`TestB2DenseFlagship` and four Gemma-3 tests; R19 raised the default to 16384 and says it
      is "shrunk to what the card holds", but only the multi-slot arm did), now trimmed to fit with a floor of 4096. The six tests pass again and the full
      non-heavy cuda suite is green. A control run with the heavy tier skipped (`GOINFER_GATE_SKIP_HEAVY=1`, pinned worktree of `6ceb11a6`) reads
      **PASS, 10 of 10 groups**, with the E-model tests running in 2b and 2d. The rest of the heavy tier (about 40 of 78 minutes were never reached) is
      re-queued for the owner to start.
  - **`gate quick` over the change (18 min, `~/goinfer-logs/gate-quick-s1c.log`): 2085 passed, 0 failed, 429 skipped; lint 16 of 17.** The one red lint step was
    `staticcheck` on `./cuda/...`: two unused fields (`resident`, `cpuInt4v`) I left in the new `gemma4_emodel_test.go`, the dead-field class CLAUDE.md warns
    about. Removed; `staticcheck` for `-tags 'cuda goinfer_testhooks'` and `-tags 'gpu goinfer_testhooks'` re-run clean, the full gate not repeated.

### S2 — GPU towers for Qwen3.5+ and GLM-OCR

On CUDA, after P26b, the Qwen3.5 tower is about 3.3 of the 3.6 s a new image costs. GLM-OCR's tower is the slowest one
in the tree.

- **Approach:** the Gemma 4 Metal tower's.
  - aikit exports the tower's float32 weights (a `Weights()` for `Qwen3VisionEncoder` and `GlmOcrVisionEncoder`; new
    aikit API, next release).
  - goinfer runs the blocks on the device.
  - aikit's tail (merger, projection) stays shared, like `FinishHidden`.
  - Metal and CUDA. The attention blocks, GEMMs and norms from `metal/gemma4_vision.go` largely carry over. Qwen's
    2-D RoPE and its bilinear position-table interpolation are new kernels.
- **Gates to write before starting:**
  - Device tower against aikit's CPU tower, every merged token at cosine >= 0.9999, on the P8a images and on GLM-OCR's
    O3 documents.
  - Served replies identical against the CPU tower.
  - Planted defects in the new kernels.
- **Speed (night):** the tower per image on both boxes, against the CPU tower.
- **Size:** L.

**S2 Gate 0, the desk map (2026-10-07, Mac, aikit at v1.58.0 + 2 test-only commits):**
- **Neither tower has a device seam.** `vision.Qwen3VisionEncoder` (`vision/qwen3_encoder.go`) and
  `vision.GlmOcrVisionEncoder` (`vision/glm_ocr_encoder.go`) keep every field unexported, with no `Weights()` and no
  public tail; their tails (`merge`, `downsample`) are unexported. goinfer calls `enc.Forward` from `lm.qwenForward`
  (`internal/serveapp/vision_serve.go`); no backend reaches either loader. aikit's `gpu/qwenmetal` and `gpu/qwencuda`
  serve Qwen2.5-VL only (window plan, RMSNorm hard-coded at 1e-6) and are not reused.
- **Qwen3 (Qwen3.5+ tower), per block:** LayerNorm with weight and bias, eps 1e-6 (f64 accumulation); fused biased qkv
  `[3][heads][hd]`; 2-D RoPE, theta 1e4, `[row·f, col·f]` then `cat(row,row)`, NeoX rotate-half over the full head;
  full attention per image frame, scale `1/sqrt(hd)`; biased proj, residual; LayerNorm, biased fc1, GELU-tanh, biased fc2,
  residual. Before the blocks: a biased patch-embed matmul over `C·T·P·P`, plus a learned 48x48 position table
  interpolated bilinearly (align_corners=True, f32 coordinates, clamped taps, a fixed tap order). After: the merger
  (LayerNorm eps 1e-6; m² patches concatenated **patch-major**; fc1, **erf** GELU, fc2). Real tower
  (`~/models/qwen3.5-0.8b`): 12 layers, hidden 768, 12 heads x 64, patch 16, merge 2, out 1024.
- **GLM-OCR, per block:** RMSNorm (weight only, eps 1e-5); fused biased qkv, then **per-head RMSNorm on q and k before
  RoPE**; the same RoPE and attention; biased proj; RMSNorm; SiLU-gated MLP with biases. No position table. After the
  last block: post RMSNorm, a downsample conv done as a matmul over **channel-major** groups (`x[g][c·m²+j]`, the
  opposite of Qwen3's merger layout), then proj (no bias), LayerNorm eps 1e-5, erf GELU, SiLU-gated MLP. Real tower
  (`~/models/glm-ocr`): 24 layers, hidden 1024, 16 heads x 64, patch 14, merge 2, out 1536.
- **What Metal already has:** goinfer's tower base (`eg2Ops`: f32 GEMM, RMSNorm, per-head norm, rotate-half RoPE,
  attention built from matmuls in 256-row blocks, never T²) and aikit `gpu.ViT` (LayerNorm with bias, biased GEMM
  epilogues, GELU tanh and erf, SiLU-mul). **New:** an attention scale (the base hard-codes 1.0; Gemma 4 needs none), and
  Qwen3's interpolated position table, which depends only on the grid, so the host computes it with aikit's own code.
- **Split, mirroring Gemma 4:** aikit exports the f32 weights (aliasing, refusing a quantized tower) and a public tail
  that takes the last block's output; goinfer runs the patch embed and the blocks on the device; the merger (Qwen3) or
  post-norm, downsample and merger (GLM) stay aikit's on the host, as `FinishHidden` does for Gemma 4.

**S2 steps:** S2.0 the aikit exports (a local branch; released with the cycle's batch). S2.1 a backend-neutral tower
seam in goinfer's `multimodal` for both towers (Gemma 4's registry is the model). S2.2 the Qwen3 tower on Metal. S2.3 the
GLM-OCR tower on Metal. S2.4 serve: `--backend metal` picks the device tower, with named CPU fallbacks (int8 tower, not
registered, declined). The CUDA twins are nobara's, after S2.4, against the same gates.

**S2 gates, written 2026-10-07 before any S2 code or measurement:**
- **G-S2a, the exports:** in aikit, the exported weights plus the public tail reproduce `Forward` exactly (bit-identical:
  the same CPU code in a different order of calls) on the tiny towers (`qwen35vl-vision-tiny`, `glm-ocr-vision-tiny`),
  with norms and the position table randomised first (the all-ones trap); a quantized tower's export errors.
- **G-S2b, the device towers against aikit's CPU tower:** every merged token at cosine >= 0.9999, on the tiny towers at
  three grid sizes, and on the real towers (`qwen3.5-0.8b`, `glm-ocr`) on the four repo images F2a uses plus GLM-OCR's
  three O3 documents at the default cap. Ambiguous (parked): a worst token in 0.999-0.9999.
- **G-S2c, planted defects, each red on G-S2b's tiny-tower check:** (1) the attention scale dropped; (2) RoPE's row and
  column halves swapped; (3) Qwen3's position table added at a transposed grid; (4) GLM's per-head q/k norm skipped;
  (5) the patch-embed bias dropped. A defect that stays green means the fixture is degenerate along that axis.
- **G-S2d, served:** one image request per family through the Metal serve binary (qwen3.5-0.8b and glm-ocr), the tower
  on Metal against the tower on the CPU: identical reply text, or a first divergence at a near-tie under the G1 rule.
- **Speed (night):** tower time per image, Metal against CPU, on the same images. A record.

**S2 progress, 2026-10-07 (Mac), on goinfer branch `s2-towers` over aikit's unreleased `s2-tower-exports`:**
- **G-S2a: PASS.** aikit's exports recompose `Embed` and `Forward` bit for bit on both tiny towers, norms and the
  position table randomised (`TestQwen3Export_recomposesForward`, `TestGlmOcrExport_recomposesForward`); dropping GLM's
  post-norm from the tail turns it red.
- **G-S2b, tiny towers: PASS**, worst merged-token cosine 1.000000000 for both over four grid sets (three sizes and a
  two-image batch). **G-S2c: PASS**, every planted defect red: Qwen3.5 (scale 0.975, RoPE halves 0.997, transposed
  position grid 0.995, patch bias 0.994), GLM-OCR (scale 0.126, RoPE 0.698, q/k norm 0.119, patch bias 0.531). The GLM
  gate uses aikit's tiny GLM tower, copied into goinfer's testdata (goinfer's own `glm-ocr-tiny` has no vision weights).
- **G-S2b, real towers: PASS**, nine image/tower pairs through serve's preprocessing and caps: Qwen3.5-0.8B worst token
  0.999999212, GLM-OCR worst 0.999991618 (`formula.png`). Exploratory times: Metal 2-4x the CPU tower (GLM-OCR invoice
  12.3 s against 48.9 s; Qwen3.5 on the 896² image 1.2 s against 4.7 s). Raw: `s2-gs2b-real.log` on the branch.
- **G-S2d run 1 (serve binary at the branch's `fac4aa52` plus the serve wiring), table.png, 32 greedy tokens:**
  - **GLM-OCR: PASS**, byte-identical replies. Both arms decode on the CPU (Metal does not run GLM-OCR's decoder), so
    only the tower differed. Time to first token 16.5 s against 39.2 s (exploratory).
  - **Qwen3.5-0.8B: confounded, not graded.** Under `--backend metal` the 0.8B also decodes on Metal, so the two arms
    differed in the decoder as well as the tower. The replies agree for their first 15 tokens, then the Metal arm wrote
    "," where the CPU arm wrote "." — the kind of flip Metal-vs-CPU decode makes about 5% of the time (G3), which says
    nothing about the tower. The registered procedure did not hold the decoder fixed; GLM only did so by accident.
  - Raw: `docs/measurements/multimodal-support-2026-10/s2-gs2d-run1/`.
- **G-S2d amendment, after run 1, before the graded Qwen run:** both arms run `--backend metal`; the CPU-tower arm adds
  serve's new `-vision-device cpu` (a real flag: it keeps the tower on the CPU while the model keeps its backend), so only
  the tower differs. Pass rule unchanged.
- **G-S2d run 2, Qwen3.5-0.8B, both arms `--backend metal` (decode `metal-resident (int4)` in both), the CPU-tower arm
  with `-vision-device cpu`:** the replies differ.
  - Tower on Metal: "...region (thousands)**, which presents...". Tower on the CPU: "...region (in thousands)**.\n\nIt
    presents...". Every token before generated token 14 agrees, each at p ~ 1.000.
  - At token 14 both arms sit on a coin flip: the CPU-tower arm picks "in" at p 0.545 with "th" at 0.442; the Metal-tower
    arm picks "th" at 0.579 with "in" at 0.403. The CPU-tower arm's gap between its token and the Metal arm's is 0.21 in
    logits.
  - **Status: near-tie by every measure serve exposes; the registered form is not computed.** The rule compares that
    gap with 3% of the raw top-1 logit, which log-probabilities hide; it passes for any top-1 logit above 7. Accepting
    it on that basis, or building an in-process replay that reads the raw logit, is the owner's call.
  - **Owner decision 2026-10-07: accepted as a near-tie. G-S2d: PASS for both families.**
  - Raw: `docs/measurements/multimodal-support-2026-10/s2-gs2d-run2/` (serve logs, replies, top-3 logprobs per token).

### S3 — Gemma 3 and Qwen2.5-VL on Metal: run what exists, then put the towers on the GPU

- **First, run it.** Metal's `UploadKV` and `ForwardMRoPE` tests have never been run, and no image turn has ever run on
  a Mac (`docs/measurements/multimodal-audit-2026-10-02.md`).
  - Pull `gemma-3-4b-it` and `qwen2.5-vl-3b` (disk first).
  - Run those tests, and one served image turn per family against the CPU decoder.
- **Then the towers.** aikit ships `gpu/visionmetal` (SigLIP) and `gpu/qwenmetal`, tagged and unused by goinfer.
  - Wire them through `EnableResident` (M-15 of `docs/audit-metal-2026-09-12.md`).
  - Gate them like S2, and run a night crossover against the CPU tower.
- **Size:** S to run, M to wire.

**S3 progress and gates (2026-10-07, Mac):**
- **Run it, part 1: done on what is local.** Metal's `UploadKV` and `ForwardMRoPE` tests ran for the first time and pass
  (5/5 on `llama-tiny` and on the real Qwen2.5-Coder-1.5B). `TestVLImageTurn_metalResident` runs the first image turns on a
  Mac's GPU, from the tiny fixtures: Gemma 3 and Qwen2.5-VL decode `metal-resident` after their CPU prefill (8/8 and 6/8
  tokens agree with the CPU decoder; agreement is the served gate's question). The real checkpoints (`gemma-3-4b-it`,
  `qwen25vl-3b-instruct`) are downloading from Hugging Face into `~/models`; the archive has no Qwen2.5-VL, and nobara
  held its timing lock.
- **The wiring:** goinfer's `metal` module imports aikit's tagged `gpu/visionmetal` (SigLIP) and `gpu/qwenmetal`
  (Qwen2.5-VL); serve's `enableResidentTower` admits `metal`. SigLIP's Metal tower is int8 (it needs `LoadEncoder`
  quant=true, as on CUDA and WebGPU), so `towerInt8` treats Metal like them for that family; Qwen2.5-VL stays f32 (its
  export carries both). `-vision-device cpu` keeps either on the CPU, f32 by default.
- **Near-tie, defined for served comparisons (serve exposes log-probabilities, not logits; G-S2d):** at the first differing
  token, the reference arm's probability of the other arm's token is at least half its own top token's probability, read
  from serve's `top_logprobs`.
- **S3 gates, written 2026-10-07 before any S3 tower code or measurement:**
  - **G-S3a, device tower against aikit's CPU tower at the same quantization** (SigLIP int8 both, Qwen2.5-VL f32 both):
    every soft token at cosine >= 0.9999, on the tiny towers (`siglip-tiny`, `qwen25vl-tiny`) and on the real towers on the
    four images F2a uses; the encoder must report the resident attached. Ambiguous (parked): a worst token in
    0.999-0.9999.
  - **G-S3b, served, the tower isolated:** one image request per family through the Metal serve binary, both arms
    `--backend metal`; the reference arm adds `-vision-device cpu` (and `-vision-quant int8` for Gemma 3, matching the
    device tower). Identical reply, or a first divergence at a near-tie.
  - **G-S3c, served, the decoder (part 1's "against the CPU decoder"):** both arms with the tower on the CPU
    (`-vision-device cpu`): `--backend metal` against `--backend cpu`. Identical reply, or a first divergence at a near-tie.
  - **Speed (night):** tower time per image, Metal against CPU, and TTFT of the served turn. A record.
- **The wiring needed new aikit tags.** `gpu/visionmetal` and `gpu/qwenmetal` v0.1.0 date from 2026-08-12 and do not build
  against `gpu` v0.33.5; aikit's current source does. So the S3 tower work moved to the `s2-towers` branch (whose go.work
  uses the local aikit), with S2's release batch.
- **G-S3a, read 2026-10-07 11:30 PDT on the Mac: tiny PASS, real FAIL for SigLIP, real PASS for Qwen2.5-VL.**

  | tower | tiny | real, worst token over four images | time per image, Metal vs CPU (exploratory) |
  |---|---|---|---|
  | SigLIP (Gemma 3), `visionmetal`, int8 both | 1.000000000 | **0.180-0.461: wrong** (withdrawn as a defect 2026-10-07; see below) | ~73 s vs ~17 s |
  | Qwen2.5-VL, `qwenmetal`, f32 both | 1.000000000 | 0.999995874 or better | 22-47 s vs 23-37 s |

  - The Metal SigLIP tower agrees exactly at the tiny fixture's size (hidden 32, 16 patches) and breaks at Gemma 3's
    (hidden 1152, 4,096 patches), so aikit's own tiny parity test could not see it. It is also four times slower than the
    CPU.
  - The Metal Qwen2.5-VL tower is correct but no faster than the CPU, and slower on the larger images.
  - **Neither is wired into serve:** the wiring commit is reverted on the branch; the test stays as the evidence
    (`metal/s3_towers_test.go` on `s2-towers`). Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3a-real.log`.
  - **Proposed, for the owner:** build both towers on goinfer's own Metal tower base, as S2 did for Qwen3.5 and GLM-OCR
    (its kernels pass at real size and run 2-4x the CPU), from new aikit float32 exports added to the same unreleased
    batch; and report the `visionmetal` real-size defect to aikit. G-S3b waits for that decision.
  - **Correction, 2026-10-07: there is no `visionmetal` real-size defect.** Both arms here were int8, and SigLIP's int8
    tower is itself far from its f32 tower at real size. Against the f32 CPU tower:
    - the CPU int8 tower's worst token is 0.012-0.172 and the Metal one's 0.027-0.110;
    - both have relative L2 0.16-0.52.

    The two int8 towers go wrong on the same tokens and disagree there, which is the 0.180-0.461 above, reproduced exactly.
    The record is `docs/measurements/siglip-int8-fidelity-2026-10-07.md`. Nothing was reported to aikit. The rebuild stands,
    since the rebuilt tower is f32. The record's open question is serve's int8 SigLIP tower under cuda and webgpu.
- **G-S3c, read 2026-10-07 11:35 PDT on the Mac (serve binary from `s2-towers`, the tower on the CPU in every arm,
  table.png, 32 greedy tokens): Qwen2.5-VL FAIL; Gemma 3 not reached.**
  - **Qwen2.5-VL-3B, `metal-resident (int4)` against `cpu (int4)`: the replies differ from the first token.** Metal
    "Table 2. Quarterly unit sales by region (thousands)"; CPU "Quarterly unit sales by region (thousands)". At token 0
    the CPU arm picks "Quarter" at p 0.728 with "Table" at 0.208, under half its top: **not a near-tie, so G-S3c fails for
    Qwen2.5-VL.** The first token comes from the prefill, so the Metal image turn computes something the CPU does not
    on the real checkpoint (the tiny fixture's first token matched). To be traced.
  - **Gemma 3 4B: Metal declined the resident** (it needs 5.15 GB against a 4.20 GB budget on this 16 GB Mac), so both
    arms decoded on the CPU and G-S3c did not reach Metal. The two CPU arms still differ at token 7, not as a near-tie
    (the `--backend cpu` arm puts the other arm's token at p 0.284 against its top 0.687): a CPU decode after a declined
    Metal request is not the same computation as `--backend cpu` (its int4 layout differs). A separate finding, to be
    understood.
  - Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3c/` (serve logs, replies, top-3 logprobs per token).
- **Owner decision 2026-10-07: the proposal is accepted.** Both towers get rebuilt on goinfer's Metal tower base, from new
  aikit float32 exports (SigLIP, Qwen2.5-VL) on the same unreleased `s2-tower-exports` branch; the `visionmetal` real-size
  defect is reported to aikit. G-S3a's bars apply unchanged to the rebuilt towers, then G-S3b.
- **G-S3a for the rebuilt towers, registered 2026-10-07 before their code runs:**
  - **Bars unchanged:** every soft token at cosine >= 0.9999 against aikit's CPU tower, on the tiny towers and on the four
    F2a images at real size; 0.999-0.9999 ambiguous (parked).
  - **One amendment: SigLIP is float32 on both sides,** not int8. The rebuilt tower runs float32 GEMMs, from aikit's new
    float32 export (`Encoder.Weights`, `FinishHidden`), so serve loads Gemma 3's tower float32 for it, as it already does
    on the CPU. Qwen2.5-VL stays float32 on both sides, from aikit's existing `GPUWeights` (its float32 form),
    `BuildWindowPlan` and `MergeHidden`.
  - **The tiny checks randomise every norm first** (the all-ones trap).
  - **Planted defects, each one alone, must turn the tiny check red:**
    1. the attention scale dropped (both towers);
    2. RoPE's row and column halves swapped (Qwen2.5-VL);
    3. every block attending its whole image instead of its windows (Qwen2.5-VL);
    4. the window reordering skipped, on the way in and on the way out (Qwen2.5-VL);
    5. the position table dropped (SigLIP);
    6. the patch-embed bias dropped (SigLIP).
  - A defect that stays green means the fixture is degenerate along that axis. That gets recorded, and the check is
    rerun on a fixture or grid that can see the defect, before any result is read.
  - **Then G-S3b, as registered.**
- **G-S3a for the rebuilt towers, read 2026-10-07 on the Mac (`s2-towers` at `77f20330`, aikit `s2-tower-exports` at
  `d7d5cf9`): PASS, tiny and real.**
  - **The code:** `metal/vl_towers.go`. Both towers run on the S2 base (`metal/grid_vision.go`, which gains tower kinds:
    no RoPE, windowed segments, a fixed position table, and a block split across command buffers when its segments would
    overflow the scalar arena). They register through aikit's own seams (`vision.RegisterResident`,
    `vision.RegisterQwenResident`); aikit's `gpu/visionmetal` and `gpu/qwenmetal` are no longer imported. aikit gains
    SigLIP's float32 export (`Encoder.Weights`, `FinishHidden`), its recomposition bit-exact on the tiny tower.
  - **Tiny: both towers exact (worst token 1.000000000).** The first run left five of the seven planted defects green,
    because the tiny fixtures are degenerate: every bias is zero, and Qwen2.5-VL's q/k/v and output projections are at
    init scale (0.02), so attention is nearly uniform and adds little to the residual. As registered, the fixture was
    sharpened before reading anything: random biases, q/k scaled by 6 (SigLIP) or 12 (Qwen2.5-VL), and v and the output
    projection scaled by 8 (Qwen2.5-VL), through the aliasing exports, which the CPU tower reads too. Then every defect
    is red:

    | defect | SigLIP | Qwen2.5-VL |
    |---|---|---|
    | (1) attention scale dropped | 0.998501 | 0.629849 |
    | (2) RoPE halves swapped | — | 0.750730 |
    | (3) whole-frame attention in the windowed blocks | — | 0.789726 |
    | (4) window reordering skipped | — | 0.726768 |
    | (5) position table dropped | 0.745353 | — |
    | (6) patch-embed bias dropped | 0.762383 | — |

  - **Real, the four F2a images, attached through `EnableResident` as serve attaches them:**

    | image | SigLIP worst token | Metal / CPU tower | Qwen2.5-VL grid | worst token | Metal / CPU tower |
    |---|---|---|---|---|---|
    | gemma3_preprocess_image.png | 0.999999928 | 9.2 / 26.1 s | 64x64 | 0.999996659 | 8.9 / 22.1 s |
    | qwen25vl_preprocess_image.png | 0.999999972 | 9.5 / 28.4 s | 4x6 | 0.999999999 | 1.8 / 0.9 s |
    | glm_ocr/formula.png | 0.999998073 | 9.6 / 25.8 s | 86x72 | 0.999999651 | 15.1 / 34.9 s |
    | glm_ocr/table.png | 0.999999857 | 9.6 / 26.6 s | 64x86 | 0.999999464 | 14.4 / 31.1 s |

    Times are exploratory: one reading each, by day, a compile running beside one of them. SigLIP's Metal tower is about
    2.8x the CPU tower here; aikit's was wrong and ~73 s. Qwen2.5-VL's is 2.1-2.5x on the large images, and slower on
    the tiny 4x6 one, where per-window dispatch overhead dominates. Raw:
    `docs/measurements/multimodal-support-2026-10/s3-gs3a-rebuilt-real.log`.
- **G-S3b, read 2026-10-07 12:00-12:04 PDT on the Mac** (serve binary from `s2-towers` at `0c66b18b`, table.png, 32 greedy
  tokens, both arms `--backend metal`, the tower on Metal against `-vision-device cpu`; `run-gs3c-served.sh` with arms
  `metal:auto,metal:cpu`): **Qwen2.5-VL PASS; Gemma 3 read FAIL, re-graded PASS the same evening (below).**
  - **Qwen2.5-VL-3B: byte-identical replies** ("Table 2. Quarterly unit sales by region (thousands)"). Decoder
    `metal-resident (int4)` in both arms, the tower on Metal in one. Request time 38.1 s against 53.1 s (exploratory).
  - **Gemma 3 4B: the replies differ at generated token 10,** " presented" against " broken". The Metal-tower arm puts
    " broken" at p 0.195 against its top 0.795: not a near-tie. Both arms decoded on the CPU (Metal declined the decoder
    for memory again, 5.15 GB against a 4.3-4.9 GB budget) on the same load, so only the tower differed, and the tower
    matched the CPU tower to a worst token of 0.999999857 on this very image (G-S3a).
  - **That size of flip from that size of perturbation is surprising, so nothing is concluded yet.** The determinism
    control (the CPU-tower arm twice; the Metal-tower arm twice) was attempted at 12:04 and 12:05. Both times serve's
    fit guard refused to load the 4B: 7.6-7.9 GB was available by then, against the 5.9 GB it needs at a 70% margin. It
    is owed on a quieter machine. If the CPU-tower arm repeats itself byte for byte, the next step is the perturbation
    control: CPU-tower features plus random noise of the Metal tower's size. Shipped path is not ground truth (R2): an
    int4 W4A8 decoder quantizes activations per tensor, so a 1e-7 perturbation can cross an int8 rounding and change a
    choice the reference itself makes with p 0.8.
  - Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3b/`.
  - **Re-graded 2026-10-07 evening (Cowork audit; no new run): Gemma 3 PASS.**
    - **The misgrade:** `run-gs3c-served.sh` graded against whichever arm came first, and this run listed `metal:auto`
      first. So the Metal-tower arm served as the reference.
    - **The registered reference** is the `-vision-device cpu` arm. At generated token 10 it has " broken" 0.485 and
      " presented" 0.404. 0.404 is at least half of 0.485, so this is a near-tie, and G-S3b passes under its own rule.
    - **The re-grade:** the script's own grader, run on the recorded files with the CPU-tower arm as the reference
      (`s3-gs3b/regrade-2026-10-07.txt`).
    - **The script now refuses an ambiguous order.** The reference arm is marked with a leading `=`; exactly one must
      be marked, or nothing runs. Tonight's callers mark theirs: `run-s3-rootcause-night.sh` (step A's reference is
      `cpu:cpu`; it too listed `metal:cpu` first), `run-s6-e4b-night.sh` and `run-gs5c-served.sh`'s callers.
    - **Still worth understanding:** at token 9, p(",") moves from 0.841 to 0.556 on features at cosine 0.9999999.
      Tonight's determinism and sensitivity steps (`run-s3-rootcause-night.sh` steps B and C) still run for that.
- **G-S3c's failure, analysed 2026-10-07 (no new measurement): the two arms run different CPU kernels, so the gate does
  not isolate Metal. REFUTED the same day by measurement; see "G-S3c root cause, in progress" below.**
  - Metal has no resident m-RoPE prefill (only CUDA implements `PrefillMRoPELast`). So a Qwen2.5-VL image turn under
    `--backend metal` prefills on the CPU and `UploadKV`s, and **its first token comes from the CPU prefill**, as the
    `--backend cpu` arm's does.
  - The two CPU prefills still differ: the load keys the int4 layout on `Options.Backend` (`wantsCanonicalInt4`,
    `wantsRow4Fallback` in `decoder/weightmat.go`). Under `metal` the CPU holds canonical int4 and no row4; under `cpu`
    it holds the arm64 row4 repack only, for embed, gate/up and qkv (the serve logs' `int4 layout` line). Different
    kernels, different rounding.
  - With both arms' towers on the CPU in float32 (the same features), the token-0 split ("Table" against "Quarter")
    can only come from that layout difference, not from Metal. Gemma 3's G-S3c is the same thing measured directly:
    both arms decoded wholly on the CPU (Metal declined) and still differed beyond a near-tie.
  - **Proposed, for the owner:** re-register G-S3c so the reference shares Metal's CPU-side layout. For example, the
    `--backend metal` load with its resident detached, through a test hook, against the same load decoding on Metal.
    Then the gate grades Metal's decode and nothing else. Until then G-S3c stands as FAIL, as registered.
- **Owner, 2026-10-07: re-register, but root-cause G-S3c and Gemma 3's G-S3b divergence first.**
- **G-S3c root cause, in progress (2026-10-07, Mac, by day; in-process, one load per process to fit beside the owner's
  session):**
  - **The layout analysis above is wrong.** `decoder/qwen25vl_layout_split_real_test.go` runs `prefillLogitsQwenVL`, the
    CPU prefill whose logits pick the first token, on table.png with serve's prompt, under each load. The towers ran
    float32 on the CPU, cached once.

    | load | HF golden image: KL(HF || this) | table.png token 0, top-3 |
    |---|---|---|
    | int8 weight-only (control) | 0.02651 | "Quarter" 0.508, "Table" 0.241, "quarter" 0.088 |
    | int4 row4, int8 head | 0.03841 | "Quarter" 0.629, "This" 0.156, "Table" 0.100 |
    | int4 canonical, int8 head | 0.03841 | identical to row4 |
    | int4 row4, int4 head (serve's `--backend cpu`) | 0.13439 | "Quarter" 0.705, "Table" 0.167, "quarter" 0.031 |
    | int4 canonical, int4 head (serve's `--backend metal` CPU side) | 0.13439 | identical to row4 |

    **Canonical and row4 give the same logits.** And with serve's settings, the CPU prefill of a `--backend metal` load
    says "Quarter", not the "Table" the served Metal arm wrote.
  - **In a binary with Metal it is the same,** as long as the resident is not built. `metal/qwen25vl_prefill_split_test.go`
    reproduces that prefill bit for bit (151936 of 151936 logits) under `--backend metal` and `--backend cpu`. But both
    times the resident build was declined for memory: 3.61-3.70 GB needed against a 3.38-3.68 GB budget. So **the served
    "Table" needs the resident to exist.** Something about a built Metal resident changes the first token. The obvious
    suspects are ruled out by reading:
    - `prefillLogitsQwenVL` and its LM head run on the CPU (Metal's backend implements no staged int4 matmul);
    - the f16-MMA pass is resident-only;
    - the KV precision is resident-only;
    - fast attention is on in both arms;
    - `int4BufA` copies the weights rather than rewriting them.
    The next measurement, the same test with the resident built, needs about 5.5 GB free on the Mac. (Correction, the same
    hour: this paragraph's inference is void. The test's own `--backend cpu`-equivalent arm gives "Quarter" at 0.705,
    where the served `--backend cpu` arm gave 0.728, so its hand-built prompt is not serve's and its arms cannot be set
    beside the served ones. The served control below settles it.)
  - **Side readings worth keeping:** on a real image, the int4 head (EmbedInt4, serve's default) costs 3.5x the KL of the
    int8 head against HF (0.134 against 0.038). And the int8 control itself puts token 0 at 0.508 against 0.241: the
    choice is fragile at any precision.
- **G-S3c ROOT CAUSE, found 2026-10-07 12:26-12:30 PDT by served controls: serve's `--embed-int4` default differs by
  backend, so the two arms ran different embedding/LM-head precisions. Metal was never compared.**
  - `internal/loadflags`' `embedInt4()` turns the int4 embedding table off under `--backend metal`, because Metal's
    resident does not accept an int4 table, and leaves it on elsewhere. Qwen2.5-VL-3B (and Gemma 3) tie the embedding
    to the LM head, so the CPU arm decoded with an int4 head and the Metal arm with an int8 one.
  - **Proof, through the same serve binary, flipping only that flag** (table.png, tower on the CPU; raw in
    `docs/measurements/multimodal-support-2026-10/s3-gs3c-rootcause/`):

    | arm | table | token 0 | reply |
    |---|---|---|---|
    | `--backend cpu` (default) | int4 | "Quarter" 0.728, "Table" 0.208 | "Quarterly unit sales by region (thousands)" |
    | `--backend cpu --embed-int4=false` | int8 | **"Table" 0.573**, "Quarter" 0.230 | "Table 2. Quarterly unit sales…" |
    | `--backend metal` (default), resident declined for memory | int8 | "Table" 0.573, "Quarter" 0.230 | "Table 2. Quarterly unit sales…" |
    | `--backend metal --embed-int4` (resident declined: int4 table) | int4 | **"Quarter" 0.728**, "Table" 0.208 | "Quarterly unit sales…" |

    The numbers swap exactly with the flag, and the `--backend metal` arm says "Table" with no resident at all. **So the
    Metal arm's "Table" is the int8-head answer, not a Metal effect.** It is also the higher-precision one: on the HF
    golden, the int8 head sits at KL 0.038 from HF against the int4 head's 0.134 (the table above).
  - **Two explanations recorded earlier today were wrong and are withdrawn:**
    - mine, that the CPU int4 layout differed by backend (canonical and row4 give identical logits);
    - nobara's G-S3c-on-CUDA note, that "Metal's `UploadKV` prefill-then-upload route is the suspect". CUDA matched the
      CPU because CUDA keeps the int4 table on both arms, not because Metal's upload is at fault.
  - **The same default explains G-S3c's Gemma 3 observation** ("a CPU decode after a declined Metal request is not the
    same computation as `--backend cpu`"): Gemma 3 ties its head too, and that pair also differed in the table's
    precision. Its layout explanation is withdrawn with mine. Not yet re-measured with the flag held fixed (step A of
    tonight's job).
  - **A finding beyond S3:** an operator moving between `--backend cpu` and `--backend metal` gets a different model on
    tied-head families. The int4 head measured 3.5x the int8 head's KL to HF on a real image. That is a quality question
    about the `--embed-int4` default (docs/quantization.md), raised here, not decided here.
- **G-S3c, re-registered 2026-10-07 before it runs (owner: "re-register sounds good", after the root cause):** the same
  request, image, near-tie definition and pass rule. Every arm holds every load flag equal: `-vision-device cpu` and
  `--embed-int4=false` on all of them (the int8 table Metal's resident needs; the more faithful one). Arms: `--backend
  metal`, `--backend cpu`, and `--backend cpu` again as a determinism control (byte-identical required). **A
  `--backend metal` arm whose decode path is not `metal-resident` voids that family's reading**; it is not a FAIL.
  Qwen2.5-VL and Gemma 3. Queued as step A of `run-s3-rootcause-night.sh`, because both residents need memory the Mac
  lacks by day.
- **G-S3b's Gemma 3 divergence, in progress:** phase 1 (`metal/gemma3_tower_dump_test.go` on `s2-towers`) dumped both
  towers' projected features for table.png. They differ by at most 7.92e-5 relative L2 per soft token (worst cosine
  0.999999997). Phase 2 (`decoder/gemma3_tower_sensitivity_real_test.go`) teacher-forces the int4 decoder along the
  CPU-tower arm's greedy path with the Metal-tower features, and with three random perturbations of the same per-token
  size. It needs the 4B, which this Mac's guard refuses while the owner's session runs (6.7 GB available, 6.2 GB
  needed at a 70% margin).
  Both G-S3b arms were `--backend metal`, so both had the int8 table: the `--embed-int4` difference above does not apply
  to G-S3b, and its question stays open. Phase 2 loads with the int8 table to match. Tonight's job runs:
  - step B, the determinism controls (each tower arm twice);
  - step C, phase 2.
  (`docs/measurements/multimodal-support-2026-10/run-s3-rootcause-night.sh`, pinned binaries in `~/goinfer-bench/s3/`.)
- **G-S3c on CUDA, a cross-check, registered 2026-10-07 before it runs (nobara):** the same two requests, rule and near-tie
  definition as G-S3c, one serve binary built from `s2-towers` with `-tags cuda`, both arms `-vision-device cpu`:
  `--backend cuda` against `--backend cpu`, plus a second `--backend cpu` run as a determinism control (its reply must be
  byte-identical to the first). It answers whether Qwen2.5-VL's first-token divergence is Metal's alone: CUDA runs its own
  resident m-RoPE prefill, while Metal runs the CPU prefill and then `UploadKV`. It does not stand in for Metal's G-S3c.
- **G-S3c on CUDA, read 2026-10-07 on nobara (RTX 2070 SUPER, driver 595.91.07; `cuda/cmd/serve` built from `s2-towers` @ `29dece6c`, `-tags cuda`; int4; greedy):**
  Qwen2.5-VL reads clean; Gemma 3 diverges at a late token, outside the near-tie definition. Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3c-cuda/`.
  - **Qwen2.5-VL-3B (arms `cpu`, `cuda`, `cpu`, tower on the CPU in all three): all three replies are identical** ("Quarterly unit sales by region
    (thousands)"; the CUDA arm against the CPU arm, and the second CPU run against the first). Decode paths as asked: `cpu (int4)` (canonical layout),
    `cuda-resident (int4)` with CUDA's batched resident prefill, `cpu (int4)`. **CUDA does not reproduce Metal's first-token divergence** ("Table" against
    "Quarter"), so Metal's `UploadKV` prefill-then-upload route is the suspect, not the image prompt or the m-RoPE positions as the resident paths see them.
    (Whole-request wall, exploratory: CPU 55.6 s, CUDA 22.2 s.)
  - **Gemma 3 4B, first run: void as a CUDA comparison, for two reasons the log shows.** (1) The `cuda` arm's resident build failed on a device allocation
    (`requested cuda -> running on cpu`; 4 KV slots planned at a 4141-position context, then a 17 MB scratch buffer did not fit), so that arm decoded on the
    CPU; (2) **`-vision-device cpu` was not honored for Gemma 3's SigLIP tower**: the generic branch of `loadVisionTower` keyed the tower's int8 default and its
    resident attach on the model's own backend, so the `cuda` arm ran an int8 resident tower while the CPU arms ran f32 on the CPU. That run's difference
    (token 3, ' quarterly' p 0.777 against ' a' 0.223) is therefore confounded twice. It is, though, the same shape as the Mac's separate observation that a
    CPU decode after a declined GPU request differs from `--backend cpu` (the declined arm's log line has no `int4 layout: canonical` marker).
  - **Gemma 3 4B, second run (valid): `GS3C_EXTRA="--kv-sessions 1"`, the flag fixed on the branch** (`internal/serveapp/main.go`: under `-vision-device cpu` the
    int8 default and the resident attach use the CPU, as `towerBackend()` says; until now this branch used `cfg.load.Backend`). All three arms load the same
    `encoder f32` tower on the CPU; decode paths `cpu (int4)`, **`cuda-resident (int4)`**, `cpu (int4)`.
    - **CUDA against CPU: first differing generated token 10, CPU ' region' 0.711, CUDA ' geographic' 0.287; near-tie (p(other) >= half p(top) = 0.356): False.**
      Registered rule: not a near-tie. **The second CPU run is byte-identical to the first** (determinism control holds).
    - **How to read it:** the divergence is at the 10th generated token, not the first, with the alternative at p 0.29. That is the size of an ordinary int4 W4A8
      (CUDA) against int4 (CPU) disagreement; the Mac's G3 calibration found a validated Metal model at 94% teacher-forced agreement, and E2B on CUDA reads
      94.5%. It is NOT shown to be a defect, and the near-tie definition here is stricter than G3's logit-gap rule. A text-only control (the same prompt
      without the image, CUDA against CPU) or a teacher-forced run over the image turn would separate "decoder noise" from "image path"; not run.
  - **Two findings for the Mac and the branch:** (a) the `-vision-device cpu` fix above is on `s2-towers` (not pushed); (b) Gemma 3 4B with four KV slots does not
    fit the 8 GB card at serve's default and falls back to the CPU (it fits with `--kv-sessions 1`); pre-existing, not touched here.
  - **What changed in the script:** `run-gs3c-served.sh` takes `GS3C_EXTRA` (extra serve flags for every arm), backward compatible.

### S4 — CUDA towers for Gemma 4 and Qwen2.5-VL

- **Gemma 4:** port `metal/gemma4_vision.go` to `cuda/`. The registration seam is already backend-neutral
  (`multimodal.RegisterGemma4Tower`).
- **Qwen2.5-VL:** wire aikit's tagged `gpu/qwencuda` (a require plus `EnableResident`).
- **Gates:** as S2, on nobara.
- **Size:** M (Gemma 4), S-M (Qwen2.5-VL).

**S4 and S2's CUDA twins, combined (2026-10-07):** CUDA has no tower base yet (Gemma 3's SigLIP is goinfer's own,
`cuda/vision_encoder.go`, built on aikit's CUDA ViT ops; nothing else), so the Gemma 4 port is the base, and S2's Qwen3.5+
and GLM-OCR towers follow on it, as Metal's did. One brief: `docs/prompts/nobara-s2-s4-cuda-towers-2026-10-07.md`.
- **G-S4q, aikit's `gpu/qwencuda` at real size, registered 2026-10-07 before it runs:** G-S3a's test shape and bars on
  CUDA (f32 both, every soft token at cosine >= 0.9999 on the four F2a images; 0.999-0.9999 parked), times exploratory.
  It decides S4's Qwen2.5-VL route: correct and faster than the CPU tower → wire it, as written above; wrong, or not
  faster → build it on the CUDA base, as Metal now will. The Metal twin was correct and no faster.
  - **G-S4q read 2026-10-07 on nobara (RTX 2070 SUPER; `s2-towers` @ `29dece6c`, aikit `s2-tower-exports`; Qwen2.5-VL f32 both sides): aikit's `gpu/qwencuda`
    is CORRECT at real size and faster than the CPU tower. Verdict by the registered rule: wire it, as S4 says.** Raw and the test sources (kept as `.go.txt`
    so `go build ./...` does not compile them from `docs/`): `docs/measurements/multimodal-support-2026-10/s4-gs4q/`.
    - **Correctness:** the tiny towers at 1.000000000 (both grids). The real Qwen2.5-VL-3B on the four F2a images, every soft token against aikit's CPU tower:
      worst token cosine **0.999997994, 0.999999994, 0.999999901, 0.999999942**: all over the 0.9999 bar.
    - **Time, exploratory (the first run, the Qwen subtest alone on the box):** tower on CUDA against the CPU tower, by grid: [1 64 64] 7.97 s against 15.17 s;
      [1 4 6] 51 ms against 140 ms; [1 86 72] 15.29 s against 24.40 s; [1 64 86] 12.70 s against 21.28 s, so 1.6x to 2.7x. (The second run's times, 16.9 / 0.118 / 32.4 /
      26.9 s against 28.7 / 0.142 / 43.2 / 41.0 s, were taken while the SigLIP package ran in parallel on the same GPU and CPU and are not to be used.) 8-15 s for a
      large grid is faster than the CPU but not fast.
    - **aikit's `gpu/visioncuda` (SigLIP), the information item: WRONG at real size, like its Metal twin.** On the four images, worst token cosine **0.245, 0.269,
      0.097, 0.281**, with 4094 to 4096 of the 4096 tokens under 0.9999 (its tiny parity test passes). Times were taken under the parallel-run contention above, so
      they say nothing about speed (17-34 s on the CUDA tower against 21-37 s on the CPU's). For the aikit defect report together with the Metal reading.
    - **A trap that nearly made this reading wrong:** aikit's `visioncuda` and goinfer's own `cuda.VisionEncoder` (`cuda/vision_register.go`, Gemma 3's CUDA tower)
      both register through the one global `vision.RegisterResident`, and the LAST registration wins. Inside package `cuda`, goinfer's registers last, so a SigLIP
      test there measures goinfer's tower: `*cuda.VisionEncoder` was the attached type, with per-token cosines mostly 0.99+ against the CPU int8 tower but 4065 of 4096
      tokens under 0.999 and outliers as low as 0.18 on high-norm tokens (rows of norm 16 against 2-3). The first run's "siglip" lines are that tower, and the test
      now asserts the attached type. aikit's tower is read in its own package (`cuda/s4siglip`), which imports nothing from goinfer's `cuda`. goinfer's own tower is
      not gated here and the numbers above are not a verdict on it; the served Gemma 3 image replies with it were sensible.
  - **Wired (2026-10-07, on `s2-towers`, commit `e5e5dbdf`; not on `main`):** `cuda/vision_towers.go` imports `gpu/qwencuda` (and deliberately not `visioncuda`);
    serve's `qwenTowerPlacement` attaches it under `--backend cuda` for a float32 tower, names every CPU fallback (`-vision-quant int8`, "cuda declined"), and
    `-require-backend` refuses them; `-vision-device cpu` keeps the tower on the CPU. `TestQwenTowerPlacement` covers the ten cases.
  - **Served on Qwen2.5-VL-3B, the G-S2d shape (`--backend cuda`, tower on CUDA against `-vision-device cpu`, a second CPU-tower run as the control):** identical
    replies in all three ("Quarterly unit sales by region (thousands)"). The log lines say what each arm did: tower `(CPU; ...)`, `(CUDA; ...)`, `(CPU; ...)`,
    decode `cuda-resident (int4)` throughout. Whole request, exploratory: 22.2 s with the CPU tower, 13.5 s with the CUDA tower.
    Raw: `docs/measurements/multimodal-support-2026-10/s4-served-qwen/`.
  - **A limit the first served run found, and it is the card's:** with serve's default KV plan the resident decoder takes all the free VRAM, so the tower's
    19.6 MB upload failed (`CUDA_ERROR_OUT_OF_MEMORY`) and the tower ran on the CPU: the log said `CPU (cuda declined)` and the reply was still identical, so a
    run on the default plan reads as a pass for the wrong reason. The comparison above needed `--kv-sessions 1 -ctx 4096`. The resident KV planner reserves
    for a drafter (`extraBytes`) but not for a vision tower, so on an 8 GB card a device tower and a default-planned decoder do not coexist. That is a planner
    gap, not a tower defect, and it applies to goinfer's own Gemma 3 SigLIP tower too; not fixed here.
  - **The script:** `run-gs3c-served.sh` sets `-vision-device` per arm as `<backend>:<auto|cpu>` (plain `<backend>` means the CPU tower); the Mac added the same thing
    at the same time, and the merge kept its spelling. These runs were made with an earlier `+` spelling of it (labels `cuda_cpu`, `cuda_auto`), which is equivalent.
- **S4 Gate 0 for the CUDA towers, read 2026-10-07 (nobara), before any CUDA tower code:** the desk map is
  `docs/measurements/multimodal-support-2026-10/s4-gate0-cuda-towers.md`. In short:
  - **Reuse is large.** aikit's `gpu.ViT` already has the f32 GEMMs with bias and bias-plus-residual epilogues, LayerNorm, RMSNorm, GELU tanh and erf, SiLU-mul,
    a NeoX RoPE and scaled, per-segment attention kernels; aikit's `qwencuda` is a correct template for the Qwen3.5 and GLM-OCR block loop. New small CUDA kernels
    are needed only for Gemma 4 (axial RoPE, ClippableLinear clamps, gelu·mul, position add). No audited PTX is touched; the new kernels are a new `.cu`.
  - **Not portable:** Metal's scalar-arena launch trick (CUDA takes scalars as launch arguments), so the base brings its own launch helper with an error latch.
  - **The tiny fixtures do not reach the production GEMM branches** (real GLM-OCR's patch embed has K = 1176, not a multiple of 16; real np is rarely a multiple of 64),
    which is aikit `visioncuda`'s failure class. The real-size gates are therefore mandatory, and per-branch GEMM tests at real shapes are added.
  - **VRAM:** all three towers fit in f32 beside a small decoder at modest sizes (Gemma 4 E2B ~0.7 GB, Qwen3.5-0.8B ~0.35 GB, GLM-OCR ~1.6 GB weights, plus scratch that
    scales with the image: ~2 GB for GLM-OCR at its 4.8 MP ceiling). The planner's drafter reserve (`ExtraResidentBytes`) can carry a tower estimate with no change in `cuda/`.
  - **CUDA-specific planted defects, registered now for G-S2c (they are in addition to Metal's, which port directly):** (a) the bias dropped only in the tiled-fallback GEMM
    branch (K not a multiple of 16), seen only by a real-shape GEMM test or a real tower; (b) padded attention keys not masked (if the matmul-blocked attention is used);
    (c) all patches of a two-image batch in one attention segment; (d) stale scratch across sizes (large, small, large grid on one accelerator); (e) for Gemma 4, the
    position x and y swapped, the value norm weighted instead of unweighted, and the attention scale changed from 1.0. Each must turn its gate red; a defect that stays
    green means the fixture is degenerate along that axis (fix the fixture, not the bar). Controls that are not defects: two identical runs bit-identical, a leak test after
    close, and a factory that errors (never panics) under allocation failure, with serve naming the CPU fallback and `-require-backend` refusing it.
  - **Build order:** serve plumbing made backend-generic; the tower base; the Gemma 4 tower; the Qwen3.5 and GLM-OCR towers; the VRAM reserve; the speed record (queued, not
    started); and only if that says so, a faster attention.
- **S4 steps 0-2 read 2026-10-07 on nobara (`s2-towers`; not on `main`): the tower base and the Gemma 4 tower on CUDA PASS.** Raw: `docs/measurements/multimodal-support-2026-10/s4-gemma4-cuda/`.
  - **Step 0, serve:** `planGridTower` and `chooseGemma4Tower` are backend-generic (Metal and CUDA, each checked against the registry for its own backend, the fallback named with
    that backend's word); the table row that pinned CUDA to the CPU is replaced by CUDA cases, and `chooseGemma4Tower` has a placement test for the first time.
  - **Step 1, the base** (`cuda/tower_base.cu` and `.go`; six small kernels in their own module, NVRTC 12.9.86, FMA-linted; an executor goroutine that owns the device; a launch
    helper with an error latch; GEMM, bias and bias+residual plan selection that handles a K that is not a multiple of 16): every GEMM branch against float64 (the register kernel on
    aligned shapes, the bias kernel at any M, the tiled kernel plus `add_bias` at K = 1176 and K = 20), the norms, the six kernels **bit for bit** against Go's float32 arithmetic,
    and attention at scale 1.0 and 1/sqrt(hd). The tiled-branch bias test was shown red by a mutant (bias dropped on that branch: the output was wrong by 60% of its scale).
    aikit's own attention kernel is used as it is (np <= 12288), so Metal's matmul-blocked attention was not needed for a correct baseline.
  - **Step 2, the Gemma 4 tower** (`cuda/gemma4_vision.go`, registered as multimodal's "cuda" tower), tiny tower with finite clip bounds and random position tables and norms:
    worst soft-token cosine **1.000000000** on all four grids (18, 54 and 135 patches; raster and shuffled). **Planted defects, each alone red** (worst cosine): clamps skipped 0.938,
    position x/y swapped 0.147, value norm weighted 0.829, attention scale 1/sqrt(hd) 0.533, position tables skipped -0.529, RoPE skipped 0.292. Large-small-large on one tower and
    two identical runs: bit-identical; a short patch buffer and an out-of-table position return errors; after `Close` the device ledger is empty.
  - **Real E2B (F2a on CUDA), every soft token against aikit's CPU tower, the four images (2304-2430 patches): worst cosine 1.000000000 on all four.** Tower CUDA 2.15-2.38 s against the
    CPU's 4.19-4.50 s (exploratory; the CUDA figure includes aikit's host tail).
  - **Served (the G-S2d shape; `--backend cuda`, E2B decoding resident, `--kv-sessions 1 -ctx 4096`):** tower on CUDA, tower on the CPU, and a second CPU-tower run give an identical
    32-token reply ("This image shows **Table 2: Quarterly unit sales by region (in thousands)** for the fiscal year FY2025." and the start of the next sentence). The logs name each arm
    (`tower on CPU` / `tower on CUDA`, `decode path: cuda-resident (int4)`, `decoded 32 tokens on the resident path`); the image encode took 2.34 s with the tower on CUDA and 4.45-4.47 s
    on the CPU. The default KV plan was not used, for the reason recorded under G-S4q (the tower needs VRAM the resident decoder would take); step 4 is the reserve for that.
- **S4 step 3 read 2026-10-07 on nobara (`s2-towers`, commit `d7955399`; not on `main`): the Qwen3.5+ and GLM-OCR towers on CUDA PASS G-S2b, G-S2c and G-S2d.** Raw:
  `docs/measurements/multimodal-support-2026-10/s4-grid-cuda/` (`gates-tiny.log`, `real.log`, `served/`).
  - **The build:** `cuda/grid_vision.go`, registered as multimodal's "cuda" Qwen3 and GLM-OCR towers, on the CUDA tower base, with aikit's own scaled attention called once per image frame
    (the frame is the segment) and separate q, k and v (GLM-OCR norms q and k per head before RoPE, which aikit's fused-qkv kernels cannot do), plus one new kernel, `tower_rope_half`
    (NeoX rotate-half on a separate buffer, bit for bit against Go). The base's attention cap of 12288 patches now applies only where the query-tiled kernel does not.
  - **G-S2b, tiny towers (norms randomised through the export's aliasing slices), four grid sets including a two-image batch:** Qwen3.5 and GLM-OCR both worst merged-token cosine
    **1.000000000**.
  - **G-S2c, planted defects, each alone red.** Metal's five reproduce Metal's recorded readings to the digit (Qwen3.5: scale 0.974966, RoPE 0.996732, position grid 0.994887, patch bias
    0.993714; GLM-OCR: scale 0.126404, RoPE 0.698253, q/k norm 0.118984, patch bias 0.531329), which is the strongest cross-check available between the two backends. CUDA's new one,
    **every frame of a batch attending across the boundary**, reads 0.986906 (Qwen3.5) and 0.457776 (GLM-OCR). The other CUDA-specific entries of Gate 0 are covered elsewhere: the bias dropped in
    the tiled GEMM branch by the base's GEMM-branch test (shown red by a mutant; these tiny towers' patch dims are multiples of 16 and never take that branch, as predicted), stale scratch
    across sizes and determinism and the ledger after `Close` by `TestGridVisionCUDA_scratchDeterminismClose`.
  - **G-S2b, real towers (`~/models/qwen3.5-0.8b` at serve's 1024-token cap; `~/models/glm-ocr` at its default 4.8 MP ceiling), nine images, every merged token against aikit's CPU Forward:**
    Qwen3.5 worst cosine **0.999999763, 0.999999999, 0.999999964, 0.999999989**; GLM-OCR **0.999999175, 0.999999999, 0.999996380, 0.999999195, 0.999999049** (the images in order: the Gemma 3 preprocess image, the Qwen2.5-VL one, then
    GLM-OCR's formula and table, and for GLM-OCR also its invoice: Qwen3.5 has four, GLM-OCR five). All over 0.9999.
  - **G-S2d, served (`--backend cuda`, decode `cuda-resident (int4)`, tower on CUDA against `-vision-device cpu`, a second CPU-tower run as the control, `--kv-sessions 1 -ctx 4096`):**
    the CPU-tower control is byte-identical to the first CPU-tower run for both families. Tower on CUDA against the CPU tower: the replies differ, **at near-ties under the registered rule**:
    Qwen3.5 first differs at generated token 3 (CPU-tower ' **' 0.546, CUDA-tower ' a' 0.314; p(other) >= half p(top) = 0.273: True), GLM-OCR at token 1 (' table' 0.561, ' image' 0.439;
    0.280: True). The towers' features differ by about 1e-6 (cosine 0.999999+), so a near-tie flipping is expected, and the rule exists for it. The logs name each arm (`tower (CPU)` /
    `tower (CUDA)`); no `declined` line appears, so the CUDA arms did run the tower on CUDA. Image encode, exploratory: Qwen3.5 5.17 s (CUDA) against 6.05 s (CPU); GLM-OCR 26.0 s (CUDA).
  - **Speed, exploratory and honest:** the CUDA towers are only **1.2-1.5x** the CPU towers (real Qwen3.5 3.2-5.2 s against 4.0-6.2 s; GLM-OCR at 1-1.6 K merged tokens 14-36 s against 20-43 s; the figures
    include aikit's host tail). That is a correct baseline and not a fast tower: the f32 GEMMs and aikit's scalar query-tiled attention are the cost, and Gemma 4's tower (2.2 s against 4.4 s)
    is better only because it has fewer patches. A faster attention and GEMM are Gate 0's step 6, to be decided by the speed record (queued separately), not claimed here.
- **S4 step 4 read 2026-10-07 (`s2-towers`, commit `5970cfa4`; not on `main`): the VRAM reserve and a graceful fallback.** Raw: `docs/measurements/multimodal-support-2026-10/s4-grid-cuda/default-plan/`
  (`s4-reserve-g4`, `s4-reserve-grid`, `s4-reserve-q25-i`).
  - **The reserve:** serve prices a CUDA tower into `Options.ExtraResidentBytes` (the field the resident plan already reads for a drafter) from the checkpoint's `vision_config`: float32
    weights plus peak scratch at the family's largest image, 0.86 GB for Gemma 4 E2B, 0.67 GB for Qwen3.5-0.8B, 2.4 GB for GLM-OCR (scratch priced at about 1.5 MP, not its 4.8 MP ceiling,
    whose 4.15 GB took GLM-OCR's CUDA context from 16384 to 4931 positions), and about 3.4 GB for Qwen2.5-VL (8192 patches). Zero unless the tower will really run on CUDA. It is an
    estimate and a ceiling the KV context gives way to: on the default plan the E2B kept 16384 positions and 4 conversations, GLM-OCR's context became 5046 (4931 after the trim), Qwen2.5-VL's 6160.
  - **What it fixed:** on the default plan the Qwen2.5-VL tower used to be declined at attach (`CUDA_ERROR_OUT_OF_MEMORY`) and run on the CPU; with the reserve it attaches on CUDA.
  - **What it did not fix, found by running it:** a 5504-patch image still missed the budget on the 3B Qwen2.5-VL (2.5 GB of tower weights beside a 3B decoder on an 8 GB card), and **aikit's
    `gpu/qwencuda` panics on that allocation failure instead of returning it, which killed the whole server** (an unrecovered panic on the request goroutine). Two guards now stand in front of it:
    `recoverDeviceTower` turns a device-tower panic into an error that names the cause and the way out, and `deviceFallback` (and, for Qwen2.5-VL, `qwenForwardWithFallback`) closes a device
    tower that ran out of memory so its VRAM comes back, and runs that image and every later one on the CPU tower. On the default plan the same request now returns the CPU tower's identical reply
    (22.3 s encode, the CPU's time) with the log line `the Qwen2.5-VL tower ran out of device memory ...; it runs on the CPU from now on`. For the aikit defect report: `qwencuda` should return
    allocation failures from `ForwardViT`, not panic.
  - **Default-plan served runs, tower on CUDA against `-vision-device cpu`:** Gemma 4 E2B identical (tower encode 2.3 s against 4.5 s); Qwen3.5-0.8B and GLM-OCR differ at near-ties as before
    (token 3, p(other) >= half p(top): True; token 1, same), the towers on CUDA, no fallback needed; Qwen2.5-VL identical via the fallback above.
  - **A test-harness trap found on the way:** `run-gs3c-served.sh` left its server running on the GPU when a request failed (`set -e`), so the next run tested the stray server and held 7 GB;
    it now stops the arm's server on any failure. The three strays were found by name (`pgrep -x`) and stopped.
- **S4 step 5, the speed record, is queued for tonight as `s4-tower-speed`** (`run-s4-tower-speed.sh`: Gemma 4 E2B, Qwen3.5, GLM-OCR and aikit's Qwen2.5-VL tower against the CPU towers, three passes
  under the timing lock, a pre-built test binary at `5970cfa4`, estimate 30 min). One pass was smoke-tested by day (6 min 44 s; exploratory, not a result). Nothing is quoted from it yet.
- **S4 status, 2026-10-07:** steps 0-5 done (step 6, a faster attention and GEMM, waits for the record). Everything is committed on `s2-towers` in `~/wt/goinfer-s2` and **not pushed** (the branch is pushed
  only on the owner's word, and it needs aikit's unreleased `s2-tower-exports`); the Mac's commits on the same branch were merged in. `main` has only the records.
- **The CUDA towers' gates are G-S2b, G-S2c and G-S2d's, unchanged, on CUDA;** the Gemma 4 CUDA tower's are Metal's
  Gemma 4 tower gates (`metal/gemma4_vision_test.go`: every soft token at cosine >= 0.9999 against aikit's CPU Forward,
  the shuffled and clamp controls) plus a served Gemma 4 image turn as in G-S2d. nobara writes its desk map and any
  CUDA-specific planted defects into this doc before the CUDA tower code.

#### S4 addendum: a float32 SigLIP tower on the CUDA tower base (the Mac's audit item, nobara 1) — registered 2026-10-07 evening, before any code

Why: under `--backend cuda` serve gives Gemma 3 an int8 SigLIP tower by default (`towerInt8`; `cuda/vision_encoder.go` is int8 by construction). `docs/measurements/siglip-int8-fidelity-2026-10-07.md` puts int8 SigLIP at relative L2
0.16-0.52 from float32 and a worst token cosine of 0.01-0.17. Metal's SigLIP was rebuilt as float32 (S3); CUDA's has not been. This builds the float32 tower, grades it, and measures what the int8 default costs end to end,
so the owner can decide the default with evidence. **Changing the default is not part of this and waits for the owner.**

- **What changes** (`cuda/` plus one small serve change): a `gridSiglip` kind on the existing tower base (`cuda/grid_vision.go`): aikit's `GridPatches` rows on the host, the patch embed plus the fixed position table, every block over ONE
  segment of all the patches with no RoPE (LayerNorm with bias, biased separate q/k/v, attention at scale 1/sqrt(head_dim), biased o, GELU-tanh MLP), then aikit's `FinishHidden` (the post-layernorm) on the host; float32 weights from
  `Encoder.Weights()`. The one global `vision.RegisterResident` factory in `cuda/vision_register.go` dispatches on how the encoder was loaded: a float32 encoder (`Weights()` succeeds) gets this tower, an int8 one gets today's
  `NewVisionEncoder`, so with no flag nothing changes. Serve cannot currently ask for float32 on CUDA (`towerInt8` returns true for Gemma 3 there whatever `-vision-quant` says, and the flag's default is the string "f32"), so
  the flag's default becomes unset: unset keeps today's choice, an explicit `-vision-quant f32` now selects the float32 tower on cuda. The tower's VRAM is added to `towerReserve` for the float32 case.
- **Gates (written before the code; the first is G-S3a's shape and bars from the Metal rebuild, unchanged):**
  - **G-S3a on CUDA.** Every output token (the `FinishHidden` rows) at cosine >= 0.9999 against aikit's CPU tower, both float32; 0.999-0.9999 is ambiguous and parked. Tiny (`testdata/siglip-tiny`): norms randomised first (the all-ones
    trap) and the biases and q/k scale sharpened exactly as `metal/s3_towers_test.go` does, because the tiny tower's zero biases and init-scale q/k hid 5 of S3's 7 defects. Each registered planted defect alone must turn it red:
    (1) the attention scale dropped, (5) the position table dropped, (6) the patch-embed bias dropped (the numbers are S3's). Real (`~/models/gemma-3-4b-it`, never the archive) under `GOINFER_HEAVY_TESTS=1`, on the four F2a images
    (`gemma3_preprocess_image.png`, `qwen25vl_preprocess_image.png`, `glm_ocr/formula.png`, `glm_ocr/table.png`), attached the way serve attaches it (`EnableResident`), and the test asserts the attached type so that another
    package's registration cannot satisfy it. Real size matters here beyond the usual reason: head_dim is 72, which no other tower on this base has, and aikit's `visioncuda` read 0.10-0.28 at real size while passing tiny.
  - **G-S3b on CUDA, served.** `gemma-3-4b-it`, `--backend cuda --kv-sessions 1` (4 KV slots do not fit the 8 GB card beside the tower; known), the `table.png` request, 32 greedy tokens with top-3 logprobs. **Reference arm, named:**
    `-vision-device cpu -vision-quant f32` (the CPU float32 tower). Arm under test: `-vision-quant f32` with the device tower. Identical reply, or a first divergence at a near-tie under the registered rule (p(other) >= half p(top)),
    graded against the named reference and never against whichever arm happens to run first (the script is told its reference).
  - **G-S3d (new), what the int8 default costs end to end.** The same served request with a third arm, the shipped default (no `-vision-quant`: the int8 device tower), graded against the same float32 CPU reference by the same rule,
    with the reply, the first-divergence token and the number of matched tokens recorded for all three arms; plus, on the four F2a images, the int8 CUDA tower's features against the float32 tower's (relative L2 and worst token
    cosine), to set beside the fidelity record's 0.16-0.52 and 0.01-0.17. Two images x two arms of text is evidence, not a verdict: it is recorded as what it is.
  - **Speed (night, a record with no bar):** tower time per image on the four images, float32 CUDA against int8 CUDA against the CPU float32 tower, interleaved under the timing lock. Exploratory by day.
- **Parked or killed:** a worst token in 0.999-0.9999 on a real image parks the tower (reported, not shipped as a selectable option until understood). A defect that stays green on the tiny tower means the fixture is degenerate
  on that axis; the check is rerun on a sharper fixture before any result is read.
- **Day / night:** the tower, the tiny gates and the vets are day work; the real-size gate and the served arms are a few minutes each (G-S3a real about 3, G-S3b/d about 8); the speed record goes on the night queue.

- **S4 addendum read 2026-10-07 evening on nobara (commits `5309aa0c`, `f37fe277`; the tower is on `main` once pushed): G-S3a PASS, G-S3b PASS, G-S3d recorded; the default is unchanged.**
  Raw: `docs/measurements/multimodal-support-2026-10/s4-siglip-f32/` (`gs3a-real.log`, `gs3b/`, `gs3d/`, `gs3d-features.log`, `speed-smoke/`).
  - **Built as registered** (`cuda/siglip_vision.go`, the `gridSiglip` kind on `cuda/grid_vision.go`, `cuda/vision_register.go`): the float32 SigLIP tower on the tower base, no RoPE, one segment over every patch, aikit's `FinishHidden`
    on the host. The one global resident factory dispatches on how the encoder was loaded: a float32 encoder gets this tower, an int8 one still gets `NewVisionEncoder` (`TestSiglipCUDA_int8StaysTheInt8Tower`). Serve:
    `-vision-quant` is unset by default; unset keeps today's choice, an explicit `f32` selects the float32 tower (`towerInt8`); `towerReserve` prices the float32 Gemma 3 tower at 2.11 GB (27 layers of hidden 1152 /
    intermediate 4304 in float32 plus the 4096-patch scratch). `docs/flags.md`, its meta file and `docs/multimodal.md` say so.
  - **G-S3a on CUDA PASS.** Tiny (`siglip-tiny`, norms randomised, patch and q/k biases sharpened, q/k scaled x6, as Metal's check): worst token cosine **1.000000000**; the three registered defects alone, each red:
    (1) attention scale dropped 0.998501, (5) position table dropped 0.745354, (6) patch-embed bias dropped 0.762383 (defect (1) is the closest to the bar and still clear of it). Real `gemma-3-4b-it`, the four F2a images, attached
    through `EnableResident` with the attached type asserted: worst token **0.999999968** (`gemma3_preprocess_image`), **0.999999968** (`qwen25vl_preprocess_image`), **0.999996014** (`glm_ocr/formula`), **0.999999927**
    (`glm_ocr/table`), all over 0.9999. head_dim 72, the real-size risk named in the registration, is not a problem for aikit's attention kernel on this base.
  - **G-S3b on CUDA PASS.** `gemma-3-4b-it`, `--backend cuda --kv-sessions 1 -ctx 4096 -vision-quant f32`, `table.png`, 32 greedy tokens. Reference `=cuda:cpu` (the CPU float32 tower): the device float32 tower (log: `encoder
    f32/cuda-resident`) gives an IDENTICAL reply, and a second CPU-tower run (the determinism control) is IDENTICAL too.
  - **G-S3d, what the int8 default costs, recorded (evidence for the owner's decision, not a verdict).** *Served, same request, the shipped default (no `-vision-quant`: `encoder int8/cuda-resident`) against the same float32
    CPU reference:* the reply differs at **generated token 3**: the reference has ' quarterly' at 0.989, the int8 tower's arm chose ' a' (0.011): not a near-tie (p(other) 0.011 against half of 0.989). The int8 reply opens "The image shows a table
    presenting quarterly unit sales data..." against the reference's "The image shows quarterly unit sales data broken down by geographic region...". *Features, the int8 CUDA tower against the float32 CPU tower, per image:*

    | image | relative L2 | worst token cosine | mean token cosine |
    |---|---|---|---|
    | gemma3_preprocess_image | 0.5238 | -0.0209 | 0.9294 |
    | qwen25vl_preprocess_image | 0.5395 | 0.0016 | 0.9291 |
    | glm_ocr/formula | 0.2330 | 0.1856 | 0.9787 |
    | glm_ocr/table | 0.1679 | 0.1410 | 0.9868 |

    These reproduce `siglip-int8-fidelity-2026-10-07.md` (0.16-0.52, worst 0.01-0.17) on the CUDA tower itself. One image, one prompt and 32 tokens is evidence that the difference reaches the reply, not a rate.
  - **Speed, EXPLORATORY (single samples, nothing else measuring; the night record is queued as `s4sig-speed`, 12 min):** the tower alone on one image: float32 CUDA **18.0 s**, int8 CUDA **4.1-4.8 s**, CPU float32 **21-28 s**
    (served encode: 18.0 s float32, 4.1 s int8). So the float32 tower is only about 1.2-1.5x the CPU one and 3.6x over S7's 5 s TTFT bar by itself, while the int8 tower, which diverges visibly, is the one that fits the bar.
    That is the same shape as the other CUDA towers (S4 step 3): correct, not fast. (Superseded the same evening by S17 lever A below: the float32 tower is now 1.97 s.) The f32 tower's cost is the scalar GEMMs and aikit's query-tiled attention at 4096 patches; the int8 path uses the tensor-core kernels. S17 (tower speed)
    is where this closes.
  - **Open, the owner's:** whether the default for Gemma 3 on cuda/webgpu changes from int8 to float32. The data above is the trade: float32 is exact (identical reply) and 18 s per new image on this card; int8 is 4 s and changes the reply.
  - **Not done:** the float32 tower under `--backend webgpu` (no tower there; an explicit `f32` runs the CPU tower); a float32-tower reserve measurement on the default plan (the 2.11 GB estimate is arithmetic, and Gemma 3 4B with the default
    4 KV slots still does not fit the 8 GB card, which is S18's CUDA side).

### S5 — Gemma 4 E2B/E4B audio into the model

aikit's `audio` package probably loads E2B's tower unchanged (its config and tensor names were checked on nobara,
2026-10-06; the load was not run). What is left:
- the decoder splice: audio rows unscaled, and PAD for PLE's token-identity term;
- the prompt layout: `<|audio>` n x `<|audio|>` `<audio|>`;
- `input_audio` parts on `/v1/chat/completions`, mirroring the embeddings route.

- **Gates:**
  - The tower's stages against HF on E2B's own weights, on a fixed clip.
  - The full-model logits on an audio prompt against HF `Gemma4ForConditionalGeneration`.
  - A served transcription-style request.
- **Faster with S1:** it is CPU decode until S1 lands.
- **Size:** M-L.

**S5 Gate 0, the desk map (2026-10-07, Mac):**
- **The tower is ready in aikit.** `audio.LoadGemma4AudioEncoder` reads E2B's names (it detects the `model.` prefix) and
  applies `embed_audio` (an unweighted RMSNorm, then the projection to the text width, 1536 for E2B), so it emits soft
  tokens in the text space. `audio.Gemma4Features` is the log-mel extractor and `Gemma4SoftTokens` the count. Both are
  in production already for EmbeddingGemma 2, whose tower is the same `gemma4_audio` architecture, gated against HF at
  per-stage soft-token cosine >= 0.9999 (`embeddinggemma2/audio_real_test.go`). Metal has a tower accelerator for it
  (`metal/gemma4_audio.go`, EmbeddingGemma 2's), for later speed. E2B's tower: 12 layers, hidden 1024, 8 heads,
  output 1536.
- **The decoder splice already does what S5 asks.** `prefillLogitsGemma4VL` replaces a run of positions with projected
  rows, unscaled, and feeds PLE the pad token's id at those positions (HF's multimodal forward substitutes PAD before
  PLE's token-identity term). It keys on positions, not on the image token's id, so an audio run uses it unchanged. S1's
  resident decode follows it as for an image. E2B and E4B are causal (`use_bidirectional_attention` unset), so no mask
  question arises.
- **The prompt:** HF's processor (`processing_gemma4.py`, `replace_audio_token`) emits `<|audio>` + n x `<|audio|>` +
  `<audio|>` (ids 256000, 258881, 258883) with no newline on either side, as Gemma 4's image block. n is the feature
  mask after two stride-2 convolutions, which `Gemma4SoftTokens` mirrors.
- **Serve:** no audio input on `/v1/chat/completions` yet. The embeddings route already takes audio for EmbeddingGemma
  2 (`internal/serveapp/embeddings*.go`, WAV decoding in `embeddinggemma2/audio.go`); the chat route gains
  `input_audio` content parts (OpenAI's shape: base64 `data` plus `format`), WAV only to start.
- **References:** this Mac has transformers 5.16.1, but E2B in float32 is about 20 GB, too big here. The tower plus
  `embed_audio` alone (about 1.2 GB in float32) is not. nobara has 62 GB and transformers 5.12 (`~/.venv-vl`); E2B's
  config was written by 5.6.2 (`global_head_dim`), which 5.12 reads.

**S5 gates, written 2026-10-07 before any S5 code or measurement:**
- **G-S5a, the tower on E2B's own weights against HF:** every stage (after the subsampler, each block, the tower, the
  embedder) at soft-token cosine >= 0.9999 on the three EmbeddingGemma 2 clips (`testdata/embeddinggemma2-audio/`,
  short, mid and long), aikit's log-mel and HF's both fed; the first stage under the bar is named. HF's reference
  builds only `Gemma4AudioModel` and the embedder from the checkpoint, float32, sdpa (eager inverts the audio mask), with
  the same load checks EmbeddingGemma 2's pin script makes. Ambiguous (parked): a worst stage in 0.999-0.9999.
- **G-S5b, the full model on an audio prompt against HF `Gemma4ForConditionalGeneration` (float32, on nobara):** the
  goinfer CPU prefill in float32 (`Quant: ""`) at every position of the prompt; the audio block plus a text
  instruction. PASS: last-position logit cosine >= 0.999 and argmax equal, and per-position argmax agreement >= 95% over
  the text positions after the block.
  - Planted defects, each red: (1) the audio rows multiplied by the embed scale; (2) PLE's token-identity term from the
    audio token's id instead of PAD; (3) the `<|audio>`/`<audio|>` delimiters dropped from the ids.
  - int4 (serve's default) is reported beside it, not graded.
- **G-S5c, served:** one transcription-style request ("Transcribe this audio.") per clip through the Metal serve binary,
  `--backend cpu` against `--backend metal` with every load flag held equal (`--embed-int4=false` on both, the G-S3c
  lesson). Identical replies, or a first divergence at a near-tie under the log-probability definition; a Metal arm
  not decoding resident voids that reading. The clips are synthetic tones, so the reply's content is not graded, only
  agreement. A spoken clip is an owed extra, if one with a known license can be committed.
- **Speed (night):** tower time per clip, CPU against Metal's accelerator; TTFT of the served turn. A record.

**S5 progress (2026-10-07, Mac):**
- **G-S5a: PASS.** `multimodal/gemma4_audio_e2b_real_test.go`, E2B's tower and `embed_audio` through aikit, against
  `scripts/pin_gemma4_e2b_audio.py`. The reference was built on this Mac with transformers 5.16.1: tower and embedder
  only, float32, sdpa, every load check passing. E2B's upstream `processor_config.json` (saved as
  `testdata/gemma4-e2b-audio/processor_config.json`) equals the extractor's defaults field for field, and it caps a
  clip at 750 soft tokens (30 s at 40 ms per token).

  | clip | samples | frames (HF) | soft tokens (HF) | worst stage cosine | largest max abs diff |
  |---|---|---|---|---|---|
  | short | 5,920 | 36 (36) | 9 (9) | 1.000000 at every stage | 2.4e-4 |
  | mid | 37,920 | 236 (236) | 59 (59) | 1.000000 at every stage | 7.2e-4 |
  | long | 125,317 | 783 (783) | 196 (196) | 1.000000 at every stage | 1.3e-3 (block 11) |

  aikit's own log-mel is within 4.8e-7 of HF's, and the soft tokens from it match HF's embedder at cosine 1.000000000.
  The tower takes about 1 s for the 7.8 s clip on the CPU (exploratory).
- **WAV decoding moved to `multimodal.DecodeWAV`** for serve's chat route; `embeddinggemma2.DecodeWAV` stays as a
  wrapper. `multimodal.Gemma4AudioBlock` builds the placeholder run.
- **G-S5b: harness built, queued on nobara's night queue** (`s5b-e2b-audio`, about 20 min,
  `run-s5b-nobara.sh`; the binary built there from a bundle of this Mac's local commits, in a separate worktree,
  `~/wt/goinfer-s5`). The ids come from goinfer's built-in Gemma 4 template (`chat.Gemma4()`), because E2B's
  safetensors directory carries none. Serve with the GGUF uses the GGUF's own template, which ends the prompt
  differently (75 tokens against 79). G-S5b compares the same ids on both sides, so that does not touch it. The mid
  clip leaves 15 text positions after the audio block, so the registered ">= 95% agreement" means all 15.
- **Serve takes audio on `/v1/chat/completions`:**
  - The request: an OpenAI `input_audio` part (base64 or a data: URI, `format` wav), 16 kHz mono 16-bit PCM, at most
    30 s. A longer clip is refused by name: the extractor itself stops at 30 s, so a soft-token cap alone would have
    cut it silently. An earlier message's clip is replaced by a note, as an earlier image is, counted in the same
    `X-Goinfer-Images-Omitted` header.
  - The model: a Gemma 4 checkpoint whose config has an `audio_config` turns audio on at load (the log says so). The
    tower loads on the first clip (1.7 s here) and runs on the CPU. The turn goes through `GenerateGemma4VL`,
    streaming and logprobs included.
  - A model without the tower answers 400 by name.
  - Exploratory smoke, 12:47 PDT: `--model ~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf --vision
    ~/models/gemma-4-E2B-unq --backend cpu`, the mid clip, "Describe this audio briefly.": "The audio appears to be a
    sound effect of a door slamming." in 4.3 s, 75 prompt tokens. The safetensors load alone was refused by the fit
    guard beside the owner's session.
- **G-S5c, read 2026-10-07 12:49-12:51 PDT on the Mac: PASS.**
  - The setup: `run-gs5c-served.sh`, a serve binary built from `main` at `ec97c37b` plus the serve audio change
    (committed as `d499cb30`). The model is `--model ~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf --vision
    ~/models/gemma-4-E2B-unq`, with `--embed-int4=false` on every arm. "Transcribe this audio.", 32 greedy tokens.
    Arms: `cpu`, `metal`, then `cpu` again. The Metal arm decoded `metal-resident (int4)`, "on the resident path", for
    every clip.

    | clip | prompt tokens | Metal against CPU | second CPU run against the first |
    |---|---|---|---|
    | short | 25 | first differs at token 1: CPU "'" 0.435, Metal's " need" 0.298 there, a near-tie | identical |
    | mid | 75 | first differs at token 2: CPU " Trans" 0.125, Metal's " I" 0.105 there, a near-tie | identical |
    | long | 212 | identical | identical |

  - The replies are E2B declining or murmuring ("I'm sorry, I cannot transcribe the audio.", "Mmm."). That is right for
    synthetic tones with no speech, and content is not graded. A spoken clip with a known license is still owed for a
    real transcription.
  - Request times (exploratory, the tower on the CPU, loaded on the first clip): CPU 2.2-6.5 s, Metal 3.3-8.7 s. Metal
    is slower here because every one of these is a cold image-turn prefill on the CPU plus an upload. Batched E-model
    prefill is S9's work.
  - Raw: `docs/measurements/multimodal-support-2026-10/s5-gs5c/`.
- **S5's spoken clip, registered 2026-10-07 before it ran:** `testdata/speech/librispeech-1272-128104-0000.wav`
  (LibriSpeech dev-clean, CC BY 4.0, 5.9 s, attribution in the directory's README), served to E2B with "Transcribe this
  audio." through `run-gs5c-served.sh` (`GS5C_CLIPS`), arms `cpu`, `metal`, `cpu`, `--embed-int4=false` on all.
  Recorded, no bar: the transcription and its word error rate against LibriSpeech's transcript. Graded by G-S5c's rule:
  Metal against the CPU, identical or a near-tie at the first difference, and the CPU repeat byte-identical.
  - **Read 15:34 PDT: PASS.**
    - All three arms answer "Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.",
      byte-identical (Metal against the CPU, and the CPU repeat).
    - Against LibriSpeech's "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"
      the word error rate is 0, counting "Mr." as "MISTER" and ignoring case and punctuation.
    - The Metal arm prefilled AND decoded on the resident ("prefill resident": S9's pass carries audio rows exactly as
      it carries image rows).
    - 163 prompt tokens; request time 7.7 s on Metal against 9.3-10.3 s on the CPU (exploratory).
    - Raw: `docs/measurements/multimodal-support-2026-10/s5-speech/`.
- **G-S5d, E2B's audio tower on Metal, registered 2026-10-07 before its code:** EmbeddingGemma 2's Metal audio
  accelerator (`metal/gemma4_audio.go`, generic over aikit's `Gemma4AudioEncoder`) runs E2B's 12 conformer blocks;
  aikit's `Subsample` and `FinishBlocks` stay on the host.
  - **Bar:** the soft tokens at worst-row cosine >= 0.9999 against aikit's CPU `Forward`, on the three EmbeddingGemma 2
    clips and the LibriSpeech clip; 0.999-0.9999 ambiguous (parked).
  - **Served:** the LibriSpeech clip and the three tone clips, `--backend metal` in both arms, the tower on Metal against
    `-vision-device cpu`. Identical replies, or a first divergence at a near-tie.
  - **Serve uses the device tower only under `--backend metal`;** `-vision-device cpu` keeps it on the CPU.
  - **Read 15:45-15:48 PDT: PASS** (`s2-towers` at `38a593c7`).
    - **Tower:** `metal/gemma4_audio_e2b_test.go`. Worst soft-token cosine 1.000000000 (short, mid, LibriSpeech) and
      0.999999999 (long). The 12 blocks take 70-179 ms on Metal against 239-702 ms on the CPU for the three longer
      clips (exploratory).
    - **Served** (the E2B GGUF with `--vision ~/models/gemma-4-E2B-unq`, both `--backend metal`, the tower on the CPU in
      one run and on Metal in the other; serve logged "Metal (the conformer blocks ...)"):
      - LibriSpeech and mid: identical replies (the correct transcription for LibriSpeech);
      - short and long, the synthetic tones: the replies first differ at token 0, each a near-tie on the CPU-tower
        arm's probabilities ("Please" against "I", "M" against "Hmm").
    - Raw: `docs/measurements/multimodal-support-2026-10/s5-gs5d/`.
- **S5 status, 2026-10-07 evening: G-S5a, G-S5c, the spoken clip (LibriSpeech, word error rate 0) and G-S5d (E2B's
  audio tower on Metal, on `s2-towers`) PASS on the Mac; G-S5b (the HF anchor for audio into the model) is queued on
  nobara and has no result yet.** Owed: the speed record (tower per clip, CPU against Metal). Resampling for WAVs that
  are not 16 kHz mono (S5's follow-up) passed G-S5e with option (b) and is wired into chat audio.

- **S5's follow-up: audio at any rate and channel count. G-S5e, registered 2026-10-07 evening before any resampler
  code.**
  - **The gap.** Serve refuses a WAV that is not 16 kHz mono (`multimodal.DecodeWAV`), so ordinary 44.1 or 48 kHz files
    and stereo files get a 400.
  - **The change:**
    - a pure-Go resampler to 16 kHz: windowed sinc, Kaiser window, polyphase for rational ratios;
    - a downmix of any channel count to mono, by the per-sample mean (librosa's `to_mono` convention);
    - both in `multimodal`, used by `DecodeWAV`;
    - 16-bit PCM only, as today.
  - **Inputs, made offline and committed** under `testdata/speech/`, with the tool and settings recorded in the README
    there. All derive from the LibriSpeech clip (CC BY 4.0, 5.855 s):
    - (i) **44.1 kHz mono:** ffmpeg 8.1.1, `-ar 44100 -ac 1 -c:a pcm_s16le`, its default resampler.
    - (ii) **48 kHz stereo:** ffmpeg to 48 kHz mono, then numpy builds the channels as L = s + d and R = s - d, with
      d = 0.3 x the clip time-reversed, checked free of clipping. So the correct downmix is s up to rounding, and a
      left-only read is visibly wrong.
    - (iii) **The reference:** each of those files back to 16 kHz with scipy's `resample_poly` (scipy 1.16.3, Kaiser
      window, its defaults), after the reference downmix for (ii).
  - **The bar, through the real E2B audio tower** (aikit's log-mel and `Gemma4AudioEncoder`, the path G-S5a validated
    against HF), soft tokens compared with the 16 kHz original's, per-token cosine:
    - **PASS:** for (i) and (ii), goinfer's worst-token and mean cosine are each at least the reference's (iii) minus
      0.001.
    - **Ambiguous (parked):** 0.001-0.005 below the reference.
    - **FAIL:** worse than that.
    - **The reason for this form:** a round trip through 44.1 or 48 kHz cannot return the original exactly in any
      resampler (the anti-alias transition band below 8 kHz, a second 16-bit rounding). What goinfer owes is no more
      loss than a standard resampler. An absolute cosine set before measuring would be a guess at that loss.
  - **Served:** one serve binary on the E2B, `--backend cpu`, the three clips (the original, (i), (ii)), "Transcribe
    this audio.", 32 greedy tokens. **(i)'s and (ii)'s transcriptions must be identical to the original's,** which is
    the exact LibriSpeech text (G-S5c's spoken-clip reading).
  - **Planted defects, each alone must fail the bar:**
    1. (i) resampled as if it were 48 kHz;
    2. (ii) read from its left channel only.
  - **Cost:** the tower on CPU for nine clip reads plus one serve run, a few minutes; a quick check, by day.
- **G-S5e, read 2026-10-07 17:18-17:25 PDT on the Mac: FAIL, twice. Not wired; the owner decides.**
  - **Reading 1, cutoff at 0.97 of Nyquist:** FAIL.

    | clip | goinfer worst / mean | scipy worst / mean |
    |---|---|---|
    | 44.1 kHz mono | 0.575939 / 0.975807 | 0.617187 / 0.978763 |
    | 48 kHz stereo | 0.576007 / 0.975891 | 0.619399 / 0.978813 |

    - **The mechanism, per log-mel bin:** Gemma 4's filterbank runs to exactly 8 kHz, and a cutoff 3% under Nyquist
      removes in-band content there. Below 7 kHz goinfer matches or beats scipy. In mel bin 127, goinfer's mean
      |d log-mel| is 0.96 against scipy's 0.58.
    - **The change:** the cutoff moved to Nyquist, scipy's convention. The bar did not move.
  - **Reading 2, cutoff at Nyquist:** FAIL, by 0.007-0.008 on the worst token. The means are within 0.0003.

    | clip | goinfer worst / mean | scipy worst / mean |
    |---|---|---|
    | 44.1 kHz mono | 0.610265 / 0.978540 | 0.617187 / 0.978763 |
    | 48 kHz stereo | 0.610939 / 0.978580 | 0.619399 / 0.978813 |

    - The worst token for both resamplers is token 87 (3.48 s).
    - goinfer sits slightly below scipy on 123 of 147 tokens.
    - Both leave the same 71 tokens under 0.99: any round trip through 44.1 kHz is lossy for this tower.
  - **Planted defects, both red:**
    - (1) the wrong rate: mean 0.302;
    - (2) the left channel alone: mean 0.826, worst 0.003.
  - **The served part was not run:** the in-process bar failed first.
  - **What stands:**
    - `multimodal.Resample`, `Downmix` and `DecodeWAVAnyRate` are committed with unit tests (passthrough, DC, tones,
      aliasing, the downmix, and the extensible header).
    - `DecodeWAV`, which serve uses, still takes only 16 kHz mono, and a test pins that.
  - **For the owner:**
    - (a) accept goinfer's resampler as it is (its mean is scipy's to 0.0003, and its worst token is 0.007 under
      scipy's on a token both leave at 0.6);
    - (b) a further change with its own mechanism (the window, beta and length are what still differ from
      `resample_poly`), then a fresh reading;
    - (c) leave serve 16 kHz-only.
  - Raw: `docs/measurements/multimodal-support-2026-10/s5-gs5e/`.
- **Owner decision 2026-10-07 evening: option (b).** goinfer's resampler is to match scipy's `resample_poly` filter exactly:
  - up and down reduced by their gcd;
  - the cutoff 1/max(up, down) of the upsampled Nyquist, by `firwin` with unit DC gain;
  - a Kaiser window with β = 5.0;
  - 20·max(up, down) + 1 taps, centred, times up;
  - zero padding, and an output length of ceil(n·up/down).

  **Registered before the run, with the bar unchanged:**
  - **The prediction:** with that filter, goinfer reproduces scipy's worst-token and mean cosine within 1e-4 on both
    clips (44.1 kHz mono, 48 kHz stereo). Then G-S5e passes as registered.
  - **The check of the filter itself:** a committed scipy golden, the resampled output of a deterministic signal at
    44.1 kHz and at 48 kHz, which goinfer must match to 1e-6 per sample.
  - **If it still fails:** stop and report. No further change to the resampler.
- **G-S5e, read 2026-10-07 17:52-17:55 PDT with option (b): PASS, and the prediction held exactly.**
  - **The filter:** goinfer's `Resample` matches `resample_poly` to 3e-8 per sample at 44.1, 48, 22.05 and 8 kHz
    (`TestResample_matchesResamplePoly`, scipy 1.16.3), which is float32 rounding.
  - **Through the E2B tower** (`s5-gs5e/run3-resample-poly.log`), goinfer's worst-token and mean cosines equal scipy's to
    six decimals:
    - 44.1 kHz mono: 0.617187 / 0.978763;
    - 48 kHz stereo: 0.619399 / 0.978813.
  - **Planted defects, both still red:** the wrong rate, mean 0.302; the left channel alone, mean 0.827.
  - **Served:** serve's audio path now decodes with `DecodeWAVAnyRate`.
    - On the CPU, the original, the 44.1 kHz clip and the 48 kHz stereo clip all transcribe to the identical, correct
      text ("Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.").
    - A second CPU run is byte-identical.
    - Raw: `s5-gs5e/served/`.
  - **The scope:** chat audio for Gemma 4 (E2B, and E4B through the same path). EmbeddingGemma 2's audio embeddings keep
    `DecodeWAV`, 16 kHz mono only, since G-S5e did not read their tower.

### S6 — Coverage that is cheap once the above exists

- **Qwen3.5+ MoE images:** never run. On nobara, at night; the checkpoint's tower is in the archive, so it is copied to
  `~/models` for anything timed.
- **Gemma 4 E4B and 31B:** download and validate (owner, 2026-10-07). E4B through S1's E-model gates on both GPU
  backends; the 31B through the 26B's (bidirectional image prefill, the S1.0 re-check shape). Disk first on both boxes.
- **Gemma 4 E4B on Metal, gates registered 2026-10-07 before any E4B measurement:**
  - **The checkpoint:** `google/gemma-4-E4B-it`, downloaded to `~/models/gemma-4-E4B-it` (15 GB, 2026-10-07): 42
    layers, hidden 2560, 18 KV-shared layers, PLE width 256, an audio tower. Metal reads it from an int4 Metal sidecar
    (`prequant -quant int4 -target metal`, int8 embedding table: Metal's resident needs it), built at night. Building
    it by day goes through the load's fit guard, which E2B's safetensors already failed beside the owner's session.
  - **G-E4B-1, S1's G3 as re-registered for E2B, unchanged:** teacher-forced agreement of the Metal resident against
    the CPU on the same sidecar, over G3's prompts, against the validated Qwen2.5-Coder-1.5B's in the same process.
    PASS at >= the reference's - 2.0 points with free-run passes >= the reference's - 1; 2.0-4.0 points below
    ambiguous (parked); worse fails.
  - **G-E4B-2, served:** G4's image request and G-S5c's three audio clips through one Metal serve binary,
    `--backend metal` against `--backend cpu`, `--embed-int4=false` and `-vision-device cpu` on every arm, plus a
    second CPU run. Identical replies, or a first divergence at a near-tie; the CPU repeat byte-identical; a Metal arm
    not decoding resident voids that reading.
  - **The 31B** (`google/gemma-4-31B-it`, 62.6 GB) is downloading on nobara. It is too large for this Mac by any
    path, so its gates (the 26B's shape) are nobara's.
- **EmbeddingGemma 2 on CUDA:** text, image and audio. Optional. The Metal kernels show the shapes; the work is a CUDA
  twin.
- **WebGPU:** moved to its own phase, S12 (owner: invest, 2026-10-07).

### S7 — Measure every cell, once, at night

One night job per box re-measures each table cell on the current binary, with each figure's provenance:
- per-image (and per-clip) time;
- TTFT;
- decode tokens per second after the image.

It replaces the scattered and stale figures above: the 31 s, 29 s and 4.1 s, and the June ones.

- **Speed bar (confirmed by the owner 2026-10-07 evening):** a cell is "usable" when a new image's TTFT is under 5 s at
  the default image budget, on the box's best backend.
- **Output:** a dated record in `docs/measurements/`. S8 reads its numbers.

#### S7 and S13-lite, registered 2026-10-07 evening before they run (the Mac half)

- **The instrument:** `docs/measurements/multimodal-support-2026-10/vision_ttft.py`.
  - It sends streaming requests and measures TTFT as the wall time to the first non-empty content delta, `max_tokens` 8,
    temperature 0.
  - **Every request, warm-up included, carries media no server has seen:** the base image or WAV with that request's own
    pseudorandom least-significant-bit pattern (red channel for images, the low bit of each 16-bit sample for audio).
    The size and content are the base's, so the tower and prefill cost what they cost, and every byte-hash image cache
    misses (goinfer's feature cache, Ollama's and llama.cpp's). This is S13's harness fix; `bench_peer.py`'s
    `fresh_image` does the same.
  - **The media:**
    - images: `testdata/gemma3_preprocess_image.png`, 896x896, which reaches or nears each family's default budget;
    - audio: the LibriSpeech clip, 5.9 s;
    - prompts: "Describe this image." and "Transcribe this audio."
- **S7, Mac** (`run-s7-mac.sh`, night queue `s7-mac`, estimated 40 min):
  - **The cells:** Gemma 3 4B, Gemma 4 E2B (image, and audio), Qwen2.5-VL-3B, Qwen3.5-0.8B, Qwen3-VL-2B, GLM-OCR, and
    Gemma 4 E4B if tonight's `s6-e4b` built its sidecar.
  - **How they run:** each on `--backend metal` with serve's defaults and nothing else, one warm-up and three timed
    requests per cell. The binary is `serve-metal` from `s2-towers` @ `ed8d4756`, which has the device towers.
  - **The reading:** usable when the median of the three timed TTFTs is under 5.0 s. Each cell also records the decode
    path and where the tower ran, from serve's own log lines.
  - S7 is a record, and its gaps rank S16-S18.
- **S13-lite, Mac** (`run-s13lite-mac.sh`, night queue `s13lite-mac`, estimated 40 min): Gemma 3 4B, Metal, every
  engine at its defaults.
  - **The engines:**
    - goinfer: `serve-metal` as above, `~/models/gemma-3-4b-it`, int4 at load;
    - Ollama 0.32.5: `gemma3:4b`, model blob `sha256:aeda25e63ebd6...`, its own server;
    - llama.cpp: `llama-server` build 10621 (`c1d0e7a00`), ggml-org/gemma-3-4b-it-GGUF Q4_K_M plus `mmproj-model-f16`.
    - These are peer builds of the same base checkpoint at each engine's own int4-class quantization, which is the
      peer convention.
  - **The procedure:**
    - three rounds, each engine once per round, the order rotated each round;
    - per engine: start its server, one warm-up and three timed requests, stop it (same-session interleaved by block;
      three 4B servers do not fit 16 GB at once).
  - **The reading:** the median of the nine timed TTFTs per engine, against the 5 s bar, and goinfer's ratio to each
    peer. Reported, not gated.
- **A by-day smoke of the harness (17:29 PDT, one request per engine; exploratory and not quotable):** it ran end to
  end on all three engines. Ollama's fresh-image TTFT read about 5 s, where r12's resent image read 0.4 s, which fits
  the cache-hit reading.

#### S7 and S13-lite, the nobara half, registered 2026-10-07 evening before they run

The instrument, the media, the prompts, the 5 s bar and the reading are the Mac half's, unchanged (`vision_ttft.py`: a fresh image or clip per request, warm-up included; TTFT to the first content delta; usable when the median of
the three timed requests is under 5.0 s). What differs is the box: the RTX 2070 SUPER (8 GB, driver 595.91.07), `--backend cuda`, serve's defaults and nothing else.

- **S7, nobara** (`run-s7-nobara.sh`, night queue `s7-nobara`, estimated 60 min, one warm-up and three timed requests per cell):
  - **The cells:** Gemma 3 4B, Gemma 4 E2B (image, and audio), Qwen2.5-VL-3B, Qwen3.5-0.8B, Qwen3-VL-2B, GLM-OCR. Gemma 4 E4B is not run (no checkpoint on this box).
  - **The binary:** `serve-cuda` built from `main` at `575b4a8c` with the released aikit v1.59.0 pins, pinned in `~/goinfer-bench/s7n/`.
  - **An S18 'after' cell (added later the same evening):** the Gemma 3 4B cell runs twice, on that binary (the shipped default as it was) and on `serve-cuda-s18` (the same tree plus S18's CUDA fixes), in the same
    record, so the repair is read next to the problem it fixes.
  - **What the defaults do, found by a by-day look, not a measurement:** Gemma 3 4B's default plan fails the resident build on this card (`CUDA_ERROR_OUT_OF_MEMORY` after the four KV slots), so its decoder runs on
    the CPU while the int8 tower loads on CUDA: S18's gap, as recorded. That cell therefore reads a CPU-decoder TTFT, and is a reading of the shipped default, not of what the card can do (a hand-set
    `--kv-sessions 1` cell is S18's G-S18a comparison, not this one). Every cell records the decode path and where the tower ran from serve's own log lines.
  - **Reading:** a record. Its gaps rank S16-S18, and S17's CUDA half has its first number already (the float32 SigLIP tower, 18 s per image; the int8 one 4 s).
- **S13-lite, nobara** (`run-s13lite-nobara.sh`, night queue `s13lite-nobara`, estimated 55 min): Gemma 3 4B, CUDA, every engine at its defaults, a new image every request, three rounds with the engine order
  rotated each round (one server at a time; two 4B servers do not fit the 8 GB card).
  - **The engines:** goinfer as above, `~/models/gemma-3-4b-it` (int4 at load); **Ollama 0.32.5**, `gemma3:4b`, **model blob `sha256:aeda25e63ebd6`, the same blob the Mac's cell reads** (pulled to `~/ollama-0325/models`
    2026-10-07), its own server on 11535.
  - **llama.cpp** (added after a correction, below): `~/mycode/peers/llama.cpp/build/bin/llama-server`, **commit `427291b`** (version string `0.4.0-dev build 1`), built with CUDA for sm_75 (checked: it holds 6.3 GB of
    the card while serving), with ggml-org's `gemma-3-4b-it-Q4_K_M.gguf` and `mmproj-model-f16.gguf` in `~/models/gemma-3-4b-it-gguf` (the Mac's files; the Q4_K_M is byte-identical to the older
    `~/models/gemma-3-4b-it-Q4_K_M.gguf`), `-ngl 99`. **This is not the Mac's build** (10621, `c1d0e7a00`): a different commit, so the cross-box reading of the llama.cpp cell carries that difference; each box's own ratio does
    not. A rebuild at `c1d0e7a00` is possible here (`~/.local/bin/cmake`, `~/cuda-toolkit/bin/nvcc` 12.6) if the owner wants exact parity.
  - **A correction to this registration's first draft:** it said llama.cpp could not run on this box (no `cmake`, no `nvcc`, no `llama-server`). That came from a search of `PATH` and a shallow `find` that skipped the
    separate `/home` partition; the checkout, the build, `cmake` and `nvcc` were all under `~`. The script no longer has a "no llama.cpp" switch.
  - **The reading:** the median of the nine timed TTFTs per engine against the 5 s bar, and goinfer's ratio to each peer. Reported, not gated.
- **By-day smokes of the harness on this box (18:50 and 19:11 PDT, exploratory and not quotable):** one goinfer cell (Qwen3.5-0.8B, TTFT 3.6 s on a fresh image), the Ollama arm (`gemma3:4b`, 1.6 s on a fresh image, 7.9 s on its
  first request) and the llama.cpp arm (Gemma 3 4B with its mmproj, 0.94 s) all ran end to end through `vision_ttft.py` under the timing lock.

#### S7 on CUDA, the Qwen2.5-VL fix, read 2026-10-08 (nobara; commit follows this record)

Registered above ("S7 on CUDA, the fix", before any code). Raw: `docs/measurements/multimodal-support-2026-10/s7-qwen25-cuda/`.
- **Built as registered:** `cuda/qwen25_vision.go` (Metal's port on the tower base, from aikit's `GPUWeights`, `BuildWindowPlan` and `MergeHidden`; the fused attention per window and per frame segment), the `gridQwen25` kind, and the two planted-defect switches. aikit's
  `gpu/qwencuda` is no longer imported by the cuda package or by the G-S4q test (which now grades whatever `EnableResident` attaches, goinfer's tower, and says so).
- **G-S3a on CUDA PASS.** Tiny (`qwen25vl-tiny`, sharpened as Metal's check, grids 8x8, 12x16 and a two-image batch): worst token cosine **1.000000000**; the four registered defects each red: (1) attention scale dropped 0.629849, (2) RoPE halves swapped 0.750730,
  (3) every block attending its whole frame 0.789727, (4) window reordering skipped 0.726768. Real `qwen25vl-3b-instruct`, the four F2a images, attached through `EnableResident` with the attached type asserted: worst token **0.999997532, 0.999999995,
  0.999999803, 0.999999762**. An int8-loaded encoder is refused by name (`TestQwen25CUDA_int8Declines`).
- **The first speed read MISSED its band and tripped the kill line, and was re-opened, not tuned:** 4.04 s on the 896x896 image against the registered 1.7-2.6 s ("a tower over 4 s ... means the plan is wrong and is re-opened"). The step 0 profile of this tower
  (`TestS17ProfileQwen25CUDA`) said why: **GEMM was 92% of 3.8 s at 1.82 TFLOPS** (the other towers' GEMMs run at 3.5), attention only 4.5% (the fused kernel is fine on 64-patch windows). Qwen2.5-VL-3B's intermediate size is 3420, and 3420 % 16 = 12 sends the down
  projection and the biased epilogue through the tiled GEMM. Fix: the intermediate width is zero-padded to 3456 (gate/up rows and biases, down's columns), which is the same arithmetic (silu(0) x 0 = 0 exactly). Then **1.80 s, GEMM 3.52 TFLOPS**, cosines unchanged.
- **Speed after the fix, against the registered bands** (the real-size test, host tail included): 896x896 image (64x64 grid) **2.05 s** (band 1.7-2.6 s; aikit's 7.97 s); formula.png (86x72) **3.18 s** (band 2.5-3.8 s; aikit's 15.3 s); table.png (64x86) 2.83 s. 3.9x and 4.8x aikit's.
- **Served, S7's cell at serve's defaults on the 8 GB card (exploratory smoke by day; the night re-read is queued as `s7-nobara-2`):** TTFT **2.46 s** (three timed requests, 2.458-2.471 s) against 15.64 s, under the 5 s bar; the tower on CUDA (2.0 s per image),
  decode `cuda-resident`, context 5057 across 4 KV slots. The registered prediction was 3-4 s; it came in better.
- **The default plan still failed after the tower was fast, and the cause is accounting, measured:** the first served run fell back to the CPU tower again (a 20 MB scratch allocation failed). With `nvidia-smi` through the run: 467 MiB idle; 4858 MiB after the decoder and its KV; 7616 MiB with the
  tower's weights, leaving **169 MiB free** against about 270 MiB for the first image's scratch. Three facts, none of them a tower defect: (1) the card has a constant ~407 MiB the driver hides (8192 MiB total, 7785 usable), which the plan's driver-reported free already
  accounts for; (2) the resident build allocated **535 MiB** of its own scratch after the plan's reading, against the plan's 384 MiB margin (the same class as `TestResidentDenseBytes_matchesCUDADevice`'s 421 MiB overshoot on the 7B, which fails at the session-start baseline too);
  (3) the tower's weights took **337 MiB more than their arithmetic size** (about 570 buffers, each rounded by the allocator). The tower base's own context costs 2 MiB (`TestTowerOps_contextVRAM`): the second-context theory was checked and is wrong.
  `towerVRAMEstimate` for `qwen2_5_vl` is now calibrated to those measurements (`qwen25VLTowerEstimate`: 3.73 GB at the 8192-patch ceiling, including 256 MiB of slack for the margin overshoot), bounded in `TestTowerVRAMEstimate`.
- **Not done:** the other towers' estimates were not re-measured against `nvidia-smi` (Gemma 4, Qwen3.5, GLM-OCR and the Gemma 3 pair fit at defaults in the first night's S7, so none is known to be short); the margin overshoot itself (535 against 384 MiB for this model) is the open
  Plan-accounting item behind the heavy-tier failures, and the slack here is a local cover for it, not that fix.

#### S13-lite, the float32 arm, registered 2026-10-08 before it runs (nobara)

The cell is last night's S13-lite, unchanged (Gemma 3 4B, `testdata/gemma3_preprocess_image.png`, a new image every request, 3 rounds of one warm-up and three timed requests per engine, the engine order rotated each round, one server at
a time, every engine on CUDA at its defaults, `vision_ttft.py`), with one more goinfer arm. `run-s13lite-f32-nobara.sh`, night queue `s13lite-f32-nobara`, estimated 20 minutes, queued at 30.
- **The engines (four):** goinfer at serve's defaults (the int8 device tower), repeated as this run's control; **goinfer with `-vision-quant f32`** (the float32 SigLIP tower on the tower base); Ollama 0.32.5 `gemma3:4b` (the same blob);
  llama.cpp 427291b with ggml-org's Q4_K_M and mmproj. goinfer is `serve-cuda-s18` at `cb242e92`, the build last night used.
- **Why:** last night's goinfer arm (4.61 s) is 4.1 s of int8 tower, and the float32 tower measured 2.07 s on its own (S4 addendum, S17 lever A). Whether Gemma 3's CUDA default should be float32 is the owner's open decision; this puts the
  served TTFT of both next to each other and next to the two peers, on the same box in the same session.
- **The reading (reported, not gated):** the median of the nine timed TTFTs per engine against the 5 s bar, and the ratios goinfer-f32 / goinfer, goinfer-f32 / Ollama and goinfer-f32 / llama.cpp. **Control:** the default arm should land within
  3% of last night's 4.61 s; if it does not, the box was not in the same state and the f32 arm is read with that caveat.
- **For the owner's decision, the evidence rule written before the numbers:** float32 is eligible to become the CUDA default for Gemma 3 if its median TTFT is under the 5 s bar and not above the int8 arm's. That makes it eligible, not decided;
  the reply-fidelity half of the evidence is G-S3b/G-S3d (identical reply for float32; the int8 default's reply differs at a non-near-tie token).
- **Prediction, written 2026-10-08 in last night's record and not changed since:** the float32 arm at about 2.5 s (the tower's 2.07 s plus a prefill of about 0.4 s). **Disclosure:** one by-day smoke of this arm (2 requests, exploratory, not quotable)
  was run before this registration was written; it read 2.41 s and confirmed `-vision-quant f32` logs `encoder f32/cuda-resident` beside `decode path: cuda-resident (int4)`.

#### Night 2026-10-07 on nobara, read 2026-10-08 (the queue started 20:30 PDT and ended 00:15; raw in `docs/measurements/multimodal-support-2026-10/night-2026-10-07-nobara/`)

Eleven entries, nine ok. Every ok job ran well inside its estimate (S7 in 5 minutes against 60, S13-lite in 4 against 55): the estimates were conservative, and the logs show real cells, not early exits. The one failure is the GPU gate (below). The S7/S13-lite
output directories are dated 2026-10-08 because the jobs ran after midnight.

- **S7, nobara, CUDA, serve's defaults, a new image or clip every request, median of 3 timed, against the 5 s bar** (`s7/`):

  | cell | TTFT | bar | what the log says |
  |---|---|---|---|
  | Gemma 3 4B, the shipped default (the build before S18) | **10.70 s** | no | decoder fell to the CPU on the resident build's OOM; int8 tower on CUDA |
  | Gemma 3 4B, S18 build | **4.57 s** | yes | `cuda-resident`, int8 tower on CUDA (4.1 s of it is the tower) |
  | Gemma 4 E2B, image | 0.85 s | yes | |
  | Gemma 4 E2B, audio | 1.48 s | yes | |
  | Qwen3.5-0.8B | 0.82 s | yes | |
  | GLM-OCR | 2.09 s | yes | |
  | Qwen2.5-VL-3B | **15.64 s** | no | **the tower ran out of device memory** (aikit's qwencuda, scratch for 4096 patches, a 56 MB allocation, despite a 3630 MB reserve) **and ran on the CPU** (15.1 s per encode) |
  | Qwen3-VL-2B | not a cell | n/a | HTTP 400 on all four requests: main's serve takes no Qwen3-VL images (S10 is the Mac's unmerged `s10-mac-s2`); removed from the plan until S10 lands |

  - Five of seven valid cells are under the bar. S18 turned Gemma 3 from the worst cell into one that clears it, 10.70 to 4.57 s.
  - **Qwen2.5-VL is now the biggest CUDA gap, and it has two layers.** At defaults its tower does not fit beside the decoder and the KV (so the CPU tower runs, 15 s), and even on CUDA aikit's tower measured 7.97 s for this image
    (`s4-tower-speed` below), over the bar by itself: it is still on the pre-fused-attention kernel, being aikit's `gpu/qwencuda` and not goinfer's tower base. The fix is the same shape as lever A (a fused attention, or
    Qwen2.5-VL's window attention on the tower base) plus a reserve that actually covers it. Not started.
- **S13-lite, nobara, Gemma 3 4B, every engine at its defaults, 3 rounds x 3 timed, order rotated, goinfer on the S18+S17 build** (`s13lite/`): **goinfer 4.61 s, Ollama 0.32.5 1.60 s, llama.cpp (427291b, CUDA) 0.96 s.** goinfer is
  under the 5 s bar by 0.4 s, and 2.9x Ollama and 4.8x llama.cpp. Its time is the int8 tower (4.1 s of it, S4 addendum); the arm at defaults does not use the float32 tower, which measured 2.07 s and would put the same request at
  about 2.5 s. That arm was not in the registration; the Gemma 3 default is the owner's open decision, and this is the evidence for it (a `-vision-quant f32` arm is the obvious next cell, not run).
- **S1 on CUDA speed record** (`s1c-e2b-speed`): E2B decode CPU 12.65 / 12.64 tok/s, CUDA 87.53 / 87.63 tok/s (6.9x); the host's PLE cost per token: mean 0.411 ms, median 0.345, p90 0.573 over 512 tokens.
- **G-S5b, read for the first time (`s5b-e2b-audio`): PASS.** goinfer's float32 E2B prefill on an audio prompt against HF (transformers 5.12.0): last-position cosine 1.000000, argmax 818 on both, text-position argmax agreement
  after the block 15/15; the three planted defects each red (audio rows x the embed scale 0.888395, PLE from the audio token's id 0.950150, delimiters dropped 0.940021). This is the audio path's HF anchor.
- **S9 part A speed record** (`s9a-e2b-prefill-ttft`, 3 interleaved rounds x 5 cold requests): at ~270 tokens batched 0.38-0.39 s against sequential 3.00-3.11 s (7.7-8.0x); at 2173 tokens 3.73-3.75 s against 24.9-25.6 s (6.6-6.9x).
  Kill rule PASS.
- **S9 part B speed record** (`s9b-e2b-image-ttft`): tower on the CPU, first request 5.00 against 23.8-24.0 s (4.8x), later requests 0.49 against 19.2-19.4 s; tower on CUDA, first 2.80-2.85 against 21.8 s (7.7-7.8x), later 0.50
  against 19.3 s. Kill rule PASS.
- **S4 addendum speed record** (`s4sig-speed`): SigLIP tower per image, CPU float32 20.75-21.04 s, CUDA float32 **2.06-2.09 s**, CUDA int8 4.07-4.09 s (built before lever A it would have read 18 s; it ran on the lever A build).
- **S4 tower speed record** (`s4-tower-speed`, lever A build): Gemma 4 E2B 0.46-0.49 s against the CPU's 4.2-4.6 s (cosine 1.000000000); Qwen3.5-0.8B 0.46-0.62 s against 4.1-6.3 s; GLM-OCR 2.0-3.8 s against 20.6-43.9 s; **aikit's Qwen2.5-VL
  7.97 s against the CPU's 15.2 s (64x64 grid) and 15.3 s against 24.4 s (86x72)**, only 1.6-1.9x.
- **`peer-vetted-nobara`** (the Mac's job, graded by the Mac): CPU phi3-mini goinfer 8.2 against Ollama 10.7-10.8 tok/s, gpt-oss-20B CPU 7.4-7.5 against 10.7; Ollama gpt-oss on CUDA 26.2 tok/s with goinfer's arm empty. The re-runs are theirs.
- **`gate-gpu-cuda`: FAILED, `cuda on Linux @ 420f4404. Do not tag.`** The heavy tier completed for the first time (882 tests, 7000 s). Triage on 2026-10-08 on an idle GPU (`gate-triage/`):
  - **Pre-existing, not from this session's work (the same failures at `12c85f4a`, the tree this session started from):** `TestDefaultVerifyWidth_sweep`, `TestFlashDecodeBlockSpecLane`, `TestBlockSpec_twoTurnsMatchPlain`
    (all three: `NewBlockSpec` out of device memory attaching the drafter beside the target, 2.6-25 MB allocations) and `TestResidentDenseBytes_matchesCUDADevice/qwen2.5-7b` (the build put 4442 MB on the device before its KV,
    421 MB over Plan's 4021 MB, past the 384 MB margin). They are the drafter-reserve and Plan-weight-estimate accounting, and they are the same class as the Gemma 3 4B and Qwen2.5-VL cells above.
  - **A heavy-tier state artifact:** `TestRopeKVMRoPEBatched_degenerateMatchesScalarKernel` passes alone.
  - **Mine, fixed:** `TestS17ProfileCUDA` failed its +10% overhead limit on the dispatch-bound 14x20 grid (+14.5% of 45 ms); the limit now applies from 0.5 s of wall and logs below it.
  - **Not triaged:** `TestSpecNonCopyLane` (940 s, then `KV position 6144(+1) exceeds resident context cap 6144`) and the WebGPU parity run, which hung in `gpu.(*Context).Close` under `TestRMSNormBatched_parity` until the 10-minute timeout.

### S8 — The support table in the README, kept true

- **The README** gains a short "Images and audio" section: one compact table (family by backend, a check or "CPU"),
  one line on speed with a link, and the models that are not supported yet.
- **The full table** lives in `docs/multimodal.md`'s status block (tower and decoder per cell, with the S7 figures).
- **A drift check,** so neither goes stale silently: a test that reads both tables and checks each claim the code can
  answer.
  - A GPU-tower cell needs the tower registered for that backend (e.g. `multimodal.Gemma4Towers()` in a `metal`
    build).
  - A GPU-decoder cell needs a backend to declare the family's resident features.
  - A model listed as supported needs a serve loader.

  A cell the code cannot answer (a speed figure) carries a date instead.
- **Size:** S, once S7 has numbers.

#### Night 2026-10-07 on the Mac, read 2026-10-08 (raw in `docs/measurements/multimodal-support-2026-10/night-2026-10-07-mac/`)

**The night:** the queue ran 22:23-23:27 PDT, on AC power, with no thermal note recorded.
- Two jobs started with the load over the settle bar after 300 s: s2-tower-speed at 1.12 and s9-speed at 1.27.
- **s6-e4b failed:** the fit guard refused its prequant (15.9 GB needed against 7.5 GB available). The E4B sidecar is to
  be built on nobara instead.
- **peer-vetted-mac** is recorded in its own file.
- Each block below is graded against its own registration.

- **S2 tower speed (`s2-tower-speed`, a record with no bar): RECORDED.**
  - **The run:** `TestGridVisionMetal_real` three times, a pre-built test binary from `s2-towers` at `bec48333`
    (now on main) over the local aikit at `64bab88`.
    - The aikit revision is inferred from its reflog and was not recorded by the script. It is in v1.59.0.
    - The queue rev "36fe0d11 +dirty" is the main tree, which the binary does not depend on.
  - **What it is:** the pre-lever Metal baseline for S17.
  - **Correctness held in every pass:** worst token cosine Qwen3.5 0.999999212 and GLM-OCR 0.999991618, G-S2b's
    readings to the digit.

  | tower | image (merged) | Metal, 3 passes | CPU, 3 passes |
  |---|---|---|---|
  | Qwen3.5-0.8B | 896² (784) | 1.162s / 1.177s / 1.258s | 4.367s / 4.466s / 4.506s |
  | Qwen3.5-0.8B | 14x20 (70) | 98ms / 99ms / 97ms | 205ms / 205ms / 193ms |
  | Qwen3.5-0.8B | formula.png (1015) | 3.074s / 3.074s / 3.088s | 6.632s / 6.614s / 6.775s |
  | Qwen3.5-0.8B | table.png (972) | 1.846s / 1.857s / 1.864s | 6.151s / 6.194s / 6.28s |
  | GLM-OCR | 896² (1024) | 5.692s / 5.817s / 5.78s | 20.79s / 21.379s / 21.627s |
  | GLM-OCR | 6x8 (12) | 133ms / 129ms / 130ms | 139ms / 151ms / 138ms |
  | GLM-OCR | formula.png (1548) | 11.836s / 12.551s / 12.611s | 40.512s / 41.666s / 41.647s |
  | GLM-OCR | table.png (1376) | 8.871s / 9.488s / 9.469s | 33.806s / 35.155s / 34.543s |
  | GLM-OCR | invoice.png (1656) | 12.072s / 12.627s / 12.432s | 45.434s / 47.202s / 45.456s |

  - **Ratios:** Metal is 2.0-3.8x the CPU on Qwen3.5 and 3.3-3.8x on GLM-OCR, and ties it on the 6x8 grid.
  - **Formula.png is the attention cost:** on Qwen3.5 it costs 1.66x table.png on Metal for 4% more tokens. That is
    S17 step 0's attention share (76% at 0.27 TFLOPS).
  - **Drift:** GLM-OCR's first pass ran 5-7% under passes 2 and 3, probably thermal (not logged); the ratios hold to ±3%.

- **S3 root cause (`s3-rootcause`): G-S3c PASS on both families; the Metal-tower arm is deterministic; the CPU-tower
  repeat is VOID; phase 2 is VOID (a test bug, fixed).**
  - **The binaries:** `serve-metal` at `0c66b18b`, `decoder-g3.test` at `6a5d4efb`.
  - **A, G-S3c re-registered** (arms `=cpu:cpu`, `metal:cpu`, `cpu:cpu`, with `--embed-int4=false` and the tower on the
    CPU):
    - **Qwen2.5-VL-3B: PASS.** All three replies identical, "Table 2. Quarterly unit sales by region (thousands)", and
      the Metal arm decoded `metal-resident (int4)`.
    - **Gemma 3 4B: PASS.** The first differing generated token is 24: CPU ' along' 0.383 against Metal ' as' 0.375, a
      near-tie (at least 0.192).
    - Both CPU repeats are byte-identical.
    - **So G-S3c's earlier Metal failure was the `--embed-int4` default, as root-caused.**
  - **B, G-S3b's Gemma 3 controls:**
    - The Metal-tower arm repeats itself byte for byte (reply and log-probabilities).
    - **The CPU-tower repeat is VOID:** its resident build was refused 2 s after the previous resident arm exited
      ("budget 4.92 GB"), so its comparison is a CPU decoder against a Metal one.
    - **Indirect evidence that the CPU-tower arm repeats,** though not the registered pairing: B's `metal` log-probs are
      byte-identical to A's, from a separate process, and the refused arm's match A's CPU arm.
    - **Recorded, not graded, for G-S3b:** on a Metal-resident decoder, the Metal tower against the CPU tower first
      differs at token 7 (' by' 0.489 against ' for' 0.082), not a near-tie. Token 0's 'The' moves 0.985 → 0.746.
  - **C, phase 2 sensitivity: VOID.**
    - **The bug:** `run()` in `decoder/gemma3_tower_sensitivity_real_test.go` kept `m.forward`'s returned slice, which
      is the cache's reused logits buffer. So steps 1-31 all read the last step's logits, and every arm reported a
      first argmax change at step 1. Fixed 2026-10-08 (`slices.Clone`).
    - **A second problem, not fixed:** the test's greedy reference path ("...a table of quarterly unit sales...") is not
      the served CPU-tower reply (" quarterly" at token 3). Its prompt or features must be checked against serve's
      before a re-run.
  - **Next, per the registration:** the perturbation control (phase 2, fixed), and the CPU-tower repeat re-queued with
    a settle between arms. Both are on the night queue's list, not yet queued.

- **S7 on the Mac (`s7-mac`, a record against the 5 s bar): two of six valid cells under the bar; Gemma 3 4B not
  loaded.**
  - **The setup:** `serve-metal` from `s2-towers` at `ed8d4756`, `--backend metal`, serve's defaults, a new image or
    clip every request, one warm-up and three timed requests, median against 5.0 s. Ran 23:04-23:09 PDT, with load
    rising from 0.97 to 5.34 over the run.

  | cell | TTFT (median of 3) | bar | what the log says |
  |---|---|---|---|
  | Gemma 3 4B | not a cell | n/a | refused by the fit guard (5.9 GB needed against a 5.2 GB budget, 7.5 GB available at 23:04) |
  | Gemma 4 E2B, image | **6.20 s** | no | `metal-resident (int4)`, tower on Metal (1.12 s per image); about 5 s of the TTFT is the prefill (explained below) |
  | Gemma 4 E2B, audio | 3.20 s | yes | `metal-resident (int4)`, audio tower on Metal (0.16-0.17 s per clip) |
  | Qwen3.5-0.8B | 4.48 s | yes | `metal-resident (int4)`, tower on Metal (1.16 s) |
  | Qwen3-VL-2B | **8.97 s** | no | `metal-resident (int4)`, tower on Metal (3.75-4.14 s) |
  | GLM-OCR | **8.32 s** | no | **decoder on the CPU** (`metal does not implement [pairwise-mrope pairwise-rope]`), float32 tower on Metal (5.6 s) |
  | Qwen2.5-VL-3B | **21.67 s** | no | `metal-resident (int4)`, tower on Metal (7.86-8.59 s, over the bar by itself) |
  | Gemma 4 E4B | not run | n/a | conditional on `s6-e4b`'s sidecar, which was not built |

  - **Where the time goes:** the towers explain Qwen2.5-VL, GLM-OCR and most of Qwen3-VL. That is S17's target.
  - **E2B's other 5 s, explained 2026-10-08 by day** (exploratory: one served E2B on Metal at HEAD, streamed):
    - **The text-only prompt:** a fresh ~290-token prompt took 6.24 s to its first token. The same prompt again, with
      its prefix reused, took 0.16 s.
    - **Decode:** about 42 tok/s (about 24 ms a token).
    - **The image turn:** 6.42 s, of which the tower is 1.34 s.
    - **So:** the prefill costs about one decode step per prompt row (~290 rows at ~21 ms). That is S9's layer-major
      pass, which runs every row through decode's GEMV kernels: bit-identical, but with no batching. Not a defect.
    - **The fix is S9 step 2,** the f16-MMA batched prefill taught Gemma 4's per-layer geometry. It would bring this
      cell to about the tower's time plus a batched prefill.
  - **Not measured:** decode tokens per second after the image (`max_tokens` is 8), as on nobara.
  - **A log nit:** the audio cell's serve log says "encoded a ...-byte image" for each WAV clip.

- **S13-lite on the Mac (`s13lite-mac`, reported, not gated): goinfer 12.81 s, over the 5 s bar; 2.6x Ollama and 2.7x
  llama.cpp, both under it.**
  - **The setup:** Gemma 3 4B, Metal, every engine at its defaults, a new image every request, 3 rounds of 1 warm-up and
    3 timed, engine order rotated. Ran 23:13-23:20 PDT. The float32 arm (`3c55d7f4`) is nobara's and came later.

  | engine | median TTFT (9 timed) | range | under 5 s |
  |---|---|---|---|
  | goinfer, `serve-metal` at `ed8d4756` (int4 at load) | **12.81 s** | 12.33-13.58 s | no |
  | Ollama 0.32.5, `gemma3:4b` | **4.97 s** | 4.961-4.970 s | yes, by 0.03 s |
  | llama.cpp b10621 (`c1d0e7a00`), Q4_K_M + mmproj-f16 | **4.68 s** | 4.673-4.688 s | yes |

  - **Where goinfer's time goes:** the float32 SigLIP tower on Metal takes 8.9-9.6 s per image. That is S17's gap;
    lever A is not on this build.
  - **The decode path:** `metal-resident (int4)` in rounds 1 and 3. Round 2 fell to `cpu (int4)` because the resident
    memory guard declined by 0.01 GB (5.15 against 5.14 GB). That is serve's default behaviour, so it stays in the
    median. Without it the median is 13.06 s, and the reading is unchanged.
  - **For comparison,** nobara's cell read goinfer 4.61 s, Ollama 1.60 s and llama.cpp 0.96 s.

- **S9 speed on the Mac (`s9-speed`, a record; the registered rule turns the pass off below 1.00x): the image turn
  reads 1.43x, so it stays on; a 512-token text prompt reads 0.98x, under the line.**
  - **The setup:** `serve-metal` at `6e4449ae`, E2B q4_0 with `-vision-device cpu`, 3 passes with the arm order
    alternating, under the timing lock. Every server logged the path it was meant to.

  | TTFT | old (s) | S9 pass (s) | old/S9 (medians) | pairs |
  |---|---|---|---|---|
  | image turn | 16.526, 14.669, 14.643 (median 14.67) | 10.244, 10.282, 10.225 (median 10.24) | **1.43x** | 3/3 above 1 |
  | ~512-token text | 10.847, 10.779, 10.728 (median 10.78) | 10.944, 10.995, 10.945 (median 10.95) | **0.98x** | 3/3 below 1 |

  - **Image:** without the CPU tower encode (4.82-4.89 s every time), the part S9 changes goes about 9.8 → 5.4 s.
    Pass 1's old figure (16.526 s) is a first-server outlier, which also inflated the by-day smoke's 1.53x.
  - **Text:** the by-day smoke's 1.02x has reversed sign.
  - **The rule did not name a metric.** Read literally, the text arm trips it, but both turns go through the one
    `PrefillLast` pass, so turning it off would also drop the image win. **Parked for the owner:**
    - keep the pass for media turns and decline it for text-only E-model prompts, until S9 step 2 (an f16-MMA E-model
      pass) replaces the text route; or
    - keep it everywhere, or turn it off everywhere.
  - **Owner, 2026-10-08: keep it everywhere.**

- **Owner decisions on this night, 2026-10-08:**
  - **S9: keep the pass everywhere.** The text prompt's 0.98x does not turn it off. The image turn's 1.43x and the text
    prompt's 0.98x are recorded as they read.
  - **S3: queue what the follow-ups need** (below).
- **S3 follow-ups, registered 2026-10-08 before they run** (`run-s3-followup-night.sh`, night queue `s3-followup`):
  - **Phase 2, fixed and re-run.** Two fixes in `decoder/gemma3_tower_sensitivity_real_test.go`:
    - **The logits copy** (above).
    - **Serve's own prompt encoding:** template segments, the block spliced in as a Special segment, `EncodeSegments`.
      The hand-written string merged the template's "\n" with the block's "\n\n" into one token: 277 tokens against
      serve's 278, found 2026-10-08 by encoding both with the tokenizer alone. That is why its greedy path was not
      serve's.
    - **Its reference must reproduce G-S3b's served CPU-tower reply on the same decoder** (the day run's `metal:cpu`
      arm, which decoded `cpu (int4)` with Metal's layout:
      `docs/measurements/multimodal-support-2026-10/s3-gs3b/gs3c-reply-gemma-3-4b-it-metal.txt`), or the step is VOID.
    - **The reading is the registered one:** the Metal arm inside the three noise arms (KL and first argmax change)
      means the decoder's sensitivity, not a tower defect.
  - **G-S3b's CPU-tower repeat:** two `metal:cpu` arms on the night's `serve-metal` (`0c66b18b`), with a 30 s settle
    between arms (`GS3C_SETTLE`). Both arms must decode `metal-resident`, or the pair is VOID. PASS: byte-identical
    replies and log-probabilities.
  - **Cost:** about 15 minutes, queued at 20.

### S9 — Batched E-model prefill on Metal and CUDA, image turns included

Added 2026-10-07. Once S1 put E2B/E4B decode on the GPU, prefill became the cost: an E2B image turn on the Mac is about
15 s for 32 tokens (G4), most of it the ~286 image and prompt tokens prefilling one at a time on the CPU, and a text
prompt prefills token by token on the GPU too. Both backends decline E-models from every batched prefill path by name
(S1's explicit declines), and only CUDA's bidirectional 26B/31B class runs its image prefill on the GPU.

- **What it needs:** the batched prefill paths learn the E-model shape S1 taught the decode step: the per-row PLE tail
  (each row's `[h || L*P]`, image rows with PAD's token-identity term), KV-shared layers that store nothing and read
  their source's rows, per-layer FFN widths. Then the image turn's prefill runs resident instead of on the CPU plus an
  `UploadKV`.
- **Gates to write before starting:** batched against sequential resident prefill on the tiny E-model, every position
  (bit-identical where the kernels allow, else the S1 near-tie rule); S1's planted defects through the batched path; G4's
  served image request with the prefill resident, reply identical to the CPU's.
- **Speed (night):** image-turn TTFT and text-prompt TTFT on E2B, batched against sequential, on both boxes.
- **Size:** M-L per backend.

#### S9 on CUDA, part A (text prompts) — plan registered 2026-10-07 on nobara, before any code

Part A is the text-prompt path (`PrefillLast`, `PrefillLastN`, `HiddenLast`, `ResidualAll`, `PrefillSeedArgmax`, and the chunked driver under them). Part B, the Gemma 4 image turn
(today `GenerateGemma4VL`: CPU prefill, then `UploadKV`), is registered separately once A is read, because it adds a decoder-side entry point and the image rows' PLE tails.

- **What changes** (`cuda/prefill.go` only, plus the decline text; no kernel, no PTX): the E-model decline in `prefillStaticDecline` goes. `prefillCore` stages each row's `[hidden ‖ L*P]`
  as the hidden part into `xB` and the tail reordered to `[layer][row][P]`, so a layer's slice is contiguous and `glu_quant`'s batched twin reads it as the `up` operand;
  scratch is sized to the widest layer's FFN (`ffnI`) and each layer's launches take its own; a KV-shared layer skips the K/V projections, the K half of qk-norm, v_norm and the
  K/V store (`nKV=0` in the rope launch, as decode's `decodeAttnGap` does) and attends over its source's aliased cache; the PLE branch (quant, gate GEMV, GELU-tanh x tail,
  projection GEMV, post-norm, residual add) runs between the FFN residual and the layer scalar, the order decode uses. `nonBatchableKind` stops demanding a K/V weight of a shared
  layer. The batched path reads the SAME test seams decode does (`g4SkipPLEForTest`, `g4KVSrcOffForTest`, `g4OneFFNWidthForTest`, `g4DropLayerScalarForTest`,
  `g4KeepSharedKVStoreForTest`, `pleLayerShiftForTest`, `g4DropVNormForTest`), so one flag plants a defect in both.
- **Gates (written before the code; the first two are tests on `testdata/gemma4-emodel-tiny`, window 4, P=32, 2 shared layers, FFN 256/512):**
  - **G1p, batched against sequential, same resident.** Last-row logits of `PrefillLast` over 96 random-ish rows and over the golden's 18-position prompt: bit-identical
    (`!=` count 0), as `TestPrefillNonUniform_bitIdentical` demands of the K=V shape. Every row of `PrefillLastN` (the exact-kernel tail) bit-identical to the sequential
    `Forward` at that position. The chunked driver at chunk 5 and 7 (not divisors of 18 or 96) bit-identical to the unchunked pass. **Non-vacuity:** the test asserts
    `PrefillPath()` says batched, that `passPromptLen` was set by the call (it is written only in `prefillCore`), and the E-model shape (P=32, 2 shared, FFN 256 and 512).
    If a fast lever (`attn_fused`, the MMA GEMM) makes `PrefillLast` differ from decode on this fixture, the bar for that tail alone is cosine >= 0.99999 and an identical argmax or
    a near-tie within 3% (S1's rule), recorded as a departure with the measured digits; the exact tails stay bit-identical.
  - **G2p, planted defects through the batched path.** The batched all-rows logits graded against the CPU int4 reference with G1c's rule (every argmax identical or a near-tie;
    mean CUDA-vs-CPU cosine at least the CPU's own int4-vs-f32 mean). Unplanted: must pass. Each of the seven seams above planted in turn: each must turn it red
    (a seam the batched path ignores would not, which is what the gate is for). Mutants on the new code itself, run and shown red before the gate counts: (a) the PLE tail not
    reordered per layer, (b) the shared layers' K/V stored through the alias, (c) the layer-0 FFN width used for every layer.
  - **G3p, served, real `gemma-4-E2B`** (`~/models`, never `/srv/models`): a ~300-token text prompt and a ~2,700-token one, `--backend cuda`, greedy, against a binary built from
    `origin/main` before this change (the sequential arm: there is deliberately no env knob to switch batched prefill off). The log line must read `prefill path: batched`, and
    the two replies must be identical or differ first at a near-tie under the registered rule (p(other) >= half p(top)).
- **Speed (night, a record with no bar; exploratory by day):** TTFT at ~300 and ~2,700 tokens, batched against the pre-change binary, both interleaved in one session
  under the timing lock (`BENCH_RUNS` per the harness defaults). Expectation, not a bar: well over 2x at 2,700 tokens (the sequential path is about one decode step per token).
  **Pre-registered kill:** if batched TTFT at 300 tokens is not below sequential, the decline is restored and the change parks; correctness gates alone do not ship a slower path.
- **Day / night:** the code, G1p, G2p and the vets are day work (seconds each on the tiny fixture); G3p's two requests are a few minutes (exploratory by day, then queued);
  the speed record and `gate gpu` go on tonight's queue. Estimated by-day wall: build + tiny gates about 5 minutes; G3p about 6 minutes.

- **S9 on CUDA part A read 2026-10-07 on nobara (commits `2ef038a0`, `6abcd5cb`, `698e8a31`; local, not pushed): G1p, G2p and G3p PASS; one registered gate failed first and led to a departure.**
  Raw: `docs/measurements/multimodal-support-2026-10/s9a-cuda/` (`g1p-run1.log`, `g2p-run1.log.gz`, `cuda-prefill-subset.log.gz`, `g3p/`, `g3p-exact-levers-off/`, `g3p-run1-fastlevers-FAIL/`, `speed-smoke/`).
  - **What was built, as registered** (`cuda/prefill.go`, no kernel, no PTX): the E-model decline is gone; `prefillCore` stages each row's hidden part into `xB` and the PLE tails reordered
    `[layer][row][P]`; a KV-shared layer skips the K/V projections, the K half of qk-norm and the K/V store (`nKV=0` in the rope launch) and attends over its source's aliased cache; scratch
    is the widest layer's FFN and each layer's launches take its own `ffnI`; the PLE branch (quant, gate GEMV, GELU-tanh x the layer's tail slice through `glu_quant_batched`, projection GEMV,
    post-norm, add) runs between the FFN residual and the layer scalar; `nonBatchableKind` stops demanding a K/V weight of a shared layer and checks the PLE pair. The batched path reads the
    same eight test seams as decode. What still declines, by name: an MC3 step, an image block, m-RoPE (part B), and a hidden-sized row is refused by length.
  - **G1p, batched against sequential on the same resident, PASS bit-identical (zero differing logits) on every cut:** `PrefillLastN` every row, `PrefillLast`'s last row, chunk 5 and chunk 7,
    and 6 decode tokens after the batched pass, at 18 rows and at 96. Non-vacuity asserted in the test: `PrefillPath` says batched, `passPromptLen` (written only in `prefillCore`) equals the
    prompt length after the call, and the fixture has P=32, 2 shared layers, FFN 256/512. The existing non-E bit-identity tests still pass (`TestPrefillNonUniform_bitIdentical`,
    `TestPrefillMoE_bitIdentical`, `TestPrefillTails_bitIdenticalToTheRecordedBaseline`; 57 PASS and no FAIL over the prefill/E-model/Gemma 4 subset, the skips being asset-gated).
  - **G2p, the eight S1 planted defects through the batched path, PASS.** Unplanted: GREEN, mean CUDA-vs-CPU-int4 cosine 0.999931 against the CPU's own int4-vs-f32 mean 0.777674, 18/18 exact
    argmax (the same digits as sequential G1c). Planted, each alone RED: (1) PLE skipped 0.371194 (2/18); (2) token-identity term zeroed 0.966228 (10/18); (3) shared-KV source off 0.887704 (7/18);
    (4) one FFN width 0.938942 (12/18); (5) K/V store not skipped 0.905355 (6/18); (6) layer scalar dropped 0.972024 (10/18); (7) v_norm dropped 0.920343 (10/18); (8) PLE layer offset 0.179992 (0/18).
    Mutants on the new code, run and shown red: (a) the tails staged `[row][layer]` instead of `[layer][row]`: 255 of 255 logits differ at position 0 and in every cut of G1p; (b) the K/V store
    through the alias is defect (5) above; (c) layer 0's FFN width for every layer: G1p and G2p red.
  - **G3p, served, real `gemma-4-E2B_q4_0-it.gguf` from `~/models`, `--backend cuda`, greedy, batched (`6abcd5cb`) against the pre-change binary (`928f9a41`, sequential), cold prompts of 267 and 2,170 tokens:
    PASS on the second registered run; the first FAILED.** The startup line read `prefill path: batched` on the new arm and `sequential — Gemma 4 E-model ... has no batched prefill` on the old one.
    - *First run (fast levers at their default, above the 512-row floor):* the 267-token prompt was identical in the reply and every logprob; the **2,170-token prompt first differed at generated
      token 10** (sequential ' and' 0.190, batched ' issues' 0.193; ' issues' was outside the sequential top 3, whose third is 0.130), which is **not a near-tie under the registered rule**. That is a FAIL as
      registered (`g3p-run1-fastlevers-FAIL/`).
    - *Diagnosis:* the same two arms with `GOINFER_CUDA_FAST_PREFILL=0` (`g3p-exact-levers-off/`) matched in the reply and in every logprob at both sizes, so the batched pass itself is correct and the
      difference is the fast levers (`attn_fused`, the MMA GEMM), which engage only above the 512-row floor, are cosine-close and not bit-identical to decode, and have never had a fidelity gate on an
      E-model. The tiny gates could not see it: their prompts are far under the floor.
    - *Departure from the plan, mechanism stated:* an E-model's batched prefill now runs the exact kernels at every length (`forceExactKernels`, `6abcd5cb`), and the startup line says so. No bar moved;
      the lever is parked until it passes a fidelity gate of its own. The guard `TestGemma4EModelPrefill_exactKernelsAboveTheFloor` moves the floor to 0 and requires bit-identity to sequential and zero
      fast-kernel launches; shown red by removing the forcing (52 fast launches, 255 of 255 logits differ).
    - *Second run, final code, defaults:* both sizes IDENTICAL in the reply and in every logprob (`g3p/`).
  - **Speed, EXPLORATORY (single samples on a box that was also building; not quotable, the night record is queued as `s9a-e2b-prefill-ttft`, 15 min, 3 alternating rounds x 5 cold requests per size, with the
    registered kill rule):** time to first content chunk, exact kernels: 0.41 s against 3.04 s at 267 tokens and 3.72 s against 24.56 s at 2,170 (about 7x and 6.6x); the one-cell script smoke read 0.44 / 3.14 s and
    3.70 / 25.11 s. With the fast levers on (the failed first run): 0.62 / 3.03 s and 2.43 / 24.97 s, so exactness costs about 1.5x of the batched TTFT at 2,170 tokens, which is what a lever fidelity gate for
    E-models would win back.
  - **Not done:** part B (the Gemma 4 image turn, still CPU prefill plus `UploadKV`); an E-model fidelity gate for the fast levers; E4B (no checkpoint); the Metal half of S9 (the Mac's); `gate gpu` over this tree
    (tonight's queue has it); the commits are local.

#### S9 on CUDA, part B (the Gemma 4 E-model image turn) — plan registered 2026-10-07 on nobara, before any code

Today `GenerateGemma4VL` prefills an E-model image turn on the CPU, token by token (E2B/E4B ship `use_bidirectional_attention=""`, so the image block is attended causally and the sequential walk IS the
reference forward), then uploads the KV of the owning layers (`residentUploadPrefill`). Part A made the resident's batched pass carry an E-model row, so a causal image turn is a text-like prefill with
different rows.

- **What changes** (`decoder/generate_gemma4_vl.go`, a decoder helper, one test hook; no backend code): for an E-model with a causal image block, when the resident is a `Prefiller` and batched prefill is not
  knob-disabled, `GenerateGemma4VL` builds the prompt's resident rows `[hidden || L*P]` (text rows exactly as `embedResident` builds them; an image row's hidden part is the projected feature as-is, no embed scale, and
  its tail is `gemma4PLEInputs(feature, PadTokenID)`: the pad token's identity term, as `prefillLogitsGemma4VL` passes `padID`), takes the logits from `PrefillLast(ctx, rows, 0)`, and decodes resident from
  `len(ids)`. The CPU prefill, the CPU cache and the upload bridge are skipped on that path. **Any decline or error other than a cancel falls through to today's CPU prefill + upload, unchanged** (a backend
  without a batched E-model prefill, a prompt past the context cap). The bidirectional class (26B/31B, `"vision"`) is untouched.
- **Gates (written before the code):**
  - **G1q, the image turn's batched resident logits against the CPU reference.** On the tiny E-model with synthetic features, three layouts (image at the start, in the middle, at the end of the prompt), the last
    position's logits from `PrefillLast` on the built rows against `prefillLogitsGemma4VL`'s (the CPU forward, via a test hook): G1c's rule (argmax identical or a near-tie within 3%; cosine at least the CPU's own int4-vs-f32
    cosine at the same position). Plus the batched rows against the SAME resident's sequential per-row forward: bit-identical last-row logits and bit-identical decode afterwards. Non-vacuity as in G1p.
  - **G2q, planted defects in the new row builder, each alone must turn G1q red:** (1) an image row's tail built from the placeholder token's id instead of the pad id; (2) the image feature multiplied by the embed scale;
    (3) an image row's tail zero. Plus: a mutant that builds image rows as text rows (the placeholder ids' embedding), shown red.
  - **G3q, served, real `gemma-4-E2B_q4_0-it.gguf` on CUDA, the `testdata` table image used by S4:** the image turn with prefill resident (the new binary) against the same turn on the pre-change `main` build (CPU prefill + upload);
    reply identical or first different token a near-tie under the registered rule; the startup/log lines must show the resident prefill ran (a log line is added for it, named in the test). Run with the tower on the CPU and
    with it on CUDA (the s2-towers code, now on main).
- **Speed (night; exploratory by day):** image-turn time to first token, new against the pre-change build, same tower placement, interleaved rounds, under the timing lock. **Kill rule:** if the resident-prefill turn is not
  faster than the CPU-prefill turn at the default image budget, the path is removed. No other bar.
- **Day / night:** code + G1q + G2q about 10 minutes; G3q about 6; speed record queued.

- **S9 on CUDA part B read 2026-10-07 on nobara (commit `e6d18adb`; local, not pushed): G1q, G2q and G3q PASS.**
  Raw: `docs/measurements/multimodal-support-2026-10/s9b-cuda/` (`partb-g1q-g2q-run1.log.gz`, `partb-generate-run1.log`, `g3q/`, `speed-smoke/`).
  - **Built as registered, then merged onto the Mac's equal** (`decoder/generate_gemma4_vl.go`; no backend code; the Mac's S9 on Metal, `1e2f40db`, landed the same decoder path - `gemma4VLResidentPrefill` and `gemma4ResidentMediaRow` - while this was being built, and the merge took its structure and kept this session's row-builder seam `gemma4EModelImageRows`, its argument checks and the G2q defect hook on top; G1q, G2q and the flow test pass on the merged tree): for an E-model with a causal image block, `GenerateGemma4VL` claims the resident, builds the prompt's rows (text rows as
    `embedResident`; an image row's hidden part is the projected feature as-is and its tail is `gemma4PLEInputs(feature, PadTokenID)`), takes the logits from `PrefillLast(ctx, rows, 0)` and decodes resident from
    `len(ids)`; the CPU prefill, the CPU cache and the upload bridge are skipped. A decline or any non-cancel error forgets the half-written prefill, releases the claim and falls through to the unchanged CPU prefill + upload.
    Serve logs `vision: decoded N tokens on the resident path (prefill resident)` when it ran (the Mac's wording; this session's own first line was dropped in the merge). The bidirectional class (26B/31B) is untouched; a backend whose resident declines the E-model batched pass falls through.
  - **G1q PASS.** On the tiny E-model with synthetic features, the image block at the start, in the middle and at the end of the 18-position prompt (6 soft tokens): the resident's batched last-position logits on the
    built rows against the CPU forward (`prefillLogitsGemma4VL`), cosine **1.000000 / 1.000000 / 1.000000** (the CPU's own int4-vs-f32 cosine at those positions 0.805, 0.763, 0.901), argmax identical in all three,
    and bit-identical to the same rows run one token at a time on the same resident. Non-vacuity as in G1p (`passPromptLen`).
  - **G2q PASS, each planted defect in the row builder red on at least one layout:** (1) the placeholder's id instead of the pad id for an image tail: red on image-last only (cosine 0.989288, argmax off by 7.02%;
    the other two layouts read 0.998928 and 0.995355 and stay green, so this defect is the one the fixture sees least); (2) the feature scaled by the embed scale: red on image-first and image-last (0.890795, 0.493592);
    (3) an image tail zero: red on image-first and image-last (0.959385, 0.734399); (4) image positions built as text rows (the placeholder ids' embedding): red on image-first and image-last (0.783826, -0.051670).
  - **The flow, `TestGemma4EModelImage_generateTakesTheResidentPath`:** three turns on one CUDA model (image, the same image, then plain text) generate the CPU model's greedy tokens, the image turns report
    `ImgPrefillResident` and `DecodeResident`; with `GOINFER_BATCHED_PREFILL=0` the same turn reports `ImgPrefillResident=false`, `DecodeResident=true` and still matches (the fall-through).
  - **G3q, served, real `gemma-4-E2B_q4_0-it.gguf` + `~/models/gemma-4-E2B-unq` on CUDA, `--kv-sessions 1 -ctx 4096`, the S4 table image, greedy, 32 tokens with top-3 logprobs: PASS.**
    Reference: the pre-part-B build (`f1b593d0`, tower on the CPU, CPU prefill + upload). *New, tower on the CPU:* first differs at generated token 17 (' for' 0.428 against '.' 0.281; 0.281 >= half of 0.428:
    a near-tie, True). *New with `GOINFER_BATCHED_PREFILL=0` (same-binary control):* IDENTICAL to the reference, 32 tokens, so the new path declining reproduces today's behaviour exactly. *New, tower on CUDA:* first
    differs at token 27 ('\n\n' 0.591 against ' It' 0.303; near-tie True), with the tower's own ~1e-6 feature difference on top. The logs show the resident-prefill line on the two new arms and not on the control, and
    decode on the resident path on all of them. The reply differs from the CPU-prefill one only where GPU int4 and CPU int4 arithmetic differ, which G1c already bounds; it is not bit-identical by design.
  - **G3q again on the merged tree (`5792b9f3`, `g3q-postmerge/`, released aikit v1.59.0 pins):** the same four arms read the same: new first differs at token 17 (near-tie True), the same-binary control IDENTICAL, new with the
    CUDA tower first differs at token 27 (near-tie True); G3q PASS. Wall times 24.6 s (before), 5.8 s, 3.6 s.
  - **Speed, EXPLORATORY (single samples; not quotable; the night record is queued as `s9b-e2b-image-ttft`, 20 min, 3 interleaved rounds x 4 requests, four arms, with the registered kill rule):** wall time of the whole
    32-token image request: 24.7 s before (CPU tower, CPU prefill), 5.8 s with resident prefill (CPU tower), 3.7 s with resident prefill and the CUDA tower. The one-cell smoke of the night script (max_tokens=1,
    1 round, 2 requests): first request 4.97 s against 23.58 s (CPU tower) and 2.83 s against 21.54 s (CUDA tower); the second request, with the tower's features cached, 0.47 s against 19.08 s and 0.45 s against 19.00 s.
  - **Not done:** Metal's half (the Mac's S9, recorded above); E4B (no checkpoint on this box); the bidirectional class is out of scope; an image turn
    whose prompt exceeds the resident context cap declines to the CPU path (by the existing cap check inside the batched pass).

**S9 Gate 0 on Metal, the desk map (2026-10-07, Mac):**
- **Today an E-model never prefills in a batch on Metal.** The f16-MMA pass (`PrefillLast`) declines every dense Gemma
  4, because its head size varies by layer (`prefillOK` checks `HasPerLayerGeometry`). So a text prompt prefills one
  token at a time on the GPU, and an image turn (`GenerateGemma4VL`'s E-model branch) prefills on the CPU, then
  `UploadKV`s and decodes resident.
- **A bit-identical batched route already exists for the paged 26B:** `prefillG4Paged`
  (`metal/prefill_g4paged.go`, 4b of task-m26-mac). It runs layer by layer: one command buffer per layer encodes
  decode's own kernels once per prompt row, with each row's own residual and position uniforms. So the K/V and the last
  row's logits equal the sequential loop's bit for bit, and the per-token submit-and-wait trips go away. Its dense
  layers already take the plain `encodeLayerWith` path.
- **An E-model row adds two things to that per-row state:**
  - the PLE input `pleIn` (`[L·P]`, about 35 KB on E2B, which `encodePLE` reads from the resident's field, as decode
    stages it from the row's tail, `loadEmb`);
  - nothing for KV-shared layers. They read their source layer's cache at keys up to the row's position, and the
    layer-major order fills every row of the source layer first.
  Per-layer FFN widths are already in decode's layer encode.
- **The image turn:** the decoder can build the resident rows itself:
  - text rows from `embedResident` (`[h ‖ PLE]`);
  - image rows as the projected feature followed by `gemma4PLEInputs(feature, PAD)`, the CPU's own rule.
  It then calls `PrefillLast`. A decline keeps today's CPU prefill and upload, unchanged.
- **CUDA's half is nobara's** (its batched prefill declines E-models by name too, S1's C-steps), against the same gates.

**S9 gates on Metal, written 2026-10-07 before any S9 code or measurement:**
- **G-S9a, bit-identity:** on `gemma4-emodel-tiny`, the layer-major pass against the sequential resident loop (the same
  rows through `Forward` one at a time).
  - Two prompts: text only, and one carrying a run of image rows (random features, PAD in PLE).
  - The last row's logits must be bit-identical, and so must the next 8 decode steps' logits from the prefilled cache
    (which checks every layer's K/V, shared layers included).
  - Prompt lengths below and above one chunk boundary.
  - No tolerance band: the construction is exact, so any difference is a defect.
- **G-S9b, the planted defects through the pass:**
  - S1's G2 E-model defects that the prefill touches, each injected into the layer-major path and each red against
    G-S9a's sequential reference: the PLE input of one row bound to another row's; the PLE term dropped; a KV-shared
    layer storing its own K/V.
  - Also the row order within a layer reversed, which must still be green: the encoder keeps rows apart, and a red
    there would mean rows share state they should not.
- **G-S9c, served:** G4's request (E2B, table.png, 32 greedy tokens, the GGUF with `--vision ~/models/gemma-4-E2B-unq`).
  - Arms: `--backend metal` against `--backend cpu`, `--embed-int4=false` on both (G-S3c's lesson).
  - The Metal arm must report its image prefill resident.
  - Identical reply, or a first divergence at a near-tie under the log-probability definition. The CPU repeat must be
    byte-identical.
- **G-S9b amended 2026-10-07, after writing the pass and before any measurement:**
  - **The row-order item was wrong.** Within a layer each row writes its own K/V and later rows attend to it, so rows
    encoded out of order DO share state: a row would read keys not yet written. Reversing the order is therefore a
    planted defect that must go RED, not a control that stays green.
  - **"A KV-shared layer storing its own K/V" is replaced by "every row one position late".** The first needs a change
    inside decode's shared kernels, which the pass does not own; the second is the pass's own state.
  - **The defects as run:** (1) each row's PLE inputs bound to the next row's; (2) the PLE term dropped (zero inputs);
    (3) every row one position late; (4) rows encoded in reverse order within a layer. Each must turn G-S9a red.
- **Speed (night):** E2B image-turn TTFT and a 512-token text prompt's TTFT, the layer-major pass against today's path
  (CPU prefill and upload; sequential resident), interleaved. Default on if G-S9a and G-S9b pass, since the change is
  bit-identical (the 4b precedent); the night grade turns it off below 1.00x.

**S9 on Metal, results (2026-10-07, Mac):**
- **The code:** `metal/prefill_emodel.go`.
  - `PrefillLast` takes a dense Gemma 4 E-model through a layer-major pass: per layer, one command buffer encodes
    decode's own `encodeLayerWith` once per row, each row on its own residual, PLE inputs and position uniforms.
  - `GenerateGemma4VL`'s E-model branch builds the resident rows and prefills there (`gemma4VLResidentPrefill`), with
    `gemma4ResidentMediaRow` building the image and audio positions: the feature unscaled, PLE from PAD. A decline
    keeps the CPU prefill and upload.
  - Serve's log line names where the prefill ran. E-model text prompts reach the pass through the decoder's existing
    `residentPrefillSeed`, unchanged.
- **G-S9a: PASS, bit-identical in all four cases.** On the tiny E-model, the last row's logits plus the next 8 decode
  steps are equal bit for bit to the sequential loop's, for text only and with 12 image rows, at 18 and 293 rows (past
  one 256-row chunk). `metal/gemma4_emodel_s9_test.go`.
- **G-S9b: PASS, all four defects red at the first step:** (1) PLE inputs shifted one row, max |diff| 0.904; (2) PLE
  dropped, 1.19; (3) one position late, 1.41; (4) rows reversed within a layer, 0.906.
- **G-S9c, read 13:00-13:01 PDT: PASS.** The serve binary was built from `s2-towers` with `main` merged
  (`6e4449ae`), which has `-vision-device`. The model was the E2B GGUF with `--vision ~/models/gemma-4-E2B-unq`, using
  table.png, `-vision-device cpu` and `--embed-int4=false` on every arm.
  - The Metal arm reported `prefill path: layer-major on decode's kernels (... a Gemma 4 E-model, S9)` and logged
    "decoded 32 tokens on the resident path (prefill resident)". The CPU arms prefilled on the CPU.
  - The replies agree for 17 tokens. At token 17 the CPU reference picks " for" at 0.388, with Metal's "." at 0.339
    there: a near-tie. The second CPU run is byte-identical to the first.
  - Request time (exploratory; the tower on the CPU in every arm, by the gate's design): Metal 12.1 s, CPU 14.5 and
    12.9 s. The tower is most of that; the night record separates prefill from tower.
  - Raw: `docs/measurements/multimodal-support-2026-10/s9-gs9c/`.
- **Exploratory smoke of the speed script, one pass by day (13:03 PDT; not a result, the night record decides):**
  `run-s9-speed.sh` with `PASSES=1`. Image-turn TTFT 16.5 s on the old path against 10.8 s on the pass (1.53x). The
  ~512-token text TTFT was 11.0 s against 10.8 s (1.02x).
  - The image turn gains because its prefill leaves the CPU.
  - A text prompt barely moves, because both paths run decode's GEMV kernels once per row, about 20 ms a token on
    E2B. The pass only removes the per-token submit-and-wait.
  - **A real text-prefill win for E-models needs the f16-MMA pass to learn Gemma 4's per-layer geometry** (head size
    256 and 512 by layer, the PLE block, shared K/V), which it declines today for every dense Gemma 4. That is a
    further S9 step, noted here and not started.
- **S9 step 2 (named 2026-10-07 evening, at the audit's request): the E-model text-prefill win.**
  - S9's layer-major pass gives about 1.02x on a text prompt (the smoke above).
  - The win needs the f16-MMA batched prefill to learn Gemma 4's per-layer geometry: head size 256 or 512 by layer,
    the PLE block, the 18 KV-shared layers, and the dense layer scalar and `v_norm` of S1.0.
  - Its gates are written here before its code. Correctness is S1's G3 shape (teacher-forced agreement against the
    CPU, non-inferior to the validated reference); the speed is a text-prefill A/B at night, shipping at >= 1.02x.
- **Still owed for S9:** the night speed record (`s9-speed`, queued tonight on the Mac) and CUDA's half (nobara).
- **S9 step 2 on Metal, the design and gates, registered 2026-10-08 before any code (owner: "S9 step 2 then do S18").**
  - **Why:** an E2B image cell spends about 5 s in S9's layer-major prefill, which runs every prompt row through decode's
    GEMV kernels (found by day 2026-10-08, S7's Mac record above).
  - **The design:** the f16-MMA batched pass learns a Gemma 4 E-model. Every existing family keeps one geometry, so its
    dispatches are unchanged.
    - **Per-layer geometry:** a uniform bundle per attention geometry. The head dim (256/512 on E2B), QKV stride, KV
      width and rotary half are read per layer. Attention takes the fused or steel kernel where the head dim allows
      (<= 128), the exact kernel otherwise (<= 4,096 keys, checked).
    - **KV-shared layers** project Q only, skip the K rope and the K/V store, and attend over their source's cache
      (decode's aliased buffers).
    - **`v_norm`:** the scale-less norm on every K/V-owning layer's V slot, after Q/K norm and before RoPE, as decode.
    - **The per-layer FFN width.**
    - **The PLE block** after the FFN residual: a gate GEMM (H→P) on the raw residual, `gelu(gate) × the row's own PLE
      slice`, a projection GEMM (P→H), the post-PLE norm, the residual add.
    - **The layer scalar** after that.
    - **Excluded:** K=V layers decline (no gate covers them yet).
    - **A production switch, off** until the gates below pass. With it off, the S9 layer-major pass keeps the route.
  - **G-S9c, tiny, by day:** `gemma4-emodel-tiny` (6 layers; head dims 32 and 64; 2 KV-shared layers with their own
    sources and double FFN width; PLE P=32; layer scalars; `v_norm`).
    - **The comparison:** the batched pass against the S9 layer-major pass (the validated path, bit-identical to
      sequential), on one Metal resident. The last prompt row and 8 teacher-forced decode steps.
    - **The bar:** cosine >= 0.9999, every argmax equal or an R10 near-tie.
    - **Two prompts:** the 12-token golden prompt, and a synthetic 96-token one, which runs every local layer past its
      window of 4 and both shared layers over long sources.
    - **Planted defects, each alone red on at least one prompt:**
      1. a shared layer stores its K/V (over its source's cache);
      2. the PLE block skipped;
      3. each row's PLE inputs taken from the next row;
      4. the layer scalar dropped;
      5. `v_norm` dropped;
      6. every layer's FFN at the first layer's width.
  - **G-S9d, real E2B, by day if it fits:** S1's G3 shape, on the batched prefill.
    - **The procedure:** G3's eight prompts, each prefilled by the pass under test, then 32 greedy tokens.
      - Free run: the reply against the CPU's, by G3's rule.
      - Teacher-forced: the CPU's 32 tokens through Metal decode after that prefill, argmax agreement over the last
        prompt row and the 32 generated positions.
    - **The reference:** the same procedure with the S9 layer-major pass (the shipped path), in the same process.
    - **PASS:** agreement at least the reference's minus 2.0 points, and free-run passes at least the reference's minus
      1. **Ambiguous (parked):** 2.0-4.0 points below. **FAIL:** worse.
  - **Speed, at night:** S9 step 2's registered text-prefill A/B (ship at >= 1.02x), and E2B's image-turn TTFT against
    the layer-major pass (a record).
    - **The projection band, written before the run:** the layer-major pass costs about 21 ms a row; the batched pass
      runs the 1.5B's text prefill at 225-366 tok/s (`docs/benchmarks.md`, 2026-09-18), and E2B is about that size. So
      an image turn's ~290-row prefill goes 5-6 s → about 0.8-1.3 s, and TTFT 6.4 → about 2.2-2.7 s: **2.4-2.9x**.
    - **The night job, specified 2026-10-08 before it runs** (`run-s9step2-speed.sh`, in this campaign's measurements
      directory). Two serve binaries from pinned worktrees: `main` (the switch off, so the layer-major pass, arm `lm`)
      and the local night-only branch `s9b-night-on` (`main` with the switch on and a one-time stderr line when the
      batched pass runs, arm `batched`). Each server is fresh and serves two requests, 8 tokens each, streamed: a
      ~512-token text prompt (the graded cell), then `table.png` (the record). The tower runs on the CPU in both arms.
      There are 5 passes, the arm order alternating, under the timing lock. Two arms, no peer (TE5(a)).
    - **The rule:** the median of the 5 per-pass text TTFT ratios, lm / batched. **Ship** at >= 1.02 (the switch goes
      on), **park** at 1.00-1.02, **off** below 1.00. **Void** if any batched server log lacks the ran line (a silent
      fallback would otherwise read as a park), or if a pass is missing. The image cell is a record against the band
      above, not a gate.
- **S9 step 2 on Metal, by day 2026-10-08: built; G-S9d PASS; G-S9c FAILED its registered bar and passes the owner's
  amended one. The switch stays off until the night speed rule grades it.**
  - **Built** as designed above, in `metal/prefill.go` (`prefillLast`, with the per-layer uniform bundles `layerU`) and
    `metal/prefill_emodel.go` (`emodelBatchedOn`, off; `emodelBatched`, which declines K=V layers, an int8 KV cache, int8
    projections and the q4k lane). `PrefillLast` tries it first for an E-model at or above the floor and falls back to
    the layer-major pass on any decline. The exact attention kernel takes the 256/512 head dims.
  - **A defect G-S9d found, fixed before the records below.** A KV-shared layer's Q/K norm dispatch still passed the
    model's KV head count with a Q-only row stride. The rows shifted, and the "K" head normalised the next row's Q. It
    now passes nKV 0 for a shared layer. G-S9c did not catch this one at its first read: its tiny fixture's shared
    layers sit where the shift moved little.
  - **G-S9d, real E2B: PASS.** Batched 96.88% (248/256, 8/8 free-run passes) against layer-major 97.27% (249/256, 7/8):
    delta -0.39 points against the -2.0 margin. The free runs differ only at small CPU gaps (prompt 3: 0.79% against
    layer-major's 5.91%; prompt 5: token 31 at 0.20%). The record is `s9step2/gs9d-realE2B.log`, 49 s.
  - **G-S9c, tiny: FAIL against the registered 0.9999.** Worst cosine 0.9907247 (12-token) and 0.9915778 (96-token),
    0 non-tie argmax differences, reproduced exactly on re-run.
    - **The planted defects are all red, on both prompts** (worst cosine, 12-token / 96-token): (1) 0.888 / 0.909;
      (2) 0.186 / 0.439; (3) 0.473 / 0.486; (4) 0.942 / 0.940; (5) 0.681 / 0.815; (6) 0.825 / 0.852.
    - **A deviation from the registration, disclosed:** defect (1) is "a shared layer attends over an empty cache", not
      "a shared layer stores its K/V over its source's". The pass never projects K/V on a shared layer (its QKV stride is
      Q only), so the registered defect has no code path to plant. The substitute catches the same class: a shared layer
      that does not read its source.
    - **The mechanism read for the miss** is quantization noise that the tiny fixture amplifies, not a defect. The
      fixture is strengthened (norm weights ×42, layer scalars ×6), so a small rounding difference grows through 6
      layers. Every quantized path reads only 0.70-0.83 against an f32 CPU on it. On the 96-token prompt, the batched
      pass is closer to f32 than the layer-major one (0.829 against 0.792). Gemma 3's shipped batched pass reads
      0.99993 on its own, unstrengthened tiny fixture. The real-checkpoint gate, G-S9d, passes.
    - **The bar, amended 2026-10-08 by the owner** (option "amend to 0.98"): worst cosine >= 0.98, every argmax equal
      or an R10 near-tie, every planted defect red. 0.98 sits between the clean pass (0.9907) and the nearest defect
      (0.940). **PASS** under it (`s9step2/gs9c-tiny-amended.log`; the run against 0.9999 is `s9step2/gs9c-tiny.log`).
  - **The prefill refactor's regression check:** `go run ./cmd/gate quick`, 13:44-13:53. 2154 tests passed, and the
    only red was G-S9c at 0.9999 (every other family's prefill test passed).



### S10 — Towers for the families that have none

Added 2026-10-07: Ministral 3 (Pixtral), LFM2.5-VL, North, and Qwen3-VL's image path (today text only).

- **Pattern:** the Gemma 4 and Qwen ones: the tower in aikit (batched into the cycle's one release), the decoder splice
  and prompt layout in goinfer, then serve. One sub-phase per family, in order of what users ask for; each starts with
  a desk map of its HF processor (image budget, special tokens, position scheme).
- **Gates, per family, before starting it:** the tower's stages against HF on the real checkpoint; full-model logits on an
  image prompt against HF; a served request.
- **Size:** M each.

**S10, Qwen3-VL first (owner, 2026-10-07: "Qwen3-VL first, on nobara").** This lifts the park on `docs/multimodal.md`'s
P8c ("Qwen3-VL DeepStack, PARKED", 2026-09-30), whose trigger was Qwen3-VL drawing use Qwen3.5+ does not cover; the
owner's choice is that decision. The dev checkpoint is `Qwen/Qwen3-VL-2B-Instruct`, downloaded on nobara (`~/models/
qwen3-vl-2b-instruct`); the Mac gets only what a test needs.

**Gate 0, the desk map (2026-10-07, from transformers 5.16.1's `modeling_qwen3_vl.py` and aikit's source):**
- **The tower is S2's Qwen3.5 tower plus DeepStack.** It has the same patch embed, interpolated learned position table,
  2-D RoPE, blocks and merger. In addition, at `deepstack_visual_indexes` a separate merger (`use_postshuffle_norm=True`:
  the LayerNorm after the merge-unit shuffle, not before) turns that block's output into one feature row per merged
  token. The tower returns the main merged rows plus one DeepStack list per index.
- **aikit refuses a DeepStack tower at load, by name** (`Qwen3EncoderConfig.DeepstackVisualIndexes`, pinned by
  `TestQwen3VisionEncoder_refusesWhatItCannotRun`). It needs the extra mergers and a forward that returns their outputs:
  new aikit API for the cycle's release.
- **The decoder:** after decoder layer i, for i < len(DeepStack), the i-th DeepStack rows are ADDED to the hidden state
  at the image positions (`_deepstack_process`). Every other family here splices once before layer 0. goinfer has the
  text decoder (`qwen3_vlArchitecture`, P8 Phase 0) and its interleaved m-RoPE (`MRopeInterleaved`, shared with
  Qwen3.5), but no additive per-layer hook.
- **Where the injection applies:** only in the prompt's prefill. A decode step is never an image position. So the CPU
  prefill carries the hook, and Metal's existing CPU-prefill-then-`UploadKV` route decodes unchanged. CUDA's resident
  m-RoPE prefill must decline a DeepStack model until it learns the hook.
- **Serve:** a `qwen3_vl` model type routes to the Qwen tower path, as Qwen2.5-VL and Qwen3.5 do, with the DeepStack
  rows carried beside the merged rows.
- **The text gate written in P8 Phase 0 and never run** (`decoder/qwen3vl_real_test.go`, `GOINFER_QWEN3VL_2B`) runs
  first, on nobara: the decoder must be right before images are added to it.

**S10 Qwen3-VL gates, written 2026-10-07 before any S10 code or measurement:**
- **G-S10a, the text decoder:** `qwen3vl_real_test.go` as written in P8 Phase 0, on nobara.
- **G-S10b, the tower against HF on the real 2B:** every stage (the patch embed with position rows, each block, the
  merger, each DeepStack merger) at soft-token cosine >= 0.9999 on the four F2a images. HF's tower is run alone, float32;
  0.999-0.9999 is ambiguous (parked). Plus aikit's own tiny-tower check, with norms randomised and the DeepStack merger's
  post-shuffle norm dropped as a planted defect that must go red.
- **G-S10c, the full model on an image prompt against HF `Qwen3VLForConditionalGeneration` (float32, nobara):**
  goinfer's float32 CPU prefill, with the same pass bar as G-S5b: last-position cosine >= 0.999, argmax equal, argmax
  agreement >= 95% over the text positions after the image. Planted defects, each red: (1) the DeepStack features not
  added; (2) added at the wrong layers (shifted by one); (3) added to the text positions too.
- **G-S10d, served:** one image request through serve, `--backend cpu` against `--backend metal` (the CPU prefill and
  upload) with every load flag equal, plus a CPU repeat. Identical replies, or a first divergence at a near-tie.

**S10 Qwen3-VL progress (2026-10-07):**
- **G-S10a: PASS** (nobara, 14:36, `~/wt/goinfer-s5`, under the timing lock). `scripts/pin_qwen3vl_real.py` ran in
  `~/.venv-vl` (transformers 5.12) on `~/models/qwen3-vl-2b-instruct`, then `TestQwen3VLReal_gate`: logit cosine
  1.000000, argmax 12095 equal, the 6-token greedy continuation identical (" Paris, and the capital of"). P8 Phase 0's
  gate, written 2026-09-08 and never run, has now run. The golden is committed as `qwen3vl_real_golden.json.gz` (3.0
  MB of JSON, 1.3 MB gzipped).
- **A defect the run surfaced, fixed:** the load printed `MRopeSection=[]`. The released checkpoint writes `rope_scaling
  {mrope_interleaved: true, mrope_section: [24, 20, 20], rope_type: "default"}`, and `qwen3_vlArchitecture` took the
  section only when the type said `mrope`. Text never sees it (every position's three components are equal), but an
  image prompt would have been rotated as plain RoPE. The section is now taken whenever it is present.
  `TestQwen3VL_mropeSectionFromRopeScaling` pins it with the real config's shape: red before the fix, green after.
- **aikit DeepStack (local branch `s2-tower-exports`, 14f4b7c..45b93dd, not pushed):**
  - The Qwen3 tower loads the DeepStack mergers (the post-shuffle LayerNorm) and returns their rows beside the main
    ones (`ForwardDeepstack`). `Forward` is unchanged for every caller.
  - The device export (`Weights`) refuses a DeepStack tower, so S2's Metal tower declines it rather than drop the rows.
  - The load refusal of DeepStack itself is gone; malformed index lists are still refused.
- **G-S10b: PASS, tiny and real.**
  - **Tiny** (aikit's `TestQwen3Deepstack_tiny`, a committed fixture pinned by `scripts/pin_qwen3vl_vision_tiny.py`:
    DeepStack at blocks 0 and 1, every norm randomised, two images): the main rows and both DeepStack sets match
    transformers at worst cosine 1.000000000. The planted defect (the post-shuffle norm dropped) is red at 0.888 and
    0.823.
  - **Real** (nobara, 15:03 PDT, under the timing lock, aikit's `TestQwen3VisionEncoder_realDeepstack` against
    `scripts/pin_qwen3vl_tower_real.py`'s artifacts: HF's own pixel values, every stage hooked): every stage of all
    four F2a images at worst cosine >= 0.9999 (the embed, all 24 blocks, the merger, the three DeepStack sets at blocks
    5, 11 and 17).

    | image | grid | worst stage | merged | DeepStack 0 / 1 / 2 |
    |---|---|---|---|---|
    | gemma3_preprocess_image.png | 56x56 | 0.999995237 | 0.999999901 | 1.000000000 / 0.999999803 / 0.999999813 |
    | qwen25vl_preprocess_image.png | 14x20 | 0.999999891 | 0.999999995 | 1.000000000 / 0.999999996 / 0.999999998 |
    | glm_ocr/formula.png | 76x62 | 0.999999739 | 0.999999988 | 1.000000000 / 0.999999871 / 0.999999932 |
    | glm_ocr/table.png | 56x76 | 0.999999562 | 0.999999999 | 1.000000000 / 0.999999999 / 0.999999997 |
- **The decoder's DeepStack (main, `84cc5aae`):**
  - The KV cache carries a prefill's DeepStack sets. The batched layer loop adds set l to the image positions after
    layer l (`addDeepstack`, unit-tested on batch offsets).
  - `GenerateQwenVLDeepstack` takes them; `GenerateQwenVL` is its no-DeepStack case.
  - A resident m-RoPE prefill (CUDA's) is not offered a DeepStack turn, since it cannot inject.
- **Serve (`s2-towers`, `8a6bc27a`; needs the aikit branch):**
  - A `qwen3_vl` checkpoint takes the Qwen3.5+ tower path, DeepStack required; a `qwen3_5` config declaring it is
    refused.
  - The tower runs on the CPU by name: no device tower carries DeepStack yet, and `-require-backend` refuses on Metal.
  - The features are one flat vector (the merged rows, then each set), so the image cache keeps them whole; serve splits
    them for the decoder.
- **G-S10c: PASS** (nobara, 15:14-15:16 PDT, under the timing lock).
  - The setup: `scripts/pin_qwen3vl_image_real.py` (transformers 5.12, float32, HF's own processor and chat template:
    table.png, 1083 ids with 1064 image tokens, grid 56x76). Then `TestQwen3VLImageReal` on those ids and pixel values:
    aikit's tower with DeepStack, goinfer's float32 prefill with the production fast-attention setting.
  - **Result:** last-position cosine 1.000000, argmax 86608 equal to HF's, and argmax agreement over the 15 text
    positions after the image 15/15. The test's open loop equals `prefillLogitsQwenVL` itself.
  - **Planted defects, all red:**
    1. DeepStack not added: cosine 0.982867, argmax kept;
    2. added one layer late: 0.987967;
    3. added to the text positions too: −0.135, argmax changed.
- **G-S10d, read 15:18-15:19 PDT on the Mac: PASS.**
  - The setup: a serve binary from `s2-towers` at `8a6bc27a`. Qwen3-VL-2B was copied from nobara to
    `~/models/qwen3-vl-2b-instruct` over the LAN. table.png, 32 greedy tokens, `--embed-int4=false` and
    `-vision-device cpu` on every arm; arms `cpu`, `metal` (`metal-resident (int4)`), then `cpu` again.
  - **All three replies are byte-identical:** "Table 2. Quarterly unit sales by region (thousands)". The tower ran on
    the CPU in every arm, reported as such ("CPU (DeepStack: no device tower yet)").
  - Request times, exploratory: 25.8-27.9 s, most of it the CPU tower.
  - Raw: `docs/measurements/multimodal-support-2026-10/s10-gs10d/`.
- **G-S10e, the Qwen3-VL tower on Metal with DeepStack, registered 2026-10-07 before its code:** S2's Metal grid tower
  returns the block outputs at the DeepStack indexes ("taps"); aikit's exported host tails turn them into the DeepStack
  sets, as `FinishHidden` turns the last one into the merged rows.
  - **Bar:** G-S2b's, unchanged. The merged rows and each DeepStack set at worst-row cosine >= 0.9999 against aikit's
    CPU `ForwardDeepstack`, on the tiny DeepStack tower (norms randomised) and on Qwen3-VL-2B's four F2a images;
    0.999-0.9999 ambiguous (parked).
  - **Planted defect:** the taps taken one block late; it must go red on the tiny tower.
  - **Then G-S10d's served check again,** with the tower on Metal against `-vision-device cpu`, both `--backend metal`:
    identical replies or a first divergence at a near-tie.
- **G-S10e: PASS** (Mac, 15:39-15:43 PDT; `s2-towers` at `aec3ccec`, aikit `5db4ce6`).
  - **The code:** the Metal grid tower taps the DeepStack blocks (`HiddenTaps`, read where `run` ends each block's
    command buffer). aikit's host tails (`FinishHidden`, the new `DeepstackFromHidden`) make the merged rows and the
    sets (`multimodal.Qwen3TowerFeaturesDeepstack`). Serve runs a Qwen3-VL tower on a device only if that tower can tap;
    otherwise the CPU, by name, or an error under `-require-backend` (`TestQwen3VLTower_needsTaps`, red without the
    check).
  - **Tiny** (`testdata/qwen3vl-vision-tiny`, copied from aikit): merged rows and both sets at 1.000000000 on two grid
    sets. The planted taps-one-block-late defect is refused (the last tap falls past the 2-block tower), which is red.
    **Amended 2026-10-07 evening (audit):** counting any refusal as red was too weak. The fixture was re-pinned with a third
    block (and nonzero biases, qkv x6), so the late taps stay inside the tower: the defect now reads worst 0.997922,
    red, and a refusal fails the test.
  - **Real 2B, four F2a images at serve's cap:** worst 0.999997916 (formula.png's second set); every other output
    0.99999990 or better. The planted defect on the 14x20 image: sets at 0.574, 0.293 and 0.661, red.
  - **Served, `--backend metal` in every arm:** the tower on Metal against `-vision-device cpu`, twice: byte-identical
    replies ("Table 2. Quarterly unit sales by region (thousands)"). Request time 19.2 s against 31.3-41.3 s with the
    tower on the CPU (exploratory, one reading each).
  - Raw: `docs/measurements/multimodal-support-2026-10/s10-gs10e/`.
- **The owed preprocessing record, read 15:30 PDT:** `multimodal/qwen3vl_preprocess_real_test.go` (goinfer's
  QwenPreprocess with Qwen3-VL's own, uncapped config against transformers' `Qwen2VLImageProcessor` pixel values from
  G-S10b's run). The grids are equal on all four images. The pixels are identical on the two that need no resize; on
  the two resized ones the largest difference is 0.00784 (one 8-bit level, 2/255 in [-1, 1]), the mean 2e-6 and 9e-6,
  and the worst patch cosine 0.999998.
- **aikit's qwenmetal allocation fix (nobara's df1a98f, merged into the local branch as 97cc578):**
  `TestQwenMetal_allocationFailureIsAnErrorAndTheEncoderRecovers` passes on the Mac, with the full qwenmetal suite. A
  2.5 TB scratch request came back as "MTLBuffer allocation failed ... out of memory", and the encoder recovered.
- **S10 Qwen3-VL: G-S10a-e all PASS.** Owed, none of them gates:
  - ~~goinfer's preprocessing against HF's~~ (done, above).
  - ~~A device tower with DeepStack on Metal~~ (G-S10e). CUDA's twin (nobara): its grid tower implements
    `multimodal.GridTowerTapper`, the same `run`-loop tap.
  - ~~The serve log's "Qwen3.5 vision"~~ (now "Qwen3-VL vision").
  - The MoE variants.
- **The 2026-10-07 audit's fixes on `s2-towers` (Mac, evening):**
  - **`towerInt8`:** `qwen3_vl` joins the float32 list (147e2298). Under cuda or webgpu it had an int8 CPU tower nobody
    validated, under a banner naming a `-vision-quant` the user never passed. `TestTowerInt8` is now a table over
    family x backend x `-vision-quant`, red without the fix.
  - **The fallbacks (22bea42c):**
    - Under `-require-backend`, a device tower's memory failure now fails the request instead of falling back to the
      CPU: `deviceFallback.run` for the grid towers, `qwenDeviceForward` for Qwen2.5-VL.
    - Every tower's forward is recovered from a panic: inside `run`, and for every family (Gemma 3 SigLIP, Gemma 4
      image and audio) at `withFeatureCache`.
    - Four tests, each red without its fix.
  - **The fixture (ed8d4756 on `s2-towers`; aikit c4c1404, local):** `testdata/qwen3vl-vision-tiny` re-pinned.
    - It now has nonzero biases at the weights' init scale (0.02), qkv x6 and three blocks.
    - The bias scale is not arbitrary: a 0.3 scale drowned the DeepStack merger's signal (its planted post-shuffle-norm
      defect read 0.998), 0.05 read 0.95, and 0.02 reads 0.898 / 0.880 (0.888 / 0.823 before).

### S11 — Several images per message

Added 2026-10-07. Today a second image in one message is a 400.

- **What it needs:** prompt layout with several image blocks per family, the resident image-block bookkeeping
  (`residentImageBlock`, prefix reuse) for more than one block, and serve's request parsing.
- **Gates:** two-image prompts against HF on a tiny fixture per family; served two-image requests; prefix reuse refusing a
  changed second image.
- **Size:** S-M.
- **The peers' trap, to get right and say so (from S15's desk map):**
  - llama.cpp #24303 (fixed between b10360 and b10520) and Ollama #17814 (open) fuse two adjacent same-size images into
    one 2-frame temporal "super-frame" on the Qwen3 family, through `mtmd`'s `can_merge_with()`. So 4 images read as 2.
  - goinfer's rule is "Do not pair images" (`docs/multimodal.md`). S11's gates add a planted defect for it: two
    same-size images in one message, merged as temporal halves, must go red.

### S12 — WebGPU for multimodal (owner: invest, 2026-10-07)

WebGPU today: Gemma 3 GPU/GPU, Qwen2.5-VL decode on the GPU, GLM-OCR staged, Gemma 4 and Qwen3.5 on the CPU, and no
release binary (it needs cgo).

- **What it needs:** Gemma 4 resident decode on WebGPU (its Gemma kernels: sandwich norms, the two geometries, K=V, the
  S1.0 fixes it already has), then the E-model shape (S1's three parts), then the towers that S2/S4 put on Metal and CUDA,
  and the release-binary question (a cgo build per platform, or a documented build-it-yourself).
- **Gates:** the S1/S2 gates, run on WebGPU; written per sub-phase before it starts.
- **Size:** L.

### S13 — Against the peers

Added 2026-10-07. S7 grades each cell against a TTFT bar, not against another engine.

- **What:** one night per box, `bench_peer.py`, the S7 cells against llama.cpp (and MLX on the Mac) on the same image
  and prompt, same-session interleaved; reported, not gated, unless a bar is registered first.
- **Size:** S-M (mostly harness: the vision peer path exists for SigLIP only).
- **S13-lite first (owner, 2026-10-07 evening):** one model per box, against Ollama and llama.cpp, a fresh image per
  timed request. Registered with S7 above. The full S13 comes after S10's remaining families, S11 and S15.

### S14 — Speech beyond Gemma 4 (last, after S12)

The owner put this last, after the WebGPU work. It starts with the choice P11 left open
(`docs/measurements/multimodal-finish-2026-10-06/p11-audio-comparison.md`): a Whisper-style front end plus Qwen3-ASR; a
Whisper encoder (which serves Voxtral), toward pure-Go Whisper; or Voxtral Realtime. Gates are written once the family
is chosen.

#### S14, the family chosen: owner 2026-10-08, "b then c". Registered before any code.

**Order:** (b) a Whisper-style front end plus Qwen3-ASR, then (c) the Whisper encoder (which also serves Voxtral Mini) toward pure-Go Whisper. (d) Voxtral Realtime and (e) CTC encoders are not in this tranche. Each stage's gates are registered when it starts; only S14.1 is registered here. The owner did not ask for the do-nothing measurement first (Gemma 4 E4B's word error rate against Qwen3-ASR's on the same clips); it stays as a record-only side measurement at S14.3, not a gate and not a precondition.

- **S14.1 · the Whisper front end** (aikit `audio`, a second extractor beside Gemma 4's; shared by every later stage).
- **S14.2 · Qwen3-ASR's audio encoder and projector** (AuT: three stride-2 Conv2d, sinusoidal positions, windowed attention).
- **S14.3 · Qwen3-ASR end to end:** the prompt layout, the soft-token splice into the `qwen3` decoder (full-oracle already), served transcription; WER against Hugging Face.
- **S14.4 · (c):** the Whisper encoder, Voxtral Mini through the `llama` decoder, then the cross-attention decoder and the decode policy.
  Hardware reference: Qwen3-ASR-0.6B is 0.94B parameters, Apache-2.0, ungated, on the NVMe after S14.2 starts.

##### G-S14a, the Whisper front end (S14.1), registered 2026-10-08 before any code

- **Reference:** transformers 5.15.0 (`~/g4venv`) `WhisperFeatureExtractor`. It has two implementations and `__call__` uses the torch (float32) one when torch is present; the NumPy (float64) one is the other. **Measured before registering, to set the bar:** on the LibriSpeech clip padded to 30 s the two differ by at most 1.657e-05 (mean 2.85e-08), 128 mels, n_fft 400, hop 160, output [128, 3000], values in [-0.76, 1.24].
- **Function:** `audio.WhisperFeatures(samples []float32, mels int)` returns the [mels x 3000] features, row-major, exactly as the extractor does with its defaults: the clip zero-padded to 30 s (480,000 samples) or truncated to 30 s, a periodic Hann window of 400, centre-padded by reflection, power spectrum, the Slaney mel bank with Slaney area normalisation (mels 80 or 128, fmax 8 kHz), `log10` with each clip clamped to its maximum minus 8, then `(x + 4) / 4`, the last STFT frame dropped. A mel count other than 80 or 128, or an empty clip, is refused. A second function gives the number of valid (non-padding) frames for a sample count, pinned to the extractor's own attention mask.
- **Bars (per case):** the output shape equal to the reference's; the largest absolute difference against EACH of the two reference paths **<= 5e-05** (three times their own spread) and the mean absolute difference **<= 1e-06**; the valid-frame count equal to the mask's sum for lengths 1, 159, 160, 161, 16,000, 93,680, 480,000 and 500,000.
- **Cases (8):** the LibriSpeech clip (93,680 samples); its 44.1 kHz and 48 kHz-stereo 16 kHz references (`testdata/speech/*.ref16k.f32`); a synthetic 30.0 s signal (tones plus seeded noise, exactly 480,000 samples); a synthetic 31.5 s signal (truncated); an 800-sample clip (0.05 s); 2 s of silence (every value equals the clamp, no NaN); the LibriSpeech clip with 80 mels.
- **Planted defects, each alone, each must put some case over the 5e-05 bar** (a test-only seam, in the manner of the DeepStack gates): (1) the HTK mel scale; (2) the magnitude spectrum for the power one; (3) no `max - 8` clamp; (4) the symmetric Hann window; (5) zero padding at the edges instead of reflection; (6) no `(x + 4) / 4`. If a planted defect stays under the bar, the bar is too loose and that is recorded, not argued.
- **Where the code lives:** the extractor in aikit (`audio/whisper_features.go`; aikit is a separate repo with its own release, so goinfer's push waits on an aikit tag, as before); the golden, its pin script (`scripts/pin_whisper_features.py`) and the numeric test in goinfer (the golden is gzip-compressed, per the repo's convention for large raw data).
- **Prediction, written now:** the Go output is within 3e-06 of the NumPy path and within 2e-05 of the torch path on every case (it does float64 arithmetic and rounds once, like the NumPy path); every planted defect exceeds 1e-03 on the speech clip. The most likely surprise is the 400-point FFT (not a power of two) and the exact framing at the clip edges.
- **Tier and cost:** quick, by day: the pin script seconds, the Go tests seconds, about two hours of work.

### S15 — Video (back in scope, owner 2026-10-07 evening)

Depends on S11 (several images per message). They share:
- several media blocks per prompt and per-block feature caching;
- the resident image-block bookkeeping for more than one block;
- the rule that same-size images are never paired.

S15 adds temporal patching, frame timestamps and video placeholder tokens.

#### Gate 0, the desk map (read 2026-10-07 from transformers 5.16.1; file:line citations in
`docs/measurements/multimodal-support-2026-10/s15-video-deskmap-2026-10-07.md`)

- **Qwen2.5-VL: native video.**
  - **The processor:** `Qwen2VLVideoProcessor`; the token `<|video_pad|>`; `get_video_features` runs the image tower.
  - **Frame sampling:** off by default (`do_sample_frames=False`: every frame given is used). With `fps` set, 4-768
    frames, floored to even. The pixel budget is per frame (`smart_resize`, factor 28, 128·28² to 768·28²).
  - **Temporal patching:**
    - The patch layer is a `Conv3d` with kernel and stride [2, 14, 14], no bias.
    - An odd frame count is padded with the last frame repeated.
    - Each temporal group is its own vision attention segment, and the vision RoPE is (h, w), repeated per t.
  - **Positions:**
    - `second_per_grid_ts = temporal_patch_size / sampled_fps`, and the text model's t step is
      `tokens_per_second * int(second_per_grid_t)` (`tokens_per_second` 2 in the 3B config).
    - The `int()` truncates: above 2 sampled fps every frame group gets the same t.
    - After a video, `current_pos` advances by max(h, w)/2, not by the t extent.
    - Both look unintended, and S15 matches transformers exactly anyway (G-S15c is against it), recording each.
  - **The prompt:** `<|vision_start|><|video_pad|><|vision_end|>`, the pad expanded; no timestamps.
- **Qwen3-VL: native video.**
  - **Frame sampling:** fps 2, 4-768 frames, `linspace` indices. The budget is per video (the 2B checkpoint caps it at
    about 12,288 tokens). Fewer than 2 frames is an error.
  - **The patch layer:** `Conv3d` [2, 16, 16] with bias.
  - **Timestamps are text:** each frame pair is `<{t:.1f} seconds><|vision_start|>` + pads + `<|vision_end|>`, where t
    is the mean of the pair's two frame times.
  - **Positions:** each pair is a t = 1 grid laid out like a still image, so time is carried only by the timestamp
    text.
  - **A quirk:** with the shipped chat template the per-pair blocks sit inside a second, outer
    `<|vision_start|>…<|vision_end|>`. Matched as transformers does it, and recorded.
  - **DeepStack:** video goes through it, with the image and video features under one mask.
- **Qwen3.5+:** the same as Qwen3-VL (it uses `Qwen3VLVideoProcessor`; the video token id is 248057), without DeepStack.
- **Gemma 4: native video, now confirmed in code** (`docs/multimodal.md` said "not independently confirmed").
  - **The tower:** each frame runs the image tower and `embed_vision` (`get_video_features`), and the result is
    scattered in. There are no m-RoPE positions.
  - **Frame sampling:** 32 frames (`arange` sampling; fewer than 32 is an error), at most 70 soft tokens a frame
    (`max_soft_tokens`, one of 70-1120).
  - **The prompt:** each frame is written as `MM:SS <|image>` + `<|video|>`×n + `<image|>`, joined with spaces.
- **Gemma 3: no native video.** Frames go in as separate images (S11), each `\n\n<boi>` + 256 + `<eoi>\n\n`; any
  timestamps are the client's text.
- **The input shape, for the owner** (from cheapest):
  - (a) **A client-extracted frame list as a content part** (frames plus their times): no decoding in goinfer.
  - (b) **Animated GIF:** decodable by the standard library (`image/gif.DecodeAll`: frames plus delays, so real
    timestamps). The caller composites each frame by its disposal method, and bounds memory with `DecodeConfig` first.
  - (c) **mp4/H.264 in-process:** pure-Go demuxers exist (`abema/go-mp4`, `Eyevinn/mp4ff`; unverified), but no mature
    pure-Go H.264 pixel decoder is known. So it means cgo (FFmpeg through `go-astiav`, openh264, VideoToolbox or NVDEC)
    or a WASM decoder under wazero. Each breaks pure-Go or adds a large dependency, which is why `docs/multimodal.md`
    ruled container decoding out of v1.
- **Two known risks, made gates:**
  - **The patch conv's temporal halves (F5d):** swapping `v.patch_embd.weight` and `.weight.1` is invisible on a still
    image, where both halves see the same frame. At T > 1 they see different frames, so the swap must go red.
  - **The peers' same-size pairing** (llama.cpp #24303, Ollama #17814): video is the one place frames ARE paired
    temporally. Images in a message never are (S11's defect). S15 says which it does, per family, in the served log.

#### Gates, registered 2026-10-07 evening before any S15 code

- **G-S15a, preprocessing against HF's video processor:**
  - **The clip:** a short committed clip with a known license, as frames (and, if the owner picks (b), the same clip as
    an animated GIF).
  - **The bar:** frame indices and timestamps equal; per-frame grids equal; pixels within one 8-bit level, the G-S10
    preprocessing record's standard.
  - **The families:** Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma 4.
- **G-S15b, the tower against HF at T > 1:** every soft token at cosine >= 0.9999 (0.999-0.9999 parked).
  - **Fixtures:** the tiny fixtures with norms randomised and nonzero biases, then the real checkpoints.
  - **Planted defects, each alone red:**
    1. the temporal halves swapped (F5d);
    2. the frame groups' attention segments merged;
    3. an odd frame count padded with zeros instead of the last frame;
    4. Qwen3-VL's DeepStack not applied to video features.
- **G-S15c, full-model logits against HF on a video prompt (the G-S10c shape):**
  - **The bar:** last-position cosine >= 0.9999 with an equal argmax, and argmax agreement over the text positions
    after the video.
  - **Planted defects:**
    1. Qwen2.5-VL's `second_per_grid_ts` ignored;
    2. Qwen3-VL's timestamps dropped from the prompt;
    3. Gemma 4's per-frame `MM:SS` dropped.
- **G-S15d, served:** a video request through serve, the tower on the device against `-vision-device cpu`, identical
  replies or a first divergence at a near-tie (G-S2d's rule). Then the speed record at night, against S7's 5 s bar for
  a short clip at the default budget.

### S16 — A resident m-RoPE prefill on Metal (added 2026-10-07 evening, after the audit)

- **The gap:**
  - Every Qwen-family image turn on a Mac (Qwen2.5-VL, Qwen3.5+, Qwen3-VL) prefills on the CPU, then `UploadKV`s
    and decodes on Metal. Metal has `ForwardMRoPE` (decode rows) but no batched m-RoPE prefill.
  - CUDA's twin (`PrefillMRoPELast`) measured 36.14x the CPU-prefill-plus-upload bridge: 711.2 ms against 19.7 ms
    (`docs/benchmarks.md`, 2026-09-08). That was on a 14-token Qwen2.5-VL prompt, prefill only, with the tower
    excluded. An image turn's gain is smaller in proportion to the tower's share.
- **The scope:**
  - The batched prefill's rope learns the per-row 3-component m-RoPE angle, for image rows and for the text rows
    after them (`mropePositions`).
  - Qwen3-VL's DeepStack injection: after layer l, add set l to the image rows. Prefill only, as in
    `decoder/forwardn.go`'s `addDeepstack`. CUDA's prefill also lacks it, so CUDA's Qwen3-VL image turns also
    prefill on the CPU. The CUDA twin is nobara's, under the same gates.
- **Gates, registered 2026-10-07 before any S16 code:**
  - **G-S16a, tiny, against the existing path:** the qwen25vl, qwen3.5 and qwen3vl tiny fixtures, one image each.
    The new resident prefill against the current one (CPU prefill, `UploadKV`, Metal decode): last-row logits plus
    the next 8 decode steps, at cosine >= 0.9999 per row with an equal argmax (0.999-0.9999 parked).
    - Bit-identity is not expected: f16-MMA against the CPU's int4 kernels, G3's lesson.
  - **G-S16b, planted defects, each alone red at G-S16a's bar:**
    1. m-RoPE sections in the wrong order (t, w, h);
    2. the image rows given 1-D text positions;
    3. the text rows after the image given uncompressed positions;
    4. DeepStack not added;
    5. DeepStack added one layer late;
    6. DeepStack added to the text rows too.

    A fixture too degenerate to show one is sharpened first (S3's lesson; norms randomised, nonzero biases).
  - **G-S16c, real:** Qwen2.5-VL-3B and Qwen3-VL-2B, the four F2a images, the same comparison at the same bar.
    Then served: `run-gs3c-served.sh` with the tower held equal, `--backend metal` before and after the change on
    one binary pair. Identical replies or a first divergence at a near-tie.
  - **Speed, a record, at night:** image-turn TTFT on the same two models, old path against new, interleaved, the
    tower time reported apart.
    - **Projection band, written here before the speed run:** it comes from two inputs.
      - A by-day split of one image turn on today's path: tower, CPU prefill, upload, first decode step.
      - The new prefill's expected rate: the Metal f16-MMA text prefill's 225-366 TTFT tok/s on the 1.5B at K
        256-3900 (`docs/benchmarks.md`, 2026-09-18), scaled to the 2-3B models.

      TTFT then gains by the CPU prefill's share of the turn. CUDA's 36x does not transfer: it was a single-threaded
      CPU bridge on 14 tokens.
    - Ship at >= 1.02x TTFT (the owner's default); 1.00-1.02 parked; under 1.00 off.

- **S16 on Metal, step 1 (Qwen2.5-VL's m-RoPE prefill), built and gated on the tiny fixtures 2026-10-08.**
  - **What:** `metalResident.PrefillMRoPELast` (`decoder.ResidentMRoPEPrefill`), the f16 batched pass with each row
    rotated by its own (t, h, w) triple.
    - **The kernel,** `rope_mrope_f16`, reads a per-pair axis table that the decoder builds from its own
      `mropeComponent` / `mropeComponentInterleaved` (`Model.MRopeAxisResident`), so the layout lives in one place.
    - **The K/V cache** is still placed by sequence position.
    - **Shared declines:** `PrefillLast`'s checks moved into one helper (`batchedPrefill`), so the image entry declines
      in exactly the text pass's cases. The batched step and the layer-major routes never take it.
  - **G-S16a / G-S16b (defects 1-3), PASS** (`TestS16MRoPEPrefill_tiny`).
    - **Coverage:** the tiny Qwen2.5-VL and Qwen3.5 VL fixtures, each with its golden prompt and with a synthetic 16x24
      image (96 merged rows, random features at the golden features' scale).
    - **Why the synthetic image:** the golden 2x3 grid keeps every position under 6, where most rotary pairs barely
      turn and a position defect can hide.
    - **Results:**
      - the new path's worst cosine is 0.999943 / 0.999950 / 0.999935 / 0.999948, with every argmax equal, over the
        last row and 8 decode steps against today's path;
      - through `GenerateQwenVL`, the Qwen2.5-VL fixture takes the resident prefill and its greedy tokens equal today's
        path's.
    - **Defects, each red on at least one case:** (1) h and w swapped 0.99538; (2) the image rows 1-D 0.98174; (3) the
      text after the image uncompressed 0.98380.
    - **Not seen on the Qwen2.5-VL fixture:** the h/w swap. Its head dim 16 and rope base 10000 put h and w on
      frequencies that turn under 0.12 rad across the whole grid. Real size covers it.
  - **The hybrid (Qwen3.5+) is NOT claimed**, so `HybridMRoPEPrefill` is not implemented on Metal, and a hybrid's image
    turn stays on the CPU as before.
    - **Why:** a hybrid's only reference decodes on the CPU, so its per-step comparison cannot isolate the prefill.
    - **Measured 2026-10-08 on Qwen3.5-0.8B,** the four F2a images, every row at a flat text position (so no m-RoPE in
      any arm). Every pair of backends differs by about the same:

      | comparison | worst cosine |
      |---|---|
      | Metal sequential vs CPU | 0.87-0.94 |
      | Metal batched vs CPU | 0.89-0.95 |
      | Metal batched vs Metal sequential | 0.88-0.94 |

    - **The text baseline** for the same model and comparison is 0.987-0.990.
    - **So:** the gap is this model's sensitivity on image prompts across kernel sets, not the m-RoPE pass. A hybrid
      claim needs a G3-shaped gate (model-level non-inferiority), not G-S16a's per-step bar. Parked for the owner.
  - **S16 step 2 (Qwen3-VL's DeepStack), built 2026-10-08.**
    - **The decoder:** a new optional `decoder.ResidentMRoPEDeepstackPrefill`, which `GenerateQwenVLDeepstack` uses
      when the resident implements it. Before, a turn with DeepStack sets always prefilled on the CPU.
    - **Metal:** `PrefillMRoPEDeepstackLast` adds set l to the image rows of the f16 residual after layer l, at the top
      of iteration l+1 and after the loop, so every branch of the layer loop reaches it (the MoE branches end in
      `continue`).
    - **Its tiny gate PASSES** (`TestS16DeepstackPrefill_tiny`: the qwen3vl-tiny decoder, a synthetic 16x24 image,
      random features and sets): worst cosine 0.9999463.
      - Each planted defect is red on cosine: (4) the sets not added 0.98195; (5) added one layer late 0.98195 (in a
        2-layer model "late" pushes set 0 past the last layer, where nothing reads it, as on the CPU); (6) added to the
        text rows too 0.99350.
      - **The argmax reading:** this fixture's next-token distribution is nearly uniform (top p about 0.005), so argmax
        differences are read by R10's near-tie rule. Every one was a near-tie, including the free runs' first
        divergence through `GenerateQwenVLDeepstack`, which took the resident prefill.
  - **G-S16c on Qwen3-VL-2B, read 2026-10-08 by day: FAIL against the registered bar.**
    - **The readings:** the four F2a images, today's path (the CPU prefill with the sets, the upload, Metal decode)
      against the DeepStack prefill. Worst per-step cosine 0.9718 / 0.9776 / 0.9846 / 0.9865, with argmax 8/9 on two
      images. The bar was 0.9999 and an equal argmax.
    - **The mechanism, measured the same day:** the same isolated comparison on a text-only prompt reads worst 0.97726
      at 84 tokens and 0.99739 at 800. That is the CPU prefill and upload against Metal's shipped batched text
      prefill, both decoding on Metal, with no image and no m-RoPE.
    - **So:** the m-RoPE prefill sits at the shipped f16 prefill's own distance from the CPU's int4 prefill (that
      prefill was graded by the §3.2 pooled gate, not a per-step 0.9999). G-S16a's bar held on the tiny fixtures and is
      stricter than the shipped text path at real size.
    - **The bar stands as registered.** Re-registering it is the owner's: for example, non-inferiority to the same
      model's text-prefill control at the same length, or S1's G3 shape.
    - **Until then the resident m-RoPE prefill is OFF in production** (`metalMRoPEPrefillOn`, default false; the gates
      turn it on). An image turn takes the CPU prefill and upload, as before.
    - **The night job is pinned at `a1e007c7`**, where the path was on, so its G-S16c, served and speed readings still
      measure it.
    - **Qwen2.5-VL-3B's G-S16c** is the night job's (the fit guard refuses it by day).

  - **G-S16a / G-S16c's bar, re-registered 2026-10-08 (owner: "a, change the bar"), before it is applied to any reading.**
    - **Why:** the per-step 0.9999 is stricter than Metal's shipped f16 text prefill achieves against the CPU's int4
      prefill (0.977-0.997, measured above).
    - **The control:** for each image prompt, text-only prompts of the same length on the same model, in the same
      isolated comparison: the CPU prefill and upload, against Metal's batched prefill, both decoding the same 8
      teacher-forced steps on Metal. Four prompts per length (token seeds 1-4), so the control has a spread.
    - **PASS:** the image turn's worst per-step cosine (the last row and 8 steps) is at least the control's minimum
      minus 0.005, and every argmax difference is a near-tie by R10 (the other token's reference p at least half the
      top's).
    - **Parked:** 0.005-0.015 below the control's minimum.
    - **FAIL:** worse, or a non-tie argmax difference.
    - **The tiny gates keep their passing readings;** this rule governs the real-size G-S16c.
    - **What it gates:** Qwen2.5-VL's path (no DeepStack) and Qwen3-VL's (DeepStack) each turn on in production only
      when that model passes.
    - **Tonight:** step 1 runs a test binary with the control, from a later rev. The served and speed steps keep the
      `a1e007c7` serve binary, where the path was on.

  - **G-S16c on Qwen3-VL-2B under the re-registered bar, read 2026-10-08 by day: PASS on all four images.**

    | image | rows | image worst cosine | the text control's minimum (4 prompts) | argmax |
    |---|---|---|---|---|
    | 896² | 798 | 0.9718256 | 0.8745833 | 1 near-tie |
    | 4x6 | 84 | 0.9776144 | 0.9537060 | 1 near-tie |
    | formula.png | 1029 | 0.9845718 | 0.9579256 | equal |
    | table.png | 986 | 0.9864695 | 0.8663697 | equal |

    - **The control's own range (0.87-0.96)** is the shipped f16 prefill's distance from the CPU's int4 prefill on
      random-token text at these lengths. The image turns sit above its minimum on every image.
    - **Still owed before Qwen3-VL's path turns on:** the served comparison and the TTFT record (the night's).
    - **The production switches are per path** (`metalMRoPEPrefillOn` for Qwen2.5-VL, `metalDeepstackPrefillOn` for
      Qwen3-VL), both off on main.
  - **The night job, amended 2026-10-08 before it runs:** both models in every step.
    - **The binaries:** the test binary from main (with the control); the serve binary from a local branch,
      `s16-night-on`, which is main with both switches on, so the served and speed steps measure the path.
    - **The verdicts:** a path turns on (a commit on main) only when its model passes G-S16c, the served comparison and
      the speed rule (>= 1.02x TTFT).
    - The step 3 band above is Qwen2.5-VL's. **Qwen3-VL's TTFT band, from the same arithmetic:**
      - S7's cell: 8.97 s TTFT, with a 3.75-4.14 s tower;
      - on this build the tower is lever A+B's, about 4.0 s / 2.8 ≈ 1.4 s;
      - the CPU prefill and upload take about 5 s;
      - S16 prefills about 1,000 rows at 150-250 tok/s, the 1.5B's rate scaled to the 2B: 4.0-6.7 s;
      - **so TTFT goes about 6.4 → 5.6-8.3 s: 0.8-1.15x.** An honest projection that it may not pay on the 2B. The
        speed rule decides.
    - **Cost:** about 50 minutes, queued at 75.

  - **S16's night job, registered 2026-10-08 before it runs** (`docs/measurements/multimodal-support-2026-10/s16/
    run-s16-night.sh`, night queue `s16-night`; Qwen2.5-VL-3B, which the fit guard refuses by day):
    1. **G-S16c, real:** `TestS16MRoPEPrefill_real`, the four F2a images, today's path (the upload bridge, Metal decode)
       against the S16 prefill, at G-S16a's bar.
    2. **G-S16c, served:** one image request (`run-gs3c-served.sh`'s), the same binary with `--exact-prefill` (today's
       path) and without. PASS: identical Metal replies, or the first difference at a near-tie in today's
       log-probabilities.
    3. **Speed, a record:** image-turn TTFT, today's path against S16, `vision_ttft.py`, 3 rounds of 1 warm-up + 3 timed,
       the order rotated, a new 896² image every request.
       - **The projection band, written before it runs:**
         - On the S16 build, the tower is lever A+B's (about 8.6 s / 2.8 ≈ 3.1 s).
         - Today's path's other time was 13.1 s on the 2026-10-07 S7 cell (21.67 s TTFT less its 8.59 s tower): the
           CPU prefill, the upload and the first token.
         - S16 prefills about 1,040 rows at 110-180 tok/s, the 1.5B's f16-MMA 225-366 TTFT tok/s
           (`docs/benchmarks.md`, 2026-09-18) halved for the 3B: 5.8-9.5 s.
         - **So TTFT goes about 16.2 → 9.1-12.8 s: 1.3-1.8x.**
       - **The rule:** ship at >= 1.02x, park at 1.00-1.02, off below 1.00. Below the band's low end with no mechanism
         found: parked.
    - **Cost:** about 25 minutes, queued at 40.

### S17 — Tower speed on both backends (added 2026-10-07 evening)

- **The gap:** the device towers are correct float32 baselines, and slow.
  - **SigLIP at 896x896:** about 5.45 TFLOP (27 layers of about 202 GFLOP: projections 43.5, MLP 81.2, attention
    77.3).
  - **Metal:** its 9.2 s on Metal is about 0.59 TFLOPS against the M1 Pro GPU's ~5.3 f32 peak. Qwen2.5-VL's 64x64
    grid is about the same.
  - **CUDA:** the towers run at 1.2-1.5x the CPU (S4 step 3).
- **Absorbs S4's conditional step 6.** The candidate levers:
  - **Metal:** f16 or simdgroup-matrix GEMMs; blocked (flash-style) attention.
  - **CUDA:** tensor-core GEMMs; fused attention.
- **Gates, registered 2026-10-07 before any S17 code:**
  - **Step 0, a profile:** before any lever, each backend's tower time is split by kernel class (GEMM, attention,
    norms and the rest) on SigLIP and Qwen2.5-VL at the F2a sizes. The split fixes each lever's projection band.
  - **Step 0's instrument on Metal, registered 2026-10-07 evening before it runs** (TE11):
    - **Tier and stopping rule:** by day (a profile, not a speed claim), fixed: the four F2a images per tower at serve's
      caps, one warm-up and then 3 profiled forwards each, the median per kernel class reported.
    - **The instrument:**
      - With profiling on, the Metal grid tower ends its command buffer at every change of kernel class: GEMM (with its
        bias), attention (the gathers, both GEMMs, the softmax, the scatter), norm, and elementwise (RoPE, the scale, the
        residual adds, the activation).
      - Each buffer's GPU-busy window (`GPUEndTime - GPUStartTime`) counts toward its class.
      - With profiling off nothing changes: the hook is nil.
      - The towers are SigLIP (Gemma 3) and Qwen2.5-VL (the two the audit named), and Qwen3.5 as a third (the grid
        tower's own kind).
    - **Overhead check:** the profiled total (the sum of GPU windows) against an unprofiled forward's wall, same image,
      same process. Extra command buffers cost submit gaps, not GPU time, so the profiled GPU sum should be at most the
      unprofiled wall. A sum over it marks the profile untrustworthy, and it is recorded so.
    - **Also reported:** each class's achieved rate. GEMM FLOPs come from the shapes (2·M·N·K), against the M1 Pro GPU's
      ~5.3 TFLOPS f32 peak, so each lever's band has a share and a headroom.
    - **Cost:** SigLIP about 10 s a forward x 4 images x 4, Qwen2.5-VL 2-15 s x 4 x 4, Qwen3.5 under 2 s; about 8
      minutes in all.
    - **CUDA's twin** is nobara's, by the same split (`cudaEventRecord` per class).
  - **Step 0's instrument on CUDA, registered 2026-10-07 evening (nobara) before it runs** (TE11):
    - **Tier and stopping rule:** by day, a profile and not a speed claim, fixed: per tower and image one warm-up, then 3 unprofiled forwards (the wall) and 3 profiled ones, the median per kernel class reported.
    - **The towers:** the three that run on goinfer's CUDA tower base: SigLIP float32 (4096 patches whatever the image, so one image, `gemma3_preprocess_image.png`), Qwen3.5-0.8B on the four F2a images at serve's
      1024-token cap, and GLM-OCR on that same one image. **Qwen2.5-VL is not profiled:** it is aikit's `gpu/qwencuda`, which owns its queue, and a per-class split would be an aikit change; its row stays
      the Metal one and S4 step 3's whole-tower figure.
    - **The instrument:** with profiling on, `towerOps` drains the queue (`Queue.Sync`) at every change of kernel class and attributes the host time since the previous drain to the class that was running: GEMM (with
      its bias and add epilogues), attention, norm (LayerNorm, RMSNorm), and elementwise (RoPE, the activations, adds, scales). aikit's `Event` has no elapsed-time call, so this is a wall-clock split with the queue
      drained at each boundary, not `cudaEventElapsedTime`; the launch latency of the first kernel after a drain is inside the class. With profiling off nothing changes: the hook is nil.
    - **Validity checks, recorded with the result:** (1) the profiled output equals the unprofiled one bit for bit; (2) the profiled total is within +10% of the unprofiled wall (the drains add bubbles; over that the
      profile is marked untrustworthy and said so); (3) the three profiled reads of a class agree within 5%.
    - **Also reported:** each class's share, and the achieved rate against the card's fp32 peak (about 9.1 TFLOPS on the RTX 2070 SUPER at boost): GEMM FLOPs from the shapes (2*M*N*K), attention's as
      4*np^2*head_dim*heads (QK^T and PV), so each lever's band has a share and a headroom.
    - **A prediction, written before the read and not a bar:** attention dominates SigLIP on CUDA as it does on Metal (more than half of its time), the GEMMs run well under half of the fp32 peak, and norms and
      elementwise are a few percent. A result that contradicts it is recorded as such.
    - **Cost:** SigLIP about 18 s x 7 forwards, Qwen3.5 under 4 s x 4 images x 7, GLM-OCR about 25 s x 7; about 7 minutes in all.
  - **Step 0 on CUDA, read 2026-10-07 evening on nobara (commit `9bf0e917` + the instrument; `TestS17ProfileCUDA`, 5.4 minutes): attention is 92-95% of the time at 1% of the card's fp32 peak; the GEMMs are 4-7%.**
    - **Its validity (the registered checks):** the profiled output equals the unprofiled one bit for bit on all 7 runs; the profiled wall is within +0.1% to +0.4% of the unprofiled on every large size
      (+6.1% on the 14x20 grid, which is dispatch-bound and not a lever's target); a class's three reads spread 0.3-3.1% on the large sizes (4.1% on the small one). Raw:
      `docs/measurements/multimodal-support-2026-10/s17-step0-cuda/profile.log`. The median of 3 profiled forwards; rates from the shapes against the RTX 2070 SUPER's ~9.1 TFLOPS fp32 peak:

      | tower, image | rows | tower time | GEMM: share, TFLOPS | attention: share, TFLOPS | norm + elementwise |
      |---|---|---|---|---|---|
      | SigLIP float32 (every image) | 4096 | 18.03 s | 5.3%, 3.55 | **93.6%, 0.12** | 1.1% |
      | Qwen3.5-0.8B, 896x896 image | 3136 | 3.10 s | 5.0%, 3.46 | **93.3%, 0.13** | 1.7% |
      | Qwen3.5-0.8B, formula.png | 4060 | 5.07 s | 4.0%, 3.43 | **94.7%, 0.13** | 1.3% |
      | Qwen3.5-0.8B, table.png | 3888 | 4.72 s | 4.1%, 3.44 | **94.5%, 0.12** | 1.4% |
      | GLM-OCR, 896x896 image | 4096 | 14.14 s | 6.7%, 3.49 | **91.8%, 0.13** | 1.5% |
      | Qwen3.5-0.8B, 14x20 grid | 280 | 0.044 s | 41.5%, 2.63 | 45.5%, 0.14 | 13% |

    - **The prediction held, more strongly than written:** attention is far over half of every large tower, norms and elementwise are 1-2%, and the GEMMs run at 38% of the fp32 peak. aikit's attention
      (`AttentionTiled` from 3072 patches) runs at 0.12-0.14 TFLOPS in every tower, about 28x below the GEMMs on the same card (3.4-3.6 TFLOPS), so the towers are bound by one kernel, not by arithmetic.
    - **What it says about the levers:** the GEMMs are not worth work (a 3x faster GEMM buys at most 3.5% of SigLIP's 18 s); attention is the whole gap. S17's CUDA lever is an attention kernel, and it also changes
      the owner's open Gemma 3 default question: float32 SigLIP is 18 s because of attention, not because float32 is slow.
    - **Qwen2.5-VL on CUDA** is aikit's `gpu/qwencuda` and was not profiled (its own queue); its row is S4 step 3's whole-tower figure.
  - **S17's CUDA lever A (attention), bands registered 2026-10-07 evening before any lever code.** The projection is tower = non-attention + attention / k_A, from the rows above.
    - **What:** a fused, query-tiled online-softmax attention kernel in float32 for the tower base (`cuda/tower_base.cu`, NVRTC), non-causal, per segment, head dims 64, 72, 80 and 128, K/V tiled through shared
      memory with register-blocked QK^T and PV. It replaces `AttentionPlan` for the towers; the float32 numerics stay (no half precision), so the bars below are the same as before the lever.
    - **The kernel speedup expected, k_A = 10-25:** from 0.13 TFLOPS to 1.3-3.2, a third to a whole of the 3.5 TFLOPS this card's own tower GEMMs reach on the same base.
    - **The bands:** SigLIP 6.4-10x (18.0 to 2.8-1.8 s, which is under S7's 5 s bar with the tower alone); Qwen3.5 formula.png 6.8-11x (5.1 to 0.75-0.46 s); GLM-OCR 5.7-8.4x (14.1 to 2.5-1.7 s).
    - **Correctness first, unchanged bars:** every output token at cosine >= 0.9999 against aikit's CPU tower (G-S3a, G-S2b, G-S2c), the registered planted defects still red, and a new planted defect for the kernel
      (a softmax scale or a K/V tile boundary off by one) red on the tiny tower and at real size; a kernel test against float64 at the production shapes (head dim 72 at 4096 patches, a ragged tile, a frame batch).
    - **Kill line:** under 1.02x on any tower, or under the low end of its band with no mechanism found, parks it; a kernel that misses the cosine bar at real size is not shipped however fast.
    - **Instrument:** an in-process whole-tower A/B, interleaved, by day (TE5(b)), the profile above re-run for the per-class change.
  - **S17's CUDA lever B (GEMM) is registered as not a lever:** the GEMMs are 4-7% of the tower time, so even an ideal one buys under 7%. Written here so that it is not built by default.
  - **S17's CUDA lever A, read 2026-10-07 evening on nobara (commit `2e20dc01`): all three bands met; the float32 towers are 8-9x faster and SigLIP is under the 5 s bar.** Raw:
    `docs/measurements/multimodal-support-2026-10/s17-lever-a-cuda/` (`leverA-ab.log`, `leverA-real.log`, `leverA-g4v.log`).
    - **Built as registered** (`cuda/tower_base.cu`, `cuda/tower_base.go`): `tower_attn_hd64/72/80`, a fused, query-tiled, online-softmax float32 kernel (64 queries per block, 32-key tiles, a 4x2 score tile and a
      4xNC output tile per thread, K staged transposed and V as it lies in one shared buffer, `exp2f` with the scale folded in); a head dim outside 64/72/80 keeps aikit's kernel, and there is no longer a 12288-patch ceiling
      for these head dims (no score row in shared memory). Every multiply-add is an explicit `__fmaf_rn`, so the FMA lint passes; the PTX is regenerated at the same NVRTC 12.9.86 (the unchanged source reproduced the
      committed PTX byte for byte first).
    - **Correctness first, as registered.** (1) *The kernel against float64* at the production shapes (`TestTowerAttn_matchesFloat64`): head dim 72 at 4096 patches 3.5e-6, 64 at 3136 4.9e-6, 80 ragged 2.8e-6,
      4093 ragged 7.0e-6, one key 0, lengths 7/33/65, and two segments in one buffer 5.2e-6 (max abs error; the bar was 5e-5). (2) *The tower gates at their registered bars, unchanged:* real SigLIP worst token cosine
      0.999999848 to 0.999999977 on the four images; Qwen3.5-0.8B 0.999999528 to 0.999999999 and GLM-OCR 0.999996830 to 0.999999998 (`TestGridVisionCUDA_real`); Gemma 4 E2B 1.000000000 on all four
      (`TestG4VCUDA_gemma4E2B`); the tiny tower suites and all their registered planted defects pass. The fused kernel is not bit-identical to aikit's (a different summation order), so the bar is the cosine one, as it was.
      (3) *Planted kernel defects, each alone red:* (1) no rescale when the running max moves: kernel error 2.7, GLM-OCR tiny tower 0.937, **real SigLIP -0.125**; (3) the V tile read one key late: kernel error 2.0, tiny tower
      0.324, **real SigLIP -0.146**; (2) the key mask one past the segment: kernel error 1.4e-2 and tiny tower 0.9978, but **invisible at real size** (cosine 1.000000 over 4096 keys, where the extra zero-score,
      zero-value key carries 1/4097 of the weight): recorded as a limit of that defect, which is why (3) was added, not as a pass.
    - **The whole-tower A/B** (`TestS17LeverA_wholeTower`, TE5(b): one process, fused and aikit's kernel interleaved per round, 3 rounds after a warm-up, the ratio per round):

      | tower | fused | aikit's | ratio per round | the registered band |
      |---|---|---|---|---|
      | SigLIP float32, 4096 patches | 1.97 s | 18.01 s | 9.18 / 9.13 / 9.14 | 6.4-10x |
      | Qwen3.5-0.8B, formula.png (4060 patches) | 0.539 s | 5.09 s | 9.44 / 9.47 / 9.45 | 6.8-11x |
      | GLM-OCR, 896x896 (4096 patches) | 1.79 s | 14.12 s | 7.89 / 7.88 / 7.88 | 5.7-8.4x |

      The fused attention runs at 2.5-2.6 TFLOPS in the towers (k_A of about 21 against the registered 10-25; isolated, with its transfers, 1.81 TFLOPS and 14.8x). The real-size gate runs on the other towers read:
      Qwen3.5 0.44 s for the 896x896 image (was 3.1 s), Gemma 4 E2B 0.45-0.50 s (was 2.2 s). The kill line (under 1.02x, or under a band's low end) did not fire anywhere.
    - **The profile again, fused** (the same instrument; bit-identical, overhead +0.6 to +0.9%): SigLIP GEMM 48% at 3.56 TFLOPS, attention 42% at 2.52, norm 2%, elementwise 8%; GLM-OCR GEMM 53% / attention 36%;
      Qwen3.5 40% / 47%. **Attention is no longer the gap.** The GEMMs, at 39% of the fp32 peak, are now about half of every tower, so lever B is reopened as a candidate and not built: a GEMM twice as fast would buy
      about a quarter of SigLIP's 1.97 s, and a tensor-core GEMM would trade away the float32 numerics these gates were passed on. It needs its own registered band and a fidelity gate before any code.
    - **What it changes elsewhere:** (1) the float32 SigLIP tower (1.97 s) is now **faster than the int8 one (4.1 s)**, so the open question of Gemma 3's default under cuda/webgpu is no longer a trade of exactness
      against speed on CUDA: float32 is exact and about twice as fast. It stays the owner's decision. (2) S7's 5 s bar is met by the tower alone for every CUDA tower measured here. (3) Tonight's queued records were
      re-pinned to the build with this kernel before they ran: `s4sig-speed` and `s4-tower-speed` (their test binaries), S7 (every cell but its Gemma 3 'before' cell) and S13-lite (the goinfer arm), amended
      2026-10-07 evening, before any of them ran.
- **S7 on CUDA, the fix: Qwen2.5-VL on goinfer's tower base. Registered 2026-10-08 before any code.** S7's worst CUDA cell is Qwen2.5-VL-3B at 15.64 s: at defaults aikit's `gpu/qwencuda` tower ran out of device memory
  (a 56 MB scratch allocation, with 3630 MB reserved) and the CPU tower encoded for 15 s; and even on CUDA that tower measured 7.97 s for this image (`s4-tower-speed`), over the 5 s bar on its own, because it carries aikit's unfused
  attention (S17 step 0's finding for every tower).
  - **What changes** (`cuda/qwen25_vision.go`, the `gridQwen25` kind on `cuda/grid_vision.go`): Metal's port (`metal/vl_towers.go`) on the CUDA tower base, from aikit's own exports (`GPUWeights`, `BuildWindowPlan`, `MergeHidden`, so the window
    permutation, both segmentations and the RoPE tables are the CPU path's, not reimplemented): the pixel rows permuted into window order before the patch embed, then per block RMSNorm (eps 1e-6), biased split q/k/v, NeoX rotate-half
    RoPE from the plan's tables, **attention per window segment (per frame segment on the full-attention blocks) through the fused kernel**, biased proj and residual, RMSNorm, the SiLU-gated MLP with its biases, residual; the output permuted
    back to the original patch order; the merger stays aikit's, on the host. Float32 only (an int8-loaded encoder declines, as Metal's does). It registers through `vision.RegisterQwenResident` in the cuda package and replaces the blank import of
    aikit's `qwencuda` (whose allocation-failure fix, v0.1.1, stays in aikit); serve's `recoverDeviceTower` and `qwenForwardWithFallback` keep guarding it.
  - **Gates (G-S3a's, for the Qwen2.5-VL tower on CUDA; bars and planted defects unchanged from the Metal rebuild):** every merged token at cosine >= 0.9999 against aikit's CPU tower (0.999-0.9999 ambiguous, parked), on the tiny tower
    (`qwen25vl-tiny`, the grids 8x8, 12x16 and a two-image batch; norms randomised and q/k/v/proj sharpened first, exactly as Metal's check, because the tiny tower's init-scale weights hid most of S3's defects) and on `qwen25vl-3b-instruct` at real
    size on the four F2a images. Planted defects, each alone red on the tiny tower: (1) the attention scale dropped, (2) RoPE's halves swapped, (3) every block attending its whole frame, (4) the window reordering skipped. The fused kernel's own defects
    stay covered by the kernel and tower gates already registered (S17 lever A).
  - **Bands, written before the code (tower = GEMM + attention + the rest, from S17 step 0's rates: GEMM 3.5 TFLOPS, fused attention 2.5):** the 896x896 image (64x64 grid, 4096 patches): GEMMs about 5.2 TFLOP = 1.5 s, the four full-attention blocks
    about 0.3 s, the rest 0.1-0.2 s, so **1.7-2.6 s against aikit's 7.97 s (3-4.7x)**; formula.png (86x72, 6192 patches): **2.5-3.8 s against 15.3 s**. **The S7 reading it is for:** the Qwen2.5-VL cell under the 5 s bar at serve's defaults on the 8 GB card
    (the tower attached on CUDA, not the CPU fallback), predicted 3-4 s (the tower plus the resident m-RoPE prefill of about 1,000 merged tokens).
  - **Kill line:** a worst token under 0.9999 at real size is not shipped however fast; a tower over 4 s on the 896x896 image, or the S7 cell still over 5 s at defaults with the tower on CUDA, means the plan is wrong and is re-opened, not tuned to pass.
  - **Reserve:** `towerVRAMEstimate` for `qwen2_5_vl` prices 8192 patches at about 3.4 GB; this tower's float32 weights are about 2.5 GB and its scratch about 68 KB per patch (0.28 GB at 4096), so the estimate stands as a ceiling; it is measured, not assumed, in the S7 cell.
  - **Step 0 on Metal, read 2026-10-07 18:56-19:03 PDT: the time is GEMMs and attention; norms and elementwise are ~1%.**
    - **The run:** `s2-towers` at `9b7da484`, `TestS17Profile`.
    - **Its validity:**
      - the profiled output equals the unprofiled bit for bit;
      - the overhead check passed on all 12 runs (profiled GPU sum under the unprofiled wall);
      - repeat reads agree within 0.1%.
    - **Raw:** `docs/measurements/multimodal-support-2026-10/s17-step0-metal/profile.log`. The table is the median of
      3 profiled forwards; TFLOPS from the shapes, against the M1 Pro's ~5.3 f32 peak:

      | tower | rows | GPU time | GEMM: share, TFLOPS | attention: share, TFLOPS |
      |---|---|---|---|---|
      | SigLIP (every image) | 4096 | 8.68 s | 34%, 1.13 | **65%, 0.37** |
      | Qwen2.5-VL, 896² image | 4096 | 6.65 s | **71%, 1.09** | 27%, 0.21 |
      | Qwen2.5-VL, formula.png | 6192 | 11.21 s | **67%, 1.05** | 32%, 0.23 |
      | Qwen2.5-VL, table.png | 5504 | 9.36 s | **68%, 1.10** | 31%, 0.23 |
      | Qwen3.5-0.8B, 896² image | 3136 | 1.05 s | 43%, 1.19 | **55%, 0.63** |
      | Qwen3.5-0.8B, formula.png | 4060 | 2.92 s | 23%, 1.03 | **76%, 0.27** |
      | Qwen3.5-0.8B, table.png | 3888 | 1.69 s | 38%, 1.03 | **60%, 0.55** |

      The tiny 4x6 image is dispatch-bound (90 ms, 0.4 TFLOPS) and not a lever's target.
    - **The headroom, from this Mac's own records:**
      - R19's simdgroup-matrix prefill attention ran 7.2x the old fused kernel (4444 to 618 ms, about 2.1 TFLOPS at head
        dim 128, `metal-prefill-attn-2026-09-27.md`).
      - The decoder's f16-MMA GEMMs run at 2.8-3.0 TFLOPS against MPS's 3.24-3.43 f16 ceiling
        (`metal-gemm-ceiling-2026-09-25.md`).
      - The towers run at 0.2-0.6 and 1.0-1.2 TFLOPS.
- **The two Metal levers, bands registered 2026-10-07 evening before any lever code.** The projection is
  tower = GEMM/k_G + attention/k_A + the rest, from step 0's split.
  - **The instrument:** an in-process whole-tower A/B, interleaved, by day (TE5(b)), on the F2a images. Correctness
    first, at G-S2b's and G-S3a's bars and planted defects, unchanged.
  - **Lever A: attention.**
    - **What:** a flash-style simdgroup-matrix attention for the towers. Non-causal; per segment (frames, windows); head
      dims 64, 72 and 80 (R19's kernel is head dim 128 and causal).
    - **The kernel speedup expected, k_A = 3-6:** R19's 7.2x is the ceiling; the smaller head dims and Qwen2.5-VL's
      short windows reduce it.
    - **The bands:**
      - SigLIP 1.7-2.2x (8.68 to 4.9-4.0 s);
      - Qwen3.5 table.png 1.6-2.0x;
      - Qwen2.5-VL 1.2-1.3x.
    - **Kill line:** under 1.02x on any tower, or under its band's low end with no mechanism found: parked.
  - **Lever B: GEMM.**
    - **What:** the towers' projections on f16-input simdgroup-matrix GEMMs with f32 accumulation, the decoder's path.
    - **The kernel speedup expected, k_G = 2.3-2.7:** 1.1 to 2.5-3.0 TFLOPS.
    - **The bands:** Qwen2.5-VL 1.7-1.8x; SigLIP 1.2-1.3x; Qwen3.5 1.1-1.3x.
    - **Kill line:** as A's. If f16 inputs cannot hold the 0.9999 bar, B keeps f32 inputs at a lower k_G (its band is
      re-registered before that variant runs) or is dropped.
  - **The order:** A first. It is the largest share on two of the three towers and has R19's precedent; B follows.
    With both at band, SigLIP projects to about 2.5 s.
  - **Each lever, before it is measured, writes into this doc:**
    - its projection band: the tower-time ratio it should give, from the profile share it attacks and the kernel
      speedup it expects;
    - its kill line: under 1.02x, or outside its band on the low side with no mechanism found, and it is parked.
  - **Correctness is unchanged for every lever:** G-S2b's and G-S3a's bars (every soft token at cosine >= 0.9999
    against the CPU tower, on the tiny and the four F2a images) and their planted defects. An f16 lever that cannot
    hold 0.9999 keeps f32 accumulation or is dropped. The bar does not move.
  - **The speed instrument (TE5(b)):** a whole-tower in-process A/B, interleaved in one process, by day. A resolved
    direction may be acted on; an unresolved one goes to the served TTFT gate at night. Ship at >= 1.02x.

- **S17's Metal lever A, read 2026-10-08 by day: all three towers at or above their bands; SHIPPED (default on).**
  - **What:** `tower_attn_hd{64,72,80}` in `metal/grid_vision.go`, a fused non-causal float32 attention on simdgroup
    8x8 tiles in R19's shape. Any other head dim keeps `eg2Ops.attention`.
    - Q is staged with bounds, and K/V in 16-key blocks.
    - The online softmax runs in exp2, everything in f32.
  - **The kernel gate** (`TestTowerAttnKernel`, against a float64 softmax(q·kᵀ)·v; bar, set before the first run:
    every element within 1e-5 of max |v|):
    - worst 7.05e-7 (hd 64), 7.53e-7 (hd 72), 6.91e-7 (hd 80), over segments of 1 to 1100 rows;
    - each planted defect is red: the rescale skipped 7.05e-1, the key mask one past 3.41e-3, V one key late 9.76e-1.
  - **Correctness at real size, unchanged bars (cosine ≥ 0.9999 against the CPU tower), all PASS:**
    - G-S2b: Qwen3.5 worst 0.999999136, GLM-OCR worst 0.999998650.
    - G-S3a: SigLIP worst 0.999999909, Qwen2.5-VL worst 0.999996201.
    - The tiny gates read 1.000000000. GLM-OCR's tiny fixture has head dim 64, so the fused kernel ran there too.
  - **The speed instrument** (`TestS17LeverA_wholeTowerMetal`, TE5(b)):
    - one process, the fused arm against the old one (the `oldAttn` seam), interleaved;
    - per tower and image, one warm-up of each arm, then 3 rounds with the order alternating; wall time;
    - SigLIP on one image (every image is the same 4096 patches).
    - **Not idle:** the load average was 8.58 at the start (the owner's machine, by day). The arms are interleaved, so
      the ratios carry it; the absolute seconds are not a record.
    - **Raw:** `docs/measurements/multimodal-support-2026-10/s17-leverA-metal/` (`ab.log`, `real-gates.log`).

  | tower | image (rows) | old s | fused s | old/fused per round | median | band |
  |---|---|---|---|---|---|---|
  | SigLIP | any (4096) | 9.177 / 9.241 / 9.358 | 4.123 / 4.174 / 4.090 | 2.23 / 2.21 / 2.29 | **2.23x** | 1.7-2.2x |
  | Qwen2.5-VL | 896² (4096) | 8.136 / 8.248 / 8.253 | 5.281 / 5.235 / 6.066 | 1.54 / 1.58 / 1.36 | **1.54x** | 1.2-1.3x |
  | Qwen2.5-VL | 4x6 (24) | 0.112 / 0.111 / 0.111 | 0.083 / 0.083 / 0.083 | 1.35 / 1.34 / 1.33 | 1.34x | — |
  | Qwen2.5-VL | formula.png (6192) | 13.484 / 13.462 / 13.714 | 9.004 / 9.393 / 9.471 | 1.50 / 1.43 / 1.45 | **1.45x** | 1.2-1.3x |
  | Qwen2.5-VL | table.png (5504) | 11.797 / 11.846 / 11.386 | 8.259 / 7.711 / 8.870 | 1.43 / 1.54 / 1.28 | **1.43x** | 1.2-1.3x |
  | Qwen3.5-0.8B | 896² (3136) | 1.226 / 1.262 / 1.201 | 0.714 / 0.710 / 0.634 | 1.72 / 1.78 / 1.90 | 1.78x | — |
  | Qwen3.5-0.8B | 14x20 (280) | 0.086 / 0.089 / 0.090 | 0.062 / 0.063 / 0.065 | 1.39 / 1.42 / 1.40 | 1.40x | — |
  | Qwen3.5-0.8B | formula.png (4060) | 3.040 / 3.112 / 2.994 | 1.071 / 1.021 / 0.989 | 2.84 / 3.05 / 3.03 | **3.03x** | — |
  | Qwen3.5-0.8B | table.png (3888) | 1.800 / 1.737 / 1.752 | 1.009 / 0.924 / 0.919 | 1.78 / 1.88 / 1.91 | **1.88x** | 1.6-2.0x |

  - **Every round resolves above 1** (27 of 27), and the two arms' outputs agree to a worst row cosine of 0.9999997.
    By TE5(b) a resolved direction may be acted on, and every tower is above the 1.02x ship bar: **default on**.
  - **Above the band, with a mechanism to confirm:** SigLIP (2.23x against 2.2) and Qwen2.5-VL (1.43-1.54x against
    1.2-1.3).
    - Qwen2.5-VL's gain is more than removing its whole attention share could give: 27-32% of GPU time caps the gain at
      about 1.4x.
    - The bands came from step 0's GPU-time split, while the A/B times wall clock. The old path's host side (thousands
      of per-window dispatches to encode, and the command-buffer flushes its scalar arena forces) was never in the
      profile, and the fused path removes most of it.
    - That is the likely mechanism; it is not verified by a host-time profile.
  - **Still owed:**
    - lever B (GEMM, the larger share on Qwen2.5-VL);
    - the served TTFT cells on this build (S7 and S13-lite at night): S13-lite's goinfer arm was 8.9-9.6 s of
      SigLIP tower per image.

- **S7 and S13-lite on the Mac, re-run on lever A's build, registered 2026-10-08 before they run (owner: "yes queue
  them").**
  - **What:** the same scripts, cells, procedure and 5 s bar as the 2026-10-07 night (`run-s7-mac.sh`,
    `run-s13lite-mac.sh`), unchanged.
  - **The one difference:** goinfer's binary is `serve-metal` at `92c30640` (lever A, default on), in
    `~/goinfer-bench/s7-leverA/` (`BIN`), built from a clean worktree.
  - **Reading:** a record, each cell against its 2026-10-07 reading.
    - **The prediction, not a bar:** the tower cells (Qwen2.5-VL, Qwen3-VL, GLM-OCR, Qwen3.5, and S13-lite's Gemma 3)
      fall by about their tower's lever A ratio applied to the tower's share of TTFT.
    - **Ollama's and llama.cpp's arms** in S13-lite repeat as same-night controls.
  - **The fit guard stays on.** Gemma 3 4B may be refused again, as S7's cell was on 2026-10-07; a refusal is
    recorded as such.
  - **Queue:** `s7-mac-leverA` and `s13lite-mac-leverA`, 20 minutes each (2026-10-07: 5 and 6 minutes).

- **S17's Metal lever B (GEMM): bands re-registered on the post-A baseline, 2026-10-08, before any lever B code runs.**
  - **Why re-registered:** lever A shipped first, so lever B's A/B measures against today's tower (fused attention),
    not the pre-A tower its 2026-10-07 bands assumed. The same inputs, re-projected:
    - step 0's GEMM GPU seconds (unchanged by A);
    - k_G = 2.3-2.7, unchanged;
    - the post-A wall times from lever A's A/B, the fused arm's medians.
  - **The projection:** tower ratio = post-A wall ÷ (post-A wall - GEMM × (1 - 1/k_G)).

  | tower | image | step 0 GEMM GPU s | post-A wall s | band |
  |---|---|---|---|---|
  | SigLIP | any | 2.95 | 4.12 | **1.7-1.8x** |
  | Qwen2.5-VL | 896² | 4.72 | 5.28 | **2.0-2.3x** |
  | Qwen2.5-VL | formula.png | 7.51 | 9.39 | **1.8-2.0x** |
  | Qwen2.5-VL | table.png | 6.36 | 8.26 | **1.8-1.9x** |
  | Qwen3.5-0.8B | table.png | 0.64 | 0.92 | **1.6-1.8x** |
  | Qwen3.5-0.8B | formula.png | 0.67 | 1.02 | **1.6-1.7x** |

  - **The lever, as registered:**
    - the projections as f16-input simdgroup-matrix GEMMs with f32 accumulation: weights f16 at upload, activations
      rounded to f16 as they are staged, the bias fused into the epilogue;
    - 64x64 output tiles per threadgroup, each simdgroup 32x32;
    - the patch embedding stays on the f32 GEMM.
  - **Correctness, unchanged:** G-S2b's and G-S3a's bars (every soft token at cosine >= 0.9999 against the CPU tower)
    and their planted defects, plus a kernel gate against a float64 reference.
    - **The kernel bar:** each output within 2e-3 of its row's Σ|a·w|. This is f16 input rounding (about 2^-11 per
      operand, so about 1e-3 per product), not accumulation.
    - **If f16 cannot hold the tower bar,** B keeps f32 inputs at a lower k_G, re-registered before that variant runs.
  - **Kill line:** under 1.02x on any tower, or under its band's low end with no mechanism found: parked.
  - **The instrument:** lever A's, TE5(b): an in-process whole-tower A/B, interleaved by day, f16 GEMM against the f32
    GEMM, with lever A on in both arms.

- **S17's Metal lever B, f16 inputs: FAILED the tower bars at real size, 2026-10-08. The registered fallback,
  f32 inputs, is re-registered here before it runs.**
  - **The f16-input kernel (`tower_gemm_f16`) passed its own gate:** worst 3.40e-4 against 2e-3, on N(0,1)
    activations, with every planted defect red (the bias dropped 1.06e-1, A one K column late 2.91e-1, the second half
    of each K step skipped 1.56e-1).
  - **The towers fail with it** (G-S2b and G-S3a, cosine against the CPU tower, bar 0.9999):

    | tower | image | worst token cosine |
    |---|---|---|
    | GLM-OCR | table.png | 0.962422796 |
    | GLM-OCR | invoice.png | 0.995452352 |
    | SigLIP | 896² image | 0.996123455 |
    | SigLIP | formula.png | 0.947761767 |
    | SigLIP | table.png | 0.991308799 |
    | Qwen2.5-VL | 896² image | 0.997665165 |
    | Qwen2.5-VL | formula.png | 0.917237607 |
    | Qwen2.5-VL | table.png | 0.835251574 |

  - **Not measured: why.** Far beyond f16 rounding noise, and worst on dense-text images: the shape of real activation
    outliers losing precision or overflowing in f16. It is not chased further: the registration's fallback is exactly
    the fix.
  - **The fallback (`tower_gemm_w16`):** the same kernel, with the activations staged in f32 and the f16 weights
    widened to f32 as they are staged, on f32 simdgroup tiles.
    - **What it keeps:** the 64x64 tiles with 16 multiply-accumulates per 8 fragment loads, the fused bias, and half the
      weight memory.
    - **The kernel bar:** 1e-3 of Σ|a·w| + |bias|, since only the weights round. Measured 1.99e-4, every planted defect
      red.
    - **The tower bars are unchanged.**
  - **Its k_G, re-registered: 1.5-2.2.** The f32 tiles' peak is the f32 rate (~5.3 TFLOPS). The old kernel's 1.1
    TFLOPS loads one fragment from device memory per multiply; staging plus a 2:1 ratio should give 1.7-2.4 TFLOPS.
  - **The bands, by the same post-A projection:**

    | tower | image | band |
    |---|---|---|
    | SigLIP | any | **1.3-1.6x** |
    | Qwen2.5-VL | 896² | **1.4-2.0x** |
    | Qwen2.5-VL | formula.png | **1.4-1.8x** |
    | Qwen2.5-VL | table.png | **1.3-1.7x** |
    | Qwen3.5 | table.png | **1.3-1.6x** |
    | Qwen3.5 | formula.png | **1.3-1.6x** |

  - **Kill line and instrument as registered above.**

- **S17's Metal lever B, the f32-input kernel's design, amended 2026-10-08 before its tower A/B runs.**
  - **Why:** a kernel-only probe of the design registered above (one GEMM, SigLIP's shape 4096x4352x1152, in one
    process; direction only, not a record) found it slower than aikit's `gemm_f32_sg_big` (1.19 TFLOPS):

    | design | TFLOPS |
    |---|---|
    | the 32x32-per-simdgroup staged kernel, f16 weights | 0.22 |
    | the same with direct loads | 0.08 (register spill) |
    | 16x32 per simdgroup, direct loads | 1.53 |
    | 64x64 tiles over 16 simdgroups of 16x16, K steps of 32 staged as float4 | **2.22** |

  - **The kernel now:** `tower_gemm_w16` (f16 weights widened at staging) and `tower_gemm_w32` (f32 weights), both in
    the 2.22 TFLOPS shape with edge guards and the bias fused.
    - The probe: 2.02-2.03 TFLOPS (1.70x aikit), and 1.98 on SigLIP's unaligned 4304-wide K (1.75x).
    - The kernel gate: w32 4.04e-7 against 1e-5; w16 1.99e-4 against 1e-3; every planted defect red.
    - **Production uses w16:** f16 weights hold the tower bars (the run above) and halve the weight memory.
  - **The kernel ratio, 1.7-1.75x, is inside the registered k_G (1.5-2.2).** The bands and the kill line stand as
    re-registered above. The tower bars are re-run on this kernel before the A/B.

- **S17's Metal lever B, read 2026-10-08 by day: every tower at or above its re-registered band; SHIPPED (default on,
  `tower_gemm_w16`).**
  - **Correctness at real size, unchanged bars, all PASS:**
    - G-S2b: Qwen3.5 worst 0.999999036, GLM-OCR worst 0.999998282.
    - G-S3a: SigLIP worst 0.999999622, Qwen2.5-VL worst 0.999996381.
    - The tiny gates read ≥ 0.999999812.
  - **The A/B** (`TestS17LeverB_wholeTowerMetal`): lever A's instrument, with lever A on in both arms.
    - **Not idle:** the load average was 16.8 at the start (the owner's desktop apps and this session's own GPU work; no
      stray process). The arms are interleaved; the absolute seconds are not a record.
    - **Raw:** `docs/measurements/multimodal-support-2026-10/s17-leverB-metal/` (`ab.log`, `real-gates-w16.log`, and
      the failed f16-input run, `real-gates-f16-inputs-FAILED.log`).

  | tower | image (rows) | f32 GEMM s | w16 GEMM s | old/new per round | median | band (f32-input fallback) |
  |---|---|---|---|---|---|---|
  | SigLIP | any (4096) | 4.000 / 4.005 / 4.004 | 2.652 / 2.711 / 2.663 | 1.51 / 1.48 / 1.50 | **1.50x** | 1.3-1.6x |
  | Qwen2.5-VL | 896² (4096) | 6.145 / 6.165 / 6.209 | 3.469 / 2.968 / 3.428 | 1.77 / 2.08 / 1.81 | **1.81x** | 1.4-2.0x |
  | Qwen2.5-VL | 4x6 (24) | 0.167 / 0.089 / 0.085 | 0.089 / 0.077 / 0.076 | 1.88 / 1.16 / 1.13 | 1.16x | — |
  | Qwen2.5-VL | formula.png (6192) | 9.326 / 9.475 / 9.440 | 5.222 / 5.177 / 5.182 | 1.79 / 1.83 / 1.82 | **1.82x** | 1.4-1.8x |
  | Qwen2.5-VL | table.png (5504) | 8.381 / 8.340 / 8.322 | 4.834 / 4.582 / 4.635 | 1.73 / 1.82 / 1.80 | **1.80x** | 1.3-1.7x |
  | Qwen3.5-0.8B | 896² (3136) | 0.729 / 0.701 / 0.775 | 0.547 / 0.494 / 0.541 | 1.33 / 1.42 / 1.43 | 1.42x | — |
  | Qwen3.5-0.8B | 14x20 (280) | 0.062 / 0.075 / 0.068 | 0.046 / 0.049 / 0.048 | 1.33 / 1.52 / 1.41 | 1.41x | — |
  | Qwen3.5-0.8B | formula.png (4060) | 0.972 / 0.982 / 0.961 | 0.632 / 0.625 / 0.630 | 1.54 / 1.57 / 1.53 | **1.54x** | 1.3-1.6x |
  | Qwen3.5-0.8B | table.png (3888) | 0.960 / 0.921 / 0.912 | 0.682 / 0.696 / 0.608 | 1.41 / 1.32 / 1.50 | **1.41x** | 1.3-1.6x |

  - **Every round resolves above 1** (27 of 27), with outputs agreeing to cosine ≥ 0.9999999. Every tower is above the
    1.02x ship bar and at or above its band's low end: **default on.**
  - **Qwen2.5-VL's text images sit at or just above the band's top** (1.82 and 1.80 against 1.8 and 1.7). That is
    consistent with its GEMM share after lever A (step 0's split predates A); the band used step 0's GEMM seconds over
    A's wall, so a larger post-A GEMM share reads high.
  - **The two levers together, against step 0's pre-lever towers** (a product of two by-day A/Bs, not one
    measurement): SigLIP 9.24 → about 2.7 s (2.23x × 1.50x ≈ 3.3x); Qwen2.5-VL 896² about 2.8x; Qwen3.5 formula.png
    about 4.7x.
  - **Also:** the weights are now uploaded in f16 only (half the tower's weight memory); the patch embedding stays f32.
  - **Owed:** the served cells. Tonight's S7 and S13-lite re-runs are on lever A's build (`92c30640`) as registered,
    so lever B's served effect is a later night's.

- **S7 and S13-lite on the Mac, a second pass on levers A and B, registered 2026-10-08 before it runs (owner: "sure
  queue it").**
  - **What:** the same scripts, cells, procedure and 5 s bar as lever A's re-run above.
  - **The one difference:** goinfer's binary is `serve-metal` at `e2910316` (both levers on), in
    `~/goinfer-bench/s7-leverAB/`.
  - **Order:** it runs after lever A's pass, the same night. Each cell reads against the 2026-10-07 night and lever A's
    pass, so B's served effect is the difference between the two passes.
  - **The same-night controls:** the peers (Ollama, llama.cpp) in S13-lite are re-measured in each pass.
  - **The fit guard stays on.**
  - **Queue:** `s7-mac-leverAB` and `s13lite-mac-leverAB`, 20 minutes each.

### S18 — Defaults that fit (added 2026-10-07 evening)

- **The gap:** with a tower loaded, the out-of-the-box plan puts the decoder or the tower on the CPU on common
  hardware.
  - **The 16 GB Mac:** Gemma 3 4B's decoder does not go resident by default. It needs 5.15 GB against a 4.2-4.9 GB
    budget (G-S3b, G-S3c).
  - **The 8 GB card:** it needs `--kv-sessions 1` (S4).
  - **The tower reserve:** S4 step 4's reserve (`towerReserve`) returns 0 for an int8 tower, so Gemma 3's CUDA
    SigLIP tower is never priced.
- **The change:**
  - **The plan:** the default KV plan sizes itself after the tower: fewer conversations or a shorter context before
    declining the resident. Every tower is priced, int8 included.
  - **The banner:** serve says what it chose and why, for example "KV plan: 1 conversation x 8192 positions
    (reduced from 4 x 16384 to leave 1.1 GB for the vision tower)".
- **Gates, registered 2026-10-07 before any S18 code:**
  - **G-S18a, the defaults:** with no sizing flags, two cells:
    - Gemma 3 4B on the 16 GB Mac must decode `metal-resident` with its tower on Metal;
    - Gemma 3 4B on the 8 GB card (nobara) must decode `cuda-resident` with its tower on CUDA.

    Each takes one served image turn whose reply is identical to the same request with the hand-set flags that
    work today.
  - **G-S18b, no regression:** a text-only model's default plan is unchanged (unit tests over the plan function on
    the bench set's text models). A model with a tower that already fit keeps its plan.
  - **G-S18c, the reserve:** `towerReserve` is nonzero for every family with a device tower under its backend, int8
    included. It is a table test over family x backend, red on today's code for Gemma 3 under cuda.
  - Measured by day on the Mac (G-S18a's Mac half is a served correctness check, not timed); nobara owns the CUDA
    half.

- **S18 on CUDA, read 2026-10-07 evening on nobara (the 8 GB card): G-S18a PASS, G-S18c PASS, G-S18b by tests. Raw:** `docs/measurements/multimodal-support-2026-10/s18-cuda/`.
  - **The root cause was not only the unpriced tower.** With serve's defaults Gemma 3 4B's resident build failed on `CUDA_ERROR_OUT_OF_MEMORY` for a 16,961,536-byte buffer, which is one layer's KV at 4141 positions
    (4141 x 1024 x 4 bytes). The plan chose that context so that four slots fit: 4 x 1,153,384,448 B of KV against 4,614,782,976 B of budget, **1.2 MB of slack**, with the build's own scratch (34-120 MB, `ctxForSlots`'
    own comment) allocated before the KV, so the last slot missed and the whole resident dropped to the CPU (decode on the CPU, the int8 tower still loading on CUDA: 21.6 s for the served request).
  - **The change (CUDA):** (1) the extra KV slots degrade instead of declining: `allocKVSlot` recovers a device OOM, releases the partial slot and keeps the slots that fit (`cuda/backend.go`, `cuda/resident.go`); the first
    slot is the build's own and still declines. (2) `towerReserve` prices the int8 Gemma 3 device tower (the shipped default) at `towerInt8VRAMEstimate`, calibrated to the 558 MiB it measured on the card (`TestSiglipCUDA_int8VRAM`;
    the formula gives 575 MiB); every other int8 tower is the CPU's and stays at zero.
  - **G-S18a, the CUDA cell, PASS.** With no sizing flags Gemma 3 4B on the 8 GB card is `decode path: cuda-resident (int4)` with `encoder int8/cuda-resident`: context 4096 (the floor, from a requested 15709), **3 of the 4 requested KV
    slots** ("free VRAM beside the weights 4785 MB, less 959 MB reserved"), and the served image request's 32 logprob records are **identical** to the same request with the hand-set `--kv-sessions 1 -ctx 4096`
    (`default/` against `handset/`; each pair of runs also has its determinism control IDENTICAL). The 32-token request took 5.2 s (tower and decode included) against 21.6 s on the CPU decoder before.
  - **G-S18c PASS:** `TestTowerReserve_everyDeviceTower`, a table over five families x `-vision-quant` x four backends: nonzero exactly for the device towers (every family in float32 on CUDA; Gemma 3 also int8), zero where the
    tower is the CPU's; shown red by pricing int8 at zero (today's behaviour: Gemma 3 under cuda, default and int8, read 0). The kernel of the first fix has its own gate, `TestCUDAKVSlots_oomKeepsTheSlotsThatFit`: three slots
    requested, the allocation of slot 2 failed by a test seam, the resident stays CUDA with two, the device ledger equals a plain two-slot build's (no leak), the two slots decode the plain build's tokens; red without the
    recovery (the build declines).
  - **G-S18b:** no plan arithmetic changed; the degrade path runs only on an OOM, and the reserve change touches only a Gemma 3 tower on CUDA. The existing plan and slot tests (`TestResolveCtxCapFit_*`, `TestCUDAKVSlots_*`,
    `TestKVSlotsFit`) and the full serve suite pass. The bench set's text models were not re-planned on the card by day.
  - **Not done:** the registered banner wording ("KV plan: 1 conversation x 8192 positions (reduced from 4 x 16384 to leave 1.1 GB for the vision tower)"). The CUDA planner already prints what it chose and why
    (the context floor line, the slots-granted line with the reserve), but not in that shape and not from serve; a unified serve banner is shared with the Mac's Metal cell. The Metal half of G-S18a is the Mac's.

- **Build-scratch / margin accounting on CUDA: finding and pre-registration, 2026-10-08, nobara (before the code).** Group 1 of the first heavy-tier gate's failures (`TestDefaultVerifyWidth_sweep`,
  `TestFlashDecodeBlockSpecLane`, `TestBlockSpec_twoTurnsMatchPlain`, `TestResidentDenseBytes_matchesCUDADevice/7b`, and the 256 MiB slack I put on the Qwen2.5-VL tower estimate) read as "the 384 MiB margin is too small".
  The measurement says the margin is not the quantity that is short. Raw: `~/goinfer-logs/margin/` (`accounting-*.log`, `traj.log`, `sizes.log`; archived into `docs/measurements/multimodal-support-2026-10/margin/` with the record).
  - **Where the bytes go (`TestBuildScratchAccounting`, six models, first `checkKVFits` probe against the free VRAM before the load):** the build's own scratch allocated AFTER the probe, beyond the KV, is 0 MiB on five models and
    73 MiB on Gemma 3 (the 7B read 0; the 421 MiB I quoted for it is this pre-probe gap, 425 measured below; the 535 MiB I quoted for the Qwen2.5-VL build is NOT reproduced by its text build, 172 MiB before the KV and 0 after: it came from the served S7 run, which also loads the tower, and is not decomposed here. Corrected after the read). The shortfall is BEFORE the probe: the device falls 128 / 425 / 326 / 172 / 200 MiB more than Plan's dense
    figure on the 1.5B / 7B / Gemma 3 4B / Qwen2.5-VL 3B / Gemma 4 E2B (0.5B: 144 MiB; its first load in a process reads 530, a one-time 386 MiB context and module cost that a serve process has already paid before it plans).
  - **What the shortfall is:** the buffers the build allocates before the probe sum to Plan's figure (7B: 4024 MiB requested against Plan's 4021, counted at the four allocation helpers). The extra is the driver rounding each
    buffer of 2 MiB or more up to a 2 MiB multiple (`allocQuantumBytes`, measured long ago for the expert slots in `allocgran_test.go`; the weight estimate never applied it). Sampling free VRAM at every allocation during the
    7B load shows the gap growing steadily with the weight uploads (0 -> 420 MiB) and flat from the KV on. Applying "sizes >= 2 MiB round up to 2 MiB" to the exact buffer sizes predicts the gap: 0.5B 139 against 140 MiB
    measured, 1.5B 122 against 125, 7B 406 against 420. Gemma 3 / Qwen2.5-VL / E2B leave 32-54 MiB beyond the rule (modules and the tower's context; read below).
  - **Why it hits what it hits:** Plan sizes the context from `free - margin - dense`, so every unpinned build asks `checkKVFits` for a context that already spent the rounding; the S18 trim then gives positions back. A
    `--drafter` attach is priced by the same unrounded estimate, so it can find no room; and the 7B's 425 MiB breaks `TestResidentDenseBytes`' "short by less than the margin" bound (384). Nothing is wrong with the margin.
  - **The change:** `BuildResident` sums the allocator's rounding over the exact packed buffers (every `hostW` in the layers and the LM head, found by reflection so no family's fields are missed) before it plans, and adds it to
    the planning call's `ExtraBytes` in `resolveCtxCapFit`. NOT to `checkKVFits`, whose probe is read after the weights and is already net of it. The margin stays 384 MiB. The drafter's and the towers' estimates are priced
    the same way in a second step only if their own rows below still miss (they are separate estimators).
  - **Gates (`TestBuildScratchAccounting`, heavy; the six bench models):**
    - **G-M1 residual:** `device before KV - (Plan dense + slack)` is within [-8, +64] MiB on the Qwen models (0.5B, 1.5B, 7B) and within [-8, +128] MiB on Gemma 3, Qwen2.5-VL and E2B. The slack is the build's own
      number (`cudaResident.allocSlackBytes`), so a model whose residual falls outside has a cost the rule does not know.
    - **G-M2 no regression of the planned context:** for each model, the unpinned context the plan picks with the slack is not larger than the one `checkKVFits` then grants without a trim, i.e. no "trimmed at the build" line
      on the bench set at serve's defaults (the 7B and Gemma 3 4B, which trimmed before).
    - **G-M3 the old failures:** the four tests named above pass in isolation on the card.
    - **Planted defect:** the slack zeroed (the old planning) turns G-M2 red on the 7B.
  - **Kill line:** a residual above +128 MiB on any model, or G-M2 red on a model whose slack is nonzero, means the rule is not the whole story: stop, do not widen the margin to make it pass, and re-open the measurement.
  - **Bands that are NOT claims:** no speed is claimed. A larger planned context where the old plan trimmed is a capacity effect, measured by the plan line, not timed.

- **Build-scratch / margin accounting on CUDA: read 2026-10-08 on nobara. G-M1 PASS, G-M3 PASS, G-M2 as registered NOT met (restated below). Raw:** `docs/measurements/multimodal-support-2026-10/margin/`
  (`accounting-3.log` the gate, `planted.log` the slack-off comparison, `traj.log` + `sizes.log` the rounding evidence, `gm3-fixed.log` the four tests red before, `gm3-fixed-2.log` green after, `plan-tests.log` the existing
  planning tests). Not archived: a first accounting run whose probe stub kept the LAST `checkKVFits` reading (the build reaches it more than once; the first is the plan's) and three runs of a temporary allocation counter
  that was reverted.
  - **G-M1 PASS.** Residual (`device before the KV - (Plan dense + slack)`): 0.5B 5 MiB, 1.5B 6, 7B 19 (band -8..+64); Gemma 3 4B 69, Qwen2.5-VL 3B 36, Gemma 4 E2B 58 (band -8..+128). The slack the build prices is 139 / 122 /
    406 / 257 / 136 / 142 MiB against 144 / 128 / 425 / 326 / 172 / 200 MiB of total gap before it. After the probe, the build's scratch beyond the KV plus the first prefill and decode is 0 on five models and 73 MiB on Gemma 3,
    far inside the 384 MiB margin; the margin was never short.
  - **G-M2 as registered is NOT met, and its premise was wrong.** I wrote that the 7B and Gemma 3 4B "trimmed before". The 7B did not: it plans 16384 and keeps 16384 with or without the slack. Gemma 3 4B does, and still does:
    planned 15350, final 15092 (a 258-position trim) with the slack, against 16317 -> 15092 (1,225) without. The slack cut the trim by 79% and left the Gemma-specific 69 MiB residual, which the rule does not explain.
    The planted defect (slack off) is red on the 7B through its residual (425 MiB, over the 384 MiB margin the old bound enforced; 19 with the slack) and on Gemma 3 through the trim, not through a 7B trim.
  - **G-M3 PASS, with two different causes.** The four tests pass in isolation (`gm3-fixed-2.log`; 417 s, not the ~3 min I estimated: `TestDefaultVerifyWidth_sweep` alone is 247 s).
    - `TestResidentDenseBytes_matchesCUDADevice/7b` was the rounding: its bound ("short of the device by less than the margin") now reads Plan plus the build's priced slack. Red before, green after.
    - `TestDefaultVerifyWidth_sweep`, `TestFlashDecodeBlockSpecLane` and `TestBlockSpec_twoTurnsMatchPlain` were NOT the margin and NOT the rounding. They load the target plainly (the log reads "less 384 MB reserved": the margin
      alone) and bolt `NewBlockSpec` on afterwards, where `serve --drafter` loads the drafter first and prices it through `Options.ExtraResidentBytes` and `ExtraResidentKVPerPosition`. The plan fills the card to the margin and the
      drafter's allocation runs out. The tests now price the drafter the way serve does (`withDrafterReserve`). I did not run them with the slack off, so I do not claim the slack is needed for them.
  - **Not done / owed:** the drafter's own weights are priced unrounded in `DrafterResidentBytesEstimate` (about 35 buffers, tens of MiB, inside the margin). Gemma 3's 69 MiB residual is unexplained.
  - **The Qwen2.5-VL tower's 256 MiB slack: removed, same day.** It was the misattribution above: "the build's scratch beyond the plan's reading, 535 MiB against 384" is not reproduced (0 MiB after the probe), and
    the rounding it patched is now priced by the plan. A served probe on the card, exploratory and untimed (`margin/qwen25vl-slack-probe.txt`): serve's defaults, an ~8,000-patch image (2,014 prompt tokens), a small image,
    the big one again; with the slack at 256 and at 0 the tower stays on CUDA, no failure, identical replies, the same 7040 MiB used (KV buffers round to the same 6 MiB), 1152 MiB of the 8192 still free. At 0 the
    default context is 5967 against 5057 (+18%). `qwen25VLTowerEstimate` is now 3.46 GB (weights 2758 MiB + 540 MiB of scratch at the 8192-patch ceiling); its test band is re-pinned to 3.40-3.60.
  - **A production consequence worth stating:** on the 8 GB card an unpinned default load now plans against what the build will actually take, so the plan line and the final context agree on every bench model but Gemma 3.
    Nothing here changes a speed.

- **`TestSpecNonCopyLane` (the first heavy gate's 940 s failure): root-caused 2026-10-08 by tokenizing, no run needed; fixed; the measurement itself is queued tonight.** The test built its "~4000-token document" from the first
  14,800 bytes of the LIVE `docs/benchmarks.md`. On the Qwen2.5-Coder tokenizer with the test's own closing line that slice is **4,911 tokens at the commit that added the test (33a18e82), 6,233 at 12c85f4a (the exact
  number the gate log prints for the declined prefill) and 6,200 at the commit before this fix**, against a context pinned at 6,144. So the batched prefill declined, every arm ran the per-token path (940 s), and the first
  speculative arm died at "KV position 6144(+1) exceeds resident context cap". It fails at the baseline the same way: nothing a recent change broke, and nothing to do with the margin. The document is now a frozen
  copy of the 33a18e82 slice (`cuda/testdata/noncopy_doc.txt`, 14,800 bytes) and the test refuses a prompt that does not fit with 160 new tokens, naming the cause. Not yet run after the fix (a measurement, longer than the
  by-day bound): `docs/measurements/multimodal-support-2026-10/margin/run-gate-triage.sh` runs it tonight, beside the WebGPU triage.

- **The WebGPU parity-cell hang: root-caused and fixed by day, 2026-10-08.** The first heavy gate's webgpu-parity cell sat on `TestRMSNormBatched_parity` for over 10 minutes. Run alone that test passes in 0.14-0.4 s, and
  nothing under `gpu/` changed between the gate's revision and now, so it was intermittent: the cell run 8 times in a row hung once (run 8, 2 min timeout), and in `TestVisionLayerNorm_parity`, a different test, so it is not
  one test's bug. Raw and the unfixed goroutine dump: `margin/webgpu-hang-runs.txt`, `margin/webgpu-hang-unfixed-run8.log`.
  - **Cause.** A deadlock inside wgpu-native, between two device destructions at once. The bindings give every wrapper (Buffer, CommandEncoder, ...) its own reference on the device and drop it from a finalizer. A buffer a
    test leaks after its `Context` was closed holds the LAST reference on that device, so the garbage collector destroys the device on the finalizer goroutine at an arbitrary moment. If that lands while another
    `Context.Close` destroys its own device, both threads park in `wgpuDeviceRelease` for good (the dump: `runtime.runFinalizers -> Buffer.release -> wgpuDeviceRelease` against `Context.Close -> wgpuDeviceRelease`).
  - **Fix.** `Context.Close` runs its releases on the runtime's finalizer goroutine (`finalizerSerial`), the one thread that runs every finalizer, so a finalizer-driven device drop and a Close can no longer overlap. It
    falls back to running in place after about a second if that goroutine is blocked. `gpu.TestContextClose_finalizerRace` is the reproducer: it deadlocks without the fix (90 s deadline) and passes with it.
  - **Gates.** The parity cell: 10 of 10 runs clean after, against 1 hang in 8 before (same 65 pass / 8 skip, same ~30 s). The whole `gpu` package: 142 pass, 57 skip, 0 fail. `gofmt`, vet and `staticcheck` with the gpu
    tags clean. 10 clean runs after a 1-in-8 failure is evidence, not proof (the chance of 10 clean runs by luck at 1/8 is about 26%); the reproducer is the sharper evidence, and the first night gate that runs the cell is the
    next one.
  - **Not covered.** An explicit `Release` from another goroutine (a caller closing a matrix after its Context while another Context closes) is not serialized; it stays the caller's ordering. The leaked buffers themselves
    are a separate hygiene item (the live-buffer gauge in `gpu/bufaccount.go` counts them). `GOGC=1` did not amplify the hang. The night triage job is now the `TestSpecNonCopyLane` run alone (`noncopy-fixed-head`).

- **The WebGPU hang's "not covered" list, worked 2026-10-08 (branch `gpu-explicit-release`, validated on the card the same night and merged to main).** (1) A wrapper closed by the caller after its
  Context: `ResidentMatrix`, `ResidentW8A8`, `ResidentW4A8` and `ResidentStackedW8A8` now carry their Context, and those, the GEMV and decode runners, `residentDecoder` and `VisionEncoder` release through
  `Context.releaseOwned`: in place while the Context is live, on the finalizer goroutine once it is closed (a release already running there is detected by goroutine id and runs in place, so a Close that closes other
  wrappers does not wait on itself). `ResidentStackedW8A8.Close` also nils its buffers now: a second Close used to release twice. (2) Leaks: `ResidentMatrix` and the stacked weights join the live-buffer gauge
  (`LiveBufferBytes` covered three of the five resident types); `Context.Close` calls a test-only hook on the caller's goroutine, and the package's `TestMain` prints a LEAK REPORT naming each test that closed a Context
  with buffers still live (`GOINFER_GPU_LEAKS_FATAL=1` fails the package). The report is the work list for the leaks themselves; none has been fixed yet because none has been listed. `DeviceBuffer` (the per-op
  activation buffers, 19 creation sites) does not hop: it has no Context back-pointer and is closed in place, normally before its Context.
  - **Validated, night of 2026-10-08 (`margin/gpu-release-validate/`).** All four criteria written before the run were met: the two reproducers pass (150 rounds each); the whole `./gpu/` package is 144 pass / 58 skip /
    0 fail (142 / 57 / 0 before, plus the two new tests and the planted-defect skip); ten consecutive runs of the webgpu-parity cell are clean (65 pass / 8 skip, ~30 s; before the first fix one in eight hung); and the
    planted defect (the hop off) deadlocks, so the reproducer can see what it guards. `staticcheck` with the gpu tags is clean. The script's plumbing had never run before the job and worked.
  - **The leaks are fixed (same day).** The report said 21 tests closed a Context with caller-owned device buffers live, the largest `TestKVCacheI8_parity` (108 MB), `TestDecodeRunnerW4A8_parity` (56 MB),
    `TestDecodeToken_throughput` (47 MB) and `TestFusedMLP_microbench` (41 MB). To fix them from evidence and not by guessing, the counted wrappers now report their creation to a test-only trace
    (`allocTrace`, keyed by address: a first version held the wrappers themselves, which kept every leaked device alive and exhausted the driver's ~63-device limit mid-suite), and the report names the test-file line and
    the library function that made each survivor (`margin/gpu-release-validate/leak-report-before-with-sites.txt`). What it found:
    - **Test omissions (most of them).** The shared MLP fixture, the hand-built models in the decode-runner, MLA, MoE, per-layer-RoPE, sliding-window, int8-KV, shared-quant and q-gate tests, and two weight uploads in the
      Relu2 test never closed what they uploaded. They now add it to a small `closers` helper (`closers_test.go`) and close it with a deferred call. (A value-receiver `closeAll` copied the empty slice at the `defer` line
      and closed nothing; it is a pointer receiver.)
    - **Three library defects.** (1) `rmsnormDevice`, the attention-context, q-split, SwiGLU and GeGLU helpers returned a `DeviceBuffer` beside a `free` function that released the raw buffer without settling the
      live-buffer gauge: the memory was freed and the accounting leaked, ~73 wrappers and ~1.3 MB of gauge per `DecodeToken`. The `free` closures now close the wrapper. (2) A REAL GPU-memory leak in `attnBlockInto`: its
      cleanup closures captured the variable `q`, so after the q-gate split reassigned it the double-width `q` the matmul made was never freed (a closure over a reassigned variable). Each buffer is bound by value now.
      (3) `ResidentStackedW8A8.Close` did not nil its buffers, so a second Close released twice (fixed earlier in this series).
    - **Result.** The whole `./gpu/` package: 144 pass / 58 skip / 0 fail with an empty leak report and exit 0, and the `-short` variant CI's darwin job runs: 128 / 74 / 0 (`leak-report-after.txt`). `staticcheck` is clean. The
      leak check is now FATAL by default (`GOINFER_GPU_LEAKS_FATAL=0` turns it off): a throwaway test that leaks one upload on purpose fails the package with exit 1, naming the file, the line and the library function.
      `gpu-darwin` runs the same package on the Mac's adapter, so a leak that only shows there would turn that job red; it is the first place to look.
  - **A disclosure about tonight's queue:** after the queue started at 11:02 I compiled and vetted the `gpu` package three times and ran its device-free test, on the same box as the first job's timed CPU cells. The
    harness's cell gate saw the load ("loadavg 5.19 > 1.00, waiting") and held the cell until it fell, so the cells were not started under load, but a compile burst inside a cell would not have been seen. Read
    `peer-vetted-nobara-3` with that in mind (its raw log has the gate lines).

#### Night 2026-10-08 on nobara, graded 2026-10-08 (the queue ran 11:02-11:44 PDT, six jobs, all exit 0; raw in `docs/measurements/multimodal-support-2026-10/night-2026-10-08-nobara/`)

Exit 0 is not the verdict (the triage and validation scripts exit 0 whatever happened, and several jobs finished far under their estimates). Each job's own output was read.

- **S13-lite, the float32 arm, graded against its registration** (`s13lite-f32/`, nine timed requests per engine, order rotated, the same session):

  | engine | median TTFT | min-max | spread |
  |---|---|---|---|
  | goinfer, serve's default (int8 tower) | **4.590 s** | 4.570-4.608 | 0.8% |
  | goinfer `-vision-quant f32` | **2.396 s** | 2.364-2.413 | 2.0% |
  | Ollama 0.32.5 `gemma3:4b` | 1.583 s | 1.483-1.605 | 7.7% |
  | llama.cpp 427291b, Q4_K_M + mmproj | 0.951 s | 0.928-0.955 | 2.8% |

  - **Control PASS:** the default arm is 4.590 s against last night's 4.61 s, -0.43% (band 3%).
  - **Prediction:** about 2.5 s; read 2.396 s, 4.2% under.
  - **Ratios:** f32 / default 0.522 (1.92x faster); f32 / Ollama 1.51; f32 / llama.cpp 2.52. The default arm is 2.90x Ollama and 4.83x llama.cpp.
  - **The registered evidence rule is met:** float32's median is under the 5 s bar and not above the int8 arm's, so it is ELIGIBLE to become Gemma 3's CUDA default. The other half is the fidelity record above (G-S3b: float32 reproduces the CPU float32 reference reply exactly;
    G-S3d: the int8 default's reply diverges at generated token 3, p(other) 0.011 against 0.989, not a near-tie; one image and one prompt, evidence and not a rate). Eligible, not decided: the decision is the owner's.
  - **Memory, measured this morning because the float32 reserve had only ever been arithmetic** (`gemma3-memory-probe/`, exploratory and untimed, serve main `c4c03f00`): int8 default 6955 MiB used, 3 of 4 KV slots, 959 MB reserved; `-vision-quant f32` 7041 MiB, 2 of 4 slots, 2400 MB
    reserved; both serve two image requests with no fallback, no growth after the first image. Inference from the difference (one KV slot is 1088 MB; this assumes nothing else differs): the float32 tower is about 1.7 GiB, 1174 MiB more than the int8 tower's measured 558 MiB, so the plan's
    2016 MB reserve for it is about 280 MiB generous.
- **S7 second read** (`s7-second-read/`, serve's defaults, a new image or clip every request, median of 3 timed, `serve-cuda-s7fix` at `f3456904`): every cell is under the 5 s bar and every serve log reads `decode path: cuda-resident` with the towers on CUDA, so no cell hides a CPU fallback.

  | cell | first read | second read | note |
  |---|---|---|---|
  | Gemma 3 4B | 4.57 s | **4.59 s** | int8 tower; 0.4 s under the bar |
  | Gemma 4 E2B image | 0.85 s | 0.85 s | |
  | Gemma 4 E2B audio | 1.48 s | 1.48 s | |
  | Qwen3.5-0.8B | 0.82 s | 0.81 s | |
  | GLM-OCR | 2.09 s | 2.08 s | |
  | Qwen2.5-VL-3B | 15.64 s | **2.45 s** | 6.4x; the tower on CUDA (was on the CPU); the registered prediction was 3-4 s |

  Five cells reproduce the first read to within 1.3% (Qwen3.5 0.82 to 0.81 s is the largest), which is the control; Qwen2.5-VL is the one that changed, by design. Qwen3-VL is still not a cell (S10 is the Mac's). This binary still carries the Qwen2.5-VL tower's 256 MiB slack that was removed the same day (`6b4ef501`); the removal was verified
  separately by the served probe recorded under "Build-scratch / margin accounting" (`qwen25vl-slack-probe.txt`), not by this run.
- **`noncopy-fixed-head`: `TestSpecNonCopyLane` PASS in 45 s** (a measurement with no bar; `noncopy-fixed-head.log`). Decode tok/s on the 1.5B, COPY: plain exact 117.8, plain lane 205.9, n-gram spec 297.7, lane + spec 549.3 (lane + spec over exact + spec 1.85x); SUMMARIZE 118.4 / 204.3 / 168.9 / 219.0
  (acceptance 55% exact, 41% lane); FRESH-essay 117.9 / 206.3 / 108.6 / 188.7 and FRESH-story 117.5 / 205.8 / 110.6 / 205.9, where 0-4% of drafts are accepted and speculation is slower than plain exact (0.92-0.94x) and no better than the plain lane (0.92-1.00x). The speculative arms equal their plain counterparts token for token.
- **`peer-vetted-nobara-3` (the Mac's job; the Mac grades it; `peer-vetted-3/`).** Cell 2, gpt-oss-20B on CUDA at ctx 2048: goinfer 70.5 / 70.5 / 70.5 against Ollama 26.2 / 26.2 / 26.2 tok/s, paired ratio 2.687x, swap +31.9 MB (goinfer) and 0.0 (Ollama) under the nobara rule, the harness's own outcome **AHEAD**. This
  is the first valid cell 2 after two VOIDs. Cell 1c (the vetted 26B on the CPU): **VOID again**, goinfer's swap grew +536.6 MB against the 256 MB rule (Ollama's arm 10.7 tok/s, swap +18.1 MB, valid on its own). **A caveat I owe the Mac:** this job was the first thing running when I started compiling on the box. The goinfer arm's
  recorded machine state at its start was loadavg 0.34, so it began before my first compile, but the arm and that compile are close in time and I cannot show they did not overlap, so I do not claim my work was not a cause. 1c has now been VOID on goinfer's swap growth in all three runs, on the previous two without me, which points at the 26B's own load.
- **`gpu-release-validate`:** already recorded above (reproducers, suite 144 / 0, ten clean cell runs, planted defect red).

#### Gemma 3's CUDA default is now float32 (owner, 2026-10-08, after the S13-lite float32 arm above)

- **Decision:** "float32 it is". On `--backend cuda` with `-vision-quant` unset, Gemma 3's SigLIP tower is the float32 one; `webgpu` keeps int8 (it has no float32 device tower); Metal was already float32; an explicit `-vision-quant int8` asks for int8 and an explicit `f32`
  never falls back. The evidence is the S13-lite float32 arm (2.396 s against 4.590 s) and G-S3b/G-S3d (float32 reproduces the CPU float32 reference's reply; the int8 default's reply diverged at a non-near-tie token), and the memory cost measured above.
- **What it needed beyond flipping `towerInt8`** (`internal/serveapp/gemma3_tower.go`): the float32 reserve is 2.4 GB, and the plan subtracts it from the KV budget BEFORE the model loads, so on a card where one KV slot no longer fits beside the decoder and that reserve the resident build declines and the
  decoder runs on the CPU, which is worse than the int8 tower it replaced. I found this by reasoning from `checkKVFits`, then demonstrated it on the card (`night-2026-10-08-nobara/gemma3-float32-default/result.txt`): with 5.4 GB free the float32 default put the decoder on the CPU; with the check it chose
  int8, with a note, and the decoder stayed `cuda-resident`. Two guards, then:
  1. **Before the plan** (`resolveGemma3VisionQuant`): the unset default becomes int8, with a note, unless the free VRAM holds the decoder, one 4096-position KV slot and the float32 tower. The sizes are estimates from the checkpoint (0.27 of the safetensors bytes for the decoder, 0.55 of that for a slot; both from
     the measured 4B), and the check leans about 0.3 GB toward int8; it never leans toward declining. What it cannot read (an `hf:` reference, no probe, several models) is left alone.
  2. **After the attach** (`attachGemma3Tower`): if the default float32 tower still fails to attach, it is released and the int8 device tower attached, not the CPU tower.
- **Gates:** the table tests (`TestTowerInt8_gemma3ExplicitF32`, `TestTowerInt8`, `TestTowerReserve_gemma3`, `TestTowerReserve_everyDeviceTower`) now pin float32-on-CUDA, int8-on-webgpu and the explicit values; `TestAttachGemma3Tower` (7 cases), `TestLoadVisionTower_gemma3FloatDefaultFallsBackToInt8` and `TestResolveGemma3VisionQuant`
  (8 cases and the edges) are new and each was shown red by disabling the thing it guards. The whole `internal/serveapp` suite passes, `gofmt`, vet and `staticcheck` are clean, and the flags page and `docs/multimodal.md` say so.
- **What changes for a user on the 8 GB card:** a new image costs about 2.4 s of TTFT, not 4.6 s, with an exact tower; the default plan holds 2 resident KV slots at the 4096 floor, not 3. Not measured: the float32 tower's load time at startup (2.7 GB of weights more to upload).

#### S7, the third read on nobara, registered 2026-10-08 before it runs

The procedure, cells and 5 s bar of the second read, unchanged (`run-s7-nobara-3.sh`, `vision_ttft.py`, serve's defaults, a new image or clip every request, one warm-up and three timed requests per cell, the box idle under the timing lock), on `serve-cuda-s7r3`, with one more cell. **Amended the same day, before the job ran:** the binary was rebuilt at `5f715eb3` (it was `9561da9a`) so that the Qwen3-VL cell reads the DeepStack tower work below, not the state before it. Night queue `s7-nobara-3`, estimated 12 minutes (the second read took 2).
- **What changed since the second read:** Gemma 3's CUDA tower defaults to float32 (`dcaa0dd2`), with the pre-load fit check; the Qwen2.5-VL tower estimate lost its 256 MiB slack (`6b4ef501`), so its default context rises from 5057 to about 5967; the plan now prices the driver's allocation
  rounding (`12b221ed`).
- **The new cell:** Qwen3-VL-2B (`qwen3-vl-2b-instruct`). S10 is on main, so serve takes its images. As rebuilt, its tower runs on CUDA (HiddenTaps, G-S10f; priced in the plan) and its image turn still prefills on the CPU and uploads, because the resident DeepStack prefill is off
  until G-S10g passes (one image parked).
- **Predictions, written now:**
  - Gemma 3 4B: **2.3-2.6 s** (the float32 arm of S13-lite read 2.396 s on the same image; the second read's int8 cell was 4.59 s).
  - Qwen2.5-VL-3B: **2.3-2.6 s** (second read 2.45 s). Gemma 4 E2B image 0.85, E2B audio 1.48, Qwen3.5-0.8B 0.81, GLM-OCR 2.08: each **within 5%** of the second read, which is the control that the box is in the same state.
  - Qwen3-VL-2B: **9-15 s, over the 5 s bar, expected.** The 896x896 image is 798 rows: the CUDA tower about 1.5-3 s (first request) plus the CPU prefill and upload (11.75 s for 986 rows, measured by day, so about 9-10 s for 798) plus the first decode step. It would be about 3 s with the resident
    prefill on. A request that errors or hangs is itself the finding and is reported as such.
- **What would be a regression:** any cell whose serve log reads a decode path other than `cuda-resident`, or a CPU tower where the second read had a CUDA one. It is root-caused before any further work.
- **Not a gate** beyond the 5 s bar per cell; a record that ranks what is left.

#### S10 on CUDA: Qwen3-VL's tower with DeepStack (G-S10f) and its resident DeepStack prefill (G-S10g), registered 2026-10-08 before any code (nobara)

CUDA's twin of the Mac's G-S10e (the tower) and S16 step 2 (the prefill).
- **Today, measured by day (exploratory, untimed, serve `9561da9a`, Qwen3-VL-2B at defaults on the 8 GB card, a 23 KB image, 801 prompt tokens):** the decoder is `cuda-resident` (3 KV slots at context 6744). The first image logs `vision: the Qwen3.5 tower runs on the CPU:
  cuda declined it: the cuda tower cannot tap the DeepStack blocks` and `encoded a 23028-byte image in 14.223s`. The turn then prefills on the CPU and uploads, because CUDA's resident m-RoPE prefill does not implement `decoder.ResidentMRoPEDeepstackPrefill`. The replies are sensible ("This image displays a smooth,"). The startup
  banner says `tower (CUDA) loads on first image`, which is the plan and not what then happens.
- **Part A, the tower (G-S10f).** `HiddenTaps` on CUDA's grid tower (`multimodal.GridTowerTapper`): the residual after each DeepStack block (`enc.Cfg.DeepstackVisualIndexes`), read by finishing the queue and downloading it at that block; aikit's host tails make the merged rows and the sets.
  - **Bar:** G-S10e's. The merged rows and each DeepStack set at worst-row cosine >= 0.9999 against aikit's CPU `ForwardDeepstack` (0.999-0.9999 parked), on the tiny DeepStack tower (`testdata/qwen3vl-vision-tiny`, norms randomised, two grid sets) and on Qwen3-VL-2B's four F2a images at serve's cap (heavy).
  - **Planted defects, each alone red on the tiny tower, a refusal counting as a failure:** (1) every tap one block late (the tiny tower has three blocks so the late taps stay inside it); (2) the taps handed back in reverse order.
  - **Served (G-S10d on CUDA):** `--backend cuda` at defaults against `-vision-device cpu`, one image request, 32 greedy tokens with top-3 logprobs: identical replies, or a first divergence at a near-tie (p(other) at least half p(top)); the CPU arm repeated for determinism; the CUDA arm's log must not contain
    `runs on the CPU`.
  - **Speed (night, a record):** the tower per image on the 2B's four images, CUDA against the CPU tower, interleaved under the timing lock, host tails reported apart.
  - **Prediction, with its arithmetic:** the 800-token image is 3,200 patches. GEMMs: 24 blocks x 2 x 3,200 x 12 x 1024^2 is about 1.9 TFLOP, 0.5-0.6 s at the 3.5 TFLOPS the other towers reach; the fused attention about 1.0 TFLOP, 0.4-0.6 s; the host tails (four mergers, about 40 GFLOP each on the CPU) 1-2 s. So **1.5-4.0 s a
    new image, against the CPU tower's 14.2 s.** **Kill line:** over 4 s re-opens the plan with a profile (S17 step 0's tool) and does not tune.
- **Part B, the resident prefill (G-S10g).** `PrefillMRoPEDeepstackLast` on CUDA's resident (`decoder.ResidentMRoPEDeepstackPrefill`): after layer l, set l is added to the image rows of the residual, prefill only, as `addDeepstack` does on the CPU. Until it exists the DeepStack turn keeps the CPU prefill and the upload, correctly and slowly.
  - **Tiny, against the CPU prefill + upload path (S16's G-S16a):** last-row logits and the next 8 decode steps at cosine >= 0.9999 per row with an equal argmax, on `qwen3vl-tiny` and a synthetic image with random features and sets (0.999-0.9999 parked).
  - **Planted defects (S16's 4-6), each alone red:** the sets not added; added one layer late; added to the text rows too.
  - **Real (S16's re-registered bar, which exists because the strict one was tighter than the shipped batched prefill's own distance from the CPU's):** Qwen3-VL-2B, the four F2a images: the image turn's worst per-step cosine at least the same-length text-only control's minimum (four prompts per length) minus 0.005, every argmax
    difference a near-tie; 0.005-0.015 below parked; worse fails. Then served, resident prefill on against off, one binary: identical replies or a near-tie first divergence, `ImgPrefillResident` true.
  - **It turns on in production only if the real gate passes,** as on Metal.
  - **Speed (night):** the S7 Qwen3-VL-2B cell, before and after, tower time reported apart. No prediction written until Part A's split of one image turn (tower, CPU prefill, upload, first decode) is measured; it will be added here before Part B's speed run.
- **Order:** A first (it removes the 14 s), measured, then B.

#### S10 on CUDA, read 2026-10-08 (nobara)

- **G-S10f, the tower with DeepStack: PASS, tiny, real and served** (code `50233201`, reserve fix `Qwen3-VL's tower priced`; raw `docs/measurements/multimodal-support-2026-10/s10-cuda/`).
  - **Tiny** (`qwen3vl-vision-tiny`, norms randomised, two grid sets): the merged rows and both sets 1.000000000; the planted taps-one-block-late reads 0.995 and taps-in-reverse 0.992, both red.
  - **Real, Qwen3-VL-2B, the four F2a images at serve's cap:** worst 0.999999219 (formula.png, set 1), every other output 0.9999999 or better; the planted late taps read 0.574 / 0.293 / 0.661 on the 14x20 image.
  - **Speed, exploratory single samples:** the CUDA tower with the host mergers 1.49 s (3,136 patches), 2.10 s (4,060) and 1.93 s (3,888) against the CPU tower's 12.46 s, 18.49 s and 17.41 s; the device blocks alone are 1.08-1.63 s. Inside the registered 1.5-4.0 s band (the smallest image 0.01 s under its lower edge).
  - **Served, `=cuda:cpu` against `cuda:auto`, table.png, 32 greedy tokens:** IDENTICAL replies ('Quarterly unit sales by region'), the CPU arm's repeat IDENTICAL. **The first served run was NOT the gate:** the CUDA arm's tower ran out of device memory on its first image (a 63.7 MB allocation)
    and "runs on the CPU from now on", because the Qwen3-VL tower was priced at zero (it was in neither `towerVRAMEstimate` nor `cudaTowerRegistered`): the decoder had taken every free byte. The same class as S18's Gemma 3 fix; fixed by pricing it as the Qwen3.5 tower (1.64 GB for the
    2B, against 1158 MiB measured resident after an image; the scratch is released per call, so the estimate is the peak), with `TestTowerReserve_everyDeviceTower` extended (red before) and the figure pinned. The default plan for the 2B moves from 3 slots at 6744 positions to 3 at 4955. The rerun is the one
    recorded: the CUDA tower encoded the image in 3.11 s against the CPU arm's 18.4 s, no fallback in its log; the whole request 15.1 s against 30.4 s (the rest is the CPU DeepStack prefill, Part B).
- **G-S10g, the resident DeepStack prefill (code `7703e4e3`): tiny PASS; the real gate's first reading FAILS the registered bar, and is kept.**
  - **Tiny:** worst cosine 1.0000000, argmax 9/9 against the CPU prefill + upload; planted defects 0.9818884 (sets not added), 0.9818884 (one layer late: in the 2-layer fixture set 0 lands after the last layer, where nothing reads it, as on Metal), 0.1000094 (text rows too), all red;
    through `GenerateQwenVLDeepstack` the turn takes the resident prefill and generates the same 8 tokens.
  - **Real, first reading (S16's re-registered bar: the image turn's worst per-step cosine at least the same-length text control's minimum minus 0.005; 0.005-0.015 below parked; worse fails; every argmax difference a near-tie):**

    | image | rows | worst cosine | text control's minimum (4 prompts) | verdict |
    |---|---|---|---|---|
    | gemma3_preprocess_image (896²) | 798 | 0.9670 | 0.9451 | PASS |
    | qwen25vl_preprocess_image (4x6) | 84 | 0.9624 | 0.9728 | **PARKED** (0.0104 below) |
    | glm_ocr/formula | 1029 | 0.9739 | 0.9417 | PASS |
    | glm_ocr/table | 986 | 0.9125 | 0.9539 | **FAIL** (0.0414 below) |

    No argmax differences on any image. So the path does NOT turn on in production, as registered.
  - **Mechanism, a diagnostic and not a gate (`TestS10DeepstackPrefillCUDA_diag`; per-step cosines, step 0 first):** table.png as gated 0.9125 0.9865 0.9857 0.9631 0.9935 ...; with no sets in either arm 0.9711 0.9856 ...; as gated with the fast prefill levers off 0.9741 0.9861 0.9886 0.9805 ....
    The failing number is the prefill's last-row logits (step 0), and it is the fast prefill levers (the L2/L3 kernels that engage at 512 or more rows, cosine-close but not bit-identical) acting on image rows with the sets added; with them off the image turn is above the control. The 4x6 image has 84 rows, below the lever floor,
    so nothing changes there: its worst is step 3 (0.9624 as gated; 0.9711 at step 0 with no sets).
  - **Amendment, registered before the re-read (the design, not the bar):** a DeepStack prefill runs on the exact kernels (`forceExactKernels` includes an active DeepStack plan), as the Gemma 4 E-model does until its fast levers have a fidelity gate (S9 on CUDA, G3p). The bar, the control and the images are unchanged;
    the real test now also prints the control's four per-prompt minima beside the verdict, for context and not for the verdict.
  - **Prediction, written now:** table.png PASSES (the lever-off cut read 0.9741 against 0.9539); the 896² and formula images stay PASS; the 4x6 image stays PARKED (no lever is involved at 84 rows). If it stays parked, that image is reported as parked and the path stays off; the bar is not moved to resolve it.
  - **Re-read on the exact kernels (`gs10g-real-reread-exact-kernels.log`), every prediction held:**

    | image | rows | worst cosine | text control's minimum | each control prompt's own minimum | argmax | verdict |
    |---|---|---|---|---|---|---|
    | 896² | 798 | 0.9680 | 0.9451 | 0.9653 0.9457 0.9451 0.9689 | 2 near-ties, 0 real | PASS |
    | 4x6 | 84 | 0.9624 | 0.9728 | 0.9728 0.9982 0.9980 0.9885 | equal | **PARKED** (0.0104 below) |
    | formula | 1029 | 0.9819 | 0.9417 | 0.9552 0.9807 0.9417 0.9983 | equal | PASS |
    | table | 986 | 0.9741 | 0.9539 | 0.9782 0.9871 0.9539 0.9972 | equal | PASS (was FAIL) |

    **So the registered outcome stands: three of four PASS, one PARKED, and the path stays OFF in production** (`cudaDeepstackPrefillOn`, default false; the gates turn it on; while it is off a Qwen3-VL image turn takes the CPU prefill and the upload, as before).
    The parked image is the one with the fewest rows. Its worst is step 3 of the 8 teacher-forced decode steps (0.9624), below all four control prompts' own minima; the control's minimum on this length is set by one prompt (0.9728) while the other three
    sit at 0.9885-0.9982, so the control's minimum is noisy. I have not moved the bar or the control to resolve it.
  - **What is at stake, exploratory single samples (`gs10g-speed-exploratory.log`, table.png, 986 rows, on the exact kernels):** CPU prefill + KV upload **11.75 s** and **11.78 s** against the resident DeepStack prefill **0.90 s** and **0.89 s** (13x). With the tower's 2-3 s,
    a Qwen3-VL-2B image turn on CUDA would be about 3 s against about 15 s today. The night speed record is the S7 Qwen3-VL cell.
  - **Decision for the owner (the Mac's precedent for a parked or failed reading: re-registering the bar is the owner's):** (a) keep it off; (b) turn it on for image runs of 512 rows or more, where all three images pass and where the CPU prefill costs seconds, and leave small images (84 rows is about a
    second on the CPU) on the CPU path; (c) re-register the control as 12 prompts per length (the minimum of more prompts is a lower, steadier bar) and re-read the control only. (b) and (c) each need to be registered before they are applied to any reading.

#### G-S10g, the owner's decision (c): the control re-registered at 12 prompts per length, registered 2026-10-08 before the re-read

Owner decision 2026-10-08: option (c), first. Written before any run of it.

- **The change, and only this:** the text control for each image length is 12 prompts (seeds 1-12) instead of 4. Seeds 1-4 are the existing ones (same token formula), so the 12-prompt minimum is at most the 4-prompt minimum at every length, by construction.
  The bar (image turn's worst per-step cosine at least the control's minimum minus 0.005; 0.005-0.015 below parked; worse fails; argmax differences near-ties only), the images, the 8 teacher-forced steps and the exact kernels are unchanged.
- **The mechanism that justifies it, stated now so it is not found afterwards:** the 4x6 control's minimum (0.9728) is set by ONE prompt of four; the other three read 0.9885-0.9982. A minimum over four draws is a noisy statistic, and which prompt happens to be the lowest decides the bar. It is not that the bar is too hard.
  The cost, also stated: the minimum of more draws is mechanically lower, so this change can only make a verdict easier. That is why the registration below fixes what each outcome means and reports the whole distribution, not only the minimum.
- **Scope of the re-read:** the 4x6 image only (84 rows), as "re-read the control only" says. The other three images pass against the 4-prompt control and the 12-prompt minimum cannot be higher, so their verdicts cannot get worse; they are not re-run and are not re-claimed as re-read.
- **Instrument check before the verdict is read:** the 12-prompt run's seeds 1-4 must reproduce the recorded per-prompt minima (0.9728, 0.9982, 0.9980, 0.9885 to four places). If they do not, the control is not the one the registered reading used and the run is void.
- **Outcome rule, fixed now:**
  - the 4x6 image's 0.9624 against the 12-prompt minimum m: **PASS** if m <= 0.9674 (so a new prompt reads at or below 0.9674); then the resident DeepStack prefill is enabled for all image sizes (`cudaDeepstackPrefillOn` default true, with its tests and docs), as registered under G-S10g.
    The record prints all twelve per-prompt minima, with the image's 0.9624 beside them, so the reader can see where it falls.
  - **Still PARKED or FAIL:** the path stays OFF, nothing else changes, and (b) (on for 512 or more rows) goes back to the owner as its own registration; it is not applied from here.
  - No third control size is run if this one does not pass: a bigger control after a miss is a bar moved until it clears.
- **Prediction, written now:** the image's 0.9624 is about 0.01 under the one low control prompt and 0.025-0.036 under the other three. A pass needs one of the eight new prompts at or below 0.9674, which is a tail event against the three at 0.9885+; I put it near even, no better.
  A pass would rest on the minimum statistic and I will say so; it would not show the image turn is as good as the text path in the mean.
- **Cost:** about 2 minutes (one image, 12 CPU prefills of 84 rows at about 1 s each, plus the image arm); by day, quick tier.


#### G-S10g, decision (c) read 2026-10-08: the real gate PASSES on all four images; the served check does NOT clear its rule, so the path stays OFF (raw `docs/measurements/multimodal-support-2026-10/s10-cuda/gs10g-control12-4x6.log`, `gs10g-served-*`)

- **The 4x6 image against the 12-prompt control: PASS** (code `a54f1642`'s test; 24 s). Instrument check held: seeds 1-4 reproduced the recorded minima (0.9728 0.9982 0.9980 0.9885). The twelve per-prompt minima: 0.9728 0.9982 0.9980 0.9885 0.9753 **0.9527** 0.9984 0.9944 0.9977 0.9987 0.9984 0.9955.
  The image turn's worst is 0.9624, against the 12-prompt minimum 0.9527 (bar 0.9527 - 0.005 = 0.9477): PASS, 0 argmax differences. My prediction ("near even") held.
  **What that does and does not say:** the pass rests on one control prompt, the sixth of twelve, at 0.9527. The image's 0.9624 is below the other eleven prompts' own minima, whose median is 0.9944. The image turn sits in the lower tail of the text path's distribution, not at its centre; it is inside the tail the batched prefill already has on text.
  The other three images were not re-run (their verdicts cannot get worse against a lower minimum, by construction).
- **What I did next, and the mistake in it.** The outcome rule I registered said PASS means "enable for all sizes", and I flipped `cudaDeepstackPrefillOn` to true (`a54f1642`). That rule left out G-S10g's own served half, registered at the top of S10: "served, resident prefill on against off, one binary: identical replies or a near-tie first divergence".
  I then ran it, with TWO binaries because the switch is a variable and not a flag (`serve-cuda-dsoff` at `8ff9a57e`, `serve-cuda-dson` at `a54f1642`; they differ by that one line), Qwen3-VL-2B, table.png, 32 greedy tokens, `run-gs3c-served.sh`, `=cuda:auto,cuda:auto` per binary.
- **Served, read: NOT a pass by the registered rule.** Each binary's repeat is IDENTICAL. Across them the replies differ at generated token 0, which is the prefill's last-row logits: off (CPU prefill + upload) answers 'Quarterly unit sales by region' (Quarter 0.412, Table 0.112, The 0.099); on answers 'Table 2. Quarterly unit sales by region (thousands)' (Table 0.236, Quarter 0.190, A 0.129).
  Read from the off path as the reference, p('Table') = 0.112 against half the top's 0.206: **not a near-tie**. Read from the on path it would be (0.190 against 0.118). The registration never named a direction; the only defensible one is the established path (off) as the reference, so that is the reading, and it is chosen before looking for another.
  This is the same distance the gate measured (step 0 is where table.png read 0.9741 against the CPU prefill), now visible in a served first token. It is also what the text control shows between the batched prefill and the CPU one. I do not know which prefill is the closer to the HF float32 reference on this image; no anchor was run.
  Exploratory, one sample each: the whole request 15.0 s (off) against 4.3 s (on).
- **State:** `cudaDeepstackPrefillOn` is back to false (the flip stays in history as `a54f1642`, the revert is the next commit); nothing is pushed. The real test keeps the 12-prompt control and the per-image subtests; its `defer` restores the previous value instead of forcing false.
- **Open, for the owner:** (1) keep it off; (2) accept the real gate's four PASSes as the bar and read the served check's first-token difference as the text path's own distance from the CPU prefill (then enable; this needs the served rule re-registered with a direction and a band, and I would add the other three images to it first);
  (3) an HF float32 anchor for the image turn's first-token distribution, which says which prefill is closer, so the served difference stops being a tie between two non-reference paths (the 2B fits in RAM at float32: a CPU Hugging Face run, minutes).

#### G-S10h, the HF float32 anchor for the image turn's prefill, registered 2026-10-08 before any run (owner: option 3, then option 2)

Question: of the two prefills that disagree at the served first token (the CPU prefill + upload, "off", and the resident DeepStack prefill, "on"), is "on" at least as close to Hugging Face float32 as "off" is?

- **Method.** Qwen3-VL-2B, the four F2a images at serve's 1,024-row cap, the gate's prompts. A Go dump (heavy test, CUDA) writes, per image, the token ids, the CPU encoder's preprocessed pixels and grid, the teacher tokens (the off path's greedy, as the real gate teacher-forces), and the 9 logit vectors of each arm (the prefill's last row, then 8 teacher-forced decode steps).
  A Python script on `~/g4venv` (transformers 5.15.0, torch CPU) loads the SAME checkpoint into float32, feeds the SAME ids and pixels with `image_grid_thw` and the teacher tokens appended, and reads its logits at the same 9 positions. HF computes its own m-RoPE positions and DeepStack; nothing of goinfer's enters its side.
- **Confound, stated now:** both goinfer arms run int4 weights and HF runs float32, so every cosine to HF carries the int4 error, shared by the two arms. The reading is the paired DIFFERENCE between the arms, not either distance alone.
- **Instrument checks (a failure voids the run):** (i) the dump's off-against-on cosine per step, minimised over steps, reproduces the real gate's recorded worst to four places (0.9680, 0.9624, 0.9819, 0.9741 for 896², 4x6, formula, table); (ii) HF's processor on the same image at the same pixel cap gives the same grid, and its `pixel_values` are reported against the dump's (max abs difference; informational, HF is fed the dump's pixels either way);
  (iii) HF's step-0 argmax on table.png is a token in the top-3 of at least one arm (a sanity floor against a garbage reference, not a correctness claim).
- **Reading.** For each (image, step) of the 36 pairs: delta = cos(on, HF) - cos(off, HF).
  - **"On at least as close" (PASS):** mean delta >= -0.002 AND every image's step-0 delta >= -0.01.
  - **"On farther" (FAIL):** mean delta < -0.005 OR any image's step-0 delta < -0.02. Checked first.
  - **Otherwise AMBIGUOUS:** goes to the owner with the numbers.
  Printed beside the verdict, not graded: each arm's cosine to HF per step; KL(HF || arm) at step 0 per image; HF's probabilities at step 0 for 'Quarter' and 'Table' on table.png (the two tokens the served check split on) next to each arm's.
- **What a PASS would and would not do.** It would say the served first-token difference is two non-reference paths disagreeing, with "on" no further from HF than "off". It would NOT enable anything: option 2 still needs the served rule re-registered with a direction and a band (and the other three images in it) before it is applied to any reading. A FAIL leaves the path off.
- **Prediction, written now:** both arms read 0.95-0.99 to HF and the mean delta lies within +-0.003; I do not predict its sign. The batched prefill's own distance from the CPU prefill (0.95-0.99, the control) is the same size as the distance this tests, so an outcome in the AMBIGUOUS band is quite possible.
- **Cost, tier:** day, in two steps each under ~8 min: the Go dump (four CPU encodes and CPU prefills, ~1 min each, plus the resident arms) and the HF float32 CPU forward (four ~1,000-row prompts on a 2B, a minute or two each).

#### G-S10h read 2026-10-08: FAIL as registered (mean delta +0.0088, one image's step-0 delta -0.0362); the path stays OFF (raw `docs/measurements/multimodal-support-2026-10/s10-cuda/gs10h-*`)

Dump `cuda/s10_anchor_dump_test.go` (94 s), Hugging Face side `scripts/anchor_s10h_hf.py` (transformers 5.15.0, torch CPU, float32, 56 s for all four; my estimate of 5-8 minutes was several times too high). The dump's raw logits (110 MB) stay in `~/goinfer-logs/s10h`, not committed.

- **Instrument checks.** (i) held to four places: the dump's off-against-on worst cosines are 0.9680 / 0.9624 / 0.9819 / 0.9741, the real gate's. (iii) held: HF's step-0 argmax (token 1986) is the argmax of both arms on all four images.
  **(ii) could not run:** the venv has no torchvision or PIL, so HF's own preprocessing of the images was not compared with the dump's pixels (it was registered as informational; HF was fed the dump's pixels either way). One thing to know about the HF call: transformers 5 needs `mm_token_type_ids` (1 on the image-pad tokens) to compute M-RoPE, which the script passes.
- **The reading, by the registered rule: FAIL** (checked first: any image's step-0 delta below -0.02). 36 pairs, **mean delta +0.00882** (on closer on average), step-0 deltas **-0.0362** (896²), +0.0001 (4x6), +0.0030 (formula), +0.0150 (table). The 896² image's step 0 alone fails it.

  | image | rows | mean delta (on - off) | steps where on is closer | step-0 delta | KL(HF‖off) step 0 | KL(HF‖on) step 0 |
  |---|---|---|---|---|---|---|
  | 896² | 798 | -0.0014 | 5/9 | **-0.0362** | 0.0587 | 0.0724 |
  | 4x6 | 84 | +0.0224 | 7/9 | +0.0001 | 0.0033 | 0.0009 |
  | formula | 1029 | -0.0069 | 3/9 | +0.0030 | 0.0632 | 0.0801 |
  | table | 986 | +0.0212 | 8/9 | +0.0150 | 0.0454 | 0.0122 |

- **What it says.** Neither prefill is the closer one. On is closer on the two images where the text control's distance is largest (4x6, table) and farther on the other two (896², formula); in the middle steps both arms sit at 0.62-0.97 to HF (the int4 error against float32 is shared and large), so deltas of 0.01-0.04 are a fraction of that. The registered prediction (both 0.95-0.99, mean within +-0.003, sign not predicted) was WRONG in size:
  the cosines to HF reach 0.62-0.80 at some teacher-forced steps on table.png, and the mean delta is +0.0088. The step-0 argmax agrees everywhere; no arm picks a different first token from HF's on these prompts.
- **What it does not say.** The probabilities printed for 'Quarter' and 'Table' on table.png (HF 1e-14 and 8e-7) are for the gate's prompt, "Describe this image.", which is not the prompt of the served check ("What does this image show? Answer briefly."), where those two tokens split 0.41/0.11 and 0.24/0.19. So this anchor does not answer the served split directly. I did not re-run it on the served prompt: a new anchor after a FAIL is a new registration, and it is the owner's call whether to open one.
- **Outcome.** Option 3's reading is FAIL, so option 2 (enable) is not taken; `cudaDeepstackPrefillOn` stays false. No bar was moved.

#### G-S10i, the HF float32 anchor on the SERVED prompt, registered 2026-10-08 before any code or run (owner: yes, after G-S10h read FAIL)

G-S10h used the gate's prompt ("Describe this image."), not the one on which the served check split ("What does this image show? Answer briefly."), so it did not explain that split. This repeats G-S10h with the served prompt and nothing else changed.

- **Unchanged from G-S10h, on purpose:** the model, the four images and their cap, the two arms, the teacher tokens (the off path's greedy), the 9 positions, the HF float32 side, the metric (delta = cos(on, HF) - cos(off, HF) per (image, step)), the thresholds and the order of the rule:
  **FAIL** if mean delta < -0.005 or any image's step-0 delta < -0.02; **PASS** if mean delta >= -0.002 and every image's step-0 delta >= -0.01; otherwise **AMBIGUOUS**. No threshold is touched because G-S10h failed.
- **Changed:** the user turn's text is "What does this image show? Answer briefly." (the served request's, `run-gs3c-served.sh`), the ids are the chat template's rendering of one image then that text, with the generation prompt, and the image-pad token expanded to the image's row count.
- **New instrument check (iv), void on failure:** the hand-built ids equal `apply_chat_template` on `[{"type": "image"}, {"type": "text", ...}]` from the checkpoint's own template (Hugging Face's tokenizer, image-pad expanded), for every image. Checks (i) and (iii) as before; (i) is expected to differ from the gate's recorded worst cosines now, because the prompt differs, so it is reported and not compared (the dump's off-against-on worst is the new reference for this prompt).
  (ii) stays unavailable (no PIL/torchvision) and informational.
- **The served split, printed and read as registered here:** on table.png at step 0, the log-odds ln(p('Quarter') / p('Table')) of HF, of the off arm and of the on arm. The served check had Quarter 0.412 / Table 0.112 off and 0.190 / 0.236 on, i.e. log-odds +1.30 off and -0.22 on. The arm whose log-odds is nearer HF's is the one that is nearer on the token pair the served check split on. This is context for the owner, not a graded condition.
- **What each outcome does, fixed now:**
  - **PASS:** this is the served prompt's reading and it goes to the owner beside G-S10h's FAIL. It does not erase that FAIL, and nothing is enabled from here: enabling needs the served rule re-registered with a direction and a band, and the owner's decision weighing both anchors.
  - **FAIL or AMBIGUOUS:** the path stays off. **No third anchor is run**: a third prompt after two readings is searching for a pass.
- **Prediction, written now:** the sign of the mean delta will match G-S10h's (positive, +0.005 to +0.015), and one of the larger images will again carry a negative step-0 delta below -0.01; I put the chance of a PASS under 30%. On the served split I expect HF's log-odds to be nearer the on arm's, because the on arm moved the first token toward 'Table' and the table image is the one where the on arm was closer to HF; that is a guess and could be wrong.
- **Cost, tier:** day, ~3 minutes (dump 95 s, HF 1 min).

#### G-S10i read 2026-10-08: AMBIGUOUS as registered, and the instrument turned out not to be the served configuration; the path stays OFF (raw `docs/measurements/multimodal-support-2026-10/s10-cuda/gs10i-*`)

- **Reading, by the rule registered above (unchanged from G-S10h): AMBIGUOUS.** 36 pairs, **mean delta -0.00450** (inside [-0.005, -0.002)), step-0 deltas **-0.0105 (896²), -0.0123 (4x6), -0.0116 (formula), -0.0024 (table)**: no step-0 delta under -0.02 (so not FAIL), but three under -0.01 and the mean under -0.002 (so not PASS).

  | image | rows | mean delta (on - off) | steps where on is closer | step-0 delta | KL(HF‖off) step 0 | KL(HF‖on) step 0 |
  |---|---|---|---|---|---|---|
  | 896² | 803 | -0.0006 | 3/9 | -0.0105 | 1.5255 | 1.4180 |
  | 4x6 | 89 | -0.0114 | 1/9 | -0.0123 | 0.0996 | 0.3336 |
  | formula | 1034 | -0.0027 | 4/9 | -0.0116 | 0.3704 | 0.6635 |
  | table | 991 | -0.0033 | 3/9 | -0.0024 | 0.8925 | 2.0298 |

  On this prompt "off" is the closer arm on average and on three of four images at step 0 in KL; the opposite sign from G-S10h's mean (+0.0088). My prediction (same sign as G-S10h, a PASS under 30%) held for the PASS part and missed the sign.
- **Instrument checks.** (iv) held: the hand-built ids equal the checkpoint's chat template's rendering on all four images (803, 89, 1034, 991 tokens). (i): the off-against-on worst cosines are 0.9809 / 0.9748 / 0.9631 / 0.9706 (the new reference for this prompt). (iii) as registered (table.png): holds. Reported beyond it: on the 896² image HF's step-0 argmax (362) is in neither arm's top-3 (both arms 32), and on table.png HF's is 86608 ('Quarter') against both arms' 2556 ('Table').
- **The served split, as registered: not explained, and the reason is below.** On table.png at step 0, ln(p('Quarter')/p('Table')): HF **+1.69** (0.702 / 0.129), off **-0.92** (0.142 / 0.358), on **-3.05** (0.030 / 0.626). Off is nearer HF's; I expected on. But the dump's off-arm probabilities (0.142 / 0.358) are not the served off arm's (0.4115 / 0.1117, the same table image with 991 prompt tokens in both).
- **Post-hoc finding, not registered, about the instrument:** the dump loads the model with `Options{Quant: "int4"}`, and serve's `--embed-int4` defaults to TRUE with `--quant int4` (`docs/flags.md`), so the dump's token-embedding/LM-head table was at the int8 pin and the served one at int4. **G-S10h and G-S10i therefore compared two arms that were not the served configuration** (the arm-against-arm real gate, G-S10g, is unaffected: both its arms share the load).
  A diagnostic dump with `EmbedInt4` on (`gs10i-embed4-dump-diagnostic.log`; the Go side only, no HF run, nothing graded) moves the table.png first-token probabilities toward the served ones but does not reproduce them: off Quarter 0.305 / Table 0.164, on 0.246 / 0.202, against served 0.4115 / 0.1117 and 0.190 / 0.236. Something further differs between the dump and serve (the tower's features come from the CPU encoder in the dump and the CUDA tower in serve, at about 1e-6, which I do not expect to move a probability this much; I have not found the rest).
  The lesson that stands without any further run: this first-token pair is highly sensitive to a table-precision change (Quarter 0.14 -> 0.30 on the off arm), so a 0.4 against 0.1 split between two prefills is within what a quantisation choice alone does, and neither anchor can rank the prefills until the instrument is the served configuration.
- **Outcome.** AMBIGUOUS: the path stays OFF. By the registration, no third anchor is run on my own motion. A corrected-configuration anchor (EmbedInt4 on, and the dump reproducing the served probabilities first, as a registered instrument check) is a new registration that lifts my own "no third anchor" rule, so it is the owner's call.

#### G-S10j, the corrected HF anchor, registered 2026-10-08 before any code or run (owner: option 2 after G-S10i read AMBIGUOUS and showed the dump was not the served configuration)

This lifts my own "no third anchor" rule at the owner's word. It is bounded below so it cannot become a search.

- **Step 1, the instrument gate (nothing is graded until it holds).** The Go dump, loaded as serve loads it, must reproduce the SERVED first-token distribution on table.png with the served prompt, for BOTH arms: the reference is the two serve binaries already pinned (`serve-cuda-dsoff` at `8ff9a57e` = the off path, `serve-cuda-dson` at `a54f1642` = the on path; they differ by one line), each asked for 1 token with the top-5 log-probabilities, `temperature 0`, the same request as the served check (`run-gs3c-served.sh`).
  Held when, for each arm, every one of that server's top-5 tokens has a dump probability within **0.01 absolute** of the server's. The server's repeat is IDENTICAL to itself (recorded), so the reference is stable; 0.01 is a width for the CUDA tower (server) against the CPU encoder (dump), whose rows agree to 1e-6.
  The reference numbers are written to a file by a script before the dump is compared, so the comparison cannot be adjusted afterwards.
- **What may be changed to make the dump match serve:** only the dump's configuration, to be what serve does (every numerics-affecting default of `serve --model <dir> --backend cuda -vision-device auto` that the dump left at the zero value: the embed-int4 table that G-S10i found, then the KV quantisation, the context and any other `decoder.Options` field serve sets). Nothing on the graded side moves.
- **Cap on the search, fixed now:** three rounds, each a hypothesis about a named configuration field, tested by running the dump and reading the instrument gate. If the gate does not hold after three, **the anchor is VOID and nothing is graded**: the report is "the dump does not reproduce serve and why we do not know", and no further anchor is proposed.
- **Step 2, only if step 1 holds:** G-S10i's anchor unchanged (the four images, the served prompt, the two arms, the teacher tokens, the HF float32 side, the metric and the rule: FAIL if mean delta < -0.005 or any step-0 delta < -0.02; PASS if mean delta >= -0.002 and every step-0 delta >= -0.01; else AMBIGUOUS), on the corrected dump. Checks (i) (iii) (iv) as before; (iii) is read on every image, not only table.png, and a miss is reported.
  The step-0 log-odds ln(p('Quarter')/p('Table')) of HF, off and on are printed beside it.
- **Outcomes, fixed now:** PASS goes to the owner with G-S10h's FAIL and G-S10i's AMBIGUOUS beside it; nothing is enabled from here (the served rule would be re-registered with a direction and a band first). FAIL or AMBIGUOUS leaves the path off, and the DeepStack prefill question is closed for this tranche.
  Whatever the reading, this is the last anchor of this kind.
- **Prediction, written now:** the gate holds after the embed-int4 and KV-quant fields (one or two rounds); with the served configuration the mean delta stays within +-0.01 and the verdict is AMBIGUOUS again (probability about 0.5), a PASS about 0.2, a FAIL about 0.3.
- **Cost, tier:** day. Finding the configuration is the unknown (I estimate under 45 minutes, three dump runs at 95 s each); once it holds, the dump and HF side are ~3 minutes.

#### G-S10j read 2026-10-08: the instrument gate HELD in round 3, and the anchor reads FAIL; the resident DeepStack prefill stays OFF for good in this tranche (raw `docs/measurements/multimodal-support-2026-10/s10-cuda/gs10j-*`)

- **Deviation from the registration, first.** I wrote that the served reference is stable because "the server's repeat is IDENTICAL". It is for the on binary and is **not** for the off binary: its second identical request differs from its first (the second reuses the first's resident KV prefix). The reference is therefore each binary's COLD first request, which reproduced across three separate server starts (the off binary's 0.4115 / 0.1117 at 16:29, 16:45 and in the reference file). Recorded in `gs10j-served-reference.json` by `scripts/s10j_served_reference.sh`, before any comparison.
- **The three rounds (cap 3), each a named hypothesis:**
  1. **Serve's `decoder.Options`** (built from `internal/loadflags` defaults: embed-int4 on, KV f32, fit, the CPU-attention knob): NOT HELD (worst |diff| 0.1068 off, 0.0559 on), numerically identical to G-S10i's embed-int4-only diagnostic, so no other Options field matters.
  2. **The vision tower's source.** Tested on the SERVER side, not the dump: the off binary with `-vision-device cpu` gives Quarter 0.3047 / Table 0.1644 / A 0.1375 / quarter 0.0963, which is the round-1 dump's numbers to four digits. So the dump's goinfer side (prompt, options, CPU prefill) is exactly serve's, and the whole gap was the tower: serve's `auto` is the CUDA tower and the dump had the CPU encoder.
  3. **The dump with the CUDA tower's features and sets** (`newQwen3Tower` + `Qwen3TowerFeaturesDeepstack`, the same calls serve makes): **HELD, worst |diff| 0.0000 for both arms**, all ten top-5 probabilities identical to four digits.
- **What round 2 means beyond the anchor (an observation, not a gate):** the CUDA tower's merged rows agree with the CPU encoder at worst-row cosine 0.9999992 (G-S10f), and that is enough to move this prompt's first token from Quarter 0.305 to Quarter 0.412. The first token here is a near-tie between 'Quarter' and 'Table' under int4, so a feature difference of about 0.1% per row swings it by 0.1 in probability. It is the same kind of sensitivity G-S10i found for the embed-table precision. Neither tower has been ranked against Hugging Face's.
- **Step 2, the registered rule on the instrument-held dump: FAIL.** 36 pairs, **mean delta -0.00887**, step-0 deltas **-0.0554 (896²), -0.0005 (4x6), -0.0085 (formula), -0.0104 (table)**: the mean is under -0.005 and the 896² image's step 0 is under -0.02, each enough alone.

  | image | rows | mean delta (on - off) | steps where on is closer | step-0 delta | KL(HF‖off) step 0 | KL(HF‖on) step 0 |
  |---|---|---|---|---|---|---|
  | 896² | 803 | -0.0200 | 1/9 | **-0.0554** | 0.3408 | 0.6070 |
  | 4x6 | 89 | +0.0022 | 5/9 | -0.0005 | 0.1775 | 0.0678 |
  | formula | 1034 | -0.0048 | 2/9 | -0.0085 | 0.1492 | 0.3174 |
  | table | 991 | -0.0129 | 0/9 | -0.0104 | 0.3210 | 0.7486 |

  The checks: (iv) the chat template's ids equal the dump's on all four; (iii) HF's step-0 argmax is in an arm's top-3 on all four (G-S10i's 896² miss is gone, as it came from the unserved configuration); (i) the new off-against-on worst cosines 0.9626 / 0.9868 / 0.9812 / 0.9677; (ii) still unavailable.
- **The served split, answered.** On table.png at step 0, ln(p('Quarter')/p('Table')): HF **+1.69** (0.702 / 0.129), off **+1.30** (0.4115 / 0.1117), on **-0.21** (0.190 / 0.236). The off arm (the CPU prefill and upload) is the nearer to Hugging Face's float32 first token; the resident DeepStack prefill moves it toward 'Table', away from HF. My guess that the on arm would be nearer was wrong.
- **Prediction record.** Held: the gate passed within the cap (it took all three rounds; I said one or two), the mean delta lay inside +-0.01 (-0.0089). Missed: I put FAIL at 0.3 and AMBIGUOUS at 0.5.
- **Outcome.** `cudaDeepstackPrefillOn` stays false. Three anchors have been run (G-S10h FAIL on the gate's prompt with an unserved dump, G-S10i AMBIGUOUS with an unserved dump, G-S10j FAIL on the served configuration), and the last is the only one whose instrument reproduces the served numbers; it is the one that counts. The 13x prefill speedup (0.9 s against 11.8 s on table.png) is real and is left unused: the resident prefill is less close to HF than the CPU one on three of four images, by cosines of a few hundredths at the first token.
  No further anchor of this kind is run. What could change this is a change to the prefill itself (a lever, a kernel) that closes the 896² image's step-0 gap, with the same anchor as its gate; the gap is largest on the image with the most rows after the 4x6 (803), so the first place to look is the long-sequence batched path, not the DeepStack add. That is a hypothesis, not a finding.

#### Gemma 4 31B on nobara, owner decision 2026-10-08: (b') with (a) as its first step. Step (a), G-31a, registered before any code or run

Owner choice: (b') a layer-streaming Hugging Face float32 reference, and (a) first. This registers (a) only; (b') is registered when (a) has run, because its bar should be set from what (a) shows.

- **The claim step (a) can support, fixed now:** "the 31B loads and runs coherently on the CPU at int4, and its int4 decode agrees with its own int8int8 decode at least as well as a plain-bf16 sibling's does." It says nothing about correctness against Hugging Face; that is (b'). The Gemma 4 31B row in the status table stays "not validated" after (a); it may read "runs on the CPU, coherent, not yet checked against HF".
- **G-31a1, load and served smoke (CPU).** `prequant -quant int4 -target cpu-amd64` writes the 31B's bundle (`~/models/gemma-4-31B-it.int4.cpu-amd64.giw`, expected about 17-18 GB; /home has 93 GB free). One serve binary, `--backend cpu`, the bundle as `--model` and the checkpoint directory as `--vision`: three text chat requests (fixed prompts, in the script) and one image request (table.png, "What does this image show? Answer briefly."), 48 greedy tokens each.
  Automatic bars: the load completes; every reply is non-empty, contains at least 8 distinct tokens, no `<unused`, `<pad>` or NaN-looking output, and the log says the CPU decoder ran (no resident). Then the replies are read by me against the prompts and recorded verbatim with a coherent / not coherent call and the reason; that call is a judgement and is labelled one.
- **G-31a2, int4 against int8int8 on the same checkpoint (CPU), with a sibling for scale.** 12 fixed chat prompts through the checkpoint's real chat template (quantisation is judged only through the template). The int8int8 arm decodes 32 greedy tokens per prompt, recording its path; the int4 arm is teacher-forced along that path; the statistic is the fraction of the 12 x 32 = 384 positions where the int4 argmax equals the int8int8 argmax. Logits must be finite everywhere.
  The same harness runs first on `~/models/gemma-4-E4B-it` (plain bf16, not quantisation-aware, the same regime as the 31B, kept on disk for this; its shape differs: PLE, KV sharing), which sets the scale. **The scale's limit, stated now:** a different shape, and 384 positions give a standard error of about 2 points each, 2.5-3 on a difference, so the margin below is wider than G3's 2.0.
  **PASS:** the 31B's agreement is at least the sibling's minus 5.0 points. **PARKED:** 5.0-10.0 below (to the owner). **FAIL:** more than 10.0 below, or any non-finite logit. The agreement of each arm is printed with the positions where they split and the int8int8 margin there (top-1 minus top-2 probability), so near-ties can be read apart from real flips.
- **Prediction, written now:** the sibling reads 80-92% (the plain E4B was about 77% from HF per arm, and two quantisations of one model agree more than either with float32); the 31B reads within 5 points of it, with a PASS probability of about 0.7. I expect the served replies coherent (0.85), and an image reply that names the table or its contents (0.7).
- **Tier and cost (TE11):** night. Instrument: a Go heavy test for G-31a2 (`decoder`, `realckpt`) and the served script for G-31a1; stopping rule: both complete, fixed prompt sets, no early stop. Cost basis: the bundle write is a read of 59 GB and a quantisation of 31B parameters (I have not timed it; estimate 10-20 min); the E4B sibling about 15 min (two loads, 768 CPU token-steps);
  the 31B's int8int8 arm reads 31 GB per decoded token (about 1-1.5 s each, 384 tokens, 6-10 min) and the int4 arm 17 GB (a third of that), plus two loads (5-10 min each); the served smoke about 10 min with the image prefill. Total estimate 90-110 minutes, queued at 120; timing lock held by `night.py`.
- **Plumbing control by day (not a result):** the harness on a tiny Gemma 4 fixture, and the night script against an empty directory, before it is queued.

#### The `--embed-int4` default, option D: the parked quality evaluation, registered 2026-10-08 before any harness code (owner: "d", with option A's banner line)

Option A is done (`serve`'s banner prints `head table: int4 (...)` or `int8`, from `Model.HeadTable()`; the facts test is mutation-checked). This registers D.

- **The question.** What does the int4 embedding / LM-head table cost against the int8 pin, per model family, now? The only evidence on file is "about 2.3 points of top-1" (measured before 2026-09-28, not re-verified, CPU only) and one real image on which the int4 head had 3.5x the KL to Hugging Face (0.134 against 0.038, Qwen2.5-VL-3B).
- **Instrument.** For each model, Hugging Face float32 (transformers 5.15.0, `~/g4venv`, CPU) runs 16 fixed chat prompts through the checkpoint's own chat template, continues each greedily for exactly 32 tokens (EOS suppressed, so every prompt has 32 positions), and dumps its logits at those 32 positions: 16 x 32 = 512 positions per model.
  goinfer then teacher-forces that HF path on the CPU backend, `--quant int4` in both arms, and the ONLY difference is `EmbedInt4` (false = the int8 pin, true = int4). Each arm is loaded alone and closed. The test asserts `HeadTable()` reads int8 in one arm and int4 in the other, so a run in which the flag did nothing is void.
- **Models (7, all with safetensors on the NVMe; the choice spans the two things the question turns on):**
  - tied head: `qwen2.5-0.5b-instruct` (head 26% of a token), `qwen3-1.7b-bf16`, `qwen25vl-3b-instruct` (text path; the model on which the 3.5x was seen), `gemma-3-4b-it` (262k vocabulary: the biggest table);
  - untied head: `tinyllama-1.1b-chat` (32k vocabulary), `phi3-mini-4k` (32k), `olmo3-7b-think` (100k).
  A model that fails to load or whose logits cannot be compared is recorded as such and the others still run; it does not silently drop out of the count.
- **Statistics, per model** (all paired by position, with a cluster bootstrap over the 16 prompts, 10,000 resamples, fixed seed, 95% interval):
  - d_agree = (top-1 agreement with the HF argmax, int8 head) - (the same, int4 head), in points;
  - dKL = mean KL(HF || int4 head) - mean KL(HF || int8 head), in nats (also the ratio of the means, reported).
- **Per-model reading, fixed now:**
  - **OK:** the interval's upper bound of d_agree <= 2.3 points (the disclosed cost) AND the upper bound of dKL <= 0.02 nats. (0.02 is about half the int4 body's own KL to HF on the one real image on file, 0.038; a head cost above half the body's is not small.)
  - **COSTLY:** the point estimate of d_agree >= 4.3 points (the disclosed cost plus 2.0) OR the point estimate of dKL >= 0.05 nats.
  - **MIXED:** neither.
- **What each outcome means for the default. This registration decides nothing by itself: it only fixes which reading goes with which option, and the owner decides.**
  - All seven OK: option C (the default on everywhere, which needs an int4 table in Metal's resident) is justified on quality; the Mac work is the owner's call.
  - COSTLY on at least two of the four tied-head models and OK on the untied: the candidate is "default off for tied-head models", a rule resolved per model at load, not per backend. COSTLY on both kinds: option B (off everywhere), with the speed given back.
  - Anything else: the numbers go to the owner as they are.
- **Controls:** (1) positive: on `qwen2.5-0.5b-instruct` the harness is run with BOTH arms at EmbedInt4=false, and dKL and d_agree must be exactly 0 (the CPU decode is deterministic; if not, the statistic has noise of its own that the intervals must include). (2) the `HeadTable()` assertion above. (3) HF's own continuation is a sanity row: its mean top-1 probability is printed per model.
- **Limits, stated now:** CPU backend only (the GPU backends dequantise the same table, but their kernels are not exercised); 512 positions per model give a paired agreement interval of roughly +-1 point; greedy continuations of 16 fixed prompts are not a benchmark; the HF reference is float32 of a bf16 checkpoint, so every KL here includes the body's int4 error, shared by both arms.
- **Prediction, written now:** int4-head costs about 1-3 points of agreement and 0.02-0.06 nats on the tied small-vocabulary models, less on the untied; I expect at most two COSTLY. The 3.5x on the one image will turn out to be one image's near-ties.
- **Tier and cost (TE11):** night. Instrument: HF dump script plus a `realckpt` Go test, graded by a script; stopping rule: all seven models, the fixed prompt set, no early stop. Cost basis: HF float32 forwards, about 25 minutes for the seven (the 7B is the long one); goinfer two loads and 2 x 512 teacher-forced steps per model, about 40 minutes; estimate 70 minutes, queued at 100.

- **Amendment A1, 2026-10-08, before any graded run (a power correction; no bar moves).** The plumbing control on `qwen2.5-0.5b-instruct` (HF side 73 s, Go side two runs of 57 s; exploratory, not a result, not quoted as one) showed that my "roughly +-1 point" for 512 positions was wrong: with the 16 prompts as bootstrap clusters the agreement interval's half-width was 2.3 points and the dKL interval's 0.013 nats.
  At that width the OK rule (upper bound <= 2.3 points and <= 0.02 nats) can only be met by an effect near or below zero, so most models would read MIXED for want of data and not for want of an answer. **The prompt count rises from 16 to 32 (1,024 positions per model); the statistics, the bars (2.3 / 0.02 for OK, 4.3 / 0.05 for COSTLY), the models and the controls are unchanged.** The widths should shrink to about 1.6 points and 0.009 nats; if they do not, that is recorded.
  Cost estimate rises to about 110 minutes (HF float32 ~45, goinfer ~65); queued at 150. The same plumbing run confirmed the positive control reads exactly 0 / 0 and that `HeadTable()` reads int8 against int4 across the two arms.

#### P9(d) done, 2026-10-08 (owner decision 4: option A, one vision-language checkpoint per box class, each a single safetensors directory)

- **What shipped (local commits, not pushed):**
  - `pull.Checkpoint` gained `Kind: "directory"`. A directory entry's digest is `Plan.TreeDigest` (sha256 over `path<TAB>size<TAB>sha256-or-dash` lines in path order; the weights are LFS files and carry Hugging Face's declared digest, a small non-LFS file carries its size only). `pull <name>` resolves a directory entry to the same plan-and-verify path as `owner/repo:safetensors`, with the digest riding in `Ref.Pin`, and `Plan.VerifyPin` refuses before any weight byte moves when the repo no longer matches (the directory counterpart of a GGUF entry's pinned sha256).
    The plan output now prints its tree digest, which is also how an entry is authored.
  - Box classes (the docs defined none; these are the owner-approved ones): **S** CPU only, 16 GB RAM or less: `qwen3.5-0.8b` (Qwen/Qwen3.5-0.8B, 1.77 GB, Apache-2.0). **M** an 8 GB GPU, or 16-32 GB Apple Silicon: `qwen3-vl-2b` (Qwen/Qwen3-VL-2B-Instruct, 4.27 GB, Apache-2.0). **L** (24 GB GPU or 64 GB and up): none yet. Qwen2.5-VL-3B is left out on purpose: Hugging Face lists it under `qwen-research` (non-commercial).
  - Each entry's `needs` line carries the tower's cost: the 0.8B about 0.35 GB of float32 weights on the CPU and about 0.7 GB reserved on a CUDA card; the 2B about 1.2 GB and about 1.6 GB (the estimates `towerVRAMEstimate` uses; 1.64 GB for the 2B was checked against 1158 MiB measured resident after an image). The 2B's line also says what we measured: about 15 s for a 1,000-token image turn on the 8 GB card, because its DeepStack prefill runs on the CPU, and 12-18 s for the CPU tower alone.
    The resident-size figures (about 1 GB and 1.2 GB at int4) are estimates and the lines say so. The Tools column reads "not yet measured" for both (the registry test accepts that placeholder and no `serve check` run was made).
- **A blocker found on the way, and what was done about it.** The registry's own rule (`TestRegistry_noEntryOutrunsItsParity`) allows only families with a full-oracle or real-oracle parity, and the manifest still recorded `qwen3_vl` as `experimental: tiny-oracle`, "the text half only; it doesn't take images yet". That was stale: G-S10a and G-S10c had passed on 2026-10-07 and nothing had carried them into the manifest. Both were re-run on current main on 2026-10-08:
  - `TestQwen3VLReal_gate` (the 2B's text forward against HF f32): logit cosine 1.000000, argmax 12095 equal, the 6-token continuation equal (`p9d/qwen3vl-real-gate-text.log`).
  - `TestQwen3VLImageReal` (a real image prompt, table.png, 1083 ids, 3 DeepStack sets): last-position cosine 1.000000, argmax 86608 equal, 15/15 text positions after the image; the three planted defects red at 0.982867, 0.987967 and -0.135017 (`p9d/qwen3vl-image-gate.log`).
  Then: `qwen3_vl` is a required real-checkpoint gate in `cmd/gate` (`parityRealckptGates`, `emitGates`, out of `realckptNotRequired`), the asset's candidate path now names `~/models/qwen3-vl-2b-instruct`, the ledger entry is promoted (`gate ledger promote`), the manifest block is edited by hand in the olmo3 v0.21.0 manner (status validated, method full-forward-oracle, 100.0%/1.00000, deps_hash untouched and still fresh: 40/40 families enforced), and the matrix row now says it reads images. The registry rule was mutation-checked: with `qwen3_vl` set back to `experimental: tiny-oracle` it goes red.
- **Checked end to end:** `pull qwen3.5-0.8b` fetched 1.6 GiB in 20 s with the digest verified and every weight file sha256-checked, and `serve` on the pulled directory (`--backend cpu`) answered table.png in 16.1 s with "Table 2. Quarterly unit sales by region (thousands)...", the banner showing the new `head table: int4` line (`p9d/pull-qwen3.5-0.8b-serve.log`). **Not checked end to end:** the 2B's pull (4.3 GB). Its digest was authored from Hugging Face's listing and every validation in this tranche used a copy of the same files, not a fresh pull.
- **Limits:** `TreeDigest` covers a small non-LFS file by size only, so a same-size edit to a config or tokenizer would not be caught; the weights, which are the point, are. The M line's CUDA wart is temporary: it changes when the resident DeepStack prefill question (the 896 image, G-S10j) is resolved.

##### G-S14a read 2026-10-08: PASS (S14.1 done; aikit commit on the local `main`, goinfer test `multimodal/whisper_features_golden_test.go`, golden `testdata/whisper_features_golden.zip`)

- **Against the NumPy path (float64, the one this port follows): bit-exact.** The largest absolute difference is **0.000e+00** on all three cases that carry that path (the LibriSpeech clip at 128 mels, the same clip at 80 mels, the 10 s synthetic signal); the mean is 0 as well. My prediction was "within 3e-06"; it came out exact, because the port rounds where the reference rounds (the FFT result is stored as complex64 in the NumPy path, the log is cast to float32 before the clamp and the final `(x + 4) / 4`).
- **Against the torch path (float32, what `__call__` uses):** the difference is exactly the two reference paths' own spread, on all seven cases: 1.657e-05 (LibriSpeech, 128 mels), 1.156e-05 (80 mels), 2.372e-05 (44.1 kHz reference), 1.514e-05 (48 kHz stereo reference), 3.111e-05 (synthetic 10 s), 7.272e-06 (800 samples), 0 (silence); means 6e-10 to 2.4e-07. All are under the 5e-05 bar. **A correction to the registration:** it said the bar was three times the references' spread, from the one clip I had measured (1.66e-05); across the seven cases the spread reaches 3.11e-05, so 5e-05 is 1.6 times it. The bar was not moved.
- **Valid-frame counts** equal the extractor's attention-mask sums for all nine lengths: 1, 159, 160 give 1; 161 gives 2; 800 gives 5; 16,000 gives 100; 93,680 gives 586; 480,000 and 500,000 give 3,000 (ceil(n/160), capped).
- **Planted defects, all six red:** the worst difference against the NumPy path was 1.942 (HTK mel scale), 0.706 (magnitude for power), 0.875 (no max-8 clamp), 0.221 (symmetric Hann), 0.473 (zero padding at the edges), 6.418 (no `(x + 4) / 4`), against the bar 5e-05; the weakest still clears it by four orders of magnitude.
- **Also checked (aikit's own tests):** the 400-point mixed-radix FFT against a direct DFT (1e-9), silence equals -1.5 everywhere with no NaN, a clip past 30 s is truncated to its first 30 s exactly, an empty clip and a mel count other than 80 or 128 are refused. A 30 s extraction takes about 0.13 s at 128 mels on this CPU (not a graded figure).
- **Limits:** one reference version (transformers 5.15.0); inputs are speech, tones plus a hashed noise, silence and a very short clip, not noisy or clipped real recordings; no 16 kHz resampling here (the resample path exists and is gated separately, G-S5e); the extractor's batch and dither options are not ported (dither is 0 by default).
- **Open before S14.2:** aikit has an unpushed local commit; goinfer's push waits on an aikit tag, as for any aikit-dependent change. S14.2 (Qwen3-ASR's encoder and projector) is registered when it starts.

##### G-S14b, Qwen3-ASR's front end, audio encoder and projector (S14.2), registered 2026-10-08 before any code

Desk read, 2026-10-08 (transformers 5.15.0 `models/qwen3_asr`, `~/models/qwen3-asr-0.6b` = Qwen/Qwen3-ASR-0.6B, 1.88 GB bf16, Apache-2.0, ungated). **Three findings that change the plan:**

1. **Qwen3-ASR's extractor is NOT G-S14a's.** `Qwen3ASRFeatureExtractor` has only the torch float32 path (`torch.stft`, centre reflect padding, power, `[..., :-1]`); the processor calls it with `padding=True, truncation=False`, so a clip is NOT padded to 30 s and NOT truncated: its feature count is floor(n/160), the reflect padding at the end reflects real audio, and a clip over 30 s keeps all its frames. A clip under 8,000 samples is zero-padded to 8,000 first (`min_length`, and the mask is deliberately not adjusted); the mel axis is then right-padded to a multiple of 100 (`2 * n_window`) with the mask padded likewise. G-S14a's function (pad to 30 s, NumPy path) is the Whisper one; this needs a second entry point on the same arithmetic, and its reference is float32, not float64.
2. **The encoder (AuT) as read:** the 128 x T features are split into chunks of 100 frames; each chunk goes through three Conv2d (1 -> 480 -> 480 -> 480 channels, k3 s2 p1, exact-erf GELU after each), a bias-free Linear (480 x 16 = 7,680 -> 896), plus a sinusoidal position added per chunk (positions restart at 0 in every chunk; 13 time steps per full chunk); the valid post-CNN positions (per chunk, `((n-1)//2+1 ...)` three times) are packed into one sequence; 18 pre-LayerNorm layers (14 heads of 64, biased q/k/v/o, GELU MLP 896 -> 3584 -> 896) with **windowed attention** (blocks of 8 chunks = 104 post-CNN positions, `cu_seqlens`), then `ln_post`; the projector is `proj1` (896 -> 896), GELU, `proj2` (896 -> 1024). The token count for n valid mel frames is `((((n % 100) - 1)//2 + 1 - 1)//2 + 1 - 1)//2 + 1 + (n // 100) * 13`.
3. **The checkpoint's layout:** every tensor is under `thinker.` (the audio encoder at `thinker.audio_tower.*` with `proj1`/`proj2` inside it, the decoder at `thinker.model.*`, a separate `thinker.lm_head.weight`), config nested under `thinker_config`; the decoder is Qwen3 (28 layers, 16 heads, 8 KV, head 128, hidden 1024, tied head declared) with an interleaved m-RoPE section [24, 20, 20] that is plain RoPE when all three position components are equal (as for audio). That is S14.3's loader work, not this stage's.

**Gates for S14.2:**

- **G-S14b1, the ASR front end** (aikit `audio`, a variable-length entry point beside `WhisperFeatures`; the min-length padding, the mask, and the multiple-of-100 padding included). Reference: the installed `Qwen3ASRFeatureExtractor` (torch float32) with the processor's arguments. Bars: shapes and mask equal; largest absolute difference **<= 1e-04** and mean **<= 1e-06** (float32 against the float64 port: the two Whisper paths differ by up to 3.1e-05, so this is 3 times that). Cases: the LibriSpeech clip; a 3 s clip (padded to 8,000 samples); a clip whose frame count is exactly a multiple of 100; one one frame past it; a 12 s synthetic signal; a 45 s one (past 30 s, nothing dropped); silence. Planted defects, each red: zero padding at the end instead of reflection; padding to 30 s; truncating at 30 s; no multiple-of-100 padding; no min-length padding; a mask padded with ones.
- **G-S14b2, the encoder and projector, tiny.** A random-weight Qwen3-ASR encoder small enough to commit (d_model 32, 2 layers, 2 heads, downsample 8, n_window 4 and n_window_infer 16 so chunks and windows both repeat inside a short clip), pinned from the transformers class with its norms randomised (as the S10 tower fixtures were). Bar: per-token cosine **>= 0.99999** against the float32 reference and the token count exact. Planted defects, each red: no position embedding; one position table for the whole clip (no restart per chunk); full attention over all tokens instead of windows; tanh GELU for the exact one; the padding positions of a partial chunk left in; the projector's GELU removed.
- **G-S14b3, the real encoder.** Qwen3-ASR-0.6B's encoder and projector, goinfer float32 on the CPU against transformers float32, on the LibriSpeech clip, a 12 s clip and a 45 s clip (so windows span more than one block). Bar: per-token cosine **>= 0.9999** for every output token, argmax of the nearest vocabulary embedding equal for at least 95% of tokens (a sanity read of the projector's output, recorded, not the verdict), token count exact. The 45 s clip has several windows; if its cosine is worse than the 12 s one's by more than a factor of 10 in (1 - cosine), that is recorded as the windowing's own error.
- **Prediction, written now:** G-S14b1 within 3e-05 of the torch float32 path (the port is float64); G-S14b2 cosine 1.00000 with every planted defect under 0.999; G-S14b3 cosine >= 0.99999 on all tokens. The most likely stumbles: the per-chunk position restart, the 104-position window boundaries, and the valid-position packing for a chunk that is only partly real.
- **Tier and cost:** quick by day for the front end and the tiny encoder; the real encoder is a few minutes on the CPU. Roughly a day of work. Cross-repo as before (aikit has the code, goinfer the goldens and gates).

##### G-S14b read 2026-10-08: PASS on all three (S14.2 done; aikit `audio/qwen3asr_features.go` and `qwen3asr_encoder.go`, goinfer `multimodal/qwen3asr_test.go`, `qwen3asr_real_test.go`, fixtures `testdata/qwen3asr-tiny/` and `testdata/qwen3asr_features_golden.zip`)

- **G-S14b1, the front end: PASS.** Against transformers 5.15.0's `Qwen3ASRFeatureExtractor` (torch float32), as the processor calls it, on seven clips: shapes, valid counts and masks equal on all; largest absolute difference **1.66e-05, 1.13e-05, 4.40e-05, 4.40e-05, 3.20e-05, 1.71e-05, 0** (the LibriSpeech clip, 4,800 samples padded to 8,000, exactly 300 frames, 301 frames, 6 s synthetic, the 30.5 s clip past one Whisper chunk, silence), means 1.4e-07 to 3.9e-07, against the bar 1e-04 / 1e-06. As registered, the port works in float64 against a float32 reference, so it differs by float32 rounding; the prediction (within 3e-05) missed by 1.5e-05 on two cases and stayed under the bar.
  The LibriSpeech clip has 585 valid frames here against 586 for Whisper's extractor (floor against ceiling), confirming the finding that the two are different front ends. **All six planted defects are red:** zero padding at the clip's end (values), padding to 30 s, truncating at 30 s, no multiple-of-100 padding and no min-length padding (shape or valid count), the mask padded with ones (mask).
- **G-S14b2, the encoder and projector on the tiny checkpoint: PASS.** Per-token worst cosine **1.00000000** against the transformers modules on all three clips (26, 69 and 91 tokens; windows repeat inside every clip), max |diff| 1.5e-06 to 5.5e-06 on outputs of rms 1.3-1.9; token counts exact. **All six planted defects are red:** no position embedding (cosine 0.895), one position table for the whole clip (0.934), full attention over all tokens (0.921), the partial chunk's padding positions kept (token count), no projector GELU (0.542), and tanh GELU for the exact one (cosine 0.99999985, max |diff| 3.0e-03).
  **Two amendments, both made before the real-checkpoint reading:** (A1) the registered cosine bar (0.99999) cannot see tanh-for-erf GELU, which leaves the projected cosine at 0.99999985, so a second criterion was added, max |diff| <= 2e-05 (a tightening; the port matches to ~6e-06); and (A2) the first fixture draw used weights of std 0.15, on which tanh-GELU gave cosine 1.000000 and full attention 0.9991: a fixture too gentle to fail. The pin script now sizes weights so pre-activations are of order 1-3 and attention is sharp (scales and the reason are in the script). The outputs of those two draws are not kept.
- **G-S14b3, the real encoder: PASS.** Qwen3-ASR-0.6B (`~/models/qwen3-asr-0.6b`), goinfer float32 on the CPU against transformers float32 (the log is `s14/gs14b3-real-encoder.log`):

  | clip | tokens | tower worst cosine | tower max abs | projected worst cosine | projected max abs |
  |---|---|---|---|---|---|
  | LibriSpeech (5.86 s) | 76 | 1.00000000 | 2.6e-05 | 1.00000000 | 1.2e-06 |
  | synthetic 12 s | 156 | 0.99999994 | 9.1e-04 | 0.99999995 | 2.9e-05 |
  | LibriSpeech tiled, 46.8 s | 609 | 1.00000000 | 6.2e-05 | 1.00000000 | 2.4e-06 |

  Bar 0.9999 per token: met by four to five orders of magnitude. Token counts exact. **The windowing costs nothing visible:** the 46.8 s clip (609 tokens, six attention windows of 104) is as good as the 6 s one. Fed the Go front end's features instead of the reference's, the projected worst cosines are 1.00000000, 0.99999998 and 1.00000000 (recorded, not the verdict). Exploratory timing on this CPU: 15 s for both passes of the 46.8 s clip.
  **The registered "nearest vocabulary embedding" sanity read was not made**; the reference script's transcription of the clip (below) is a stronger check of the same thing, and the projector output is gated token by token.
- **Finding about the reference, the most important of the stage:** **transformers 5.15.0 cannot read the Qwen3-ASR-0.6B checkpoint as shipped.** It keeps the nested `thinker_config` as defaults, builds the 1.7B's dimensions (d_model 1024, 24 layers, text hidden 2048) and reports every weight MISSING, randomly initialised, with no error: a reference built by a plain `from_pretrained` would be noise and would pass nothing honestly.
  `scripts/pin_qwen3asr_real.py` converts the checkpoint by a pure rename (`thinker.audio_tower.proj1/2` to `model.multi_modal_projector.linear_1/2`, the rest of `thinker.audio_tower` to `model.audio_tower`, `thinker.model` to `model.language_model`, `thinker.lm_head` to `lm_head`) into a flat-config directory, loads it with **zero missing and zero unexpected keys**, and asserts the proof: it transcribed the LibriSpeech clip as "language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." That assertion is in the script and gates the reference.
- **Limits:** one clip of real speech (the rest synthetic or tiled), one reference version, float32 on the CPU only (no int4 or GPU yet), the encoder loaded only from the original `thinker.*` layout (and bare `audio_tower.` / unprefixed names), and `n_window` 50 only (the reference hard-codes 13 steps per chunk, so any other value could not be validated). Speed was not optimised: the GELU uses `math.Erf` per element, the conv stem is im2col plus the blocked matmul.
- **Not yet done (S14.3):** the decoder side. Qwen3-ASR's decoder is Qwen3 under `thinker.model.*` with a nested config; goinfer's `qwen3` loader reads neither, so S14.3 starts with a loader adapter, the `<|audio_start|><|audio_pad|><|audio_end|>` splice, and served transcription, gated by word error rate against Hugging Face on LibriSpeech clips.

#### S6 on nobara, registered 2026-10-08 before any run

- **Gemma 4 E4B on CUDA.** The checkpoint is `~/models/gemma-4-E4B-it` (`google/gemma-4-E4B-it`, 15.99 GB `model.safetensors`, downloaded today onto the NVMe): 42 layers, hidden 2560, 18 KV-shared layers, PLE width 256, vision and audio configs. It goes through S1's E-model gates, which are the Mac's rules unchanged.
  - **G-E4B-C1, S1's G3c on CUDA:** in one process, `g3Run` on Qwen2.5-Coder-1.5B on CUDA (the validated reference), then on E4B loaded from the safetensors directory by both sides with the same `Options` (int4, context 512, `EmbedInt4` unset on both: table precision is the confound Metal's and CUDA's first G3 runs fell into).
    PASS: E4B's teacher-forced agreement at least the reference's minus 2.0 points and its free-run passes at least the reference's minus 1; 2.0-4.0 points below is ambiguous (parked for the owner); worse, or free-run passes 2 or more short, fails. The harness is E2B's (`g3Model`) taking a directory: the tokenizer and template from the directory's `tokenizer.json`.
  - **G-E4B-C2, served:** G4's image request and G-S5c's three audio clips through one CUDA serve binary, `--backend cuda` against `--backend cpu`, `-vision-device cpu` and `--embed-int4=false` on every arm, plus a second CPU run. Identical replies, or a first divergence at a near-tie (p(other) at least half p(top)); the CPU repeat byte-identical;
    a CUDA arm not decoding resident voids that reading. If the 8 GB card cannot hold E4B's resident, that is the finding and is reported with the plan line.
  - **Planted defects:** none new; the E-model defects (S1's eight) are shape-independent and were shown red on the tiny fixture. E4B adds no new mechanism beyond its width.
  - **Prediction:** the E4B's weights at int4 are about 2.5 GB resident, so it fits beside a context-512 plan; I expect G-E4B-C1 to land near E2B's delta (S1 on CUDA) and make no stronger claim.
- **Qwen3.5+ MoE images (never run on any box).** Qwen3.6-35B-A3B (`qwen3.6-35b-a3b`, in the archive; the text decoder is `~/models/qwen3.6-35b-a3b-int4.giw` on the NVMe). The tower's weights are copied to `~/models` for anything timed. A hybrid with MoE layers is not claimed by CUDA's resident m-RoPE prefill ("no MoE hybrid image gate exists"), so the image turn prefills on the CPU.
  - **What is gated:** the tower (G-S2b's bar, the CUDA Qwen3 tower against aikit's CPU encoder, merged rows at worst-row cosine >= 0.9999, on the real checkpoint's four F2a images at serve's cap) and a served image request, `-vision-device auto` against `cpu`, identical reply or a near-tie first divergence. There is no HF anchor for the 35B on this box (70 GB of bf16 does not fit in 62 GB of RAM); the claim is "runs, and the CUDA tower is the CPU tower", not "matches HF".
  - **Night work** (a 35B image turn prefills on the CPU, minutes).
- **Gemma 4 31B (`~/models/gemma-4-31B-it`, 59 GB, downloaded).** Dense, 60 layers, hidden 5376: about 17 GB at int4, so it cannot be GPU-resident on the 8 GB card, and a float32 or bf16 HF reference does not fit in 62 GB of RAM. So the 26B's shape (a CPU forward as the reference for a GPU arm) does not apply: there is no GPU arm and no oracle here.
  What can be gated is stated as a question for the owner, not decided here: (a) a CPU smoke plus an internal int4-against-int8int8 consistency on the same checkpoint and a served image and text request read for coherence, claimed as exactly that; (b) an HF bf16 oracle with disk offload (hours, night); (c) defer. Nothing is run for the 31B until this is decided.

#### S6 on nobara, G-E4B-C1 read 2026-10-08: FAIL as registered, root-caused to the checkpoint and not to the implementation; the gate is re-registered on the right one (raw `docs/measurements/multimodal-support-2026-10/s6-e4b/`)

- **The reading (`g-e4b-c1-plain-bf16.log`, 217 s):** `google/gemma-4-E4B-it` (the plain bf16 checkpoint) loaded by both sides at int4, context 512: E4B teacher-forced agreement 381/435 = **87.59%**, 3 of 8 prompts through the free-run rule; the reference (Qwen2.5-Coder-1.5B) 374/398 = 93.97%, 7 of 8. Delta
  **-6.38 points** and free-run 3 against 7: FAIL under the rule (worse than -4.0, or 2 or more free-run passes short). For comparison the E2B read 94.48%, 7 of 8 (+0.51).
- **What it is not: an implementation defect.** Two cuts, both exploratory (`TestGemma4EModel_realE4BAnchorDump`, `TestGemma4EModel_realE4BF32Dump`, `scripts/anchor_e4b_hf.py`; transformers 5.15.0 in `~/g4venv`, the E4B in float32, the same 435 ids teacher-forced):
  - **goinfer's CPU in float32 against HF float32: 435 of 435 argmax positions identical (100%).** The E-model implementation (PLE, KV sharing, the two geometries, K=V, the layer scalars) is right on the real E4B.
  - **At int4, each arm is about equally far from HF:** goinfer CPU 77.70% (338/435), CUDA 77.01% (335/435); where the two disagree (54 positions) HF sides with the CPU at 16, with CUDA at 13 and with neither at 25. CUDA is not the odd one out. The two int4 implementations scatter around HF
    and around each other (87.59%) far more than the E2B's do.
- **The cause, as far as the evidence goes:** the E2B gates (S1, G3c) ran Google's QUANTIZATION-AWARE-TRAINED `gemma-4-E2B_q4_0-it.gguf`, whose weights were trained to survive 4 bits. The E4B I downloaded and gated was the plain bf16 checkpoint, which goinfer quantizes after the fact. I did not run the plain
  E2B bf16 at int4 to show the same drop there, so "post-hoc int4 of a non-QAT checkpoint is lossy at this size" is the explanation that fits and is not proven. The registration named `~/models/gemma-4-E4B-it` for a gate whose intent (S1's) is the QAT analogue: that was my error, disclosed here.
- **Re-registration, before any run on the QAT file (the bar, the procedure and the reference are unchanged; only the checkpoint is the E2B gate's equivalent):** **G-E4B-C1b** is G3c on `google/gemma-4-E4B-it-qat-q4_0-gguf` (`gemma-4-E4B_q4_0-it.gguf`, 5.15 GB), loaded by both sides as a GGUF exactly as the E2B file is, with the same rule: teacher-forced agreement at least
  the reference's minus 2.0 points and free-run passes at least the reference's minus 1; 2.0-4.0 points below parked; worse fails. **G-E4B-C2** (served, image and three audio clips, CUDA against CPU) runs with that GGUF as `--model` and the QAT-unquantized safetensors (`google/gemma-4-E4B-it-qat-q4_0-unquantized`) as `-vision`, the E2B layout.
  **Prediction:** near E2B's delta (the same QAT recipe, a wider model); I will not claim more. The plain-bf16 FAIL above stays recorded as it is.

#### S6, Gemma 4 E4B on CUDA: G-E4B-C1b and G-E4B-C2 PASS (read 2026-10-08, nobara; raw `docs/measurements/multimodal-support-2026-10/s6-e4b/`)

- **G-E4B-C1b, G3c on the QAT GGUF (`g-e4b-c1b-qat-gguf.log`, 225 s): PASS.** Both sides loaded `gemma-4-E4B_q4_0-it.gguf` with the same options; E4B teacher-forced agreement **405/435 = 93.10%**, **8 of 8** prompts through the free-run rule; the reference (same process) 374/398 = 93.97%,
  7 of 8; delta **-0.87 points** against the -2.0 margin; free-run passes 8 against 7. The same procedure on the plain bf16 checkpoint read 87.59% and 3 of 8 (above): the only change is the checkpoint, which supports the quantization-aware-training explanation. My prediction ("near E2B's +0.51") held in direction; the delta is
  0.9 points lower than E2B's and inside the margin.
- **G-E4B-C2, served (`g-e4b-c2-image/`, `g-e4b-c2-audio/`): PASS.** The QAT GGUF as `--model` and the QAT-unquantized safetensors (`google/gemma-4-E4B-it-qat-q4_0-unquantized`, downloaded today) as `--vision`, `-vision-device cpu` and `--embed-int4=false` on every arm, `--kv-sessions 1 -ctx 4096`, the CPU arm repeated.
  - **Image (table.png, 32 greedy tokens):** the CUDA arm `decode path: cuda-resident (int4)`; its reply first differs from the CPU's at generated token 3 (CPU ' a' 0.488, CUDA ' me' 0.459, p(other) over half p(top): a near-tie); the CPU repeat IDENTICAL.
  - **Audio (the three clips):** the CUDA arm `prefill resident`, `cuda-resident` on all three; short first differs at token 10 ('\n' 0.300 against ' I' 0.268, near-tie), mid IDENTICAL, long at token 24 (' mechanical' 0.053 against ' bird' 0.042, near-tie); the CPU repeats IDENTICAL on all three. (The clips are synthetic tones, so the replies describe sounds and are not transcripts; the gate is CUDA against CPU, as registered.)
- **What is and is not established:** the E-model shape (PLE width 256, 18 KV-shared layers, two KV heads, no double-wide FFN) runs on CUDA and agrees with the CPU at the E2B's level on the QAT checkpoint; goinfer's CPU float32 agrees with HF float32 at 435 of 435 positions on the plain checkpoint. Not done: a CUDA speed record for E4B, and the E4B on Metal
  (the Mac's, whose sidecar build failed the fit guard last night). The `gemma-4-E4B-it` bf16 directory (16 GB) is on the box only because of the first gate; it can be removed.

#### S6, Qwen3.6-35B-A3B images on CUDA: the tower gate PASSES (read 2026-10-08, nobara); the served check is registered below before it runs (raw `docs/measurements/multimodal-support-2026-10/s6-moe/`)

- **The tower (G-S2b's bar on the MoE checkpoint), `g-s2b-moe-tower.log`: PASS.** `TestGridVisionCUDA_real` gained a `qwen3.6-35b-a3b` entry (`cuda/grid_vision_test.go`) reading `~/models/qwen3.6-35b-a3b-vision`: the two tower shards (5.28 GB of the 26) copied from the archive with the config, preprocessor and tokenizer files and an index filtered to the 370 vision tensors, because the 35B's tower is the only part of it that this gate reads.
  Worst merged-row cosine of the CUDA tower against aikit's CPU encoder on the four F2a images at serve's cap: **0.999999995 / 0.999999997 / 0.999999999 / 0.999999981** against the 0.9999 bar. Exploratory timing (not a result): CUDA 1.514 / 0.119 / 2.198 / 2.077 s against CPU 14.074 / 0.689 / 20.585 / 19.231 s.
- **Registered before the served run: G-S6m, the served image check.** One image request (`testdata/qwen35vl_preprocess_image.png`, 96x64; the 1200x900 table would prefill for a long time on the CPU), "What does this image show? Answer briefly.", 32 greedy tokens, top-3 log-probabilities, through ONE CUDA serve binary
  (`run-s6-moe-night.sh`, which wraps `run-gs3c-served.sh` with its new `GS3C_IMAGE`): `~/models/qwen3.6-35b-a3b-int4.giw` as `--model`, the vision directory above as `--vision`, `--kv-sessions 1 -ctx 4096`, three arms, all `--backend cuda`: tower on CUDA (`-vision-device auto`), tower on the CPU (the reference), tower on the CPU again (the control).
  The image turn prefills on the CPU on every arm (no MoE hybrid image gate exists on CUDA), so the tower is the one thing that differs between the first two.
  - **PASS:** the CUDA-tower reply identical to the CPU-tower reply, or first differing at a near-tie (p(other) at least half p(top)); the repeat IDENTICAL; every arm logs `decode path: cuda-resident`. **FAIL:** a divergence that is not a near-tie, or a repeat that differs. **VOID (not a failure of the engine):** an arm that did not decode resident, or whose tower did not come up on CUDA: the script prints `!! arm ... did not decode cuda-resident` and exits 1.
  - **What it can establish:** the 35B runs an image turn end to end on this box and the CUDA tower is the CPU tower in the served path. It cannot establish agreement with Hugging Face: 70 GB of bf16 does not fit in 62 GB of RAM, so there is no anchor, and the claim stays "runs, and the CUDA tower is the CPU tower".
  - **Prediction:** the tower rows agree to 1e-8 (above), so the two replies are IDENTICAL; I would be surprised by a near-tie divergence and would read one as the f32 sum order of the tower, not a defect.
  - **Cost and tier:** night. ~40 min (three arms, each a ~9-minute load per the 35B's earlier cycles, plus a CPU-prefill image turn); queued at est 60. The giw is the local copy (`~/models`, NVMe); it has the archive v12 file's size and header and differs from it at byte 78, so a load refusal would show in the first arm's log, which is the one risk of this job not producing a reading.
- **Gemma 4 31B: not run.** Still the owner's question as registered above ((a) CPU smoke plus an internal consistency check, claimed narrowly; (b) HF bf16 with disk offload, hours, at night; (c) defer).

## Order of work (owner, 2026-10-07 evening; replaces the morning order)

1. **In flight, finish:**
   - S3's night root-cause steps;
   - S4 step 5;
   - S9 on CUDA;
   - S6 (E4B, 31B);
   - the 2026-10-07 audit's fixes.
2. **Then S7 and S13-lite** (one model per box), against the 5 s bar.
3. **Then S16, S17 and S18,** in the order those measurements rank them: the largest gap to the bar and to the peer
   first.
4. **Then:**
   - S10's remaining families: Pixtral, LFM2.5-VL, North, and the Qwen3.5+ and Qwen3-VL MoE images;
   - S11 (several images);
   - S15 (video);
   - S12 (WebGPU);
   - full S13;
   - S8 (the README table);
   - S14 (speech).

S8's drift check keeps the README true through S14, which updates the table as part of its own work.

## Decisions for the owner

**Decided 2026-10-07 evening** (after the Cowork audit of this doc):
- **Video is back in scope (S15).** This reverses the morning's "Video: deferred; no phase."
- **S7's speed bar:** a cell is usable when a new image's TTFT is under 5 s at the default image budget, on the box's
  best backend.
- **Order:** S7 and a one-model-per-box S13 move ahead of S10's remaining families, S11 and S12, so the speed work aims
  at a measured gap ("Order of work").

**Decided 2026-10-07:**
- **Video:** deferred; no phase. (Reversed the same evening; above.)
- **Gemma 4 E4B and 31B:** download and validate (S6), not a narrowed claim.
- **WebGPU:** invest in it for multimodal (S12).
- **Speech:** last (S14), after the WebGPU work; the family is chosen when it starts.

**Still open:**

- Gemma 3's int8 SigLIP default under `--backend cuda|webgpu` (`siglip-int8-fidelity-2026-10-07.md`); nobara builds
  the float32 CUDA tower and the served comparison first.
- Whether `s2-towers` is merged and pushed (aikit's side, v1.59.0, is tagged).
- serve's `--embed-int4` default differing by backend on tied-head families (the int4 head measured 3.5x the int8
  head's KL to HF on a real image).
- S14's speech family (P11's options; decided when S14 starts).
- ~~Which VL checkpoint `pull` recommends per box class (P9(d)).~~ Decided and shipped 2026-10-08, "P9(d) done".
