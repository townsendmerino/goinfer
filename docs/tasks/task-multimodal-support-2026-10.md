# Multimodal support: every image and audio model on every backend, at a usable speed (2026-10)

**Status: ACTIVE (refreshed 2026-10-10).** The phases' state:
- **Done:**
  - S1 (E-model decode on the GPU), Metal and CUDA.
  - S2 (Qwen3.5+ and GLM-OCR towers); `s2-towers` is merged into `main` (`471a500e`, 2026-10-08).
  - S3 (Gemma 3 and Qwen2.5-VL on Metal), root cause and follow-ups included.
  - S4 steps 0-5 (the CUDA towers) and the float32 SigLIP addendum.
  - S5 (E2B audio into the model; G-S5b PASS).
  - S8 (the README and doc support table, computed from code and drift-checked) and S11 (several images per
    message), both 2026-10-09.
  - S9 on Metal and CUDA.
  - S10 for Qwen3-VL, Ministral 3 (Pixtral) and LFM2.5-VL. "North" was dropped (2026-10-09).
  - S16 for Qwen2.5-VL and Qwen3-VL.
  - S17's levers A (Metal, CUDA) and B (Metal), and Gemma 3's resident image prefill on Metal, with the f16 residual
    fix it needed (2026-10-09). The owner kept the prefill on over G-IP4's registered FAIL.
  - S7 and S13-lite on both boxes. The Mac's third pass (read 2026-10-10) has six of eight S7 cells under the 5 s bar.
    - Qwen2.5-VL-3B at 5.51 s is accepted as it is (owner, 2026-10-10: "we will just accept 5.51 as OK").
    - Gemma 3 4B read 6.2 s on a build that predates its image prefill, which read 4.14 s by day (G-IP4).
  - The `--embed-int4` default's evaluation (option D): 7 of 7 models graded, none costly.
- **In flight:**
  - S10 for Qwen3-VL MoE: G-S10q-a, b and c PASS. The served check G-S10q-d is queued on nobara for the night of
    2026-10-10 (`gs10q-d-2`; the first run was void).
  - S18: every gate has read. G-S18g read FAIL as registered, on a rule that cannot grade a tower on Gemma 3 (the f16
    reference fails it too). The owner withdrew its consequence; G-S18g2 replaces it and is queued on the Mac for the
    night of 2026-10-10.
  - S6: the 35B's served image check (G-S6m) and the 31B's step (b') PASS; Gemma 4 E4B PASS on CUDA. E4B on Metal
    (G-E4B-1, G-E4B-2) has not run: its sidecar build was refused for memory on 2026-10-07. A Metal sidecar built by
    serve has been on the Mac since 2026-10-09.
  - S14 (speech): Qwen3-ASR, the Whisper encoder and Voxtral Mini's gates have passed, except Qwen3-ASR's
    word-error-rate gate G-S14c4b, which failed again on its re-read (2026-10-09).
- **With the owner:**
  - The `--embed-int4` default, from option D's numbers. To be asked after G-S18g2 reads (owner, 2026-10-10).
  - The int8 Metal tower as the tight-memory default, from G-S18g2.
- **Not started:** S15 (video), S12 (WebGPU), full S13.

The order is in "Order of work". Each phase writes its own gates into this doc, and commits them, before its first
measurement (CLAUDE.md, "Pre-registration").

The owner, 2026-10-06, after the multimodal plan (`docs/multimodal.md`) was finished and aikit v1.58.0 shipped:
"we want better overall support." Three axes:
- **Which models** take images and audio.
- **Which backends** run them on the GPU (CPU, CUDA, Metal, WebGPU).
- **Whether they are fast enough** to use interactively.

The last phase puts the answer where users look first, the README, with a check that keeps it true.

## Where it stands (refreshed 2026-10-09, from the gate readings below; the speed list is not refreshed)

"Tower / decoder": where the image or audio encoder runs, then where the language model runs after it.
- **Branches:** everything is on `main`. `s2-towers` was merged on 2026-10-08 (`471a500e`).
- **Image turns on Metal:** "prefill CPU" means the image turn prefills on the CPU and uploads its KV before decoding on
  the GPU. Since 2026-10-09 only the Qwen3.5+ hybrid still does this, because S16's hybrid half is parked. Qwen2.5-VL
  and Qwen3-VL (S16), Gemma 3 (S17) and GLM-OCR (pairwise RoPE, Part A) prefill on the GPU.

| | CPU | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| Gemma 3 | CPU / CPU | GPU, float32 by default (int8 when the card cannot hold it; 2026-10-08) / GPU | GPU, float32 / GPU, resident by default on the 16 GB Mac (G-S18a), image prefill on the GPU (S17) | GPU (int8) / GPU |
| Gemma 4 E2B | CPU / CPU | GPU / GPU, batched image prefill (S9 on CUDA) | GPU / GPU, layer-major prefill (S9) | CPU / CPU |
| Gemma 4 E2B audio | CPU / CPU | CPU / GPU (not run served) | GPU / GPU | CPU / CPU |
| Gemma 4 E4B | CPU / CPU, not yet validated | validated (S6: G-E4B-C1b, G-E4B-C2) | not validated (the Mac's sidecar build was refused by the fit guard) | CPU / CPU |
| Gemma 4 26B | CPU / CPU | GPU (checked on E2B) / GPU | GPU (checked on E2B) / GPU | CPU / CPU |
| Gemma 4 31B | CPU / CPU: runs, coherent (G-31a: text and one table image); text logits agree with Hugging Face float32 as well as the E4B's do ((b'), 12 prompts, CPU); the image path is not checked against Hugging Face | not validated (8 GB card) | too large for the Mac | not validated |
| Qwen2.5-VL | CPU / CPU | GPU / GPU | GPU / GPU, image prefill on the GPU (S16) | CPU / GPU |
| Qwen3.5+ dense | CPU / CPU | GPU / GPU | GPU / CPU: an image turn prefills and decodes on the CPU (Metal has no hybrid m-RoPE prefill; S16's hybrid half is parked); text turns are resident | CPU / CPU |
| Qwen3.5+ MoE | not run | GPU: tower gate passed, served check passed (G-S6m, 2026-10-09) | not run | not run |
| Qwen3-VL (dense) | CPU / CPU | GPU with DeepStack (S10) / GPU, resident DeepStack prefill on (owner, 2026-10-09, over G-S10k's FAIL) | GPU with DeepStack / GPU, image prefill on the GPU (S16) | not run |
| GLM-OCR | CPU / CPU | GPU / GPU | GPU / GPU, batched image prefill (pairwise RoPE, 2026-10-09) | CPU / staged |
| Ministral 3 (Pixtral) | CPU / CPU | CPU / GPU (not run) | CPU / GPU, image prefill on the CPU then uploaded (G-S10m-d) | not run |
| EmbeddingGemma 2 (text, image, audio) | CPU | CPU | GPU | CPU |

**Gaps in coverage:**
- **Models:**
  - Qwen3-VL MoE images have never been run. Qwen3.5+ MoE images passed the served check on CUDA (G-S6m, 2026-10-09).
  - Gemma 4 E4B is validated on CUDA, not on Metal. Gemma 4 31B: CPU only, text agreement with Hugging Face read 2026-10-09 ((b'), below); images not checked against Hugging Face.
  - Ministral 3 (Pixtral) and LFM2.5-VL read images since 2026-10-09 (S10: G-S10m-a to d; G-S10l-a, b, d and c's reading, its planted defect 1 blind), their towers on the CPU in float32 on every backend; LFM2.5-VL's decoder is CPU only. North was dropped, never identified.
  - Video (S15) is not supported. Several images per message are, since S11 (2026-10-09).
- **Audio:**
  - Gemma 4 E2B audio into the model works on CPU and Metal.
  - Chat audio takes 16-bit WAV at any rate, mono or stereo (G-S5e). EmbeddingGemma 2's audio embeddings still take 16 kHz
    mono only.
  - E4B audio rides on S6.
  - G-S5b, the HF anchor for audio into the model, PASS.
  - Speech beyond Gemma 4 is S14: Qwen3-ASR's decoder runs (S14.3); Whisper and Voxtral are not started.
- **Precision:** under `--backend webgpu`, serve gives Gemma 3 an int8 SigLIP tower (no float32 WebGPU tower; S12). At
  real size that tower is relative L2 0.16-0.52 from float32, worst token 0.01-0.17
  (`docs/measurements/siglip-int8-fidelity-2026-10-07.md`). CUDA's default became float32 on 2026-10-08 (owner).
- **Release binaries:** WebGPU needs cgo and is in no release binary.

**Speed** (the most recent read of each, exploratory unless marked; S7 measures every cell against the 5 s bar; not
refreshed 2026-10-09. Gemma 3's Metal image turn has prefilled on the GPU since then, and G-IP4 reads its TTFT):
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

#### S8, plan and gates, registered 2026-10-09 before any code (owner: "G-IP4, then S11, then S8")

**Why the design changed from the description above.** Writing this morning's "Where it stands" table by hand put a
wrong cell in it within hours. It said Qwen3.5+ on Metal decodes an image turn on the GPU, but a recurrent family takes
neither the resident reuse path nor the upload bridge unless the backend implements `ResidentHybridMRoPEPrefill`, and
only CUDA does. So every cell the code can answer is computed from one declaration in code, and each backend module
proves its implementation matches that declaration.

**Steps:**
1. **Declarations, in one place each:**
   - **Towers:** a static table in `multimodal` of which tower families each backend registers. `metal`, `cuda` and
     `gpu` each get a test that its live registries equal its row: `Gemma4Towers`, `Qwen3Towers`, `GlmOcrTowers`,
     `AudioAccelerators`, `embeddinggemma2.Accelerators`, and a new goinfer-side name for the SigLIP and Qwen2.5-VL
     resident towers, whose aikit slots cannot be queried.
   - **The hybrid image-prefill capability:** declared beside the backends' resident features in `decoder`, with each
     backend's test asserting its `ResidentHybridMRoPEPrefill` implementation matches.
2. **The table:** `docs/multimodal.md`'s status block holds it between markers, one row per family: a `model_type`,
   a tiny fixture whose config resolves the architecture, and per backend `tower / decoder` from {GPU, CPU}. A notes
   column holds what the code cannot answer (validation status, speed), each with a date.
   - The README gets a short "Images and audio" section with the same cells.
3. **The drift test, in `decoder` (root, no GPU):**
   - Every row's decoder cell is GPU exactly when `ResidentEligible(arch, backend)` holds and, for a recurrent family,
     the backend declares the hybrid image prefill.
   - Every tower cell is GPU exactly when the backend declares that tower family; the CPU column is all CPU.
   - The README's cells equal the doc's.
   - In `internal/serveapp`: every row's `model_type` reaches a case of serve's vision dispatch (`loadVisionTower`).

**Gates:**
- **G-S8a:** the root drift test passes on the committed table.
- **G-S8b, planted defects** (each run on a mutated copy of the table, each must fail):
  - Qwen3.5+ on Metal marked GPU for decode, this morning's real error;
  - a WebGPU tower marked GPU for Gemma 4;
  - a README cell that differs from the doc;
  - a row whose `model_type` has no loader.
- **G-S8c:** the backend declaration tests pass:
  - `metal` on the Mac;
  - `cuda` and `gpu` on nobara.

  The same check is planted red once on Metal, a tower dropped from the declaration.
- **G-S8d:** `gate quick` green.

**Cost:** half a day by day.

#### S8 results (2026-10-09, by day)

- **Declarations:**
  - `multimodal.DeclaredTowers` and `LiveTowers` / `TowerDrift`.
  - `MarkResidentTower`, called beside aikit's unnamed SigLIP and Qwen2.5-VL slots on Metal, CUDA and WebGPU.
  - `decoder.HybridImagePrefillDeclared` (CUDA, dense hybrids).
  - `imageTurnDecodesResident`, the decoder cell. It shares Gemma 4's rule with `GenerateGemma4VLSpans`
    (`gemma4ImageDecodeResident`), so the two cannot drift.
- **The table:** generated into `docs/multimodal.md`, after the title, with the 2026-10-02 audit block below it kept as
  the record. The README has a new "Images and audio" section with the same cells, without notes.
- **G-S8a PASS:**
  - `TestSupportTable_fresh` (decoder).
  - `TestSupportTable_everyFamilyHasALoader` (serveapp, over `loadVisionTower`'s `mt ==` branches, gemma3 by fallback).
- **G-S8b PASS** (`TestSupportTable_plantedDefects` and the loader test's planted row):
  - Qwen3.5+ dense on Metal reads GPU / CPU, and no Metal hybrid prefill is declared. That is this morning's error,
    now computed.
  - Qwen3.5+ dense on CUDA reads GPU / GPU.
  - A Gemma 4 tower on WebGPU reads CPU.
  - A causal non-E Gemma 4 decodes on the CPU.
  - The README and doc rows agree.
  - A planted `mistral3` row is reported as having no loader.
- **G-S8c PASS:** `TestSupportTable_metalDeclarations` on the Mac (planted: a tower dropped from the declaration is
  reported), and `_cudaDeclarations` and `_webgpuDeclarations` on nobara. Each build's live towers equal its declared
  row; the hybrid image prefill is implemented exactly where declared.
- **G-S8d PASS:** `gate quick` green (2,261 passed, 0 failed); vet and staticcheck clean on every touched module.
- **What the computed table says that the hand-kept one did not:**
  - Qwen3.5+ MoE decodes an image turn on the CPU on CUDA too: CUDA's hybrid prefill excludes MoE hybrids.
  - Gemma 4 31B on WebGPU is CPU tower, GPU decoder.

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
- **S3 follow-ups, read 2026-10-09 by day: phase 2 PASS (the decoder's sensitivity, not a tower defect); G-S3b's
  CPU-tower repeat PASS.** The night run was VOID on memory (above). Re-pinned by the owner to `c2891555` and run by
  day; raw `s3-followup-2026-10-09/`.
  - **A disclosed deviation:** the repeat's serve binary is built at `c2891555`, not the registered `0c66b18b`. That
    older binary predates the directory sidecar, and its heap load was what the guard refused. Phase 2's test loads
    Gemma 3 through its sidecar (`gemma-3-4b-it.int4.metal.giw`, the file serve's Metal load reads).
  - **Phase 2: not VOID.** The reference path (CPU tower) reproduces the served reply exactly: "The image shows
    quarterly unit sales data by region, broken down into North, South, East, West, and Central, along with a total
    for all regions."
  - **The registered reading holds.** The Metal tower's arm sits inside the three noise arms:

    | arm | KL mean (nats) | KL max | first argmax change |
    |---|---|---|---|
    | Metal tower | 0.0563 | 0.489 | step 10 |
    | CPU tower + noise, seed 1 | 0.0661 | 0.484 | step 7 |
    | CPU tower + noise, seed 2 | 0.0307 | 0.323 | step 7 |
    | CPU tower + noise, seed 3 | 0.0462 | 0.558 | step 7 |

    **So G-S3b's red was the decoder's sensitivity to the tower's float32-order differences, not a tower defect.**
  - **G-S3b's CPU-tower repeat: PASS.** Both arms decoded `metal-resident (int4)` with a 30 s settle. Replies and
    log-probabilities are byte-identical (`cmp`).

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



- **The Mac night of 2026-10-08, graded 2026-10-09** (the runner started 22:16, ended 23:48; 6 of 10 jobs ok; raw logs in
  `~/goinfer-logs/night/runs/2026-10-08/`, archived into this campaign's measurements as noted per job).
  - **S9 step 2, the night speed rule: SHIP.**
    - **Text cell:** text-prompt TTFT 10.991 s (layer-major) against 1.565 s (batched), medians. The five per-pass
      ratios are 7.418 / 6.841 / 7.046 / 7.038 / 7.004, median **7.04x** against the 1.02 bar.
    - **Not void:** the batched arm's ran line is in all five server logs. Binaries: `79631cb7` (lm) and `90de0779`
      (`s9b-night-on`).
    - **The image cell, a record:** 10.223 s against 5.603 s, **1.82x**, under the 2.4-2.9x projection band. Both
      arms run the tower on the CPU (`-vision-device cpu`, about 4.5 s), which the band's 6.4 s baseline (S7's,
      tower on Metal) did not include. The prefill itself fell from about 5.7 s to about 1.1 s.
    - **So `emodelBatchedOn` goes on** (G-S9c and G-S9d passed by day).
  - **S7 and S13-lite on levers A and A+B, records against the 5 s bar** (serve-metal `92c30640` = A,
    `e2910316` = A+B; the 2026-10-07 night is the baseline):

    | cell | 10-07 | lever A | levers A+B |
    |---|---|---|---|
    | Gemma 3 4B (S7) | not loaded (fit guard) | not loaded (fit guard, at that minute) | 6.50 s |
    | Gemma 4 E2B, image | 6.20 s | 6.24 s | 6.17 s |
    | Gemma 4 E2B, audio | 3.20 s | 3.23 s | 3.14 s |
    | Qwen3.5-0.8B | 4.48 s | 4.03 s | 3.94 s |
    | Qwen3-VL-2B | 8.97 s | 7.74 s | 7.30 s |
    | GLM-OCR | 8.32 s | 6.40 s | 5.30 s |
    | Qwen2.5-VL-3B | 21.67 s | 20.50 s | 17.72 s |
    | **S13-lite, Gemma 3 4B: goinfer** | 12.81 s | 7.79 s | **6.60 s** |
    | S13-lite: Ollama / llama.cpp | 4.97 / 4.68 s | 4.97 / 4.70 s | 4.97 / 4.69 s |

    - The towers' levers show up where the tower dominates: GLM-OCR -36%, Qwen3-VL -19%, Gemma 3 -48% in S13-lite.
    - E2B's image cell does not move, because its time is the prefill (S9 step 2's, above).
    - Qwen2.5-VL is still the prefill: these binaries predate S16, whose own speed record below takes it to 5.73 s.
    - Two cells under the bar, as before.
  - **S16 (`s16-night`): the served gate PASS on both models; the real gate VOID (memory); speed a record.**
    - **G-S16c served:** today's path against S16, both Metal arms. **IDENTICAL replies** on Qwen2.5-VL-3B and on
      Qwen3-VL-2B.
    - **Speed** (image-turn TTFT, `vision_ttft.py`, 9 timed each, rotated):
      - Qwen2.5-VL-3B: 19.46 s today against **5.73 s** S16 (3.4x);
      - Qwen3-VL-2B: 8.25 s against **2.86 s** (2.9x, under the 5 s bar).
    - **G-S16c real: VOID, not FAIL.** Neither model went Metal-resident in the test process. Qwen2.5-VL's resident
      build needed 3.77 GB against a live budget of 2.80 GB, and Qwen3-VL's load saw 3.2 GB available. No
      comparison ran.
  - **VOID on memory** (the load-time fit guard, with 6.7-7.4 GB available all night):
    - **`s3-followup`:** at `1be0c274`, before directory sidecars. Phase 2's test loads Gemma 3 4B's directory in the
      heap ("needs ~5.6 GB ... + 0.6 GB KV", budget 4.8 GB), and the G-S3b repeat's servers were refused the same
      way.
    - **`metal-q4k-gq`** (`docs/tasks/task-metal-q4k-2026-10.md`): G-Q2's Phi-3 load at q4k needs 4.0 GB + 0.4 GB
      KV + 2.2 GB "reading the checkpoint", against 5.2 GB. G-Q3's serve was refused the same way, and its
      automatic streaming retry has no q4k form. The reference arm ran (Qwen2.5-Coder-1.5B, 94.22%, 7/8).
  - **`gs18g`: VOID by design error.** It served Gemma 3 4B from the ggml-org Q4_K_M GGUF, whose tokenizer has no
    `<image_soft_token>`. All three servers loaded Metal-resident, then exited at the vision setup ("tokenizer has no
    `<image_soft_token>` token"). The directory decoder, whose sidecar now exists, serves the same request: G-S18a's
    two arms ran that way by day.

- **By day 2026-10-09 (owner: "re-pin and run now", "run with latest code now").**
  - **A silent defect in S18 part 1, found and fixed: a Qwen-VL loaded through a sidecar lost its m-RoPE section.**
    - **The mechanism:** the qwen2_5_vl and qwen3_vl adapters read `mrope_section` from `rope_scaling` (where the
      released checkpoints carry it), clear `rope_scaling`, and keep the section in `Config.MRopeSection`, which was
      `json:"-"`. A `.giw` therefore dropped it, and a model loaded from one ran plain RoPE on image positions, with
      text unaffected (the three axes coincide on text).
    - **Live since part 1:** serve loads a Qwen-VL directory through its sidecar by default. GGUF-sourced Qwen-VL
      sidecars had the same loss before part 1.
    - **Why G-S18d missed it:** its text-only greedy streams cannot see image positions, and the tiny fixtures carry
      the section in `rope_parameters`, a raw field that survives. CLAUDE.md's "minimal in exactly the dimension that
      hides the bug".
    - **Found by** S16's real gate loading through the new sidecars: "not Metal-resident with an m-RoPE axis table"
      (`s16/real-2026-10-09/real-run1-sidecars-without-mrope.log`).
    - **The fix:**
      - `MRopeSection` is serialized (`json:"mrope_section,omitempty"`), and the two adapters take it from a `.giw`'s
        config.
      - A Qwen-VL config with no section is refused ("no m-RoPE section ... rebuild it"), so a stale sidecar fails
        its freshness self-check and rebuilds instead of being trusted.
      - `prequant.TestDirSidecar_keepsMRopeSection`: each Qwen-VL fixture is copied with the section moved into
        `rope_scaling` as released, and its m-RoPE axis table through the sidecar must equal the direct load's (red
        with the old tag); a config with no section must be refused.
  - **G-S16c real: PASS on both models, through the rebuilt sidecars** (`s16/real-2026-10-09/real.log`, 535 s), under
    the re-registered bar:

    | model | image | worst cosine | text control min | argmax |
    |---|---|---|---|---|
    | Qwen2.5-VL-3B | gemma3_preprocess_image | 0.913 | 0.335 | 0 real |
    | Qwen2.5-VL-3B | qwen25vl_preprocess_image | 0.950 | -0.022 | 0 real (3 near-ties) |
    | Qwen2.5-VL-3B | formula | 0.836 | -0.356 | 0 real (1 near-tie) |
    | Qwen2.5-VL-3B | table | 0.892 | 0.387 | 0 real |
    | Qwen3-VL-2B | gemma3_preprocess_image | 0.972 | 0.875 | 0 real (1 near-tie) |
    | Qwen3-VL-2B | qwen25vl_preprocess_image | 0.978 | 0.954 | 0 real (1 near-tie) |
    | Qwen3-VL-2B | formula | 0.985 | 0.958 | 0 real |
    | Qwen3-VL-2B | table | 0.986 | 0.866 | 0 real |

    - **But Qwen2.5-VL's text control is itself broken.** Its minimum cosine against the CPU falls to 0.33, -0.02 and
      -0.36 on same-length text prompts (about 1,000-1,600 rows): unrelated logits. The relative bar then passes
      vacuously.
    - **The plan:**
      - Qwen3-VL's PASS stands on healthy controls (0.87-0.95).
      - Qwen2.5-VL's long-prompt Metal prefill is a defect to find before S16 is claimed for it. It is not S16's own:
        the control is today's path.
      - S16 goes on for Qwen3-VL only, pending the owner.
  - **Qwen2.5-VL's control, investigated 2026-10-09 by day** (owner: "investigate now"; exploratory diagnostics, logs
    `s16/real-2026-10-09/q25vl-diag.log` and `q25vl-dec.log`, through the rebuilt sidecar):
    - **The last prompt row is fine.** CPU prefill against Metal's batched f16 pass reads cosine 0.95-0.99 on the
      control's random tokens and 0.82-0.99 on real text, 32 to 1,500 tokens. Metal's sequential path reads
      0.95-0.999. No non-finite values; max |logit| 12-28.
    - **The collapse is in the decode steps that read the batched pass's K/V.** The control's random tokens (seed 1),
      decoded 8 teacher-forced steps from four K/V sources, cosine against CPU-throughout:
      - the CPU prefill uploaded to Metal, the control's reference: 0.948-1.0;
      - Metal sequential: 0.82-0.998;
      - **Metal batched: steps at 0.70-0.75** (n = 512 step 7-8, n = 1024 step 1, n = 1500 step 5 and 8), each a
        different argmax; elsewhere 0.95-0.99.
      - The S16 run's -0.36 is the same effect over four seeds and up to 1,562 rows.
    - **The likely mechanism, not yet measured:** the batched pass rounds activations to f16 for its MMA GEMMs, and
      Qwen2.5's large activations lose precision there. That is the failure S17 lever B's f16-activation kernel hit on
      the towers, which f32 activations fixed. A precision defect of the batched text prefill on this model, live in
      production for long Qwen2.5-VL text prompts. Not S16's.
    - **The decision (owner, 2026-10-09):**
      - S16 is ON for Qwen3-VL (`metalDeepstackPrefillOn`: real PASS on healthy controls, served IDENTICAL, TTFT
        8.25 -> 2.86 s).
      - It stays OFF for Qwen2.5-VL (`metalMRoPEPrefillOn`): turning it on would move image turns from the CPU prefill
        (0.95-1.0) onto the lossy batched pass.
      - **Owed:** a task for Qwen2.5-VL's batched prefill precision (f32 activations, as lever B), with a decode-step gate
        against the CPU on the control.
    - **CORRECTION, 2026-10-09 (later the same day): the "likely mechanism" above is refuted, and so is "lossy".** Step 0
      of `docs/completed/task-metal-prefill-precision-2026-10.md` measured the batched pass as the MOST precise prefill:
      layer 0's V against an f64 truth on the same int4 weights is 0.00045 against 0.0245 for the CPU prefill and the
      sequential path. Both of those run W4A8, one int8 scale per activation row. The control's reference was the
      less precise arm. The collapses come from W4A8 on this model, and the CPU-throughout arm shows them too (n =
      1,562 seed 1 step 4: -0.41 against the near-truth, while the batched arm holds 0.94). The reason S16 stays off for
      Qwen2.5-VL is therefore void. Re-grading it needs a control whose reference is not W4A8, which is the owner's
      call (that task doc, "What follows").
    - **S16 ON for Qwen2.5-VL, 2026-10-09 (owner: "switch on").** Before the flip, the image turns themselves were graded
      against two references that are not W4A8 (exploratory, by day; S16's four images, last row plus 8 teacher-forced
      steps, 36 steps; `docs/measurements/metal-prefill-precision-2026-10/s16-q25vl/`):

      | arm | mean vs weight-only int8 (E) | worst | mean vs per-32 int4 (F) | worst |
      |---|---|---|---|---|
      | S16 off: CPU W4A8 image prefill, uploaded | 0.9485 | 0.841 | 0.977 | 0.855 |
      | S16 on: Metal m-RoPE batched prefill | 0.9480 | 0.823 | 0.990 | 0.956 |

      - The two arms are even against the near-truth, and S16 on is closer to the same weights. The worst step is the
        same one in both arms (`qwen25vl_preprocess_image.png`, step 7).
      - Reach: the switch serves no-DeepStack turns of a Metal resident with an m-RoPE axis table. That is Qwen2.5-VL
        only:
        - Qwen3.5+ is recurrent, and Metal does not claim `ResidentHybridMRoPEPrefill`;
        - GLM-OCR runs on the CPU on Metal (pairwise m-RoPE).
      - `gate quick` was green on the flip.

### S10 — Towers for the families that have none

Added 2026-10-07: Ministral 3 (Pixtral), LFM2.5-VL, North, and Qwen3-VL's image path (today text only).

**2026-10-09: "North" is dropped** (owner: "i have no idea what north meant"). No checkpoint or repo was ever recorded
for it; re-add it with one if it resurfaces. Order from here (owner, 2026-10-09: "on to s10s families, s12 and s15, and
then full s13"): Ministral 3 (Pixtral), whose 3B is on nobara; then LFM2.5-VL; then Qwen3-VL MoE. Qwen3-VL MoE's 30B
cannot hold a float32 HF reference in nobara's 62 GB, so it would need the 31B's layer-streaming reference (step (b')).

- **Pattern:** the Gemma 4 and Qwen ones: the tower in aikit (batched into the cycle's one release), the decoder splice
  and prompt layout in goinfer, then serve. One sub-phase per family, in order of what users ask for; each starts with
  a desk map of its HF processor (image budget, special tokens, position scheme).
- **Gates, per family, before starting it:** the tower's stages against HF on the real checkpoint; full-model logits on an
  image prompt against HF; a served request.
- **Size:** M each.

**S10, Ministral 3 (Pixtral): desk map, plan and gates, registered 2026-10-09 before any code** (owner: "on to s10s
families"). Checkpoint `mistralai/Ministral-3-3B-Instruct-2512` (bf16, `~/models/ministral3-3b-bf16` on nobara).

**Gate 0, the desk map** (transformers 5.15.0's `pixtral` and `mistral3` sources and the real checkpoint, read 2026-10-09):
- **Preprocessing** (`PixtralImageProcessor`):
  - Scale down so the longest side is at most 1540, then round each side UP to a multiple of 28 (patch 14 x merge 2).
  - Bicubic with antialias on uint8, then CLIP mean/std.
  - Measured with the real processor: a 1024x768 image gives a 784x1036 tensor.
- **Token layout:** one merged 2x2 unit per `[IMG]` (id 10), each merged row ending in `[IMG_BREAK]` (12), the last
  `[IMG_END]` (13). The 1024x768 example is 1,036 `[IMG]`, 27 `[IMG_BREAK]`, 1 `[IMG_END]`.
- **The tower** (`PixtralVisionModel`, 24 layers, hidden 1024, 16 heads, head dim 64):
  - Patch conv with no bias, then RMSNorm `ln_pre` (eps 1e-5).
  - 2-D rotate-half RoPE (theta 10000, position row*110+col). Rows take the even-indexed frequencies, columns the
    odd-indexed ones.
  - Pre-norm blocks: separate q/k/v/o with no bias, then a SiLU-gated MLP. No final norm.
  - Several images run as one sequence with a block-diagonal mask.
- **Projector:** RMSNorm, then a 2x2 patch merger (channel-major unfold over the row-major grid, a 4096->1024 linear),
  then linear 1024->3072, exact GELU, linear 3072->3072. No biases.
- **Decoder:**
  - Features replace the `[IMG]` positions only; `[IMG_BREAK]` and `[IMG_END]` are ordinary text.
  - Plain causal attention and plain 1-D positions: no bidirectional block, no m-RoPE.
  - goinfer's `ministral3` text decoder exists, is resident on Metal and CUDA, and has a real-checkpoint gate.
- **Chat:**
  - The checkpoint ships no chat template. The HF one (fetched and read) puts a bare `[IMG]` per image inline inside
    `[INST]`, in content order, and moves a [text, image] message's image first.
  - Its long default system prompt is deliberately not replicated (`chat/templates.go`, Ministral()), so the
    reference harnesses send an explicit system message and goinfer's own token ids.
- **Today:** serve has no `mistral3` loader. An image request is a 400 "no vision tower", and the support table's
  loader check plants exactly this row as its defect.

**Plan:**
1. **aikit** (local branch `s10-pixtral`, off `origin/main`, pushed and tagged by the owner): `PixtralVisionEncoder`
   (loader, conv, the RoPE table, blocks reusing the Qwen tower's attention and MLP helpers, multi-image segments) and
   `PixtralPreprocess` (the resize rule on `ResizeBicubicAA`, normalise, the grid), with a tiny-tower test.
2. **goinfer** (worktree branch `s10-pixtral`, merged after the aikit release):
   - the projector (`multimodal`);
   - the prompt block (rows of `[IMG]` with breaks and the end token);
   - a CAUSAL span entry, one span per merged row, splicing only the `[IMG]` positions, with no image block. Its
     resident path is the plain batched prefill over the spliced rows.
   - serve's `mistral3` branch;
   - the support-table row.

**Gates:**
- **G-S10m-a, preprocessing:** against HF's `PixtralImageProcessor` (`~/.venv-vl`, torchvision 0.27) on the four
  images. Sizes equal, pixel values within rounding, token counts equal. Also the layout edge cases: a single merged
  row (no break), a 3000x2000 image (downscaled), two images.
- **G-S10m-b, the tower and projector against HF on the real 3B:**
  - Every stage (conv, ln_pre, each block, the projector's norm, merger, linear_1, linear_2) at soft-token cosine
    >= 0.9999 on the four images. HF's tower runs alone in float32 from its own pixel values; 0.999-0.9999 is ambiguous
    (parked).
  - A two-image run must equal each image's single-image output.
  - aikit's tiny tower with norms randomised, planted defects each red:
    1. column frequencies from the even list;
    2. row and column halves swapped;
    3. the block mask dropped (two images);
    4. the merger in position-major order.
- **G-S10m-c, the full model on an image prompt against HF `Mistral3ForConditionalGeneration`:**
  - Float32, nobara, an explicit system message, goinfer's ids. G-S5b's bar: last-position cosine >= 0.999, argmax
    equal, argmax agreement >= 95% over the text after the image.
  - Planted defects, each red:
    1. features also written over `[IMG_BREAK]`/`[IMG_END]`;
    2. a bidirectional image block;
    3. the rows shifted by one.
- **G-S10m-d, served:** one-image and two-image requests, `--backend cpu` against `--backend metal` with equal load
  flags, plus a CPU repeat. Identical replies, or a first divergence at an R10 near-tie.
- **Tier:** b and c hold about 17 GB in float32 on nobara. Each runs by day only if its cell estimate is under 10
  minutes, otherwise on the night queue.

**Size:** aikit about 500 lines, goinfer about 800, both with tests: M, as registered for S10.

**S10 Pixtral progress (2026-10-09, Mac and nobara): G-S10m-a, b, c and d PASS. S10 Ministral 3 is DONE.**
- **The code:**
  - aikit, local branch `s10-pixtral` (96259e5, 419c9b3; not pushed): `PixtralVisionEncoder` and `PixtralPatchify`.
    The tiny tower's weights now ship, un-ignored like the other tiny towers' (a fresh checkout on nobara failed both
    tiny tests without them).
  - goinfer, worktree branch `s10-pixtral`:
    - `multimodal.PixtralProjector` (with `ForwardStages`), `PixtralPreprocess` and `PixtralImageBlock`;
    - `decoder.GenerateVLCausalSpans`: one span per merged row, only `[IMG]` spliced, no image block. Its resident
      path is the backend's plain batched prefill (`Prefiller.PrefillLast`) over the spliced rows;
    - serve's `mistral3` tower, CPU float32, with one span per merged row (`imageSpans`);
    - the support-table row.
- **A real defect found by the ids step:** `chat.Ministral()` rendered the whole prompt as one Special segment, so the
  vision splice never found the image block and refused every Pixtral request. It also let a typed `[/INST]` become a
  control token. It now renders its six markers as control tokens and the texts as content (M25). All six are single
  special tokens of the 3B's tokenizer (ids 1, 2, 3, 4, 17, 18). `TestRenderSegments_ministral` on that tokenizer
  (nobara): byte identity with the whole-string encode on legitimate turns, and a forged `[/INST]` stays literal.
- **A reference mistake, caught by the reference's own check:** the first pin called `PixtralImageProcessorFast`
  alone, which rounds to the patch (14). `PixtralProcessor` passes it patch x merge (28). `table.png` came out 910 high
  instead of 924, and the script stopped at "1419 [IMG] in goinfer's ids, the tower gives 1376 rows". The references
  now come from the full processor. goinfer's preprocessing was right throughout.
- **G-S10m-a, preprocessing: PASS.** transformers 5.12.0 (`~/.venv-vl`), the four F2a images and the two layout cases
  (a 1000x20 strip, one merged row with no break; a 3000x2000 image, downscaled to 1540x1036). Every size equal. The
  pixels are bit-identical on all six (max |diff| 0, the mean of order 1e-7 being float rounding of the normalisation).
  Every [IMG], [IMG_BREAK] and [IMG_END] count equal.
- **G-S10m-b, tower and projector: PASS.** From HF's own pixels, every stage (patch conv, ln_pre, 24 blocks, the
  projector's norm, merger, linear_1, linear_2), worst row cosine:

  | image | patches | worst stage |
  |---|---|---|
  | gemma3_preprocess_image.png | 64x64 | 0.999999994 (block 13) |
  | qwen25vl_preprocess_image.png | 4x6 | 0.999999999 (the merger) |
  | glm_ocr/formula.png | 86x72 | 0.999999999 (block 17) |
  | glm_ocr/table.png | 66x86 | 0.999999998 (block 13) |

  Two images in one call (table, then the 896x896): each bit-identical to itself alone, and 0.999999999 against HF's
  own two-image call. The planted defects are the tiny tests' (aikit's three, goinfer's position-major merger at
  0.809).
- **G-S10m-c, the full model: PASS.** `table.png`, an explicit system message, "What does this table show? Answer in
  one sentence.": 1,474 ids, a 33x43 merged grid (1,419 [IMG]), 13 text positions after the image. goinfer float32
  CPU, its own preprocessing end to end: last-position cosine 1.000000, argmax 1784 equal, agreement 13/13. The
  production entry's last logits: cosine 1.000000 against HF.

  | planted defect | last cosine | argmax | agreement | verdict |
  |---|---|---|---|---|
  | features also over [IMG_BREAK]/[IMG_END] | 0.952522 | equal | 8/13 | red |
  | a bidirectional block per row | 0.996371 | equal | 13/13 | red, on the cosine bar only |
  | rows shifted by one position | 0.738214 | equal | 8/13 | red |

  Defect 2 is red by the cosine bar alone: per-row bidirectionality moves the last logits but not the greedy reading
  of this prompt.
- **Wall time on nobara:** the references 3 min (one float32 load), a and b 2.7 min, c 4.5 min. Started 16:29:27 PDT,
  done 16:39:46.
- **Raw:** `docs/measurements/multimodal-support-2026-10/s10m-pixtral/` (the driver, every step's log, the two refused
  runs, and the reference summary without its float arrays).
- **Released and merged:** aikit v1.63.0 (the Pixtral tower; tagged on the Mac 2026-10-09 19:05 PDT, releasegate 5/5,
  vulncheck 16/16 clean at 0f5a008 on nobara, perfgate exception: no `linalg/`/`mmap/` change), and goinfer `main` at
  62009e1e on it. The parity manifest refresh for the `forwardn.go` edit is 443e0d05 (43 forward goldens green, 0 failed). The
  site's Ollama snapshot moved mistral-small3.2 from T to S (`docs/measurements/ollama-coverage-2026-10-09.md`, f31e1435).
- **A third defect, found by G-S10m-d's first arm:** the checkpoint ships no chat template, and `chat.Detect`'s vocab
  fallback had no Ministral case. Serve refused every image request ("this model has no chat template for vision"), and
  text from this directory fell back to raw completion. `Detect` now takes `[SYSTEM_PROMPT]` with `[INST]` as control tokens
  for `Ministral()`. Mistral v0.3's `[INST]`-only vocab is not taken for it (`TestDetect_fallback`).
- **G-S10m-d, served: PASS.** Read 2026-10-09 19:42-19:49 PDT on the Mac. The Metal serve binary was built from `main` plus
  the template fix. `--model` and `--vision` were `~/models/ministral3-3b-bf16` (the HF files copied from nobara, byte
  counts checked), with `--embed-int4=false` on every arm. An explicit system message, 48 greedy tokens. Arms `=cpu`,
  `metal`, then `cpu` again.

  | request | prompt tokens | Metal against CPU | second CPU run against the first |
  |---|---|---|---|
  | one image (`table.png`) | 1,474 | first differs at token 2: CPU " displays" 0.253, Metal's " presents" 0.247 there, a near-tie | identical |
  | two images (`table.png`, then the 896x896) | 2,527 | first differs at token 25: CPU `)"` 0.420, Metal's `),"` 0.396 there, a near-tie | identical |

  - The Metal arm decoded `metal-resident (int4)`. Its resident image prefill declined ("prefill not implemented for this
    arch's FFN shape"), so the prefill ran on the CPU and was uploaded. The support table's Metal cell (CPU tower, GPU
    decode) says what ran, and its note now says where the prefill runs.
  - The one-image prompt is exactly G-S10m-c's 1,474 ids, so the served path built the same prompt the reference graded.
  - Request times were 55.6-69.9 s, the CPU float32 tower being most of it (exploratory, one reading each).
  - Two refused runs before the reading are kept in `gs10m-d/refused/`. The first is the template defect above. The second
    found the port held by the first run's server, which the driver had leaked on the failed request; it now kills its
    server on exit.
- **Raw:** `docs/measurements/multimodal-support-2026-10/s10m-pixtral/gs10m-d/` (the run, each arm's serve log, replies and
  log-probabilities).
- **Owed, none of them gates:**
  - A Metal resident prefill for this FFN shape (the image turn's prefill is on the CPU today).
  - A GPU Pixtral tower (none declared on any backend).
  - Mistral Small 3.x's 24B: the same `PixtralVisionModel` architecture, never run.

**S10, LFM2.5-VL: desk map, plan and gates, registered 2026-10-09 before any code** (owner: "on to s10s families", then
"continue"). Checkpoint `LiquidAI/LFM2.5-VL-1.6B` (bf16, `~/models/lfm25-vl-1.6b` on nobara; the LFM Open License v1.0,
`license: other` on the hub). The 450M (a smaller SigLIP2) and the 3B are the same `lfm2_vl` architecture and are not part of
these gates.

**Gate 0, the desk map** (transformers 5.12.0's `lfm2_vl` and `siglip2` sources on nobara, and the checkpoint's own
`config.json`, `processor_config.json` and `chat_template.jinja`, read 2026-10-09):
- **Preprocessing** (`Lfm2VlImageProcessorFast`, through the full `Lfm2VlProcessor`):
  - Bilinear resize with antialias on uint8 (`resample` 2), then (x/255 - 0.5)/0.5.
  - An image is "too large" when its size, rounded to multiples of 32, exceeds 256 tokens x 32² x 2.0 (the pixel
    tolerance). A too-large image is resized to a grid of 512x512 tiles. The grid is the (w, h) pair with 2 <= w·h <= 10
    whose aspect is closest to the image's; on a tie, the later pair wins when the image covers more than half its area.
    The tiles are followed by a thumbnail.
  - Otherwise, or as the thumbnail, it is smart-resized: each side rounded to a multiple of 32, then scaled so the token
    count lies between 64 and 256 (floor on shrink, ceil on growth).
  - Each tile or thumbnail is patchified into 16x16 patches, each flattened (row, column, channel), channels innermost,
    and runs through the tower on its own (the padding to 1,024 patches is masked).
- **Token layout:** `<|image_start|>`, then for a multi-tile image each tile in row-major order as `<|img_row_R_col_C|>`
  followed by its 256 `<image>` (id 396), then `<|img_thumbnail|>` and the thumbnail's tokens, then `<|image_end|>`. A
  single-tile image is `<|image_start|>`, its tokens, `<|image_end|>`. The features replace the `<image>` positions only.
- **The tower** (`Siglip2VisionModel`, NaFlex, 27 layers, hidden 1152, 16 heads, no head):
  - A linear patch embedding over the 768-value patch vector.
  - A learned 16x16 position table, resized to each tile's patch grid with `F.interpolate(bilinear, align_corners=False,
    antialias=True)`.
  - SigLIP's pre-norm blocks (LayerNorm eps 1e-6, biased q/k/v/o, GELU-tanh MLP), then `post_layernorm`. The output is
    `last_hidden_state`.
- **Projector:**
  - A 2x2 pixel unshuffle over each tile's patch grid. Unit (r, c) takes patches (2r+j, 2c+k) laid out
    `[(2j+k)·1152 + channel]`: position-major, the opposite of Pixtral's merger.
  - No LayerNorm, then linear 4608->2048 (bias), GELU (erf), linear 2048->2048 (bias).
- **Decoder:**
  - `lfm2` (LFM2-1.2B's shape: 16 layers, 10 short-conv and 6 attention).
  - Plain 1-D positions and causal attention; the image rows also run through the conv layers in order.
  - goinfer's `lfm2` is CPU only (no backend implements its short convolution) and runs one token at a time
    (`runLayersLFM2(id)`). It has no entry that takes an embedding.
- **Chat:** the checkpoint's template is ChatML after a BOS. An image part is `<image>` in content order, with no newline
  around it.
- **Today:** serve has no `lfm2_vl` loader. aikit has no SigLIP2 NaFlex tower and no antialiased bilinear resize.

**Plan:**
1. **aikit** (a local branch `s10-lfm2vl`, released when done):
   - `ResizeBilinearAA`: the existing antialiased uint8 resize with a triangle filter.
   - `Siglip2NaFlexEncoder`: the linear patch embedding, the antialiased position-table resize in float32, SigLIP's
     blocks and `post_layernorm`, one tile per call.
   - A tiny-tower test against HF.
2. **goinfer** (worktree branch `s10-lfm2vl`):
   - The preprocessing (`multimodal`): too-large, the grid, smart resize, tiles plus thumbnail, patchify.
   - The projector and the prompt block.
   - `runLayersLFM2FromEmbed` and a per-token causal image prefill for `lfm2` behind the causal span entry, with one span
     per tile and thumbnail. It is CPU only, so no resident path.
   - Serve's `lfm2_vl` branch and the support-table row.

**Gates:**
- **G-S10l-a, preprocessing:** against the full HF processor (`~/.venv-vl`) on the four F2a images and three layout
  cases:
  - a 1000x20 strip (one tile, smart resize);
  - a 3000x2000 image (the grid search);
  - 84x56 (upscaled to the 64-token floor).
  - Grid, tile count, thumbnail size and token counts equal; pixels within one 8-bit level (0.0078 after normalising).
    torchvision's uint8 bilinear is fixed point, so exactness is the target and one level the bar.
- **G-S10l-b, the tower and projector on the real 1.6B:**
  - Every stage (patch embedding plus positions, each block, `post_layernorm`, the unshuffle, linear_1, linear_2) at
    worst row cosine >= 0.9999, on every tile and thumbnail of the four images. HF's tower runs alone in float32 from its
    own pixels; 0.999-0.9999 is ambiguous (parked).
  - aikit's tiny tower (norms randomised, a position table that both shrinks and grows) and goinfer's tiny projector,
    with planted defects each red:
    1. the position table resized without antialias;
    2. patches flattened channel-major;
    3. `post_layernorm` dropped;
    4. the unshuffle channel-major (Pixtral's order).
- **G-S10l-c, the full model on an image prompt against HF `Lfm2VlForConditionalGeneration`:**
  - Float32, nobara, an explicit system message, goinfer's ids, a multi-tile image. G-S5b's bar: last-position cosine
    >= 0.999, argmax equal, argmax agreement >= 95% over the text after the image.
  - Planted defects, each red:
    1. the tiles' features in column-major order;
    2. the thumbnail's features placed first;
    3. features written over the `<|img_row_R_col_C|>` markers.
- **G-S10l-d, served:**
  - One-image and two-image requests through serve on the Mac (`--backend cpu`) and a second CPU run: identical replies.
  - The one-image prompt's ids equal G-S10l-c's.
  - A `--backend metal` arm must decline the image turn to the CPU path by name, not fail, since `lfm2` has no GPU
    decoder.
- **Tier:** b and c hold about 8 GB in float32 on nobara, each well under 10 minutes, so by day.

**Size:** aikit about 400 lines, goinfer about 900, both with tests: M, as registered for S10.

**S10 LFM2.5-VL progress (2026-10-09, Mac and nobara): G-S10l-a, b and d PASS; c PASS on its bar with one of its three
planted defects BLIND, the cover accepted by the owner. S10 LFM2.5-VL is DONE.**
- **The code:**
  - aikit, local branch `s10-lfm2vl` (6f73bb5; not pushed):
    - `Siglip2NaFlexEncoder` and `PatchifyNaFlex`: SigLIP's blocks through a `runBlocks(h, np)` split out of
      `forwardBlocks`, so Gemma 3's tower is unchanged;
    - `ResizeBilinearAA` (uint8) and `ResizeBilinearAAFloat`.
    - Against HF on a tiny random tower (three tiles that grow, shrink and keep the position table): every stage at
      cosine 1.000000000, planted defects red. Without antialias the shrinking tile reads 0.999581; the control, the
      growing tile, agrees. Channel-major patches are red, and so is a dropped post-layernorm. The uint8 resize is
      bit-identical to torchvision on four sizes.
  - goinfer, worktree branch `s10-lfm2vl`:
    - preprocessing, the projector and the prompt block (`multimodal`);
    - `lfm2_vl` registered as the `lfm2` architecture, with `runLayersLFM2FromEmbed` and a per-token causal image
      prefill;
    - serve's tower, with one span per tile and thumbnail;
    - the support-table row.
  - The tiny tests:
    - the layout against HF's processor on 21 sizes. Planted: dropping the area rule changes 3 layouts.
    - the projector at 7.15e-7. Planted: the channel-major unshuffle reads 3.13.
    - the lfm2 image prefill on `lfm2-tiny`. With the text's own rows spliced in it is bit-identical to the token
      forward.
- **The ratio order is not a defect site.** HF sorts `_target_ratios` from a Python set, and its order within one tile
  count looked like an exactness trap. It cannot decide a layout: two grids of one count tie only at an aspect midway
  between theirs, and the square grid of that range is always nearer. The planted "plainly sorted" defect was blind for
  that reason. It was replaced by the rule that does decide ties, the area test: 1200x1000 and 1000x1250 are exact ties
  that it resolves to 3x3.
- **Three defects found on the way, all fixed:**
  1. **No BOS.** goinfer's ChatML rendered no BOS, while LFM2's template opens with `bos_token` and serve encodes with
     `addBOS=false`, so every LFM2 chat prompt (the text family's too) lacked HF's `<|startoftext|>`. A ChatML template
     that opens with `bos_token` now renders it when the vocab has that token (`TestDetect_chatMLBOS`).
  2. **The tokenizer would not load.** The VL checkpoint's tokenizer.json wraps its ByteLevel decoder in a `Sequence`,
     which routed it to the SentencePiece path ("required token `<unk>`", G-S10l-c's ids step). A Sequence of ByteLevel
     decoders now loads as byte-level.
  3. **The FFN width.** LFM2-1.2B's text_config sets `block_auto_adjust_ff_dim`, and HF's `Lfm2MLP` then derives the
     width: 12288 becomes 8192. goinfer read the stated width, and the checkpoint's shapes refused the load (G-S10l-c's
     first compare). The rule is ported exactly, including HF's rounding sitting inside the multiplier branch
     (`TestLfm2FFNDim`).
  - With all three fixed, goinfer's 1,810 prompt ids equal those from HF's own chat template and processor.
- **G-S10l-a, preprocessing: PASS.** transformers 5.12.0, `~/.venv-vl`. The four F2a images and three layout cases
  (a 1000x20 strip, a 3000x2000 image, 84x56 upscaled to the 64-token floor) give every tile, grid and run equal and
  every pixel bit-identical (0 levels). The tiles: 896x896 is 2x2 plus thumbnail; formula.png 3x3 plus thumbnail;
  table.png 2x3 plus thumbnail; the rest single.
- **G-S10l-b, the tower and projector: PASS.** From HF's own patches, every stage of every tile (embeddings, 27 blocks,
  post_layernorm, the unshuffle, linear_1, linear_2):

  | image | tiles | worst stage |
  |---|---|---|
  | gemma3_preprocess_image.png | 5 | 0.999999950 (tile 3, block 25) |
  | qwen25vl_preprocess_image.png | 1 | 0.999999995 (post_layernorm) |
  | glm_ocr/formula.png | 10 | 0.999999985 (tile 7, block 25) |
  | glm_ocr/table.png | 7 | 0.999999973 (tile 6, block 25) |

- **G-S10l-c, the full model: the reading PASSES; planted defect 1 is BLIND.** `table.png` (2x3 tiles plus a thumbnail,
  1,810 ids, 17 text positions after the image). goinfer float32 CPU, end to end: last-position cosine 1.000000, argmax
  7491 equal, agreement 17/17. The production entry also reads 1.000000.

  | planted defect | last cosine | argmax | agreement | verdict |
  |---|---|---|---|---|
  | the tiles' features in column-major order | 0.999541 | equal | 17/17 | **BLIND** |
  | the thumbnail's features first | 0.997643 | equal | 13/17 | red |
  | features over the tile markers | 0.990344 | equal | 16/17 | red |

  - As registered, each planted defect must be red, and defect 1 is not. This prompt's answer does not depend on which
    tile's features sit in which tile's slot, beyond 0.9995. The bar is not moved.
  - What does carry tile order:
    - G-S10l-a checks tile k's pixels against HF's tile k (bit-identical);
    - the layout golden checks the markers' row-major order;
    - goinfer concatenates the features in that same layout order (`lfm2vlPrep`).
  - A per-position check over the image span would see the swap. That is an amendment for the owner, not something to
    add after the reading.
  - **Owner decision, 2026-10-09 21:30 PDT: accept the cover.** G-S10l-c stands as PASS with planted defect 1 BLIND,
    carried by G-S10l-a's per-tile pixel identity and the layout golden (the Qwen2.5-VL two-image BLIND precedent).
    No amendment, no re-run.
- **G-S10l-d, served: PASS.** Read 2026-10-09 20:59-21:17 PDT on the Mac. The Metal serve binary was built from the
  branch, with `--model`/`--vision ~/models/lfm25-vl-1.6b` (copied from nobara, byte counts checked) and
  `--embed-int4=false` on every arm. Arms `=cpu`, `cpu`, `metal`.
  - Both requests give identical replies on every arm: "Quarterly unit sales by region." for one image; for two, "The
    first image is a table of quarterly unit sales by region in thousands, while the second image is a gradient of
    colors ranging from black to pink."
  - The one-image prompt is 1,810 tokens, G-S10l-c's count.
  - The Metal arm declined by name ("arch is not eligible for the resident decode runner, and metal has no staged decode
    path") and ran on the CPU, as registered.
  - Request times were 121-237 s, exploratory: the lfm2 image prefill is per token on the CPU.
- **Tier, a miss to record:** the registration said b and c were each well under 10 minutes. b took 1.2 min, but c took
  30 min. Each 1,810-token float32 per-token prefill is about 6 minutes on nobara (0.19 s a token), and c runs five of
  them (the reading, the production entry, three planted).
- **Raw:** `docs/measurements/multimodal-support-2026-10/s10l-lfm2vl/` (the drivers, each step's log including the two
  refused runs, the reference summary, and G-S10l-d's run in `gs10l-d/`).
- **Released and merged:** aikit v1.64.0 (tagged on the Mac 2026-10-09 21:38 PDT: releasegate 5/5, preflight 10/10,
  vulncheck 16/16 clean at c605ba6 on nobara, perfgate exception, root CI green at 302bca2; backends re-pinned), and
  goinfer on it.
- **Owed:**
  - A batched lfm2 prefill: the per-token image prefill is the slow part of a served image turn.
  - The 450M and 3B sizes, never run.

**S10, Qwen3-VL MoE: desk map, plan and gates, registered 2026-10-09 before any code** (owner: "continue", the next family
in the S10 order). Checkpoint `Qwen/Qwen3-VL-30B-A3B-Instruct` (Apache-2.0, bf16, 13 shards, about 62 GB,
`~/models/qwen3-vl-30b-a3b-instruct` on nobara). There is no smaller Qwen3-VL MoE; the 235B is out of reach.

**Gate 0, the desk map** (transformers 5.12.0's `qwen3_vl_moe`, `qwen3_vl` and `qwen3_moe` sources, and the checkpoint's
`config.json`, index and one shard's header, read 2026-10-09):
- **The decoder** (`Qwen3VLMoeTextModel`, 48 layers, hidden 2048, 32 heads, 4 KV heads, head dim 128):
  - Qwen3's attention (per-head RMSNorm QK-norm, no bias) with Qwen3-VL's interleaved m-RoPE (section [24, 20, 20],
    theta 5e6).
  - A sparse MoE on every layer: 128 experts, top 8, intermediate 768, no shared expert. The router is softmax over all
    128, then top-8, then renormalised (transformers 5.12 always renormalises; `norm_topk_prob` is true anyway).
  - DeepStack: three sets, added to the hidden state at the image positions after decoder layers 0, 1 and 2, as in the
    dense Qwen3-VL.
  - An untied head.
- **The experts on disk are in transformers 4.57's layout:** `mlp.experts.gate_up_proj` `[128, 2048, 1536]` (expert,
  hidden, gate then up) and `mlp.experts.down_proj` `[128, 768, 2048]`. transformers 5.12's module holds the transpose
  (`[128, 1536, 2048]`) and converts on load.
  - **goinfer's `loadFusedExperts` (Qwen3.5-MoE's) checks element counts, not shapes, and the two layouts have equal
    counts.** Loading this checkpoint through it as it stands would give transposed experts with no error. The loader
    must read the shape and transpose this layout.
- **The tower** (`Qwen3VLMoeVisionModel`): the Qwen3-VL tower aikit already runs (`Qwen3VisionEncoder` with DeepStack,
  S10), here 27 blocks, hidden 1152, 16 heads, patch 16, a 48x48 position table, out 2048, DeepStack taps at blocks 8,
  16 and 24, under `model.visual.`.
- **Preprocessing and prompt:** Qwen3-VL's (`Qwen2VLImageProcessor` settings from the checkpoint, the
  `<|vision_start|>` + `<|image_pad|>`… + `<|vision_end|>` block), the same as the dense 2B's.
- **Today:**
  - goinfer has no `qwen3_vl_moe` decoder; `qwen3_moe` (attention, experts, router) and `qwen3_vl` (m-RoPE) exist
    separately.
  - Serve has no `qwen3_vl_moe` tower branch.
  - The support-table loader test plants `qwen3_vl_moe` as its row with no loader.
- **Reference:** a float32 Hugging Face forward of the 30B is about 120 GB, beyond nobara's 62 GB. goinfer cannot run it
  in float32 either. So the full-model gate is step (b')'s design: HF's own decoder-layer modules run layer-major, one
  layer's weights at a time.

**Plan:**
1. **goinfer:**
   - A `qwen3_vl_moe` architecture: `qwen3_moe`'s attention, experts and router, with `qwen3_vl`'s m-RoPE.
   - `loadFusedExperts` reads the shape: the Qwen3.5 layout as now, the 4.57 layout transposed, anything else refused.
   - The image path through the existing DeepStack entry (`GenerateQwenVLDeepstack`).
   - Serve's `qwen3_vl_moe` branch on the Qwen3-VL tower path, and the support-table row.
2. **aikit:** the tower is expected to load as it is. If its loader refuses the `qwen3_vl_moe` vision config or the
   tensor prefix, the change goes on a branch for a release.
3. **The streaming reference:** `g31b_hf_stream.py`'s design for Qwen3-VL MoE text layers with the image features and
   DeepStack spliced in.

**Gates:**
- **G-S10q-a, the decoder against HF on a tiny random `Qwen3VLMoeForConditionalGeneration`** (text config at toy size;
  the experts written to disk in the released 4.57 layout). Two inputs:
  - a text prompt;
  - an image prompt with random merged rows and three DeepStack sets, through `prefillLogitsQwenVLSpans`'s entry.
  - Every position at cosine >= 0.99999.
  - Planted defects, each red:
    1. the experts loaded without the transpose (the element-count path);
    2. the router's softmax taken after the top-8;
    3. the m-RoPE not interleaved;
    4. each DeepStack set added one layer late.
- **G-S10q-b, the tower on the real 30B:** HF's `Qwen3VLMoeVisionModel` alone, in float32, from its own pixel values, on
  the four F2a images. Every stage (the embedding with positions, each block, the main merger, each DeepStack set) at
  worst-token cosine >= 0.9999; 0.999-0.9999 is ambiguous (parked).
- **G-S10q-c, the full model against the layer-streaming HF float32 reference** (step (b')'s design):
  - **The sequences:** 8 prompts (the four F2a images x two questions, an explicit system message, the checkpoint's
    template), each followed by goinfer's own int8int8 greedy continuation of 32 tokens, recorded once and pinned:
    256 teacher-forced positions.
  - **Arms:** goinfer CPU at int8int8 (the primary) and int4 (serve's default), the image features from goinfer's tower.
  - **Statistic:** per arm, argmax agreement with HF over the 256 positions, and mean KL(HF || arm), with a cluster
    bootstrap over the 8 prompts (10,000 resamples, a fixed seed).
  - **The sibling the bar is read against:** the dense Qwen3-VL-2B, same prompts, same arms, against an ordinary HF
    float32 forward (it fits), measured in the same run.
  - **PASS** if the 30B's agreement is at least the 2B's minus 5.0 points, per arm. More than 5 points below is
    ambiguous (parked, to the owner). More than 15 below is FAIL.
  - **Controls, run by day before anything is queued:**
    1. The streaming script against an ordinary HF float32 forward of the tiny fixture, on an image sequence: relative
       max|diff| <= 1e-4.
    2. Planted defects in the stream, each red against control 1's bar: DeepStack skipped; two layers swapped; the
       router not renormalised.
- **G-S10q-d, served on nobara:** one-image and two-image requests through serve at int4 on the CPU, twice: identical
  replies. A `--backend cuda` arm on the 8 GB card is reported (resident with expert paging, or a named decline), not
  graded.
- **Tier:**
  - a, b and the controls by day.
  - c on the night queue: the 30B streamed through 48 layers on 8 sequences, plus goinfer's two 30B loads. Estimated
    90 minutes, firmed up from the controls' timings before queueing.
  - d by day if its two arms fit in 10 minutes, else at night.

**Size:** goinfer about 600 lines (the architecture, the loader's shape check, serve), the streaming script about 250,
aikit expected 0: L, as registered for S10's MoE variants.

- **Amendment A1 (2026-10-09 22:10 PDT, before G-S10q-a ran; no bar moves).** G-S10q-a's planted defect 2, "the
  router's softmax taken after the top-8", is not a defect. A softmax over the top-8 logits equals softmax-over-all,
  top-8 and renormalise, exactly, so it would read green by construction. It is replaced by **the router not
  renormalised over the top 8**, which changes the weights.

**S10 Qwen3-VL MoE progress (2026-10-09, Mac):**
- **G-S10q-a: PASS.** `decoder/qwen3vlmoe_test.go` on `testdata/qwen3vlmoe-tiny` (committed; the experts on disk in the
  4.57 layout, `[8, 64, 48]`, so the shape decides the layout as on the 30B).
  - Text: every position at cosine 1.000000000 (14 positions).
  - Image: one 4x6-patch image with three DeepStack sets; goinfer's m-RoPE positions equal HF's `get_rope_index`;
    every position at cosine 1.000000000 (15 positions).
  - Planted defects, each red: the experts read without the transpose 0.795; the router not renormalised 0.945; the
    m-RoPE chunked 0.966; DeepStack one layer late 0.913.
- **A defect found by G-S10q-a: the batched prefill's MoE branch never added DeepStack.** Its `continue` skipped the
  add, the same shape as audit C-07's capture seam in the same branch. The image prompt read 0.666, equal to the
  planted one-layer-late defect because both added nothing. Fixed in `runLayersFromEmbedN`. No family that has
  DeepStack was MoE until now, and no MoE family without DeepStack is affected (its `cache.deepstack` is nil).
- **The fused-expert loader reads by shape.** `fusedLayout` decides `[E, 2I, H]` against `[E, H, 2I]` from the
  tensor's shape and refuses anything else. Element counts alone could not tell the two apart. The square case falls
  to the family's declared layout (`TestFusedLayout`). Qwen3.5-MoE's and Granite's real fixtures load as before (58
  tests green with the fixtures present).
- `num_local_experts` (a transformers 5.x save) is accepted beside `num_experts` (the released config).
- **Serve:** `qwen3_vl_moe` takes the Qwen3-VL tower path with DeepStack (`qwen3TowerFamily`). It is its own family in
  the capability matrix (43 families), with qwen3_moe's feature profile and admission, a census entry (the `.giw` round
  trip keeps the m-RoPE section) and the support-table row.
- **G-S10q-b: PASS** (nobara, 22:13-22:16 PDT). aikit's `Qwen3VisionEncoder` loads the 30B's tower unchanged, so no aikit
  change is needed. HF's `Qwen3VLMoeVisionModel` ran alone in float32 (sdpa), built from the config with only the
  shards' `model.visual.*` (`pin_qwen3vl_tower_real.py --tower-only`), on its own pixel values. aikit's
  `TestQwen3VisionEncoder_realDeepstack` was pointed at the 30B and these artifacts. Every stage is graded: the
  embedding, 27 blocks, the merger and the three DeepStack sets.

  | image | grid | worst stage |
  |---|---|---|
  | gemma3_preprocess_image.png | 56x56 | 0.999999753 |
  | qwen25vl_preprocess_image.png | 14x20 | 0.999999987 |
  | glm_ocr/formula.png | 76x62 | 0.999999999 |
  | glm_ocr/table.png | 56x76 | 0.999999980 |

  Raw: `~/goinfer-logs/qwen3vlmoe/tower/` on nobara (aikit-test.log). The test's log label reads "G-S10b", the dense
  model's gate, whose code it is.
- **G-S10q-c's reference: built, controls PASS (nobara, by day).**
  - `scripts/q3vlmoe_hf_stream.py` runs HF's own decoder layers layer-major. The image rows come from HF's own tower,
    the positions from HF's own `get_rope_index` (run on a stub, since it reads no weights), and the experts are
    converted as transformers converts them (`Transpose(1, 2, check_dims=True)`).
  - Controls on the tiny (`scripts/q3vlmoe_stream_controls.py`), two image sequences:
    - stream against ordinary: 1.95e-7 relative (bar 1e-4);
    - DeepStack skipped: 0.55; two layers swapped: 0.80; the router not renormalised: 0.22. Each is beyond the bar.
  - **Two defects of the stream itself, found by control 1 (1.26 relative) by differencing per layer, and fixed:**
    1. A standalone decoder layer needs its experts implementation named (`from_pretrained` picks `grouped_mm`; HF's
       three implementations agree to 1.3e-6 on the tiny).
    2. Given three-row m-RoPE positions, HF builds the causal mask and calls the layers with `position_ids` None. The
       stream passed the temporal row, which `create_causal_mask` read as packed sequences.
  - goinfer's side is `decoder/qwen3vlmoe_logits_real_test.go` (paths, then the teacher-forced arms); the grader is
    `scripts/q3vlmoe_grade.py`.
  - **Smoke on the 2B sibling, by day:** the paths step took 198 s, with sensible continuations (for example "Table 2.
    Quarterly unit sales by region (thousands)"). HF ordinary ran 179 positions with the image-token check passing.
    Those 2B sequences are the pinned ones; the night job reuses them.
  - **Queued on nobara 2026-10-09 22:34 PDT:** `s10q-c`, est. 2 h 30 min (timeout 5 h), output
    `~/goinfer-logs/s10q-c-run/`, binaries pinned in `~/goinfer-bench/s10q-c` (branch rev `12b1a93d`).
- **G-S10q-c: PASS on both arms** (nobara, the night of 2026-10-09; job `s10q-c`, 1 h 11 min, exit 0). Raw:
  `docs/measurements/multimodal-support-2026-10/s10q-qwen3vlmoe/s10q-c/`.
  - The controls ran again inside the job and passed (stream 1.95e-7; the planted defects 0.55, 0.80, 0.22).

  | model | positions | arm | argmax agreement with HF [95%] | mean KL(HF \|\| arm) [95%] |
  |---|---|---|---|---|
  | Qwen3-VL-30B-A3B | 200 | int8int8 | 98.00% [95.0, 100.0] | 0.0113 [0.0066, 0.0190] |
  | Qwen3-VL-30B-A3B | 200 | int4 | 92.50% [87.9, 97.1] | 0.0691 [0.0338, 0.1353] |
  | Qwen3-VL-2B (the sibling) | 179 | int8int8 | 95.53% [91.9, 98.8] | 0.0187 [0.0083, 0.0268] |
  | Qwen3-VL-2B (the sibling) | 179 | int4 | 90.50% [86.6, 95.8] | 0.1699 [0.1035, 0.2185] |

  - **The rule:** the 30B's agreement minus the 2B's is +2.47 points at int8int8 and +2.00 at int4; PASS is at least
    -5.0. Every logit is finite.
  - HF's own argmax equals the path token at 0.980 on the 30B and 0.933 on the 2B (HF's mean top-1 probability 0.897
    and 0.888).
  - **A deviation from the registration:** it says 256 teacher-forced positions (8 prompts of 32 tokens). Three of the
    30B's continuations ended their turn early (8, 31 and 1 tokens), which leaves 200 positions; the 2B's own
    continuations leave 179. The statistic and the rule are as registered; the intervals are wider than 256 positions
    would give.
  - The binaries were pinned at branch rev `12b1a93d`, before the branch was rebased onto main.
  - Wall time, not a speed record: the 30B's paths 1,103 s, its two arms 2,612 s, the HF stream 348 s.
- **G-S10q-d: no reading (VOID, a fault in the job), nobara, the night of 2026-10-09** (job `gs10q-d`, 22 min, exit 1;
  serve `cuda/cmd/serve` at `f5fa5a30`). Raw: `docs/measurements/multimodal-support-2026-10/s10q-qwen3vlmoe/gs10q-d-void/`.
  - **The graded CPU arms never answered.** The 30B's first int4 load transcodes its sidecar, and that outlasted the
    served driver's 600 s wait for the server (`run-gs10m-served.sh`). The driver's first request was refused, and its
    exit trap stopped serve in the middle of the transcode. A partial
    `qwen3-vl-30b-a3b-instruct.int4.cpu-amd64.tmp.giw` (16.4 GB) is left in `~/models` on nobara.
  - **The reported CUDA arm:** its sidecar (`.int4.cuda.giw`, 16.7 GB) built in 11 min 5 s. The resident build then
    declined by name: `CUDA_ERROR_OUT_OF_MEMORY; the model does not fit this GPU's memory. Try -moe-cache-experts …`,
    "continuing on the CPU/staged path". That is the decline the registration allowed for the 8 GB card.
  - **The exit after the decline, root cause found and fixed 2026-10-10 by day.** It is not about CUDA or the
    decline. Serve exited with `--quant "int4" cannot apply to the prequantized .giw bundle …int4.cuda.giw — it is
    baked at "int4mix"`.
    - **Mechanism:** a `.giw` header records `Weights.quantLabel`, which reads `int4mix` whenever an int4 weight sits
      beside any other kind among the body matmuls, and those include a MoE's routers. The routers stay float32 at
      every quant, so every qwen3_moe-family model at `--quant int4` is labelled `int4mix`. A direct load reports the
      quant that was asked for (`int4`); a load through a sidecar reports the header's label. `modelload.Load` then
      runs `CheckGiwQuantMatch` on the user's explicit `--quant` against that label, and refuses the sidecar it has
      just built.
    - **Reproduced on tiny fixtures (the Mac, throwaway tests, not committed):**
      - `qwen3vlmoe-tiny` and `qwen3moe-tiny` at int4: `quantLabel()` is `int4mix`, body kinds 112 int4 and 4 f32
        (one router per layer). `llama-tiny`: `int4`, 28 int4.
      - `modelload.Load` on a copy of `tiny-qwen2-moe`, backend CPU and Metal alike: with no explicit quant it loads
        through `…int4.canonical.giw` / `…int4.metal.giw` and `Quant()` says `int4mix`; with `ExplicitQuant: "int4"`
        it returns the refusal above.
    - **Reach:** any MoE with float32 routers, named as a directory or a GGUF (both resolve to a sidecar), on any
      backend, when `--quant int4` is typed. The default (no `--quant`) loads. The first run's CPU arms would have
      failed the same way had the transcode finished.
    - **The label is read elsewhere too (by reading the code, not run):** `autoMetalPrecision` keeps an `int4mix` model
      on the CPU under `-backend auto` on a Mac, `residentQuantLabel` prints "int4mix→int4, no Metal int8 GEMV
      kernel", and the KV-snapshot fingerprint carries it. The S6 night log shows the label on a prequant file named
      for int4: `qwen3.6-35b-a3b-int4.giw` decodes "cuda-resident (int4mix)".
    - **Fixed 2026-10-10 (owner: "fix now"), in the same format version.**
      - `Weights.quantLabel` returns `int4mix` only when int4 sits beside an int8 body weight, which is what
        `--quant int4mix` produces. A float32 body weight beside int4 is a by-design pin, as the int8 logit tables
        already are.
      - `LoadSerializedWeights` takes a header that says `int4mix` from the weights instead, so sidecars written
        under the old rule load with the right label and need no rebuild.
      - **No new `.giw` version (the owner asked).** The field keeps its values and the one the old rule got wrong
        can be checked at load, so a version would buy nothing here; it would make every older binary, each pinned
        night binary included, refuse the sidecars a new build writes and rebuild them.
      - **Tests (each red before the fix except the control):** `TestQuantLabel_float32BesideInt4IsNotAMix` (the
        committed `qwen3moe-tiny`: label, a baked bundle, an explicit int4 accepted, an explicit int8int8 still
        refused), `TestQuantLabel_oldInt4MixHeaderIsRederived` (a bundle written with the old label),
        `TestQuantLabel_int8BesideInt4IsAMix` (the control: a real mix is still a mix and still refuses int4), and an
        explicit `--quant int4` through `modelload.Load`'s sidecar path in
        `TestLoad_safetensorsDirGoesThroughSidecar`.
      - **What changes for users:** an int4 MoE through a sidecar now reports `int4`. Its KV snapshots saved under the
        old label no longer match and are prefilled again. Under `-backend auto` on a Mac such a model is no longer
        held on the CPU for being "int4mix" (by reading `autoMetalPrecision`; not run).
  - **Queued again on nobara for the night of 2026-10-10 as `gs10q-d-2`** (est. 1 h 30 min; the same serve binary). The
    job now builds the CPU sidecar first under a 40 min wait, and leaves `--quant` at its int4 default.

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

#### S11, plan and gates, registered 2026-10-09 before any code (owner: "G-IP4, then S11 (it unblocks S15), then S8")

**What exists, mapped 2026-10-09:**
- **Already handles several blocks:**
  - the CPU bidirectional mask (`KVCache.SetImageBlocks` / `attendHi`);
  - `mropePositions`, which consumes one grid per image run in order;
  - `reuseLenOf`, which iterates blocks and claims;
  - the Qwen resident m-RoPE prefill (`PrefillMRoPELast` takes no block argument).
- **One block only:**
  - serve (`maxImagesPerTurn = 1`, `visionInput`, `FindImageRun`, `SpliceImageBlock`);
  - every `Generate*` signature;
  - `residentCommitIDs`;
  - the Gemma 4 bidirectional range (`gemma4AttendRange`, documented as a v1 non-goal);
  - DeepStack rows;
  - Metal's `attention_prefill_img` (one `uint2`) and CUDA's `attn_img_batched` (two ints).
- **Image order is lost:** `contentPartsText` joins the text parts and drops where the images sat.
- **The Responses API silently drops an `input_image` part.** That is a bug, not a limit.

**Choices made here, each changeable by the owner:**
- **Order is kept.** Each image's block goes where its part sat among the text parts, as the HF processors do. Two images
  with no text between them stay two blocks.
- **Never pair images.** Every image is its own block; for Qwen, its own t = 1 grid (`docs/multimodal.md`).
- **The cap:**
  - Several images in the newest message, up to 8 (`maxImagesPerTurn`, a guard on tower work per request).
  - The context must hold every block, as one block's check does today.
  - The newest-message media rule (`image_history.go`, owner 2026-10-01) is unchanged.
- **Audio stays one clip per message.** A message with images and audio stays a 400. S11 is about images.
- **The Responses API** gets `input_image`, with the same rules as chat. Batches and jobs keep refusing images.
- **GLM-OCR stays one image per request,** with a 400 that says so. Its task prompts are written for one page.
- **Resident prefill:**
  - Two-block image prefill on Metal and CUDA comes after the CPU path.
  - Until then a multi-image turn takes the CPU prefill and upload bridge, which handles any number of blocks, and the
    log says why.
  - The Qwen family's resident m-RoPE prefill takes several images from the start.

**Steps:**
1. Two-image tiny goldens from HF (transformers 5.15.0, `~/g4venv` on nobara), for Gemma 3, Qwen2.5-VL, Qwen3.5 and
   Gemma 4 (bidirectional tiny). Two layouts each: A, text, B, text; and A and B adjacent and the same size.
2. Decoder: span slices through the `Generate*` entries, the CPU prefills, DeepStack, `gemma4AttendRange`, the window
   check, and reuse claims and commits.
3. Serve: part order, N images, per-family builders, the Responses API.
4. Metal and CUDA: the image block as a list.

**Gates:**
- **G-S11a, tiny, CPU, against the two-image HF goldens:** each family at its existing single-image golden's bars, on
  both layouts.
- **G-S11b, planted defects** (each must read red; a fixture that is blind to one says so, and G-S11e carries it):
  1. The same-size adjacent pair merged as temporal halves (Qwen: one t = 2 grid).
  2. The two images' features swapped.
  3. Gemma 3 and Gemma 4: the second block left causal.
- **G-S11c, resident:**
  - Each backend's two-image prefill, against the CPU's, on the tiny fixtures: cosine >= 0.995, relL2 <= 0.15, as
    G-IP2.
  - Where a backend declines, the test asserts the decline and its stated reason.
- **G-S11d, reuse:**
  - The same two images again reuse the whole prompt.
  - A changed second image reuses no further than the second block's start.
  - A changed first image reuses no further than the first block's start.
- **G-S11e, served, real, by day** (Gemma 3 4B, Qwen2.5-VL-3B, Qwen3-VL-2B on the Mac; Gemma 4 E2B; `table.png` and
  `formula.png` in one message):
  - The reply names both: it contains "table" and one of "integral", "Gaussian", "formula".
  - Its first 32 greedy tokens match the CPU decoder's two-image path, or first differ at an R10 near-tie.
  - The single-image reply is unchanged: identical to the binary before S11.
- **G-S11f, API:**
  - OpenAI chat, Anthropic messages and the Responses API each carry part order. A unit test checks the block positions.
  - A ninth image is a 400 that names the cap.
  - GLM-OCR's second image is a 400 that names the family.
- **G-S11g:** `gate quick` green; `GOOS=windows go vet`; the tagged vets.

**Cost:** about two days by day. Step 1 is an hour on nobara, step 2 most of a day, step 3 half a day, step 4 half a day
per backend. Nothing in it needs the night queue.

#### S11 results, steps 1 and 2 (2026-10-09, by day)

- **Step 1, goldens** (`scripts/pin_two_images_tiny.py`, from the Mac's fixture bytes copied to nobara):
  - Gemma 3, Qwen2.5-VL and Qwen3.5 are pinned under transformers 5.15.0.
  - Gemma 4 (bidirectional tiny) is pinned under 5.12, for the reason in the finding below.
  - Each golden holds both layouts, every position's logits and a 4-token greedy continuation. Gemma also holds HF's
    own second-block-causal forward: the defect moves its logits by up to 0.29-0.65.
- **Step 2, decoder:**
  - `ImageSpan` and span-slice entries (`GenerateVLSpans`, `GenerateQwenVLDeepstackSpans`, `GenerateGemma4VLSpans`);
    the single-image entries are now one-span wrappers.
  - The CPU prefills splice every span; DeepStack rows cover several runs; `gemma4AttendRange` takes the span list (a
    query sees its own block).
  - Reuse claims and commits take every block.
  - `ResidentImageBlocksPrefill` is the optional multi-block resident interface. Without it a multi-image turn takes
    the bridge, with the reason in `Generation.ImgPrefillDecline`. The resident DeepStack prefill (one run) declines a
    multi-image turn the same way.
- **G-S11a PASS** (`decoder/two_images_test.go`):
  - Gemma 3: every position at cosine 1.000000.
  - Qwen2.5-VL and Qwen3.5: last position 1.000000; m-RoPE positions and rope delta exact; continuation exact.
  - Gemma 4: last position 1.000000; continuation exact.
- **G-S11b:**
  - **Features swapped:** red on Gemma 3 (0.24), Qwen3.5, Gemma 4 (0.992 and 0.995 against 0.999), and Qwen2.5-VL's
    adjacent layout.
  - **Known blind: Qwen2.5-VL's interleaved layout** reads 0.9933 against its 0.99 bar. Its fixture keeps the released
    rope theta, so the last position barely sees the image rows. Logged as BLIND, as registered.
  - **Pairing** (the adjacent pair as one image's temporal halves): red through the exact-position check on both Qwen
    families.
  - **Second block left causal:** red on Gemma 3 (0.85 / 0.88) and Gemma 4 (0.76 / 0.93). On both, the mutant
    reproduces HF's own defect forward at 1.000000.
- **G-S11d PASS** (`TestTwoImages_reuse`, Gemma 3 through the fake resident):
  - The same two images reuse 22 of 23 positions without re-running the tower.
  - A changed second image stops at its block's start (12); a changed first image stops at 4.
- **Regression:** the decoder's image, Qwen, Gemma 4, reuse and resident tests are green. The parity hashes were
  refreshed through `refresh_parity_hashes.sh` (43 forward goldens passed, 0 failed).
- **Found on the way: transformers 5.15 changed a "vision" Gemma 4's bidirectional mask.**
  - 5.12 applied Gemma 3's mask on every layer. That is what goinfer's `gemma4AttendRange` implements, and what the
    committed single-image golden was pinned under.
  - 5.15 makes global layers causal only and, on sliding layers, applies the window after the block OR. Its own
    docstring: "Gemma 4 explicitly disables bidirectional attention on global attention layers".
  - Measured: the single-image golden re-pinned under 5.15 reads cosine 0.99982 against the committed one. Its image
    features and causal-only logits are identical.
  - **This concerns goinfer's 26B/31B image turns, not S11.** Which semantics the real checkpoints were trained with
    is open (Google's reference, not transformers, decides). It is recorded for the owner, not fixed here.

#### S11 results, step 3 (serve, 2026-10-09, by day)

- **What changed:**
  - `maxImagesPerTurn` is 8.
  - Each family's one-image preparation is `prepImage`, and the prompt is `imagesPrompt`. It places every block where
    its part sat, using an offset recorded at parse time (`imageRef.at`). It splices them with
    `multimodal.SpliceImageBlocks`, which works from the end so V-19's last-occurrence rule holds per block, and finds
    one run per image with `FindImageRuns`.
  - One image takes the same code with N = 1.
  - When the media message is not the last user turn, every block leads that turn, as one image always did.
  - GLM-OCR keeps its one-page builder. A second image there is a 400 that names the family.
  - An audio clip must be the only media in its request.
  - `/v1/responses` now takes `input_image` (a data URI; `file_id` is refused) through a new `serveVisionResponses`. A
    multi-image turn whose resident prefill declined logs why.
- **G-S11f PASS** (`internal/serveapp/two_images_test.go`): part order through the chat, Anthropic and Responses parsers
  into the placed blocks (interleaved, adjacent, a merged turn, an earlier media message); the ninth image's 400 names
  the cap; GLM-OCR's and the audio-mix refusals.
- **Kept honest along the way:**
  - The M-38 newline guard (`TestVision_imageBlockNewlinesMatchTheRealProcessors`) now reads each family's case in
    `prepImage`. A planted `+ "\n"` on Gemma 3's block turned it red.
  - The tokenizing-route guard lists the new builders.
  - The SSE heartbeat guard lists the new streaming site.
  - The history test now expects two images in one message to reach the vision path.
- `go test ./internal/serveapp/ ./multimodal/` green; gofmt, vet (plain, realckpt, windows) and staticcheck (linux
  target) clean.

#### S11 results, G-S11e (served, real, 2026-10-09 by day, the M1 Pro; `docs/measurements/multimodal-support-2026-10/s11-gs11e/`)

**Arms.** `serve-metal` at `c8f1b475` (S11) and `80be9921` (before S11). The two-image request is `table.png`, then
`formula.png`, interleaved with text, "What does each image show? Answer briefly, one line per image.":
- Metal, with the tower on the CPU (`-vision-device cpu`, G-S3c's method, so only the decoder differs);
- `--backend cpu`.

The one-image request is `table.png`, Metal, default flags, on both binaries. Greedy, top-3 logprobs, replies to 96
tokens, the first 32 compared.

- **Two driver fixes after the first Gemma 3 run, neither a bar change:**
  1. Its Metal arm ran the tower on the GPU, unlike the CPU arm. That differed at token 18, not at a near-tie.
  2. Its 32-token reply ended before the second image's line.

  The re-run fixed both.

| model | names both | two images, Metal against CPU (first 32) | one image, against before S11 |
|---|---|---|---|
| Gemma 3 4B | yes (the table; the Gaussian integral) | identical | identical |
| Qwen2.5-VL-3B | **no, as registered** (the table only) | identical | identical |
| Qwen3-VL-2B | yes | first differs at token 0, an R10 near-tie (0.270 against 0.492) | identical |
| Gemma 4 E2B | yes | first differs at token 31, an R10 near-tie (0.349 against 0.520) | identical |

- **Qwen2.5-VL's miss is a template gap, not S11's:**
  - HF (transformers 5.12, `~/.venv-vl` on nobara, `hf_two_qwen25.py`) answers the same two images with both: "Image
    1: Quarterly unit sales by region (thousands) / Image 2: A Note on the Gaussian Integral".
  - Its prompt is 2,969 tokens against goinfer's 2,958, and the grids match ([1,64,86], [1,86,72]). The 11 tokens are
    the checkpoint template's default system message ("You are a helpful assistant."), which serve's detected
    `chatml` rendering leaves out.
  - Sent explicitly, goinfer's prompt is HF's 2,969 tokens and the reply names both ("The first image shows a table
    of quarterly unit sales by region in thousands. The second image shows a note on the Gaussian integral.").
  - The gap affects every Qwen2.5-VL request, text included, and predates S11. Recorded for the owner, not fixed here.
- **The swap guard tripped** on that follow-up request: the Mac started at 7.0 GB of swap and grew past it by 1 GB. The
  guard refused the request with a 503, as designed; it was not an S11 defect.
- **G-S11e reads:** single-image identity and Metal-against-CPU agreement hold on all four models. "Names both" holds on
  three, and on Qwen2.5-VL only with the model's own default system message.

#### S11 results, step 4, Metal (2026-10-09, by day)

- **Change:**
  - `attention_prefill_img` takes a block list and a count in place of one `uint2`; a query takes its own block's end.
  - `prefillLastImg` and `batchedPrefillImg` take `[][2]int`.
  - `PrefillImageLast` is a one-block call of the new `PrefillImageBlocksLast` (`decoder.ResidentImageBlocksPrefill`).
- **Kernel (G-IP1, extended):**
  - Every case matches its float64 reference at max |diff| <= 0.00024 against a 4e-3 tolerance: three single blocks,
    two blocks with text between, two adjacent blocks, and two blocks under a sliding window of 6.
  - The causal control misses by 0.31-0.86.
  - **New planted defect, the adjacent blocks merged into one:** misses by 0.347 on the first block's rows, so the
    kernel can see pairing.
- **G-S11c PASS on Metal** (`TestImagePrefillResident_twoImagesTiny`): the resident two-image prefill against the CPU's
  at int4 per-32, the last token plus 8 teacher-forced steps. Worst cosine 0.999765 (interleaved) and 0.999796
  (adjacent), relL2 0.0217 / 0.0202. G-IP2's single-image numbers are unchanged (0.999674, 0.0256).
- **Exploratory real check, not a registered gate** (`TestImagePrefillResident_gemma3TwoImagesReal`; Gemma 3 4B,
  `table.png` and `formula.png`, 551 positions; `s11-two-images-real-metal.log`):
  - Against the CPU per-32 reference, the resident two-block prefill's worst cosine is 0.988801. The bridge's is
    0.984450 (G-IP3's bar 0.979450 holds).
  - **One argmax difference is not an R10 near-tie.** At step 1, after "Here", the reference gives " are" 0.819 and "'"
    0.143; the resident arm gives "'" 0.673 and " are" 0.084. The bridge has none.
  - This is G-IP4's pattern: on Gemma 3 the f16 pass can swing a stylistic token hard while the whole vector stays
    closer to the reference. The test now asserts the cosine bar and logs the flips.
  - **Open for the owner:** whether G-IP4's "keep it on" covers several images too. It is on as built.
- **Served (G-S11e's Gemma 3 arms on this build, `s11-gs11e-step4`):**
  - The two-image Metal reply names both images.
  - Against the CPU arm it now first differs at token 1 (" are" 0.811 against "'" 0.069): that same token.
  - The single-image reply is still identical to the pre-S11 binary.

#### S11 results, step 4, CUDA, and G-S11g (2026-10-09, built and tested on nobara in a detached worktree at `be41cc50`)

- **Change:**
  - `attn_img_batched` takes a block list and a count; its PTX was regenerated through `build_ptx.sh`, and every other
    PTX is byte-identical.
  - `prefillCore` takes `[][2]int`; shared memory is sized at the widest block (`imgBlocksMaxNWin`).
  - `checkPrefillShmemImg` is a one-block call of `checkPrefillShmemImgBlocks`.
  - `PrefillImageLast` is a one-block call of the new `PrefillImageBlocksLast`.
- **G-S11c PASS on CUDA** (`TestImagePrefillResident_twoImagesTinyCUDA`): worst cosine 0.999784 (interleaved) and
  0.999627 (adjacent), relL2 0.0208 / 0.0273.
- **The one-block path unchanged:**
  - The real-checkpoint kernel tests pass (`TestAttnImgBatched_*`): rows outside the block are bit-identical to causal,
    and the decoupled window reaches its poison key.
  - CUDA's real Gemma 3 4B single-image resident gate passes (`TestGemma3ImgPrefillResidentReal_gate`, cosine 0.996878,
    argmax equal, `GenerateVL` on the resident path).
  - The shared-memory guards pass. The structural test that `prefillCore` calls the image guard now names
    `checkPrefillShmemImgBlocks`.
- **G-S11g PASS:**
  - `gate quick` green on the Mac (2,256 passed, 0 failed).
  - The CUDA module's tagged vet and staticcheck are clean on nobara, as are the Metal and decoder vets and staticcheck.

**S11 is done.** Several images per message work on every API surface and backend. What remains:
- **Three items for the owner:**
  1. Whether G-IP4's "keep it on" also covers the resident two-block prefill's stylistic-token flip on Gemma 3 (the
     exploratory real check above).
  2. Qwen2.5-VL's missing default system message, a template gap that predates S11.
  3. transformers 5.15's change to a "vision" Gemma 4's bidirectional mask (step 2's finding).
- **Not built:** a resident DeepStack prefill for several images (Qwen3-VL's multi-image turns take the bridge).

**Owner decisions on S11's three open items, 2026-10-09:**
1. **Keep the resident two-block prefill on**, as G-IP4's single-image one.
2. **Fix the default system message, for every template that has one.** Done: `Detect` reads the default from the
   checkpoint's own ChatML template (`chat/default_system.go`), and every render path applies it when a request has
   none. Qwen 2.5's no-system rendering is now its template's byte for byte. The prompt change for bench comparability
   is recorded in `docs/benchmarks.md`, "Prompt changes that move comparability".
3. **transformers 5.15's Gemma 4 mask:** recorded, unchanged ("sounds good"). Until then a
Gemma 3 multi-image turn on CUDA prefills on the CPU and uploads (logged); on Metal it takes the resident two-block
prefill (below). Qwen2.5-VL and Qwen3.5 take their
resident m-RoPE prefill with several images already; Qwen3-VL's DeepStack resident prefill declines to the bridge; a
Gemma 4 E-model's resident prefill takes them directly.

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

- **S7 and S13-lite on the Mac, levers A and A+B, read 2026-10-09** (the night of 2026-10-08; all four jobs exit 0; raw
  `~/goinfer-logs/night/runs/2026-10-08/s7-mac-leverA*.log` and `s13lite-mac-leverA*.log`).
  - **Conditions:** median of three timed requests per cell, new media every request. The load average sat at 2.3-6
    during the cells (the instant idle gate), so the seconds are a record of the night, not an idle measurement. The
    passes are same-night and interleave nothing, so a small difference between them is not a resolved effect.

  | cell (median TTFT, s) | 2026-10-07 | lever A | levers A+B | under 5 s |
  |---|---|---|---|---|
  | gemma-3-4b (S13-lite, 9 requests) | 12.81 | 7.79 | 6.60 | no; Ollama 4.97, llama.cpp 4.69 |
  | gemma-4-e2b | 6.20 | 6.24 | 6.17 | no |
  | gemma-4-e2b audio | 3.20 | 3.23 | 3.14 | yes |
  | qwen2.5-vl-3b | 21.67 | 20.50 | 17.72 | no |
  | qwen3.5-0.8b | 4.48 | 4.03 | 3.94 | yes |
  | qwen3-vl-2b | 8.97 | 7.74 | 7.30 | no |
  | glm-ocr | 8.32 | 6.40 | 5.30 | no |

  - **Reading:**
    - The tower levers took Gemma 3 from 12.8 to 6.6 s, still 1.3-1.4x the peers.
    - Every Qwen cell moved with them.
    - Gemma 4 E2B did not move (its tower was already on Metal; the time is elsewhere).
    - Two cells pass, five do not.
  - **All of this predates 2026-10-09's changes,** which reach three of the five failing cells:
    - S16 put Qwen2.5-VL's and Qwen3-VL's image prefill on the GPU; the served comparisons read 19.46 -> 5.73 s and
      8.25 -> 2.86 s.
    - GLM-OCR's decoder and prefill went resident (`docs/tasks/task-metal-pairwise-followups-2026-10.md`).

- **S7 and S13-lite on the Mac, a third pass on 2026-10-09's tree, registered 2026-10-09 before it runs** (owner:
  "continue with image/audio track").
  - **What:** the same scripts, cells, procedure and 5 s bar as the two passes above.
  - **Binary:** `serve-metal` at `74830779`, in `~/goinfer-bench/s7-2026-10-09/` (`BIN=`).
  - **Purpose:** re-rank the failing cells for S17/S18's next lever, since three of the five moved by day.
  - **Queue:** `s7-mac-0909` and `s13lite-mac-0909`, 20 minutes each, tonight.
  - **Read 2026-10-10** (the night of 2026-10-09; both jobs exit 0, 3 and 4 minutes; raw
    `docs/measurements/multimodal-support-2026-10/night-2026-10-09-mac/`).
    - **Conditions:** as the earlier passes: the median of three timed requests per cell (nine for S13-lite), new media
      every request, every cell `metal-resident (int4)`. The load average was 3.2-4.1 during S7's cells and 1.6-4.0
      during S13-lite's (the instant idle gate; a sidecar build and G-S18g had just run), so the seconds are a record
      of the night.
    - The two Gemma 4 E2B copies had been deleted from `~/models` that evening and were pulled back from the archive
      before the run (13,590,544,823 bytes, equal on both sides). The Metal sidecars for E2B, Qwen3.5-0.8B and E4B
      were built with this same binary before the queue started. The E4B cell ran because its sidecar existed.

    | cell (median TTFT, s) | levers A+B (2026-10-08) | third pass | under 5 s |
    |---|---|---|---|
    | gemma-3-4b (S7's cell) | — | 6.27 | no |
    | gemma-3-4b (S13-lite, 9 requests) | 6.60 | 6.23 | no; Ollama 4.98, llama.cpp 4.70 |
    | gemma-4-e2b | 6.17 | 1.97 | yes |
    | gemma-4-e2b audio | 3.14 | 0.57 | yes |
    | gemma-4-e4b | — | 2.57 | yes |
    | qwen2.5-vl-3b | 17.72 | 5.51 | no |
    | qwen3.5-0.8b | 3.94 | 4.21 | yes |
    | qwen3-vl-2b | 7.30 | 2.83 | yes |
    | glm-ocr | 5.30 | 3.09 | yes |

    - **Reading:** six of eight S7 cells are under the bar, against two of seven on the pass before. The two over it
      are Gemma 3 (6.2-6.3 s, 1.25x Ollama and 1.33x llama.cpp in S13-lite) and Qwen2.5-VL (5.51 s).
    - The moves follow 2026-10-09's changes: the E2B cells log S9 step 2's batched prefill for the E-models;
      Qwen2.5-VL and Qwen3-VL prefill their images on the GPU (S16); GLM-OCR's decoder went resident.
    - **This binary (`74830779`) predates Gemma 3's resident image prefill on Metal.** G-IP4 read that lever by day at
      6.88 s before and 4.14 s after, so Gemma 3's row here is the 'before' build.
    - **Ranking for the next lever:** Qwen2.5-VL is the one cell over the bar with no lever already measured under it.
      **Owner, 2026-10-10: accepted as it is** ("we will just accept 5.51 as OK"); no lever is owed for it.
    - The passes are on different nights and interleave nothing; a small difference (Qwen3.5-0.8B's 3.94 to 4.21 s)
      is not a resolved effect.

- **S7 and S13-lite on the Mac, a fourth pass on 2026-10-10's tree, registered 2026-10-10 before it runs** (owner: "do
  we need to queue or run something to get the 4.14 speed on the serve path?").
  - **Nothing has to change in serve:** Gemma 3's resident image prefill is on by default since `f6b113ba`, which is
    in `main`. What is missing is a night reading of it: the third pass's binary predates it, and 4.14 s is one
    exploratory reading by day.
  - **What:** the same scripts, cells, procedure and 5 s bar as the three passes above.
  - **Binary:** `serve-metal` at `eb882131`, in `~/goinfer-bench/s7-2026-10-10/` (`BIN=`).
  - **The reading it is for:** Gemma 3 4B's median TTFT in S7's cell and in S13-lite (nine requests, beside Ollama and
    llama.cpp the same night), against the 5 s bar. The other cells repeat as a record.
  - **Queue:** `s7-mac-1010` and `s13lite-mac-1010`, 20 minutes each (the third pass took 3 and 4), for the night of
    2026-10-10.

- **S17's next Metal lever: Gemma 3's image turn prefills on the CPU. A resident image prefill on Metal, registered
  2026-10-09 before any code** (owner: "continue with image/audio track").
  - **Measured by day (exploratory, one run each; `metal/zz_g3_prefill_probe_test.go`, not committed):**
    - Gemma 3 4B's image turn (282 positions, 256 of them the image) prefills on the CPU in 3.67-3.86 s, then
      uploads its K/V (0.04 s).
    - Metal implements no `ResidentImagePrefill`, so every non-Qwen image turn takes that bridge.
    - The same 282 positions through Metal's batched pass take 0.93 s (causal, so for scale only).
    - With the tower at about 2.8 s, that CPU prefill is most of the gap between S7's 6.6 s and the peers' 4.7-5.0 s.
  - **Change:**
    - `metalResident.PrefillImageLast` (`decoder.ResidentImagePrefill`): the batched f16 pass with the image block
      `[imgStart, imgEnd)` attending bidirectionally, as the CPU's `KVCache.attendHi` and CUDA's
      `PrefillImageLast` do.
    - `attention_prefill` (the exact kernel) takes the block. A query inside it sees keys up to `imgEnd - 1`; every
      other query stays causal; the sliding window's lower bound stays at the causal position's.
    - When a block is present the pass runs the exact kernel on every layer; the fused and steel kernels carry no
      mask. Gemma 3's head dim (256) uses the exact kernel anyway.
    - The whole prompt in one pass (no chunking across a bidirectional block, as on CUDA).
  - **G-IP1, kernel:** `attention_prefill` with a block against a float64 reference, covering causal rows, block
    rows, a block at a non-zero start and a sliding window that cuts in. Max abs diff <= 4e-3 (f16 output).
    Control: the same kernel with no block must miss by at least 10x on the block's rows.
  - **G-IP2, tiny** (`gemma3-vl-tiny`, its image golden's prompt): the resident image prefill's last-token logits
    and 8 teacher-forced decode steps after it, against the CPU's image prefill (`prefillLogitsVL`) at int4 per-32.
    - Bars: cosine >= 0.995, relL2 <= 0.15.
    - Planted defect: the block ignored (causal everywhere) must read red, or the fixture is blind and the gate moves
      to G-IP3 alone, recorded.
  - **G-IP3, real** (Gemma 3 4B, S3's four images, through its sidecar): last row plus 8 teacher-forced steps, against
    the CPU image prefill at per-32.
    - Bar: the resident path's worst cosine >= today's production path's worst minus 0.005, where today's path is the
      CPU W4A8 prefill uploaded, graded against the same reference. Every argmax difference is an R10 near-tie.
    - Planted defect as in G-IP2.
  - **G-IP4, served** (by day, exploratory): S3's served comparison (`run-gs3c-served.sh`, arms `metal` against the
    CPU bridge). Replies identical or first diverging at a near-tie. TTFT reported; tonight's S7 pass, if queued
    after this lands, carries the record.
  - **Speed:** ships at >= 1.02x TTFT (the bar), which the measurement above expects to clear by about 1.7x.

- **G-IP3's first run found a shipped bug: Gemma 3 4B's batched pass on Metal returns garbage, text or image** (by day,
  2026-10-09; probes in `metal/zz_g3_prefill_probe_test.go`, not committed).
  - The resident image prefill and the planted no-block arm both read cosine NaN. That was a cosine of all-zero logits.
  - **Text alone does it too.** A 31-token text prompt through `Generate` on Metal answers `<pad>` and then unrelated
    tokens; the sequential path's first token is 818.
    - Every Gemma 3 prompt of 16 tokens or more on Metal has taken this pass since `84c3f29f` (2026-09-13, audit M-06,
      which admitted Gemma 3 and graded it on tiny fixtures only).
    - Image turns were spared: they prefill on the CPU (above).
  - **Mechanism, measured:** the pass keeps the residual in f16 (max 65,504), and Gemma 3 4B's residual outgrows it.
    - The `<bos>` row reaches 50,752 by layer 5 and is inf at layer 6. At layer 7 its K/V are NaN, attention spreads
      NaN to every row, and the int8 LM head turns a NaN row into finite zeros.
    - So A-C02's non-finite-logit check (`batchedPrefillImg`) passes it.
    - Computing `<bos>` on decode's f32 kernels and the rest batched does not help: every ordinary row climbs to about
      60,000 by layer 26 and overflows at layer 27.
  - **Fix 1, safety (registered before code):** the pass declines when its last residual row is non-finite after the
    trunk, and the decoder re-runs the prompt sequentially, as A-C02 does for a non-finite logit.
    - The last row is enough. A row that overflows before the final layer writes NaN K/V, which reaches the last row
      through attention. A row that overflows only in the final layer writes no further K/V. A NaN K/V row that only
      sliding-window layers hold, outside the last row's window, is never attended again.
    - **G-NF1, real** (Gemma 3 4B, the 31-token prompt): the batched pass declines, naming the overflow, and
      `Generate`'s 24 greedy tokens equal the sequential path's (fast prefill off). The planted defect, the guard off,
      is today's garbage (above).
    - **G-NF2, tiny:** a test seam that writes inf into the last residual row makes the pass decline. Every family
      that does not overflow keeps its path; `gate quick` green.
  - **Fix 2, the lever (registered before code): store the residual scaled by 1/s.**
    - **Applies to:** a sandwich-norm family with no PLE, MoE, post-only norm or DeepStack (Gemma 3, on this pass).
      - Every read of its residual there is an RMSNorm (pre-attention, pre-MLP, final), and RMSNorm ignores scale
        except through eps.
      - Every write is an embedding upload or a post-normed branch's `residual_f16` add.
      - So the pass uploads the embeddings times 1/s and adds each branch times 1/s.
      - s is a power of two, so the scaling is exact in floating point. The scaled pass equals today's wherever today's
        does not overflow, except through eps and subnormals.
    - **s:** the smallest power of two that leaves at least 4x headroom over the largest residual measured on Gemma 3
      4B (the pass's largest finite value times s) across the G-RS2 prompts and S3's four image turns. Recorded.
    - **G-RS1, identity (tiny, `gemma3-vl-tiny`, which does not overflow):** last-token logits scaled against unscaled,
      cosine >= 0.99999. Families without the scale run the unchanged kernels, so they are identical by construction,
      and their existing tests are unchanged.
    - **G-RS2, real Gemma 3 4B text:**
      - Batched last token against the CPU at int4 per-32: cosine >= 0.98 (G-B3's real tier).
      - 16 teacher-forced decode steps after it, with no non-tie flip.
      - The guard of Fix 1 does not fire on prompts of 31, about 450 and about 2,000 tokens.
      - Planted defect: s = 1 must trip the guard.
    - **G-RS3, speed, exploratory by day:** TTFT on about 512 tokens, batched against sequential. Ships at >= 1.02x.
    - G-IP3 then runs as registered.
  - **Results, 2026-10-09 (by day, the M1 Pro; logs `docs/measurements/multimodal-support-2026-10/s17-*.log`, probes
    archived as `zz_g3_prefill_probe_test.go.txt` there):**
    - **G-NF1 PASS** (`TestGemma3PrefillOverflow_declinesReal`, run at s = 1): the batched pass declines, naming the
      overflow, and `Generate`'s 24 greedy tokens equal the sequential path's ("The bicycle's origins can be traced back
      to 1817 in Germany, ...").
    - **G-NF2 PASS** (`TestPrefill_declinesNonFiniteLogits`): an inf in the last residual row declines; the next clean
      pass does not.
    - **s = 32.** The measured peak was 295,936 on every prompt (the `<bos>` row, after layer 31; prompts of 31, 716 and
      3,094 tokens and an image turn). s = 16 would leave 3.5x headroom, under the registered 4x; s = 32 leaves 7.1x.
    - **G-RS1 PASS, bit-identical** (`TestPrefillResidScale_identityTiny`): s = 32 against s = 1, max |diff| 0.
      - The first build read cosine 0.99994, under the 0.99999 bar. **Mechanism:** the tiny fixture's residual is
        small, so after dividing by s² its mean square sat near eps (1e-6).
      - **Fix, inside the registered design:** the three norms that read the residual (pre-attention, pre-MLP, final)
        take eps/s². Then x/s / sqrt(ms/s² + eps/s²) is x / sqrt(ms + eps) exactly.
    - **G-RS2 PASS** (`TestGemma3PrefillResidScaleReal`; batched last token against the CPU at int4 per-32, then 16
      teacher-forced steps). The overflow guard fired on none of the three prompts at s = 32 and on all of them at
      s = 1.

      | prompt | last token | worst over the 16 steps | flips |
      |---|---|---|---|
      | 31 tokens | 0.999596 | 0.982823 | 0 |
      | 468 tokens | 0.999085 | 0.960446 | 2 near-tie, 0 real |
      | 1,894 tokens | 0.999030 | 0.958636 | 0 |

    - **G-RS3, exploratory:** 538 tokens, batched 2.06-2.33 s against sequential 15.05-15.16 s (three runs each, warm),
      about 7x. Ships (bar 1.02x).
    - **G-IP3 PASS** (`TestImagePrefillResident_gemma3Real`, S3's four images through the sidecar):
      - The resident image prefill's worst cosine is 0.991388, against a bar of 0.968103 (production's worst, 0.973103,
        minus 0.005), with 0 non-tie argmax differences.
      - It sits closer to the reference than today's CPU W4A8 bridge on every image (0.991-0.993 against 0.973-0.986).
      - The planted no-block arm reads 0.962159 over the four, under the bar, so the gate is not blind. On one image
        alone (table.png) it reads 0.989773, which is above it.
      - The prompt is serve's shape (`Gemma3PromptBlock`, `<bos>`). The first run used the bare block without `<bos>`;
        corrected before grading.
    - **G-IP2 PASS** (tiny), as before: 0.999674 / relL2 0.0256. The planted defect is blind there, as registered, so
      G-IP3 carries it.
    - `gate quick` green (2,237 passed, 0 failed); staticcheck clean on the darwin `metal` package and the linux target.
    - **G-IP4, served, by day (exploratory): FAIL as registered** (`docs/measurements/multimodal-support-2026-10/s17-gip4/`).
      The arms are `serve-metal` at `74830779` (before; the CPU bridge) and at `80be9921` (after), `--backend metal`,
      default flags, the tower on Metal.
      - **TTFT** (`vision_ttft.py`, three interleaved rounds, one warm-up and three timed requests each, fresh images):
        median 6.88 s before, 4.14 s after, 1.66x. Gemma 3's Metal cell is now under the 5 s bar.
      - **Replies** (S3's four images, 32 greedy tokens, top-3 logprobs, after against before):
        - `qwen25vl_preprocess_image.png` is identical.
        - `gemma3_preprocess_image.png` first differs at token 10 (" bright" 0.495, " green" 0.324): a near-tie.
        - `formula.png` first differs at token 2 (" demonstrates" 0.384, " shows" 0.377): a near-tie.
        - **`table.png` first differs at token 5, and not at a near-tie:** before " presenting" 0.625, after " of",
          which before gave 0.104. **This is the FAIL.** The bar is not moved.
      - **Diagnostic, not a regrade** (`diag-table*.txt`). Served prompt, CPU tower, " The image shows a table"
        teacher-forced, then p at token 5:

        | arm | " presenting" | " of" | " detailing" |
        |---|---|---|---|
        | CPU int4 per-32 (G-IP3's reference) | 0.440 | 0.271 | 0.168 |
        | the bridge | 0.418 | 0.328 | 0.223 |
        | the resident pass | 0.172 | 0.494 | 0.267 |

        - Under the reference the position is close: " of" is an R10 near-tie (0.271 >= 0.220).
        - At this token the bridge sits nearer the reference than the resident pass, which reverses G-IP3's
          whole-vector reading for this image.
        - The served bridge's 0.625/0.104 split comes from the Metal tower's features; with CPU tower features the
          bridge is itself near a tie.
        - **Not the residual scale:** at s = 8, 16, 32 and 64 the resident pass picks " of" (p 0.44-0.50), with
          " presenting" at 0.16-0.31.
      - **Both replies are correct descriptions of the table.** Whether the lever stays on, which it is since
        `f6b113ba`, is the owner's decision.
      - **Owner decision, 2026-10-09: keep it on**, over G-IP4's registered FAIL. G-IP3's logit gate passed, and against
        the reference the divergence is an R10 near-tie. The FAIL stays recorded as read.

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

- **S18 on the Mac: the design amended and its gates registered 2026-10-08, before any code (owner: "sidecar for dirs
  too").**
  - **Why the registered change cannot pass alone.** Gemma 3 4B from its safetensors directory prices about 5.15 GB on
    Metal:
    - about 1.7 GB of int4 layers on the device;
    - the same again as the heap host copy, because a safetensors load quantizes into the heap, and Metal's unified
      memory then holds both;
    - about 1.2 GB of int8 LM head and embedding;
    - 0.53 GB of KV at the 4096 default (34 layers x 4 KV heads x 256 x 2 x f16 = 136 KB a position).

    Shrinking the context or the slot count saves at most about 0.45 GB, which leaves about 4.7 GB against a live budget
    of 4.2-4.9 GB (G-S3b, G-S3c, S7). The load guard read 3.9 GB at 14:05 today. Metal's guard does not price the
    tower at all: `towerReserve` is CUDA-only, and Metal never reads `ExtraResidentBytes`.
  - **The lever: a `.giw` sidecar for a safetensors directory too.** A `.gguf` on darwin already loads through its
    sidecar (S1, `prequant.DefaultToSidecar`). The weights are then mmap aliases, which the guard prices once (M-24's
    exemption in `ResidentHostCopyBytes`). That takes about 1.7 GB off the need.
  - **The change, four parts:**
    1. **The sidecar for a directory,** where `DefaultToSidecar` holds (darwin, linux) and no `-direct-load`:
       - `modelload.Load` resolves a safetensors directory to `<dir>.<quant>.<target>.giw` through
         `prequant.EnsureCachedGIW`, which already transcodes a directory (`transcodeDir`).
       - Not for a `--lora` load (the merge needs the safetensors base), and not for `--quant q4k` (no `.giw` form).
       - A transcode that fails falls back to the direct load with a one-line note, never a refused start.
       - Freshness for a directory: the sidecar must be newer than every regular file in the directory, not just the
         directory's own mtime.
       - The disk check for a directory: a projection from the safetensors files' sizes at the quant, not the
         directory entry's 4 KB.
    2. **Metal prices the tower.** `towerReserve` returns the Metal tower's resident bytes when the tower runs on Metal,
       and Metal's guard (`residentNeedBytes`, so also `metalKVSlots`) adds `ExtraResidentBytes`. The figure comes from
       the tower's dims at the precision Metal uploads, calibrated against a measured RSS delta by day (below).
    3. **The context before the decline.** If an unpinned build still prices over the budget, the context halves from
       4096 down to a floor of 2048 before the guard declines. An explicit `-ctx` is never shrunk.
    4. **The banner:** serve prints one line naming the KV plan and why it shrank, as registered, when part 3 or the
       slot count reduced it.
  - **Gates (G-S18a as registered, plus Mac-side additions):**
    - **G-S18a, the Mac cell:** `serve --model ~/models/gemma-3-4b-it --backend metal`, no sizing flags. The decode path
      is `metal-resident` and the tower is on Metal. One served image turn (`table.png`, 32 greedy tokens, top-3
      logprobs) whose reply is identical to the same request with hand-set flags (`--model <the sidecar> -ctx 2048
      -kv-sessions 1`). By day, a served correctness check, not timed. It is recorded with the live budget it ran
      under, because the budget moves with the owner's load.
    - **G-S18d, the directory sidecar, by tests:** on a tiny safetensors fixture, a directory loads through a sidecar
      built once; the second load reuses it; CPU logits through the sidecar are bit-identical to the direct load's;
      `-direct-load` and `--lora` stay direct; a touched safetensors file makes the sidecar stale. Red on today's code
      (it loads the directory directly).
    - **G-S18c-Mac, the reserve:** `towerReserve` is nonzero for every family with a Metal device tower under `metal`,
      zero under `-vision-device cpu`. The Metal guard's need grows by exactly the reserve. Gemma 3's figure is within
      ±25% of the RSS delta its tower load measures on the Mac.
    - **G-S18b-Mac, no regression:** for the bench set's text models, the Metal plan's context and slot count are never
      lower than today's (the sidecar can only lower the need). The `prequant` and `modelload` suites pass.
  - **Cost:** one transcode of Gemma 3 4B (minutes, about 2.4 GB of disk; 10 GB free at registration).
- **S18 on the Mac, by day 2026-10-08: part 1 built and G-S18d PASS; two findings that part 1 cannot pass G-S18a
  alone. Parts 2-4 wait on the owner.**
  - **Part 1, built:**
    - `modelload.Load` resolves a safetensors directory to its sidecar where the default holds; `dirSidecarApplies`
      excludes `--lora`, q4k and `-stream-weights`. A failed build keeps the direct load, with a note.
    - The tokenizer stays the directory's. A sidecar's tok half is `tokenizer.json` alone, so `tokenizer_config.json`'s
      chat template and BOS/EOS flags would otherwise be lost silently.
    - In `prequant`: the cache path keeps a dotted directory name whole; freshness is against every file in the
      directory (`sourceFiles`); the disk projection for a directory is `projectedDirSidecarBytes`.
  - **G-S18d PASS.** `TestDirSidecar_matchesDirectLoad`: 42 tiny safetensors fixtures, every family `decoder.Load`
    reads directly, generate identical greedy streams through the sidecar (int4, embed-int4, Metal target) and direct.
    No transcode was refused. The 6 skipped are towers, an audio encoder and `embedding_gemma2`, which `decoder.Load`
    does not read. `TestDirSidecar_cachePathAndFreshness` and `TestLoad_safetensorsDirGoesThroughSidecar` cover the cache
    path, freshness, reuse without a rebuild, the chat template, `-direct-load` and the exclusions. The modelload,
    prequant, serveapp, chatapp, clef and decidecmd suites pass, and none wrote a sidecar into `testdata/`.
  - **Finding 1: the one-time build loads the whole directory into the heap.** There is no streaming transcode for
    safetensors; each family's loader builds the whole model. So the build costs the same RAM as today's direct load,
    and the load guard refused Gemma 3 4B's build at 14:09: "needs ~5.6 GB resident at quant int4 + 0.0 GB KV",
    against a 3.4 GB budget (70% of 4.9 GB available). That 5.6 GB includes the bundled vision tower priced at f32
    (about 1.6 GB), which a text-only build never loads. Without it, about 3.9 GB is still over budget. Today's direct
    load is refused the same way at this hour. The sidecar has to be built once while memory is free (at night, or
    with the owner's apps closed); every load after that is an mmap.
  - **Finding 2: the tower is bigger than the decoder's saving.** An exploratory probe on Gemma 3's Metal SigLIP alone
    (`vision.LoadEncoder` then `newSiglipVResident` then one forward, the process's `phys_footprint` after each, a
    throwaway test, not committed):
    - the host f32 encoder: **2.17 GB**;
    - plus the Metal resident: **3.23 GB** (the f16 device copy, +1.06 GB);
    - after one forward: **3.45 GB** (+0.22 GB of scratch).

    The resident keeps the host encoder referenced (`siglipVResident.enc`, for `FinishHidden`), and serve keeps it as
    the CPU fallback tower. With the sidecar, the decoder needs about 3.3 GB, so decoder plus tower is about 6.8 GB
    against a 4.2-4.9 GB budget. Part 2 would price this honestly and then decline, which is correct but not a pass.
  - **What would pass:**
    - dropping the host f32 encoder after the Metal upload (about -2 GB, but it loses the CPU fallback tower and needs
      an aikit API to free the block weights while keeping the post-LN ones);
    - an int8 SigLIP on Metal (CUDA's int8 tower holds 558 MiB);
    - or both.

    Each is the owner's call.
- **S18 on the Mac, the tower: owner chose "both" (c) 2026-10-08. A design probe, then the design and gates,
  registered before any tower code.**
  - **The probe (exploratory, a throwaway test, CPU, not committed):** Gemma 3's SigLIP with its projection weights
    rounded to int8 and activations kept f32, against the f32 tower, on the four F2a images (`last_hidden_state`,
    4,096 tokens):

    | int8 granularity | mean token cosine | relative L2 | worst token |
    |---|---|---|---|
    | groups of 32 | 0.99914-0.99962 | 0.020-0.040 | 0.39-0.99 |
    | groups of 128 | 0.99845-0.99939 | 0.028-0.054 | 0.39-0.91 |
    | per row | 0.99664-0.99863 | 0.046-0.073 | 0.49-0.83 |
    | W8A8, serve's int8 tower (`siglip-int8-fidelity-2026-10-07.md`) | 0.932-0.987 | 0.16-0.52 | 0.01-0.17 |

    - **What it shows:** most of W8A8's loss is the activations. Weight-only int8 in groups of 32 is about 10x closer.
    - **No int8 form holds the tower bar** (every token at cosine >= 0.9999): a few outlier tokens stay far off, and
      they do not shrink monotonically with the group size. So the int8 tower is graded at the decoder (the served
      reply), not by the tower bar.
  - **The design:**
    1. **Build the tower block by block from disk (the "free the host copy" half).** Today `vision.LoadEncoder`
       materializes all 27 blocks in f32 (2.17 GB). The Metal resident then uploads its own copy, and serve keeps the
       encoder as the CPU fallback. Freeing the blocks after the upload would still peak at about decoder + 2.17 GB +
       the device copy at the first image, the swap case. So:
       - aikit gains a head-only load (config, patch embed, position table, post-layernorm; no blocks) and a per-block
         f32 reader;
       - the Metal resident builds from that, one block at a time, so the host never holds more than one block;
       - the encoder's CPU forward loads its blocks on first use, so the CPU fallback still works (slower the first
         time).

       That is an aikit release (the owner's to push and tag); goinfer develops against the local checkout until then.
    2. **A weight-only int8 SigLIP on Metal:** int8 weights in groups of 32 along K, each with an f32 scale, widened to
       f32 as they are staged in lever B's GEMM (`tower_gemm_w8`, the same 64x64 tiles); activations, attention and
       norms stay f32. Device weights are about 0.46 GB, against the f16 tower's 0.82 GB.
    3. **Serve's choice on Metal, mirroring CUDA's (`resolveGemma3VisionQuant`):**
       - unset `-vision-quant`: the f16 tower when the decoder plus that tower fits the Metal budget, otherwise the
         int8 one, with a note;
       - `f32`: the f16 tower, as today;
       - `int8`: the Metal int8 tower (today it is the CPU W8A8 tower).
    4. **S18's parts 2-4 as registered above** (Metal prices the tower at the form chosen, the context before the
       decline, the banner).
  - **Gates, registered before the code:**
    - **G-S18e, the kernel:** `tower_gemm_w8` against a float64 reference over the same dequantized weights, each output
      within 1e-5 of Σ|a·w| + |bias| (only accumulation differs). Planted defects red: a group's scale taken from the
      next group, the bias dropped, A one K column late.
    - **G-S18f, the tower is exact on its weights:** the Metal int8 tower against the CPU f32 tower running the same
      group-32-rounded weights; every token at cosine >= 0.9999 (the tower bar, on the quantized weights), on the
      four F2a images. The quantization's own cost against the true f32 tower is recorded (relative L2 per image). A
      relative L2 above 0.06 on any image is red, a regression guard at 1.5x the probe's worst.
    - **G-S18g, the reply (decoder-level, night: it needs the 4B decoder):** a served image turn (`table.png` and the
      four F2a images, 32 greedy tokens, top-3 logprobs), the Metal int8 tower against the f16 Metal tower on the
      same decoder. PASS: identical replies, or each first difference at an R10 near-tie (p(other) >= half p(top)).
      **If it fails, the int8 tower is not a default:** explicit `-vision-quant int8` only, and Gemma 3 on the
      16 GB Mac keeps the f16 tower where it fits, or the CPU.
    - **G-S18h, host memory:** with the tower attached, its share of the process's `phys_footprint` is within 10% of
      device weights + scratch + head. No 2.17 GB block copy stays, and none is transiently held (the peak sampled
      during the attach). The CPU fallback still produces the f32 tower's output after a forced device failure.
    - **G-S18a as registered,** read by day after the sidecar is built at night, with whichever tower serve chose,
      named.
- **S18 on the Mac, the tower, by day 2026-10-08: built; G-S18e, G-S18f, G-S18h and G-S18c-Mac PASS; part 3 built and
  tested. G-S18g (the served reply) and G-S18a wait on the night and on an aikit release.**
  - **aikit v1.60.0, released 2026-10-08 (owner: "yes push / release"),** from branch `s18-encoder-head`:
    - `vision.LoadEncoderHead` (the encoder without its blocks) and `Encoder.ForEachSiglipBlock` (the blocks read
      one at a time from the checkpoint).
    - The CPU `Forward`, `Weights` and `GPUWeights` load the blocks on first use, so the CPU path and the CUDA and
      WebGPU towers are unchanged.
    - `Quantized` and `HasBlocks` accessors.
    - `TestLoadEncoderHead_matchesLoadEncoder`: bit-identical to `LoadEncoder` at both precisions, and the streamed
      blocks equal `Weights()`. The vision suite passes.
  - **goinfer is on aikit v1.60.0 since its merge** (every module's require bumped).
  - **goinfer moved to Go 1.27.2 the same evening (owner: "lets move to 1.27.2").**
    - aikit's release had moved aikit to `toolchain go1.27.2`, because govulncheck v1.8.0 now finds the Go 1.27.0
      standard library's advisories GO-2026-6607..6617 reachable (fixed in 1.27.2). It also moved aikit's
      golangci-lint to v2.14.0. goinfer's govulncheck read the same advisories at 41814649 (serve uses net/http).
    - No staticcheck release (v0.8.0 or v0.8.1) reads Go 1.27.2's export data (version 5). A canary that imports the
      standard library fails on both: "export data version 5 is greater than maximum supported version 4".
    - So CI now builds staticcheck v0.8.1 against golang.org/x/tools v0.51.0 from `.github/actions/staticcheck/go.mod`
      instead of downloading the release. That costs the ~24 s compile C5 had removed, until a staticcheck release
      reads Go 1.27.2.
    - The root go.mod gains `toolchain go1.27.2`. A workspace's go.work needs the same line, because the module's line
      is ignored in workspace mode.
    - CLAUDE.md's install line and `cmd/gate`'s hint now name the source build.
    - `tower_gemm_w8` in `metal/grid_vision.go`. `tower_gemm_body` takes the scale under a compile-time flag, so w16
      and w32 compile as before.
    - The SigLIP resident streams the blocks for a head-only encoder and keeps uploading in-memory float32 blocks
      otherwise (the tiny gates sharpen weights through `Weights()`). An encoder loaded with quant=true gets the int8
      tower.
    - Serve:
      - on Metal it loads a head-only encoder;
      - the unset default is chosen by `resolveGemma3VisionQuantMetal` (f16 when the budget holds the decoder, one
        KV slot at 2048 and the f16 tower, else int8, with a note);
      - a failed f16 default attach falls back to the int8 device tower, as on CUDA.
    - `towerReserve` prices Metal towers (`metalTowerEstimate`). Metal's guard adds `ExtraResidentBytes`, the drafter's
      included.
    - `BuildResident` halves an unpinned context to the 2048 floor before it declines (`shrinkCtxToFit`) and prints
      the KV plan line.
  - **G-S18e PASS** (`TestTowerGemmKernel/tower_gemm_w8`): worst 3.26e-7 against 1e-5 over 8 shapes. The planted
    defects read: the bias dropped 1.06e-1, A one column late 2.91e-1, a quarter of each K step skipped 1.08e-1, the
    next group's scale 6.46e-2.
    - Found on the way: a `char4` load from an unaligned int8 row (K = 9 or 17) reads the wrong bytes. The int8 path
      now takes the element path unless K is a multiple of 4.
  - **G-S18f PASS** (`s18-mac/gs18f-real.log`; tiny `TestS18Int8Tower_tiny`).
    - On the same rounded weights the Metal int8 tower's worst token is 0.999999965 / 0.999999947 / 0.999999881 /
      0.999999849 on the four F2a images.
    - Against the true float32 tower, the relative L2 is 0.0254 / 0.0285 / 0.0399 / 0.0200, the probe's figures
      exactly, under the 0.06 guard.
    - Tiny: 1.000000000, with every planted defect red; the scale defect at 0.999829, the narrowest.
  - **G-S18h PASS** (`s18-mac/gs18h-memory-{f16,int8}.log`, each form in its own process; the first run had both in
    one process and is `gs18h-memory.log`, which confounds the second form's footprint):

    | tower | Go heap peak during the attach (bar 217 MB) | phys_footprint after one image |
    |---|---|---|
    | f16 | 215 MB | 1172 MB |
    | int8 | 196 MB | 843 MB |

    - The encoder holds no blocks after either attach.
    - The f16 form's 215 MB against 217 is a narrow pass: one block's float32 plus its f16 copy and GC lag.
    - The CPU fallback (the device tower detached) matches `LoadEncoder`'s float32 tower bit for bit.
    - Gemma 3's tower was 3.45 GB before (finding 2 above).
  - **G-S18c-Mac PASS** (`TestTowerReserve_everyDeviceTower` with Metal rows): nonzero for every Metal device tower,
    zero for the CPU's. Gemma 3's figures are 1291 MB (f16) and 932 MB (int8) against the measured 1172 and 843, about
    10% over and inside the ±25% bar. The old code priced every Metal tower at zero, so the Metal rows fail on it.
  - **Part 3** (`TestS18ShrinkCtxToFit`): with a budget between the 4096 and 2048 needs, an unpinned build goes
    resident at 2048 and prints "metal: KV plan: 1 conversation(s) x 2048 positions (context reduced from 4096 ...)".
    An explicit `-ctx 4096` still declines, and the ceiling's entry goes with the resident.
    - The test probes with a Metal load: a CPU load prices more, since it keeps arm64-repacked weights.
  - **The arithmetic for G-S18a now:** the decoder through the sidecar is about 2.9 GB plus KV (to be measured), and
    the int8 tower 0.84 GB, so about 4.0 GB at a 2048 context, against a 4.2-4.9 GB budget.
  - **G-S18a, the Mac cell: PASS, read 2026-10-08 17:57 by day** (`s18-mac/gs18a-run2/`; serve-metal at `95201adb`
    plus the fix below).
    - **The default arm** (`serve --model ~/models/gemma-3-4b-it --backend metal`, no sizing flags):
      - `decode path: metal-resident (int4)`;
      - `encoder f32/metal-resident` (the f16 Metal tower: the budget held it, so the default did not take int8);
      - "metal: KV plan: 2 conversation(s) x 4096 positions (1.2 GB left for the vision tower or drafter)";
      - `table.png`'s 32-token reply in 7.5 s.
    - **The hand-set arm** (`--model <the sidecar> --vision <dir> -ctx 2048 -kv-sessions 1 -vision-quant f32`): the
      same decode path and tower. Its reply and every token are **IDENTICAL**.
    - **The budget it ran under:** 8.5 GB reclaimable, in the evening with the owner's apps open. The int8 tower and
      the context shrink were not needed here. The tight case is G-S18g's tonight and the unit tests'.
    - **The sidecar:** built once with the guard bypassed on the owner's word ("so do this now"), in 32 s. Max RSS
      was 6.7-8.3 GB with the bf16 source mapped; swap grew 1-2 GB during each build.
    - **A defect the first run found, fixed before the second:**
      - On `--backend metal` the load's embed-int4 resolves OFF (the Metal resident takes no int4 head), so the
        sidecar a Metal load reads is `<dir>.int4.metal.giw`, not `.int4.e4h.metal.giw`. The first build used
        `-embed-int4`, and the default arm, finding no plain sidecar, tried to transcode one and was refused by the
        guard (`s18-mac/gs18a/`).
      - `resolveGemma3VisionQuantMetal` had the same fault: it asked about the `.e4h` sidecar, using the raw flag. It
        now reads `cfg.load.Options()`, and `TestResolveGemma3VisionQuant` checks what it asks for.
      - The two `.e4h` sidecars built today (the directory's and the GGUF's, 2.7 GB each) are read by no Metal load.
        (Gone from the Mac's `~/models` by 2026-10-09.)
  - **Owed:**
    - G-S18g (the served reply, int8 tower against f16 on the same decoder: night);
    - nothing else for the Mac half: G-S18a passed (above).
  - **G-S18g: FAIL as registered, read 2026-10-10** (the night of 2026-10-09, job `gs18g`, 3 min; `serve-metal` at
    `c2891555`, Gemma 3 4B from its directory sidecar). Raw:
    `docs/measurements/multimodal-support-2026-10/s18-mac/gs18g-2026-10-09/`.
    - Three arms on one decoder (`metal-resident (int4)`): the f16 tower, the int8 tower, the f16 tower again.
    - **The control holds:** the two f16 arms give identical replies on all four images.
    - **int8 against f16, first difference per image (f16's probabilities):**

      | image | token | f16 | int8 | p(f16's token) | p(int8's token) | near-tie |
      |---|---|---|---|---|---|---|
      | `glm_ocr/table.png` | 3 | " a" | " quarterly" | 0.830 | 0.170 | no |
      | `gemma3_preprocess_image.png` | 10 | " bright" | " green" | 0.495 | 0.324 | yes |
      | `qwen25vl_preprocess_image.png` | 9 | "," | " and" | 0.486 | 0.337 | yes |
      | `glm_ocr/formula.png` | 12 | " distribution" | " (" | 0.747 | 0.236 | no |

    - Two of the four first differences are not near-ties (the rule needs every one to be), so the gate fails.
    - **The registered consequence, owed in code:** the int8 tower is not a default. `resolveGemma3VisionQuantMetal`
      takes it today when the budget does not hold the f16 tower; that choice has to go, leaving `-vision-quant int8`
      as the explicit option and the f16 tower where it fits, or the CPU tower.
  - **G-S18g follow-up, 2026-10-10 by day (owner: "can we investigate before we just turn off int8"). Exploratory: not
    registered, one run each, the Mac in use.** Raw:
    `docs/measurements/multimodal-support-2026-10/s18-mac/gs18g-followup-2026-10-10/`.
    - **The gate's reference is not ground truth.** A third served arm, the exact float32 tower on the CPU
      (`-vision-device cpu`), same binary (`c2891555`), images and request. All three arms decode
      `metal-resident (int4)` and prefill the image the same way (this build predates the resident image prefill), so
      only the tower differs. Today's f16 and int8 arms repeat last night's byte for byte. First differences against
      the CPU tower:

      | image | f16 Metal tower | int8 Metal tower |
      |---|---|---|
      | `glm_ocr/table.png` | token 3, 0.844 against 0.156: not a near-tie | token 24, 0.400 against 0.295: near-tie |
      | `gemma3_preprocess_image.png` | token 12, 0.840 against 0.078: not | token 10, 0.731 against 0.206: not |
      | `qwen25vl_preprocess_image.png` | token 8, 0.771 against 0.213: not | token 8, 0.771 against 0.213: not |
      | `glm_ocr/formula.png` | token 2, 0.241 against 0.000: not | token 2, 0.241 against 0.000: not |

      **The f16 tower, serve's default, fails G-S18g's rule on four of four images when the exact tower is the
      reference; the int8 tower fails on three.** The rule separates neither tower from the other. This is S3's
      finding again (the decoder's sensitivity, read 2026-10-09 after G-S18g was registered): on Gemma 3 4B a tower
      change at cosine 0.9999999 already moves a greedy reply at a token the reference holds with p 0.84.
    - **Size, on S3's instrument** (`TestGemma3TowerDump` then `TestGemma3TowerSensitivity`, both extended with the
      int8 tower and a noise control per tower): the int4 decoder on the CPU, teacher-forced 32 steps along the CPU
      tower's own greedy path. Mean KL(CPU tower || arm), nats; the noise arms are the CPU tower's features plus
      Gaussian noise at that tower's per-soft-token relative L2, three seeds:

      | image | f16 tower | noise at its size | int8 tower | noise at its size |
      |---|---|---|---|---|
      | `glm_ocr/table.png` | 0.030 | 0.044-0.058 | 0.032 | 0.034-0.062 |
      | `glm_ocr/formula.png` | 0.084 | 0.070-0.080 | 0.149 | 0.079-0.102 |
      | `gemma3_preprocess_image.png` | 0.161 | 0.141-0.249 | 0.197 | 0.078-0.283 |
      | `qwen25vl_preprocess_image.png` | 0.058 | 0.086-0.138 | 0.420 | 0.095-0.136 |

      - **Feature space:** the f16 tower's mean relative L2 from the CPU tower is 4e-6 to 8e-6 per soft token; the
        int8 tower's is 0.016-0.018, with single soft tokens at 0.04-0.51 (`tower-dump.txt`).
      - **The decoder has a floor.** Noise 2,000-4,000 times larger moves it no more (0.03-0.28 at either size). The
        f16 tower sits at that floor on every image.
      - **The int8 tower sits at the floor on two images and above it on two:** `formula.png` (0.149 against 0.102 at
        most) and `qwen25vl_preprocess_image.png` (0.420 against 0.136 at most, worst step 7.1 nats). Its error is not
        noise-like there. Over the four images its mean KL is 0.199 against the f16 tower's 0.083.
      - **Limits:** four images, one prompt, three seeds, one run. The decoder ran on the CPU (the served arms decode
        on Metal). On the two gradient images the reply ends well before 32 tokens, so the later steps are forced past
        the end of the turn, for every arm alike.
    - **What turning the fallback off costs:** the CPU tower encodes these images in 25-34 s against 2.7-2.8 s on
      Metal; today's served requests took 30-40 s with it and 7.0-7.6 s with the int8 tower. The fallback is taken
      only when the budget does not hold the f16 tower.
    - **Reading:** G-S18g's FAIL is not evidence against the int8 tower, because the tower it would be replaced by
      fails the same rule. The int8 tower does cost more than the f16 one on some images (about 2.4x its mean KL on
      these four), by an amount this exploratory run sizes but does not grade.
    - **For the owner:** the registered consequence (int8 explicit-only) rests on a rule that cannot grade it. The
      alternative is to keep the fallback and register a graded replacement for G-S18g on this instrument: more
      images and prompts, the noise control, a margin written first, at night. Nothing in the code has changed.
  - **Amendment to G-S18g, 2026-10-10 (owner: "do your recommendation").** G-S18g's registered consequence ("if it
    fails, the int8 tower is not a default") is withdrawn. Its rule cannot grade a tower on Gemma 3: the f16 tower
    fails it against the exact tower (the follow-up above). The int8 fallback stays as it is until G-S18g2 reads.
    G-S18g's result stands in the record as FAIL as registered.
  - **G-S18g2, registered 2026-10-10 before it runs: is the int8 Metal tower fit to stay the tight-memory default?**
    - **Instrument:** `TestGemma3TowerDump` (the CPU float32 tower, the f16 Metal tower and the int8 Metal tower's
      projected features for one image), then `TestGemma3TowerSensitivity` (Gemma 3 4B's int4 decoder on the CPU,
      through its sidecar, teacher-forced along the CPU tower's own greedy reply). The graded positions run to the
      reply's end of turn, 32 at most.
    - **Units:** 24. Twelve committed images, each with two prompts.
      - Images: the four F2a images; `qwen25vl_preprocess_image_resize.png`; `qwen35vl_preprocess_image.png`;
        `glm_ocr/invoice.png`; `glm_ocr/invoices/inv01.png` to `inv05.png`. Six of the twelve are invoices and four
        are synthetic gradients, so this is not a sample of ordinary photographs.
      - Prompts: "What does this image show? Answer briefly." (G-S18g's) and "Describe this image in one sentence."
    - **Statistics,** each with a 95% cluster bootstrap over the twelve images (10,000 resamples, seed 20261010):
      - **D:** the mean over units of the int8 tower's mean KL(CPU tower || arm) minus the f16 tower's, nats per
        position.
      - **A:** the f16 tower's argmax agreement with the CPU tower's reply minus the int8 tower's, positions pooled,
        percentage points.
    - **Bands, read on each interval's upper bound; the verdict is the worse of the two:**

      | | PASS | PARKED (to the owner) | FAIL |
      |---|---|---|---|
      | D, nats per position | at most 0.05 | above 0.05, at most 0.20 | above 0.20 |
      | A, points | at most 5 | above 5, at most 15 | above 15 |

      - 0.05 nats is the decoder's own floor on the least sensitive image in the follow-up (noise at either tower's
        size reads 0.03-0.06 there): an excess under it is not distinguishable from a perturbation of any size.
      - 0.20 nats is about what int4 quantization itself costs a decoder against a float reference (G-S10q-c read
        0.07 and 0.17): an excess above it is a second quantization as large as the first.
      - 5 and 15 points are G-S10q-c's bands.
    - **Consequences:** PASS: the fallback stays. FAIL: the int8 tower becomes explicit-only (`-vision-quant int8`),
      as G-S18g first registered. PARKED: the owner decides, with D, A and the cost of the alternative beside them
      (the CPU tower: 30-40 s a request against 7.0-7.6 s, the follow-up's served arms).
    - **Controls, per unit; a failure of either makes the run VOID:**
      1. The CPU tower's features run again as an arm read KL exactly 0 with full agreement.
      2. The CPU tower's features negated read at least 10 times that unit's f16 arm, and at least 0.5 nats.
    - **Reported, not graded:** per tower, three noise arms (the CPU tower's features plus Gaussian noise at that
      tower's per-soft-token relative L2), which show the decoder's floor on each unit.
    - **What the exploratory run predicts, said before the graded one:** on the four F2a images with the first prompt
      D was +0.116 (per image +0.002, +0.065, +0.036, +0.362), with positions forced past the end of turn on two of
      them. That is inside the PARKED band. The graded run may differ: it stops at the end of turn and has 24 units.
    - **Limits of the instrument:** the decoder runs on the CPU, while a served request decodes on Metal; the reply is
      scored by distribution and argmax, not by a judgement of its quality.
    - **Tier, cost, stopping rule:** night, the Mac. A fixed 24 units, no early stop. About 35 minutes, from the
      follow-up's own times (a dump about 40 s an image, a nine-arm unit about 57 s), queued at 60.
      `docs/measurements/multimodal-support-2026-10/s18-mac/run-gs18g2-night.sh`, graded by
      `scripts/gs18g2_grade.py`; binaries pinned at the commit that carries this registration.
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

##### G-S14c, Qwen3-ASR end to end (S14.3), registered 2026-10-08 before any code

Owner: "yes" to continuing after G-S14b. What the code reading shows is needed: (1) the decoder must load the checkpoint's layout (`thinker_config.text_config` nested; tensors under `thinker.model.*` and `thinker.lm_head.weight`; the family key `qwen3_asr`; the text model is exactly Qwen3, whose interleaved m-RoPE is plain RoPE when the three position components are equal, and for ASR they always are); (2) a causal soft-token prefill (the audio embeddings replace the `<|audio_pad|>` run in the input embeddings; unlike goinfer's image prefill for Gemma 3 there is no bidirectional block, and unlike its Qwen image prefill there is no m-RoPE grid); (3) the prompt layout `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`, N = `QwenASRTokenCount(valid)`, with an optional forced prefill `language <Name><asr_text>`; and the model's output form `language English<asr_text>transcription`.

- **G-S14c1, the decoder loads and its text path is the reference's.** On the tiny checkpoint (the decoder half of `testdata/qwen3asr-tiny`, new golden: last-position logits and an 8-token greedy continuation from transformers on a fixed token sequence): last-row logit cosine **>= 0.99999**, argmax and the continuation equal. On the real 0.6B in float32 against transformers float32 on the same kind of token sequence: cosine **>= 0.9999**, argmax and a 6-token continuation equal.
- **G-S14c2, the composition, tiny.** Audio samples -> the Go front end -> the Go encoder and projector -> the splice -> the Go decoder prefill, against transformers' `Qwen3ASRForConditionalGeneration` forward on the same prompt and the same features: last-position logits cosine **>= 0.99999** and an 8-token greedy continuation equal, on the three tiny clips (26, 69 and 91 audio tokens). **Planted defects, each red:** the splice one position early; the splice one token short (the last audio row dropped); the audio embeddings added to the pad embeddings instead of replacing them; the m-RoPE grid positions an image would get instead of plain positions (the wrong prefill reused for audio); the projector skipped (the tower output spliced raw).
- **G-S14c3, the composition, real, float32.** The 0.6B on the LibriSpeech clip: the greedy transcription from goinfer float32 on the CPU **byte-equal** to transformers float32's ("language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."), and the teacher-forced last-position logit cosine **>= 0.9999** at every generated position (HF's path forced through goinfer). A first divergence is a FAIL unless it is a near-tie under the project's R10 definition (p(other) >= half p(top)); that is recorded as PARKED.
- **G-S14c4 (registered when reached): word error rate at int4** against transformers float32 on the LibriSpeech clips of `hf-internal-testing/librispeech_asr_dummy` (73 short clips), and the record-only comparison with Gemma 4 E4B on the same clips. **G-S14c5 (registered when reached): served** `input_audio` transcription equals the library path's.
- **Prediction, written now:** G-S14c1 and c2 at cosine 1.00000 with the five planted defects far below 0.999; c3 byte-equal. The likeliest stumble is the tensor prefix and the nested-config flattening (the generic loader has several prefix special cases), then the end-of-sequence set (`<|im_end|>` and `<|endoftext|>` come from `generation_config.json`).
- **Tier and cost:** quick by day for c1/c2 and c3 on a 0.6B (a few minutes of CPU); the family also needs a capability-matrix row and a parity-manifest entry (new registry keys carry both), which is bookkeeping with tests, not a gate.

##### G-S14c1-3 read 2026-10-08: PASS on all three (decoder `qwen3_asr` family, `decoder/generate_audio.go`, `multimodal/qwen3asr.go`; tests `decoder/qwen3asr_test.go` and `qwen3asr_real_test.go`)

- **What was built:** the `qwen3_asr` registry family (the config flattened from `thinker_config.text_config`, the tensors found under `thinker.*` by one more prefix case in the generic loader, the architecture exactly `qwen3` with the checkpoint's m-RoPE section dropped); a causal soft-token prefill, `Model.prefillLogitsAudio`, and `Model.GenerateAudio` (CPU decode, no resident claim); the transcription prompt builder `multimodal.QwenASRPrompt`; the family's capability-matrix row (summary tag `audio`, which the site shows under "all" until it grows a pill), its manifest row and the README's family count (41).
- **G-S14c1, the text path: PASS.** Tiny: last-row logit cosine **1.000000000**, argmax and the 8-token continuation equal. Real Qwen3-ASR-0.6B (the checkpoint's own weights, unquantized, on the CPU) against transformers float32: cosine **1.000000000**, argmax 12095, the 6-token continuation [12095 13 576 6722 315 15344] equal.
- **G-S14c2, the composition on the tiny checkpoint: PASS.** Samples -> the Go front end -> the Go encoder and projector -> the splice -> the Go decoder: last-position cosine **1.000000000**, argmax and the 8-token continuation equal, on all three clips (26, 69, 91 audio tokens). **All five planted defects red:** the splice one position early (worst cosine 0.9930), one audio row short (0.9936), added to the pad embeddings (0.9613), a bidirectional image block (Gemma 3's prefill) (0.9966), the projector's GELU removed (0.4568); the first four also change the continuation.
  **Two of the registered defects were replaced, with the reason:** "m-RoPE grid positions" cannot differ here because the architecture carries no m-RoPE (that is the finding that made the family plain `qwen3`), so the wrong-prefill defect is the bidirectional block instead; and "projector skipped" cannot splice (the tower's width is not the decoder's), so it is the projector's GELU removed, through the encoder's own seam.
- **G-S14c3, the composition on the real checkpoint: PASS.** The LibriSpeech clip through the Go front end (585 valid frames), encoder and projector (76 audio tokens, 1.1 s on this CPU), the splice and the decoder: **the Go-built prompt (91 ids) equals the processor's exactly**; with transformers' path forced, **26 positions at worst logit cosine 1.000000000, and no position where goinfer's argmax is not transformers' token**; the free-run greedy transcription (3.4 s, exploratory) is **token-for-token the same**: "language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." (transformers' ends with `<|im_end|>`, goinfer's stop id, not emitted). Logs `s14/gs14c-tiny.log`, `s14/gs14c-real.log`.
- **Bookkeeping a new family required, all done:** `testdata/parity_manifest.json` row (full-forward-oracle, 100.0%/1.00000, keys kept sorted: the refresh script writes them sorted and refuses a manifest whose diff is more than deps_hash); the loader edits restaled all 40 other validated families' hashes (the whole-file hash is deliberately conservative), so `scripts/refresh_parity_hashes.sh` ran: **64 forward goldens passed, 0 failed, 0 skipped, then 40 deps_hash lines refreshed and nothing else** (`Deps-Hash-Refresh: db4a3039`); the two tiny gates are required gates with ledger entries; the real gate is listed as not required with its reason (its reference needs a pin output directory, not a plain asset); the identity tool lists the family under "no tiny fixture it can read" with the reason.
- **Open, and found on the way:** (1) the Qwen3-ASR repo ships **no `tokenizer.json`**, only `vocab.json` and `merges.txt`, and goinfer's tokenizer reads the JSON; the reference script generated one with transformers (written into `~/models/qwen3-asr-0.6b`), but a user who pulls the repo and serves it gets no tokenizer. Reading vocab and merges directly is part of making this servable. (2) `pull` cannot yet fetch this family: its model_type is not in `pull`'s generative list until the matrix row ships in a build (it does now: the row exists), but the nested layout is also not what the loader in `serve --model <dir>` has been shown to open end to end. (3) Served `input_audio` (G-S14c5) and the WER gate (G-S14c4) are not built.
- **Limits:** one real clip of speech; float32/native weights only (no int4, no GPU); a CPU decode of 0.6B is fast but nothing here says how the 1.7B behaves; the front end's float32 reference spread (up to 4.4e-05) was inside the composition on the real clip (cosine 1.000000000 there), not yet on noisy audio.

##### G-S14c5, served `input_audio` transcription, registered 2026-10-08 before the served run

Built first (and gated by the tests that exist): the tokenizer assembled from `vocab.json` + `merges.txt` when a Qwen2-class repo has no `tokenizer.json` (`tokenizer.LoadVocabMerges`: ids and decodes equal to the tokenizer.json transformers builds from the same files on a 15-text corpus, on the committed Qwen2.5 tokenizer taken apart and on the real Qwen3-ASR files; its regex text held to Qwen2.5's declared pattern; other classes refused), and the serve wiring (`--model <qwen3-asr dir>` auto-detects `model_type qwen3_asr`, loads the encoder on the first clip, `input_audio` parts take the fixed ASR prompt, `driveVL` calls `GenerateAudio`).

- **The request:** `POST /v1/chat/completions` on `serve --model ~/models/qwen3-asr-0.6b --backend cpu`, one user message with an `input_audio` part (`format: wav`, base64) and no text, `temperature 0`, `max_tokens 64`, for the three LibriSpeech files in `testdata/speech/` (the 16 kHz original, the 44.1 kHz mono, the 48 kHz stereo).
- **PASS:** (a) the original's reply is **byte-equal** to the library path's greedy transcription (G-S14c3's, "language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."); (b) both resampled forms transcribe the clip with a word error rate of **0** against its reference words (case and punctuation ignored, "Mr." read as "mister", as G-S5c), and their replies are recorded verbatim; (c) the response's `usage.prompt_tokens` equals the prompt's length (91 for the original); (d) a request with no audio tower for the model (the same server asked a text-only message) still answers, and a clip over 600 s is refused with a message, not cut.
- **FAIL / PARKED:** any other text on (a) is a FAIL; a resampled form with WER above 0 is PARKED for the owner (the resample path is gated by G-S5e at the feature level, not at the word level for this model).
- **Prediction, written now:** (a) byte-equal; (b) WER 0 on both (the resampled references differ from the 16 kHz clip at the 1e-3 level, far under what moves a transcription); the likeliest stumble is the chat route treating a message with no text as empty.
- **Tier and cost:** quick by day (a few seconds of decode per request on the CPU; a minute for the server to load).

##### G-S14c5 read 2026-10-08: PASS, in two configurations (the tokenizer from vocab.json + merges.txt, the serve wiring, `scripts/s14c5_served.py`; logs `s14/c5/gs14c5-int4.log` and `gs14c5-f32.log`)

- **Setup:** `serve --model ~/models/qwen3-asr-0.6b --backend cpu`, once at the DEFAULTS (int4 weights, int4 head) and once at `--quant f32 --embed-int4=false`. Those two runs had the `tokenizer.json` I generated beside the checkpoint, so they did not exercise the vocab+merges loader; **a third run did**: the same server on `~/goinfer-bench/qwen3-asr-notokjson`, a directory of symlinks to the repo's own eight files (no `tokenizer.json`), loaded without complaint and gave the same replies byte for byte (the 16 kHz original's, the 44.1 kHz clip's, and the stereo clip's with the same int4 control token), prompt_tokens 91.
- **(a) PASS in both:** the 16 kHz original's reply is byte-equal to the library path's: "language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." (int4 and float32 alike). **(c) PASS:** `usage.prompt_tokens` 91.
- **(b) PASS in both:** the 44.1 kHz mono form's reply is identical to the original's; the 48 kHz stereo form's WER is 0 in both. **An int4 finding:** in the int4 run the stereo clip's reply is `<|im_start|>language English<asr_text>Mr. Quilter ...`, a control token before the language tag (the words after `<asr_text>` are right, so the WER is 0, but the raw text carries it); at float32 the reply is clean. So it is the quantization on that input's first-token choice, not the audio path.
  Not fixed here: stripping or forbidding control tokens in the output is a decoding constraint the reference does not have, and how often int4 does this is exactly what G-S14c4's 73 clips will count.
- **Exploratory speed:** 2.2-2.8 s per request for the 5.9 s clip at int4 (encoder, prefill of 91 tokens, 25 tokens of decode), 4.5-4.7 s at float32, on this CPU.
- **Corrections to my own checker, recorded:** its first draft compared lowercased replies against an uppercase reference and reported WER 1.000 on texts that were right; it is fixed (the reference words are lowercase) and the figures above are from the fixed script. No bar moved.
- **Not covered:** the request path with text in the user turn (ignored by design), a clip over 600 s (the refusal is in the code, no test), several audio parts per message, streaming, and a GPU. A text-only message to the ASR model answers `language None<asr_text>` (int4) or echoes the text (float32); it is an ASR model.

##### G-S14c4, word error rate at int4, registered 2026-10-08 before any run

- **Data:** the 73 clips of `hf-internal-testing/librispeech_asr_dummy` (LibriSpeech clean validation, CC BY 4.0), converted to 16 kHz mono 16-bit WAV with ffmpeg, with the dataset's transcripts. Not committed (the clips are fetched by `scripts/s14c4_data.py`); the WAV conversion's tools and versions are written beside them.
- **Arms:** **R** transformers 5.15.0 float32 (the converted reference of `scripts/pin_qwen3asr_real.py`), greedy, no forced language, 256 new tokens at most. **F** goinfer's library path at the checkpoint's own weights (native, the CPU): Go front end, encoder, projector, soft-token splice, `GenerateAudio`. **I** the same path at serve's defaults (`Quant int4`, int4 head). **S** a control: F with each clip's audio replaced by the NEXT clip's (the last takes the first's), same prompt. **D** (record-only, no bar): Gemma 4 E4B (the QAT GGUF with its tower, CPU, served) asked "Transcribe this audio." on the same clips.
- **Metric:** corpus word error rate (word edit distance summed over clips, divided by reference words). Reference and hypothesis are normalised the same way: the `language <Name><asr_text>` prefix and any `<|...|>` control token removed from the hypothesis, then lowercased, `Mr.`/`Mrs.`/`Dr.` read as the LibriSpeech words (mister, missus, doctor), hyphens as spaces, every other character outside letters, digits and apostrophes dropped, whitespace collapsed. (Not Whisper's English normalizer: its spelling table is a separate download, and this gate is relative, arms against arms, so a fixed simple normalizer is enough; every arm pays the same spelling and number-word errors.)
- **G-S14c4a, goinfer float32 is the reference:** F's transcription byte-equal to R's on **at least 95% of clips (70 of 73)**, and |WER_F - WER_R| **<= 0.3 points**. Every unequal clip is listed with its first differing word.
- **G-S14c4b, int4 non-inferiority:** WER_I - WER_F (points, paired by clip, cluster bootstrap over clips, 10,000 resamples, seed 20261008). **PASS:** the point estimate **<= +0.5** and the interval's upper bound **<= +1.5**. **PARKED:** the point estimate in (+0.5, +1.5]. **FAIL:** above +1.5. **Also PARKED whatever the WER:** more than **3 of the 73** int4 replies carry a control token (the stray `<|im_start|>` G-S14c5 found on one input), because the owner decides whether serve forbids them. Reported: clips where I's text differs from F's, and the control-token count in each arm.
- **Controls:** (1) F against F is exactly 0 points; (2) S's WER must be **above 80%** (a pipeline fed the wrong audio cannot score), which shows the metric sees a broken pipeline; (3) R's, F's and I's WER are each printed against the transcripts, so a gate that agrees with itself at a high error rate is visible.
- **Prediction, written now:** WER_R and WER_F near 2-4% on these clips (short clean read speech); F byte-equal to R on 90-100% of clips; WER_I - WER_F inside +0.5 points (probability about 0.6; int4 mostly flips near-ties on punctuation and capitalisation, which the normaliser removes); control tokens in at most two clips; S above 95%; D (Gemma 4 E4B) worse than the Qwen3-ASR arms, by how much I do not predict.
- **Tier and cost:** night. HF float32 on 73 clips (about 6-8 min), goinfer F (about 6 min), I (about 3 min), S (about 3 min), Gemma 4 E4B served on the CPU (20-25 min); estimate 55 minutes, queued at 90. The harness is checked by day on 3 clips.

#### S14.4, part (c): the Whisper encoder, then Voxtral Mini, then pure-Go Whisper. Stages and G-S14d registered 2026-10-08 before any code

Desk read, 2026-10-08 (transformers 5.15.0 `models/whisper`, `models/voxtral`; HF repo metadata fetched the same day):

- **The Whisper encoder (`WhisperEncoder`):** the 128- or 80-bin features [mels, 3000] (the S14.1 front end, padded to 30 s: the encoder REFUSES any other length) go through `Conv1d(mels -> d, k3, p1)`, exact-erf GELU, `Conv1d(d -> d, k3, s2, p1)`, GELU; the result [1500, d] gets the checkpoint's own learned `embed_positions` (1500 x d; sinusoids at initialisation, a stored parameter in the file) ADDED; then `encoder_layers` pre-LayerNorm layers (biased `q_proj`, `v_proj`, `out_proj`, **no bias on `k_proj`**, GELU MLP, biased LayerNorms, attention over all 1500 positions, no windows), then a final `layer_norm`. There is no attention mask.
- **Voxtral's audio tower is the same module** (`VoxtralEncoder`: same convs, same fixed length, same layers; an unused `avg_pooler` is built and never called) and its projector `VoxtralMultiModalProjector` is `Linear(5120 -> 3072, no bias)`, GELU, `Linear(3072 -> 3072, no bias)`, applied to the encoder output **reshaped to rows of 5120 = 4 frames x 1280** (so 1500 frames become 375 audio tokens per 30 s chunk). The decoder is `llama` (goinfer has it). The repo `mistralai/Voxtral-Mini-3B-2507` is ungated, Apache-2.0, 4.68 B parameters in HF-format safetensors, and ships **`tekken.json` (Mistral's tokenizer) and no `tokenizer.json`**: a tokenizer gap of the Qwen3-ASR kind, bigger (a different tokenizer family). Clips over 30 s are chunked by the processor.
- **Whisper's decoder** is its own class (cross-attention to the encoder's output in every layer, learned positions of 448, tied head, a 51,866-token BPE); its decode policy (language and task tokens, timestamps, long-form windows, temperature fallback) is where most of "pure-Go Whisper" lives.

**Stages, in the owner's order:** **S14.4a** the Whisper encoder in aikit (any Whisper size, 80 or 128 mels); **S14.4b** Voxtral Mini 3B (the encoder, the stacked-frames projector, the `llama` decoder, the Tekken tokenizer, 30 s chunking); **S14.4c** the Whisper cross-attention decoder, short-form greedy, no timestamps; **S14.4d** the decode policy (timestamps, long-form, fallback). Each stage's gates are registered when it starts; only S14.4a is registered here.

##### G-S14d, the Whisper encoder (S14.4a), registered before any code

- **G-S14d1, tiny.** A random-weight Whisper encoder from transformers' own class (d_model 32, 2 layers, 2 heads, ffn 64, 80 mel bins, `max_source_positions` 1500 so the input is 3000 frames), weights sized so the planted defects can be seen (the lesson of G-S14b2), written as a committed fixture with a golden of the encoder output [1500, 32] on two clips (features from the S14.1 front end applied to synthetic signals, taken from transformers' own extractor). Bar: per-frame cosine **>= 0.99999 and max |diff| <= 2e-05** (both criteria from the start, as G-S14b2 learned). **Planted defects, each red:** no positional embedding; the positions added before the second convolution; `k_proj` WITH a bias; the first convolution's stride 2; tanh GELU; the final LayerNorm skipped; attention restricted to 8 s windows (Qwen3-ASR's habit carried over).
- **G-S14d2, real, small.** `openai/whisper-small` (241.7 M parameters, Apache-2.0, 80 mels, d_model 768, 12 encoder layers) in goinfer float32 against transformers float32 on the LibriSpeech clip and a 25 s synthetic signal: per-frame cosine **>= 0.9999** over all 1500 frames, max |diff| recorded; and the S14.1 front end feeding it (Go features, not the reference's) recorded beside. The reference is built the ordinary way (this checkpoint loads in transformers 5.15.0; verified by its own transcription of the clip, so a silently random-initialised reference cannot pass).
- **G-S14d3, real, large (night).** `openai/whisper-large-v3` (the encoder is Voxtral's shape: 1280, 32 layers, 128 mels): the same comparison on the clip. Registered now so that S14.4b does not start on an encoder checked only at the small size.
- **Prediction, written now:** tiny at cosine 1.00000000 with the seven defects far outside the bars; small at worst-frame cosine above 0.99999; the cost is the 1500-position attention (about 1.9 TFLOP for large-v3: tens of seconds on this CPU). The likeliest stumbles: the learned position table's name and shape in the checkpoint (`encoder.embed_positions.weight`), and the Conv1d weight layout [out, in, 3].
- **Tier and cost:** tiny and small by day (minutes); large-v3 at night with the S14.4b reference.

##### G-S14d1 and G-S14d2 read 2026-10-08: PASS (S14.4a done; aikit `audio/whisper_encoder.go`, goinfer `multimodal/whisper_encoder_test.go` and `whisper_encoder_real_test.go`, fixture `testdata/whisper-tiny-rand/`, `scripts/pin_whisper_tiny.py` and `pin_whisper_real.py`; G-S14d3, large-v3, is still to be read)

- **G-S14d1, tiny: PASS after two corrections made before any real-checkpoint reading.** The port matches transformers' WhisperEncoder at per-frame worst cosine **1.000000000 / 0.999999999** on the two clips (6.5 s and 25 s of synthetic signal), max |diff| 2.2e-05 and 1.3e-04. Reloading the saved fixture through `from_pretrained` gives the same tensors (all 89 compared equal); its forward differs from the pinned one by 4e-06, which is threading noise.
  - **Correction 1, the max-difference bar.** I carried over G-S14b2's 2e-05 and the 25 s clip read 1.29e-04. Measured before changing it: transformers' OWN float32 differs from a float64 evaluation of the same weights by **1.9e-05 and 1.16e-04** on the two clips (a second float32 run agrees with the first to 4e-06 and 7e-06), the same size as the port's difference from the reference, so the bar sat below the reference's own float32 error on a 1500-position attention. It is now **5e-04** (about 4x that noise; the weakest planted defect, tanh GELU, is 3.1e-02, 60x above). Cosine bar unchanged.
  - **Correction 2, a planted defect that was not one.** I had planted "a bias on k_proj". A bias on k adds the same q.b to every score in a query's row, and softmax is shift-invariant, so it changes nothing (the max difference of the "defect" was the noise floor, 1.27e-04). That is why Whisper's `k_proj` has no bias. It is replaced by "the attention scale omitted".
  - **All seven planted defects red:** no position embedding (cosine 0.691), positions shifted by one frame (0.487), attention scale omitted (0.858), tanh GELU (0.99995508 with max |diff| 3.1e-02), final LayerNorm skipped (0.883), 8 s attention windows (0.627), post-norm instead of pre-norm (-0.183). A length other than 3000 frames is refused.
- **G-S14d2, real, small: PASS.** `openai/whisper-small` (241.7 M parameters, 80 mels, 768, 12 layers; `~/models/whisper-small`) against transformers float32, which loaded with **zero missing and zero unexpected keys and transcribed the LibriSpeech clip correctly** ("Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.", asserted in the script), so the reference is the real model. Worst per-frame cosine over all 1500 frames: **0.999999995** (LibriSpeech, max |diff| 1.5e-03) and **0.999999935** (25 s synthetic, 6.0e-03), against the bar 0.9999; with the Go front end instead of the reference's features: 0.999999996 and 0.999999806. About 12 s for a 30 s chunk on this CPU while a gate was also running (exploratory). Log `s14/gs14d2-whisper-small-encoder.log`.
  The absolute differences are larger than the tiny's because these activations are larger (rms 1.6, with the large outlier dimensions Whisper encoders carry); the registered criterion for the real size is the cosine.
- **Not yet:** G-S14d3 (large-v3, the Voxtral-shaped encoder: 1280, 32 layers, 128 mels) is registered and not run; `openai/whisper-large-v3` is a 3 GB download and a reference of tens of seconds, queued for the S14.4b night with the Voxtral reference.

#### S14.4b, Voxtral Mini 3B: desk read and gates G-S14e, registered 2026-10-09 before any code (owner: "S14 (speech) code steps")

**A correction first.** The owner's choice was worded from my own stale list ("served `input_audio`, tokenizer-from-vocab and the WER harness"). All three exist and were graded (G-S14c5 PASS 2026-10-08; the tokenizer from `vocab.json` and `merges.txt` shipped with it; G-S14c4 re-read 2026-10-09). The S14 code that is actually open is this stage, S14.4b, which the 2026-10-08 registration lists next in the owner's own order ("b then c"). The int4 first-token problem in Qwen3-ASR is a finding with an open owner decision (above), not a code step.

**Desk read, 2026-10-09** (transformers 5.15.0 `models/voxtral`; `mistral_common` 1.12.0 in an isolated venv `~/goinfer-bench/s14e/venv`, not `g4venv`; repo metadata fetched the same day; the checkpoint is `mistralai/Voxtral-Mini-3B-2507`, ungated, Apache-2.0, 4,676,271,104 parameters, two bf16 shards of 4.98 and 4.38 GB):

- **The checkpoint layout (HF format, 762 tensors):** `audio_tower.*` (conv1, conv2, `embed_positions` 1500 x 1280, 32 layers, `layer_norm`; the same names as the Whisper encoder minus the `encoder.` prefix), `multi_modal_projector.linear_1.weight` and `linear_2.weight` (no biases), `language_model.model.*` and `language_model.lm_head.weight` (a SEPARATE head, not tied). `config.json` nests `audio_config` (hidden 1280, 20 heads, `intermediate_size` 5120, 128 mel bins, 1500 source positions, `scale_embedding` false) and `text_config` (llama: hidden 3072, 30 layers, 32 query / 8 KV heads, **`head_dim` 128, which is not hidden/heads (96)**, intermediate 8192, **`rope_theta` 1e8**, no scaling, rms eps 1e-5, vocabulary 131,072), `audio_token_id` 24, `projector_hidden_act` gelu.
- **The audio path:** the encoder output [1500, 1280] is reshaped to rows of `audio_config.intermediate_size` = 5120 = 4 consecutive frames (so 375 rows per 30 s chunk), then `linear_1`, exact GELU, `linear_2` into 3072. The rows replace the embeddings at the positions whose id is 24 (the count must match exactly; transformers raises when it does not).
- **Chunking and padding:** the processor pads each clip to the next multiple of 480,000 samples (30 s) and splits the 3000-frame mel along time into chunks of `max_source_positions` x 2 = 3000 frames; every chunk is one encoder batch row and yields 375 audio tokens.
- **The transcription request** (`mistral_common`'s `encode_transcription`, which transformers' `apply_transcription_request` calls): `<s> [INST] [BEGIN_AUDIO] [AUDIO] x 375 [/INST] lang : en [TRANSCRIBE]` for a language, and the same without `lang : en` when none is given (measured on a 2 s clip: 383 and 380 ids, 375 of them id 24; ids at the tail `4, 9909, 1058, 1262, 34` and `4, 34`).
- **The tokenizer is Tekken (`tekken.json`, no `tokenizer.json`):** 1,000 special tokens at ids 0-999 (`<unk>`, `<s>`, `</s>`, `[INST]`, `[/INST]`, `[AUDIO]` = 24, `[TRANSCRIBE]` = 34, ...), then 150,000 byte-level BPE entries `{rank, token_bytes (base64)}` at id = rank + 1000, of which only the first 130,072 are in the model's 131,072-id vocabulary (`default_vocab_size`), split by an o200k-style regex. It is tiktoken's algorithm (merge the adjacent pair whose concatenation has the lowest RANK), not an ordered merges list. A **tokenizer gap of the Qwen3-ASR kind but a different family**: goinfer has no Tekken reader.
- **Not in this stage:** the served `input_audio` route for Voxtral (an S14.4b follow-up with its own gate, as G-S14c5 was for Qwen3-ASR), clips over 30 s in the served path beyond the processor's chunking, the Whisper cross-attention decoder (S14.4c) and the decode policy (S14.4d).

**Gates, in the order they are built (each is one commit, tiny first):**

- **G-S14e1, the Tekken tokenizer and the transcription prompt.** A reader for `tekken.json` in `tokenizer/` and a prompt builder in `multimodal/` (`VoxtralPrompt(audioTokens, language)`). Reference: `mistral_common` 1.12.0's `MistralTokenizer` on the real `tekken.json`, pinned by `scripts/pin_tekken.py` into a gzip golden of about 70 strings (English prose, code, numbers, the multilingual set the repo already uses for Qwen, emoji, runs of spaces and newlines, a string containing the text `[INST]` and `<s>` (which must encode as plain text, not as the special ids), the empty string, a 3,000-character string) and the full request ids for `en`, `fr`, no language, and 1, 2 and 3 chunks. **Bars:** token ids EXACTLY equal on every string and every request; decode(encode(s)) equals s for every string; the model's id range respected (no id >= 131,072). **Planted defects, each must read red** (test seams in the manner of the earlier gates): no +1000 rank offset; the vocabulary not truncated at 130,072; greedy longest-match instead of lowest-rank merging; special-token strings parsed out of plain text; the o200k pattern with its `\p{M}` classes dropped.
- **G-S14e2, the composition on a tiny Voxtral.** A random-weight `VoxtralForConditionalGeneration` built by transformers' own class and committed as a fixture of well under 1 MB, in the real layout (so the loader's prefixes are the real ones), with the properties the real model has and the planted defects need: audio `intermediate_size` = 4 x its hidden size, a text `head_dim` that is NOT hidden/heads, a `rope_theta` far from 1e4, an untied head, `audio_token_id` inside the vocabulary. A golden pinned by `scripts/pin_voxtral_tiny.py`: the last-row logits and the 8-token greedy continuation after a prefill containing audio tokens, for two synthetic clips (one 6 s, one 35 s so the chunking path runs: two chunks, 750 audio tokens), through goinfer's Go front end, encoder, projector and `llama` decoder. **Bars:** worst per-row last-layer cosine >= 0.99999 over every audio row, last-row logit cosine >= 0.99999 and argmax plus the 8-token continuation equal, the max |diff| criterion registered with it from the start (G-S14b2's lesson). **Planted defects, each red:** the 4 frames not stacked (every frame projected alone, then truncated); the stack order interleaved across chunks; no GELU between the projector's two linears; the audio rows spliced one position late; the splice keyed on the wrong id; the head dimension taken as hidden/heads; `rope_theta` left at the llama default; the projector given a bias. A clip whose padded length is not a multiple of 30 s is padded, not refused.
- **G-S14e3, the composition on the real checkpoint (the long step, by night).** Voxtral-Mini-3B-2507 in goinfer float32 on the CPU against transformers float32 (the official class and `apply_transcription_request`), on the LibriSpeech clip: the greedy transcription **byte-equal**, and the encoder output, projector output and prompt-row logits compared per row (cosine >= 0.9999 over all 375 rows, the first and last rows named). **The reference is proved real the way G-S14d2's was:** zero missing and zero unexpected keys, and the transcription asserted in the script ("Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."), because a nested checkpoint can load into a silently random-initialised model (the Qwen3-ASR trap). **This also settles G-S14d3 as registered:** the Voxtral tower IS the large-v3 shape (1280, 32 layers, 128 mels) and is the weights that will be used, so the encoder check runs on it directly, in situ, and the 3 GB `whisper-large-v3` download is not needed. (An amendment to G-S14d3, written before it is read.)
- **Prediction, written now:** G-S14e1 exact on every string in the first build with the planted defects each over 1,000 ids wrong on at least one string (the likeliest stumble: the regex's `\p{M}` handling and the one-character-at-a-time rank search on the 3,000-character string's speed); G-S14e2 at cosine 1.00000000 and every planted defect far under 0.999 (the likeliest stumble: the loader's nested prefixes and an explicit `head_dim`, the two things `llama` has not been loaded with here); G-S14e3 byte-equal with the encoder at cosine >= 0.99999 (`rope_theta` 1e8 in float32 is the one place I expect a last-digit difference, and the cosine bar absorbs it; the byte-equal bar might not).
- **Tier and cost:** e1 and e2 by day (tokenizer: about 3 hours; the loader and composition: about 4 hours); e3 by night: the 9.4 GB download is done by day (network, not timed), the run is about 15 minutes of CPU float32 (about 6 TFLOP of prefill and encoder, 19 GB resident, then a short decode) and the reference run is of the same size, in a separate process so the two never hold 19 GB each at once. Queued after e1 and e2 pass.

##### G-S14e1 read 2026-10-09: PASS, in both halves, after one amendment made before the first reading and one gap the first draft of the golden had (`tokenizer/tekken.go`, `split_o200k.go`, `splitshape.go`, `tekken_test.go`; `multimodal/voxtral.go`, `voxtral_test.go`; `scripts/pin_tekken.py`; `testdata/tekken_golden.json.gz`, 11.5 KB)

- **Amendment A1 (before the first reading; the bars did not move).** Planted defect 5 is "the o200k walker (contractions attach, digit runs of three) instead of Tekken's" in place of the registered "the o200k pattern with its `\p{M}` classes dropped": the Tekken regex has NO attached contractions and a single-digit `\p{N}`, which is the real, plausible mistake (the pattern LOOKS like o200k's), whereas the marks are the same class text in both. The walker is `splitO200kVariant(s, contractions, maxDigits)`, with `splitO200k` (true, 3) unchanged and `splitTekken` (false, 1) new, and `classifySplit` names the new shape `tekken`.
- **The tokenizer: exact on every string (463 of 463),** including the 3,000-character one: ids equal to `mistral_common` 1.12.0's, decode(encode(s)) equal to s, no id at or above 131,072, BOS 1 and EOS 2. It is built in memory from `tekken.json` (14.9 MB, sha256 asserted equal to the golden's; not committed): ids 0-999 the specials, ids 1000 up the first 130,072 ranks, a merge list derived from the ranks the way transformers' `TikTokenConverter` does (every multi-byte token is the merge of the two parts its own rank splits into), `ignore_merges` on.
- **A gap in the first golden, found by its own planted defect.** With 63 hand-written strings the o200k-walker defect read **0 of 63 differ**, i.e. the gate was blind to it, and a probe found why: the two pre-tokenizers differ only on strings that glue a contraction onto a word or sit digits against letters (`THEY'REx`, `we'redon't`: 68 of 4,000 random glues differ), which natural prose never produces and the vocabulary otherwise hides (Tekken has no multi-digit tokens). The golden now carries 400 seeded fuzz strings (`random.Random(20261009)`, 26 fragments) pinned from `mistral_common` beside the 63; the same defect now reads **11 of 463**, and the tokenizer is still exact on all of them. So the pre-tokenizer is checked against the reference on adversarial input, not only against prose.
- **All six planted defects red:** no rank offset (462 of 463 differ) and an offset of 100 (462), the vocabulary not truncated at 130,072 (7), merges ordered by length instead of rank (74), special-token spellings parsed out of plain text (4), the o200k walker (11).
- **The prompt builder `multimodal.VoxtralPrompt`:** the ids equal `mistral_common`'s `encode_transcription` on all **9 requests** (2 s, 35 s and 65 s: 1, 2 and 3 thirty-second windows and 375, 750 and 1,125 placeholders; languages `en`, `fr` and none). Note for S14.4b: `mistral_common` returns ONE audio array per request padded to a whole number of 30 s windows, not one per window. **Six planted defects red** (no BOS 9 of 9; `[INST]`/`[/INST]` swapped 9; language without the colon 6; one placeholder too few 9; one too many 9; no `[TRANSCRIBE]` 9), and floor instead of ceil for the window count is caught by the 35 s and 65 s clips.
- **Prediction scored (written before):** exact on every string in the first build (held); every planted defect over 1,000 ids wrong on at least one string (**the bar was a poor one: counts are of strings, not ids; the o200k defect was invisible until the fuzz set**); the likeliest stumbles (the regex's `\p{M}` handling and the speed on the 3,000-character string) did not occur (0.7 s for the whole golden).
- **Cost:** about 2.5 hours against the 3 estimated. Still open for S14.4b: G-S14e2 (the tiny composition) and G-S14e3 (the real one, by night); the 9.4 GB checkpoint is downloading into `~/models/voxtral-mini-3b-2507` (network, not timed).

##### G-S14e2 read 2026-10-09: PASS (the composition on a tiny Voxtral), with a loader bug it found, a fixture-layout correction, and one disclosed blind spot (`decoder/voxtral_test.go`, `decoder/registry.go`, `decoder/weights.go`; aikit `audio/voxtral.go`, `whisper_features.go`, `whisper_encoder.go`; `scripts/pin_voxtral_tiny.py`; `testdata/voxtral-tiny/`, 2.1 MB)

- **The result.** transformers' `VoxtralForConditionalGeneration` against goinfer's whole chain: Go front end (one STFT over the whole padded signal, then split into 3000-frame windows) -> Go tower -> four-frame stacking -> projector -> soft-token splice -> Go `llama` decoder.
  - **6 s clip** (1 window, 375 audio tokens): worst per-row projector cosine **0.999999999**, max |diff| 3.8e-4; last-position logit cosine **0.999999999**, argmax and the 8-token greedy continuation equal.
  - **35 s clip** (2 windows, 750 audio tokens): worst row cosine **0.999999999**, max |diff| 6.0e-4; logit cosine **0.999999999**; argmax and continuation equal.
  - **The text-only path** (the loader, the explicit `head_dim` 16 that is not hidden/heads 12, `rope_theta` 1e8, the untied head): last-row cosine **1.000000000**, continuation equal.
  - The max |diff| bar on the projector rows is 4e-3, written into the test before the first reading and not moved.
- **A real loader bug, found by the text gate and fixed.** The first reading of the text path was cosine **0.15**. Per-layer differencing (the residual after the embedding and after each layer: cosine 1.000000 at all three) put it after the last layer, and logits recomputed in transformers from goinfer's own last-layer residual matched the TIED head (cosine 0.98) and not the file's (0.16): `decoder/weights.go` looked the output head up as `lm_head.weight` WITHOUT the checkpoint's top-level prefix, so an untied head at `language_model.lm_head.weight` was never found and the embedding silently became the head. It never mattered for the two prefixed families that existed (Gemma 3 VL and Qwen3-ASR both tie their heads). It would have made the real Voxtral Mini 3B emit garbage with every other gate green. **Fix:** one line, `embedTable(topPrefix+s.LMHead, ...)`. **Untested, flagged:** a Qwen3-ASR checkpoint with an UNTIED head (the 1.7B, if it carries `thinker.lm_head.weight`) was affected the same way before this fix; only the 0.6B (tied) has been run.
- **A fixture-layout correction, found by reading my own claim.** I had written that the fixture is "in the real checkpoint's layout"; transformers 5.15 saves the text tensors as `language_model.model.model.*` (an extra `.model`), and the hub checkpoint has `language_model.model.*`. The pin script now renames to the hub layout and the reload through `from_pretrained` (audio max |diff| 1.5e-05, text 2.6e-06) proves transformers reads that layout and gets the same model.
- **Planted defects.** Registered and red: the four frames not stacked (worst cosine -0.009), the windows in the wrong order (0.66), no GELU in the projector (0.05), a bias after `linear_1` (0.95), the splice one position late (0.34), the splice keyed on the wrong id (0.18), `rope_theta` at the Llama default (0.38), and the head dimension derived as hidden/heads (refused at load, loudly, because every attention tensor has the real shape). Added beyond the registration and also red: the second linear skipped (-0.37), the tower's positions omitted (0.29), one audio row short (0.61), the audio added to the pad embeddings (0.50).
- **A blind spot, disclosed and NOT tuned away.** Tanh-GELU in the projector (a defect I added) reads cosine 1.000000 and max |diff| **2.2e-3** against the 4e-3 bar, and the logits and continuation do not move at all. Moving the bar to catch it would be moving it with the data in hand; it is recorded as a blind spot of this composition gate (the test logs it and asserts only that the seam changes the rows). A projector-only gate with a max |diff| bar near 1.5e-3 would see it; it is not built.
- **Two additions G-S14e2 needed that the registration did not name:** `audio.WhisperFeaturesWindows` (aikit; the whole-signal front end, because the clamp's maximum and the reflect padding are the whole padded signal's and not each window's), and Voxtral's config spelling and `audio_tower.` prefix in the shared encoder loader. Neither changes a result of the gated Whisper path (aikit's tests pass).
- **Prediction scored (written before):** G-S14e2 at cosine 1.00000000 with every planted defect far under 0.999 (held for the registered ones; the tanh-GELU defect I added is the exception); the likeliest stumble "the loader's nested prefixes and an explicit `head_dim`" (the prefixes were fine; the head's prefix was the bug).
- **Still open:** G-S14e3 (the real checkpoint by night; the 9.4 GB is on disk at `~/models/voxtral-mini-3b-2507`), served `input_audio` for Voxtral, a GPU path. The aikit commit `e5d586e` is local and goinfer's push waits on an aikit tag, as for the earlier S14 pieces.

##### G-S14e3 read 2026-10-09, the graded case: PASS on the first run, every bar met by a wide margin (`decoder/voxtral_real_test.go` under `-tags realckpt`, `scripts/pin_voxtral_real.py`; reference `~/goinfer-logs/s14e/ref-smoke/`, gate log `~/goinfer-logs/s14e/gate-libri-smoke.log`; pinned binary `~/goinfer-bench/s14e/real/decoder-real.test` at `8494de25`; 115 s)

The real Voxtral Mini 3B (`mistralai/Voxtral-Mini-3B-2507`, shards sha256 `1facdc4c...` and `0a103ba7...`), goinfer float32 on the CPU, against transformers 5.15.0 / torch 2.12.0+cpu float32:
- **The reference is the real model.** It loaded with **zero missing and zero unexpected keys** and transcribed the LibriSpeech clip as "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." (23 tokens, 62 s), asserted in the script before anything was written.
- **The prompt:** the Go prompt (Tekken tokenizer on the real `tekken.json` and `multimodal.VoxtralPrompt`) equals the processor's 383 ids exactly (instrument).
- **The front end:** the Go features (one pass over the padded signal) against the processor's: worst cosine 1.000000000, max |diff| 1.7e-05.
- **The tower, 1500 rows:** worst row cosine **0.999999849** (row 309); first and last rows 1.000000000. **The projector, 375 rows:** worst **0.999999930** (row 77); first and last 1.000000000. Bar 0.9999 on each.
- **The prompt logits:** cosine **1.000000000**, argmax 27118 equal (bar 0.9999 and argmax equal).
- **The transcription: byte-equal.** 23 tokens against 23, no difference at any position, the decoded text identical; the stop token is `</s>` (2).
- **Teacher-forced:** the reference's own tokens through goinfer, 23 positions, worst logit cosine **1.000000000**, **0** positions where goinfer's argmax is not the reference's token (bar 0.9999).
- **Timings (exploratory, one run on an otherwise idle CPU):** tower and projector 55 s, prompt prefill 11 s, the whole gate 115 s.

##### G-S14e3, the record-only second case, read 2026-10-10 (night run `voxtral-real`, 23:31-23:43 PDT, queued at `8494de25`, pinned `~/goinfer-bench/s14e/real`; logs `~/goinfer-logs/night/runs/2026-10-09/voxtral-real.log`, `~/goinfer-logs/s14e/real-2026-10-09/`)

The graded case was run again on the night queue and reproduced its reading (23 tokens byte-equal, worst tower row 0.999999849, worst teacher-forced logit cosine 1.000000000), so the PASS above is a two-run PASS. The second case, **the clip six times with half a second of silence between (36 s, two 30 s windows, 750 audio rows, 758 prompt ids), registered as RECORD ONLY and never graded**, exercises the chunked front end and the multi-window tower on real weights:
- features, Go against the processor: worst window cosine 1.000000000, max |diff| 1.7e-05 (two windows);
- tower, 3,000 rows: worst row cosine 0.999999569 (row 314); projector, 750 rows: worst 0.999999578 (row 70); first and last rows 1.000000000;
- prompt logits: cosine 1.000000000, argmax 27118 equal; prompt prefill 18.8 s;
- free-run greedy: **133 tokens against 133, no difference at any position**, the six repetitions transcribed; teacher-forced worst logit cosine **0.999999996**, 0 positions where the argmax differs;
- the reference transcribed six sentences too (the transformers float32 run took 177 s for the 133 tokens).

Nothing here moves a bar; it is the record the registration asked for, and it reads clean. What it does not cover: a signal longer than two windows, and any speech but this one clip.
- **Prediction scored (written before):** byte-equal with the encoder at cosine >= 0.99999: held, and the one place I expected a last-digit difference (`rope_theta` 1e8 in float32) did not show: the logit cosines read 1.000000000. The e1/e2 stumbles (the untied head's prefix, which the tiny gate caught and fixed before this run) are the reason this one passed first time.
- **The manifest row** moves from `experimental` / `tiny-golden` to **`validated` / `full-forward-oracle`** at `8494de25` (a real-checkpoint T3 method; the capability matrix, the hardware matrix and the pull copy are regenerated). **Not claimed:** Voxtral is not served (no `input_audio` route, no Tekken wiring in `serve`), and no GPU path has been run; the matrix entry says so.
- **Settles G-S14d3 as amended:** the Voxtral tower IS the large-v3 shape (1280, 32 layers, 128 mels) and it matches transformers at worst row cosine 0.999999849 in situ, so the 3 GB `whisper-large-v3` download is not needed.
- **Still to read:** the RECORD-ONLY two-window case (the clip six times, 36 s, 750 audio tokens: chunking on real weights) is in tonight's queue as `voxtral-real` (est 45 min, about 20 expected); its readings are logged and never graded. The local aikit commits (`e5d586e` and the tower accessor) wait on an aikit tag before goinfer can be pushed.

#### G-S14e4, Voxtral Mini served (`input_audio`): registered 2026-10-10 before any code (owner: "all of it" for S14, in order: Voxtral served first)

**What is wrong now (measured today):** `serve --model ~/models/voxtral-mini-3b-2507 --backend cpu` transcodes the 3B in 58 s and then stops: `load tokenizer ... tokenizer.json: no such file`. The checkpoint ships `tekken.json` and no `tokenizer.json`, and `tokenizer.Load(dir)` knows only the JSON and (for Qwen3-ASR) `vocab.json` + `merges.txt`. Behind that there is no Voxtral audio path in `serve` at all (`internal/serveapp` has no reference to it).

**What is built, in this order.** (1) `tokenizer.Load(dir)` falls back to `tekken.json` when a directory has no `tokenizer.json` and no vocab/merges pair. (2) `serve` recognises `model_type: voxtral` and turns audio input on the way it does for Qwen3-ASR: the encoder (`audio.LoadVoxtralAudio`, CPU float32) loads on the first clip, the WAV (16-bit PCM, any rate, downmixed and resampled to 16 kHz mono) goes through `audio.WhisperFeaturesWindows(samples, 128)`, the prompt is `multimodal.VoxtralPrompt(tk, VoxtralAudioTokens(n), "")` (no language: the model detects it; the processor's own `apply_transcription_request(language=None)` ends `[/INST] [TRANSCRIBE]`, checked today: 380 ids for a 3 s clip against `language="en"`'s 383), and generation is `GenerateAudio`. A clip over 600 s is refused, not cut. The reply is the model's raw text; text in the user turn is ignored. (3) Not claimed: a text-only chat message to Voxtral (the checkpoint ships no chat template), a language parameter, the GPU.

**Gates.**
- **G-S14e4a, controls, by day.** (i) `tokenizer.Load(dir)` and `tokenizer.LoadTekken` of the same `tekken.json` give identical ids on a fixed list of strings, on the real file. (ii) The Go prompt with no language equals the processor's `language=None` ids on the registered clip (the reference ids are written by a tokenizer-only run of `scripts/pin_voxtral_real.py`, no model). (iii) On `testdata/voxtral-tiny`, **amended before it ran (2026-10-10): a unit test of `voxtralPrompt`, not an HTTP request**, because the fixture has no tokenizer and a real `tekken.json` is not committed (the test builds a Tekken file of the 256 byte tokens and 40 special tokens, the prompt's seven at their real ids): the served prompt's ids equal `multimodal.VoxtralPrompt`'s, the placeholder count is 375 per 30 s window, the prompt ends `[/INST] [TRANSCRIBE]`, the features equal the encoder's own output bit for bit, and a generation through it runs; planted defects, each red against it (run): the language word left in the prompt, the placeholder count off by one. The HTTP path is G-S14e4b's.
- **G-S14e4b, the real checkpoint, float32, graded, by day (about 5-8 min, not timed).** `serve --quant f32 --backend cpu` on `~/models/voxtral-mini-3b-2507` and the registered LibriSpeech clip (`testdata/speech/librispeech-1272-128104-0000.wav`, 5.9 s) as `input_audio`: **PASS** if the reply is byte-equal to transformers 5.15.0 float32's greedy generation for `apply_transcription_request(language=None)` on the same clip (the reference is `scripts/pin_voxtral_real.py` with a no-language switch, the model loaded with zero missing keys, the transcription asserted against the corpus text first), the stop token `</s>` ends it, and the served prompt token count is the processor's. **FAIL** otherwise.
- **G-S14e4c, record only, by night (about 20 min).** The same request on (a) the six-times clip (36 s, two windows, 750 audio tokens) at float32, byte-compared with the reference's 133 tokens; (b) serve's DEFAULT quant (int4, int4 head) on the single clip and the six-times clip, compared word by word with the float32 reply, with the first-generated-token class tabulated (a transcript word, a control token, a stop). Nothing is graded: int4 on Voxtral is unvalidated (Qwen3-ASR's int4 first-token problem is why this is looked at at all), and the default users get is int4, so the reading goes to the owner. The word-error gate for int4 (G-S14c4's shape, on a subset of the 73 clips because the tower is 55 s per clip on this CPU) is the next registration if (b) shows damage.

**Prediction (written now).** G-S14e4a (i)-(iii) PASS on the first build at 0.8; (iii)'s planted defects all red at 0.95 (the likeliest stumble is the tokenizer fallback's special-token handling, which the real-file comparison would show). G-S14e4b byte-equal at 0.85: the library path was byte-equal on this clip with `language="en"` (G-S14e3), and the only new thing is the empty-language prompt, which the same tokens end with on both sides. If the no-language reply differs from the English one, that is a reading about the model, not a defect, as long as serve equals the reference. G-S14e4c(b): int4 within three words of float32 on the single clip at 0.5; a control token or an empty first position in at least one of the two int4 replies at 0.3.

**Tier and cost:** (a), (b) by day (the tokenizer-only reference, the tiny fixture, one real request); (c) night, queued at 30. **What each result does next:** (b) PASS closes the served Voxtral claim as "serves audio on the CPU, float32 verified against the processor's own request; int4 and the GPU not claimed"; the matrix row and `docs/` support table are regenerated from code by S8's generator.

##### G-S14e4a and G-S14e4b read 2026-10-10 (by day, nobara; branch `s14c4d-partial-force`, code at `efc9815a`; reference and logs `~/goinfer-logs/s14e4/`)

- **The code, as registered:** `tokenizer.Load(dir)` falls back to `tekken.json`; `serve` recognises `model_type: voxtral` (`loadVoxtral`) and turns on audio input the way Qwen3-ASR does (`voxtralPrompt`: the Whisper-style 128-mel windows, the tower and projector lazily on the first clip, `multimodal.VoxtralPrompt` with no language, `GenerateAudio`), a clip over 600 s refused. Nothing changed for any other model.
- **G-S14e4a: PASS, first build.** (i) `Load(dir)` on a directory holding only the real `tekken.json` gives mistral_common's ids on every golden string and `[AUDIO]` = 24 (`TestLoad_tekkenDirectory`). (ii) `VoxtralPrompt(tk, n, "")` equals transformers 5.15.0's `apply_transcription_request(language=None)` ids on all three lengths (3 s: 380 ids, the 5.9 s clip: 380, the 36 s clip: 755; `TestVoxtralPrompt_noLanguageMatchesProcessor`, reference `testdata/voxtral_nolang_ids.json`). (iii), amended before it ran (above): `TestVoxtralPrompt_wiring` and `_refusals` on the tiny fixture with a built Tekken file; the wiring test goes red under the two planted defects I ran (the language word left in the prompt; the placeholder count one over).
- **G-S14e4b: PASS.** `serve --quant f32 --backend cpu` on the real checkpoint and the registered 5.9 s clip as `input_audio`: the reply "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." is **byte-equal** to transformers float32's greedy generation for the no-language request (the reference loaded with zero missing and zero unexpected keys, 58 s); `prompt_tokens` 380 as the processor's; 22 completion tokens then `</s>` against the reference's 23 (serve does not count the stop token); `finish_reason` stop; 80 s for the request (tower, prefill, decode on the CPU). The registered prediction (byte-equal at 0.85) held. What it does not cover: int4, the GPU, a clip over one window (all G-S14e4c, by night), a language parameter, and a text-only message (the checkpoint's template comes up as `ministral`; not looked at).
- **Queued:** `s14e4c` (record only; est 40, queued at `891a51ba`): the 36 s clip at float32 against the reference, and serve's default int4 on both clips against float32.

#### G-S14f, the Whisper cross-attention decoder (S14.4c): registered 2026-10-10 before any code (owner: "all of it" for S14)

**Desk read (transformers 5.15.0 `models/whisper`).** `WhisperDecoder`: learned token embeddings `embed_tokens` [vocab, d] (no embedding scale for the checkpoints here: `scale_embedding` false) plus learned positions `embed_positions` [448, d] indexed by absolute position; then per layer, pre-norm: `self_attn_layer_norm`, causal self-attention (`q_proj` and `v_proj` and `out_proj` with bias, `k_proj` without; the query is multiplied by `head_dim ** -0.5` before the product), residual; `encoder_attn_layer_norm`, cross-attention (the same projections, queries from the decoder, keys and values from the encoder's last hidden state, no mask: all 1,500 frames), residual; `final_layer_norm`, `fc1`, exact-erf GELU, `fc2`, residual; after the last layer `layer_norm`, then the head, which is `embed_tokens` transposed (`proj_out` tied). The encoder side is aikit's `audio.WhisperEncoder` (S14.4a, validated). Generation (`generate`, short form, greedy, `return_timestamps=False`): the prompt is `<|startoftranscript|>`, a language token, `<|transcribe|>`, `<|notimestamps|>`; with no language given, `detect_language` runs the decoder over `<|startoftranscript|>` alone and takes the argmax over the language tokens; the logits processors are `suppress_tokens` (always) and `begin_suppress_tokens` (`[220, 50257]`, at the first generated position), both from `generation_config.json`; generation ends at `<|endoftext|>` (50257) or at `max_length` 448 counting the prompt.

**What is built.** `internal/whisper` (inside goinfer, not aikit: it uses aikit's public `embed` safetensors reader, `linalg` and `audio.WhisperEncoder`, so no aikit change and no tag): the decoder with a self-attention KV cache and the cross-attention keys and values computed once per request from the encoder output; `Detect`, greedy generation with the two processors and the stop rules; a `Transcriber` joining features, encoder, decoder and the tokenizer (`tokenizer.Load(dir)` reads Whisper's `tokenizer.json`) for ONE 30 s window. A clip over 30 s is refused by name (long form is S14.4d). No serve route here (S14.4d ends in `/v1/audio/transcriptions`, where timestamps surface). float32 on the CPU only.

**Gates.**
- **G-S14f1, tiny, by day.** `testdata/whisper-tiny-rand` (d_model 32, 2 decoder layers, 2 heads, ffn 64, vocab 160, 64 target positions: already holds the decoder's weights) with a new golden from transformers' own `WhisperDecoder` on the fixture's encoder output for its two clips (`scripts/pin_whisper_decoder_tiny.py`): teacher-forced logits at every position for a 12-token input, and a 12-token greedy continuation with no processors. **PASS** if the per-position worst cosine is >= 0.99999 and max |diff| <= 2e-3, the incremental (cached, one token at a time) logits equal the full forward's to 1e-5, and the greedy ids are exactly transformers'. **Planted defects, each red against it (run):** the query unscaled; positions shifted by one; cross-attention keys and values taken from the decoder's own hidden state; no final `layer_norm`; self-attention not causal; the `k_proj` bias the checkpoint does not have (a no-op for softmax, noted as the one defect that cannot be seen, as the encoder gate recorded); the head untied (a different matrix).
- **G-S14f2, real, `openai/whisper-small` (241.7 M, 80 mels, d 768, 12 + 12 layers; `~/models/whisper-small`), by day, one clip.** The registered LibriSpeech clip, goinfer float32 on the CPU against transformers float32 (loaded with zero missing keys, the transcription asserted against the corpus text first): **(a)** language detection gives `<|en|>`, as transformers'; **(b)** the logits at the last prompt position, cosine >= 0.9999 and the same argmax; **(c)** the greedy transcription **byte-equal** (token ids and decoded text); **(d)** teacher-forced on the reference's own tokens, worst logit cosine >= 0.9999 and zero positions where goinfer's argmax is not the reference's token. **FAIL** otherwise.
- **G-S14f3, WER, by night (record the estimate when it is measured; about 15 min).** The 73 LibriSpeech clips of G-S14c4: goinfer's `Transcriber` against transformers float32, each greedy with the language detected by each: **PASS** if byte-equal on at least 70 of 73 and |WER_goinfer - WER_transformers| <= 0.30 points (G-S14c4a's bars, same normaliser); the unequal clips are listed with their first differing word. Record only: the language each side detected on every clip.

**Prediction (written now).** G-S14f1 PASS on the first build at 0.7 (the likeliest stumble: the positions offset or the tied head's transpose); every planted defect red except the `k_proj` bias at 0.9. G-S14f2 (a)-(d) PASS at 0.8: the encoder is already at cosine 0.9999999 against transformers on the real small model (G-S14d2), so what is new is the decoder, and the generation rules are short; the likeliest miss is the begin-suppress rule applied one position early or late, which byte-equality would show. G-S14f3 PASS at 0.75; if 70 of 73 is missed, the first differing word names whether it is a numerical near-tie or a rule.

**Tier and cost:** f1 and f2 by day (a few minutes each; whisper-small's encoder is 12 layers at d 768 over 1,500 frames, seconds on this CPU); f3 at night. **What each result does next:** f1 and f2 PASS: the decoder is the stage's claim ("pure-Go Whisper, short form, greedy, float32, CPU"); f3 is its WER record. Then S14.4d (timestamps, long-form windows, temperature fallback, the `/v1/audio/transcriptions` route), registered when it starts.

##### G-S14f1 and G-S14f2 read 2026-10-10 (by day, nobara; branch `s14c4d-partial-force`; code `internal/whisper` at `df2523a2` and `20ff51dd`; reference `~/goinfer-logs/s14f/ref/`)

- **G-S14f1, tiny: PASS on the first build.** Against transformers' `WhisperDecoder` on `testdata/whisper-tiny-rand` for its two clips (golden `dec_golden.zip`, `scripts/pin_whisper_decoder_tiny.py`): worst per-position cosine **1.000000000**, max |diff| **5.3e-05** (bar 2e-3); the incremental (one token at a time, cached) logits equal the full forward's **exactly** (max |diff| 0); the 12-token greedy continuation equals transformers' on both clips. **Planted defects, all six red:** the query unscaled (cosine 0.861), positions shifted (0.268), cross-attention keys and values from the decoder's state (0.077), no final layer norm (0.833), self-attention not causal (0.379), the head untied (-0.108); the `k_proj` bias is the one defect no test can see, as registered. `TestGenerate_rules` also pins the generation mechanics on the tiny model (prompt order, suppress at every position, begin-suppress at the first only, stop token, length cap, unknown language and task refused, detection).
- **G-S14f2, `openai/whisper-small`, the LibriSpeech clip: PASS, all four parts, first run (29 s).** The reference loaded with zero missing keys and transcribed the clip, asserted against the corpus text: " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." (23 generated tokens incl. `<|endoftext|>`). **(a)** goinfer's language detection `<|en|>` (id 50259), the reference's. **(b)** the logits at the last prompt position: cosine **1.000000000**, argmax 2221 on both. **(c)** the greedy transcription: the 23 ids, the 4-token prompt `[50258 50259 50359 50363]` and the decoded text byte-equal to transformers'. **(d)** teacher-forced on the reference's tokens over 23 positions: worst logit cosine **1.000000000**, **0** positions where the argmax (after the same suppression) differs. The registered prediction (0.8) held; the begin-suppress rule I named as the likeliest miss is exercised by the mechanics test and not distinguished by this clip (its first token is not in the begin-suppress list).
- **What this establishes:** a pure-Go Whisper decoder at float32 on the CPU, short form, greedy, one 30 s window, agreeing with transformers to cosine 1.0 on whisper-small. **Not established:** any language but English, any clip but this one (G-S14f3 reads 73), a long clip (S14.4d), the GPU.
- **Queued:** `s14f` (G-S14f3; est 40, queued at `55dbc0d8`): the 73 clips, byte-equal on at least 70 and |WER diff| <= 0.30.

#### G-S14g, Whisper timestamps and long-form (first half of S14.4d): registered 2026-10-10 before any code (owner: "all of it")

**Scope, split on purpose.** S14.4d is four things: timestamps, long-form windows, the temperature fallback (with its thresholds and conditioning on earlier text), and the `/v1/audio/transcriptions` route. This registration is the first two, because they are deterministic and can be held to transformers byte for byte. The fallback re-samples at a higher temperature, which no seed can make match transformers' random stream, so it gets its own registration (G-S14h) with a different instrument; the route is G-S14i. Nothing below touches sampling.

**Desk read (transformers 5.15.0, `generation_whisper.py`, `logits_process.py`; the reference is its `generate(return_timestamps=True, return_segments=True)` with no `temperature`, no thresholds, `condition_on_prev_tokens` off).**
- **Features for a clip over 30 s are not the short-form's.** The extractor is called with `truncation=False, padding="longest", return_attention_mask=True`: no padding to 30 s, the log-mel of the WHOLE clip (the centre padding reflects at the true ends, the clamp's maximum is the whole clip's), the last STFT frame dropped, so T = floor(n / 160) frames. Window k is frames [seek, seek + 3000) of those, padded with ZEROS in the feature domain (not with the log-mel of silence) when shorter. aikit's `WhisperFeatures` pads the waveform to 30 s and `WhisperFeaturesWindows` pads it to a multiple of 30 s (Voxtral's processor), so neither is this; the 150 lines of log-mel are ported into `internal/whisper` (not into aikit: no release is needed) and held to the extractor.
- **Timestamps.** The prompt omits `<|notimestamps|>` (`<|startoftranscript|> <|lang|> <|transcribe|>`; `begin_index` is its length). The processors run in this order, each seeing the previous one's output: `begin_suppress_tokens` (first generated position only), `suppress_tokens`, then `WhisperTimeStampLogitsProcessor`: `<|notimestamps|>` is -inf; over the tokens generated so far, if the last was a timestamp and the one before was too, no timestamp may follow; if the last was a timestamp and the one before was not, no text token (ids below the stop token) may follow; timestamps may not decrease (the lower bound is the last timestamp, or last + 1 unless it just opened a pair); at the first generated position everything below `timestamp_begin` is -inf and nothing above `timestamp_begin + max_initial_timestamp_index` (50 on the OpenAI checkpoints) is allowed; then, on the log-softmax of the result, if the log of the summed probability over all timestamps exceeds the largest single text token's log-probability, every text token is -inf. `timestamp_begin` is `<|notimestamps|>` + 1.
- **Segments (`_retrieve_segment`).** Over the generated tokens (the stop token removed): a "single timestamp ending" is `[text, timestamp]` at the end; a segment ends wherever two timestamps are adjacent; the segment tokens run from one cut to the next, start = the window's offset + 0.02 x (the first token's timestamp index), end likewise from the last timestamp in the segment (the last segment's own trailing timestamp when the text ended on a pair). With no adjacent pair at all, the whole generation is one segment ending at the last timestamp (or at the window's length when there is none). The window then advances by the whole window when the text ended on a single timestamp, else to the last timestamp that closed a segment (`last_timestamp_pos * input_stride`, input_stride 2). `seek_num_frames = min(max_frames - seek, 3000)`; the window's time offset is `seek * 0.02 / 2`. The language is detected once, on the first window; every window uses the same prompt (no conditioning on earlier text).

**What is built (`internal/whisper`).** `LongestFeatures` (the port); timestamp-mode generation (the three processors, in that order); `Segments` (the rule above); `TranscribeLong` (the window loop, one request, greedy). float32, CPU, English or detected language as before. Not in this stage: `temperature` and the three thresholds (G-S14h), prompts and `condition_on_prev_tokens` (G-S14h), word-level timestamps (not built, not planned), the route (G-S14i).

**Gates.**
- **G-S14g1, the features, by day.** `LongestFeatures` against the extractor (`scripts/pin_whisper_longest.py`: `WhisperFeatureExtractor`, `truncation=False, padding="longest"`) on three clips: 5.9 s (T = 585), the 36 s clip (T = 3,813), a 70 s composite of LibriSpeech dummy clips with half-second gaps: the same shape, **max |diff| <= 1e-4 and mean |diff| <= 1e-6** (G-S14a's bars). Planted defects red: padded to 30 s as `WhisperFeatures` does, padded to a multiple of 30 s, the last frame kept, the clamp's maximum taken per window.
- **G-S14g2, the rules on random weights, by day.** A second tiny random-weight Whisper whose vocabulary has the real layout in miniature (3,110 ids: text 0-1,599, stop 1,600, start 1,601, three languages, translate, transcribe, `<|notimestamps|>` 1,607, 1,501 timestamp tokens from 1,608 so 0.00-30.00 s exists), with a `generation_config.json` carrying the language and task tables, the suppression lists and `max_initial_timestamp_index` (`scripts/pin_whisper_ts_tiny.py`), and transformers' own `generate(return_timestamps=True, return_segments=True)` as the reference on two clips: **7 s (short form) and 68 s (three windows)**. **PASS** if goinfer's segments equal transformers' exactly: the same count, the same token ids in each, the same start and end (to 1e-9: both are 0.02 x an integer index), the same final token sequence, the same detected language. Random weights make the processors' branches fire in every combination (adjacent pairs, single endings, empty windows), which a real checkpoint's clean speech would not. **Planted defects red:** the pair rule ignoring the penultimate token; no non-decreasing rule; no first-position restriction; the logprob rule off; suppress run after the timestamp processor; the window always advancing by 30 s; the single-ending case treated as a pair; the time offset not added.
- **G-S14g3, the real checkpoint, by day.** `openai/whisper-small` (float32, CPU) on (a) the registered 5.9 s clip with timestamps and (b) a 70 s composite of LibriSpeech dummy clips with half-second gaps, against transformers float32 `generate(return_timestamps=True, return_segments=True)`: **PASS** if for each the segments are identical (count, tokens, start and end) and the decoded text byte-equal. **FAIL** otherwise; a differing first segment names whether it is the features (g1), a rule (g2) or a numerical near-tie in the model.

**Prediction (written now).** G-S14g1 PASS after at most one correction at 0.7 (the likeliest stumble: the frame count, floor against ceil, and the reflect at the end); g2 PASS on the first run with every planted defect red at 0.6 (the rules are long and the order-of-processors defect may not be seen on random weights: it is registered as red, so if it is not red the gate is amended to say what cannot be seen, as the `k_proj` bias was); g3 PASS at 0.7 (a near-tie flipped by 1e-6 numerical noise over 30-100 tokens is the way it fails, and would be a reading about float32 order, not a rule).

**Tier and cost:** all by day (each a few minutes; whisper-small over 70 s is three encoder passes and about 120 decoded tokens). **What each result does next:** g1-g3 PASS: timestamps and long-form are the stage's claim ("pure-Go Whisper with timestamps and sequential long-form, greedy, float32, CPU, equal to transformers"); then G-S14h and G-S14i are registered.

##### G-S14g1, g2 and g3 read 2026-10-10 (by day, nobara; branch `s14c4d-partial-force`; `internal/whisper` at `3c9bac5c` and the commits before it; references `~/goinfer-logs/s14g/`)

- **G-S14g1, the features: PASS.** `LongestFeatures` against the extractor (`truncation=False, padding="longest"`) on the 5.9 s clip (T = 585), the 36 s clip (3,813) and a 70 s composite (7,446): the same shapes, **max |diff| 1.6e-05, mean |diff| 1.0e-07** (bars 1e-4, 1e-6). The attention mask counts exactly T frames. Planted defects red: padded to 30 s and padded to a multiple of 30 s (the shape differs), the last frame kept (the shape differs), no clamp (max |diff| 0.86), the clamp's maximum per window (max |diff| 0.049). No correction was needed. The registered prediction (PASS after at most one correction, 0.7) held.
- **G-S14g2, the rules on random weights: PASS.** `testdata/whisper-tiny-ts` (3,110-id vocabulary, the real layout in miniature) against transformers' `generate(return_timestamps=True, return_segments=True, task="transcribe")` on three clips (7 s, 8 s, 68 s; the long one is three windows): **the same segments (count, tokens, start and end to 1e-9), the same sequence, the same language on all three** (5, 7 and 5 segments; the 8 s clip ends a window on a single timestamp). Two corrections before it passed, both mine and both in how I called the reference or read it: the reference call omitted `task`, and transformers adds the task token only when it is passed (the prompt was `[start, lang]`, not `[start, lang, task]`); and a short clip's padded 3,000 frames are stepped through by `seek` like any other window (I had reused the unshifted features for the second window). **Planted defects:** seven are red on the clips (the pair rule ignoring the penultimate, timestamps allowed to decrease, no first-position restriction, the logprob rule off, the window always advancing 30 s, a single ending treated as a pair, no time offset). **One the random clips cannot see, as registered it might not be:** `suppress_tokens` run after the timestamp rule: 56 clips of two lengths were searched and none differed, because a suppressed id is rarely the best text id. `TestProcessLogits_order` covers it with a hand-built score vector in which the suppressed token leads the text and the timestamps' mass sits between it and the next token: red under the defect, and it says in its comment that it checks the order I read from `_retrieve_logit_processors` and not transformers' behaviour. The single-ending defect needed a third clip (found by search; it is in the golden), so the registered "two clips" became three.
- **G-S14g3, `openai/whisper-small`: PASS first run.** The registered clip (short form, 15 s here) and a 70 s composite of LibriSpeech clips (long form, three windows, 59 s): the 1 and 13 segments have **identical tokens and identical start and end times to 1e-9**, the sequences and the decoded texts are byte-equal, and the detected language is `<|en|>` on both. The registered prediction (0.7) held; the numerical near-tie I named as the likeliest failure did not occur.
- **What this establishes:** pure-Go Whisper with timestamps and sequential long-form (greedy, float32, CPU) equal to transformers on a real checkpoint and on random weights that fire every branch of the rules. **Not established:** the other checkpoints (large-v3, 128 mels), a language but English, a clip with real silences and music (the composite has half-second gaps only), the fallback and thresholds (G-S14h), the route (G-S14i).

#### G-S14h, Whisper's fallback, thresholds, silence skipping and conditioning on earlier text (second half of S14.4d, part 1): registered 2026-10-10 before any code (owner: "go")

**Desk read (transformers 5.15.0 `generation_whisper.py`), the parts a gate can hold to it exactly.**
- **Thresholds (`_need_fallback`).** After a window's generation at some temperature, on the generated tokens INCLUDING the stop token: the compression ratio is `len(bytes) / len(zlib.compress(bytes))` where each token id is written as `int(log2(vocab_size) / 8) + 1` little-endian bytes (2 for 51,865); above `compression_ratio_threshold` the window needs a fallback. The average log-probability is the sum over the generated tokens of the log-softmax of the scores the generation produced (the scores AFTER the logits processors, multiplied back by the temperature when it is above 0) at the chosen token, divided by the token count; below `logprob_threshold` the window needs a fallback. With `no_speech_threshold` set: if the average log-probability is below `logprob_threshold` AND the no-speech probability is above `no_speech_threshold`, the fallback is cancelled and the window is skipped: no segment, and `seek` advances by the whole window. The no-speech probability is the softmax of the RAW logits (no processors) at the `<|startoftranscript|>` position of the decoder input, taken at the token `<|notimestamps|>` minus one (`<|nospeech|>`).
- **Fallback.** The temperatures are tried in order; a window that needs a fallback is generated again at the next temperature, and the last temperature's result is kept whatever the thresholds say. A temperature above 0 samples: the scores go through the processors, then `TemperatureLogitsWarper`, then top-k (the generation config's default 50), then a multinomial draw.
- **Conditioning (`condition_on_prev_tokens`).** From the second window on, the decoder input is `<|startofprev|>` (`prev_sot_token_id`, else the second-to-last `suppress_tokens` entry), then the previous segments' tokens (a segment that ends on two timestamps drops its last token), cut to the last `max_target_positions / 2 - 1` (223), then the usual prompt; the window's `begin_index` is the full input's length; the maximum length is raised by `min(223, input length)`. A window decoded at a temperature of 0.5 or more does not feed the next window.

**What is built (`internal/whisper`).** `TranscribeOptions` (temperatures, the three thresholds, conditioning, a seed) and `Transcriber.TranscribeWith`; the no-speech probability, the two decision statistics, the fallback loop, sampling (temperature, top-k 50, a seeded multinomial draw), and the conditioned prompt. One implementation: `TranscribeTimestamps` becomes `TranscribeWith` with the zero options.

**What cannot be held to transformers, said now.** (1) A sampled attempt: its random stream is torch's and no seed here reproduces it, so the gates below stop at the decision to fall back and treat the sampled attempt by invariants. (2) The compression ratio: Go's zlib is not byte-identical to CPython's, so the compressed length can differ by a few bytes and a ratio within a hair of the threshold could flip; the real gate prints both ratios per window and a flip there is read as this, not as a rule.

**Gates.**
- **G-S14h1, the rules on random weights, by day.** The tiny model of G-S14g2, its `generation_config.json` gaining a `prev_sot_token_id`; transformers' own `generate` as the reference, `temperature=(0.0,)` (one attempt, so no sampling is reachable) on the 68 s clip and the 8 s clip, in four configurations: (a) `condition_on_prev_tokens=True`; (b) `logprob_threshold` and `compression_ratio_threshold` set so some windows "need a fallback" (nothing to fall back to: the result is kept); (c) `no_speech_threshold` and `logprob_threshold` set so some windows are SKIPPED; (d) all of them together. **PASS** if the segments (count, tokens, start and end to 1e-9) and the final sequence equal transformers' in every configuration, and the number of skipped windows is positive in (c). **Planted defects red:** conditioning feeding the whole previous text uncut; the double ending timestamp not trimmed; no `<|startofprev|>`; the no-speech probability taken from the processed scores; the skip advancing by nothing; the average log-probability divided by the count without the stop token.
- **G-S14h2, the real checkpoint, by day.** `openai/whisper-small` on the 70 s composite and on a composite with long stretches of silence (10 s of zeros before and between two clips, so a window is silent), `temperature=(0.0,)`, `condition_on_prev_tokens=True`, `logprob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `no_speech_threshold=0.6` (OpenAI's defaults): **PASS** if the segments and the sequence equal transformers' on both. The windows' two compression ratios and average log-probabilities are printed.
- **G-S14h3, the fallback and sampling, by day, invariants only.** With temperatures `(0.0, 0.2, 0.4)` and thresholds forced to fail until the last attempt (`logprob_threshold` above anything reachable): every window is attempted at 0.0, 0.2, 0.4 in that order and the last attempt's result is kept; the same seed gives the same result and different seeds can differ; every sampled window obeys the timestamp structure (the three rules of G-S14g's desk read hold on its tokens); at temperature 1e-6 the sampled result equals the greedy one; the sampled token is always inside the top 50 of the processed scores (checked on a hand-built score vector); a window that stays under the thresholds is attempted once.

**Prediction (written now).** G-S14h1 PASS on the first run for (a)-(d) at 0.5: conditioning has two trimming rules and a length formula, and the likeliest miss is the cut or the double-timestamp trim; (c) leaves a skipped window on the random model at 0.7 (if not, the thresholds are moved before the run and the move is recorded, not the bar). G-S14h2 PASS at 0.7. G-S14h3 PASS at 0.9 (it asserts what I build). The silent composite makes the no-speech probability visible on a real model for the first time.

**Tier and cost:** all by day (each a few minutes). **What each result does next:** PASS: the pure-Go Whisper has transformers' decode policy except the sampled stream; the route (G-S14i) is registered next.

##### G-S14h1, h2 and h3 read 2026-10-10 (by day, nobara; branch `s14c4d-partial-force`; `internal/whisper` at `fc602c0e` and `TranscribeWith`; references `~/goinfer-logs/s14h/`)

- **G-S14h1, the policy on random weights: PASS on the first run.** The tiny model of G-S14g2 (its generation config gaining `prev_sot_token_id`), transformers' `generate` with `temperature=(0.0,)`: the five registered configurations (`cond`, `thr`, `skip`, `skip2`, `all`) on the 8 s and 68 s clips: **the same segments (count, tokens, start and end to 1e-9) and the same sequence in all ten**; windows skipped as silence: 1 on the 8 s clip in each skip configuration (every window of it is skipped, so transformers returns no segment: matched), 1 and 2 on the 68 s clip. Planted defects red on the clips: conditioning uncut (2 configurations differ), the double ending timestamp not trimmed (3), no `<|startofprev|>` (3), the no-speech probability read from the processed scores (6), a silent window not skipped (6). **One the clips cannot see, as the registration allowed for the processor order:** leaving the stop token's log-probability out of the average. The tiny model never writes the stop token in these windows (they end on the length cap), so the average is identical either way. `TestPolicy_avgLogprobCountsTheStopToken` covers it with a stop token chosen among the tokens the model does write: red under the defect.
- **G-S14h2, `openai/whisper-small` with OpenAI's decode policy: PASS.** Conditioning, `logprob_threshold` -1.0, `compression_ratio_threshold` 2.4, `no_speech_threshold` 0.6, one greedy attempt: **identical segments, sequences and texts to transformers' on all three cases**: the 70 s composite (13 segments), the silent composite (35 s of zeros, a clip, 35 s of zeros, two clips: 3 segments), and the same silent composite with `logprob_threshold` -0.4 (3 segments, starting at 30.0 s: **the first window was skipped as silence**, no-speech probability 0.944, average log-probability -0.547, the rule firing on the real model). The registration's silent case as first run (threshold -1.0) did NOT skip the silent window: its average log-probability is -0.547, above -1.0, and the rule needs both conditions; the third case was added for that reason, the same audio, before it was graded. The windows' statistics are logged: compression ratios 0.38-0.98 and average log-probabilities -0.12 to -0.55, none near the 2.4 and -1.0 bars, so Go's zlib against CPython's, which the registration flagged, did not come into play here.
- **G-S14h3, fallback and sampling by invariants: PASS.** Every window is attempted at 0, 0.2, 0.4 in order under a threshold nothing clears, and once under one everything clears; the accepted first attempt equals the greedy run; the same seed gives the same result and two seeds differ; sampled segments open on a timestamp, never decrease, never contain `<|notimestamps|>`; a temperature of 1e-6 equals greedy; top-k keeps exactly the 50 largest processed scores and a draw never leaves them.
- **What this establishes:** the pure-Go Whisper has transformers' decode policy, greedy, on a real checkpoint. **What it does not:** a sampled attempt against transformers' random stream (cannot be held to it), a clip that really triggers the compression-ratio rule (none of the three does), other checkpoints and languages.

#### G-S14i, `POST /v1/audio/transcriptions` and a Whisper model in `serve` (S14.4d, last part): registered 2026-10-10 before any code (owner: "go")

**What is wrong now.** `serve --model ~/models/whisper-small` fails to start (Whisper is an encoder-decoder, not a `decoder.Model`), and no route takes an uploaded audio file.

**Design.** A Whisper directory (`config.json` `model_type` "whisper") named by `--model` is loaded as a **speech model**, kept in its own registry (`server.speech`) and not in `server.models`, because everything keyed on `s.models` assumes a `decoder.Model` (sessions, KV slots, templates, adapters, the resident path). It is listed by `GET /v1/models` (`id`, `owned_by`, and a `capabilities` entry `["audio.transcriptions"]`), named in the 404 for an unknown model, and reachable only through the new route. Everything else about startup is unchanged; a server may hold speech models and decoders together.

**The route (OpenAI's `audio/transcriptions`, multipart/form-data).** `file` (required): a 16-bit PCM WAV at any rate and channel count (downmixed and resampled to 16 kHz mono as for the other audio models), over 600 s refused; any other format is a 400 that names this limit. `model`: required when more than one speech model is served, else optional. `language`: an ISO code ("en") or the English name ("english"); unknown is a 400; absent means detected. `response_format`: `json` (default, `{"text"}`), `text`, `srt`, `vtt`, `verbose_json` (`task`, `language` as the English name, `duration`, `text`, and `segments` with `id`, `seek`, `start`, `end`, `text`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio`, `no_speech_prob`). `temperature`: absent or 0 means OpenAI's policy, the ladder 0, 0.2, 0.4, 0.6, 0.8, 1.0 with `compression_ratio_threshold` 2.4, `logprob_threshold` -1.0, `no_speech_threshold` 0.6 and conditioning on earlier text; above 0 is one attempt at that temperature. `timestamp_granularities[]`: `segment` only (`word` is a 400). `prompt` is a 400 (not built), as is `stream`. Texts are stripped of the leading space Whisper writes; segment times are seconds from the start of the clip. The transcriber runs one request at a time per model; the route goes through the same auth, halt and in-flight gates as the other inference routes.

**Gates.**
- **G-S14i1, the shapes and refusals, by day.** Handler tests on a stub engine through the real mux: every response format against OpenAI's documented shape (`srt` and `vtt` timestamp formats `HH:MM:SS,mmm` and `HH:MM:SS.mmm`, `vtt`'s `WEBVTT` header, `verbose_json`'s fields and the language as a name); each refusal (no file, a non-WAV, over 600 s, unknown language, unknown model, `prompt`, word granularity, an unknown `response_format`) with the right status and a message that names the field; the model defaults; `temperature` mapping to the options. **PASS** if every case holds; a planted defect for each of three (the leading space kept, the srt separator a dot, the language as a code) is red.
- **G-S14i2, the loading, by day.** `newServer` with a Whisper directory in `--model` loads it as a speech model, lists it in `/v1/models`, and still loads a decoder beside it; a duplicate name is refused; a Whisper directory with no `tokenizer.json` or no `generation_config.json` is refused at startup by name.
- **G-S14i3, served on the real checkpoint, by day (about 5 minutes).** `serve --model ~/models/whisper-small` and, by HTTP: the registered 5.9 s clip and the 70 s composite, `response_format=verbose_json`: **PASS** if the segments (count, start, end, tokens) and the text equal G-S14h2's references for OpenAI's policy on the same audio (the 5.9 s clip one window; the 70 s composite the 13 segments), the language is "english", and `srt`, `vtt` and `text` of the same request are consistent with those segments (the same cuts and times, formatted). A second request on the same server returns the same result (the transcriber holds no state between requests).

**Prediction (written now).** i1 PASS after at most one correction at 0.8; i2 PASS at 0.8 (the likeliest stumble: some startup code that iterates `cfg.models` and expects a decoder: `resolveBodyCaps`, the banner, `towerReserve`); i3 PASS at 0.8 (the library is already equal to transformers; what is new is the policy defaults and the formatting).

**Tier and cost:** by day. **What each result does next:** PASS: S14.4d is done as registered, and S14 is complete but for the items it names as not built (word timestamps, the sampled stream, `prompt`, formats other than WAV, a GPU). A docs section in `docs/server.md` describes the route.

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
  (the Mac's, whose sidecar build failed the fit guard last night). The `gemma-4-E4B-it` bf16 directory (16 GB) is on the box only because of the first gate; it can be removed. **Moved to
the archive 2026-10-09** (`/srv/models/gemma-4-E4B-it`, sizes verified, synced). It was never archived before, and the
G-31a/G-31b scripts read it as a control, so it was moved rather than deleted. `models-pull gemma-4-E4B-it` brings it
back.

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

#### The night of 2026-10-08, read 2026-10-09 (nobara; five jobs, started 21:51 PDT, done 00:30; logs under `~/goinfer-logs/night/runs/2026-10-08/` and the per-job dirs named below; the Mac's queue is a separate queue)

One job failed outright (S6, a plumbing error of mine, below), one passed its gate, one met its registered bars with a disclosed deviation, and two returned readings that are graded below as registered. Two things came out of the grading that no gate asked about: a decoder bug that stopped every Qwen3-ASR transcription at its first "!", and the fact that `--embed-int4` does nothing on nine families.

##### S7, third read: every registered prediction held (`~/goinfer-bench/s7n/s7-nobara-3-2026-10-08/`, `serve-cuda-s7r3` at `5f715eb3`, 3 min)

| cell | second read | third read | registered |
|---|---|---|---|
| Gemma 3 4B | 4.59 s | **2.40 s** | 2.3-2.6 s |
| Qwen2.5-VL-3B | 2.45 s | **2.48 s** | 2.3-2.6 s (its KV plan reads 5,963 positions; I predicted about 5,967) |
| Gemma 4 E2B image | 0.85 s | 0.87 s (+2.4%) | within 5% |
| Gemma 4 E2B audio | 1.48 s | 1.46 s (-1.4%) | within 5% |
| Qwen3.5-0.8B | 0.81 s | 0.82 s (+1.2%) | within 5% |
| GLM-OCR | 2.08 s | 2.07 s (-0.5%) | within 5% |
| **Qwen3-VL-2B (new)** | - | **10.79 s** (warm-up 12.43; timed 10.83 / 10.76 / 10.79) | 9-15 s, over the bar |

- **The regression check:** all seven serve logs read `decode path: cuda-resident (int4)` and every tower is on CUDA (Qwen3-VL's included: it encoded a 283,430-byte image in 3.169 s on the first request). No cell hides a CPU fallback.
- **What the table says:** six of seven cells are under the 5 s bar; Qwen3-VL-2B is the one over it, 10.8 s, for the reason registered: its image turn prefills on the CPU because the resident DeepStack prefill is off (G-S10j, the 896 image, below). The control (five cells within 2.4% of the second read) says the box was in the same state.

##### G-31a read: G-31a2 PASS; G-31a1's automatic bars NOT HELD as registered, on one bar that a correct answer fails (`~/goinfer-logs/g31a-2026-10-08/`, 42 min against 120 queued)

- **G-31a2, int4 against int8int8 on the same checkpoint: PASS.** The 31B agrees with its own int8int8 decode at **352 of 373 positions, 94.37%** (95% interval 91.5-96.3); the E4B sibling at **352 of 384, 91.67%** (88.5-94.0). The 31B is **2.70 points ABOVE the sibling**; the bar was "at least the sibling minus 5.0". Non-finite logits: none in either arm. Splits: 21 on the 31B (4 at an int8int8 margin below 0.1, so 17 are not near-ties), 32 on the sibling (3 below 0.1).
  - **Two deviations, disclosed.** (1) The registered count was 12 x 32 = 384 positions; the 31B's int8int8 path for prompt index 8 ended at 21 tokens (EOS), so it has 373. The sibling has 384. (2) `scripts/g31a_grade.py` prints "the 31B is -2.70 points below the sibling": the number is sibling minus 31B and the word "below" is backwards. The verdict logic compares that number against the bar correctly, so PASS stands; the line is corrected in the same commit as this record.
  - **Prediction:** the sibling at 80-92% (read 91.67%, at the top edge), the 31B within 5 points of it (above it), PASS probability 0.7. Held.
- **G-31a1, the served smoke (CPU, `smoke/`): bars NOT HELD as registered; the reading below is mine and is labelled a judgement.** Three text requests and one image request, 48 greedy tokens. Three of four held every automatic bar. **text3 failed `distinct_tokens >= 8`:** the reply is "The capital of Australia is Canberra." (7 tokens, then EOS), which is correct and complete. The bar was written to catch degenerate output, and a one-fact question has a legitimate seven-token answer: the defect is in the bar's reach, not in the model. It is not moved here: the registered outcome is NOT HELD, and any change to what the bar applies to is registered before the next reading that uses it.
  - **Read against the prompts (judgement): coherent 4 of 4.** text1 explains Rayleigh scattering correctly; text2 (the Fibonacci prompt) offers an iterative, a recursive and a matrix method; text3 is right; the image reply names "Table 2. Quarterly unit sales by region (thousands)" and the five regions (North, South, East, West, Central) and the total row, which are on the page.
  - **Cost, for planning:** the int4 bundle took 152 s to write (17.3 GB, kept in `~/models`), the int8int8 bundle 414 s (deleted); decode on the CPU is 2.25 s per token at int8int8 and 1.39 s at int4; the smoke's text replies took 48.5, 49.6 and 19.4 s, the image turn 144.9 s (290 prompt tokens).
- **The claim step (a) supports, as fixed in advance:** "the 31B loads and runs coherently on the CPU at int4, and its int4 decode agrees with its own int8int8 decode at least as well as a plain-bf16 sibling's does." Both halves hold (the second by G-31a2, the first by the judgement above). The Gemma 4 31B row stays "not validated against HF"; it may read "runs on the CPU, coherent, int4 agrees with int8int8 at 94.4%, not yet checked against HF".
- **Next, as registered:** (b') the layer-streaming Hugging Face float32 reference is registered below (2026-10-09), now that (a) has run; its bar is set from what (a) showed (the 31B's own int4-int8int8 agreement is 94.4% with 21 splits, the quantisation-against-quantisation figure the Hugging Face comparison is read beside, not against).

##### Option D read: four of seven models graded, none COSTLY, the registered map does not match; three models not run, each for a different reason (`~/goinfer-logs/head-precision-2026-10-08/`, 57 min against 150 queued)

The positive control reads **exactly 0 / 0** (`qwen2.5-0.5b-instruct`, both arms at the int8 pin), and `HeadTable()` read int8 against int4 across the two arms on the four graded models. 32 prompts x 32 positions = 1,024 per model (amendment A1).

| model (head) | agree int8 -> int4 | d_agree, points [95% interval] | dKL, nats [95% interval] | KL ratio | reading |
|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct (tied) | 82.8% -> 83.1% | -0.29 [-1.95, +1.37] | +0.0182 [+0.0090, +0.0273] | 1.11x | MIXED |
| qwen3-1.7b-bf16 (tied) | 93.2% -> 93.1% | +0.10 [-0.78, +1.07] | +0.0190 [+0.0048, +0.0353] | 1.17x | MIXED |
| gemma-3-4b-it (tied, 262k vocabulary) | 91.1% -> 89.5% | +1.66 [+0.39, +2.93] | +0.0453 [+0.0196, +0.0713] | 1.23x | MIXED |
| tinyllama-1.1b-chat (untied) | 91.0% -> 90.5% | +0.49 [-0.68, +1.66] | +0.0042 [+0.0002, +0.0079] | 1.07x | OK |

- **The registered map:** not all seven OK; not COSTLY on two or more tied-head models; so "anything else: the numbers go to the owner as they are." This registration decides nothing by itself.
- **What the numbers say, read plainly:** the int4 head costs at most **+1.66 points** of top-1 agreement (Gemma 3's 262k-vocabulary table; the interval excludes zero) and **+0.004 to +0.045 nats** of KL, a **1.07x to 1.23x** ratio. The three tied-head models read MIXED because their dKL upper bounds (0.027, 0.035, 0.071) sit above the 0.02 bar, not because their point estimates are near the COSTLY line (0.05): Gemma 3's 0.0453 is the only one close. The "about 2.3 points" cost on file is not reproduced on text: the largest point estimate here is 1.66. The "3.5x the KL on one real image" is not reproduced either (1.07-1.23x), **but the model it was seen on, Qwen2.5-VL-3B, is one of the three not run**, so this is not yet a test of that claim.
- **Prediction, written before:** 1-3 points of agreement and 0.02-0.06 nats on the tied small-vocabulary models, less on the untied; at most two COSTLY. Observed: agreement cost below the predicted range on the two smallest (-0.29, +0.10), inside it on Gemma 3; KL cost 0.018-0.045, at or just below the predicted range's floor on two; zero COSTLY. Direction right, size smaller than I predicted. **Interval widths (recorded as amendment A1 asked):** d_agree half-widths 0.9-1.7 points (predicted about 1.6); dKL half-widths 0.004-0.026 nats (predicted 0.009: wider on the large-vocabulary models, which is why three read MIXED).
- **The three that did not run (the registration said a model that cannot be read is recorded, not dropped):**
  - **`qwen25vl-3b-instruct`: HF side, `AutoModelForCausalLM` does not take a `Qwen2_5_VLConfig`.** A harness gap. The script now falls back to `AutoModelForImageTextToText` for the same checkpoint's text path (no image given).
  - **`olmo3-7b-think`: the HF side, "tokenizer.chat_template is not set".** The checkpoint directory on this box was downloaded without its `chat_template.jinja` (its repository publishes one). The registration says every model goes through its OWN chat template, so the template is taken from the model's own repository (`allenai/Olmo-3-7B-Think`, `chat_template.jinja`, 1,652 bytes, fetched 2026-10-09, sha256 `6d549883...`), kept beside the HF script, and used only when the directory ships none. The directory itself is not edited.
  - **`phi3-mini-4k`: VOID, "EmbedInt4=true loaded an int8 head table (the flag did nothing)".** The registered assertion did its job. **Root cause, found in the grading, and it is wider than phi3:** `Options.EmbedInt4` is read only by the generic safetensors loader (`buildWeightsFromSafetensors`, `weights.go`) and the GGUF loader. Nine family loaders call `quantizeWM(..., quant.embedding())` and never see it: `buildGPT2Weights`, `buildGraniteWeights`, `buildNemotronWeights`, `buildPhi3Weights`, `buildGlmOcrWeights`, `buildSpark25Weights`, `buildLlama4Weights`, `buildGptOssWeights`, `buildInternLM2Weights`. On those families `--embed-int4` is inert: the head stays int8 whatever the flag says (confirmed on phi3: `HeadTable()` reads "int8" with the flag on and off). That matters to the owner's option (C, the default on everywhere): the default would not reach them either, and the banner line (option A) is the only place it shows. **Fixed 2026-10-09, at the owner's request ("fix inert flag"); the record is "The inert `--embed-int4` flag, fixed" below.**
- **Amendment A2 (2026-10-09, before the second night's run; no bar, statistic, control or model moves).** The second night runs only `qwen25vl-3b-instruct` and `olmo3-7b-think`, with the two harness repairs above, into its own directory, and the grader reads both nights together (`scripts/head_precision_grade.py <root1> <root2>`; a model read in a later root replaces the same name in an earlier one). `phi3-mini-4k` was to stay "NOT RUN: VOID", **but its cause was fixed the same day (below), so it runs on the second night too** (its HF float32 dump is re-made, about 8 min, because the second night writes its own directory; its Go side is two loads and 2 x 1,024 steps, about 12 min): the untied evidence is `tinyllama`, `phi3` and `olmo3`. Cost: HF float32 of a 3B and a 7B model (about 45 min) and the Go side (about 50 min); queued at 150.

##### G-S14c4 read: G-S14c4a NOT MET, G-S14c4b FAIL, controls ok, both as registered; two findings that change what a re-read should show (`~/goinfer-logs/s14c4-2026-10-08/`, 49 min against 90 queued)

| arm | WER | control tokens in replies |
|---|---|---|
| R, transformers float32 | 4.09% | 0 |
| F, goinfer native weights | 4.96% | 0 |
| I, goinfer int4 (serve's defaults) | 9.13% | **33 of 73** |
| S, control: F on the next clip's audio | 126.96% | 0 |
| D, Gemma 4 E4B raw replies (record only) | 123.04% | - |
| D', its longest quoted span (best-effort extraction) | 18.78% | - |

- **G-S14c4a: NOT MET.** F is byte-equal to R on **71 of 73** clips (bar: at least 70 of 73, met), but |WER_F - WER_R| = **0.87 points** against a bar of 0.30; both were required. The two unequal clips are listed by the grader with their first differing word.
- **G-S14c4b: FAIL.** WER_I - WER_F = **+4.17 points** (95% interval [+0.61, +9.52]); above +1.5 fails. 15 clips differ in words between I and F. **Independently, 33 of the 73 int4 replies carry a control token (all `<|im_start|>`, always at the start), against the 3 that park the verdict.**
- **Controls:** F against F is +0.00 (required 0); S is 127.0% (required above 80). The metric sees a broken pipeline. D's raw replies are unusable as WER (123%: they carry a preamble and thinking markers); the extraction is best-effort and is record-only.
- **Prediction, written before:** R and F near 2-4% (R 4.09, F 4.96: at and just above the range); F byte-equal on 90-100% (97.3%, held); WER_I - WER_F inside +0.5 with probability 0.6 (**missed: +4.17**); control tokens in at most two clips (**missed: 33**); S above 95% (held); D worse than the Qwen3-ASR arms (held).
- **Finding 1, a decoder bug, and it is why G-S14c4a is not met.** Both unequal clips stop at a "!": F reads "Oh no" where R reads "Oh no! I'm quite sure he didn't." and "...put that away" where R continues "! You're being a fool." Out." Of the replies whose reference contains a "!", F stopped at it in **2 of 2**; "?" and quotes were fine. **Cause:** Qwen3-ASR's `thinker_config.text_config` carries `"eos_token_id": null` (transformers writes the key with `None` and leaves the ids to `generation_config.json`). `Config.EOSIDs()` unmarshalled the JSON `null` into an int, which succeeds and leaves 0, so the model's stop ids were `[0, 151643, 151645]`, and **id 0 is "!" in Qwen's vocabulary**: generation ended at the first "!". Float32 and int4 both did it, so it cancels in the G-S14c4b difference. **Fix:** `EOSIDs()` treats `null` as absent (`decoder/config.go`), with `TestConfigEOSIDs_nullIsAbsent` (the unit, including that a written `0` is still an id) and `TestQwen3ASR_loadedModelDoesNotStopOnIDZero` (through the loader on the tiny fixture); **both were red on the unfixed code** (`EOSIDs() = [0], want []`). **Checked on the real checkpoint** (exploratory, the two clips, both arms, by day): the float32 and the int4 replies are now byte-equal to transformers' text on both ("Oh no! I'm quite sure he didn't." and "He must have drawn his gun because the intruder said quickly, \"Put that away! You're being a fool.\" Out."). This is a plumbing check, not the graded reading: **G-S14c4a stays NOT MET as read, and is re-read in full tonight.** Other families whose released config carries a null `eos_token_id` were affected the same way if a stop id of 0 mattered to them; the tiny fixtures that carry one (`glm-tiny`, `qwen3moe-tiny`, `qwen35-tiny` and others) are exercised by the decoder suite, run on this commit.
- **Finding 2, where the int4 damage is, and what it is not.** Of the 15 clips where I differs from F, the words are mostly proper nouns and near-homophones that a 4-bit 0.6B model plausibly gets wrong (linnell, mantle, rigado, breon, perry, needing). **But two int4 replies are empty or one garbage byte** (`1272-141231-0003`: an empty string; `1272-141231-0010`: a replacement character and a newline), and a third has lost its `language English` prefix (`<asr_text>He laments ...`); together the two empty ones are 38 of the 1,150 reference words, **3.3 of the 4.17 points.** The 33 `<|im_start|>` prefixes are the same kind of event (the first generated token is wrong, then the model recovers). All of it is at the first token. **Two readings that this data cannot separate:** (i) the int4 body or the int4 head damages the first-token distribution; (ii) for the empty replies, the first token was id 0 and the bug above ended the reply. The second is testable and the re-read tests it.
- **Amendment A1 (2026-10-09, before the re-read; no bar, statistic, arm or control moves).** The decoder fix changes F, I and the byte-equality count, so G-S14c4 is re-read in full on the fixed tree, with **one added record-only arm, `int4h8`: serve's int4 with the head table at the int8 pin** (`EmbedInt4=false`). It moves no verdict. It answers one question the first reading raised: whether the int4 head is what damages the first token (the control tokens, the empty replies). Cost: +3 min; estimate 60, queued at 75. The harness change is in `decoder/qwen3asr_wer_real_test.go` and `scripts/s14c4_grade.py`.

##### S6's served check: the first night's run is VOID (a plumbing error of mine); the fixed script is queued (`~/goinfer-logs/s6-moe-2026-10-08/`, 2 min)

All three arms were void by the harness's own check (`decode path` was `cpu (int4mix)`, not `cuda-resident`): the 35B's int4 weights are 18.6 GB on an 8 GB card, and `run-s6-moe-night.sh` did not pass `--moe-cache-experts`, the C' expert streaming its own header says it uses, so the CUDA build declined with `cuMemAlloc ... CUDA_ERROR_OUT_OF_MEMORY` and said so ("Try -moe-cache-experts"). The harness stopped after the first arm. **No reading was made and none is claimed.** The script now passes the flag. **Plumbing control, by day (not a result):** the same pinned binary with the same flags loads in 1m44s and logs `decode path: cuda-resident (int4mix)`, KV plan 1 x 4,096 positions, 2.1 GB held back for the tower; stopped by pid, GPU back to 468 MiB. The registration (G-S6m above) is unchanged; it is queued again at est 60.

**Second night's run (`s6-moe-image-2`, run by day 2026-10-09 09:43): no reading again, for a different reason. It was
re-queued as `s6-moe-image-3`.**
- **What failed:** the first arm's image request returned HTTP 500 and the harness stopped. "Arm cuda/cuda2 log is
  missing" followed from that; those arms never ran.
- **The error,** reproduced by day with the same pinned binary and flags: `qwen3_5_moe has no mrope_section in
  rope_parameters; refusing an image turn`.
- **The cause:** `~/models/qwen3.6-35b-a3b-int4.giw` was built on 2026-09-24 from `qwen3.6-35b-a3b-Q8_0.gguf`, before
  45ac1c07 (2026-10-06) made a Qwen3.5+ GGUF carry its m-RoPE split. Its embedded config had no `mrope_section`.
  The binary (8ff9a57e) is newer and was not at fault.
- **The fix:** the bundle was rebuilt from the same Q8_0 GGUF by `prequant -quant int4` at `origin/main` afb50500
  (2.5 min). It now carries `mrope_section [11,11,10]`, interleaved, and replaces the old file under the same name.
  Re-running the request got past the refusal.
- **That check is plumbing, not a result.** That run's swap guard then tripped, because `g31b` started loading the 31B
  at the same moment.
- **Same staleness, not fixed here:** all four of nobara's Qwen3.5-9B sidecars (`Qwen3.5-9B-Q4_K_M.int4.cuda`,
  `.int4.e4h.cuda`, `.int4.e4h.cpu-amd64`, `.int8int8.e4h.cpu-amd64`) lack the section. The sidecar freshness check
  does not see it (source mtime and layout only), so a 9B image turn there would be refused until they are rebuilt.
  - **Rebuilt 2026-10-09 by day** (`prequant` at `ebce2b4d`, from the same GGUF, the same quant, target and
    embed-int4 for each). Each now carries `mrope_section` and passed its self-check (98, 119, 119 and 27 s).
  - A header scan of every `.giw` on both boxes found no other Qwen-family sidecar without it.
  - Of the nine families' sidecars (below), none holds a stale head. The only one keyed embed-int4 is nobara's phi3
    `int8int8.e4h`, and int8int8 never takes an int4 head (`embeddingWith`). The Mac's `glm-ocr.int4.e4h.cpu-arm64`
    was built after the fix.

##### Gemma 4 31B, step (b'): the 31B against a layer-streaming Hugging Face float32 reference. Registered 2026-10-09 before any code (owner: "register it")

Owner choice of 2026-10-08, "(b') with (a) as its first step"; (a) is G-31a (read above), and this bar is set from what it showed. **Why it exists:** the 31B is 59 GB in bf16, so an ordinary float32 Hugging Face run (about 124 GB) cannot be held in this box's 62 GB of RAM, and the 8 GB card cannot hold it either. Every other family was validated by comparing against Hugging Face; the 31B has no such comparison, so its status row says "not validated". (b') builds the comparison by running Hugging Face's own decoder layers **one at a time, layer-major**: all 12 sequences go through layer 0 (its weights read from the safetensors shards, cast to float32, used, dropped), then through layer 1, and so on, keeping only the hidden states ([tokens, 5376] float32, a few MB) between layers; the token embedding (262,144 x 5,376, 5.6 GB in float32) stays resident and the tied head is applied to the needed positions at the end.

- **The claim (b') can support, fixed now:** "goinfer's Gemma 4 31B, run on the CPU at int8int8 and at int4, agrees with Hugging Face's float32 forward on the same token sequences at least as well as the plain-bf16 E4B does (minus 5.0 points), teacher-forced, 12 prompts." It does not say the 31B is correct on images, on the GPU (it cannot be resident), on long contexts (the sequences are under 100 tokens, so the 1,024-position sliding window never engages: a limit, stated below), or anywhere outside these 12 prompts.
- **The sequences (fixed, pinned).** The 12 prompts of `decoder/quant_consistency_real_test.go` (`qcPrompts`) through the checkpoint's own chat template, each followed by the continuation recorded in G-31a: goinfer's own int8int8 greedy path, **`docs/measurements/multimodal-support-2026-10/g31a/qc-31b.json`** (sha256 `660afa59d8362374...`: 373 positions, prompt 8 ends at 21 tokens) for the 31B and **`qc-e4b.json`** (sha256 `4d21d193203c1d2d...`: 384 positions) for the E4B. The reference is teacher-forced along that path (no generation on the Hugging Face side: a layer-major pass has no per-token loop to pay for).
- **Instrument.** `scripts/g31b_hf_stream.py` (new): transformers 5.15.0, `~/g4venv`, CPU, float32. It builds the model on the `meta` device, then per layer loads that layer's tensors from the shards (`safetensors.safe_open`, bf16 to float32, `load_state_dict(..., assign=True)`), and runs **Hugging Face's own decoder-layer module** with the position embeddings and the sliding or full mask that Hugging Face's model code builds, so no layer arithmetic is re-implemented; what the script owns is the loop order, the embedding scale, the final norm, the tied head and the final logit softcap (30.0). For the E4B (sibling) the ordinary `AutoModelForCausalLM` float32 forward is used (15 GB of bf16, 32 GB of float32, fits), because its per-layer inputs and shared KV are not what the 31B has and a streaming port of them would be a second instrument to validate.
  goinfer's side is a new `realckpt` test beside `TestQuantConsistency_real`: it loads the checkpoint at `int8int8` and at `int4` (CPU, `EmbedInt4` at serve's default for the 31B bundle, the same as G-31a), teacher-forces each prompt plus the recorded continuation, and writes the logits (float32) at the continuation positions.
- **Statistics.** At each of the N positions: the arm's argmax equals Hugging Face's argmax (**agreement**), and KL(Hugging Face || arm) in nats; per model and arm the mean of each, with a cluster bootstrap over the 12 prompts (10,000 resamples, seed 20261009, 95% interval). Also printed: Hugging Face's mean top-1 probability (a sanity row), the positions where the arm and Hugging Face split with Hugging Face's top-1 minus top-2 margin there (so near-ties read apart from real flips), and how often Hugging Face's own argmax equals the path token.
- **Controls, all before any 31B reading, and a failed one voids the reading rather than failing the model:**
  1. **Streaming against ordinary on a 31B-shaped tiny.** `testdata/gemma4-dense-twogeom-tiny` (dense, two attention geometries) is run through the streaming script and through an ordinary Hugging Face float32 forward in the same process, on sequences longer than its sliding window so the window engages: per-position logits **cosine at least 0.9999999 and max |difference| at most 1e-5** at every position. (If this fixture lacks K=V on its global layers, a second tiny with `attention_k_eq_v` is built by the same recipe as the committed ones; the control is not run on a shape that omits a mechanism the 31B has.)
  2. **Planted defects, each alone red against control 1's bar:** the final logit softcap skipped; the final norm skipped; two layers visited in swapped order; the sliding-window layers given a full causal mask; K=V's tie dropped on the global layers (if the fixture has it).
  3. **Determinism:** the streaming script run twice on one prompt, bit-identical logits.
- **Bars, set from (a).** For each arm separately (int8int8 is the primary, closest to float32; int4 is serve's default), against the E4B's same arm measured by the same statistic: **PASS** if the 31B's agreement is **at least the E4B's minus 5.0 points**; **PARKED** (to the owner) between minus 5.0 and minus 10.0; **FAIL** if worse than minus 10.0, if any logit is non-finite, or if control 1, 2 or 3 does not hold (then "no reading"). The 5.0 is G-31a2's, for G-31a2's reason: 373 and 384 positions with 12 clusters put the standard error of a difference at about 2.5-3 points. **Both arms must PASS for the row to change;** the primary arm alone passing reads as "int8int8 only".
- **What each outcome means for the status row.** Both PASS: the 31B row may read "agrees with Hugging Face float32 as well as the E4B does, teacher-forced on 12 prompts, CPU; not validated on images or on contexts past the sliding window"; which parity tier that earns under the matrix's own rules is the owner's call, not this registration's. PARKED or FAIL: the numbers go to the owner with the splits; a FAIL with controls holding is a finding about the 31B's loader or quantisation and is root-caused before anything else is claimed.
- **Prediction, written now.** The only figure on file for the plain E4B is "about 77% from Hugging Face per arm", quoted in G-31a's registration (not re-measured here), well below G-31a's 91.7% quantisation-against-quantisation agreement, because float32 is a different reference from int8int8. I expect the E4B's agreement at **70-85%** on both arms, the 31B's int8int8 within 5 points of it with probability **0.65**, and the 31B's int4 within 5 with probability **0.55**; the likelier way to miss is the 31B reading *lower* than the E4B (60 layers accumulate more quantisation drift against float32 than 42 do). Controls 1-3 pass with probability 0.8; the likeliest stumble is the sliding-window mask Hugging Face builds for this class (its construction moved between transformers releases), which control 2's fourth defect is there to see.
- **Limits, stated now.** (1) **Sequences under 100 tokens never reach the 1,024-position sliding window**, so the window's behaviour at depth is exercised only by control 1's tiny; a deep-context check on the 31B is a separate, larger gate. (2) 12 prompts and a path taken from goinfer's own int8int8 decode: teacher-forced agreement along one model's greedy path is not agreement of free-running generations. (3) The reference is float32 of a bf16-trained checkpoint, so it carries no quantisation and every KL here includes the arms' whole quantisation error. (4) CPU only; the 31B has no resident arm on this box. (5) The E4B sibling is a different shape (per-layer inputs, shared KV, 8 heads); that is why the bar is wide, and it is the weakest part of the design.
- **Tier and cost (TE11):** night; **not queued until its controls pass by day** (control 1 and 2 are seconds to minutes on a tiny, run by day as plumbing controls, not results). Instrument: the streaming script plus a `realckpt` Go test, graded by a script; stopping rule: both models, both arms, the fixed 12 sequences, no early stop. Cost basis (estimates, not measured): the streaming pass reads 59 GB once (a few minutes at the NVMe's 2-3 GB/s) and does about 43 TFLOP of float32 for 12 sequences of about 57 tokens, 15-30 min at this CPU's rate; the E4B ordinary forward about 10 min; goinfer's teacher-forced forwards are sequential per token at 2.25 s (int8int8) and 1.39 s (int4) per token on the 31B, about 700 positions each, so about 45 min, plus the int8int8 bundle's 7-minute re-quantisation (it was deleted after G-31a) and the loads; total about **110 minutes, to be queued at 150.**

##### The inert `--embed-int4` flag, fixed (2026-10-09, owner: "fix inert flag")

- **What changed.** `Options.EmbedInt4` and the canonical-layout flag now reach all nine family loaders (`buildGPT2Weights`, `buildGraniteWeights`, `buildNemotronWeights`, `buildPhi3Weights`, `buildGlmOcrWeights`, `buildSpark25Weights`, `buildLlama4Weights`, `buildGptOssWeights`, `buildInternLM2Weights`), which now quantize their embedding and head with `quantizeEmbedWM(w, quant.embeddingWith(embedInt4), needCanonical)`, the call the generic loader makes. With the flag off the result is identical to before (`quantizeEmbedWM` falls through to `quantizeWM` for a non-int4 mode). **The behaviour change is the point and the default carries it:** `--embed-int4` defaults on everywhere but Metal, so on these nine families a default load now holds an int4 head where it held an int8 one.
- **Tests, each red on the unfixed loaders (24 failing lines) and green on the fix:**
  - `TestEmbedInt4_reachesEveryFamilyLoader` (decoder): `Load` each family's tiny fixture (gpt2, granite-tiny, granite-dense-tiny, nemotron-tiny, nemotron3nano-tiny, phi3-tiny, glm-ocr-tiny, llama4-tiny), flag off then on: `HeadTable()` reads int8 then int4, the logits are finite, differ, and keep a cosine of at least 0.5 (a mis-laid int4 table reads near zero; the working tables here read 0.86-0.999, measured, and the same measurement on the established generic path reads 0.95-0.98; **the first bar I wrote, 0.9, was a guess, failed on glm-ocr-tiny at 0.862, and was replaced after measuring that a ragged last int4 group, glm-ocr-tiny's hidden 48, reconstructs at the same ~9% relative RMS as 64 columns**).
  - `TestEmbedInt4_noLoaderPinsTheHeadItself` (decoder): no non-test source file may call `quantizeWM(x, <anything>.embedding())`. This is the structural cover for the three loaders with no fixture on disk (spark2_5, gpt_oss, internlm2); it proves "no loader pins the head by itself", not "every head is int4".
  - `TestEmbedInt4_residentDecisionDoesNotChange` in `cuda` and in `gpu`: for each fixture the decode path is the same with either table and the head reads int8 then int4. **Measured on this box (RTX 2070 SUPER, WebGPU on Vulkan):** the resident families keep their path with an int4 head (`cuda-resident (int4)`: llama-tiny, granite-dense-tiny; `webgpu:vulkan-resident (int4)`: those two and nemotron-tiny), and every family that declines or runs staged does so for the same reason with either table. Metal is the backend that cannot take an int4 head (its default stays off; unchanged).
- **What is NOT covered, stated plainly.** gpt-oss, spark2_5 and internlm2 have no tiny fixture here: only the structural guard covers them. gpt2 is the one real checkpoint among the fixtures. No quality number was measured on these nine families at an int4 head (option D's evidence is on four other families, at most +1.66 points and 1.23x KL); phi3 is read on the second night.
- **Cached bundles.** A `.giw` sidecar built before this change for one of the nine families holds an int8 head under a key that says embed-int4 (the key records the flag, not what the family did with it); it keeps loading with the int8 head until it is rebuilt, and the banner's `head table:` line tells the truth either way. The format version is not bumped: nothing about the file's layout changed, and `minInt4CacheGIWVersion`'s own rule is to raise it only when an old file loads at a real cost.

##### The 896-pixel image prefill, first pass (2026-10-09, exploratory: four cuts, no gate; raw `docs/measurements/multimodal-support-2026-10/s896/`)

**The question** (owner, 2026-10-08: "figure out the prefill problem with the 896 image"): G-S10j read the resident DeepStack prefill FAIL on the 896-pixel image (803 rows; step-0 cos(on,HF) - cos(off,HF) = -0.0554, KL(HF||on) 0.607 against 0.341) and the hypothesis written down was "the long-sequence batched path, not the DeepStack add". Every cut below uses G-S10j's round-3 instrument unchanged (serve's `loadflags` Options, the CUDA tower's features and sets, the served prompt, `Qwen3-VL-2B`), whose step-0 numbers it reproduces to four digits (cut 0). Nothing here is graded; what it can and cannot support is at the end.

- **Cut 0, the instrument reproduces.** The dump re-run at the default chunk gives G-S10j's off-against-on worst cosines (0.9626 / 0.9868 / 0.9812 / 0.9677), and the per-layer script's step-0 numbers equal its table (KL against HF 0.3408 for the CPU arm and 0.6070 for the resident).
- **Cut 1, the chunk boundary: REFUTED.** The resident prefill splits a prompt over 512 rows into passes (`prefillChunked`); the 803-, 991- and 1,034-row prompts are the three that fail, the 89-row one is the one that does not. With `GOINFER_PREFILL_CHUNK=2048` those prompts run as ONE pass (confirmed by a temporary print of the chunk and the rows, removed) and **every logits file of both arms is byte-identical** to the two-pass run (`chunk-bit-identity.txt`). The chunked path is exact with DeepStack sets and m-RoPE, as `mropePosWindow`'s and the plan's absolute-index handling intended.
- **Cut 2, per layer against Hugging Face.** Both arms' residual after every layer (and its DeepStack add) was captured through the two seams that already exist (the CPU cache's batched capture, newly reachable from tests by `SetCaptureLayersForTest`; the resident's `SetBatchedCapture`) and differenced against Hugging Face's float32 text model fed the SAME embeddings, positions and sets (`scripts/s896_hf_layers.py`; per-layer tables `layers-*.txt`).
  - **A bug of mine caught on the way:** transformers' `output_hidden_states` records layer OUTPUTS and omits the DeepStack add (it differed from the state entering the next layer by 8.05 and 16 on the image rows), so a first run read 39% error at layer 0 and was wrong; the script now reads the state entering each next layer with pre-hooks and checks the convention.
  - **Layer 0 is identical between the arms** (same inputs, positions, set 0), so positions and the DeepStack add are not the problem. Both arms sit the same distance from Hugging Face from the start (relative error 12-13% on the image rows and 17% on the text rows at layer 0, about 10% and 20% by layer 4, on the 89-row image as on the 803-row one); the cause of that shared gap was not isolated here (serve's int4 embedding table is the likeliest, and it is common to both arms).
  - **On the 784 image rows the two arms are indistinguishable from Hugging Face at every layer** (cosines agree to three or four digits). **On the text rows the resident arm is, on average, slightly CLOSER** (mean cosine 0.9013 against 0.8998 at the last layer). Per row, the arm-against-arm difference is small and of both signs across the whole image block and the text rows (+-0.00x), with isolated swings in the last layer's text rows (-0.047, +0.052).
  - **The first-token row is the exception, and it is one row:** its residual is slightly farther from Hugging Face on the resident arm at every sampled layer from 4 on (cosine 0.9394 against 0.9516 at layer 27; identical at layer 0). Splitting body from head: the exact float32 head applied to each arm's own last residual gives cos 0.886 (CPU) against 0.830 (resident), KL 0.58 against 0.96, nearly the actual logits' 0.884 / 0.828 and KL 0.341 / 0.607, so **the gap is the body's last row, not the head**.
- **Cut 3, an A/A: how far does one correct CPU implementation move by itself?** The same dump with ONLY the CPU arm's attention changed from the fast float32 kernel (served's default, an operator-accepted divergence from the exact kernel, `GOINFER_CPU_FAST_ATTENTION=1`) to the exact one (`=0`), the resident arm untouched, against the same Hugging Face run (`aa-table.txt`):

  | image | cos(off, HF) step 0, fast | exact | A/A shift | cos(on, HF) | registered delta, on - fast | on - exact |
  |---|---|---|---|---|---|---|
  | 896² (803 rows) | 0.8836 | 0.8253 | **-0.0583** | 0.8282 | **-0.0554** | **+0.0029** |
  | 4x6 (89) | 0.9429 | 0.9429 | 0.0000 | 0.9423 | -0.0005 | -0.0005 |
  | formula (1,034) | 0.9652 | 0.9503 | -0.0149 | 0.9567 | -0.0085 | +0.0064 |
  | table (991) | 0.9586 | 0.9531 | -0.0055 | 0.9482 | -0.0104 | -0.0048 |

  Over all 36 (image, step) pairs the CPU variant alone shifts cos(off, HF) by a mean -0.0240 with a standard deviation of 0.0506 and a largest move of 0.1995. **The registered rule's mean delta is -0.0089 (FAIL below -0.005) against the fast kernel and +0.0124 (PASS) against the exact one: the choice of an equivalent CPU variant swings it by +0.0213, more than the whole band between its PASS and FAIL lines.** The 4x6 image shows no shift because the fast kernel does not engage on short prompts (its floor is by prompt length), which is also why that image has always behaved.
- **What this supports, and what it does not.**
  - **Supports:** G-S10j's FAIL is not evidence of a defect in the resident DeepStack prefill. The 896² image's -0.0554 is the same size as the shift one correct CPU kernel makes against itself on that image (-0.0583); at the residual level on this image the resident prefill is as close to Hugging Face as the CPU prefill on the image rows and on average on the text rows; its chunking is exact; against the exact CPU kernel it passes the registered rule on all four images (mean +0.0124, every step-0 delta within -0.0048). The registered statistic (one near-tied first token and eight teacher-forced steps per image, 4 images) cannot resolve a difference of the size it was asked to call, and its bars were set without measuring that noise floor.
  - **Does not support:** that the resident prefill is as good as the CPU one in general, or that it should be turned on. One image per size, one served configuration, one A/A variant (the fast kernel and the exact one differ by design, so this is a measure of what that difference does at step 0, not of every source of noise), one Hugging Face reference, and the last-row difference above is consistent in sign across layers on the one image where it was read. The sign of the near-tie that decides the first token (`ln p(Quarter)/p(Table)`: Hugging Face +1.69, CPU fast +1.30, CPU exact -0.36, resident -0.21) is a coin flip at int4, which is what an A/A of this size says.
- **The registration this calls for (not written yet, nothing queued): a noise-calibrated gate.** Same arms and instrument; many prompts per image instead of one (each prompt's last row is an independent draw, 16 or more per image); the decision rule's bar set from the A/A spread of two equivalent CPU variants measured in the same run, as a non-inferiority margin over it, with the do-nothing arm (the CPU prefill and upload, the production path today) as the comparator; an explicit parked band. It is about two minutes per image-and-prompt batch of dump plus Hugging Face, so it is a day-sized job for a few dozen prompts and a night job beyond that.
- **The owner's decision it feeds, unchanged in kind:** `cudaDeepstackPrefillOn` stays off until the owner decides; this pass removes one reason for it (a measured defect) and shows the other (a FAIL) rested on an instrument with no noise floor. The 13x prefill speedup (0.9 s against 11.8 s on table.png) is what is at stake on one side; on the other, the resident arm's last row is, on this one image, measurably farther from Hugging Face in every layer, and that stays unexplained.
- **Housekeeping found:** `cuda/s10_deepstack_prefill_test.go` still says in two comments that the production default is ON since G-S10g passed; `cudaDeepstackPrefillOn` is false and its own comment says why. Not edited here.

##### G-S10k, a noise-calibrated gate for the resident DeepStack prefill. Registered 2026-10-09 before any harness code or run (owner: "run the noise gate now")

**Why this gate and not G-S10j again:** the first pass above showed that G-S10j's bars (mean delta below -0.005 FAIL, any step-0 delta below -0.02 FAIL) sit inside the shift one correct CPU kernel makes against itself (the A/A: mean -0.0240, sd 0.0506 per (image, step) pair), and that the registered reading flipped from FAIL to PASS when only the comparator's attention variant changed. This gate measures that noise in the same run and sets the margin from it, with 16 independent prompts per image instead of one.

- **The claim it can support, fixed now:** "on the four F2a images at serve's configuration, the resident DeepStack prefill (ON) is no farther from Hugging Face float32 than the production prefill (the CPU prefill with the sets and the upload, OFF) by more than M, where M is how far two CPU kernels the project already ships differ from each other against the same reference." It says nothing about speed (the 13x is measured and not in question), about other images, other models, other quantisation, or about whether to enable it; that is the owner's decision.
- **Units.** 4 images (`gemma3_preprocess_image.png` 803 rows, `qwen25vl_preprocess_image.png` 89, `glm_ocr/formula.png` 1,034, `glm_ocr/table.png` 991) x 16 fixed prompts (`scripts/s896_prompts.json`, sha256 `a89dae6d0a8f46ffa3b421c456a0d1127522429ad4527ee0cbed42878b44de66...`, the served prompt first) = **64 units**. Each unit is one prompt through the chat template with the image, exactly as serve builds it; the prompts differ only in the question, so the image rows are identical across a unit's image and the last rows differ.
- **Arms, per unit** (serve's `loadflags` Options, the CUDA tower's features and sets, as G-S10j round 3; the context 2048):
  - **OFF** (the do-nothing arm, production today): the CPU prefill with the sets at the fast attention kernel (`GOINFER_CPU_FAST_ATTENTION=1`, serve's default), then the upload; the teacher path is OFF's greedy continuation, 3 tokens.
  - **OFFX** (the A/A arm): the same with the exact kernel (`GOINFER_CPU_FAST_ATTENTION=0`), teacher-forced along OFF's path.
  - **ON:** the resident DeepStack prefill, teacher-forced along OFF's path.
  - **ONX** (planted defect, red control): ON with `deepDefectOneLayerLate` (each DeepStack set added one layer late), the seam G-S10g planted.
  - The reference **HF**: Hugging Face transformers 5.15.0 float32 on the SAME ids and pixels (its own vision tower, one pass per image, the language model per unit), logits at the 4 positions.
- **Statistic.** For each arm and unit, c = the mean over the 4 positions (the prefill's last row and 3 teacher-forced steps) of the cosine between the arm's logits vector and HF's. Per unit: **d = c(ON) - c(OFF)** and **a = c(OFFX) - c(OFF)**. Reported with them, never in the verdict: KL(HF || arm) per arm, and the argmax agreement with HF. Cluster bootstrap, 10,000 resamples, seed 20261009, resampling units WITHIN each image (stratified, images weighted equally), 95% interval.
- **The margin, set from the A/A and fixed now.** **M = max(0.005, -mean(a))**: the distance, in the same statistic, by which the exact kernel (a shipped option, `--exact-prefill`) sits farther from HF than the fast default, if it does; 0.005 is the floor (G-S10j's own bar, the smallest margin worth naming). **This rule is the one judgement in the design** (what counts as "an implementation the project already accepts"), chosen before any data and open to the owner's challenge before the run.
- **Instrument sufficiency (a gate on the gate).** The bootstrap standard error of mean(a) must be at most M/2. If not, the reading is "no reading: the instrument cannot resolve M at this N" and ON is neither passed nor failed.
- **Decision rule.**
  - **PASS:** the lower bound of mean(d) is at least -M, and no single image's mean(d) is below -2M.
  - **FAIL:** the upper bound of mean(d) is below -M (confidently worse than the margin), or any single image's mean(d) is below -2M.
  - **PARKED (to the owner):** everything else, which includes a point estimate below -M whose interval still reaches it.
- **Controls, all before the verdict can count:**
  1. **Reproduction.** For each image's FIRST prompt (the served one) the OFF and ON logits files are **byte-identical** to G-S10j round 3's dump (the sha256 table in `s896/chunk-bit-identity.txt`): the instrument is the one that reproduced the served numbers.
  2. **Determinism.** ON dumped in both runs (below) is byte-identical across them; OFF run twice in the same configuration is byte-identical.
  3. **The red control.** ONX must read FAIL under the same rule (a planted defect that this gate cannot see would mean the gate cannot see anything).
  4. **HF sanity.** The chat template's ids equal the dump's on every unit; Hugging Face's step-0 argmax lies in the top 3 of at least one arm on at least 90% of units (G-S10j's check iii).
- **Execution.** Two runs of the same harness that differ only in the CPU attention knob (the second supplies OFFX); ON, ONX and the teacher path come from the first; the second's ON must equal the first's (control 2).
- **Prediction, written now.** A/A mean(a) between -0.01 and -0.03 (the 36-pair measurement gave -0.0240), so **M is about 0.02**; per-unit sd of d about 0.05, so the standard error of mean(d) at 64 units about 0.006; mean(d) between -0.02 and +0.01 (G-S10j's -0.0089 against OFF and +0.0124 against OFFX bracket it). Verdict: **PASS 0.45, PARKED 0.40, FAIL 0.15**; the control I expect to be hardest is the instrument-sufficiency check at the unit level (a per-unit sd of 0.05 is large).
- **Limits, stated now.** Four images are four clusters, so image-level conclusions rest on four; the 16 prompts vary only the question; one served configuration (int4, embed-int4 on, f32 KV); one pair of CPU kernels defines M, and they differ BY DESIGN (the fast one is an accepted divergence), so M measures what that choice costs, not every source of noise; cosine of logits at four positions is not a task metric; Hugging Face is one float32 reference; the units' prefills share an image, so the 64 are not 64 independent draws of the image side.
- **Tier and cost (TE11).** Estimate about **50 minutes** (HF float32 for 4 vision passes and 64 language-model forwards at 800-1,100 rows, about 15 min; goinfer's CPU prefill for OFF and OFFX at about 12 s per unit per run, 2 runs, about 26 min; ON and ONX about a second a unit). By the project's rule this is a night job; **the owner's "run it now" overrides it**, so it runs by day, detached, with a heartbeat every 30-60 s and an explicit timeout, and a 4-unit pilot first (plumbing, not a result, not quoted).

- **Amendment A1 to G-S10k, 2026-10-09, after the 8-unit pilot and before any graded run (control 3 only; no bar, statistic, margin, arm, unit or decision rule moves).** The pilot (plumbing; not a result and not quoted as one) measured how far each planted defect moves the statistic, c(planted ON) - c(OFF), mean over its 8 units: **sets one layer late -0.0073, sets not added at all -0.0119, sets added to the text rows too -0.5204.** The margin on the same 8 units was M = 0.0112. So the registered red control ("sets one layer late must read FAIL") is below M by construction: it read FAIL on the pilot only through the per-image guard, which is a reading of four noisy image means, not a test that the gate sees a defect of that size. A control a defect smaller than the margin can pass or fail by luck is not a control. **Control 3 is now:** the ON arm with the sets added to the text rows (a gross wrong-row defect, 70 times the margin on the pilot) **must read FAIL**, and the other two defects are dumped and **reported with their measured size** as the gate's sensitivity statement (what is NOT claimed: that this gate resolves a defect in the DeepStack add smaller than about M; removing the sets altogether moved the statistic by about M). The instrument-sufficiency check, the reproduction and determinism controls and the HF checks are unchanged, and all three pilot controls 1, 2 and 4 held (8 of 8 byte-identical to G-S10j round 3; ON identical across the two runs; template ids equal; HF argmax in an arm's top 3 on 100%). The pilot's A/A had se(mean a) 0.0079 against M/2 = 0.0056 (NOT HELD at 8 units, as expected: it shrinks with the root of the units).

##### (b') controls read 2026-10-09: HELD; the step is built and queued (not a reading of the 31B)

- **Control 1, streaming against ordinary on `testdata/gemma4-dense-twogeom-tiny`** (dense, two attention geometries, K=V on the global layer, softcap 30, tied head, sliding window 4; sequences of 16, 18 and 15 tokens so the window engages): per-position logits **min cosine 1.0000000000, max |difference| 4.17e-07**, against the registered 0.9999999 and 1e-5. **HELD.**
  - **The first attempt did NOT hold (min cosine 0.074) and that is the control working:** the streamer built standalone layers from a config that had never gone through a `PreTrainedModel`, so it had no attention implementation: the mask helpers returned no mask (the sdpa convention, `is_causal` carries it) while the layer ran the eager kernel, which attends to the future. Found by the control within a minute, fixed by naming `sdpa` for both (what `from_pretrained` gives an ordinary run).
- **Control 2, the six planted defects, each alone red against that bar:** no softcap (cosine 1.0000000, max |diff| 1.5e-3: only the absolute bar sees it, which is why the registration required both), no final norm (0.994, 1.9), the first two layers visited in swapped order (-0.100, 1.5), sliding layers given a full causal mask (-0.009, 1.65), the K=V tie dropped (a V projection of its own on the global layer: 0.576, 1.0), the embedding unscaled (0.609, 1.1). **All red.**
- **Control 3, determinism:** two streaming runs, bit-identical logits.
- **Plumbing, not results:** the whole chain (goinfer's arms in synthetic-ids mode on the tiny, the streamer, the grader with the tiny standing in for both models) connects end to end; its readings are meaningless and are not quoted. The real E4B loads and runs through `AutoModelForCausalLM` (21 s; the first draft's `Gemma4ForCausalLM` was wrong for its multimodal config).
- **Built:** `scripts/g31b_hf_stream.py`, `g31b_controls.py`, `g31b_grade.py`, `decoder/g31b_logits_real_test.go`, `docs/measurements/multimodal-support-2026-10/run-g31b-night.sh` (it runs the controls first and aborts on a failure; it refuses a G-31a path file whose sha256 is not the pinned one). **Queued as `g31b`, est 150 min** (tonight's queue is now 7 h 15 min, which fits only if the owner starts it by 23:15; drop `s14c4-2` or `g31b` to start later).

##### G-S10k read 2026-10-09: the registered verdict is FAIL, on the per-image guard alone; the pooled test passes (raw `docs/measurements/multimodal-support-2026-10/s896/gate/`; 64 units, `rev` in `provenance.txt`)

| | result |
|---|---|
| **units / controls** | 64 units, 16 per image. Control 1 (the served-prompt unit of each image byte-identical to G-S10j round 3): **8 of 8**. Control 2 (ON byte-identical across the two runs): **64 of 64**. Control 3 (the sets added to the text rows must read FAIL): **FAIL, mean -0.4712 [-0.5249, -0.4224], as required.** Control 4 (template ids equal on every unit; HF step-0 argmax in an arm's top 3): **True; 100% of units.** |
| **the A/A** | mean(a) = c(OFFX) - c(OFF) = **-0.0039** [-0.0088, +0.0010], unit-level sd 0.0201, per image gemma3 -0.0071, 4x6 0.0000, formula -0.0022, table -0.0062. **M = max(0.005, 0.0039) = 0.0050** (the floor). **Instrument sufficiency: se(mean a) = 0.002473 against M/2 = 0.0025: HELD, by 0.000027.** |
| **ON against OFF** | mean(d) = **+0.0007** [**-0.0044**, +0.0060], se 0.0027. Per image (registered): **896² (803 rows) -0.0111**, 4x6 +0.0097, formula -0.0027, table +0.0069. |
| **the rule** | PASS needs LB(mean d) >= -M **and** no image mean below -2M. LB = -0.0044 >= -0.0050: **the pooled half holds.** The 896² image's -0.0111 is below -2M = -0.0100: **the guard fires, so the registered verdict is FAIL.** (UB(mean d) = +0.0060 is not below -M, so the other FAIL clause did not.) |

- **The reading, stated without softening or excusing.** The registered rule says FAIL and the bar is not moved. What the data say underneath it is narrower than a FAIL suggests: pooled over the four images the resident DeepStack prefill is no farther from Hugging Face than the production prefill by more than the margin (point estimate +0.0007, interval inside it); on three of four images it is not measurably farther (4x6 is measurably CLOSER, +0.0097 [+0.0010, +0.0198], ON closer in 12 of 16 units); on the **896² image alone** it is farther in cosine, **-0.0111 (post hoc, that image's own bootstrap interval [-0.0200, -0.0031], ON closer in 3 of 16 units)**, and that is a real, repeatable, small difference, not one unit (13 of 16 prompts negative), and it is of the size of the shift one equivalent CPU kernel makes against itself on the same image (A/A -0.0071).
- **Two other metrics, reported and never in the verdict, point the other way, on all four images pooled and on the 896² image itself:** mean KL(HF || arm) **0.2560 for ON against 0.2715 for OFF** (896² alone 0.248 against 0.261), and argmax agreement with HF **0.832 against 0.816** (896² alone 0.797 against 0.781). Cosine of the logits vector, the registered statistic, and KL/argmax disagree about the 896² image; the registration chose cosine for continuity with G-S10g/j and I did not choose among them after seeing the data.
- **What moved the verdict, in the open.** M came out at its **floor** because the A/A averaged only -0.0039; I had predicted -0.01 to -0.03 and written M about 0.02 (on the 36-pair first-pass measurement of -0.0240). **That figure was dominated by one unit: the served prompt on the 896² image, at -0.081 here (the 4-position mean of the -0.0583 step-0 shift), against -0.0071 for that image over its 16 prompts.** The served prompt turned out to be an atypical draw, which G-S10j (one prompt per image) could not have known. A margin of 0.02 would have made the guard -0.04 and the verdict PASS; the rule I registered ties the margin to the A/A in the same run, which is what makes the gate honest and also what makes it stricter when the noise is smaller than predicted.
- **Prediction scored.** Verdict: PASS 0.45, PARKED 0.40, FAIL 0.15 (**the least likely outcome I gave came**). mean(a) -0.01 to -0.03 (**missed: -0.0039**); per-unit sd of d about 0.05 (**-0.0201 for a, d similar: smaller**); se(mean d) about 0.006 (**0.0027**); mean(d) between -0.02 and +0.01 (held: +0.0007). The instrument-sufficiency check, which I expected to be the hardest, held, narrowly.
- **What the gate can and cannot see (control 3's reported arms).** Sets added one layer late: mean -0.0010, **PARKED** (the gate cannot see it). Sets not added at all: mean -0.0054, **FAIL only through the per-image guard** (896² -0.0316). Sets added to the text rows: -0.4712, FAIL. So this gate resolves a gross wrong-row defect and, through the guard on one image, the total absence of the sets; it does not resolve a subtle DeepStack-add defect, and a pass here would not have been evidence that the add is exactly right (that evidence is G-S10g's tiny gates and the per-layer residuals, which match Hugging Face's on the image rows at every layer).
- **Effect on the first pass's conclusion, which stands as written with one correction:** the first pass said the resident prefill is as close to Hugging Face as the CPU prefill "on this image" at the residual level and that G-S10j's FAIL rested on an instrument with no noise floor. Both remain true of what they measured. This gate adds that, with the noise floor measured and 16 prompts per image, the 896² image still reads **farther in cosine by about 0.011 for the resident prefill**, so "no defect there" is not what the data support: "no localized fault found, and a small residual difference on that one image that the registered rule calls FAIL".
- **Outcome.** `cudaDeepstackPrefillOn` stays false: the registered reading is FAIL and this record does not overrule it. The owner's decision is now better informed than G-S10j's was: the resident prefill is pooled non-inferior at 0.005, measurably closer on the small image, within noise on two, and measurably (about 0.011 in cosine, better in KL and argmax) different on the 896² image; the 13x prefill speedup (0.9 s against 11.8 s on table.png) is what the other side of that choice is worth.
- **What this does not support:** that the resident prefill is safe to enable; that it is worse in any way a user would see (the 896² image's argmax agreement with Hugging Face is HIGHER for ON); anything about other images, other models or other quantisation. Four images are four clusters, one served configuration, one reference, one CPU-kernel pair defining M.
- **The follow-up this points at, not registered:** the 896² image is the only one that is measurably different, and it is the only one that is a square grid of 28 x 28 merged rows with no text structure around it; a gate with 64 prompts on that image alone and on one or two other large square images would say whether the difference is about the image's shape, and is a day-sized job (64 units of one image take about 13 minutes per dump run, two runs, and about 14 minutes of Hugging Face: roughly 40 minutes).

##### Owner decision, 2026-10-09: the resident DeepStack prefill is ON by default on CUDA ("just turn it on"), over G-S10k's registered FAIL

- **What changed.** `cudaDeepstackPrefillOn` in `cuda/prefill.go` is now `true`. A Qwen3-VL image turn on CUDA takes the resident m-RoPE prefill with the DeepStack sets (when every layer is a plain dense one) instead of the CPU prefill and the upload. Setting the variable false restores the old path exactly; there is no operator knob for it (a new one would be a new production env read or a new `Options` field, and nobody asked for one).
- **What it overrides, stated plainly.** G-S10j read FAIL (an instrument with no noise floor) and G-S10k read FAIL on its per-image guard alone (the 896-pixel image, -0.0111 against -0.0100). Both records stand as written; this is the owner's call to accept the difference they describe, not a finding that they were wrong.
- **What it rests on, all in this document.** The chunk boundary is exact (byte-identical logits at one pass against two); per-layer residuals against Hugging Face show no localized fault (the arms are indistinguishable on the 784 image rows at every layer); an A/A shows one equivalent CPU kernel moves the registered statistic as much as the "defect" did; G-S10k pooled: +0.0007 [-0.0044, +0.0060] against a margin of 0.0050, with the resident prefill measurably CLOSER on the 4x6 image and, on the 896 image, farther in logits cosine by about 0.011 but closer in KL and in argmax agreement with Hugging Face. What it buys: the prefill is about 13x faster (0.9 s against 11.8 s on table.png, G-S10g/j).
- **Measured after the flip (exploratory; a speed check, not a gate).** `serve` built from the flipped tree, Qwen3-VL-2B at serve's defaults on the RTX 2070 SUPER, the S7 harness's one cell (a new image per request, one warm-up and three timed): **median TTFT 2.20 s** (2.17, 2.22, 2.20; warm-up 3.98), against **10.79 s** in S7's third read with the CPU prefill, the one cell that was over the 5 s bar. The serve log reads `decode path: cuda-resident (int4)` and `prefill path: batched`. So every S7 cell is now under the bar on CUDA.
- **Housekeeping done with it.** The recommended-checkpoint line for `qwen3-vl-2b` said the image turn took about 15 s because the DeepStack prefill runs on the CPU; it now states the 2.2 s measurement and the G-S10k caveat (the capability-matrix copies regenerated). The gates (`TestS10DeepstackPrefillCUDA_*`, the dumps) set the variable explicitly and put it back, so their meaning is unchanged; the tiny gate passes with the new default.
- **What this does not claim:** that the resident prefill is as close to Hugging Face as the CPU one on every image (it is not, on the 896 image, by a small cosine margin), or anything about Metal (its own switch, `metalDeepstackPrefillOn`, is the Mac's) or about other Qwen3-VL sizes.

#### The queue of 2026-10-09, run by day (09:43-12:45 PDT, 3 h 02 min of 7 h 15 min estimated), read the same day (nobara; logs under `~/goinfer-logs/night/runs/2026-10-09/`)

Four jobs ran and read; the fifth, `s6-moe-image-3`, was queued onto this box's queue by the Mac session (the 35B bundle rebuilt at 11:56 for the m-RoPE section) and failed for the same reason as the first (below), so it is superseded, not graded. Everything below is read against the registrations above.

##### G-S6m, the 35B's served image check: PASS, after a server bug the check itself found (raw `~/goinfer-logs/s6-moe-fixed-2026-10-09/`, binary rev in `provenance.txt`)

- **First, the failures.** `s6-moe-image-2` (this box) and `s6-moe-image-3` (the Mac's) both failed in about two minutes, with the first arm correctly `cuda-resident (int4mix)` (the `--moe-cache-experts` repair held) and then the harness client dying on `JSONDecodeError: Expecting value: line 1 column 1`. **Replayed by hand against the same pinned binary: the server answered the image request with HTTP 200 and `Content-Length: 0`**, on a first send and on a resend; the same request without `logprobs` returned a normal answer ("This image shows a collection of ..."); text requests were fine.
- **Cause, in two layers.** (1) `writeJSON` wrote the 200 status and then streamed the encoder's output with its error dropped, so a body JSON cannot carry became an empty 200. (2) The body could not be carried because a logprob was -Inf: at step 24 of the 32-token answer the **thinking budget forces the end-of-thinking token** (id 248069, probability 1, logprob 0) and the other two top-3 entries, filler ids 0 and 1, have probability zero (logprob -Inf). That is a normal state, not a numerical fault. Found by building the server with the encode error surfaced ("json: unsupported value: -Inf") and then a temporary print naming the entries (removed).
- **Fixed** (`43e35056`): `writeJSON` encodes first and answers an unencodable body with a 500 that names the problem (red on the old code: status 200, empty body; `TestWriteJSON_*`); the logprobs builder reports **-9999** for a zero-probability token, OpenAI's own convention (`TestJSONLogprob_*`; NaN is deliberately left to be reported). The whole `serveapp` package passes.
- **G-S6m, as registered** (`~/models/qwen3.6-35b-a3b-int4.giw` rebuilt 11:56; one image request, 32 greedy tokens, top-3 logprobs; three arms, `--backend cuda --moe-cache-experts`): the CUDA-tower reply **IDENTICAL** to the CPU-tower reply; the CPU-tower repeat **IDENTICAL**; **every arm decoded `cuda-resident (int4mix)`**. 37-47 s per request (the image turn prefills on the CPU: no MoE hybrid image gate exists). **PASS.** Prediction ("identical") held. What it establishes: the 35B runs an image turn end to end on this box and the CUDA tower is the CPU tower in the served path; there is no Hugging Face anchor for the 35B (the claim was never "matches HF").
- **Two limits, and a trap.** The binary was built from `70de7994` plus the serve fix before it was committed (`serve-cuda.rev` read `70de7994`; the source is `43e35056`). The server's swap guard tripped on the second request in one process when the box already had 5.3 GB of swap in use ("swap grew 0.58 GB over baseline; refusing new requests"): serving the 35B on this 62 GB box next to other work is marginal, and the harness's clean first request is what avoided it.

##### G-S14c4 re-read: G-S14c4a PASS, G-S14c4b FAIL again, and the added arm overturns a hypothesis (`~/goinfer-logs/s14c4-2026-10-09/`, 51 min, binaries `s14c4b` at `c7df2f72`)

| arm | WER | control tokens in replies | first-run WER |
|---|---|---|---|
| R, transformers float32 | 4.09% | 0 | 4.09% |
| F, goinfer native weights | **4.09%** | 0 | 4.96% |
| I, goinfer int4 (serve defaults) | 8.26% | **33 of 73** | 9.13% |
| S, control: F on the next clip's audio | 127.39% | 0 | 126.96% |
| A1, record only: int4 body, int8 head | **16.78%** | 3 | (new) |

- **G-S14c4a: PASS.** F is byte-equal to R on **73 of 73** clips and |WER_F - WER_R| = **0.00** (bars: 70 of 73, 0.30). The first reading's NOT MET (71 of 73, 0.87) was the null-EOS stop-id bug (`9775a315`) and nothing else; it is superseded by this re-read, not erased.
- **G-S14c4b: FAIL, unchanged:** WER_I - WER_F = **+4.17** [+0.61, +9.52], 15 clips differ, **33 control tokens (parks regardless)**. The stop-id bug cancelled in this difference, as predicted before the re-read; so the two empty int4 replies were NOT that artefact (2 empty replies remain with the fix).
- **The added record-only arm A1 refutes "the int4 head damages the first token".** With the head at the int8 pin the stray `<|im_start|>` prefixes fall from 33 to 3, but **WER rises from +4.17 to +12.70 points** because 7 replies come back as an empty string and 3 more as garbage, against 1 and 2 with the int4 head.
  | first generated token | float32 | int4 body + int4 head | int4 body + int8 head |
  |---|---|---|---|
  | `language ...` | 73 | 37 | 60 |
  | `<|im_start|>language ...` | 0 | **33** | 3 |
  | empty (a stop token first) | 0 | 1 | **7** |
  | other (garbage, a lost prefix) | 0 | 2 | 3 |
  So the int4 **body** leaves the first position a near-tie among `language`, `<|im_start|>` and a stop token, and the head's precision only changes which wrong one wins. The stray control token is the MILD symptom: the metric's normaliser strips it, so it costs no WER; the costly failures are a premature stop and a garbage first token.
- **Prediction scored.** R and F near 2-4% (held: 4.09, 4.09); F byte-equal on 90-100% (held: 100%); WER_I - WER_F within +0.5 (**missed**, +4.17); control tokens in at most two clips (**missed**, 33). The registered hypotheses for the first reading ((i) the int4 body or head damages the first token; (ii) the empties are the stop-id bug) read: (i) true of the body and false of the head as the cause; (ii) false.
- **What this means for the control-token decision (still the owner's).** Forbidding or stripping control tokens would remove the visible symptom and cost nothing in WER terms, and would leave the premature and garbage first tokens (3 of 73 replies with the int4 head, 10 of 73 with the int8 head) and the 15 clips with changed words. What would address the cause is a decode constraint on the first tokens (the assistant turn must open `language <name><asr_text>`, which the Qwen3-ASR prompt layout already supports as a forced prefix), which the reference does not do and which is a product choice, not a repair; or finding why the int4 body flattens that one position.

##### G-S14c4d, the partial forced prefix for Qwen3-ASR: registered 2026-10-10 before any code (owner: "yes do that", on the option list after the G-S14c4 re-read)

**What is tried, and why this one.** The G-S14c4 re-read put the int4 damage at ONE position: the first generated token after `assistant\n`. Float32 writes `language` there on 73 of 73 clips; the int4 body (head int4) writes `language` on 37, `<|im_start|>` on 33, a stop token on 1, something else on 2. **The partial force prefills that one token**: the assistant turn is opened with `language` already in it (the id the model itself would have generated first), and the model writes ` English<asr_text>` and the transcript itself. It keeps what the full `language <Name><asr_text>` force would lose: the model still detects the language (no API surface, no wrong-language risk), and serve can keep the raw reply shape by emitting `language` first. The full force stays an option and is not tried here.

**What it can and cannot show.** It removes the contested position, so the 33 stray `<|im_start|>` and the stop-first and garbage-first replies there cannot occur. It moves the fragility, it does not remove it: the new first generated token is the language NAME, and the first transcript token follows `<asr_text>`; whether the int4 body is steady at those positions is exactly what is not known and is what the gate reads. Nothing here speaks for non-English audio: the data is English (LibriSpeech), so "the model still detects the language" is a design property, **not a result**; a non-English check is out of scope and stated as not covered.

**Arms (one binary, `TestQwen3ASRWER_arms` with `GOINFER_S14C4_PF=1`, which adds three arms to the three it has; the reference R and its refs are the 2026-10-09 `hf.json`, same 73 clips, checked by sha256 of `refs.json`):** `f32`, `int4`, `int4h8` as before (re-run in this binary, and compared byte for byte with 2026-10-09's `go.json` as a control that this change did not touch the unforced path); new `f32pf`, `int4pf` (serve's defaults plus the partial force), `int4h8pf` (record only). A partial-force reply is stored as `"language" + decode(generated tokens)`, which for a model that would have said `language` itself is the same string the unforced arm stores, so the arms compare directly.

**Bars (the existing ones, applied to the new arms; nothing is re-based).**
- **G-S14c4d-a, the force is harmless where the model was right:** `f32pf` byte-equal to `f32` on at least 72 of 73 clips and |WER_f32pf - WER_f32| <= 0.30 points (G-S14c4a's width). Float32 wrote `language` first on all 73, so any larger difference is the prompt, not the model. FAIL otherwise, and a FAIL here stops the idea.
- **G-S14c4d-b, the point:** G-S14c4b's rule on `int4pf` against `f32`: **PASS** if WER_int4pf - WER_f32 <= +0.5 and its bootstrap 95% upper bound <= +1.5 (the same 10,000-resample cluster-by-clip bootstrap, seed 20261008); **PARKED** if the difference is <= +1.5; **FAIL** above. Control tokens in `int4pf` replies: more than 3 parks a PASS (G-S14c4b's rule). For reference the unforced arm read +4.17 [+0.61, +9.52] with 33 control tokens.
- **Record only, never graded:** `int4h8pf`; the table of what the model writes at each position after the forced `language` (the name token: ` English` expected on 73; then `<asr_text>`; then the first transcript token: empty, a control token, garbage, a word); the count of clips where the detected language differs from `f32`'s; and the same table for `int4` and `f32` for comparison.

**Stopping rule:** fixed (73 clips, greedy, 256 tokens at most). **Cost:** the Go side was 510 s for three arms on 2026-10-09 (f32 242 s, int4 86 s, int4h8 82 s), so six arms are about 14-16 minutes; no HF or Gemma rerun; queued at 30 on the night queue (over the by-day 10 minutes). **Tier:** night, correctness gate, not timed.

**Prediction (written now).** d-a PASS at 0.90. d-b: PASS at 0.50, PARKED at 0.25, FAIL at 0.25. The reasoning: the empties and garbage first tokens were 3.3 of the 4.17 points and live at the removed position, which would leave about +0.9; against that, the int4 body is flat at one position and a new flat position (the first transcript token) would put empties back. Control tokens at most 2 in `int4pf` at 0.85.

**What each result does next.** PASS: the partial force becomes serve's behaviour for Qwen3-ASR with the reply shape unchanged (serve emits `language` ahead of the model's text and counts one fewer completion token), after the owner's say; the unforced path stays reachable. PARKED or FAIL with the first-token table showing the fragility moved to the transcript's first token: the partial force is dropped and the full force or a constraint on the first K tokens is the next registration. d-a FAIL: the idea ends there.

##### Option D complete: 7 of 7 models graded, none COSTLY; the registered map matches nothing, so the numbers go to the owner (`~/goinfer-logs/head-precision-2026-10-09/`, 1 h 12 min, binary `hp2` at `9a021498`)

| model (head) | agree int8 -> int4 | d_agree [95%] | dKL [95%] | ratio | reading |
|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct (tied) | 82.8 -> 83.1 | -0.29 [-1.95, +1.46] | +0.0182 [+0.0086, +0.0275] | 1.11x | MIXED |
| qwen3-1.7b-bf16 (tied) | 93.2 -> 93.1 | +0.10 [-0.78, +0.98] | +0.0190 [+0.0047, +0.0356] | 1.17x | MIXED |
| qwen25vl-3b-instruct (tied) | 89.5 -> 87.8 | **+1.66** [+0.20, +3.12] | +0.0216 [+0.0123, +0.0317] | 1.31x | MIXED |
| gemma-3-4b-it (tied) | 91.1 -> 89.5 | **+1.66** [+0.39, +2.93] | **+0.0453** [+0.0196, +0.0713] | 1.23x | MIXED |
| tinyllama-1.1b-chat (untied) | 91.0 -> 90.5 | +0.49 [-0.68, +1.66] | +0.0042 [+0.0003, +0.0079] | 1.07x | OK |
| phi3-mini-4k (untied; **now read**) | 89.3 -> 89.6 | -0.39 [-1.95, +1.07] | +0.0107 [-0.0012, +0.0248] | 1.07x | MIXED |
| olmo3-7b-think (untied, its own repository's chat template) | 96.3 -> 96.7 | -0.39 [-1.46, +0.78] | **-0.0218** [-0.0392, -0.0072] | **0.54x** | OK |

- **The plain reading:** the int4 head costs at most **+1.66 points** of top-1 agreement (Qwen2.5-VL-3B and Gemma 3) and **at most +0.045 nats** of KL (Gemma 3); on olmo3 it is *better* in KL than the int8 head (0.54x) with the interval excluding zero. The "3.5x the KL to Hugging Face on one real image" on Qwen2.5-VL-3B is **not reproduced on text: 1.31x**. The "about 2.3 points" on file is not reproduced either (largest 1.66). **0 of 4 tied-head models and 0 of 3 untied are COSTLY; the registered map ("COSTLY on two or more tied-head models" etc.) matches nothing, so the decision is the owner's, as registered.**
- **Prediction scored:** at most two COSTLY (held: 0); 1-3 points and 0.02-0.06 nats on the tied small-vocabulary models (the agreement cost came in at or below the range: the two smallest read -0.29 and +0.10; the KL cost 0.018-0.045, mostly at its floor).
- **What the fix of the inert flag did, measured:** phi3's int4-head arm is no longer VOID (`HeadTable()` read int8 then int4 across the arms on all three of this night's models), and its reading is MIXED with the interval of dKL touching zero.
- **A control that did not run, disclosed.** The positive control step (both arms at the int8 pin must read exactly 0 / 0) failed in 0 s: it reads `qwen2.5-0.5b-instruct`, which is not among this night's three models, so there was no HF dump for it. **The exact-zero control was therefore not re-observed on this night's binary** (night 1's read exactly 0 / 0 on its own); what held on this binary is the `HeadTable()` assertion. A repeat would add the 0.5B to `HP_MODELS`.

##### (b'), the Gemma 4 31B against a layer-streaming Hugging Face float32 reference: both arms PASS as registered (`~/goinfer-logs/g31b-2026-10-09/`, 45 min against 150 queued, binaries `g31b` at `9d5e7943`)

- **Controls (run first by the job, held):** streaming against ordinary on the 31B-shaped tiny at max |diff| 4.17e-07, the six planted defects red, bit-identical twice; the sequences are G-31a's pinned path files (sha256 checked by the script).
- **Result** (12 prompts, 373 positions for the 31B and 384 for the E4B, cluster bootstrap over the prompts):

  | model, arm | agreement with HF float32 | mean KL(HF || arm) |
  |---|---|---|
  | 31B int8int8 | **94.64%** [91.5, 97.1] | 0.0955 [0.0592, 0.1430] |
  | 31B int4 | **94.37%** [91.2, 96.6] | 0.1444 [0.0832, 0.2283] |
  | E4B int8int8 | 97.92% [96.6, 99.2] | 0.0083 [0.0050, 0.0121] |
  | E4B int4 | 92.97% [89.1, 96.4] | 0.1114 [0.0632, 0.1655] |

  **int8int8: the 31B is 3.28 points below the E4B (bar: PASS at -5.0 or better) -> PASS. int4: the 31B is 1.40 points ABOVE the E4B -> PASS. Both arms PASS, so the status row may change** as the registration worded it: "agrees with Hugging Face float32 as well as the E4B does, teacher-forced on 12 prompts, CPU; not validated on images or on contexts past the sliding window" (the status table is updated to that, and which parity tier it earns under the matrix's rules is the owner's call, not this record's).
- **Said plainly, beside the PASS:** the registered statistic is agreement, and on KL the 31B's int8int8 arm is **11 times** the E4B's (0.0955 against 0.0083, intervals not overlapping); the E4B's int8int8 is nearly float32, the 31B's is not. That is the failure mode I named in advance (60 layers accumulate more quantisation drift than 42) showing in the divergence and not in the top-1. It is not in the rule and does not change the verdict; it does say "as well as the E4B" holds for the argmax and not for the distribution.
- **Prediction scored:** the E4B at 70-85% on both arms (**missed: 97.9% and 93.0%**; the "about 77% from Hugging Face" in G-31a's registration was a different quantity); the 31B's int8int8 within 5 of it with probability 0.65 (held), its int4 with 0.55 (held); controls pass with 0.8 (held, after the first attempt's bug, found and fixed by the control before any reading).
- **Limits (registered, unchanged):** sequences under 100 tokens never reach the 1,024-position sliding window (control 1's tiny is the only place the window engages); 12 prompts along goinfer's own int8int8 path; the E4B is a different shape; CPU only.

##### Findings for the owner from this night (none is a gate; each is stated with its evidence)

- **`--embed-int4` was inert on nine families** (list above), found because the registered assertion caught phi3. **Fixed 2026-10-09** (above); on those families the default now gives an int4 head. It bore directly on option C, which is no longer blocked by it.
- **The int4 head costs less than the file says on text** (at most 1.66 points and 1.23x the KL on four models), and the 3.5x on one image is untested on text until Qwen2.5-VL-3B is read tonight. The decision is still the owner's.
- **Qwen3-ASR at int4 (serve's default) has a first-token problem**: 33 of 73 replies start with a stray `<|im_start|>`, two come back empty, and WER is +4.2 points over float32 before the stop-id fix is accounted for. The re-read separates the stop-id artefact from real damage and the `int4h8` arm says whether the head is the cause. Until then, the registered question "should serve forbid control tokens in ASR output" has data on its side (33 of 73), and the answer is still the owner's.

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

**Where the order stands, 2026-10-09:** item 1 is done except S6. Item 2 is read (the Mac's third pass is queued). In
item 3, S16 and S17 have shipped their levers and S18 owes only G-S18g. Next by the owner's word ("lets do your
suggestion"): G-IP4, then S11 (it unblocks S15), then S8.

**Where the order stands, 2026-10-10:** in item 1, S6 owes E4B on Metal. Item 2 is read three times on the Mac. In
item 3, S18 owes G-S18g2. In item 4, S10's families are done but for Qwen3-VL MoE's served check ("North" dropped, the
Qwen3.5+ MoE images covered by S6's G-S6m), S11 and S8 are done, and S14 is in flight. What has not started, in the
list's order: S15, S12, full S13.

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

- ~~Gemma 3's int8 SigLIP default under `--backend cuda|webgpu`.~~ Decided 2026-10-08: float32 on CUDA ("float32 it
  is"); WebGPU stays int8 until S12 gives it a float32 tower.
- ~~Whether `s2-towers` is merged and pushed.~~ Merged 2026-10-08 (`471a500e`).
- serve's `--embed-int4` default differing by backend on tied-head families (the int4 head measured 3.5x the int8
  head's KL to HF on a real image). Option D's first night (2026-10-09 read, above): at most +1.66 points and 1.23x KL on
  four models; the flag is inert on nine families; Qwen2.5-VL-3B (where the 3.5x was seen) is read on the second night.
- S14's speech family (P11's options; decided when S14 starts).
- ~~Which VL checkpoint `pull` recommends per box class (P9(d)).~~ Decided and shipped 2026-10-08, "P9(d) done".
