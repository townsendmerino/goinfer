# Multimodal audit, 2026-10-02: what image input actually does, per family and per backend

**What this is.** `docs/multimodal.md`'s status paragraph was dated 2026-09-15 and known stale (`docs/tasks/task-glm-ocr-2026-10.md`, O7 RESULT: "NOT done:
the per-backend audit"). This record re-derives the per-family x per-backend table from the tree and from what ran on `nobara-pc` today, says which cells
were run, which were only read, and which could not be checked, and lists every claim in the docs that turned out wrong. The docs were then rewritten from it
(`docs/multimodal.md` status block, `docs/server.md` vision paragraph, and the others named under "Claims that were wrong").

**Provenance.** 2026-10-02, 10:08-10:30 PDT. `nobara-pc`, linux/amd64, RTX 2070 SUPER 8 GB, driver 595.91.07, Ryzen 3700X, idle box (load 0.13) but 14.75 GB of
swap already in use before any load. goinfer `9692b547`, tree clean. `go.work` in place, so aikit is the sibling checkout `v1.52.0-3-g9e8e242` (goinfer's five
modules pin `v1.52.0`; the three commits on top are unreleased). WebGPU ran on the box's Vulkan adapter (the same RTX card), through a cgo build. **Not run:
anything on a Mac.** Every Metal cell is read from code. Nothing here is a benchmark: the timings below are one-off wall times from single served requests
("exploratory, never quoted as a result", CLAUDE.md run budget) and the recorded numbers are carried with their own dates.

**Labels.** `verified-run` = I ran it today and read the result (test `--- PASS` line or a served request). `read-from-code` = read at the cited line, not run
here. `recorded <date>` = a dated record in the tree supports it; not re-measured. `unverified` = nothing I could run or read settles it.

## What ran (all of it; everything else in this file was read)

Logs and the probe harness are in `docs/measurements/multimodal-audit-2026-10-02/` (`probe.py`, `probe-console.txt`, the `t-*.log` test logs, the `probe-*.log`
server logs). The test image is `redsq.png` there (a 336x336 red square on blue with a yellow disc) except Gemma 3, which always resizes to 896x896
(`testdata/gemma3_preprocess_image.png`). Prompt "What shapes and colours are in this image? Answer briefly.", temperature 0, `max_tokens` 24, each request sent twice.
No numeric result below is a quality claim; the replies only show the image was seen.

| id | what | result |
|---|---|---|
| B1 | `CGO_ENABLED=0 go build -tags cuda ./cmd/serve` in `cuda/`, then `go version -m` and `go list -deps` | builds; `build CGO_ENABLED=0`, `-tags=cuda`; `--version` prints `backends: cpu cuda`; the aikit packages linked are `gpu`, `vision` and the CPU libraries, **not** `gpu/visioncuda` or `gpu/qwencuda` |
| B2 | `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build ./cmd/serve` in `metal/`, `go list -deps` | builds; aikit packages linked: `vision`, `gpu`; **not** `gpu/visionmetal` or `gpu/qwenmetal` |
| B3 | `CGO_ENABLED=0 go build -tags gpu ./cmd/serve` in `gpu/`; then with `CGO_ENABLED=1` | the first fails inside `oliverbestmann/webgpu/wgpu` (`undefined: Texture`...); the second builds, `--version` prints `backends: cpu webgpu`, and the binary links libc |
| P1 | `serve-cuda --backend cuda --model gemma-3-4b-it`, one image twice | banner `encoder int8/cuda-resident`, `decode path: cuda-resident (int4)`; cold request 4.9 s end to end, repeat 0.3 s with `prefill_reused_tokens` 280 |
| P2 | `serve-cuda --model qwen2.5-vl-3b` (no `--backend`, so auto) | `backend: cuda (auto: a CUDA device answered)`; tower line has no "resident"; cold 2.0 s, repeat 0.1 s with reuse 165 |
| P3 | `serve-cuda --backend cuda --model qwen3.5-0.8b` | text `decode path: cuda-resident (int4)`; image turn 7.0 s and again 6.7 s, **reuse 0 both times** (the tower runs again on the repeat) |
| P4 | `serve-cuda --backend cuda --model gemma-4-E2B_q4_0-it.gguf --vision gemma-4-E2B-unq` | `decode path: cpu (int4) - requested cuda -> running on cpu: cuda does not implement [gemma4-e-model]`; 25.3 s and 24.4 s per image, reuse 0 |
| P5 | `serve-cuda --backend cuda --model glm-ocr` | banner `tower (f32, CPU) loads on first image`; `decode path: cuda-resident (int4)`; cold 2.8 s, repeat 0.1 s with reuse 163 |
| P6 | `serve-cuda --backend cuda --model ministral3-3b-bf16` (`model_type` mistral3), an image | HTTP 400 `this model has no vision tower` |
| P7 | `serve-cuda --backend cuda --moe-cache-experts --model gemma4-26b-int4.giw --vision gemma-4-26b-a4b-it` | `decode path: cuda-resident (int4)`, Gemma 4 tower line printed. **Run 2: correct reply in 68.5 s.** Run 1: HTTP 200, `completion_tokens` 0, empty reply, 31.4 s, not reproduced and not explained (see "Code and matrix observations") |
| P8 | `serve-gpu --backend webgpu --ctx 4096 --kv-sessions 1 --model gemma-3-4b-it` | `decode path: webgpu:vulkan-resident (int4)`, banner `encoder int8/webgpu-resident`; cold 27.4 s, repeat 0.8 s with reuse 280, same reply text as P1. **Without `--ctx 4096` the server exited at startup** (the tower upload ran out of VRAM after the decoder took it) |
| P9 | `serve-gpu --backend webgpu --ctx 4096 --kv-sessions 1 --model qwen2.5-vl-3b` | tower line has no "resident"; cold 6.1 s, repeat 0.1 s with reuse 165 |
| P10 | `serve-gpu --backend webgpu ... --model glm-ocr` | `decode path: webgpu-staged (int4) ... webgpu does not implement [pairwise-mrope pairwise-rope]`; reuse 0 on the repeat |
| T1 | `cuda`: `TestGemma3VisionResidentReal_gate` (`-tags 'cuda goinfer_testhooks'`) | `--- PASS` 26.78 s, CPU int8 tower vs resident CUDA tower cosine 0.913292, max abs diff 31.2 |
| T2 | `cuda`: `TestQwen25VLMRoPEPrefillResidentReal_gate` | `--- PASS` 11.50 s, CPU argmax 264 = resident 264, cosine 0.998644, `ImgPrefillResident=true` |
| T3 | `cuda`: `TestQwen25VLResidentReal_gate` | `--- PASS` 11.85 s, mropeDelta -3, one decode step past the image: argmax 1697 = 1697, cosine 1.000000 |
| T4 | `cuda`: `TestGemma3ResidentReal_gate`, `TestGemma3ImgPrefillResidentReal_gate` | **`--- SKIP`** both: `BuildResident declined ... default resident context 16168 positions does not fit (4.50 GB of KV, 4.57 GB free)`. Not run. |
| T5 | `cuda`: `TestGemma4VLResident_bidirParity` | `--- PASS` 1.69 s, **scaled synthetic fixture**, CUDA vs CPU-int4 mean cosine 0.842, exact argmax 6/6 |
| T6 | `cuda`: `TestGlmOcrResidentParityCUDA`, `TestGlmOcrResidentAdmissionCUDA` | both `--- PASS` (tiny fixture: pairwise kernels text decode worst cos 0.999772, image m-RoPE prefill cos 1.000000; NeoX control -0.34) |
| T7 | `decoder`: `TestGemma3VL_ TestGemma4VL TestGemma4VLBidir_ TestQwen25VL_ TestQwen35VL_ TestGenerateVL_ TestGenerateQwenVL_ TestGenerateGemma4VL_ TestGlmOcr` (tiny fixtures and fake residents) | 41 top-level `--- PASS` (49 with subtests), 0 SKIP, 0 FAIL |
| T8 | `internal/serveapp`: `TestIsQwen35VisionDir TestSetupQwen35Vision TestIsGlmOcrVisionDir TestSetupGlmOcrVision TestImageFitsContext TestGlmOcr_ TestLoadVisionTower` | 11 `--- PASS` |
| T9 | `gpu` (`-tags 'gpu goinfer_testhooks'`, cgo): `TestVisionEncoder_parity TestVisionEnableResident_parity`, and with `GOINFER_HEAVY_TESTS=1` `TestUploadKV_matchesSequentialForward TestForwardMRoPE_*` | all `--- PASS`: tower cosine 1.000000 vs CPU W8A8 and 0.999959 vs the HF golden (tiny SigLIP fixture); `EnableResident` cosine 1.000000; UploadKV bridge decode cosine 1.00000000 (base 0 and ring-wrapped); ForwardMRoPE shift invariance cosine 1.00000000 |

Not run and why: the real 26B served probe a third time (swap guard, above); the Gemma 3 cosine gates (T4, VRAM); a real GLM-OCR page (O3's record stands, 44 s of CPU
tower per page); everything on a Mac; the CPU-tower timings (not re-measured); `--backend cpu` serving (nothing in this audit depends on it beyond what T7 covers).

## The truth table

Reading it: **(1)** where the vision tower runs, **(2)** where the decoder runs after the image, **(3)** which release binaries can do it. The "binaries" column
is the same for a backend whatever the family, so it is stated once here:

| backend | release binaries that carry it | cgo |
|---|---|---|
| CPU | all six (`goinfer-serve-{linux,darwin,windows}-{amd64,arm64}`) | none |
| CUDA | `goinfer-serve-linux-amd64` and `-arm64`, built from `cuda/cmd/serve` with `-tags cuda` (`.github/workflows/release-assets.yml:204-205`) | **none**: `CGO_ENABLED=0` (`release-assets.yml:193`); the driver is opened at run time. B1 + P1 (verified-run) |
| Metal | `goinfer-serve-darwin-arm64` and `-amd64`, built from `metal/cmd/serve` (`release-assets.yml:202-203`) | none |
| WebGPU | **no release binary** (`release-assets.yml:155`: "`-tags gpu` (WebGPU) is deliberately NOT here: it needs cgo"); self-build `gpu/cmd/serve` with `-tags gpu` and a C toolchain | cgo (B3, verified-run) |

`goinfer-chat` has the same per-platform binaries (`release-assets.yml:79-84`) but its `--image` is **GLM-OCR only** (`internal/chatapp/image.go:111`, tower f32 on the
CPU always). `demo/agent` takes images for **Gemma 3 only**, with the tower on `cpu` or `webgpu` (the `--vision-backend` flag in `demo/agent/cmd/agent-web/main.go`, `demo/agent/agent/agent.go:285-307`): no CUDA tower there.

*Changed 2026-10-02 (item 2 below): the tower-quant cells in the tables describe the rule as audited. Today Qwen2.5-VL, Qwen3.5+ and Gemma 4 load f32 unless `-vision-quant int8`, on every backend.*

**The tower-quant rule, exactly as `internal/serveapp/main.go` does it today** (`:1102`): `int8Tower := visionQuant == "int8" || backend == "webgpu" || backend == "cuda"`
(`--backend auto` is resolved to a concrete backend first, `internal/loadflags/loadflags.go:110-152`, verified by P2's banner). That value is passed to the Gemma 3,
Qwen2.5-VL, Qwen3.5+ and Gemma 4 loaders (`:1103-1118`). **GLM-OCR is the exception**: it gets `visionQuant == "int8"` only (`:1109-1112`). Only the Gemma 3 tower
has a resident GPU path that needs int8 weights; for the other three the rule gives a **CPU int8 tower** on CUDA and WebGPU. `EnableResident` is called in exactly one
place in production code (`:1125-1127`, the Gemma 3 `vision.Encoder`, on `webgpu` and `cuda`); a `grep -rn EnableResident` over the non-test Go files finds only that and `demo/agent/agent/agent.go:303`.

### Gemma 3 (SigLIP; `model_type` gemma3 with a `vision_config`)

| backend | (1) tower | (2) decoder after the image | verdict |
|---|---|---|---|
| CPU | CPU SigLIP, f32 unless `--vision-quant int8`; 31.3 s/image | CPU | read-from-code (`internal/serveapp/main.go`); tower time recorded 2026-09-08 (`docs/benchmarks.md` §A "Vision tower CPU prefill"), not re-measured; tiny-fixture parity verified-run (T7: `TestGemma3VL_imageParity`) |
| CUDA | **goinfer's own resident CUDA tower** (`cuda/vision_encoder.go`, registered at `cuda/vision_register.go:13`; not aikit's `visioncuda`), int8, `EnableResident` called (`internal/serveapp/main.go`). **4.1 s/image** with the fused attention that is default since 2026-09-21 (26.0 s with `GOINFER_CUDA_VISION_ATTN=exact`) | resident: `PrefillImageLast` (`cuda/prefill.go:174`, tried first at `decoder/generate_vl.go:159`) when the prompt fits one pass, else CPU prefill then `UploadKV` (`:210`); decode `cuda-resident (int4)` | tower + serving **verified-run** (B1, P1, T1); the two decoder cosine gates **skipped today** (T4), so their figures (decode-step cosine 0.998167, prefill cosine 0.997042, decode 13.26x, prefill 22.27x) are **recorded 2026-09-08** (`docs/benchmarks.md` "Vision-language resident decode (gap 0)"); tower speed **recorded 2026-09-21** (`docs/measurements/vision-tower-mma-2026-09-21.md`, the owner's default override is `docs/tasks/red-october.md` R8) |
| Metal | **CPU**: nothing on Metal registers a vision tower (`metal/` has no `RegisterResident`; `internal/serveapp/main.go` is webgpu/cuda only) and aikit's `gpu/visionmetal` is not imported (B2). f32 unless the flag | resident-capable: Metal has `UploadKV` and `ForwardMRoPE` (`metal/backend.go:1235,585`) but no `PrefillImageLast`, so CPU prefill then `UploadKV` then Metal decode | **read-from-code**, never run on a Mac in any record I found (`docs/benchmarks.md` "Not yet measured: CUDA is the only backend measured") |
| WebGPU | resident WebGPU tower (`gpu/vision_register.go:11`), int8, `EnableResident` called; recorded 18.8 s/image, ~9x, 2026-06-11 (a pre-re-anchor row, "the absolute 18.8 s is not current", `docs/legacy-benchmarks.md`); today's cold request took 27.4 s end to end (exploratory) | resident (`gpu/residency.go:1429` `UploadKV`; no `PrefillImageLast`): `webgpu:vulkan-resident (int4)` | **verified-run** (P8, T9) with `--ctx 4096`; at the default context an 8 GB card aborts startup (below) |

### Gemma 4, E2B and E4B (`use_bidirectional_attention` unset)

| backend | (1) tower | (2) decoder after the image | verdict |
|---|---|---|---|
| CPU | CPU `Gemma4Encoder`, f32 unless the flag | CPU, one token at a time (`decoder/generate_gemma4_vl.go:29-67`) | recorded 2026-09-10 (Phase E, real E2B, `docs/multimodal.md`); tiny fixtures verified-run (T7) |
| CUDA | CPU, **int8** (the rule above; the encoder has no resident seam) | **CPU, the whole model**: `cuda does not implement [gemma4-e-model]` | **verified-run** (P4: 25.3 s per 336x336 image, no reuse) |
| Metal | CPU, int8 only by flag | CPU: no backend declares `FeatGemma4EModel` (`decoder/features.go:100,268`; a grep of `cuda/ gpu/ metal/` finds no declaration) | read-from-code |
| WebGPU | CPU, int8 | CPU, same reason | read-from-code |

### Gemma 4, 26B-A4B and 31B (`use_bidirectional_attention: "vision"`)

| backend | (1) tower | (2) decoder after the image | verdict |
|---|---|---|---|
| CPU | CPU | CPU bidirectional batched prefill (`runLayersGemma4FromEmbedN`), CPU decode | recorded 2026-09-10 (Phase E, real 26B-A4B on the CPU: "Blue."); tiny fixtures verified-run (T7) |
| CUDA | CPU, int8 | CPU bidirectional prefill, then `residentUploadPrefill` and **resident decode** (`decoder/generate_gemma4_vl.go:161-168`) | 26B-A4B **verified-run** served (P7 run 2: `cuda-resident (int4)`, correct reply, 68.5 s cold; run 1 anomaly noted) and the bridge on a **synthetic scaled** fixture (T5). 31B: no checkpoint on this box, **unverified** |
| Metal | CPU | the same generic bridge; resident Gemma 4 is a matrix claim (`docs/hardware-matrix.md`) | read-from-code; unverified end to end |
| WebGPU | CPU | the same bridge; the 2026-09-17 commit message `26f64807` says Gemma 4 + MoE stays CPU-only on WebGPU, the matrix says resident | **unverified** (the two disagree and I did not settle it) |

### Qwen2.5-VL

| backend | (1) tower | (2) decoder after the image | verdict |
|---|---|---|---|
| CPU | CPU `QwenVisionEncoder` | CPU | recorded; tiny fixtures verified-run (T7) |
| CUDA | **CPU**, int8 by the rule. aikit now has the seam (`vision/qwen_resident.go:38,44`) and the module (`gpu/qwencuda`), but **goinfer registers no Qwen factory and imports neither** (B1; `docs/tasks/task-aikit-boundary-2026-09.md:33`), and serve never calls `EnableResident` on it | resident m-RoPE prefill (`cuda/prefill.go:227`) then `ForwardMRoPE` decode (`cuda/resident.go:2317`), else CPU prefill + `UploadKV`; 3.86x decode recorded 2026-09-08 | **verified-run** (P2, T2, T3: real 3B checkpoint, cosine 0.998644 prefill, 1.000000 decode step, reuse 165) |
| Metal | CPU (`gpu/qwenmetal` exists, not imported, B2) | `ForwardMRoPE` + `UploadKV` exist (`metal/backend.go:601,1127`); no m-RoPE prefill kernel, so CPU prefill then bridge | read-from-code |
| WebGPU | CPU, int8 | `ForwardMRoPE` + `UploadKV` (`gpu/residency.go:1250,1428`), CPU prefill then bridge | **verified-run** (P9: reuse 165; T9 primitives cosine 1.0). At temperature 0 the cold and the reused turn give different text, every time (a CPU-prefill versus GPU-last-token arithmetic gap; item 8 below) |

### Qwen3.5+ (`qwen3_5` dense; `qwen3_5_moe`)

| backend | (1) tower | (2) decoder after the image | verdict |
|---|---|---|---|
| CPU | aikit `Qwen3VisionEncoder` on the CPU, **loaded on the first image**, f32 unless the flag, at most 1024 merged tokens per image (`internal/serveapp/qwen35_vision.go:24,83`) | CPU, one token at a time (the Gated-DeltaNet recurrence has no batched form) | **recorded 2026-09-30**: 0.8B 32/32 tokens identical to HF f32 on three images (`docs/measurements/p8a-qwen35-vl-2026-09/g1-g4-results.md`) and **9B 32/32 on three images** (night run `~/goinfer-logs/night/runs/2026-09-30/p8a-g2-9b.log`, `--- PASS: TestQwen35VLReal_G2_9B (767.58s)`). The old "9B leg open" text is stale |
| CUDA | CPU, **int8** by the rule (the 0.8B/9B gates ran f32; I found no int8 gate for this tower) | **CPU prefill and CPU decode**: every resident branch is refused for a recurrent family (`decoder/generate_vl.go:297-299,364,420`) | **verified-run** (P3: reuse 0, 7.0 s, the tower reruns on a repeat; `TestGenerateQwenVL_recurrentTakesNoResidentBranch` T7) |
| Metal | CPU | CPU (the refusal is in `decoder/`, not backend-specific) | read-from-code |
| WebGPU | CPU, int8 | CPU, same | read-from-code |

**MoE sizes (`qwen3_5_moe`) are unverified.** Serve's auto-discovery accepts the model type (`internal/serveapp/qwen35_vision.go:93-99`, and a `qwen3_5_moe` row in `TestIsQwen35VisionDir`), but
no image has been run through a MoE checkpoint (P8a: "the MoE checkpoint last and separately"; the tiny `qwen3_5_moe` fixture is text-only), the capability matrix says modality
`text` for it, and the Ollama-coverage notes say "dense sizes only". The supported claim stays dense (0.8B and 9B gated).

### GLM-OCR (`glm_ocr`)

| backend | (1) tower | (2) decoder after the image | verdict |
|---|---|---|---|
| CPU | aikit `GlmOcrVisionEncoder`, f32, lazy, **never forced to int8** (`internal/serveapp/main.go`) | CPU `GenerateQwenVL` | **recorded 2026-10-02** (O3: 64/64 tokens identical to transformers f32 on three rendered pages); T7 |
| CUDA | CPU f32 (about 44 s for a 1,656-token page, O5, exploratory) | **resident** (pairwise rope kernels): `ForwardMRoPE` decode, m-RoPE prefill | **verified-run** (P5: `tower (f32, CPU)`, `cuda-resident (int4)`, reuse 163; T6). Real pages and the 6,144-token ceiling are recorded 2026-10-02 (O3), not re-run |
| Metal | CPU f32 | declined, CPU (`pairwise-mrope` is not declared on Metal; `docs/hardware-matrix.md`) | read-from-code |
| WebGPU | CPU f32 | declined; **`webgpu-staged (int4)`** (decode matvecs on the device kernel, prefill on the host, no resident KV) | **verified-run** (P10) |

### The negatives

| family | what happens | verdict |
|---|---|---|
| Qwen3-VL (`qwen3_vl`) | text decoder only: no tower loader matches it (`internal/serveapp/main.go:1129-1148`), `multimodal.LoadProjector` fails on it | read-from-code; no checkpoint on the box |
| Ministral 3 / `mistral3` | the checkpoint's tower is ignored; an image gets HTTP 400 "this model has no vision tower" | **verified-run** (P6) |
| anything else | the capability matrix marks `vision` for exactly `qwen3_5`, `glm_ocr`, `gemma4`, `qwen2_5_vl` in `tasks`, and names a vision tower in the modality text of those plus `gemma3` and `mistral3` (ignored); nothing else | read from `docs/capability-matrix.json` |

All five working families share one route (`internal/serveapp/vision_serve.go`, dispatch at `internal/serveapp/openai.go:1717-1721`), and the Anthropic `image` block goes through
the same `driveVL`; the Anthropic surface was **not run** here.

## Claims in the docs that were wrong

Line numbers into `docs/multimodal.md` below are the file's numbers **before** this rewrite (`git show 9692b547:docs/multimodal.md`).

| # | the old claim | where | what is true |
|---|---|---|---|
| 1 | "images work end-to-end for three families" | `docs/multimodal.md` status (line 6); `docs/completed/audit-2026-09-02.md` and `audit-2026-09-10.md` carry the same count (history, left) | five: Gemma 3, Gemma 4, Qwen2.5-VL, Qwen3.5+ (dense), GLM-OCR |
| 2 | "CUDA resident-serves Gemma 3's tower ... 1.58x, 41.3 s -> 26.1 s" as the current figure | `multimodal.md` status; `docs/server.md` vision paragraph | superseded 2026-09-21: the fused attention is the default, 26.0 s -> 4.1 s/image (6.4x); 1.58x is the 2026-09-08 first version. `GOINFER_CUDA_VISION_ATTN=exact` gives the old 26 s |
| 3 | "the cgo-free release binaries still have no GPU vision for any family ... a downloaded `goinfer-serve` does every image at CPU speed regardless of which family" | `multimodal.md` status and gap 1 (line 157) | **false for CUDA**: the CUDA backend is cgo-free (`CGO_ENABLED=0`, B1) and the Linux release binary carries the resident SigLIP tower (P1: `encoder int8/cuda-resident`, 4.9 s cold image on Gemma 3). True for the macOS and Windows binaries, and for every family but Gemma 3 |
| 4 | "Both CUDA's and WebGPU's resident towers need cgo" | `multimodal.md` status; `docs/ARCHITECTURE.md` is silent | only WebGPU needs cgo (B1, B3). CUDA opens `libcuda` at run time |
| 5 | "Gemma 4's `GenerateGemma4VL` is CPU-only v1 with no resident decode bridge at all" | `multimodal.md` status | the bridge exists since P7 Phase D (2026-09-10) for 26B-A4B and 31B (`decoder/generate_gemma4_vl.go:161-168`); only E2B/E4B are CPU for the whole turn, and that is not a vision gap (no backend implements the E-model features) |
| 6 | "Qwen2.5-VL's and Gemma 4's own towers stay CPU-only (no `EnableResident` path yet)" | `multimodal.md` status | **still true of what ships**, but the reason changed: aikit has had a Qwen seam and `qwencuda`/`qwenmetal` modules since, and goinfer does not import them (`task-aikit-boundary-2026-09.md:33`). Gemma 4 and Qwen3.5+ and GLM-OCR towers have no resident seam in aikit at all |
| 7 | "`--backend webgpu`/`--backend cuda` force the int8 tower regardless of `-vision-quant`" (stated for every family) | `docs/server.md` | true for Gemma 3, Qwen2.5-VL, Qwen3.5+ and Gemma 4; **not** for GLM-OCR (`internal/serveapp/main.go`). For the three that have no resident tower the rule produces a CPU int8 tower, which the comment "needed for the resident GPU matmul weights" does not cover |
| 8 | "Metal has no vision tower for any family yet" | `multimodal.md` status | true of the **tower**. Not true of Metal image turns generally: Metal has had a real `UploadKV` and `ForwardMRoPE` since 2026-09-17 (`26f64807`), so the CPU-prefill-then-resident-decode bridge is available there (read-from-code, unrun) |
| 9 | "Metal is out of scope: `UploadKV` is unimplemented there" and "Metal has no `UploadKV`/resident image-turn decode at all" | `multimodal.md` gap 0 (line 117); `docs/benchmarks.md:838`; `decoder/generate_vl.go` (a code comment, reported, not edited) | true on 2026-09-08 (the stub returned an error); implemented 2026-09-17 |
| 10 | "Vision is one family ... Qwen-VL, the most-pulled VL line, is still text-only here" | `multimodal.md` gap 2 (line 164-173) | Qwen2.5-VL has served images since P5; the line meant Qwen3-VL, which is still text-only |
| 11 | P8a: "Open: the 9B leg (night job `run-g2-9b.sh`), aikit tag + five-module pin bump" | `multimodal.md` P8a status (line 776-781) | the 9B leg ran 2026-09-30 and passed (32/32 on three images); aikit v1.52.0 is tagged and pinned. Still open: resident image decode, the MoE checkpoint, resize fidelity, the think-block decision |
| 12 | "a resident vision tower (CUDA, WebGPU)", "The other towers run on the CPU, and the text decode after them stays resident"; towers listed as SigLIP, Gemma 4, Qwen2.5-VL | `docs/ARCHITECTURE.md` (Multimodal section, "Registration") | the decode after Qwen3.5+, Gemma 4 E2B/E4B and (off CUDA) GLM-OCR is **not** resident; Qwen3.5+ and GLM-OCR towers were missing from the list |
| 13 | "MLA / Mamba / hybrid / vision -> declines; runs staged" | `docs/cuda-backend.md` | MLA and the DeltaNet hybrids are resident (the same page lists them above); a vision checkpoint's text decoder goes resident and an image turn rides the `UploadKV` bridge |
| 14 | "expect a minute or two per image ... a faster int8 tower is the planned follow-on"; `agent-web` is Gemma 3 only and `--vision-backend` offers webgpu only | `demo/agent/README.md` | CPU tower 31.3 s/image (2026-09-08); the int8 tower shipped and is a wash on AVX2; agent images are Gemma 3 only and have no CUDA tower |
| 15 | "vision in (Gemma 3 VL + Qwen2.5-VL ...)" as the feature row | `docs/benchmarks.md:372` (a dated comparison row, 2026-09-08) | five families now; the row's tower numbers are the 2026-09-08 ones (note added, row left as the dated record it is) |
| 16 | "the SigLIP prefill is CPU-heavy ... 31.3 s/image ... correct but slow" as the general state | `docs/server.md` | true on CPU and Metal; on a CUDA box the Gemma 3 tower is 4 s |
| 17 | "Shipped opt-in ... the default is unchanged" | `docs/measurements/vision-tower-mma-2026-09-21.md` "State" | written before the owner's override the same day made the fused kernel the default (`cuda/vision_encoder.go:138-143`, `red-october.md` R8, `docs/env-vars.md`); a dated note was added at the top of that record |

## Contradictions between docs, and which one I aligned to

| contradiction | aligned to |
|---|---|
| `server.md` "`--backend webgpu`/`--backend cuda` force the int8 tower" vs `server.md`'s own O7 note "GLM-OCR is not forced" vs `multimodal.md` | **`internal/serveapp/main.go`** (the code), stated per family in the rewritten paragraph |
| `multimodal.md` "Metal has no vision tower" / "`UploadKV` unimplemented" vs `docs/hardware-matrix.md` (Metal resident for Gemma 3 and Qwen2.5-VL) vs `metal/backend.go` | the code and the matrix: Metal has the decode bridge, no tower |
| `multimodal.md`, `benchmarks.md`, `server.md`: "CUDA tower 26.1 s / 1.58x" vs `docs/env-vars.md`, `docs/releases/v0.20.0.md`, `red-october.md` R8, `docs/benchmarks.md:676` ("superseded") | the later, dated records: 4.1 s default |
| `multimodal.md` "cgo-free release binaries have no GPU vision" vs `README.md` ("CUDA and Metal are cgo-free ... WebGPU is the one cgo build") vs `release-assets.yml` | README and the workflow |
| capability matrix `qwen3_5_moe` modality `text` vs `main.go`/`qwen35_vision.go` auto-discovering it vs the Ollama-coverage note "dense sizes only" | "dense is gated, MoE unverified" (the matrix is generated from `decoder/`; not edited) |
| capability matrix `gemma3` `tasks` lacks `vision` while its modality names it; the site works around it (`site/internal/site/ollama.go:96-100`) | reported, not edited |
| `ARCHITECTURE.md` towers list vs the five families | the five families |
| `docs/hardware-matrix.md` reads "Gemma 4 ... resident" on all three backends; E2B/E4B decline on all three | reported (a generator note would say it; `decoder/` not edited) |

## Could not verify

- **Every Metal cell.** No Mac. B2 proves only what the darwin binary links. `metal/uploadkv_parity_test.go` and `metal/forwardmrope_parity_test.go` are darwin-only and were not run; no record of a Metal image turn exists.
- Gemma 3's two decoder gates on CUDA (T4 skipped): decode-step cosine, prefill cosine, 13.26x and 22.27x stay "recorded 2026-09-08".
- Gemma 4 31B and E4B (no checkpoint); 26B-A4B's decode actually on the GPU beyond the banner (the repeat was refused by the swap guard); Gemma 4 on WebGPU (the commit message and the matrix disagree).
- Qwen3.5+ MoE images; Qwen3-VL (no checkpoint); int8 fidelity of the Qwen2.5-VL, Qwen3.5+ and Gemma 4 towers (no gate found).
- The CPU tower's 31.3 s and the WebGPU tower's 18.8 s were not re-measured (27.4 s cold request on WebGPU today is end to end, one sample).
- The Windows and Linux-arm64 release binaries (read from the workflow only); `--backend auto` on a Mac.
- The Anthropic `/v1/messages` image route, `demo/agent` and `goinfer-chat --image` were not run (read; O5 recorded the chat demo).
- A real GLM-OCR page through the CUDA resident today (O3 recorded it).

## Code and matrix observations (found, not fixed: this was a docs-only pass)

1. **FIXED 2026-10-02:** a failed attach now warns and runs the tower on the CPU (`enableResidentTower`; `TestEnableResidentTower`, and the old and new serve binaries run side by side: exit 1 before, serving with the warning after). **`internal/serveapp/main.go`: a failed `EnableResident` aborts startup.** With `--backend webgpu` and the default context on this 8 GB card the decoder's KV took the VRAM and the tower upload failed with "Not enough memory left"; serve exited (P8, the first attempt). A fall-back to the CPU tower with a warning is a code change and the owner's call. `--ctx 4096 --kv-sessions 1` makes it fit.
2. **FIXED 2026-10-02 (measured first):** int8 is not faster on these CPU towers and sits at relative L2 0.14 to 0.31 from f32, so the three now follow GLM-OCR's rule (`towerInt8`, `TestTowerInt8`; `docs/measurements/vision-tower-int8-fidelity-2026-10-02.md`). **`internal/serveapp/main.go`**: the int8 rule applies to three towers with no resident path (Qwen2.5-VL, Qwen3.5+, Gemma 4). GLM-OCR was excluded for exactly that reason ("its int8 form is not gated on the real checkpoint"); the same reasoning has not been applied to the others, and the P8a gates ran f32.
3. **FIXED 2026-10-02.** **`decoder/generate_vl.go`**: the comment "Metal, which also lacks UploadKV" is stale (finding 9).
4. **FIXED 2026-10-02.** **`internal/serveapp/main.go`**: the usage line says `--backend cpu (default)`; the flag's default is `auto` (`internal/loadflags/loadflags.go:62`).
5. **PARTLY FIXED 2026-10-02:** `gemma3` now lists `vision` (and its summary says the 4B and larger read images), and the hardware matrix has a Gemma 4 E-model footnote; `qwen3_5_moe` stays modality `text` on purpose, because its images are unverified and the supported claim is dense only. **Matrix generator (`decoder/capability_matrix_test.go:560-562`)**: `qwen3_5_moe` modality `text`; `gemma3` `tasks` without `vision`; the hardware matrix's per-checkpoint note covers Nemotron-H but not Gemma 4's E-models, which decline on every backend.
6. **FIXED 2026-10-02:** both pin a 1024-position context and fail, not skip, if the resident declines; they ran for real (`--- PASS`, cosine 0.998271 and 0.996878, argmax-identical) and go red when unpinned. **`cuda/gemma3_resident_real_test.go` and `cuda/gemma3_img_prefill_resident_real_test.go` skip silently on an 8 GB box** (`Options` without a pinned context asks for 16,168 positions, 4.50 GB of KV against 4.57 GB free), so the gates behind the 13.26x and 22.27x claims do not run here as written. A SKIP is not a pass.
7. **P7 run 1: HTTP 200 with `completion_tokens` 0 and an empty reply** on the real 26B-A4B image request (a swap-guard trip happened during it; the guard only refuses new requests, `internal/serveapp/halt.go:73-83`, so it is not the explanation). **One occurrence, not reproduced, cause not established.** Run 2 answered correctly, and three further runs on 2026-10-02 (one with 1 GB of VRAM held back, so the C′ cache was capped at 16 slots instead of 29) all returned HTTP 200, 16 tokens, finish `stop` and the same correct text (66.7 s and 69.1 s for the two timed; `followup/g26-rerun1.txt`, `followup/g26-hog1.txt`). What the code allows: `vlDecodeLoop` ends with no tokens and no error exactly when the first sampled token is a stop token or the budget is 0, and a GPU error would have been a 500, so the likeliest reading is a first-token stop; that was not tested directly. The kernel log has nothing in run 1's window. Run 1's second request also got `Connection refused`, i.e. the server was gone after the empty reply; nothing explains that either. The finish reason was not captured by that version of the probe.
8. **WebGPU Qwen2.5-VL (P9): the cold and the prefix-reused turn differ at temperature 0. Explained: a CPU-versus-GPU arithmetic gap, not a reuse defect.** Reproduced and deterministic (3 cold turns "square, red, yellow, blue"; 4 reused turns "square, red, circle, yellow, blue"; `followup/seq-webgpu.txt`). CUDA gives one text on all nine turns of the same sequence, with bit-identical logprobs cold and reused. Cause: on WebGPU the cold turn is prefilled on the CPU (no `ResidentMRoPEPrefill`) and its first-token logits are CPU arithmetic, while a reused turn recomputes the last prompt token on the GPU over the CPU-computed KV. Evidence, from per-token logprobs read through a temporary diagnostic (`followup/lp-webgpu.log`, `lp-cpu.log`, `lp-cuda.log`): token 0's runner-up is -1.652 on WebGPU-cold, -1.676 CPU-only, -1.254 on WebGPU-reused; the texts part at token 4, a near-tie (cold: yellow -0.669 over circle -0.801; reused: circle -0.707 over yellow -0.779). A pure-GPU path is as far from the CPU as the reused turn is: CUDA's token 0 is -0.839 against the CPU's -1.144. The reuse path's position handling is correct (`mropeDelta` applied at both sites). So on WebGPU a cold and a reused turn are not token-identical; on CUDA they are. Not changed.
9. **Qwen3.5+ re-runs its tower on a repeat** (P3: 6.7 s again, reuse 0): the resident image-reuse path does not apply to a recurrent family, and nothing caches the tower output. Documented, not a defect of the claims.
10. **An image request with `logprobs: true` returned 200 with no `logprobs` field** (found while chasing item 8: `driveVL` discarded them). **Fixed 2026-10-02:** the buffered vision reply now carries them (`choices[0].logprobs`, one entry per completion token, reused turns included), and a streamed image request with `logprobs` is a 400, as on the text route. Gates: `TestHandleChat_imageStreamLogprobsRejected` (red with the check disabled) and `TestGemma4VLReal_E2B_logprobs` (real checkpoint, 107 s, red with `driveVL` returning no logprobs); end to end on the real CUDA server with Qwen2.5-VL (`followup/e2e_lp.py`, `followup/e2e.log`).
