# nobara, 2026-10-07: CUDA image towers (S4 and S2's CUDA twins), plus two quick cross-checks

For a session on `nobara-pc`. The task doc is `docs/tasks/task-multimodal-support-2026-10.md`. Read S2 (its Gate 0, the
gates and the Mac's results), S3 (G-S3a, G-S3c and the owner's decision under them) and S4 (the combined block) first.

The work splits into two quick items, then the main one:
- Item 1 is a cross-check of a Mac failure.
- Item 2 is a quick test that decides S4's Qwen2.5-VL route.
- Item 3 is the main work, on the GPU towers.

All three are daytime work. Item 3's speed record goes on the night queue.

## 0. Setup: an unreleased aikit branch, so work in worktrees

S2's towers need new aikit API: the `vision` exports in aikit commit `64bab88` on the local branch `s2-tower-exports`.
- It is **not pushed and not tagged**. aikit pushes and tags are the owner's, once per cycle.
- It is copied here as a bundle: `~/bundles/aikit-s2-tower-exports.bundle`. The bundle needs `7cce915`, which your aikit
  `origin/main` already has.
- goinfer's side is the branch `s2-towers`, on origin.
- Neither may reach goinfer `main` until aikit is tagged. Docs-only records (gates, results) go to `main` directly.

```sh
cd ~/mycode/goinfer && git fetch && git merge --ff-only origin/main     # leave the uncommitted run-gate-gpu.sh edit as it is
cd ~/mycode/aikit/aikit && git fetch ~/bundles/aikit-s2-tower-exports.bundle s2-tower-exports:s2-tower-exports
git worktree add ~/wt/aikit-s2 s2-tower-exports
cd ~/mycode/goinfer && git worktree add ~/wt/goinfer-s2 origin/s2-towers -b s2-towers
```

Give `~/wt/goinfer-s2` its own `go.work` (it is gitignored):
- `use` these goinfer modules: `.`, `./cuda`, `./gpu`, `./demo/agent`;
- and these aikit modules: `../aikit-s2`, `../aikit-s2/gpu`, `../aikit-s2/gpu/qwencuda` (the last one for item 2).

Build there with `GOWORK=$PWD/go.work`. Your global GOWORK points at the main checkout's.
- A `GOWORK=off` build will fail on missing aikit API. That is expected.
- The Mac's `go.work` for the same branch is the model.

## 1. G-S3c on CUDA, a cross-check (quick: about 15 minutes)

On the Mac, Qwen2.5-VL-3B decoding on Metal disagreed with the CPU at the very first generated token, and not as a
near-tie ("Table" against "Quarter", p 0.208 against 0.728).
- On the Mac, the image turn's prefill runs on the CPU and then `UploadKV`s to Metal.
- CUDA runs its own resident m-RoPE prefill.
- So CUDA tells us whether the defect is Metal's alone.

Gemma 3 4B never reached Metal on the Mac (memory guard). Its two CPU-decoding arms also disagreed beyond a near-tie,
which is a separate oddity that the control arm below speaks to.

It is registered in the task doc (S3, "G-S3c on CUDA"). Run it as written:

```sh
cd ~/wt/goinfer-s2 && GOWORK=$PWD/go.work go build -tags cuda -o ~/goinfer-bench/s3/serve-cuda ./cuda/cmd/serve
git rev-parse --short HEAD > ~/goinfer-bench/s3/serve-cuda.rev
cd ~/mycode/goinfer && bash docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh ~/goinfer-bench/s3/serve-cuda \
  ~/goinfer-logs/s3c-cuda cpu,cuda,cpu ~/models/qwen25vl-3b-instruct ~/models/gemma-3-4b-it
```

- Both checkpoints are already in `~/models`. Never use `/srv/models`.
- Every arm keeps the tower on the CPU (`-vision-device cpu`).
- The script compares the first `cpu` arm against `cuda`, and against the second `cpu` run (the determinism control,
  which must be byte-identical). For each it prints the first differing token and the near-tie read.
- Check that each log's `decode path` line says what the arm asked for: CUDA resident for `cuda`, `cpu` for `cpu`. An
  arm that fell back is not that arm.
- **Record it in the task doc** under the registration, with the raw files copied to
  `docs/measurements/multimodal-support-2026-10/s3-gs3c-cuda/`.
  - If CUDA also says "Table": that points at something shared (the image prompt or the m-RoPE positions as the resident
    paths see them), not at Metal's `UploadKV`.
  - If CUDA matches the CPU: Metal's path is the suspect, and the Mac traces it.

## 2. G-S4q: aikit's `gpu/qwencuda` at real size (quick: about 10 minutes)

On the Mac, aikit's Metal SigLIP tower was wrong at real size (worst token 0.18–0.46) although its tiny parity test
passes. aikit's Metal Qwen2.5-VL tower was correct but no faster than the CPU. The owner has decided that both Metal
towers get rebuilt on goinfer's own base.

S4 says to wire aikit's `gpu/qwencuda` on CUDA. G-S4q, registered in S4, decides whether that still holds.
- Port the Qwen2.5-VL half of the Mac's `metal/s3_towers_test.go` (on `s2-towers`) to `cuda/`. Use the same images,
  bars and grading helper. Copy `gvWorst` from `metal/grid_vision_test.go` or share it.
- It needs a blank import of `github.com/townsendmerino/aikit/gpu/qwencuda` and a `require` for it in `cuda/go.mod`, as
  the Mac's commit `29dece6c` did for Metal.
- Run it under `GOINFER_HEAVY_TESTS=1` with `-v`.
- Record the worst token per image and the exploratory times (CUDA tower against CPU tower) in S4, with the verdict:
  - **correct and faster:** wire `qwencuda`, as S4 says;
  - **wrong, or not faster:** Qwen2.5-VL's CUDA tower is built on the CUDA base in item 3, as Metal's will be.
- If aikit's `gpu/visioncuda` (SigLIP) can be attached the same way, read it at real size too. It is unused here, because
  goinfer's own `cuda/vision_encoder.go` is Gemma 3's CUDA tower, so this is information for the aikit defect report,
  not a gate.

## 3. The CUDA towers: Gemma 4 first (the base), then Qwen3.5+ and GLM-OCR (S4 + S2's CUDA twins; size L)

CUDA has no general tower base. Metal's is `metal/gemma4_vision.go`, its `eg2Ops`: f32 GEMM, RMSNorm, per-head norm,
rotate-half RoPE, and attention built from matmuls in 256-row blocks, never T². S2 built `metal/grid_vision.go` on top of
it for Qwen3.5+ and GLM-OCR, adding:
- an attention scale;
- LayerNorm with bias, biased GEMM epilogues and GELU (tanh and erf) from aikit `gpu.ViT`;
- the host-computed Qwen3 position table;
- per-segment attention for multi-image batches.

On CUDA:
1. **Gate 0, a desk map, recorded in S4 before any CUDA tower code.**
   - What CUDA already has: `cuda/vision_encoder.go` and `attn_fused_vit.cu` (goinfer's SigLIP tower), and aikit's
     `gpu/cuda_vit.go` ops.
   - What carries over from `metal/gemma4_vision.go` and `metal/grid_vision.go`.
   - What is new.
   - Any CUDA-specific planted defects to add to G-S2c's list.
2. **The Gemma 4 tower on CUDA**, registered through `multimodal.RegisterGemma4Tower` (already backend-neutral). Its
   gates are Metal's Gemma 4 tower gates (`metal/gemma4_vision_test.go`), plus a served Gemma 4 image turn, tower on
   CUDA against `-vision-device cpu`, as in G-S2d. That turn runs on the E2B, which CUDA now decodes resident.
3. **Qwen3.5+ and GLM-OCR on CUDA**, through `multimodal.RegisterQwen3Tower` / `RegisterGlmOcrTower`, mirroring
   `metal/grid_vision.go`. Gates: G-S2b, G-S2c and G-S2d unchanged, on CUDA.
   - The tiny GLM tower is `testdata/glm-ocr-vision-tiny` (on the branch).
   - The real ones are `~/models/qwen3.5-0.8b` and `~/models/glm-ocr`, both already here.
   - Randomise the norms through the export's aliasing slices before comparing: the tiny towers' norms are all ones,
     which hides a dropped or swapped norm.
4. **Serve:** `planGridTower` (on the branch) places these towers on Metal or the CPU today. Admit CUDA the same way:
   - `--backend cuda` picks the CUDA tower;
   - CPU fallbacks are named, and `-require-backend` refuses;
   - `-vision-device cpu` keeps any tower on the CPU.
   - Its tests are in `grid_tower_plan_test.go`.
5. If item 2 said build, **Qwen2.5-VL's tower** goes on the same base. Its gates are G-S3a's bars on CUDA, then a
   G-S3b-shaped served turn.

**Branch discipline:**
- Commit on `s2-towers` in `~/wt/goinfer-s2`. Push that branch to origin only on the owner's word, as with `main`.
- `git fetch` and merge before pushing. The Mac is committing Metal work (SigLIP and Qwen2.5-VL on the Metal base) to
  the same branch, mostly under `metal/`.
- If you need a change in aikit (an export the CUDA side lacks), commit it on `~/wt/aikit-s2`'s branch and hand it back
  as a bundle (`git bundle create … 7cce915..s2-tower-exports`). Never push aikit.
- Before each commit: `gofmt -l`, `go vet -tags 'cuda goinfer_testhooks' ./cuda/`, and CI's staticcheck
  (`~/go/bin/staticcheck`, v0.8.0) on `./cuda/...` with the same tags. Prove the staticcheck gate can go red first.

## 4. Night: the CUDA tower speed record

Copy `docs/measurements/multimodal-support-2026-10/run-s2-tower-speed.sh`, the Mac's version, to a CUDA script.
- Use a pre-built test binary from the branch, with its rev recorded; three passes; under the timing lock.
- Cover every CUDA tower that passed its gates, each against the CPU tower, on the same images.
- Queue it with `night.py add` and an estimate.
- Do not start the queue: the owner does. Tell the owner tonight's total.
