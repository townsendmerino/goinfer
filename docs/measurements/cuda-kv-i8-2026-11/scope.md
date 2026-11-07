# CUDA `--kv i8` (P2 of `docs/tasks/task-kv-precision-everywhere-2026-10.md`): scope read and staged plan (nobara-sb01, 2026-11-07)

Francis's decision (via the coordinator, 2026-11-06 21:33): CUDA `--kv i8` is in v0.25.0. The brief: P2 and G3, delivered bytes tested against `ResidentKVBytes`, no silent fallback, red first, the identity gate, a fidelity check against f16 at the doc's registered bar, timed cells by day under the lock with the wall time stated, one PR, no auto-merge. **This note is the read of what that touches, written before any code, and a proposal about "one PR".**

## What the tree has today

- `--kv i8` on CUDA is **refused by name** at load (`docs/kv-precision-matrix.md`: "refused: CUDA has no int8 KV cache until phase 2"), for every family. Nothing is silent; this work turns refusals into deliveries.
- The f16 build is the template: `backend/cuda/kvtype.cuh` types the cache pointer per module (`KV_T`, with `KV_LD`, `KV_LD4V`, `KV_LD2V`, `KV_ST`, `KV_LDH` macros) and each module is compiled twice, `-DKV_F16`, giving `_kv16` PTX twins.
- Size of that surface: **12 `.cu` modules, 61 macro sites** (`decode_fa` 6, `attn_fused` 2, `attn_fused_bm` 2, `decode_splitkv` 2, `prefill_batched` 8, `prefill_splitkv` 5, `attn_img_prefill` 6, `rope_pairwise` 12, `gemv_fwd` 6, `glue` 3, `rope_mrope_prefill` 3, `mla` 6), **15 embedded `_kv16`/`_cap` PTX artifacts**, and 28 Go sites that branch on `kvHalf`/`kvF16`. Paged addressing is `kvpage.cuh` (`KV_ROW`, 64-row blocks, null block 0).
- Plan side: `decoder.cudaKVBytes` (`decoder/residentneed.go`) prices f32/f16 only; `ResidentKVBytes("cuda", ctx, kvF16, kvI8)` ignores `kvI8`. Metal's `metalKVParts` already prices int8 with per-head f32 scales and is the pattern.

## Why the f16 macros do not extend by themselves

int8 is symmetric per (position, KV head) with one f32 scale each for K and V (the contract's one scheme, shared with CPU, Metal and WebGPU). A load of an element needs its scale, so a kernel needs the scale pointers and the (row, head) of every access: the macros cannot be pointer-only. Every KV reader gets two more parameters (K and V scale pools, laid out like the cache, one f32 per row per head), every store quantizes (absmax over the head's `hd` elements, reduced inside the store kernel), and the same scale pools have to follow the blocks wherever the blocks go.

## Everything the scale pools must follow (each is a place a silent fallback or silent wrong answer can hide)

1. KV pool alloc/release/zeroing and the null block (`kvpool.go`); the per-slot tables; the windowed ring and rollback (`kvwindow.go`); shared-layer aliasing.
2. Block copy and COW (`kvcopy.go`, `kvCopyBlockRows`), KV7 snapshot/restore (`kvsnap.go`), demotion to host and restore (`kvdemote.go`), prefix reuse and the share root (`resident_reuse.go`: precision is part of it).
3. Decode attention: `decode_fa`, `decode_splitkv`, `attn_fused`; prefill attention: `prefill_batched`, `prefill_splitkv`, `attn_fused_bm`; the batched step (MC3); speculative verify (`ForwardN`); the drafter's own cache (f32/own, declared); image paths (`attn_img_prefill`, `rope_mrope_prefill`); MLA and pairwise RoPE (`mla`, `rope_pairwise`).
4. Fit and pricing: `cudaKVBytes`, `ResidentKVBytes`, `Plan`, `kvChargedBytes`/`kvChargedSlotsFit`, the MM7/MM8 memory gate rows, the banner and `docs/kv-precision-matrix.md` (generated and ratcheted); the attention-digest census (`testdata/attn_digests.json`).
5. PTX: every new twin is regenerated at NVRTC 12.9.86 from the fixed staging dir (`build_ptx.sh`), and the frozen-module identity tests (`TestKVTwins_f32ModulesUnchanged`) must still hold for the f32 and f16 modules.

## Proposed stages (each reviewable; each removes refusals from the matrix and adds none)

| stage | content | gate | rough size |
|---|---|---|---|
| S1 | `KV_I8` mode in `kvtype.cuh`; `_kv8` twins of the dense decode and prefill attention and the rope/store kernels; Go-free kernel tests | G1 exact-input identity (int8 values x power-of-two scale = the f32 kernel, bit-exact), tier 0 | 1 to 2 days |
| S2 | scale pools beside the K/V pools; quantizing store; `cudaKVBytes`/`ResidentKVBytes` price int8; named refusals for every family/path not yet covered; `--kv i8` delivered on dense (llama, mistral, qwen2, qwen3, phi3) decode + prefill on one slot | G3 delivered bytes read back against `ResidentKVBytes`; MM7 gate rows; matrix regenerated | 1 to 2 days |
| S3 | multi-slot, MC3 batch step, spec verify, prefix reuse/COW, KV7 snapshots, demotion; windowed ring | G1 on the new kernels; the state-grid rule (scales stored beside their rows); tier-0 identity per path | 2 to 4 days |
| S4 | fidelity G4 against f16 at the f16 doc's bars (per cell: hard flips <= ref + 2*sqrt(ref); paired agreement >= ref - 1.0 pt and >= on half the prompts; KL <= 1.10x), cells D7 (7B at K=8000, the decision cell) and S (1.5B at K=3900); speed A B B A at 4k/16k | G4 and speed; by day under the exclusive lock | cards: about 2 h of timed cells; wall time stated before each run |
| S5 | MoE, MLA, pairwise RoPE, VL/image, Gemma 4 families (a "second arm" of the f16 gates) | G1 + G3 per family | per the doc, as f16 was widened |

**About "one PR":** S1 to S3 touch the kernels, the pools, the plan and the matrix together and S4 is a different kind of evidence. The doc's own estimate for this phase is two weeks; I would ship it as the stages above, S1+S2 first (the smallest thing that is honestly "delivered on dense decode and prefill, everything else refused by name"), each a PR on the same no-auto-merge rule, unless Francis wants a single branch that merges only when S4 reads. **That is a decision for the coordinator and Francis; I start S1 now on the staged reading unless told otherwise.**

## What I need decided
1. Staged PRs (S1+S2, then S3, S4, S5) or one branch.
2. Whether the v0.25.0 line is "dense families only, everything else refused by name" (S1 to S4) or needs S5's families too.

## S1a, 2026-11-07: the design proof on `attention` (glue.cu) — G1 green

- **Mechanism** (`backend/cuda/kvtype.cuh`): `-DKV_I8` makes `KV_T` `signed char`; readers go through `KV_LDS(p, sc)` (the element widened and multiplied by its scale; `sc` is ignored, and not evaluated, in the f32 and f16 builds) and each i8-capable kernel takes `KV_SCALE_PARAMS` (the K and V scale pools, empty in the other builds) as trailing parameters. `attention` reads K's scale at `[row * nKV + kvh]` and V's at `[KV_ROW(kvt, s) * nKV + kvh]`: the pools are laid out like the cache, one f32 per (physical row, head), so the block table addresses them as it addresses the rows.
- **Toolchain.** `glue.ptx` is one of the audited 12.6.85 artifacts (`backend/cuda/testdata/REGEN.md`). Control first: rebuilding the unchanged `glue` with the pinned NVRTC 12.6.85 (`~/goinfer-bench/venv-nvrtc12685`) reproduced all four existing glue modules byte for byte; after the edit to `glue.cu` and `kvtype.cuh` they are byte-identical again (`sha256` before and after), and only the new `glue_kv8.ptx` is added, at the same toolchain (`TestKV8_glueTwinIsBuiltAtTheAuditedToolchain`). An ambient 12.9.86 regen would have rewritten all four. `build_ptx.sh` builds `_kv8` only for the modules listed there (glue so far); a module joins the list when every KV access in it goes through `KV_LDS` and the quantizing store.
- **G1** (`backend/cuda/kernels/kv8_identity_test.go`, `s1-g1-run*.log`, `s1-green.log`): int8 values times power-of-two scales (distinct per (position, head), 2^-9 to 2^-5 for K and 2^-8 to 2^-4 for V; an outlier channel at 127) against the same values written as floats through the f32 module, under a **scrambled block table** with the cache and both scale pools laid out under it, at GQA shapes (12/2/64, 8/2/128, 4/4/64), key counts 1, 63, 64, 65 and 300, window 0 and 70: the contexts are bit-identical. The plant (K and V scale pools swapped in the i8 launch) is red.
- **A defect of the test, found and recorded:** my first version copied to the device with `gc.CopyHtoD` and launched on a non-blocking stream; one cell (4/4/64, 300 keys) differed in 1 run of the first 2 and then never again after a `Synchronize`. The package already documents that race (`docs/measurements/cuda-kernels-red-2026-10-30/README.md`) and lints for it (`TestKernels_uploadsSynchronize`): the test now uses `allocBuffer` / `copyHtoD`, as the other kernel tests do. The kernel was right; the f32 arm and the i8 arm both matched a float64 CPU reference to 2e-7 when run alone.
- **Repo rules this met on the way:** the decoder-import ratchet (the test lives in the `kernels` package, which imports no decoder); `TestPTX_matchesSourcesAndBindings` (a `_kv8` twin's declared parameters are the base's plus the two scale pools); `TestKernelFMALint_coversEmbeddedPTX` (`launchgeom.PTXSourceName` knows `_kv8`).
- `touched.py` 16 checks and `cmd/lint` 7/7 pass; `go run ./cmd/gate ptx` GREEN, 59/59 at the NVRTC each records.

**Not done in S1:** the remaining i8 readers and stores (`decode_fa`, `decode_splitkv`, `attn_fused`, `attn_fused_bm`, `prefill_batched`, `prefill_splitkv`, the store in `gemv_fwd`/`glue` rope_kv, the image and pairwise-RoPE kernels, MLA stays f32), and everything in S2 to S5.
