# Environment variable registry

**Convention: every goinfer environment variable is prefixed `GOINFER_`.** (The stray `GINFER_`
prefix — 10 test/bench knobs — was consolidated into `GOINFER_` on 2026-08-11; `GINFER_SPEC_TARGET`
and `GINFER_TEST_MODEL` merged into their pre-existing `GOINFER_` twins, which the decoder and gpu
test suites had been setting inconsistently.)

This registry curates the **operator-facing** knobs — the ones you might set when running or tuning a
deployment. The full set (≈130, most of them per-family test-model overrides and diagnostic probes)
is grep-derivable and enumerated at the bottom.

## Serving

| Var | Purpose |
|---|---|
| `GOINFER_API_KEY` | Bearer token the OpenAI-compatible server requires on requests (unset = open). |

(Most serve configuration is CLI flags on `cmd/serve`, not env vars — see the README Build section.)

## Residency & MoE (device-memory tuning)

| Var | Purpose |
|---|---|
| `GOINFER_P13_OFF` | Keep the safetensors SOURCE mapping resident for the model's life, as the loader did before P13. The loader now closes it at end of load when no tensor dtype can alias it (BF16/F16 widen on read; anything else may be a zero-copy view). Set this only to diagnose a suspected use-after-free, or to reproduce the old memory profile — it is the control arm the P13 measurement used. |
| `GOINFER_NO_RESIDENCY` | Force the staged (non-resident) GPU path — disables whole-model device residency. |
| `GOINFER_MOE_CACHE_SLOTS` / `GOINFER_MOE_CACHE_EXPERTS` | Size the resident MoE expert-slot cache (VRAM ↔ per-token DMAs trade). |
| `GOINFER_MOE_NOCACHE` | Disable the MoE expert cache (always stage experts per token). |
| `GOINFER_MOE_WILLNEED` / `GOINFER_MOE_PREAD` | MoE expert-paging readahead strategy (madvise WILLNEED / pread). |
| `GOINFER_METAL_MOE_SLOTS` | Metal resident MoE slot count. |
| `GOINFER_METAL_ALIAS` | S6 (`docs/tasks/task-never-swap-2026-09.md`) — `=1` opts a Metal load of a **`.giw`** into aliasing its dense int4 nibbles out of the file mapping (a no-copy MTLBuffer over each tensor's, or each fused q‖k‖v / gate‖up group's, page window) instead of copying them into a second MTLBuffer. Unset/`0` is the shipped copy path, unchanged. **Experimental, opt-in, not yet gated**: logits are byte-identical to the copied path on the checkpoints tested (`metal/alias_test.go`), but the registered rule's depth-128/2048 tok/s bands and memory-hog arm have not been run. Only a v13 `-target metal` sidecar has adjacent fused groups (`docs/giw-bundles.md`); older files alias their unfused tensors and copy the rest. Metal wires the pages a decode touches, so the aliased pages are file-backed but not evictable. Prints one `[metal] weights aliased …` line at load. |
| `GOINFER_SPLITKV_MIN_KEYS` | Override the split-KV decode-attention key-count threshold (per-geometry default otherwise). |

## Decode-path escape hatches (default is the fast/bit-exact path; these opt out for A/B or safety)

| Var | Purpose |
|---|---|
| `GOINFER_NO_GREEDY_FASTPATH` | Disable the on-device greedy argmax fast path (force full-logits readback). |
| `GOINFER_NO_KVONLY_PREFILL` | Disable KV-only prefill (run the full-logits prefill on every prompt token). |
| `GOINFER_BATCHED_PREFILL` | Toggle the batched prefill path (`=0` disables). |
| `GOINFER_PREFILL_CHUNK` | Rows per batched CUDA prefill pass (default 512). A long prompt is ingested in several weight-stationary passes over the positional KV — bit-identical to one pass, bounded scratch. Raise it only if the card has room; the path halves the width itself on an OOM. |
| `GOINFER_PREFILL_IMAGE_CHUNK` | Rows for the resident image-block prefill call specifically (default 2048, separate from `GOINFER_PREFILL_CHUNK` above). A bidirectional image block cannot span more than one pass, so this prices the WHOLE turn (image + surrounding chat text) in one weight-stationary pass rather than the general prompt-chunking width. |
| `GOINFER_METAL_BATCHED_PREFILL` | **Deprecated** — superseded by `GOINFER_METAL_FAST_PREFILL`. Still honoured for backward compat: `=1` opt-in (now the default), `=0` opt-out. |
| `GOINFER_METAL_FAST_PREFILL` | Toggle Metal's f16-MMA batched prefill. Default ON above 64 tokens since the §3.2 gate passed at that depth (R3, 2026-09-20; `docs/measurements/metal-prefill-floor-2026-09-20.md`). `=0`/`false`/`off` forces the sequential path everywhere; `=1`/`true`/`on` forces it on (including below the floor). Server flag: `--exact-prefill` sets this to `0`. |
| `GOINFER_METAL_FAST_PREFILL_FLOOR` | Prompt-length floor (tokens) below which the fast Metal prefill declines even when enabled (default **64**, lowered from 256 by R3, 2026-09-20 — `docs/measurements/metal-prefill-floor-2026-09-20.md`). `=0` disables the floor entirely. |
| `GOINFER_METAL_FUSED_ATTENTION` | Toggle `attention_prefill_fused` (the simdgroup_matrix flash-attention twin of `attention_prefill`, L2-Metal, `docs/completed/task-prefill-gap.md` §4), used inside the batched path above. Default ON since §3 gate passed (2026-09-10, `docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md` §5). `=0`/`false`/`off` falls back to the exact scalar kernel; `=1`/`true`/`on` forces it on. Requires hd%8==0 && hd<=128 (`ATTN_MAXHD`) regardless of this flag. Server flag: `--exact-prefill` covers it transitively (disables the whole batched path, so this kernel never dispatches). |
| `GOINFER_METAL_DECODE_LANE` | R1 (`docs/tasks/red-october.md`) — `=w4f16` opts a dense (non-MoE, non-paged, non-sandwich/postOnly/parallelBlock/qGate/outBias/layerNorm, no compute-time LoRA) layer's QKV/o-proj/gate-up decode GEMVs into the f16-activation lane instead of the shipped W4A8 (int8-activation) path; unset is the default and unaffected. **Experimental, opt-in, not yet gated**: the 2026-09-19 "catastrophic divergence at layer 26" that first parked this lane was an instrument error — the shipped W4A8 path was used as ground truth at the attention-sink position, where its per-tensor int8 activation scale is the coarse arm; against an f64 reference the f16 lane is exact and against a CPU reference it is at least as faithful as W4A8 (see [`r1-layer26-rootcause-2026-09-20.md`](measurements/r1-layer26-rootcause-2026-09-20.md), superseding [`w4f16-decode-investigation-2026-09-19.md`](measurements/w4f16-decode-investigation-2026-09-19.md)). It stays off by default because R1's real gates — the pooled teacher-forced fidelity gate and the served-tok/s band — have not been run on it; `metal/r1_gu_reference_test.go` and `metal/r1_lane_vs_cpu_test.go` are its keeper tests. |
| `GOINFER_METAL_ATTN_FA` | R2 (`docs/tasks/red-october.md`) — decode attention on Metal, `attention_fa`/`attention_fa_combine` (a kernel gridded by kvHead×split, cooperatively coalescing a GQA group's K/V reads) for dense-GQA, hd=128, window/sink-free layers past `attnFADepthFloor` (1536 keys); the shipped `attention` kernel runs below the floor or when this is off. **DEFAULT ON since 2026-09-21.** The kernel is correctness-proven in isolation and against a real CPU f32 reference (gate (3) PASSES — [`r2-attn-fa-rootcause-2026-09-21.md`](measurements/r2-attn-fa-rootcause-2026-09-21.md)); a 2026-09-19/20 "divergence at the 3rd decode token" that parked the Build attempt was an end-to-end-logits instrument crossing one int8 rounding boundary, not a kernel defect — see that record. It is a real, deterministic 1.11–1.19× at depth ([`r2-attn-fa-speed-2026-09-21.md`](measurements/r2-attn-fa-speed-2026-09-21.md)) — **below the brief's own peer-parity band** (needed ≥60 tok/s at depth 4000, measured 44.9), **shipped anyway by owner decision** as an incremental win despite missing that bar. NOT bit-identical to the shipped kernel (reduction/combine order differs by design — moves argmax at the margin on some inputs; `TestMetalSnapshotGolden` now covers this kernel via `llama-attnfa-tiny` — closed 2026-09-21, checkpoints straddling the 1536 floor exactly); also a second source of decode/`ForwardN` divergence on top of the pre-existing one `docs/spec/08-dspark-dflash.md` already names, on a Metal spec-decode verify path that was already not a legal oracle for that other reason (P10 — "not a build target" independent of this). `0`/`false`/`off` opts out to the shipped kernel unconditionally; `1`/`true`/`on` is explicit and harmless (matches the default). |
| `GOINFER_INT4_SLOWPATH` / `GOINFER_INT4_F16_SCALES` | int4 unpack path selectors. |
| `GOINFER_CUDA_NO_FUSE` | Disable CUDA kernel fusion (debug/A-B). |
| `GOINFER_MLA_NAIVE` | Use the naive (un-optimized) MLA attention path. |

## Prefill/MoE defaults (2026-08-31 → 09-01) — the four default-ON changes and their opt-outs

**These four turned ON by operator decision and every one of them can change greedy output.** The
only contract naming them used to be a CHANGELOG bullet (N-42). Each is a developer A/B handle
rather than a user setting, and each is listed here because an operator who sees an output change
after upgrading needs one place to look.

| Var | Default | Opt out with | What it changes |
|---|---|---|---|
| `GOINFER_CPU_FAST_ATTENTION` | ON | `=0` | f32 prefill attention above a 512-token suffix. NOT split-invariant: a warm session's long suffix can differ in the last ulps from a cold prefill, which at temperature 0 can flip a near-tie. `serve` sets this from `--cpu-exact-prefill`. |
| `GOINFER_FUSED_ATTENTION` | ON | `=0` | FlashAttention-style fused prefill schedule (P19). Re-associates the softmax denominator, so it is not bit-identical to the materialized path. |
| `GOINFER_MOE_EXPERT_MAJOR` | ON | `=0` | Expert-major MoE MLP traversal (P18). |
| `GOINFER_ATTN_GROUPED` | ON | `=0` | R13 decode attention: one `MatmulQKAcc64Group`/`MatmulAVAcc64Group` kernel call per kv head (aikit's G=6 NEON port, `linalg` commit `af926e3`) instead of one `MatmulQKAcc64`/`MatmulAVAcc64` call per query head, sharing the K/V load across a kv group. Applies only when the model's GQA ratio is exactly 6 and the attended span is at least 128 keys; every other shape runs the unmodified per-head path regardless of this var. BIT-IDENTICAL by construction (`TestAttendGroupedHeads_matchesPerHead`, `decoder/attn_grouped_test.go`, compares on/off byte for byte) — softmax and every other reduction stay untouched, only the QKᵀ/scores·V loads are shared. Measured on qwen2.5-coder-1.5b: parity at depth 2048 (~1% overhead, noise-level), **1.32× served speedup at depth 8192** (`docs/measurements/r13-served-decode-2026-09-20.md`) — the isolated kernel win (1.53-2.39×, `r13-neon-kernel-ab-2026-09-20.md`) only fully shows up once softmax runs as parallel as it already does on the ungrouped path (an earlier wiring bug serialized it; fixed). |
| `GOINFER_CPU_FUSED_GATEUP` | ON (non-arm64) / OFF (arm64) | `=0` (`=1` forces on) | Fused gate+up+SwiGLU decode fork/join for W4A8 SiLU-gated MLPs (`decoder/cpu_gateup_fused.go`): every worker computes gate AND up for its own column chunk and applies the activation to that chunk, so a layer pays ONE barrier instead of two and the SwiGLU's scalar float64 exp runs in parallel inside it instead of serially after. Declines (runs the normal path) for a non-SiLU activation, non-int4 weights, LoRA, a repacked layout (arm64 row4, `GOINFER_W4A8_SPLITHALF`), or a scratch shorter than N. BIT-IDENTICAL: `TestGatedMLPFusedGateUp_bitIdentical` (mutation-checked) plus, on real models, 48 decode steps × ~152k logits × 0.5B/1.5B/7B compared exactly with 0 differing. Paired ABBA on the Ryzen 7 3700X: 1.5B 1.066×, 0.5B 1.113×, 7B 1.029× (`docs/measurements/cpu-decode-roofline-2026-09-23.md`). arm64 stays off — measured on amd64 only. |
| `GOINFER_CUDA_FAST_PREFILL` | **ON above 512 prompt tokens** | `=0` (exact everywhere); `=attn` / `=gemm` select one lever | CUDA fast prompt prefill: `attn_fused` (FlashAttention-style, tensor-core) for the attention term and `gemm_w4a8_mma` (tensor-core int4 GEMM) for the weight term. **3.91× end-to-end prefill at K=3900** on a 1.5B int4, and the deficit vs Ollama at depth goes 12.1× → 3.16× behind (`measurements/prefill-l2l3-phase{2,4}-*`). Not bit-identical: L2 uses f16 K/V with an online-rescaled softmax, L3 re-associates the cross-group float sum. **The 512-token floor is not a guess** — §3's fidelity gate passes at 512/1024/3900 on both bench models and FAILS at 256, so short prompts stay exact (`measurements/prefill-l2l3-phase3-2026-09-05.md`). At depth the fast path is measurably CLOSER to an f32 reference than the exact path. `=0` restores `attn_batched` + `gemv_w4a8_rn`, which spec-decode verify and the parity gates always use regardless of this knob. Server flag: `--exact-prefill` sets this to `0`. |
| `GOINFER_NO_OPTFWD` | (unset) | `=1` | Disables the optimistic-forward speculation gate. Same convention as `GOINFER_NO_GREEDY_FASTPATH`. `GOINFER_OPTFWD_MAX_TEMP` bounds the temperature at which it engages. |
| `GOINFER_NO_TOPK_FASTPATH` | (unset) | `=1` | Disables the CUDA device top-K sampling fast path (`top_k` / `top_p` / `min_p` at temperature > 0): the sampler reads the whole logits row instead of the K best. Same convention as `GOINFER_NO_GREEDY_FASTPATH`; for A/B checks and as an escape hatch. Token streams are identical either way at a fixed seed (except, in principle, at a rounding-boundary `top_p` draw). |
| `GOINFER_NO_SAMPLE_FASTPATH` | (unset) | `=1` | Disables the device temperature-only sampling fast path (`temperature` > 0 with no `top_k` / `top_p` / `min_p`): the sampler reads the whole logits row and draws on the host. The draw is the same Gumbel-max either way, so the token stream is identical except where two candidates' scores are within an f32 rounding (measured ~1e-6 per token); it is an A/B switch and escape hatch, like `GOINFER_NO_GREEDY_FASTPATH`. |
| `GOINFER_CUDA_FLASH_DECODE` | (unset = off) | `=S` (integer >= 1) | **Opt-in, experimental, NOT bit-identical.** CUDA decode attention runs a flash-decode lane (`cuda/decode_fa.cu`: one CTA per kv-head x S key-splits, online softmax, fixed-order combine) instead of the exact split-KV path, for layers with hd 64/128/256, GQA group <= 8 and no attention sink, once the attended span reaches `GOINFER_CUDA_FLASH_DECODE_MIN_KEYS`. Not fidelity-cleared: gated by `docs/measurements/attn-decode-fa-fidelity-PREREGISTERED.md`. Speculative decoding (`--spec ngram`, `--drafter`, the two-model path) still works with it set: each speculative generation holds an exact-attention scope, so decode and verify both run the exact tree and stay token-identical to plain greedy on that tree; the lane serves non-speculative requests. |
| `GOINFER_CUDA_FLASH_DECODE_MIN_KEYS` | `2048` | `=<n>` (>= 0) | Attended-span floor for the flash-decode lane above; shorter spans keep the exact path. Only read when `GOINFER_CUDA_FLASH_DECODE` is set. |
| `GOINFER_CUDA_FLASH_DECODE_VERIFY` | (unset = on) | `=0` | Only read when `GOINFER_CUDA_FLASH_DECODE` is set. Speculative verify rows run on a multi-row version of the flash-decode lane whose output is bit-identical to the single-row lane at each row's position, so speculative output equals plain lane greedy and speculation gets the lane's speed. `=0` disables that and falls back to running speculative generations on the exact attention tree (lane bypassed for their decode and verify). |

## Operator knobs not in the sections above

| Var | Purpose |
|---|---|
| `GOINFER_NO_RESIDENT_MEM_GUARD` | Bypass Metal's resident memory guard — the only remedy the decline message prints. |
| `GOINFER_NO_FIT_GUARD` | Bypass the **load-time fit guard** (`decoder/fitguard.go`): the CPU/GGUF sibling of the line above, which refuses a model whose estimated resident weights exceed 70% of physical RAM instead of letting it page to swap. Set it when the threshold is wrong about your machine. Also bypasses `guardGIWFit` (S4 item 1, `task-never-swap-2026-09.md`), the same idea for a `.giw` load's KV + prefill scratch — a `.giw`'s weights are file-backed and never priced by this guard either way; only the KV/scratch term is what this variable would be bypassing there. |
| `GOINFER_SWAP_GUARD` | S3 (`docs/tasks/task-never-swap-2026-09.md`, `internal/serveapp/swapguard.go`): both halves of the swap tripwire — the SERVING side (armed once, for the process's life, after startup) refuses NEW requests with a 503 (`{"error":"halted","reason":"swap guard tripped: ..."}`, the same shape K2's halt uses) WITHOUT cancelling anything already running, and clears automatically once swap-used has stayed within threshold of the baseline for 30s continuously; the LOAD-TIME side (armed fresh around each `.gguf` direct-build resident load) aborts that load instead, naming the swap delta and the fit guard's own priced terms. Both default to a 512 MB threshold over their own baseline swap-used reading, polled every 2s. Set to a positive integer to change the threshold in MB (e.g. `GOINFER_SWAP_GUARD=1024` for 1 GB); set to `off` to disable both. Keys on **swap-used**, never RSS — darwin RSS reports what survived reclaim under memory pressure, not what was actually asked for (measured on the R11(c) Metal-pager runs: RSS read "7 MB → 892 MB" while swap genuinely grew 2.3 → 12 GB in the same build), so an RSS-keyed version of this guard would have judged that run recovering while it was still losing memory — see `decoder/swapwatch_test.go`'s `TestSwapWatch_keyingOnRSSWouldHaveMissedR11c`. The real gpt-oss-20b positive control (`docs/measurements/swap-tripwire-2026-09-22.md`) found a genuine LIMIT: the load-time trip fires on time, but a fast burst of in-flight worker allocation can outrun the poll interval and overshoot the +1 GB bound before the load actually stops — an external backstop is still advisable for a deliberately-bypassed (`GOINFER_NO_FIT_GUARD=1`) large load. |
| `GOINFER_GGUF_DIRECT` | S1 (`docs/tasks/task-never-swap-2026-09.md`, `internal/prequant.DefaultToSidecar`): on darwin, a plain `.gguf` given to `goinfer-serve` or `goinfer-chat` resolves to its sidecar `.giw` by default (file-backed resident weights instead of a fresh heap copy — `docs/measurements/sidecar-default-2026-09-22.md` measured the sidecar's anonymous footprint at 16–21% of the direct load's, with zero swap growth and byte-identical output, on two real models). Set to any non-empty value (or pass `-direct-load`) to opt back out to the pre-S1 direct-heap-dequant behavior, which is still the default on every other platform. Ignored with `-stream-weights`, which always needs the sidecar's mmap regardless of platform. `--embed-int4` implies this for the same load (the int8 embed/head pin has no sidecar-cache representation yet). `fit` is unaffected by this var: it never builds a sidecar itself, only reuses one that is already fresh (`prequant.SidecarPathIfFresh`), so there is nothing for -direct-load to opt out of there. |
| `GOINFER_GIW_VERIFY` | `always` forces the `.giw` whole-payload CRC-32 on EVERY load. Default (unset): the CRC runs once per file — after a load passes it, `<file>.giw.verified` records the file's size and mtime, and later loads of an unchanged file skip the pass (`decoder/giwverify.go`). Why: the CRC reads every byte of the mapping, which is the entire load time and page-cache footprint of a large streamed `.giw` — measured 27–28 min for a 22 GB file over a ~10 MB/s link (`docs/measurements/moe-pager-m35-smb-2026-09-23.md`), and ~100% of `LoadSerializedWeights` on a 5.17 GB local file. A missing, malformed, mismatched or unwritable marker just means the CRC runs, as it always did. Truncation is caught regardless (the header records the lengths); what the marker trades away is re-detecting silent bit-rot inside an unchanged-size, unchanged-mtime file — set this to `always` for a suspect disk or a file restored by a tool that preserves mtimes. |
| `GOINFER_NO_FIT_DEFAULT` | Precursor to `--fit=false` (`decoder.Options.DisableFit`, `decoder/model.go`'s `Model.FitDisabled()`): restores ONE of the two "fit by default" behaviors — CUDA's unpinned resident context — to its pre-Phase-2 exact default. N-98 (docs/audit-2026-09-10.md): it does NOT also restore the CPU auto-retry into weight streaming (`internal/serveapp/main.go`'s `!opts.DisableFit` check ahead of that retry) — that check reads `opts.DisableFit` directly, which only `--fit=off` sets; the env var alone reaches `Model.FitDisabled()`'s CUDA-side check but never reaches this earlier, flag-only check. On `goinfer-serve`, use `--fit=off` for the full pre-Phase-2 default on both paths; this var is kept working only for anything driving `decoder.Load` directly without going through serve's own `--fit` flag at all. |
| `GOINFER_SSM_RESIDENT` | Opt Granite/Nemotron SSM layers into the resident path. |
| `GOINFER_SPLITKV_ATTN` | CUDA split-KV attention rollback knob (sibling of `GOINFER_SPLITKV_MIN_KEYS`). |
| `GOINFER_CUDA_FLASH_DECODE` | **ON (S=16)** | `=0` / `off` / `false`; a positive integer picks S | CUDA flash-decode attention lane (R6, `cuda/decode_fa.cu`): one CTA per kv-head × S key splits with online softmax and a fixed-order combine, replacing decode attention's three-launch exact path. **Default ON since 2026-09-23 (owner decision); NOT bit-identical to the exact path.** Acts only once a layer's attended span reaches `GOINFER_CUDA_FLASH_DECODE_MIN_KEYS` (default 2048), so shallow decode is untouched; head dim outside 64/128/256, GQA > 8 and attention-sink models stay exact. A non-numeric value is OFF, never a different S. The default does NOT reach a C′ expert-cache model (VRAM is sized to the byte there); set the variable explicitly to force it on. Served 1.5B@3900 124.7→194.1 tok/s, 7B@8000 39.6→61.1; fidelity gate passed on held-out prompts (`docs/measurements/attn-decode-fa-fidelity-2026-09-20.md`). |
| `GOINFER_CUDA_FLASH_DECODE_MIN_KEYS` | 2048 | any integer ≥ 0 | Attended-span floor for the flash-decode lane. 2048 is the lowest floor with no measured regression (0.5B and gemma3-1b lose 3-9% up to 1024 keys). `0` forces the lane on at every depth (the gate/ladder tests' arm). |
| `GOINFER_CUDA_FLASH_DECODE_VERIFY` | ON | `=0` | Speculative-verify rows run on the lane's multi-row kernel (bit-identical to the single-row lane, so speculation equals plain lane greedy). `=0` runs speculation on the exact-attention scope instead. |
| `GOINFER_CUDA_FAST_PREFILL_FLOOR` | Move the CUDA fast-prefill prompt-length floor (default **512**). `0` disables the floor entirely, which is how the §3 gate measures cells the floor excludes. **Lowering it is not free**: 512 is the shallowest depth with a PASSING fidelity cell, and the gate FAILS at 256 — moving it down needs a passing cell at the new depth, not a smooth-looking curve. |
| `GOINFER_MODELS` / `GOINFER_MODEL_TMP` | Model asset root / temp dir for downloads. |
| `GOINFER_PREFILL_ATTN_WORKERS` / `GOINFER_ATTN_ROW_TILE` | Prefill attention fan-out, row tile. |
| `GOINFER_WEBGPU_GEMM` | WebGPU **prefill** W8A8 GEMM kernel selector. Default (unset): the R10 register-blocked 64×64 kernel (`gpu/gemm_rb.go`, each thread a 4×4 block of outputs). `tiled16` restores the pre-R10 16×16 kernel — bit-identical output either way (exact i32 K-sum, same dequant expression; `TestTiledRB64_bitIdentical`), so this is a speed A/B knob, not a fidelity one. Band, profile and A/B: `docs/measurements/webgpu-prefill-profile-2026-09-22.md`. |
| `GOINFER_ATTN_KEYS` | WebGPU **decode** key-split attention kernel kill switch — set to `0` to force the old dim-split kernel instead (A/B in the same binary). Not a prefill knob and not a key count (V-23, docs/completed/review-2026-09-04.md). |
| `GOINFER_W4A8_SPLITHALF` | Select the split-half W4A8 kernel. |
| `GOINFER_W4A8_BATCH` | Opt into the fused q/k/v and gate/up batched W4A8 matmul (audit R-06). Default off — measured 1.08x decode on this box, ambiguous against the 1.05x park / 1.15x ship bar, so it ships parked rather than as the default. |
| `GOINFER_MOE_RESIDENCY` / `GOINFER_MOE_RESIDENCY_SCOPE` | Metal MoE residency mode and scope. |
| `GOINFER_PRECISE_MATH` | Metal: precise (non-fast) math in generated shaders. |
| `GOINFER_NO_RESIDENT_REUSE` | Disable resident-forward reuse across requests. |
| `GOINFER_NO_LORA_CACHE` | CUDA: re-upload a LoRA adapter on every bind and free it on every clear (the pre-cache behaviour; audit P-10 A/B switch). |
| `GOINFER_NVRTC_DIRS` | Extra directories to search for NVRTC when building CUDA kernels. |
| `GOINFER_CUDA_GRAPHS_SYNC` | Force a synchronize around CUDA graph launches. |

## Diagnostics and experiment knobs — NOT contract

Listed so the registry is complete and `TestEnvVars_docAndCodeAgree` has somewhere to put things
that are not operator-facing. These may change or disappear without notice:

`GOINFER_A10_PROBE`, `GOINFER_DELTANET_TIMING`, `GOINFER_ROUTER_CAPTURE`,
`GOINFER_MOE_PREFILL_SCRATCH`, `GOINFER_MOE_CACHE_PROF`,
`GOINFER_CUDA_L01_CPU_OFFLOAD` (L-01 prototype: hybrid CPU/GPU MoE expert offload,
docs/tasks/task-l01-hybrid-moe-cpu-gpu.md — synchronous only, no overlap yet, default off),
`GOINFER_FAKEQUANT_ACT`, `GOINFER_FAKEQUANT_EXPERTS`, `GOINFER_FAKEQUANT_PERROW`,
`GOINFER_SSM_W8A16`, `GOINFER_SSM_F16MAMBA`, `GOINFER_SSM_NOMUL`, `GOINFER_SSM_Q8CPU`,
`GOINFER_SSM_SKIPFFN`, `GOINFER_SSM_STOP_LAYER`, `GOINFER_CUDA_L01_CPU_OFFLOAD`,
`GOINFER_GEMMA4_RESIDENT` (M-56, audit-2026-09-10.md: a Gemma-4 bring-up gate that is now a
no-op — `decoder/gemma4_admission_test.go` pins that admission is unconditional regardless of
its value; kept only so tests can still force both branches while the code path exists),
`GOINFER_MOE_PREAD_CPU` (an unpromoted spike: switches the CPU decoder's MoE expert pager from
mmap+madvise to an owned-buffer pread pool — Lever 1b, docs/completed/task-moe-streaming.md. Bit-exact
either way (`TestExpertBufferPool_refillIsByteExact`), but performance is not yet established on
real hardware at scale; default off),
`GOINFER_CUDA_ATTN_FUSED_TILE` (CUDA: overrides the attn_fused kernel's tile-size selection —
`64x64`, `128x64`, `32x64`, or `32x32`; an R5-phase investigation knob, default unchanged, the
32-row tiles are a measured regression kept only for A/B comparison),
`GOINFER_CUDA_VISION_ATTN` (CUDA vision tower: `bm128` is the DEFAULT since 2026-09-21 — the R8 fused non-causal attention kernel for the SigLIP tower, 6.4× faster (26.0 → 4.1 s/image). Its output differs from the pre-2026-09-21 `attn_img_batched` path at tower level (cosine 0.96 vs the old resident output; cosine vs the CPU int8 reference unchanged, ~0.91-0.93 either way) and a pre-registered served downstream check found it perturbs greedy generation somewhat MORE than a 1-LSB pixel jitter control does (no defect found in either check) — the owner chose the speedup anyway, overriding both pre-registered rules; see `docs/measurements/vision-tower-mma-2026-09-21.md` and `vision-tower-downstream-2026-09-21.md`. `exact` restores the old kernel; `bm64` selects the other fused arm),
`GOINFER_MOE_PIN_REGISTER` (CUDA C′: DEFAULT ON. Stages the expert-stack DMA source by pinning the
already-populated host bytes in place (aikit `Device.RegisterMappedHostBuffer`, gpu/v0.33.3) instead
of allocating pinned memory first and copying into it — cuMemHostRegister on already-resident pages
needs no page-cache reclaim, unlike cuMemAllocHost's fresh pinned allocation. Measured
1.33x-4.46x in an isolated microbenchmark and 1.105x (10.5%) on a real 26B load, fresh process per
sample, 5/5 trials won — below the pre-registered 15% ship bar (park 5-15%), so this is an OWNER
OVERRIDE default-on despite the park verdict, not a promoted measurement; the DMA source's bytes
are identical either way (no numerics risk). `0` restores the allocate-then-copy order. See
`docs/measurements/lead3-pin-order-2026-09-22.md`),
`GOINFER_MOE_DMA_OVERLAP` (CUDA C′ decode: DEFAULT ON when the expert-slot cache is on; `0` restores
the draining path. The routing readback waits on an event recorded after the router instead of
draining the stream; cache misses are DMA'd on a second stream while the kernels issued after the
router (Gemma-4's dense branch, then the hit-expert ranks of segC) execute; each MoE rank waits
device-side only if it missed. Launch order and arithmetic are unchanged, so it is bit-identical to
the draining path. Off under `GOINFER_CUDA_GRAPHS` and `GOINFER_CUDA_L01_CPU_OFFLOAD`. Ceiling, decision
rule and A/B: `docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md`),
`GOINFER_CUDA_MOE_EXPERT_MAJOR` (CUDA: DEFAULT ON since 2026-09-21 — expert-major restructuring of
the batched prefill MoE FFN, both the generic path (`cuda/moe_expert_major.go`) and Gemma-4's own
parallel dense‖MoE FFN (`cuda/moe_expert_major_gemma4.go`). Routes every row of a chunk first, then
admits/DMAs each DISTINCT expert once instead of once per (row, routing rank). Bit-identical by
construction (`TestMoEExpertMajorCUDA_bitIdentical`, `TestMoEExpertMajorGemma4CUDA_bitIdentical`,
mutation-checked). Measured on the real 26B (Gemma-4, the tight-VRAM case this was built for): 2.66x
/ 2.50x / 2.39x / 2.26x at M=512/2048/4096/8012, sequential control unmoved (`docs/measurements/
p20-expert-major-m26-2026-09-21.md`); the generic path's own measured win (Mellum2, a looser-VRAM
model) is smaller, 3.5-4.3% (`p20-expert-major-2026-09-21.md`). Declines to the per-row path on a
shared expert or a gpt-oss per-expert bias table (the generic path only — gemma4 has neither).
`GOINFER_CUDA_MOE_EXPERT_MAJOR=0` restores the per-row path, mirroring the CPU precedent this
mirrors throughout (`decoder/mlp.go`'s P18, `GOINFER_MOE_EXPERT_MAJOR`, same `!= "0"` convention)),
`GOINFER_SPLITKV_VSUM_SPLIT` (an unpromoted spike: splits the decode split-KV V-sum across the
key axis, measured +40-43% on the attention block and +15.6% served on one geometry, and it is
**NOT bit-identical** to the default path. Unset, the pipelines are not loaded and no scratch is
allocated. Its fidelity is NOT established — docs/measurements/vsum-split-spike-2026-09-13.md
says so in as many words — so do not set it on anything whose output matters). **Setting it refuses
speculative decoding**: `serve --spec ngram` and `--drafter` fail at startup, and the decoder's resident
spec entry points return an error, because decode would sum values in a different order from the batched
verify and the lossless guarantee would silently break (`decoder.SpecDecodeConflict`).

Gate/CI knobs read by `cmd/gate` and the harnesses: `GOINFER_GATE_BACKEND`,
`GOINFER_GATE_HEARTBEAT`, `GOINFER_GATE_SKIP_HEAVY`, `GOINFER_GATE_SKIP_WEBGPU`,
`GOINFER_REQUIRE_FIXTURES`, `GOINFER_TEST_NOTHINK`, `GOINFER_SPEC_PROBE_GIW`, and the
heavy-cell knobs `cmd/gate` reads through its `env()` helper: `GOINFER_GATE_MODELS` (the models
root), `GOINFER_HEAVY_RUN` (a -run filter), `GOINFER_HEAVY_TIMEOUT`, `GOINFER_HEAVY_PKGS`. `GOINFER_RELEASE_TAG` names the version being
tagged; set, it makes `TestParity_noPendingGateOutlivesARelease` enforce (RELEASING.md pre-flight 6,
and `release-assets.yml`), and unset it skips.

## CUDA graphs (perf, opt-in)

| Var | Purpose |
|---|---|
| `GOINFER_CUDA_GRAPHS` | Enable CUDA graph capture/replay on the static decode segments. |
| `GOINFER_CUDA_GRAPHS_ONLY` | Restrict graph capture to a scope (investigation flag). |
| `GOINFER_CUDA_GRAPHS_SYNC` | Force a synchronize around graph launches (investigation flag). |
| `GOINFER_CUDA_GRAPHS_UNSAFE` | Bounds-check-elision variant. **It also BYPASSES THE TENANCY GATE** (`cuda/graphs_safe.go`), so it is not a pure performance switch — N-42. |

## Diagnostics, probes & parity debugging

| Var | Purpose |
|---|---|
| `GOINFER_DECODE_TIMING` | Per-phase decode timing on the serve/decode path. |
| `GOINFER_DUMP_LOGITS` | Dump logits for cross-checking. |
| `GOINFER_MEM_PROBE` | RSS/heap attribution probe around the test suite. |
| `GOINFER_FAKEQUANT` (`_ACT` / `_EXPERTS` / `_PERROW`) | Parity-debugging: simulate a quantization on the f32 path to isolate a numeric gap. |
| `GOINFER_G4_CAPTURE` / `GOINFER_MOE_PROF_SPLIT` | Gemma-4 / MoE internal capture + profiling splits. |
| `GOINFER_GPU_CAPTURE` | Per-layer WebGPU resident-decode capture (attention context, post-attention and post-MLP residuals) into `DecodeRunner.ReadCapture` — the WebGPU twin of `decoder.Model.ForwardSubCapture`, for localising a resident-vs-CPU divergence to one sublayer instead of arguing from final logits (`docs/completed/task-webgpu-nogqa-decode-bug.md`). Nothing allocated or dispatched when unset. |
| `GOINFER_NORM_ULP_NOISE=<seed>` | Nudges every f32 norm vector a loaded model carries by an independent ±1/0 ULP per element (`decoder/normnoise.go`) — noise of exactly f32-rounding size, the same magnitude two correct implementations that reduce in a different order differ by. Measures a checkpoint's OWN sensitivity to that noise, the floor below which a resident-vs-CPU cosine on a quantized (W4A8/W8A8) forward carries no information about the kernels (`docs/completed/task-webgpu-nogqa-decode-bug.md`). |
| `GOINFER_ATTNFA_DEBUG=1` | R2 (`docs/tasks/red-october.md`) — prints each `attention_fa` dispatch's parameters (layer, depth, split count, group size, hd) to stderr from `canUseAttnFA`'s call site in `metal/model.go`, kept from the 2026-09-19/20 investigation (see `GOINFER_METAL_ATTN_FA` above) so dispatch parameters can be re-verified without rebuilding the print. No effect if `GOINFER_METAL_ATTN_FA=0` — since the kernel is default-on, this now has effect by default too. |
| `GOINFER_ATTN_TIMING_DEBUG=1` | R13 (`docs/tasks/red-october.md`) — wraps `attendBatchedHeads` in a `time.Now()`/`atomic.AddInt64` timer (`attnElapsedNanos`, `decoder/forwardn.go`), read out by the fast diagnostic `TestZZDiagGroupedFires` (`decoder/zz_diag_test.go`, `GOINFER_DIAG_STEPS`/`GOINFER_DIAG_CPUPROFILE`/`GOINFER_DIAG_TRACE`). Kept (not reverted like most temporary R-brief instruments) because it's the tool that made R13's served measurement tractable — a full `BenchmarkDecodeAtDepth` run costs 3-44 minutes per shape; this diagnostic costs 15-25 seconds and isolates attention's own wall time from the rest of the forward pass. It is what let R13 discover the wiring never fired before trusting a "no effect" reading, and what let a `go tool trace` per-goroutine breakdown find the real cause of a "slower once wired in" result (an accidentally-serialized softmax, not scheduler contention — a plain CPU pprof pointed at the wrong subsystem first; see `docs/measurements/r13-served-decode-2026-09-20.md`). No effect on output; adds one `time.Now()` call per `attendBatchedHeads` invocation when set. |


## CI & test gates

| Var | Purpose |
|---|---|
| `GOINFER_HEAVY_TESTS=1` | Opt into the ~120 heavy (real-checkpoint) tests that `go test ./...` otherwise skips. |
| `GOINFER_MANIFEST_EMIT` / `GOINFER_MANIFEST_MACHINE` | Emit parity-manifest rows / stamp the generating machine. |
| `GOINFER_PAR_THRESHOLD` / `GOINFER_PAR_WIDTH` | Sweep the CPU matmul parallel threshold (MACs) / fan-out width. |
| `GOINFER_PREFILL_GATE_PROMPTS` | `docs/completed/task-prefill-gap.md`'s §3 fidelity gate: `a` (default) or `b` selects which snapshotted 10-prompt set (`testdata/prefill-gate-prose-<label>/`) `TestPrefillGateReference`/`TestPrefillGateVsReference` build their reference and score against. |

## Test-model overrides (a large class — point tests at your local checkpoints)

Generic: `GOINFER_TEST_MODEL`, `GOINFER_PREQUANT_GGUF`, `GOINFER_MODELS_DIR`, `GOINFER_CUDA_MODEL`,
`GOINFER_METAL_MODEL`, `GOINFER_E2E_MODEL`, `GOINFER_EMBED_MODEL`. Per-family: one `GOINFER_<FAMILY>_GGUF`
(and occasional `_DIR`/`_INT4`/`_LABEL`) for deepseek, glm, granite, gptoss, llama3/4, nemotron, phi3,
gemma3/4, cohere/aya, mellum, mla, moe*, eagle/spec, giw, chatml, matrix, cpuint4, … — each skips its
test when unset.

## The exhaustive list

This registry is curated, not generated — but it is no longer allowed to drift from the code.
`TestEnvVars_docAndCodeAgree` (decoder/envvars_registry_test.go) fails when a variable is read by
non-test code and missing here, or documented here and referenced by no `.go` file at all. Both
directions had drifted: 40 variables were missing (including the escape hatches for all four
default-ON changes), one was documented while nothing in the tree read it, and three CUDA graph
knobs were written as suffix shorthand so their full names never appeared at all (N-42).

To enumerate every variable the code reads:

```sh
grep -rhoE 'GOINFER_[A-Z0-9_]+' --include='*.go' . | sort -u
```
