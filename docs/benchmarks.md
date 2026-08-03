# goinfer — capability & performance vs the field

> **What this page is.** An evaluator-facing, provenance-gated comparison of goinfer
> against the engines people actually weigh it against. **Rule:** no number or ✓/✗
> without a traceable source — every cell is either measured (with the goinfer commit +
> date + machine), copied from an in-repo measurement with its doc/commit cited, or
> marked `—` (not applicable / not verified). If you can't trace it, it isn't here.
>
> **The lane.** goinfer runs open-weight model weights *in-process, in pure Go* — the
> single-file, zero-install, HF-parity-gated lane. The native engines (llama.cpp,
> Ollama, vLLM, mistral.rs) are far broader, and faster everywhere except the one lane
> the cgo-free CUDA backend now wins outright (dense 4-bit decode — §B2); the Go *bindings*
> (gollama.cpp, yzma) reach llama.cpp's speed but still ship its native `.so`; the
> pure-Go *ports* (llama2.go lineage) are archived toys. Among the peers surveyed,
> goinfer is the only maintained runtime that executes production-scope weights in
> pure Go with **no native dependency** — and the only one that can compile the model
> **into** the binary. It trades peak throughput and breadth (no continuous batching,
> vision-in only — no audio, CPU-slow — 11 architectures) for a static binary that
> boots in ~0.5 s.

---

## Methodology (non-negotiable — applies to every number on this page)

A measurement enters a table **only if it satisfies all of**:

- **Same machine** for goinfer and any peer it's compared against (rig named per row).
- **Same model checkpoint** and **same quant** — the exact GGUF/`.giw` named.
- **Greedy decode, fixed seed** (we measure the engine, not sampling luck).
- **Pinned versions**: goinfer commit + each peer's version, inline.
- **Date** of the run, and a **thermal note** (plugged in, warm, repeated runs, median).

Anything not matching all of these is `—` and a re-measure, never a guess. This page
exists *because* a sloppy comparison is worse than none: `docs/gpu-assessment.md`
caught one of its own early runs comparing a **Qwen1.5-1.8B q4** against the
**Qwen2.5-1.5B** target (a 191 tok/s number) and discarded it. Same-checkpoint,
same-quant, same-machine is the whole discipline.

Reproduce goinfer's side end-to-end (and get the verbatim peer commands) with
[`scripts/bench_compare.sh`](../scripts/bench_compare.sh).

**Forward vs serve split (new — future rows may state both).** Every published row here is a
**serve-path, client-wall-clock** number (prefill, sampling, detok, HTTP all inside it) — the
methodology-symmetric bar, and it stays. But a direct decomposition now exists
(`GOINFER_DECODE_TIMING=1`, `docs/task-moe-streaming.md`): on the 26B the greedy **forward** is
~36 ms/tok while the **serve** number is ~57 ms/tok, the gap being **prefill amortization** (sequential
full-logits prefill, no batched `PrefillLast`) plus context-depth growth — *not* a per-token serve tail
(greedy `sample`/`embed` ≈ 0). So future rows may state both a forward rate and a wall-clock rate, the
way §B2 already prints Ollama's decode-only rate alongside its wall clock — with the wall-clock number
remaining the one that counts for a peer comparison. Existing rows (incl. §B4's 16.98 tok/s, re-confirmed)
are correctly measured and unchanged.

---

## Table 1 — Capability matrix

Durable, mostly-boolean axes. goinfer's `✗` stay in the table. Legend: **✓** yes ·
**✗** no · **~** partial / caveated · **—** not verified or N/A. Peer cells verified
against each project's repo/docs on **2026-06-10** (citations in *Sources*); re-verify
at each goinfer tag.

| Capability | **goinfer** | llama.cpp | Ollama | mistral.rs | vLLM | Go bindings¹ (gollama.cpp · yzma) | pure-Go ports² (llama2.go) |
|---|---|---|---|---|---|---|---|
| Runs weights in-process, **no native dep** | ✓ pure Go, no cgo ᵃ | ✗ *is* the C/C++ lib | ✗ spawns native `llama-server` (GGUF) + MLX ᵇ | ✗ Rust; GPU links CUDA/Metal | ✗ Python + PyTorch + CUDA | ✗ dlopen prebuilt llama.cpp `.so/.dll/.dylib` ᶜ | ✓ genuinely pure Go |
| Single static binary, zero install | ✓ ᵃ | ~ standalone binary, model separate | ~ binary + `llama-server` companion ᵇ | ~ one CLI binary; GPU needs CUDA/Metal | ✗ pip / Python env | ✗ needs the native lib ᶜ | ✓ (but toy) |
| Model **embeddable in the binary** | ✓ `.giw` mapped from the image ᵈ | ✗ | ✗ | — | ✗ | ✗ | ✗ |
| HF logit-parity gate (a contract) | ✓ ᵉ | — | — | — | — | — | — |
| Constrained / structured decode | ✓ **struct-derived** grammar ᶠ | ✓ GBNF + JSON-schema | ✓ JSON schema | ✓ grammar + strict schema | ✓ xgrammar / `guided_json` | — | ✗ |
| OpenAI-compatible server | ✓ pure stdlib ᵍ | ✓ `llama-server` | ✓ | ✓ (+ Anthropic) | ✓ (heavy deps) | ✗ (library only) | ✗ |
| LoRA adapters | ✓ PEFT, merged at load ʰ | ✓ | ✓ | ✓ | ✓ | — | ✗ |
| GPU | ~ WebGPU (broad residency) + **cgo-free CUDA & Metal** (dense + MoE; `features.go`-gated) ⁱ | ✓ CUDA/Metal/Vulkan | ✓ CUDA/ROCm/Vulkan/Metal | ✓ CUDA/Metal | ✓ CUDA/TPU/+ | ✓ inherits llama.cpp | ✗ CPU only |
| Continuous batching | ✗ | ✓ | ~ parallel slots via llama-server ᵇ | ✓ | ✓ PagedAttention | — | ✗ |
| Multimodal (vision/audio) | ~ **vision in** (Gemma 3 VL, pure-Go SigLIP → serve + agent; ~171 s/image CPU or **18.8 s on `-tags gpu`**, no audio) | ✓ | ✓ | ✓ | ✓ | ~ (yzma VLMs; gollama —) | ✗ |
| Model coverage | ~ **11 architectures** ʲ | ✓ dozens | ✓ broad | ✓ broad | ✓ 200+ | ✓ inherits llama.cpp | ✗ Llama-2 toy |

**Reading it:** goinfer wins cleanly on *no-native-dep pure-Go execution* and
*model-in-binary* (no surveyed peer does either at production scope), and on
cold-start/footprint (Table 2). It ties on the library surface (constrained decode,
LoRA, OpenAI server). It **loses**, honestly, on GPU breadth, continuous batching,
multimodal, and model coverage — those are the native engines' and vLLM's turf.

¹ Go *bindings*: "no CGO" (purego/ffi) but still download/bundle and `dlopen` a native
llama.cpp shared library — not pure-Go inference. ² The one genuinely pure-Go,
in-process port (nikolaydubina/llama2.go) is **archived (2024-11-30)**, CPU-only,
Llama-2-in-`llama2.c`-format, ~0.87 tok/s on 7B — educational, not production.

Cell provenance: ᵃ `README.md` (pure Go, no cgo, cross-compiled static binary) ·
ᵇ Ollama PR #16031 (CGO runner removed → upstream `llama-server` for GGUF + MLX for
safetensors; merged 2026-05-29 — so continuous batching is whatever `llama-server`'s
`-np` slots give, not an Ollama-documented feature) · ᶜ gollama.cpp / yzma READMEs
(purego/ffi that `dlopen` a prebuilt llama.cpp lib) · ᵈ `ARCHITECTURE.md` §2 (`.giw`
weights mapped from the binary's read-only image) · ᵉ `README.md` + `CHANGELOG.md`
(forward-pass numerics parity-gated vs HF; per-family logit-parity tests) ·
ᶠ `README.md` (`GrammarFromStruct` — grammar derived from a Go struct) · ᵍ `README.md` /
`cmd/serve` (OpenAI-compatible server in pure stdlib) · ʰ `README.md` (LoRA PEFT,
merged at load) · ⁱ `ARCHITECTURE.md` §2 + `docs/gpu-assessment.md` (WebGPU full
residency; cgo quarantined behind `-tags gpu`) + §B2/§B3 below (`cuda/`, `metal/`:
driver-JIT / MSL, **CGO_ENABLED=0**, admission-gated by
`decoder/features.go`) ·
ʲ `decoder/registry.go` — **13 registered `model_type` keys, 11 distinct architectures**
(`gemma3_text`/`qwen3_5_moe_text` are text-decoder aliases of `gemma3`/`qwen3_5_moe`).

---

## Table 2 — Measured performance

Two rigs, both the repo's existing ones. Apple-Silicon CPU is the pure-Go lane; the
RTX section is the GPU-residency story. Peer columns are filled **only** from
same-machine/same-quant runs; where we have none, the cell is `—` and the script +
peer commands are how you fill it.

### A. Apple Silicon CPU (the pure-Go lane)

Rig: **Apple M1 Pro** (6P+2E), prequant `.giw` int8, greedy + fixed prompt/seed,
plugged in. Source: `docs/ARCHITECTURE.md` §2 + `docs/completed/perf-campaign.md`, "after the
v0.5.0 perf work." goinfer commit for these: the v0.5.0-era CPU campaign (see
completed/perf-campaign.md). Peers were **not** run on this rig → `—` (use the script).

**Re-measured under load 2026-08-02 (`14dfc47`) — consistent with the baseline.** The
table's ~70/~36 remain the headline: they were measured on a clean rig (v0.5.0 campaign).
A spot re-run of `BenchmarkDecode` this pass came back at ~68 (best 69.8) tok/s for 0.5B
and ~34 (best 34.5) for 1.5B (best-of-8 × 2 s, `qwen2.5-coder-{0.5b,1.5b}-instruct-q4_k_m.gguf`
loaded `int8int8`), but that run had the IDE holding ~4 cores (load avg 3.8), so those are a
*lower bound*, not a replacement — landing just under the clean baseline is exactly what a
contended re-run should do. The takeaway is that the ~70/~36 held across both
decode-parallelism-threshold fixes (the M1-Pro int8 crossover was already optimal, so unlike
the Ryzen path it had no room to move; see `tune.go` sweep note); a clean re-measure would be
needed to *revise* the headline. The end-to-end demo row (~57/~26) was **not** re-measured.

| metric (M1 Pro · `.giw` int8 · greedy/fixed seed) | goinfer 0.5B | goinfer 1.5B | peers |
|---|---|---|---|
| decode tok/s — `BenchmarkDecode` (pure forward+sample) | ~70 | ~36 | — |
| decode tok/s — end-to-end demo (incl. UI/stream) | ~57 | ~26 | — |
| cold start → first token | **0.48 s** | **1.23 s** | — |
| resident heap (`phys_footprint`) | **77 MB** | **87 MB** | — |
| install footprint | one static binary, model included | one static binary, model included | — |

Context for the footprint win (same 0.5B, M-series, `docs/ARCHITECTURE.md` §2):
embedded-GGUF path → prequant `.giw` cuts **cold start 2.30 s → 0.48 s (~5×)** and
**resident heap 772 MB → 78 MB (~10×)** — the weights are mapped from the binary's
read-only image, not heap-copied.

**Prefill (relative, CPU):** vectorizing prefill attention onto the SIMD path gave
**~3.4× dense prompt prefill** and **~1.7× Gemma 4** (M1 Pro; `CHANGELOG` v0.5.0,
`7fa82c2`/`88b7aaa`). Sparse-MoE prefill is now batched too — **2.4×** on Mellum2
12B-A2.5B (**3.36 → 8.11 tok/s** at a 1024-token prompt), measured on the RTX-box CPU
(**Ryzen 7 3700X**, `08acc11`, 2026-06-10) — a *different* rig, listed separately so it
isn't conflated with the M1 numbers above.

**Vision prefill (SigLIP, gemma-3-4b-it, 896², 4096 patches):** ~171 s/image on
CPU (compute-bound matmul; int8 is a wash on AVX2 — no VNNI). On `-tags gpu` with
`--backend webgpu` the resident GPU encoder runs the whole tower on-device:
**18.8 s/image** (~9×) on an **RTX 2070 SUPER**, parity cosine 1.000000 vs the CPU
W8A8 encoder (`886c8fd`/`5d7c572`, 2026-06-11). The attention matmuls are still
naive f32 — a tiled GEMM there is the next lever (`docs/completed/task-gpu-vision-tower.md`).

### B. GPU residency vs native CUDA, at equal quant

Rig: **RTX 2070 SUPER / Ryzen 7 3700X**, warm (`ollama ps` 100% GPU), greedy. goinfer
WebGPU residency (`-tags gpu`), bit-exact vs CPU decode on the first tokens. Source &
lab notebook: `docs/gpu-assessment.md` §0.0 (goinfer commit `eaf9a6c`, **2026-06-08**);
final corrected numbers only. The 1.5B row (89.7 / 147 / 61%) is from gpu-assessment;
the **7B row's peer figure (72.8 / 71%) is from `CHANGELOG.md` v0.5.0**, not
gpu-assessment — cited there.

| model · quant (same card, warm, greedy) | goinfer (WebGPU) | peer (native CUDA) | goinfer vs peer |
|---|---|---|---|
| Qwen2.5-Coder-1.5B · int8 vs q8_0 (~1.55 GB/tok) | **111.6 tok/s** | Ollama-CUDA **147** | **76%** (equal int8) |
| Qwen2.5-7B · int4 vs q4 | 51.7 tok/s *(pre-coalescing, stale)* | llama.cpp-CUDA **72.8** | 71% (equal 4-bit) |

> **These WebGPU rows are int8/q8_0 and are NOT comparable to the cgo-free CUDA rows
> below**, which are 4-bit on both sides. Two different backends, two different quants,
> two different peer measurements — read them as separate experiments, never as a
> before/after of the same thing.

### B2. cgo-free CUDA (`-tags cuda`) vs Ollama-CUDA — 4-bit both sides

The `cuda/` backend is a **driver-JIT, CGO_ENABLED=0** path (dlopen `libcuda` via
purego; PTX embedded, no toolkit at build or run time — re-verified by `ldd` at the
commit below showing no `libcuda`/`libnvrtc` linked). It admits an arch only when it
implements every feature that arch needs, else declines to the staged/CPU path
(`decoder/features.go`, the authoritative set). **The numbers below are the dense lane.**
At this commit (`7557723`, 2026-07-16) the backend was dense-only, but MoE (routed +
ungated shared expert) and partial rotary landed the next day (2026-07-17) — `features.go`
now carries `cuda`'s `FeatMoE`/`FeatPartialRotary` plus the full Gemma norm/GeGLU/embed set —
so those lanes are real but **unmeasured here** (still declined: MLA, SSM, the gated shared
expert, YaRN mscale, logit softcap).

Provenance, all rows: **RTX 2070 SUPER**, driver **595.58.03** / Ryzen 7 3700X ·
goinfer commit **`7557723`**, **2026-07-16** · peer **Ollama 0.5.7** · both sides
q4_K_M, warm (`/api/ps` shows `size_vram == size`, 100% GPU), greedy (`temperature:0`),
256-token completions (both hit the cap), **best of 3 warm runs** (the first run after
load is discarded as a warmup outlier on both sides).

**Method — server-to-server, identical on both sides.** Each engine is driven through
its *own* HTTP server (goinfer `cmd/serve` `/v1/chat/completions`; Ollama
`/api/generate`) and timed by **client wall clock**, so prefill, sampling, detokenize,
JSON, and HTTP are inside *both* numbers. This is the only methodology-symmetric
comparison on this page.

| model · q4_K_M (same card, warm, greedy, wall-clock) | goinfer (`-tags cuda`) | Ollama-CUDA 0.5.7 | goinfer vs peer |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | **429.8 tok/s** | **211.1** | **2.04×** |
| Qwen2.5-Coder-1.5B | **210.0 tok/s** | **149.3** | **1.41×** |

- **Conservative cross-check.** Ollama also self-reports a *decode-only* rate
  (`eval_count / eval_duration`, excluding prefill): **216.8** / **153.4** tok/s. Scoring
  goinfer's all-in wall clock against Ollama's prefill-free figure — a bar tilted
  *against* goinfer — still gives **1.98×** / **1.37×**. The ratio is not a
  measurement artifact. (Ollama's wall and decode-only agree within ~3% here, so its
  server overhead is negligible and the two framings barely differ.)
- **Correctness is gated, not assumed.** CUDA decode is held to the repo's own 3%
  near-tie parity rule against the CPU path on a real q4_k_m checkpoint (9/10 exact
  argmax, 0 hard fails) — the speed is only meaningful because the tokens match.
- **Why the 0.5B ratio is bigger:** small-model decode is launch/issue-bound, and the
  cgo-free path's per-token dispatch is cheap (measured executor tax **15.3 µs**/token).
  The advantage narrows as the model grows and work becomes bandwidth-bound — expect
  the trend to continue past 1.5B. **Do not extrapolate these ratios to 7B**; that row
  is unmeasured.
- **Scope, stated plainly:** these are dense-lane numbers (0.5B/1.5B dense) — a real-speed
  claim about the dense lane, **not** a coverage claim. Coverage is `decoder/features.go`,
  which since 2026-07-17 grants `cuda` MoE (routed + ungated shared) + partial rotary + the
  Gemma set; MoE-resident speed is unmeasured. (Earlier drafts of this bullet said "dense
  architectures only / No MoE" — stale as of 2026-07-17.)

### B3. cgo-free Metal (darwin, `--backend metal`) vs Ollama-Metal — 4-bit both sides

Provenance: **Apple M1 Pro, 16 GB**, macOS **26.5.2** · peer **Ollama 0.32.0** · both
sides q4_K_M, warm, greedy, 256-token completions (both hit the cap), server-to-server
wall clock — the same method as §B2. Measured on the Mac, **2026-07-16**. The `metal/`
package is `//go:build darwin` (no extra tag) and is selected at runtime with
`--backend metal`; it registers via `decoder.RegisterBackend` and must be blank-imported
by the binary.

| model · q4_K_M (M1 Pro, warm, greedy, wall-clock) | goinfer (Metal) | Ollama-Metal 0.32.0 | goinfer vs peer |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | **~128 tok/s** | **~124** | **1.03×** |
| Qwen2.5-Coder-1.5B | **~61 tok/s** | **~79** | **0.77×** |

- **Metal's story is not raw speed — it's cgo-free / no-Xcode.** goinfer is at parity on
  0.5B and **loses** on 1.5B. Apple GPUs have no DP4A, so integer decode is issue-bound;
  the CUDA ratio does **not** carry over. An earlier "~85% of Ollama-Metal" estimate is
  **superseded** — the real figure is size-dependent (1.03× → 0.77×).
- Do not quote a Metal speed *multiple* as a headline. The defensible Metal claims are
  portability (no Xcode, no toolchain, static binary) and correctness parity.

### B4. Host↔VRAM MoE streaming — a 26B that does not fit the card (cgo-free CUDA)

This is a **capability** row, not a speed row: it has **no peer number** because the peers do
not produce one. **Gemma 4 26B-A4B** (128 experts, top-8, ~11.4 GB of int4 experts) does not fit
an 8 GB card; **llama.cpp and Ollama fail to load it**, they do not run it slower. goinfer decodes
it GPU-resident by keeping the ~1.3 GB non-expert core in VRAM and streaming the experts from
pinned host memory into a VRAM slot cache (host↔VRAM "C′" path, `docs/task-moe-streaming.md`).

| RTX 2070 SUPER · 8 GB · driver 595.58.03 · 2026-08-02 | value |
| --- | --- |
| model | Gemma 4 **26B-A4B**, int4 (128 experts, top-8, 30 layers) — experts ~11.4 GB, **does not fit 8 GB** |
| decode | **16.98 tok/s** (64-tok greedy, capture-free, synchronous H2D) |
| expert cache | 38 slots/layer (auto-capped from 48 to measured free VRAM), **81.6% hit rate** (17816 / 4024) |
| resident VRAM | ~1.3 GB core + ~3.8 GB slots + KV — the 11.4 GB of experts live in host RAM |
| coherence | greedy through the real chat template: distinct-trigram 0.818, *"…**Paris**… the Eiffel Tower, the Louvre Museum… **Gastronomy:**"* |
| peers | **llama.cpp / Ollama: fail to load** (model exceeds VRAM) |
| commit | `cuda/gemma4_26b_cache_test.go` (`GOINFER_HEAVY_TESTS=1`), host↔VRAM track through `2f51449` |

- The number is a **floor**: synchronous H2D with no async overlap yet, and it still clears
  fieldfare's 5.1–6.3 tok/s on its own 8 GB M2 Air (the closest analogue either project has —
  *different silicon, so a floor in the peer's constrained regime, not a comparison row*).
- The 81.6%-hit-rate-at-30%-residency is the empirical basis of the design: trained MoE routing
  is a stationary skew, so an LRU cache over a small resident fraction captures most reads
  (`turbo-fieldfare` reaches the same conclusion via 16-slot LFU — two mechanisms, one property).

> **serve caveat (both GPU backends).** GPU-resident models bypass the session cache in
> `cmd/serve` (`7557723`), so they skip prompt-prefix reuse and speculative decode. That
> is a deliberate trade — resident decode is worth far more than the TTFT optimization —
> but it means a resident server re-prefills each request. The numbers above include
> that re-prefill on goinfer's side.

> **Fresh re-measure (2026-07-14, CUDA-spike step 0):** the 1.5B-int8 row is re-measured
> on this 2070 SUPER after the buffer-coalescing win (`f8ef42b`/`5c3777f`) that postdated
> the 89.7 figure — **89.7 → 111.6 tok/s** (best of 6 × 48-tok greedy, resident int8,
> `TestDecodeRealModel_throughput`), closing the native-CUDA gap **61% → 76%**. This is
> the number the CUDA-megakernel-spike GO bar keys off: **≥1.3× = 145 tok/s int8**. The
> 7B-int4 row was not re-measured (also predates coalescing — treat as stale).

- The 7B-int4 row is the *footprint* headline: it **fits and decodes pure-GPU on an
  8 GB card** — the model class that does not fit at int8.
- goinfer-GPU is portable **WebGPU** (cgo quarantined behind `-tags gpu`) measured
  against years-tuned **native CUDA** — 60–70% of the CUDA engines at equal quant is
  the ceiling `gpu-assessment.md` predicted and hit (decode is glue-serialization-
  bound near a WebGPU wall, not bandwidth-bound; see §0.5 there).
- **Provenance gap (disclosed):** the 2026-06-08 source run did not pin the exact
  Ollama / llama.cpp **version**. Treat the ratios as "as-of-2026-06-08"; re-pin peer
  versions on the next run via `scripts/bench_compare.sh`. The checkpoints, quants,
  card, and greedy/warm conditions *are* matched.

---

## Reproduce it

`scripts/bench_compare.sh` runs **goinfer's** side end-to-end on your machine —
`BenchmarkDecode` (decode tok/s), `BenchmarkPrefillLong` (prefill), cold-start wall
clock to first token, and resident memory (`phys_footprint` on macOS, max-RSS on
Linux) — stamping each with the goinfer commit + date + machine. It then **prints the
peer commands verbatim** (`ollama run --verbose`, `llama-bench -ngl 99`, vLLM) for you
to run yourself on the same machine; it never drives the peers (their install is
yours). Peers absent → it still emits goinfer's column.

```bash
GINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q8_0.gguf \
  scripts/bench_compare.sh            # add GINFER_GPU=1 for the -tags gpu residency row
```

---

## Maintenance rules (so this page never rots into a lie)

- **Every number carries its date + goinfer commit + peer version, inline.** No
  floating "~90 tok/s" without the run that produced it.
- **Re-run `scripts/bench_compare.sh` at each tag.** A number more than one minor
  version stale is re-measured or struck — never silently carried forward.
- **Re-verify the capability matrix against peer release notes at each tag** (cheap —
  it's mostly booleans). Peers move: Ollama dropped its CGO runner for a `llama-server`
  subprocess (PR #16031, merged 2026-05-29); mistral.rs shipped multi-model + an OpenAI
  Responses API. A stale ✓/✗ is as misleading as a stale tok/s.
- **The bar is provenance, not flattery.** If a cell can't be traced to a measurement
  or a peer's own docs, cut it.

---

## Sources

**goinfer (in-repo):** `docs/gpu-assessment.md` §0.0 (GPU residency: 89.7 / 51.7 tok/s;
61% / 71% of CUDA at equal quant; the discarded wrong-model 191 tok/s; commit `eaf9a6c`,
2026-06-08) · `docs/ARCHITECTURE.md` §2 (cold-start / heap / binary tiers; embedded-GGUF
vs `.giw`) · `docs/completed/perf-campaign.md` (M1 Pro CPU decode, measurement conventions) ·
`CHANGELOG.md` v0.4.0/v0.5.0 (prefill 3.4× / 1.7×; MoE prefill `08acc11`) ·
`decoder/registry.go` (13 `model_type` keys, 11 distinct architectures) · `CHANGELOG.md`
v0.5.0 (the 7B int4 row: **51.7 vs llama.cpp-CUDA 72.8 tok/s = ~71%** — this peer
figure lives in CHANGELOG, not gpu-assessment) · `README.md` (capabilities) ·
**§B2 (cgo-free CUDA)** — measured 2026-07-16 on this repo at commit `7557723`, RTX 2070
SUPER / driver 595.58.03, peer Ollama 0.5.7 via `/api/generate`; goinfer via `cmd/serve`
`/v1/chat/completions`; both wall-clock, best of 3 warm; lab notes in
`docs/task-cuda-cgofree-spike.md` · **§B3 (cgo-free Metal)** — measured 2026-07-16 on
Apple M1 Pro 16 GB / macOS 26.5.2, peer Ollama 0.32.0, same server-to-server method;
notes in `docs/metal-model-coverage.md`.

**Peers (verified 2026-06-10, against each project's repo/docs):**
llama.cpp — github.com/ggml-org/llama.cpp (README; `tools/server/README.md`;
`grammars/README.md`; `docs/multimodal.md`).
Ollama — github.com/ollama/ollama PR #16031 (CGO runner removed → `llama-server` + MLX,
merged 2026-05-29); docs.ollama.com (modelfile/LoRA, structured outputs, OpenAI API).
mistral.rs — github.com/EricLBuehler/mistral.rs (README; `guides/serve/openai-responses-api.md`;
`guides/serve/multiple-models.md`; v0.8.3, 2026-06-01).
vLLM — github.com/vllm-project/vllm (README, 200+ archs); docs.vllm.ai
(structured outputs / xgrammar); en.wikipedia.org/wiki/VLLM (V1 engine).
gollama.cpp — github.com/dianlight/gollama.cpp (README: purego bindings, dlopen native
llama.cpp; v0.2.2-llamacpp.b6862).
yzma — github.com/hybridgroup/yzma (README: purego+ffi, `llama.Load(libPath)`, VLM
examples; v1.16.1, 2026-06-08).
llama2.go — github.com/nikolaydubina/llama2.go (README; **archived 2024-11-30**, CPU-only).

> *Peer-state note:* the web sweep that populated the peer columns read each project's
> canonical repo files/READMEs (GitHub) and search summaries; a couple of cells were
> marked "unverified" by the sweep and are `—` here rather than guessed. Before a
> release that leans on this page, re-open the flagged primary URLs to confirm wording.
