# goinfer — capability & performance vs the field

> **What this page is.** An evaluator-facing, provenance-gated comparison of goinfer
> against the engines people actually weigh it against. **Rule:** no number or ✓/✗
> without a traceable source — every cell is either measured (with the goinfer commit +
> date + machine), copied from an in-repo measurement with its doc/commit cited, or
> marked `—` (not applicable / not verified). If you can't trace it, it isn't here.
>
> **The lane.** goinfer runs open-weight model weights *in-process, in pure Go* — the
> single-file, zero-install, HF-parity-gated lane. The native engines (llama.cpp,
> Ollama, vLLM, mistral.rs) are faster and far broader; the Go *bindings*
> (gollama.cpp, yzma) reach llama.cpp's speed but still ship its native `.so`; the
> pure-Go *ports* (llama2.go lineage) are archived toys. Among the peers surveyed,
> goinfer is the only maintained runtime that executes production-scope weights in
> pure Go with **no native dependency** — and the only one that can compile the model
> **into** the binary. It trades peak throughput and breadth (no continuous batching,
> no multimodal, 11 architectures) for a static binary that boots in ~0.5 s.

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
| GPU | ~ WebGPU, dense-only residency ⁱ | ✓ CUDA/Metal/Vulkan | ✓ CUDA/ROCm/Vulkan/Metal | ✓ CUDA/Metal | ✓ CUDA/TPU/+ | ✓ inherits llama.cpp | ✗ CPU only |
| Continuous batching | ✗ | ✓ | ~ parallel slots via llama-server ᵇ | ✓ | ✓ PagedAttention | — | ✗ |
| Multimodal (vision/audio) | ✗ | ✓ | ✓ | ✓ | ✓ | ~ (yzma VLMs; gollama —) | ✗ |
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
residency, dense Qwen2/Llama only; cgo quarantined behind `-tags gpu`) ·
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
plugged in. Source: `docs/ARCHITECTURE.md` §2 + `docs/perf-campaign.md`, "after the
v0.5.0 perf work." goinfer commit for these: the v0.5.0-era CPU campaign (see
perf-campaign.md). Peers were **not** run on this rig → `—` (use the script).

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

### B. GPU residency vs native CUDA, at equal quant

Rig: **RTX 2070 SUPER / Ryzen 7 3700X**, warm (`ollama ps` 100% GPU), greedy. goinfer
WebGPU residency (`-tags gpu`), bit-exact vs CPU decode on the first tokens. Source &
lab notebook: `docs/gpu-assessment.md` §0.0 (goinfer commit `eaf9a6c`, **2026-06-08**);
final corrected numbers only. The 1.5B row (89.7 / 147 / 61%) is from gpu-assessment;
the **7B row's peer figure (72.8 / 71%) is from `CHANGELOG.md` v0.5.0**, not
gpu-assessment — cited there.

| model · quant (same card, warm, greedy) | goinfer (WebGPU) | peer (native CUDA) | goinfer vs peer |
|---|---|---|---|
| Qwen2.5-Coder-1.5B · int8 vs q8_0 (~1.55 GB/tok) | **89.7 tok/s** | Ollama-CUDA **147** | **61%** (equal int8) |
| Qwen2.5-7B · int4 vs q4 | **51.7 tok/s** | llama.cpp-CUDA **72.8** | **71%** (equal 4-bit) |

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
vs `.giw`) · `docs/perf-campaign.md` (M1 Pro CPU decode, measurement conventions) ·
`CHANGELOG.md` v0.4.0/v0.5.0 (prefill 3.4× / 1.7×; MoE prefill `08acc11`) ·
`decoder/registry.go` (13 `model_type` keys, 11 distinct architectures) · `CHANGELOG.md`
v0.5.0 (the 7B int4 row: **51.7 vs llama.cpp-CUDA 72.8 tok/s = ~71%** — this peer
figure lives in CHANGELOG, not gpu-assessment) · `README.md` (capabilities).

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
