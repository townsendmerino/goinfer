# Task (goinfer/gpu): Ollama/llama.cpp calibration — the one missing measured number

> **For:** Claude Code on the RTX 2070 SUPER / 3700X box. The GPU decode
> campaign closed at **89.7 tok/s** (1.5B int8, `docs/gpu-assessment.md`
> §0.0), but every external-comparison figure in the doc ("~52% of the CUDA
> ceiling") is **estimated, not measured** — no Ollama/llama.cpp number was
> ever run on this hardware. This task fills that in so the assessment's
> competitive claim rests on a real number. ~30 min; no toolkit build
> (Ollama ships its own CUDA runtime).

## What we have vs what's missing

Measured on this box, all real: goinfer CPU 8.5 tok/s · goinfer GPU **89.7**
· staged hybrid 25.6. **Missing:** any llama.cpp/Ollama decode rate on the
same card. The doc's "CUDA ceiling ~170 tok/s" is roofline-scaled from a
generic "4090 / 7B ~150–200 tok/s" figure — replace it with a measurement.

## The apples-to-apples trap (read before running)

goinfer's 89.7 is **int8 weights (~1.55 GB streamed/token)**. Ollama's
default 1.5B is **q4_K_M (~0.9 GB)**. Decode is bandwidth-bound, so q4 reads
~1.7× fewer weight bytes and *should* be ~1.5–1.7× faster — that's quant,
not engine quality. Two honest comparisons; run both:

1. **Same-quant (engine vs engine):** pull/serve an **int8 / q8_0** 1.5B in
   Ollama so both stream ~the same bytes. This isolates "goinfer GPU vs
   llama.cpp CUDA at equal quant" — the real engine-efficiency number.
2. **As-shipped (what a user gets):** Ollama default q4_K_M vs goinfer int8.
   This is the honest "what each tool does out of the box" gap, and it
   motivates the **W4A8** future-work item (the only lever on goinfer's
   4.3 ms gemv floor).

Match everything else: same model family (Qwen2.5-Coder-1.5B — goinfer's
demo model), greedy/temp 0, a warm run (discard the first, cold-load
skews), and read the **eval/decode tok/s** (not prompt-eval/prefill).

## Steps

```bash
# Ollama ships a bundled CUDA runtime — no nvcc/toolkit needed.
# (install per ollama.com if absent; confirm it sees the GPU)
ollama --version
nvidia-smi            # confirm the 2070S is visible to Ollama

# (2) as-shipped q4_K_M
ollama pull qwen2.5-coder:1.5b
ollama run --verbose qwen2.5-coder:1.5b "Write a Go function that reverses a slice." 
#   → read "eval rate: N tokens/s" from the --verbose footer; rerun once warm.

# (1) same-quant q8_0 (int8-equivalent) — the fair engine comparison
ollama pull qwen2.5-coder:1.5b-instruct-q8_0    # or `ollama show` to find the q8 tag
ollama run --verbose qwen2.5-coder:1.5b-instruct-q8_0 "Write a Go function that reverses a slice."
```

- Confirm Ollama is actually on the GPU (`nvidia-smi` shows the process; or
  `ollama ps` shows `100% GPU`). A CPU-fallback Ollama number is worthless
  here.
- If a q8_0 tag isn't published for this model, note it and use q4_K_M only,
  clearly labelled — don't silently compare int8-goinfer to q4-Ollama as if
  same-quant.
- Optional, only if cheap: the same-API Vulkan llama.cpp build
  (`-DGGML_VULKAN=ON`, `llama-bench`) — the fair WebGPU-vs-Vulkan number.
  Skip if it's a yak-shave; the Ollama CUDA number is the priority.

## Record (in `docs/gpu-assessment.md`)

Replace the estimated "~52% of the CUDA ceiling" in §0.0 and the open item
(§0.5 list, the "llama.cpp calibration — blocked" bullet) with a small table:

| engine | quant | bytes/tok | decode tok/s | vs goinfer-GPU |
|---|---|---|---|---|
| goinfer GPU (WebGPU) | int8 | ~1.55 GB | 89.7 | 1.00× |
| Ollama (CUDA) | q8_0 | ~1.55 GB | ? | ? (engine gap) |
| Ollama (CUDA) | q4_K_M | ~0.9 GB | ? | ? (+ quant) |

Then state the takeaway honestly: the q8_0 row is the engine-efficiency
verdict (how close pure-Go/WebGPU gets to llama.cpp's tuned CUDA at equal
quant); the q4 gap quantifies what W4A8 would recover. Update the §0.0
"~52% of CUDA ceiling" line to cite the measured ratio, or delete it if the
number contradicts the estimate.

## Done when

- Both Ollama runs measured warm, confirmed GPU-resident, eval-rate
  recorded.
- §0.0 + §0.5 updated: estimate replaced by the table; the calibration open
  item closed.
- One-line verdict written: "at equal (q8) quant, goinfer-GPU is N% of
  Ollama-CUDA on the 2070S" — the honest competitive number the whole
  assessment was missing.

## RESULT (2026-06-08) — DONE

Qwen2.5-Coder-1.5B (qwen2 arch, embedding 1536 = goinfer's shape), warm, two
runs each, `ollama ps` = 100% GPU, Ollama 0.30.6 bundled CUDA, RTX 2070 SUPER.

| engine | quant | bytes/tok | decode tok/s | vs goinfer-GPU |
|---|---|---|---|---|
| goinfer GPU (WebGPU) | int8 | ~1.55 GB | 89.7 | 1.00× |
| Ollama (CUDA) | q8_0 | ~1.6 GB | 147 (146.2, 149.1) | 1.64× *(engine, equal quant)* |
| Ollama (CUDA) | q4_K_M | ~0.9 GB | 186 (187.5, 184.9) | 2.07× *(engine + quant)* |

**Verdict: at equal quant (q8/int8), goinfer-GPU is 61% of Ollama-CUDA** (89.7 vs
147) — top of the predicted 40–60% band, thesis confirmed with margin. Ollama's
own q4→q8 is only 1.27× (186/147), so the as-shipped gap is mostly quant width.
W4A8 sizing (see §0.0): it cuts *only* goinfer's gemv (4.3→2.2 ms), not the fixed
glue/encode, so goinfer-int4 ≈ 110 tok/s — ~59% of Ollama's q4 186, i.e. the same
~60% engine ratio, NOT parity. Caveat: an earlier `qwen15-q4` run (191 tok/s) was
a Qwen1.5-**1.8B** q4 — wrong model, discarded.

## 7B CALIBRATION (2026-06-08) — the engine gap NARROWS at scale

Same box (RTX 2070 SUPER), Ollama bundled CUDA, warm, `ollama ps`-confirmed
placement. Qwen2.5-Coder-7B (qwen2 7.6B, embedding 3584 — goinfer's 7B shape).

| engine | quant | placement | tok/s | eff. GB/s | % roofline |
|---|---|---|---|---|---|
| goinfer / WebGPU | int4 | 100% GPU | **51** | ~204 | **58%** |
| Ollama / CUDA | q4_K_M | 100% GPU | **72.8** | ~310 | ~89% |
| Ollama / CUDA | q8_0 | **31% CPU / 69% GPU** | — | — | — (does NOT fit 8 GB) |

- **q8_0 7B (8.5 GB) does not fit 100% GPU on the 8 GB card** — Ollama offloads
  31% to CPU, so there is no valid pure-GPU q8 number at 7B. This is the int4
  value prop restated from the engine side: at 7B, *only 4-bit runs pure-GPU on
  8 GB*; int8/q8 can't. (At 1.5B both fit; at 7B the int8 class is out.)
- **goinfer's 7B engine ratio NARROWS the gap vs llama.cpp at scale — now
  MEASURED end-to-end (commit 6d8e681), not shape-bench.** A real int4 `.giw`
  through `decoder.Generate` on the webgpu residency path (greedy 16/16 matching
  the CPU oracle): **1.5B 102.4 tok/s, 7B 51.7 tok/s** (TTFT 0.66 s / 1.3 s for a
  ~30-token prompt, option-(a) GPU prefill). At equal 4-bit quant, goinfer-GPU /
  Ollama-CUDA tok/s is **55% → 71%** (1.5B 102.4/186 → 7B 51.7/72.8). Both
  engines amortize their fixed per-token overhead at scale, but goinfer gains
  MORE — it had more fixed overhead (the WebGPU encode/glue tax) to amortize — so
  it closes ~⅓ of the gap going 1.5B→7B. (The earlier 52%/70% shape-bench
  estimates were right: the CPU sampler is 0.14 ms/token, negligible, so
  end-to-end ≈ shape-bench. The apples-to-oranges worry — shape-bench excludes a
  sampler Ollama's eval-rate includes — does not bite.) The structural wall
  (WebGPU dispatch vs CUDA megakernel) remains, but its *relative* cost shrinks
  with model size.

**Headline:** the 7B is where the capability story and the closing engine gap
meet — int4 is the *only* way to run a 7B pure-GPU on 8 GB, and at that scale
goinfer reaches **~71% of llama.cpp's tok/s end-to-end** (up from ~55% at 1.5B).
Measured through the production `decoder.Generate` path on the webgpu backend, not
a shape-bench.
