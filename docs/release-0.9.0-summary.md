# goinfer — work summary for 0.9.0 planning (last 2 days, both backends)

> ~95 commits across two machines: **Metal** (Mac / Apple Silicon) and **CUDA** (Linux / NVIDIA),
> plus shared decoder + admission infrastructure. This is the "what landed + what's left before we
> tag 0.9.0" brief. Both new GPU backends are **cgo-free** (`CGO_ENABLED=0`, pure Go via
> purego/dlopen) and **opt-in** (build tags + blank import + `--backend`), with graceful CPU
> fallback — so 0.9.0 can ship them without risk to the default pure-Go build.

## 1. Two new cgo-free GPU decode backends — both shipped, opt-in

### CUDA (Linux box) — `cuda/`, `-tags cuda`
- cgo-free CUDA via **gocudrv** (dlopen libcuda) + **NVRTC**. Full resident decode + prefill.
- **Perf (measured, real q4_k_m):** 1.5B **218.6 tok/s** (~1.23× Ollama at int4); 0.5B **507.5 tok/s**.
- Optimization arc: coalesced W4A8 GEMV (43%→71%→80% of peak), ILP unroll, launch diet
  (18→13 dispatches/layer), f16 group scales, on-device greedy argmax, pinned logits readback,
  and **super-kernels K1/K3a** (fold rmsnorm into the QKV / gate-up GEMV) — K1+K3a shipped
  (+21.3% on 0.5B); K2 measured null and reverted.
- **Qwen3 now runs GPU-resident on CUDA** (QK-norm kernel added). Validated on Qwen3-1.7B: 6/8
  teacher-forced, coherent generation incl. the `<think>` block. qwen2.5 unchanged.
- Wired into serve/demo (opt-in, fallback), CI builds/vets/tests `-tags cuda`.

### Metal (Mac) — `metal/`, `-tags metal`, darwin-only
- cgo-free Metal via **purego-objc** (dlopen Metal.framework, `objc_msgSend`, MSL 3.1 compiled at
  runtime). Full resident decode + prefill. Defused the `LC_BUILD_VERSION`/MSL-2.4 landmine.
- **Decode perf arc: ~20 → 73.6 tok/s** (0.98–1.04× the ~71 GO bar = 85% of Ollama-Metal; 2.2×
  the WebGPU-on-Metal baseline). Levers: threadgroup-per-head attention (fixed a hidden 68%),
  coalesced + Fable "Stage A" W4A8 GEMV (uint4 + staged activation), encode-ahead executor.
- **Diagnosis (measurement-grounded): int-MAC / issue-bound**, not bandwidth/dispatch. Confirmed
  via cgo-free GPU-timestamp capture + the f16-scales-flat-GPU-time test. **Speculation ruled out**
  (batch-k GEMM doesn't amortize on this issue-bound kernel — the make-or-break came back
  negative). **Megakernel closed on Metal** (no grid-sync primitive; redundant-recompute is
  net-negative). ~73–77 tok/s is the practical ceiling for W4A8 batch-1 decode on M1 without DP4A.
- **Prefill: f16 `simdgroup_matrix` (MMA) path — 3.7× faster TTFT** (140-tok prompt 2034→543 ms).
  Blocked MMA GEMM with **in-kernel int4→f16 dequant (zero extra RAM)**, f16 activation path,
  batched causal attention, wired via optional `decoder.Prefiller`.
- **KV-f16** (halves KV cache memory).
- Wired into generation (`--backend metal --quant int8int8`); coherent generation verified.

## 2. Model-architecture coverage (the big new-families push)

Actually running new models caught a real bug the unit tests couldn't — **the As-cap bug**:
Qwen3's `head_dim` is independent of `hidden` (`nH·hd ≠ H`), which overflowed a hardcoded
activation-staging buffer and would have **silently broken every real Qwen3/Mistral size**. Fixed
on Metal with dynamic threadgroup memory (validated bit-exact 1536→4096); the CUDA audit found it
**immune** (dynamic shared memory, K taken from the weight, not a dim constant).

| Family | Metal | CUDA | Notes |
|---|---|---|---|
| Qwen2 / Llama | ✅ | ✅ | flagship, both tuned |
| **Qwen3** | ✅ e2e | ✅ e2e | per-head QK-norm; cross-checked correct on both |
| **Mistral** | ✅ components¹ | ❌ declines | sliding-window (Metal); CUDA has no window yet |
| **Phi-3** | ✅ | (n/a) | windowing free via decoder fix; partial-rotary unit-validated |
| Gemma 2/3 | ❌ declines | ❌ declines | structural (softcaps, sandwich norm, GeGLU, embed-scale) |
| MoE / MLA / YaRN | ❌ declines | ❌ declines | correctly declined by the shared taxonomy |

¹ Metal Mistral: As-cap@4096 bit-exact + sliding-window unit-tested; full **7B coherence run needs
a >16 GB Mac** (int8+int4 doesn't fit 16 GB comfortably).

**Silent-wrong-output class eliminated:** admission is now one **shared feature taxonomy**
(`decoder/features.go`) — requirements derive from arch flags, each backend declares its
implemented set (`ResidentBackendFeatures`), admission is a subset check, and a **registry-driven
test** fails when a new arch is unclassified. Metal + CUDA + WebGPU all use it. Also fixed a
decoder bug it surfaced: `phi3Architecture` silently dropped `sliding_window` on **every** backend
(CPU included) — now honored.

## 3. Cross-cutting / hygiene
- `go fix` modernization sweep (range-over-int) + gofmt across the repo; parity manifest re-blessed
  (audited numerics-neutral).
- Decoder arch-flag validation tests + registry-driven admission coverage gate.
- Legal: `THIRD_PARTY_LICENSES.md` regenerated, gocudrv/purego attribution.
- The **Qwen3 parity "gap"** (Metal 0.6B 15/24) was chased down and **confirmed NOT a bug** — two
  unrelated backends hit the identical 62.5% ratio, CUDA's int4-vs-int4 gate rules out
  quantization, and generation is coherent on both. It's adversarial teacher-forced ids (near-ties)
  + f32-vs-CPU-f64 reduction (Metal GPUs have no `double`, so unbridgeable — cosmetic).

## 4. Open items / decisions before tagging 0.9.0

**Likely release-blocking to decide:**
1. **Ship posture for the GPU backends.** Both are opt-in, cgo-free, fallback-safe. Is 0.9.0 the
   "GPU backends land as opt-in" release? If so, the README/docs need a clear "how to enable
   Metal/CUDA + which models" section.
2. **Coverage messaging.** Qwen2/Llama/Qwen3 are GPU-resident on both; Mistral GPU-resident on
   Metal only; Gemma/MoE/MLA decline to CPU. Decide what we *advertise* vs what silently falls
   back, so users aren't surprised.

**Nice-to-close before or shortly after 0.9.0 (non-blocking):**
3. **Mistral-7B full e2e on Metal** — needs a >16 GB Mac (only the coherence run is unverified;
   components are). CUDA Mistral needs a sliding-window kernel (contained, like the CUDA QK-norm).
4. **Gemma** — the biggest remaining family; structural (5+ features). Correctly declines today.
5. **MoE** — high-value, isolated (router + top-k + per-expert MLP). Declines gracefully today.
6. **Metal decode ceiling** — ~73–77 tok/s is issue-bound; a "Stage C" GEMV or long-context KV
   work are the only levers, both modest. Prefill (3.7×) is the bigger realized win.

**My read:** the substance for 0.9.0 is done and solid — two cgo-free GPU backends, a real
coverage expansion (Qwen3 on both, Mistral/Phi-3 on Metal), and a shared admission taxonomy that
makes the silent-wrong-output class structurally impossible. The main pre-tag work is **docs +
release messaging** (items 1–2), not more kernels. Everything is on `main`, synced, tests green.
