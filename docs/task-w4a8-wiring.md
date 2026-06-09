# Task (goinfer/gpu): wire W4A8 into the GPU decode path — a FOOTPRINT feature

> **For:** Claude Code on the RTX 2070 SUPER / 3700X box. The W4A8 GPU int4
> GEMV kernel is built, bit-exact, and de-risked (`gpu/gemv_w4a8.go`, commits
> `5c3777f`, `196bd6d`, `d828ae5`; probe in `gemv_w4a8_test.go`). Read
> `docs/gpu-assessment.md` §0.0's W4A8 block first. **What remains is not risky
> code — it is contracts and forks, cheap to pin here and expensive to discover
> mid-build.** This doc pins them, then gives the wiring order.
>
> **Note (supersedes an earlier draft):** investigation of the actual codebase
> found the int4 `.giw` format, `prequant --quant int4`, and the int4 quantizer
> **already exist and ship** (CPU int4 path). So there is **no `.giw` version
> bump, no new prequant emit, and no aikit change** — the earlier draft's D1/D2/D3
> assumed those were new work; they are not. Pin to what exists.

## Why W4A8 (set the bar correctly — it is NOT the decode number)

The probe measured **~96 tok/s** (1.5B, ~1.07× over int8's 89.7) — a byproduct,
not the case. A 1.07× on the 1.5B does not justify a multi-surface build; that's
the "chase 100 on the 1.5B" vanity §4 already closed. **The case is footprint:**
int4 = **0.5625 B/weight** (56% of int8), so the model that *does not fit at
int8* becomes runnable. Fit arithmetic (Qwen2.5-7B, GPU KV f32):

| | int8 | int4 (W4A8) |
|---|---|---|
| 7B resident weights | 7.07 GB (won't fit) | **3.98 GB** |
| total @ 4k / 16k / 32k ctx | — | 4.55 / 5.96 / 7.84 GB |

7B int4 fits 8 GB **comfortably at ≤16k context**; 32k is tight → f16 KV (out of
scope, F2). W4A8 does **not** move the ~60% engine ratio vs Ollama (the WebGPU
encode/glue wall is unchanged) — it ships 4-bit at Ollama's footprint, a
parity-of-**capability** story. It unifies one int4 `.giw` *format* across the
GPU and CPU paths — but int4's win is GPU **footprint**, NOT CPU speed: the
decision matrix (`gpu-assessment.md` §1) measures CPU int4 **decode** at
**~0.15–0.18 tok/s** (20–45× slower than CPU int8int8). **aikit v1.0.1 rewrote
`MatmulBTQ4`** (dequant-row-once + single vectorized dot + column-outer M-reuse) —
a real win for **prefill / M>1** (at the 7B down shape: M=4 → 1.6× Q8, M=16 → 0.5×,
i.e. *faster* than Q8), but the §1 number is **decode (M=1)** where there is no
M-reuse, so it is unchanged from v1.0.0 (~7× slower than even `MatmulBTQ8`, and
`int8int8` decode uses the much faster `MatmulBTW8A8` quantized-activation kernel
→ the ~20–45× gap). Closing CPU int4 **decode** needs a genuine int4×int8 integer
kernel (W4A8-on-CPU, no f32 dequant) — its own aikit task, flagged 2026-06-08;
**keep avoiding CPU int4 for decode.**

## Contracts to pin BEFORE code (the irreversible / forky bits)

### C1. The int4 `.giw` serialization contract — ALREADY EXISTS; pin to it, do not invent

Shipping today (CPU int4 path). `decoder/serialize.go`, weightMat **kind 3 = q4**:

```
uint8  kind = 3
u32    rows (N)            // out features
u32    cols (K)            // in features
u32    group  = 32         // int4GroupSize, GGUF Q4_K granularity
uint8  w8a8   = 0
f32[]  q4s                 // [rows * ceil(K/32)] per-group scales (u32 len + LE f32)
bytes  q4                  // [rows * ceil((K+1)/2)] packed nibbles
```

- Nibble layout: element k → byte `k>>1`, **low nibble if k even**, high if odd;
  value = **nibble − 8**. (Matches `linalg.QuantizeGroupInt4Row` / `MatmulBTQ4`;
  the GPU probe already reproduces it bit-for-bit — that's why parity passed
  cosine 1.0.)
- Bundle `quantMode` tag = `quantInt4`; `Model.Quant()` already returns `"int4"`.
  `bundleVersion`/`giwVersion` are **unchanged** — int4 is a weightMat *kind*,
  not a format version.
- **Decision: the GPU reads this EXISTING layout unchanged.** No new version, no
  field reorder. The one genuinely irreversible surface is already frozen by the
  shipping CPU path — the discipline is "match it," not "design it."

### C2. f16 scales are VRAM-ONLY — NO new `.giw` version

The probe stores group scales at f16 *in the resident GPU buffer* for footprint
(62%→56%). The `.giw` on disk keeps **f32 `q4s`** (C1). `UploadW4A8` reads the
f32 `q4s` and packs f16 into the resident buffer at upload — disk format and CPU
path untouched. **W4A8-v1 adds no `.giw` version.** (An f16-scales-on-disk `.giw`
variant — smaller files — is a possible future optimization, **out of scope**;
the kernel is ALU-bound so it wouldn't help speed anyway, only file size.)

### C3. aikit boundary — NO bump needed for W4A8-v1

The f32→int4 quantizer (`linalg.QuantizeGroupInt4Row`) is in the **already-pinned
aikit** and already used by the shipping CPU int4 path (`decoder/weightmat.go`
`quantizeInt4`). The GPU W4A8 path only **consumes** the emitted int4 `.giw` — it
never quantizes. So no new aikit surface, no v1.1.0. (Prefer keeping any future
int4 helper goinfer-side next to serialize; a *deliberate* aikit minor is only on
the table if a later item needs a new shared primitive — flag it then, never reach
in silently.)

## Forks to decide HERE (not mid-wiring)

### F2. f16 KV — DECISION: v1 ships f32 KV, caps context at 16k

The fit table shows 32k needs f16 KV. The GPU KV cache is f32 everywhere
(`decodelayer.go` `NewKVCache`). **v1 keeps f32 KV and the 7B fit gate runs at
≤16k context** (~6 GB, comfortable). f16 KV is its own task (unlocks 32k for 7B
and headroom for 12B) — decided out of scope, not discovered mid-build.

### F3. Which matmuls go int4

Only the **projection GEMVs** (q/k/v/o, gate/up/down — 7/layer) and the **LM
head** are int4 in the `.giw`; logit-critical norms/embeddings stay higher
precision (`decoder/weightmat.go`). So the W4A8 DecodeRunner swaps **only the
gemv** for `gemv_w4a8`; rmsQuant / attn / swiglu / kv-store and the whole decode
graph are **unchanged** (activations stay int8). This is why the kernel was the
only real risk and it's already retired.

## Preconditions (list, don't hit mid-build)

- **P1. A 7B int4 `.giw` staged on the box.** The throughput/fit gate can't run
  without it. `cmd/prequant --quant int4 -o qwen2.5-7b-int4.giw <7B.gguf>`
  already supports this — but **verify it round-trips a 7B end-to-end first**
  (int4 prequant has mostly been exercised on small models; confirm it emits and
  the CPU decoder loads + decodes a 7B int4 before trusting the GPU gate).
  Fetching a 7B GGUF (~4–5 GB) is the one external dependency.
- **P2. Confirm `prequant --quant int4` output loads** via the existing CPU
  decoder on the 1.5B (sanity that C1 is exactly what prequant writes) before the
  GPU consumes it.

## Wiring order (the de-riskable core first)

1. **`ResidentW4A8` upload from a kind-3 weightMat → GPU model.** The GPU model
   builder / backend currently assumes int8 (`UploadW8A8` call sites). Add the
   int4 branch: a kind-3 weightMat (q4 + q4s + group) → `UploadW4A8`. Decide the
   `ModelW`/`LayerW` shape so a layer's projections are *either* `ResidentW8A8`
   *or* `ResidentW4A8` (interface or tagged union — keep it boring).
2. **W4A8 `DecodeRunner` path + 1.5B bit-exact gate** (the de-riskable core).
   Swap the projection `gemv` builder for `gemv_w4a8` when the weight is int4
   (F3 — nothing else changes). Gate: GPU W4A8 DecodeRunner vs a CPU oracle
   (`MatmulBTQ4` over the same q4/q4s, then the rest of the forward) on the 1.5B,
   **cosine ≥ 0.9999 / argmax-exact** (the int4 quant error is shared by both
   sides, so this is a clean kernel/wiring oracle, not an accuracy test). Real HW
   (`newOrSkipHW`), skips on software.
3. **Backend selection (small).** `decoder.Options` / `cmd/serve` / `demo/chat`
   route an int4 `.giw` to the W4A8 GPU path (auto from the bundle's `quantInt4`
   tag, or `--quant int4`); runtime `--quant` + GGUF fallback behave like the
   int8 path. No new public API beyond the quant selector. Verify
   prequant-emit → GPU-load decodes bit-identically to step 2's in-memory path
   (closes C1 end-to-end: disk → GPU).
4. **7B int4 fit + throughput gate** (the actual value prop). Load the P1 7B int4
   `.giw` on the GPU at ≤16k context, confirm it **fits** (no OOM; log resident
   bytes + `nvidia-smi` headroom vs the 8 GB budget) and decodes at usable tok/s
   (report effective GB/s, not just tok/s). The 7B has no fast CPU oracle (like
   Mellum2) — the bit-exact gate stays on the 1.5B; the 7B gate is fit+throughput.

## Gates / done when

- **Bit-exact (1.5B):** GPU W4A8 DecodeRunner vs CPU `MatmulBTQ4` oracle, cosine
  ≥ 0.9999 / greedy-argmax-exact, on real HW. The fast oracle, kept on the 1.5B.
- **Fit + throughput (7B):** a 7B int4 model loads, fits 8 GB at ≤16k context
  (resident-bytes logged under budget, no OOM), and decodes at usable tok/s.
  **This is the gate that tests the footprint claim** — not the 1.5B speed; the
  capability shown, not asserted.
- Existing int8 GPU path + all parity gates stay green (the int4 branch is
  additive; int8 `.giw` stays loadable, int8 models unaffected).
- `docs/gpu-assessment.md` §0.0 updated: probe → wired, measured 7B fit + tok/s
  replacing the projection. **CHANGELOG** entry framed as footprint/capability
  ("7–12B class now runs on 8 GB"), NOT speed.
- `docs/roadmap.md`: GPU arc closed (Track C) — note W4A8 done, then **pivot**. The
  higher-leverage work is Track A (qwen3_5_moe GGUF, real-checkpoint parity) and
  Track B serve, not more GPU decode tuning.

## Explicitly OUT of scope

- f16-scales-on-disk `.giw` variant (C2) — files only, no speed, defer.
- f16 KV cache (F2) — separate task; unlocks >16k-ctx 7B and 12B headroom.
- Closing the engine gap vs Ollama — structural WebGPU encode/glue wall,
  unchanged (recorded in §0.0).
- int4 prefill kernel / `dot4I8Packed` — upstream-blocked; a *prefill* lever.
- Any kernel re-touch — it's de-risked; if step 4 shows it slow at 7B shapes,
  that's a *measurement*, not a v1 blocker (the footprint claim stands
  regardless of the exact tok/s).
- Chasing the 1.5B decode number. W4A8's value is fit, not speed.

## RESULT (2026-06-08) — steps 1–2 + 4 done; step 3 (real .giw→GPU) scoped

Commits `ecfe527` (steps 1–2), `3331d1c` (MaxBufferSize + 7B fit bench), giw v2 +
bundle fix.

**Steps 1–2 (de-risked core): DONE, bit-exact.** `decodeWeight` interface →
one DecodeRunner builder for W8A8 or W4A8; W4A8 kernel got the addResidual
epilogue. Full-token int4 forward vs a CPU oracle (same int8-act×int4-weight
math): **cosine 1.000000, maxAbs 3.8e-6**. W8A8 path transparent (still
cosine 1.0 / maxAbs 0).

**Step 4 (the value prop): DONE — footprint PROVEN.** Real Qwen2.5-7B shape
resident int4 + 16k f32 KV on the 8 GB card:
- **FIT: 5.86 GB resident (3.98 weights + 1.88 KV), peak VRAM 6937/8192 MiB,
  ~1.25 GB headroom.** int8 weights alone = 7.07 GB → won't fit. A model that
  can't run at int8 runs at int4.
- **THROUGHPUT: 51 tok/s, 204 GB/s = 58% roofline** — HIGHER per byte than the
  1.5B int4 (~84 GB/s, ~24%); fixed per-token overhead amortizes at scale.
- Footprint is shape-determined → measured on the real 7B shape with synthetic
  int4; correctness gated on the 1.5B.

**Two 7B-scale findings, both resolved:**
- **MaxBufferSize** (`3331d1c`): device kept the 256 MB WebGPU default (the
  guarded raise never fired — `DefaultLimits()` u64-max sentinel). 7B head is
  272 MB int4 / 545 MB int8. Now unconditionally 2 GB.
- **.giw 4 GiB ceiling** (giw v2): bundle stored `uint32(len(weights))`,
  truncating the 5.17 GB 7B int4 blob → CRC corruption. Fixed (u64 length, v1
  readable). The 7B int4 `.giw` now emits (4.93 GB) and **decodes coherent
  tokens on CPU** — real disk→CPU closed.

**Still OPEN — real `.giw` *on GPU* (step 3 end-to-end):** needs a decoder→GPU
residency bridge (DecodeRunner is not wired into `decoder.Generate`; the webgpu
backend only exposes per-matmul primitives) + Qwen2 q/k/v **bias** support in the
DecodeRunner. Both are their own pieces; the footprint claim and CPU int4 path
are proven without them.
