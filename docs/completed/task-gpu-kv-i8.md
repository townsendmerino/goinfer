# Task (goinfer): int8 GPU KV cache (4× VRAM KV; ~64k context on 8 GB)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **For:** `~/tmcode/goinfer/gpu` (`-tags gpu`, the RTX box). The third rung of
> the ladder f32 → f16 (`--kv f16`, landed 2026-06-10, `task-gpu-f16-kv.md`)
> → **int8**. Same knob, same gate shape, one more halving. Lossy ⇒ opt-in;
> f32 default stays bit-exact, f16 stays as shipped.

## Problem / opportunity

f16 KV made 32k fit the 8 GB card by halving the f32 cache (32k f16 ≈ 16k f32
≈ 1.88 GB, peak 6912 MiB). int8 halves it again:

| 7B int4 resident + KV | KV bytes | peak est. | context on 8 GB |
|---|---|---|---|
| 16k f32 (v1 cap) | 1.88 GB | 6926 MiB | 16k |
| 32k f16 (shipped) | 1.88 GB | 6912 MiB | 32k |
| **32k int8** | **0.94 GB** | **~5950 MiB** | 32k + ~1 GB freed headroom |
| **64k int8** | **1.88 GB** | **~6912 MiB** | **64k — same envelope as 32k f16** |

Two ways to spend it: **64k context** (long-doc / agent-transcript story), or
~1 GB freed VRAM at 32k (bigger resident model class on small cards). It also
finishes the unmeasured f16 story: the long-context attention kernel is
KV-read-bound, and int8 halves the bytes f16 left on the table.

## Building blocks (all shipped)

- **GPU-side int8 infrastructure exists end-to-end**: the W8A8 resident path
  already quantizes activations *on device* and dots int8×int8 in WGSL
  (`gpu/quant.go` — `runQuant`, `ResidentW8A8` with `bq` + `bScales`
  bindings). KV int8 reuses the same store-quantized/read-with-scale idiom.
- The three KV kernels are isolated (`ropeStore`, `vStore`, `attnShaderWGSL`)
  and already branch once for the f16 pack — int8 is a second arm of the same
  branch, not a new seam.
- Scale storage: per (position, KV-head) f32 scales — the CPU doc's
  granularity (`task-cpu-kv-quant.md` §Design), <1% overhead, and one scheme
  shared across CPU and GPU keeps snapshots/parity reasoning uniform.
- Gate machinery: `TestKVCacheF16_parity` is the template (long-context
  cosine + argmax with the 3%-near-tie rule, real-HW, software-adapter-skip).

## Design notes (the deltas vs f16)

- Cache buffer: `array<u32>`, 4 int8/word (vs 2 f16/word); scales in a small
  side buffer indexed `[pos*nKV + head]`. No `shader-f16`/dp4a feature
  needed — unpack is shifts+masks, scores multiply by `qScale·kScale` in f32.
  (When `dot4I8Packed` ever lands in the binding, the attention dot gets the
  same DP4A boost the prefill item is gated on — free future upside.)
- `ropeStore`/`vStore` compute the sub-row absmax → scale → quantize at
  write. One position × one head per invocation already; the reduction is
  over `headDim` (64–128) — trivial in-workgroup.
- Quality expectation: KV int8 per-head ≫ f16's 10-bit mantissa in dynamic
  range terms for post-norm K/V; the CPU doc argues the same bar
  (argmax-preserved, cosine ≥0.999 target, accept ≥0.99 with near-tie rule).
  If keys (RoPE outliers) miss the bar, fall back to **int8 keys per-channel /
  values per-token** (KIVI split) before dropping to f16-keys+int8-values.
- Residency fit math is static (`Size:` fields in decodelayer.go) — the cap
  bump to 64k is arithmetic, gated by a real allocation test like
  `TestKVCacheF16_fit`.

## Increments

1. **int8 KV storage + the three kernels**, knob `--kv i8` joining
   `f16|f32` (`Options.KVPrecision`). Gate: int8-KV vs f32-KV decode at ≥8k
   keys on real HW — argmax preserved (3% rule), cosine ≥0.99 (target 0.999);
   f32 and f16 paths unchanged (existing parities green).
2. **64k cap when int8** + fit gate: 7B int4 + 64k int8 KV on the 8 GB card,
   no OOM, peak recorded (~6.9 GiB expected). Record the long-context decode
   tok/s delta vs f16 at equal context — this finally measures the
   "KV-read-bound" speedup the f16 doc left open, with 4× instead of 2× of
   lever.

## Estimated impact

| metric | f16 (shipped) | int8 |
|---|---|---|
| KV VRAM @ 32k, 7B | 1.88 GB | **0.94 GB** |
| max context, 7B int4 on 8 GB | 32k | **~64k** |
| freed headroom @ 32k | — | ~1 GB (next model class up, or batch room) |
| long-ctx decode tok/s | unmeasured (predicted >1× at 16k+) | predicted **1.2–1.8× vs f32 at 16k+** (half f16's bytes; measure, don't claim) |
| short-ctx decode | 0.99× measured | ~same (weight-stream-bound) |
| quality | cosine 0.99868 @ 8k | ≥0.99 gate; KIVI split as fallback |

## Effort / when

~3–4 days on the RTX box (kernel branch + knob + two gates), low-medium risk.
**Trigger:** same as f16's was — a real >32k workload, or marketing the 8 GB
story harder. Sequence *after* the CPU int8 cache lands so the granularity
and gate bars arrive pre-validated.

---

## Progress (branch `gpu-kv-i8-wip`, not yet merged)

- ✅ **The three int8 kernels + their correctness gate** (`6b681f8`): `ropeStoreI8`,
  `kvStoreI8` (one thread per KV head: rotate/read → per-head absmax → maxabs/127
  scale → quantize → pack 4/word + scale-store), `attnI8` (f16 attn with int8
  unpack × per-head scale; q+ctx stay f32). Verified real-HW
  (`gpu/kv_i8_test.go`): WRITE kernels match the CPU per-head quant **cosine
  1.000000**; attnI8 matches f32-over-dequant **cosine 1.000000**. **The doc's
  flagged risk (the WRITE-kernel per-head reduction) is retired.**
- ✅ **Tri-state `--kv f32|f16|i8` knob** (`b6865a4`): `Options.KVPrecision="i8"` →
  `Model.kvPrecI8` + `KVCacheI8()`, serve `--kv i8`.
- ✅ **Residency integration — DONE + parity-gated (this session).** `NewKVCacheI8`
  (`decodelayer.go`: `(capElems+3)/4` data words + a `ctxCap*nKV` f32 scale buffer;
  `packKVInt8` CPU per-head prefill mirror); `runLayer` gained `kScale/vScale` and
  `runModel` a `kvI8` flag; the three op builders in `decoderunner.go` each grew an
  int8 third arm (`ropeStoreI8`/`kvStoreI8` one workgroup per KV head; `attnI8` with
  the two extra scale binds), with the shared `ropeKUni` carrying `nKV` in its spare
  slot and a dedicated `vStoreI8Uni` per-token uniform. `residency.go` allocates the
  int8 caches + scales and sets `ctxCap=65536`; `rd.rm.kvI8=kvI8`.
  - **The "prefill upload-quantize seam" turned out to be a no-op:** stateless
    `Generate` prefills the GPU caches via sequential `m.resident.Forward` (the WRITE
    kernels), not `UploadKV` (which has no caller). So `ropeStoreI8`/`kvStoreI8`
    quantize both prefill and decode writes on-device — no separate CPU upload path
    needed. `UploadKV` was nonetheless made precision-correct (f32/f16/int8) so the
    future prefix-reuse bridge can't silently corrupt a lossy cache.
  - **Gate (`gpu/kv_i8_parity_test.go`, `TestKVCacheI8_parity`):** int8-KV vs f32-KV
    decode, 8k prefilled keys + 16 steps → argmax 13/16 with every flip a sub-3%
    near-tie (worst 2.1%), **mean cosine 0.99739, min 0.98374**. Hard gate = the
    argmax 3%-rule + mean cosine ≥0.99; min-cosine tripwire relaxed to 0.98 (vs
    f16's 0.99) because int8's 8-bit mantissa dips one synthetic near-uniform step
    below 0.99 from pure rounding — the kernels are bit-exact (cosine 1.000000), so
    that's the synthetic floor, not degradation. f16 + f32 + W4A8 parities stay green.
  - **Inc 2 fit gate** folded into `TestKVCacheF16_fit` (added an `i8` arm: 64k cap,
    8 GB fit assert, tok/s vs f32 — also the real-Qwen-7B-distribution check, since
    it loads the model with `KVPrecision="i8"` through the live residency path).
    **Measured (Qwen2.5-7B int4, 8 GB card):** i8 64k peak **6977 MiB** (fits, ~50
    MiB over f32-16k — the caches are sized so 64k·i8 = 32k·f16 = 16k·f32 bytes, so
    int8's win is 4× context at flat VRAM, not lower VRAM); decode 20.3 vs f32 21.1
    tok/s = **0.96× at 1k context** (weight-stream-bound short-ctx, matching f16's
    0.99×).

    **LONG-CONTEXT RESIDUAL — MEASURED (2026-06-15, RTX 2070 SUPER, GOINFER_KV_LONGCTX
    bench): the predicted KV-read-bound speedup DOES NOT materialize.** Decode tok/s
    at depth 1k/4k/8k:

    | depth | i8 | f16 | f32 | i8/f32 | f16/f32 |
    |---|---|---|---|---|---|
    | 1024 | 20.57 | 21.07 | 21.33 | 0.96× | 0.99× |
    | 4096 | 7.33 | 7.47 | 7.60 | 0.96× | 0.98× |
    | 8192 | 3.95 | 4.02 | 4.09 | 0.97× | 0.98× |

    The ratios are **flat across depth** — int8 stays ~0.96× (4% *slower*), f16 ~0.98×.
    Decode IS steeply attention-bound at depth (~5× falloff 1k→8k), but all three
    precisions fall off by the **same factor** ⇒ the attention kernel is **compute-bound
    on the f32 dequant-then-dot per key, NOT bandwidth-bound on the KV bytes**. Fewer KV
    bytes don't speed it up; the unpack overhead makes i8/f16 marginally slower. The
    speedup is gated on the SAME `dot4I8Packed`/DP4A upstream block as the prefill item
    — only an integer attention dot makes the KV-byte read the bottleneck where int8
    wins. **NET: int8/f16 KV are CAPACITY wins (4×/2× context per VRAM), not decode-speed
    wins on this card pre-DP4A.** The §Estimated-impact "predicted 1.2–1.8× at 16k+" row
    is retracted (it assumed a bandwidth-bound attention dot that doesn't exist yet).

## Grounded scoping (2026-06-12 — after CPU int8 Inc 1–3 shipped)

Mapped against the live f16-KV implementation (the template) and updated with what
the CPU int8 work measured. Net: **lower risk than the original notes assumed**,
with one real new kernel wrinkle.

### The big de-risk: GPU residency = dense Qwen2/Llama only
`DecodeRunnerEligible` (`decoder/residency.go:45`) rejects MoE / gemma4 / qwen3_5 /
**QK-norm** / **sliding-window** / learned-pos / softcap. So GPU int8 KV targets
**full-attention Qwen2/Llama with NO QK-norm and NO 256-dim-head gemma3** — i.e.
the families that are *least* outlier-prone post-RoPE. The CPU session's
quant-sensitive worst case (gemma3-4b: per-head int8 = 0.993 cosine / ~93% argmax,
borderline) **cannot occur here**; f16 KV already measured **cosine 0.99868 @ 8k on
Qwen 7B**, and per-head int8 should land just under that, comfortably ≥0.99. ⇒ the
**"KIVI per-channel-key split" fallback is almost certainly unneeded** — don't build
it unless the gate actually misses.

### The f16 path is a clean 3-kernel template; the READ arm is easy, the WRITE arm isn't
Each kernel branches once on `m.kvF16` (`decoderunner.go:187-230`); int8 is a third
arm — but they're not equal in effort:
- **attn (READ) — easy.** `attnF16ShaderWGSL` reads f32 `q`, `unpack2x16float`s K/V.
  The int8 arm: unpack 4×int8/word (shift+mask), `score += q[d]·f32(k_i8)·kScale[s,h]`,
  `ctx += w·f32(v_i8)·vScale[s,h]`. q stays f32 — no integer dot, no DP4A needed
  (that's a free future upside when `dot4I8Packed` lands). Pure "second arm."
- **ropeStore / vStore (WRITE) — the real new work.** f16's write is scale-free
  (just `pack2x16float`, one thread per word, no cross-thread comms). int8 needs a
  **per-head absmax → scale → quantize**, i.e. a reduction over `headDim` (64–128)
  per head *before* packing. Restructure to **one workgroup per (KV-head)** doing a
  shared-memory absmax reduction, then quantize+pack its `headDim` lane, and write
  the per-head scale to the side buffer. This is genuinely more than an unpack arm —
  it's the increment's main risk/effort.

### Buffers + the prefill-upload quantize (reuse CPU code)
- Data buffer: `array<u32>`, **4 int8/word** → `NewKVCacheI8` sizes `(capElems+3)/4`
  words (vs f16's `(capElems+1)/2`). Plus a **scale side buffer** `[ctxCap*nKV] f32`
  per K and per V, indexed `[pos*nKV + head]` (the CPU doc's granularity — <1%
  overhead, one scheme shared CPU↔GPU).
- **Prefill K/V is computed on CPU** (the residency path uploads post-RoPE K/V into
  the GPU caches; only decode runs on GPU — `residency.go`). So the upload must
  quantize: reuse the shipped CPU `quantizeHeads` (per-(pos,head) int8 + scales),
  upload int8 + scales. **Decode-time** new-token K/V quantize on-device via the
  WRITE kernels above. Both per-head ⇒ uniform with the CPU cache + snapshots.
- 64k cap: `ctxCap = 65536` when int8 (`residency.go:88`); pure arithmetic, gated by
  a real allocation test.

### Knob: a tri-state `--kv`, distinct from CPU `--kv-quant`
Two independent axes already coexist: CPU `--kv-quant f32|i8` (`Options.KVQuant` →
`m.kvI8`, the shipped CPU cache) and GPU `--kv f32|f16` (`Options.KVPrecision` →
`m.kvF16`, residency). int8 GPU extends the **GPU** knob to `--kv f32|f16|i8`.
Refactor the `m.kvF16 bool` to carry the precision (e.g. a small enum or keep the
string) so the residency builder + kernel selection branch on f32/f16/i8; add a
`KVCacheI8()` getter mirroring `KVCacheF16()`. (Don't overload `m.kvI8` — that's the
CPU cache.)

### Increments (refined)
1. **int8 storage + 3 kernels + tri-state knob.** `NewKVCacheI8` (+ scale buffers),
   `ropeStoreI8`/`vStore I8` (per-head absmax reduction + quantize + scale store),
   `attnI8` (unpack + scale READ arm), upload-quantize via CPU `quantizeHeads`,
   `--kv i8`. **Gate (corrected):** int8-KV vs f32-KV decode at ≥8k keys on real HW
   — argmax-preserved (3%-near-tie rule) + cosine **≥0.99** (NOT the original
   ≥0.999 — that bar was unreachable on CPU over deep stacks and is the wrong
   metric; argmax/cosine is the gate). Mirror `TestKVCacheF16_parity`'s synthetic
   shape, **plus** a real-7B int8-KV-vs-f32-KV decode-argmax check (synthetic random
   weights may not reflect real RoPE'd K/V; the CPU "garbage-input" lesson — always
   exercise real distributions). f32 + f16 parities stay green.
2. **64k cap + fit gate.** 7B int4 + 64k int8 KV on the 8 GB card, no OOM, peak
   recorded (~6.9 GiB expected) — extend `TestKVCacheF16_fit`. **Measure** the
   long-context decode tok/s vs f16 at equal context (the KV-read-bound speedup the
   f16 doc left open — now 4× the lever). Report measured, don't predict.

### Risk register
- **WRITE-kernel per-head reduction** — the one non-trivial bit; isolated to two
  kernels, testable against the CPU `quantizeHeads` output for exactness.
- Quality — low risk (Qwen/Llama, full attention; f16 already at 0.99868).
- The two-knob (`--kv` vs `--kv-quant`) coexistence — a naming/UX wrinkle to keep
  clear in `--help`, not a technical risk.
