# Task (goinfer +maybe aikit): W4A8 decode wiring — footprint feature

> **For:** Claude Code on the RTX 2070 SUPER box. The W4A8 GPU int4 GEMV
> kernel is built + parity-validated + de-risked (`5c3777f`, `196bd6d`,
> `d828ae5`; see `docs/gpu-assessment.md` §0.0). This task wires it into a
> production decode path. **The justification is FOOTPRINT, not speed:**
> W4A8 is ~96 tok/s (~1.1× int8's 89.7 — a byproduct), but int4 weights are
> 56% of int8, which is what makes the **7–12B class fit on the 8 GB card at
> all** (7B int8 = 7.07 GB doesn't fit; 7B int4 = 3.98 GB does, ≤16k ctx).
> It does NOT move the ~60% engine ratio vs Ollama — it ships 4-bit at
> Ollama's footprint. Gate accordingly (footprint/fit, not the 1.5B number).

## Decisions to settle in THIS doc before touching code

### D1 — the int4 `.giw` format contract (irreversible; pin it first)

The on-disk int4 weight format must be specified before `prequant` emits it,
because shipped models lock it. Match the kernel exactly:
- group = 32, value = `nibble − 8`, **per-group f16 scales** (f16 chosen for
  footprint: 56% vs 62%; it does not help speed — kernel is ALU-bound).
- A **new `.giw` version** (bump the magic/version; keep the int8 version
  loadable). Write the field order + header explicitly here, CRC as today.
- Decide: does the bundle carry int4 weights directly, or int8 + a flag?
  (Direct int4 — that's the footprint win; don't store int8 and quantize at
  load.)

### D2 — KV precision / context cap (a fork, not a discovery)

Fit table (7B int4, 8 GB / ~7 usable): 4k=4.55 GB, 16k=5.96 GB comfortable;
**32k=7.84 GB needs f16 KV.** Decision for W4A8-v1:
- **Recommended:** ship **f32 KV, cap context at 16k**, document the limit.
  Clean, fits, ships now.
- f16 KV is a **separate follow-on** (touches the cache + attn kernels +
  its own parity gate) — out of scope here unless explicitly pulled in.

### D3 — aikit boundary

`.giw` emit (`cmd/prequant`) and the `DecodeRunner` are goinfer. **If** the
f32→int4 group quantize helper must come from aikit (frozen v1.0), that's a
**v1.1.0 minor** (new feature, never a 1.0.x patch) — flag it as a
deliberate bump, don't reach into aikit silently. Prefer: the int4 quantize
lives goinfer-side next to the existing serialize path if it can.

## Increments (proposed order — de-riskable core first)

### 1. `ResidentW4A8` → W4A8 `DecodeRunner` path · bit-exact on the 1.5B

Mirror `ResidentW8A8`/`DecodeRunner` for the int4 kernel. Oracle already
exists: aikit `MatmulBTQ4` (CPU). Gate: **cosine 1.0 / argmax-exact vs the
CPU int4 path on the 1.5B**, the fast parity check. Greedy decode tokens
match CPU. This is the core; everything else is plumbing around it.

### 2. `cmd/prequant` int4 `.giw` emit (per D1)

Emit the int4 bundle (direct int4 weights + f16 group scales + tokenizer
metadata GGUF, as the int8 path does). Round-trip test: emit → load →
weights bit-match the in-memory quantized arrays (zero-copy alias like the
int8 `.giw`).

### 3. Decoder backend selection

`decoder.Options` / `cmd/serve` / `demo/chat` select the int4 `.giw` path
(`--quant int4` or auto from the bundle version). The runtime `--quant` flag
and GGUF fallback behave like the int8 path. No new public API churn beyond
the quant selector.

### 4. 7B int4 E2E fit + throughput gate (the feature's real test)

**Precondition:** stage a 7B int4 `.giw` on the box (prequant a 7B GGUF via
increment 2). Gate:
- **Fit:** 7B int4 loads resident on the 8 GB card with KV (f32, 16k) +
  DecodeRunner scratch + 150k-vocab logits — confirm `nvidia-smi` headroom,
  no OOM. This is the capability claim; it's the gate that matters.
- **Throughput:** record decode tok/s (expect the int4 kernel's ~250–280
  GB/s → a usable rate; the absolute number is secondary to "it runs").
- Keep the **bit-exact parity gate on the 1.5B** (increment 1) — the 7B has
  no fast CPU oracle, same as Mellum2.

## Out of scope

- f16 KV cache (D2 follow-on; needed only for >16k context on 7B).
- Closing the engine gap vs Ollama (structural WebGPU encode/glue wall,
  unchanged — recorded in §0.0).
- int4 prefill kernel / `dot4I8Packed` (upstream-blocked; prefill lever).
- Any change to the int8 path (additive only; int8 `.giw` stays loadable).

## Done when

- D1–D3 settled and written; int4 `.giw` format pinned with a version bump.
- Increments 1–3 landed, each with its gate; 1.5B bit-exact green.
- A 7B int4 model demonstrably loads + decodes on the 8 GB card (the
  footprint win, shown not asserted).
- `docs/gpu-assessment.md` §0.0 updated: W4A8 measured ~96 tok/s (not the
  estimated ~110), footprint = 56% of int8, 7B-class now fits ≤16k. CHANGELOG
  entry framed as footprint/capability, not speed.
- `feature-plan-v0.2.md`: GPU arc closed; note W4A8 done, then pivot —
  higher-leverage work is Track A (qwen3_5_moe GGUF, real-checkpoint parity)
  and Track B serve, not more GPU decode.
