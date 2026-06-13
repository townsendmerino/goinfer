# Program (goinfer): KV-cache memory reduction — order + index

> **Audience:** internal planning. Umbrella for the four memory docs below;
> each is independently shippable with its own gates. Default decode path
> stays **bit-exact** throughout the program — every lossy step is an opt-in
> knob, per house rule.
>
> **Status (2026-06-12): steps 1 + 2 SHIPPED to main** (ring eviction + CPU int8
> KV Inc 1–3 — the ~20×-stacked Gemma-local headline below is live, opt-in,
> default bit-exact). **Next: step 3 (GPU int8 KV).** Step 4 (TurboQuant spike)
> open; int4 KV (step 2 Inc 4) deferred — see roadmap.md.

## Suggested order

1. ✅ **DONE (commit `4a9c2b5`).** **[Ring-buffer eviction for sliding-window layers](task-kv-ring-eviction.md)**
   — was masking only:
   `WindowStart` bounds reads, but `Append` stores every position forever. On
   Gemma (5 local : 1 global, window ≤1024) at 32k context ~81% of the KV
   cache is provably dead — the existing window goldens pin that invariance.
   Ring buffers on local layers: **~5.2× KV reduction on Gemma-class,
   LOSSLESS (bit-exact gate)**. The one real seam: `TruncateTo` rewinds
   deeper than the window → cold-prefill fallback. ~3–5 days. Free memory
   with zero precision argument beats cheap memory with one.

2. ✅ **DONE (commits `71d4ef1` / `d78eea9` / `d4e7830`).** **[CPU int8 KV cache, Increments 1–3](task-cpu-kv-quant.md)**
   — per-(position, KV-head) symmetric int8 on shipped aikit kernels
   (`DotI8`/SDOT, `QuantizeRowInt8`, dequant-into-scratch for prefill). 4× KV RAM
   on full-attention families, ~4× smaller `.giw-kv` warm sessions; opt-in
   `--kv-quant i8`, snapshot-v2 merged with step 1's windowed persistence (format
   bumped once). **Gate (measured, corrected):** argmax ~87–93% / cosine ~0.993 on
   gemma-3-4b-it, coherent generation (the 0.999 bar was unreachable over 34 layers
   and the wrong metric; a garbage-input test bug once faked a 0.73 "failure").
   **int4 (Inc 4) deferred** — ~1.6× over int8, likely needs per-channel keys; flip
   on a real >32k / multi-session RAM wall (roadmap.md).

3. ⏳ **NEXT — [GPU int8 KV cache](task-gpu-kv-i8.md)** — third rung after the shipped
   `--kv f16`, via the existing `runQuant`/W8A8 WGSL idiom. **~64k context
   on the 8 GB card in the 32k-f16 envelope**, or ~1 GB freed at 32k.
   ~3–4 days. Sequenced after doc 2 so the quant granularity and gate bars
   arrive pre-validated from the CPU side.

4. ⏳ **OPEN — [TurboQuant spike](../task-turboquant-spike.md)** — the only
   research-risk item; 2–3 day timebox, pointed at KV (3-bit, the published
   near-zero-loss claim) rather than weights, with a written go/no-go bar.
   **Go** ⇒ it replaces doc 2's deferred int4 increment; **no-go** ⇒ the
   watch item closes for KV.

## Combined impact (1 + 2 landed)

| family | KV RAM at long context | quality |
|---|---|---|
| Gemma-class local layers (rings × int8) | **~20× smaller** | rings bit-exact; int8 gated |
| Mellum2 | ~3.7× (rings) × 4 (int8) | same |
| Qwen2/Llama (full attention) | **4×** (int8) | gated, opt-in |
| warm sessions on disk | ~4× smaller (+O(W) on local layers) | exact restore |

Assessed and rejected across the program (reasons in
[task-cpu-kv-quant.md](task-cpu-kv-quant.md) §Non-goals): CPU f16 KV
(dominated by int8 on CPU), XQuant rematerialization (wrong direction when
compute-bound), attention-score eviction / AhaKV (quality-risky, breaks exact
prefix-reuse; sliding windows already serve the families that want bounded
attention).
