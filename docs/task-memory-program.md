# Program (goinfer): KV-cache memory reduction — order + index

> **Audience:** internal planning. Umbrella for the four memory docs below;
> each is independently shippable with its own gates. Default decode path
> stays **bit-exact** throughout the program — every lossy step is an opt-in
> knob, per house rule.

## Suggested order

1. **[Ring-buffer eviction for sliding-window layers](task-kv-ring-eviction.md)**
   — ship first. Today "sliding-window eviction" is masking only:
   `WindowStart` bounds reads, but `Append` stores every position forever. On
   Gemma (5 local : 1 global, window ≤1024) at 32k context ~81% of the KV
   cache is provably dead — the existing window goldens pin that invariance.
   Ring buffers on local layers: **~5.2× KV reduction on Gemma-class,
   LOSSLESS (bit-exact gate)**. The one real seam: `TruncateTo` rewinds
   deeper than the window → cold-prefill fallback. ~3–5 days. Free memory
   with zero precision argument beats cheap memory with one.

2. **[CPU int8 KV cache, Increments 1–3](task-cpu-kv-quant.md)** — the core
   quant step. Per-(position, KV-head) symmetric int8 on shipped aikit
   kernels (`DotI8`/SDOT, `QuantizeRowInt8`, dequant-into-scratch for
   prefill). **4× KV RAM on full-attention families (Qwen2/3, Llama), 4×
   smaller `.giw-kv` warm sessions, ~1.5–2.5× long-context decode on small
   models**; opt-in `--kv-quant i8`, gate argmax-preserved + cosine ≥0.999.
   ~6–9 days. Coordinate the snapshot-v2 format bump with doc 1's windowed
   persistence so the format bumps once.

3. **[GPU int8 KV cache](task-gpu-kv-i8.md)** — third rung after the shipped
   `--kv f16`, via the existing `runQuant`/W8A8 WGSL idiom. **~64k context
   on the 8 GB card in the 32k-f16 envelope**, or ~1 GB freed at 32k.
   ~3–4 days. Sequenced after doc 2 so the quant granularity and gate bars
   arrive pre-validated from the CPU side.

4. **[TurboQuant spike](task-turboquant-spike.md)** — the only
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
