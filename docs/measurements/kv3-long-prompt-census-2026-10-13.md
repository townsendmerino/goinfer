# Which models still fall off the batched prefill past 12,160 keys, and what that costs (2026-10-13)

Follow-up to `docs/measurements/kv3-cuda-2026-10-13.md` §2.4 ("still declines, by name"). nobara-pc, RTX 2070 SUPER 8 GB, CUDA; serve built from branch
`kv3-cuda-2026-10`. A measurement, no code change.

## 1. Question and decision rule (registered before the runs)

**Question.** For each model a user is likely to point a 20k-token agent at on this card, does a 20,000-token prompt (a) fit the resident
context the server resolves, (b) take the batched prefill past 12,160 keys, and (c) if not, what does the per-token path cost?

**Models:** Qwen3.6-35B-A3B (hybrid Gated-DeltaNet + attention, paged experts), Gemma-3-4B (head dim 256, sliding window), Gemma-4-E4B (E-model),
gpt-oss-20b (head dim 64, sinks, paged experts: the model KV0 found declining), Mellum2.1 (paged experts). Qwen3-4B is the control (fused, measured).

**Readings.**
1. *Banner:* the context the server resolves with no `-ctx` and with `-ctx 24576`, and the decode path (resident or not). A prompt that cannot fit
   the resolved context is not a prefill question on this card.
2. *Decline, by name:* `PrefillLast` of 12,400 tokens at a resolved context that allows it: batched or declined, and the reason. A decline in the
   guard costs nothing (it fires before any compute).
3. *Rates:* time to prefill 1,500 tokens batched (where it runs) and per token (`ForwardNoLogits` over the same 1,500), each in ms a token. The per-token
   figure at shallow depth is the **projection base** for a 20,000-token prompt (labelled a projection: attention cost grows with depth, so it
   is a lower bound of the time).
4. Where a model does take the batched path at 12,400 tokens, its time for that pass (a measurement).

**Decision rule.** A tiled exact kernel (or a head-dim-256 fused instantiation) is **worth building** for a model class if some model of it **fits a
20,000-token prompt in its resolved context on this card, falls to the per-token path past the bound, and its projected 20,000-token time
exceeds 60 s** (the speed gate's ambiguous band). It is **not worth building yet** for a class whose models cannot hold 20,000 tokens of context on
this card at all. Costs are not in the rule: the build is a PTX regeneration and a bit-identity gate against decode's split-KV.

**Estimates.** Banners: about 5 minutes (loads). Rates: about 1 to 3 minutes a model. All by day.
