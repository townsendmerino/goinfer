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

## 2. Readings (2026-10-13, one run each, by day: exploratory figures, none is a result row)

Raw JSON and the serve logs' tails: `docs/measurements/kv3-long-prompt-census-2026-10-13/` (`census.py`, `census2.py`, `run-census.sh`, `run-census2.sh`).
Serve is branch `kv3-cuda-2026-10`'s CUDA binary; every run is a fresh process; prompts are unique (counter prefix) so no slot reuse serves them.

| model | resolved context, no `-ctx` | decline past 12,160 keys | batched 1.5k (ms/token) | per-token 1.5k (ms/token) | the ~12.4k prompt | projected 20,000 tokens |
|---|---|---|---|---|---|---|
| Qwen3-4B (control) | not read | no | 0.58 (0.86 s / 1,486) | 8.4 (12.45 s) | 12,250 tokens batched in 13.68 s | at least 22 s (1.12 ms/token at 12.2k) |
| Gemma-3-4B | **16,384** (backend default) | **yes**, at 12,241 keys, layer 5 (the first global layer) | 0.87 (1.30 s / 1,487) | 9.17 (13.63 s) | per-token; unfinished at the 150 s timeout | per-token ≥ **183 s**; batched ≥ 17 s |
| Gemma-4-E4B | **16,384** | **yes**, at 12,215 keys, layer 5 | 2.43 (3.62 s / 1,491) | 13.78 (20.55 s) | per-token; unfinished at 150 s | per-token ≥ **276 s**; batched ≥ 49 s |
| gpt-oss-20b (paged experts) | not read | **no**: 12,759 tokens taken batched (§3) | 9.70 (9.33 s / 962) | 11.41 (10.98 s) | 12,759 tokens in 122.1 s (9.6 ms/token) | about 190 s, from expert paging |
| Mellum2.1 (paged experts) | not read | no | 9.79 (14.55 s / 1,486) | 13.75 (20.43 s) | 12,250 tokens in 118.3 s (9.7 ms/token) | about 190 s, from expert paging |
| Qwen3.6-35B-A3B (hybrid, paged experts) | not read | none logged in 150 s (inference: a decline is logged at once, the Gemma logs show it) | 21.4 (31.91 s / 1,488) | 28.2 (41.94 s) | 12,242 tokens, unfinished at 150 s | at least 260 s at the 1.5k rate, from expert paging |

The Gemma-3 and Gemma-4 request logs name the decline the same way (`batched prefill declined for a 12241-token prompt, falling back to the per-token path`); at layer 5
the attention needs 49,476 B / 49,372 B of shared memory against the device's 49,152 B, and the fused attention does not serve the pass (head dim 256 on Gemma-3; Gemma-4's
global layers are head dim 512, local 256). The sliding-window layers never reach the bound; the global layers do.

Rates are shallow-depth (1.5k) figures: attention cost grows with depth, so each projection is a lower bound. Timeouts are the census's, not the server's.

## 3. What each reading does and does not say

- **Banner.** With no `-ctx` both Gemmas resolve **16,384** tokens (KV f32, 1.0 GB / 0.6 GB), so a 20,000-token prompt does not fit by default; with `-ctx 24576` it does
  (the runs above). The decline is reachable with no flag at all: any prompt in (12,160, 16,384] attended keys fits the default context and takes the per-token path.
- **gpt-oss.** The first run's "long" prompt tokenized to 11,482, under the bound, and tested nothing; `census2.py` re-ran it at 12,759 tokens: batched, no decline (head dim 64 is
  served by the fused kernel). Its cost is the same ~9.6 ms/token at 1k and at 12.7k and the batched path is only 1.18x the per-token one: expert paging, not attention, is its
  prefill. The same holds for Mellum2.1 (1.4x) and Qwen3.6-35B-A3B (1.3x).
- **Qwen3.6-35B-A3B.** No decline was logged, so by inference it is not taken off the batched path at 12,242 keys. Whether the pass completes was not measured (the 150 s timeout is
  shorter than the roughly 260 s the 1.5k rate implies). Whatever the path, the model's time is the expert paging.
- Not measured: a 20,000-token prompt on any model (every 20k figure is a projection from the 1.5k rate); a second sample of any cell; f16 KV.

## 4. Verdict against the registered rule

The rule as registered asks for a model that **fits a 20,000-token prompt in its resolved context**. Read literally with the default context, neither Gemma does (16,384), so the rule
returns "not worth building yet". With `-ctx 24576` (the setting a 20k agent needs anyway) both fit, both fall to the per-token path past the bound, and their projected times (183 s,
276 s) clear 60 s: **worth building**. The two readings disagree, and I do not choose between them by moving the bar; the decline also costs 112 s (Gemma-3) and 168 s (Gemma-4) at
12.2k keys, which the default context reaches, so on the rule's own 60 s measure the cost is already over the line at a length that fits with no flag. **Recommendation, the owner's call:**
build one tiled exact attention kernel (no head-dim limit, bit-identical to decode's split-KV) rather than a head-dim-256 fused instantiation: Gemma-3 needs 256, Gemma-4's global layers
need 512, and the exact kernel is what removes the 12,160 bound for any family. It is a PTX regeneration (pinned `nvidia-cuda-nvrtc-cu12==12.6.85`) and a bit-identity gate against
decode, plus a speed gate at night; this note registers none of that.

For the three paged-expert models the bound is not the cost, and a kernel would not change their 20k times: 190 s and up come from expert paging. That is a different lever (the C′
prefill pass), not part of this decision.
