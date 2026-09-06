# demo/chat Tier 2 gates — Qwen3.5-0.8B as the embedded 0.5B replacement

`docs/task-demo-refresh.md` Tier 2, run in the doc's own kill order. **Gates 1 and 2 pass; gate 4
kills the straight swap**, and the doc pre-registered exactly this outcome.

## Gate 1 — license: PASS

All four small-series repos exist and are `apache-2.0` with a `LICENSE` file, checked per-repo on the
repo itself rather than the list endpoint (the doc's rule, after P10):

    Qwen/Qwen3.5-0.8B  Qwen/Qwen3.5-2B  Qwen/Qwen3.5-4B  Qwen/Qwen3.5-9B     apache-2.0, ungated

Embedding is permitted.

## Gate 2 — loader: PASS

The checkpoint is a multimodal wrapper (`Qwen3_5ForConditionalGeneration`, `pipeline_tag:
image-text-to-text`, with image AND video preprocessors), exactly as the doc anticipated — and every
piece it worried about is already handled:

| what | found | goinfer |
|---|---|---|
| config nesting | `text_config` | flattened at `decoder/config.go:1162` |
| weight prefix | `model.language_model.*` | detected at `decoder/weights.go:549` |
| DeltaNet tensors | `in_proj_qkv` / `_z` / `_a` / `_b`, `A_log`, `conv1d`, `dt_bias` | the separate-tensor branch, `decoder/weights.go:968` |
| layer pattern | `DDDSDDDSDDDSDDDSDDDSDDDS` — 18 DeltaNet + 6 softmax | the 3:1 hybrid this adapter was built for |
| ignored cleanly | `model.visual.*` (153 tensors), `mtp.*` (15) | loaded text-only without complaint |

Loads and generates through the existing dense `qwen3_5` adapter. **No loader work is required.**

## Gate 3 — template: thinking-capable, needs the non-thinking render

`chat_template.jinja` carries 11 `think` / 9 `reasoning` markers. Not a kill, but the P10 standing
rule applies: render the model's own template, non-thinking, or the demo gives a visibly worse first
impression at double the latency.

## Gate 4 — decode speed: **KILL for a straight swap**

Same box, same quant (int8int8), same commit, same harness as the incumbent measurement
(`demo-chat-incumbent-2026-08-22.md`) — `BenchmarkDecode`'s exact loop, prompt, greedy sampler and
`DefaultDecodeParallelThreshold`. 5 runs each, spread under 0.5%.

| model | tok/s | vs the 0.5B it would replace |
|---|---|---|
| qwen2.5-coder-0.5B (incumbent) | **28.1** | — |
| **Qwen3.5-0.8B (candidate)** | **13.5** | **2.08× slower** |
| qwen2.5-coder-1.5B (incumbent) | 12.1 | the candidate barely beats the tier ABOVE it |

The demo's pitch is "tiny + fast". A replacement that runs at half the speed and only just outpaces
the 1.5B tier fails that on its own terms.

### And the cause is mostly the VOCABULARY, not DeltaNet

The doc pre-registered the risk as "the hybrid's DeltaNet layers may decode slower per-token". The
arithmetic points elsewhere:

| | hidden | vocab | LM-head params |
|---|---|---|---|
| qwen2.5-coder-0.5B | 896 | 151 936 | 136.1 M |
| Qwen3.5-0.8B | 1024 | **248 320** | **254.3 M** |

The LM head alone is **1.87× larger**, against a measured **2.08×** slowdown. At this size the head
is a third of the model and is touched in full every token, so it dominates decode. Stated as a
strong attribution from parameter counts, **not** a measured decomposition — a profile would settle
how much of the residual 0.21× is DeltaNet.

This matters for what to do next: if the penalty were DeltaNet, a different hybrid member might
escape it. Because it is a 248 320-token vocabulary shared across the whole Qwen3.5 line, **no
member of the family escapes it**, and the conclusion generalizes beyond the 0.8B.

## Recommendation — the doc's own fallback

> "If the 0.8B loses the feel test, keep the Qwen2.5 tier alongside rather than replacing it."

That is the outcome. Gates 1 and 2 passing means the family is *usable* — it is a legitimate
flag-loaded example and a legitimate future tier — it just cannot be the "tiny + fast" headline. The
cheap follow-up, if Tier 2 is still wanted, is **Gemma 4 E2B**: apache-2.0 and ungated (see
`f21324b`-era Tier 1 notes), a dense architecture without the 248 K vocab penalty, and already
carrying a full-oracle parity row in the capability matrix.
