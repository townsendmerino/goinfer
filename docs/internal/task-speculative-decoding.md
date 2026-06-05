# Task (goinfer): speculative decoding — the 0.5B drafts for the 1.5B

> **For:** Claude Code, in `~/tmcode/goinfer`. **Post-launch** — do NOT block
> v0.1.3 on this. This is the one workload change that beats the batch=1 CPU
> ceiling for *single-stream* chat (per `perf-campaign.md`'s closing note), and
> it's distinctively on-brand because goinfer already ships both model sizes.
> Phased; **parity is the gate** (see Phase B).

## What it is (and why it's a pure speedup, not a quality trade)

A small **draft** model (Qwen2.5-Coder-**0.5B**) cheaply generates K candidate
tokens autoregressively. The large **target** model (Qwen2.5-Coder-**1.5B**) then
runs **one** forward pass over those K positions and checks them. Accepted tokens
are kept; the first rejection is replaced by the target's own token (free, from
the same pass). One expensive target pass yields *(accepted + 1)* tokens instead
of 1.

For **greedy** decoding the rule is trivial and the output is **token-identical to
plain target-greedy**: accept draft token *i* iff it equals the target's argmax at
position *i*; stop at the first mismatch and emit the target's argmax there. That
exactness is the whole point — build greedy first; it's the parity-clean version.

Expected win: ~**1.5–2×** on the **1.5B** (target), acceptance-rate dependent.
Rough model: speedup ≈ (accepted+1) / (1 + K·c), c = draft/target cost ≈ 0.33
(both bandwidth-bound, cost ∝ size). Code is predictable and both are Coder
models sharing a tokenizer, so acceptance should be high. **Measure it** (Phase C)
— ship only if it's a real win.

Hard precondition (already satisfied): draft and target **share the exact
tokenizer/vocab**. Qwen2.5-Coder 0.5B and 1.5B both use vocab 151936 — ✓. Assert
this at setup and refuse mismatched pairs.

## Phase A — decoder primitives (the enabling machinery)

Two capabilities the speculative loop needs that single-token decode doesn't:

1. **Multi-position forward returning per-position logits.** Verification is a
   prefill of K draft tokens that must return logits at **every** one of the K
   positions (today `forward` keeps only the last; prefill discards intermediates).
   Add a `forwardN(ids []int, cache) → [K][vocab]` (or expose per-position logits
   from the existing prefill path). Reuse the batched W8A8 path; this is the same
   matmuls with M=K instead of M=1, so it *also* amortizes fork/join better.

2. **KV-cache rollback.** Verification appends K positions to the target's cache,
   but only *(accepted+1)* are real — the rest must be dropped before the next
   round. Add `cache.TruncateTo(pos)`. (Confirm the cache's append is
   position-indexed so truncation is just resetting the length + Pos.)

Both are numerics-neutral; existing `TestDecodeParity` must stay green.

## Phase B — the speculative loop (greedy first; parity is the gate)

Add a generation path (e.g. `decoder.GenerateSpeculative(ctx, prompt, maxTok,
draft *Model, K int, sp)`), or a `Speculator` wrapping a draft+target pair:

1. Draft autoregressively generates K tokens (K cheap forwards on its own cache).
2. Target `forwardN` over those K tokens (one pass) → K position-logits.
3. Greedy accept: walk i=0..K-1, accept while `draftTok[i] == argmax(targetLogits[i])`;
   at the first mismatch emit `argmax(targetLogits[i])` and stop. If all K accept,
   also emit the target's argmax at position K (the bonus token).
4. `cache.TruncateTo` both caches to the accepted length; advance; repeat until
   EOS / maxTok / ctx-cancel.

**THE GATE — exactness:** add `TestSpeculativeGreedyParity` — greedy speculative
output must be **token-identical** to plain target greedy on a fixed prompt+seed,
for several K (e.g. 1, 4, 8) and several prompts. If it ever differs, it's a bug,
not a quality knob. (K=1 should degenerate to plain decode.)

Defer **sampled** speculative (temperature>0) to a follow-up — it needs the
rejection-sampling residual rule (accept w.p. min(1, p_t/p_d), else sample from
the normalized positive part of p_t−p_d) to preserve the target distribution.
Greedy is the launchable, clearly-correct version.

## Phase C — surface it + measure (mind the asset cap)

- **Library API + `--draft` flag.** `demo/chat` gains `--draft <gguf>`: load a
  second (draft) model and route generation through the speculative path. Print
  the **measured acceptance rate** and **tok/s vs plain target** to stderr.
- **Honest asset-cap reality:** the single-file *embedded* demo **cannot** bake in
  both models — 1.5B (~1.7 GB) + 0.5B (~0.5 GB) ≈ 2.2 GB **exceeds GitHub's
  2 GiB/asset cap**. So speculative ships as a **library feature** demoed via
  `--model <1.5B> --draft <0.5B>` (user supplies both), *not* as a new
  single-file release binary. Don't promise "both baked into one downloadable
  file." (If a combined binary is ever wanted, it needs split assets or an
  external host — out of scope.)
- **Measurement / ship gate:** on code prompts (the favorable case), report
  acceptance rate and end-to-end tok/s for plain-1.5B vs speculative-1.5B(draft
  0.5B). **Ship only if it's a clear win (say ≥1.3×)**; record the numbers (and a
  "no win, parked" outcome if it isn't) in `docs/perf-campaign.md`.

## Done

- [ ] Phase A: `forwardN` (per-position logits) + `cache.TruncateTo`; existing
      parity green.
- [ ] Phase B: greedy speculative loop; `TestSpeculativeGreedyParity` **token-
      identical** to plain target greedy across K∈{1,4,8} and several prompts.
- [ ] Phase C: `--draft` flag + library API; tokenizer/vocab match asserted;
      acceptance rate + speedup measured and recorded; asset-cap reality
      documented (no dual-embedded binary).
- [ ] `gofmt`/`vet`/`go test ./...` green; sampled speculative noted as a
      follow-up.
- [ ] Decision recorded: ship (≥1.3× on the 1.5B) or park, with numbers.
```
