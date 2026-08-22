# Gemma-3 on Metal — localize the mid-channel sign-flip (within-layer bisect)

> **STATUS: DELIVERED — the mechanism was localized and the harness kept.** `metal/bisect_test.go`
> and `metal/gemma_sublayer_test.go` are the seams this prompt asked for, and the finding is
> written up in `docs/parity-hunt-playbook.md` (the sign-flipped channels amplified by the final
> norm). Gemma 3 is resident on Metal. Kept as the worked example the playbook cites.


> The Gemma-dense residual bug is localized to a *mechanism*, not yet to an *op*. This
> bisect names the op. **Gemma stays DORMANT** (declines to CPU) — nothing ships; this is
> diagnostic. Companions: `gemma3-metal-debug-report.md`, `gemma3-metal-debug-fable.md`,
> and the CUDA cross-check (dim 443).

## What's established — do NOT re-derive

- Metal-int4 vs CPU-int8 residual cosine = **0.818** (control 0.990). The split: **int4
  costs Gemma 0.99→0.92** (CPU-int4 vs CPU-int8 = 0.9217 — Gemma is int4-fragile, ~10× the
  control), and **Metal adds −0.104 on top** (Metal 0.818 vs CPU-int4's 0.922, both vs
  CPU-int8). The **−0.104 is the target**; it's Metal-compute, not quant — **CPU-int4 and
  CPU-int8 track each other (0.997); Metal alone diverges.**
- The **trunk is clean** through all 34 layers (all-dims cosine 0.981). The logits collapse
  in the **final-norm → LM-head** step, because the final norm's `(1+w) ≈ 16–20` amplifies a
  handful of **sign-flipped mid-magnitude channels** into the 262k head.
- **REFUTED — it is NOT the massive channel.** At the shared tap, dim **443** (|~70k|, the
  dominant massive-activation channel) is **fine on Metal** — tracks CPU ~8% low, same sign,
  no flip, exactly like CUDA (+0.95%). The earlier "int16 group-sum overflow on massive
  activations" hypothesis is **dead**.
- **The bug is sign-flipped SECONDARY channels.** At the tap, Metal flips **1698** (−536 vs
  CPU +83), **1723** (+371 vs −505), **227** (+369 vs −424). Mid-magnitude (|hundreds|),
  amplified by the final norm.
- **Reference + oracle.** **CPU-int4** is the correct reference (weights exonerated; it's
  the int4 quant-twin of Metal's W4A8). **CUDA-int4 is a proven-clean oracle** — nails 443 to
  0.95% and shows *no* sign-flips on the mid channels (its mid drift is generic symmetric
  noise). **The fix must make Metal match CUDA/CPU-int4 on 1698/1723/227.**

## The tap — verbatim (both harnesses must match)

Pre-final-norm residual, **position 5**, token-ids **[2 669 5279 529 7001 563]**, ordinary
token (**NOT BOS**). Always report **cosine AND ‖·‖ AND the raw signed value** on the
**fixed shared channel set {443, 19, 172, 1698, 1730, 2482, 1723, 227}**, plus **top-16 by
magnitude** and **top-16 by |Metal − CPU-int4|**, both labeled. (Never cosine alone — the
sink lesson; never all-dims cosine alone — it hid 6/2560 channels.)

## Fork 1 — which sublayer carries the flip (attention vs MLP)

The flip builds from ~layer 11, catastrophic ~22–24. Pick the **first layer where a target
channel (1698/1723/227) flips sign** vs CPU-int4. Within that layer, tap the two sublayer
**contributions separately, before the residual add**:
- after the **attention contribution** (o-proj output),
- after the **MLP contribution** (down output).

Compare **Metal-int4 vs CPU-int4** on the fixed channel set. Whichever contribution already
carries the sign-flip names the sublayer (bit-near on one, flipped on the other = clean).

## Fork 2 — where in that sublayer the flip originates (quant vs GEMV)

Mechanism hypothesis: the **443 outlier sets the per-vector int8 activation scale
(~70k/127 ≈ 558), crushing the mid channels to near-zero int8** — and at near-zero magnitude
a **±1 rounding difference *is* a sign-flip**. Discriminate at the input to the culprit GEMV:
- Compare **Metal's int8-quantized activation vs CPU's** (aikit `QuantizeRowsInt8`), channel
  by channel. **Differ by ±1 on the crushed channels ⇒ origin is the activation-quant
  rounding** (Metal `quant_vec`/`rmsnorm_quant` rounds near-zero differently than CPU).
- **Quantized activations match but the GEMV OUTPUT sign-flips ⇒ origin is the int4 GEMV
  kernel** (Metal UNP8 accumulation on near-zero sums — sign/order/narrow-int). CUDA's dp4a
  path + CPU `MatmulBTW4A8` are the correct-answer references.

## Discipline

- **Reference: Metal-int4 vs CPU-int4** (not int8). **Oracle: CUDA-int4's value on 1698** =
  the correct answer.
- Always **norm + cosine + raw signed value** on the fixed channel set.
- Ordinary token, **pos 5**, the exact ids above.
- Nothing ships — Gemma stays dormant. If you commit the harness, gate it with
  `requireDeviceAndFixture` (skips cleanly in CI, like the other device gates), and label it
  diagnostic (prints; it isn't a regression gate).

## When it lands — fix framing (don't fix blind)

- **Activation-quant rounding:** match Metal `quant_vec` rounding to CPU on near-zero, OR —
  the real Gemma fix — **outlier-aware / per-channel activation quant** so the 443 outlier
  doesn't crush the others (this is *why* Gemma is quant-hostile; prefer it if the crush is
  the root).
- **GEMV accumulation:** fix Metal UNP8 near-zero handling to match CUDA dp4a / CPU
  `MatmulBTW4A8`.
- **Verify:** re-run the tap — 1698/1723/227 must match CUDA-int4's sign and magnitude, and
  Metal-vs-CPU-int4 residual cosine must close toward the ~0.99 the control shows. Then the
  ship gate (dNLL < 0.3, clean free-run, Metal ≥ CPU-int4 − 0.03) decides declaration.
