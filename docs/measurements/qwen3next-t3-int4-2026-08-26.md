# qwen3_next T3: the reference method works; the int4 result misses an int8 bar

**2026-08-26 · `linux` (nobara-pc) · goinfer `3b7facd`** · work order `docs/prompts/qwen3next-t3-validation.md` (`b6335bf`)

## What was measured

| | |
|---|---|
| checkpoint | `Qwen/Qwen3-Next-80B-A3B-Instruct`, 41 shards / 162.6 GB bf16, in `~/models` (NOT `/srv/models`) |
| reference | HF bf16, **full model**, via `accelerate` disk offload — placement **11 layers CPU / 41 disk** |
| goinfer | same checkpoint at **int4 weights, f32 activations** |
| machine | 62 GB RAM, loadavg 0.39 at start, 400 GB free disk, driver 595.91.07 / Nobara 44 |
| prompt | `"The capital of France is"` → ids `[785, 6722, 315, 9625, 374]` |

    argmax        got=12095  want=12095                    EXACT
    continuation  [12095 13 576 6722 315 9856]             EXACT, all 6 tokens
                  -> " Paris. The capital of Germany"
    logit cosine  0.989876   against the gate's 0.99       FAIL by 0.000124

## The recorded blocker was about co-residency, and co-residency was not required

G5 carried this since 2026-08-17: *"no full reference forward of a 163GB bf16 model fits 62GB."*
True of a **resident** reference and only of that. The reference and goinfer never have to be alive
at the same instant — pin the bf16 logits to a file in one process, read them back in another.
That is already how every tiny golden here is made; the earlier slice route generalised the *slice*
when what needed generalising was the *pinning*.

**It works, and the cost is now known rather than guessed:**

| phase | time |
|---|---|
| load + place 759 tensors (127 GB spilled to disk) | 1179 s |
| prompt forward | 70 s |
| continuation steps 1–2 | 6 s, 12 s |
| continuation steps 3–6 | 73, 81, 83, 88 s |
| **stage total** | **26 min** |

The load dominates and is paid once. Steps 1–2 being cheap and then jumping to ~80 s is the offload
cache warming and then thrashing as the sequence grows — so for anyone reusing this, **`N_NEW` is
the expensive knob, not prompt length**: six steps cost 343 s against the prompt's 70.

**This method is not specific to this family.** Any family whose reference will not co-reside can
now have a real T3 instead of a slice.

## Why this is quantization and not a defect — measured, not asserted

The pre-registered rule (G5, fixed before the number existed) required the cause be established
from evidence rather than argued. It is:

- The family's **slice oracle loads the same real checkpoint at f32** (`Options{}`, *"this is the
  NUMERIC gate, no quantization"*) and gets **cosine 1.00000000** against a 0.9999 bar. Same loader,
  same real tensor layout, same fused `in_proj_qkvz`/`in_proj_ba` split. The loader and forward are
  exact on these weights.
- The full-model run differs from that by exactly two things: **depth** (48 layers vs 4) and
  **int4**.
- **Every discrete check stayed exact** — argmax, and six greedy steps. That is what quant noise
  looks like. A loader or forward defect does not survive six argmax decisions intact.

This is also what was *predicted* before the run, from the config: 10 of 512 experts is 1.95%
active, sparsest in the table by 2×, and `nemotron3nano` at 6/128 already measured 0.978 with int8
activations against 0.9977 with f32 ones.

## Why it is NOT recorded as validated

The gate's threshold is at the assertion site in `decoder/real_oracle_test.go`:

    if cos < 0.99 { // int8 W8A8 vs bf16 — same bar as the deepseek real gates

**The bar is documented as an int8 bar, and this is the repo's first int4 T3.** goinfer's precision
here is forced by capacity, not chosen: 80B at int8 is ~80 GB against 62 GB of RAM, so there is no
int8 run to fall back to on this box. The run falls outside the population the threshold was
calibrated on.

That is a reason to *decide* an int4 bar deliberately — not a reason to read 0.989876 as ≥ 0.99.
The pre-registered rule said ≥0.98 passes; the gate says 0.99; **quoting the looser of the two after
seeing the number is precisely what pre-registration exists to prevent**, so the stricter committed
standard governs and the row stays `experimental`. Filed as its own queue item.

## The manifest row was NOT hand-written, on purpose

`emitParityRow` is a no-op when the gate fails, so `EMIT_MANIFEST=1` will not record this run — and
the work order's rule is that rows are written **by measurement, never by hand**. Hand-editing the
measured numbers in would satisfy the letter of "these were measured" while breaking the mechanism
that makes the manifest trustworthy. The row therefore still reads `tiny-golden`, which is
conservative and true; when the int4 bar is set, the gate writes the row itself.

**A side effect worth naming:** a gate that fails cannot record its own result, so an informative
near-miss leaves no manifest trace at all. That is right for a failing gate and wrong for a
*calibration* miss, and it is part of the bar question rather than a separate one.

## Artifacts

`qwen3next-t3-reference-pin_2026-08-26.log` (progress bars stripped) ·
`qwen3next-t3-oracle-int4_2026-08-26_FAIL-0.9899.log` · `qwen3next-t3-run-status_2026-08-26.txt` ·
golden at `testdata/qwen3next_real_golden.json` (1.6 MB, committed; weights are not).
