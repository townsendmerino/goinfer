# D6a: is label scoring (Route A) enough, or do decision heads earn their build? (nobara-pc, run 2026-09-28, graded 2026-09-30)

**Status: GRADED 2026-09-30. Decision by the pre-registered rule: BUILD D2–D4. The control FAILED as registered; the investigation
below explains most of it, and the explanation strengthens the decision rather than weakening it.** The pre-registration and its two
amendments are in [`task-constrained-confidence.md`](../tasks/task-constrained-confidence.md) (D6a) and were committed before the run;
nothing here re-grades against them.

## The run

Night queue on nobara-pc, 2026-09-28 20:20–20:55 PDT (`d6a-amended-cuda`, rc=0; [`results/night-job.log`](decisions-d6a-2026-09-28/results/night-job.log)).
Pinned binary `goinfer-chat-cuda-a4e16c43` (the batched DeltaNet prefill), CUDA, `--ctx 4096`, [`run-d6a.sh`](decisions-d6a-2026-09-28/run-d6a.sh)
on the amended samples (600 calibration / 872 OOD eval rows, `select.py`). Validity: no arm's stderr shows a declined GPU path, and all
872 rows per arm passed validation. Graded with [`analyze.py`](decisions-d6a-2026-09-28/analyze.py), unchanged
([`results/analysis.txt`](decisions-d6a-2026-09-28/results/analysis.txt)).

| arm | model, template | raw top-1 | calibrated top-1 | calibrated ECE |
|---|---|---|---|---|
| **A** (graded) | Qwen3.5-9B Q4_K_M, chat-v1 | 0.4197 | 0.4197 | 0.1656 |
| **B** (control) | Qwen3.5-9B Q4_K_M, bare-v1 | 0.3378 | 0.3378 | 0.2199 |
| C (reported) | Qwen2.5-Coder 1.5B, chat-v1 | 0.2774 | 0.2774 | 0.2102 |

Top-1 is combined with the OOD split's kind proportions (noul 9,767 / choice 3,219 / score 72). The ECE floor at this sample is
0.0386, so the owner's bias-adjusted bar is sqrt(0.05² + 0.0386²) = 0.0632. Arm B's score-kind temperature hit the search bound
(49.9999), which the tool flags as "not a calibration".

**Decision (arm A, calibrated):** top-1 0.4197 against JEV-9B's published 0.9181, below the 0.8181 build line, and ECE 0.1656 against
the 0.0632 bar → **BUILD D2–D4**.

**Control (arm B, raw):** 0.3378 against the authors' B0 0.5180, outside the registered 0.05 → **does not reproduce**, and by the
rule arm A is not trusted until that is explained.

## Explaining the control: same bytes, different numbers

The D0 fixture holds 150 items with transformers' own Route A B0 run on them, in f32 (`testdata/decisions/route_a_b0.jsonl`), and
JEV-9B's trained head on the same items (`jev9b_ref_f32.jsonl`, `jev9b_ref_bf16.jsonl`), all from `scripts/pin_decisions_d0.py`. They
were untracked on nobara until this record; they are committed with it. goinfer was compared on identical items, 2026-09-30 on nobara-pc.

**1. The inputs are identical.** [`b0cmp.go.txt`](decisions-d6a-2026-09-28/b0cmp.go.txt) renders every item's bare-v1 prompt through
goinfer's `decide.Render` and tokenizes it with the GGUF's tokenizer: prompt text identical **150/150**, token ids identical **150/150**,
and the label token ids match the fixture's `verbalizer_ids` (noul and score by id; choice by its letters A, B, C…). So the template,
the tokenization and the readout position are not the cause.

**2. The distributions are not.** goinfer's `decide` (same binary, bare-v1, int4, CUDA) on the 150 items against transformers' f32
([`b0dist.py`](decisions-d6a-2026-09-28/b0dist.py), [`results/b0cmp-goinfer-bare.jsonl`](decisions-d6a-2026-09-28/results/b0cmp-goinfer-bare.jsonl)):

| kind | n | argmax agreement with transformers | mean total-variation distance |
|---|---|---|---|
| noul | 51 | 0.765 | 0.100 |
| choice | 50 | 0.520 | 0.194 |
| score | 49 | 0.510 | 0.144 |
| all | 150 | 0.600 | 0.146 |

With the inputs proven equal, that is numeric: the Q4_K_M file's 4-bit weights, plus goinfer's re-quantization of them to its own
int4 layout with 8-bit activations (the CUDA default for this model). goinfer's f32 forward matches HF on this family (the parity
manifest), so the gap is quantization, not the forward. **How it splits between the file and goinfer's re-quantization is not
measured:** `--quant q4k` would isolate it, but on CUDA it declines for this model ("per-32 activations: the Gated DeltaNet path is
not implemented") and runs on the CPU at about 6 items in 10 minutes. That is a night job.

**3. Against gold, on the same 150 items** ([`b0ref.py`](decisions-d6a-2026-09-28/b0ref.py); OOD-weighted like `analyze.py`):

| computation | noul | choice | score | OOD-weighted top-1 |
|---|---|---|---|---|
| JEV-9B trained head (transformers, f32) | 0.980 | 0.800 | 0.653 | **0.934** (published 0.918) |
| Route A bare-v1 (transformers, f32) = B0 | 0.608 | 0.420 | 0.102 | **0.559** (published 0.518) |
| Route A bare-v1 (goinfer, Q4_K_M int4) | 0.529 | 0.340 | 0.184 | **0.481** |

The fixture reproduces both published figures, so it is a sound reference. On identical items, **label scoring is about 37 points
behind the trained head in the reference implementation itself**, and goinfer's port is about 8 points behind the reference.

## What that does to the verdict

The control asked "does goinfer's Route A reproduce theirs?" The answer is: the inputs do, exactly, and the numbers are about 8 points
lower on the same items, from quantization. That does not rescue Route A. A perfect port would read about 0.56, still far below the
0.8181 line, and the trained head reads 0.93 on the same items. **So BUILD D2–D4 holds whatever the numeric gap turns out to be.**

**The 14-point gap between the two samples (explained 2026-09-30).** On D6a's own OOD sample goinfer's bare-v1 read 0.338, where
the 150 fixture items (100 from the calibration split, 50 from OOD) put it at 0.481. [`gap.py`](decisions-d6a-2026-09-28/gap.py)
([`results/gap.txt`](decisions-d6a-2026-09-28/results/gap.txt)) reads only committed files and finds three things:

1. **All of it is noul.** noul falls from 0.529 to 0.340, while choice (0.340 → 0.335) and score (0.184 → 0.167) hold. noul carries
   75% of the OOD weight, so its 19 points are the whole 14.
2. **The noul items differ, and goinfer leans "true".** goinfer answers "true" on 85% of the OOD sample's noul items, where gold is
   true on only 21.5% (the fixture: 72.5% against 37.3%). Two environments that the fixture barely samples make up 65% of the OOD
   noul items, and gold is mostly "false" in both:

   | environment | share of OOD noul | gold true | goinfer says true |
   |---|---|---|---|
   | painting-geometry | 39% | 33% | 100% |
   | snake | 26% | 8% | 88% |

   The fixture's largest noul group, 17 `v3_*` items on which goinfer reads 0.65, does not occur in the OOD sample at all.
   Discrimination is not what changed: noul AUROC is 0.695 on the OOD sample against 0.719 on the fixture.
   The fixture shows the same split effect inside itself: its 34 calibration noul items read 0.559 and its 17 OOD ones 0.471.
3. **The quantization shift is largest exactly there.** On the fixture's own noul items, goinfer's P(true) sits above the transformers
   f32 reference's by:

   | environment | P(true) shift |
   |---|---|
   | `v3_*` | −0.002 |
   | workflow-controls | +0.042 |
   | painting-geometry | +0.074 |
   | snake | +0.143 |

   On snake the reference says "true" on 3 of the 7 items and goinfer on all 7. These are near-margin decisions: the median
   |P(true) − 0.5| on the OOD sample is 0.17.

**What that makes the gap.** It is the same ~8-point quantization shift measured above, falling hardest on the environments the OOD
split is made of. It is not a second defect. The size of that share is an **estimate**:

- **The counterfactual.** Subtracting each environment's measured shift from goinfer's OOD probabilities lifts noul from 0.340 to
  0.492, and the OOD-weighted top-1 from 0.338 to 0.451. That is about 11 of the 14 points.
- **Why it is only an estimate.** The shifts come from 7–17 items per environment, and choice and score are left as measured.
- **The rest** is item-level difference that samples this small cannot resolve.
- **The check that would settle it** is the transformers reference on D6a's 400 OOD noul rows. Queued on nobara 2026-09-30 as
  `d6a-b0-ood-noul` ([`run-b0-ood-noul.sh`](decisions-d6a-2026-09-28/run-b0-ood-noul.sh) at 3e2f9ea3, `pin_decisions_d0.py b0rows`).
  A two-row probe matched goinfer's token counts exactly (516 and 124).

**The settling check, run 2026-09-30, graded 2026-10-01: the gap is goinfer's, not the sample's, and larger than
estimated.** `d6a-b0-ood-noul` ran the transformers f32 reference (Route A, bare-v1) on the same 400 OOD noul rows, in
73 minutes on nobara-pc ([`results/b0-ood-noul-f32.jsonl`](decisions-d6a-2026-09-28/results/b0-ood-noul-f32.jsonl)).
Every prompt's token count matches goinfer's.

| on the 400 OOD noul rows | top-1 against gold | answers "true" |
|---|---|---|
| gold | | 21.5% |
| transformers f32 (the reference) | **0.580** | 53.5% |
| goinfer, arm B (bare-v1, raw) | **0.340** | 84.5% |

- **The sample explains little.** The reference reads 0.580 here against 0.608 on the fixture's noul items, so the
  harder environments cost it under 3 points.
- **goinfer against the reference, on identical prompts, is 24 points on noul.** The two agree on 65% of items. By
  environment, goinfer's P(true) sits above the reference's by +0.147 (painting-geometry, n=156), +0.069
  (workflow-controls, 121) and +0.132 (snake, 103). These are the fixture's measured shifts (+0.074, +0.042, +0.143), at
  scale, and on near-margin items they flip far more decisions than the counterfactual above assumed. Its "about 11 of
  the 14 points" understated goinfer's share.
- **This is not yet a quantization number.** Arm B ran CUDA-resident at the default int4 (`run-d6a.sh`, `--backend
  cuda`). D6b (`decisions-d6b-2026-09/results.md`) then found that this path deviates from the f32 reference more than
  CPU int4 does: on Route B, KL 0.038 against 0.030, with one confident flip that the CPU arms do not make. Which part of
  these 24 points is int4 and which is the CUDA path is not separated. Route A at CPU int4 and CPU int8int8 on these 400
  rows would separate them, and that is night work.
- **D6a's decision is unaffected.** BUILD D2–D4 rested on the trained head beating Route A, which this widens.

**Two consequences for Route A (D1), neither of which touches D6a's decision:**

- **A constant beats every arm on noul here.** Answering "false" every time scores 0.785 on this sample, above every arm and above
  the fixture reference's 0.608.
- **The calibration cannot correct a lean.** It fits a temperature only, and a temperature never changes which label wins. A per-kind
  bias term (a prior correction, fitted on the calibration split) would change the argmax. Whether it beats the trained head's 0.98
  on noul is a separate question, and it is not run here.

**Measured 2026-10-02 (night job `d6a-q4k-150`; registered below, result in "The owed q4k measurement: RESULT"):** the 150 items at `--quant q4k` on the CPU, to split the 8-point gap between the Q4_K file and
goinfer's re-quantization. It matters for D1's Route A quality on 4-bit models, not for D6a's decision.

## The owed q4k measurement, registered 2026-10-02 before it ran

**Question.** The gap between the transformers f32 Route A reference (0.559 OOD-weighted top-1 on the 150 fixture items) and goinfer's Q4_K_M int4 on CUDA (0.481) is about 8 points. How much of it is the Q4_K FILE's own 4-bit loss, and how much is goinfer's
re-quantization of that file to its int4 layout? `--quant q4k` computes the file's native 4-bit blocks without re-quantizing them, so it sits between the two. **A record, not a gate:** nothing here changes D6a's verdict (BUILD D2-D4 holds whatever the numeric gap is, as above).

**What runs.** `run-d6a-q4k.sh`: the pinned `goinfer-chat` `goinfer-chat-cuda-a4e16c43` (sha256 `c9974a122c8755fca901f53b2deec0fdb8b7fe85f8ec1f666c938b2105d4ee5d`), `decide --backend cpu --quant q4k --template bare-v1 --ctx 4096`, the 150 items
(`testdata/decisions/items.jsonl`, sha256 `2a5f37e57d51a0b0139b70381c98ce19053aff0a8f0935f50a9452f1382da7fc`, pinned by copy), Qwen3.5-9B Q4_K_M from `~/models`, raw label scoring (no head, no calibration), as the int4 arm's `results/b0cmp-goinfer-bare.jsonl` was
produced. The CUDA resident declines this combination ("per-32 activations: the Gated DeltaNet path is not implemented"), so it runs on the CPU. A first attempt on 2026-09-30 produced 6 rows in 10 minutes and was stopped (kept aside as `*.PARTIAL-6rows-2026-09-30.*`; not mixed in).
**Estimate: 2.25 hours** = 27,861 prompt tokens x the first attempt's 290 ms per token (rows 1-6 averaged 341 tokens against 186 for all 150, which is why the earlier "4 hours" was an over-estimate); queued at 150 minutes.

**What is read (primary, paired, per item).** Per-item argmax agreement with the transformers f32 reference and mean total-variation distance, overall and per kind, for the q4k arm beside the int4 arm already on record (`b0dist.py`): int4 CUDA reads agreement **0.600** overall (noul 0.765, choice 0.520, score 0.510), TV **0.146**.
**Secondary:** top-1 against gold, OOD-weighted as in the table above (per-kind noul 9,767 / choice 3,219 / score 72), for q4k against 0.481 and 0.559.
**How to read it.** If q4k's agreement with the reference is clearly above int4's, a large part of the gap is goinfer's re-quantization; if it is about equal, the file's own 4-bit loss accounts for it. With 51 / 50 / 49 items per kind, a per-kind difference smaller than about 10 points is inside sampling noise, so the claim is stated only to that resolution, and the gold-top-1 split
(dominated by noul at n = 51) is quoted with its interval, not as a point.

**Confound, stated before the run.** q4k runs on the CPU and the int4 arm on CUDA, so the comparison also changes the backend. The control is Route A at CPU int4 over the same items (about another 2.25 hours); it is NOT queued, and a split that the CPU/CUDA difference could explain is reported as unresolved.

## The owed q4k measurement: RESULT (run 2026-10-02 18:16–20:13 PDT on nobara-pc, 1 h 56 min; read 2026-10-03)

Run exactly as registered above: the pinned `goinfer-chat-cuda-a4e16c43`, `decide --backend cpu --quant q4k --template bare-v1 --ctx 4096`, the 150 pinned items, Qwen3.5-9B Q4_K_M from `~/models`. 150 rows, 0 failed validation, in 1 h 56 min (the registered estimate was 2.25 h). Rows and stderr are [`results/b0cmp-goinfer-bare-q4k.jsonl`](decisions-d6a-2026-09-28/results/b0cmp-goinfer-bare-q4k.jsonl) and `.stderr.txt`; the comparison is [`q4k_compare.py`](decisions-d6a-2026-09-28/q4k_compare.py) with its output in [`results/q4k_compare.txt`](decisions-d6a-2026-09-28/results/q4k_compare.txt). The stderr records a swap-used baseline of 12.67 GB on the box at the start; this is not a timing measurement, so it does not touch the numbers read here.

**Primary (per-item argmax agreement with the transformers f32 reference, paired on the same 150 items):**

| kind | n | q4k (CPU) agrees | int4 (CUDA) agrees | q4k-only / int4-only discordant | exact sign test | mean TV, q4k vs int4 |
|---|---|---|---|---|---|---|
| noul | 51 | 0.804 | 0.765 | 6 / 4 | p = 0.75 | 0.063 vs 0.100 |
| choice | 50 | **0.800** | **0.520** | 16 / 2 | **p = 0.0013** | 0.084 vs 0.194 |
| score | 49 | 0.612 | 0.510 | 10 / 5 | p = 0.30 | 0.077 vs 0.144 |
| all | 150 | **0.740** (Wilson 95% 0.66 to 0.80) | **0.600** | 32 / 11 | **p = 0.0019** | 0.074 vs 0.146 |

**Secondary (top-1 against gold, OOD-weighted noul 9,767 / choice 3,219 / score 72):** transformers f32 **0.559**, goinfer q4k **0.549**, goinfer int4 CUDA **0.481**. q4k is within 1 point of the reference and int4 8 points under it. The weighted figure is dominated by the noul cell (31 of 51 for f32 and for q4k, Wilson 95% 0.47 to 0.73; int4 27 of 51, 0.40 to 0.66), so per the registration it is quoted with that interval: the f32/q4k/int4 gold figures are NOT separable at this sample size, and the claim rests on the agreement result above.

**Reading, by the registered rule.** q4k's agreement with the reference is clearly above int4's (0.74 against 0.60; 32 items where only q4k agrees against 11 where only int4 does), so a large part of the 8-point gap is not the Q4_K file's own 4-bit loss. The effect is carried by the choice kind (0.80 against 0.52, 16 against 2), where it is well beyond the registered 10-point resolution; noul (4 points) and score (10 points, p = 0.30) are inside the noise the registration named.

**What this does NOT show, as registered before the run.** The registration named the confound: q4k runs on the CPU and the int4 arm on CUDA, so the comparison also changes the backend, and "a split that the CPU/CUDA difference could explain is reported as unresolved". It is unresolved. The result is consistent with goinfer's int4 re-quantization costing most of the gap, and equally with something specific to the CUDA int4 path (its activation quantization, which a CPU int4 run would not share). Only the control separates them: Route A at CPU int4 over the same 150 items (about another 2 hours; NOT queued). Until then the defensible statement is "the native-block CPU path reproduces the reference's agreement and gold top-1 to within sampling noise; goinfer's CUDA int4 path does not, mostly on choice questions", and nothing about WHERE in the int4 path the loss enters.

**What it changes.** Nothing in D6a's verdict (BUILD D2-D4 holds, as registered). It bears on D1's Route A on 4-bit models: Route A through the CUDA int4 path reads about 8 gold points under the reference, and the q4k path does not. D13's quantization arms for Clef-flash include q4k and int4, so the same question will be asked again there with the backend held fixed by design.
