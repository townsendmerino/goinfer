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

**Not explained:** on D6a's own OOD sample goinfer's bare-v1 read 0.338, where the 150 fixture items (calibration split) put it at
about 0.48. The two samples differ in split and items, and the size of that difference (about 14 points, roughly four standard errors
on 400 noul rows) is more than sampling alone predicts. It does not change the decision (both are far below 0.8181) and is left open.

**Owed, by night:** the 150 items at `--quant q4k` on the CPU (about 4 hours), to split the 8-point gap between the Q4_K file and
goinfer's re-quantization. It matters for D1's Route A quality on 4-bit models, not for D6a's decision.
