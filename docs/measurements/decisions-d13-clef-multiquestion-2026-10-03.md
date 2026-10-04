# D13 addendum — Clef-flash on FIVE-question records, real weights: PRE-REGISTERED 2026-10-03 (before any graded run)

**Why.** D13's 150 records have one question each (`decisions-d13-clef-fidelity-2026-10-03.md`). Clef's point is that one backbone pass scores every question, and the part of the pipeline that does that (the head's self-attention over the questions, the per-question option split, the encoder's field numbering for several questions) has only ever met a SYNTHETIC golden (`decisions-d12-clef-encoder-2026-10-02.md`) and a tiny model. The owner tagged Clef-flash at `int8int8` on 2026-10-03 as his decision, not a gate pass; this is the check that the tag's page can fairly say "many questions in one pass". It is a fidelity check of a path, **not an estimate of a rate**: 5 records, 25 questions.

## 1. What runs

- **Records:** `scripts/pin_clef_mq.py records` — D7's frozen states (`decisions-d7-2026-09-28/prompts.json`, sha256 `2d56d87f…0dd9`; K = 256 states 0, 1, 2 and K = 1,024 states 0, 1), each asked the five questions D14 sent (`d7_bench.Q`/`FIVE`: noul, score, choice, noul, choice), by the same wording. 5 records, 25 questions, about 5,700 input tokens (D14 measured ~836 and ~1,595 tokens per record).
- **Reference:** Cloudflare's `joint_schema_model.py` (sha256-checked) on `~/models/clef-flash` at f32, `scripts/pin_clef_mq.py model --dtype f32`, per-question probabilities in request order.
- **Arms:** goinfer's `clef.Model.Decide` on the CPU, the backbone at `f32` and at `int8int8` (the default), through `modelload` (the serve path), `internal/clef/mq_fidelity_test.go`. Both are asked because int8int8 alone cannot tell a multi-question PORT defect from quantization noise: the f32 arm is the port check and the int8int8 arm is the question the tag cares about.
- **Cost, estimated before the run (by day, each step under 10 minutes):** reference about 5 min plus a load of about 2 min; goinfer int8int8 about 5 min at ~55 ms per token plus a 20 s load; goinfer f32 about 8 min at ~84 ms per token. Run one at a time; the box is idle (no night queue). The box's 62 GB holds the f32 reference (~38 GB) only if nothing else is loaded.
- **Not timed, not a speed measurement.**

## 2. The rule (written before the run; `decisions-d13-clef-multiquestion-2026-10/grade_mq.py` applies it)

Per question: max |ΔP| over its options, KL(reference ‖ arm), top-1 argmax agreement. Structural checks first: every record's token count equals the reference's, and every question's id and option order equal the reference's; **any structural mismatch is a FAIL of that arm.**

| arm | PASS | AMBIGUOUS (reported, goes to the owner, never a pass) | FAIL |
|---|---|---|---|
| goinfer f32 (the port check) | max \|ΔP\| ≤ 1e-4 over all 25 questions and top-1 25/25 | max \|ΔP\| ≤ 1e-3 and top-1 25/25 | otherwise |
| goinfer int8int8 | mean KL over the 25 ≤ 0.03 (D13's int4/int8int8 band) and no position's mean KL > 0.06 | mean KL ≤ 0.06 and no position's mean KL > 0.06, not PASS | otherwise |

- **Basis for the f32 bar:** D13's single-question f32 arm agreed to ~1e-7 in probability on 150 records. 1e-4 is a thousand times that and still far below what a real multi-question defect does (a wrong cross-question mask or a mis-split option range moves probabilities by 1e-2 or more). If the f32 arm is not a PASS, **the int8int8 arm is not read**: the port is suspect first.
- **Basis for the int8int8 bar:** the single-question int8int8 mean KL was 0.0165 (D13). The question is whether five questions in one pass degrade it; the registered band is the same 0.03. Position is checked (each position has only 5 questions) because the head's attention could make later questions worse.
- **Top-1 for int8int8 is REPORTED, with no bar:** on 25 questions one flip is 4 points, and the single-question rate was 0.907. Each flip is listed with the reference's top-two margin.
- **Predictions, written now:** goinfer f32 PASS with max |ΔP| well under 1e-5; int8int8 mean KL between 0.005 and 0.03, no position over 0.06. A prediction that fails is recorded as failed.
- **What this does NOT establish:** whether five questions asked in one record give the same answers as the same five asked one at a time (the reference's own single-versus-joint difference; not run); any accuracy against gold (these states have no gold); anything at K = 4,096; the GPU path.

## 3. Result

*Not yet run.*
