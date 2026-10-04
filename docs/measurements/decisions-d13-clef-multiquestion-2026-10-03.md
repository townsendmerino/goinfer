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

**Run:** 2026-10-03, 21:17–21:34 PDT, nobara-pc (Ryzen 7 3700X, 8 threads), by day, box otherwise idle. `~/models/clef-flash` (rev `17f0b0ad…`, head sha256 `19cdcec8…`, module sha256 `0e304cf7…`), reference torch 2.14.0+cpu / transformers 5.16.1 (`data/clef_env_f32.json`). The records file has sha256 `0c62417f…c3b4` and the reference rows `data/probs_f32.jsonl`. The grader's output is `data/grade-2026-10-03.txt`; all rows, the reference log and the `go test -v` log are in `data/`. The code under test is the tree at `4bcc2afa`.

Cost against the estimate: reference 3.5 min (est. 7, with a 13 s load), int8int8 5.8 min (est. 5), f32 7.4 min (est. 8). Every record's token count and every question's id and option order matched the reference (0 structural problems); 5 records, 25 questions, 5,704 tokens.

| arm | max \|ΔP\| | mean KL | top-1 | verdict |
|---|---|---|---|---|
| goinfer f32 (the port check) | 1.22e-06 | < 1e-5 | 25/25 | **PASS** |
| goinfer int8int8 | 0.124 | 0.00554 | 24/25 (reported, no bar) | **PASS** |

- **f32:** the multi-question path in the Go port agrees with Cloudflare's code to about 1e-6 on real weights, at every position (largest on the `choice` questions, 1.22e-06 and 9.4e-07; the `noul` first question 6.8e-08).
- **int8int8, by position (mean KL, top-1):** 1 noul 0.00001, 5/5; 2 score 0.00340, 5/5; 3 choice 0.01096, 5/5; 4 noul 0.00057, 5/5; 5 choice 0.01278, 4/5. The worst position, 0.0128, is a fifth of the 0.06 line. The one flip is `mq-K256-s0` question 5 (a choice), where the reference's own top-two margin was 0.090: a near-tie.
- **Predictions, graded as written:** f32 PASS with max |ΔP| well under 1e-5: HELD (1.22e-06). int8int8 mean KL between 0.005 and 0.03, no position over 0.06: HELD (0.0055 is just inside the lower edge; the interval of 25 questions from 5 states is wide, so the lower edge is not evidence of anything).

**Reading.** The multi-question path is correct on the real weights, and int8int8 does not degrade when five questions share one pass: its mean KL here (0.0055) is under the single-question mean (0.0165). The `choice` questions carry the error again (as the `score` questions did in the single-question set), which fits the same cause (those probabilities are spread over several options), but nothing here tests that. So the owner's tag can say that the many-questions-in-one-pass path was checked against the reference on real weights, with the sample stated: **5 records of 5 questions each, three at K = 256 and two at K = 1,024, one reference and no gold.**

**Still not shown:** whether five questions asked in one record give the same answers as the same five asked one at a time (the reference's own joint-versus-single difference); accuracy against gold on multi-question records; K = 4,096; the GPU path; a question count other than five; images.

## 4. Joint versus single — a CHARACTERIZATION, pre-registered 2026-10-03 (before it ran; no pass/fail bar)

**The question** (named under "still not shown" above, and the owner's instruction on 2026-10-03): when five questions are asked in one record, do they get the same answers as the same five asked one at a time? Clef's head attends across the questions, so a difference is certain and is a property of the MODEL, not of the port. This section measures how large it is. **There is no bar and no verdict**: the readings are reported as numbers, and nothing here gates the tag (the owner's decision stands either way). What the numbers do is let the tag's page say how far joint answers sit from single ones.

- **Runs:** the 25 questions of section 1's five records, each as a record of its own (`scripts/pin_clef_mq.py records --single`: the same state and the same question object, ids `<record>::<question>`), through Cloudflare's reference at f32 (`model --dtype f32 --single`), compared with the joint reference run that already exists (`data/probs_f32.jsonl`). The reference is used on both sides, so numeric noise between runs is the f32 level seen before (about 1e-6 in probability); any difference far above that is the model's. goinfer is not run here: it matches the reference jointly to 1e-6 (section 3), and a goinfer int8int8 single-question arm would add quantization noise to a question about the model.
- **Estimated cost:** 15 single requests at K = 256 (~430 tokens each, D14) and 10 at K = 1,024 (~1,190), about 18,300 input tokens at the reference's ~37 ms per token (section 3: 839 tokens in 31 s): about 11 to 12 minutes plus a 13 s load. By the owner's instruction; it is above the usual ten-minute daytime line and he asked for it.
- **Reported (`grade_joint_single.py`):** per question, KL(single ‖ joint), max |ΔP| over the options, top-1 agreement; then the mean, median and max, by position in the joint record, and by K. Each top-1 flip is listed with the single run's top-two margin. Token counts are printed beside each pair (a single request has fewer tokens, and that is the point).
- **Prediction, written now, not a bar:** the mean KL(single ‖ joint) will be well under the int8int8 quantization error (0.0055 jointly, 0.0165 single-question in D13), top-1 agreement at least 23 of 25, and the differences concentrate on the `choice` questions (probability spread over several options). A prediction that fails is recorded as failed.
- **What it still does not show:** joint versus single under quantization (int8int8), on the device, at K = 4,096, with other question counts, or with different question orders in the record (the position effect here is confounded with the question type, since the order is fixed at noul, score, choice, noul, choice).

*Result: not yet run.*
