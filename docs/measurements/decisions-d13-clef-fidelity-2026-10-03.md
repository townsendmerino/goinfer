# D13 — Clef-flash fidelity: goinfer's Clef pipeline against the reference, PRE-REGISTERED 2026-10-03 (before any graded run)

The question (D13 of `docs/tasks/task-constrained-confidence.md`, Route C): on the real Clef-flash weights, how closely does goinfer's pipeline (record encoder, one backbone pass, joint head; `internal/clef`) reproduce Cloudflare's own code, at each backbone precision? The method is D6b's (`decisions-d6b-2026-09/`), applied to Clef.

Nothing in this record has a graded result yet. Section 7 is filled in after the run, below a line that says so.

## 1. What runs

- **Reference.** `scripts/pin_clef_d10.py`'s `probs_f32.jsonl`: Cloudflare's `joint_schema_model.py` (sha256-checked) on `Cloudflare/clef-flash` at f32, over the 150 D10 records (`testdata/decisions/clef/records.jsonl`, sha256 `5942ccb997012d69c37945ef8cf3a2dd1974f62009ef14d9a668f5d4c49f9b7c`; 150 prompts, 44,823 tokens). Produced by the night job `d10-clef-f32` (re-queued 2026-10-03 after its first attempt died on a script bug, `7a38ba53`). **If that job does not finish, the arms below are still written but nothing is graded: the run script refuses to start an arm without the 150 reference rows.**
- **Arms.** goinfer's `clef.Model.Decide` on the CPU backend, the backbone loaded at `f32`, `int8int8` or `int4` through `internal/modelload` (the path `goinfer-serve` uses), the head always f32. One row per record (`TestFidelityArm_run`, `internal/clef/fidelity_test.go`); the prompt token count of every row is checked against the encoder dump (a template or tokenizer drift would grade a different prompt).
- **Checkpoint.** `~/models/clef-flash`, copied from the archive on 2026-10-03 and verified byte for byte against it (`rsync -c`, 0 differences). Never read from `/srv/models`.
- **Sets.** All 150 records for every arm.

## 2. The rule (registered before any graded run; `decisions-d13-clef-fidelity-2026-10/grade.py` applies it)

Per arm: mean KL(reference || arm), top-1 argmax agreement with the reference, top-label ECE against gold on the 84 gold records (15 equal-width bins) with a paired bootstrap of (arm ECE − reference ECE) (2000 resamples, seed 0).

| arm | PASS | AMBIGUOUS (reported, never a pass; goes to the owner) | FAIL |
|---|---|---|---|
| f32 | KL ≤ 0.01 and top-1 ≥ 0.98 | KL ≤ 0.02 and top-1 ≥ 0.95, not PASS | otherwise |
| int8int8, int4 | KL ≤ 0.03 and top-1 ≥ 0.98 | KL ≤ 0.06 and top-1 ≥ 0.95, not PASS | otherwise |

The bands are the task doc's (mean KL ≤ 0.01 at f32, ≤ 0.03 at int4, top-1 ≥ 98%) and D6b's (int8int8 graded on int4's band). The ambiguous band is new here and is the zone below the threshold where motivated reading lives: an arm in it is not called a pass.

- **Calibration** (D6b's amendment): an arm FAILS calibration if the bootstrap 95% interval of (arm ECE − reference ECE) lies wholly above 0; an interval that reaches 0 is UNRESOLVED, not failed.
- **If the f32 arm is not a PASS, the port is suspect before any quantization finding is read.** The goinfer f32 arm differs from the reference only by summation order and by the encoder and head being ported, so a miss there is a bug or a drift to find, not a result.
- **The default quantization** for decision models is the owner's decision already made, `int8int8` (D6b). This run does not reopen it unless int8int8 FAILS calibration or is outside its KL band; a top-1 miss alone leaves the default standing with the disagreement stated, which is D6b's precedent. Anything else goes to the owner.
- **Context, reported beside the arms:** the reference at bf16 (the release's own dtype, `d10-clef-bf16`, with f32 GEMMs rounded to bf16) against the f32 reference on the same metrics: the numeric noise a correct port is held against.

## 3. Clef against JEV-9B (the task doc's comparison)

On the 84 gold records, both at f32 reference (no goinfer arm involved, nothing extra to run): top-1 accuracy against gold and ECE, Clef (`probs_f32.jsonl`) against JEV-9B (`testdata/decisions/jev9b_ref_f32.jsonl`, D6b's reference), with a paired bootstrap of each difference. **Rule, from the task doc:** "if Clef-flash is at least level on both, JEV's route is kept but not extended". Registered reading of "at least level on both": point estimates, accuracy Clef ≥ JEV and ECE Clef ≤ JEV. With 84 rows the intervals will probably reach 0; if both do, the verdict is stated as a point-estimate reading and UNRESOLVED, and the decision on JEV's route goes to the owner with the numbers. It is not auto-resolved either way.

## 4. Predictions, written before the run

Basis: D6b's JEV-9B results on the same 150 items (f32 exact; int8int8 KL 0.0088, top-1 0.927; int4 KL 0.0298, top-1 0.940; calibration no worse in any arm), and the tiny end-to-end fixture, where the whole pipeline matched the official reference to 1.8e-7 (`decisions-d12-clef-encoder-2026-10-02.md`).

- **f32: PASS**, with KL well under 0.001. If it is not, that is a finding about the port.
- **int8int8 and int4: mean KL inside their 0.03 band, top-1 below 0.98.** The expected grade is FAIL or AMBIGUOUS on top-1, as for JEV. The Clef head is f32 and reads the backbone's hidden states at every position, which may make it more or less sensitive to the backbone's quantization than JEV's last-token head; I do not predict which.
- **Calibration: not failed in any arm** (intervals reaching 0, as for JEV).
- **Clef against JEV:** no prediction on the direction. Clef is a newer, larger-trained model with a joint head; the 84-row sample cannot resolve a few points.

A prediction that fails is recorded as failed, not reread.

## 5. Deviations from the task doc, disclosed

- **The q4k arm is not run tonight.** `--quant q4k` computes a GGUF's native Q4_K blocks, and there is no Clef GGUF here: the only ones are third parties' on Hugging Face, and a download would be both an external fetch and a different artifact from this checkpoint. It is owed. Making the same choice as D6a made for Qwen3.5-9B (a Q4_K_M file) needs the owner to say whether to fetch one.
- **Clef 27B on the CPU is not part of this run** (the task doc's last D13 bullet). It needs the 27B checkpoint (not on this box) and a smaller reference set; it is registered separately if wanted.
- **Not a speed measurement.** The throughput the harness logs is for planning only; D14 owns speed.

## 6. Cost, from an exploratory smoke (by day, one run, 2 records, labelled exploratory and never quoted as a result)

2026-10-03, nobara, 2 records (827 tokens), CPU: int4 loaded in 19 s and ran at about 68 ms per token; int8int8 loaded in 14 s and ran at about 54 ms per token. Both rows had token counts equal to the encoder dump (589 and 238), probabilities summing to 1, and the same top option as gold; the two arms differed by 0.01 to 0.025 in P(true). So the first run on the real weights exercised the whole serve load path, the real tokenizer, the 32-layer backbone and the real head without error. f32 was not smoked (36 GB of RAM by day).

| job | tokens | estimate | basis |
|---|---|---|---|
| `d13-clef-f32` | 44,823 | 90 min | no measurement: assumed 1.5x int8int8's rate (D6b's JEV ran f32 about 1.3x slower per item); timeout 3 h |
| `d13-clef-int8int8` | 44,823 | 45 min | 54 ms per token + load |
| `d13-clef-int4` | 44,823 | 60 min | 68 ms per token + load |

Queued after the two D10 jobs (55 + 65 min), about 5 h 15 min of estimates in all. The runner does not start a job whose estimate would end after 06:30. Each job is resumable (rows already written are skipped), so a timeout loses nothing.

## 7. Result

*Not yet run.* Filled in after the night run, below this line, with the grader's own output.
