# D13 — Clef-flash fidelity: goinfer's Clef pipeline against the reference, PRE-REGISTERED 2026-10-03 (before any graded run)

The question (D13 of `docs/tasks/task-constrained-confidence.md`, Route C): on the real Clef-flash weights, how closely does goinfer's pipeline (record encoder, one backbone pass, joint head; `internal/clef`) reproduce Cloudflare's own code, at each backbone precision? The method is D6b's (`decisions-d6b-2026-09/`), applied to Clef.

Section 7 is the graded result (run 2026-10-03, nobara-pc; filled in the same evening from the grader's own output).

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

## 3. Clef against JEV-9B (informational; the owner decided 2026-10-03 to keep and extend BOTH routes equally)

On the 84 gold records, both at f32 reference (no goinfer arm involved, nothing extra to run): top-1 accuracy against gold and ECE, Clef (`probs_f32.jsonl`) against JEV-9B (`testdata/decisions/jev9b_ref_f32.jsonl`, D6b's reference), with a paired bootstrap of each difference. **There is no decision rule attached to it.** The task doc had one ("if Clef-flash is at least level on both, JEV's route is kept but not extended"); the owner replaced it on 2026-10-03 with "both equally": JEV's route and the Clef route are both kept serving and both get further work, so the comparison informs how to describe each route to a user (the docs say which is more accurate, on which sample, with the interval) and does not choose between them. With 84 rows the intervals will probably reach 0, and the record says so when they do.

## 4. Predictions, written before the run

Basis: D6b's JEV-9B results on the same 150 items (f32 exact; int8int8 KL 0.0088, top-1 0.927; int4 KL 0.0298, top-1 0.940; calibration no worse in any arm), and the tiny end-to-end fixture, where the whole pipeline matched the official reference to 1.8e-7 (`decisions-d12-clef-encoder-2026-10-02.md`).

- **f32: PASS**, with KL well under 0.001. If it is not, that is a finding about the port.
- **int8int8 and int4: mean KL inside their 0.03 band, top-1 below 0.98.** The expected grade is FAIL or AMBIGUOUS on top-1, as for JEV. The Clef head is f32 and reads the backbone's hidden states at every position, which may make it more or less sensitive to the backbone's quantization than JEV's last-token head; I do not predict which.
- **Calibration: not failed in any arm** (intervals reaching 0, as for JEV).
- **Clef against JEV:** no prediction on the direction. Clef is a newer, larger-trained model with a joint head; the 84-row sample cannot resolve a few points.

A prediction that fails is recorded as failed, not reread.

## 5. Deviations from the task doc, disclosed

- **The q4k arm was first left out, then added by the amendment below (2026-10-03, before any graded run).** It needs a Clef GGUF, and the only ones are third parties'; the owner said to fetch one.
- **Clef 27B on the CPU is not part of this run** (the task doc's last D13 bullet). It needs the 27B checkpoint (not on this box) and a smaller reference set; it is registered separately if wanted.
- **Not a speed measurement.** The throughput the harness logs is for planning only; D14 owns speed.

## 5a. Amendment (2026-10-03, before any graded run): the q4k arm

The owner asked for the q4k arm the task doc names, so a third-party GGUF was fetched. **What it is, exactly:** `bartowski/Cloudflare_clef-flash-GGUF`, revision `d7f376ea88c05e7bb1014dd5351a93df9dd8029e`, file `Cloudflare_clef-flash-Q4_K_M.gguf` (5,841,052,992 bytes, sha256 `45f803cbcb6144784653bc31cde957e0d184d963a5198d423dc589a79e178d45`, checked against the sha256 the hub publishes), quantized by bartowski with llama.cpp release b11279 **using an importance matrix** from Cloudflare/clef-flash. It is the backbone only (no head), in `~/models/clef-flash-gguf/`.

- **Not the same kind of arm as the other three.** f32, int8int8 and int4 are goinfer's own loads of the checkpoint; q4k is llama.cpp's imatrix-guided Q4_K_M of the same weights, run by goinfer's `--quant q4k` CPU path (native 4-bit blocks, no re-quantization, as D6a ran Qwen3.5-9B). It measures that artifact, not goinfer's quantizer, and an imatrix quantization is calibrated on text the others never saw. The head is the checkpoint's own f32 head; the lm_head rows it reads are the GGUF's, as loaded.
- **Set:** all 150 records. (This amendment first registered every 2nd record, 75 of 150, on the strength of D6a's CPU rate for Qwen3.5-9B Q4_K_M, about 250 ms per token, which would have put the full set at 3 to 4 hours. A smoke on this GGUF the same day, exploratory, 2 records and 827 tokens, ran at about 68 ms per token and loaded in 8 seconds, so the full set is about 51 minutes and fits. Nothing had been queued or graded when this was corrected.)
- **Rule:** int4's band (KL <= 0.03 and top-1 >= 0.98; ambiguous band KL <= 0.06 and top-1 >= 0.95), on all 150 records, against the same f32 reference rows.
- **Validity as for the other arms**, including that every prompt's token count equals the encoder dump's. That also checks the tokenizer: it is the GGUF's own, not Clef's `tokenizer.json`.
- **Prediction, written now:** KL inside the band, top-1 below 0.98 (JEV's int4 read 0.940 and its CPU int4 KL 0.0298). The load is smoked (2026-10-03, exploratory): the GGUF loaded through `modelload`, the head read its quantized lm_head rows, token counts equalled the encoder dump (589 and 238), the probabilities summed to 1, and P(true) sat within 0.03 of the other arms' on both records. That says the plumbing works, not how close the arm is.
- **Cost:** 44,823 tokens at about 68 ms per token (the smoke's, exploratory) is about 51 minutes; queued at 65, resumable.

## 5b. Amendment (2026-10-03, before any graded run): the CUDA-resident arm

D11's follow-up (`decisions-d11-resident-hidden-2026-10-03.md`) lets a CUDA-resident model answer the Clef route's all-positions hidden state on the device. It needs its own graded arm: the device kernels are not the CPU's, and a few exploratory records are not a grade.

- **Arm `cuda-int4`:** goinfer's Clef pipeline with the backbone loaded `-backend cuda -quant int4` through `modelload` (the serve path), resident on the 8 GB RTX 2070 SUPER; the head is the checkpoint's f32 head on the CPU; the lm_head rows are the quantized model's. Run by `cuda/clef_fidelity_test.go` (a harness in the cuda module, which may import the root's internal packages where the root module may not import cuda), full-precision rows in the same format as the CPU arms. int8int8 does not fit the card (9.5 GB), so int4 is the only device arm.
- **Set:** all 150 records. **Rule:** int4's band (KL <= 0.03 and top-1 >= 0.98; ambiguous KL <= 0.06 and top-1 >= 0.95), against the same f32 reference rows, graded as `cuda-int4=PATH`.
- **Validity (D6b's amendment 2):** the run is VOID unless the model's decode path reads `cuda-resident` (the harness fails otherwise) AND the mean request time is under 20 ms per input token (the device measured about 3, the CPU 54 to 68), so a CPU fallback cannot pass as a GPU result. Its verdict is its own and does not change the CPU arms'.
- **Prediction, written now (revised the same day, before any graded run; see 5c for why):** the device agrees with the CPU when both are configured alike, so `cuda-int4-pin` should land near the CPU int4 arm run with the pin, and `cuda-int4` near the CPU int4 arm run as served. Both int4 arms are predicted inside the KL band and below 0.98 top-1, as JEV's CPU int4 was (KL 0.0298, top-1 0.940), graded FAIL or AMBIGUOUS on top-1. D6b's CUDA-resident JEV int4 read KL 0.038 and top-1 0.907 (worse than its CPU int4); that is the case this arm could repeat, and the three exploratory records, once the table setting is matched, do not suggest it will (the device sat within 0.005 of the CPU in P(true)).
- **Cost:** 44,823 tokens at about 3.2 ms per token (exploratory) is about 2.5 minutes plus a load of about a minute; queued at 20 minutes. It needs the GPU and about 6 GB of its memory and nothing else at the same time.

## 5c. Amendment (2026-10-03, before any graded run): the int4 arms and `--embed-int4`

**Found while smoking the GPU arm.** `--embed-int4` is ON by default in serve and the CLIs (`internal/loadflags`; off only on Metal) and applies only with `-quant int4`: it stores the token-embedding/LM-head table at int4 instead of the int8 pin. The Clef head reads LM-head rows, so the flag changes int4 answers. The harness as first written never set it, so its int4 arms would have run with the pin while serve runs with the table at int4. On three records the flag moved P(true) by up to 0.04 (0.9186 with the pin, 0.8765 as served, on the GPU). The f32, int8int8 and q4k arms are unaffected.

- **Policy:** the graded int4 arms run **as served** (table at int4, the harness default; `CLEF_FIDELITY_EMBED_INT4=0` selects the pin). This is also what D6b's JEV int4 arms ran, since they used the CLI default. The f32, int8int8 and q4k arms do not change.
- **The CPU `int4` arm queued earlier ran the old harness (pin).** It is re-queued with a rebuilt binary so it runs as served. Its exploratory smoke numbers in section 6 (P(true) 0.9217, 0.9424) were the pin and are not that arm's.
- **Diagnostic arm `cuda-int4-pin`:** the CUDA arm with the pin, graded as `cuda-int4-pin=PATH` on int4's band, same validity rule. It exists to separate the device from the flag: `cuda-int4` against `cuda-int4-pin` is the flag's effect on the device, `cuda-int4-pin` against a CPU int4 arm is the device's. It costs about 5 minutes. No CPU pin arm is run (an hour for a diagnostic); the CPU pin is represented by the two exploratory records only.
- **What this changes in the record:** an earlier note read a 0.045 gap between the device and the CPU as a possible GPU-int4 discrepancy like D6b's. It was this flag (see `decisions-d11-resident-hidden-2026-10-03.md`, corrected).

## 6. Cost, from an exploratory smoke (by day, one run, 2 records, labelled exploratory and never quoted as a result)

2026-10-03, nobara, 2 records (827 tokens), CPU: int4 loaded in 19 s and ran at about 68 ms per token; int8int8 loaded in 14 s and ran at about 54 ms per token. Both rows had token counts equal to the encoder dump (589 and 238), probabilities summing to 1, and the same top option as gold; the two arms differed by 0.01 to 0.025 in P(true). So the first run on the real weights exercised the whole serve load path, the real tokenizer, the 32-layer backbone and the real head without error. f32 was not smoked (36 GB of RAM by day).

| job | tokens | estimate | basis |
|---|---|---|---|
| `d13-clef-f32` | 44,823 | 90 min | no measurement: assumed 1.5x int8int8's rate (D6b's JEV ran f32 about 1.3x slower per item); timeout 3 h |
| `d13-clef-int8int8` | 44,823 | 45 min | 54 ms per token + load |
| `d13-clef-int4` | 44,823 | 60 min | 68 ms per token + load |
| `d13-clef-q4k` | 44,823 | 65 min | 68 ms per token + load (smoke on the GGUF) |
| `d13-clef-cuda-int4` | 44,823 | 20 min | 3.2 ms per token + load (exploratory, 3 records) |
| `d13-clef-cuda-int4-pin` | 44,823 | 20 min | as above |

Queued after the two D10 jobs (55 + 65 min), about 5 h 15 min of estimates in all; the q4k arm (section 5a) adds about 1 h 05 min and the CUDA arms (5b, 5c) about 40 min. The runner does not start a job whose estimate would end after 06:30. Each job is resumable (rows already written are skipped), so a timeout loses nothing.

## 7. Result

**Run:** the night queue started 2026-10-03 11:18 PDT; all nine jobs ok, finished 18:59 (`night.py morning`). Reference `d10-clef-f32` (31 min), arms f32 54 min, int8int8 40, int4 50, q4k 48, cuda-int4 and cuda-int4-pin 3 each; the bf16 context row 42. nobara-pc, Ryzen 7 3700X (8 threads) and RTX 2070 SUPER, NVIDIA driver 595.91.07, `Cloudflare/clef-flash` rev `17f0b0ad64efb65d273590632833508766b2aae6`, head sha256 `19cdcec8…5ba0`, module sha256 `0e304cf7…c3a3`, torch 2.14.0+cpu, transformers 5.16.1 (the release was tested with torch 2.11 / transformers 5.10.2; recorded in `testdata/decisions/clef/clef_env_f32.json`). Greedy, no sampling; every prompt's token count matched the encoder dump on every arm. Both cuda arms: decode path `cuda-resident (int4)` on all 150 rows, 3.2 ms per input token, so both are valid under 5b's rule. The grader's full output is `decisions-d13-clef-fidelity-2026-10/grade-2026-10-03.txt`; the per-arm rows are in `decisions-d13-clef-fidelity-2026-10/rows/`, the reference rows in `testdata/decisions/clef/probs_f32.jsonl` (sha256 `b0250cb4…5688`) and `probs_bf16.jsonl`.

| arm | mean KL | top-1 vs ref | worst item KL | ECE (ref 0.1070) | arm − ref ECE, 95% bootstrap | verdict |
|---|---|---|---|---|---|---|
| f32 | < 1e-5 | 1.0000 | 0.0000 | 0.1070 | [−0.0000, +0.0000] | **PASS** |
| int8int8 | 0.01653 | 0.9067 | 0.2270 | 0.1244 | +0.0174 [−0.0506, +0.0650] | **FAIL** (top-1; KL inside the band) |
| int4 (as served) | 0.05115 | 0.8400 | 0.8809 | 0.1376 | +0.0306 [−0.0532, +0.0821] | **FAIL** (KL and top-1) |
| q4k (bartowski imatrix GGUF) | 0.02067 | 0.8867 | 0.9220 | 0.0916 | −0.0154 [−0.0648, +0.0349] | **FAIL** (top-1; KL inside the band) |
| cuda-int4 (as served) | 0.04403 | 0.8400 | 0.4277 | 0.1427 | +0.0357 [−0.0512, +0.0799] | **FAIL** (KL and top-1) |
| cuda-int4-pin (diagnostic) | 0.04512 | 0.8133 | 0.6837 | 0.1243 | +0.0173 [−0.0488, +0.0721] | **FAIL** (KL and top-1) |

No arm reaches the AMBIGUOUS band: the best non-f32 top-1 is 0.907 (band edge 0.95). Calibration is UNRESOLVED in every arm (every interval reaches 0); none FAILS it. Context row, the release's own dtype: the reference at bf16 against the f32 reference reads **KL 0.00005, top-1 0.9933** (1 flip in 150), so the quantized arms' misses are quantization, not numeric noise a correct port would show. By kind (top-1 vs the reference, int8int8 / int4 / q4k): choice 0.920 / 0.900 / 0.920, noul 0.941 / 0.902 / 0.941, **score 0.857 / 0.714 / 0.796**: the `score` questions are the fragile ones.

### What the f32 arm says

KL below 1e-5, top-1 1.0000 on all 150 records, ECE identical to the reference's: **the port is correct.** Rows differ from the reference in the 8th digit (P(true) 0.93546729 against 0.93546724 on the first record), so it is an independent computation, not a copy. This is the premise of every other row in the table.

### Predictions (section 4 and the amendments), graded as written

- **f32 PASS, KL well under 0.001:** HELD.
- **int8int8 and int4: KL inside the 0.03 band, top-1 below 0.98:** int8int8 HELD (KL 0.0165, top-1 0.907). **int4 FAILED the prediction**: KL 0.0512 is outside the 0.03 band (inside the 0.06 ambiguous KL edge, but top-1 0.84 is under its 0.95 floor); I predicted the same shape as JEV's int4 (KL 0.0298), and the Clef int4 is worse on KL (0.051) and top-1 (0.840 against JEV's 0.940).
- **Calibration not failed in any arm:** HELD (six arms, all intervals reach 0).
- **q4k: KL inside the band, top-1 below 0.98:** HELD (0.0207, 0.887).
- **cuda-int4 / cuda-int4-pin: both inside the KL band, top-1 below 0.98:** FAILED on KL (0.0440 and 0.0451, both over 0.03). The top-1 half held.
- **"cuda-int4 lands near the CPU int4 as served":** HELD in aggregate (KL 0.0440 against 0.0512, top-1 0.840 against 0.840). The companion claim, `cuda-int4-pin` near a CPU int4 *pin* arm, could not be graded: no CPU pin arm was run (5c).
- **"The device could repeat D6b's CUDA-int4-worse-than-CPU case":** NOT repeated: the device's KL is lower than the CPU's, not higher.
- **Clef against JEV:** no prediction was made, and the result is unresolved (below).

### Clef against JEV-9B (informational, no rule attached)

On the same 84 gold rows, both at f32 reference: top-1 against gold **Clef 0.690, JEV 0.774**, Clef − JEV −0.083, 95% paired bootstrap [−0.202, +0.024]; ECE Clef 0.1070, JEV 0.1222, Clef − JEV −0.0153, [−0.0536, +0.1094]. Both intervals reach 0. The point estimates favour JEV on accuracy and Clef on ECE; **neither difference is resolved on 84 rows**, and neither is a statement about which route is better. Under the owner's 2026-10-03 decision (both routes equal) this changes nothing.

### What follows from the rule

- **The default stays `int8int8`, with the disagreement stated.** Section 2: the default is reopened only if int8int8 FAILS calibration or is outside its KL band. Its KL is 0.0165 (inside 0.03) and calibration is unresolved, so a top-1 miss alone leaves it standing, as in D6b. The statement: on the Clef route int8int8 disagrees with the f32 reference's argmax on **14 of 150** records (9.3%); 9 of the 14 are records where the reference's own top-two margin is under 0.1, the largest margin among them is 0.445. (That tie-margin count is a post-hoc diagnostic, not a registered statistic, and is not a pass.) Its KL is about twice JEV's int8int8 (0.0165 against 0.0088).
- **`int4` is not a quality-neutral option on the Clef route.** It FAILS outright, on both CPU and device, and the device does not rescue it. Whether to keep offering it is the owner's call; the data says it moves 24 of 150 argmaxes.
- **Post-hoc paired comparisons, labelled as such** (not registered; the paired design is rule 7 of the measurement discipline, and the intervals are 2000-resample bootstraps of per-record differences): CPU int4 against int8int8, mean KL difference **+0.0346, [+0.0174, +0.0552]** (int4 is resolvably worse than int8int8); q4k against CPU int4, **−0.0305, [−0.0467, −0.0163]** (the third-party imatrix 4-bit file is resolvably closer to the reference than goinfer's own int4; this is a statement about that artifact, 5a, not about q4k as a format); device against CPU, both as served, **−0.0071, [−0.0270, +0.0098]** (unresolved); `--embed-int4` on the device (served − pin), **−0.0011, [−0.0193, +0.0147]** (unresolved: on 150 records the flag's effect is not distinguishable from zero, although it moved single records by as much as 0.53 in P(true)). So the earlier worry that the GPU is worse than the CPU at int4 on this route is **not supported**; neither is the claim that it is the same, at this sample.

### Owner decision after the result (2026-10-03)

**Clef is not served at int4.** `quant=int4` on a Clef model is refused at load, with an error naming these figures; `int8int8` stays the default. **The decision-model tag may go on Clef-flash at `int8int8`, as the owner's decision and not as a pass of this record's gate** (the registered top-1 bar of 0.98 is missed, 0.907; the gate stays graded FAIL and no bar is moved). Not yet built: the site changes only at a release, and every real-weights record here has one question. The int4 arms stay in this record as the measurement behind the int4 decision, and the harnesses still run int4 (they load through `modelload`, not through serve's check).

### Not shown by this run

Nothing here grades speed (D14), the Metal backend, the WebGPU backend, the 27B Clef, or a `chat-v1` template. The int8int8 arm is CPU only (it does not fit the card). The f32 row is agreement to about 1e-7 in probability, not a claim of bit-identity.
