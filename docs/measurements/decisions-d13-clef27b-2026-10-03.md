# D13, the last bullet — Clef 27B on nobara's CPU: PRE-REGISTERED 2026-10-03 (before any graded run)

**Why.** The task doc's D13 ends: "Then Clef 27B on the Linux box's CPU, at f32 against a smaller reference set. Report its
speed as measured, with the machine named." D13 graded Clef-flash (9B); its record says the 27B "is registered separately if
wanted" (`decisions-d13-clef-fidelity-2026-10-03.md` §5). The owner asked for it on 2026-10-03 ("can you do the Clef 27B on
the CPU").

## 1. Deviations from the task doc, disclosed before the run

- **Not f32, on either side.** A 27B at f32 is about 108 GB, for goinfer and for the reference alike; nobara has 62 GB.
  - **The reference** runs at **bf16**, the release's own dtype, with D10's f32-GEMM emulation (bf16 operands, f32
    products and accumulation, a bf16-rounded result; `scripts/pin_clef_d10.py`, `install_bf16_gemm_emulation`). On
    Clef-flash that reference read **KL 0.00005 and top-1 0.993 against the f32 one** (D13 §7, the context row): its own
    error is about 300 times smaller than the int8int8 arm's, so it can stand in for f32 here.
  - **goinfer** runs **int8int8**, the decision-model default and the quant the owner tagged Clef at. int4 is refused for
    Clef at load (owner decision after D13). No f32 port check is possible at 27B. The port is the one D13 checked at f32
    on Clef-flash to < 1e-5 KL: the same `internal/clef` code, a head of the same shape at width 5120 instead of 4096,
    and a backbone of the same family at 64 layers instead of 32.
- **"A smaller reference set":** every 5th of D10's 150 records by position, 30 records (10 choice, 11 noul, 9 score; 18
  with gold), 8,659 input tokens (the longest record 624). `records.jsonl` sha256 `5942ccb9…9b7c`, D13's.

## 2. The checkpoint

`Cloudflare/clef` at revision `2f3de3dd85f379784083b0814d997ab627200f0c`: Qwen3.8-27B, `qwen3_5_text`, 64 layers, hidden
5120, vocab 248,320, untied LM head; the joint head at width 1024, 2 routing layers, 4 layers, 16 heads.
- Downloaded 2026-10-03 to the archive, `/srv/models/clef`: 25 files, every size equal to the hub's listing.
- `joint_head.safetensors` sha256 `a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`, equal to the hub's
  LFS hash.
- `joint_schema_model.py` sha256 `0e304cf7…a3c3`, **byte-identical to Clef-flash's**, and `tokenizer.json` (`06b95093…e523`)
  and `tokenizer_config.json` (`91a08f82…db72`) are byte-identical too. So the reference code is the one D10 pinned, and
  every record's token count must equal D10's encoder dump: the grader's validity check stands unchanged.
- `chat_template.jinja` differs from Clef-flash's. The Clef encoder builds its own prompt and does not read it.
- Both sides read the copy on nobara's NVMe, `~/models/clef` (the archive is storage; goinfer's harness refuses it).

## 3. What runs

On nobara's night queue, `docs/measurements/decisions-d13-clef27b-2026-10/run-clef27b.sh`, in order, each step its own
process (a bf16 load of the 27B is about 55 GB, an int8int8 one about 28 GB; they never share RAM):
1. **The reference:** `pin_clef_d10.py model --dtype bf16 --every 5` with the 27B's four pins, in `~/d0venv` (torch
   2.14.0+cpu, transformers 5.16.1, D10's), rows to `probs_bf16.jsonl`.
2. **goinfer int8int8:** D13's harness binary (`TestFidelityArm_run`, `~/goinfer-bench/decisions-d13/clef-fidelity.test`, sha256
   `ca71f9e8dffbc51046f5fc52afea9acb9f26de0bb61612646dcd914c7e651437`, the binary whose f32 arm passed D13) with
   `CLEF_FIDELITY_EVERY=5`, through `modelload`, the serve path.
3. **The grade:** D13's `grade.py --ref probs_bf16.jsonl --ref-every 5 --arms int8int8=…:5`.

A one-record probe of step 1, labelled exploratory, comes first, to measure the reference's peak memory and its time per
token; the night estimate is written from it before the job is queued.

**The probe (2026-10-03 22:30 PDT, nobara, exploratory, not a result; its row is in a separate probe directory and is not
graded):** the first registered record, 589 tokens: loaded in 4 s (memory-mapped), 122.2 s for the record, about 207 ms
per token, **maximum resident set 49.0 GB** of 62, probabilities finite and summing to 1 (0.984 / 0.016).

**Estimate:** the reference, 8,659 tokens at about 207 ms, about 30 min. goinfer int8int8: no 27B measurement; Clef-flash
ran 54 ms per token (D13 §6), and the 27B streams about 3 times the weights, so about 160 ms per token, about 25 min plus a
load of 1 to 2 min. About 55 min in all; queued at 90, inside the 3-hour job limit. Both phases resume if interrupted.

## 4. The rule (written before the run)

D13's int8int8 band, applied by D13's grader as written:
- **PASS:** mean KL(reference ‖ goinfer) ≤ 0.03 and top-1 agreement ≥ 0.98.
- **AMBIGUOUS** (reported, never a pass, to the owner): KL ≤ 0.06 or top-1 in [0.95, 0.98), with the other criterion passing.
- **FAIL:** anything else. Calibration as D13's (ECE against gold on the 18 gold rows, paired bootstrap; an interval that
  reaches 0 is UNRESOLVED).
- **Validity:** exactly the 30 registered records, every option list equal to the reference's, finite probabilities summing
  to 1, and every token count equal to the encoder dump's.

**Context, already measured (D13's own rows on these same 30 records):** Clef-flash at int8int8 against its f32 reference
reads KL 0.01765, top-1 0.833. **Prediction, written now:** the 27B's int8int8 arm, like Clef-flash's, has KL inside the
0.03 band and top-1 below 0.98, so it FAILs on top-1. A larger model's int8 error need not be smaller per position, and 30
records leave a wide interval: one flip is 3.3 points.

## 5. Speed

Reported, with no bar, as the task doc asks: goinfer int8int8's mean request time per input token over the 30 records,
from the harness's own per-record timing, on nobara-pc (Ryzen 7 3700X, 8 threads, 62 GB) with the box otherwise idle (the
night queue), from `~/models`. One pass. It is a request time (prefill of the whole record plus the head), not a decode
rate, and not a `benchmarks.md` row. The reference's time per record is reported beside it as fixture-generation cost, not
as a peer number.

## 6. What this does not establish

The f32 port at 27B (impossible here); anything on the GPU (the 27B fits neither card); multi-question records; gold
accuracy beyond the 18 gold rows' reading; and how the 27B compares with Clef-flash on the same records, which is
reported but has no rule.

## 7. Result (graded 2026-10-04 morning, from the night job `d13-clef27b-cpu`, 2026-10-03 22:39–23:32 PDT)

**Run:** nobara-pc (Ryzen 7 3700X, 8 threads, 62 GB), the night queue, the box otherwise idle, the timing lock held by the job. Pinned worktree at `f3b50cda`, harness binary sha256 `ca71f9e8…1437`, `~/models/clef` (never the archive), `~/d0venv` (torch 2.14.0+cpu, transformers 5.16.1). 52 minutes against the 90 queued (the estimate in section 3 was about 55). Grader output `data/grade.txt`, the rows `data/probs_bf16.jsonl` (reference) and `data/probs_goinfer_int8int8.jsonl`, the job log `data/night-job.log`, the environment `data/clef_env_bf16.json`.

**Validity (checked against the registered list, not assumed):** exactly the 30 registered records in both files, in record order; every option list equal to the reference's; every probability finite and each row summing to 1; every token count equal to the encoder dump's (8,659 tokens in all). 0 failures.

| arm | n | mean KL | top-1 vs reference | worst item KL | verdict |
|---|---|---|---|---|---|
| Clef 27B, goinfer int8int8 (reference at bf16) | 30 | 0.01209 | 0.8667 (26 of 30) | 0.0784 | **FAIL** on top-1 (KL inside the 0.03 band; below the 0.95 ambiguous floor) |

By kind (KL, top-1): choice 0.00680, 0.800 (n=10); noul 0.00485, 1.000 (n=11); score 0.02683, 0.778 (n=9). Calibration on the 18 gold rows: ECE 0.1869 against the reference's 0.2356, arm − reference −0.0487, 95% paired bootstrap [−0.0925, +0.0312]: **UNRESOLVED** (the interval reaches 0). Top-1 against gold: arm 0.667, reference 0.722.

- **Prediction (section 4), graded as written:** KL inside the 0.03 band and top-1 below 0.98, so FAIL on top-1: **HELD.**
- **The f32 port check at 27B was impossible here (section 1), so a miss cannot be split into port and quantization at this size.** What supports reading it as quantization: the same `internal/clef` code passed the f32 check on Clef-flash to under 1e-5 KL, the error here has the same shape (concentrated on `score`, none on `noul`), and its size is the Clef-flash int8int8 size.
- **Reference is bf16, not f32** (section 1): on Clef-flash that substitution read KL 0.00005 and top-1 0.993, so its own error is far below this arm's.

**Speed, reported with no bar (section 5), one pass:** goinfer int8int8 took **1,382 s for the 8,659 tokens: 160 ms per input token, 46.1 s per record**, the longest record (624 tokens) 101 s. That is a request time (prefill of the whole record plus the head), not a decode rate and not a `benchmarks.md` row. It matches the estimate (about 160 ms) and is 2.96 times Clef-flash's 54 ms per token (D13 §6). The reference at bf16 took 1,707 s: 197 ms per token, 56.9 s per record, fixture-generation cost, not a peer number.

**Against Clef-flash on the same 30 records (reported, no rule; `data/flash-same-30-grade.txt`):** Clef-flash int8int8 against its f32 reference reads KL 0.01765 and top-1 0.833 (25 of 30); the 27B reads 0.01209 and 0.867 (26 of 30). That is one record apart, and the references differ in dtype, so it is not a difference. Against gold (18 rows), the reference models read top-1 0.722 (27B, bf16) and 0.778 (Clef-flash, f32), and ECE 0.2356 and 0.2049; against JEV-9B's 0.778 and 0.137 on the same 18 rows, both Clef intervals reach 0 (27B ECE, Clef − JEV +0.0986 [−0.0008, +0.2448]; top-1 −0.056 [−0.222, +0.111]). Eighteen rows resolve nothing about which model is more accurate.

**Not established (section 6, unchanged):** the f32 port at 27B; anything on a GPU; multi-question records at this size; gold accuracy beyond 18 rows. **The owner's 2026-10-03 tag decision names Clef-flash at `int8int8`; it says nothing about the 27B, whose int8int8 arm fails the same bar by the same metric.** Whether the 27B is offered, and how it is described, is not decided here.
