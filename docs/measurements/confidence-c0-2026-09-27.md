# C0 — does a per-field confidence number on constrained output mean anything? (2026-09-27)

C0 of [`task-constrained-confidence.md`](../tasks/task-constrained-confidence.md). It is a measurement only: no API, and
no code beyond the harness. It was pre-registered in `d805dd7c`, before either graded model ran; the gates below are
that registration's, computed by its `analyze.py`.

**Result: C clears for enum, boolean and integer fields. Number and string fields are parked (too few wrong answers
to judge), not failed. Gates 0 and 1 pass on both graded models.**

| gate | enum (`category`) | boolean (`urgent`) | integer (`order_count`) | number (`refund_amount`) | string (`customer_name`) |
|---|---|---|---|---|---|
| 0: ≥ 80% of instances have a free value token | PASS (100%) | PASS (100%) | PASS (100%) | PASS (100%) | PASS (100%) |
| 2: AUROC, 1.5B (correct/wrong) | **0.847** PASS (48/12) | **0.727** PASS (38/22) | **0.680** PASS (43/17) | insufficient (60/0) | insufficient (57/3) |
| 2: AUROC, 7B (correct/wrong) | 0.967, insufficient (55/5) | 0.905, insufficient (58/2) | 0.942, insufficient (57/3) | insufficient (60/0) | insufficient (60/0) |
| **verdict** | **ships** | **ships** | **ships** | parked | parked |

- **Gate 1 (cost):** the in-situ readout costs a mean **4.20%** of a mask-only decode token on the 1.5B (0.570 ms
  against 13.56 ms) and **1.44%** on the 7B (0.542 against 37.55 ms). Both are under the 5% bar.
  - The median is 1.88% and 0.72%. The mean runs higher because inside a free string nearly the whole vocabulary is
    legal: p90 1.17–1.19 ms against a p10 of 0.24–0.25 ms.
  - The ungraded 0.5B smoke run read 7.8%. On small models the cost is at the bar, so C1 should read only at free
    positions (a forced token needs no readout) or use a float32 exponential.
- **How to read gate 2:** it is directional only, not a calibration claim (§2 of the task). AUROC 0.5 is chance.
  - The 7B makes too few mistakes on this set to qualify on any kind. Its AUROCs point the same way but decide
    nothing.
  - Integer passes on one qualifying model, 0.03 above the bar.
  - Number and string need a harder labelled set before they can be judged. A number or string confidence may not
    ship until then.
- **Item 3, "is the enum/boolean case the whole viable feature?":** not quite. Integer passes too, but it rests on one
  model and 17 wrong answers. The pre-registered consequence ("C1 ships enum/boolean only") applies only if integer,
  number and string all do not pass; integer did pass.
  - What the data does support: every graded kind that passed decides its value in **one free token**. For a closed
    set, that position's option-level distribution *is* the answer distribution (see "Forced fractions").

## Setup

- MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2. Metal backend, `-quant int4`, greedy, at most 200 tokens per answer.
- **Models:**
  - qwen2.5-coder-1.5b-instruct q4_k_m (`.gguf`);
  - qwen2.5-7b-instruct q4_k_m (`.int4.metal.giw`, tokenizer from its `.gguf`);
  - both from `~/models`.
- **Data:** [`confidence-c0-2026-09-27/tickets.jsonl`](confidence-c0-2026-09-27/tickets.jsonl), 60 support tickets
  written for this, sha256 `71787d8f…`.
  - Each gold label follows from rules stated in the system prompt (in `metal/confidence_c0_test.go`).
  - Category counts: billing 15, technical 15, shipping 13, account 10, other 7. 13 are urgent, 14 request a
    refund, 20 mention orders, and 5 are unsigned.
- **Schema:** one field of each kind, `additionalProperties: false`. The prompt asks for a JSON object; the grammar
  (`constrain.JSONSchema`) enforces it.
- **Harness:** `TestConfidenceC0` wraps the masker.
  - Per token it records: *p*, the probability among grammar-legal tokens at T = 1; whether the grammar forced the
    token byte for byte; the field whose value it overlaps; and the readout's in-situ time.
  - For enum and boolean fields it also records the option-level distribution at the deciding position: the mass
    of every legal token, summed by the option it spells.
  - A second, mask-only pass per ticket prices the readout.
- **Session:** one, 22:29–22:37 PDT. Each model ran after an idle gate (load1 ≤ 2.0).
- **Raw data:** [`confidence-c0-2026-09-27/`](confidence-c0-2026-09-27/):
  - `c0-1.5b.jsonl` and `c0-7b.jsonl` (per-token records);
  - `analysis.md` (the script's full output, including the secondary aggregations);
  - `analyze.py`, `run-c0.sh` and `run.log`.

## Forced fractions (item 1)

| field kind | value tokens (mean, 1.5B / 7B) | free tokens (mean) | free fraction | what is free |
|---|---|---|---|---|
| enum | 3.00 / 3.00 | 1.00 / 1.00 | 0.333 | the first letter of the option; the rest of the literal and its quotes are forced |
| boolean | 1.00 / 1.00 | 1.00 / 1.00 | 1.000 | `true` / `false`, one token |
| integer | 1.00 / 1.00 | 1.00 / 1.00 | 1.000 | one digit (Qwen splits numbers into digits; every count here is < 10) |
| number | 1.58 / 1.52 | 1.58 / 1.52 | 1.000 | every digit and the point |
| string (name) | 4.92 / 4.98 | 3.92 / 3.98 | 0.80 | everything but the closing quote |
| string (summary) | 19.2 / 14.7 | 18.2 / 13.7 | 0.93–0.95 | as above |

- Averaging over all of an enum's tokens would put two forced 1.0s beside each real choice. That average measures
  the schema, which is why the registration computes confidence over free tokens only.
- The grammar accepts properties in any order, so each key's first letter is also a real choice. It is
  scaffolding, not a field value, and it enters no number.

## Aggregation (item 2)

The primary aggregations were pre-registered: *decision* (option-level mass) for enum and boolean, the minimum
over free tokens for integer and number, and the geometric mean for string. On this set:
- **enum, boolean, integer:** one free token, so every aggregation gives the same number. The only difference is
  *decision* against the raw first-token probability on enum (0.847 against 0.845): summing the mass of every
  token that spells the same option barely moves it here, because the options' first letters differ.
- **string:** the aggregations differ (1.5B: geometric mean 0.895, minimum 0.819, product 0.842, first 0.789), but
  on 3 wrong answers. They decide nothing.

## Observations

- **Repeating an identical prompt does not reproduce greedy output on Metal.** The mask-only pass re-sends each
  ticket's prompt right after the data pass, and emitted identical ids on only 36/60 (1.5B) and 24/60 (7B).
  - The second run reuses the resident prefix and prefills only a suffix. On Metal that takes a different
    (f16-MMA) prefill path from a cold prompt, as recorded in the resident-reuse work.
  - The data pass always ran first, after a different ticket, so only the system-prompt prefix was reused there,
    the same for every ticket.
  - The pricing pass is only a token clock, so this does not touch any number above. But a confidence number
    inherits it: the same request can carry a slightly different confidence depending on what the cache held.
- **The 7B is nearly error-free on this set** (category 5/60 wrong, the other fields ≤ 3). The set cannot grade a
  7B-class model's confidence; D0's gold-labelled `openjev_v2` rows are the next labelled set (1,109 boolean
  calibration items and 9,767 out-of-domain, plus choice items), per the task doc.

## What this does not show

- **Calibration.** AUROC says low-confidence answers are wrong more often, not that 0.8 means right 80% of the time.
- **Number and string fields.** Parked for want of errors.
- **Sampling.** Every answer here is greedy. Under temperature sampling, the answer and its confidence both vary.
