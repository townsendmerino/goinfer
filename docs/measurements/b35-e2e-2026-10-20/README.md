# B35 end-to-end gate: Qwen3.5-9B chat turns continue from the kept state (pre-registered 2026-10-20)

B35 (docs/queue-engineering.md) keeps a recurrent model's state at KVScope.Keep, the point the next chat turn's prompt repeats, so turn 2
of a conversation stops re-prefilling the whole prompt. The tier 0 gates passed: the decoder's hash fake (`decoder/b35_snapshot_test.go`),
the serve routes' Keep (`internal/serveapp/b35_keep_test.go`), and on CUDA a restore equal to a cold prefill on `qwen3next-tiny`
(`cuda/b35_snapshot_test.go`, nobara 2026-10-20 08:16, PASS with its plant red). This is the merge gate Francis set: the real model
through `/v1/chat/completions`, where turns 2 and 3 reuse the saved state with a time to first token near a warm one, and match a cold
run bit for bit. Merge needs Francis's yes after it.

Written and committed before the run. The rules below do not move after the data.

## Design

- **Box:** nobara, RTX 2070 SUPER, under the timing lock (a `night.py` job, run by the box's `night.py serve`). Model:
  `~/models/Qwen3.5-9B-Q4_K_M.gguf`, `-backend cuda -ctx 16384`, serve's defaults otherwise (thinking: the template's default, off).
- **Builds:** the B35 branch head and its base on `main`, pinned in the queued command (`run-b35-e2e.sh <B35> <base>`), built there with
  `-tags cuda`.
- **Conversation** (`gate.py`, the 2026-10-18 reuse probe's shape): a system prompt of about N tokens of notes; turn 1 asks one question;
  turns 2 and 3 append the previous reply and a new question. Greedy, 24 new tokens. N = 4096 and 8192.
- **Arms**, per size:
  - A: the B35 build, one server, streamed: turns 1-3 (time to first token, `usage.prefill_reused_tokens`), then a cold control of turn
    3's length (a nonce first in the system prompt);
  - B: the B35 build, one server, `logprobs`: turns 1-3, the warm path;
  - cold: the B35 build, a fresh server for each of turns 2 and 3;
  - base: the base build, a fresh server for each of turns 1, 2 and 3.
- **Bit for bit** means the served tokens and the exact text of each logprob.

## Rules (`grade.py`)

| check | rule | on failure |
|---|---|---|
| I1 | B's turns 2 and 3 (restored state) equal the cold arm's, every token and logprob | FAIL |
| I2 | B35's cold turn 1 (B), turns 2 and 3 (cold) equal the base build's | FAIL |
| I3 | each streamed reply (A) equals the logprob run's (B) | FAIL |
| R1 | turns 2 and 3 report reused tokens > 0 on A and B; every cold request (A's control, B's turn 1, the cold and base arms) reports 0 | FAIL |
| T1 | a warm turn's time to first token ≤ 1.5 × (the cold control's time × rows still prefilled ÷ the control's prompt) + 0.25 s | between 1.5× and 2.5×: PARKED for Francis; past 2.5×: FAIL |

PASS is every row green. T1 asks that a warm turn cost what its leftover prefill costs and nothing hidden: the saved state lands on the
pass grid (Keep rounded down to the CUDA pass width), so a warm turn still prefills up to one pass plus the reply and the new question.
The record also reports each warm turn's time as a fraction of cold.

**Cost:** 7 server loads per size (about 12 s each, the 2026-10-18 probe) and 8 cold prefills per size (about 15 s at 4k, 39 s at 8k);
about 4 + 7 minutes, plus two builds: queued at 20 minutes, timeout 45.

## Result: PASS (2026-10-20, nobara, 09:59:42–10:10:48 PDT)

`b35-e2e` on nobara's `night.py serve`, under the timing lock, B35 `fbae3175` against base `fa8efaa0`, driver 595.91.07, load
average 0.34 at the start. Every row of `grade.py` is green at both sizes (`raw/grade.txt`; the record is `raw/gate.json`, the log
`raw/run.log`).

| size | turn | time to first token | prompt tokens | reused | of cold |
|---|---|---|---|---|---|
| 4096 | 1 (cold) | 15.62 s | 4,098 | 0 | |
| 4096 | 2 | 2.36 s | 4,130 | 3,584 | 0.149 |
| 4096 | 3 | 2.48 s | 4,150 | 3,584 | 0.157 |
| 4096 | cold control | 15.82 s | 4,172 | 0 | |
| 8192 | 1 (cold) | 39.06 s | 8,146 | 0 | |
| 8192 | 2 | 3.05 s | 8,177 | 7,680 | 0.077 |
| 8192 | 3 | 3.19 s | 8,197 | 7,680 | 0.080 |
| 8192 | cold control | 39.76 s | 8,219 | 0 | |

- **I1, warm = cold:** turns 2 and 3 restored from the kept state gave the same tokens and the same logprobs, digit for digit, as a
  cold prefill of the same request on a fresh server, at both sizes.
- **I2, cold output unchanged:** B35's cold turns 1, 2 and 3 equal the base build's, digit for digit.
- **I3:** the streamed and logprob runs gave the same replies.
- **R1:** warm turns reused 3,584 (4k) and 7,680 (8k) prompt tokens; every cold request reused none.
- **T1:** each warm turn's time to first token is 1.02–1.17 × the cold control's time scaled to the rows it still prefilled (bar 1.5).

**What the warm turns still prefill.** The kept state lands on the CUDA pass grid (512 rows): turn 1's Keep (4,094 at 4k) rounds
down to 3,584, so turn 2 prefills 546 rows. Turn 2's own prefill, 3,584 to 4,130, is a single pass with no boundary below its Keep,
so it keeps nothing newer and turn 3 restores turn 1's state again (566 rows). The kept point advances only when a later prefill
crosses a pass boundary below its Keep, so a warm turn re-prefills up to about one pass plus whatever the conversation has added since
the last state kept. Placing the save at Keep itself (a pass split there) would bring a warm turn down to the new messages alone; that
is a follow-up, not part of this gate.

**Lock overlap:** none during the run. The job took nobara's lock at 09:58:42, when tack's C6 4.1 screen released it. A read-only
`night.py status` over ssh ran at about 09:59, during the job's settle and before its first timed request, and the next one at 10:17,
after the job had ended. Earlier that morning the B35 day checks ran on the Mac beside tack's screen, which timed nobara: load on
tack's harness box only. The three tack cells that overlapped this session's nobara touches at 08:35–08:47 were kept, with the overlap
listed in tack's own record (Francis, 2026-10-20).

## Rerun on KV8a: PASS (2026-10-21, nobara, 08:17:27–08:28:28 PDT)

B35 rebuilt on main with KV8a (`80938f4d`, the merge of main into B35) against base `fac209c2` (main, KV8a merged), the same `gate.py`
and `grade.py`, driver 595.91.07. Every row green at both sizes (record `~/goinfer-bench/b35-e2e/results-2026-10-21-0817/` on nobara).
Francis's yes for it: 2026-10-21 07:50 PT, "yes to all four as recommended".

| size | turn | time to first token | reused | first e2e | cold control |
|---|---|---|---|---|---|
| 4096 | 2 | 2.41 s | 3,584 of 4,130 | 2.36 s | 15.86 s |
| 4096 | 3 | 2.43 s | 3,584 of 4,150 | 2.48 s | |
| 8192 | 2 | 3.06 s | 7,680 of 8,177 | 3.05 s | 39.66 s |
| 8192 | 3 | 3.17 s | 7,680 of 8,197 | 3.19 s | |

I1 (warm = cold), I2 (cold output equal to the KV8a base build), I3 and R1 held; T1 at 1.01–1.17 of the predicted cost (bar 1.5).
On the same combined build KV8a's gate 2(a) tests passed (08:15:15–08:16:27). The job's one failure was
`TestNemotronResidentCUDA_guards`, which pinned the pre-B35 snapshot rule (a Gated-DeltaNet prefix snapshot declined); its check was
updated to B35's rule on Francis's word ("yes to the B35 test fix with the full CUDA package run", 10:11 PT), and the full CUDA package
then passed at `d190005f` (cuda-device-suite, 10:14:28–10:26:06: 742 passed, 0 failed, 0 device skips). #112 merged as `5cb1925d`.

