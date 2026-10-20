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

## Result

(after the run)
