# B35-TL — does B35's kept state serve a new question after a tool loop? (registered 2026-10-24)

This check replaces KV8a-Q (`docs/tasks/task-kv-architecture-2026-10.md` § "KV8a-Q", closed on Francis's decision of 2026-10-24
3:33 pm). Gate 1 (`docs/measurements/kv8a-q-gate1-2026-10/README.md`) found two things:

- **The divergence:** a new question's request leaves the previous request right after the previous question.
- **What already serves it:** serve's `KVScope.Keep` names that point for every request of a question's tool loop, and B35
  keeps the recurrent state there on CUDA.

What gate 1 could not say, and this check measures, is whether the restore works end to end on the real model.

Written and committed before the run. The rules below do not move after the data.

## What the code reading predicts, stated before the data

- **The prediction:** the second question restores the state kept during Q1's prefill. `decoder/kvsnap.go:Model.kvPrefillMark`
  takes no new mark on a request that restored a snapshot (`acq.snapRows > 0`). So Q2's own point may never be kept, and the
  third question would restore Q1's point again.
- **What it would cost:** Q1's whole tool loop is re-prefilled at every later question.
- **How the check reads it:** R2 tests this prediction. Its outcome is reported as its own verdict (`Q1-ONLY`), not as a
  correctness failure.

## Design

- **Box:** nobara, RTX 2070 SUPER, under the timing lock (a `night.py` job run by the box's `night.py serve`), by day.
  - Model: `~/models/Qwen3.5-9B-Q4_K_M.gguf`, `-backend cuda -ctx 16384`, serve's defaults otherwise.
  - Thinking on (`enable_thinking`), every assistant turn's `reasoning_content` sent back: the case KV8a-Q was for (Codex,
    Claude Code, Flint's card). Gate 1 found the same divergence point with thinking off.
- **Build:** one `serve`, `main` at the pinned commit, built with `-tags cuda`. B35 is on `main`, and this check asks what
  `main` does.
- **Conversation** (`gate.py`):
  - a system prompt of about 4,096 tokens of notes, and two tools;
  - Q1, two tool calls each with a result of about 1,500 tokens, and an answer;
  - Q2, one tool call and its result, and an answer;
  - Q3.
  - The six requests a client sends are r1–r6. r4 (Q2) and r6 (Q3) are graded. The assistant turns are the client's history,
    scripted. Greedy, 24 new tokens.
- **Arms:**
  - **A:** one server, streamed. r1–r6 in order: time to first token, `usage.prompt_tokens`, `usage.prefill_reused_tokens`.
    Then a cold control of r6's length (a nonce first in the system prompt).
  - **B:** one server, `logprobs`. r1–r6 in order, the warm path.
  - **cold:** a fresh server each, `logprobs`, for r4 and then r6. This is the do-nothing arm: what the question costs with no
    kept state.
- **Bit for bit** means the served tokens and the exact text of each logprob.

## Rules (`grade.py`)

| check | rule | on failure |
|---|---|---|
| R3 | every cold request (the control, the cold arm) reused 0 tokens, and A's replies equal B's | VOID |
| I1 | B's r4 and r6 (restored) equal the cold arm's, every token and logprob | FAIL |
| R1 | r4 (Q2) reused > 0 tokens on A and on B. Neither the live state nor KV8a's end-of-prompt checkpoint lies at or before r4's divergence, so only a kept state can serve it | FAIL |
| R2 | r6 (Q3) reused more tokens than r4 did, on A and on B: the kept point moved past Q1's loop | Q1-ONLY |
| T1 | B35's rule: a graded request's time to first token ≤ 1.5 × (the cold control's time × rows still prefilled ÷ the control's prompt) + 0.25 s | 1.5–2.5×: PARKED for Francis; past 2.5×: FAIL |

- **The verdict:** VOID, then FAIL, then PARKED, then Q1-ONLY, then PASS. The first that fires is the verdict; every check is
  printed.
- **Reported beside it, with no bar:** each graded request's time to first token as a fraction of the cold control's, scaled
  to its length. KV8a-Q's sketched gate 5 ("falls by at least half its cold time") is not a rule here. At Q2 the reusable
  prefix is the system prompt, tools and Q1, and the whole loop after it is re-prefilled whatever is held, so that fraction
  depends on the fixture's proportions.
- **After a Q1-ONLY verdict,** the follow-up is to let a restoring request mark its own Keep. It is product code and is not
  started without Francis's yes.

## Pre-registration (`docs/measurements/noise-registry.md` §8)

- **Tier:** day. **Testing tier:** 1, one small real model.
  - Tier 0 already covers the pieces: `internal/serveapp/b35_keep_test.go`, `cuda/b35_snapshot_test.go`, and gate 1's route
    readings.
  - It cannot show the restore on a real tool-loop conversation, where three requests pass between the save and the restore.
- **Interleaving:** none needed. The bars are identity and a ratio against a cold control on the same server, as in B35's own
  gate.
- **Cell validity:** the job records the load average and `MemAvailable` at its start. Above 8.0 load it is void and re-run.
- **Reused readings:** B35's own end-to-end gate (`docs/measurements/b35-e2e-2026-10-20/README.md`, PASS) covered plain chat
  turns, and is not re-run.
- **Fit check:** the same model and flags as B35's gate, which passed on this box on 2026-10-20 at `-ctx 16384`. The job's
  first server start prints the `fit:` banner, and a refusal ends the job void before any graded request.
- **Never-swap:** not a memory gate. The servers' banners and the box's `free -m` at the start are recorded.
- **Instrument:** B35's gate, reused with a different conversation. It is the cheapest instrument that answers "does a
  restored question equal cold, and what does it reuse".
- **Stopping rule:** fixed. One pass, every arm.
- **Ambiguous band:** T1's 1.5–2.5×, parked for Francis.
- **Cost:**
  - 4 server loads (about 12 s each).
  - Cold prefills: r1 (about 4.1k tokens), the cold control and cold r6 (about 9k tokens each), cold r4 (about 7.6k), at about
    4 s per 1k tokens.
  - A and B's warm requests.
  - About 6 minutes, plus a 2-minute build. Queued at 15 minutes, timeout 40.

## How it runs

```sh
# on nobara, from ~/goinfer-runner (its checkout may predate this script, so it is read from the pinned commit):
python3 scripts/night.py add b35-toolloop --est 15 --by "mac-kapitsa, KV8a-Q replacement" --doc docs/measurements/b35-toolloop-2026-10/README.md -- \
  bash -c 'cd ~/goinfer-runner && git fetch -q origin && mkdir -p ~/goinfer-bench/b35-toolloop && git show <sha>:docs/measurements/b35-toolloop-2026-10/run-b35-toolloop.sh > ~/goinfer-bench/b35-toolloop/run.sh && bash ~/goinfer-bench/b35-toolloop/run.sh <sha>'
```

The record is `~/goinfer-logs/b35-toolloop-2026-10/results-<date>/` on nobara (`gate.json`, `grade.txt`, `run.log`, the server
logs). The verdict is copied here.
