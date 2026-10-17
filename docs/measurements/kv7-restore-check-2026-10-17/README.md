# Is a restored prefix the cold prefix? The pilot's failing requests, replayed (2026-10-17)

Tier 1, by day, GPU runs under the timing lock. Question (owner, 2026-10-17): the three tool-calls-as-text replies of tack's E1 pilot job 1 each came on the first
large request of a run, which restored a 1,792-row snapshot (`snap=1792`). Is a restored prefix the cold prefix, on the pilot's own binary, model and settings?

## Setup

- Binary `goinfer-serve-cuda-0e598d3f` (the pilot's), `Qwen2.5-7B-Instruct q4_k_m` (the `.int4.e4h.cuda.v17.giw` the server picks), `-backend cuda -kv-sessions 1 -ctx 32768 -log-requests`,
  nobara-pc, RTX 2070 SUPER, driver 595.91.07.
- The requests are opencode 1.19.0's own, captured through a recording pass-through proxy (`proxy.py`, `capture.sh`, `run-t05.sh`) from tack's pilot tasks 1, 2 and 5 (arm A0k for 1 and 2, A0 for 5).
  Task 5's is the request after the compaction: tack's session database with the garbage reply and its trigger message deleted, then the same synthetic continue message sent. Bodies are not committed
  (they carry tack's task text); the SHA-256 of the ones replayed:

  | body | sha256 | prompt ids |
  |---|---|---|
  | task 1 first request (B) | `c00ef0554a8c30bb…` | 15,008 |
  | task 2 first request (A, the snapshot's source) | `06a2793d1b8f05336…` | 20,177 |
  | task 1 title request (C, flushes the live slot) | `97521837622ca391…` | 821 |
  | task 5 post-compaction request | `fa40d999e67ed2f2…` | 15,907 (the pilot's failing one: 15,925, a longer working-directory path) |

- Restored arm: A (cold, takes the snapshot) → C (the title request, so the one slot no longer holds A) → B. The server logs `kv snapshot: restored 1792 rows of 20160` and `snap=1792`, as in the pilot.
  Cold arm: `-kv-host-bytes 0`, B alone on a fresh server. Greedy (`temperature 0`), non-streamed, `-log-token-ids` for the ids.

## Readings

1. **First differing token: none.** Task 1: restored ×2 and cold ×2 give the same 61 greedy ids and the same prompt hash (`run-arms.sh`, `compare_ids.py`). Task 5, the reply that came back with garbage markers:
   restored ×2 and cold ×2 give the same 112 ids, the same hash, a normal reply and a valid tool call (`run-t05-arms.sh`). No restored reply was garbage.
2. **Largest logit difference at the first generated position: 0.** (Secondary: the tools route returns no logprobs, so these use tool-free copies of the same two requests, the tools block
   written into the system message as text, 14,220 ids; not byte-identical to the pilot's prompt.) Top-20 logprobs and the chosen token's are bit-identical, restored against cold, cold against cold, at
   pass width 512 and at 256 (`run-arms3.sh`, `compare_lp.py`).
3. **Pass alignment.** The default pass is 512 rows. The cold prefill's passes start at 0, 512, 1,024 …; the restored one starts at 1,792 = 3.5 × 512, so its passes are 1,792, 2,304 …: **they do not line up
   at the default width**. At `GOINFER_PREFILL_CHUNK=256` they do (1,792 = 7 × 256). Both widths read bit-identical, so bit-identity does not need the alignment on this model, consistent with the pass-width
   invariance KV0 recorded (`docs/measurements/kv0-invariance-2026-10-13.md`) while the kernel class (a function of the whole prompt's length and the pass's rows) is the same. The tier 0 test pins the
   aligned case, and pins it on the tiny fixture only: `cuda/kvsnap_cold_test.go:TestKV7_restoreMatchesColdAtPassBoundaries`.

## Verdict

Restore is cleared for the pilot failures: on the three requests that restored, the restored and cold runs are the same bits where it was measured (4 greedy runs of task 1, 4 of task 5, 6 first-position
distributions). The garbage in task 5 is therefore not cache damage. The sampling the pilot ran with is the next suspect: opencode sent no `temperature`, `top_p` or `top_k` (the captured bodies carry only
`model, max_tokens, messages, tools, tool_choice, stream, stream_options`), and `internal/serveapp/openai.go:prepare` then samples at temperature 1.0 over the whole vocabulary. That is a reading of the code
and of the bodies; whether it produced `ồng` and `tolua` was not replayed (a sampled run is not reproducible without its seed).

## Limits

One model, one snapshot source (task 2's prompt), 1,792 rows. Not the Mellum windowed-ring path (that has its own gates: the KV7 and KV7-step-2 tests). A sampled run was not compared.
