# Empty-answer reasons against live clients, 2026-10-14

Owner check (b): an empty answer that carries goinfer's empty-answer reason (docs/server.md, "Empty answers say why") must
not make a coding agent error out or retry-loop.

- **Setup.** nobara-pc, no GPU: `TestScriptedServe_forHarness` (internal/serveapp/scripted_serve_test.go) serves a scripted
  model on Mellum2.1's chat template whose every turn is `<think>\nThe file is unreadable, but Add surely returns 5.\n`
  and then the end of turn, so every reply is `ended_in_reasoning`. No real model ends a turn that way on demand. Clients
  as in the I3 harness, from `~/harnesses`: Codex 0.162.1 (`exec --json`, `/v1/responses`) and opencode 1.19.0 (`run
  --format json`, `/v1/chat/completions`), each with a fresh HOME, the `read` task prompt, a 90 s timeout and stdin from
  `/dev/null`. Run under the timing lock (label `responses-empty-live-scripted`). Logs: `~/goinfer-bench/responses-empty-live-2026-10-14/`.
- **Codex.** 1 request. The response was `status: "incomplete"` with `incomplete_details.reason: "ended_in_reasoning"`.
  Codex printed `agent_message` with text `""`, then `turn.completed`, and exited 0 within a second. No error item and no
  retry.
- **opencode.** 2 requests, both carrying `goinfer_empty_answer.reason: "ended_in_reasoning"` in the streamed finish chunk.
  The second is presumably the session-title request. opencode wrote `step_finish` with reason `stop` and exited 0 within
  2 s. No retry.
- **Verdict.** Neither client errors or loops on the reason. Neither shows it to the user either: both end the turn with
  an empty answer, as before the change.
- **An artifact of the first attempt, not of the change.** The scripted tokenizer counts one token per byte, so opencode's
  first prompt read as 44,487 tokens against the 32,768 window its configuration declared. opencode then compacted, and
  after each empty reply it sent its synthetic "Continue if you have next steps" message: about 1,200 requests in 2
  minutes, until the timeout. With the window raised to 1,000,000 (run 2, above) it did not compact and did not loop. Not
  checked: whether opencode loops the same way when a real compaction turn gets an empty answer without the field.
