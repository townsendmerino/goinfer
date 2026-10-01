# Qwen3.5 tool-call prompt format — result (2026-09-30)

Decision rule: `PREREGISTERED.md`, committed (`a4e32c1c`) before arm B existed. Arm B ran from the build `f9ae61c6` with `-tool-format template`;
arm A's reproduction ran from the same binary with `-tool-format hermes`. Probe, prompts, seeds, flags: as registered. `serve-cuda`,
`-backend cuda`, default quant, one request at a time.

| | C: correct calls (of 21) | E: loops answered (of 21) | K: control in prose (of 3) |
|---|---|---|---|
| **0.8B, A** (Hermes JSON, goinfer's prompt) | 14 | 9 | 3 |
| 0.8B, A rerun on the B binary (`-tool-format hermes`) | 14 | 9 | 3 |
| **0.8B, B** (the model's own template) | 12 | **11** | 3 |
| **9B, A** | 21 | 20 | 3 |
| **9B, B** | 21 | **21** | 3 |

**Rule applied.** 9B guard (B.C ≥ 20, B.E ≥ 19, B.K = 3): holds (21, 21, 3). 0.8B ADOPT threshold (B.E ≥ 12, B.K = 3): **11 — not met.**
0.8B B.E is in the PARK band (8 to 11). **Outcome: PARK.** `-tool-format hermes` stays the default; the native renderer and the XML parser
stay in the tree, behind the flag.

**What the numbers say, beyond the verdict** (not part of the rule, and not grounds to move it):
- The native form reads its own history better: of the calls that were made, 11 of 12 loops were answered on the 0.8B (A: 9 of 14) and 21 of
  21 on the 9B (A: 20 of 21). The one case where the model took the result and called again (`pick` s2) is the same behaviour A shows five times.
- On the 0.8B it makes the model call less often: 12 correct calls against 14. The extra prose replies are refusals ("I cannot access files
  on your local system") and the model answering a tool question from its own knowledge, on `calc`, `read` and `multiline`. The JSON prompt's
  gain there is real but is the instruction being more insistent, not the model understanding the tool better.
- Neither arm ever produced an unparsed `<function=` or `<tool_call>` in the text: 0 of 96 replies across both arms and both models. The
  suspected defect — the model writes XML and the call is dropped silently — did not occur under either prompt on these 8 prompts.
- 0.8B misses ADOPT by one reply (11 vs 12) on 21 greedy-and-seeded samples. That is inside what 21 samples can resolve, which is the
  reason the rule has a PARK band and a threshold set before the run.

**What was and was not measured.** Eight prompts, seven tools, one-turn calls and one follow-up turn, two models, fixed seeds. Not measured:
long agent transcripts, large tool sets, a client with its own long system prompt (Claude Code), other Qwen3.5 sizes, or Qwen3-Coder (its
template is not the layout the native renderer is pinned to, so it keeps the Hermes renderer whatever the flag says).

**Open, per the registration.** Forced tool_choice under the native form: it has no JSON wrapper, so a named `tool_choice` is a 400 and
required/auto run unconstrained. That cost is real (the Hermes form constrains a call so it cannot be malformed) and is one more reason the
default does not move.
