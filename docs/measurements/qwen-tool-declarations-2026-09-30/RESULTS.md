# Qwen tool declarations in the model's own bytes — result (2026-09-30)

Rule: `PREREGISTERED.md`, committed (`ccf51aba`'s parent) before arm B existed. Two `serve -tags cuda` builds of commit `ccf51aba`, differing only by
`-ldflags -X …/chat.toolDeclarations=template`; each arm's load log carries (B) or lacks (A) the line "tool declarations: the template's bytes"
(`verdict.py` checks it). `--backend cuda`, default quant, one request at a time, probe as registered. Both arms measured in the same
session; fixed seeds. The whole sweep took about 5.5 minutes (8 runs); the estimate made beforehand, 38 minutes, was anchored on a first 9B load
and was about seven times too high.

| model | A: C / E / K | B: C / E / K |
|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct q4_k_m | 7 / 2 / 3 | 8 / 0 / 3 |
| Qwen2.5-7B-Instruct q4_k_m | 21 / 21 / 3 | 21 / 21 / 3 |
| Qwen3.5-0.8B | 14 / 9 / 3 | 17 / 13 / 3 |
| Qwen3.5-9B | 21 / 20 / 3 | 21 / 18 / 3 |

(C correct calls and E loops answered, of 21; K control answered in prose, of 3.)

**Verdict: PARK** (`verdict.py`, exit 3). REJECT needed a drop of three on C or E (or two on K) on some model: the largest drops were two
(0.5B and 9B on E). ADOPT needed no model to be lower on any of C, E, K: the 0.5B and the 9B were each lower on E. Applied as registered; the
thresholds were not moved.

**What the numbers say, beyond the verdict** (not grounds to move it):
- The effect is not uniform. The smallest Qwen3.5 gained clearly (+3 correct calls, +4 follow-ups answered); the 7B did not move at all (the same
  21/21/3 in both arms); the 0.5B and the 9B each lost two follow-ups (0.5B: all eight of its loops called again; 9B: three calls again where
  arm A had one).
- 21 replies per model cannot separate a two-reply difference from the model changing its mind at a different sampled seed; the 0.8B's gain is
  the only difference here larger than that, and it is one model.
- None of the four is "wrong bytes make the model fail": the 7B, whose tool use is strongest, is indifferent to the formatting. That is itself
  the finding the registration was written to be able to report: goinfer's compact form was never a measured liability.

**Consequence, as registered.** The default stays goinfer's compact declarations. The B implementation (`chat/tool_decls.go`, the
`pyDeclarations` template field and detector, the renderer's extra parameter, the load-log note, and the golden test of the bytes) is deleted;
`qwenToolJSON`, which the native Qwen3.5 form uses, stays. The HF-rendered declarations are still in `testdata/chat_goldens/tools_qwen25_grouped.json`
(including the `declare_special` case with its own tool list), so a later attempt starts from a verified oracle.

**Not measured.** Prompts beyond these eight, nested or enum-heavy tool schemas, descriptions containing `<`, `>`, `&`, `'` (the two arms
escape differently; the probe never exercises it), Qwen3 (no checkpoint), and the other pre-existing difference (a blank line before a call with no
text), which is a history-rendering question and would need its own run.
