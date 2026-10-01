# Qwen tool declarations in the model's own bytes — pre-registration (2026-09-30, before arm B exists)

**Question.** goinfer's Hermes tool renderer writes each declared tool inside `<tools>` as sorted-key compact JSON, HTML-escaped:
`{"function":{"description":…,"name":…,"parameters":…},"type":"function"}`. The Qwen2.5, Qwen3 and Qwen3.5 chat templates write
`{"type": "function", "function": {"name": …, "description": …, "parameters": …}}` — `tool | tojson` under HuggingFace's environment: Python's
separators, this key order, the schema's own key order, no HTML escaping. Same content, different bytes. Does writing the model's own bytes help,
hurt, or do nothing to a tool-using model?

**Arms** (two binaries from the SAME commit, differing only by `-ldflags -X …/chat.toolDeclarations=`):

| | declarations | where |
|---|---|---|
| **A** | goinfer's current compact form (`compact`, the default) | every ChatML family |
| **B** | the template's bytes (`template`) | only where the template's own text writes `tool | tojson` (Qwen2.5 / 3 / 3.5) |

Everything else in the prompt is identical in the two arms — in particular the `-tool-format` is `hermes` in both, the instruction text, the
call wrapper and the history rendering are untouched. (Two other differences from the Qwen2.5 template — a blank line before a call with no
text, and the default system prompt — are NOT changed in either arm.)

**Instrument.** `scripts/tool_format_probe.py --samples 3 --loop --max-tokens 600` (`--thinking-budget 400` for Qwen3.5-9B, whose template
default is thinking on): 8 prompts, 7 that need a tool and 1 control, × 3 samples (greedy; T=0.7 with seeds 1001, 1002); each correct call is
replayed with a canned result and the second reply must be prose that uses it. Per model and arm, over the 21 tool-prompt replies:
**C** correct calls, **E** loops answered, **K** control prompts answered in prose (of 3). Models, `serve -backend cuda`, default quant:
Qwen2.5-Coder-0.5B-Instruct q4_k_m, Qwen2.5-7B-Instruct q4_k_m, Qwen3.5-0.8B, Qwen3.5-9B. Both arms are re-measured in the same session on the
new binaries; earlier arm-A numbers are not reused. The load log must say, for arm B, that declarations are in the template form.

## Decision rule (fixed now)

- **REJECT** if on ANY model B.C ≤ A.C − 3, or B.E ≤ A.E − 3, or B.K ≤ A.K − 2.
- **ADOPT** if on EVERY model B.C ≥ A.C, B.E ≥ A.E and B.K ≥ A.K: no regression anywhere. (Parity is enough: the template's bytes are what the
  model was trained to read, and goinfer's compact form has no evidence behind it either.)
- **PARK** otherwise, including every result the numbers do not land in.

What each does to the code. ADOPT: the default becomes the template's bytes for those templates, and the experiment variable is removed.
PARK or REJECT: the default stays and the B implementation is deleted (it is a few lines and the result is recorded), not left behind a
switch no operator can reach. Thresholds are not moved after the run; the first complete run under these settings stands.

## What this does not decide

Prompts beyond these eight and tools beyond simple schemas (nested objects, enums, long descriptions); a description containing `<`, `>`, `&`
or `'` (the escaping differs in B and the probe does not exercise it); Qwen3 (no checkpoint to run); the two other differences above.
