# Qwen3.5 tool-call prompt format — pre-registration (written 2026-09-30, before arm B exists)

**Question.** goinfer prompts Qwen3.5 for tool calls in the Qwen2.5/Hermes form (a system block of function signatures, "return a json
object … within `<tool_call>`"), and parses `<tool_call>{json}</tool_call>`. The model's own chat template asks for a different form:
`<tool_call><function=NAME><parameter=K>value</parameter></function></tool_call>`. Should serve render the model's own form?

**Why this is a question and not a fix.** The suspected defect — "the model emits XML and the call is dropped silently as prose" — is
**not** what arm A shows. With goinfer's JSON prompt neither model wrote an XML call once in 48 replies; they follow the instruction
in the prompt. So the native form is a change of prompt for a hypothetical gain, and it has to earn its place.

## Arms

| | what it renders | build |
|---|---|---|
| **A** (baseline, measured below) | goinfer's Hermes-JSON tool prompt and JSON call history | `a8bf4860` (`serve-cuda`, `-backend cuda`) |
| **B** | the model's own template, byte for byte (declarations, XML call history, grouped tool responses), with an XML call parser | the commit that adds it; the probe is unchanged |

Same models, same prompts, same seeds, same probe (`scripts/tool_format_probe.py`), same flags. Only the prompt format differs.

## Instrument

`scripts/tool_format_probe.py --samples 3 --loop`: 8 prompts (7 that need a tool, 1 control that does not) × 3 samples (greedy; T=0.7 with
seeds 1001, 1002), through `/v1/chat/completions` on a running serve. `--max-tokens 600` for Qwen3.5-0.8B (thinking off by its template
default); `--max-tokens 900 --thinking-budget 400` for Qwen3.5-9B (thinking on by its template default). A call is *correct* only when it names
the expected tool and every argument has the expected JSON type; with `--loop` each correct call is replayed with a canned tool result and the
second reply must be prose that uses it (*answered*).

Per model, over the 21 tool-prompt replies: **C** = correct calls, **E** = loops answered (end to end: right call, then a history the model
reads correctly), **K** = control prompts answered in prose (of 3). Sampling is seeded and the machine runs one request at a time, so arm A
reproduces exactly (0.8B measured twice: 14/21 and the same 14/21 replies).

## Arm A, measured 2026-09-30 (build `a8bf4860`; logs in this directory)

| model | C | E | K |
|---|---|---|---|
| Qwen3.5-0.8B | 14/21 | 9/21 | 3/3 |
| Qwen3.5-9B | 21/21 | 20/21 | 3/3 |

0.8B's misses: a hallucinated weather answer, a refusal, three `json`-fenced pseudo-calls in a code block, and one plain code answer.

## Decision rule (fixed now)

- **ADOPT** — serve renders the model's own form for every checkpoint whose template declares it — only if ALL hold:
  1. 9B does not regress: B.C ≥ 20/21 **and** B.E ≥ 19/21 **and** B.K = 3/3;
  2. 0.8B improves end to end: B.E ≥ 12/21 (A's 9 plus 3) **and** B.K = 3/3.
- **PARK** — keep A as the default, keep the native form and the XML parser in the tree, off by default — if the 9B guard (1) holds and
  0.8B's B.E is 8 to 11 of 21 (within one below to two above A), or B.K < 3/3 on 0.8B only.
- **REJECT** — remove the native renderer, keep only the XML parser — if the 9B guard (1) fails, or 0.8B's B.E ≤ 7/21.

A rule the numbers do not land in falls to PARK. The thresholds are written down before B runs and are not moved afterwards; if B is
re-measured, the first complete run under the registered settings stands.

## What this does NOT decide

- The XML **parser** is built regardless (it only affects output that today is dropped as prose, and cannot change a reply that parses
  now), so a model that emits the form — under another prompt, or Qwen3-Coder — is not silently ignored.
- Forced / required tool choice: the Hermes form has a JSON call grammar; the XML form would not. That is decided separately, after B.
- Anything about models other than these two, or about prompts beyond these eight.
