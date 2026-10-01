# Gemma 4 canonical template: where an assistant message's text goes beside a tool call — pre-registration (2026-09-30, before either arm runs)

**Question.** goinfer's own tool rendering for Gemma 4 writes an assistant message's text BEFORE its call and leaves the model turn open.
The canonical Gemma 4 chat template writes it AFTER the tool results and closes the turn, and writes no turn header after a tool result
(`-tool-format template`; ported byte for byte, matching HuggingFace on every golden case). For an agent client that replays a turn with a
preamble ("Let me check…" plus a call), the template's prompt ends `…<tool_response|>text<turn|>` with nothing after it. Which prompt gets
better follow-up answers from the model? The default flips to the template's only if it does not do worse.

## Setup (amended 2026-09-30 before any arm ran; the thresholds below are untouched)

No local checkpoint pairs a small model with the canonical template, with two exceptions found while setting this up. The E2B GGUF carries the
OLDER template (goinfer serves it generically: `thinking: unmanaged`) and the 26B's `.giw` bundle has none (built from safetensors; same
`unmanaged`). My first plan — the E2B safetensors with the 26B's `chat_template.jinja` copied beside it — cannot load: goinfer does not
implement safetensors loading for Gemma 4's per-layer-embedding inputs (`this checkpoint has per-layer-embedding (PLE) inputs … GGUF loads
it`). The model is therefore **Gemma-4-12B-it QAT q4_0 (`~/models/gemma-4-12b-gguf/gemma-4-12b-it-qat-q4_0.gguf`)**, whose embedded template is
the canonical one (checked: "Canonical Chat Template", `enable_thinking | default(false)`, `thinking_gate` all present). It is not the 26B and
nothing below speaks for it. The load log must say `thinking: template default off` (managed) for the run to count; `unmanaged` voids it.

## Arms — the same binary, differing only by flag

| | flag | renders |
|---|---|---|
| **A** | `-tool-format hermes` (the default) | goinfer's own order: text before the call, turn left open |
| **B** | `-tool-format template` | the canonical template's order |

## Instrument

`scripts/tool_format_probe.py --samples 3 --loop --preamble "Let me check that for you." --max-tokens 600`: 8 prompts (7 need a tool, 1 control)
× 3 samples (greedy; T=0.7 with seeds 1001, 1002); each correct call is replayed with a canned result AND the preamble text beside the call,
and the second reply must be prose that uses the result. Per arm, over the 21 tool-prompt replies: **C** correct calls, **E** loops answered,
**K** control prompts answered in prose (of 3). The preamble is the one thing that differs between this and the Qwen probe: it is what puts text
beside the call, so it is what the two orders disagree about.

## Decision rule (fixed now)

- **ADOPT** (the default becomes the template's order for the canonical Gemma 4 template) if B.E ≥ A.E, B.C ≥ A.C and B.K ≥ A.K: no regression
  on any of the three. Reason for adopting on parity: the template is the contract everywhere else in this repo, and goinfer's order has no
  evidence behind it either.
- **PARK** (default stays; the flag stays) if B.E is 1 or 2 below A.E, with B.C ≥ A.C − 1 and B.K ≥ A.K − 1.
- **REJECT** (default stays; and the doc says the template's order is measurably worse on this model) if B.E ≤ A.E − 3, or B.K ≤ A.K − 2.

A result the numbers do not land in falls to PARK. Thresholds are not moved after the run; the first complete run under these settings stands.

## What this does not decide

Anything about the 26B or any Gemma 4 size but this one; long transcripts; a client with its own long system prompt; thinking on (the
template's default for this checkpoint is off, and so is the probe).
