# Gemma 4 canonical template: text beside a tool call — result (2026-10-01)

Rule: `PREREGISTERED.md` (committed before either arm ran; setup amended once, before any arm, when the E2B safetensors route proved unloadable). Run as
the night job `gemma4-text-order` (`run-ab.sh`, binary pinned at `c09fe1e3`), 2026-10-01 00:15–01:06 PDT, 51 minutes, on **Gemma-4-12B-it QAT q4_0**
(its embedded template is the canonical one) on the CPU backend. Both arms' load logs say `thinking: template default off` (the managed path); arm A
says "tools: goinfer's own form", arm B "the model's own template form". Results: `results/` here.

| arm | what it renders | C correct calls (of 21) | E loops answered (of 21) | K control in prose (of 3) |
|---|---|---|---|---|
| A (`-tool-format hermes`) | goinfer's own order: text before the call, turn left open | 21 | 21 | 3 |
| B (`-tool-format template`) | the canonical template's order: text after the results, turn closed, no header after a result | 21 | 21 | 3 |

**Verdict: ADOPT** (`ab_verdict.py`, exit 0): B.E ≥ A.E, B.C ≥ A.C, B.K ≥ A.K. The two arms agree on all 24 replies (kind, correctness and loop outcome); the
follow-up wording differs in 9 of the 24 (sampled replies), so the prompts did differ and the model's answers landed in the same place.

**What this does and does not show.** It rules out the worry that motivated the measurement — that a replayed agent turn ending
`…<tool_response|>text<turn|>` with no turn header would confuse the model — for this model, with a one-line preamble, on eight prompts, at T=0 and T=0.7
with fixed seeds. It does not show the template's order is *better* (the arms tie), and it says nothing about the 26B (no template in its `.giw`), longer
transcripts, or a client with a long system prompt. The rule adopted on parity because the template is the contract everywhere else in this repo and
goinfer's own order had no evidence behind it either.

**Consequence, as registered.** The template's order becomes the default for canonical Gemma 4 (the `-tool-format` default is now per-family: `auto`);
`-tool-format hermes` keeps goinfer's earlier order as an escape hatch.
