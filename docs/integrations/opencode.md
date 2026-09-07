# opencode → goinfer

opencode speaks the OpenAI-compatible chat API through an AI-SDK provider, so it points at
`/v1/chat/completions` via a project-local `opencode.json`. This page did not exist through three
releases even though the README named opencode alongside Claude Code as "a real agent" target
(R14, docs/measurements/cold-user-2026-09-07-macbook-arm64.md) — a cold user had to reconstruct
the config below from outside knowledge of opencode itself, and that reconstruction produced the
run's only safety incident (see "What actually happened" below). This page exists so the next
person does not have to reconstruct it.

```json
{
  "provider": {
    "goinfer": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": { "<served-model-name>": {} }
    }
  }
}
```

`<served-model-name>` is whatever `goinfer-serve -model <name>=<path>` names it (default: the
file/dir basename). Point opencode at `goinfer/<served-model-name>`.

**Run `serve check` before opencode, not after.** `goinfer-serve check <url>`'s "tools,
harness-scale" row sends a dozen-tool schema shaped like opencode's own "build" agent — the same
shape a real opencode turn sends — and reports `ok` or `skip` *before* you configure anything. In
both runs this page draws on, that row's answer matched what opencode did:

```
tools, OpenAI .............  ok    call get_weather({"city": "Paris"}) → result → answer in 2 turns
tools, harness-scale ......  skip  model did not call the tool under a harness-scale (12-tool)
                                    schema — model answered without calling the tool
                                    (finish_reason="stop") — too small for tool use, or its
                                    template has no tool section
```

A `skip` here means stop and pick a different checkpoint — don't spend the wall-clock finding
out from opencode itself.

## What actually happened, twice — and what is still open

No run in this project has yet completed a full opencode tool-call turn end to end. That is
stated plainly because both attempts so far failed for two DIFFERENT reasons, and conflating them
would hide which one you are actually protecting against:

1. **nobara-pc (RTX 2070 SUPER, CUDA), Qwen2.5-Coder-1.5B-Instruct** (R11,
   docs/measurements/cold-user-2026-09-06-nobara-pc.md): opencode's real "build" agent printed a
   fake JSON tool call **as prose**, twice, instead of issuing a real one. Not a memory or
   networking problem — the model itself did not hold up under a harness-scale schema. `serve
   check`'s harness-scale row predicts exactly this case, which is why the recipe above puts it
   first.
2. **MacBook, M1 Pro, 16 GB RAM, Qwen2.5-7B-Instruct q3_k_m** (R13/R14,
   docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the load-time check reported a
   comfortable-sounding `"79% of budget"` (8.9 GB resident estimate against an 11.2 GB budget).
   The actual opencode turn — its real system prompt plus tool schemas — pushed RSS to **14 GB**
   and drove the OS into **heavy sustained swapping** (Swapouts +621,588 pages, ~9.7 GB, in under
   two minutes; 577% CPU; no tokens produced). The run was killed for machine safety at ~182 s
   before tool-calling itself could even be observed. **This model was never actually
   disqualified as a tool-caller — it was never given the chance to try.**

The gap in case 2 was a real bug, since fixed: the load-time check priced KV cache at 0 for a
context that had not been explicitly pinned, so a request's *own* prompt size — the one thing
that actually varies per turn — was invisible to it. `AdmitPrefillMemory` (decoder,
`prefill_budget.go`) now prices KV + prefill scratch for the request's actual prompt and
`max_tokens` against the remaining budget, and refuses with a 413 **before prefill starts**
rather than letting the process page. The startup banner also now prints a `fit:` line stating
the context cap's own KV cost and what remains of the budget, at every load — not only when
something is already tight.

**This fix has not yet been re-verified live against the exact scenario that found it.** The
targeted re-run (this same MacBook, same 7B/q3_k_m model, same opencode two-turn task) is still
outstanding — track it against this page before treating case 2 as closed. Until that re-run
lands, treat a 7B-class model on a 16 GB machine as "the load-time number is no longer wrong, but
the actual behavior under a real agent prompt is not yet re-measured here."

## Picking a model, honestly

The closest evidence for a model class that *can* hold up under a real harness-scale tool schema
is [claude-code.md](claude-code.md)'s own table: Qwen2.5-7B-Instruct completed a real
`glob → read → answer` loop in 3 turns — but that was through the Anthropic Messages protocol
(`/v1/messages`), CUDA-resident with ample VRAM headroom, not through opencode's OpenAI-compatible
path. It is suggestive that the model class is capable; it is not the same claim as "this model
tool-calls correctly under opencode," which nothing has measured yet.

What `goinfer-chat models`' `tools:` line records per registry checkpoint today (2026-09-07):

| checkpoint | tools (minimal schema) | tools (harness-scale, 12 tools) |
|---|---|---|
| `qwen2.5-coder-0.5b` | ok | **skip — too small** (measured 2026-09-07, nobara-pc) |
| `phi3-mini-4k` | not yet measured | not yet measured |
| `granite-4.0-h-tiny` | not yet measured | not yet measured |
| `gpt-oss-20b` | not yet measured | not yet measured |
| `gemma-4-26b-a4b` | not yet measured | not yet measured |

None of the registry's own checkpoints have a recorded harness-scale `ok` yet. Until one does,
the honest recommendation is: run `serve check` against whichever checkpoint you are about to
point opencode at, and stop at the first `skip` rather than assume size alone predicts success.

## Retiring this page

Per `docs/task-embed-and-harness-ux.md` §3.5, a recipe is retired when `serve check` covers what
it says. It already covers the tools-schema prediction above; it does not yet cover an actual
opencode-driven multi-turn loop, RSS/swap behavior under a real agent prompt, or a registry
checkpoint with a measured harness-scale `ok` — this page stays until those exist.
