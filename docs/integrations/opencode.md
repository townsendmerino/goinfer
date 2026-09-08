# opencode → goinfer

opencode speaks the OpenAI-compatible chat API through an AI-SDK provider, so it points at
`/v1/chat/completions` via a project-local `opencode.json`. This page did not exist through three
releases even though the README named opencode alongside Claude Code as "a real agent" target
(R14, docs/measurements/cold-user-2026-09-07-macbook-arm64.md) — a cold user had to reconstruct
the config below from outside knowledge of opencode itself. This page exists so the next person
does not have to reconstruct it, and it now carries a real, measured success, not only failures.

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
every run this page draws on, that row's answer matched what opencode did:

```
tools, OpenAI .............  ok    call get_weather({"city": "Paris"}) → result → answer in 2 turns
tools, harness-scale ......  ok    call get_weather({"city": "Paris"}) among 12 tools
```

A `skip` here means stop and pick a different checkpoint — don't spend the wall-clock finding
out from opencode itself. An `ok` means proceed; it has predicted every real outcome measured so
far, including the one success below.

## The working configuration, measured

**Qwen2.5-7B-Instruct, `q4_k_m` → `--quant int4 --backend cuda --ctx 16384`, on a CUDA GPU with ≥8
GB VRAM** (measured on nobara-pc, Ryzen 3700X + RTX 2070 SUPER 8 GB, 2026-09-08, v0.17.2):

```
goinfer-serve -model coder=<path>/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384
```

- Load: 26.2 s, `decode path: cuda-resident (int4)`, weights 4.8 GB resident, **6,824 / 8,192 MiB
  VRAM used** — comfortable margin, not a near-miss.
- `serve check`: **all 8 checks passed**, including `tools, harness-scale`, for the first time any
  checkpoint in this project has recorded that.
- opencode's real "build" agent, given *"list the files in this directory using your tool, then
  tell me what's in notes.txt"*: called `Read notes.txt` and `Glob "*.txt"` — real tool calls, not
  prose — and answered correctly, matching the file's actual content. **Two turns, 7,165 and 7,399
  input tokens, 44 and 16 output tokens, VRAM unchanged (6,824 MiB) through the whole exchange.**

This is a **CUDA-VRAM** story, not a host-RAM one, and the two are governed by different guards.
`decoder/fitguard.go`/`prefill_budget.go` (R13/R13-follow-on) price *host* RAM and ran here too,
but were never the relevant gate: with 4.8 GB of weights this box's real host-RAM headroom was
never in question. The load-bearing check for this scenario is CUDA's own `checkKVFits`
(`cuda/resident.go`), which queries actual free VRAM via the driver and is what already measured
this exact card's ceiling (`docs/task-kv-cache-streaming.md`: this same RTX 2070 SUPER runs a
dense 7B at int4 fine at `-ctx 20000`, refuses at `-ctx 24576`) — `-ctx 16384` sits well inside
that, which is why the margin above is comfortable rather than tight. It has no live per-request
re-check (only the fixed cap `checkKVFits` validated at load), so something else competing for
VRAM *after* load is the one failure mode it does not see — check `nvidia-smi` before starting, the
VRAM analogue of the host-RAM `ps aux` audits used elsewhere in this document's sibling reports.

## What actually happened before this, and what it taught

No run in this project completed a full opencode tool-call turn end to end until the one above.
Three earlier attempts failed for three DIFFERENT reasons, and conflating them would have hidden
which one the eventual fix needed to target:

1. **nobara-pc (RTX 2070 SUPER, CUDA), Qwen2.5-Coder-1.5B-Instruct** (R11,
   docs/measurements/cold-user-2026-09-06-nobara-pc.md): opencode's real "build" agent printed a
   fake JSON tool call **as prose**, twice, instead of issuing a real one. Not a memory or
   networking problem — the model itself did not hold up under a harness-scale schema.
2. **MacBook, M1 Pro, 16 GB RAM, Qwen2.5-7B-Instruct q3_k_m** (R13/R14,
   docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the load-time check reported a
   comfortable-sounding `"79% of budget"`, then the real opencode turn pushed RSS to 14 GB and
   drove **heavy sustained swapping** (Swapouts +621,588 pages, ~9.7 GB, in under two minutes) —
   killed for machine safety before tool-calling could even be observed. Root cause: the guard
   priced against a fixed fraction of TOTAL host RAM, which a real machine with other work open
   can exceed regardless of the model's own footprint. Fixed in two passes
   (`docs/task-first-hour.md`, "R13-follow-on"), each re-verified live on the same Mac.
3. **Same Mac, third live re-run, after both fixes**: `Load` correctly *refused* this exact
   model — cleanly, zero Swapouts — because it genuinely does not fit that machine's real
   available memory under ordinary desktop load. Confirms the fix; does not reach opencode either,
   for an entirely different (capacity, not correctness) reason.
4. **Same Mac, Qwen2.5-Coder-3B-Instruct**: loads comfortably (2.4 GB margin, zero Swapouts,
   `serve check` clean) — proving memory was never the reason smaller sizes hadn't been tried —
   but `tools, harness-scale` skipped, same failure mode as attempt 1's 1.5B: too small to
   tool-call under a real schema. A capability gap, not a memory one.

**The pattern across all four: memory-fit and tool-calling capability are independent axes, and
this project had never had a data point that was true on both until nobara-pc's CUDA-resident 7B
above.** The Mac's real available memory tops out well under what a 7B model needs; nobara-pc's
GPU has real headroom to spare for exactly that size. Different hardware, different bottleneck.

## Picking a model, honestly

**Qwen2.5-7B-Instruct on a CUDA GPU with ≥8 GB VRAM is now a measured, working recommendation**,
not a suggestive data point — see "The working configuration, measured" above. Off that specific
hardware profile, the honest picture is narrower:

| checkpoint | hardware tried | tools (harness-scale, 12 tools) |
|---|---|---|
| `qwen2.5-coder-0.5b` | nobara-pc (CUDA) | **skip — too small** |
| Qwen2.5-Coder-1.5B-Instruct | nobara-pc (CUDA) | **skip — too small** (prose, not a real call) |
| Qwen2.5-Coder-3B-Instruct | Mac (CPU, fits comfortably) | **skip — too small** |
| Qwen2.5-7B-Instruct q3_k_m | Mac (CPU) | never reached — does not fit this Mac's real available memory |
| **Qwen2.5-7B-Instruct q4_k_m** | **nobara-pc (CUDA, 8 GB VRAM)** | **ok — real tool calls, real completion** |
| `phi3-mini-4k`, `granite-4.0-h-tiny`, `gpt-oss-20b`, `gemma-4-26b-a4b` | — | not yet measured |

Run `serve check` against whichever checkpoint and hardware you actually have before pointing
opencode at it — it has predicted every outcome in this table, including the success. If your
hardware is CPU-only or a smaller GPU, the honest state is: nothing between 3B and 7B has been
tested, and 7B has only been shown to fit on a discrete GPU with real VRAM headroom, not on a
16 GB Mac's host RAM.

## Retiring this page

Per `docs/task-embed-and-harness-ux.md` §3.5, a recipe is retired when `serve check` covers what
it says. It now covers the tools-schema prediction *and* that prediction has been confirmed
against a real, completed opencode-driven multi-turn loop with real token/VRAM numbers — the two
things this page previously said were still missing. What is not yet covered: a registry
checkpoint (rather than an ad hoc pull) with a measured harness-scale `ok`, and any result on
CPU-only or smaller-GPU hardware. This page stays until those exist, but its central claim — that
opencode can work against goinfer at all — is no longer open.
