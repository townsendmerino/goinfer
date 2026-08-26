# dsh × goinfer Tier-0 run log — 2026-08-25 (mac)

Queue G14. The findings from the Tier-0 end-to-end attempt, recorded as they happened. The recipe
in the scoping doc may only be written from this run (claim-discipline rule 7), so this file is the
evidence and the recipe is downstream of it.

> **INCOMPLETE — the qualifying agent run is still in flight as of this commit.** Everything below
> is recorded; the multi-turn agent task and the pass-2 run on `gemma4-26b-int4` are not yet done,
> and **no recipe may be written from this file until they are** (that is the Tier-0 gate). Committed
> mid-run deliberately, so the findings are not sitting only in a terminal.

**Box:** MacBook, Apple Silicon, darwin 25.6.0. **goinfer:** `4ca19e9` + queue commits, built from
`cmd/serve`. **dsh:** `@deepseek-ai/dsh@0.1.1-rc.2` (the current published version; the scoping doc
read the repo at 2026-08-21). **node:** v25.1.0, **npm:** 11.11.0.

**Model, pass 1 (wire-level shakeout):** `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, `-quant int4`,
`-ctx 32768`, CPU backend. Chosen because an iteration costs seconds; the qualifying run is meant
to be pass 2 on `gemma4-26b-int4`. Nothing at the scoping doc's recommended bar (Qwen3.8-27B,
gpt-oss-20b, DeepSeek-lite class) is on this box.

## Pass 1 — goinfer's OpenAI surface, driven directly. GREEN.

Serve came up in 5s. `/health` and `/v1/models` both report the model with its decode path
(`cpu (int4)`, batched prefill), which is more than the OpenAI spec requires and is useful to a
harness author.

**The G12 alias works on the wire.** With a `role: "developer"` system prompt, a
`role: "system"` one, and the same text as a `user` turn, all three returned identical completions
at `temperature: 0` and identical `prompt_tokens` (32).

> **The `user` row matching is the finding, not a footnote.** This probe cannot discriminate role
> *placement* on this model — the 1.5B ignores the instruction in all three forms, so identical
> output across all three is consistent both with "the alias works" and with "roles do not matter
> here". The alias is proven byte-identically at the unit level instead
> (`TestDeveloperRoleAliasesSystem`, mutation-checked both directions). **Do not use the 1.5B to
> validate anything semantic** — it can only show that bytes move.

**Tool calling: GREEN, first try.** A `tools` request with a `developer` system prompt returned
`finish_reason: "tool_calls"` and a well-formed call (`get_weather`, `{"city": "Paris"}`, an `id`,
`type: "function"`, `content: null`). No schema-shape friction of the kind the scoping doc
anticipated.

**Tool-result round trip: plumbing GREEN, model RED.** Feeding the `assistant` tool_call + `tool`
result back produced *another identical tool call* instead of an answer — the classic agent-loop
failure. `prompt_tokens` rose 162 → 203, so the history did reach the prompt. Rendering was then
inspected directly rather than inferred, and it is the correct Hermes/Qwen2.5 shape:

```
<|im_start|>system … # Tools … <tools>{…}</tools> … <|im_end|>
<|im_start|>user  What is the weather in Paris right now?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp_c": 14, "conditions": "light rain"}
</tool_response><|im_end|>
<|im_start|>assistant
```

So the re-call is model weakness, **not** a goinfer defect. This is the discrimination the recipe
has to help a reader make, and the reason pass 1 exists: had this been found first at 26B, it would
have cost a long re-run to reach the same conclusion.

## Pass 1 — dsh itself. BLOCKED, on dsh's side.

**`npx @deepseek-ai/dsh web` does not work on this setup, and neither does `--help`.** `npm exec
@deepseek-ai/dsh@0.1.1-rc.2 --help` spun at ~100% CPU and 1.6 GB RSS for **6m16s of CPU time
without producing a byte of output**, and ignored SIGTERM (SIGKILL required). The npx cache
directory for the invocation was left holding only a `concurrency.lock` — the install never
completed, so this is npm's dependency resolver, not dsh's code.

The likely mechanism is the shape of the dependency graph: `dsh` declares **~30 first-party
`@deepseek-ai/dsh-*` packages all pinned at `^0.1.1-rc.2`**, plus `@deepseek-ai/cordis`. A large
graph of prerelease ranges is the classic pathological case for npm's SAT resolver, and prerelease
ranges are exactly what a developer-preview publishes.

**Resolution, and the mechanism, measured.** `npm install --legacy-peer-deps` reached **184
packages while the plain `npm install` was still at zero** at the same wall-clock, and completed.
The cause is peer-dependency resolution: the `@deepseek-ai` packages declare **1063 peer-dependency
edges** between them, all on prerelease ranges. That is the graph npm's resolver cannot get through.

**But `--legacy-peer-deps` produces a BROKEN install**, because it skips peer deps and cordis
plugins *are* peers: `dsh --help` then died with `ERR_MODULE_NOT_FOUND: Cannot find package
'@deepseek-ai/cordis-plugin-group'`. Nineteen peers were missing. Installing those nineteen
explicitly (also with `--legacy-peer-deps`) reached closure in 1s, and `dsh --help` then worked.

So the working install is two commands, neither of which is the documented one:

```
npm install @deepseek-ai/dsh@0.1.1-rc.2 --legacy-peer-deps
npm install --legacy-peer-deps <the 19 peers it skipped>   # see the recipe for the list
```

## Pass 1 — dsh driving goinfer. The provider seam.

Config lives at `$DSH_HOME/settings.yaml`, and three details cost a cycle each:

1. **`providers` is a dict keyed by provider route, not an array.** The scoping doc recorded the
   array shape on 2026-08-21. rc.2 rejects it with migration directions — so the doc was already
   stale after four days, which is the churn its Tier-1 precondition (b) was written about.
2. **The section is namespaced `llm-pi-ai:`**, not top-level `providers:`. A top-level block loads
   without complaint and then fails at request time with `NO_ADAPTER: no adapter registered for
   provider "goinfer"` — a silent-until-used misconfiguration.
3. **`apiKeyEnv` is REQUIRED for a hand-declared route**, despite the README saying an omitted one
   leaves the route unauthenticated. Omitting it fails with `PI_AI_ERROR: No API key for provider`.
   goinfer on loopback needs no key, so the recipe must tell readers to set `apiKeyEnv` to any
   variable and give it any non-empty value. The scoping doc's "any apiKeyEnv — loopback needs no
   key" was right in spirit and wrong in shape: it is not optional.

Pointing the agent at the route needs a second file — a `--patch` overlay setting
`agent-default-model`'s `provider`/`model` — because the default is `deepseek-official` /
`deepseek-v4-flash` and that is plugin config, not a settings section.

## The compat finding that justifies G12, from dsh's own README

`@deepseek-ai/dsh-llm-pi-ai`'s README states outright that for an endpoint pi-ai does not
recognize, detection **answers as though it were OpenAI itself**:

> a reasoning model's system prompt goes out as `developer`, the output cap as
> `max_completion_tokens`, the thinking level as a bare `reasoning_effort` — and most
> OpenAI-compatible gateways reject at least one of those.

goinfer is an endpoint pi-ai does not recognize. All three were checked against the running server:

| what pi-ai sends to an unrecognized endpoint | goinfer at `4ca19e9` |
|---|---|
| system prompt as `role: "developer"` | **aliased to `system`** — G12, landed today |
| `max_completion_tokens` instead of `max_tokens` | **already supported** (`openai.go`, preferred when set) |
| bare `reasoning_effort` | **accepted and ignored** — no `DisallowUnknownFields` on the request path |

Verified live in one request carrying all three: HTTP 200, correct completion, `finish_reason:
"length"` at the requested cap of 8.

**This is the strongest available answer to "was G12 worth sequencing first".** dsh does not send
`developer` only for DeepSeek's own reasoning models — it sends it to *any endpoint pi-ai cannot
identify*, which is every goinfer deployment. The compat flag was a per-model escape hatch for a
default that fires on our endpoint by construction. "Most OpenAI-compatible gateways reject at
least one of those" — goinfer now rejects none.

## What this already tells the recipe

1. **goinfer's side is not the friction.** Every goinfer-side thing the scoping doc worried about
   — developer-role, tool-call schema shape, streaming, the tool-result round trip — was green on
   first contact. The one red was the model, and it was diagnosable in one render.
2. **The recipe must pin dsh's version and name the install path that actually worked**, not
   reproduce `npx …` from dsh's README. A preview that hangs a package manager is precisely the
   churn the scoping doc's Tier-1 precondition (b) was written about, showing up in Tier 0.
3. **Model guidance needs a floor, stated as a floor.** "≥32k context" is not sufficient: the 1.5B
   has 32k and cannot hold a tool loop. The recipe should say what a model must *do* (return a
   tool call AND then answer from its result), and say that a model which re-calls the same tool is
   the model's limit, not the server's — with the render check above as how to tell.
