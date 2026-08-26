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

---

# The agent run FAILED, and the failure is ours. Three findings.

The headless agent task (`"List the files in the current directory, then tell me how many there
are."`) never produced an answer. dsh's own session transcript
(`$DSH_HOME/sessions/*/session-*/session.jsonl.zstd`, zstd-compressed JSONL) gives the timeline:

| t | event |
|---|---|
| 0.1s | agent request sent — 4214-char system prompt, **25 tool declarations**, `maxTokens: 2048` |
| 3.7s | **session-title LLM call SUCCEEDS** — a small request answers fine |
| 300.1s | agent request: `pi-ai stream idle timeout after 300000ms`, **zero bytes received** |
| 300.6s | retry 1/5 (`TIMEOUT` is in dsh's retryable set) → times out again at 600.6s |
| 601.6s | retry 2/5 → killed by the harness wall clock |

The session-title call succeeding at 3.7s is the control that makes the rest diagnosable: goinfer
answers a small request promptly, so nothing is wedged.

## Finding 1 — the tool path buffers even when streaming, so a slow generation is INVISIBLE

`internal/serveapp/tools.go`, stated in its own comment: *"Tool decisions need the whole output, so
buffer (even when streaming)."* The whole generation is accumulated into a `strings.Builder`, and
only then is SSE opened and the entire message sent as one delta plus a finish. This is correct for
tool-call parsing and it is honest SSE — but it means **`stream: true` with tools declared emits
zero bytes until the generation completes**.

dsh's default `streamIdleTimeoutMs` is **300000ms**, and `TIMEOUT` is in its retry set. Any
generation slower than five minutes therefore cannot succeed through dsh, no matter how correct the
output is.

**Replayed the exact request** (same system prompt, same 25 tools translated to OpenAI function
tools, `stream: true`, 32887 request bytes) against a freshly restarted server:

```
FIRST BYTE at 1682.6s
TOTAL 1682.6s, 495 bytes
```

Every byte arrived at the end. 28 minutes to first byte against a 5-minute client timeout.

## Finding 2 — an abandoned request keeps running; PREFILL is not cancellable

`tools.go` passes `r.Context()` into `lm.drive`, so cancellation is wired correctly at the seam.
It does not help, because the time is spent in prefill, and prefill is one long call that does not
observe the context.

Measured twice, unambiguously:

- After dsh was killed, goinfer kept a core saturated and climbed from **14:59 → 19:53 → 22:01** of
  CPU time with no client attached.
- After a second client (the replay) was killed, goinfer was still burning at **47:38** of
  accumulated CPU.

Each of dsh's five retries opens a *new* generation while the abandoned one keeps prefilling, so a
single agent turn can pin the server for far longer than any client is waiting. **This is
reachable from an ordinary client that gives up** — no hostile input required.

## Finding 3 — CPU prefill is superlinear, and is not delivering a batched speedup

The server advertises `prefill path: batched (CPU forwardLayersN, one weight stream for the whole
prompt)`. Measured (system prompt of N filler words, `max_tokens: 1`, so the number is prefill plus
one token):

**`-quant int4`, dense 1.5B, CPU backend:**

| prompt_tokens | prefill+1tok | effective | vs previous point |
|---|---|---|---|
| 170 | 4.5s | 37.7 tok/s | — |
| 620 | 24.4s | 25.4 tok/s | 3.6× tokens → 5.4× time |
| 1520 | 99.9s | 15.2 tok/s | 2.5× tokens → 4.1× time |
| 3020 | **1587.1s** | **1.9 tok/s** | **2.0× tokens → 15.9× time** |

(The 4000-word point was abandoned: at this trajectory it is hours, and the shape is already
established.)

**This is not a smooth exponent — there is a CLIFF.** An earlier reading of the first two points
called it ~n^1.3; the full curve refutes that. Doubling the prompt from 1520 to 3020 tokens
multiplies the time by **15.9×**, and the effective rate collapses by 8× to below 2 tok/s. Something
falls off between ~1.5k and ~3k tokens; the first three points are merely bad, the fourth is a
different regime.

Two further things are wrong independent of the cliff. The absolute rate (~38 tok/s at its *best*)
is **the same order as this model's decode rate** — batched prefill is not buying the speedup the
label implies. And the process sits at **~105% CPU** — one core — throughout, on a machine with
many.

The agent request's ~8k-token prompt is far past the cliff, which is why a `max_tokens: 1` request
against it exceeded 600s.

**This is the root cause of the whole failure.** Findings 1 and 2 are what turn a slow prefill into
a silent, uncancellable, retry-amplified one.


---

# Experiment 1 (int8int8 comparison) — the cliff is INT4-SPECIFIC, and the prediction was wrong

The decisions doc named a cheap first experiment: re-run the prefill points at `-quant int8int8`,
on the theory that int4's per-token nibble unpacking was the suspect. It also registered a
prediction — that the 3020-token int8int8 point would land in the same ~1500s class as int4,
because the *decaying* int8 advantage (1.36x → 1.24x → 1.07x) said the dominant term at depth was
quant-independent.

**The prediction is refuted, and the reasoning behind it inverted at the last point.**

| prompt_tokens | `-quant int4` | `-quant int8int8` | int8 advantage |
|---|---|---|---|
| 170 | 4.5s (37.8 tok/s) | 3.3s (51.5 tok/s) | 1.36x |
| 620 | 24.4s (25.4 tok/s) | 19.7s (31.5 tok/s) | 1.24x |
| 1520 | 99.9s (15.2 tok/s) | 93.2s (16.3 tok/s) | 1.07x |
| 3020 | **1587.1s (1.9 tok/s)** | **334.9s (9.0 tok/s)** | **4.74x** |

| step | int4 | int8int8 |
|---|---|---|
| 170→620 | n^1.31 | n^1.38 |
| 620→1520 | n^1.57 | n^1.73 |
| 1520→3020 | **n^4.03** | n^1.86 |

int8int8 shows **no cliff**: it converges on ~n^2, which is simply what attention costs. int4 goes
to n^4 across the same step. The decaying advantage was real but was a fact about the *smooth*
regime only; it reversed into a 4.74x explosion precisely where int4 falls over.

**Both branches of the experiment were half right.** int8int8 is faster — but by a shrinking
constant, not multiples, through the smooth regime; and it is **also single-threaded**, which is the
escalation branch. But the cliff, the thing that actually made the agent run impossible, is int4's
alone.

**Two mechanism hypotheses died on this data**, both raised from the CPU-sample fingerprint and the
RSS: allocation-rate-driven GC, and per-head attention scores spilling L2/SLC. `runLayersFromEmbedN`
allocates the same ten K-sized buffers and the same K×(startPos+K) attention scratch **regardless of
quant**, so either story predicts a quant-independent cliff. There isn't one.

**Filed as three separable items** rather than one, so the cheap fixes do not wait on the hard
investigation: **G15** (the int4 cliff — profile first, no mechanism filed), **G16** (the missing
parallelism — independent of the cliff, true pre-cliff, ~4-5x on the pure-Go lane), **G17** (the
`prefill path: batched` label — fixed same day, gated on neither).

**What this does NOT change:** the pass-2 GPU re-scope stands. It only explains the long-prompt
penalty from the inside — and it adds one line to the eventual recipe that could not have been
written before: on the CPU backend, `int8int8` is not merely the fast lane, it is the only quant
whose prefill cost stays predictable as an agent transcript grows.

**One more confirmatory point.** The int8int8 run's 4000-word case (~6020 tokens) hit the client's
1200s ceiling and aborted. That is not a second cliff: extrapolating int8int8's own n^1.86 from the
3020-token point predicts **~1216s** at 6020 tokens, so a timeout just past 1200s sits *on* the
smooth curve. int4's cliff remains the only discontinuity in either dataset.


---

# Disposition of the three findings (updated 2026-08-25)

This file is the evidence, not the tracker — but a cold reader should not have to guess whether any
of it was acted on.

| finding | disposition |
|---|---|
| **1 — tool path buffers, so streaming is silent** | **Stage 1 SHIPPED** (`43a3fdb`, queue G19, v0.15.0): SSE comment frames while the buffer holds; 71 heartbeats before the first data frame, mutation-checked. Stage 2 (incremental tool-call-aware streaming) queued, deliberately out of the release. |
| **2 — abandoned prefill keeps running** | **FIXED** (`3a16a4b`, queue G18, v0.15.0): context threaded through the prefill chain, checked per layer. Bound is one layer (~12.3 s at 3072 tokens), not immediate — stated as such. |
| **3 — CPU prefill cliff** | **INT4-SPECIFIC** (queue G15, in flight). Not fixed and not blocking the release; the CHANGELOG now tells long-prompt CPU workloads to prefer `int8int8`. The parallelism half is G16, whose two documentation halves shipped (`0c84d1d`). |

**What this run cost and bought.** One end-to-end session with a real consumer surfaced three
defects no gate had caught — two of them shipped fixes the same day, and the third produced a
measured curve, three eliminated hypotheses, and a corrected recommendation in the release notes.
The recipe still is not written, because the qualifying run still has not happened. That ordering is
the point: the gate held.


---

# RETRACTION (2026-08-25): the int4 prefill cliff was a measurement artifact

**Everything above about a cliff is withdrawn.** The 3020-token int4 number in this file
(**1587.1 s**) does not reproduce. Three later measurements of the same thing give ~350 s:

| tokens | int4 as recorded above | int4 re-run (verified-idle box) | int8int8 | int4 direct call |
|---|---|---|---|---|
| 1520 | 99.9s | 100.3s | 93.2s | — |
| 3020 | **1587.1s** | **355.5s** | 334.9s | **348.9s** |

The other three points reproduce within 3%; only that one moved, by 4.5×. **int4 and int8int8 scale
identically at ~n^1.85** — attention's own cost curve. The `n^4.03` step, the "INT4-SPECIFIC"
framing, and the "cliff" language in the Finding-3 section above are all void. Withdrawn in
`docs/queue-performance.md` G15, with the full comparison and the process change it earns.

**What does NOT change — this is the part that matters for this document.** The Tier-0 verdict is
unaffected. An ~8k-token agent prompt extrapolates to **~2200 s** at the *corrected* n^1.85 curve,
which is still far past dsh's 300 s idle timeout and past any harness's. Findings 1 and 2 were
measured by different means, reproduced, and shipped fixes. The run's conclusion — that goinfer's
API surface was green and its engine was not — stands on the corrected numbers.

**What it cost.** A single contaminated number survived into three documents and a release note
before a disagreeing instrument caught it. The instrument was doubted first, because a microbench
that fails to reproduce the real path is usually the microbench's fault. Here it was not.
