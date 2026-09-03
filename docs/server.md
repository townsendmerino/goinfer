# HTTP server — OpenAI, Anthropic, and the rest of the surface

Everything `goinfer serve` exposes: the OpenAI-compatible surface, Anthropic Messages,
multi-model serving, the Responses API, vision, embeddings, prompt-prefix KV caching, and the
admin endpoints. Moved out of the README so the front page stays short.
Back to the [README](../README.md).

## OpenAI-compatible server

[`cmd/serve`](../cmd/serve) is a pure-stdlib (`net/http`, no deps) OpenAI-compatible
server — point Open WebUI, LangChain, or the OpenAI SDKs at it:

```bash
go run ./cmd/serve --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
# OpenAI base URL: http://localhost:8080/v1
```

> **Default quantization is `int4`** (smallest, and fastest on the GPU backends). Override with `--quant
> int8int8|int8|int4mix|""`: `int8int8` is more accurate at ~2× the RAM and is **required for
> `--backend metal`** (int4 declines to CPU there). All quantized modes get batched CUDA prefill
> (fast TTFT); only native f32 falls back to the sequential path. `--quant -h` explains all five.
> A prequantized `.giw` model ignores `--quant` (it carries its own).
>
> **On `--backend cpu` on Apple Silicon, `int4` is now the right default for speed, not the wrong
> one.** A 2026-08-22 measurement (below, kept for the record) found `int8int8` ~60% faster than
> `int4` — correct at the time, but for a reason that has since been fixed: `int4`'s LM head ran a
> slow weight-only-Q8 path while `int8int8`'s happened to already run full W8A8, so the gap was the
> head's drag, not the W4A8 matmul kernel. Both the NEON W4A8 kernel and the `int4`-mode LM head
> shipped (`docs/task-w4a8-neon-bandwidth.md`) and the ranking flipped: measured on the same M1 Pro,
> goinfer commit `a11c56b` (2026-08-24), `int4` now decodes **at or above `int8int8`'s speed** (1.5B:
> 39.1-40.7 vs 37.56 tok/s; 0.5B: 81.9-83.75 vs 85.25 tok/s) at **half the weight RAM** — see
> `docs/benchmarks.md` for the full table. `int8int8` remains **required for `--backend metal`**
> (int4 declines to CPU there) and is still the higher-accuracy choice if precision matters more
> than either speed or RAM.

`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`
(Anthropic — see below), `/v1/models`, and `GET /health` — which is **auth-gated like every
other route**, so a liveness probe must send the API key when one is configured (N-36);
streaming (SSE); the sampling knobs (`temperature`/`top_p`/`top_k`/`seed`/
`frequency_penalty`/`presence_penalty`/`stop`/`logprobs`); and **`response_format`**
— `{"type":"json_schema", …}` or `{"type":"json_object"}` gives schema-constrained
output the model cannot violate (the same grammar as above). The chat template is
auto-detected per model.

On the OpenAI-compatible routes, **`role: "developer"` is accepted as an alias for
`role: "system"`** — same position, and the same last-one-wins precedence two `system`
messages already have. OpenAI's newer APIs send the system prompt under that role for
reasoning-class models, and agent harnesses have followed. `/v1/messages` is unaffected:
the Anthropic API carries the system prompt in its own top-level field and has no
developer role.

> **No auth by default — on loopback only.** `--addr` defaults to loopback, and there
> that only keeps other *machines* out, not other browser *tabs* on yours: any web page
> open while `serve` is running can silently `fetch()`/`POST` to this API (the request
> is sent regardless of CORS; CORS only gates whether the page can *read* the response
> back). That's the deliberate default for the common single-user desktop case — no
> friction for `curl`/local tools — but if it's not the threat model you want, pass
> `--api-key <secret>` (or set `$GOINFER_API_KEY`): every route then requires
> `Authorization: Bearer <key>` or `x-api-key: <key>`, and the server prints a startup
> warning whenever it's running without one.
>
> **A non-loopback `--addr` (e.g. `0.0.0.0:8080`) requires `--api-key`** — `serve`
> refuses to start otherwise, the same hard-fail `--allow-admin` already gets. Even
> with a key, the connection is plaintext by default: the bearer token and every
> prompt/completion travel unencrypted to anyone on the network path. Pass
> `--tls-cert <cert.pem> --tls-key <key.pem>` for plain stdlib HTTPS, or put a
> TLS-terminating reverse proxy (Caddy, nginx, Traefik) in front instead — the better
> answer if you want ACME/auto-renewal. `serve` warns at startup if it's serving
> non-loopback without TLS either way.

**Multi-model.** `--model` is repeatable as `name=path` to serve a model zoo from
one process; requests route on the OpenAI `model` field, `/v1/models` lists all,
and distinct models run in parallel (per-model mutex). Resident int8 models are
expensive — prequant `.giw` maps weights zero-copy for a cheap zoo. With
`--allow-admin` (off by default — it loads attacker-named paths), `POST
/admin/models/{load,unload}` manage the registry at runtime — an unload makes the
model unroutable immediately and frees its device memory once in-flight requests
finish, returning `200` if that completes within `--unload-drain-wait` (default 5s)
and `202` otherwise. `--max-queue N` (default 8) bounds each model's queue: a full queue
returns 429 + Retry-After (single decode worker per model; no continuous batching).

**Sampling: pass `top_k` alongside your temperature.** Since v0.10.3, `top_k`/`top_p`/`min_p` use
bounded selection instead of a full-vocabulary sort, so they are cheap. Plain `temperature` with
*neither* set is the one configuration that still normalizes over the **entire** vocabulary every
token, which makes it now the **slowest** sampled configuration — roughly **3× behind `top_k=20` on
a 152k-vocabulary model**, and worse as the vocabulary grows. If you are setting a temperature,
adding `top_k` is faster than leaving it off. (Removing that remaining cost is scoped in
`docs/ollama-chase.md` §8 D6.) Greedy (`temperature=0`) stays the fastest path and is unaffected.

> **Tie-break (changed in v0.10.3).** Tokens with *equal* probability now resolve by **ascending
> token id**. Before v0.10.3 the order came from an unstable sort and was arbitrary — an
> unspecified part of the result, since that order feeds the cumulative-probability draw. The
> distribution is unchanged, but a sampled sequence from a given seed may differ from v0.10.2 at
> tie points. Greedy argmax is unchanged.

**Request-body limits.** Every request body is capped, and an over-cap body is rejected with `413`
on `Content-Length` **before a byte is read**. `--max-body-bytes` sets the cap explicitly for every
route; left at `0` (the default) it is derived per route: the text cap from the largest served
model's context window (floored at 4 MiB, since a body that could never fit the window is not worth
reading), the vision routes get 32 MiB on top for base64 image data, and `/v1/embeddings` gets its
own 64 MiB — independent of any decoder, because a batch embeddings body scales with batch count,
not with a chat model's context. The resolved caps are printed on the startup line.

**Responses API.** `/v1/responses` honors `input` (string or message items),
`instructions`, `text.format` (→ the same constrained grammar), `tools`, and
streaming (`response.created`/`output_text.delta`/`completed`). `store` +
`previous_response_id` continue a conversation from an in-memory ring — by
construction a prompt-prefix extension, so it rides the warm-KV cache below.

**Anthropic Messages API.** `/v1/messages` and `/v1/messages/count_tokens` speak
the second de-facto standard (the one llama.cpp, Ollama, and LM Studio also
serve), so Anthropic-speaking tools — **Claude Code** included — can point at a
pure-Go single-binary runtime. It honors `system` (string or block array),
content blocks (`text`, `tool_use`/`tool_result` replay), `tools` (note:
`input_schema`), `tool_choice` (`auto`/`any`/`tool` — `any`/`tool` ride the same
constrained decoding, so a malformed tool call is impossible), `stop_sequences`,
and streaming (the named-event SSE protocol: `message_start` → `content_block_*`
→ `message_delta` → `message_stop`, no `[DONE]`).

The `messages` array accepts **only `user` and `assistant`**, as upstream does; any
other role is a `400 invalid_request_error` naming the offending role. In particular
`system` is **not** a message role on this API — it is the top-level `system` field —
and `developer` (accepted as a `system` alias on the OpenAI-compatible routes) is not
one either. Point Claude Code at it — all three env vars are required:

```bash
go run ./cmd/serve --model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=goinfer \
  ANTHROPIC_MODEL=qwen2.5-coder-1.5b-instruct-q4_k_m claude
```

Compatible, not full-spec (llama.cpp's bar): `thinking` / `cache_control` /
`metadata` are accepted and ignored. Agentic use wants a roomy-context model
(≥32k).

**Vision (image→text), pure Go.** With a Gemma 3 VL checkpoint loaded behind
`--vision <dir>` (auto-discovered when `--model` is a VL dir), `cmd/serve` accepts
images on both surfaces — OpenAI `image_url` content parts and Anthropic `image`
blocks — **base64 / `data:` URIs only** (a remote URL is never fetched: an SSRF
guard, returns 400). An image runs through the pure-Go `vision` tower (SigLIP
encoder + projector, HF-parity-gated) into the decoder's embed-by-vector seam;
image tokens count in `usage`. `demo/agent`'s web UI takes a dropped/pasted image
too. Caveat: the SigLIP prefill is CPU-heavy (~3 min/image at 896²) — correct but
slow; an int8 tower is the planned speedup (`docs/completed/task-cpu-vision-prefill.md`).

```bash
go run ./cmd/serve --model ~/models/gemma-3-4b-it --vision ~/models/gemma-3-4b-it
# then POST an image_url data: URI to /v1/chat/completions, or an image block to /v1/messages
```

**Prompt-prefix KV caching.** Across requests the server reuses the KV cache for
the longest token prefix a new prompt shares with a recent one, prefilling only
the new suffix — so a continuing chat (or an agent loop with a fixed system
prompt + tool specs) skips re-encoding the whole history.

Reuse is exact — bit-identical to a cold prefill — under `GOINFER_CPU_FAST_ATTENTION=0`.
With the default fast prefill attention ON, a suffix of 512 tokens or more is prefilled
with a kernel whose reassociated arithmetic is not split-invariant, so a warm continuation
can differ in the last ulps from a one-shot generate of the same full prompt, and at
temperature 0 that can change a near-tie token. This is an accepted trade (2026-08-31): the
exact kernel costs 1.43x on a cold 2048-token turn. Below 512 tokens the fast path does not
engage and reuse is exact either way. `--kv-sessions N` sets how many conversations
to keep warm (default 4; 0 disables); `--session-dir DIR` persists the warm
sessions to disk and restores them on restart.

**Embeddings.** Point `--embed-model` at a [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
HF snapshot to serve `/v1/embeddings` (`--embed-quant f32|q8`). `--model` and
`--embed-model` are each optional and can run together — generation and
embeddings from one process, or either alone:

```bash
go run ./cmd/serve --embed-model ~/models/coderankembed         # /v1/embeddings only
```

`input` (string or array), `encoding_format: float|base64`, and `dimensions`
(truncate + renormalize) follow the OpenAI shape; vectors are L2-normalized. For
this encoder's asymmetric query/document encoding, an optional `input_type:
"query"|"document"` (default `document`, the Cohere/Voyage convention) selects the
query instruction prefix.

### Use goinfer with DeepSeek Harness (dsh)

A fully local agent stack: dsh's harness, goinfer's single binary, no cloud. **Verified end to end
on 2026-08-26** — dsh drove a multi-turn, tool-using task (`glob` → `read` → answer) against a
CUDA-resident goinfer across a network, completing in 277 s with no retries. Every step below comes
from that run; see `docs/measurements/dsh-tier0-run-2026-08-25.md` for what it cost to learn.

**1. Install dsh.** The documented `npx @deepseek-ai/dsh` **hangs** — its ~30 first-party packages
are all prerelease-pinned with 1000+ peer edges, which npm's resolver cannot get through (observed:
6 m of CPU, no output, SIGTERM ignored). Install with peer resolution off, then add the peers it
skips:

```bash
npm install @deepseek-ai/dsh@0.1.1-rc.2 --legacy-peer-deps
# --legacy-peer-deps skips peers, and cordis plugins ARE peers, so the first run
# dies with ERR_MODULE_NOT_FOUND. Add them explicitly (19 at rc.2 — the error names
# one at a time; installing them together is one command):
npm install --legacy-peer-deps @deepseek-ai/cordis-plugin-group @deepseek-ai/dsh-fs \
  @deepseek-ai/dsh-shell @deepseek-ai/dsh-sandbox @deepseek-ai/dsh-workflow ...
```

**2. Start goinfer.** A model that can hold an agent loop is the requirement — not just a context
window. It must emit a tool call *and then answer from the result*; a model that re-calls the same
tool forever looks like a server bug and is not one. A 1.5B failed this; **Qwen2.5-7B-Instruct
passes**. Prefill dominates an agent turn (the harness sends a ~4 KB system prompt plus ~25 tool
schemas, ~8k tokens), so a GPU backend is strongly preferred: measured **270 tok/s prefill on an
RTX 2070 SUPER** vs ~30 tok/s on an M1 Pro CPU.

```bash
# Loopback:
goinfer -model coder=~/models/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384

# Across a network: non-loopback REQUIRES an API key (serve refuses to start otherwise).
GOINFER_API_KEY=<secret> goinfer -model coder=... -backend cuda -addr 0.0.0.0:8080 -ctx 16384
```

**3. Point dsh at it** — `$DSH_HOME/settings.yaml`. Three details each cost a debugging cycle:
the section is namespaced **`llm-pi-ai:`** (a top-level `providers:` block loads fine and then fails
at request time with `NO_ADAPTER`); `providers` is a **dict keyed by route**, not a list; and
**`apiKeyEnv` is required** for a hand-declared route even on loopback, despite the docs — omit it
and you get `PI_AI_ERROR: No API key`. On loopback the value is unused, so any non-empty string
does; off loopback it must match goinfer's `-api-key`.

```yaml
llm-pi-ai:
  providers:
    goinfer:
      apiKeyEnv: GOINFER_API_KEY
      api: openai-completions
      baseURL: http://127.0.0.1:8080/v1     # or http://<host>:8080/v1
      defaultContextWindow: 32768
      defaultMaxTokens: 2048
      models:
        - id: coder
          contextWindow: 32768
          maxTokens: 2048
```

**4. Select the model.** dsh defaults to `deepseek-official`; that is plugin config, not settings,
so it needs a `--patch` overlay:

```yaml
# goinfer-patch.yml
- id: agent-default-model
  name: '@deepseek-ai/dsh-agent-default-model'
  config: {provider: goinfer, model: coder}
```

```bash
GOINFER_API_KEY=<secret> dsh --profile web --patch ./goinfer-patch.yml      # browser UI
GOINFER_API_KEY=<secret> dsh --profile headless --patch ./goinfer-patch.yml "your task"
```

**No compatibility flags are needed.** dsh sends a reasoning model's system prompt as
`role: "developer"`, the output cap as `max_completion_tokens`, and a bare `reasoning_effort` to any
endpoint it does not recognize — which is every goinfer deployment. goinfer accepts all three, so
leave `compat.supportsDeveloperRole` at its default. (Before v0.15.0, `developer` was silently
demoted to a *user* turn, delivering the agent scaffold as the user's first message; if you are on
an older build, set that flag to `false`.)

**What to expect.** Agent turns are prefill-heavy and mostly silent while the model decides on a
tool call — goinfer streams SSE keep-alives during that window so harness idle timeouts do not fire,
and abandons the work if the client disconnects. Deep context slows decode (see the benchmarks
below). If a turn hangs and no keep-alives arrive, you are on a pre-v0.15.0 build.
