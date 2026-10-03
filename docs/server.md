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

> **Default quantization is `int4`** (fastest on every backend, including Metal, which consumes it
> directly). Override with `--quant int8int8|int8|int4mix|""`: `int8int8` is more accurate. All
> quantized modes get batched CUDA prefill (fast TTFT); only native f32 falls back to the
> sequential path. `--quant -h` explains all five. A prequantized `.giw` model ignores `--quant`
> (it carries its own).
>
> **`int4` is not the smallest option everywhere.** On Apple Silicon and non-VNNI amd64 hosts, the
> loader keeps a second, repacked copy of the int4 nibbles beside the canonical ones (the layout
> the fast NEON/AVX2 kernels need), so `int4` measures ~1.25 bytes/element resident against
> `int8int8`'s ~1.02 — **more** RAM, not less, on exactly those platforms (`decoder/fitguard.go`).
> `int4` is still the default there, for speed, not for RAM: if RAM is the constraint on Apple
> Silicon, `int8int8` is the smaller choice, not the fallback for "more accuracy, more RAM" its
> name suggests.
>
> **On `--backend cpu` on Apple Silicon, `int4` is now the right default for speed, not the wrong
> one.** A 2026-08-22 measurement (below, kept for the record) found `int8int8` ~60% faster than
> `int4` — correct at the time, but for a reason that has since been fixed: `int4`'s LM head ran a
> slow weight-only-Q8 path while `int8int8`'s happened to already run full W8A8, so the gap was the
> head's drag, not the W4A8 matmul kernel. Both the NEON W4A8 kernel and the `int4`-mode LM head
> shipped (`docs/completed/task-w4a8-neon-bandwidth.md`) and the ranking flipped: measured on the same M1 Pro,
> goinfer commit `a11c56b` (2026-08-24), `int4` now decodes **at or above `int8int8`'s speed** (1.5B:
> 39.1-40.7 vs 37.56 tok/s; 0.5B: 81.9-83.75 vs 85.25 tok/s) — see `docs/benchmarks.md` for the full
> table. `int8int8` is still the higher-accuracy AND (on this platform) lower-RAM choice if either
> matters more than speed.

`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`
(Anthropic — see below), `/v1/models`, and `GET /health` — which is **auth-gated like every
other route**, so a liveness probe must send the API key when one is configured (N-36). Each `/v1/models`
entry also carries goinfer-only fields: the resolved `decode_path`/`prefill_path`, and `context_window`,
the exact token limit a text request is held to (the resident KV cap on a GPU backend when that is lower than
the model's own maximum), from the same function that enforces it; and `vision`, whether the model
accepts image content parts (from the same check that refuses them otherwise);
streaming (SSE); the sampling knobs (`temperature`/`top_p`/`top_k`/`seed`/
`frequency_penalty`/`presence_penalty`/`stop`/`logprobs`); and **`response_format`**
— `{"type":"json_schema", …}` or `{"type":"json_object"}` gives schema-constrained
output the model cannot violate (the same grammar as above). The chat template is
auto-detected per model.

**Per-field confidence (`goinfer_confidence`, opt-in).** Add `"goinfer_confidence": true` to a
`/v1/chat/completions` or `/v1/completions` request that carries `response_format: {"type": "json_schema", …}`,
and the response gains a top-level `goinfer_confidence` array, one record per reported field. Streaming sends the
array as one event, `{"id": …, "goinfer_confidence": […]}`, after the finish chunk. For example:

```json
{"path": "category", "kind": "enum", "value": "shipping", "confidence": 0.60,
 "distribution": {"billing": 0.37, "shipping": 0.60, "technical": 0.03}, "free_tokens": 1, "calibrated": false}
```

- **What the number is.** It is the model's probability over what the schema allowed, at the position that
  decided the field. It is **not** the probability that the value is right, and `calibrated` is `false`.
  - It *discriminates*: a low-confidence value is wrong more often. That was measured on a labelled set as AUROC
    0.85 / 0.73 / 0.68 for enum / boolean / integer fields on a 1.5B
    (`measurements/confidence-c0-2026-09-27.md`).
  - It was not measured for calibration. Use it to rank or to route ("ask a person below 0.7"), not as a
    probability of being right.
- **Which fields.**
  - Enum, boolean and integer fields are reported, at any depth (`items[2].status`).
  - Number and string fields are not: the labelled set could not grade them yet.
  - A field whose value the schema forced (a one-option enum) is not reported either; its value says nothing about
    the model.
- **Enum and boolean** fields carry `distribution`, the mass of every legal token summed by the option it spells,
  so a literal split across tokens does not dilute it. `undecided` is the mass that did not yet pick an option.
  **Integer** fields report the lowest probability among their digits.
- **Cost.** It adds about 0.3 ms at each position that decides structure or a value, and nothing inside a free
  string.
  - It turns off grammar-fused speculative decoding for that request, which drives the grammar without the hook
    the capture needs.
  - The answer itself is unchanged: the same request without the flag returns the same content.
- **Where it is refused (400):** without `response_format` `json_schema` (the schema is what says each field's
  kind), and on routes that would drop it: tools, `/v1/jobs`, batches, `/v1/responses` and `/v1/messages`.
  *Updated 2026-10-02 (GLM-OCR O5): an image request on `/v1/chat/completions` is no longer refused. The vision chat
  route always honoured `response_format` (the grammar masks every decode step there too); it now also writes
  `goinfer_confidence` back, in both the buffered reply and the stream's trailing event. Measured on GLM-OCR in
  `docs/measurements/glm-ocr-o5-2026-10/`.*
- **In Go:** `constrain.NewMasker(…).CaptureConfidence(constrain.ConfidenceOptions{})`, then
  `masker.FieldConfidence(generatedIDs)` after the generation.

**The server refuses new work before the machine swaps.** At startup it prints `swap guard: armed,
threshold +512 MB over baseline` and the baseline swap-used it measured; if swap-used then grows past
that threshold while serving, every generation route answers **503** with
`{"error":"halted","reason":"swap guard tripped: …"}` (the same shape as an admin halt) until swap has
stayed back within the threshold for 30 s — requests already running are left to finish, nothing is
cancelled. A `.gguf` loaded directly (not through its sidecar) is guarded during the load too: the load
aborts with a message naming the swap growth and the model's priced memory terms. `GOINFER_SWAP_GUARD`
sets the threshold in MB or `off` (`docs/env-vars.md`); `goinfer-chat` has the load-time half only.

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
returns 429 + Retry-After (no continuous batching).

**`--max-concurrent N` (default 4) lets one CPU model run N generations at once** (Experimental; MC3c of
`docs/tasks/parked/task-concurrency-2026-09.md`, 2026-09-26; default 4 by owner decision, and `1` restores strict
serialization). Each generation runs on its own session KV, so each
conversation's output is byte-identical to serving it alone, and admission stays FIFO. N is capped by `--kv-sessions`
(each running generation holds a session). A weight-streaming or vision model always runs one.

**`--cpu-batch auto|on|off` (default `auto`) batches those generations' decode tokens** (MC3c step 2, 2026-09-27).
Concurrent CPU generations of one model join their decode tokens into one batched forward, which reads each weight once
for all of them. Every reply is bit-identical either way, and a lone request takes exactly the unbatched path.
- `auto` batches a plain dense model with at least 2 GiB of weights and leaves smaller models on the independent
  workers, where they are faster.
- On a 7B (qwen2.5-7b, Ryzen 7 3700X, W7, 4 clients): **2.19×** the workers' aggregate (5.4 → 11.9 tok/s), with the
  p99 turn 96 → 46 s and a lone request unchanged. See `measurements/concurrency-mc3c-step2-2026-09-27.md`.
- On the 0.5B, `on` reads 0.73× the workers.
- On macOS too: on an M1 Pro's CPU the 7B's batched step reads 1.54× (depth 128) and 1.38× (depth 512) the workers
  (`measurements/concurrency-mc2-2026-09-26.md`, "Mac 7B cell"). A served W7 run on the Mac has not been made.
- Not batched (they run as workers): MoE, recurrent, hybrid and non-standard-block families, `--kv i8`, adapters,
  speculative decode and vision.
- The banner's concurrency line says which mode runs, and serve logs the batcher's step counts at shutdown.

**On Metal and CUDA, a dense resident model batches concurrent generations** (MC3: Metal 2026-09-26, CUDA
2026-09-27). From 2026-09-28 to 2026-09-30 the `--embed-int4` default kept a default Metal load off the resident, so it did
not batch; `--embed-int4` now defaults off on Metal (`quantization.md`). Under the same flag, each running generation holds its own resident KV slot (`--kv-sessions` sets the
count, 4 by default). Their decode tokens run together in one step on the GPU, every logit bit-identical to serving
that conversation alone. *(Established per decode step and on 4x3-turn tests. End to end on CUDA with the default fast prefill, a long prompt's reply at a near-tied token can differ from the same prompt served alone, because rows it reused from the KV cache may have been computed by the exact kernels (a short earlier request) while the rest run on the fast ones; under concurrent load which rows it reuses was a race. Since 2026-10-02 a reuse under 64 tokens is declined on such a prompt, which removed every difference in the schedules tested; a longer reuse whose rows were computed by a different kernel class can still differ. `GOINFER_CUDA_FAST_PREFILL=0` removes it entirely. `cuda-backend.md` "Tensor-core fast prefill"; `measurements/mc4-candidate-cuda-2026-10-01.md`, ROOT CAUSE.)*
- On CUDA (RTX 2070 SUPER, W7, 4 clients): qwen2.5-coder-1.5b reads 1.38× the one-at-a-time build, with p99 per turn
  from 2.68 s to 2.02 s. qwen2.5-7b-instruct reads 1.83×, with p99 from 7.2 s to 4.1 s. A lone request is unchanged.
  - Greedy and sampled requests both batch.
  - Long prompts that arrive mid-decode are prefilled in chunks, as on Metal.
  - The bullets below are Metal's, except where they say CUDA. See `measurements/concurrency-mc3-cuda-2026-09-27.md`.
- Measured on qwen2.5-coder-1.5b (W7, 4 clients): 1.59× the serialized aggregate, and p99 per turn from 7.1 s to
  4.7 s; a lone request is unchanged. See `measurements/concurrency-mc3-2026-09-26.md`.
- Greedy and temperature-only sampled requests both batch; the latter at 1.62× at 4 clients
  (`measurements/concurrency-mc3-s2-2026-09-27.md`). A sampled draw is the same one a lone request would make.
- Since S3 (2026-09-27), a batched step's per-row work runs as one dispatch over all rows: 4 clients 1.26× faster
  again, and a turn under 4-client load takes ~1.9× a lone request's (`measurements/concurrency-mc3-s3-2026-09-27.md`).
- On qwen2.5-7b-instruct (W7, the whole stack against the serialized build), 4 clients read 1.79× and p99 per turn
  fell from 20.9 s to 12.4 s (`measurements/concurrency-mc3-7b-w7-2026-09-27.md`).
- Since S4 (2026-09-27), a two-sequence step runs its largest projections as production's own per-row GEMVs where the
  load-time calibration finds that cheaper. On the 7B, 2 clients read 1.12× the previous build (before, they gained
  ~2% over one at a time); on the 1.5B, 1.06× (`measurements/concurrency-mc3-s4-2026-09-27.md`). The choice is printed
  at load.
- A newcomer's long prompt no longer stalls the others (`-prefill-chunk`, default 512). A prompt that arrives while
  others are decoding is prefilled in 512-token chunks, with a decode step between chunks.
  - Measured with a ~3k-token newcomer, the decoders' longest wait fell from 5.4 s to 1.2 s, and the newcomer's
    time to first token rose ~11%.
  - Replies are unchanged, because Metal's prefill is chunk-invariant (`measurements/chunked-prefill-2026-09-27.md`).
  - `0` prefills whole.
- A newcomer's prefill runs between steps, in chunks when it is long (below).
- A model serving `--spec`, a `--drafter` or an adapter keeps one generation at a time, as does any non-dense family.
  - On Metal, `--spec ngram` verifies its drafts on the batched step's kernels (since 2026-09-27), bit-identically.
    A lone greedy request measured 2.08× plain decode on verbatim-copy requests and 1.07× on chat (1.5B); the 7B
    measured 1.85× and 1.01× (`measurements/metal-spec-step-verify-2026-09-27.md`).
  - Its replies are identical to plain decode's on every turn. Before 2026-09-27, turns after the first diverged on
    Metal, because a generation ending at `max_tokens` left its last token unforwarded, and the next turn
    re-prefilled it through the f16 prefill. That is fixed.
  - It still takes the resident exclusively: under 4-client load it gives up batching, at 0.61× / 0.49× the batched
    aggregate (`measurements/spec-vs-batching-metal-2026-09-27.md`).
  - So on Metal, turn it on for one user or an agent loop (copy-heavy edits pay most), and leave it off for a server
    shared by several clients.
- The line printed after load (`"<name>" concurrency: …`) says which applies. A request's prefill shares the memory safety margin with the generations running or queued
ahead of it when it arrives (up to N), so a lone request keeps the whole margin. The trade: aggregate throughput rises (4 decode workers measured 2.0–2.5× on an M1 Pro's CPU,
`measurements/concurrency-mc2-2026-09-26.md`), while each request, sharing the cores, takes longer than it would
alone.

**`top_k` + `top_p` together are not HF/llama.cpp semantics** (N-03). When both are set, the
nucleus mass is computed over the FULL distribution and the result is then intersected with the
top-k set. HuggingFace and llama.cpp apply `top_k` first and take the nucleus over the
RENORMALIZED survivors, which is a strictly wider set for the same `top_p`. Either is a defensible
reading of two filters that OpenAI's API never defined together, but a client that tuned `top_p`
against HF will see a tighter cut here. Each filter alone matches.

**Sampling performance (corrected 2026-09-11, audit-2026-09-02.md N-36 — this section used to
recommend adding `top_k` for speed; that was right before P2b, 2026-08-09, and is backwards now).**
`top_k`/`top_p`/`min_p` use bounded
selection instead of a full-vocabulary sort (since v0.10.3), and plain `temperature` alone
normalizes over the entire vocabulary — but P2b made that full-vocabulary normalization itself
~3-4.7× cheaper (parallel chunked reduction), closing the gap this section used to describe.
Measured directly just now (`benchSample`, `decoder/sampler_selection_test.go`'s own harness,
best-of-3, same machine): at a 152k vocab, temperature-only and `temperature`+`top_k=20` cost
about the same (**0.99×**, noise-level); at 262k, temperature-only is **~15% FASTER** than adding
`top_k=20` (**0.86×**), not 3× slower. **Do not add `top_k` purely for speed** — it no longer
helps and can cost a little. `top_k`/`top_p`/`min_p` are still cheap in their own right (bounded
selection, not the old O(V·log V) sort) and remain the right choice when you want their actual
sampling behavior, just not as a performance workaround for plain `temperature` anymore.
(Removing the last full-vocabulary-normalization cost entirely is scoped in
`docs/ollama-chase.md` §8 D6.) Greedy (`temperature=0`) stays the fastest path and is unaffected.

**Update 2026-09-20 (R7, R7b) — on CUDA and WebGPU sampled decode now runs at greedy speed.** Two changes:
with `top_k`, `top_p` or `min_p` set, the resident reduces the logits row on-device and the host samples from the K
best; with none of them set (plain `temperature`, the OpenAI default), the resident draws the token on-device and
returns just its id. In both cases nothing applies that needs the whole row: no logit bias, repetition/presence/
frequency penalty, logprobs or constrained decoding. Measured on qwen2.5-coder-0.5b, same session, paired against
greedy: **CUDA plain `temperature` 0.744 → 1.008; `top_p` 0.95 0.653 → 0.951; `top_k` 40 0.969; `min_p` 0.05 0.972;
WebGPU plain `temperature` 0.796 → 1.035.** So the advice above no longer holds on those backends in either
direction: adding a filter is neither needed nor faster, and plain `temperature` is the fastest sampled shape. CPU
and Metal still sample on the host (Metal has no device kernel yet); their plain-`temperature` draw got ~1.8x cheaper
in the same change. `GOINFER_NO_TOPK_FASTPATH=1` and `GOINFER_NO_SAMPLE_FASTPATH=1` turn the device paths off.
Records: `docs/measurements/sampled-topk-2026-09-20.md`, `docs/measurements/sampled-gumbel-2026-09-20.md`.

> **Plain-`temperature` sampling draws a different stream (changed in the release after v0.19.0).** The draw is now
> Gumbel-max — the token is `argmax(logit/T + noise)`, the noise from a counter-based Philox generator keyed by
> the seed and the draw index — on every backend. The **distribution is unchanged** (statistically tested against the
> exact softmax and against the old sampler), but **for a given seed the tokens are not the ones earlier releases
> produced.** It applies only to temperature > 0 with no `top_k` / `top_p` / `min_p`; sampling with any of those
> is unchanged token for token. Logprobs, penalties and bias no longer change which token a seed yields: every
> plain-`temperature` request draws the same way. Speculative decoding keeps drawing accept/reject from its own
> stream (it needs explicit probabilities); its first token equals plain decoding's under the same seed.

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
`input_schema`), `tool_choice` (`auto`/`any`/`tool`), `stop_sequences`,
and streaming (the named-event SSE protocol: `message_start` → `content_block_*`
→ `message_delta` → `message_stop`, no `[DONE]`).

**What `tool_choice` actually constrains** (N-18, closed 2026-09-24): on the families with a JSON call form — chatml and mellum2
(`<tool_call>`), mistral (`[TOOL_CALLS]`) and llama3 (bare JSON) — a tool call cannot be malformed and cannot name a tool you did not
supply, with any number of tools:
- a NAMED tool (`{"type":"tool","name":…}`, OpenAI's `{"type":"function",…}`) is constrained to that tool from the first token, and
  naming a function that is not in `tools` is a 400;
- `required` / `any` is constrained from the first token to a call to ONE of the supplied tools, with that tool's argument schema;
- `auto` is constrained the same way from the moment the model writes its call opener, so it can still answer in prose — and a turn
  that answers in prose decodes exactly as it would with no constraint (same speed, byte-identical output). llama3 has no opener:
  under `auto` its call is constrained when the reply BEGINS as one (`{"name": "`, optionally after `<|python_tag|>`); a call that
  follows prose in the same reply is still only parsed. A server started with speculative decoding leaves `auto` unconstrained, to
  keep its drafter.

Families without a JSON call form (Gemma 4's own call syntax, gpt-oss's channel message, and families with no tool template) are parsed only. This constrains the
call's FORM, not the model's choice of tool. `GOINFER_TOOL_UNION=0` turns the multi-tool constraint off. Measured in
`docs/measurements/tool-union-2026-09-24.md`: on the Qwen2.5-7B agent transcript that produced 14 unusable calls and 10 calls with
invalid arguments in 111, it produced none. (This paragraph previously said that with two or more tools the output was not
grammar-constrained, which was true until T1–T3.)

**Qwen3.5's own tool form is read too, and available on request (`-tool-format`).** The model's chat template asks for
`<tool_call><function=NAME><parameter=K>value</parameter></function></tool_call>`; goinfer prompts the ChatML families in the Hermes
JSON form instead (signatures and a JSON call, which `tool_choice` can constrain). The reply parser reads **both** forms whichever was
prompted, typing each XML parameter from the request's schema (a `"123"` for a string parameter stays a string). `-tool-format
template` prompts the model's own form instead, rendered byte for byte from its template; a `tool_choice` naming a function is then a
400, because that form has no JSON wrapper to constrain a decode to. **The default stays `hermes`**: measured on Qwen3.5-0.8B and 9B
(eight prompts, three samples each, with a follow-up turn after each call), the native prompt read its own tool history better (loops
answered 11/12 against 9/14 on the 0.8B, 21/21 against 20/21 on the 9B) but made the 0.8B call a tool less often (12/21 correct calls
against 14/21), and missed the pre-registered bar for changing the default by one reply. Neither prompt ever produced an unparsed XML call.
`docs/measurements/qwen35-tool-format-2026-09-30/RESULTS.md`.

**Gemma 4's tool loop follows its template where it is the canonical one.** Gemma 4 ships in two template versions; goinfer's managed rendering
(thinking, history, and this loop) is for the canonical one (Google's "Gemma 4 Canonical Chat Template", 2026-07-09 — the 26B-A4B checkpoint's and the
12B QAT's GGUFs). The older template (the E2B GGUF has it) is managed for thinking only: its thinking-off prompt is a bare `<|turn>model\n` (it has no closed scaffold — writing one made the model close a channel it never opened, and `<channel|>` leaked into `content`), thinking on is the `<|think|>` system line, and a reply's reasoning is split out of `content`, held to HuggingFace's rendering of that template; its tool loop stays the generic rendering. For the canonical template the
loop is the template's own, byte for byte against HuggingFace: consecutive assistant messages are **one model turn**; a tool result that carries only
`tool_call_id` names its function (OpenAI clients often omit `name`); and an assistant message with text beside a call has its **text after the
results, the turn closed, and no turn header after a tool result** — the template's order, **adopted as the default** after a pre-registered A/B
(Gemma-4-12B QAT: 21/21/3 on both arms, identical outcomes on every reply; `docs/measurements/gemma4-tool-text-order-2026-09-30/RESULTS.md`).
`-tool-format hermes` restores goinfer's earlier order (text before the call, the turn left open); `-thinking asis` keeps the pre-thinking bytes.
The A/B ruled out a worry, it did not show the template's order is better: the arms tied, on one model, eight prompts, a one-line preamble.

**An unwrapped call is accepted on the `<tool_call>` families (chatml, mellum2).** Qwen2.5-Coder at every size tested
practically never writes the `<tool_call>` wrapper under `auto`; it emits the call object alone, which earlier versions
returned as prose, so an agent got no call at all. An output whose first non-space byte opens a JSON object whose `name` is
exactly one of the supplied `tools` and whose arguments are an object is now returned as that one call (the first object only;
anything else stays prose, and a name you did not supply never becomes a call). This fixes the call's form, not the model's
choice of tool. Measured in `docs/measurements/tool-call-failure-t0-2026-09-23.md` (0 → 219–236 parsed calls of 300 on the
0.5B/1.5B; no other output changed).

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

**Vision (image→text), pure Go.** With a vision-capable checkpoint loaded behind
`--vision <dir>` (auto-discovered when `--model` is a VL directory: `internal/serveapp/main.go`'s
`loadVisionTower` detects Gemma 3's SigLIP projector, Qwen2.5-VL's own ViT, Gemma 4's vision
tower, a Qwen3.5+ checkpoint and a GLM-OCR checkpoint, and routes each to its own loader),
`cmd/serve` accepts images on both surfaces — OpenAI `image_url` content parts and Anthropic
`image` blocks — **base64 / `data:` URIs only** (a remote URL is never fetched: an SSRF guard,
returns 400), one image per generation (serve keeps the newest image in a conversation and replaces earlier ones with a visible note and an `X-Goinfer-Images-Omitted` header; several images in the latest message are a 400). An image runs through the matching pure-Go vision tower into the
decoder's embed-by-vector seam; image tokens count in `usage`. `demo/agent`'s web UI takes a
dropped/pasted image too, for Gemma 3 only. Qwen3-VL is its text decoder only (no tower), and a
`mistral3` checkpoint's tower is ignored: an image on either is a 400 "this model has no vision tower".

*Rewritten 2026-10-02 from an audit of the tree and a run on the CUDA box (`docs/measurements/multimodal-audit-2026-10-02.md`; the previous text of this paragraph said "SigLIP
path, CPU-heavy, 31.3 s" and "`--backend webgpu`/`--backend cuda` force the int8 tower", both true only of Gemma 3 on some backends). **Every Metal statement below is read from code; no Mac
was available.** Per family:*

| family | the vision tower | the decoder after the image |
|---|---|---|
| **Gemma 3** (SigLIP) | CPU, f32. **On `--backend cuda` a resident CUDA tower (goinfer's own), on `--backend webgpu` a resident WebGPU tower**, both int8; Metal: CPU | CUDA and WebGPU: resident decode; CUDA also a resident image prefill. Metal: resident decode through the `UploadKV` bridge (read, unrun) |
| **Gemma 4 E2B, E4B** | CPU, f32 unless `-vision-quant int8` | **CPU for the whole model on every backend** (no backend implements the E-model features) |
| **Gemma 4 26B-A4B, 31B** | CPU, f32 unless `-vision-quant int8` | CPU bidirectional prefill, then resident decode through the bridge (26B-A4B run on CUDA; 31B unverified; Metal and WebGPU unverified) |
| **Qwen2.5-VL** | CPU, f32 unless `-vision-quant int8`: aikit has `gpu/qwencuda` and `gpu/qwenmetal`, goinfer does not use them | CUDA: resident m-RoPE prefill and decode; WebGPU and Metal: CPU prefill, then resident decode |
| **Qwen3.5+ dense** (0.8B, 9B gated; MoE sizes accepted but never run) | CPU, f32 unless `-vision-quant int8`, loaded on the first image, at most 1,024 image tokens per image | **CPU prefill and CPU decode on every backend** (a recurrent family refuses every resident branch), so a repeated image re-runs the tower |
| **GLM-OCR** | CPU, **f32 on every backend** | CUDA: resident (pairwise rope); WebGPU: staged (no resident KV); Metal: CPU |

**The tower-quant rule, exactly as `towerInt8` in `main.go` applies it.** `-vision-quant int8` gives an int8 tower. `--backend cuda` and `--backend webgpu` (including `--backend auto` when it
resolves to CUDA) **also** give Gemma 3 an int8 tower whatever `-vision-quant` says, because only its resident tower has a GPU path and that path needs int8 weights. **Every other tower
(Qwen2.5-VL, Qwen3.5+, Gemma 4, GLM-OCR) is CPU-only and f32 unless `-vision-quant int8`:** until 2026-10-02 the first three were forced to int8 under cuda/webgpu, which was not faster and was
not close to f32 (relative L2 0.21 / 0.14 / 0.31 against each tower's own f32 on one image; `docs/measurements/vision-tower-int8-fidelity-2026-10-02.md`), and GLM-OCR's int8 form is not gated
on the real checkpoint. The f32 tower holds more host memory. `EnableResident` is called in one place only (the Gemma 3 tower on cuda/webgpu); if it fails, serve warns and runs that tower on
the CPU rather than refusing to start. `-vision-quant int8` (default `f32`) is a CPU option: `docs/completed/task-cpu-vision-prefill.md` found it speeds the compute-bound
prefill only on AVX512-VNNI hardware and measures a wash on plain AVX2, which is why f32 is the default.

**Speed, each figure with its record.** CPU SigLIP tower ~31.3 s/image at 896² (recorded 2026-09-08, `docs/benchmarks.md` §A "Vision tower CPU prefill", not re-measured). CUDA Gemma 3 tower
**4.1 s/image** (recorded 2026-09-21: 26.0 s with the exact attention kernel, 6.4x with the fused one that is the default since the owner's override; `GOINFER_CUDA_VISION_ATTN=exact`
restores the 26 s kernel; `docs/measurements/vision-tower-mma-2026-09-21.md`); a cold Gemma 3 image request took 4.9 s end to end on the 2070 SUPER on 2026-10-02 (exploratory). The WebGPU tower
is recorded at ~9x over the CPU (2026-06-11, before the 2026-08-25 re-anchor; not re-measured), and its cold request took 27.4 s end to end on 2026-10-02 (exploratory). On an 8 GB card
`--backend webgpu` needs `--ctx 4096 --kv-sessions 1` or more so that the tower fits beside the decoder's KV: at the default the load aborts at startup ("Not enough memory left"). A CPU
tower costs what the CPU costs: tens of seconds per Gemma 3 image; on 2026-10-02 a 336x336 image took 7 s on Qwen3.5-0.8B and 25 s on Gemma 4 E2B (the whole model on the CPU) on this box,
one sample each (exploratory). Nothing was timed on Metal.

**Release binaries.** The Linux `goinfer-serve` is built `CGO_ENABLED=0` with CUDA, so **a downloaded Linux binary on an NVIDIA box gets the resident Gemma 3 tower and resident decode after an image
for Gemma 3, Qwen2.5-VL and GLM-OCR (and Gemma 4 26B-A4B)** with no C toolchain. The macOS binaries carry Metal (no vision tower: the tower runs on the CPU) and the Windows binaries are CPU. WebGPU, the one cgo build,
is in no release binary.

**GLM-OCR** (a 0.9B document model; `model_type` `glm_ocr`) is auto-discovered from its checkpoint directory and uses aikit's own tower (`vision.GlmOcrVisionEncoder`, aikit v1.52.0), loaded on the
first image. An image is at most 6,144 image tokens (about 4.8 megapixels) and there is no smaller default cap yet; one that does not fit the resolved context is refused with
`image_too_large_for_context` before the tower runs. The CPU tower costs about 44 s for a 1,656-token page on this class of machine (exploratory, one rendered invoice). Structured extraction with
`response_format` is described below; the measured accuracy is in `docs/measurements/glm-ocr-o5-2026-10/`.

```bash
go run ./cmd/serve --model ~/models/gemma-3-4b-it --vision ~/models/gemma-3-4b-it
# then POST an image_url data: URI to /v1/chat/completions, or an image block to /v1/messages
```

**Images with `response_format` json_schema; structured extraction from a document (GLM-OCR, 2026-10-02, O5).** The
vision chat route constrains decoding to the request's `response_format` exactly as the text route does, on every vision
family: the grammar masks each decode step, so the reply is JSON that has the schema's shape. On a **GLM-OCR** checkpoint
(`zai-org/GLM-OCR`; auto-discovered like the other towers) the request is also told *what to extract*, because that model
is prompted for extraction with a JSON **template**, not a JSON Schema: the card's instruction
`请按下列JSON格式输出图中信息:` followed by an object of blank values. Serve builds that template from the request's own
schema (`constrain.TemplateFromSchema`, one source with the grammar) and uses it as the prompt, with this rule:

| the request's text part | what the model is sent |
|---|---|
| absent, empty or whitespace | the instruction and the schema's template |
| exactly `Text Recognition:`, `Table Recognition:` or `Formula Recognition:` | the instruction and the schema's template (the task prompt would contradict the grammar) |
| anything else | **the user's text, unchanged** (their own extraction prompt, a question); the grammar still constrains the reply |

The template convention: properties in the schema's order, every property shown (optional ones too), nested objects
nested, `""` for a string, `0` for a number or integer, `false` for a boolean, one example element for an array (so a list of
line items shows what one looks like). **Type the amounts as numbers.** The model answers a numeric column with a bare
number whatever the template shows for a string field, and a string grammar forbids a bare number: the only legal tokens
left are whitespace, so on `"quantity": ""` the measured reply was whitespace until `max_tokens`, with no error
(`docs/measurements/glm-ocr-o5-2026-10/string_typed_quantity_whitespace_runaway.txt`). With `integer` and `number` fields the
same page extracted cleanly. An invalid schema is a 400; `json_object` has no shape to build a template from and is left
alone. The reply is `finish_reason: "length"` and unparseable if `max_tokens` ends it first, since the grammar guarantees a valid
*prefix*; an invoice with many lines needs a generous `max_tokens` (the example reply is about 420 tokens for six lines).

```bash
python3 - <<'PY'
import base64, json, urllib.request
png = base64.b64encode(open("testdata/glm_ocr/invoice.png", "rb").read()).decode()
schema = json.load(open("testdata/glm_ocr/invoice.schema.json"))
body = {"model": "glm", "temperature": 0, "max_tokens": 1500, "goinfer_confidence": True,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "invoice", "schema": schema}}}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}))
print(json.load(r)["choices"][0]["message"]["content"])
PY
```

The same flow in the terminal: `goinfer-chat --model ~/models/glm-ocr --image invoice.png --schema invoice.schema.json`
(one image, one answer, then exit). In Go: [`examples/invoice`](../examples/invoice/main.go).

**Prompt-prefix KV caching.** Across requests the server reuses the KV cache of a
warm session **whose entire token history is a prefix of the new prompt** —
not the longest prefix the two merely share (`internal/serveapp/sessions.go`'s
`bestExtend`: a session that diverges anywhere, even one token in, is not
reused at all; among the sessions that DO qualify, the longest — most
reused — one wins) — prefilling only the new suffix. So a continuing chat (or
an agent loop with a fixed system prompt + tool specs) skips re-encoding the
whole history, as long as each new turn's prompt still starts with the exact
tokens of a session already warm.

Reuse is exact — bit-identical to a cold prefill — under `GOINFER_CPU_FAST_ATTENTION=0`.
With the default fast prefill attention ON, a suffix of 512 tokens or more is prefilled
with a kernel whose reassociated arithmetic is not split-invariant, so a warm continuation
can differ in the last ulps from a one-shot generate of the same full prompt, and at
temperature 0 that can change a near-tie token. This is an accepted trade (2026-08-31): the
exact kernel costs 1.43x on a cold 2048-token turn. Below 512 tokens the fast path does not
engage and reuse is exact either way. `--kv-sessions N` sets how many conversations
to keep warm (default 4; 0 disables); `--session-dir DIR` persists the warm
sessions to disk and restores them on restart.

**GPU-resident models reuse prefixes too, by a different mechanism.** A `-backend
cuda`/`metal`/`webgpu` resident model has no session cache (the two are mutually
exclusive) — instead, resident prefix reuse (`decoder/resident_reuse.go`) compares
the new prompt's committed token ids directly against the resident KV's own positions
and prefills only the divergent suffix, reported to the client as
`prefill_reused_tokens`. Its suffix goes through the same batched prefill path an
ordinary cold prefill uses, so it inherits that path's own exactness knob, not
`GOINFER_CPU_FAST_ATTENTION`: **Metal and CUDA** — `--exact-prefill` covers both
(`GOINFER_METAL_FAST_PREFILL=0`/`GOINFER_CUDA_FAST_PREFILL=0`). **WebGPU** — no
fast/exact split exists on this backend, so there is nothing to opt out of.

**On Metal, CUDA and WebGPU, several conversations stay resident** (MC1 of `docs/tasks/parked/task-concurrency-2026-09.md`:
Metal 2026-09-26, CUDA and WebGPU 2026-09-27). `--kv-sessions N` also asks the resident for N GPU KV slots, one
conversation each. Each generation binds the slot that already holds its prompt's prefix; a new conversation
takes an empty slot, else the least recently used one. A slot that shares only a chat template's lead with the
prompt is never truncated to serve it. With `--max-concurrent` above 1, a dense model's generations also run at once,
one per slot, on Metal and CUDA (MC3, above); WebGPU runs them one at a time. The memory guard clamps N to what fits
(each slot is the full KV at the resident context, e.g. ~117 MB for Qwen2.5-1.5B at 4k on Metal, 448 MB at 8k in
CUDA's f32 KV), and the banner says what it allocated. On CUDA the slots alone, before MC3 batched the generations,
stopped interleaved conversations from evicting each other (qwen2.5-coder-1.5b, 4 clients: 1.25× the one-slot build,
`measurements/concurrency-mc1-cuda-2026-09-27.md`). WebGPU does the same, where a thrash costs more because Qwen2.5
prefills one token at a time there: 2.81× at 4 clients on an M1 Pro (`measurements/concurrency-mc1-webgpu-2026-09-27.md`).
- **On WebGPU**, each slot is the full f32 KV at the resident context (~0.94 GB for Qwen2.5-1.5B at the default 16k).
  WebGPU has no free-memory query. On a Mac, slots are clamped to 70% of RAM and to what was available before the
  build, and **slots come before context there as on CUDA** (owner decision 2026-09-27): when `--ctx` is not set, the
  context gives up positions, down to 4096, until every requested slot fits, and it logs the shrink. A context the
  load-time fit guard chose for you counts as not set; an explicit `--ctx` is never shrunk. Elsewhere (a discrete
  GPU) a slot that fails to allocate, or that would leave under 384 MiB free, ends the count, and the context is not
  shrunk yet; pass a smaller `--ctx` for more slots. That half waits on a measurement of the clamp on real Vulkan
  hardware (`docs/prompts/nobara-mc1-webgpu-2026-09.md`).
- **On CUDA, slots come before context** (owner decision 2026-09-27). When `--ctx` is not set, fit by default gives up
  context, down to 4096, until every requested slot fits, and it logs the shrink. Below 4096 the slots are clamped
  instead.
  - On an 8 GB card the 7B starts at about 4,980 tokens with 4 slots (1.35× at 4 clients against 2 slots at 8192).
  - The build makes the final trim against real free VRAM, and it logs both steps.
  - An explicit `--ctx` is never shrunk: it keeps that context and the slots that fit beside it.
- **One slot:** the recurrent families (Gated DeltaNet, Mamba-2, LFM2), and a CUDA load with expert streaming on.

**Decisions (`POST /v1/systemone`).** This is TypeSafe's decisions wire shape: a state and named questions with a
closed answer set, returning a probability distribution per question. goinfer answers it by **label scoring on the
served model**: one prefill per question, reading the model's probabilities for the option labels, with no decode.
It exists so [jevx](https://github.com/muthuishere/jevx) and TypeSafe's SDKs work against goinfer unchanged. Point
them at goinfer, and set `model` to goinfer's served name; an unknown name such as `jev-latest` is refused with the
names that are served.
- **jevx:** a profile's `url` is the full endpoint, e.g. `"url": "http://127.0.0.1:8080/v1/systemone", "model": "local"`.
- **`typesafe-sdk` (Python) and `@typesafe-ai/sdk` (JS):** `base_url=` / `baseURL`, or the `TYPESAFE_BASE_URL`
  environment variable, set to the server root (`http://127.0.0.1:8080`).
- **The Vercel provider:** `baseURL` must include `/v1`.
- **LangChain:** the `base_url` / `baseUrl` field.

```json
{"model": "local", "state": "Help! My payouts have been failing for 3 days.",
 "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                              "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": null}},
               "is_urgent":  {"type": "noul", "instructions": "Is this urgent?"}}}
```

```json
{"model": "local",
 "answers": {"department": {"type": "choice", "choice": "billing", "confidence": 0.77,
                            "probabilities": {"billing": 0.85, "technical": 0.12, "sales": 0.03}},
             "is_urgent":  {"type": "noul", "noul": 0.91}},
 "usage": {"input_tokens": 212, "output_tokens": 0},
 "goinfer": {"route": "label", "template": "chat-v1", "calibrated": {"choice": false, "noul": false}}}
```

- **Kinds:**
  - `noul` returns P(true). Its optional `criteria` (`{"true": …, "false": …}`) is shown to the model.
  - `choice` takes 2–16 options, the letters A–P that label scoring reads; TypeSafe allows 255. The options come
    from the `criteria` keys, in the order sent, and each description is shown beside its option.
  - `score` takes 2–10 levels, one per `criteria` entry. It returns the expected level, a `legend` and the
    probabilities by level.
- **`confidence`** (choice and score) is the top probability's margin over uniform, `(n·p − 1)/(n − 1)`: 0 when the
  model cannot tell the options apart, 1 when it is certain. TypeSafe does not publish its formula; this is the
  form its docs' demo uses for three options. jevx compares it against its `min_confidence` (0.6).
- **`usage`:** `output_tokens` is 0, because nothing is decoded.
- **`goinfer`** says how the answer was made: the route, the template, and which kinds a fitted temperature
  calibrated.
- **What it is not: TypeSafe's hosted model.** Without a head (below), the probabilities are the served model's own over
  the options it was shown.
  - They are calibrated only for a kind whose temperature `--decisions-calibration` supplies: a `calibration.json`
    from `goinfer-chat decisions-calibrate`, fitted under the same template.
  - How good they are depends on the model. On Qwen3.5-9B, the authors of the open JEV models measured this method
    with their bare template at choice top-1 0.53, against their trained head's 0.90
    (`measurements/decisions-d0-prior-art-2026-09-27.md`).
  - goinfer's own measurement (D6a, `measurements/decisions-d6a-2026-09-28.md`) on Qwen3.5-9B Q4_K_M with its chat
    template, calibrated, is top-1 0.42 and ECE 0.17 on an out-of-distribution sample, against the trained JEV-9B's
    0.92. Label scoring is not a substitute for a trained head.
- **`--decisions-template`** is `chat-v1` (the default: the model's chat template, for instruct models) or
  `bare-v1` (JEV's own, no chat template). One prefill per question; each question re-prefills the state.
- **Many questions about one state cost one full prefill each.** On the hybrid families (Qwen3.5 and the other
  Gated-DeltaNet models) no prefix is reused between questions today. A request's cost therefore grows with its
  question count: measured on Qwen3.5-9B Q4_K_M (CUDA resident int4, batch 1, 2026-10-02), five questions cost five
  times one, and take 1.76x (95% interval 1.72 to 1.80), 3.21x (3.16 to 3.26) and 4.47x (4.43 to 4.51) as long as
  one schema-constrained generation that answers all five fields, at 256, 1,024 and 4,096 state tokens (4.6, 16.0 and
  78.1 s against 2.6, 5.0 and 17.5 s). For a single question a decision is about as fast as a schema-constrained
  answer (1.24x, 1.07x and 1.01x faster at those sizes), so for a fixed set of questions on this family the
  one-pass schema request is the cheaper shape. The measurement is
  [`measurements/decisions-d7-2026-09-28.md`](measurements/decisions-d7-2026-09-28.md) section 5; the work that would
  remove the repeated prefill is D8 in `tasks/task-constrained-confidence.md`, whose trigger this measurement met.
- **Refused:** a question that breaks a rule is a 422 before any prefill. A compute-time adapter entry cannot answer,
  since label scoring would read the base model. `/v1/models` lists each entry's `decisions` support.
- **With a trained decision head (Route B):** `--model jev=~/models/JEV-9B,head=~/models/JEV-9B`. `head=` takes a
  directory in autotrust's JEV layout (`judge_config.json`, `head.safetensors`, `calibration.json`, and `adapter/`
  when the head's weights are unmerged). That entry then answers `/v1/systemone` the way the JEV reference does:
  - **The answer:** the final-norm hidden state at the last prompt token goes through the head, then the kind's slots,
    then the head's own per-kind temperatures. `goinfer.route` is `"head"`, and `--decisions-template` /
    `--decisions-calibration` do not apply to it.
  - **The prompt is JEV's template, bare-v1**:
    - option descriptions are left out, as JEV's template has none, and `goinfer.descriptions_dropped` names the
      questions that had them;
    - a score must have JEV's six levels (0–5), or the question is a 422;
    - a state over 1024 tokens is cut to its first and last parts as the reference cuts it.
  - **The adapter is merged at load.** That entry's text generation is the decision model's, not the base model's.
    Serve the base model as another entry if you need both.
    - A `.giw` built with `prequant -lora DIR` carries the adapter already, so no merge happens at load.
    - `head=` checks the bundle's `.lora.json` sidecar and refuses a bundle built without that exact adapter.
  - **Precision: a model with `head=` loads at `int8int8` unless `quant=` or `--quant` says otherwise** (the owner's D6b decision, 2026-10-02). Against the transformers f32 reference on 150 items it agrees on 92.7% of top answers (mean KL 0.009; every one of its 11 disagreements is a near-tie, reference margin 0.10 or less)
    and is no worse calibrated; it does **not** meet the registered 98% top-1 bar, and nothing here says it does. f32 is exact (`quant=f32`, about 36 GB for JEV-9B, so it fits a large-RAM box and no GPU). That grading is of the CPU path: **int8int8 on a CUDA or Metal resident has not been graded**
    (`measurements/decisions-d6b-2026-09/results.md`).
  - **It runs on the GPU when the model is resident** (CUDA or Metal), through the same headless forward embeddings
    use, and falls back to the CPU otherwise.
    - On nobara's RTX 2070 SUPER, JEV-9B at int4 measured about 3 ms per prompt token, against about 65 ms on its
      CPU (exploratory, five items).
    - The GPU's int4 kernels are not the CPU's, so the two answers differ: slightly on most items, and on one of the 150 by a lot (a 0.956 option read as 0.362 on CUDA int4, unexplained). Until that is explained, do not rely on decisions from a CUDA-resident int4 model (D6b).
  - **How closely it tracks the reference** is D6b in `tasks/task-constrained-confidence.md`, not yet graded.

**Reasoning models (thinking).** Qwen3, Qwen3.5 and Gemma 4 can think before they answer, and their own chat templates
disagree about the default — Qwen3 and Qwen3.5-9B think unless told not to, Qwen3.5-0.8B and Gemma 4 do not. serve reads
the default from the checkpoint's own template (nothing is hard-wired per family) and keeps one rule on every route:
**`content` is the answer and never carries `<think>` markup; the reasoning travels in a separate field a client is free
to ignore.**

| client | what it sends | what it gets |
|---|---|---|
| OpenAI SDK / curl, reasoning-unaware | nothing special | clean `message.content`; an extra `reasoning_content` it ignores |
| Reasoning-aware OpenAI-compatible | `chat_template_kwargs: {"enable_thinking": true\|false}` | `reasoning_content` (streamed as `delta.reasoning_content`, all of it before any `content`) |
| Clients that parse `<think>` tags themselves | `reasoning_format: "none"` (or `-reasoning-format none`) | the raw text, tags and all, in `content` — exactly what serve sent before |
| Both at once | `reasoning_format: "deepseek-legacy"` | `reasoning_content` filled and the tags kept in `content` |
| Anthropic `/v1/messages` | `thinking: {"type": "enabled"\|"adaptive"\|"disabled"}` | a `thinking` block first, then `text` (and `tool_use`); with no `thinking` field, text only — reasoning is dropped |
| `/v1/responses` | `reasoning: {"effort": "none"}` turns it off | clean `output_text`; reasoning items are not returned |

- **Defaults.** `-thinking template` (the default) renders what each checkpoint's own chat template renders when thinking
  is unspecified, so serve behaves as the model's card says: Qwen3.5-0.8B and Gemma 4 do not think; Qwen3.5-9B and Qwen3 do.
  `-thinking asis` renders the prompt serve produced before thinking was modelled (nothing written after the assistant tag,
  the model decides); `on` and `off` force it. A request overrides the flag. What changed from that older behaviour: a small
  Qwen3.5 now gets its closed empty think block and answers directly, as HuggingFace's default does; Qwen3, Qwen3.5-9B and
  Gemma 4 prompts decode the same as before. `reasoning_effort: "none"` turns thinking off; any other value changes
  nothing, because clients such as dsh send a bare `reasoning_effort` to every endpoint and must not have their prompts
  flipped by it.
- **A thinking reply always gets room to answer (the reasoning budget).** `max_tokens` counts thinking too, so a thinking
  reply could run out of tokens inside its block and return nothing (empty `content`, `finish_reason: length`). serve prevents
  that: once the block has used its budget, the next token is forced to be the block's closing token and the model writes its
  answer in what remains. The budget is the smaller of what the request asked for (`thinking_token_budget` in an OpenAI-style
  body, or Anthropic's `thinking.budget_tokens`; else `-reasoning-budget N`) and what leaves room to answer: thinking takes at
  most three quarters of `max_tokens` (and leaves at least 2 tokens). `-reasoning-budget auto` (the default) applies only that
  room rule; `unlimited` applies none of serve's own (a request's budget still holds). The budget costs nothing until it is
  due — the decode fast paths stay on until then. **It does not apply** under `-spec` or `-drafter` (any logit processor would
  silently turn the drafter off for every thinking request), to a turn shorter than 4 tokens, to a model with no recognised
  thinking control, or to a model whose tokenizer has no single token for the block's delimiters. A reply that ends inside its
  block anyway (one of those cases) has empty `content`, `reasoning_content` holding what was written, and `finish_reason:
  length` (`stop_reason: max_tokens` with only a thinking block on `/v1/messages`).
- **Constrained requests render thinking-off.** A `response_format` of `json_object`/`json_schema`, and a tool call the
  server forces from the first token (a named or lone tool, or `required`), are grammar-constrained from token 1; a prompt
  that ends inside an open `<think>` would contradict the grammar, so those requests are rendered with thinking off.
- **Stop strings are matched against the answer only.** A `stop` (OpenAI) or `stop_sequences` (Anthropic) string is a request
  about what the model says to you, so it is checked against the answer, not the thinking: a stop string that appears in the
  model's reasoning no longer ends the reply there with no answer — the reasoning stays whole and the answer is written. A
  stop string in the answer ends the reply exactly as before, and one split across the thinking/answer boundary is not a
  match. Exceptions, where `content` carries the thinking too: `reasoning_format: "none"` and `deepseek-legacy` return the raw
  text, so a stop string in the thinking still stops the reply there.
- **Replayed reasoning (history).** A client that sends the model's reasoning back — `reasoning_content` or `reasoning` on an
  OpenAI assistant message, `thinking` blocks on an Anthropic one (Claude Code sends them with every turn of a tool loop) —
  has it rendered the way the model's own chat template would: **kept for the turns of the tool loop in progress** (those after
  the last user query) and **dropped for turns before it**. The families differ in the details and follow their own templates:
  Qwen3.5 always writes the `<think>` block for those turns (empty when there is no reasoning), Qwen3 only for the last message
  or a turn that has reasoning, Gemma 4 a thought channel when there is reasoning. A client that left `<think>…</think>` inside
  `content` instead has it extracted the way the Qwen templates do, so an old turn's tags are stripped rather than re-sent as prose.
  An Anthropic user message carrying `tool_result` blocks plus reminder text (Claude Code's shape) does **not** count as a new
  query, or the turn after the first tool result would lose its reasoning. Under `-thinking asis` replayed reasoning is ignored
  (the pre-thinking prompt). Consequence for the KV cache: when a new user query arrives, the previous loop's turns lose their
  reasoning, so the cached prefix is reusable only up to the first such turn — the templates' own design, not serve's.
  Not replicated: the templates also trim every message's content and Gemma 4 has a `preserve_thinking` option.
- **Not split:** a model whose template has no recognised thinking control (everything else, and any template shape not
  read from a real checkpoint) is served exactly as before. `goinfer-chat` and the demo agent apply the same separation (below).
- **`goinfer-chat` and the demo agent.** Both run the same machinery. `goinfer-chat` takes `--thinking template|asis|on|off` (default
  `template`, as serve; `/think <mode>` changes it mid-session) and `--show-thinking` (default true): the reasoning prints dimmed
  before the answer, the answer in cyan, and only the answer enters the conversation history; `--show-thinking=false` prints a
  `(thinking…)` marker instead. The reasoning budget applies (three quarters of `--max`), except under `--schema` / JSON mode (whose
  prompt is rendered thinking-off), `--draft` and `--spec ngram`. The demo agent takes the same `-thinking` flag; its DECIDE phase
  (a JSON grammar) is always thinking-off; `stdlib-agent` prints the reasoning dimmed, `agent-web` drops it (its page keeps its own
  "Thinking…" indicator).
- **Jobs and batches** apply the same split; a job's result and a batch line's `content` are the clean answer.
- **gpt-oss (Harmony).** Its reply is not one `<think>` span but channel messages — `analysis`, then `final`, and `commentary`
  when it acts — and the same rule holds: `content` is the answer, `reasoning_content` the `analysis` channel, no `<|channel|>`
  markup in either. `final` and a `commentary` preamble are content (several messages are joined by a blank line); a message
  addressed to a function (`to=functions.…`) is a tool call (below), shown in neither stream.
  gpt-oss always reasons: its prompt has no off form, so `-thinking off` / `enable_thinking: false` cannot stop it and the
  reasoning still arrives in `reasoning_content` (the load log says so). What it does take is an effort:
  **`reasoning_effort: "low" | "medium" | "high"`** (chat completions, and `reasoning.effort` on `/v1/responses`, jobs and batches)
  is written on the template's own `Reasoning:` line — medium when the request says nothing — and a lower effort spends fewer tokens
  before the answer. It acts for gpt-oss only: on every other model `reasoning_effort` still means just `"none"`, so a client that
  sends a bare `"high"` to every endpoint (dsh does) changes no other prompt; an unrecognised value is ignored, never a 400.
  `goinfer-chat` has `--reasoning-effort` and `/effort`, and a `--batch` line's own `reasoning_effort` overrides the flag. **The reasoning budget applies to it** (`-reasoning-budget`,
  `thinking_token_budget`, on by default): once the analysis has used its share, serve forces `<|end|><|start|>assistant<|channel|>
  final<|message|>` — the whole sequence, not one token, or the model could open another channel and still never answer — so a
  small `max_tokens` returns an answer instead of an empty `content`. For gpt-oss the budget counts the channel headers too (about
  nine tokens), because the answer starts only after them. It is off under speculative decoding and JSON grammars, as for the
  others. `-reasoning-format none` restores the raw text, markers included. A conversation's earlier
  assistant turns are replayed on the `final` channel, as the model's own template does, with their reasoning dropped.
- **gpt-oss tool calls** work on `/v1/chat/completions`, `/v1/responses` and `/v1/messages` with the same request and response shapes as
  every other family. Tools are declared in the developer message the way the model's own template writes them — a TypeScript-like
  `namespace functions { … }` — and a call and its result replay as the template renders them; both are pinned byte for byte against
  HuggingFace's rendering of the template from the GGUF, quirks included. Limits, stated plainly:
  - **A call's form is parsed, not constrained.** gpt-oss's call is a channel message, not a JSON wrapper a grammar can govern, so a
    named `tool_choice` is a 400 (as for Gemma 4) and `required` / `auto` run unconstrained: the model may answer in prose where
    `required` asked for a call. An argument object that is not valid JSON is left out rather than passed on.
  - **Prose is buffered, not streamed** on a tool-capable request; the reasoning still streams as `reasoning_content`.
  - Only `functions.NAME` recipients are calls: gpt-oss's built-in `browser` and `python` are never declared and are ignored if the
    model reaches for them. The model stops at the first call, so parallel calls in one turn do not occur; a history that carries
    several is replayed as several call messages.
  - The reasoning budget can end the analysis before the model has chosen to call a tool; with a tool-using agent, give it room
    (`-reasoning-budget unlimited`, or a larger `max_tokens`).
  - **Run live with a real Claude Code** (2026-10-01, `docs/measurements/claude-code-gptoss-2026-10-01/`): a `Read` tool loop completed through `/v1/messages` — thinking, tool call,
    result, answer. One finding: the model's first call named the tool `read`, not `Read`; Claude Code's own error ("tool names are case-sensitive: call Read instead") was enough for it to retry. A
    harness that does not say so would stop there. One run, one tool; not a measurement.

**Batch files — over HTTP, or locally with `goinfer-chat --batch` and no server.** One JSONL file, two ways to run it. Each
input line is `{"custom_id": "a1", "method": "POST", "url": "/v1/chat/completions", "body": {chat request}}`; `custom_id` is
required and must be unique (a duplicate is refused, naming both positions). Over HTTP it is `POST /v1/files` then `POST /v1/batches` (OpenAI) — or inline requests to
`POST /v1/messages/batches` (Anthropic). Locally:

```sh
goinfer-chat --model ~/models/qwen2.5-7b-instruct-q4_k_m.gguf --batch in.jsonl -o out.jsonl
```

The output is the real API's: `out.jsonl` holds a line per request that produced a response, `{"custom_id", "response":
{"status_code", "body"}, "error": null}`, and the lines that failed — `{"custom_id", "response": null, "error": {"code",
"message"}}` — go to `out.errors.jsonl` beside it (over HTTP, the batch's output and error files). The format is one piece of
code (`internal/batchio`) used by both, and `TestBatch_cliAndServeAgree` runs a file through both on the same model and checks
the replies are identical, line for line.

- **Resumable.** Each finished line is appended to `-o` and fsynced before the next starts. Rerun the same command after a
  crash, a Ctrl-C or a power cut and lines whose `custom_id` is already in the output are skipped; a final line the process died
  in the middle of is cut off and run again. A line that failed is not done, so a rerun retries it (`out.errors.jsonl` is
  rewritten each run and holds only that run's failures). A run with nothing left to do exits without loading the model.
- **A line that states no sampling settings gets the API's defaults** (temperature 1, no top-k/top-p, 512 tokens, a random
  seed, no system prompt), not `goinfer-chat`'s interactive ones — unless you pass the flag (`--temp 0`, `--seed`, `--max`,
  `--system`…), which is an instruction about this run. Put `temperature` and `seed` in the line for a reproducible file.
- **Scope is text chat,** as over HTTP. Supported per line: `messages` (string or text-part content, `system`/`developer`,
  replayed `reasoning_content`), `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `seed`,
  `frequency_penalty`, `presence_penalty`, `stop`, `response_format` (`json_object`, `json_schema`), and the thinking controls
  (`chat_template_kwargs.enable_thinking`, `reasoning_effort: "none"`, `thinking_token_budget`, `reasoning_format`
  `deepseek`|`none`). A line asking for what the runner cannot do — tools, tool messages, images, `logprobs`, `n` above 1,
  `goinfer_confidence` — is refused with an error line naming it, never answered without it.
- **Progress** goes to stderr, one line per finished line (`[137/20000] id ok  212 tok  18.3 tok/s  elapsed 4m12s  eta ~38m`;
  the estimate appears after ten lines and is a mean — lines differ in cost) and a `… still running` line each minute inside a
  slow one. The exit status is 0 when every line produced a response, 1 when some failed, 130 when interrupted.

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
RTX 2070 SUPER** vs ~30 tok/s on an M1 Pro CPU. This trades the session-based prompt-prefix KV
caching above for a different mechanism, not a loss of prefix reuse itself: GPU-resident models
(`-backend cuda`/`metal`/`webgpu`) take a STATELESS decode path with no session, but still get
resident prefix reuse (above) against their own resident KV — so a typical agent turn still
reprefills only the divergent suffix (the new tool result and reply), not the whole ~8k-token
history, as long as the fixed system prompt and tool schemas stay a genuine prefix of the next
turn. That is still the right trade here (measured 13 tok/s stateless
CPU/staged fallback vs ~460 resident on a 0.5B, RTX 2070 SUPER — the loss from going stateless is
far smaller than the loss from leaving the GPU's resident decode path); the doc's usual advice to
expect reuse across turns still applies once `-backend cuda` is in play, just via resident prefix
reuse rather than the session cache.

```bash
# Loopback:
goinfer-serve -model coder=~/models/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384

# Across a network: non-loopback REQUIRES an API key (serve refuses to start otherwise).
GOINFER_API_KEY=<secret> goinfer-serve -model coder=... -backend cuda -addr 0.0.0.0:8080 -ctx 16384
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
and abandons the work if the client disconnects. On families whose tool syntax allows it (ChatML/Qwen,
Mellum, Gemma 4), any prose the model writes before the call streams as it is generated, on all three
routes — `/v1/messages` (a text block, then the `tool_use` blocks), `/v1/chat/completions` and
`/v1/responses` — rather than arriving in one piece at the end. Deep context slows decode (see the benchmarks
below). If a turn hangs and no keep-alives arrive, you are on a pre-v0.15.0 build.
