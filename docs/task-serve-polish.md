# Task (goinfer): serve polish — multi-model, dynamic load, Responses API

> **For:** Claude Code, in `~/tmcode/goinfer`. Track B of the v0.4 plan
> (`feature-plan-v0.2.md`). Increments are ordered and independently
> shippable; `go test -race ./...` and the serve integration tests stay
> green after every step. Pure stdlib `net/http` stays the rule — no deps.

## Current state (read first)

`cmd/serve` (~2k lines): one decoder + optional one encoder per process,
loaded at startup (`newServer` → `loadDecoder`/`loadEncoder` in `main.go`).
One global `mu sync.Mutex` in `openai.go` serializes generations. Warm KV
reuse via `sessionLRU` (`sessions.go`, `--kv-sessions`, `--session-dir`
snapshots bound to a `modelFingerprint`). Endpoints: chat/completions,
completions, models, embeddings. Tool calls buffer fully, then emit deltas
(`tools.go`).

## Increment 1 — per-model serialization (refactor, no features)

Prep for multi-model: move the global `mu` + sessionLRU + template + sampler
defaults into a `loadedModel` struct (decoder, name, fingerprint, mutex,
sessions). `server` holds exactly one for now. Behavior identical.

- [ ] Pure refactor; all existing serve tests pass unchanged.

## Increment 2 — multi-model serving (static)

Serve N generative models from one process.

- Flags: `--model` becomes repeatable (keep single-flag compat), each with
  optional `name=path` syntax for the served id; `--quant`/`--lora` stay
  global for now (per-model overrides only if trivially clean).
- Registry: `map[servedName]*loadedModel`; requests route on the OpenAI
  `model` field; unknown model → OpenAI-shaped 404 error body.
  `/v1/models` lists all (decoders + encoder).
- Memory note in `--help`: N resident int8 models are expensive; prequant
  `.giw` (`--model name=path.giw`) maps weights zero-copy and is the
  intended way to keep a model zoo cheap.
- Sessions: per-model sessionLRU (already keyed by fingerprint — keep
  `--session-dir` layout `<dir>/<fingerprint>/…` so models don't collide).
- [ ] Gate: integration test serving two tiny GGUFs; concurrent requests to
      different models proceed in parallel (per-model mutex); cross-model
      `model` field routing verified; `/v1/models` lists both.

## Increment 3 — dynamic load/unload

- Admin endpoints (mirror llama.cpp/mistral.rs conventions, but keep it
  small): `POST /admin/models/load {name, path, quant, lora?}`,
  `POST /admin/models/unload {name}`, both refusing while the target model
  is mid-generation (try-lock, 409 on busy). Gate behind `--allow-admin`
  (default off — this is RCE-adjacent surface; loading attacker-named paths
  must be a deliberate opt-in).
- Unload: drop the `loadedModel`, snapshot its warm sessions to
  `--session-dir` first if set. Document that Go's GC + mmap means RSS
  shrinks lazily.
- [ ] Gate: load → generate → unload → 404 → reload cycle in a test;
      unload-while-busy returns 409; `--allow-admin` off → 403.

## Increment 4 — OpenAI Responses API (`/v1/responses`)

The API surface is shifting under Chat Completions; mistral.rs already has
initial support. Scope tightly:

- **Phase A (stateless):** `input` (string | message items), `instructions`,
  `max_output_tokens`, sampling params, `text.format` (maps to the constrain
  grammar exactly like `response_format`), `tools`/`tool_choice` (reuse
  `tools.go` machinery), `stream` with the Responses event shapes
  (`response.created`, `response.output_text.delta`,
  `response.completed`, …). Non-stream returns the `response` object with
  `output` items.
- **Phase B (stateful):** `store`/`previous_response_id` via an in-memory
  ring of recent responses (id → rendered conversation tokens). This is a
  natural fit: a continued response is by construction a prompt-prefix
  extension, so it rides the per-model sessionLRU for warm KV. No disk
  persistence; document the lifetime.
- Explicitly out: hosted tools (web_search etc.), reasoning items, file
  inputs.
- [ ] Gate: OpenAI Go/Python SDK pointed at `/v1` exercises create +
      stream + previous_response_id; tool-call round-trip via Responses
      shapes; constrain via `text.format` produces schema-valid output.

## Increment 5 (stretch) — decode workers + honest backpressure

Per-model: bounded queue (`--max-queue`, default small) + single decode
worker (today's behavior) → N workers later only if real demand; queue-full
returns 429 with `Retry-After`. Do NOT attempt continuous batching — still
explicitly deferred.

- [ ] Gate: burst test gets 429s, never deadlocks; latency of queued
      requests bounded by queue depth × generation budget.

## Out of scope (recorded so it isn't relitigated)

- True incremental tool-call streaming (buffered-then-deltas stays; revisit
  on demand).
- MCP host surface, OAuth — wrong layer for goinfer; consumers compose.
- Multi-encoder embeddings; multimodal; continuous batching.

## Definition of done

- [ ] Increments 1–4 landed, each with its gate; 5 optional.
- [ ] `cmd/serve` README section updated (multi-model flags, admin API +
      its security note, Responses coverage matrix).
- [ ] CHANGELOG entries per increment as they land.
- [ ] Greedy decode parity (`TestDecodeParity`) untouched throughout — none
      of this touches the forward pass.
