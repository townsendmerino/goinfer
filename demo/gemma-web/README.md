# gemma-web — local web chat GUI for the pure-Go Gemma 3 decoder

A single-binary, single-page chat UI over the `decoder` + `tokenizer` packages.
**Pure Go standard library** — `net/http` + Server-Sent Events, one embedded
HTML page (`//go:embed`), no external assets/CDNs, no cgo, no new dependencies.

## Run

```bash
# from the repo root, against the checkpoint already under testdata/:
go run ./demo/gemma-web --model testdata/gemma-3-270m
# → gemma-web listening on http://127.0.0.1:8080

# or point --model at any HF-layout Gemma 3 dir you have:
go run ./demo/gemma-web --model ~/models/gemma-3-270m-it --quant int8
```

Then open the printed URL in a browser.

Flags:

| flag | default | meaning |
|---|---|---|
| `--model` | (required) | Gemma 3 checkpoint dir (`config.json` + `model.safetensors` + tokenizer) |
| `--addr` | `127.0.0.1:8080` | listen address |
| `--backend` | `cpu` | `cpu` or `webgpu` (webgpu needs `-tags gpu`) |
| `--quant` | `` | `` (f32) or `int8` (per-row weight quant) |

The model + tokenizer load **once** at startup and are reused for every request.

## UI

Multi-turn chat with user/model bubbles, plus:

- **System prompt** — folded into the first user turn (Gemma has no system role).
- **Sampling** — temperature (0 = greedy), top-k, top-p, max tokens, seed.
- **Live stats** while streaming — token count, tokens/sec, time-to-first-token.
- **Stop** — aborts the in-flight request (cancels generation server-side via the
  request context); the partial reply is kept.
- **Regenerate** — drops the last model reply and re-runs the same user turn.

Conversation state lives in the browser; the full `messages` array is sent each
request (the model is stateless per call). Nothing is persisted.

## A note on the 270M checkpoint

`google/gemma-3-270m` is a **base** model, not instruction-tuned. Under a chat
template it tends to meander or repeat rather than answer — that's the model,
not the GUI. For coherent chat use an instruction-tuned checkpoint (a `*-it`
variant) or a larger model; the server, chat template, and streaming are
identical regardless.

## How it works

- `GET /` serves the embedded `index.html`.
- `POST /chat` takes `{messages, system, temperature, topK, topP, seed,
  maxTokens}`, builds Gemma's chat-template prompt
  (`<start_of_turn>user\n…<end_of_turn>\n…<start_of_turn>model\n`, `addBOS=true`),
  generates with `StopIDs=[<end_of_turn>]`, and streams `text/event-stream`:
  `event: token` (newly-completed UTF-8 text — partial byte-fallback tokens are
  held back so no broken multibyte ever ships) then `event: done` (final stats).
- One generation at a time (the CPU backend is single-stream); a concurrent
  request gets `409`.
- Client disconnect / **Stop** cancels the per-request context, which stops
  `Generate`.
