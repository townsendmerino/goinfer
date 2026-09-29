# jevx and TypeSafe's SDKs → goinfer

goinfer serves TypeSafe's decisions wire shape at `POST /v1/systemone`: a state and named questions with a closed
answer set in, a probability distribution per question out (`docs/server.md`, "Decisions"). The point is that clients
written for TypeSafe work against a local model unchanged, once they are pointed at it.

**Not yet run end to end.** Each setting below is what the client's own docs or source say it reads, recorded in
[`decisions-d0-prior-art-2026-09-27.md`](../measurements/decisions-d0-prior-art-2026-09-27.md). None has been driven
against a live goinfer yet. When one is, it goes here with its date and version, as the other recipes do.

## Serve a model

```
goinfer-serve -model local=<path>/Qwen3.5-9B-Q4_K_M.gguf -ctx 4096
```

- **`model`:** every client below must send goinfer's served name (`local` here). An unknown name such as
  `jev-latest` is refused, and the refusal names what is served.
- **`--decisions-template`:** `chat-v1` (the default, for instruct models) or `bare-v1` (JEV's own template, for
  base models).
- **`--decisions-calibration calibration.json`** applies per-kind temperatures from
  `goinfer-chat decisions-calibrate`, fitted on your own labelled examples under the same template. Without it the
  probabilities are the model's raw ones, and the response says so (`goinfer.calibrated`).

## Point the client at it

| client | setting |
|---|---|
| [jevx](https://github.com/muthuishere/jevx) | a profile's `url` is the full endpoint: `"url": "http://127.0.0.1:8080/v1/systemone", "model": "local"` |
| `typesafe-sdk` (Python) | `base_url="http://127.0.0.1:8080"`, or `TYPESAFE_BASE_URL` |
| `@typesafe-ai/sdk` (JS) | `baseURL: "http://127.0.0.1:8080"`, or `TYPESAFE_BASE_URL` |
| the Vercel AI SDK provider | `baseURL` must include `/v1`: `http://127.0.0.1:8080/v1` |
| LangChain | `base_url` / `baseUrl` |

## What you get, and what you do not

- **Label scoring on the served model:** one prefill per question, with the model's probabilities read at the option
  labels. Nothing is decoded, so `usage.output_tokens` is 0.
- **Not TypeSafe's hosted model, and not a trained decision head.** How good the answers are depends on the model.
  goinfer's measurement on Qwen3.5-9B (D6a in
  [`task-constrained-confidence.md`](../tasks/task-constrained-confidence.md)) is pending.
- **Many questions about one state re-read the state once per question.** On Qwen3.5 no prefix is shared between
  questions today, so a request's cost grows with its question count. If the questions are fixed, one
  schema-constrained generation that answers every field in a single pass may be the faster shape
  (`response_format` with `"goinfer_confidence": true`, `docs/server.md`). The speed comparison is D7; its projection
  is recorded, and the measurement is not yet run.
