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
- **Not TypeSafe's hosted model.** This recipe serves a model by label scoring (Route A), and how good the answers are depends on the model. **Measured (D6a, 2026-09-28,
  [`decisions-d6a-2026-09-28.md`](../measurements/decisions-d6a-2026-09-28.md)):** Qwen3.5-9B with the `chat-v1` template, calibrated, reads top-1 0.4197 and ECE 0.1656 on the fixture where JEV-9B's trained head reads top-1 0.9181, so label scoring is far behind a trained head.
  A model served with a trained head (`head=DIR`: the JEV-9B and Clef routes) answers by that route instead: `docs/server.md`, "Decisions".
- **Many questions about one state.** On the GPU-resident path each question is a full prefill of the state, so a request's cost grows with its question count. On the CPU path, `D8` (built 2026-10-03) shares the state across a request's questions
  ([`decisions-d8-shared-state-2026-10-03.md`](../measurements/decisions-d8-shared-state-2026-10-03.md)). **Measured (D7, 2026-10-02,
  [`decisions-d7-2026-09-28.md`](../measurements/decisions-d7-2026-09-28.md)):** for one question a decision was faster than a schema-constrained answer (1.24x, 1.07x and 1.01x at 256, 1,024 and 4,096 state tokens), and for five questions it took 1.76x, 3.21x and 4.47x as long as one
  schema pass. If the questions are fixed, one schema-constrained generation that answers every field in a single pass (`response_format` with `"goinfer_confidence": true`, `docs/server.md`) is the faster shape for many questions.
