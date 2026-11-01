- **Per-turn `timings` on every response, and a `cost` report on request (experimental).** Chat completions,
  completions, responses, messages, batches and jobs carry a `timings` object beside `usage`: `queue_wait_ms`,
  `prefill_ms`, `decode_ms` and `ttft_ms`, read at the server's stage boundaries, each absent rather than 0 when its
  boundary was not crossed. A stream carries it in its final usage event. A request body with `"goinfer": {"cost": true}`
  also gets `cost`, the turn's per-phase and per-operation counts (calls, bytes, kernel launches, syncs, allocations,
  time), counted on the CPU and on CUDA's resident `Generate` (with `--max-concurrent 1`; the default MC3 is not counted);
  other paths answer `cost: {}`. Both are vendor extensions,
  experimental until v1.0 (`docs/server.md`, "Per-turn timings and cost").
