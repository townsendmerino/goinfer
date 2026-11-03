# A q4k serve smoke on nobara's CUDA (release item 3b for v0.24.0, 2026-11-03)

**Why:** every q4k `.gguf` writes a `.giw` sidecar on its first load, Phi-3's CUDA default included, and that path has never run on CUDA.

**Registered 15:22 PST, before any run (nobara-sb01).** Serve at `0f5cb935a` (main), driver 595.112.07, from `~/models` on NVMe, greedy, under the timing lock, shared, as nothing here is timed for a row. Expected finish about 15:30 PST.

| case | command |
|---|---|
| Phi-3, its CUDA default | `--model ~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf --backend cuda` |
| Phi-3 at q4k | the same with `--quant q4k` |
| the 0.5B at q4k | `--model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --backend cuda --quant q4k` |

Each case: load, serve one short chat, stop; load again, serve the same chat, stop.

**It passes when, in each case:**
- the load is resident on CUDA, by its banner;
- at q4k, the first load writes a `.q4k` sidecar and its `.verified` mark, and the default case's sidecar is whichever the banner names;
- the second load writes nothing, by size and modification time, and opens the sidecar;
- the reply is non-empty and reads as an answer, and **the second reply equals the first, byte for byte**;
- the server's peak `VmSwap` is at most 512 MB.

**A failure is a fix or a named report, not a pass.** A refusal to load is a failure here, not a NOT RUN, unless the fit check names the card. This is a smoke test: one prompt per case says nothing about quality, and no timing here is a benchmark.

**Side effect:** the sidecars this writes stay beside their models in `~/models`, as any user's first load leaves them; they are listed in the record.
