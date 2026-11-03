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

## The reading, 2026-11-03 15:22 PST: the path works; one registered clause fails, on my own case order

`run/smoke.out`, `run/results.json`, and each load's log in `run/`.

| case | first load | second load | reply, both loads | peak `VmSwap` |
|---|---|---|---|---|
| Phi-3, its CUDA default | 4.5 s, **writes** `….q4k.e4h.cuda.v17.giw` (2.81 GB) and its `.verified` | 1.5 s, writes nothing | ` The capital of France is Paris.` | 0 MB |
| Phi-3 at `--quant q4k` | 2.0 s, writes nothing | 2.0 s, writes nothing | the same | 0 MB |
| the 0.5B at `--quant q4k` | 1.5 s, **writes** `….q4k.e4h.cuda.v17.giw` (616 MB) and its `.verified` | 1.0 s, writes nothing | `Paris is the capital of France.` | 0 MB |

Every load's banner reads `decode path: cuda-resident (q4k)`, and each second load opens the sidecar with no transcode line.

**The script's verdict is FAIL, on one clause: the Phi-3 `--quant q4k` case wrote no `.q4k` sidecar.** It had none to write. Phi-3's default on CUDA **is** q4k (its banner says so), so the case before it wrote that sidecar and this one reused it. That is the sharing a user wants, and my clause did not foresee it: the two cases should have run with the sidecar removed between them, which I did not do, so the Phi-3 `--quant q4k` first write was not read on its own. The default case's first write is the same file by name.

**Worth knowing, not failures:**
- On q4k every prompt here took the per-token path: `batched prefill declined … per-32 activation quantization`. Fine at 17 and 42 tokens; a long prompt's first token on this lane is slower, and nothing here measures by how much.
- Phi-3 at this context got 2 KV slots of 4, by its own banner: 1,664 MB a slot.

**What this does not cover:** one prompt per case says nothing about quality. No long prompt. No other q4k family. No timing here is a benchmark: page cache held the source files. Driver 595.112.07.

**Left on disk,** as a user's first load would leave them: `Phi-3-mini-4k-instruct-q4.q4k.e4h.cuda.v17.giw` (2.81 GB) and `qwen2.5-coder-0.5b-instruct-q4_k_m.q4k.e4h.cuda.v17.giw` (616 MB), each with its `.verified`, in `~/models`. Remove them if the space matters; the next first load writes them again.
