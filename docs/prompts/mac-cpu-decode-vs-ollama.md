# goinfer decodes ~2x SLOWER than ollama on CPU — find out why, on Apple Silicon

> Written 2026-08-22 against goinfer `f5430ed`, from the v0.15.0 peer benchmark measured on
> `linux-62gb` (AMD Ryzen 7 3700X, 16 threads, amd64/AVX2). **This is a diagnosis task, not a
> fix-it task** — the deliverable is a cause with evidence, and a recommendation. Do not optimise
> anything until §1 has answered the confound, because the most likely single explanation makes
> most of the obvious optimisations pointless.

## The measurement

`scripts/bench_peer.py`, both engines driven over their own HTTP server, decode-only (rate timed
client-side from the first streamed token, so prefill is excluded on both sides by construction),
interleaved cell by cell with a server restart between cells, greedy, `NGEN=64`, `NCOMP=8`,
`NRUNS=2`. Depth 128. Raw data: `/home/francis/bench-v0.15.0/results.json` on the Linux box.

| model (q4_k_m) | goinfer CPU | ollama CPU | ratio |
|---|---|---|---|
| qwen2.5-coder-0.5b | 27.4 tok/s | 50.7 tok/s | **0.54x** |
| qwen2.5-coder-1.5b | 10.6 tok/s | 24.2 tok/s | **0.44x** |
| qwen2.5-7b-instruct | 4.2 tok/s | 5.9 tok/s | **0.71x** |

Spread across runs was 0.1–0.9 tok/s, so this is not noise. The gap is worst at 1.5B and narrows
at 7B, which is itself a clue: a fixed per-token overhead would shrink as the model grows, and a
pure kernel-throughput deficit would not.

**On CUDA goinfer is ahead at the same depth** (1.08x / 1.13x / 1.00x), so whatever this is, it is
CPU-path specific and not a scheduling or serving-layer defect shared by both backends.

## What is already verified — do not re-derive it

- **Both engines run bit-identical weights.** `scripts/gguf_same_weights.py` compares per tensor at
  each file's own offsets: 291/291, 339/339, 339/339 tensors identical for the three models. The
  ollama models were built from goinfer's own GGUFs with a `FROM` Modelfile.
- **A file-md5 check cannot pass and is not evidence of anything.** `ollama create` repacks the
  container in a different tensor ORDER — metadata identical, all offsets different. Do not "fix"
  a failing md5; that is expected. See `docs/measurements/bench-peer-v0.15.0-RUNNING.md`.
- **The harness excludes prefill on both sides.** This is a decode-rate gap, not a TTFT artifact.
- **Ollama was genuinely on CPU**, forced with `options.num_gpu=0`. Without that it silently uses
  the GPU; the force is in `ollama_payload` in `scripts/bench_peer.py`.

## §1 — Rule this out FIRST. It may be the whole answer, and it changes the task.

**The two engines were not running the same quantization.** goinfer was launched with
`-quant int4`, which is its own W4A8 path (`w.MatmulBTW4A8Into`, see
`decoder/weightmat.go:270`) — 4-bit weights requantized at load, int8 activations. Ollama ran the
file's **native Q4_K_M** k-quant with llama.cpp's hand-written kernels for that exact format.

Those are different weight formats, different activation precisions, and different kernel families.
So the headline may be measuring *"goinfer's W4A8 vs llama.cpp's Q4_K_M"* rather than
*"goinfer's CPU backend vs llama.cpp's"* — and only the second is a fair statement of the deficit.

**Do this before anything else:** re-run the goinfer CPU cells at the other quantizations the CPU
path supports (at minimum `int8`, and f32 if it fits) and see how much of the gap survives.

- If goinfer at some other quant closes most of the gap → the finding is *"our int4/W4A8 CPU kernel
  is the slow path"*, which is a narrow, actionable kernel result, and §2/§3 below are where to look.
- If every goinfer quant is ~2x down → the deficit is structural (threading, per-token overhead,
  memory traffic) and the quant choice is a red herring. §4 becomes the priority.

Report which of these it is **before** optimising. Getting this backwards means tuning a kernel
that was never the problem.

## §2 — Prior art that bears directly on this, and constrains the answer

- `docs/measurements/aikit-w4a8-opsperbyte.md` is the existing W4A8 inner-loop measurement.
  **Caveat: its method citations point at aikit files that do not exist at any commit** — see
  `docs/prompts/goinfer-w4a8-opsperbyte-citations.md`. Use its numbers as a hypothesis, not as
  something you can re-run.
- **Low-bit unpack was already found compute-bound on NEON** and the two weight-memory items that
  depended on it (#1-A, #6) were shelved for that reason. If §1 lands on "the int4 kernel", that
  shelved result is the most relevant prior evidence and probably predicts what you will find.

## §3 — If it is the kernel

The question to answer is ops-per-byte, not "is there a faster instruction". Q4_K_M's superblock
layout and llama.cpp's kernels amortize the unpack differently than W4A8 does. Measure:

- unpack cost per weight byte, goinfer vs the equivalent llama.cpp path
- whether goinfer's decode GEMV is bandwidth-bound or issue-bound on this Mac (they need different
  fixes, and on Apple Silicon the memory system makes it easy to guess wrong)
- whether the int8 activation quantization step (`QuantizeActivationsInto`) is on the per-token
  critical path and how much of the token it costs

## §4 — If it is structural

Check, in this order, cheapest first:

1. **Thread count actually used.** goinfer derives workers from `GOMAXPROCS` (see
   `decoder/weights.go:281`, `decoder/sampler_chunked.go:111`). Confirm what it really runs with on
   the Mac and what ollama runs with — llama.cpp defaults to physical cores and Apple Silicon's
   P/E-core split makes "all cores" the wrong answer. **An E-core-inclusive thread count that
   ollama avoids and goinfer does not would produce exactly this shape of result.**
2. **Per-token overhead independent of model size** — the gap narrowing from 0.44x at 1.5B to 0.71x
   at 7B is consistent with a fixed cost per token. Time a decode step with the matmuls stubbed out.
3. **Sampler and detokenization** on the per-token path.

## Scope, and what NOT to chase

- **arm64 numbers must be measured, not assumed.** Every figure above is amd64/AVX2. goinfer and
  llama.cpp both have separate NEON paths, so **re-measure the baseline on the Mac first** and
  report it — the gap there may be a different size, or absent. If the Mac shows no gap, that is a
  publishable finding on its own and the task ends there.
- **Out of scope: the CUDA depth curve.** The same benchmark found goinfer degrading with context
  depth on CUDA (0.70–0.78x of ollama at depth 3900, while ollama stays nearly flat). That is real
  and it is being tracked separately. Do not fold it into this task.
- Do not change published figures anywhere. This task produces a diagnosis; any re-measurement of
  a claim is a separate change with its own evidence.

## Done looks like

A short measurement doc under `docs/measurements/` that states: the Mac's own goinfer-vs-ollama CPU
baseline, the §1 answer (quant confound or not), the identified cause with the measurement that
pins it, and a recommendation with a rough size — including "not worth fixing" if that is what the
evidence says. A negative result costs the same to obtain and is worth as much.
