# aikit task: measure W4A8's ops-per-byte before anyone redesigns a quant format

> **STATUS: OPEN — no result recorded anywhere.** Checked 2026-08-21: nothing in
> `docs/queue-performance.md`, `docs/ollama-chase.md` or the benchmarks page records an
> ops-per-byte or arithmetic-intensity measurement for W4A8. The premise the prompt guards
> ("measure it before anyone redesigns a quant format") is therefore still unguarded.


## Why this task (read first)

goinfer decodes **Qwen3.8-27B at 0.656 tok/s on CPU** where **Ollama/llama.cpp does 1.674** on the
same box — 2.55×. Two explanations were proposed and **both have been ruled out with measurements**,
which is why this task exists: what remains is a claim about aikit's kernel, and it should be
measured before it is acted on.

**Ruled out #1 — the parallelization threshold.** aikit reasonably suspected the `int4ParThreshold`
bug class that cost gemma4-26B a measured 2.3× (decode-time M=1 matmuls falling under aikit's
16.78M-MAC serial default). On Qwen3.8 every projection is far above the 1<<20 threshold:

| projection | shape | MACs | vs 1<<20 |
|---|---|---|---|
| FFN gate/up/down | [17408,5120] | 89.1M | ×85 |
| softmax `q_proj` | [12288,5120] | 62.9M | ×60 |
| DeltaNet `in_proj_qkv` | [10240,5120] | 52.4M | ×50 |
| DeltaNet `in_proj_z` / `out_proj` | [6144,5120] | 31.5M | ×30 |
| softmax `k`/`v_proj` | [1024,5120] | 5.24M | ×5 |
| LM head | [248320,5120] | 1271M | ×1212 |
| `in_proj_a`/`b` | [48,5120] | 0.25M | **below** — but f32, 0.002% of per-token MACs |

And the CPU profile shows the **opposite** of the gemma4 signature: **832% CPU** (the fan-out is
firing), `runtime.park_m` **1.22%**, and **no `chanrecv` in the top 60 by cumulative**. Time is in
`dotW4A8FoldAVX2` (58.3% flat), `dequantI8AVX2` (2.9%), `dotFMA` (1.0%).

**Ruled out #2 — quant format bytes.** aikit's own analysis: int4 packs one f32 scale per 32-weight
group = 5.0 bits/weight; Q4_K_M shares a 6-bit-quantized scale+min across 8 sub-blocks of a
256-weight superblock ≈ 4.5 bits/weight. **~11%.** Nowhere near 2.55×.

**What is left.** Per token: 25.62 GMAC, 17.9 GB of quantized weights streamed.

```
goinfer   17.9 GB / 1.524 s = 11.7 GB/s   at 832% CPU   (16.8 GMAC/s)
ollama    16.5 GB / 0.597 s = 27.6 GB/s
DDR4-3200 dual-channel peak ≈ 51 GB/s
```

**Neither engine saturates memory bandwidth**, so neither is bandwidth-bound at this size — the
kernel is **compute-limited**, doing roughly 2.4× more work per weight byte than llama.cpp's. That is
a statement about the inner loop, and it is the one thing nobody has measured.

## What to measure (a micro-benchmark, not a rewrite)

**One kernel, one shape, against its own theoretical.** Take `dotW4A8FoldAVX2` at
**[17408, 5120] M=1** (the FFN shape, 89.1M MACs, the largest per-token contributor) and answer:

1. **Achieved ops/byte vs theoretical.** For each 32-weight group the kernel must load 16 bytes of
   nibbles + 4 bytes of f32 scale and produce 32 MACs. What fraction of peak int8-dot throughput
   (VPMADDUBSW/VPMADDWD chains) does it reach, single-threaded, weights hot in L3 and weights cold
   from DRAM — report both, they answer different questions.
2. **Where the extra work goes.** The three candidates, in the order I would check them:
   - **per-group scale handling** — one f32 load + broadcast + FMA per 32 weights, versus Q4_K's
     6-bit scales shared across a 256-weight superblock (8× fewer scale loads and no f32 broadcast
     in the inner loop);
   - **nibble unpack** — shift/mask/`vpunpck` cost per 32 weights, and whether it is on the critical
     dependency chain or hidden under the loads;
   - **accumulator width and reduction** — int16 vs int32 accumulation, and how often the horizontal
     reduction runs.
3. **Whether the activation quantization is on the hot path.** `dequantI8AVX2` is 2.9% of the
   profile; confirm it is per-matmul rather than per-token, and whether reusing a quantized
   activation across the gate/up pair (they share an input) is already happening.

## What NOT to do

- **Do not start a Q4_K-style superblock format.** It is the expensive answer to a question that has
  not been asked yet: if the inner loop is at 80% of theoretical, the format is the only lever left
  and the project is justified; if it is at 25%, the format is a distraction from whatever is
  costing the other 55%. Measure first.
- **Do not tune `int4ParThreshold` for this model.** It is not implicated — see above.
- Do not chase the DeltaNet recurrence. Measured at **4.5%** of total CPU; a perfect 10× buys ~4%.

## Arch caveat — the numbers above are amd64/AVX2, and the Mac is not

Every figure here was measured on **linux/amd64, Ryzen 7 3700X (8c/16t), DDR4-3200 dual channel**,
in `dotW4A8FoldAVX2`. On the MacBook the equivalent path is the NEON kernel on unified memory with a
very different bandwidth profile, so:

- **The method transfers; the numbers do not.** Ops-per-byte against theoretical is the right
  question on either arch, but do not expect 11.7 GB/s to reproduce.
- If you want to reproduce the **goinfer-vs-llama.cpp gap** rather than just profile the kernel, you
  need both engines on the same machine with the same model — and on the Mac that means an
  arm64 Ollama build plus the 16.5 GB `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M.
- Cross-arch conclusions are the trap here: a NEON inner loop can be at 80% of its theoretical while
  the AVX2 one is at 25%, and "the kernel is fine" would then be true and useless.

## Reproducing the goinfer side

```
GOINFER_HEAVY_TESTS=1 GOINFER_QWEN38=~/models/qwen3.8-27b \
  go test -tags realckpt ./decoder/ -run TestQwen38Real_gate -v
```

The profile above came from 12 decode tokens under `pprof.StartCPUProfile` on the same model at
`Options{Quant:"int4"}`. Two goinfer-side changes landed on the way here and are already in main, so
re-measure against current main rather than an older number: the qwen35 family's DeltaNet/softmax
projections are now quantized (they were f32 at every quant — 1.60× decode, 7.4× TTFT), and the
dense `qwen35` GGUF loader exists (16.5 GB instead of 55.6 GB of bf16).

## Done means

A number: **achieved ops/byte as a fraction of theoretical**, for one shape, hot and cold, with the
breakdown of where the remainder goes. Plus a recommendation that follows from it — including
"the kernel is near its ceiling, the format is the lever" if that is what the measurement says.
