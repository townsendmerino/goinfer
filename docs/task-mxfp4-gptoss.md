# task: MXFP4 quantization + gpt-oss model family, and the CPU decode scheduler

> Status: **NOT STARTED.** Drafted 2026-07-26 from a review of
> [`arizqi/cpubrrr`](https://github.com/arizqi/cpubrrr).
>
> Depends on `aikit/docs/internal/task-q8k-integer-accum.md` for §4 only. §1–§3 and §5
> are independent and can start immediately.

---

## 1. The gap

```
$ git grep -il -E 'mxfp4|gpt.?oss|gptoss'
(no output)
```

goinfer supports f32/bf16/f16 + int8/int4 from safetensors/GGUF/GPTQ/AWQ across ~20
architectures. It does **not** support MXFP4, and therefore cannot run **gpt-oss:20b** or
**gpt-oss-120b** at all.

This is a missing model family, not a performance knob. It is also the family where
cpubrrr measured its largest margin — **~77 tok/s vs llama.cpp's ~14 tok/s on an M4 Max
CPU, ~5×** — which says llama.cpp's MXFP4 MoE CPU path is leaving most of the available
memory bandwidth unused. *(Those are cpubrrr's own published figures on an M4 Max — an
external CPU-vs-CPU citation, not a goinfer measurement; used only to size the opportunity.)* That is an unusually large, unusually specific opening.

The capability matters more than the speed: "runs gpt-oss:20b and 120b, in pure Go, one
binary" is a headline. "Runs it 5× faster than llama.cpp on CPU" is the second sentence.

## 2. MXFP4 format

Microscaling FP4 (OCP MX spec): blocks of 32 elements, each element an `e2m1` 4-bit float
(1 sign, 2 exponent, 1 mantissa — 16 representable values), plus one shared 8-bit `e8m0`
power-of-two block scale. 4.25 bits/weight.

**Do not infer the layout.** cpubrrr's README is explicit that they recovered the
architecture math from llama.cpp source and verified their 4-bit unpacking **bit-for-bit
against the official `gguf` Python library before writing the forward pass** — which is
why their new architecture produced correct output on the first run. Replicate that order:

1. Write the unpacker.
2. Verify bit-exact against the reference `gguf` library on a real gpt-oss checkpoint.
3. Only then write the forward pass.

`scripts/gguf_index.py` and `scripts/extract_experts.py` in cpubrrr show the shape of the
extraction; the file layout is GGUF, which `embed/gguf.go` in aikit already parses.

### Kernel strategy
`e2m1` has 16 values, so the dequant table fits a NEON `tbl` lookup — cpubrrr's approach
is exact 4-bit arithmetic on **integer** hardware via `sdot`/`tbl`, with **no float
dequant in the inner loop**. Same principle as the Q8_K work: stay integer, convert once.

This is a natural sibling to the existing W4A8 path (`aikit/linalg/dot_w4a8_arm64.s`,
`quant_w4a8*`). Decide early whether MXFP4 becomes a new `WeightMat` kind in aikit or a
goinfer-local kernel; **prefer aikit** for consistency with int4, unless the block-scale
semantics force otherwise.

## 3. gpt-oss architecture

Sparse MoE. `gpt-oss:20b` ≈ 21B total / ~3.6B active per token. `gpt-oss-120b` ≈ 117B /
~5.1B active.

Register it through the existing architecture-descriptor path so the feature taxonomy
handles backend admission correctly. **Critically: it must be declined at load by the
CUDA and Metal backends until they implement MXFP4, and fall back to the CPU path — never
mis-run.** That guarantee is the load-bearing claim in the README; do not weaken it for a
new family.

**Memory.** 120b does not fit in RAM. cpubrrr runs it via `mmap`'d weights so it pages in
under pressure instead of OOM-killing. goinfer already has `aikit/mmap` (`MapReadOnly`,
`Advise`, `SpanCache` — an LRU of page-aligned spans under a byte budget), which is
exactly this substrate. Wire 120b through `SpanCache`; do not write a new mmap path.

## 4. Interleaved expert layout *(after aikit A1 §4)*

cpubrrr's largest MXFP4 win was quad-interleaving the weight bytes so each worker core
reads one sequential stream. For MoE decode — where per-token you touch a *subset* of
experts — the interleave must be **per-expert**, so that activating experts {3, 17, 22}
still yields one sequential stream per core within each.

Do this at prequant time and cache it in the `.giw` bundle. `cmd/prequant` already owns
that pipeline, and `.giw` already maps zero-copy, so the reordering cost is paid once.

Measure the kernel win and the layout win **separately**.

## 5. CPU decode scheduler — the yielding spin-barrier

Independent of everything above. Ship it on its own.

cpubrrr's most transferable debugging result: a **52 tok/s regression** traced to
thread-pool dispatch, where condvar wakeups silently cost **7.5 ms/token**. Their fix was
12 persistent workers running the whole forward pass behind a *yielding* spin-barrier —
spin briefly, then let the OS in.

The two failure modes they mapped:

- **Condvar/channel per layer:** wakeup latency dominates. At ~30 layers × a few barriers
  each, microseconds of scheduler latency become milliseconds per token.
- **Pure spin barrier:** collapses under jitter — one descheduled worker burns every other
  core spinning, and the failure is nonlinear.

The middle path is spin-then-yield. In Go: bounded spin on an atomic, then
`runtime.Gosched()`, then park. Persistent workers for the whole forward pass, not a pool
re-entered per layer.

**Audit `decoder`'s CPU path for both patterns before changing anything.** Instrument
per-token barrier wait time first — if goinfer's current path is already
worker-persistent, this is a smaller task than it looks, and the measurement tells you so
in an hour.

## 6. Acceptance criteria

1. MXFP4 unpacking bit-exact vs the reference `gguf` library, on a real checkpoint,
   **committed as a test with a golden fixture** — matching the existing parity discipline.
2. gpt-oss:20b runs end to end on the pure-Go CPU path and is parity-gated against the
   HuggingFace reference (argmax-exact + logit cosine), same bar as every other family.
3. gpt-oss-120b loads and decodes through `SpanCache` on a machine with less RAM than the
   model. Document the tok/s honestly, including the paging cost.
4. CUDA and Metal **decline** gpt-oss at load and fall back to CPU. Assert this in a test.
5. `docs/capability-matrix.md` and `docs/hardware-matrix.md` regenerate cleanly and pass
   the CI freshness gate.
6. Decode tok/s for gpt-oss:20b in `docs/benchmarks.md` with full provenance, measured
   server-to-server against llama.cpp/Ollama CPU-only. **Verify GPU placement from the
   server's own logs** — cpubrrr's README documents that Ollama's `num_gpu:0` is a
   *request, not a fact*, and that this contaminated their first published numbers. Check
   the logs; do not trust the flag.
7. Scheduler change (§5) measured independently: per-token barrier wait before and after.

## 7. Non-goals

- MXFP4 on CUDA or Metal. CPU first; GPU residency is a follow-up once the CPU path is
  parity-gated.
- Training or fine-tuning in MX formats.
- The other MX types (MXFP6, MXFP8, MXINT8).
- gpt-oss multimodal or tool-calling specifics beyond what the existing `chat` package
  already provides per-family.

## 8. First move

`./scripts/setup_gptoss.sh` in cpubrrr shows how they extract runtime data from a local
Ollama copy without copying weights. Pull `gpt-oss:20b`, write the unpacker, and get the
bit-exactness test against `gguf` green. That single test de-risks the whole task — it is
the step that let cpubrrr's forward pass work on the first run.
