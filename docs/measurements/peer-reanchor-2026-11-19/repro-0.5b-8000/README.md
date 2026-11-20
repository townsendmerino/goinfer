# The 0.5B @ 8000 VOID: why goinfer's greedy reply stops at 8 tokens on CUDA (2026-11-20)

**The cell:** sweep cell a, CUDA, Qwen2.5-Coder-0.5B Q4_K_M, depth 8000, `BENCH_CTX=8192`, greedy.
- Every one of goinfer's 24 completions, and its warm-up, returned 8 of 64 tokens. Both peers returned 64.
- The token gate voids the cell ([`../README.md`](../README.md), night 2).
- Reproduced here with one request at a time, on the pinned `serve-cuda-e29da447` and the cell's flags
  (`-backend cuda -ctx 8192 -quant int4`).
- Each check held the timing lock exclusive for under 2 minutes, from 04:59 to 05:30 PST. The scripts beside this file
  are the exact commands.

## What the request returns

From [`repro.sh`](repro.sh):
- `prompt_tokens` 8052 and `completion_tokens` 8, so 8060 of 8192: **not a context cap**.
- `finish_reason: stop`; the content is `'thanks\n<|im_start|>\n<|im_start|>\n<|endoftext|>'`.
- So the model emits its own end-of-text token after a few special tokens.

**The prompt** (`scripts/prompts.json` `0.5B:8000`, format essay-v2): about 8,000 repetitions of "the", then *"The text
above is only filler. Now write a long, detailed essay about the history of the printing press."*

**Other engines on the same prompt, greedy:**
- **Both peers continue the filler:**
  - Ollama 0.32.5: 64 tokens, "the the the…";
  - llama.cpp `a37cc46`: 64 tokens, " the the the…".
- **goinfer CPU** (same binary, `-backend cpu`): 64 tokens, "import the the the…".

## The first-token distribution

Top-k logprobs of the first token, cold, on a fresh server each time, from [`len.sh`](len.sh) and [`prec.sh`](prec.sh):

| prompt | goinfer CUDA, int4 head (default) | goinfer CUDA, int8 head (`--embed-int4=false`) | goinfer CPU, default | llama.cpp |
|---|---|---|---|---|
| 0.5B:128 | `The` −0.39 | — | `The` −0.38 | `The` −0.39 |
| 0.5B:1024 | `import` −1.77, `#` −1.89 | `The` −0.84, `###` −2.16 | `The` −2.97, ` the` −3.07 | — |
| 0.5B:8000 | `th` −4.41, ` Thursday` −4.84, `the` −4.91 | `#` −1.64, `import` −2.20 | `import` −2.64, `T` −2.82 | ` the` −1.31, `the` −2.02 |

- **The head's precision decides it; the KV's does not:**
  - `-kv f32` gives byte-identical logprobs to the default f16 KV, at 1024 and at 8000.
  - `--embed-int4=false` moves the top-1 logprob from −4.41 to −1.64 at 8000.
  - With that int8 head, greedy runs 20 tokens; with the int8 head and f32 KV, 64 ([`flags.sh`](flags.sh)).
- **On this prompt, past about 1k tokens, the distribution is near-flat under the int4 head,** so rounding decides the
  greedy path:
  - CUDA's own result changes with how the prompt is prefilled ([`cw.sh`](cw.sh), [`lp2.sh`](lp2.sh)): cold in one go,
    or after a 128-token prompt whose prefix it reuses.
  - `-exact-prefill` changes the tokens but not the shape.
- **The 1.5B** agrees between CUDA and CPU through 1024 tokens and diverges at 2048 (`The` −1.31 against `` `` `` −1.58).
  Its replies at 2048 to 8000 ran the full 64 tokens in the sweep.

## When it started

The same cold probe on the CUDA builds still on disk ([`old.sh`](old.sh)):

| build | date | 0.5B:8000 greedy |
|---|---|---|
| `7a44a58e` | 2026-09-27 | 64 tokens, "\n\nthe the the…" |
| `754f12d3` (the 09-29 sweep) | 2026-09-29 | 64 tokens in all 24 completions (`peer-sweep-2026-09-29/nobara/a-depth8000.json`) |
| `fac209c2` | 2026-10-21 | 8 tokens, the same reply as today |
| `7545a43c`, `f5f06f4b2`, `e29da447` | 11-08, 11-09, 11-19 | 8 tokens, byte-identical replies |

The change is between `754f12d3` and `fac209c2`. `--embed-int4` became the default on 2026-09-28, so it was already on
in both good builds. What moved after it is not identified.

## Reading

- **This is not a speed defect, and it is not context-length truncation.** The 0.5B's int4 head, on an adversarial
  filler prompt past about 1k tokens, gives a near-flat first-token distribution: top logprob −4.4 at 8000, against
  −1.6 with the int8 head and −1.3 for llama.cpp's Q4_K_M. Greedy decoding then wanders into special tokens and stops.
- **Whether that is a fidelity problem of the int4 head on real long prompts is open.** This prompt is filler by design.
  No HF f32 reference was run here.
- **The VOID stands.** By the token gate, a reply that ends early is not graded.

## Next steps

These are not done here; they are the owner's call:
1. **Bisect `754f12d3..fac209c2`** with `old.sh`'s probe. About 1–2 minutes a build, and about 11 builds.
2. **Measure the int4 head's long-context fidelity on real prompts** against an HF f32 reference, for the 0.5B and the
   1.5B, before deciding anything about `--embed-int4`'s default. Changing a default is Francis's call.
3. **Give the next sweep's 0.5B:8000 cell a prompt that is not near-flat at the head:** re-calibrate the essay-v2 prompt
   with a natural-text filler. That is a protocol change, registered before its run.
