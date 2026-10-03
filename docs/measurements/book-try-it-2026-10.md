# The book's "Try it" figures — 2026-10

Every figure a `docs/book/` chapter's "Try it" section prints is measured here first, then entered in
`site/data/claims.json` as a book claim, which the site's claims check holds to this record and to the chapter
(`site/internal/site/claims.go`, `CheckClaims`). A chapter's number moves only with this record.

**Deterministic figures** (this file's day run): token counts, sizes, fit-plan lines, next-token probabilities and draft
acceptance under greedy decoding, the compiled-in backends, the server's self-check. They repeat exactly on one machine:
the day script ran twice on 2026-10-02 and every figure below was the same both times. On another architecture, the
probabilities and the acceptance can move in their last digits (arm64 fuses multiply-adds that amd64 does not;
`docs/parity-coverage-policy.md`), and the sizes cannot.

**Timed figures** (tok/s, seconds) are night-queue work and get their own section from that run.

## Setup (day run, 2026-10-02)

- Machine: Apple M1 Pro, 16 GB, macOS 26.6.2 (the owner's MacBook).
- Binaries: the released `v0.20.0` darwin-arm64 assets, downloaded with `gh release download v0.20.0`:
  `goinfer-chat-darwin-arm64` (sha256 prefix in `book-try-it-2026-10/day-2026-10-02/provenance.txt`) and
  `goinfer-serve-darwin-arm64` (`31cb4c0be2424cbf`). Both report `v0.20.0 (890ca565f982-dirty)`: the release assets'
  known build stamp (`docs/tasks/task-first-hour.md` R18), not a local edit.
- Checkpoints, from `~/models`: `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (`1d9614638d18024d`),
  `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (`cc324af070c2ecbf`), and the Gemma 4 26B-A4B as
  `gemma4-26b-int4-v14st.metal.giw` (`b514bf5fe9cd2633`).
- The two Qwen files are the ones a reader's `hf:Qwen/Qwen2.5-Coder-{0.5B,1.5B}-Instruct-GGUF:q4_k_m` fetches: their sha256
  equals the published LFS object's (checked 2026-10-02 against the Hugging Face API).
- Script: `book-try-it-2026-10/run-day.sh`; every command's raw output is in `book-try-it-2026-10/day-2026-10-02/`.

## Chapter 1: tokens (2026-10-02)

`POST /v1/messages/count_tokens` on `goinfer-serve --model qwen2.5-coder-0.5b… --backend cpu`, one user message, "The quick
brown fox jumps over the lazy dog.": `{"input_tokens":18}` on 2026-10-02. That is the sentence's tokens plus the chat
template's wrapper around one user turn; the same request through `goinfer-serve check`'s own row reports
"23 tokens, matches usage" for its longer test message.

## Chapter 3: next-token probabilities (2026-10-02)

`/v1/chat/completions` on the same server, "What is the capital of France? Answer in one word.", `temperature` 0,
`max_tokens` 1, `logprobs` with `top_logprobs` 3, on 2026-10-02. The model answers `Paris`; its top three next-token
probabilities (the model's own distribution, before any temperature) are `Paris` 82.84%, `London` 2.30% and `Br`
2.26%.

## Chapters 4 and 5: sizes (2026-10-02)

`goinfer-chat fit qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, on 2026-10-02:

- at the default int4, the CPU row reads `dense 0.93 GB + KV@8192 0.44 GB`, and the Metal row `dense 0.93 GB +
  KV@8192 0.22 GB` (Metal keeps its KV cache in f16, the CPU in f32);
- with `-quant int8int8`, the CPU row reads `dense 1.66 GB + KV@8192 0.44 GB`.

0.44 GB is 8192 positions × 28 layers × 2 (K and V) × 256 × 4 bytes = 0.4375 GiB; it is the figure chapter 4 quotes.
Chapter 5 quotes the int8int8 line's 1.66 GB.

**Not quoted: the int4 dense figure, because `fit` reports it from whichever sidecar is cached.** `fit` reads a fresh
sidecar if one exists, trying the backend's default head precision first and then the other
(`internal/fitcmd/fit.go`, `sidecarIfFresh`). Before this session's chapter-9 run wrote
`…int4.e4h.cpu-arm64.giw` at 10:02 (the int4-embedding sidecar every default CPU load writes since 2026-09-28), the
same command read the older plain-head sidecar and printed `dense 1.12 GB`; after it, `dense 0.93 GB`. So a reader's int4
figure depends on what they ran before. Reported to the owner; not changed here. The rest of each line ("fits N GB
free", or a decline) is this machine's free memory at that moment and is not a figure the book quotes either.

## Chapters 6 and 7: the 26B MoE's plan (2026-10-02)

`goinfer-chat fit gemma4-26b-int4-v14st.metal.giw` on 2026-10-02: both rows say the expert-cached plan `needs at least
top-k=8 slots/layer (0.75 GB)` — eight expert slots per layer, because the router picks 8 experts per token, at 0.75 GB
for all layers together. On this 16 GB machine, with the owner's applications open, the CPU row was `WEIGHT-PAGED` and
the Metal row `DECLINE`; that verdict depends on free memory and is not quoted.

**Not quoted yet.** This was measured on a `.giw` bundle of the 26B, the only copy on this Mac's disk; a reader runs
`fit` on the published GGUF (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, 14.44 GB), which does not fit this disk's
14 GB free. Chapters 6 and 7 get their "Try it" from a run on that file.

## Chapter 9: draft acceptance (2026-10-02)

`goinfer-chat --model qwen2.5-coder-1.5b… --draft qwen2.5-coder-0.5b… --temp 0 --max 64 --backend cpu`, with "Write a Go
function that reverses a string." on stdin, on 2026-10-02: the closing line reads `[spec: 95% accepted, 4.6 tok/pass]`
(K = 4 proposed per pass). The `tok/s` on the same line is a timing and is not quoted from this run.

## Chapter 10: backends (2026-10-02)

`goinfer-serve --version` on 2026-10-02: `backends: cpu metal` for the darwin-arm64 asset.

## Chapter 11: the server's self-check (2026-10-02)

`goinfer-serve check -long-prompt 0` against the 0.5B server above, on 2026-10-02: `2 of 8 checks FAILED`.

- `tools, OpenAI` failed: "turn two asked for the tool again instead of answering — the agent-livelock shape (M-18)".
- `stop sequences` failed: "the reply never reached the stop sequence, so this proves nothing about it".
- `vision` was skipped, with no vision tower loaded.
- The other five passed: models list, streamed chat, harness-scale tools, structured output and `count_tokens`.

The 0.5B is the smallest model the catalogue lists, and the check names what failed and why. Not reconciled here:
`goinfer-chat models` lists the 0.5B's tools as "minimal schema: ok; harness-scale (12 tools): skip — too small
(measured 2026-09-07, nobara-pc)", while this run passed harness-scale and failed the OpenAI tools row.

## Timed figures: chapters 2, 8 and 9 (night run, 2026-10-02)

`book-try-it-2026-10/run-night.sh` on the night queue, 18:37–18:39 PDT: the same released `v0.20.0` binaries and
checkpoints as the day run, on the same machine. No thermal warning was recorded. Every command's raw output is in
`book-try-it-2026-10/night-2026-10-02/`. A chapter quotes the median of three.

**The run had no idle gate of its own.** The runner's settle wait gave up at a load average of 2.0, with the owner's
editor open. That bears on the CPU rows most. Each row's three runs agree within 1% after the first, and the record
gives all three.

| Figure | Run 1 | Run 2 | Run 3 | **Median** |
|---|---:|---:|---:|---:|
| ch2/ch8, 1.5B on the CPU: `check`'s "chat, streamed" decode | 70.9 | 62.6 | 62.4 | **62.6 tok/s** |
| ch2/ch8, 1.5B on the CPU: 2,000-word prompt, time to first token | 10.96 | 11.11 | 10.98 | **10.98 s** |
| ch2/ch8, 1.5B on Metal: "chat, streamed" decode | 93.4 | 94.4 | 94.8 | **94.4 tok/s** |
| ch2/ch8, 1.5B on Metal: 2,000-word prompt, time to first token | 2.08 | 2.07 | 2.07 | **2.07 s** |
| ch9, 1.5B plain, CPU, greedy, 64 tokens | 51.7 | 51.1 | 51.3 | **51.3 tok/s** |
| ch9, 1.5B drafted by the 0.5B, same | 46.8 | 46.7 | 46.6 | **46.7 tok/s** |

- **ch9: drafting is slower here, 0.91× plain,** at 95% acceptance and 4.6 tokens per pass (the day run's figures
  again). On the CPU the 0.5B's draft passes cost more than the verify saves on a 64-token reply. The chapter must say
  that, not that drafting speeds this up.
- The CPU decode's first run (70.9) is above the other two. It is kept, and the median sits with the two that agree.
- **`check` on the 1.5B reads `1 of 9 checks FAILED`, every run, on both backends.** The failing row is `tools, OpenAI`:
  "turn two asked for the tool again instead of answering — the agent-livelock shape (M-18)", the same row the 0.5B
  failed in chapter 11. The 1.5B passes `stop sequences`, which the 0.5B did not. A chapter that shows `check` must
  show this line.

