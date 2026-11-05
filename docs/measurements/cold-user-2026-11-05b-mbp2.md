# Cold-user run, goinfer v0.24.1, scenarios D, E, F (mbp2, 2026-11-05, second run)

## Contamination note

- I am a Claude session (Sonnet 5.5) started by the project's coordinator. I was started in ~/tmcode/wgpu only because it is a trusted folder. I did not read or use that repo; all work was in ~/cold-user-v0.24.1/run-b.
- I have not read the goinfer source tree, its docs/ or its git history. I did not open the morning record (cold-user-2026-11-05-mbp2.md), which sits in the parent directory.
- The brief said to work in an EMPTY directory. ~/cold-user-v0.24.1 was NOT empty: it held morning leftovers (binaries, logs, chat.html.txt, embed/). I used a fresh subdirectory, run-b, and re-downloaded everything. I did not read the leftovers.
- Prior knowledge: I know goinfer only as a name from this brief. The brief names Ollama, `-stream-weights` and the goinfer-status path. Everything else came from the allowed sources below.
- Sources used: the GitHub README (raw, main), the release API for v0.24.1, the binaries' `--version`, `--help` and `models` output, and runtime log text. I did not use pkg.go.dev.
- Not from an allowed source: Pillow generated the test image, and for E I tried to fetch Ollama's own release (see E).
- No Anthropic API key or token was used. Nothing was pushed or committed.

## Binaries and versions

Downloaded from release v0.24.1. sha256 matched checksums.txt for both.

| binary | --version |
|---|---|
| goinfer-serve-linux-amd64 | v0.24.1 (7f19e161c4f1), backends: cpu cuda, go1.27.3 linux/amd64 |
| goinfer-chat-linux-amd64 | v0.24.1 (7f19e161c4f1), backends: cpu cuda, go1.27.3 linux/amd64 |

## Baseline (11:36-11:37 PST)

- Host MBP2, Kubuntu, 8 cores, 15833 MB RAM, swap 511 MB (used 0), 56 GB disk free at the start.
- uptime load average was 0.10 / 0.09 / 0.04 at 11:36:32.
- Other processes: only Claude's own daemon and this session. No other goinfer or ollama process.
- Swap counters at 11:37:24: pswpout 28, pswpin 0, swap used 0.
- The box was quiet, though I was downloading while measuring.
- No CUDA device. Every run reported "backend: cpu (auto: cuda is built in, but no CUDA device answered)".

## Scenario D: a model that does not fit (RESULT: dead end, two attempts, both stopped by the swap guard)

Model: `goinfer-chat pull gemma-4-26b-a4b`, a 13.4 GiB q4_0 GGUF on a 15.8 GB box. The README and `models` list both say ~20 GiB peak RSS on the reference box.

Guard: my script `guard.sh` polls `/proc/vmstat` pswpout and `free` every 2 s and kills serve on ANY pswpout increase or on swap used above baseline + 512 MB. I read rule 2 literally.

| attempt | command | outcome |
|---|---|---|
| D1 | `goinfer-serve --model <gguf>` (no flags) | CPU backend. Printed "stream-weights: transcoding ... .giw (int4, one-time — minutes + ~model-size on disk)". Guard saw pswpout 33→46 (13 pages, about 52 KB), swap used 0 MB, and killed serve. |
| D2 | `goinfer-serve -stream-weights -weight-cache 6 --model <gguf>` (the README line) | Same transcode path. Guard saw pswpout 64→1418 (about 5.5 MB), swap used 0→8 MB, and killed serve. 8.7 GB of a partial .giw had been written; I deleted it. |

- Serve never reached the load stage, so the `swap guard: baseline` line was not printed for D. (It was printed for F.)
- No tokens were decoded, so there is no tok/s figure.
- Findings:
  - The README's "bigger than RAM" flags do not skip a one-time transcode of the whole model to a second ~model-size file.
  - The transcode alone moved pages to swap on a 15 GB box with about 12 GB available.
  - D1 and D2 both tripped before serving began, so I cannot say whether the flags avoid swap once running.
  - The guard may have tripped on small, unrelated pageouts: D1's 13 pages were tiny. That is a limit of the literal-guard method.
- Guard log: guard-d1.log, guard.log. Serve logs: d1.log, d2.log.

## Scenario E: control through a peer tool (RESULT: dead end, one attempt, blocked by the harness)

- Ollama was not installed (no `ollama` on PATH, no ~/.ollama).
- `curl -I https://ollama.com/download/ollama-linux-amd64.tar.zst` redirected to a GitHub release .tar.zst. `zstd` is present, so a no-sudo install looked possible.
- I tried to download, extract and run it from a fresh directory. Claude Code's auto-mode permission classifier denied the call as "Code from External".
- I did not retry or work around the denial. Nothing was installed and no peer numbers were recorded.
- The reason is a harness permission block, not a sudo or tool limit. The control is therefore missing, and the D and F numbers have no peer comparison.

## Scenario F: show it a screenshot (RESULT: reached)

- Model: `pull qwen3.5-0.8b`, 1.6 GiB safetensors, 39 s download, sha256 verified. The `models` list calls it "box class S: reading a screenshot".
- The README says `goinfer-chat --image` is GLM-OCR only (from `--help`). I used `goinfer-serve --model <dir> --log-requests` and POSTed an `image_url` part to /v1/chat/completions.
- Image: a synthetic 800x500 PNG made with Pillow, 9613 bytes. It shows a terminal window with a failing test line and a "Retry Cancel" box.
- Swap guard line from serve: `swap guard: baseline 0.01 GB (system swap)` and `swap guard: baseline 0.00 GB (server swap)`.
- The first-run transcode took 1 s (419 MB cache).

| turn | request | prompt tok | completion tok | ttft | total | KV reused |
|---|---|---|---|---|---|---|
| 1 | image + "What does this screenshot show? What failed?" | 423 | 200 | 8245 ms | 17.6 s | 0 |
| 2 | SAME image + history + "What buttons are visible?" | 660 | 152 | 7924 ms | 15.1 s | 0 |
| 3 | identical resend of turn 1, max_tokens 20 | 423 | 20 | 5040 ms | 5.9 s | 0 |

- Turn 1 answer: it correctly read the terminal log and named `FAIL tokenizer: TestMerge`. A bug was visible in the output: "exitsstatus".
- Turn 2 answer: it listed "Retry Cancel" as one button, plus non-button items (the title bar, the prompt, the log output).
- Serve logged `vision: reused the cached encode of this image (9613 bytes, sha256 119b2326)` on turns 2 and 3. The vision tower is therefore not recomputed, but the text prefill is, every time.
- `reused=0` and `prefill_reused_tokens=0` held on every request, including the identical resend. Turn 1 logged `canon=0`.
- Serve RSS was 1.05 GB at the end. Swap was untouched: pswpout stayed at 1418 and swap used stayed at 8 MB.
- Cause of the zero KV reuse: not investigated, and I do not know it.
- Turn 1's ttft includes the tower load ("loads on first image").

## Friction log (verbatim from friction.txt, plus additions)

Tags: Guessed / Wanted and absent / Error.

- 11:36:38 README is 442 lines and long. The "Bigger than your RAM or your GPU" section is last. `-stream-weights` is mentioned in `--help` as "the one flag to know". (Observation.)
- 11:37 `goinfer-serve --help` was denied once by the harness classifier. A retry with output redirected to a file worked. [Error, harness]
- 11:37 `goinfer-chat models` has a gpt-oss-20b entry (12 GB, "too big to hold resident on an 8 GB GPU") that does not apply to a CPU-only 15 GB box. I chose gemma-4-26b-a4b as the does-not-fit model. [Guessed]
- 11:37 E: Ollama tarball download, extract and run was DENIED by the classifier ("Code from External"). Not retried. E = dead end, one attempt. [Error, harness]
- 11:44:26 D1 stopped. The plain `serve --model` on a 13 GiB file silently entered the stream-weights transcode ("-fit retry"). My guard saw pswpout rise 33→46. [Wanted: a warning before a minutes-long, model-size disk write]
- 11:44:58 D2 stopped. The README flags did not avoid the transcode, and pswpout went 64→1418. [Wanted and absent: a "this will use X GB disk and Y GB page cache" estimate. The run-time guard is armed only after load.]
- 11:44 serve prints "swap guard: baseline" only at the end of startup. The transcode phase has no guard line. [Wanted and absent]
- 11:47 F: `goinfer-chat --image` supports GLM-OCR only. The README's vision line points at the server. [Guessed, then verified by `--help`]
- 11:47:48 F: KV reuse was 0 for the same image and for an identical resend. No log line says why. [Wanted and absent]
- Harness note: my first guard script's pkill -f pattern matched my own shell and killed the call (exit 144). Fixed it with an anchored pattern. Not goinfer friction.

## Not done / limits

- D never decoded. The does-not-fit behaviour of the running server is unknown.
- E was not run, so there is no peer table.
- The 25-minute time-box was not reached. Total elapsed time was about 12 minutes, apart from downloads.
- gpt-oss-20b and `-moe-cache-experts` (CUDA only, no GPU here) were not tried.
- Files in run-b: friction.txt, guard*.log, d1.log, d2.log, f.log, serve-help.txt, chat-help.txt, shot.png.
