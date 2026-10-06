# Cold-user run: goinfer v0.20.0 on nobara-pc, 2026-10-05

Run window: 19:20:27 to 19:44:14 PDT (02:20:27 to 02:44:14 UTC, 2026-10-06), about 24 minutes.
Machine: nobara-pc. Linux 7.2.0-202.nobara.fc44.x86_64, amd64, 16 threads, 64238 MiB RAM, swap of 8 GiB zram0 plus 69 GiB NVMe partition (swappiness 100), NVIDIA RTX 2070 SUPER 8 GiB, driver 595.91.07.
Run directory: `~/cold-user-2026-10-05`. It was created empty at the start, and everything below happened inside it, apart from the one A1 incident noted below.
Target: the published release tag v0.20.0, which `releases/latest` reports as the latest.

## Contamination declaration

- **Prior contact with goinfer.** I had not run goinfer before this run. But the agent harness that started me runs from a checkout of the goinfer repository on another machine (a Mac), and it put project material into my context before the run began. That material was:
  - The project's CLAUDE.md agent notes. They cover the five-module layout and build tags; the benchmarking methodology and its provenance rules; the model-storage roots on both machines (`~/models`, `/srv/models`, an SMB archive); a "night queue" and timing-lock workflow; and long sections on measurement discipline.
  - An index of roughly 80 auto-memory notes about goinfer internals. Topics include Metal/CUDA kernels, MoE expert paging, `.giw` sidecar formats, swap-guard and fork/copy-on-wire incidents, the parity manifest, release steps, tiny-fixture identity, and "nobara box access".
  - The repo's git status, with recent commit subjects about a v0.21.0 release and its Metal device gate.

  I did not open any project file, doc or source tree, and I did not use any of this material to get past an obstacle. Where it may have shaped a choice, the friction-log line says so.
- **Things I saw that point at project internals (all from allowed sources):**
  - The `--help` text of both binaries cites internal docs and task IDs: `docs/measurements/cold-user-2026-09-06-nobara-pc.md`, `task-never-swap-2026-09.md`, `audit-2026-09-10.md`, `audit-metal-2026-09-12.md M-13`, `cuda/resident.go's fitDefaultCtx`, and the codes M35/M26/R9/R10/N-35/MC3/K2/K5/J2.
  - `serve check` names "M-18".
  - The pkg.go.dev doc for `Model.Generate` cites "R10, docs/measurements/cold-user-2026-09-06-nobara-pc.md", an earlier cold-user run on this same machine.
  - The `--version` string says the release binaries were built from a dirty tree (`890ca565f982-dirty`).
- **Pre-existing state on the box.** `~/.cache/goinfer` already held goinfer downloads from earlier runs, which I did not create. My first scenario-A attempt resolved `demo:0.5b` to a cached file there and wrote a 344 MB sidecar beside it. To diagnose this I listed that one model directory once. Then I deleted the two files I had created, and re-ran A with an isolated cache (see A1/A2). I did not look at anything else in it. I did not touch `~/mycode`, `~/models`, `/srv/models`, `~/goinfer-*` or `~/cold-user-run-2026-09-18`.
- **Sources outside the allowed list, used out of necessity:**
  - go.dev/dl, for a Go 1.27.1 toolchain (none was on PATH, and the README requires Go 1.27+).
  - The huggingface.co API and `resolve/main` URLs, to fetch and sha-check a safetensors checkpoint that v0.20.0's `pull` cannot fetch (scenario F).
  - opencode's provider-config shape, from general knowledge.
  - The Python `openai` SDK (3.16.2), which was already installed on the box.
- **Two README versions.** "The GitHub README" a stranger sees is `main`'s, and it differs from the v0.20.0 tag's README in 68 diff lines. I read both. One piece of advice that exists only in `main`'s README, starting the server with `-ctx 16384` for an agent, I used in scenario B.
- **General knowledge used:**
  - `XDG_CACHE_HOME` relocates Go's `os.UserCacheDir` on Linux.
  - Piping stdin into an interactive REPL.
  - The OpenAI SDK streaming API.
  - opencode's `@ai-sdk/openai-compatible` provider config.
  - Generating test PNGs with PIL.
  - Linux swap counters in `/proc/vmstat` and `/proc/meminfo`.

## Versions of every binary used

| binary | `--version` / version output |
|---|---|
| goinfer-chat-linux-amd64 | `goinfer-chat-linux-amd64 v0.20.0 (890ca565f982-dirty)` / `backends: cpu cuda` / `go: go1.27.0 linux/amd64` |
| goinfer-serve-linux-amd64 | `goinfer-serve-linux-amd64 v0.20.0 (890ca565f982-dirty)` / `backends: cpu cuda` / `go: go1.27.0 linux/amd64` |
| goinfer-chat-0.5b-linux-amd64 | same three lines, plus `embedded: tier=0.5b quant=int8int8 (baked at build time; --quant has no effect on this binary)` |
| Go toolchain (installed into the run dir) | `go version go1.27.1 linux/amd64` |
| C's program `embed` (`go version -m`) | `dep github.com/townsendmerino/goinfer v0.20.0 h1:Ogj4Qih0g4J8Gk7j1LcNafca2JrBDvMQxD1aCTEzwJU=`, aikit v1.51.1 |
| opencode | `1.18.29` |
| python3 / openai SDK | `Python 3.14.7` / `openai 3.16.2` |
| curl | `curl 8.18.0 (x86_64-redhat-linux-gnu)` |
| NVIDIA driver | `595.91.07` |

Every release asset I downloaded passed `sha256sum -c` against the release's `checksums.txt`:
- goinfer-chat-linux-amd64 `fd4a2567…`
- goinfer-serve-linux-amd64 `7c6ee4c8…`
- goinfer-chat-0.5b-linux-amd64 `a2cbf5e2…`

## Friction log

Tags: **[Guessed]** I had to guess. **[Wanted and absent]** I looked for something and it was not there. **[Error]** The tool or a step failed or misled. Times are PDT.

- 19:20:27 Start. Created an empty `~/cold-user-2026-10-05`. Baseline: swap 1310 MiB used, 59.6 GB available, GPU 547 MiB used, 72 GB disk free.
- 19:20:30 Read the README (tag and `main`) and the release JSON for v0.20.0: 27 assets, including `checksums.txt`.
- 19:20:53 Downloaded `goinfer-chat-linux-amd64` and `checksums.txt`; the sha256 matched.
- 19:20:55 **[Error, minor]** `--version` reports `v0.20.0 (890ca565f982-dirty)`: the published binary was built from an uncommitted tree.
- 19:20:55 **[Guessed]** `--help` has no one-shot prompt flag (chat is a REPL). I guessed that piping the question on stdin would work, and it did. **[Wanted and absent]** a `-p "prompt"` flag.
- 19:20:55 **[Error, cosmetic]** The help prints the flag as `-backend -tags cuda` and `-version backends:`. Go's flag package turns backticked words in the usage text into the argument name.
- 19:20:55 **[Wanted and absent]** Plain-language help. Flag descriptions cite internal docs, task IDs and audit codes (see the contamination section) that a stranger cannot resolve.
- 19:20:55 **[Wanted and absent]** GPU by default. The binary has CUDA compiled in, but `-backend` defaults to `cpu`, so A ran on the CPU while an RTX 2070 SUPER sat idle. (`main`'s README describes a later `--backend auto`; v0.20.0 does not have it.)
- 19:21:02 **[Error / contamination]** A1: `--model demo:0.5b` resolved to a file already in `~/.cache/goinfer` from earlier runs on this box, so nothing downloaded. It answered in 9.3 s, but that time is invalid as a cold start. It also wrote a 344 MB `…int4.e4h.cpu-amd64.giw` sidecar into that shared cache. I deleted the two files I had created.
- 19:21:33 **[Guessed]** A2: I set `XDG_CACHE_HOME` to a fresh directory inside the run dir. Neither the README nor `--help` says how to relocate the download cache ("your user cache dir" is all it says).
- 19:21:53 A2 answered: "The capital of France is Paris." Total from nothing was 20.1 s (numbers below).
- 19:21:53 **[Wanted and absent]** An explanation of the startup line `stream-weights: transcoding … (int4, one-time — minutes + ~model-size on disk)`. I never asked for stream-weights. A 469 MB model took 813 MB on disk (gguf plus a 344 MB `.giw`). The release notes explain the sidecar cache; the v0.20.0 README does not.
- 19:22:22 **[Error]** `goinfer-serve -model hf:Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:q4_k_m` exited with `quant "q4_k_m" is ambiguous … name the file exactly`, because the repo has both a single-file and a two-part q4_k_m. **[Guessed]** The message lists the files but not the syntax for choosing one. I guessed `hf:owner/repo:<filename.gguf>`, which worked first try.
- 19:22:33 to about 19:25:55 Server start with the 7B on CUDA at `-ctx 16384` (I took the 16384 from `main`'s README). This covered the 4.4 GiB download, a 1m43s transcode to `.cuda.giw` and an 11.9 s load. The banner reported 1 resident KV slot instead of the 4 requested ("free VRAM … allows 1"), "one generation at a time", and swap-guard baseline 3.21 GB.
- 19:26:00 **Observation** Swap-used rose from 1310 MiB (19:20) to 3050 MiB (19:26), on zram, while the 7B downloaded and transcoded, with about 59 GB still available. goinfer printed nothing about it. Its swap guard arms only after the server starts, which matches the release notes.
- 19:26:10 OpenAI Python SDK against `/v1`: `/v1/models` listed the model, streaming chat worked, and a one-tool request returned a correct structured `tool_calls` entry `get_weather({"city": "Paris"})`.
- 19:26:31 **[Guessed]** I wrote an `opencode.json` with a `goinfer` provider (`@ai-sdk/openai-compatible`, baseURL `http://127.0.0.1:8080/v1`) from general knowledge. The README links a recipe under `docs/integrations/`, which is outside the allowed sources. **[Wanted and absent]** a recipe inside the README.
- 19:26:52 **[Error]** opencode attempt 1, "fix the bug in calc.go": opencode made one real `read` tool call, then wrote the `edit` call as a ```json fenced block in prose. The file was not edited, and opencode exited 0. (My own harness slip: `opencode run` swallowed the rest of my stdin script; I re-ran with `</dev/null`. This was not goinfer's fault.)
- 19:27:17 **[Error]** opencode attempt 2, with an explicit "use the edit tool, do not describe the change": the same fenced-JSON prose, and no edit. **DEAD END #1** (agent CLI on this checkpoint). **[Wanted and absent]** `goinfer-chat models` names no checkpoint measured OK at agent-harness scale. Its only "tools:" measurement is for the 0.5B ("harness-scale (12 tools): skip — too small"), and no 7B appears in the list.
- 19:27:30 `goinfer-serve check` against the running server: 2 of 9 rows failed. "tools, OpenAI … turn two asked for the tool again instead of answering — the agent-livelock shape (M-18)" and "stop sequences … the reply never reached the stop sequence". This is consistent with the opencode failure, and nice to have. The internal code "M-18" means nothing to a stranger.
- 19:23:01 (in parallel with B's download) C: `go mod init` and `go get …/decoder@v0.20.0 …/tokenizer@v0.20.0` (7.6 s). I fetched the pkg.go.dev pages for `decoder`, `chat` and `tokenizer`. **[Wanted and absent]** a runnable example on pkg.go.dev. Both the README and the `Generate` doc point to `examples/embed/main.go`, which is in the repo tree, outside the allowed sources. **[Guessed]** I composed the sequence from the signatures: `decoder.Load`, `tokenizer.LoadGGUF`, `chat.Detect(chat.Meta{…})`, `Template.Render`, `Encode(…, false)`, `Stops()` to `StopIDs`, then `Generate`.
- 19:23:50 C built on the first compile. **[Error, minor]** The pkg.go.dev text for `Model` still says "Model is a loaded Gemma 3 checkpoint". The first version was 42 lines, which I trimmed to 40 (gofmt-clean, `go vet` clean).
- 19:23:52 C printed a completion from the 0.5B GGUF (a weak but on-topic haiku), ending with `<nil>` for `gen.Err()`.
- 19:28:00 **[Error]** F: `goinfer-serve pull Qwen/Qwen3.5-0.8B` printed "0 GGUF file(s)", then the hint "fetch one with: … pull Qwen/Qwen3.5-0.8B:<quant>", which cannot work. **[Error]** `pull Qwen/Qwen3.5-0.8B:safetensors` failed with "goinfer-chat pull: repo … has no .gguf files". That is the wrong binary name in a `goinfer-serve` error. **[Wanted and absent]** a way in v0.20.0 to fetch the original-weights checkpoint that the release notes say Qwen3.5 vision needs. (`main`'s README describes `:safetensors`, so it lands after v0.20.0.) **[Guessed]** I fetched the 10 repo files with curl from `huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/…` and checked the 1.75 GB shard against the HF API's LFS sha256 (OK).
- 19:28:38 F server on CPU (the default) with `-model <dir>`: ready in 8 s. The banner said "tower loads on first image" and "prefill path: sequential — this arch has its own per-token forward".
- 19:28:56 to 19:31:18 F on CPU, three requests: 41.95 / 37.88 / 37.78 s to first token. **[Wanted and absent]** Re-encode savings for the byte-identical resend: TTFT was the same as for a new image, and `prefill_reused_tokens=0`. **[Wanted and absent]** A server log line saying how long the image encode took and where it ran.
- 19:31:40 to 19:33:55 F retried with `-backend cuda`: 37.75 / 37.50 / 37.53 s, so the GPU gave no change. As a control, a text-only request on the same CUDA server had TTFT 0.12 s (51 chunks in 0.43 s). The image requests also decoded slowly (about 120 tokens in 8 s).
- F answer quality on the 0.8B:
  - F1 named the right error, but claimed the app "successfully connects" to Postgres.
  - F2 gave the right file and line (`/src/cmd/server/main.go:42`).
  - F3 got the got/want values right (104.99 / 109.99), but named the package instead of `TestCheckoutTotal`.
- 19:34:25 to 19:38:26 D: `goinfer-chat pull gemma-4-26b-a4b` (the short name from `goinfer-chat models` and the README's "bigger than your GPU" section) downloaded 13.4 GiB in 3m55s, sha256 verified. Disk went from 61 GB to 48 GB free.
- 19:38:40 D: `goinfer-chat fit <gguf>` reported "cuda EXPERT-CACHED … fits 6.97 GB free" and "cpu RESIDENT … fits 58.45 GB free". **[Error, minor]** Above that table it printed "fit is tight … Re-run goinfer-serve with -stream-weights", while the table says the model fits on both backends.
- 19:39:10 D baseline (`free -m`, `vmstat 1 5`): swap used 1605 MiB; si/so 0 on every live sample; pswpin 1129360, pswpout 4771466. A 30 s idle check showed zero counter movement and SwapFree drifting slightly up, not down.
- 19:39:56 D attempt 1: `goinfer-serve -backend cuda -moe-cache-experts -model <gguf>` (the README's command), under a watcher sampling every 0.25 s that kills the server on any pswpin/pswpout increase or any rise in swap-used. The server transcoded to a 13.6 GB `.cuda.giw` ("cache ready (13589 MB) in 1m6s", 19:41:02).
- 19:41:22 **D attempt 1 STOPPED** at +86.7 s: swap-used +18956 kB, with pswpin/pswpout +0 (vmstat si/so stayed 0 throughout). Server RSS was about 11.9 GB. The server log had only its 2 transcode lines: no warning, no refusal and no swap-guard banner before the growth.
- 19:41:59 D attempt 2 (second attempt at the same obstacle; the sidecar now existed, so no transcode), same command and watcher.
- 19:42:20 **D attempt 2 STOPPED** at +20.3 s: swap-used +14516 kB, pswpin/pswpout +0, server RSS 16.7 GB. The server log was still empty (0 lines). **DEAD END #2** under the swap-safety rule. Model output was never reached, and the server never printed its swap-guard banner in either attempt.
- 19:42:51 Extra A data point: the model-included `goinfer-chat-0.5b-linux-amd64` (654.8 MB) went from nothing to an answer in 23.5 s.
- 19:44:14 End. Every process I started was stopped by its saved PID (servers, watchers, vmstat, tail). `pgrep -x` finds no `goinfer-serve-l`/`goinfer-chat-li`/`vmstat`. GPU at 547 MiB, 0 % (the same as the start). Swap 1650 MiB used. 34 GB disk free. The run dir holds 39 GB, mostly the 26B gguf and its sidecar.

Watcher bug of my own: its periodic lines print the label `MemAvailable` with no value. This does not affect the stop logic.

## Numbers per scenario

### A: try it, from nothing to an answer

| run | path | download | model fetch | first-run convert | load | answer after prompt | **total** |
|---|---|---|---|---|---|---|---|
| A1 (void, pre-cached model on the box) | goinfer-chat + `--model demo:0.5b` | 1.3 s | 0 (cache hit) | 8.6 s | 0.11 s | 0.45 s | 9.3 s (invalid) |
| **A2 (the cold start)** | goinfer-chat (16.0 MB) + `--model demo:0.5b`, isolated cache | 0.61 s incl. sha256 | 10.3 s (468.6 MiB) | 8.5 s (344 MB `.giw`) | 0.12 s | 0.44 s | **20.1 s** |
| A3 | goinfer-chat-0.5b (654.8 MB, model baked in) | 22.9 s incl. sha256 | none | none | 0.18 s | 0.44 s | **23.5 s** |

Decode was 7 tokens at 15.7 tok/s (A2) and 16.0 tok/s (A3), on the CPU backend. A2 used 813 MB of disk for the cache. The totals exclude my reading time. Wall clock from the first download (19:20:53) to the first clean answer (19:21:53) was 60 s, including the A1 detour.

### B: an OpenAI client and an agent CLI

| measure | value |
|---|---|
| model | Qwen2.5-Coder-7B-Instruct q4_k_m (4.4 GiB), `-backend cuda -ctx 16384` |
| server start to "serving on" | about 3m22s (download plus a 1m43s transcode plus an 11.9 s load) |
| VRAM after load | 6696 MiB of 8192; 1 KV slot (4 requested) |
| OpenAI SDK, streamed chat | TTFT 0.181 s, 107 chunks, 1.58 s total |
| OpenAI SDK, 1-tool request | correct structured tool call, 141 prompt / 24 completion tokens |
| `serve check` | TTFT 0.05 s · 84.2 tok/s; long prompt (about 2000 words) TTFT 1.61 s · 59.0 tok/s; **2 of 9 FAIL** (tools OpenAI turn two, stop sequences) |
| opencode 1.18.29 | 2 runs (21 s, 15 s), 1 real tool call (read), **0 edits**: dead end |

### C: embed it in Go

| measure | value |
|---|---|
| program length | 40 lines (gofmt and vet clean), built on the first compile |
| `go get` decoder + tokenizer @v0.20.0 | 7.6 s (fresh module cache) |
| `go build` (cold build cache) | 5.3 s |
| run (0.5B GGUF, CPU, 64 tokens max) | 2.0 to 2.4 s wall, max RSS 976 MB |
| first `go mod init` to first printed completion | 51 s, including reading pkg.go.dev |

### D: a model bigger than the hardware

| measure | value |
|---|---|
| model | gemma-4-26b-a4b, `gemma-4-26B_q4_0-it.gguf` 13.4 GiB (bigger than the 8 GiB GPU) |
| pull | 3m55s, sha256 verified |
| `fit` verdict | cuda EXPERT-CACHED, fits 6.97 GB free; cpu RESIDENT, fits |
| baseline before D (`free -m`) | swap 1605 MiB used of 78863; mem available 59962 MiB; vmstat si/so 0 |
| attempt 1 | stopped at +86.7 s, swap-used +18956 kB (pageouts 0), RSS about 11.9 GB, during load after the transcode |
| attempt 2 | stopped at +20.3 s, swap-used +14516 kB (pageouts 0), RSS 16.7 GB, during load |
| tool warned or refused before swap growth? | **No.** Nothing was printed in attempt 2, and only transcode progress in attempt 1. |
| swap baseline the banner printed | **None in D**: the server never reached its `swap guard: baseline` line in either attempt. Elsewhere in this run the banner printed 3.21 GB (B), 1.54 GB (F, CPU) and 1.52 GB (F, CUDA). |
| tokens generated | none |

### F: screenshots

Model: Qwen3.5-0.8B (original safetensors, fetched with curl). Images: two synthetic 1024x640 terminal screenshots. The request: one user turn with an image plus a question, temperature 0, `max_tokens` 120, streamed.

| request | prompt tokens | TTFT, CPU backend | TTFT, `-backend cuda` | total, CPU / CUDA | `prefill_reused_tokens` |
|---|---|---|---|---|---|
| F1 new image (shot1.png) | 662 | **41.95 s** | 37.75 s | 50.26 / 46.06 s | 0 |
| F2 byte-identical shot1.png, new question | 667 | **37.88 s** | 37.50 s | 46.07 / 45.78 s | 0 |
| F3 different image (shot2.png) | 667 | **37.78 s** | 37.53 s | 45.04 / 42.12 s | 0 |
| control: text only, CUDA server | — | — | 0.12 s | — / 0.43 s | — |

The first CPU request includes loading the vision tower. The resend of identical bytes saved nothing measurable over a new image.

## Dead ends

1. **B, agent CLI.** opencode with Qwen2.5-Coder-7B on goinfer-serve: in both attempts the model wrote its edit tool call as fenced JSON prose, so no edit was made. `serve check` independently fails the OpenAI tools row.
2. **D, bigger than the GPU.** gemma-4-26b-a4b with `-backend cuda -moe-cache-experts`: in both attempts the swap-safety stop fired during load (swap-used growth with zero pageouts) before the server had printed any warning, refusal or banner.

Count: **2 dead ends.** The F fetch obstacle (no safetensors pull in v0.20.0) was worked around with curl, so it is recorded as friction, not as a dead end.
