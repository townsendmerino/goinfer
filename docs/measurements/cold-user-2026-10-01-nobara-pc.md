# goinfer cold-user run, 2026-10-01, nobara-pc

Run by Claude (Sonnet 5.5) inside Claude Code, working from the prompt Francis pasted. All times are local (-07:00) on 2026-10-01.

## Contamination declaration (read this first)

I am **not** a clean stranger. Three things:

1. **Prior contact with the project, through the harness.** The session I ran in lists a pile of `~/mycode/goinfer/...` directories as additional working directories (chat, decoder, cuda, metal, multimodal, cmd/serve, docs/measurements, scripts, and more). A `goinfer-doc-review` skill is also loaded. So I knew from the first minute that a local goinfer checkout exists on this machine and that its docs/measurements folder holds files. **I did not open, list or read any of it.** The same goes for the session's own working directory (`~/mycode/gocudrv-fork`, a different project) and the memory folder the harness gave me. I cannot prove I was uninfluenced by merely knowing those paths exist.
2. **Things I saw in tool output that point at project internals.** `--help` and `models` text cites internal doc paths (for example `docs/measurements/cold-user-2026-09-06-nobara-pc.md`), and the `models` list says a tools row was "measured 2026-09-07, nobara-pc". So an earlier cold-user run on this same machine exists. I did not open it or any other doc. I mention it so you can judge whether I was nudged.
3. **General knowledge, not goinfer knowledge.** I know opencode's config format and Ollama's CLI from general background, and I used that without looking either up. I had no goinfer-specific knowledge beyond the pasted prompt and the README.

Beyond the prompt I was given nothing. I did not use the local clone, the existing models in `~/models` (about a dozen older files, which I left alone), or any previously installed goinfer binary.

## Where I deviated from "write nothing elsewhere"

- **XDG_CACHE_HOME** was redirected to `~/gi-cold-2026-10-01/cache` for every goinfer pull, because the README says downloads go to the user cache dir, which is outside the allowed paths. `--help` offered no flag for this on `goinfer-chat` (`pull` has `-o`; I used the env var instead).
- **Ollama** was installed from its manual tarball into `~/gi-cold-2026-10-01/ollama-inst`, not with the official install script. The script needs a sudo password and writes to `/usr/local`, `/etc/systemd` and creates a system user. I downloaded the script and read it, but did not run it. The server ran on port 11435, not the default 11434.
- **opencode** ran with XDG_CONFIG/DATA/CACHE/STATE redirected into `oc-xdg/`, so that it would not write to `~/.config` or `~/.local`.
- **Go** ran with GOPATH and GOCACHE redirected into the working directory. The auto-downloaded Go 1.27 toolchain therefore landed there too.
- **Scenario F** model files went to `~/models/qwen2.5-vl-3b` (7.1 GB, plain curl from HuggingFace, because goinfer's `pull` only handles GGUFs).

Disk: 80 GB free at the start, 54 GB at the end. The floor was never close to 20 GB. The working directory holds about 22 GB and `~/models/qwen2.5-vl-3b` holds 7.1 GB. I left everything in place.

## Versions

| binary | --version |
| --- | --- |
| goinfer-serve | `goinfer-serve v0.19.0 (c7f8eff76c7c-dirty)`, backends: cpu cuda, go: go1.27.0 linux/amd64 |
| goinfer-chat | `goinfer-chat v0.19.0 (c7f8eff76c7c-dirty)`, backends: cpu cuda, go: go1.27.0 linux/amd64 |
| Ollama | 0.35.0 (client and server) |
| opencode | 1.18.29 |
| system Go | go1.26.5; the module forced a switch to a Go 1.27 toolchain (see C) |
| Python / openai package | 3.14.7 / `openai.__version__` printed `3.16.2` |
| curl | 8.18.0 |
| Pillow (to make test images) | 12.3.0 |
| NVIDIA driver | 595.91.07 |

Latest release found on GitHub (releases/latest API): **v0.19.0**, published 2026-09-19T02:08:35Z. I downloaded `goinfer-serve-linux-amd64` and `goinfer-chat-linux-amd64`, and both sha256 values matched `checksums.txt`.

Machine at the start: swap used 6,618 MB (zram 4.6 GB of 8 GB, plus 1.9 GB on the NVMe swap partition), 57 GB RAM available, GPU 639 MiB used of 8192.

## Scenario A: nothing to an answer (goinfer)

Question asked, same in A, B, C and E: "Write a Go function that reverses a string." Model: `goinfer-chat pull demo:1.5b`, which is Qwen2.5-Coder-1.5B-Instruct q4_k_m.

| leg | result |
| --- | --- |
| Download binaries (serve + chat, 26 MB) | 2.05 s |
| Download model (1.0 GiB, sha256 verified) | 16.8 s at about 65 MiB/s |
| First run, default backend (cpu) | 19.3 s wall: load 5.5 s, 198 tokens at 14.4 tok/s, maxrss 2.4 GB |
| **Total, default path** | **38.1 s** (my wall clock including command overhead was 56 s) |
| Same run with `--backend cuda` (not part of the first-run total) | 9.1 s wall: load 7.9 s, 178 tokens at 173.8 tok/s |

The answer was correct, runnable Go (a rune-swap reverse with a main). I did not do a model-included binary run (`goinfer-chat-1.5b`). I used the "pull" route.

## Scenario B: OpenAI-compatible clients

Server: `goinfer-serve --backend cuda --model <gguf>`, later `--ctx 16384`.

| client | result |
| --- | --- |
| curl, non-streaming | OK, 1.09 s for 200 tokens. usage: prompt 18, completion 200 |
| curl, streaming | OK. The first reply to "Say hi." began " Query issued," at default sampling, which looks odd. At temperature 0 it said "Hello! How can I assist you today?" |
| Python openai package | OK. Non-stream 0.08 s. Stream TTFT 0.015 s (a warm prefix: `prefill_reused_tokens` showed up in usage) |
| opencode, run 1 (default ctx 8192) | **No output at all for 240 s, then my timeout killed it.** The server logged no request. I guessed this was opencode waiting on stdin; I did not confirm it. |
| opencode, run 2 (stdin closed, default ctx 8192) | The very first request was rejected: `prompt is 11137 tokens but the model's context window is 8192 (context_length_exceeded)`. opencode then retried through its own compaction path: 34 context-overflow errors in 120 s, and I killed it at my 120 s timeout. |
| opencode, run 3 (server `--ctx 16384`, config limit 16384) | Completed in 37 s wall including opencode startup. **First-request prompt token count: 11,137** (the server's own count in the 400 error and opencode's usage `input` field agree). 126 output tokens. The reply was a JSON tool-call-shaped blob, not code. A 1.5B model does not cope with an agent harness. |

The server wrote no per-request log line, so the token count came only from the client and from error bodies.

## Scenario C: embed in 40 lines or fewer

I got the API from pkg.go.dev only. Program: 38 lines (`embed/main.go`). It uses `tokenizer.LoadGGUF`, `decoder.Load(path, Options{Quant:"int4"})`, `chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})`, `tmpl.Render`, `tok.Encode`, the template's stop strings mapped to token ids, `m.Generate`, and `tok.DecodePiece`. It takes the GGUF path as its argument.

| step | result |
| --- | --- |
| `go get` of decoder, tokenizer and chat | 22.9 s. Go printed `goinfer@v0.19.0 requires go >= 1.27.0; switching to go1.27.1` and downloaded the toolchain on its own (system Go is 1.26.5). This worked without my doing anything. |
| `go build` | 4.8 s, 7.2 MB binary |
| run | 19.1 s wall, a correct reverse-string answer with an explanation, CPU backend (maxrss 2.4 GB) |

It worked on the first compile. The path into `chat.Meta` took guessing from the method index (see friction).

## Scenario D: bigger than my hardware

Model: `gemma-4-26b-a4b` (`goinfer-chat pull gemma-4-26b-a4b`, 13.4 GiB, q4_0). **Constraint: the 8 GB of GPU memory** (it fits the 62 GB of RAM). Download 3 min 17 s at about 70 MiB/s.

Swap baseline just before: 6,618 MB used, `pswpout` 181,239,879, `pswpin` 62,640,471, 57.4 GB RAM available. I logged swap, `pswpout`, available RAM and GPU memory every 2 s into `d-watch.log`.

| step | what the tool said | swap |
| --- | --- | --- |
| `goinfer-chat fit <gguf>` (35.9 s) | A warning, "fit is tight ... needs ~14.7 GB resident ... + 10.4 GB KV + 13.4 GB reading the checkpoint = 38.5 GB ... (96% of budget). Re-run goinfer-serve with -stream-weights". Then a table: cpu RESIDENT, fits 57.38 GB free; cuda EXPERT-CACHED, 16/128 experts, 6.88 GB free. | no change |
| `goinfer-serve --backend cuda --model <gguf>` (no `-moe-cache-experts`) | The same "fit is tight" warning first, then `[cuda] resident path DECLINED ... cuda: unsupported projection kind ""`, then loaded on CPU in 46 s, ctx 262144. **No refusal.** | grew, see below |

The swap growth started at 13:48:53 (6,663 to 6,773 MB) and `pswpout` first moved at 13:48:55. It peaked at 8,523 MB (+1.9 GB over baseline) around 13:49:20. I killed the server at about 13:49:30 and swap settled at 7,275 MB (+650 MB over baseline). `pswpout` rose by about 2,100 pages in total, which is oddly small next to the MB growth. I do not know why (zram accounting, maybe). Other processes on the machine may have contributed.

- **Did the tool tell me before the machine did?** There was a budget warning before any swap growth, in the `fit` command and again at the top of the serve log, which printed before the first growth. There was **no refusal**, and the load went ahead and grew swap anyway.
- **"swap guard: baseline ... swap-used" line:** never printed (`grep -ci "swap guard"` on the serve log gave 0).
- **Who noticed swap growth first:** neither the tool nor I did in time. My watcher logged it, but I was polling inside one blocking command and did not see it until the sample at 13:49:22. The load had already finished by then. Under rule 2 I stopped scenario D, so the README's `-moe-cache-experts` recommendation for this model was **not tried**. That is a dead end set by my own stop rule, not a finding about that flag.

A dense model bigger than RAM, or a 70B download that would break the 20 GB floor, was not attempted.

## Scenario E: Ollama control

Ollama's install script needs sudo and writes outside the allowed directories (see the deviations section), so I used the manual tarball.

| leg | result |
| --- | --- |
| Install (tarball, 2.1 GB unpacked) | 13.5 s |
| Server start with `OLLAMA_MODELS` pointing at an empty folder | about 4 s, not counted in the legs below |
| `ollama pull qwen2.5-coder:1.5b` (941 MB on disk) | 13.2 s |
| First `ollama run ... --verbose` | 54.8 s wall: load 23.7 s, prompt eval 21.7 s (reported as 1.79 tok/s on 39 tokens), 359 tokens at 39.0 tok/s |
| **Total** | **81.5 s**, against 38.1 s on the goinfer default path |

Ollama used the GPU (CUDA, 7.6 GiB). The answer was correct Go, longer than goinfer's, with a `check` function. This was a single cold run on each side. It is not a benchmark, and the Ollama run included runner startup. I did not repeat it warm.

## Scenario F: screenshots

Vision route: the model has to be a directory with a vision tower, not a GGUF plus an mmproj file.

| attempt | result |
| --- | --- |
| 1: Gemma 3 4B q4_k_m GGUF + `mmproj-...-f16.gguf` passed to `--vision` | Error: `vision: read config: open .../mmproj-google_gemma-3-4b-it-f16.gguf/config.json: not a directory`. The `--vision` help text does say "dir". |
| 2: same files, no `--vision`, mmproj sitting next to the model | Server started, `/v1/models` said `"vision":false`. |

Those two attempts ended the GGUF route (dead end). For a third route I treated it as a new obstacle (getting a vision-tower directory): I downloaded the HF safetensors checkpoint `Qwen/Qwen2.5-VL-3B-Instruct` with curl (7.1 GB, 13:52:55 to 13:54:22) and served it with `--backend cuda --model ~/models/qwen2.5-vl-3b`. That loaded in 11.3 s, printed `loaded Qwen2.5-VL vision tower`, and `/v1/models` said `"vision":true`.

Test images: two PNGs I drew with Pillow (a fake `go build` error terminal in `a.png`, a fake disk-space dialog in `b.png`). Time to first token, streamed through the openai package:

| step | TTFT | prompt tokens | notes |
| --- | --- | --- | --- |
| 1. image A + question 1 (cold) | 7.62 s | 533 | |
| 1 again, the identical request one run later | 0.044 s | 533 | `prefill_reused_tokens` 532: a whole-request prefix cache hit |
| 2. same A bytes, question 2, same conversation (text before image) | 7.54 s | 673 | no reuse |
| 3. different image B, question 3 | 7.57 s | 702 | no reuse |
| Variant: image placed first in each message | A 7.54 s / A again 7.52 s / B 7.53 s | 529 / 570 / 599 | no reuse |

- **Same conversation with the image still in history:** `400: v1 supports 1 image per request, got 2`. To send the identical file a second time I had to drop the first image from the history and keep it as text. That is why the figures above use text-only history.
- Re-sending the same bytes saved nothing: about 7.5 s every time, apart from the exact-repeat case. Compare the 11,137-token text prompt in B, which prefilled in about 2 s. So the image path (the vision encoder) dominates.
- The answers were right: the line with the undefined error was "line 12", and the text on image B was read correctly ("System Update ... 97% full ... 63 GB").

## Friction log

Tags: Guessed, Wanted and absent, Error.

| time | tag | entry |
| --- | --- | --- |
| 13:30:43 | Wanted and absent | `--version` ends in `(c7f8eff76c7c-dirty)`. A release binary reporting "dirty" is a small trust blip. |
| 13:30:59 | Guessed | No `--help` flag for the model cache location on chat or serve (only `pull -o`). I guessed `XDG_CACHE_HOME` would be honoured. It was. |
| 13:31:20 | Guessed | `goinfer-chat` has no prompt argument. I guessed that piping a line into stdin would answer once and exit. It did, but the answer prints after a `you>` label with ANSI escapes and a trailing `bye`. |
| 13:31:20 | Wanted and absent | The README says "GPU built in", but the default backend is `cpu` (14.4 tok/s against 173.8 with `--backend cuda`). I wanted it to pick the GPU on a machine that has one. |
| 13:32:09 | Error | Ollama's official install script: `sudo: a password is required`. Not run. |
| 13:34:20 | Guessed | The odd " Query issued," streamed reply at default sampling (guessed it was sampling noise; temperature 0 was fine). |
| 13:34:34 | Error | opencode run 1: no output, no request at the server, 240 s lost. I suspect stdin; closing it let run 2 proceed. |
| 13:38:48 | Error | opencode's first request is 11,137 tokens, and goinfer's default ctx is 8192, so every request returned 400 and opencode looped. No hint in the server log; the info came only through the 400 body. |
| 13:38:48 | Wanted and absent | No per-request log line from `goinfer-serve`. I wanted one showing prompt tokens and time. |
| 13:41:11 | Guessed | `--ctx 16384` fixed it. The help text for `--ctx` helps. |
| 13:42:41 | Guessed | The chat-template route (`chat.Meta` fed from `tok.ChatTemplate()` and `tok.Has`) came from guessing off the pkg.go.dev method index. |
| 13:42:41 | Wanted and absent | pkg.go.dev `decoder.Load` says it "reads a Gemma 3 snapshot (config.json + model.safetensors)" and that "webgpu falls back to CPU", yet it loaded a Qwen2.5 GGUF path fine. The doc comment looks stale. `Options.Backend` is documented as "cpu or webgpu"; I did not try `cuda` in the library path. |
| 13:48:23 | Error | On the 26B model with `--backend cuda`: `[cuda] resident path DECLINED ... unsupported projection kind ""`. The reason is cryptic, and it is not a VRAM reason. The CPU fallback then ran with ctx 262144 and a "10.4 GB KV" figure, while `fit` said "KV@8192 3.44 GB". |
| 13:48:53 | Error | Swap grew during that load while the tool only warned. No "swap guard" line was printed. |
| 13:49:59 | Error | `--vision <mmproj>.gguf` fails with `.../config.json: not a directory`. `pull` can fetch the mmproj file but nothing consumes it. |
| 13:52:00 | Guessed | I guessed that a vision tower directory means an HF safetensors checkpoint, and fetched it with curl. |
| 13:55:10 | Error | `400: v1 supports 1 image per request, got 2` when a conversation's history still holds the first image. |
| 13:55:42 | Wanted and absent | No visible reuse of an already-encoded identical image: about 7.5 s each time. |

## Dead ends

- Ollama via its official script (needs sudo): used the tarball instead.
- Scenario F via GGUF + mmproj: dead after two attempts.
- Scenario D: stopped by the swap rule, `-moe-cache-experts` untried.
- opencode as an agent on a 1.5B model: it ran, but output was not usable.
