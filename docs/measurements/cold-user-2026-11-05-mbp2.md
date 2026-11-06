# Cold-user run, goinfer v0.24.1, mbp2, 2026-11-05 (INCOMPLETE: stopped by swap rule)

## Contamination note
- I am a Claude session (Sonnet 5.5) started by the project's coordinator. I was launched in ~/tmcode/wgpu only because it is a trusted folder; I did all work in ~/cold-user-v0.24.1 and ignored that wgpu repo (not opened, apart from the harness reading nothing from it).
- I have NOT read the goinfer source tree, its docs/ or its git history. Sources used: the GitHub README (raw main), the GitHub release API for v0.24.1, each binary's --help/error text (I only skimmed the first lines of --help), and pkg.go.dev (v0.24.1 `decoder`, `chat`, `tokenizer` pages). **pkg.go.dev shows doc comments lifted from source**, so scenario C was written from source-derived doc comments. I did NOT open apps/examples/embed/main.go (linked from the README, but not on the allowed list).
- Prior knowledge: I knew goinfer only as a name in this coordinator's project context; no specifics of its CLI/API. I know Ollama/llama.cpp generally.
- Machine: Kubuntu, 8 cores, 15.8 GB RAM, 512 MB swap, CPU only (`no CUDA device answered`), Go 1.27.0, Python 3.14.4 without pip. Disk 56 GB free.
- Baseline at start: swap used 0, `pswpout 0`.

## Versions (rule 4)
- goinfer-serve-linux-amd64 v0.24.1 (7f19e161c4f1), backends: cpu cuda, go1.27.3. sha256 matched checksums.txt.
- goinfer-chat-linux-amd64 v0.24.1 (7f19e161c4f1), same.
- Go toolchain go1.27.0; claude (Claude Code) 2.1.318. Ollama: not installed (not reached).

## Stop event (rule 2)
03:52:18: `grep pswpout /proc/vmstat` read **28** (baseline 0) right after the scenario C embed binary ran (RSS 760 MB; a goinfer-serve 0.5B, a `claude -p` and the embed build/run were overlapping). `free -m`: swap used 0 MB, 12 GB buff/cache, 12-13 GB available. 28 pages = 112 KB, so almost certainly unrelated noise, but the brief says stop at the first pageouts, so I stopped. Scenarios D, E, F were not attempted. Serve's own line: `swap guard: baseline 0.00 GB (system swap)` and `swap guard: baseline 0.00 GB (server swap)`; armed at server +512 MB, system +4096 MB backstop.

## Scenario A: try it (reached, ~2 min of 25)
| step | time | result |
|---|---|---|
| download serve+chat binaries (33 MB + 29 MB) | 03:46:55-03:46:59 | 4 s |
| `chat pull qwen2.5-coder-0.5b` (469 MiB) | 03:47:20-03:47:34 | 14.6 s, sha256 verified, max RSS 20 MB |
| `chat --model <gguf> --temp 0 -p "What is the capital of France? Answer in one sentence."` | 03:47:43 | "The capital of France is Paris." 7 tok, 15.8 tok/s, wall 1.48 s (second run incl. one-time 1 s .giw cache build), max RSS 736 MB |
| nothing -> first answer (no-model-yet start to answer) | 03:46:41-03:47:45 | about 64 s total including my reading time of README/help |

Cold-start note: path from README worked first try. Model cache goes to ~/.cache/goinfer/models (outside the working dir; README says so).

## Scenario B: point my tools at it (partial)
| item | result |
|---|---|
| goinfer-serve -model <gguf> -addr 127.0.0.1:8080 | up in <6 s, 5 ms load, endpoints listed |
| raw OpenAI chat/completions via curl | works; reply "The capital of France is Paris."; timings in response: prefill 270 ms (39 tok), decode 146 ms (7 tok) |
| Official `openai` Python client | **dead end**: no pip on machine, `python3 -m venv venv` failed (ensurepip), then `--without-pip` has no pip. 2 attempts. Not a goinfer problem. |
| Claude Code CLI (/v1/messages) via ANTHROPIC_BASE_URL + dummy key, isolated HOME | 1 attempt, `claude -p "Reply with the single word: pong"`: no output at all after the 120 s harness limit (and until I killed it at ~03:52); no request line visible in serve log (serve logs requests only with a flag I did not look up). Inconclusive; counts as 1 of 2 attempts. No real Anthropic credentials used. |

## Scenario C: embed it (reached)
- Wrote 39-line Go program (kept in ~/cold-user-v0.24.1/embed/main.go) from pkg.go.dev signatures: `tokenizer.LoadGGUF`, `chat.Detect(chat.Meta{ChatTemplate, HasToken})`, `decoder.Load(path, Options{Backend:"cpu",Quant:"int4"})`, `tmpl.Render`, `tk.Encode`, `tmpl.Stops()` -> `tk.TokenID`, `m.Generate(..., SamplingParams{StopIDs})`.
- `go mod init`, `go get .../decoder@v0.24.1 .../tokenizer@v0.24.1 .../chat@v0.24.1` (README's "name the packages" advice): 03:51:33-03:51:55 includes download+build, about 22 s. First build compiled with no errors on first try (guessed `SamplingParams.StopIDs` after reading the type on pkg.go.dev).
- Run: prints "The capital of France is Paris."; wall 3.97 s (incl. 2 s one-time .giw sidecar transcode, 459 MB written beside the GGUF), max RSS 760 MB.
- Friction: my first draft was 43 lines; trimmed blank lines to 39. gofmt -l flags the file (blank lines removed).

## Scenarios D, E, F: NOT REACHED (stopped by swap rule)

## Friction log
| time | tag | entry |
|---|---|---|
| 03:48:19 | Error | `model "x" not found (served: qwen2.5-coder-0.5b-instruct-q4_k_m)`: the server demands the exact model id; useful error, it names the id. |
| 03:47:43 | Wanted and absent | chat auto-"stream-weights: transcoding ... one-time — minutes + ~model-size on disk" printed for a 0.5B that took 1 s; message wording alarming vs reality; 344 MB then 459 MB sidecars appear next to the model (two different .giw files, one per entry point). |
| 03:48:19 | Wanted and absent | serve prints "warning: no -api-key set" on localhost; fine, noted. |
| 03:48:26 | Error | python venv/pip unavailable (host issue). |
| 03:48:40 | Wanted and absent | no way found quickly to confirm whether Claude Code reached the server; no per-request log by default. |
| 03:52:18 | Error | pswpout 0 -> 28 with swap used 0 MB; stop rule applied. |

## Cleanup
Killed goinfer-serve and the claude process. Left in ~/cold-user-v0.24.1: binaries, checksums.txt, embed/, embedbin, serve1.log. Model and sidecars remain in ~/.cache/goinfer/models and beside the GGUF.
