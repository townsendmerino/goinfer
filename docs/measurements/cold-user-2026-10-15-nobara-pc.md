# Cold-user run: goinfer v0.22.0 on nobara-pc, 2026-10-15

| | |
|---|---|
| Run window | 06:36:59 PDT to 06:56:47 PDT, 2026-10-15 (13:36:59Z to 13:56:47Z), about 20 min |
| Machine | nobara-pc: Linux 7.2.0-202.nobara.fc44.x86_64, amd64, 16 threads, 62 GiB RAM (`free -g`), swap 77 GiB (zram 8 GiB + NVMe 69 GiB), NVIDIA GeForce RTX 2070 SUPER 8 GiB, driver 595.91.07 |
| Run directory | `/home/francis/cold-user-2026-10-15` (created empty). `XDG_CACHE_HOME=/home/francis/cold-user-2026-10-15/.cache` was set for every goinfer command except one `--help` (see friction log). The Go module cache (`GOMODCACHE=…/gomod`) and Ollama's store (`OLLAMA_MODELS=…/ollama-models`) were isolated too. |
| Target tag | v0.22.0. GitHub's `releases/latest` also resolved to v0.22.0, published 2026-10-06T20:55:08Z. |
| README read | **The v0.22.0 tag's README**, `raw.githubusercontent.com/townsendmerino/goinfer/v0.22.0/README.md`. I did not read main's README. |
| Other sources used | The GitHub release API and assets for v0.22.0 (`checksums.txt`, binaries), `--help` output, error text, and pkg.go.dev for `decoder`, `chat` and `tokenizer` at v0.22.0. Outside sources, used where the scenario needed them: huggingface.co, fetched by goinfer's own `pull`; the Go module proxy, which delivered go1.27.0 through `GOTOOLCHAIN=auto` rather than go.dev/dl; ollama.com's registry, through `ollama pull`; and npm, configured for `npx` but never reached, because tiny-agents failed on its missing `mcp` module first. |
| Swap baseline | At start: `pswpout=83397355`, swap-used 8,736,828 kB. The serve guard's printed baselines were **8.93 GB** (B) and **8.83 GB** (F), each printed only after the first request. |

## Contamination declaration

**What I was given.** My context held project material that the rules forbid me to use:

- the project's `CLAUDE.md` agent notes (benchmarking rules, build tags, the model archive paths, the night queue);
- the owner's auto-memory index (`MEMORY.md`), whose one-line summaries name internal results, including:
  - gemma-4-26b and 35B on this 8 GB card via expert streaming;
  - the swap guard;
  - a CUDA "12.2K-key prefill decline";
  - Qwen2.5-VL as a shipped vision family;
  - Ollama and llama.cpp being peers on this box;
  - earlier cold-user runs;
- a git status of the repo (main, with recent v0.23.0 release commits).

**What I did not use.** I did not open, read, grep or list anything under `~/mycode`, `~/models`, `/srv/models`, `~/goinfer-*` or `~/.cache/goinfer`, or any git history, and I did not use the above to get past an obstacle. Every command, flag and path came from the README, `--help`, the release assets, pkg.go.dev or error text.

**Where it may have shaped a choice.**

1. **D: picking gpt-oss-20b.** `goinfer-chat models` lists it as the "too big for an 8 GB GPU" case, but I already knew this card was the project's big-MoE test bed.
2. **F: picking Qwen2.5-VL-3B.** `--help` names Qwen2.5-VL's ViT under `-vision`, but my memory index also says that family is validated.
3. **B: reading the server message "12191 attended keys … 49152 B limit".** I recognised it as a known CUDA limit rather than investigating it.
4. **E: knowing where to look for Ollama.** I knew Ollama is the usual peer, so I looked in `~/.local/bin`, which is not a project directory.
5. **B: starting at `-ctx 16384`.** That is both the README's agent advice and something already in my context.

## Versions of every binary used

| binary | `--version` |
|---|---|
| goinfer-chat-0.5b-linux-amd64 | `v0.22.0 (053bc7110a1e)` · `backends: cpu cuda` · `go: go1.27.0 linux/amd64` · `embedded: tier=0.5b quant=int8int8 (baked at build time; --quant has no effect on this binary)` |
| goinfer-chat-linux-amd64 | `v0.22.0 (053bc7110a1e)` · `backends: cpu cuda` · `go: go1.27.0 linux/amd64` |
| goinfer-serve-linux-amd64 | `v0.22.0 (053bc7110a1e)` · `backends: cpu cuda` · `go: go1.27.0 linux/amd64` |
| Go | go1.26.5 on PATH; `GOTOOLCHAIN=auto` switched to go1.27.0 |
| Ollama | `ollama version is 0.32.5` (`~/.local/bin/ollama`, not on PATH) |
| tiny-agents | huggingface_hub 1.17.0 (`~/.local/bin/tiny-agents`) |
| OpenAI Python SDK | openai 3.16.2, Python 3.14.7 |
| node / npx | v22.22.2 (npx was configured for an MCP server; the run never got that far) |

All three goinfer assets passed `sha256sum -c` against the release's `checksums.txt`.

---

## A. Try it: from nothing to an answer. Reached the answer.

**Path.** I followed the README's "model included — nothing else to fetch" route:

1. Downloaded `checksums.txt` and `goinfer-chat-0.5b-linux-amd64` (659 MB).
2. Verified the checksum and ran `chmod +x`.
3. Ran: `goinfer-chat-0.5b-linux-amd64 --temp 0 -p "What is the capital of Australia? Answer in one sentence."`

**Answer** (stdout): `The capital of Australia is Canberra.`

**Stderr:**
```
backend: cuda (auto: a CUDA device answered; -backend cpu to use the CPU)
mapping weights…
[resident] BuildResident declined (ok=false err=resident path declined: cuda: layer 0: MoE experts are "int8"/"int8" — the resident MoE GEMVs are int4-only (load with Quant: "int4")) — continuing on the CPU/staged path
loaded 24-layer model (hidden 896, vocab 151936) in 1.398s [backend=cuda decode=cuda-resident (int8int8)]
chat template: chatml
[7 tok, 89.8 tok/s]
```

**Finding (Error): the load messages contradict each other.**
- The model is a dense Qwen2.5 0.5B, yet the binary says it declined a *MoE* resident path and is "continuing on the CPU/staged path".
- The next line says `decode=cuda-resident`.
- `nvidia-smi` showed the process holding 1,114 MiB on the GPU, so the second line is the true one.

| A numbers | value |
|---|---|
| Scenario start to answer | **35 s** (13:37:40Z → 13:38:15Z) |
| Asset download (checksums + 659 MB) | 19.9 s |
| Answer wall, load included | 1.97 s (load 1.398 s) |
| Peak RSS | 1.31 GB |
| Short answer | 7 tok, 89.8 tok/s |
| 400-token story, GPU | 158.2 tok/s (binary's own counter) |
| Same question, `--backend cpu` | 17.7 tok/s |
| Swap | no movement |

## B. Point my tools at it. OpenAI client: reached. Agent CLI: reached an answer with no tools.

**Server launch.** I used the README's one-liner: `goinfer-serve-linux-amd64 -model hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m`. The server:

1. downloaded 1.0 GiB;
2. printed `stream-weights: transcoding … → …int4.e4h.cuda.giw … cache ready (959 MB) in 22s`, although I never passed `-stream-weights`;
3. loaded in 3.5 s, resident on CUDA int4, with 16384 context;
4. printed `swap guard: armed, threshold +512 MB over baseline` with **no baseline figure**. `swap guard: baseline 8.93 GB swap-used` appeared only after the first request.

**OpenAI client** (official Python SDK; general knowledge of the API shape):
- `models.list` returned `['qwen2.5-coder-1.5b-instruct-q4_k_m']`.
- A non-streamed chat returned `'The capital of Australia is Canberra.'` with `prompt_tokens=20, completion_tokens=7`, plus a non-standard `prefill_reused_tokens=0` field.
- A streamed chat returned `'1,2,3,4,5,6,7,8,9,10'` in 20 chunks.
- A tool call came back well-formed: `finish_reason=tool_calls`, `get_weather({"city": "Paris"})`.
- `/v1/messages`, the Anthropic shape, sent with curl, returned `{"content":[{"text":"The capital of Australia is Canberra.","type":"text"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":7}}`.

**Agent CLI.** No codex, claude, opencode, aider, goose or crush was on PATH. `tiny-agents` (huggingface_hub 1.17.0) was installed in `~/.local/bin`.

- **Attempt 1** used the filesystem MCP server via `npx`. It died inside tiny-agents before any request reached goinfer: `ModuleNotFoundError: No module named 'mcp'`.
- **Attempt 2** used no MCP servers. It printed `Agent loaded with 0 tools:` and answered the first turn `The capital of Australia is Canberra.` through goinfer.
  - With stdin piped, tiny-agents then looped on EOF, re-sending a growing conversation (568 "Canberra" lines).
  - goinfer refused cleanly: `prompt is 18804 tokens but the model's context window is 16384 (context_length_exceeded); this is the server's GPU context (-ctx), not the model's limit (32768 tokens): restart goinfer-serve with -ctx 18805 or more (up to 32768) to accept it`.
  - I killed the agent by PID. The loop is tiny-agents' bug; the 400 message is a good one.
- On the long reused session the server also logged that batched prefill was declined: `cuda prefill: layer 0 attention at 12191 attended keys needs 49276 B of shared memory, past this device's 49152 B limit — batched prefill has no split-KV path`. It then fell back to the per-token path.
- **An agent doing real tool use was not reached.**

| B numbers | value |
|---|---|
| Launch to "serving on" | ≈57 s (13:38:47Z → by 13:39:44Z), including the 1.0 GiB download, 22 s transcode and 3.5 s load |
| Non-streamed chat (20 in / 7 out) | 0.08 s |
| Streamed chat, time to first token / total | 0.020 s / 0.13 s |
| Tool call | correct |
| Scenario wall | 252 s |
| Swap | no pageouts |

## C. Embed it: ≤40-line Go program. Reached the answer.

**Toolchain.** Go on PATH is go1.26.5, and the README says 1.27+.
- `go mod init`, then `go get …/decoder@v0.22.0 …/tokenizer@v0.22.0`, as the README instructs.
- `GOTOOLCHAIN=auto` raised `go.mod` to `go 1.27.0` and silently fetched go1.27.0 from the module proxy.
- It added aikit v1.57.0, x/sys and x/text, all marked `// indirect`, and the build needed no `go mod tidy`.

**The program.** `examples/embed/main.go` was outside the allowed sources, so I wrote 38 lines from pkg.go.dev signatures:
```
tokenizer.LoadGGUF(path) → chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
→ tmpl.Render("", []chat.Turn{{Role:"user",Content:q}}) → tok.Encode(prompt, true)
→ decoder.Load(path, decoder.Options{Quant:"int4"}) → m.Generate(ctx, ids, 64, decoder.SamplingParams{Temperature:0})
→ tok.Decode(ids); gen.Err()
```

**Output:** `The capital of Australia is Canberra. <nil>`. It ran on the CPU, because I did not import the cuda module.

**Doc findings (minor):**
- pkg.go.dev's `decoder.Model` doc reads "Model is a loaded Gemma 3 checkpoint", which is stale.
- `Generate`'s doc cites internal tracker ids ("R10, docs/measurements/…").

| C numbers | value |
|---|---|
| `go get` (with toolchain download) | 14.1 s |
| `go build` | 5.1 s, 11.9 MB binary |
| Run (CPU int4, 1.5B q4_k_m) | 3.04 s wall, 2.33 GB peak RSS |
| Program length | 38 lines |
| Scenario wall | 82 s, plus reading pkg.go.dev |

## D. Run bigger than my hardware. STOPPED by the swap-safety rule (not a goinfer dead end).

**What happened, in order.**

1. `goinfer-chat models` lists gpt-oss-20b (12.11 GB mxfp4) as "too big to hold fully resident on an 8 GB GPU … bring -moe-cache-experts or -stream-weights".
2. `goinfer-chat pull gpt-oss-20b` took 4m6s for 11.3 GiB, sha256 verified. It exited at 13:48:37Z.
3. **Pageout.** `pswpout` rose 83397355 → 83397362 (+7 pages, 28 KB) between **13:48:55Z and 13:49:00Z**, and swap-used grew about 3 MB. **No goinfer process was running at that moment**: it was the idle gap after the 11.3 GiB write and verify. I did not see it until 13:50:35Z, from my own 5-second monitor. goinfer's guard was not running.
4. `goinfer-chat fit` printed its budget at 13:49:08Z, ~10 s *after* the pageout:
   ```
   cpu     RESIDENT        dense 1.42 GB + experts 10.01 GB + KV@16384 1.50 GB fits 54.71 GB free
   cuda    EXPERT-CACHED   dense 0.88 GB + 16/32 experts 5.01 GB + KV@16384 1.50 GB fits 7.49 GB free
   ```
5. **D1, the naive route** (README: just `-model`), started at 13:49:14Z:
   - It wrote an 11.81 GB `.giw` sidecar (49 s), with no flag asked for.
   - It cut the resident context to 14240 so 4 KV slots fit.
   - It tried full CUDA residency, which failed with **a full goroutine stack trace** (`[cuda] resident build failure, detail: goroutine 1202 … cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY`).
   - Then came the useful line: `the model does not fit this GPU's memory. Try -moe-cache-experts …, a smaller -quant (int4) or -ctx, or run on the CPU on purpose with -backend cpu) — continuing on the CPU/staged path`.
   - It served on CPU at int4mix. Load wall was 81 s, with no further pageouts.
6. I killed D1 by PID per rule 2. **`-moe-cache-experts` was not run** and no request was sent.

**Verdict.**
- **Not passed, by the letter.** Swap moved before any goinfer budget output.
- That movement happened while no goinfer process was running.
- The first refusal-style warning goinfer gives a naive user is buried under a stack trace.
- `fit` correctly predicted the outcome (cuda EXPERT-CACHED), but the default load ignored that prediction and attempted full residency first.

**Interpretation I applied.** I read rule 2 as stopping the scenario in progress. I continued E and F, with the monitor running, on the condition that any further `pswpout` increase would stop the whole run. None occurred. Re-run D with `-moe-cache-experts` on a quiet box.

| D numbers | value |
|---|---|
| Pull | 11.3 GiB in 4m6s (~45 MiB/s) |
| `fit` | instant; cuda EXPERT-CACHED |
| D1 load (no flags) | 81 s: 49 s transcode + 30.3 s load, CUDA OOM, then CPU |
| Disk | 12.11 GB gguf + 11.81 GB sidecar |
| pswpout | +7 at 13:48:55–13:49:00Z (no goinfer process running), none afterwards |
| Swap-used | +3 MB peak, then below baseline |
| Guard | never fired; printed no baseline because D1 got no request |
| Scenario wall | 403 s until the stop |

## E. Control: the same question through a peer. Reached.

**Setup.** Ollama 0.32.5 at `~/.local/bin/ollama`, run isolated: own `OLLAMA_MODELS` in the run directory, port 11435, own pidfile.

**Run.**
- `ollama pull qwen2.5-coder:0.5b` fetched 398 MB in 7.86 s.
- `ollama run … --verbose "<same question>"` printed `The capital of Australia is Canberra.`
- Its `/v1/chat/completions` returned the same text.

**Same-weights check.** I copied Ollama's q4_k_m blob into the run directory and ran goinfer-chat on it. It gave the same answer.

**Caveats.** This is exploratory and not to the provenance standard:
- the run was not interleaved;
- the two tools count rate differently: Ollama's eval rate is decode-only, and goinfer's `[N tok, X tok/s]` definition is not documented in `--help`;
- Ollama's binary was already installed, while goinfer's time in A includes downloading its binary.

| E numbers | goinfer | Ollama 0.32.5 |
|---|---|---|
| Nothing to answer | 35 s (download 659 MB binary with model baked in, run) | 20 s (start daemon, pull 398 MB, run) |
| Weights for that answer | baked 0.5B int8int8 | qwen2.5-coder:0.5b q4_k_m |
| Answer | Canberra | Canberra |
| Run wall / load | 1.97 s / 1.40 s | 2.81 s / 2.41 s |
| Short-answer rate | 89.8 tok/s (7 tok) | 170.5 tok/s eval (8 tok) |
| 400-token decode, **same q4_k_m blob** | 339.4 tok/s | 262.9 tok/s |
| Placement | CUDA resident int4 | 100% GPU, context 4096 |

## F. Show it a screenshot, then resend it. Reached; the resend gets no reuse.

**Choosing a model.** No vision model is listed by `goinfer-chat models`. `--help` says `-image` is GLM-OCR only, and that `-vision` auto-discovers Qwen2.5-VL's ViT.

**Setup.**
- `goinfer-chat pull Qwen/Qwen2.5-VL-3B-Instruct:safetensors` fetched 7.0 GiB of bf16 in 1m30s, sha256 verified.
- `goinfer-serve -model <dir>` loaded in 14.6 s: CUDA resident int4, with `loaded Qwen2.5-VL vision tower … (merge 2, image-pad id 151655)`.
- The screenshot was a 960×540 PNG fake terminal, synthesized with PIL because this session has no display: psql "Connection refused" on db.internal (10.0.0.7):5432, `Active: inactive (dead)`, and "BUILD #4127".
- I used the OpenAI SDK with a base64 `image_url` part.

**Turn 1:** the answer was correct (the error, the IP, port 5432, inactive postgres, BUILD #4127). It took 10.96 s for 675 prompt tokens and 200 output tokens (my cap).

**Turn 2** (the same image resent, "What IP address and port?"):
- It took 10.04 s for 920 prompt tokens and 72 output tokens, with **`prefill_reused_tokens=0`**.
- The answer mislabelled the fields (`IP Address: db.internal`, `Port: 10.0.0.7, port 5432`). That is the 3B model's quality.

**Time-to-first-token probes** (max_tokens 8):

| request | time to first token | prompt tokens | reused |
|---|---|---|---|
| single image, first send | 9.46 s | 662 | 0 |
| byte-identical resend | **0.04 s** | 662 | 661 |
| conversation extended, same image again | 9.42 s | 703 | **0** |
| conversation extended, text-only follow-up | 9.47 s | 685 | **0** |

**Finding.** Only an exact repeat hits the cache. Every follow-up turn in an image conversation pays the full ~9.4 s again.

| F numbers | value |
|---|---|
| Pull | 7.0 GiB in 1m30s |
| Launch to serving | 18 s |
| Turn 1 / turn 2 | 10.96 s / 10.04 s |
| Scenario wall | 207 s |
| Swap | no pageouts; guard baseline 8.83 GB, printed after the first request |

---

## Dead ends

1. **B, agent CLI with tools.** tiny-agents lacks the `mcp` package (`ModuleNotFoundError: No module named 'mcp'`). The no-tools fallback answered through goinfer, then looped on piped-stdin EOF. Both attempts are spent. No other agent CLI was found on PATH.
2. **D, `-moe-cache-experts`.** Not attempted: the swap-safety rule stopped the scenario. This is a stop, not a goinfer failure.

## Top findings

1. **D.** Under default flags a too-big MoE first attempts full CUDA residency, dumps a goroutine stack on OOM, and then prints good advice and falls back to CPU. That is despite `fit` having predicted EXPERT-CACHED.
2. **D, B.** A `.giw` sidecar roughly the size of the model (11.8 GB for gpt-oss-20b) is written by default, under a message headed "stream-weights: transcoding", without `-stream-weights` being passed.
3. **F.** Follow-up turns in an image conversation get zero prefix reuse (~9.4 s each). Only a byte-identical request is cached.
4. **A.** The bundled dense 0.5B prints "BuildResident declined … MoE experts … continuing on the CPU/staged path", then runs CUDA-resident.
5. **B, F.** The startup banner says the swap guard is armed but gives no baseline. The baseline line appears only after the first request.
6. **B.** The context-overflow 400 is excellent: it names the exact `-ctx`.
7. **E.** goinfer-chat, which has no server, prints "the serving watch arms once the server is up".

## Friction log (verbatim; PDT | UTC | tag | note)

```
06:37:35PDT | 13:37:35Z | Info | Read README at the v0.22.0 tag (raw.githubusercontent.com/.../v0.22.0/README.md) and release API for v0.22.0
06:37:40PDT | 13:37:40Z | Info | Scenario A start
06:38:07PDT | 13:38:07Z | Guessed | --help printed cache path under ~/.cache/goinfer because XDG_CACHE_HOME was not yet exported in that one shell (help only; nothing written). Exporting it for every command from here.
06:38:15PDT | 13:38:15Z | Error | A: 0.5B bundled binary on a dense Qwen2.5 0.5B printed '[resident] BuildResident declined (... layer 0: MoE experts are "int8"/"int8" — the resident MoE GEMVs are int4-only ...) — continuing on the CPU/staged path' and then 'decode=cuda-resident (int8int8)'. The model is not MoE; the two lines contradict each other about where it ran.
06:38:15PDT | 13:38:15Z | Info | A: answer 'The capital of Australia is Canberra.' reached, 1.97 s wall, 89.8 tok/s, RSS 1.31 GB; no swap movement
06:38:33PDT | 13:38:33Z | Info | Scenario B start
06:39:54PDT | 13:39:54Z | Wanted and absent | B: serve banner prints 'swap guard: armed, threshold +512 MB over baseline' but no baseline figure (no 'baseline N GB swap-used' line).
06:39:54PDT | 13:39:54Z | Guessed | B: banner says 'stream-weights: transcoding ... (int4, one-time)' though -stream-weights was not passed; assumed this is the default .giw sidecar the --help -direct-load text describes.
06:40:12PDT | 13:40:12Z | Info | B: agent CLI = tiny-agents (huggingface_hub 1.17.0) found in ~/.local/bin (not on PATH); MCP filesystem server fetched via npx from npm (outside source, declared)
06:40:18PDT | 13:40:18Z | Error | B: tiny-agents attempt 1 failed before reaching goinfer: ModuleNotFoundError: No module named 'mcp' (tiny-agents' own extra not installed on this box). Attempt 2: no MCP servers.
06:42:37PDT | 13:42:37Z | Info | B: tiny-agents attempt 2 reached goinfer: first turn answered 'The capital of Australia is Canberra.' (0 tools loaded). With stdin piped, tiny-agents then looped on EOF re-sending the conversation (568 'Canberra' lines) until the prompt reached 18804 tokens and goinfer returned a clean 400 context_length_exceeded naming -ctx 18805; killed by PID. Loop is a tiny-agents stdin artifact, not goinfer.
06:42:37PDT | 13:42:37Z | Info | B: the serve log printed 'swap guard: baseline 8.93 GB swap-used' only AFTER the first request (line 50), not in the startup banner (line 44 says armed, no figure).
06:42:37PDT | 13:42:37Z | Error | B: server log: 'batched prefill declined for a 10-token prompt ... layer 0 attention at 12191 attended keys needs 49276 B of shared memory, past this device's 49152 B limit' — a long reused-session context drops to the per-token prefill path on this card; reported once
06:42:55PDT | 13:42:55Z | Info | Scenario C start
06:44:17PDT | 13:44:17Z | Wanted and absent | C: Go on PATH is go1.26.5; README needs 1.27+. GOTOOLCHAIN=auto silently fetched go1.27.0 from the Go module proxy on 'go get' (Go's own mechanism; not go.dev/dl). No error, but a stranger without network to proxy.golang.org would hit a wall.
06:44:17PDT | 13:44:17Z | Guessed | C: could not read examples/embed/main.go (outside allowed sources); wrote the program from pkg.go.dev signatures: tokenizer.LoadGGUF + Tokenizer.ChatTemplate/Has -> chat.Detect(chat.Meta) -> Template.Render -> decoder.Load(path, Options{Quant:int4}) -> Model.Generate. Guessed that Encode(addBOS=true) is harmless for Qwen (docs say byte-level families ignore it).
06:44:17PDT | 13:44:17Z | Error | C (doc): pkg.go.dev decoder.Model doc says 'Model is a loaded Gemma 3 checkpoint' though Load serves any family; Generate's doc cites internal paths (R10, docs/measurements/...).
06:44:17PDT | 13:44:17Z | Info | C: 38-line program printed 'The capital of Australia is Canberra. <nil>' on CPU (no cuda import), 3.04 s wall incl. load, RSS 2.33 GB; build 5.1 s after 14 s go get
06:44:24PDT | 13:44:24Z | Info | Scenario D start
06:51:07PDT | 13:51:07Z | Error | D: SWAP-SAFETY STOP. pswpout rose 83397355 -> 83397362 (+7 pages, 28 KB) between 13:48:55Z and 13:49:00Z, i.e. after 'pull gpt-oss-20b' finished (13:48:39Z, sha256 verify of 11.3 GiB) and during/just after 'goinfer-chat fit' — BEFORE the D1 serve started (13:49:14Z). swap-used +3 MB then fell. Noticed by me at 13:50:35Z from my 5-s monitor, not by goinfer (the serve swap guard was not running yet; it armed at the end of the D1 load and reported nothing). D1 serve killed by PID at once; D stopped per rule 2.
06:51:17PDT | 13:51:17Z | Info | D: 'goinfer-chat fit' (13:48:4xZ, before the pageout) printed a per-backend budget: cpu RESIDENT 54.71 GB free; cuda EXPERT-CACHED dense 0.88 + 16/32 experts 5.01 + KV 1.50 fits 7.49 GB free.
06:51:17PDT | 13:51:17Z | Error | D: D1 serve with no flags (README path: just -model): CUDA resident build hit CUDA_ERROR_OUT_OF_MEMORY and printed a full goroutine stack trace ('[cuda] resident build failure, detail: goroutine 1202 ...') before the useful line 'the model does not fit this GPU's memory. Try -moe-cache-experts ...' — then served on CPU (int4mix). The fit command had already predicted EXPERT-CACHED for cuda, but the default load tried full residency anyway.
06:51:17PDT | 13:51:17Z | Guessed | D: banner 'stream-weights: transcoding ... (one-time — minutes + ~model-size on disk)' wrote an 11.26 GB .giw sidecar beside the 12.11 GB gguf without -stream-weights being asked for (disk use doubles silently).
06:51:17PDT | 13:51:17Z | Info | D: decision — rule 2 applied to scenario D (stopped, -moe-cache-experts attempt NOT run). E and F continued with the 5-s monitor, on the reading that the pageout preceded any goinfer load and the rule stops the scenario in progress; any further pswpout increase stops the whole run.
06:51:25PDT | 13:51:25Z | Info | Scenario E start: peer = ollama 0.32.5 at ~/.local/bin/ollama (installed on the box, not on PATH); isolated OLLAMA_MODELS in the run dir, port 11435
06:52:24PDT | 13:52:24Z | Info | Scenario F start
06:56:47PDT | 13:56:47Z | Info | CORRECTION to the 13:51:07Z and 13:51:17Z D lines (from file mtimes): pull exited 13:48:37Z; D-fit.txt written 13:49:08Z. The +7-page pageout (13:48:55-13:49:00Z) therefore fell in the gap when NO goinfer process was running, ~10 s BEFORE fit printed its budget, not during or after it. By the letter of rule 2, swap moved before any goinfer budget output.
06:56:47PDT | 13:56:47Z | Info | E: ollama from nothing (binary pre-installed): serve start + pull qwen2.5-coder:0.5b (398 MB, 7.86 s) + run (2.81 s) = 20 s to 'The capital of Australia is Canberra.'; 400-token decode 262.9 tok/s (ollama eval_duration). goinfer-chat on the SAME blob (q4_k_m, copied from ollama's store): same answer, '[400 tok, 339.4 tok/s]' (goinfer's own counter; definition not documented in --help). Exploratory, not to provenance standard.
06:56:47PDT | 13:56:47Z | Error | E: goinfer-chat (no server) printed 'swap guard (load): not armed ... the serving watch arms once the server is up' — a server message in the one-shot chat binary.
06:56:47PDT | 13:56:47Z | Guessed | F: no vision model in 'goinfer-chat models'; --help says -image is GLM-OCR only and -vision auto-discovers Qwen2.5-VL's ViT in the --model dir, so picked hf Qwen/Qwen2.5-VL-3B-Instruct:safetensors (7.0 GiB bf16, 1m30s pull). Screenshot synthesized with PIL (no display in this session).
06:56:47PDT | 13:56:47Z | Info | F: turn 1 (image + question) answered correctly (connection refused to db.internal 10.0.0.7:5432, postgresql inactive, BUILD #4127) in 10.96 s; turn 2 resent the SAME image: 10.04 s, prefill_reused_tokens=0, and the answer mislabelled the IP ('IP Address: db.internal', 'Port: 10.0.0.7, port 5432').
06:56:47PDT | 13:56:47Z | Error | F: TTFT probes: first send 9.46 s; byte-identical resend 0.04 s (661 reused); conversation extended with the same image again 9.42 s (0 reused); extended with a TEXT-ONLY follow-up 9.47 s (0 reused). Any follow-up turn in an image conversation re-pays the full ~9.4 s; only an exact repeat hits the cache.
06:56:47PDT | 13:56:47Z | Info | Cleanup: all servers (goinfer x3, ollama) killed by PID; nvidia-smi shows no compute apps; swap monitor stopped 13:56:34Z (pswpout 83397362, swap-used 8.62 GB, below the 8.74 GB start).
```

The run directory now holds about 35 GB of downloads and sidecars: gpt-oss-20b and its `.giw`, Qwen2.5-VL-3B, the 1.5B, the Ollama blob and an isolated Go module cache.
