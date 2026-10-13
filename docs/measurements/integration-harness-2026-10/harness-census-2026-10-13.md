# First-request census: Pi, Oh My Pi, Hermes, OpenClaw, Cline, Goose, Aider, Kilo (I4), 2026-10-13

`docs/tasks/task-integration-harness-2026-10.md` I4, the eight harnesses the 2026-10-12 census
(`harness-census-2026-10-12.md`) did not install. nobara-pc, all pinned under `~/harnesses`. Same two phases and the
same scripts as that census, in this directory:

1. **Capture** (`harness_capture.py`, `capture.py`): each harness runs once, headless, from an empty work directory with
   a fresh `HOME`, against a recorder that stores every request and answers 400. No model answers, so no harness runs a
   tool. Each harness has its own port (18483 + its index in `PORTS`). The bodies are under `~/harnesses-captures/<harness>/`,
   outside the repo.
2. **Replay** (`harness_replay.py`): each recorded POST goes to a CPU goinfer (`serve` at `e5267a7c` plus another
   session's uncommitted edits to `internal/connect/` and `internal/serveapp/responses.go`, built `CGO_ENABLED=0`),
   Qwen2.5-Coder-0.5B q4_k_m, `-ctx 32768`, streaming off, output capped at one token. The size is goinfer's own
   `prompt_tokens`. Log: `replay-nobara-2026-10-13.log`. serve's swap guard tripped twice during the replay
   (other processes on the box grew swap past the guard's 512 MB threshold; it then answers 503 `halted`). Each time
   serve was restarted and the unfinished harnesses replayed; the log keeps only the status-200 lines. opencode was not
   re-replayed (its 2026-10-12 count stands).

Cross-check against the 2026-10-12 CUDA numbers, same tokenizer: Claude Code 20,196 (was 20,199) and Codex 5,848
(was 5,850) replayed on this CPU server. The counts move by a few tokens between servers, not by a few percent.

Sizes are under the Qwen2.5 tokenizer and move by a few percent on another family. Nothing here is a timing.

## Result

All eight configured to say 32,768 to the harness (so the window the harness believes is not the window that bounds
the request). "Fits 16,384" applies I4's rule: first request plus 25% must be at most 16,384, so the request must be at
most 13,107 tokens.

| harness, version | route | first request (main) | tools sent | schema | fits 16,384 with I4's 25% margin? |
|---|---|---|---|---|---|
| Pi 1.2.0 (`@earendil-works/pi-coding-agent`) | `/v1/chat/completions` | **1,330 tokens** | 4 | 3 KB | yes (1,663) |
| Oh My Pi 18.10.3 (`@oh-my-pi/pi-coding-agent`) | `/v1/chat/completions` | **6,970 tokens** | 11 | 16 KB | yes (8,713) |
| Hermes Agent 0.21.7 (source install, `71d34bad`) | `/v1/chat/completions` | **11,276 tokens** | 24 | 40 KB | the request fits (14,095), **but Hermes refuses to run under a 64,000-token window**, so not viable at 16,384 |
| OpenClaw 2026.10.3 (`59cb447`) | `/v1/chat/completions` | **7,166 tokens** | 11 (41 cataloged behind a compact surface) | 10 KB | yes (8,958) |
| Cline 3.0.71 | `/v1/chat/completions` | **4,606 tokens** | 25 | 18 KB | yes (5,758) |
| Goose 1.54.0 | `/v1/chat/completions` | **4,001 tokens** (+ a 413-token title request before it) | 17 | 14 KB | yes (5,001) |
| Aider 0.86.2 | `/v1/chat/completions` | **593 tokens** (empty directory; plus the repo map in a real one, up to 4,096) | 0 (no `tools` field) | 0 | yes (741, repo map aside) |
| Kilo Code CLI 7.8.8 (`@kilocode/cli`) | `/v1/chat/completions` | **15,144 tokens** | 23 | 51 KB | **no** (18,930 needed) |

Harnesses already measured (2026-10-12, not re-run): Claude Code 2.1.296 20,199 / 23 tools; opencode 1.19.0 7,165 /
10; Codex 0.162.1 5,850 / 9.

- **None needed a login, a paid account or a license click-through to start.** Each ran from a fresh `HOME` with only the
  config shown below. Nothing was skipped.
- **Hermes is not viable at any window goinfer is likely to serve here.** Configured with `context_length: 32768` it
  refuses before sending a chat request: "Model coder has a context window of 32,768 tokens, which is below the minimum
  64,000 required by Hermes Agent." It had sent five GET probes (below) and no POST. The census used 65,536.
- **Oh My Pi's first user message carries the date and working directory** in a `<system-reminder>` block, and its system
  prompt is 10.7K characters; tool schemas are 16 KB. Two sessions are not token-identical.
- **OpenClaw's tool surface is deliberately compact.** The log line `tool-search: cataloged 41 tools behind compact prompt
  surface` says 41 tools exist and 11 are sent; the rest are found through a search tool. Its system prompt is the
  largest of the eight (21.8K characters). It sends `tool_choice`.
- **Goose sends a title request first**: a no-tools request ("Generate a short title (four words or less)…", 413
  tokens, system plus user) before the main request, and it repeats both after the 400: seven requests were recorded, the title request three times and the
  17-tool request four times (the replay skips the byte-identical repeats). Goose's
  process exits 0 on a 400 and prints the error as an assistant message, so a harness-level success check must read
  the JSON, not the exit code.
- **Hermes probes the server before it chats**: `GET /api/v1/models`, `GET /api/tags`, `GET /v1/props`, `GET /props`,
  `GET /version`, in that order, then the POST. These look like probes for Ollama, llama.cpp and similar servers (window
  detection). What goinfer answers to each was not tried; the recorder answered 400 to all of them and Hermes went on
  to POST regardless, with `context_length` set in its config. Its POST has no output cap and sends `reasoning_effort`.
- **Cline asks for `max_tokens` equal to the window minus its own estimate of the input**: 24,758 with a 32,768 window
  and an 8,010-token estimate, whatever `maxTokens` is in its provider settings (4,096 set, ignored). At a 16,384
  window it would ask for about 8,374. A server that rejects `prompt + max_tokens > window` will see that on every turn
  of a long session; goinfer's behaviour for it is I1's question.
- **Cline leaves a daemon behind.** Every run starts a detached `cline --cline-hub-daemon --cwd <dir> --port <n>`
  (PPID 1) that outlives the CLI. `harness_capture.py` stops the one it started, by PID, after the run. An unattended
  matrix has to do the same or it leaves one listener per run.
- **Pi, Oh My Pi and OpenClaw send `max_completion_tokens`**, Kilo and Cline send `max_tokens`, Hermes, Goose and Aider
  send none. Pi, Oh My Pi, OpenClaw and Kilo took 4,096 from the `maxTokens` / `limit.output` the census config gave
  them, so the number is the config's, not the harness's own default.
- **Aider uses no tools at all.** It speaks edit formats in text ("whole" for an unknown model) and its request has five
  messages of few-shot examples, so a goinfer tool-call path is never exercised by it. It sends `temperature: 0`. It
  created a git repository, a `.gitignore` and `.aider*` files in the work directory without being asked
  (`--yes-always` answered its prompt). Its exit code is 0 on a 400.
- **Kilo sent no title request in this run** (one request recorded). opencode, the harness it forks, sent one in the
  2026-10-12 census; do not read the absence as a Kilo property, since the recorder's 400 may end the session first.
- **Pi sent one request, no side request.** Its 4 tools are `read`, `bash`, `edit`, `write`.

## How to run each headless against goinfer

`$GI` is the server root with no path (`http://127.0.0.1:8080`), `$M` the model id `/v1/models` lists, `$W` the served
window. Every harness below was exercised for real only against the recorder, so each line is "the first request reached
the recorder at the path shown", not "a turn completed". All use a fresh `HOME`; none needs the real one.

| harness | one-shot argv | config |
|---|---|---|
| Pi | `pi -p --provider goinfer --model $M --no-session "<prompt>"` | `$HOME/.pi/agent/models.json` (override the directory with `PI_CODING_AGENT_DIR`): `{"providers":{"goinfer":{"baseUrl":"$GI/v1","api":"openai-completions","apiKey":"goinfer","models":[{"id":"$M","name":"$M","contextWindow":$W,"maxTokens":4096}]}}}`. Env: `PI_OFFLINE=1` (or `--offline`) turns off its startup network use. |
| Oh My Pi | `omp -p --model goinfer/$M --no-session "<prompt>"` with `bun` on `PATH` | `$HOME/.omp/agent/models.yml`: `providers: {goinfer: {baseUrl: $GI/v1, apiKey: goinfer, api: openai-completions, models: [{id: $M, name: $M, contextWindow: $W, maxTokens: 4096}]}}` (written as block YAML in `harness_capture.py`). `PI_CODING_AGENT_DIR` overrides the agent directory. |
| Hermes | `hermes -z "<prompt>"` (add `--yolo --accept-hooks` for an unattended run that must execute tools) | `$HOME/.hermes/config.yaml`: `model: {default: $M, provider: custom, base_url: $GI/v1, api_key: goinfer, context_length: $W}`. `$W` must be at least 65,536. |
| OpenClaw | `openclaw agent exec --config $HOME/.openclaw/openclaw.json --model goinfer/$M --cwd <dir> --json "<prompt>"`, **Node 24** (`>=24.16 <25`); no Gateway is started | the `--config` file: `{"models":{"mode":"replace","providers":{"goinfer":{"baseUrl":"$GI/v1","apiKey":"goinfer","api":"openai-completions","models":[{"id":"$M","name":"$M","reasoning":false,"input":["text"],"contextWindow":$W,"maxTokens":4096}]}}}}`. `agent exec` uses a temporary state directory and deletes it; `--code-mode direct\|auto\|code` and `--local-model-lean` choose the tool surface. |
| Cline | `cline -P openai-compatible -m $M --json --timeout 600 "<prompt>"` (act mode, auto-approve is the default; `--yolo` for the limited unsupervised set) | first, once: `cline auth -p openai-compatible -b $GI/v1 -k goinfer -m $M`, which writes `$HOME/.cline/data/settings/providers.json`; then add `"contextWindow": $W` to `providers["openai-compatible"].settings` there (`--data-dir <dir>` or `--config <dir>` isolate it). Stop its hub daemon afterwards. |
| Goose | `goose run --no-session -q --output-format stream-json -t "<prompt>"` | environment only: `GOOSE_PROVIDER=openai OPENAI_HOST=$GI OPENAI_BASE_PATH=v1/chat/completions OPENAI_API_KEY=goinfer GOOSE_MODEL=$M GOOSE_CONTEXT_LIMIT=$W`. No keyring needed. `GOOSE_MODE` defaults to `auto` (tools run without asking). It logs every LLM request under `$HOME/.local/state/goose/logs/llm_request.N.jsonl`. |
| Aider | `aider --model openai/$M --message "<prompt>" --yes-always` (add `--no-check-update --no-analytics --no-auto-commits`; `--no-git` is not needed, it makes a repo anyway) | env `OPENAI_API_BASE=$GI/v1 OPENAI_API_KEY=goinfer`, and `<work dir>/.aider.model.metadata.json`: `{"openai/$M":{"max_input_tokens":$W,"max_output_tokens":4096,"litellm_provider":"openai","mode":"chat"}}` (or `--model-metadata-file`). |
| Kilo | `kilo run -m goinfer/$M --format json "<prompt>"` (`--auto` approves permissions that are not denied; `--pure` skips external plugins) | `$HOME/.config/kilo/kilo.json`, the same shape as opencode's: `{"$schema":"https://opencode.ai/config.json","provider":{"goinfer":{"npm":"@ai-sdk/openai-compatible","name":"goinfer","options":{"baseURL":"$GI/v1","apiKey":"goinfer"},"models":{"$M":{"name":"$M","limit":{"context":$W,"output":4096}}}}}}`. |

## Compaction and overflow settings seen, by harness

For the night job that drives each harness past its window. "Default" is what the installed version's own docs or source
say; none of these was exercised, because no harness got a model reply.

| harness | knob (where) | default | note |
|---|---|---|---|
| Pi | `compaction.enabled`, `compaction.reserveTokens`, `compaction.keepRecentTokens` (`~/.pi/agent/settings.json` or `<dir>/.pi/settings.json`; `compaction.modelOverrides` keys on `provider/modelId`) | true, 16384, 20000 | triggers when `contextTokens > contextWindow - reserveTokens`, so by that formula at a 16,384 window with the default reserve it compacts on every turn: set `reserveTokens` well below the window. A context-overflow error or an early `stopReason: length` selects one compact-and-retry. `retry.enabled/maxRetries` 3, `retry.provider.maxRetries` 0. A model with no `contextWindow` is given 128,000. |
| Oh My Pi | `compaction.enabled`, `compaction.thresholdPercent`, `compaction.thresholdTokens`, `compaction.reserveTokens`, `compaction.keepRecentTokens`, `compaction.modelThresholds`, `compaction.autoContinue` (`config.yml` in the agent dir) | true, -1 (legacy reserve-based), -1, unset, 20000, none, true | percent form `"80%"`, token form `90000`, fixed form `"f90000"`. A model with no `contextWindow` is given 128,000. |
| Hermes | `compression.enabled`, `compression.threshold`, `compression.threshold_tokens`, `compression.target_ratio`, `compression.protect_last_n`, `compression.protect_first_n`, `compression.max_attempts` (`config.yaml`) | true, 0.50 (floored at 0.75 for windows under 512K), null, 0.20, 20, 3, 3 | refuses a window under 64,000. Auxiliary work (compression, titles, vision) goes to the main model unless `auxiliary.*.provider` says otherwise. |
| OpenClaw | `agents.defaults.compaction.enabled`, `.mode` (`default` or `safeguard`), `.keepRecentTokens`, `.recentTurnsPreserve`, `.midTurnPrecheck.enabled`, `.model`; per model `contextWindow` and `contextTokens` (an active-input cap) | true, `safeguard` in new configs, 20000, 3, false | on a context-overflow error it compacts and retries. A summary needs a model call, so a summary model that is the same 16K model is part of the test. |
| Cline | `--compaction agentic\|basic\|off` (CLI flag); `providers.json` `settings.contextWindow` / `maxInputTokens` | `agentic` | the installed binary triggers at 0.9 times the model's `maxInputTokens` or `contextWindow`, and falls back to 128,000 when neither is set. An overflow error triggers one compaction. `--retries` is the consecutive-mistake limit (3). |
| Goose | `GOOSE_AUTO_COMPACT_THRESHOLD`, `GOOSE_AUTO_COMPACT_TOKEN_LIMIT`, `GOOSE_CONTEXT_LIMIT`, `GOOSE_TOOL_CALL_CUTOFF`, `GOOSE_MAX_TURNS` | 0.8, 225000, model default or 128000, computed, 1000 | the token limit is an absolute trigger and the earlier of the two applies, so at a small window only the percentage matters. |
| Aider | `--max-chat-history-tokens`, `--map-tokens`, model metadata `max_input_tokens` | `min(max(max_input_tokens / 16, 1024), 8192)`, `min(max(max_input_tokens / 8, 1024), 4096)` | Aider does not compact on overflow; it summarizes history past the soft limit using a weak model (the same model when none is named) and ends the turn on a context error. |
| Kilo | `compaction.auto`, `compaction.prune`, `compaction.reserved` (`kilo.json`), `limit.context` and `limit.output` per model | true, false, unset | like opencode: with no `limit.context` it never compacts. |

## Network and filesystem behaviour that matters in a sandbox

Measured two ways. (1) The whole capture run inside `bwrap --unshare-net` with no route anywhere: every one of the eight
(and Claude Code, opencode, Codex) still reached the recorder and sent its first request, so none needs the network
beyond the endpoint to start. (2) The same run with a stub resolver that answers NXDOMAIN to everything and logs the names
asked (`dnslog.py`, `bwrap --unshare-user --uid 0 --cap-add CAP_NET_BIND_SERVICE`, `/run/systemd/resolve` masked so
the lookups reach it), one harness at a time. Names looked up, per harness, in a fresh `HOME` with an empty work directory:

| harness | looked up | slower with no network? |
|---|---|---|
| Pi | nothing | no |
| Oh My Pi | `api.commandcode.ai`, `api.kilo.ai`, `api.venice.ai`, `catalog.stencil.so`, `coding-intl.dashscope.aliyuncs.com`, `hyper.charm.land`, `zenmux.ai` (other providers' model catalogs, at startup) | no |
| Hermes | nothing | no |
| OpenClaw | nothing | no (9 s either way; its own startup) |
| Cline | `data.cline.bot`, `otel.cline.bot` (13 lookups: telemetry, retried) | **yes: 9 s instead of 0.9 s** |
| Goose | `models.dev` (its model registry) | no |
| Aider | `pypi.org` (update check), `raw.githubusercontent.com` (model metadata) | no; `--no-check-update` and a metadata file cover both |
| Kilo | `api.kilo.ai`, `models.dev` (6), `us.i.posthog.com` (analytics) | a little (6.5 s instead of 4.6 s) |
| (Claude Code) | `api.anthropic.com` (4) | no |
| (opencode) | `models.opencode.ai`, `registry.npmjs.org` | no |
| (Codex) | `chatgpt.com`, `github.com` | no |

Lookups are attempts; whether a connection would have carried anything was not captured (no packet capture without
root). Names are from the harness version in the table at the top.

- **Writes outside the work directory.** Every harness wrote its config, cache, sessions and logs under the fresh
  `HOME`, nothing in the real one, with these exceptions: Cline's detached daemon (above); one Bun-compiled binary (which was not
  attributed; Cline, Kilo and Oh My Pi run on Bun) left a native `.so` directly under `/tmp`; Aider and Goose write inside the work directory
  (Aider a git repo and `.aider*` files; Goose a session database and an `apps/` directory under its `HOME`).
- **Goose records full LLM requests** under `$HOME/.local/state/goose/logs/` (not an engine-side concern, but a log
  that can hold the whole prompt).
- **The installs**, all under `~/harnesses` with `HOME` redirected at run time: npm packages in `~/harnesses/package.json`
  (Pi, Oh My Pi, OpenClaw, Cline, Kilo exact; the three from 2026-10-12 still caret ranges); OpenClaw needs Node 24 and
  Oh My Pi needs Bun, so `node-v24.21.0-linux-x64` and `bun` are under `~/harnesses`.

## How each was installed, pinned

| harness | command | pin |
|---|---|---|
| Pi | `npm install --prefix ~/harnesses --save-exact @earendil-works/pi-coding-agent@1.2.0` | `@mariozechner/pi-coding-agent` is deprecated in favour of this name |
| Oh My Pi | `npm install --prefix ~/harnesses --save-exact @oh-my-pi/pi-coding-agent@18.10.3`; the `omp` script needs `bun` on `PATH` | Bun 1.4.3, `bun-linux-x64.zip` sha256 `3e730528d89775f03f87f05c06787c4df4099d0a37f3f252e7b829cb23816e18` (matches the release's `SHASUMS256.txt`) |
| OpenClaw | the same command with `openclaw@2026.10.3`, **run with Node 24**: its `preinstall` refuses Node 22 (the box's default), and a failed `npm install` with several packages leaves `package.json` unchanged | Node v24.21.0, `node-v24.21.0-linux-x64.tar.xz` sha256 `fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6` (matches `SHASUMS256.txt`) |
| Cline | `npm install --prefix ~/harnesses --save-exact cline@3.0.71` | a Bun-compiled native binary (`@cline/cli-linux-x64`) |
| Kilo | `npm install --prefix ~/harnesses --save-exact @kilocode/cli@7.8.8` | a Bun-compiled binary; the CLI package's name is `@kilocode/cli`, the command `kilo` |
| Hermes | `uv venv --python 3.14 ~/harnesses/hermes-venv`, `git clone --branch v0.21.7` into `~/harnesses/hermes-src` (commit `71d34bad1045610252eda02ee79549436896e784`), `uv pip install -e ~/harnesses/hermes-src` | a normal `pip install hermes-agent` from PyPI is 0.19.0 and `pip install git+…` fails by design ("Building wheels or sdists for hermes-agent is not supported"); the project's supported installs are its shell installer, Docker and Nix. The npm package `hermes-agent` is an unofficial bridge, not used. System Python 3.14.7 is within its `>=3.11,<3.15`. |
| Goose | `goose-x86_64-unknown-linux-gnu.tar.bz2` from release `v1.54.0` into `~/harnesses/goose/` | sha256 `bf01cb6359f7a2b1acb54ca4827e1d059b18aabb3d2b75202e21066cfd3a3396` (equal to the release's published digest); recorded in `~/harnesses/goose/SHA256` |
| Aider | `uv venv --python 3.12 ~/harnesses/aider-venv`, `uv pip install aider-chat==0.86.2` | `aider-chat` needs Python `<3.13`; the box has 3.14 only, so uv fetched CPython 3.12.15 into `~/harnesses/pythons` (`uv` itself is `~/harnesses/uv-venv`, uv 0.13.0) |

## What was not read

One request a harness and no reply, so nothing about a second turn, compaction, overflow, tool-result handling, or what
each harness does with goinfer's stream; that is I3. No second-session TTFT comparison (I4's prefix-reuse question) was
run: it needs a model. Window sizes in the config above are what the harness was told, not what it assumes by default
(defaults are in the compaction table where the source gave one). Cline's `maxTokens` setting and Goose's
`GOOSE_CONTEXT_LIMIT` were set but their effect on the request was not separately isolated, beyond Cline's `max_tokens`
above.
