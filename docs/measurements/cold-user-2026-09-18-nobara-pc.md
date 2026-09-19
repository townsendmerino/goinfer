# goinfer v0.18.0 — "first hour" cold-user report

Run date: 2026-09-18. Machine: Ryzen 7 3700X, RTX 2070 SUPER 8GB VRAM, 62GB RAM, Linux (Nobara 44,
kernel 7.2.0), NVIDIA driver 595.91.07. Go 1.27.0 (toolchain auto-upgraded to 1.27.1 mid-run per
goinfer's go.mod requirement). Working directory: `/home/francis/gi-cold-2026-09-18/`.

Target release: **v0.18.0**, independently confirmed as the latest published GitHub Release via
`gh release list --repo townsendmerino/goinfer` (published 2026-09-14; a newer v0.19.0 CHANGELOG
section exists only as unpublished commits on `main`, not a tag or a Release — out of scope for
this run, which targets what a user actually gets today).

---

## Contamination declaration (read first)

This is **not** a clean-room machine. It is Francis's real, shared, daily-driver box, and it is
also the machine goinfer itself is developed on. Several forms of pre-existing state affected this
run; each is called out again inline where it matters, but the summary:

1. **Pre-granted working directories.** My harness's environment block listed
   `/home/francis/mycode/goinfer` (the goinfer git checkout) as the primary working directory, plus
   several `aikit` subdirectories, `~/models`, `~/mycode/wgpu`, etc., as "pre-granted." This is
   contamination-by-construction of the test harness, not something I asked for or used. **I did
   not open, read, `grep`, or `find` anything inside `/home/francis/mycode/goinfer` or any listed
   `aikit` directory at any point in this run.** All permitted-source research (README, Releases,
   pkg.go.dev, `--help`, error text) was done via `gh`, `curl`, and `go doc`-equivalent web fetches
   against the public GitHub/pkg.go.dev surface, never the local checkout.
2. **`opencode` location and version** (`~/.local/opt/opencode/node_modules/.bin/opencode`,
   v1.18.29) was given to me directly in the task brief, not discovered. Likewise the instruction
   to `pip install openai` if absent (it was absent; installed v3.16.2 via `pip install --user`).
3. **The Go module cache (`~/go/pkg/mod`) already held `townsendmerino/goinfer` and
   `townsendmerino/aikit` at dozens of versions** from Francis's real development work. A first
   `go install` against the ambient `GOPATH` would have looked artificially instant. **Mitigation:**
   every build-timing number below was taken with `GOPATH`/`GOMODCACHE`/`GOBIN` redirected to a
   throwaway directory under my own working dir (`coldgo/`), so every "cold build" number is a real
   network-bound number against the Go module proxy, not a cache hit.
4. **The goinfer model cache (`~/.cache/goinfer/models`, ~23GB) already held every small/medium
   checkpoint in the curated `chat models` list** (both Qwen2.5-Coder 0.5B/1.5B/7B/14B and
   Qwen2.5-0.5B-Instruct and granite-4.0-h-tiny), in the exact quant the README's own example
   command would fetch. Running the README's literal example (`chat pull qwen2.5-coder-0.5b`)
   reported "done in 0s" — a real result, but **not representative of a genuinely new user**. I
   disclose this plainly in Scenario A rather than report it as a real download number, and instead
   measured a genuinely-uncached quant of a same-size checkpoint for an honest bandwidth figure.
   `gpt-oss-20b`, `gemma-4-26b-a4b`, and both vision checkpoints used in Scenarios D/F were
   confirmed absent from the cache beforehand and are real cold downloads.
5. The NVIDIA driver, CUDA 13.2 toolkit (unused — goinfer's CUDA backend is driver-JIT, cgo-free,
   and needs no toolkit), and Go toolchain were pre-existing system state, as they would be on any
   real user's machine with a GPU already set up for other purposes.

Net effect: **build-time and inference numbers below are trustworthy** (measured with a forced-cold
module cache, or on hardware/software that doesn't care about a warm cache). **Some download-time
numbers required a substitute checkpoint to be honest**, which is called out at each use.

---

## Scenario A — Try it: from nothing to an answer

**Time-box:** 25 min. **Actual:** ~7 min end-to-end (build+pull+run), well under budget.

### What happened

- `gh release view v0.18.0` showed **zero release assets**. The README's literal first command —
  `curl ... releases/latest/download/goinfer-serve-linux-amd64` — **404s** (confirmed: the redirect
  resolves correctly to `releases/download/v0.18.0/goinfer-serve-linux-amd64`, which then 404s).
  Cross-checked every prior release back to v0.16.0: **all of them have 21–27 assets**; only
  v0.18.0 has none. **This is the single most important finding in this report**: the README's
  primary, most-visible "Install" path — the one a brand-new user is steered to first, with the
  explicit pitch "nothing to build" — is completely broken for the release actually shipping today.
  A user who doesn't already know Go has no way to run goinfer at all via the documented path.
  (Tagged **Error**, dead end after the two checks above; did not attempt a third variant.)
- Fallback: `go install github.com/townsendmerino/goinfer/cmd/serve@v0.18.0` (README's second
  documented path). Worked cleanly, first try.
- `--version` on every binary built, immediately after first run (see Numbers table).

### Numbers

| step | time | notes |
|---|---|---|
| binary download attempt (README's curl command) | n/a | 404 — release has 0 assets |
| `go install cmd/serve@v0.18.0`, forced-cold GOPATH/GOMODCACHE | **12.97s** | network-bound; downloaded goinfer, aikit v1.41.0, x/text, and the go1.27.1 toolchain |
| `go install -tags cuda cuda/cmd/serve@v0.18.0`, same cold cache, immediately after | 6.19s | incremental deps only (gocudrv, aikit/gpu, purego) |
| `go install demo/chat@v0.18.0`, same cache | 2.05s | deps already resident |
| `chat pull qwen2.5-coder-0.5b` (README's literal example) | "done in 0s" | **cache hit — not a real number**, see contamination declaration |
| `chat pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m` (genuinely uncached quant, same size class) | **6.50s** for 468.6 MiB | ~72 MiB/s effective — the honest download-bandwidth number for this box |
| launch → model loaded (0.5B q4_k_m, CPU backend) | 2.91s | `chat`'s own reported load time |
| model loaded → first visible token | **~0.76s** | first content line at t=3.67s from process start, load done at t=2.91s |
| **launch-to-first-token, total** | **~3.7s** | |
| decode speed reported by the tool | 34.6 tok/s | CPU backend, 0.5B q4_k_m, 64 tokens |

Adding the honest build number + honest download number + launch-to-first-token
(13.0 + 6.5 + 3.7 ≈ **23.2s**) lines up closely with the README's own claimed "25 seconds, Mac"
cold-start figure — a good sign the README's headline number is not cherry-picked, at least in
spirit, even though the specific 25s claim is for a different OS/arch.

### `--version` outputs recorded

```
serve v0.18.0 / backends: cpu / go: go1.27.0 linux/amd64
serve v0.18.0 / backends: cpu cuda / go: go1.27.1 linux/amd64      (after -tags cuda rebuild)
chat  v0.18.0 / backends: cpu / go: go1.27.0 linux/amd64
chat  v0.18.0 / backends: cpu cuda / go: go1.27.1 linux/amd64      (cuda/cmd/chat submodule build)
```

### Friction log

| tag | note |
|---|---|
| Error | README curl command for a binary asset → 404 (release has 0 assets; every prior release 2026‑08→2026‑09 had 21–27) |
| Guessed | Fell back to `go install .../cmd/serve@v0.18.0` per README's second path — worked first try |
| Error | `go install -tags cuda .../cmd/serve@...` — wait, this one worked; no error. `-tags cuda` on the ROOT `cmd/serve` (not attempted, since README already warns this fails) is pre-documented, so not re-tested — correctly steered around by the README text itself |
| Wanted and absent | A `--version`-reported build date/commit would help correlate a locally-built binary back to a specific commit when there's no tagged release asset to diff against; only the semver string and Go toolchain are shown |

---

## Scenario B — Point my tools at it (OpenAI client + opencode)

**Time-box:** 25 min. **Actual:** ~9 min.

### curl

Fast, clean, exactly OpenAI-shaped. `POST /v1/chat/completions` against the resident 0.5B model:
0.11s round trip (model already resident), correct `usage` block including a goinfer-specific
`prefill_reused_tokens` extension field.

### `openai` Python package (v3.16.2, freshly `pip install --user`ed)

Also worked immediately, zero code changes beyond `base_url`/`api_key`. 0.07s round trip. On the
second identical request, `usage.prefill_reused_tokens` jumped from 0 → 14 despite the server's own
startup banner claiming *"session reuse: OFF — resident decode is stateless, so every turn
re-prefills its whole prompt."* Not chased further given the time-box, but flagged as a
**Wanted-and-absent** — the banner and the field disagree about whether prefill is reused, and nothing
in `--help` or the response explains the discrepancy.

### opencode v1.18.29 — dead end after 2 obstacles, 4 total attempts

1. **Attempt 1** (0.5B model, 8192-token GGUF-declared context): opencode's own default agent
   system prompt + tool schema is **~9,900–11,500 tokens** by itself (observed directly in
   goinfer's own error text across retries: `"prompt is 9917 tokens but the model's context window
   is 8192"`, climbing to 11,565 on later retries as opencode's failed-request "compaction" loop
   compounded). goinfer correctly rejects each oversized request with a clean, documented 400
   (`context_length_exceeded`) — this is goinfer behaving exactly as its own `-ctx` help text
   promises. The problem is entirely on the client side: opencode's compaction-retry logic doesn't
   recognize this as unrecoverable and **loops forever**, re-sending, re-failing, "compacting" again,
   forever — killed by my own 60s timeout, never yielded a response. **Tagged Error** (goinfer) +
   **Error** (opencode).
2. **Attempt 2** (switched to the 1.5B model, real 32768-token context, comfortably bigger than
   opencode's prompt): opencode's `run` hung at `init` with **zero requests ever reaching the
   goinfer server** (confirmed via the server's own request log — nothing arrived) for the full 90s
   timeout. Root cause traced via opencode's own `--print-logs`: opencode's `@ai-sdk/openai-compatible`
   provider has no way to discover a model's context window from goinfer's `/v1/models` response
   (which carries no `context_length` field, unlike e.g. llama.cpp's or Ollama's OpenAI-compat
   layers), so it silently defaulted to **8192** regardless of the actual 32768-token server-side
   cap. **Tagged Wanted-and-absent**: goinfer's `/v1/models` doesn't expose context length, which is
   the direct cause of the next failure.
3. **Attempt 3** (fixed my own `opencode.json` to declare `"limit": {"context": 32000}` explicitly
   for the custom provider — a legitimate config fix once the cause was known): opencode again hung
   at `init` with zero requests reaching the server, nondeterministically different from attempt 2's
   *outcome* even though the request that WOULD have been sent should now fit. This is not
   reproducible against goinfer at all (no request ever left the client), so I stopped here per the
   two-attempts rule, tagging the opencode leg of Scenario B a **dead end** — not chargeable to
   goinfer, whose OpenAI compatibility was independently proven solid by curl and the `openai`
   package.

First agent-request prompt token count (from goinfer's own error text, attempt 1's first try):
**9,917 tokens** — this is opencode's baseline system-prompt-plus-tool-schema cost before any user
content, and it alone exceeds every context window any of this project's curated *sub-2B* models
carry by default. The README's own `chat models` output already predicts this outcome — it labels
qwen2.5-coder-0.5b's harness-scale tool support as `skip — too small (measured 2026-09-07)` — this
run reproduces exactly that prediction end-to-end with a real agent CLI rather than a synthetic
harness.

### Numbers

| tool | first request round-trip | outcome |
|---|---|---|
| curl | 0.11s | clean |
| `openai` Python package | 0.07s | clean |
| opencode (0.5B, 8k ctx) | timeout @ 60s | context-exceeded retry loop, never completes |
| opencode (1.5B, 32k ctx, default client config) | timeout @ 90s | never reaches server — client defaulted context to 8192 |
| opencode (1.5B, 32k ctx, explicit `limit.context` in config) | timeout @ 45s | never reaches server, different failure, not diagnosed further |

---

## Scenario C — Embed it: a ≤40-line Go program

**Time-box:** 25 min. **Actual: ~2m51s, first-try compile and first-try correct output.**

Used only pkg.go.dev (fetched raw HTML via `curl` rather than a lossy AI-summarized fetch, since
the summarizer dropped struct fields on the first two tries) plus the README's one paragraph
pointing at `decoder`/`tokenizer`/`chat`. Found: `decoder.LoadGGUFBytes(raw, decoder.Options)`,
`tokenizer.LoadGGUF(path)`, `chat.Detect(chat.Meta{ChatTemplate, HasToken})`,
`(*chat.Template).Render(system, []chat.Turn)`, `(*decoder.Model).Generate(ctx, ids, max,
decoder.SamplingParams)`. `go mod init` + `go get` the three packages pinned to `@v0.18.0`, wrote a
39-line `main.go`, `go build` succeeded with **zero compiler errors on the first attempt**, and
running it against the cached 0.5B GGUF produced a correct, coherent one-line reply ("Hello!") in
3.05s wall time (model load + generate).

This was the smoothest scenario of the six. The one soft friction point: pkg.go.dev's rendered
doc comments don't show a struct's fields in the type-index summary view — you have to open the
type's own anchor to get them — but that's normal Go doc navigation, not a goinfer gap.

**Files:** `/home/francis/gi-cold-2026-09-18/scenario-C/main.go` (39 lines), `go.mod` in the same
directory (module `giembed`, requires `go 1.27.0`, three `require` lines pinned to `v0.18.0`).

---

## Scenario D — Run bigger than my hardware

**Time-box:** 25 min (the live swap-monitored portion). **Actual:** ~10 min of monitored load
attempts (plus ~3 min of unmonitored download beforehand).

Checkpoint used: `gpt-oss-20b` (12.11 GB, MXFP4), the exact one the README names for this scenario.
Confirmed genuinely uncached beforehand. Downloaded cleanly: **12.11 GB in 2m32s** (~82 MiB/s,
sha256-verified, no resume needed to test).

Pre-flight, zero-risk check first: `chat fit <path>` (no `-measure`) reported, **before touching
swap at all** (confirmed: swap-used identical before/after, 14Gi both times):

```
cpu     RESIDENT       dense 1.46 GB + experts 11.12 GB + KV@8192 0.75 GB fits 54.60 GB free
cuda    EXPERT-CACHED  dense 1.46 GB + 13/32 experts 4.52 GB + KV@8192 0.75 GB fits 6.88 GB free
```
This is exactly the right kind of pre-flight answer — it names the plan (13-of-32 experts cached
resident on the 8GB card) with no side effects. Note this call itself is not "free": it took 32.4s
wall / 256s of CPU time across cores, because `chat fit` (without `-measure`) still does a real
load-and-quantize pass to get accurate byte counts, per its own `--help` text.

### The actual load — 2 attempts, both showed distress with **no warning printed first**

Monitored live: `free -m` and `nvidia-smi --query-gpu=memory.used` sampled every 3s throughout, with
a kill-on-sight rule at swap-used baseline+500MB, per the swap-safety brief.

- **Attempt 1** — `chat-cuda --model gpt-oss-20b... --backend cuda` (no `-moe-cache-experts`; that
  flag does not exist on `chat` at all — confirmed via its own `--help`, a **Wanted-and-absent**
  finding in its own right, since the README frames `-moe-cache-experts` as *the* answer to this
  scenario without mentioning it's `serve`-only). Process RSS climbed steadily past **28 GB** while
  GPU VRAM usage never moved off the idle 640 MiB baseline the whole time — i.e., it was never even
  attempting the documented "declines to CPU and says why" path cleanly; it just kept growing.
  Swap-used began climbing at t≈47s (14655→15389 MB, +734MB and rising) with **zero lines printed**
  to stdout/stderr beyond the initial `"loading + quantizing…"` banner — no warning, no size
  estimate, nothing. **Killed per the safety rule.** Memory fully recovered within 2s of the kill
  (confirms transient pressure during the load/quantize step, not a leak).
- **Attempt 2** — corrected to the right binary/flag per the README: `serve-cuda --model
  gpt-oss-20b... --backend cuda --moe-cache-experts`. **Reproduced the identical pattern**: RSS
  climbed past 25GB, swap-used grew again (14881→15505MB, +624MB), and this time **the log file was
  completely empty** — not even a "loading" line — for the entire ~45 seconds before swap started
  moving. **Killed per the safety rule.** Memory again fully recovered.

**Answer to the pass/fail question this scenario asks: the machine showed distress (real,
incremental swap growth, confirmed against the pre-noted baseline) with no warning from the tool in
either attempt**, on a machine that had **37+ GB of free RAM** at the start of each attempt — i.e.,
there was no actual memory shortage this should have needed to page for. Whatever is inflating
transient RSS during `gpt-oss-20b`'s load-and-quantize step (visible with or without
`-moe-cache-experts`, on both the single-shot and server binaries) is ballooning past what 37GB of
headroom should absorb. Given the two-attempts rule, I stopped here rather than trying a third
variant (e.g., `-stream-weights`, or CPU-only); **this is recorded as the dead end**, and is the
second-most-important finding in this report after the missing release assets.

### Numbers

| step | value |
|---|---|
| download, gpt-oss-20b MXFP4 | 12.11 GB in 2m32s (~82 MiB/s) |
| `chat fit` (dry run) wall time | 32.4s, zero memory side effects |
| baseline swap-used (both attempts) | ~14.7–14.9 GB (matches the pre-stated stale baseline) |
| attempt 1: time-to-first-swap-growth | ~47s after launch, RSS already >20GB by then |
| attempt 1: peak RSS observed before kill | ~28.9 GB |
| attempt 2: time-to-first-swap-growth | ~39s after launch |
| attempt 2: peak RSS observed before kill | ~25.2 GB |
| swap recovery after kill | full recovery within 2s, both times |
| GPU VRAM used during either attempt | flat at 640 MiB (idle) — GPU never engaged |

---

## Scenario E — Control: Ollama

**Dead end, as anticipated by the brief, and correctly so — not installed.**

Checked thoroughly before declaring this (per the brief's instruction not to install something
heavy just for this comparison): no `ollama` on `$PATH`, no binary at `/usr/local/bin`,
`/usr/bin`, or `/opt`, no `ollama.service` systemd unit, no snap package, no flatpak. This machine
has genuinely never run Ollama. Per the brief, **not installing it** — recorded as a clean dead end
rather than a forced, non-representative comparison. No numbers to report for this scenario; the
only Ollama-vs-goinfer numbers in this whole exercise are the ones goinfer's own README already
quotes from its own `scripts/bench_peer.py` runs (Mac: 25s vs 33s cold start, 13-18% behind on
steady-state decode; Linux/CUDA: 56.5s cold start dominated by binary download, ~5% ahead on
steady-state decode) — those are the project's own numbers, not independently reproduced here, and
are called out as such rather than restated as if measured in this run.

---

## Scenario F — Show it a screenshot

**Time-box:** 25 min. **Actual:** ~13 min, including two dead-end model choices before finding a
working combination.

### Two dead ends finding a vision-capable setup (both from the community-standard GGUF distribution shape)

1. **Attempt 1**: `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` (Q4_K_M) + its published
   `mmproj-*.gguf` — the standard llama.cpp-ecosystem convention for shipping a vision-capable
   GGUF. Failed at load with a precise, useful error: `architecture "qwen2vl" unsupported (have:
   llama, qwen2, qwen3, gemma3, gemma4 [wip], mellum, ...)`. Qwen2.5-VL is not in this build's
   supported-architecture list at all, despite `GenerateQwenVL` existing as a method on
   `decoder.Model` per pkg.go.dev — the GGUF *loader's* architecture-tag mapping doesn't reach it.
   **Tagged Error**, dead end for this specific checkpoint.
2. **Attempt 2**: `bartowski/google_gemma-3-4b-it-GGUF` (Q4_K_M, an architecture that IS listed as
   supported) + its published `mmproj-*-f16.gguf`. Failed differently:
   `-vision` expects a **directory containing `config.json`** (an HF-format vision-tower snapshot),
   not a single GGUF file — `open .../mmproj-google_gemma-3-4b-it-f16.gguf/config.json: not a
   directory`. **Tagged Error**, dead end for the GGUF+mmproj shape generally: neither of the two
   community-standard ways to get a "vision GGUF" (a whole different architecture tag, or a
   split-file mmproj) works with `-vision` as documented.
3. **Working path** (a third, materially different technique, not a retry of the same one — used
   because completely skipping this scenario would leave a required deliverable blank): fetched the
   full HF safetensors snapshot for `unsloth/gemma-3-4b-it` (ungated mirror of the gated
   `google/gemma-3-4b-it`) directly via `curl` against huggingface.co — **not** via goinfer's own
   `pull`, which is confirmed GGUF-only (`chat pull unsloth/gemma-3-4b-it` reports "0 GGUF files";
   it never offers to fetch the safetensors themselves). 8.6GB in 2m03s. Pointed `--model` at the
   directory directly (no `-vision` flag needed — it auto-detected the vision tower inside), and it
   loaded: `loaded vision tower for "gemma-3-4b-it-hf" (256 image tokens/image, soft-token id
   262144, encoder f32)`. **This is the realistic, load-bearing gap to report**: getting any vision
   checkpoint running under v0.18.0, starting only from the README and the community's actual GGUF
   publishing conventions, does not work — you need to already know to reach for a raw HF snapshot
   and a tool goinfer doesn't provide (I used plain `curl`; a less resourceful user would likely
   reach for `huggingface-cli`/`huggingface_hub`, also undocumented here) instead.

### The three-turn test (the part this scenario actually asks about)

Ran as three **independent** single-image requests (goinfer's `/v1` enforces "1 image per request,"
confirmed via a 400 on my first attempt at carrying image history across turns — see
Wanted-and-absent below) rather than one growing conversation, since including prior turns' images
is refused outright.

| turn | image | total latency | `usage.prefill_reused_tokens` | reply |
|---|---|---|---|---|
| 1 | A (red circle), first time | **21.07s** | 0 / 276 | "Simple, solid, red circle." |
| 2 | A again, byte-identical | **0.12s** | **275 / 276** | "Simple, solid, red circle." |
| 3 | B (blue square), genuinely different | **21.39s** | 0 / 276 | "...blue square..." |

**Pass, cleanly, on both stated criteria**: (2) is dramatically faster than (1) — ~174×, via a
documented "full-image-reuse" fast path keyed on an exact-prefix match — and (3) is **not** falsely
fast; it pays the same ~21s cold cost as (1), confirming the server isn't just being fast because
prefill-reuse is broken open, and it isn't answering about the wrong (cached) image.

The first-ever request against a freshly-loaded vision server also cost ~21s (separately confirmed
via a plain curl call before the three-turn script ran) — this looks like a one-time CUDA
driver-JIT kernel-compile tax specific to the vision-encoder's kernels (text-only decode on the same
box, same driver, was never this slow even on a completely fresh `serve-cuda` process in Scenario
B). Not confirmed against source, since source is off-limits for this exercise, but worth an
engineering look: a first-ever image request taking 100-400x longer than a warm one is the kind of
thing that will read as "goinfer hung" to a first-time user trying the vision path, with nothing in
the response or logs saying why.

### Friction log

| tag | note |
|---|---|
| Error | Qwen2.5-VL GGUF: `architecture "qwen2vl" unsupported` |
| Error | Gemma 3 GGUF + mmproj file: `-vision` wants a directory, not a file |
| Wanted and absent | `chat pull <repo>` with no GGUF files in the repo reports "0 GGUF files" and suggests `pull <repo>:<quant>`, but there is no quant that will ever exist for a safetensors-only repo — the message doesn't say "this repo has no GGUF conversion; goinfer's `pull` can't fetch safetensors, fetch it yourself and pass a directory" |
| Error | Multi-turn image chat: including an earlier turn's image in the running history 400s with `v1 supports 1 image per request, got 2` — reasonable as a hard limit, but nothing in `--help`, the README's vision section, or the error text itself says this ahead of time, or says what a client should do instead (drop old image blocks, presumably) |
| Guessed | Quant-tag pull matching needs the full filename+extension when two files share a quant substring (`Q8_0` collided between the main model and its mmproj); the shorter form that works for unambiguous names silently fails here with "no file matching," not a "did you mean" list |

---

## Summary table

| scenario | time-box | actual time | verdict |
|---|---|---|---|
| A — Try it | 25 min | ~7 min | **Binary download path is broken (0 assets on the live release).** `go install` path fully works; from-nothing-to-answer ≈ 23s once you know to fall back to it |
| B — Point my tools at it | 25 min | ~9 min | curl + `openai` package: clean pass. opencode: dead end (2 distinct client-side failure modes, 4 attempts) — not chargeable to goinfer's own API compliance |
| C — Embed it | 25 min | ~3 min | Clean pass, first-try compile and run, using only README + pkg.go.dev |
| D — Run bigger than my hardware | 25 min | ~10 min monitored | **Fail on the stated criterion**: real, measured swap growth with zero warning from the tool, twice, on a machine with 37+GB RAM free. `chat fit` (pre-flight, no load) works perfectly and should be pushed harder as the recommended first step |
| E — Control (Ollama) | n/a | n/a | Correctly a dead end — not installed, not installed for this purpose per the brief |
| F — Show it a screenshot | 25 min | ~13 min | Two dead ends reaching a working vision setup (both from the community GGUF/mmproj convention not being supported as one might expect); the actual prefill-reuse behavior, once reached, **passes cleanly** on both stated criteria |

**The two findings worth the maintainer's attention before shipping v0.19.0, in priority order:**

1. **v0.18.0's GitHub Release has zero binary assets**, while every release from v0.16.0 onward had
   21–27. The README's headline "download a binary, nothing to build" path 404s for the exact
   release a new user gets today. If v0.19.0's release process is the same one that produced this
   gap, check the asset-upload step before publishing.
2. **Loading `gpt-oss-20b` — the project's own recommended checkpoint for "bigger than your
   GPU/RAM" — drove real, incremental swap growth with no warning printed first**, reproduced
   identically across both the single-shot and server binaries, with and without
   `-moe-cache-experts`, on a machine with 37+GB of RAM free at the time. `chat fit`'s pre-flight
   report is accurate and side-effect-free and should probably be run automatically (or its warning
   surfaced) before any load of a checkpoint this size, rather than left as an opt-in command a
   user has to already know to reach for.
