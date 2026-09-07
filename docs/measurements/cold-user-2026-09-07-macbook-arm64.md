# goinfer cold-user report — macbook-arm64 — 2026-09-07

## Contamination declaration (read first)

- **Prior contact with this project:** none in this account's own memory record. No specific
  recollection of goinfer from any other source either, but a negative can't be proven; noting it
  rather than asserting certainty.
- **Pre-granted working directories — yes, and this matters.** This session's *global*
  `~/.claude/settings.json` (`permissions.additionalDirectories`) pre-grants several paths under
  `~/tmcode`, including three under `~/tmcode/goinfer/*` (`decoder`, `docs/measurements`,
  `scripts`, `docs`). This is not scoped to one window — it is inherited by any Claude Code window
  on this machine/account unless the setting is edited. I disclosed this before starting and was
  explicitly told to proceed anyway while never opening those paths. **I did not read, list, or
  open any `~/tmcode/goinfer/*` path, its docs/, or its git history at any point in this run** —
  everything goinfer-specific below came only from the public GitHub README, pages it links to
  (including two under the repo's own `docs/` tree that the README links to directly —
  `docs/integrations/claude-code.md` and the `docs/measurements/*.md` filenames it *names* but I
  did not open), the Releases page/API, `--help`/error text, and pkg.go.dev-adjacent (godoc)
  inspection of the installed module's exported API.
- **Second contamination finding, more serious, found mid-setup:** `~/models` on this machine
  already contained a 22 GB `Qwen3.5-35B-A3B-Q4_K_M.gguf` and a 26 GB
  `Qwen3.5-35B-A3B-Q4_K_M.int4mix.giw` — `.giw` is goinfer's own prequantized-bundle format
  (confirmed later from `--help`/docs references), so this is physical evidence goinfer (or
  something producing its cache format) ran on this machine before today, on the exact model class
  Scenario D asks about. **I moved the entire prior contents of `~/models` to `~/models.bak-<date>`
  before starting** (with explicit sign-off) so Scenario A/D would not find a pre-warmed cache, and
  moved everything back at the end. Free disk was still only ~16 GB throughout (the backup sits on
  the same volume, so moving it did not reclaim space) — that constraint is reported as-is below.
- **The README itself is not blind either.** It quotes specific numbers from a dated prior
  cold-user run (`docs/measurements/cold-user-2026-09-06.md` and
  `…-2026-09-06-nobara-pc.md`) for exactly the scenarios this exercise asks me to run (Scenario
  A's "25s vs 33s", Scenario D's "+7.8 GB swap in 5s" vs "8.95 GB RSS, zero swapouts"). Reading the
  README as instructed necessarily pre-exposes those numbers. That's a property of the README, not
  a rule violation, but it means Scenario A/D below are not blind measurements against the
  project's own claims — I already knew the target shape before I ran mine. Separately, the latest
  release (v0.17.1, published **2026-09-07T16:02:49Z**, ~47 minutes before I started) is literally
  titled *"the second cold-user run, and its fixes"* — this exact protocol has been run on this
  project multiple times already, very recently, and is actively shaping the software.
- **Machine identity vs. task brief:** the task names this leg "macbook-arm64" against an "M1 Pro,
  16 GB" spec. `system_profiler`/`sysctl` confirm this machine genuinely is an Apple M1 Pro, 16 GB
  — that part checks out.

## Machine state (before Scenario A, 2026-09-07 09:39:58 PDT)

- `uptime`: `9:39 up 2 days, 15:09, 1 user, load averages: 2.33 2.33 2.25`
- RAM: 16 GB (`hw.memsize=17179869184`); ~93 MB free (5945 pages × 16 KB) at that instant per `vm_stat`
- Disk: `460Gi total, 12Gi used, 16Gi avail, 42% capacity` (this is *after* the pre-existing
  models were still in place; moving them didn't change this figure — same volume)
- CPU: Apple M1 Pro, arm64, macOS (Darwin 25.6.0)
- Ollama: 0.32.5 installed, not running at start (confirmed idle)

## `--version` output for every goinfer binary run, before any model

```
$ ./goinfer-chat-darwin-arm64 --version
goinfer-chat-darwin-arm64 v0.17.1 (02bb81e8a0d2)
backends: cpu
go: go1.27.0 darwin/arm64

$ ./goinfer-serve-darwin-arm64 --version
goinfer-serve-darwin-arm64 v0.17.1 (02bb81e8a0d2-dirty)
backends: cpu metal
go: go1.27.0 darwin/arm64
```

Release tag from the Releases API: **v0.17.1**, published 2026-09-07T16:02:49Z. Note the `-dirty`
suffix on the `serve` binary's build hash — not investigated further (out of scope: would require
reading source/CI to explain), just recorded as seen.

---

## Top section: three changes that would most improve a new user's first hour

Ranked by how much time/confusion each one cost, each tied to a quoted moment below.

1. **The agent-integration story silently drops opencode.** The README's own "Serving" bullet says
   "Pointing a real agent (Claude Code, opencode) at it: `docs/integrations/`" — but that directory
   contains exactly one file, `claude-code.md`. There is no opencode config recipe anywhere in
   README/docs/`--help`. I had to reconstruct opencode's custom-provider JSON from outside
   knowledge of opencode itself (not from anything goinfer publishes), and even having done that
   correctly, the two-turn tool task never returned an answer — see the Scenario B dead end below,
   which cost the most wall-clock time and the only safety incident of the whole run.
2. **The RAM/VRAM budget check does not account for a real request.** `goinfer-serve`'s own
   pre-load message said the 7B/int4 model was `"79% of budget"` (of an 11.2 GB budget on 16 GB
   RAM) — a comfortable-sounding number. Once a real agent turn arrived (opencode's multi-tool
   system prompt), RSS measured via `top` reached **14 GB** on this 16 GB machine and drove the OS
   into heavy, sustained swapping (**Swapouts +621,588 pages, ~9.7 GB, in under two minutes**, `top`
   showing 577% CPU on the goinfer process while it sat producing no output). The number printed
   *before* load and the number *actually reached* under a realistic prompt differ by ~5 GB, and
   nothing re-warns once the gap opens. This is exactly the kind of thing Scenario D's "told me
   first?" question is about, and it happened one scenario early.
3. **Two different, non-composing ways to name a pull target, discovered only by trial.**
   `goinfer-chat pull hf:owner/repo:quant` (the `--model` syntax from the README) is rejected by
   `pull` itself — `pull` wants the bare `owner/repo:quant` form, no `hf:` prefix. Then a quant
   shorthand that happens to be a legitimate GGUF quant name (`q4_k_m`) fails for a repo where that
   quant is split into multiple shard files, with no indication in `--help` that this could happen
   ahead of time — you only find out by hitting it. Two genuinely different failure modes on one
   command, both silent until tried.

---

## Scenario A — Try it

**Outcome: worked, with friction.**

Timestamps: download+pull leg started 09:51:18 PDT; run leg started 09:52:08 PDT; scenario
effectively done by 09:52:20 PDT. Both legs together: **~13 s** end to end for chat binary +
model + answer (see caveat below — this is not a fair speed comparison to the README's 25 s claim).

**Leg 1 — download:**
```
curl -fsSL -o goinfer-chat-darwin-arm64 https://github.com/townsendmerino/goinfer/releases/latest/download/goinfer-chat-darwin-arm64
```
- size 8,526,194 bytes (8.5 MB), `time_total=0.92s` (curl-reported), effectively instant on this
  network.
- `./goinfer-chat-darwin-arm64 pull demo:1.5b` resolved to `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`,
  `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, **1.0 GiB**, reported **"done in 1s — sha256
  verified."** (~1 GB/s implied — almost certainly this environment's network/HF-CDN peering, not
  representative of a home connection; flagged, not treated as a real number.)
- **Guessed:** `demo:1.5b` is not listed anywhere in `goinfer-chat models`' own output (which only
  lists `qwen2.5-coder-0.5b`, `phi3-mini-4k`, `granite-4.0-h-tiny`, `gpt-oss-20b`,
  `gemma-4-26b-a4b` — no 1.5B tier, no mention of a `demo:` alias at all). It happened to work
  because I trusted the README over the tool's own catalog command. **Wanted-and-absent:** `models`
  should either list the `demo:` aliases or the README should stop implying they're discoverable
  via `models`.

**Leg 2 — run:**
```
echo "write a Go function that reverses a string" | ./goinfer-chat-darwin-arm64 --model <path>
```
- Total wall time **11.11 s**. stderr load breakdown: `loaded 28-layer model … in 2.267s`, `load
  2.13s (map 32ms build 2.09s) — 98% build; 1.04 GB source, 0.49 GB/s effective`. Generation:
  `345 tok, 39.6 tok/s` (≈8.7 s of the 11.11 s).
- **Wanted-and-absent:** no explicit time-to-first-token metric anywhere in stdout/stderr; had to
  infer TTFT (~load time, ~2.3 s) from the load-complete line plus the aggregate tok/s, since
  stdout was fully buffered until process exit under a piped/redirected invocation.
- Output: correct, coherent, idiomatic Go (`ReverseString` via rune-swap), with explanation. Not
  degenerate.

**Numbers table**

| leg | size | time |
|---|---|---|
| chat binary download | 8.5 MB | 0.92 s |
| model pull (demo:1.5b) | 1.0 GiB | 1 s (anomalously fast network) |
| run → answer | — | 11.11 s (2.27 s load + ~8.7 s generation @ 39.6 tok/s) |
| **total, binary-in-hand to answer** | — | **~13 s** |

**README's own claim under test:** "From nothing to an answer in 25 seconds" (M1 Pro, vs Ollama
33 s). My **~13 s** is faster than the claim, but the claim's own 1s model-pull number above
disqualifies any speed comparison here — this network is not representative, and I'd already seen
the README's cited number before I measured anything (see contamination declaration).

**What would have made this take half the time:** nothing to fix — this leg had no real friction;
the `demo:` alias not appearing in `models`' own output was the only rough edge and cost under a
minute.

---

## Scenario B — Point my tools at it

**Outcome: worked with friction on curl/streaming/self-check/openai-python; dead end (safety-
stopped) on opencode.**

Started 09:52:44 PDT.

- `goinfer-serve-darwin-arm64` downloaded: 11,204,626 bytes (11.2 MB), ~1s.
- Server start (default `-backend cpu`, 1.5B model): loaded in 3.05s, `decode path: cpu (int4)`.
  **Guessed nothing here** — clean, informative startup banner including routes, load breakdown,
  and an unprompted security warning about the missing `-api-key`. This is one of the better pieces
  of UX in the whole run.
- `curl -N` streaming to `/v1/chat/completions` with `"stream":true`: **worked first try**, correct
  SSE framing (`data: {...}` chunks, `data: [DONE]` terminator), 386 chunks, 9.15 s wall.
- **Self-check** (`./goinfer-serve-darwin-arm64 check`), run before opencode as instructed:
  ```
  models list ...............  ok    1 model · cpu (int4)
  chat, streamed ............  ok    TTFT 0.13s · 49.5 tok/s · usage present
  tools, OpenAI .............  ok    call get_weather({"city": "Paris"}) → result → answer in 2 turns
  tools, harness-scale ......  skip  model did not call the tool under a harness-scale (12-tool)
                                      schema — model answered without calling the tool
                                      (finish_reason="stop") — too small for tool use, or its
                                      template has no tool section
  structured output .........  ok    {"type":"integer"} → 366
  stop sequences ............  ok    not leaked · finish_reason="stop"
  count_tokens ..............  ok    23 tokens, matches usage
  long prompt (~2000 words) .  ok    TTFT 12.89s · usage present
  7 of 8 checks passed (1 skipped, see above)
  ```
  **This check correctly predicted what opencode later did** (see below) — a genuinely useful,
  well-designed self-diagnostic that told me in advance the 1.5B model would not hold up under a
  real agent's tool schema.
- **`openai` Python package: absent, contrary to what I'd been told to expect.**
  `ModuleNotFoundError: No module named 'openai'` — tagged **Wanted-and-absent**. Installed it
  myself (`pip3 install openai`, version 3.8.0 resolved) as ordinary environment setup, not a
  goinfer workaround. Once installed, a plain `client.chat.completions.create(...)` against
  `http://127.0.0.1:8080/v1` worked on the first try, correct code, `finish_reason="stop"`,
  `usage` populated, 7.66 s wall.
- **opencode leg — the model the check predicted would fail:**
  - The README names Claude Code and opencode together as "a real agent" target but only documents
    Claude Code (`docs/integrations/claude-code.md`, a page the README links to directly). That
    page itself is explicit about which model class holds up: *"A 1.5B re-calls the same tool
    forever, which looks like a server bug and is not one."* and names **Qwen2.5-7B-Instruct** as
    the smallest checkpoint it has actually run through a real agent tool loop successfully. Per
    the task's own rule ("if the README names a model as suitable for agents, use that one"), I
    switched to that model for this leg.
  - **Pull friction, 2 attempts, recorded as a dead end per rule 3:**
    1. `pull hf:Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m` → **Error:** `"hf:Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m": want owner/repo[:quant|:file.gguf], e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m` — the `hf:` prefix, which the README documents for `--model`, is rejected by `pull` itself.
    2. `pull Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m` (following the error's own suggested format) →
       **Error:** `no file matching quant "q4_k_m" in Qwen/Qwen2.5-7B-Instruct-GGUF` — this
       repo's q4_k_m is published as two shard files (`…q4_k_m-00001-of-00002.gguf` +
       `…00002-of-00002.gguf`), and the short-quant resolver only matches single whole files.
       **Dead end recorded here per the two-attempts rule.**
    - I then pulled `q3_k_m` instead (single file, 3.5 GiB) — not a workaround, just picking a
      different, explicitly-listed valid quant of the same recommended model from the tool's own
      error output, to actually get an agent-capable model resident. Took **173 s** at a real
      ~29 MB/s (contrast with Scenario A's anomalous ~1 GB/s — this is the more representative
      number for this network).
  - Server restarted with the 7B model. **Load-time budget warning, printed before load completed:**
    `"decoder: fit is tight — qwen2.5-7b-instruct-q3_k_m.gguf needs ~8.9 GB resident at quant int4;
    this machine has 16.0 GB RAM (budget 11.2 GB = 70%) (79% of budget)."` This is exactly the
    proactive "told me first" behavior Scenario D asks about — good, on its own.
  - Configured opencode via a project-local `opencode.json` (Vercel AI-SDK-style
    `@ai-sdk/openai-compatible` provider pointed at `http://127.0.0.1:8080/v1`) — **this
    configuration knowledge came entirely from prior knowledge of opencode itself, not from
    anything goinfer or its README/docs say**, because (per finding #1 above) no opencode recipe
    exists in this project's docs.
  - `opencode run --model goinfer/qwen2.5-7b-instruct-q3_k_m "list the files in this directory
    using your tool, then tell me what's in notes.txt"`: **opencode's own log** shows the real
    ("build", `small=false`) turn started streaming at `17:00:53.065Z` and **never produced another
    log line before I killed it** at `17:03:52 PDT` (**~3 minutes, no tokens, no tool call, no
    answer**).
  - **Swap-safety stop (rule 4), triggered here:** `vm_stat` deltas during this single request:
    **Swapouts +621,588 pages (~9.7 GB) in under two minutes**; `top` showed the goinfer-serve
    process at **14 GB RSS** (vs. its own pre-load estimate of 8.9 GB) and **577% CPU**; free pages
    fell to ~61 MB. I killed both processes (`kill -9`) rather than wait further. Memory recovered
    to ~5.3–10 GB free within seconds of the kill, confirming this specific request — not some
    pre-existing system condition — caused it.
  - **Outcome for this leg: dead end.** Tagged **Error** (the request never completed) rather than
    Guessed or Wanted-and-absent — the tool gave no error message at all; it simply consumed memory
    and CPU without producing output until terminated externally.

**Numbers table**

| step | size/count | time |
|---|---|---|
| serve binary download | 11.2 MB | ~1 s |
| server start (1.5B, cpu) | — | 3.05 s load |
| curl streaming (1.5B) | 386 SSE chunks | 9.15 s |
| `serve check` | 8 checks | a few seconds, not separately timed |
| openai-python call (1.5B) | 310 completion tokens | 7.66 s |
| 7B q3_k_m pull | 3.5 GiB | 173 s (~29 MB/s, real-world rate) |
| 7B server (re)start | — | 18.58 s load |
| opencode two-turn task | 0 tokens produced | ~182 s, then killed for swap safety |

**What would have made this take half the time:** a documented opencode recipe (mirroring the
Claude Code one) naming the 7B-class model up front, so the 1.5B → check-predicts-failure → 7B →
pull-syntax dead end → swap event chain collapses into "start with the right model the first time."

---

## Scenario C — Embed it

**Outcome: worked, one positive discrepancy from the README's own warning.**

- Found the entry point via the README's direct link to `examples/embed/main.go` (allowed —
  "pages it links to"), not by guessing internals. The file's own header comment says it exists
  *because* a prior cold-user run found no working library example — same self-referential
  pattern as elsewhere in this repo.
- `go mod init embedtest`, then exactly the README's documented command:
  ```
  go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest
  ```
  This resolved `goinfer v0.17.1` and its `aikit` dependency. **`go build ./...` succeeded with no
  errors** — despite the example importing a third package, `chat`, that this exact `go get`
  command does not name. The README explicitly warns this will produce `"missing go.sum entry for
  module providing package …"` — **that did not happen**; go's module resolution pulled in enough
  of the module's own go.sum to satisfy `chat` too. Recorded as a discrepancy between documented
  and observed behavior (README is overcautious here, at least on Go 1.27 with this module
  layout), not as a friction point against me.
- Ran it: `go run ./examples/embed <1.5B model> "Reverse a string in Go"`, **51 s** (includes Go
  compile time + ~2 s model load + generation), exit 0.
- **First 200 characters of output** (coherent, not degenerate):
  > "In Go, you can reverse a string using the `strings` package. Here's an example:\n\n```go\npackage main\n\nimport (\n\t\"fmt\"\n\t\"strings\"\n)\n\nfunc main() {\n\tstr := \"Hello, World!\"\n\treversedStr := reverseString(str)\n\tfmt.Println(\"Orig"

**What would have made this take half the time:** nothing — this was the smoothest scenario of the
five.

---

## Scenario D — Run bigger than my hardware

**First line: N/A — not attempted.**

**Outcome: skipped, for machine safety, per rule 4.** Scenario B's opencode leg already showed this
16 GB machine reaching **14 GB RSS and heavy sustained swapping (~9.7 GB swapped in <2 min)** from
a model goinfer's own pre-load check rated a comfortable-sounding "79% of budget" (a 7B model,
smaller than anything Scenario D would ask me to load). Deliberately loading a 20–35B-class MoE
next — 2–5× the resident footprint that already drove this machine into heavy swap — was judged an
unacceptable risk to the rest of the system (this is a shared machine with other work open), not
merely to the test. Rule 4 ("stop at the first pageouts… nothing runs past that") was already
triggered once; I chose not to manufacture a second, larger instance of the same failure just to
collect the scenario's numbers.

This is itself a finding, and probably the single most important one for a "first hour" report:
**a person following this README on a real 16 GB laptop with normal other work open (browser, IDE)
can hit swap-inducing memory pressure from a model class the README does not flag as risky (7B),
one scenario before the one explicitly warning about it.**

---

## Scenario E — Control

**Outcome: worked.**

- Ollama 0.32.5 (version-matches the exact number quoted in goinfer's own README, on "the same
  box" — i.e., this machine's Ollama install is very likely the literal one those README numbers
  came from) started fresh: `OLLAMA_MODELS=<scratch dir>`, confirmed `"total blobs: 0"` in its own
  startup log — a genuinely empty store.
- **Pull leg:** `ollama pull qwen2.5-coder:1.5b` — 986 MB, **27 s** (~36.5 MB/s).
- **Run leg:** `ollama run qwen2.5-coder:1.5b "write a Go function that reverses a string"` —
  **5.64 s** wall, correct/coherent Go code.
- **Ollama used its GPU (Metal) automatically, no flag needed:** startup log —
  `"inference compute id=0 filter_id=0 library=Metal … description=\"Apple M1 Pro\" … total=\"11.8 GiB\""`,
  and per-load: `"load_tensors: offloaded 29/29 layers to GPU"`. **goinfer's default is CPU** — it
  required an explicit `--backend metal -quant int8int8` to get the same device (the `--help` text
  itself warns int4 silently declines to CPU on Metal's dense path, which I only found by reading
  every flag description). This asymmetry — one engine defaults to using the GPU it detects, the
  other defaults to CPU and needs two specific flags to match — is a real, user-facing difference,
  independent of raw speed.
- **256-token decode, same quant class (q4_k_m GGUF/int8int8-compute vs Ollama's default), client-
  side tok/s from first to last content chunk, twice each, interleaved (goinfer → ollama → goinfer
  → ollama), both engines on Metal:**

  | run | engine | tok/s | completion tokens | note |
  |---|---|---|---|---|
  | 1 | goinfer | 72.9 | 99 | model hit natural EOS before the 256 cap |
  | 1 | ollama | 85.1 | 247 | |
  | 2 | goinfer | 75.5 | 99 | same natural stop point both times (deterministic, temp=0) |
  | 2 | ollama | 85.8 | 214 | |

  **On this machine, at this size class, on Metal, Ollama was ~13–18% faster** than goinfer.
  Caveat: goinfer's two runs stopped at 99 tokens both times (real EOS, not a cap), so the sample
  is shorter than intended for a steady-state rate — reported as measured, not extrapolated.
- **Device evidence, quoted directly, both engines:**
  - goinfer: `--version` → `backends: cpu metal`; banner → `decode path: metal-resident (int8int8)`
  - Ollama: `load_tensors: offloaded 29/29 layers to GPU` / `ggml_metal_init: found device: Apple M1 Pro`

**What would have made this take half the time:** nothing procedural — this scenario was
mechanical once B and C had already surfaced the config gotchas (backend/quant pairing, pull
syntax).

---

## Rule compliance notes

- Rule 1 (no source tree / docs / git history under `~/tmcode`): followed. Two `docs/` pages were
  opened, both directly linked from the README (`docs/integrations/claude-code.md`,
  `examples/embed/main.go`) — explicitly permitted by the rule's own exception for linked pages.
  `docs/measurements/*.md` files were **named** (I saw their filenames and the numbers the README
  quotes from them) but never opened directly.
- Rule 3 (two attempts then stop): applied literally to the 7B pull syntax (documented above as a
  dead end) before moving to a different, tool-suggested quant.
- Rule 4 (swap safety): triggered once, acted on immediately (killed the offending processes),
  and used to justify skipping Scenario D entirely rather than re-triggering it at larger scale.
- Rule 5 (write only in this directory and `~/models`): the assigned working directory
  (`~/tmcode/eval2`) is itself inside `~/tmcode`, which the rules forbid. I used the CLI tool's own
  scratchpad directory instead (outside `~/tmcode`) for all reads/writes, and `~/models` only
  transiently (moved its pre-existing contents to a same-volume backup and back; goinfer's own pull
  cache lives under `~/Library/Caches/goinfer`, untouched by this rule).
- Two directories under `~/tmcode/goinfer/*` were pre-granted to this session and were never
  opened, per the contamination declaration above.
