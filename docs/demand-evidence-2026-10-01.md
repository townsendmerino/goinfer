# What Go developers struggle with — evidence from the Go inference peers' issue trackers (2026-10-01)

> **Status: an evidence pass, not a plan of record.** Read-only: no goinfer code or measurement changed. The owner asked for real
> evidence behind the 2026-09 product analysis, which was desk reasoning. Issues were collected from GitHub on 2026-09-30 and
> 2026-10-01 (counts and reactions are as of then). The sample is small and skewed in ways §2 lists; read the numbers as
> directions, not as a market size. HN, Reddit, Discord and Slack could not be reached from the collection environment.

## 1. What the evidence says

1. **In the Go-native engine trackers, the biggest group of user complaints is the native layer, not speed.** Of 143
   issues filed by people other than the repos' named maintainers (99 distinct authors, seven trackers), 62 (43%) are about the
   native library not building, installing or loading (38), GPU or hardware enablement (17), or crashes at the C boundary (7).
   53 of the 99 authors (54%) filed at least one of these. The share is the same in 2023–24 (32 of 74) and 2025–26
   (30 of 69). Without the largest tracker (go-skynet, 57 issues) it is 30 of 86 (35%), still the largest group.
2. **Performance is a small share of what people file.** 4 of 143 (3%), all from 2023–24; none of the 69 issues from 2025–26. On
   Ollama, issue titles about speed are 310 of 11,319 (2.7%), against 992 (8.8%) for titles about a GPU not being used or detected and 692 (6.1%)
   for install, download or pull. This weakens the premise that speed gates adoption, but it cannot disprove it: a tracker only
   shows people who got far enough to file (§5).
3. **Tool calling is the live pain on the incumbent, and it is mostly parser and template bugs.** Of 181 Ollama issues returned by
   tool-call searches, at least 81 name a parser, renderer, template, dropped-call or leaked-token fault in the title; 58 of those
   are from 2026. The fixes are per-model-family hand-written parsers. Three users explicitly ask for what goinfer's constrained
   decoding does (a 2024 request to "combine all possible Tools into one JSON Schema", a 2026 report that a tool `enum` is not
   enforced during decoding, and a yzma request for JSON-schema-forced output), but each has at most 3 reactions. goinfer's constrained tool calls cover the Qwen families, which
   are 23 of the 56 titles that name a family, and not gpt-oss, GLM or (constrained) Gemma 4 (§6).
4. **Embedding inference in a Go program is asked for, but thinly.** Six issues by five users across Ollama and Kronk
   (the highest has 5 reactions). One asks for exactly goinfer's pitch: "use the //go:embed model/* to already have the tool
   embedded in the Golang binary". Go agent tools show a larger want for *local models behind an OpenAI-compatible endpoint*
   (Charm's crush: "Offline Mode" 28 reactions, "Ollama integration" 35, "llama.cpp support" 17).
5. **Explicit "cgo-free" demand is rare in this evidence: 3 issues by 2 users out of 143, all in trackers that were already
   cgo-free by design.** The pain that cgo-free would remove is common; people asking for it by name are not. That is
   consistent with a single-static-binary, no-shared-library-matching claim being a good pitch, but this evidence does not show
   that people look for it.
6. **Three gaps and one risk against goinfer's current posture** (§6): `-backend` defaults to `cpu`, which is the shape of the most
   common complaint class; there are no integration recipes or Docker example for the Go clients and tools that people actually
   use; split-GGUF checkpoints are refused (the most-reacted import request on Ollama, 217 reactions); and a young single-maintainer
   project is read as possibly abandoned in the pure-Go trackers.

## 2. What was read, and how far to trust it

**Primary set (coded issue by issue).** Seven Go-native trackers: `go-skynet/go-llama.cpp` and `tcpipuk/llama-go` (cgo bindings),
`hybridgroup/yzma` and `dianlight/gollama.cpp` (purego/FFI bindings), `ardanlabs/kronk` (a Go model server built on yzma),
`computerex/dlgo` and `gotzmann/llama.go` (pure Go). 295 issues; 152 were filed by the repos' own maintainers and are excluded from
demand counts, leaving **143 user-filed issues by 99 distinct authors**. Each was given one theme by hand, from its title and, where the
title was unclear, its body.

**Secondary sets (read for patterns, not coded into the table).** Ollama: keyword counts over issue titles (11,319 issues),
a 181-issue tool-call set, and targeted searches for embed-as-library and import requests. langchaingo (68 issues across
three searches); `sashabaranov/go-openai` (13) and `openai/openai-go` (23) issues that mention local servers; Charm's `crush`
(200 returned, 35 about local runtimes) and `mods` (32); Genkit (60) and Eino (9) Ollama issues; four GitHub-wide searches
(567 issues) for "no cgo", "without cgo", "pure go" LLMs and embedding a model in a Go app. Collected but **not analysed** (left for a next pass):
`knights-analytics/hugot`, `kstruzzieri/go-llm`, LocalAI's Go-library issues, `zerfoo` (299 of 300 issues by one author, so
unusable as demand), and one project that surfaced in the "no cgo" search with Metal and Vulkan work items filed as issues
(`anthony-chaudhary/fak`), which I did not examine.

**Limits that matter.**

- **Small and concentrated.** 23 of the 99 authors filed 62 of the 143 issues; one go-skynet user filed 7. Distinct authors are
  the safer unit, and the headline above uses them where it can.
- **Time skew.** 74 of the 143 are from 2023–24 (53 from go-skynet, 21 from llama.go): the early ggml-binding wave. Everything is
  reported for both eras.
- **Survivorship.** Trackers show people who got far enough to file. Someone who tried a tool, found it slow, and left never
  appears; neither does someone who failed to install and gave up before filing. This cuts against reading a low "performance"
  share as proof that speed does not matter.
- **Not all "users" are outsiders.** 11 of the 143 are marked contributor or collaborator by GitHub (3 are Kronk collaborators), who sit
  closer to the project than a typical user. Without them: 132 issues, 92 authors, A + B + C is 59 (45%, 50 authors), performance still 4.
- **One coder.** The themes were assigned by one reader (an AI assistant), single-label, from title and body, with no second coder.
  Borders blur (a missing GPU build can be A, B or L); expect a few issues to move.
- **Ollama's population is not Go developers.** Its counts describe local-LLM users in general. Its title counts are lower bounds
  from OR-queries over titles only, they overlap, and "tool" can match non-LLM uses.
- **Not reached:** HN, Reddit, Discord, Gophers Slack, X; any private feedback. The GitHub search API returns issue bodies but
  comment threads were mostly not read (the unauthenticated rate limit is 60 requests an hour).

## 3. The primary set: what users file

Themes are mutually exclusive. "Authors" is distinct people per theme. 2023–24 is go-skynet and llama.go almost entirely.

| | Theme | Issues | Authors | 2023–24 | 2025–26 |
|---|---|---:|---:|---:|---:|
| A | Native library won't build, install or load | 38 | 33 | 21 | 17 |
| F | Capability gaps and API ergonomics | 21 | 17 | 8 | 13 |
| B | GPU / hardware enablement | 17 | 15 | 10 | 7 |
| G | Generation control and concurrency | 10 | 8 | 10 | 0 |
| K | Other (examples, README, one-offs) | 10 | 9 | 4 | 6 |
| I | Docs, getting started | 9 | 9 | 6 | 3 |
| C | Crashes at the native boundary | 7 | 7 | 1 | 6 |
| E | Server API behaviour and compatibility | 7 | 5 | 1 | 6 |
| H | Maintenance and trust | 6 | 6 | 6 | 0 |
| J | Model, architecture or chat-template support | 6 | 6 | 3 | 3 |
| L | Fit and memory (VRAM, OOM, sizing) | 5 | 5 | 0 | 5 |
| P | Performance | 4 | 4 | 4 | 0 |
| D | Tool calls and structured output | 3 | 2 | 0 | 3 |
| | **Total** | **143** | **99** | **74** | **69** |

A + B + C = 62 issues, 53 authors (the authors are not additive across themes). 2025–26 alone: 28 of the 46 authors.

### What each large theme looks like

**A — native layer (38).** The library cannot be found, is the wrong version, or will not load. yzma #106 (Jetson, 45 comments):
"I don't know where to find the necessary library (or how to build it)". yzma #190, "Mismatched release tags with llama.cpp", ends in
`failed to download llama.cpp: bad response code: 404`. Kronk #574 (Docker): "even though all the required libraries and model files
exist" the load fails. Kronk #377: the runtime library download is rate-limited by GitHub ("I get rate limited by Github pretty
quickly"). In the cgo bindings (go-skynet and llama-go, 21 of the 38) it is `CGO_LDFLAGS`, cuBLAS and compiler errors. The same
problem shows from the maintainers' side: 44 of yzma's 53 maintainer-filed issues are (by my reading of titles) crashes tied to
llama.cpp struct or ABI changes (14), bindings for new llama.cpp functions (16), or library download and verification (14). A user
asks for the fix by name in gollama.cpp #39, "Pin libllama beyond b6862 (and refresh Go struct layouts)".

**B — GPU and hardware (17).** The user has a GPU and the tool does not use it. Kronk #778: "it seems stuck on my CPU". Kronk #433
(filed by a Kronk collaborator) names the cause: "users always get CPU-only libraries unless they explicitly configure the env var". In the
older cgo trackers: "Apple Silicon Metal Support not working" (go-skynet #91), "when will NVIDIA (CUDA) on windows will be supported"
(gollama.cpp #12).

**C — crashes at the C boundary (7).** Segfaults and bus errors that Go cannot recover. gollama.cpp #38: "the process crashes with
SIGBUS (bus error)" after about 400 embeddings on Metal. yzma #329 (a panic on Darwin/Metal outside the repo directory), yzma #21.

**F — capability and ergonomics (21).** A scatter: LoRA adapters (Kronk #712), session save/load, an "interactive mode"
(llama.go #5), Agent Client Protocol (Kronk #562), ONNX/speech/vision models via purego (yzma #129, #130). Kronk #513 and #608 are the
embedding-without-a-server asks (§4). Mostly outside a decoder-only LLM runtime's scope.

**G — generation control (10), all 2023–24.** Streaming, stop words, a "global mutex/callbacks" in the cgo binding (go-skynet #226).
Artifacts of the binding design, not of Go-native engines.

**H — trust (6), all in the pure-Go trackers, all 2023–24.** "is this proj abandon?" (llama.go #23), "Anyone taking this project
forward for real?" (#29).

**I — docs (9).** Even for a working server: Kronk #700, "I think there is a huge cognitive barrier to getting started with kronk",
and the writer asks for one place that says how to start.

**L — fit and memory (5), all 2025–26.** Kronk #701 asks the server to say "which model is best for this hardware"; Kronk #688,
"it adds an extra 7GB vram on top of the 10GB vram occupied by the model itself"; dlgo #3, a false out-of-memory because it reads free
rather than available RAM.

**E and D — server behaviour and structured output (10), 9 from 2025–26, mostly Kronk.** An oversized prompt answered with a 500 (#861) or a
200 carrying an in-band error (#862) instead of a 400; gpt-oss tool calls that fail the whole request (#931); Qwen2.5-Coder tool
calls emitted as prose (#944). yzma #116 (3 reactions, completed) asks for what goinfer's first feature does: "Support json schemas to
force the model to generate valid JSON".

## 4. Beyond the engine trackers

### Ollama (the incumbent)

Issue-title counts over all 11,319 issues, collected 2026-09-30. Overlapping, lower bounds, titles only.

| Title mentions | Issues | Share |
|---|---:|---:|
| GPU not used / not detected / CPU fallback | 992 | 8.8% |
| model or architecture "support" | 985 | 8.7% |
| install, download, pull, digest | 692 | 6.1% |
| Windows | 457 | 4.0% |
| tool calls / function calling | 406 | 3.6% |
| out of memory, OOM, VRAM | 369 | 3.3% |
| crash, panic, segfault | 312 | 2.8% |
| slow, performance, speed, latency, tokens/s | 310 | 2.7% |
| context, num_ctx, truncation | 281 | 2.5% |
| OpenAI or Anthropic compatibility, `/v1` | 275 | 2.4% |
| JSON, schema, structured, grammar | 175 | 1.5% |

The highest-reacted items in the slow/performance search are about **hardware reach and model support**, with a few that are speed- or
capacity-motivated: an MLX backend (#1730, 381 reactions), Pixtral support (#6748, 231), Gemma 3n (#10792, 195), AMD Ryzen NPU
(#5186, 168), Intel Arc (#1590, 139), older AMD GPUs (#2453, 83), MoE weight offload to CPU (#11772, 78), parallel requests (#358, 59),
KV-cache quantization (#5091, 33). The long-running MCP request (#7865) has 220.

**Import.** The most-reacted import request is "Allow importing multi-file GGUF models" (#5245, 217 reactions, 110 comments, still open;
its body now notes `ollama create` supports it): "larger models are sometimes split into separate files".

**Embed as a library.** Few and quiet. #7450 (5 reactions, 2 comments): "use the //go:embed model/* to already have the tool embedded in
the Golang binary", to avoid "an extra Docker Container or VM". #3234 (0 reactions, closed): use Ollama "as a library, not through
network". #505 and #11309 are developers trying to import Ollama's own packages.

**Tool calls (181 issues returned by the searches; not a census).** Coded by hand from titles: 96 are parser, renderer or
template faults (a call dropped, mangled, leaked into content, or a 500), 19 are history, thinking or multi-turn interactions,
19 are client or API-compatibility issues, 12 are model requests for tool support, 9 are about constraints not being enforced,
8 are streaming, 18 are other. Separately, 81 titles say so by keyword; 58 of those are from 2026, 56 are still open, 77 distinct authors.
Of the 56 titles that name a family: Qwen 23, Gemma 12, Llama 10, gpt-oss 4, GLM 4, DeepSeek 3, Kimi 2 (a title can name two).
Two are about the thing goinfer's grammar does: #6002 (2024, closed, 2 reactions, 17 comments) — "One would need to combine all possible
Tools into one JSON Schema" — and #17597 (2026, open, no reactions): "Tool-parameter `enum` reaches the model but is not enforced during
decoding (unlike `response_format`)". #18509 ("Ollama refusing toolcalls, which always worked fine in llama.cpp with qwen") is a
message-role incompatibility, not a parser fault.

### Go agent tools and frameworks

- **Charm `crush`** (a Go coding agent): of the 200 issues returned, 35 name a local runtime. Most-reacted: #190 "Add Ollama
  integration with CLI-based auto-discovery" (35), #453 "Offline Mode" (28: "This could become a fully offline system when paired with
  something like ollama"), #447 "Local LM Studio/Ollama Custom Providers Support" (23 reactions, 40 comments), #365 "llama.cpp support"
  (17: "connect it directly with llama.cpp running at the host/ip of our choosing"), #749 "API error on tool use with local models"
  (13), #824 "Context limit not respected in requests" (13). #392 asks for docs on connecting local models.
- **Charm `mods`**: #162 "Ollama as an API option" (38 reactions); #561 "Output loops forever when using ollama" (10).
- **langchaingo** (68 issues): users ask for a local model path and are told Ollama or an OpenAI-compatible URL. #1347: "i wanna know if i
  can use llama.cpp or only can use ollama to run local model file". Its Ollama client is HTTP and drifts from Ollama's API
  (#1035, #1514 `think` ignored); multi-tool agents fail on local models (#960 "Only the first tool is recognized", #1045). Separately, #1071
  ("Community development", 25 reactions, 38 comments) is a maintenance-capacity complaint: "PRs are piling up recently".
- **Go OpenAI clients** (`go-openai`, `openai-go`): the recurring items are base-URL handling (openai-go #103, #105, #134, #240, #448
  "WithBaseURL is not concurrency-safe"), `reasoning_content` support (#224), and streaming tool calls against Ollama (#417). They are
  what a goinfer server would be called from.
- **Genkit (Go) and Eino** (Ollama issues): "Support tool calling for any arbitrary (ollama?) model" (Genkit #663), "which models supports
  tools?" (#661), a hard-coded 30-second timeout in the Go Ollama plugin (#3506); a "llama.cpp suport" request in eino-ext (#805).
- **GitHub-wide searches** for "no cgo", "without cgo", "pure go" LLMs and embedding a model in a Go app (567 issues) returned mostly unrelated
  projects plus issues from the peer trackers already read. I found no thread outside those trackers that asks for a pure-Go inference
  engine.

### Explicit cgo-free and embed demand (the rarest, most relevant signal)

- yzma #130 (2 reactions): "from Go without CGO", for ONNX models (speech, vision, embeddings); #129 the same user, "All CGO-free,
  all on-device".
- gollama.cpp #39: "the purego approach is a great fit for us (Windows built cross compiled to linux and darwin in particular)".
- Embedding without a server: Ollama #7450, #3234; Kronk #513, #608 (a wrapper so Kronk works with langchaingo "without needing"
  a server). #513: "I really like the idea of embedding the inference engine directly into a Go application without needing a model server".

Together: seven issues by five users (cgo-free by name, 3 issues by 2 users; embedding without a server, 4 issues by 3 users). Most chose
their tool because it was cgo-free or embeddable, so they are not a random sample of Go developers.

## 5. Does speed gate adoption?

The owner's stated position (2026-09-11) is not to promote goinfer until it is at least as fast as Ollama. This evidence bears on it
from both sides and decides nothing.

**Leans against speed being the first barrier.**

- Performance is 3% of outside-user issues in the Go-native trackers and 0% of the 2025–26 ones; 2.7% of Ollama titles.
- What stops people first is getting the native layer to load (27% of issues, 33 authors), getting the GPU used (12%), and then
  fitting the model (L) and getting tool calls right (D, and 81 Ollama titles).
- The highest-reacted speed-adjacent Ollama items are hardware reach, which a single static binary does not obviously help with.

**Cannot rule out.**

- Survivorship (§2): a person who found it slow and left files nothing.
- The Go-native trackers mostly serve people already willing to run a llama.cpp-based tool; a Go developer weighing embedding goinfer
  against calling Ollama over HTTP may compare tokens per second first. This evidence does not reach that person.
- Hardware reach and Apple-Silicon speed are among the highest-reacted Ollama requests (MLX 381), so speed on specific hardware does
  matter to part of the audience.

**Where goinfer stands on the gate as measured.** On the first-run measure that matches theme A, the README records nothing to first
answer in 25 s against Ollama's 33 s on an M1 Pro (cold-user run, scenario E). The README also records decode on CUDA ahead of Ollama in all 12 measured cells (1.05–1.47×)
and on Metal 1.03–1.19×; it is behind on Phi-3 mini decode on CUDA (0.72–0.90×) and on Mac CPU decode against llama.cpp (0.84× on
the 0.5B). Long-prompt prefill on Metal, 1.96× slower at 3,900 tokens on 2026-09-25, measured level on 2026-09-30. The gate of being at least as fast as Ollama is therefore already true or
false depending on the metric; restating the gate by regime (for example: decode at short and mid context on CUDA and Metal, with the prefill
gap stated) would make it checkable. That is the owner's call.

**What would settle it.** A small first mention to an audience that will say what they hit (the install and first run, not the
benchmark) and a note of what they raise. Failing that, a handful of conversations with Go developers who have tried yzma, Kronk or Ollama's
Go client.

## 6. Themes against goinfer's current posture

Status: **by design** (the architecture removes it), **built** (shipped; whether people find it is open), **partial**, **gap**,
**out of scope**. Evidence is in the repo; paths are relative to the repository root.

| Theme (evidence) | goinfer today | Status |
|---|---|---|
| A. Native library won't build/install/load (38 issues; Ollama install/pull 692) | Pure Go, no cgo, no libllama; CPU, CUDA (driver-only) and Metal (system frameworks) with `CGO_ENABLED=0`. See `docs/positioning.md`. GPU builds still need the host driver or framework. | **By design** |
| B. GPU used? (17; Ollama 992) | `-backend` defaults to `cpu` (`internal/loadflags`). A user with an NVIDIA GPU who does not pass `-backend cuda` runs on CPU. The cold-user run in `docs/measurements/cold-user-2026-09-06-nobara-pc.md` hit this ("no `--backend`, so CPU"). `--require-backend` exists for batch jobs. Whether `serve` prints a hint when a GPU is present and the backend is `cpu` was not checked. | **Gap** (same shape as Kronk #433, #778) |
| C. Crashes at the C boundary (7; Ollama 312) | No C library on the CPU path. Metal and CUDA reach the system framework or driver without cgo (the Metal path uses purego), so a crash class remains there; see `docs/audit-metal-2026-09-30.md`. | **Partial** |
| D. Tool calls (3; Ollama 406, 81 parser-fault titles) | Constrained on Qwen families, Nemotron-3-Nano, Mellum2, Granite 4.2; Llama 3.x under `required`; Gemma 4 parsed only; gpt-oss, Gemma 3 and Ministral none; many families "template not recognised". `docs/tool-call-coverage.md`. Of the 56 Ollama titles naming a family, 33 name Qwen or Llama, 12 Gemma, 13 gpt-oss, GLM, DeepSeek or Kimi (a title can name two). | **Partial**; scoped further in `docs/tasks/task-harness-reliability-2026-10.md` |
| E. Server compatibility (7; Ollama 275; Go OpenAI clients) | OpenAI chat, Responses, embeddings and Anthropic routes; an oversized prompt returns a 400 `context_length_exceeded` (`internal/serveapp`), where Kronk #861/#862 returned a 500 and an in-band 200 error. No test against `openai-go`, `go-openai` or langchaingo was found. | **Built**; unchecked against the clients |
| F. Capability gaps (21) | MCP, Agent Client Protocol and non-LLM ONNX models are not engine concerns; MCP support was found only in `demo/agent`. LoRA was not checked. | **Out of scope** mostly |
| G. Generation control (10, all 2023) | A cgo-binding artifact. Streaming, stop sequences and batched decode exist (`docs/tasks/task-concurrency-2026-09.md`). | **By design** |
| H. Is the project abandoned? (6, pure-Go trackers) | Tagged releases (v0.19.0, 2026-09-18) and a changelog; a single maintainer. A perception risk with no code fix. | **Risk** |
| I. Docs, first run (9) | README quick start; first-hour fixes in `docs/tasks/task-first-hour.md`. No Dockerfile in the repository and no Docker instructions in any tracked markdown file; `docs/integrations/` has recipes for Claude Code, OpenCode and typesafe-jevx, none for crush, mods, langchaingo or the Go OpenAI clients. | **Gap** (small) |
| J. New model or chat template (6) | Templates are fingerprinted into eight renderers rather than interpreted as Jinja; an unrecognised template means raw completion and no tools (`docs/tool-call-coverage.md`). Each new family's template needs a renderer. Kronk #317 and yzma #340 are the Jinja-failure shape. | **Partial** |
| L. Fit and memory (5) | `goinfer-chat fit`, fit guard, never-swap defaults, 39 families with a generated hardware matrix (`docs/tasks/task-fit-to-hardware.md`, `docs/hardware-matrix.md`). | **Built**; discoverability open |
| P. Performance (4) | Ahead of Ollama on decode on CUDA and Metal, behind on Phi-3 mini decode on CUDA and on Mac CPU decode against llama.cpp; Metal long-prompt prefill level since 2026-09-30 (README, `docs/benchmarks.md`). | See §5 |
| Import: split GGUF (Ollama #5245, 217 reactions) | `pull` refuses split GGUF checkpoints and no loader assembles them (`pull` package, `CHANGELOG.md`). Safetensors shards load. | **Gap** |

## 7. What this suggests, cheapest first

None of these is decided. Each lists the evidence it rests on and what it would cost to check.

1. **Decide what a GPU user gets by default.** `-backend` is `cpu`; the commonest complaint class in the peers is a GPU that is not used.
   Options: pick the backend compiled into the binary when a device is present, or print one line when a GPU is present and the
   backend is `cpu`. Cost: small. The default affects parity-gated paths, so it is the owner's call. Evidence: B (17), Kronk #433, #778,
   Ollama 992.
   **Decided and built 2026-10-01** as R17 (`docs/tasks/task-first-hour.md`): `-backend auto` is the CLIs' default.
2. **Run the Go clients against `goinfer-serve`.** `openai-go`, `go-openai`, langchaingo (OpenAI mode), crush and mods against chat,
   streaming, tools, `json_schema`, errors on an oversized prompt, a base URL with and without `/v1`, and `reasoning_content`. Cost: a script,
   and it belongs beside the harness-breadth work already scoped in `docs/tasks/task-harness-reliability-2026-10.md`. Evidence: E, the client
   issues in §4, crush #749 and #824.
3. **Write the missing recipes.** crush, mods, langchaingo, a Go OpenAI client, and a Dockerfile (a static binary suggests a very short one).
   Cost: small. Evidence: crush #392, Kronk #574 and #700, I (9).
4. **Weigh tool-call coverage by where the pain is, not by family count.** Qwen is covered; Gemma 4 (constrained), gpt-oss and GLM are not.
   The scoped task already covers more families; this evidence adds that the pain concentrates in Qwen, Gemma and Llama titles first.
   Evidence: 81 title-flagged Ollama issues.
5. **Split-GGUF `pull` and load.** The largest import request in the incumbent's tracker. Cost: medium, and it interacts with the `.giw`
   conversion proposal. Evidence: Ollama #5245.
6. **State the first-hour message for the pure-Go claim in terms people complain about**: no libllama to match, no toolchain, one file. It is
   consistent with theme A, but nobody in this evidence asked for it by name, so treat it as a hypothesis the first public mention can test.
7. **Revisit the speed gate (§5)** by regime, or test it with a small first mention, rather than leaving it a global bar.

**Evidence that would change these conclusions:** threads from HN, Reddit and Gophers Slack (not reachable from here); goinfer's own issue
tracker and install failures once it is mentioned publicly; comment-level reads of the 20 most-reacted issues; a second reader coding a
sample of the 143.

## 8. Appendix A — membership of each theme (primary set, user-filed issues)

Issue numbers by tracker. Base URLs: go-skynet `github.com/go-skynet/go-llama.cpp/issues/N`, yzma `github.com/hybridgroup/yzma/issues/N`,
kronk `github.com/ardanlabs/kronk/issues/N`, gollama.cpp `github.com/dianlight/gollama.cpp/issues/N`, dlgo `github.com/computerex/dlgo/issues/N`,
llama.go `github.com/gotzmann/llama.go/issues/N`, llama-go `github.com/tcpipuk/llama-go/issues/N`.

- **A — Native library won't build, install or load (38):** go-skynet #2, #5, #67, #77, #78, #110, #115, #136, #142, #147, #150, #173, #190, #216, #218, #290, #291, #295, #328, #335; llama-go #2; yzma #106, #177, #190, #192, #272; kronk #52, #269, #370, #377, #459, #502, #574, #671, #672; dlgo #1; llama.go #17, #31.
- **B — GPU / hardware enablement (17):** go-skynet #42, #91, #98, #125, #209, #219, #230, #259, #261, #265; gollama.cpp #12; kronk #433, #674, #680, #778; dlgo #2, #4.
- **C — Crashes at the native boundary (7):** go-skynet #143, #343; llama-go #4; yzma #21, #329; gollama.cpp #18, #38.
- **D — Tool calls and structured output (3):** yzma #116; kronk #931, #944.
- **E — Server API behaviour and compatibility (7):** kronk #685, #705, #719, #759, #861, #862; llama.go #19.
- **F — Capability gaps and API ergonomics (21):** go-skynet #1, #63, #212, #236, #239, #326, #334; llama-go #1; yzma #129, #130, #409; gollama.cpp #15; kronk #513, #562, #608, #618, #712, #971; dlgo #7; llama.go #2, #5.
- **G — Generation control and concurrency (10):** go-skynet #3, #4, #20, #183, #205, #221, #226, #240, #325, #329.
- **H — Maintenance and trust (6):** llama.go #20, #21, #23, #25, #29, #30.
- **I — Docs, getting started (9):** go-skynet #52, #297; yzma #22; kronk #700; dlgo #8; llama.go #1, #14, #18, #24.
- **J — Model, architecture or chat-template support (6):** go-skynet #314, #331; yzma #340; gollama.cpp #39; kronk #317; llama.go #27.
- **K — Other (10):** go-skynet #102, #332, #333; yzma #341; kronk #435, #645, #970; llama.go #16, #22, #32.
- **L — Fit and memory (5):** kronk #371, #688, #701, #809; dlgo #3.
- **P — Performance (4):** go-skynet #109; llama.go #3, #11, #28.

## 9. Appendix B — how the numbers were produced

- **Collection.** GitHub's unauthenticated search API from the owner's Mac VM, one query per call with a pause between calls (search is
  limited to 10 requests a minute, core API 60 an hour). Issue bodies come back in the search results; pull requests were excluded.
  Maintainer-filed issues are those by each repo's named maintainers, chosen by login (go-skynet: `mudler`; yzma and Kronk:
  `ardan-bkennedy`, `deadprogram`, `dlsniper`; gollama.cpp: `dianlight` and a dependency bot; none for dlgo, llama.go or llama-go). They
  are excluded from the demand counts. A yzma maintainer filing on gollama.cpp (#18) counts as a user there.
- **Ollama title counts.** Queries of the form `repo:ollama/ollama is:issue in:title <terms joined by OR>` for each row of the §4 table;
  the total (11,319) is `repo:ollama/ollama is:issue` at the time. A body-keyword version was tried first and discarded because
  issue templates put words such as "GPU" and "install" in every body.
- **Coding.** Primary set: one theme per issue, by hand, after a keyword pass proved too noisy. Ollama tool-call set: 181 issues,
  coded by title into parser or template fault, history or thinking, API compatibility, model request, constraint not enforced,
  streaming, other; the keyword count of 81 uses a title pattern for parse, render, template, malformed, dropped, leaked, 500,
  XML and similar words, so it is a lower bound on faults and can include a few false hits.
- **Reproducing.** The collector, raw issue files (`*.jsonl`) and counting scripts were left in the uncommitted scratch directory
  `_to_delete/demand-evidence/` in the working tree. They are not part of the repository. Move them somewhere durable if a second pass or
  a second coder is wanted.
