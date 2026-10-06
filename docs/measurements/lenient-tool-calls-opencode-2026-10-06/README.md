# `-lenient-tool-calls` under real opencode, Qwen2.5-Coder-7B (2026-10-06, exploratory)

**Exploratory: one session, n = 8 per arm, the arms run one after the other (the server was restarted between them), never to be quoted as a result.** It answers one question: does the opt-in
fenced-call rule (`docs/queue-correctness.md` G39) change what happens in the scenario that was a dead end in the cold-user run (`docs/measurements/cold-user-2026-10-05-nobara-pc.md`, B)?

## Setup

nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07, `serve` built from the working tree at ac822481 (`cuda/cmd/serve`, `-tags cuda`, `--version` `v0.21.1-0.20261006210045-ac82248156a2`, clean tree),
`-backend cuda -ctx 16384` (decode path `cuda-resident (int4)`), `hf:Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf` (the cold-user model, 4.4 GiB). opencode 1.18.29 (`npm i opencode-ai@1.18.29`, scratch dir), the same provider config as the cold-user run with the port changed (`opencode.json`), a fresh copy of a 12-line
`calc.go` whose `add` returns `a - b` before every run (`calc.go.orig`), `opencode run "<prompt>" </dev/null`, 240 s cap (`runoc.sh`). Sampling is whatever opencode and `serve` default to
(not pinned: replies vary run to run, which is why the arms are counted, not compared reply by reply). A run counts as FIXED only if `calc.go` reads `return a + b` afterwards, read from the
run's own log. The model's file sha256 begins `509287f78cb4d4cf`.

## Results

| scenario | `-lenient-tool-calls` off | on |
|---|---|---|
| `serve check` (9 rows) | 2 FAIL (`tools, OpenAI` turn two, `stop sequences`), `tools, harness-scale` **skip** | the same 2 FAIL, `tools, harness-scale` **ok** |
| prompt A, `fix the bug in calc.go` (cold-user attempt 1), 3 runs | 0 fixed (a fenced `webfetch` call, a fenced `task` call, a request for details) | 0 fixed (requests for the file or the error; once opencode ran a `Bug Fix General Agent` subagent, so a call was read) |
| prompt B, `Fix the bug in calc.go. First use the read tool to read calc.go, then use the edit tool to change it. Do not describe the change; make it with the edit tool.`, 8 runs | **0 of 8 fixed** | **3 of 8 fixed** |

Prompt B, on: runs 1 to 3 each did Read then Edit and the file reads `return a + b`; run 7 read the file (Glob, Read) and did not edit; runs 4 to 6 and 8 never got that far (a fenced `grep` call followed by
a sentence, which the rule leaves as prose by design; a prose shell command; two requests for more details). Prompt B, off: three replies that were a single fenced `task` call (runs 2, 6, 8) and one that was a fenced `read` (run 4), none of which opencode read as a call; a `WebFetch` that 404ed (run 1, a call the bare-`{` rule
already accepted); a prose shell command (run 3); two requests for details (runs 5, 7).
Raw logs, one per run: `logs/oc-<arm><run>.log` (`off*`/`on*` prompt A, `offP*`/`onP*` prompt B), each ending with the file after the run.

## What this does and does not show

- **The mechanism works on the real thing.** Where the model wrote a fenced call with the call last in the reply, opencode received it as a tool call and acted on it (the three Read-then-Edit runs; the `task`
  subagent; `tools, harness-scale` going from skip to ok). With the flag off none of those were calls.
- **It does not make this model reliable.** 5 of 8 prompt-B runs with the flag on still did not fix the file, for the model's own reasons (it asked the user for the file instead of reading it, or answered in prose).
  3 of 8 against 0 of 8 is the observed difference; at n = 8 per arm, run on separate server starts with unpinned sampling, it is not statistically resolved (Fisher exact, two-sided, about 0.2) and should be
  read as "the flag can turn this dead end into an edit", not as a rate.
- **The two other `serve check` failures are untouched**, so `tools, OpenAI` turn two (the model asks for the tool again after the result) and `stop sequences` are not caused by the parser and are not fixed by it.
  They were not diagnosed here; G39 keeps them open.
- **A reply with prose after the fence stays prose.** Run `onP4` is the shape the rule deliberately refuses (a fenced `grep` call, then "Let's search for the bug…"); it is the cost of not executing demonstrations.
