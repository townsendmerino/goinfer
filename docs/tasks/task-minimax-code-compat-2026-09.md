# Task: goinfer as a minimax-code backend (M0–M7) — 2026-09

> **Status: SCOPED 2026-09-25, unstarted.** M0 is a measurement and gates the rest: every band below
> is a prediction from reading both codebases, and M0 is where it becomes a number. M1 is docs and can
> land with M0. M2–M4 are the fixes the reading says matter; M5–M7 are verify-then-decide.
>
> **What minimax-code is.** MiniMax's terminal coding agent (`github.com/MiniMax-AI/minimax-code`,
> MIT, read at `4198174`, 2026-09-25). Besides its own MiniMax account mode it has a bring-your-own-model
> mode (`mcode provider add --base-url … --api-format …`) speaking `openai-completions`,
> `openai-responses` or `anthropic-messages`, all three of which goinfer serves. Its LLM client is a
> vendored `pi-ai` (`third_party/pi-mono/packages/ai`). It is the same user as mode 3 in
> [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) — "point my tools at it" — on a harness
> this repo has not tested.
>
> **Siblings.** [`../integrations/opencode.md`](../integrations/opencode.md) and
> [`../integrations/claude-code.md`](../integrations/claude-code.md) (the two harness pages this one
> copies the shape of) · [`task-tool-grammar-union-2026-09.md`](task-tool-grammar-union-2026-09.md) (the
> many-tools grammar M0 relies on) · [`task-fit-to-hardware.md`](task-fit-to-hardware.md) (owns the
> context-default policy M2 amends) · [`task-memory-accounting-2026-09.md`](task-memory-accounting-2026-09.md)
> (the pricing M2 uses) · `docs/server.md` · `docs/api-tiers.md`.

---

## 0. What the reading established (2026-09-25, goinfer `a5d66ffd`)

**Works as-is, by the code on both sides** (not yet run together — M0):

| minimax-code sends (openai-completions, custom URL) | goinfer |
|---|---|
| system prompt as `role: "developer"` when its thinking toggle is on | treated as `system` (`internal/serveapp/openai.go:messagesToTurns`) |
| `max_completion_tokens` (its default for an unrecognised URL) | honoured, preferred over `max_tokens`; clamped to the context, not refused; ceiling 131072 (`internal/serveapp/openai.go:maxOutputTokensCeiling`) |
| `stream: true` + `stream_options.include_usage` | supported; final usage chunk |
| `store: false`, `prompt_cache_key`, tools' `strict: false`, `reasoning_effort` | unknown fields ignored (plain `json` decode) |
| tool calls read from `delta.tool_calls[i]` with `index` | emitted in that shape (`internal/serveapp/tools.go`) |
| `Authorization: Bearer <key>` (its `--api-key-env` is mandatory) | ignored when serve has no `--api-key` |
| overflow detection by error text (`utils/overflow.ts`, generic `/context[_ ]length[_ ]exceeded/i`) | the 400 reads "… context window of N tokens (context_length_exceeded)" (`internal/serveapp/openai.go:promptByteBudgetError`) — matches, so its compaction should fire |

**Will bite, in order of likelihood:**

1. **Context size.** minimax-code assumes a 200,000-token window for a custom model unless given
   `--context-limit`, and sends its system prompt (`packages/local-runtime-v2/assets/agents/_default/prompt-base-all.md`
   alone is 7.7 KB) plus the tool schemas on every turn — roughly 5–8k tokens before the first user
   message, by estimate from the files (M0 measures it). goinfer's resident defaults were, when this was written, 4096 on Metal
   (`decoder/fitplan.go:MetalCtxDefault`) and 4096→8192 fit on CUDA (`cuda/resident.go:cudaCtxCapDefault`); since M2 both grow toward 16384 when memory holds it. The first turn may not fit;
   the second almost certainly will not. It does not read goinfer's `/v1/models` `context_window`.
2. **Thinking models answer the connection test slowly.** `--use` sends a **non-streaming** `ping` with
   `max_tokens` = the configured output limit (default 16,384) and a **10 s** timeout
   (`packages/local-runtime-v2/src/service/model-system/connectivity/test-connection.ts`). goinfer has no
   way to turn a model's thinking off — nothing reads `enable_thinking`, `chat_template_kwargs` or
   `reasoning_effort` — so a Qwen3-class model thinks about "ping" until the timeout. The same absence
   costs every real turn.
3. **Thinking arrives as answer text.** goinfer never emits `reasoning_content` (the llama.cpp
   convention pi-ai reads, `providers/openai-completions.ts`, `reasoningFields`), so `<think>` text is
   shown as the reply and is resent as assistant history, spending context.
4. **Images with tools are refused.** minimax-code always sends tools; a pasted screenshot gets goinfer's
   deliberate 400 "tools are not supported together with image inputs" (`internal/serveapp/openai.go:server.handleChat`,
   N-16 / R-08).
5. **The other two formats are unverified.** `openai-responses` sends item types the route declares out
   of scope ("reasoning items", `internal/serveapp/responses.go` header); `anthropic-messages` sends
   pi-ai's Anthropic shape (cache_control, thinking config, beta headers) — not traced.

---

## The items

| # | item | size | status |
|---|---|---|---|
| M0 | Run it for real — both boxes, a scripted task set, a pre-registered pass rule | S | gates M2–M7 |
| M1 | `docs/integrations/minimax-code.md` + a `serve check` row shaped like its first turn | S | with M0 |
| M2 | A context default an agent harness fits in — Metal fit-by-default, a harness warning | M | **done 2026-10-13** (run on its own, before M0) |
| M3 | A thinking switch: `enable_thinking`, `chat_template_kwargs`, `reasoning_effort`, `--thinking` | M | after M0 |
| M4 | Emit reasoning separately: `reasoning_content` deltas and Anthropic `thinking` blocks | M | after M3 |
| M5 | Images together with tools on the vision path | L | verify-then-decide |
| M6 | `/v1/responses` tolerates what pi-ai sends | S–M | verify-then-decide |
| M7 | `/v1/messages` tolerates what pi-ai's Anthropic client sends | S | verify-then-decide |

---

### M0 · Run it for real

**Goal.** Replace §0's reading with a measured compatibility record, and size M2–M4 from it.

**Setup.** Mac (Metal) and nobara (CUDA), each: `goinfer-serve --model coder=<qwen2.5-coder-7b-instruct
q4_k_m> --ctx 32768` (CUDA: the largest `--ctx` the card admits for 7B — the `--ctx` help records 20000
fitting on the 2070 SUPER), `mcode provider add --name goinfer --base-url http://127.0.0.1:8080/v1
--api-format openai-completions --model coder --api-key-env MCODE_PROVIDER_API_KEY --context-limit <ctx>
--output-limit 4096 --use`. A second model with thinking (a Qwen3-class checkpoint goinfer serves) for
the thinking cells. Pin the mcode version in the record (`mcode --version`).

**Task set.** Five scripted tasks in a scratch repo with a failing test each (read → edit → run tests),
the shape of minimax-code's own quick-start example, run with `mcode "<task>"` non-interactively; the
same five on both boxes.

**Record, per turn** (serve's request log plus `usage`): prompt tokens, `usage.prefill_reused_tokens`,
TTFT, decode tok/s, whether the reply was a tool call, parse failures, any 4xx/5xx with its body, and
every compaction minimax-code performs. Per task: pass/fail, turns, wall-clock.

**Pre-registered rule** (commit `docs/measurements/minimax-code-compat-PREREGISTERED.md` before the first
run): **compatible** = `--use` passes on the non-thinking model on both boxes, ≥4/5 tasks complete on at
least one box, zero goinfer 5xx, zero malformed tool calls. **Partially compatible** = connects and
completes ≥1 task. Anything else is a finding to fix before M1 ships. Also registered, so they can be
wrong in public: the first-turn prompt is 5–8k tokens; `--use` fails within 10 s on the thinking model;
at least one `context_length_exceeded` → compaction cycle occurs by task 5 at ctx 32768.

**Record.** `docs/measurements/minimax-code-compat-2026-MM-DD.md` + raw logs under `~/goinfer-logs/`;
`benchmarks.md` untouched (this is a compatibility record, not a speed row).

### M1 · The integration page and a `serve check` row

**Build.** `docs/integrations/minimax-code.md` in the opencode page's shape: the exact
`mcode provider add` line, why `--context-limit` must equal goinfer's `--ctx` (and what happens if it
does not), `--output-limit`, which API format to use (openai-completions until M6/M7 say otherwise),
the thinking caveat until M3 lands, the images-with-tools refusal, and M0's measured result — including
failures, at full value. Link it from the README's harness list and `docs/server.md`.

`internal/servecheck/check.go` already has a harness-scale tools row built from opencode's agent; add a
"minimax-code first turn" row: a system prompt and tool set of the size M0 measured, sent against the
served model's context, reporting `ok` / `won't fit at ctx N` before the user configures anything.

**Gate.** The page's commands are copied from a run, not written from memory; the new check row
reproduces M0's fits/doesn't-fit answer at two `--ctx` values.

### M2 · A context default an agent harness fits in

**Goal.** A user who starts `goinfer-serve` without `--ctx` and points an agent at it gets a context the
agent's first turns fit in, or a message saying so at startup — never a mid-session wall.

**Read first.** `task-fit-to-hardware.md` Phase 2 (CUDA's fit-by-default context: up to 8192 when the
card affords it, and the 2026-09-15/16 regression it caused on `-moe-cache-experts` loads — a larger
default ate the expert cache's VRAM; the fix keeps 4096 there), `metal/model.go` `resolveMetalCtxCap`,
the S4 live memory ceiling (`task-never-swap-2026-09.md`), `task-memory-accounting-2026-09.md` (the one
KV formula to price with), `docs/tasks/task-embed-and-harness-ux.md` mode 3.

**Build.**
1. **Metal fit-by-default context**, the CUDA Phase-2 shape: an unpinned Metal load grows its context
   from 4096 toward `metalCtxCapMax` (32768) while `Plan("metal")`'s need stays under the live ceiling
   with the documented margin; never on a paged-MoE load (the same companion-allocation lesson). The
   banner names the resolved number and why.
2. **A harness floor warning** at startup when the resolved context is below 16384 tokens (the M0
   number decides the exact floor): one line naming the harnesses that will not fit and the `--ctx` that
   would, priced by the same function.
3. **The 400 names both numbers** — "prompt is N tokens; context window is M" — so a harness's log says
   what to set. Keep `context_length_exceeded` in the text (it is what pi-ai matches).

**Decision rule.** Ships when M0's task set, re-run with no `--ctx`, reaches the same completion count as
M0's explicit `--ctx` run on the Mac, and the dense decode rate at depth 128 is within 3% of today's
(`TestProdThroughput`-class; a larger KV must not slow short turns).

**M2 started 2026-10-12, on its own (owner's brief of that day: do it now, without waiting for M0).** This
registration was committed before the code was, and before G-M2-speed ran. It was written after the code and after
the day's first Mac reads (the 1.5B and the 7B under gate 3), so those two are results and carry no prediction;
only the night gate's predictions below were made blind. The results follow under "M2 results".

*What the brief changes in the plan above.*
- The candidate is **16384** (`decoder/fitplan.go:MetalFitDefaultCtx`, CUDA's figure), not growth toward the 32768
  ceiling item 1 names. An unpinned load asks `Plan("metal")` for it, clamped to the model's window; the answer is
  never under 4096; a genuine `-ctx` is untouched up to 32768.
- **The budget** is the registered `metal` memory probe (S4's live ceiling: the smaller of 0.70 of RAM and the live
  available figure) less `decoder/metal_fitctx.go:MetalFitMarginBytes` (384 MiB, CUDA's margin). The margin is there
  because the load-time guard reads the live figure again after the context is chosen, and a context taken at the
  exact edge would be halved by S18's shrink on a drift of a few megabytes.
- **KV is priced as Metal allocates it** (`decoder/residentneed.go:Model.ResidentKVBytes`): f16, or int8 with
  `--kv i8`, for every requested slot (Metal's default is 2). The brief says "f32 KV"; Metal has no f32 KV kernels,
  so its examples' KV sizes are twice what will be printed.
- **Opt-outs keep 4096:** `--fit=off`; an unreadable memory probe; any load that pages MoE experts
  (`-moe-cache-experts`, `-moe-cache-slots`, or the deprecated environment knob), for CUDA's reason: the pager lives
  on the memory a longer context would take. A context the load-time guard pinned lowers the candidate and the
  floor, never raises them.
- **The decision rule above cannot be read** (it re-runs M0's task set, and M0 has not run). For this standalone M2
  the brief's gates replace it; re-running M0's set with no `--ctx` stays owed to M0.

*Gates.*
1. Unit tests, the CUDA set's Metal twins, on tiny fixtures: an unpinned load grows to 16384 when it fits and takes
   what fits when it does not; slots trade context, never under 4096; a guard pin only lowers; paged MoE, `--fit=off`
   and an unknown probe keep 4096; an explicit `-ctx` is untouched; `fit` and the load name the same context for one
   memory figure; a real tiny build holds the context it resolved and decodes an 11,137-token prompt in it. One
   mutation shown: with the paged-MoE opt-out removed a test goes red.
2. A real-checkpoint unit test (tag `realckpt`, heavy): on a 16 GB Mac's static budget the 1.5B and the 7B at int4
   resolve, with two slots, a context that holds 11,137 tokens.
3. On the Mac, no `-ctx`, by day where memory allows and otherwise in the night job: the 1.5B and the 7B at int4
   resolve at least 16384 with 2 slots; serve answers a request of about 11k tokens with no swap growth from before
   the load to after the answer (`docs/measurements/minimax-code-compat-2026-09/m2/m2_served_check.py`); a paged 26B
   load still resolves 4096.
4. **G-M2-speed**, below.
5. The Metal suite and the Mac device gate (`RELEASING.md` §C1-M) pass on the branch.

*G-M2-speed (night, the Mac).* A larger KV must not slow short turns.
- **Tier and instrument:** night; `scripts/bench_peer.py` phase A (depth 128, greedy, 64 tokens x 8 completions x 2
  runs a cell), `BENCH_BACKENDS=metal`, `BENCH_MODELS=1.5B,7B`, two arms `BENCH_ENGINES=goinfer,goinfer_old` (no peer
  arm: the decision is new / old), `BENCH_ABBA=3` (three counterbalanced quads of restarted cells a model),
  `BENCH_REQUIRE_BACKEND=1`. `goinfer` is serve built at this branch's head, `goinfer_old` serve built at main
  `c75a283a`; neither is given `-ctx`, so the old arm holds 4096 positions and the new arm what the fit gives.
  Script `docs/measurements/minimax-code-compat-2026-09/m2/run-g-m2-night.sh`, grader `grade_m2_speed.py` beside it.
- **Statistic:** per model, a block's rate is the mean of its runs; the median over the six blocks of new / old.
- **Rule:** SHIP at a median of 0.97 or more on both models (the brief's "within 3%"; a faster new arm also ships).
  **PARKED** in [0.95, 0.97) on either: one repeat on another night before any decision. **FAIL** under 0.95 on
  either: the default does not grow on Metal until the cost is explained.
- **Void:** the new arm's serve log does not show a context above 4096 in every cell (nothing was compared), the old
  arm's is not 4096, a cell errored, or a block lacks an arm. A void run is re-queued, not graded.
- **Stopping rule:** fixed, 24 cells; no early stop. **Cost basis:** about 30 s a 1.5B cell and 90 s a 7B cell with
  its load, 24 cells, plus the instant idle gate: about 30 min; with the served reads, a 45 min job.
- **Predictions, written before the run:** the median is within 1% of 1 on both models, 0.75; a slowdown between 1%
  and 3% on either, 0.15; over 3%, 0.10. At depth 128 a step reads the first 129 positions of each KV buffer, so
  the buffer's size should only matter through allocation. The 7B resolves 16384 with 2 slots at night, 0.75 (it
  needs about 6.8 GB of live budget: 4.5 GB of weights, 1.9 GB of KV and the margin).

**M2 results, 2026-10-12 (the day half; the branch `m2-metal-fit-ctx` is not merged until the night half reads).**

- **Built** as registered: `decoder/metal_fitctx.go:Model.MetalFitCtx` (the one function the Metal load and
  `goinfer-chat fit` ask), `metal/backend.go:metalFitFor` (chosen once a build), the banner's reason and `harness:`
  line (`internal/serveapp/banner.go:harnessLine`), the 400's wording, `internal/fitcmd/fit.go:planFor`. The design
  notes are in `docs/code-notes/metal.md`, "metalFitFor".
- **Gate 1, unit tests: PASS.** `decoder`'s six `TestMetalFitCtx_*` (grow and shrink on the 8-position grid, slots
  trade context, a limit only lowers, the three opt-outs with a control, the window, a companion's bytes);
  `metal`'s `TestResolveMetalCtxCap_fitByDefault` (fourteen loads, with the explicit `-ctx` cases),
  `TestResolveMetalCtxCap_autoPinIsACeiling` (re-read under a roomy and an unknown probe),
  `TestMetalBuildResident_fitByDefault` (a real tiny build holds 16384 in 2 slots and decodes an 11,137-token
  prompt; with room for 20,000 positions it holds 2 slots of 10,000; both are the context `fit`'s function names)
  and `TestMetalBuildResident_pagedMoEKeepsTheDefault` (a tiny MoE grows when resident and keeps 4096 paged);
  `internal/serveapp`'s banner and 400 tests; `internal/fitcmd`'s `TestPlanFor_metalPlansAtTheLoadsOwnContext`.
  - **Mutations, both red:** the paged-MoE case removed from `MetalFitCtx` (`TestMetalFitCtx_optOuts`: ctx 16384,
    want 4096), and `metalPagesMoE` made to return false (the two `-moe-cache-*` loads resolve 16384).
- **Gate 2, real checkpoints:** the 1.5B at int4 on a 16 GB Mac's static budget resolves **16384** with 2 slots,
  0.9 GB of KV (weights 1.81 GB and as much again in host copy, of 11.20 GB). The 7B's case loads it on the CPU and
  is in the night job.
- **Gate 3, the Mac by day** (M1 Pro 16 GB, the desktop open, 5.7 GB available; `m2/served-day-2026-10-12.txt`):
  - **The 1.5B, no `-ctx`: 16384 positions, 2 slots, 0.9 GB of KV.** A request of 11,492 tokens was answered in
    14.7 s; swap used was 4,459 MB before the load and after the answer.
  - **The 7B did not go resident at all:** the build needs 4.65 GB at its smallest context and the live budget
    was 3.21 GB. That is the memory guard, which this change does not touch (inferred from the code, not read on
    main today). Its read is in the night job.
  - **The paged 26B keeps 4096.** The owner had it pulled (17:00; 14.6 GB, bytes equal to the archive's). Loaded
    under the day-use rule with `-moe-cache-experts` and no `-ctx`:
    `context: 4096 tokens (paged MoE keeps the default; model maximum 262144 — raise with --ctx) · KV f16`. It went
    resident with 1 of its 2 KV slots (the guard's count at 880 MB a slot), the `harness:` line priced `--ctx 16384`
    at 3.4 GB of KV, and swap did not move. Banner only; no request was sent.
- **Gate 5, the Metal package's suite on the branch: PASS**, 318 tests pass and 173 skip (the heavy ones), by day
  (`m2/metal-suite-day-2026-10-12.txt`). A first run failed three tests for fixtures the fresh worktree lacked;
  they were cloned from the main checkout and the run repeated. The device gate (§C1-M) is a night job.
- **The night harness, one run a cell, exploratory and not a result:** the new arm resolved 16384 and the old arm
  4096, at 94.23 and 94.22 tok/s.
- **Owed, queued on the Mac for tonight:** G-M2-speed, the 7B's served read and real-checkpoint case
  (`m2/run-g-m2-night.sh`), and the Metal device gate at the branch's head. The merge waits for them.
- **For the owner:** the brief's "f32 KV" is f16 on Metal, so 2 slots of 16384 on a 7B are about 1.9 GB and not
  3.7. The 26B's sidecar (14.6 GB) and the 7B's Metal sidecar (4.8 GB) are now in `~/models`; the disk has about
  6 GB free.

**M2 results, the night of 2026-10-12 (the Mac, nothing else running): every gate that could be read passes.**
Records: `docs/measurements/minimax-code-compat-2026-09/m2/night-2026-10-12/`.

- **G-M2-speed: SHIP.** `bench_peer.py` phase A at depth 128, six counterbalanced blocks a model; the new arm
  resolved 16384 in every cell and the old arm 4096.

| model | new (tok/s) | old (tok/s) | new / old by block | median |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 93.40 | 93.39 | 1.0004 0.9992 0.9974 1.0025 1.0002 1.0016 | **1.0003** |
| Qwen2.5-7B | 30.70 | 30.68 | 1.0035 1.0003 0.9993 1.0011 1.0017 0.9997 | **1.0007** |

- **Gate 3, the 7B: 16384 positions, 2 slots, 1.8 GB of KV.** A request of 11,492 tokens was answered in 54.0 s;
  swap used was 962 MB before the load and after the answer. The 1.5B read the same as by day (13.6 s).
- **Gate 5, the Metal device gate at the branch's head: PASS** (`docs/measurements/gpu-gate/`, the record for
  `88647c7d`).
- **Gate 2's 7B case was not read, and the night job shows FAILED for it.** My test loads the 7B's GGUF onto the
  heap, and the load-time fit guard refused that on this Mac (12.8 GB needed, 6.4 GB available). That is the
  guard doing its job on my instrument, not a reading of the fit. The served read above is the direct evidence
  for the 7B: a real Metal load choosing its own context. The test now skips, with the reason, when the guard
  refuses; the 1.5B case passed.
- **Predictions scored.** "The median is within 1% of 1 on both models (0.75)": yes. "The 7B resolves 16384 with 2
  slots at night (0.75)": yes.
- **M2 is done.** Still owed to M0, not to M2: its task set re-run with no `--ctx`.

### M3 · A thinking switch

**Goal.** A client can turn a family's thinking off, and a server operator can set the default.

**Read first.** `chat/templates.go` (how each family's template renders; Qwen3's template takes an
`enable_thinking` variable, gpt-oss's harmony format takes a reasoning level), the tool-call union's
template census (`task-tool-grammar-union-2026-09.md` T4 — which families render what), and what pi-ai
sends per `thinkingFormat` (`providers/openai-completions.ts`: `enable_thinking` for `qwen`,
`chat_template_kwargs.enable_thinking` for `qwen-chat-template`, `reasoning_effort` for `openai`).

**Build.** Accept, in priority order, `chat_template_kwargs.enable_thinking`, top-level `enable_thinking`,
then `reasoning_effort` (`none`/`minimal` → off; `low`/`medium`/`high` → on, and mapped to gpt-oss's
reasoning level). A server flag `--thinking auto|on|off` sets the default when the request says nothing
(`auto` = the family's own default, today's behaviour). Families with no switch ignore it and say so in
`/v1/models` (`"thinking_control": false`). Same fields on `/v1/responses` (`reasoning.effort`) and
`/v1/messages` (`thinking: {type: "disabled"}`).

**Gate.** Per family with a switch: the rendered prompt with thinking off is byte-identical to the HF
template rendered with `enable_thinking=False` (the template-parity tests' method); a greedy reply to
"ping" on a Qwen3-class model contains no think block. M0's thinking-model `--use` cell passes with
`--thinking off`.

### M4 · Emit reasoning separately

**Goal.** Thinking text leaves the answer channel: minimax-code (and every pi-ai / llama.cpp-convention
client) shows it as reasoning and does not resend it as history.

**Build.** Parse the family's think delimiters at the stream boundary (the same holdback machinery that
already keeps stop strings and partial UTF-8 out of chunks) and emit `delta.reasoning_content` on
`/v1/chat/completions`, `thinking` content blocks on `/v1/messages`, reasoning items on `/v1/responses`;
non-streaming responses carry `message.reasoning_content`. On input, drop `reasoning_content` from resent
assistant messages before templating (the templates that re-render thinking decide that themselves).
`--reasoning-format none` keeps today's inline behaviour.

**Gate.** Stream reassembly tests: the concatenation of reasoning + content deltas equals today's single
content stream byte for byte, across every chunk boundary inside a delimiter; a tool call emitted after
a think block still streams as `delta.tool_calls`. `usage.completion_tokens` unchanged (reasoning counts,
as OpenAI counts it).

### M5 · Images together with tools

**Status: verify-then-decide.** The refusal is deliberate: the vision path renders no tools, so a
tools+image request would silently drop the tools. Lifting it needs the VL templates (Gemma 3, Qwen2.5-VL,
Qwen3-VL) to render tools and the vision drive path to run the tool turn. Decide after M0 whether
minimax-code users actually paste images; if the answer is rarely, keep the refusal and document it in M1.

### M6 · `/v1/responses` tolerates what pi-ai sends

**Build (verify first).** Capture one real mcode turn with `--api-format openai-responses` against
goinfer (M0's harness, request logging on); list every input item type and top-level field; make the
route accept them — `reasoning` items skipped on input, `include`/`store`/`prompt_cache_key` ignored,
`function_call`/`function_call_output` round-trip (already implemented) — and record which were already
fine. **Decision:** if the captured shape needs more than skip-and-ignore, say so in M1 and recommend
openai-completions instead.

### M7 · `/v1/messages` tolerates what pi-ai's Anthropic client sends

**Build (verify first).** Same capture with `--api-format anthropic-messages` (note minimax-code rewrites
the base URL for this format — `normalizeMessagesBaseUrlForPi`; confirm the path it actually posts to).
Expected: `cache_control` markers, `anthropic-version`/beta headers, `thinking` config, tool definitions
with `input_schema`. Ignore what goinfer cannot use; M3 maps `thinking`. Record the result in M1.

---

## Sequencing

M0 and M1 together (a day; the page ships with the measurement in it). Then M3 (it fixes both the
connection test and every thinking-model turn), M2, M4. M6 and M7 are an hour each once M0's capture
harness exists; M5 waits on M0's evidence.

## What this doc does not claim

- That goinfer has been run with minimax-code. §0 is both codebases read side by side; M0 is the run.
- Anything about agent quality. A 7B coder model's decisions inside a harness with eight-plus tools
  are the model's, not the server's; M0 records completion rates so the page can say what to expect.
- That minimax-code should change. Reading `/v1/models`' `context_window` would remove item 1 of §0 on
  the client side; that is theirs to decide, and M2 does not depend on it.

<!-- doc-reviewed: 2026-09-25 -->
