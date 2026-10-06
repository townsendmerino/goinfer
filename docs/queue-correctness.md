# Correctness queue

Parity, numerics, goldens, quantization, model families. Anything whose success criterion is **agreement with a reference** — a cosine, an argmax match, a golden. If the question is *does it compute the right thing*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.


## Queued

> **ELEVEN closed entries are archived** in
> [`docs/completed/queue-correctness.md`](completed/queue-correctness.md) — G4, G5, G6, G9, G10,
> G11 and Q2 (2026-08-31, first pass), then **G7, G1, Q3 and Q1** (2026-08-31, second pass, when
> the last of them closed).
>
> **What is below is the open work, and it is one PARKED item.** G8 is not queued-and-waiting: it
> is blocked on hardware that does not exist here (V4-Flash is 0.16 TB against 62 GB of host RAM),
> so nothing in this queue is startable today. That is a real state, not an empty-queue
> formality — read G8's entry for what would unpark it.

**G12 · LFM2/LFM2.5 GGUF loader — safetensors-only today, llama.cpp already supports it** — `cpu`.

`docs/completed/scoping-lfm2.md`'s own scoping never owned this — it is the one remainder its
"GGUF status" line named and did not scope. G1 (`7098769`, 2026-08-31) shipped LFM2/LFM2.5
safetensors-only; the capability matrix still shows only `safetensors` for `lfm2`
(`docs/capability-matrix.md:111`), and `ggufConfig`'s architecture switch
(`decoder/gguf.go:50-88`) has no `"lfm2"` case — a GGUF checkpoint for this family fails to load
today with "architecture unsupported", listing every arch that DOES have one. llama.cpp supports
arch `lfm2` natively and an official `LiquidAI/LFM2.5-2.6B-GGUF` checkpoint exists, so the gap is
real: a user reaching for LFM2.5 via `goinfer-chat pull` (the GGUF-first surface most pulls use)
cannot get it. The family's numerics are already validated one tier past this doc's own
experimental ceiling (T3 full-forward-oracle, cosine 1.0, `testdata/parity_manifest.json`'s
`lfm2` row, 2026-09-07), so this is pure loader plumbing — an `lfm2Config` case in the same shape
as any other GGUF-native family already there (see `ggufMellumConfig`/`ggufGraniteDenseConfig`
for the closest-shaped precedent: dense, no MoE routing table to reconcile), not a new numeric
frontier.

**G8 · DeepSeek V4-Flash as a new family** — `any`, **PARKED — and the reason has CHANGED
(2026-08-31). It is no longer "lowest priority"; it is UNVALIDATABLE on any hardware here.**

> **Two things were checked before treating Q3 as the unblock, and both move the item.**
>
> **1. The fp8 prerequisite is only HALF lifted.** Q3 shipped e4m3 with blockwise **f32** scales
> read from `*.weight_scale_inv`. V4-Flash's config declares **`scale_fmt: "ue8m0"`** — an
> exponent-only 8-bit scale encoding — and `grep` for `ue8m0`/`scale_fmt` across `decoder/`
> returns NOTHING. So "Q3 unblocks G8" is true of the element format and unproven of the scale
> format. Anyone picking this up should confirm what V4 actually stores before assuming the
> reader covers it.
>
> **2. The decisive blocker is now HARDWARE, and it is not close.** V4-Flash is **0.16 TB** — the
> SMALLEST real member (V4-Pro 0.86 TB, Kimi-K3 1.56 TB). Against 62 GB of host RAM on the box,
> 8 GB of VRAM, and 16 GB unified on the Mac, there is no configuration in which a whole V4
> forward runs here. Expert streaming does not rescue it either: `--moe-cache-experts` needs the
> weights resident in HOST memory, which is the 62 GB that 160 GB does not fit in.
>
> **WHY THAT IS DISQUALIFYING RATHER THAN INCONVENIENT — G7 just paid for this lesson.** G7's
> gate was ONE real forward, and running it found THREE silent defects (a slot-indexed bias
> table, a never-applied per-expert bias, an unloaded router kernel) that `./cuda/...` at 100
> PASS could not see, because every one was a term the WIRING dropped rather than a kernel
> computing it wrongly. G8 is EIGHT new primitives — DSA sparse attention over a learned Indexer,
> strided KV compression, sliding-window + attention sink, grouped low-rank output projection,
> hash routing, `sqrtsoftplus` scoring, hyper-connections, clamped SwiGLU. Building that with no
> possible end-to-end run reproduces G7's failure mode across eight times the surface, with no
> way to detect it. Kernel-level gates would all be green.
>
> **So the honest state is not "later, it is low value" but "not yet, the proof is unreachable".**
> The strategic case in the original entry stands — DSA is where V3.2/GLM-5.1/V4 are converging,
> and building the compressor path once plausibly buys several frontier releases. What changed is
> that the entry read as a priority call when it is really a hardware precondition, and a reader
> deciding what to do next should see the difference.
>
> **What would unpark it,** in order of cost: a machine with ≥256 GB host RAM (V4-Flash then fits
> for a CPU-reference run, which is the arm the gate actually needs); or a released sub-100B
> member of the family; or upstream publishing a small DSA reference the primitives could be
> validated against piecewise — the last being the only one that does not require new hardware,
> and the one worth watching for.

**Original entry follows.**

**G8 (original) · DeepSeek V4-Flash as a new family — blocked on fp8 support, post-1.0** — `any`.

Scoping already done: `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`'s Phase 0 verdict.
**Not** a `deepseekArchitecture` alias — eight new primitives (DSA sparse attention over a learned
Indexer, strided KV compression, sliding-window + attention sink, grouped low-rank output
projection, hash routing, `sqrtsoftplus` router scoring, hyper-connections, clamped SwiGLU).
**Hard prerequisite, not a subtask:** V4-Flash ships fp8 e4m3 blockwise-quantized weights and
**there is no fp8 support anywhere in the tree today** — file/estimate the fp8 reader as its own
piece of work before scoping the primitive additions. MIT license, DeepSeek's brand pulls the
whole local community, and native sparse attention is where the field (V3.2, GLM-5.1, V4) is
converging — building the DSA/compressor path once plausibly buys the next several Chinese
frontier releases, which is the strategic case for filing this now even though it's not a
near-term ship. Lowest priority of the five items filed alongside this one (`G4`-`G7`).

**G39 · A chatml model that answers a tool request with a fenced JSON call in prose makes no tool call — an owner decision before any code** — `any`, **filed 2026-10-06**

Found by the cold-user run (`docs/measurements/cold-user-2026-10-05-nobara-pc.md`, dead end B): opencode on Qwen2.5-Coder-7B-Instruct (q4_k_m, CUDA, `-ctx 16384`) made one real `read` call, then wrote its `edit`
call as a ```json fenced block in prose, twice, and no edit happened. On the same server `serve check` failed 2 of 9 rows: `tools, OpenAI` (turn two asked for the tool again instead of answering: the agent-livelock
shape) and `stop sequences` (the reply never reached the stop sequence). Qwen2.5-7B-Instruct on the same card passes all of it (`docs/integrations/opencode.md`).

What the code does today: on chatml a call is constrained from the `<tool_call>` opener, and `AcceptsBareToolCall` additionally parses an UNWRAPPED call, but only when the output's first non-space byte
is `{` (`chat/tools.go`, `NewBareAwareProseStreamer`). Prose that later contains a fenced `{"name": "edit", "arguments": {...}}` is parsed as prose. The README and the opencode recipe now say which checkpoint is
measured to work (70148d89); that is the whole of the fix so far.

**The option, and why it is not just built.** Accept a ```json fence whose object names a supplied tool and parses as `{name, arguments}` as a call. It would have turned both opencode attempts into edits. It also
turns a model that was asked to SHOW an example call, in a fenced block, into one that executes it, and an agent that executes a tool it was only asked to describe is worse than one that does nothing. That
trade (tolerance against a false execution) is the owner's, not an implementation detail. If taken: pre-register a held-out set of fenced-example prompts that must NOT parse as calls and the opencode
transcripts that must, restrict it to a fence that is the whole reply or its last block, and gate it on `serve check`.

**The two real replies (from the archived raw logs, `~/goinfer-logs/cold-user-raw/cold-user-2026-10-05-raw.tgz`, `B-agent/oc-run1.log`, `oc-run2.log`).** Attempt 1: one sentence of prose ("The `add` function should subtract instead of add. I'll update the function and verify it."), then the fenced
`{"name": "edit", "arguments": {...}}` block, then nothing. Attempt 2, told to use the edit tool and not describe the change: the whole reply was that one fenced block. So a "reply must start with the fence" rule (the analogue of the bare-`{` rule) catches only attempt 2,
and "the fence is the last thing in the reply" catches both. A demonstration ("Here is an example call: ```json ...```") has the same shape as attempt 1, so no parse rule separates a shown call from a meant one; the residual false execution is a demo that ends on the fence. Options in order of
strength: opt-in flag (off by default); the narrow match above plus validating the arguments against the named tool's schema; and the client's own confirmation prompt, which is not verified here for opencode.

**BUILT 2026-10-06 (owner chose "opt-in flag plus the narrow match with schema validation"): `serve -lenient-tool-calls`, off by default** (`chat/fenced_tool_calls.go`, `chat.Template.WithLenientToolCalls`, a fence-aware prose streamer, `docs/server.md`).
The pre-registered cases (`chat/fenced_tool_call_test.go`, committed as 4015c5ec before the code) pass: both real replies parse, 22 shown-example and near-miss shapes stay prose, other families are untouched, and
the streaming prefix guarantee holds under byte-at-a-time chunking; each clause of the rule goes red when removed. One pre-registered assertion was wrong (it claimed no family parses a fenced call by default, but llama3's
parser has always found a bare JSON call anywhere) and was narrowed to the families the rule is for. **Measured the same day under real opencode** (`docs/measurements/lenient-tool-calls-opencode-2026-10-06/`, exploratory, n = 8 per arm, not statistically resolved): with an explicit read-then-edit prompt the file was fixed in 3 of 8 runs with the flag and 0 of 8 without, with a
vague prompt in 0 of 3 either way, and `serve check`'s `tools, harness-scale` row went from skip to ok. The flag turns the fenced calls into calls; the model still fails most runs for its own reasons (asks the user for the file, answers in prose),
and `tools, OpenAI` turn two and `stop sequences` are unchanged, so those two are not the parser's. The residual false execution (a demonstration that ends on one valid call) is accepted as the price of the option and is why it is off by default.

**DIAGNOSED AND FIXED 2026-10-06 (`docs/measurements/g39-lone-tool-livelock-2026-10-06/`): the `tools, OpenAI` failure was goinfer's, not the model's.** A lone tool under `tool_choice` auto was forced on every turn, including after the tool result; the Instruct model
failed the same row on this tree. Fixed so the convenience ends at the tool result (c69a6f87 pre-registration, then the fix). The `stop sequences` failure is the Coder model's own (it answers "The count is now 10." to a count prompt) and is not a goinfer defect. G39 is closed apart from
the parser's residual risk, which is the owner's call and is behind an off-by-default flag.
