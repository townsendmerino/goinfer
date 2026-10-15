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

**G44 · Metal re-quantizes a Gated-DeltaNet model's int8 weights to int4 instead of declining** — `metal`, **filed 2026-10-15
(owner); built 2026-10-16 (branch `g44-deltanet-int8-decline-2026-10`)**

Metal's native int8 path covers no recurrent family (`metal/model.go:w8Eligible`: its first slice excludes DeltaNet), so a Qwen3.5,
Qwen3-Next or Olmo Hybrid model loaded at int8int8 runs resident with its attention, MLP and DeltaNet projections re-quantized to
int4 (`upload` and `fuse` fall back to `int4BufA` / `int4ConcatA`). Checked 2026-10-15 on `qwen35-tiny`: resident at int8int8, decode
path "metal-resident (int8int8→int4, no Metal int8 GEMV kernel)". The path label names it, but the model does not run at the int8 it
was loaded with.

**The fix, as Nemotron-H's (OP2b, `metal/mamba.go:buildMamba`):** at int8 Metal declines the model as resident and the CPU runs it
at real int8. Native int8 for recurrent families is a later slice of `docs/tasks/task-metal-int8-2026-10.md`.

**Gate (tier 0):** `qwen35-tiny` at int8int8 does not go resident on Metal; at int4 it does, unchanged. Planted defect: no decline,
which must turn it red.

**Built 2026-10-16:**
- **The decline:** `metal/deltanet.go:buildDeltaNet` declines when `recurrentInt8Load` says the load is int8, int8int8 or int4mix.
  - That predicate reads the load's quant label (`decoder.Model.Quant`), not `ResidentProjsInt4`. The latter reads `qwen35-tiny`
    (a MoE: no MLP up-projection, attention outside `QProj`) as non-int4 even at int4, which would decline every load.
  - `metal/mamba.go:buildMamba` takes the same predicate as well, so Nemotron-H at int4mix (int4 MLP, int8 attention) now
    declines too, where its int8 attention used to be re-quantized.
- **Gate:** `TestG44_deltaNetInt8Declines` PASS. int8int8 gives the int4-only decline, and through the backend it runs on
  the CPU. int4 builds. The planted defect (`dnetInt8DeclineOffForTest`) turns it red.
  - int4mix is GGUF-only, so the tiny safetensors fixtures cannot load it. Its decline is the same predicate, untested here.
- **`TestQwen35ResidentParityMetal`'s dense fixture moves from int8int8 to int4,** the precision Metal now runs. It had been
  measuring the re-quantized path. Worst cosine 0.999889 over 128 tokens, argmax 125/128; the MoE fixture reads 0.991138, unchanged.

**G41 · An answer is lost inside the reasoning when a thinking model ends its turn mid-reasoning** — `any`, **filed 2026-10-15
(owner), not started**

A thinking model can end its turn while still inside its reasoning block, with the answer written there and nothing after it. Since
PR #11 the reply is no longer silently empty: `/v1/responses` returns `incomplete` with `ended_in_reasoning`, and a tool call written
in the reasoning is taken from it. But a prose answer written in the reasoning still never reaches the client. Seen in the I3 after
run at `695ce16c`: Codex x Mellum2.1 `read` r3 failed on an empty final message
(`docs/measurements/integration-harness-2026-10/i3-night-2026-10-13.md`, the after run).

**Proposed mechanism (owner):** when the model ends its turn inside an open reasoning block, close the block server-side and let the
model continue, as the thinking budget already forces a close.

**Gate, registered before any code:** answers appear where there were none; turns that closed their reasoning on their own are
unchanged, byte for byte. The rule, the prompt set and N are written before any run.


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
(`decoder/gguf.go:ggufConfig`) has no `"lfm2"` case — a GGUF checkpoint for this family fails to load
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

**G40 · CPU cold prefill: a long prompt's scratch is not bounded and is not returned to the OS, and it breaks never-swap** — `cpu`, **filed 2026-10-14**

Found by KV9 step 0 (`docs/measurements/kv9-2026-10/README.md`, Finding 3), by day on the M1 Pro 16 GB:

- One cold 20,193-token prompt on Qwen2.5-Coder-0.5B (`serve -backend cpu`, int4 weights from the CPU sidecar, f32 KV)
  took the server to a 3.1 GB footprint for 0.47 GB of KV, and grew swap by 1.27 GB.
- At 5,778 tokens it reached 1.0 GB for 0.14 GB of KV.

The rest is the batched prefill's scratch, which Go's heap keeps after the request:

- the footprint did not fall in 60 s of idle;
- later sessions' KV landed in it without moving it.

That breaks the never-swap rule (`docs/tasks/task-never-swap-2026-09.md`). Serve's fit line budgets "a request's own
prefill", but nothing bounds the prefill by that budget.

**Two parts:**

1. **Bound the prefill scratch per chunk.** A long prompt's batched pass allocates in proportion to its length; cap
   it by chunking the pass.
2. **Return memory to the OS after a long prompt,** as the GPU residency build already does
   (`decoder/residency.go:releaseHostMemory`).

**Gate:**

- bit-identity of logits and KV against today's unchunked pass at the chunk boundaries;
- the 20K cell of `kv9_mem.py` re-read with no swap growth and a footprint bounded by weights + KV + one chunk's
  scratch.

**G41 · CPU serve: interleaved conversations that share a preamble take over each other's session and lose their tails** — `cpu`, **filed 2026-10-14; revisit with KV2's first-prompt anchor**

Found by KV9 step 0 (`docs/measurements/kv9-2026-10/README.md`, Finding 1).

**What happens.** `internal/serveapp/sessions.go:pickSession` gives a new conversation the only idle session that
holds the preamble, rewound in place, whenever the prompt keeps more of that session than it discards. So:

- a second conversation sharing a 5.7K preamble with a short tail took over the first's session;
- the first, continuing, re-prefilled its own tail: 5,761 of 5,801 tokens reused, against 5,786 when it kept its
  session.

On a 32K-context model with a 20K agent preamble, every second conversation qualifies: its tail is always shorter than
the preamble. Two agents taking turns on CPU serve therefore re-prefill each other's whole conversation past the
preamble on every switch. That costs minutes on the CPU.

**The other trade.** KV2's copy (on `kv2-2026-10`, not merged) makes it possible: a fresh session with the preamble
copied in, keeping both tails, at one preamble's memory per conversation.

Revisit when KV2 merges, together with its first-prompt anchor, which keeps a conversation's own rollback in place.
The rule is the owner's to choose:

- takeover (one preamble's memory, tails lost);
- copy (tails kept, one preamble per conversation);
- KV9's host blocks (both). This is the case the KV9 record names for reopening KV9.

**G42 · CPU fast prefill attention on a sliding-window layer is not chunk-invariant, so G40 cannot chunk those caches** — `cpu`, **filed 2026-10-15**

Found by G40-1 (`docs/tasks/task-cpu-prefill-memory-2026-10.md`, "Amendment, 2026-10-15, after G40-1's first read", on
branch `g40-cpu-prefill-memory-2026-10`):

- On `gemma2-tiny` (sliding window 4), a prefill run in chunks under fast attention differed from the single pass.
  At a suffix of 529 with C = 256, 446 of the last row's logits differed. The first difference was layer 1's keys at
  the second chunk's rows, so layer 0, the sliding-window layer, gave a different attention output.
- The exact path chunked identically. Dense and MoE fixtures chunked identically under fast attention too.

The fast local attention's result depends on where the pass's assembled key window starts
(`decoder/forwardn.go:Model.runLayersFromEmbedN`, its local branch: `KVCache.batchReadLocal` then
`attendBatchedHeads`), and a chunk boundary moves that start.

**Consequence:** G40 keeps the single pass for a sliding-window cache under fast attention. So the memory bound G40
adds does not reach Gemma 2, 3 and 4, or any other family with sliding-window layers, prefilled fast on the CPU. That
is where G40's high-water stays.

**To fix:** make the fast local attention's arithmetic independent of the window's start, for example by aligning its
key blocks to absolute positions rather than to the assembled window. Then drop the exclusion in
`decoder/forwardn.go:Model.prefillLogits`.

**Gate:** G40-1 unchanged, with `gemma2-tiny`'s fast cases chunking, and its pass count asserting they do. The fast
path's own fidelity record is re-read on the 1.5B if the arithmetic changes.
