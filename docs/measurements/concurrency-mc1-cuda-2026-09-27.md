# MC1 on CUDA — multi-slot resident KV: interleaved conversations hold the 1-client figure (1.25× at 4 clients) (2026-09-27)

MC1 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), on the CUDA backend: N conversations'
KV stay resident at once, so interleaved turns reuse their own prefix instead of evicting each other. Generation is
still one at a time; CUDA has no batch stepper. Built in `9fddaf7e`. The grading was pre-registered in `8c8db0fe`
before any timing, and the Metal twin is [`concurrency-mc1-2026-09-26.md`](concurrency-mc1-2026-09-26.md).

**Result: every hard gate passes, and it ships.**
- With 4 resident KV slots (`-kv-sessions` default 4), the aggregate at 2 and 4 clients is **1.008× / 1.014×** the
  1-client figure.
- The previous build, with one slot, reads 0.805× / 0.808×.
- **4-client aggregate: 217.9 against 174.2 tok/s, 1.250× (median of 3 interleaved pairs: 1.250, 1.255, 1.249).**
  2 clients: 1.254×.
- Every client reuses its whole history on every turn. The slowest turn at 4 clients falls from 3.72–3.74 s to 2.67–2.68 s.
- A lone request is unchanged: p50 1.007×, p99 1.002×.
- **On the 7B the clamp bound at the graded build (`9fddaf7e`), and the win did not appear at the default
  context.** 2 of 4 slots fit at 8192 tokens, and 4 clients rotating over 2 slots thrash: 55.8 against 55.5 tok/s.
  An exploratory pair, *not pre-registered*, at `-ctx 4096` fits 4 slots and reads 74.7 against 56.0 tok/s (1.33×).
- **Owner decision, the same day: a slot request shrinks the unpinned default context** (`947e06ce`). The 7B now
  starts at 4096 with 4 slots and reads **74.7 against 55.6 tok/s (1.34–1.35×, two interleaved pairs)** at its
  default. The 1.5B keeps 8192 and 4 slots. See "The 7B" and "The context policy".

## Setup

- nobara-pc: RTX 2070 SUPER 8 GB, NVIDIA driver **595.91.07**, compute mode Default. Ryzen 7 3700X, Nobara 44.
- Model: qwen2.5-coder-1.5b-instruct q4_k_m from `~/models` (NVMe), `-quant int4`, and serve defaults. The context
  is fit-by-default 8192 on both builds, and the KV is f32.
- `scripts/bench_w7_plain.py --engines goinfer --backend cuda --fixed-nonce`: 6 plain turns per client, 128
  greedy tokens per turn, and a fresh server per cell.
- Two builds, each built once with `CGO_ENABLED=0 go -C cuda build -tags cuda` from a clean worktree
  (`vcs.modified=false` in both binaries), and named by hash:
  - **old** = `serve-cuda-b2168bdc`, the tree just before this work, where the CUDA resident has one KV slot;
  - **new** = `serve-cuda-9fddaf7e`. Its banner reads "4 conversations kept resident" (448 MB of KV each at 8192).
- The harness JSON header's `goinfer_dirty: true` is the harness's working tree, which held untracked suite logs. It
  is not the binaries.
- Cells: 1, then 2, then 4 clients, each old/new × 3 pairs in the order old new new old old new. Then one 4-client
  pair on the 7B, where the clamp should bind.
- Each server start waited for all of: load1 ≤ 2.0 (`/proc/loadavg`); no CUDA compute process beyond those present
  at the start (`kwin_wayland`, the compositor, is always one); and GPU memory within 256 MiB of the 639 MiB
  baseline.
  - The first launch required *zero* compute processes, which the compositor never allows. It was stopped before
    any cell ran and relaunched with the gate as described (`w7-run.log`).
  - The committed pre-registration's `run-w7.sh` carries the first version of the gate; the archived one carries
    the fix. The gates are unchanged.
- One session, 2026-09-27 09:25–09:33 PDT.
- Raw data: [`concurrency-mc1-cuda-2026-09-27/`](concurrency-mc1-cuda-2026-09-27/):
  - `w7.json`, `w7-7b.json` and `w7-7b-ctx4096-exploratory.json`;
  - `w7-run.log`, and `w7-servers*.log` (every banner);
  - `gates.py` and `gates-output.txt`;
  - the identity-test and suite logs.

## Results (qwen2.5-coder-1.5b)

| build | clients | aggregate tok/s (3 cells) | ÷ its 1-client (median) | prompt − reused, turns 2–6 (every client) | slowest turn |
|---|---:|---|---:|---|---:|
| old | 1 | 206.90, 215.17, 215.42 | — | 26, 27, 24, 20, 27 | 0.67 s |
| old | 2 | 172.92, 173.14, 173.01 | 0.805× | 176, 331, 483, 631, 786 | 1.87 s |
| old | 4 | 174.64, 173.80, 174.07 | 0.808× | 176, 331, 483, 631, 786 | 3.74 s |
| **new** | 1 | 215.99, 215.06, 213.36 | — | 26, 27, 24, 20, 27 | 0.67 s |
| **new** | 2 | 217.36, 216.85, 217.02 | **1.008×** | 26, 27, 24, 20, 27 | 1.34 s |
| **new** | 4 | 218.30, 218.11, 217.41 | **1.014×** | 26, 27, 24, 20, 27 | 2.68 s |

"Prompt − reused" is how many tokens each turn prefilled. At 2 and 4 clients the old build re-prefills each
conversation's whole history on every turn, reusing only the 7-token template lead. The new build prefills only the
new user turn, exactly as at 1 client.

The first old 1-client cell (206.9) is the session's first server start; the other five 1-client cells sit at
213–216.

The new build's 2- and 4-client figures sit slightly *above* its 1-client one (1.008× / 1.014×). This is the top edge
of the expected 0.90–1.0× band, not a finding; the band's floor is where a finding would be. Metal's MC1 read 1.007×
at 4 clients the same way. The likely reason is that the 1-client aggregate carries each turn's fixed per-request cost
alone, while more clients overlap one request's HTTP and tokenization with another's decode.

## Gates (pre-registered in `8c8db0fe`; `gates-output.txt`)

| gate | result |
|---|---|
| 1. identity: tests; at 1 client old == new `content_sha` every turn; at 2 and 4 clients new's client 0 == new's 1-client run | **pass.** `TestCUDAKVSlots_interleavedMatchesAlone`: ids identical on llama-tiny, mistral-tiny-window, deepseek-tiny (MLA), and qwen2.5-coder-1.5b int4 (48 tok × 3 turns × 2), and the one-slot control thrashes. `_graphs` (graphs forced on), `_clampedBuild` and `_pricedAgainstWhatIsLeft` pass. In W7 every compared turn matches. |
| 2. reuse: new's prompt − reused at 2 and 4 clients equals its 1-client run's, turn for turn | **pass**: turn 1 reuses 0 on every client, and turns 2–6 prefill 26, 27, 24, 20, 27 on every client of every cell |
| 3. *(expected band)* new 2/4 clients ÷ new 1 client: 0.90–1.0× | 1.008× / 1.014× (above the top edge; see above) |
| 4. ship: 4-client aggregate new ÷ old, median of 3 ≥ 1.03× | **1.250×** (1.250, 1.255, 1.249) |
| 5. solo guard: 1-client p50 and p99 turn new ÷ old, median ≤ 1.05× | **p50 1.007×, p99 1.002×** |

Reported, not gated: at 2 clients new ÷ old is 1.254×. Old against new at 2 and 4 clients differs in content on 3 of
12 and 7 of 24 turns, the same turns in every pair. The old build re-prefills a conversation's history through the
batched prefill, and the new build keeps the KV that decode wrote. The two need not agree bit for bit, and the
pre-registration made this report-only for that reason.

## The 7B: the clamp binds (reported)

qwen2.5-7b-instruct q4_k_m, 4 clients, one pair:

| build | context | slots (banner) | aggregate | prompt − reused, turns 2–6 | slowest turn |
|---|---|---|---:|---|---:|
| old | 8192 (fit by default) | 1 | 55.47 | 176 … 786 | 12.06 s |
| new | 8192 (fit by default) | **2 of 4** ("the memory guard allowed 2") | 55.79 | 176 … 786 | 12.05 s |
| *old (exploratory)* | *4096 (`-ctx`)* | *1* | *55.96* | *176 … 786* | *12.04 s* |
| *new (exploratory)* | *4096 (`-ctx`)* | *4* | ***74.71*** | *26, 27, 24, 20, 27* | *7.20 s* |

- **The clamp works.** The log names its numbers: "2 resident KV slots of 4 requested — each costs 896 MB of KV at
  the resident context 8192, and free VRAM beside the weights (2565 MB, less 384 MB reserved) allows 2". The banner
  names the clamp.
- It is the first clamping load of MC1 on any backend.
- **Two slots do not help four round-robin clients.** Under LRU, each turn's slot was taken by the two
  conversations in between, so every turn re-prefills, as with one slot.
- **The context is resolved first, then the slots.** That order is inherited: fit by default picks the largest
  context up to 8192 for one slot, then the slots are clamped to what is left.
- The exploratory pair shows the other trade. At 4096 tokens all 4 slots fit, and the 7B gets MC1's full effect:
  1.33×, full reuse, and the slowest turn 12.0 → 7.2 s.
- Which to prefer when slots are requested, context or conversations, was put to the owner. The owner chose
  conversations the same day; see the next section.
- This pair was run after the pre-registered cells, as exploration. It is one pair, and no gate reads it.

## The context policy (owner decision 2026-09-27, `947e06ce`)

"Should asking for slots shrink the default context — yes." An unpinned CUDA load that requests N slots now gives up
context before conversations. `resolveCtxCapFit` → `ctxForSlots` takes the largest context in [4096, the one-slot
choice] at which `Plan` fits the weights plus N copies of the KV. If not even 4096 holds them, it takes 4096 and the
build clamps the slot count, as before. It is untouched by:
- an explicit `-ctx`;
- `--fit=off`;
- expert streaming.

The shrink is logged, and the banner's context line shows the result.

Checked on the 7B, 4 clients, default context, interleaved old new new old, idle-gated as above (`w7-7b-ctxpolicy.json`,
`w7-servers-ctxpolicy.log`):

| build | context (banner) | slots | aggregate | prompt − reused, turns 2–6 | slowest turn |
|---|---|---:|---:|---|---:|
| `9fddaf7e` | 8192 | 2 of 4 | 55.45, 55.80 | 176 … 786 | 12.06 s |
| `947e06ce` | **4096** (logged: the floor, not 8192, for the 4 requested slots) | **4** | **74.72, 74.75** | 26, 27, 24, 20, 27 | 7.21 s |

- That is 1.35× and 1.34× per pair, with every client reusing its whole history.
- The 1.5B's banner at `947e06ce` is unchanged: 8192 tokens and 4 slots (`banner-1.5b-947e06ce.log`). Its graded
  cells above stand.
- The tagged CUDA suite at `947e06ce` has 172 pass and 0 fail (`cuda-suite-tagged-947e06ce.log`).
- `TestResolveCtxCapFit_slotsShrinkTheContext` pins the rule on a forced budget:
  - one slot takes 8192 and 4 slots take 5000, where exactly 4 fit;
  - checkKVFits grants 4 there and fewer at 8192;
  - a floor-bound budget returns 4096 and the build clamps;
  - an explicit 8192 stays 8192.

**The shrink is conservative.** `Plan` prices the 7B's device weights at 4930 MB, but the build allocates about
4476 MB before its KV (7041 MB free before the load, 2565 MB beside the weights). So `Plan` says 4 slots do not fit
even at 4096, and `ctxForSlots` takes the floor. The build, pricing against real free VRAM, grants all 4 there, and by
its own arithmetic would have held them to about 4,870 positions. The result is 4 slots at 4096 rather than at about
4,870. Tightening `Plan`'s weight estimate is its own item.

## Scope, stated

- **CUDA, one generation at a time.** CUDA has no `ResidentBatchStepper`, so `-max-concurrent` stays at one there,
  as its banner says. MC3's batching is a separate item, only on its own measurement.
- **One slot for:**
  - a DeltaNet resident, and the recurrent families generally (decoder rule, and held on CUDA too);
  - expert streaming (`MoECacheExperts`), whose cache takes whatever VRAM is left.
- **CUDA graphs** are unaffected. The captured segments touch no KV, and the identity test runs with graphs forced
  on.
- **The decoder suite** was started before timing and stopped at 366 pass / 0 fail, by owner decision, so the timing
  could run on an idle box. No `decoder/` file changed. The full suite was re-run after timing at `9fddaf7e`:
  **700 pass, 0 fail, 103 skip, 1227 s** (`decoder-suite.log`).
