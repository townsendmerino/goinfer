# Task: fit to hardware — the machine picks the configuration, not the user

> **Status (corrected 2026-09-09 — the line below was stale since 2026-09-06 and nobody updated
> it): Phase 0 (accounting) is DONE for the CPU staged path** (`decoder/fitguard.go`, `db61c83`,
> 2026-09-06 — landed three days before this doc's own last edit and was never checked off here)
> **and its M-01/M-02 dependency is DONE/mostly-done** (M-01 fixed 2026-09-03; M-02's Metal
> ordering/paged-accounting half fixed 2026-09-03, its KV/scratch/host-copy gaps closed
> 2026-09-09, and CUDA's previously-nonexistent fixed-term guard added 2026-09-09 — see
> `docs/task-gpu-paths-2026-09.md`'s G11 entries for both). Phase 0's shape differs from what this
> doc originally specified (no single `WeightBudget()`; the CPU path uses `fitCheck`/
> `estimateGGUFWeightBytes` instead, and CUDA/Metal each kept their own separate guard rather than
> unifying on one). **Phase 1 is DONE for its `plan()` core and `goinfer-chat fit` dry run**
> (`decoder/fitplan.go` + `internal/fitcmd`, 2026-09-09; verified end-to-end on real Mac/Metal and
> nobara/CUDA hardware, including the exact gemma-4-26B expert-cache scenario §4 uses as its worked
> example) — Load()-based rather than header-only, see `docs/task-gpu-paths-2026-09.md`'s entries
> for the scope decision. **The startup banner (Phase 1's third surface) is NOT started**, nor are
> `pull`'s verdict / the web UI listing (both need the still-deferred header-only work). **Phase 2
> is DONE for CUDA and Metal** (context + slots + the real `--fit=off` switch, all measured on real
> hardware, 2026-09-09) **and for CPU's placement piece** (a dense `.gguf` that will not fit
> resident RAM gets one automatic `-stream-weights` retry, gated by `--fit`, MoE deliberately
> excluded after `docs/benchmarks.md`'s "M35/M26 on the Mac" measured that CPU path as
> catastrophic — 2h10min/zero completions; the dense retry itself measured at streamed/resident =
> 0.944, real hardware, 2026-09-09 — see `docs/task-gpu-paths-2026-09.md`'s entries) **and for the
> drafter-aware companion-allocation ctx sizing** (§2's own motivating example — a `--drafter`
> attach after `BuildResident` grabbing VRAM an MoE expert cache already claimed — fixed 2026-09-09
> WITHOUT the `ResidencyBackend` interface change the doc originally expected: an out-of-band hint
> on `decoder.Model` was enough, CUDA-only since Metal hosts no drafter today, verified on real
> nobara hardware — see `docs/task-gpu-paths-2026-09.md`'s entries). **Phase 2 is now fully closed.
> Phase 3 (WebGPU) is DONE as of 2026-09-10** (`decoder/fitplan.go` admits `"webgpu"`; M-32's
> family+precision decline and the shared ctx-ceiling helper are what made that safe — see §7's
> own entry). **Phase 4 is PARTIAL as of 2026-09-10** — `fit -measure`'s self-measure half is
> done and untested against G1/G5's real hardware reference cells (scoped to "this machine"
> deliberately); its `pull`/web-UI half stays blocked on Phase 1's unstarted header-only
> checkpoint reading. **Phase 5 (host-computed experts) remains entirely unstarted.** Design
> record for the
> "run something bigger than my hardware" mode of use; reads
> `task-model-pull.md` (phase 1 shipped 2026-09-02) as the step immediately before this one in the
> user's hour. Sibling: `task-embed-and-harness-ux.md` (modes 2 and 3). Nothing in this doc changes
> numerics; every gate is an admission/accounting gate, and the do-nothing arm is today's
> hand-tuned configuration.

**Who this is for.** One person with one consumer machine — an 8 GB RTX card, a 16 GB Apple
Silicon laptop, a 32 GB desktop with no GPU — who has just pulled a 26B-A4B or 35B-A3B checkpoint
and wants it to run. They do not know what an expert slot is, they should never learn what
`int8int8` means, and the question they are asking is exactly one sentence long: *does this fit,
and how fast will it go?*

**The problem in the repo's own words.** `docs/positioning.md`'s "what it's for" section turns
into a slot-count troubleshooting guide by its second screen — two environment variables, a table
of slot counts against hit rates, a note about which build has the cap fix, and a paragraph
explaining why the published number is not reproducible on the card that produced it. Every
sentence in it is true and measured. None of it should be the user's job. The audit found the
guard that was meant to make it automatic inverting on both sides (M-01: admits the stacked 26B
that swaps; M-02: declines the paged 35B that fits), the WebGPU backend ignoring `-ctx` (M-32),
Metal's slot count reachable only through an environment variable, and 38 undocumented
`GOINFER_*` knobs (N-42), four of them the escape hatches of default-ON changes.

---

## 0. What must not change

- **Decline rather than run silently wrong.** The taxonomy rule (a backend declares a feature only
  when it ships the kernel; otherwise it declines to CPU and says so) is the one thing the user
  can already trust. Fitting must never admit a configuration the runtime cannot honour.
- **Explicit flags stay, and win.** Every knob this doc automates remains a flag and overrides
  the plan on its own dimension. Fitting is a default, not a mode.
- **Bit-exact by default.** The plan never selects a lossy KV precision (`f16`, `i8`) or a lossy
  quant step (`--embed-int4`) to make something fit without saying so in the plan and requiring the
  flag. Shrinking context is free of numerics; changing precision is not.
- **The measurement rules.** A plan's *expected rate* is a band with provenance or nothing. No
  multiplied projections (`docs/queue-performance.md` retracted two of those on 2026-09-01).
- **Single static binary, stdlib only.** Reading a checkpoint header and a machine's memory is
  already stdlib; nothing here adds a dependency.
- **`pull` stays "get bytes onto disk."** Fitting reads what `pull` fetched; it does not change
  what `pull` does (`task-model-pull.md` §0).

## 1. What already fits automatically — and what the user is still asked to decide

Scoped against what exists, the way `task-model-pull.md` was.

**Already automatic, per backend, in pieces:**

- CUDA expert cache: `--moe-cache-slots 0` means "ask for all and auto-cap to free VRAM"
  (`decoder/model.go:170-158`, the cap that accounts for 2 MiB allocation quanta and the first-launch
  reservation — `docs/positioning.md`'s own history of it).
- CPU weight paging: `--weight-cache 0` is "auto, ~half of available RAM"
  (`internal/serveapp/main.go:448`).
- Metal: a memory-fit guard that refuses a model whose weights exceed 70% of RAM
  (`metal/backend.go:136`, `:142`) — the guard whose arithmetic M-01/M-02 found wrong in both
  directions, with `GOINFER_NO_RESIDENT_MEM_GUARD=1` printed as the remedy (`metal/backend.go:252`).

**Still the user's decision, with no basis offered for it:**

| decision | today's surface | what the user has to know |
|---|---|---|
| which *mode* — resident, expert-cached, CPU-paged | `--moe-cache-experts` (`internal/serveapp/main.go:428`), `--stream-weights` (`:372`), or neither | that a 26B's experts exceed 8 GB "even at 4-bit"; that without the flag it declines to CPU |
| Metal slot count | `GOINFER_METAL_MOE_SLOTS` (`metal/gemma4_moe.go:207`, `metal/moe.go:319`), env only, no flag, no auto | the measured optimum was N=64 (`docs/task-metal-expert-streaming-at-scale.md`), and the doc's "default to 64" has no code behind it |
| context cap and KV precision | `-ctx` (`:361`, ignored by WebGPU — M-32), `-kv` (`:360`, breaks three families on WebGPU — M-32), `-kv-quant` (`:362`) | the VRAM a 16k f32 KV costs on their card |
| quant | `-quant int4` default (`:340`), `--embed-int4` (`:374`) | that int4 is now as fast as int8int8 on CPU (the in-repo guidance was reversed 2026-08-25) |
| whether it worked | `-require-backend` (`:354`) or reading the decline line | that "declined to CPU" is the failure they are looking for |
| how fast it will be | nothing | `docs/benchmarks.md`, 1,300 lines, provenance-gated, per machine |

The shape of the failure is the same on every row: the runtime *can* compute the answer and asks
the user to guess it instead.

## 2. The plan — one pure function

The centre of this design is a function with no side effects, so it can be unit-tested on
synthetic headers without a checkpoint or a GPU, printed as a dry run, and reused by `pull`, the
web UI and the startup banner without three copies.

```
plan(checkpoint, machine, request) → placement
```

**`checkpoint`** — read from the header, never by loading: tensor bytes by class — dense
(attention + shared FFN + norms + embeddings/head), routed experts (nE, top-k, bytes per expert
at the requested quant), per-layer recurrent state (DeltaNet/Mamba/conv), KV bytes per position per
precision, and the family's residency eligibility per backend (`decoder.ResidentEligible`, the
generator behind `docs/hardware-matrix.md`). `internal/prequant/ggufmeta.go` already reads GGUF
metadata without loading; the `.giw` header carries the same. This is the accounting M-01 asks
for: the same walker the `.giw` writer uses, so the accountant and the writer cannot disagree.

**`machine`** — backend availability and its budget: free VRAM as CUDA already measures it; unified
memory (`hw.memsize`) for Metal; available RAM for CPU; the host copy of the weights when the load
is not file-backed (M-02's doubling); disk class only as a warning (the SMR-archive trap in
`CLAUDE.md` is an operator rule, not something the planner can detect).

**`request`** — what the user asked for, with the defaults filled: context wanted (default: the
smaller of the model's window and 8192 — see §8), quant (default int4), anything they set
explicitly, which pins that dimension — and every companion the configuration will attach
(`--drafter`, a vision tower), because those allocate on the same device.

**Every allocation is a term of the plan, including the ones that attach after load.** Today's
CUDA cap sizes the expert cache against *free VRAM at that moment* (`capSlots`,
`cuda/resident.go`), keeping back `slotMarginBytes` — 384 MiB, by its own comment the only
unmeasured constant on that path — and nothing else; it cannot see what the same configuration
allocates next. Measured 2026-09-02 on the 8 GB card, in the verify-vs-pager run: gemma-4-26b-a4b
auto-sized to 31 slots/layer, the server came up, then `--drafter` attached and `NewBlockSpec`
failed on a 15.9 MB typed-len buffer — `cuda: device allocation failed` — because the cache had
already taken the room. So the plan sums the **fixed** terms first — dense weights, KV at the
chosen ctx, the drafter's weights (~500 MB for the 4B pairing, uploaded to the target's device at
attach) and the verify and capture buffers `NewBlockSpec` allocates, the host copy when M-02
applies, the first-launch reservation the A9 fix pays early — and gives the one **elastic** term,
the expert slot count, whatever remains. Elastic last: the thing that can shrink is sized after
the things that cannot, which is the same rule the priority order below applies to context. The
greedy load-time cap stays as the backstop for what the plan did not foresee, not as the plan.

**`placement`** — one of, per backend, with every number that justified it:

1. **resident** — everything on the device; KV at the requested ctx.
2. **expert-cached** — dense resident, routed experts in N slots per layer; N chosen as the largest
   count whose bytes fit beside every fixed term above — dense + KV + scratch + drafter + verify
   buffers (the CUDA cap already does the VRAM half of this, against whatever was allocated
   before it ran; Metal has no equivalent).
3. **host-computed experts** — reserved for L-01 (misses computed on the CPU from the pinned host
   copy). Not built; the placement enum carries it now so the planner is not rewritten when it lands.
4. **weight-paged** — CPU with `--weight-cache` sized from RAM (today's auto), for the model that
   fits no device.
5. **decline** — with the reason as numbers and the nearest configuration that would fit
   (§4).

**Priority order** (llama.cpp's `--fit`, adapted): shrink context toward a floor of 4096 before
moving anything; then move routed experts into a cache, dense weights stay resident; then fall to
weight-paging; never trade precision silently. Context first because it costs no numerics and the
KV at 16k is often the difference; experts second because the C′ work already showed a partial
cache is a large fraction of a full one (57% hit at 16 slots vs 82% at 38, `docs/positioning.md`).

## 3. The surfaces

- **`serve` and `goinfer-chat` fit by default.** `--fit=off` restores today's behaviour exactly.
  Any explicit flag pins its own dimension (`-ctx 32768` is honoured or refused with the numbers,
  never silently ignored — M-32's WebGPU case becomes a plan error).
- **`goinfer-chat fit <path-or-ref>` — the dry run.** Prints the plan without loading: bytes by
  class, the chosen placement per available backend, the ctx cap, and what was pinned. Two
  seconds, no model in memory. This is also the test surface.
- **The startup banner prints the plan** — model, backend, placement and why, ctx cap, KV
  precision, sessions on/off, and the expected-rate band when one is available (§5). The banner is
  the closest thing the product has to a UI (`task-embed-and-harness-ux.md` §3.3 owns its full
  shape; this doc owns the placement lines).
- **`pull` prints the verdict for the file it fetched**, on the line that already prints the
  `--model` command (`internal/chatapp/pull.go`): "fits resident on this machine at int4 (9.1 GB
  of 16 GB); expect the 1.5B–7B class". Cheap: the planner reads the header of the file it just
  wrote.
- **The web UI's Models tab shows fit before download.** `pull.File` already carries `Size`
  (`pull/pull.go:179`), and a GGUF's size is within a few percent of its resident
  bytes at the same quant, so the file table can say *fits / needs streaming / will not fit* per
  row from the listing alone, before the multi-gigabyte transfer. The exact plan comes after the
  header is on disk.
- **Metal slots become an `Option` and a flag**, the same `--moe-cache-slots` CUDA has, with
  `GOINFER_METAL_MOE_SLOTS` kept one release as a deprecated alias. The Metal guard moves inside
  `buildResident`, after the slot count is resolved, and computes `dense + N·L·perExpert + KV +
  scratch (+ host copy)` — M-02's fix, and the test that exists for it (`metal/qwen35_35b_paged_test.go`)
  then exercises the guard instead of bypassing it.

## 4. What the user never has to know, and what the decline says instead

Never: slot counts; any `GOINFER_*` variable; the words `int8int8`, `W4A8`, `.giw`, `sidecar`,
`transcode`; that the resident path exists; `-require-backend`; `--metal-fast-prefill`;
`--stream-weights`/`--weight-cache`; that "declined to CPU" is what failure looks like.

A decline is three lines, every number computed by the plan:

```
cannot fit gemma-4-26b-a4b at int4 on this GPU (8.0 GB free):
  dense 2.1 GB + experts 11.9 GB + KV@8192 0.6 GB = 14.6 GB
nearest that fits: expert-cached, 30 slots/layer (7.4 GB) — chosen automatically; or --ctx 4096
```

The remedy is a configuration, not an environment variable, and the automatic choice is stated
so the user knows the runtime is about to take it.

## 5. The expected-rate band — measured, or absent

Two defensible options, and the plan may carry either:

- **Self-measure.** After load, decode a fixed 32-token probe on a fixed 64-token prompt and
  print the rate as "measured on this machine". Costs one to two seconds on the paths this doc
  targets, nothing is multiplied, and the number is this machine's. The probe runs through the
  same path as a request, so it is also the first-hour gate (§6, G1) running itself.
- **A provenance table** keyed on (backend, active parameter count, quant, placement) with the
  machine and date, generated from `docs/benchmarks.md`'s rows — printed as a class ("the 2–4B
  active class on a 2070-SUPER-class card ran 11–17 tok/s expert-cached"), never as a point.

Not an option: a formula. The queue's own record is that every projected number this year was
retracted by the measurement (`docs/queue-performance.md` P18/P19).

## 6. Gates — pre-registered, with the do-nothing arm

The do-nothing arm throughout is the **hand-tuned configuration** from the measurement record
(`GOINFER_MOE_CACHE_EXPERTS=1 GOINFER_MOE_CACHE_SLOTS=128` on the 8 GB card;
`GOINFER_METAL_MOE_SLOTS=64` on the Mac; the documented `--stream-weights` invocation on CPU).

- **G1 · zero-flag admission on the reference cells.** `serve --model <ckpt>` with nothing else,
  on: gemma-4-26b-a4b int4 / 8 GB CUDA; qwen3.5-35b-a3b int4 / 16 GB Mac (paged, measured 2.19
  tok/s at N=64); Mellum2 12B int4 / 8 GB WebGPU; the 1.5B on CPU. Passes when each is admitted
  to the same placement the hand-tuned arm used and decodes at **≥ 0.90×** its rate, paired and
  interleaved in one session. Below 0.80× the planner's choice is wrong and the item does not ship;
  0.80–0.90× is ambiguous → parked pending a mechanism.
- **G2 · the guard's two directions, as a matched pair.** The paged 35B on the Mac is *admitted*
  (M-02) and the unpaged, slot-less 26B on the Mac is *declined with the numbers* (M-01), both from
  the same accounting function, in one test that runs the arithmetic on the recorded geometries
  without a device. A guard that passes one half and fails the other has inverted again.
- **G3 · accounting within 10%.** Planned bytes vs measured device allocation after load, on the
  G1 cells, and once more on the 26B cell with `--drafter` attached — the allocation the load-time
  cap could not see: every class within 10%, total within 5%. Keyed on quantities the plan
  computes, never on RSS or "available" (the `CLAUDE.md` rule the last guard broke).
- **G4 · the plan is a pure function.** A table-driven unit test on synthetic headers — dense,
  MoE, hybrid, every backend, every budget from 6 to 64 GB — runs in CI with no assets and pins the
  placement and the ctx cap. A change to the priority order is a change to this table, reviewed.
- **G5 · the band contains the rate.** If the plan prints an expected band, the G1 measured rate
  falls inside it on every cell, or the band is removed. A band that misses is worse than no band.
- **G6 · nothing pinned is overridden.** `-ctx`, `-kv`, `--moe-cache-slots`, `--quant` set
  explicitly are honoured or refused with numbers; a table test over every flag.

## 7. Phasing — each step independently droppable

0. **Accounting** (closes M-01/M-02, prerequisite): `WeightBudget()` by class from the writer's
   walker, with the drafter and verify buffers as named terms; the Metal guard moved after slot
   resolution; G2 + G3.
1. **Dry run**: `goinfer-chat fit`, the pure `plan` and its table test (G4), the banner lines. No
   behaviour change yet — the plan is printed beside today's decision so the two can be compared
   on the reference cells before anything flips.
2. **Fit by default** on CUDA, Metal and CPU: placement + ctx chosen when not pinned; `--fit=off`;
   Metal slots as an Option/flag; G1 and G6.
3. **WebGPU — DONE (2026-09-10).** M-32's `-ctx` half was already fixed 2026-09-02 (declining,
   not implementing); `Plan()` now admits `"webgpu"` (`decoder/fitplan.go`), mirrors
   `gpu/residency.go`'s own Nemotron/Qwen3.5/MLA + KVF16/KVI8 decline so it never promises a
   precision `BuildResident` will refuse, and enforces the SAME fixed per-precision ctx ceiling
   (16k/32k/64k) via one shared `decoder.WebGPUCtxCeiling` both the planner and the backend call
   — no drift possible between what a plan promises and what gets allocated. `goinfer-chat fit`
   reports it like any other compiled backend; it just has no live free-memory probe yet (WebGPU
   exposes no portable query), so it prints "no memory probe available... skipped" until one
   exists — a separate, harder problem this step did not need to solve. Table-tested
   (`decoder/fitplan_test.go`): generic admit, ctx capped at the ceiling even with byte-budget
   room, a pinned ctx above the ceiling declines, and the family decline fires on a real MLA
   fixture before any byte accounting runs.
4. **The band — self-measure half DONE (2026-09-10), `pull`/web-UI half still blocked.**
   `goinfer-chat fit -measure` (`internal/fitcmd/fit.go`) does §5's self-measure: after the dry
   run, picks the best ADMITTED backend (any non-cpu over cpu — cpu is always eligible and
   always tried last), does a REAL `decoder.Load` + `Generate` on it (64-token deterministic
   prompt, 32 decode steps, greedy), and prints the measured tok/s. Off by default (a second
   real load is not free); tested on the tracked `llama-tiny` fixture (CPU-only build) plus a
   real Metal binary on this machine, both real decode paths, not projected. **NOT gated** — G1/
   G5 (§6) need the SAME reference cells (26B/8GB CUDA, 35B/16GB Mac, Mellum2/8GB WebGPU) real-
   hardware-measured and compared, which spans the nobara CUDA box and a working WebGPU setup
   this pass didn't reach; scoped down to "this machine" deliberately (2026-09-10 decision). The
   `pull` / web-UI verdicts are UNCHANGED — still blocked on Phase 1's still-unstarted
   header-only checkpoint reading (parsing param counts/MoE geometry without a full `Load()`,
   which `decoder/gguf.go`'s ~2900 lines of format logic does not separate out today); that is
   its own scoping pass, not attempted here.
5. **Host-computed experts** as a placement when L-01 lands — the enum slot exists from step 1.
   L-01's own design pass (`docs/task-l01-hybrid-moe-cpu-gpu.md`, 2026-09-10) found the naive
   mechanism likely does NOT beat the shipped path at today's measured numbers — no code yet,
   this phase stays blocked on that resolving, not just on L-01 "landing."

## 8. Open questions, deliberately left open

- **The default context.** 8192 is the agent-turn size `docs/server.md`'s dsh section measured;
  the model's full window is what a user expects to "just work". The plan can print both costs;
  which is the default is a product call, not a measurement.
- **Lossy KV as a last resort.** `f16` KV doubles context at the same VRAM and is where llama.cpp
  and Ollama default. This doc says never silently; whether the plan may *offer* it in the decline
  line ("or `-kv f16` for 16k") is a one-line decision once G1 is green.
- **Disk-backed paging.** The Mac 35B path reads experts from SSD; a planner cannot see an SMR
  archive or a network mount. It can print the path it will read from, which is what the
  `CLAUDE.md` rule needs a human to check.
- **Windows.** Untested for every GPU backend; the planner should say "CPU only" there until a
  row exists, not guess.
- **This planner plans slots and context, not layer placement — R9 found the cost of that gap
  (docs/measurements/cold-user-2026-09-06-nobara-pc.md).** The `placement` enum (§2) has five
  states — resident, expert-cached, host-computed-experts, weight-paged, decline — and none of
  them is "N dense layers on the GPU, the rest on CPU": on cuda/metal a dense model that does not
  fit the card declines to CPU in full, because neither backend has a staged/partial GPU path at
  all (R9's finding — this doc's "adapted from llama.cpp's `--fit`" priority order borrowed the
  ordering, not that capability). The peer matrix already shows what that costs: llama.cpp's
  `--fit` partial offload beat goinfer on both the M35 and M26 8 GB-card cells (32.9 vs 23.5,
  27.8 vs 24.6 tok/s) after going from "never loaded" to winning, by dropping `-ngl 99` for a
  computed partial placement. Real hybrid CPU/GPU layer placement is the audit's top program item
  (FreeToken Lead 5); this planner does not attempt it and should not be read as covering it —
  MoE expert streaming (`expert-cached`, above) is the only bigger-than-the-card story a dense
  goinfer model has on cuda/metal today.

## Sources

`docs/positioning.md` (the slot-count section this doc exists to retire) · `docs/audit-2026-09-02.md`
M-01, M-02, M-31, M-32, N-42, L-01, L-04 · `docs/task-model-pull.md` (phase 1 shipped; the step
before this one) · `docs/task-metal-expert-streaming-at-scale.md` (N=64, 2.19 tok/s; "default to
64" with no code) · `docs/task-moe-streaming.md` §C′ (the CUDA cache and its cap) · `docs/QUEUE.md`
G31–G33 (the DMA term, capacity misses) · `docs/hardware-matrix.md` (residency eligibility, generated) ·
`internal/serveapp/main.go:409-376` (the flags the plan subsumes) · `decoder/model.go:170-186`
(`MoECacheSlotsRequest`, `Options`) · `metal/backend.go:115-194` (the guard) ·
`decoder/weightbytes.go:56` (`ResidentWeightBytes`, the accountant to replace) ·
`pull/pull.go:179` (`File.Size`) · llama.cpp `--fit` (discussion #18049, the
priority order borrowed).
