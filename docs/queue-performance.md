# Performance queue

> **Four open items as of 2026-09-23** (P21, P23, P24, P25) — was briefly empty on 2026-08-31, when
> every item then on the page was closed, refuted, withdrawn, or moved to the track that owns it.
> P20 closed 2026-09-21 (both its generic and gemma4-specific paths built and shipped default-on,
> 2.26–2.66× on the real M26); P22 closed 2026-09-23 (WebGPU Theta measured at 0.978–1.028, the
> same over-drafting defect Metal had, fixed and confirmed with a real A/B). This line was
> originally "Four open items as of 2026-09-05 (P20–P23)" and had already drifted once (missing
> P24/P25) before P20 and P22's own closures were caught and corrected here.
>
> **The closed record is [`docs/completed/queue-performance.md`](./completed/queue-performance.md)**
> (2,245 lines, G15–G24 · P1–P16 · A1–A11 and the A9-\* series). It moved rather than being deleted
> because the reasoning is the point, negative results included. Per `docs/README.md`'s archival
> rule, `completed/` is not scanned by the citation lint, so archiving it also retired its citations
> from the live gate — that is deliberate, not an oversight. Pages that linked to
> `docs/queue-performance.md` still resolve here.

## What is open

- **P35 · Gemma-4 prefill falls back to one token at a time past about 12k attended keys on a 48 KB shared-memory card
  (sm_75) — SUPERSEDED 2026-10-18: branch `kv3-tiled-2026-10` already covers it, and nobara is bringing that branch onto
  main. Option A is dropped (owner).** `linux`.

  - **The branch covers it:** its row-batched split-KV prefill kernels (`cuda/prefill_splitkv.cu` on that branch) take the
    head dim per layer. `TestPrefillSplitKV_rowsEqualAttnBatched` holds them bit-equal to `attn_batched` for head dims 20 to
    512. Its record (`docs/measurements/kv3-tiled-exact-2026-10-13.md` on that branch) has Gemma-4-E4B at 12,400 tokens, and
    a served smoke of 12,215 tokens ran in 55.9 s with no decline.
  - **Not covered there:** the image-block (bidirectional) pass past the bound.
  - **Left:** bringing the branch onto main, a dense Gemma-4 gate there, and the image-block twin. Not a new kernel. The
    scoping below is history.

  - **Found by:** tack's C1 calibration (gemma-4 E4B, RTX 2070 SUPER, goinfer `0de92cd9`). Every request past the limit
    logged `prefill_fallback=sequential`. Time to first token was 48–128 s on 11–26k-token prompts: batched ran about
    227 tok/s, the fallback about 52.
  - **Cause:** the exact batched attention keeps a whole row of scores in shared memory (`(nWin+128)*4` bytes,
    `cuda/resident.go:attnShmemBytes`), so it cannot run past about 12,160 keys on 48 KB. The fused attention, whose
    shared memory does not grow with the span, serves only head dims 64 and 128 (`cuda/prefill.go:cudaResident.checkPrefillShmem`).
  - **Scoped options (none built):**
    - A, a tiled exact kernel: one pass for the max, a second tile by tile for the exponent sum and the value sum, in the
      original order. Bit-identical by construction, about 1 day.
    - B, a batched split-KV kernel like decode's: about 1.5–2 days, because its score scratch needs sub-batching and pricing.
    - C, the fused kernel for head dims 256 and 512: fast-class, needs a fidelity gate, several days.
    - The owner approved A, then parked it, then dropped it once `kv3-tiled-2026-10` was found to cover the case.

- **P34 · Two 32k candidates never reuse a prefix on CUDA — reopened 2026-10-19 at high priority (owner). Two causes, one per
  model, both fixed on `main`: cause 1 by B35 (#112, 2026-10-21), cause 2 by `1ea85681` (2026-10-19). Not re-run: tack's E0
  cells on a fixed build.** `linux`.

  - **Seen (tack's E0, nobara):** first-token time with the prefix reused equals cold at every size.
    - Qwen3.5-9B: C3 on `f20f0d93`, at 4k 15.3 vs 15.7 s, 8k 38.8 vs 38.8 s, 31.7k 421.3 vs 421.5 s.
    - Mellum2.1-12B-A2.5B with `--moe-cache-experts`: amendment 10 on `b2b5e816`, about 93 tok/s prefill either way (tack
      `docs/measurements/e0-nobara-mellum-2026-10-19.md`, tack 70a5193). The 31.7k probes took about 343 s each.
  - **Cause 1, the Gated-DeltaNet hybrids (Qwen3.5-9B; Qwen3.6-35B-A3B has the same layers):** the chat template drops the empty
    think block from history turns, so each chat request diverges from where the recurrent state stopped
    (`docs/measurements/qwen35-reuse-probe-2026-10-18/README.md`). **Fixed by B35, merged 2026-10-21 as #112.**
  - **Cause 2, Mellum2.1 (21 of 28 layers sliding-window, KV windowed): an expert-cache load got no rollback grant. FIXED
    2026-10-19, `1ea85681` on `main`, by the proposed fix below; the probe PASSED**
    (`docs/measurements/p34-mellum-rollback-2026-10-19/README.md`: at `-ctx 8192` a 600-token edit re-prefilled on the old build,
    17.4 s, and reused 6,412 tokens on the new one, 1.9 s; the load is granted 576 positions and a follow-up may diverge 1,024
    back). The reading below is as filed, before the fix.
    - `cuda/kvwindow.go:cudaResident.kvGrantRollback` returns at once when `cacheExperts` is set ("its cache was sized to what
      was free"). The rollback stays at the base slack less the step: 448 positions, up to 512 at the best point of the
      compaction cycle.
    - E0's reused cell resends the previous prompt with its last 512 tokens changed. With the template's end-of-turn and
      assistant-header tokens and the one generated token after them, the resend diverges about 518 positions before the
      newest. Replaying `kvWindowNextBase` for Mellum's 1,024 window at 4k, 8k, 15k and 31.7k: every resend that diverges 518
      or more positions back is refused (a cold prefill) without the grant, and reused with it (rollback 1,024).
    - **Expert streaming does not disable reuse:** `Model.residentReuseDisabled` reads only a test knob.
    - **What it means for tack's own loop:** a request that appends to the previous one is unaffected. One that rewrites more
      than about 448 tokens of the end (a client that drops the previous reply, as Codex does) is refused on any
      expert-cache load of a windowed model.
  - **The fix for cause 2 (built as proposed, `1ea85681`):** price the rollback grant before the expert cache is sized, so an
    expert-cache load gets it too. The cost is the grant's few MB of KV per slot (576 positions across the windowed layers)
    taken from the expert cache. Gate: a tier 0 replay of `kvWindowNextBase` and the plan over E0's shape, with and without the grant, plus
    a plant; then one by-day probe on nobara, Mellum at 8k with an edit 400 and 600 tokens from the end, reused after the fix.
  - **Data:** tack `docs/measurements/e0-nobara-c3-2026-10-18.md` and `e0-nobara-mellum-2026-10-19.md`; raw records under
    `~/tack-runs/e0/` on nobara. The Mellum run kept no request log (`reused=`, `div=`), so the confirming probe should run with
    `-log-requests`.

- **P33 · Metal decode is 0.67–0.76 of mlx-lm's single-request path at matched settings — OPEN, filed 2026-10-18 (owner).
  Measure first; no work yet.** `mac`.

  - **The gap** (`docs/benchmarks.md` §B11, same-session, depth 128, 3 runs × 8 × 64 tokens, the same seed and sampling
    fields on both arms; mlx-lm 0.31.3 on mlx-community 4-bit):
    - greedy: 1.5B 0.666 (92.0 vs 138.2 tok/s), 7B 0.742 (29.7 vs 40.0);
    - the shared preset: 0.71 / 0.76;
    - untruncated: 0.69 / 0.75.
  - **The quants differ in kind.** goinfer runs `int4` (its group-32 W4A8 weights), and mlx-lm runs its own 4-bit
    group quantisation. Both are 4-bit, so the classes match, but the kernels and layouts are each engine's own.
  - **First, where the decode time goes on each arm:**
    - goinfer's per-token split: the GEMVs by projection, attention, norms and the head, and the CPU-side time between
      command buffers;
    - the same split for mlx-lm, from its own trace;
    - each GEMV's achieved bandwidth against the M1 Pro's.
    The July "dispatch-bound" reading was refuted 2026-09-26 (GEMVs are weight-work-bound), so this starts from a
    measurement, not that assumption.
  - **Not before the split:** any kernel work, or any claim about which side's quantisation is "faster".

- **P32 · Gemma 4 E-models prefill at 144 tok/s on Metal where Qwen2.5-1.5B does 900: a fused attention kernel for head dims
  256 and 512 — OPEN, filed 2026-10-17 (owner). Not built.** `mac`, Metal kernel work.

  - **The gap** (`docs/measurements/p28-gemma4-metal-ttft-2026-10/README.md`, steps 1 and 2): under 4,096 prompt tokens
    E2B's batched prefill runs at 143.5 tok/s, against Qwen2.5-Coder-1.5B's 900 on the same Mac.
  - **Why the kernel is the suspect:** E2B's head dims are 256 and 512, above `attention_prefill_fused`'s 128
    (`metal/prefill.go:resident.prefillAttnKernels`). So every attention layer runs the exact kernel, `attention_prefill`:
    one threadgroup per row and head, a scalar dot per key, and a serial P·V scan.
  - **The work:** raise the fused (flash) kernel's `ATTN_MAXHD` to 512. That means a threadgroup and register budget per
    head dim, and tuning. Its online-softmax rescale is not bit-identical to the exact kernel (the P19 category, as the
    fused kernel is today).
  - **First measurement, before building:** E2B's prefill time split between attention and the rest (projections, FFN),
    at 4,096 tokens. If attention is not most of it, a fused kernel cannot close the gap.

- **P31 · CUDA decode is behind llama.cpp at matched 4-bit quants: about 9% on the small models, 12% on phi3-mini — OPEN,
  filed 2026-10-17 (owner). Measure where each arm's decode time goes first; no work on it yet.** `linux`.

  - **What was measured** (nobara-pc, RTX 2070 SUPER, driver 595.91.07, depth 128, 3 runs × 8 × 64 tokens, same session,
    llama-server 0.4.0-dev `427291b` with `-ngl 99`, the same Q4_K_M files):
    - goinfer `int4` on the 0.5B and gemma3-1b, and `q4k` on phi3-mini: 4-bit on every arm.
    - Every arm was sent the same seed, the same context (4096) and the same sampling fields, plus `min_p 0` for llama.cpp.

    | model | shared preset 0.8 / top_k 40 / top_p 0.9 | greedy | no truncation (T = 1.0) |
    |---|---|---|---|
    | qwen2.5-coder-0.5b (`int4`) | **0.917** (343.4 vs 375.5 tok/s) | 0.948 | 0.978 |
    | gemma3-1b (`int4`) | **0.910** (195.7 vs 213.8) | 0.927 | 0.972 |
    | phi3-mini (`q4k`) | **0.881** (112.6 vs 127.8) | 0.890 | 0.888 |

    goinfer ÷ llama.cpp. The shared-preset column is the record
    `docs/measurements/peer-sharedtrunc-cuda-2026-10-17/README.md` (harness `e8e8c95d`). The other two are
    `docs/measurements/peer-notrunc-2026-10-17.md` (harness at `63038121`, before the context and `min_p` fix, so
    goinfer ran at its own default context and llama.cpp at its default `min_p`).
  - **The same cells against Ollama:** goinfer is AHEAD on the small models (shared preset 1.312 / 1.272) and BEHIND on
    phi3-mini (0.895).
  - **Not known:** where the gap is. It could be in the matvecs, attention, the sampler and host round trip, or launch
    overhead, and it may differ between the int4 and q4k paths (phi3's q4k path also declines batched prefill, which this
    decode-only rate does not include).
  - **First measurement, before any change:** one decode step on each arm on the same model and prompt, split by kernel.
    For llama.cpp, Nsight Systems or its own timings. For goinfer, its kernel timeline or an `ncu` pass. Compare the
    totals per token, then the matvec share against the bytes each arm reads per token. A step the two arms do not share
    is named, not estimated.

- **P30 · An armed tool-call constraint costs about 1.2 ms per token after `<tool_call>`, beyond the masker — OPEN, filed
  2026-10-17 (owner). Not timed by part yet; not being worked on.** `linux` (seen on CUDA; Metal and WebGPU unmeasured).

  - **What was measured** (nobara, Qwen2.5-7B-Instruct q4_k_m GGUF on CUDA, the pinned build at e0b1a966, tack's E1 tool
    list of 19 tools, `tool_choice: auto`, a greedy 121-token `write` call, n=8 per arm, interleaved off/on/off/on under
    the timing lock, 2026-10-17): decode ms per call **1804 with the constraint off (`GOINFER_TOOL_UNION=0`), 2011 with
    it on: +207 ms per call** (66.5 vs 59.7 tok/s). Both arms produced the same 121 tokens.
  - **What the masker accounts for:** about 62 ms of it. That is the masker's own per-step time over the same call on
    nobara's CPU with Qwen's 152k vocabulary (`constrain.Masker.Process`, the minimum of 5 runs: 62.1 ms with 19 tools,
    55.9 ms with one).
  - **The other ~145 ms is unattributed.** The working guess, not checked: once the lazy constraint arms, the
    `LogitProcessorGate` opens and each step leaves the on-device greedy path. It then copies the full logits to the host
    and picks the token there. That is about 1.2 ms per token over the ~115 tokens after the opener.
  - **Why it matters:** an agent turn pays it on every call, scaled by the call's length. The pre-name cost this was found
    beside (about 0.5 s per call) was fixed by the masker prefilter (31418112).
  - **First measurement, before any change:** split one armed step on CUDA into the logits copy, the masker and the host
    pick, at tier 0 or 1, on the same request. A part that is not near 1.2 ms per token means the guess is wrong.
  - Records: the run directory `e1-prefilter-onoff-20261017-1007` under `~/goinfer-bench` on nobara (not committed).

- **P29 · Phi-3 at q4k on Metal prefills one token at a time when the memory guard allows one KV slot, and the decline
  message blames geometry — SHIPPED 2026-10-18: a one-slot load prefills on the batched step (measured on the 0.5B, 2.27×).
  Filed 2026-10-17 (owner).**
  `mac`.

  - **The tier 1 A/B (`p29-one-slot-ab`) voided three times, 2026-10-17 and 2026-10-18.** No guard was bypassed.
    - **Twice (19:23, 21:49), the fit guard refused** its phi3 `q4k` load from the `.gguf`. The load needs 5.6 GB: 2.6 GB
      of weights, 0.8 GB of KV and 2.2 GB for the mapped checkpoint. The guard's budget is 70% of available memory, and
      available was 6.4 GB, then 7.7 GB.
    - **The third time (00:08, after a restart with nothing else open), the fit guard passed, and Metal's resident memory
      guard refused.** The build needs 5.98 GB (weights, host copy and KV), against a budget of 4.34 GB once live-available
      memory tightened the static 11.20 GB ceiling.
    - `q4k` has no `.giw` form that would take the mapped file out of the fit guard's price.
    - Not re-queued: it is back with the owner for options (owner, 2026-10-17).
  - **The A/B on the 0.5B, forced to one slot: registered 2026-10-18, before it runs (owner: queue it on the 0.5B).**
    - **Counter test first** (`metal/p29_route_test.go`, by day, the 0.5B q4_k_m):
      - main reads the sequential loop at one resident KV slot (1 row per step) and the batched step at two (8);
      - the lift branch `p29-one-slot-step-2026-10` (88a08574) reads the step at both.
      The batched step does not apply on main, so P29 stays open.
    - **Job:** `run-p29-ab-qwen05.sh` (`docs/measurements/`), a Mac night job, running `TestOneSlotStep_qwen05PrefillAB`
      from a worktree pinned at 88a08574.
      - The 0.5B q4_k_m at `--quant q4k`, `-ctx 2048`, one resident KV slot.
      - A 512-token prompt, prefilled one token at a time (production's Forward) against the one-slot step (PrefillLast).
      - One untimed warm-up of each arm, then 5 interleaved pairs in alternating order.
      - r = sequential ÷ step per pair; the median is read.
      - The one-slot step is bit-identical to the sequential loop at tier 0 (`TestOneSlotStep_q4kLaneBitIdentical`, with
        its plant), so this measures speed only.
    - **Bands:**
      - **SHIP:** a median of at least 1.02 with every pair above 1.00. The lift fast-forwards to main.
      - **PARK:** a median in [1.00, 1.02), or pairs straddling 1.00. Reported, not merged.
      - **OFF:** a median under 1.00. Reported, not merged.
      - **VOID:** a guard declines the load (the test skips).
    - **No stop rule** beyond the five pairs, and no re-run until the job has run once.
    - **Result, read 2026-10-18: SHIP.**
      - Run 12:47–12:52 PDT as night job `p29-ab-qwen05-one-slot`, at 88a08574. The owner started the queue by day, with VS
        Code and the Claude app open (load about 3.5).
      - The median r is 2.270. The five pairs are 2.269, 2.275, 2.256, 2.283 and 2.270, all above 1.00: sequential about
        192 tok/s, the step about 438 tok/s over the 512-token prompt.
      - The desktop's load cannot move a paired ratio that far from 1.02.
      - The lift fast-forwarded to main as registered, rebased on main with its tier 0 re-run. Its Phi-3 A/B moved to
        `metal/one_slot_step_phi3_test.go` so the asset registry resolves the file.
      - Log: `~/goinfer-logs/night/runs/2026-10-18/p29-ab-qwen05-one-slot.log` on the Mac.

  - **What was seen** (peer-vetted M1's fit check, 2026-10-16 23:21, the pinned `serve-metal` at 42d72319, `-quant q4k
    -ctx 2048`): the load went `metal-resident (q4k)`, and the banner said `prefill path: sequential — arch/geometry not
    supported by f16 MMA prefill kernel`. `-require-backend` then refused it. The record is
    `docs/measurements/peer-vetted-2026-10-07-macbook.md`, "Amendment, 2026-10-17", on branch
    `m1-phi3-q4k-cell-2026-10` until that branch merges.
  - **The cause, from the source at that commit, is not geometry:**
    - The q4k lane never takes the f16 MMA pass: `metal/model.go`'s `prefillOK` excludes `q4kLane` outright.
    - The fallback is OP6's batched step ("decode rows on the batched step kernels"). It needs at least two resident KV
      slots: `metal/batch.go:resident.batchIneligible` returns "one resident KV slot". At `-ctx 2048` each slot cost
      768 MB, and the memory guard allowed one ("1 resident KV slots of 2 requested").
    - `metal/backend.go:metalResident.PrefillPath` reports both cases with the same "arch/geometry" text.
  - **First, from the code (owner, 2026-10-17): why does OP6's batched step need two resident KV slots, and can a single-slot
    load use it for its own prompt (one sequence, 8 rows per step)?** Answered from main at 3c845917:
    - **The minimum belongs to MC3, not to the step's arithmetic.** The step was built for concurrent decode, one sequence
      per slot. Its eligibility says so ("on at least two resident KV slots": `metal/batch.go:resident.batchIneligible`).
      OP6 then reused it for one sequence's prompt and inherited the gate. `metal/batch.go:resident.forwardMultiOut`
      already accepts several rows on one slot when they are one sequence's consecutive positions (the speculative verify's
      shape). That is exactly a prompt's prefill.
    - **One slot fails on construction, not semantics.** `metal/model.go:buildResident` fills `kvSlotBufs` and sets `kvContig`
      only when `kvSlots > 1`. The step reaches every row's KV as `kvSlotBufs[0]`'s buffer plus a per-row offset (one
      allocation per layer, because a dispatch binds at most 16 buffers). With one slot, `kvSlotBufs` is empty, so
      `metal/batch.go:metalResident.VerifyCost` returns nil and the prompt routes in
      `metal/backend.go:metalResident.PrefillPath` and `metal/backend.go:metalResident.PrefillKernelFloor` fall back to
      sequential. Yet a one-slot resident's `kc`/`vc` already have the step's layout: slot 0 at offset 0 of a one-slot
      allocation.
    - **What lifting it would take:**
      1. At one slot without int8 KV, build `kvSlotBufs` with that one entry and mark it contiguous. Already correct at
         one slot: `resident.useKVSlot` (a no-op), `resident.kvHostOff` (offset 0) and `metal/kvcopy.go`'s CopyKV, which
         keeps its "one KV slot" decline.
      2. Let `batchIneligible` take one slot.
      3. Price the step's scratch (8 rows' buffers) in the one-slot fit plan, since the one-slot load is the tight one.
      4. Decide one side effect: a non-nil `VerifyCost` also gives the speculative paths a step verify cost
         (`decoder/spec_adaptive.go`). Either accept that and gate it, or split "prefills on the step" from "verifies on
         the step".
      - **Nothing in the decoder needs changing for concurrency:** `decoder/mc3_batch.go:Model.EnableResidentConcurrency`
        and the batcher already require two slots. `metal/batch.go:metalResident.BatchStepRange` should still report
        (0, 0) under two slots.
      - **The gates would be:** tier 0 bit-identity of a one-slot step prefill against sequential, on both sides of the floor;
        the same-slot verify test at one slot; and a planted defect.
    - **Why this likely matters more than the 1-vs-2 slot measurement:** the cliff is not Phi-3's. Every Metal resident
      whose memory guard allows one slot loses the step for its prompt, and the step is what serves:
      - the q4k lane, always;
      - a family the f16 MMA pass does not take (`!prefillOK`);
      - the short warm-turn suffix above the floor, for every model (`PrefillKernelFloor` requires `VerifyCost`).

      On a 16 GB Mac that is the large models: the loads where prefill time is largest.
  - **Built (owner, 2026-10-17, "build the one-slot lift, ahead of OP4 on the Mac"):** branch `p29-one-slot-step-2026-10`.
    - **The step at one slot:** a one-slot resident fills `kvSlotBufs` with its slot, marked contiguous, and
      `batchIneligible` takes it.
    - **The capability is split.** `metalResident.stepPrefill` ("prefills on the step") drives the prompt routes.
      `VerifyCost` ("verifies on the step"), `BatchStepRange` and `BatchGreedyDraw` still need two slots, so speculation's
      verify cost on one slot stays nil.
    - **The step's scratch is priced** against the headroom the memory guard leaves beside a one-slot build. A miss declines
      the step for that load, logged, and never the load.
    - **Tier 0 is in `metal/one_slot_step_test.go`:**
      - bit-identity against sequential (last-row logits, every K/V byte, a continuation), on the MC3 fixture at 2-64
        tokens and on the q4k 0.5B;
      - a planted row offset, which goes red;
      - verify and batched decode off at one slot;
      - the guard's exact edge through `metalBackend.BuildResident`: the load is admitted with the step declined at need, and
        the step is built at need plus its price. The price covers the driver's count (4.28 MB priced, 3.13 MB counted on
        the fixture).
  - **The measurement (tier 1, by day; registered 2026-10-17 before it runs):** `TestOneSlotStep_phi3PrefillAB`.
    - **The run:** Phi-3 mini 4k at q4k on one KV slot (`-ctx 2048`), one 512-token prompt from position 0. The two arms are
      production's sequential decode and the step route, interleaved in one process with the arm order alternating:
      1 warm-up each, then 5 pairs.
    - **The quantity:** each pair's sequential ÷ step wall time; the reading is their median. Expected about 2x (owner).
      It is reported, not graded against a bar.
    - **The setup:** the load passes Metal's memory guard first or the test skips, and the run holds the timing lock.
  - **Then the decline message:** it names the condition that applied, which is a one-line fix and needs no measurement.

- **P28 · Gemma 4 E2B and E4B: time to first token on Metal is 20-35x Qwen2.5-1.5B's at the same prompt — OPEN, filed
  2026-10-16 (owner). Measure first, by day; report before any fix.** `mac`.

  - **Step 1 read 2026-10-17 (`docs/measurements/p28-gemma4-metal-ttft-2026-10/README.md`): not per-token below 4,096
    tokens.** E2B ran the batched pass: 144 tok/s at 4,096 tokens, against the 1.5B's 900. At 8,192 it fell to 34 tok/s,
    8.4x the time for twice the prompt. The code says why the knee should sit at 4,097 keys: E2B's head dims have no fused
    kernel, the exact kernel holds 4,096 scores, and past that the pass declines silently to the layer-major one. Step 2,
    registered there, tests that against memory pressure.
  - **Step 2 read 2026-10-17 (same record): it is the 4,096-key bound, not memory.**
    - E2B ran 143.5 tok/s at 4,096 tokens, 37.5 at 6,144 and 34.3 at 8,192.
    - At the two slow lengths there were no swap-outs and no compressions, and the peak footprint did not climb (1,626 /
      1,358 / 1,564 MB).
    - Nothing logged the switch to the slow path.
    - Fixes scoped there, not built: log the decline (small); tile the exact kernel's scores past 4,096 keys (the smaller
      kernel change, bit-identical); a fused kernel for head dims 256 and 512 (larger; the only one that also speeds up
      prompts of 4,096 or fewer).
  - **Fixes 1 and 2 built 2026-10-17 (owner's yes); fix 3 filed as P32.** The decline was already logged on main from
    `292211eb`, after step 2's pinned build. Fix 1 makes the stderr note once per request, with the prompt's length, and puts
    the reason in the `-log-requests` line. Fix 2 (`attention_prefill_tiled`) runs a pass past 4,096 keys on the batched pass:
    - at ≤ 4,096 keys the pass still runs `attention_prefill` unchanged;
    - the tiled kernel's output does not depend on its tile size;
    - against `attention_prefill` it is bit-identical at head dims up to 256 and within 1 f16 ulp at 512 (fast-math codegen
      of the per-thread output sums, not the tiling);
    - past 4,096 keys it is within 4e-3 of a float64 reference;
    - on the tiny E-model it is bit-identical to the untiled pass and within the batched pass's bar of the layer-major one.
    The timed reading (4,096 / 6,144 / 8,192 on E2B) is a night job.

  - **What was seen** (tack's E0 survey on the MacBook, 2026-10-16, goinfer 0e598d3f; tack's harness, exploratory, not
    this repo's provenance gate). Both Gemma 4 E-models loaded `metal-resident (int4)` at `-ctx 16384` with fit headroom
    (E2B: 3.5 GB of weights against a 6.1 GB budget). Cold time to first token, median of 3 (1 at 15,360), as prefill tok/s:

    | model | 4,096 | 8,192 | 15,360 tokens |
    |---|---|---|---|
    | Gemma 4 E2B q4_0 | 41 (99.8 s) | 34 (238 s) | 26 (582 s) |
    | Gemma 4 E4B q4_0 | 26 (160 s) | 22 (365 s) | 18 (855 s) |
    | Qwen2.5-Coder-1.5B q4_k_m | 915 (4.5 s) | 795 (10.3 s) | 660 (23.3 s) |
    | Qwen2.5-7B q4_k_m | 206 (19.8 s) | 191 (43.0 s) | 172 (89.3 s) |

    The E-models' rate falls with length, and even the 7B prefills five times faster than E2B. That fits a one-token-at-a-time
    prefill, but it is not established: no prefill path was recorded. Both E-model arms then ran past their tack timeouts.
  - **The measurement (tier 1, by day):** prefill tok/s on Metal for Gemma 4 E2B (`gemma-4-E2B_q4_0-it.gguf`, at tack's
    int4) against Qwen2.5-Coder-1.5B q4_k_m (tack's 1.5B file), cold, at 4,096 and 8,192 tokens. For each, the prefill path
    the server takes (its banner's `prefill path:` line: batched, or one token at a time) and any decline reason logged.
    No logged reason, where the path is per-token, is itself a finding.
  - **Cost and stopping rule:** E2B cold at 4,096 and 8,192 is about 340 s, and the 1.5B about 15 s: about 6 minutes with
    loads. 15,360 alone took 582 s on E2B, past tier 1's 10 minutes, so it runs only if 4,096 and 8,192 disagree on the path,
    or the rate's fall with length is the question. Otherwise it goes to the night queue with a reason. Runs hold the timing
    lock and not beside a timed job.
  - **Report before any fix:** the two rates, each model's prefill path and decline reason, and where E2B's time goes if the
    path is batched. A fix is a separate decision.

- **P27 · CPU f16 KV, way B: widen the stored rows inside the attention kernels — PARKED 2026-10-16 (owner). Revisit if CPU KV
  memory becomes the limit for a real user.** `mac`, aikit kernel work.

  - **The finding that parked way A as the default:** CPU f16 KV way A (`--kv f16`, opt-in since #55) narrows rows to f16 on
    append and widens the whole stored history into scratch on every decode step. F3's in-process A/B put it slower at depth:
    0.925 on the 0.5B at 4,096 keys, 4 of 4 pairs under 1 (resolved), and 0.861 on the 1.5B (3 of 4). So `auto` stays f32
    (`docs/completed/task-cpu-kv-f16-2026-10.md` §5b).
  - **Way B** (that doc's §4) reads f16 in the register, inside the kernels. Because the float64 path reads K and V in place, this
    is not one kernel: it is an f16 form of each of the four attention kernels (two of them arm64-only), plus the fast prefill's
    gather. Those kernels are aikit's, and its pushes and tags are the owner's.
  - **What reopens it:** a real user whose CPU context is bounded by KV memory (f16 halves it: 0.94 GB to 0.47 GB at 16,384
    positions on Qwen2.5-1.5B). Not speed: way B's best case is f32's speed at half the bytes.
  - **Gates when it reopens:** the task doc's F1–F5 as registered, with F3 against way B; F1's exactness rule (the f32 path with
    each stored row rounded to f16 by hand) carries over unchanged.
  - **Added 2026-10-21 (the close-out sweep, from `docs/completed/task-cpu-kv-f16-2026-10.md`):** way A's widening runs on one
    goroutine, while int8's was split across cores by R-15 (`dequantKVRows`), so F3 timed a slower pass than int8 pays.
    Splitting the f16 widening by rows the same way is bit-identical by construction and should narrow the 0.925 at depth
    without closing it (R-15 made int8 1.234× faster at depth 2,000 and left it at about 0.89 of f32). It speeds up the
    opt-in; it is not a reason to reopen the default. Either box, S, read by an in-process A/B by day.

- **P25 · `breakEvenTokensPerRound` is one constant (2.5) for every (target, backend); it does not
  transfer to a CPU MoE target and would NOT have caught a real regression.** Filed 2026-09-13 from
  `docs/task-laguna.md`'s DFlash pairing measurement, moved here at archival so the finding has a
  live home.

  **Measured, not inferred:** `poolside/Laguna-XS.2-speculator.dflash` against real
  `Laguna-XS.2` @int4, CPU BlockSpec with a real batched verify — 3.20 tok/round accepted (above
  the 2.5 constant) but **0.82× — SLOWER** end to end (5m58.7s speculative vs 4m54.8s plain
  greedy, output token-identical). Working backwards from the measurement, the real break-even on
  this target is ≈3.9 tok/round, not 2.5. **The runtime guard that disables drafting below
  `breakEvenTokensPerRound` would have happily kept this configuration running** — it is sound in
  shape and wrong in calibration for this regime.

  **The mechanism:** the 2.5 constant was calibrated in P10 for the GPU-resident (dense) regime.
  A batched verify over a sparse MoE does not amortize the way a dense one does — each verified
  row routes to its own top-k of experts, so an 8-row verify approaches 8× the expert traffic
  rather than the near-free extra rows a dense model gives you. Laguna is A3B: one decode step
  activates ~3B of 33B total, and the batching that makes speculation pay elsewhere is exactly
  what MoE routing defeats.

  **Not fixed on the strength of one target's measurement** — changing a shipped guard's
  semantics from a single data point would be the same over-generalization that put a
  GPU-calibrated constant in the CPU path to begin with. Two candidate fixes, either measured
  before landing: (a) calibrate `breakEvenTokensPerRound` per (target, backend) rather than one
  global constant, or (b) have the guard compare wall-clock directly instead of inferring a
  decision from an acceptance rate. **Do NOT wire Laguna's DFlash pairing into `serve --drafter`**
  until one of these lands — the pairing itself is lossless and works (`decoder/laguna_dflash_test.go`),
  it is simply not worth running today.

- **P24 · `attn_fused` runs at 1.72% of tensor peak and 12.6% occupancy — the L2 kernel that
  shipped is nowhere near its ceiling, and the first suspect is the GRID, not the inner loop.**
  Filed 2026-09-05 from the L2/L3 prefill campaign. Named and deliberately not chased at the time:
  the kernel had already cleared its pre-registered band (≥1.4× end-to-end; measured 1.70×), and
  tuning a kernel *before* its band is met is how `gemv_w4a8_batched` accumulated five successive
  attributions that were each recorded as a conclusion and refuted by the next measurement.

  **Measured, not inferred** (`ncu`, `attn_fused_hd128`, dense 1.5B int4 at K=2048, RTX 2070 SUPER,
  driver 595.91.07, median of 3 launches — `docs/measurements/prefill-l2l3-phase1-2026-09-05.md` §4):

  | metric | value |
  |---|---|
  | tensor pipe utilisation | **1.72%** of peak |
  | achieved occupancy (warps active) | **12.6%** |
  | SM throughput | 6.98% |
  | DRAM / L1TEX throughput | 8.3% / 9.8% |

  **Nothing is saturated**, so the kernel is still latency-shaped — far less so than the
  `attn_batched` it replaced, but the 3.76× it won on the attention category is not a ceiling.

  **The mechanism, counted rather than guessed.** The grid is `(nH, ceil(M/64))` = **12 × 8 = 96
  blocks**. `ptxas` reports **178 registers/thread**, so only **2 blocks fit per SM**, and 40 SMs
  hold **80 concurrently** — **1.2 waves**, with a large tail in which most of the card is idle.
  That alone bounds achieved occupancy near what was measured. A 32-row query tile (`BM=32`, two
  warps) doubles the block count to 192 while leaving registers/thread unchanged (the O
  accumulator is `hd/2 = 64` f32 per lane regardless of BM, since a warp owns 16 query rows either
  way), which should give ~5 blocks/SM and put the whole grid inside one wave.

  **Pre-registered decision rule, with the ambiguous band and the do-nothing arm:**
  - **Cell:** `TestPrefillDecomp` attention category, dense 1.5B int4, K=3900. Baseline is the
    shipped `BM=64` kernel at **805.2 ms** (the same instrument and depth §4 L2's band used).
  - **Arms:** `BM=64` (do-nothing, the shipped kernel) · `BM=32` · and, only if BM=32 disappoints,
    a K-split (flash-decoding-style) arm. **The do-nothing arm is not optional here:** the shipped
    kernel is already 3.76×, so "BM=32 is faster than nothing" is not the question — "BM=32 is
    faster than what ships" is.
  - **Ships** at ≥1.25× on the category *and* no regression at K=512 (where the grid is smaller
    and a narrower tile may lose). **Ambiguous → parked** at 1.05–1.25×, or any gain that comes
    with a K=512 regression. **Parks** below 1.05×.
  - **Any change must re-run the §3 fidelity gate at the floor and above.** BM is not a numerics
    knob in principle — the online softmax is per-row and tile-width-independent — but that is a
    claim, and the gate is how it gets checked. `TestAttnFused_vsF16Reference` is the cheap
    pre-check: it compares against exact f64 on the kernel's own f16 operands, so it isolates a
    tiling bug from operand precision.

  **What would make this NOT worth doing**, stated up front so the negative is publishable: at
  K=3900 attention is already down to ~25% of prefill after L2 shipped (805 ms of 3255 ms catSum),
  so even a *perfect* attention kernel is now capped at ~1.33× end-to-end. A 1.25× category win is
  ~1.06× end-to-end. **The category headroom is large and the end-to-end headroom is not** — L3
  moved the bottleneck back to the weight term, and that asymmetry is the reason this is filed as
  an open item rather than started.

- **P23 · A3 f32-attention fan-out at K=8192 — ANSWERED 2026-09-23, not on the original box; the
  Mac's own point stays open, and the trend did NOT extrapolate the way it looked like it would.**
  Still counted as open below: this settles whether the monotone-extrapolation assumption is safe
  (no), not the Mac's own missing cell. Filed
  2026-09-05, carried over from the 2026-09-01 A3 fan-out work. The shipped measurement
  (`docs/measurements/a3-f32-attention-fanout-2026-09-01.md`, MacBook) covers K=1024/2048/4096
  (1.58× at 2048, 1.92× at 4096) and stopped there on purpose — extrapolating past measured points
  is the exact error that same campaign spent a morning retracting elsewhere. The prediction on
  file was "the trend is monotone in K, so 8192 is *likely* higher than 1.92×." **Measured on
  nobara-pc (Linux/amd64), not the Mac — a fresh, self-contained, same-box curve, not a filled-in
  Mac cell (same-machine comparison rule):** 2.11× → 2.90× → **3.73×** → **3.12×** at
  1024/2048/4096/8192 — **the trend peaks around 4096 and recedes at 8192**, the opposite of what
  the monotone prediction expected, tight across 3 pairs (3.07/3.13/3.13, not a single-pair
  outlier). Checked and ruled out, not left as a guess: `prefillAttnWorkers`'s own scratch-budget
  ceiling (`decoder/scratch.go`) does not bind at either depth for this model's real geometry (both
  land at the `maxAttnWorkers=6` cap, computed from the GGUF's own head_count/embedding_length) —
  the rollover's real cause is NOT identified. Full method and numbers:
  `docs/measurements/a3-fanout-k8192-2026-09-23.md`. **The Mac's own K=8192 point is still
  unmeasured** — this doesn't close that, only the cross-box question of whether the shape
  generalizes, and it does not.

- **P22 · WebGPU Theta — CLOSED 2026-09-23, and it WAS another over-drafting regression like the
  pre-fix Metal one, not assumed, measured.** Filed 2026-09-05, carried over from the same
  2026-09-01 work as P21. CPU (0.5), CUDA (0.251) and Metal (≈1.02) all got real probes
  (`docs/measurements/theta-per-backend-2026-09-01.md`, `theta-cuda-ab-2026-09-01.md`); WebGPU did
  not — `gpu.residentDecoder` implemented neither `VerifyPathReporter` nor `PrefillPathReporter`,
  so `decoder.verifyTheta()` fell all the way through to the unmeasured 0.5 default. Measured
  (`gpu/theta_probe_test.go`, new): **Theta 0.978–1.028** across four real configurations —
  `ForwardN` records every row into one command-buffer submit, but that removes Go-side dispatch
  overhead only, not GPU-side compute, so the marginal wall-clock cost of an extra verify row
  stays ~1×, same shape as Metal's own defect. Fixed by wiring `VerifyPath()` on
  `gpu.residentDecoder` (returns `false`, routing to the shared `sequentialVerifyTheta = 1.02`
  constant — no new per-backend value needed). **Confirmed with a real A/B, not asserted from the
  probe alone** (`gpu/theta_ab_test.go`, mirrors Metal's own): under the old 0.5 default,
  speculative decode measured **1.22× SLOWER than not speculating at all** on real hardware; wired
  correctly, it lands within noise of plain generation (0.95×) and **1.28× faster than the broken
  default**. Full method: `docs/measurements/theta-webgpu-2026-09-23.md`.

- **P21 · Metal `ForwardN` batching — the controller now tells the truth about Metal's Theta;
  batching is what would CHANGE the truth.** Filed 2026-09-05, carried over from the
  2026-09-01 Theta-per-backend work — measured then, never filed. The shipped fix
  (`docs/measurements/theta-per-backend-2026-09-01.md`) wired Metal's Theta from a real probe
  (≈1.02, so speculation correctly declines to draft rather than running the shipped-default 0.5
  and over-drafting). That is honest reporting of the current mechanism, not a change to it.
  Batching `ForwardN` into one command buffer (one Submit/Poll) is the item with the actual
  upside — it is what would move Theta itself, at which point the Metal constant gets
  re-measured and replaced. Note `ResidentForward`'s interface doc already claims `ForwardN` runs
  "K tokens in ONE command buffer (one Submit/Poll)": Metal satisfies the bit-identity half of
  that contract and the batching half not at all, so the doc is currently writing a cheque the
  implementation doesn't honour. `metal/backend.go` is where the CUDA/Metal split lives.

- **P20 · CUDA batched prefill for the MoE families — CLOSED 2026-09-21, both paths BUILT AND
  SHIPPED DEFAULT ON.** The body below is kept in full as the reasoning trail (the ~8%/1.085×
  reading was real but was never the ceiling — see "What the 8% refutes"), but the header itself
  went stale: the generic path (`GOINFER_CUDA_MOE_EXPERT_MAJOR`) and the gemma4-specific extension
  it needed are both built and shipped, default on since 2026-09-21 (`84d113b8`) — **2.26–2.66× on
  the real M26** (`docs/measurements/p20-expert-major-m26-2026-09-21.md`), closely matching the
  ~2.32× projection this section itself derived from real distinct-expert-count data. Credited to
  the flagship page 2026-09-23: `docs/benchmarks.md`'s W3 M26 cell moved from 368.4s to **234.4s
  wall-clock (1.57× further)** — `docs/measurements/w3-m26-rerun-2026-09-23.md`. **M35 remains
  sequential and out of scope — it needs a batched Gated-DeltaNet, a different and harder problem
  (a recurrent state cannot be reordered across rows the way a stateless MoE FFN can); not started,
  not this item's closure condition.**

  **The one-line verdict, so a scanner does not have to reconstruct it:** all three blockers that
  kept Gemma-4 off the batched prefill path are removed and none needed a new CUDA kernel; the
  resulting speedup on the real 26B is 47.382 → 43.681 ms/token (M=512, same process, paired),
  **1.085×**. That lands in the pre-registered AMBIGUOUS band, and the reason it is not larger is
  the finding — see "What the 8% refutes" below.

  **What the W3 depth-8000 cells cost, measured** (`docs/measurements/peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json`,
  the `secs` field — not the tok/s column, which is a different deficit):

  | cell | goinfer | Ollama | llama.cpp |
  |---|---|---|---|
  | M35 @ 8000 | **1528.8 s** | 86.1 s | 61.7 s |
  | M26 @ 8000 | **384.3 s** | 85.4 s | 67.0 s |
  | D7 @ 8000 | 200.7 s | 34.0 s | 22.8 s |

  **~~`cuda/prefill.go`'s `prefillStaticDecline` refuses `r.moe || r.gemma4Moe` outright~~ — removed
  2026-09-04.** A MoE layer's FFN now runs row by row off the batched residual, so the attention
  half batches while the routed experts keep decode's exact per-token sequence.

  **The prize is bounded from two directions, both measured, neither on this path.** On CUDA's
  dense path the same restructuring is worth **12.480 → 2.777 ms/token** at shallow depth and
  **3.02× at 8012 tokens** (`docs/measurements/prefill-chunking-d7-2026-09-04.md`). On the CPU
  path P18's expert-major MoE batching measured **4.364× end to end** on a real 28-layer MoE
  prefill, bit-identical. Neither transfers as a number — they say the mechanism works twice, on
  both sides of the machine, not what it is worth here.

  **M26's blockers are now MEASURED, not read off the source** (probe:
  `cuda/prefill_longprompt_test.go` against `~/models/gemma4-26b-int4.giw` with
  `-moe-cache-experts`, `-ctx 8192`; log `docs/measurements/prefill-chunking-2026-09-04/m26-guards.log`).
  Loading it takes 2m11s and the C′ cache self-caps to 12 slots/layer:

  ```
  guards: prefillReady=true moe=true gemma4Moe=true sandwich=true qkNorm=true
  layers=30 non-uniform-vs-L0=5 kEqV=5 nonBatchableKinds=map[]
  (L0: hd=256 nKV=8 qDim=4096 kvDim=2048 rhalf=128 window=1024)
  hidden=2816 inter=2112   VRAM free after load: 0.44 GB
  sequential prefill: 46.701 ms/token at M=512, 46.434 ms/token at M=2048
  ```

  **M26's sequential prefill is FLAT with depth, and that is the single most fundable fact here.**
  46.701 → 46.434 ms/token from M=512 to M=2048 — no growth at all, against D7's 12.480 → 13.878 →
  15.665 → 19.211 over the same axis. The mechanism is visible in the geometry above: 25 of its 30
  layers are sliding-window at 1024, so their attention never grows, and the 5 global layers'
  growth is invisible under a weight term of ~46 ms.

  **That inverts the usual objection.** P19's whole argument is that attention dominates dense
  prefill at depth (55% at K=3900 on CUDA), which caps what batching the weight term can buy — and
  it is why D7's 8k win came out at 3.02× rather than the 4.5× the shallow point suggested.
  **M26 has no such ceiling: essentially all of its 46.5 ms/token is the weight term batching
  removes.** Whatever fraction of that traffic gets batched converts almost directly.

  What that is worth is still a projection, not a measurement, and is flagged as one: if the
  ~70% dense share holds and its weight term compresses the way D7's did (12.48 → 2.78 ms/token,
  ~5× on the weight term at shallow depth), 46.5 ms/token → ~20, i.e. the 6.4 min cell → ~2.8 min,
  with expert-major on top taking it further. **Do not quote those two numbers as results.**

  **Three of the five flags were already handled and two were not, which was smaller than it
  looked.** `sandwich` and `qkNorm` are batched today (`rmsnorm_f32_batched` / `qk_norm_batched`).
  `nonBatchableKinds` read empty, but partly as an artifact — see the sentinel note under (3), now
  fixed. What blocked, and where each stands:

  1. ~~**Per-layer geometry — 5 layers of 30.**~~ **DONE 2026-09-04.** `prefillCore` no longer
     hoists layer 0's `hd/nKV/qDim/kvDim/rhalf`; each launch binds its own layer's, and the M-sized
     scratch is allocated at the max across layers (`prefillMaxGeom`). Removing the hoist removed
     the only thing the uniformity assertion protected.
  2. ~~**`kEqV` — the same 5 `full_attention` layers.**~~ **DONE 2026-09-04**, in the ~10 lines
     `segA` uses: a second k-projection GEMV into the V buffer, then a scale-less `qk_norm` over it
     before `rope_kv_batched` rotates k. No new kernel — `r.bGemvB` and `r.bQKN` already take an M
     dimension.
  3. ~~**The FFN.**~~ **DONE 2026-09-04, route (a).** Gemma-4's parallel dense‖MoE branch
     (`gemma4MoeMLPPre/Post`) and the generic `moeMLPPre/Post` read and write `r.x`, the M=1
     residual, and so does everything between them (`segB`, `layerTail`, `segC`). Two ways in, and
     the choice matters:
     - **(a) Thread the residual buffer through** `segBFFN` / `layerTail` / `segC` /
       `moeMLPPre/Post` / `gemma4MoeMLPPre/Post` as a parameter, callers passing `r.x`. Cleaner, and
       it is the discipline the tree already applies one level down — `kvDim/rhalf` were *removed*
       from `cudaResident` specifically "so a launch site physically cannot bind the wrong (uniform)
       source". Cost: it touches the decode hot path.
     - **(b) Swap `r.x` to `xB.At(m*hidden*4)` for the duration of each row.** Three lines, and
       correct as written — the whole pass runs inside one executor job on one thread. But it is
       hidden mutable state of exactly the kind a captured graph bakes in, and this runner does
       capture `r.x` into `gSegB`/`gSegC`. Safe only while prefill never replays graphs, which is
       true today and is not a property anything asserts.

     **(a) was taken.** The residual buffer is now a parameter of `segB` / `segBFFN` / `layerTail` /
     `segC` / `moeMLPPre/Post` / `gemma4MoeMLPPre/Post` (17 `r.x` references), decode passes `r.x`
     unchanged, and prefill passes `xB.At(m*hidden*4)`. The routed experts still run per row, so this
     wins the attention half and nothing else — which is exactly what step 2 is measured against.

     **~~ONE REGRESSION THIS INTRODUCES, not yet fixed.~~ FIXED 2026-09-05.** `Prefiller.PrefillLast`
     took no `context.Context`, so a chunk was uncancellable once started — against a sequential
     loop that checks `ctx.Err()` **per token** (G18: "an abandoned client leaves the whole prompt
     streaming through the device"). On M26 that was ~23 s of uninterruptible work against ~46 ms.

     The interface now carries a context (Experimental tier, so the signature change is permitted),
     checked in **three** places: at entry, at each chunk boundary, and between rows of the MoE FFN
     loop. Measured on the real fixture: cancel at 237 ms returns at **239 ms**, against 1.183 s
     uncancelled — per-row granularity, i.e. roughly what the per-token fallback gave. A cancelled
     prefill returns `ctx.Err()` rather than falling through to the sequential loop the caller was
     also cancelling.

     **Each of the three checks is separately mutation-proven, and that is not ceremony — two of
     them were untested when first written.** The gate used a MoE fixture, whose per-row check
     catches a cancel before the chunk boundary can, so deleting the chunk-boundary check left the
     suite GREEN. A dense fixture (no row loop) at a forced chunk width was needed to make it fail.
     Chasing that surfaced a real hole in the shipped code: a prompt at or under the chunk width
     skips the loop entirely and had **no check at all** — fully uncancellable on a dense model.
     Hence the entry check, which needed a third sub-test because the other two stayed green
     without it.

     **RESIDUAL, still open:** `PrefillSeedArgmax` (block-spec prompt seed) ingests a whole prompt
     and remains uncancellable — `decoder.ResidentSeedArgmax` carries no context. It is a second
     interface on a different seam, deliberately not changed as a side effect of this one, and
     `cuda/prefill.go`'s `context.Background()` there points at this paragraph.

     **~~A trap that goes LIVE the moment the `r.moe` decline is removed.~~ FIXED in the same
     change**, as this said it must be. For the record, because the shape recurs: `nonBatchableKind` returned
     the first projection whose kind is neither int4 nor int8 — and an ABSENT weight has kind `""`,
     which it returns, and which the caller reads as "no problem" (`if k := …; k != ""`). On a pure
     MoE layer (Mixtral-class) `Ly.g/u/d` are unset, so the check passes vacuously, and it never
     looks at `Ly.expGU` / `Ly.expDown` at all — the weights that would actually be batched. It is
     harmless only while `r.moe` declined two lines earlier. It now checks exactly the projections the
     batched path binds — skipping `Ly.v` on a K=V layer and `g/u/d` on a pure-MoE one, both
     legitimately absent — and returns `"absent/unquantized"` instead of the empty sentinel. The
     expert stacks are deliberately still NOT checked: the per-row FFN issues decode's launches on
     decode's weights, so whatever kind decode accepts it accepts here; only the M-wide GEMVs
     constrain anything. The M26 probe's `nonBatchableKinds=map[]` was partly this artifact and is
     now a real reading.

  **(3) has a no-new-kernel route that the tree's own comment says does not exist.**
  `resident.go` states "gocudrv exposes no buffer view/offset, so the split is the kernel's
  gOff/uOff rather than Go-side pointer arithmetic" — but `aikit/gpu.Buffer.At(byteOff)` returns a
  zero-copy sub-view, is already used for the C′ expert-slot DMA (`cuda/resident.go:cudaResident.appendExpertSlot`), and its
  `arg()` binds the offset as a raw device pointer. So the existing per-row MoE kernels can be fed
  row *m* of a batched residual as `xB.At(m*hidden*4)`, and the first slice — batch the attention
  half, loop the FFN per row — becomes a Go refactor (thread the residual buffer through
  `moeMLPPre/Post` and `gemma4MoeMLPPre/Post` instead of reading `r.x`) rather than a PTX project.
  **Correct that comment when this is picked up; it is what scoped this item wrong once already.**

  **Chunking is a PREREQUISITE here, not a coincidence.** M26 has 0.44 GB free after load. Its
  dense scratch is ~104 KB/row, so an 8k single pass would need ~855 MB and could never have run;
  at the 512-row chunk it is ~53 MB and fits.

  **WHERE M26'S PREFILL ACTUALLY GOES, measured with `GOINFER_MOE_CACHE_PROF=1`** (M=512, both arms
  one process, `docs/measurements/prefill-moe-m26-2026-09-04.md`):

  | arm | wall | C′ DMA | share | `loadRoutedExperts` calls | syncs |
  |---|---|---|---|---|---|
  | batched | 22.217 s | **12.839 s** | **59.5%** | 15,360 | 15,360 |
  | sequential | 24.231 s | 12.819 s | 54.5% | 15,360 | 15,360 |

  **The DMA is identical in both arms to within 0.16%.** Strip it out and the batchable remainder is
  11.412 s → 9.378 s, **−17.8%** — that is the true size of the change, diluted to 8% by a floor it
  cannot move. **Amdahl on the MEASURED share: with 59.5% untouchable, everything else going to zero
  caps any further batching at 1.68×.** The transfer is per-row and synchronizes once per
  (row, layer). Expert-major is the one restructuring that changes per-row expert traffic — a layer
  would fetch each distinct expert once per chunk instead of once per row.

  **No number is projected for it.** The bytes moved were not captured, only calls and time, so the
  fetch-count reduction cannot become a time estimate without assuming the driver is fetch count and
  not per-call overhead — the same shape of assumption this item has already had refuted once today.
  Capture `UploadProfForTest`'s byte counters in the next probe and size it from those.

  **MEASURED 2026-09-21** (`docs/measurements/p20-expert-locality-2026-09-21.md`): a 512-row chunk touches
  28-117 distinct experts per layer (median 66) against this box's 10-slot C′ cache — infeasible to hold
  ALL of them resident at once (that reading of "stage each distinct expert once per chunk" was this
  record's own first, wrong one), but a real expert-major design processes experts sequentially/in small
  waves and needs no such simultaneous residency. Real numbers: current per-row LRU already gets 61.3% hit
  rate (47,575 misses); a perfect once-per-chunk fetch would need only 2,090 (98.3% hit rate) — a **22.8x**
  reduction in remaining DMA calls. Calls-proportional on the 59.5% DMA share measured above: **~2.32x
  projected (46.5 -> ~20 ms/token)**, inside R11(b)'s own ships band — a projection, not a result, same
  caveat as the one below it.

  **BUILT AND SHIPPED OPT-IN 2026-09-21** (`docs/measurements/p20-expert-major-2026-09-21.md`,
  `GOINFER_CUDA_MOE_EXPERT_MAJOR`, default off): route+bucket+admit-once-per-distinct-expert instead of
  once per (row, rank), no new kernel (reuses the frozen `moe.ptx` GEMVs and `residual_batched`),
  bit-identical by construction, mutation-checked on two fixtures. **Generic MoE path only — declines
  `Ly.g4moe`, so M26 itself does not take this path and the ~2.32x above is still unmeasured.** The only
  real checkpoint on this box that reaches the built path is Mellum2 (a much looser 51-slot / 98.1%-hit-rate
  cache than M26's), where it measures a real but small 3.5-4.3%.

  **BUILT AND SHIPPED 2026-09-21** (`cuda/moe_expert_major_gemma4.go`, `docs/measurements/p20-expert-major-m26-2026-09-21.md`):
  the gemma4-specific extension, same mechanism against `gemma4MoeMLPPre/Post`'s parallel dense‖MoE shape. Measured
  on the real 26B: **2.66x / 2.50x / 2.39x / 2.26x at M=512/2048/4096/8012** — the ~2.32x/~20ms projection held, and
  every cell clears the ships band by a wide margin. Sequential control unmoved. `GOINFER_CUDA_MOE_EXPERT_MAJOR`
  flipped to DEFAULT ON (both the generic and gemma4 paths; `=0` opts out), mirroring the CPU P18 precedent.

  **Sequenced work, cheapest first:**
  1. ~~**M26, steps 1–3 above**~~ **DONE 2026-09-04 — 1.085×, bit-identical.** Kept for the
     structure, not the number. — per-layer geometry, `kEqV`, and a per-row-fed FFN off a batched
     residual. No new `.cu`. Do this first: M26 is the cheaper of the two models and its sequential
     cell is 6.4 min rather than 25.5.
  2. ~~**Expert-major routed experts — NOW THE WHOLE ITEM**~~ **DONE 2026-09-21 — 2.26–2.66× on
     the real M26, default on.** Kept for the structure, not the plan: it turned out to need NO new
     `.cu` after all — the projected "batched router/gather/indexed GEMV" below was superseded by a
     simpler real design (route every row, bucket by expert, admit/DMA each distinct expert once,
     fold in rank order), reusing the frozen `moe.ptx` GEMVs unchanged. See the BUILT AND SHIPPED
     notes above for the real numbers.
     <details><summary>Original plan text, superseded, kept for the record</summary>
     (the P18 shape on the GPU): a batched
     router over M rows, a per-expert row gather, an indexed batched GEMV. This one DOES need a new
     `.cu` — `moe.ptx` is the audited 12.6.85 artifact and must stay untouched, the same way
     `decode_splitkv.cu` and `router_f32.cu` were added beside it. NVRTC is available on this box
     (`~/.venv-vl/…/libnvrtc.so.12`, `~/cuda-toolkit/…`), so `build_ptx.sh` runs. **On a model that
     fits the card this is a compute lever; on M26 it is a DMA lever, and the DMA is 59.5%** — so it
     is worth pre-registering the two cases separately rather than assuming one number covers both.
     </details>
  **A GUARD THAT EXISTED ONLY BY ACCIDENT, now explicit (2026-09-04).** Removing the categorical
  `r.moe` refusal also removed the only thing keeping **M35** off the batched path — and the batched
  pass has no notion of recurrent state, so a DeltaNet layer's conv ring and matrix state would have
  been advanced M rows at a time, out of order. `ForwardN` had excluded this deliberately since it
  was written (`r.prefillReady && r.dnet == nil`); `PrefillLast` never had to, because qwen3_5_moe is
  MoE and `r.moe` was refusing it for an unrelated reason.

  It was still refused after the change — but only because a DeltaNet layer loads no q/k/o, so
  `nonBatchableKind` reported the absence. **That is a fact about this family's weight layout, not a
  statement about recurrence**, and a hybrid whose recurrent layers also carried q/k/o would have
  sailed past it into the dense attention stack: the LFM2 bug class (audit-2026-09-02 C-01), which
  CLAUDE.md records as having reached `main` twice. `prefillStaticDecline` now refuses `r.dnet != nil`
  for the reason that is true, gated by `TestPrefillPath_recurrentDeclines` — whose fixture is a
  DeltaNet model *with* valid int4 q/k/o, precisely the case the accidental guard cannot catch, plus
  a no-dnet control so the test cannot pass against a guard that refuses everything.
  Mutation-proven: disabling the check turns it red.

  3. **M35 additionally** needs a batched Gated-DeltaNet: 30 of its 40 layers are
     `linear_attention` with in-place recurrent state, which is a chunked delta-rule scan and real
     new math. It is the only part of this item that is not a restructuring.

  **What is still NOT known:** how M26's 46.7 ms/token splits between attention, the dense FFN
  branch, and the routed experts. Active-parameter arithmetic on its config says ~45% attention and
  ~25% dense FFN, but that is arithmetic on shapes, not a profile, and this repo has retracted an
  Amdahl projection twice this fortnight. `r.prof` only instruments the batched path, so the
  attribution seam has to be extended (or use `ncu`). **Measure the split before building to it.**

  **~~Decision rule, pre-registered~~ — RESOLVED, AND THE RULE WAS MIS-SPECIFIED.** It read: measure
  the attention-half slice end-to-end on M26 at depth 8000, paired and interleaved; fund
  expert-major if it lands ≥15%, park if <8%, 8–15% ambiguous → parked pending a profile naming a
  second mechanism. Measured **8.3–8.5% ⇒ ambiguous band**, and the profile above names the second
  mechanism, so on its own terms the item continues.

  Two things to record rather than gloss. **It was not evaluated as written**: the measurement was
  in-process at M=512/2048, paired within one process but not interleaved and not at depth 8000 —
  a close proxy (both arms are flat with depth) is not the same as the rule being satisfied.
  **And its inference ran backwards.** "A small attention-half win ⇒ do not fund expert-major"
  assumes the two halves compete for one bottleneck. They do not: a small attention-half win means
  the experts dominate, which argues *for* expert-major. Per CLAUDE.md's own corollary the fix is a
  second, independent pre-registration — for step 2 that means pre-registering the DMA-bytes
  reduction and the wall-clock win separately, since on a model that fits the card only the second
  exists.

  **Bit-identity is required, not optional.** Both landed precedents are bit-identical (P18 by
  preserving per-row rank-order folding; the CUDA chunking by construction over a positional KV),
  and an MoE path that re-associates the expert fold would change output for a speed win — a
  different, worse trade that would need its own flag and its own argument.

  **What landed on 2026-09-04 (the dense half), so it is not rebuilt:** CUDA batched prefill was
  all-or-nothing on M and its scratch is O(M·inter), so on an 8 GB card a D7-class model at
  `-ctx 8192` OOM'd at ~7000 rows and fell back — silently, and only for the prompts long enough to
  need it. `prefillChunked` now runs the prompt in passes of ≤512 rows over the positional KV
  (bit-identical, `TestPrefillChunked_bitIdentical`), the load-time report states the width instead
  of claiming "one pass", and a call-time decline warns once. **This does nothing for M35/M26** —
  they decline at a different gate, which is this item.

- **P24 · Decode throughput falls off with KV depth ~2× faster than the peers' — OPEN, and it is
  the deficit the W3 table actually shows.** Filed 2026-09-05 because it was not filed anywhere:
  *(Filed as P24, not P21: P21–P23 were taken by another session's filing of the 2026-09-01
  measurements while this was in flight. Renumbered rather than shadowing an existing entry — the
  same call this queue records making for P19-not-P17.)*
  it existed only as prose in `docs/tasks/task-peer-benchmarks.md` §8 and as a mechanism note inside P19,
  neither of which is a queue.

  **Measured, `nobara-pc` CUDA, `docs/measurements/peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json`:**

  | model | goinfer @128 | goinfer @8000 | retained | llama.cpp @128 → @8000 | retained |
  |---|---|---|---|---|---|
  | D7 (dense 7B) | 73.0 | **35.7** | 0.49 | 80.4 → 58.7 | 0.73 |
  | M26 | 24.6 | **15.5** | 0.63 | 26.0 → 22.7 | 0.87 |
  | M35 | 23.5 | **21.4** | 0.91 | 32.4 → 30.8 | 0.95 |

  **This is a DECODE deficit and no amount of prefill work touches it** — P20 moved M26's cell wall
  clock 384.3 s → 368.4 s and left its decode rate at 15.5 tok/s, exactly as a prefill-only change
  should. Quoting the tok/s table as evidence for or against P20 conflates the two, which is the
  mistake §8 now warns against.

  **The design record already exists — do NOT write a second one.**
  `docs/tasks/task-decode-splitkv-attention.md` is this problem, written up and cited from
  `cuda/resident.go`, `cuda/kernels.go` and `cuda/splitkv_bitident_test.go`: decode attention at M=1
  launches one block per query head, 11.9% achieved occupancy on a 40-SM card, "latency-bound purely
  because there are too few blocks to hide memory latency", against Ollama's flash attention. Its
  P6a section also records the threshold's own history, including that a constant characterized on
  one geometry and applied to all of them cost 18–25%. This entry holds the open work; that doc
  holds the design, and its new "OPEN: every threshold in the table stops around 3900 keys" section
  is the specific gap.

  **The mechanism is named but NOT confirmed as the dominant term.** Per-token attention cost at 8k:
  goinfer ~14.3 ms against llama.cpp's ~4.6 ms (back-solved from the rates above, so treat as
  indicative). D7 has nH=28, and `splitkvThreshold` returns `splitkvNever` for nH ≥ 24 — so at depth
  8000 its attention runs the single-block kernel, 28 blocks of 128 threads on a 40-SM part, ~70% of
  the device idle. **That threshold was anchored on phi3-mini measured to depth 3900 and extrapolated
  to 8000, where it was never measured.** P19 independently recorded the same shape from the other
  side: goinfer's marginal cost per PREFILL token rises with depth (0.377 → 0.932 ms/tok) while
  Ollama's stays flat (0.064 → 0.063).

  **Cheapest first step, and it needs no code:** `GOINFER_SPLITKV_MIN_KEYS=0` force-enables the
  split path on a stock binary — the A/B handle that variable exists for. Run it e2e at depth 8000
  on D7 against the default. **Do NOT set the shipped threshold from
  `TestSplitKVCrossover`**: its own doc comment records that it is an in-process microbenchmark
  whose "break-even at 256" reading was refuted e2e on that very geometry, and thresholds are set
  from the e2e table in `docs/benchmarks.md` §B6.

  **Pre-register before running**, and pre-register two things that can disagree (the P18/P20
  lesson): the depth at which split-KV wins on THIS geometry, and the end-to-end tok/s delta at
  8000. A kernel-level win that does not move the e2e number is the outcome this repo has already
  had twice.

- **P19 · Fused (FlashAttention-style) tiled attention — SHIPPED 2026-09-01, default ON under
  `--cpu-fast-attention`. 1.69–1.73× at the kernel, +8.0% end to end.**

  **The end-to-end number is the one to quote, and it is modest.** Dense 1.5B, K=4096, paired and
  interleaved: **1.080×**. The kernel win does not carry because A3's head fan-out already
  collapsed attention's share to ~18% of this prefill (Amdahl back-solve) — P19 is competing for
  what A3 left. On the MoE profiled today attention is 17.4%, so it is no better there.

  **It ships on an operator decision, not on the measurement.** +8% bought with a user-visible
  output change is a worse ratio than the flip that introduced the flag (1.43–2.28×), and the
  recommendation from the measurement was to leave it off. Recorded so the decision is
  attributable.
  *(Filed as P19, not P17: this queue already has a P17. The item arrived carrying that number
  from outside the repo; renumbered here rather than shadowing an existing entry.)*

  **Step 0 asked whether G20's tile loop fuses the softmax or materializes per-tile scores, and
  said the item closes at zero cost if already fused. It is NOT fused.** The tile is over QUERY
  ROWS only — `scores` is `tile × nKeys`, the full key extent per row — and `attendBatchedHeads`
  forbids the fused schedule explicitly: *"No key-dimension split happens here and none may: that
  would re-associate the softmax denominator and the AV fold, the exact thing acc64 exists to
  prevent."* G20's own stated purpose is *"The point is memory, not speed"* — bounding scratch so
  the worker pool can still fan out. So the N-wide score row makes three trips through memory per
  tile, which is the traffic a fused schedule removes. **Written down because nobody knew.**

  **The prize, measured rather than assumed** (`docs/measurements/cuda-prefill-attention-share-2026-09-01.md`).
  Attention's share of CUDA prefill, dense 1.5B int4, RTX 2070 SUPER:

  | K | 128 | 512 | 2048 | 3900 |
  |---|---|---|---|---|
  | attention % | 5.3% | 14.3% | **39.0%** | **55.0%** |

  K=2048 reproduces the 2026-08-04 figure of 39% exactly, across a driver/distro re-anchor — the
  instrument validated itself before the new cell was read. Amdahl at K=3900: a perfect fusion caps
  at **2.22×**, a plausible 2.5× on the attention term gives **~1.49×**.

  **THE DECIDING VARIABLE IS MODEL CLASS, NOT BACKEND**, and getting that wrong nearly killed the
  item. A first pricing used the 2026-09-01 full-model profile's **17.4%** attention share and
  concluded a 1.21× ceiling — but that is **Mellum2, a MoE**, whose expert FFN takes 42.1%. Dense
  models at depth run ~70% (Mac CPU acc64, K=8192) and 55% (CUDA, K=3900). Attention dominates at
  depth on dense on BOTH backends; the 17.4% is a property of the model, not of prefill.

  **It coheres with the CUDA prefill re-anchor.** That measured goinfer's marginal cost per prefill
  token RISING with depth (0.377 → 0.932 ms/tok) while Ollama's stays FLAT (0.064 → 0.063). A
  rising marginal cost is the O(K²) attention term; a flat one is what a tiled/fused attention
  produces. Three independent measurements today point at the same mechanism.

  **Cost, unchanged from the filing:** the running-max rescale re-associates, so a fused path is
  NOT bit-identical — same category as `--cpu-fast-attention`, breaking the same A1 guarantees
  (spec-decode verify == sequential greedy, decode == prefill).

  **Two notes on the item's own text, now stale.** Its confound warning — *"the f32 gather branch
  is single-threaded by construction (shared mutable kh/vt state), so any end-to-end A/B that moves
  both precision and fusion at once measures the parallelism confound"* — was fixed on 2026-09-01
  (A3 fan-out: each worker gathers into its own buffers). And its sequencing condition, "A3 first",
  is satisfied: A3's kernel measurement and its end-to-end result both landed.

  **THE PROTOTYPE WAS BUILT AND MEASURED THE SAME DAY, AND THE VERDICT REVERSED WITHIN IT**
  (`docs/measurements/p19-fused-attention-2026-09-01.md`). At production shapes (kt=256, hd=128,
  nKeys=8192), f32 both arms so the schedule is the only variable:

  | configuration | materialized | best fused | ratio | verdict implied |
  |---|---|---|---|---|
  | serial | 52.9 ms | 51.3 ms | 1.031× | wash |
  | column-parallel (over `MatmulBT`) | 37.8 ms | 54.0 ms | 0.700× | ✗ close |
  | row-parallel, unmasked | 15.7–16.2 ms | 9.0–9.1 ms | 1.73–1.81× | clears |
  | row-parallel, causal, last tile only | 11.0 ms | 9.2 ms | 1.19–1.25× | ✗ park |
  | **row-parallel, causal, ALL 32 TILES** | **243–257 ms** | **144–148 ms** | **1.69–1.73×** | **✓ CLEARS** |

  **Three of five configurations imply the wrong verdict**, all for methodological reasons — the
  arithmetic is correct f32 in every one (cosine 1.000000000 throughout). The bottom row is what
  production runs: row-parallel is the shape A3 ships, causal is prefill's only masking, and a
  prefill is the SUM over its tiles.

  Pre-registered bar ≥1.30× clears / <1.10× closes. Correctness held everywhere: cosine
  1.000000000, max|diff| ~1e-8.

  **This item was briefly closed on the 0.700× row, and that was wrong.** Composing fusion over a
  COLUMN-parallel matmul forfeits parallelism by construction — materialized presents N=8192 to
  `MatmulBT`, fused presents N=kb. The page had written that down as a caveat and drawn a verdict
  anyway. **A schedule that is bad at exploiting one parallelism axis has not been shown to be
  bad.** Re-run with both arms row-parallel at equal worker count and an identical serial inner
  primitive, fusion wins.

  **The mechanism, which only the parallel arm reveals.** Serially the schedule is neutral, so
  fusion saves no arithmetic. It wins on SCALING: materialized hands each worker a `[n, nKeys]`
  score array (1.4 MB here) and scales 3.7× on 8 workers; fused hands it `[n, kb]` (~44 KB) and
  scales 7.3×. Eight workers streaming 1.4 MB arrays are bandwidth-bound; on 44 KB blocks they stay
  in cache. That is the FlashAttention argument, and it is invisible without parallelism.

  **Correctly sequenced against A3, as the item demanded.** The column-parallel arm is the *pre-*A3
  f32 shape; row-parallel is analogous to the head fan-out A3 shipped the same day (3.27× at the
  kernel). So 1.75× is a win **on top of** A3's, not a re-measurement of it.

  **Next step: production.** `attendBatchedHeads` already fans out over heads with a serial inner
  matmul (A3), which is the configuration where fusion wins — so the schedule change lands in a
  shape that already exists rather than requiring a new parallelism model. It costs the A1
  guarantees (the running-max rescale re-associates; same category as `--cpu-fast-attention`), so
  it needs the same flag-and-floor treatment and its own golden.

  **Still not tested:** the CUDA form (where the 55% share was measured and the score matrix goes to
  HBM — the CPU result makes that MORE interesting, not less), and a hand-written SIMD fused inner
  kernel (these arms deliberately keep `MatmulBT` inside, so kernel quality is controlled rather
  than confounded).

- **P18 · Expert-major MoE prefill batching — FUNDED AND SHIPPED 2026-09-01, default ON.
  4.364× end to end, bit-identical.**

  **Why it is back.** The full-model K=8192 profile
  (`docs/measurements/mellum2-fullmodel-profile-RESULT.md`) moved the target after A3 collapsed
  attention: `moeMLP` is now **42.1%** of prefill against attention's 17.4%, and inside it
  `swiGLUExpert` is **93.1%** — so the expert weight matmuls are **~39% of prefill and the largest
  single bucket**. `routeExperts` is 1.7% of `moeMLP`; routing is not the cost and never was.

  **Why the existing verdict does not close it.** `docs/completed/task-moe-streaming.md` Lever 4 is parked on
  "expert-major MoE prefill batching is NOT a compute lever", whose ceiling was `uniform` (every
  row picks the same experts) vs `varied`. **Both arms call `moeMLP` per row at M=1**, so that
  experiment varies *which weights are touched* — the bandwidth/locality axis — and holds the
  M=1→M=N axis fixed. Measured 2026-09-01 at real Mellum2 expert shapes with int4 weights and
  locality already perfect (i.e. under the parked ceiling's own condition): **1.55× at N=8 rising
  to 2.13× at N=256**
  (`docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md`). Real N is ~10³ rows per expert
  per layer at K=8192.

  **What is NOT yet known, and is the whole item.** The gather/scatter cost of collecting an
  expert's scattered rows is unmeasured, and it is the most likely reason the real win lands below
  2.13×. Uneven routing puts small groups at the left of that table (1.55×), not the right. And
  the microbenchmark is cache-warm on three matrices while the real forward ran with swap growing
  to 18 GB — a benchmark that omits the real pressure can promote a mechanism as easily as
  exonerate one ([[synthetic-reproduces-shape-not-pressure]] is about the same trap in the other
  direction).

  **No end-to-end speedup is claimed.** Multiplying 39% by 2.13× is the projection this repo
  retracted twice on 2026-09-01. **Decision rule, pre-registered here:** measure permutation cost
  against batching win on the real forward at K≥4096; **fund if the net is ≥15% end-to-end,
  park if <8%, and treat 8–15% as ambiguous → parked pending a second mechanism.**

  **MEASURED: +336.4%, i.e. 4.364×** (full 28-layer Mellum2, K=4096, paired and interleaved;
  1206.9 s → 276.6 s, second pair 4.50×) — clears the bar by more than twenty times. Shipped
  default ON; `GOINFER_MOE_EXPERT_MAJOR=0` disables. **Bit-identical**, so no golden changes, no
  documented divergence, no user-facing flag: it changes speed and nothing else.

  **The mechanism I proposed is REFUTED, which is the durable part.** Amdahl on the microbenchmark
  (2.13× on matmuls that are ~39% of prefill) predicts 1.26×; measuring 4.36× said something else
  was doing the work. I proposed per-row allocation — `moeMLP` allocates ~5 slices per row per
  layer and the K=8192 profile recorded 339,293 GCs / 20.9 GB. Measured as its own arm, changing no
  arithmetic: **0.99× and 1.02×. Nothing.** The cheap five-line alternative is dead.

  Leading hypothesis for the remaining gap, **not measured**: the microbenchmark reused one
  expert's weights cache-warm, while production touches 8 of 64 experts per row with consecutive
  rows differing, on a machine whose working set drove swap to 18 GB. If so my own benchmark
  understated the win by omitting the pressure — the "synthetic reproduces shape, not pressure"
  trap, in the direction that would have killed the item rather than oversold it.

  Record: `docs/measurements/p18-expert-major-e2e-2026-09-01.md`.

  Structural note for whoever picks it up: `canBatchN`'s comment states the per-row design as a
  deliberate constraint ("the MoE FFN itself stays per-row"), so this is a restructuring of the
  prefill loop, not a kernel swap.

An observation rather than a work item — nothing below is claimed as a bottleneck to go fix.

- **P17 · `TestSamplingThroughputGate` has ~1% headroom on arm64 and fails under suite load** —
  **REOPENED 2026-09-06: the best-of-3 fix is still in the code and the x86 spread is back.**
  The 2026-08-31 diagnosis below stands and the bar still must not move; what does not hold is the
  claim that best-of-3 cut the run-to-run spread to 0.39x.

  Measured 2026-09-06 on `nobara-pc`, V=262144, five runs of the unchanged tree — one under load
  from another session's CUDA benchmark, four on an idle box (load average 0.48):

  | run | temp-only (denominator) | temp+top_p (numerator) | ratio |
  |---|---|---|---|
  | under load | 1.068 ms | 6.268 ms | **5.04 FAIL** |
  | idle 1 | 1.359 ms | 6.430 ms | 4.73 |
  | idle 2 | 1.682 ms | 6.525 ms | 3.88 |
  | idle 3 | 1.433 ms | 6.473 ms | 4.52 |
  | idle 4 | 1.234 ms | 6.407 ms | **5.19 FAIL** |

  **Spread 1.31x, against the 0.39x the fix was recorded as achieving** — essentially the pre-fix
  1.35x. The gate fails roughly one run in four on a tree nobody has touched.

  **Why best-of-3 cannot fix it: the variance is BETWEEN PROCESSES, not within one.** Fifteen
  consecutive samples inside a single test binary put temp-only at min 1.611 / p50 1.724 / max
  1.909 ms — a 19% spread around a stable floor, and that process's ratio-at-floor was 4.05x. A
  minimum taken inside one process faithfully reports *that process's* floor, and the floor itself
  moves ±30% from one process to the next. More repetitions inside the binary buy nothing.

  **And the failure direction is perverse.** The numerator is the stable quantity (6.27–6.52 ms,
  4%); every failing run is one where the machine ran the DENOMINATOR *well*. The gate fires when
  the thing it divides by gets faster — which is exactly the mode its own comment already records
  from P2b, when a 4.7x denominator improvement forced the bound up.

  Not fixed here, and deliberately not re-bounded: raising the bar to fit today's spread is the
  move `docs/benchmarks.md`'s methodology exists to prevent. The shape of a real fix is to stop
  dividing by a co-measured arm — gate the numerator's absolute cost against a recorded floor with
  provenance — which is a redesign of the gate, not a constant.

  ---

  *Superseded status line, kept because its measurement and its refuted prediction are still the
  useful record:* **DIAGNOSED AND FIXED 2026-08-31. The bar was NOT moved.**

  **The question was: is full-vocab selection genuinely load-sensitive, or was the gate set with no
  headroom on this machine class? Neither, quite.** The gate is a RATIO, and both its level and its
  variance come from the DENOMINATOR (temp-only) — which is not the thing under test. Measured, three
  consecutive isolated runs at V=262144:

  | | temp-only (denominator) | temp+top_p (numerator) | ratio |
  |---|---|---|---|
  | x86 | 1.356 / 1.694 / 1.994 ms — **47% spread** | 6.54 / 6.72 / 6.93 ms — 6% | 4.83 / 3.97 / 3.48 |
  | arm64 | 558.7 / 564.6 / 558.6 µs — 1% | 2.682 / 2.685 / 2.680 ms — **0.2%** | 4.80 / 4.76 / 4.80 |

  The numerator — the "full-vocab selection" the gate claims to protect — is the most stable
  quantity in the measurement, on both machines. The ratio moves because the baseline moves, which
  is what the gate's own comment already warned it does ("a gate whose baseline moves is measuring
  two things at once", written when P2b made that denominator 4.7x faster and the bound had to be
  raised).

  **Fix: best-of-3 instead of a single `testing.Benchmark` mean.** A mean tracks jitter upward — it
  has a floor but no ceiling — and the quantity is floored, so the minimum is the right estimator.
  Under full-suite load, the exact condition that produced the original failure: **5.13x FAIL →
  4.82x PASS.** x86 run-to-run spread cut from 1.35x to 0.39x.

  **A prediction I made and the measurement refuted**, recorded because the correction is the
  useful part: I expected best-of-N to make the ratio machine-independent (predicted x86 4.83 vs
  arm64 4.80). It does not — x86 settles near 4.0 and arm64 near 4.83, a real 0.7x gap on the same
  code. The comment in the test said the wrong thing for one commit and now says the measured one.

  **Residual, NOT fixed:** the bound is effectively set by whichever machine runs hottest on this
  ratio (arm64, ~3% margin), and the denominator remains an optimization target — any future
  speedup to temp-only re-tightens this gate for everyone. The structural answer is to gate the
  NUMERATOR, which is stable to 0.2%; the reason that was not done here is that an absolute
  numerator bound is machine-dependent (2.7 ms arm64 vs 6.5 ms x86), which is the problem the ratio
  was chosen to avoid. Filing that trade rather than resolving it.

  Measured while landing G1, on a tree whose diff contains **no sampler file** — the benchmarked
  code is byte-identical to `main`, so this is not a regression from that work:

  | context | V=262144 temp+top_p ÷ temp-only | vs gate 5.0x |
  |---|---|---|
  | inside the full `./decoder/` suite | **5.13x** | **FAIL** |
  | isolated, `-count=1`, four runs | 4.81 / 4.95 / 4.79 / 4.97 | pass, by 0.6-4% |

  So the gate passes alone and fails under concurrent load, with a mean margin under 3%. Two
  traps worth naming before anyone picks this up:

  - **`go test` CACHES a passing result.** The first three "repeats" here returned byte-identical
    timings (566997 / 2759883 ns/op) because only the first actually ran. Any repeat-measurement
    of this gate needs `-count=1`, or it measures nothing and looks stable while doing it.
  - **Do NOT raise the bar to 5.5x to make this green.** Re-baselining because a number moved is
    how a regression gets blessed; a bar moves only with a mechanism. The open question is whether
    full-vocab selection is genuinely load-sensitive (a cache-pressure story) or whether the gate
    was set with no headroom on this machine class in the first place. Answer that first.

## Where the last items went

| item | disposition |
|---|---|
| **P16** · re-anchor to Nobara 44 / driver `595.91.07` | **DONE 2026-08-31.** Six legs were already re-anchored on 2026-08-27; the seventh (the v0.11.0 qualification) was **retired** rather than re-measured — its numbers were §B6/§B7 by code-identity, and both are current. |
| **P14** · the CPU gap is kernel arithmetic | **DONE.** Items 1+2 refuted (centering), item 3 built, wired, measured (+2.10% decode) and **parked default-OFF** against its pre-registered 4% bar. |
| **A2** · 26B documentation correction | **DONE 2026-08-31**, published as `docs/benchmarks.md` §B4.2. |
| **A10** · the ~150 MiB allocation floor | **CLOSED 2026-08-12** — fully decomposed, nothing unattributed. |
| **D3b** · the expert-cache default | **SHIPPED 2026-08-20** (`8f3c5e7`). Lives in `docs/queue-release.md`. |
| **P10** · DSpark / DFlash block drafters | **MOVED to the speculation track** — see below. Not finished; not performance-queue debt. |
| **P15** · DFlash 2 | **MOVED to the speculation track** — see below. |

## P10 and P15 moved rather than closed, and the distinction matters

They are **not done**, and nothing here should be read as saying they are. They left this queue
because they were never really performance-queue items: the rest of this queue was *find a
bottleneck, measure it, fix or refute it*, and those two are an ongoing research program with
pre-registered kill-gates.

Their substance already lived in [`docs/spec/08-dspark-dflash.md`](./spec/08-dspark-dflash.md)
(~25.5k words — the kill-gates, the increment log, the licensing correction, the Metal verdict);
the entries here were a second, thinner copy that could drift from it. The spec track owns them,
[`docs/spec/README.md`](./spec/README.md) indexes them, and
[`docs/spec/experiments.md`](./spec/experiments.md) is the dated run log.

**Open state, carried over so it is not lost in the move:**

- **P10 · increment 4.** Kill-gates 1 and 2 cleared 2026-08-15 (6.78 tok/verify against a ≥3.0
  bar). Remaining: gate 3 — end-to-end **≥1.3× vs plain resident decode on ≥1 GPU backend** — and
  gate 4's mixed-workload width router. The **Metal leg is measured not-ready**: ~1.13× ceiling
  even at `draft_ms=0`, and `PrefillLast` is not bit-identical, so the lossless contract cannot be
  met there. `gpt-oss` is blocked on a missing harmony chat template, not on the seam.
- **P15 · DFlash 2.** Filed 2026-08-20, **gates before code**, not started. Sequenced to land
  **before** P10's gate-4 width router — doing gate 4 first would mean redoing it.

## Filing a new item

Read `docs/completed/queue-performance.md` first — it is 2,245 lines of what was already tried,
including the negative results, and several of its entries exist because something was rebuilt that
had already been measured and rejected. Then follow the same discipline the archive does: state the
mechanism, pre-register the decision rule and its ambiguous band, include the do-nothing arm, and
record a negative with the same care as a win.

**One recurring defect that archive documents four times over, worth knowing before you add
anything:** an entry gets its resolution *appended* while the stale conclusion is left standing in
verdict position, so the item reads as open long after it closed. A2 (a pre-registration answered
four days earlier), D3b (shipped eleven days earlier), A10 (resolved, header still said OPEN) and
P16 (a stale-list four items out of date) all failed this way. **When you close something, correct
the sentence a scanner stops at — not only the body.**

**P26 · An image turn on a Gated-DeltaNet hybrid (Qwen3.5) prefills one token at a time, so a resend costs the same as a new image** — `linux`, **filed 2026-10-06**

Found by the cold-user run (`docs/measurements/cold-user-2026-10-05-nobara-pc.md` F: 37.5 to 42 s to first token for a 662-token image prompt on Qwen3.5-0.8B, the same for a byte-identical
resend, the same on `-backend cuda`). Measured to its cause on 2026-10-06 (`docs/measurements/image-resend-2026-10-06.md`, exploratory): the vision tower is ~3 s of ~38 s; the rest is
`prefillLogitsQwen35VL`'s per-token loop (~20 tokens/s, decode speed), because the recurrent state needs every token in order and `canBatchN` is false for the family. Image-prefix reuse
(P9a) and the resident image prefill are both resident-only and exclude recurrent families, so neither engages. A per-image tower-output cache shipped (serve, `featureCache`) and removes the
~3 s. **P26a DONE 2026-10-06:** the image prefill now runs the existing batched hybrid forward (`runLayersQwen35N`, taught to rotate by m-RoPE positions) instead of one token at a time: 38 s to 13.8 s for a new image and 34.5 s to 10.7 s for a resend on the 0.8B CPU, exploratory (`docs/measurements/image-resend-2026-10-06.md`). **P26b DONE 2026-10-06:** the CUDA-resident image prefill and decode for the dense hybrids (an `interleaved` mode in `rope_kv_mrope_batched`, a `ResidentHybridMRoPEPrefill` marker, and the branch in `GenerateQwenVL`): 37.5 s to 3.6 s for a new image and to 0.30 s for a resend on the 0.8B (`docs/measurements/p26b-cuda-hybrid-image-prefill-2026-10-06/`). **P26c DONE 2026-10-06:** the ~10 s on the CPU was `deltaNetCore` run per token on one thread (6.8 s of 11); `deltaNetCoreN` fans it out bit-identically: 11 s to 4.6 s, serve TTFT 13.8 s to 7.7 s new / 10.7 s to 4.4 s resend (`docs/measurements/p26c-cpu-deltanet-fanout-2026-10-06/`). What remains on the CPU is the compute-bound W4A8 matmuls (aikit's kernel) and the 3.3 s CPU vision tower. Still open under P26: **(d)** an MoE hybrid on the GPU image path (no gate exists), and the 9B, whose CPU gate `TestQwen35VLReal_G2_9B` is queued at night.
The original sketch follows. Next, in order of size: (1) a batched hybrid prefill (chunked DeltaNet scan, the same state at the end, so bit-identity against the per-token loop is the gate), which also speeds every
long text prompt on the family; (2) a snapshot of the recurrent state and KV at the image block's end that a resend restores, for the repeated-image case only. Both are decoder edits and
re-stale the parity manifest. Measure the per-token prefill on the 9B before sizing either: the 0.8B's 20 tokens/s is not the 9B's.

