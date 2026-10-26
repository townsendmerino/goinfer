# SP2 on the 35B: Qwen3.6-35B-A3B with DFlash on nobara

Registered 2026-10-26 by nobara-sp2, before any run, on the owner's order of 2026-10-25 (Francis, relayed: the #232 fix, then the
width follow-up, then the 35B). The design and the 9B's readings are `docs/measurements/sp2-2026-10/README.md`; this record asks
whether the same path pays on the 35B, and is read against the owner's merge rule.

## Why now, and what SP0 said

- **SP0** (branch `sp0-2026-10`, `docs/measurements/sp0-2026-10/README.md`, not merged) projected the 35B + DFlash, experts
  streamed, at **1.22× at L = 512 (w = 3)** and **1.05× at L = 4096**, and parked it: 2.76 tokens a cycle, a 36.37 ms w = 3
  verify and 3.35 ms of drafting, against a **17.57 ms** step.
- **The step was optimistic.** SP0's step arm repeated one token at one position, so every step hit the same experts. SP2's
  fairness check (`docs/measurements/sp2-2026-10/README.md`, "The 35B expert-cache fairness check") read real plain decode at
  **23.6 ms a token** (42.3 tok/s, 64 expert slots a layer with or without the drafter). Against that step, SP0's own cycle
  projects about 1.6× at 512; whether its verify reading was optimistic the same way only a real run says. So the projection is
  not a verdict either way, and the gates below measure the whole generation.

## Scope

- **Target:** `qwen3.6-35b-a3b-q4_k_m.int4.e4h.cuda.v17.giw` (mapped), resident on CUDA, experts streamed
  (`Options.MoECacheExperts`), resident context 4,608, f32 KV (the CUDA drafter refuses f16), as SP0 and the fairness check
  loaded it.
- **Drafter:** `z-lab/Qwen3.6-35B-A3B-DFlash` at `f181eece646affea2c38b2765f1aaa01a9734ccd` (`~/models/qwen36-35b-dflash`; 6
  layers, hidden 2,048, block 16, 8 taps; 0.77 GB bf16), opened as `serve` opens it (`decoder.OpenDFlashDrafter`, #263).
- **Width:** w = 3, SP0's best for the 35B, fixed before any data. G1 also runs w = 16 (the block, the widest fold).
- **Harness:** SP2's (`cuda/recurrent_spec_real_test.go`), unchanged but for two knobs: `SP2_WIDTH` (the verify width of G1c
  and G4, default 4) and `SP2_MOE_STREAM=1`. Run script: `run-sp2-35b.sh` here.
- **Prompts:** SP2's. G1: SP0's 12 code and math prompts with thinking off, its 6 math prompts with thinking on, and two long
  prompts of about 3,000 tokens. G4: the 12, as they are and behind a 3,800-token preamble.
- **Out of scope:** the Mac (Metal hosts no block drafter); the served A/B (`bench_peer.py`), which follows only if G4 delivers.

## Gates

**Fit check first (seconds, tier 0 of the run):** the load G1 makes is the fairness check's load plus SP2's arena, at the same
context. G1's `load` record states the arena, the reserve, the planned context and free VRAM, and its log has the `fit:`, `KV
plan:` and expert-cache banners. G4 is not started unless G1's load planned 4,608 positions.

- **G1 — identity on the 35B (tier 1, by day, the timing lock shared).**
  - **Instrument:** `TestSP2_identity` at w ∈ {3, 16}: greedy `BlockSpec.Generate` against plain greedy decode on the same
    resident, 256 new tokens a prompt; tokens and the Gated-DeltaNet state must be equal. The planted defects P1 and P2 run at
    w = 4 and must go red, as on the 9B. G5 (KV8a and kvmgr) runs as on the 9B.
  - **Bar:** every prompt at both widths equal, both plants red, G5 passes → G4 runs.
  - **A divergence is not waved through.** The MoE verify scores rows in one batched pass where plain decode scores one row a
    step, and SP2 never ran a MoE target. A divergence is recorded with its first token, its row and the served plain decode's
    own agreement (`served_plain_diverge_at`), and G4 does not run: it goes to the owner as a decision, a defect until shown to
    be the batched MoE pass's numerics.
- **G3 — memory (tier 1, read from G1's load).**
  - **Stated:** the arena's bytes and rows, the drafter's device reserve, the planned context, the expert cache's slots per
    layer, free VRAM after attach.
  - **Bar:** the planned context is 4,608 and the expert cache keeps the fairness check's 64 slots a layer. Fewer slots is
    recorded with the figure, and G4 runs at what fits; the reading then names it.
  - **Host:** `serve --drafter` on the 35B, VmHWM and steady VmRSS against plain `serve` (#263's instrument), recorded beside G4;
    no bar.
- **G4 — speed with DFlash against plain decode (tier 2, by day, the timing lock exclusive).**
  - **Instrument:** `TestSP2_speed` with `SP2_WIDTH=3`, as registered for the 9B: the whole-generation in-process A/B (TE5(b))
    on one resident, plain greedy against `BlockSpec` at w = 3, 256 new tokens; 10 A B B A pairs per context, a pair being one
    pass of each arm over the 12 prompts and its reading the plain time over the speculative time, tokens equal by G1.
  - **Contexts, read separately:** short (the prompts as they are, about 100–250 tokens) and long (behind the 3,800-token
    preamble, so decode starts near 4,096).
  - **Void cells:** a pair started above 8.0 load, or under the fit need plus 2 GB of `MemAvailable`, is void and re-run in
    order (CLAUDE.md, "Run budget").
  - **Bar, per context: the owner's merge rule** (Francis, 2026-10-25 8:46 am; lossless and opt-in): the median pair ≥ 1.05×
    and at least 9 of 10 pairs above 1.0 → `--drafter` **delivers** on the 35B at that context. Anything else → **does not
    deliver** there. Also reported, deciding nothing: the reading against SP0's 1.30× bar.
  - **What a verdict changes:** no code is waiting on it. It says whether `--drafter` is worth using on the 35B (docs and the
    task doc record it), and whether a served A/B follows; that is the owner's call.
- **Stopping rule:** fixed. All prompts, widths, pairs and contexts; no early stop.

## Cost, stated before the run (cost basis: the fairness check's 23.6 ms a token; SP0's prefill and verify readings)

- **G1:** 20 prompts × (one plain and two speculative generations) × 256 tokens at about 6 s a plain generation, plus the
  plants' 12 prompts × 2 and G5, and one load of about 1 min: **about 15 min.**
- **G4:** short, about 2 min a pair (12 × (6 s plain + about 4 s speculative)); long, about 3.5 min a pair with the 3,900-token
  prefills; 10 pairs each plus warm-ups: **about 60 min.**
- **Host memory:** 3 arms of about 2 min: **about 6 min.**
- **Total about 80 min, by day,** through the timing lock, behind whatever holds it.

## Readings

(none yet)
