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
- **Width:** w = 3, SP0's best for the 35B, fixed before any data. G1 also runs w = 16 (the block, the widest fold). **G4 amended
  to w = 4 before its run** (Readings, "G4's width amended").
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

### G1's first start (2026-10-26, 07:16:30–07:16:44 PDT, at `c69545d6`): void, no data

The resident build declined: "dense weights need 1.16 GB plus 0.47 GB reserved for a companion attach (--drafter) but only 0.71 GB
is free on the device". My own host-memory measurement for #263's follow-ups held 6.6 GB of the card at the time, through a
`serve` it had started: both runs held the timing lock shared, which lets two loaders share the box but not one card. Nothing was
read. **Amended before the re-run:** `run-sp2-35b.sh` takes the lock exclusive for G1 too (it needs the card to itself), and G1
re-runs as registered once the lock is free. Log: `~/goinfer-logs/sp2-35b-2026-10/identity-2026-10-26-0716.log` on nobara.

### G1, the re-run (2026-10-26, 07:19–07:37 PDT, at `7129a5ab`, the timing lock exclusive): PASS

- **Identity:** 0 failures over 20 prompts × 2 widths (w = 3 and 16) plus the guard runs: tokens and the Gated-DeltaNet state equal
  plain greedy decode on every prompt. The served plain decode agrees on every prompt too. So the MoE verify, which scores a
  pass's rows in one batched pass, did not diverge from per-token decode here.
- **The plants:** P1 (the fold skips the conv window) caught on 12 of 12 prompts, P2 (verify writes the state) on 11 of 12.
- **G5:** passes (kvmgr reuses the 119 rows it should; state and checkpoint equal).
- **G3, from the load:** context 4,608 planned; the fold's arena 79,151,104 B (16 rows × 30 DeltaNet layers, "75.5 MiB on the
  device"); the drafter's device reserve 466,116,608 B; 0.94 GB of VRAM free after attach, the fairness check's figure.
- **Host:** the test process peaked at 38.7 GB VmHWM (the mapped `.giw` with experts streamed), 0 swap.
- Raw: `identity-2026-10-26-0717.jsonl` and `.log` under `~/goinfer-logs/sp2-35b-2026-10/` on nobara.

### G4's width amended to w = 4, before G4's run (2026-10-26, 07:39 PDT)

**Why:** `BlockSpec`'s acceptance guard (`decoder/blockspec.go:breakEvenTokensPerRound`, 2.5 committed tokens a round, judged
after 6 rounds) does not scale with the width. At w = 3 a round commits at most 3, and SP0's mean at w = 3 was 2.76, so the guard
switches many turns to plain decode: in G1, 5 of 20 turns at w = 3 tripped it within 6–7 rounds (1 of 20 at w = 16). A G4 at w = 3
would largely time the guard, not the drafter. I missed this when I registered w = 3.

**The amendment:** G4 runs at **w = 4** (`run-sp2-35b.sh` sets `SP2_WIDTH=4` for `MODE=speed`). SP0's projection is within a few
percent of w = 3 (1.17× against 1.22× at SP0's 17.57 ms step; about 1.57× against 1.64× scaled to the real 23.6 ms step), its mean
of 3.52 committed tokens a cycle sits well above the guard, and it is the width the 9B's G4 ran. Nothing else changes: the
contexts, pairs, void rule and bars stand. G4 records any token divergence from plain decode on its 12 prompts at w = 4, so w = 4's
identity is checked there. No G4 data existed when this was written.

**A finding for the owner, beside SP2:** the guard's fixed 2.5 makes widths of 3 or less nearly unusable. `serve` verifies at 8 by
default and its `-drafter-width` takes any width from 2, so `-drafter-width 3` would trip on many turns. Whether the bar should
scale with the width is a separate decision; nothing here changes it.

### G4 (2026-10-26, 07:50–09:03 PDT, at `79c91b64`, w = 4, the timing lock exclusive): short DELIVERS; long stopped, not graded

Graded by the merge rule, as registered. The harness printed "NOT ADVANCED" for the short cell: it still carried SP0's bands from
the 9B's G4, which the 9B's record had already corrected for grading. The harness now prints the merge rule (`705fff57`), and the
readings stand as recorded (the owner, 2026-10-26 09:01).

| context | plain decode | `BlockSpec`, w = 4 | median of the pairs | pairs faster | spread | the merge rule |
|---|---|---|---|---|---|---|
| short (SP0's 12 prompts, about 100–250 tokens) | 40.1 tok/s | 43.6 tok/s | **1.089×** | 10 of 10 | 1.086–1.092 | **delivers** (≥ 1.05×, ≥ 9 of 10) |
| long (behind the 3,800-token preamble) | 38.7 tok/s | 39.4 tok/s | 1.018× (one pair) | 1 of 1 | | **not graded**: stopped after one pair |

- **Cells:** every pair started at load 2.8–5.8, with at least 40.3 GB available; none void. No token diverged from plain decode
  at w = 4 on any prompt, so w = 4's identity holds on these 12 prompts.
- **Against SP0's 1.30× (deciding nothing):** under it. The short cell's 1.09× is far below the ~1.6× that SP0's cycle projects
  against the real step: its verify reading (a repeated token, so the same experts every row) was optimistic too, as the
  registration feared. The 35B's verify rows each bring their own experts on an 8 GB card.
- **Why the long cell stopped:** my cost estimate was wrong. One long pair took 35 minutes, not 3.5: its 24 prefills of about
  3,900 tokens took about 80 s each with experts streamed (about 49 tok/s), and decode is a small part of the pair. Ten pairs would
  have been about 6 hours, past the 3-hour cap on a night job. I stopped it at 09:03 (exit 143, 0 swap). The one pair read
  1.018×; it is recorded and not graded.
- Raw: `speed-2026-10-26-0739.jsonl` and `.log` under `~/goinfer-logs/sp2-35b-2026-10/` on nobara.

### `--drafter`'s memory cost on the 35B (2026-10-26, 09:04–09:09 PDT, `serve` at `705fff57`)

Plain `serve` against `serve --drafter -drafter-width 4`, one binary, experts streamed (`GOINFER_MOE_CACHE_EXPERTS=1`), context
4,608, one warm and one timed greedy request (`measure-host.sh` beside the raw data):

| | VmHWM | VmRSS 60 s after | timed request | reply |
|---|---|---|---|---|
| plain `serve` | 37.00 GB | 37.00 GB | 6.60 s | |
| `serve --drafter` | 37.49 GB (+0.49) | 37.02 GB (+0.02) | 5.79 s | identical to plain |

- **Device:** the drafter's reserve, 0.47 GB, and the fold's arena, 79 MB. The context stays at 4,608, and the expert cache keeps
  its 64 slots a layer (the fairness check). The drafter needs an f32 KV cache, as on the 9B.
- **Host:** +0.49 GB at the peak and +0.02 GB steady, with the drafter streamed from its file and released after attach (#259,
  #263).
- **The longest context that fits:** unchanged at the 4,608 measured; the fit banner prices it the same with the drafter (0.2 GB
  of KV, 18.2 GB of weights).
