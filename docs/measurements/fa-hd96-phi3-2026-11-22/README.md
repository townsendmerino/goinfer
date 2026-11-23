# The CUDA flash-decode lane at head dim 96 (Phi-3 mini): registration (2026-11-22)

**Status: REGISTERED 2026-11-22, APPROVED (Francis, 2026-11-22, about 15:40 PST: "Yes build it", relayed by the coordinator).** Gates in order, each passing before the next, fixed N, no early stop; if it ships it lands behind an opt-in, and the default is Francis's separate call.

**Question.** Does a head-dim-96 variant of CUDA's flash-decode lane decode Phi-3 mini (`q4k`, f16 KV) as well as the path it
decodes on today, and faster at depth? Today `faEligible` admits hd 64, 128 and 256 only, so Phi-3's 32 layers (hd 96, MHA, G = 1)
decode on the exact attention path at every depth; split-KV stays off for it (Francis, 2026-11-19, the split-KV record). Review
finding 1(b) and [`splitkv-phi3-2026-11-19/record.md`](../splitkv-phi3-2026-11-19/record.md) name this lane as the next lever: about
half the 16.1 ms token at depth 3,900 is depth-dependent, and Phi-3 is BEHIND both peers on CUDA (0.79× Ollama at 3,900).

**The change.** `fa_partial_96` and `fa_partial_rows_96_g1` (and its ragged twin) in `decode_fa.cu`: the score dot is unchanged (NQ = 6
float4 steps), and the P·V step takes its own lane mapping, since 96 dims over 32 lanes is 3 a lane (the template's VEC = 2 gives a
non-integer NCH). `faEligible` admits hd 96. The partial buffer is already sized to the model's largest head dim and allocated for
Phi-3, so the device bytes do not change; the PTX modules grow by the new kernels.

**Gates, in order. Each must pass before the next runs.**
1. **Tier 0, correctness:** `fa_partial_96` (+ `fa_combine`) and the rows kernel against the float64 reference
   (`TestAttnSoftcap_flashDecodeAgainstFloat64`'s pattern: cosine ≥ 0.9999, max abs ≤ 1e-2; windows 0 and 100; 4 and 16 splits;
   verify batches 1, 5 and 16), with a plant (a lane dimension dropped) going red. `TestAttnKernelDigests` (G1-K): every existing
   kernel's digest unchanged.
2. **Fidelity, registered now:** `TestFlashDecodeGateVsReference` with a new `P3M` cell, `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`,
   `q4k` on CUDA, K = 3,900, held-out set B, against a new `TestPrefillGateReference` `P3M` row (CPU f32 from the same GGUF, as G22B).
   The criteria, preconditions and ambiguous band of
   [`attn-decode-fa-fidelity-PREREGISTERED.md`](../attn-decode-fa-fidelity-PREREGISTERED.md), unchanged: (a) hard flips ≤ exact +
   2√exact; (b) agreement ≥ exact − 1.0 pt and lane ≥ exact on at least half the prompts; (c) mean KL ≤ 1.10 × exact's.
   AMBIGUOUS → PARKED: (b) within its last 0.2 pt, or (c) in 1.05–1.10×. The GGUF declares no `sliding_window`, so goinfer attends
   the full context (`phi3Architecture`), and the lane's 2,048-key floor is crossed from depth 2,048 on.
3. **Speed (TE5(b), then TE5(a)):** an in-process whole-token A/B, `BenchmarkResidentDecode` with serve's f16 KV, lane on against off
   (`GOINFER_CUDA_FLASH_DECODE=0`) in one binary, A B B A × 5 blocks (10 pairs) at depths 3,900 and 128. Then a served two-arm A/B (`goinfer` against
   `goinfer_old`, one binary, the flag on one arm), depths 128 and 3,900, `BENCH_RUNS=2`, `BENCH_ABBA=1`.
   **SHIP:** the served paired median ≥ 1.02 at 3,900 and ≥ 0.98 at 128. **REGRESSION:** < 0.98 at 3,900. **Between:** parked.
   An in-process result that does not resolve its direction (pairs straddle 1) still goes to the served cell.

**Stopping rule:** fixed N at every gate; no early stop. A failed gate stops the sequence there.

**What each outcome means.**
- All three pass and SHIP: the lane takes hd 96 behind an opt-in until Francis decides the default; the PR states device and host
  bytes and the longest context that fits.
- Gate 1 fails: fixed before gate 2 is spent (gate 1 is a defect detector, not a decision).
- Gate 2 FAILs: hd 96 stays off the lane; the record names the criterion, and the kernel is not merged.
- Gate 3 parks or regresses: the kernel is not merged; the record says which.

**Cost:** gate 1 about 15 min of build and card tests. Gate 2's reference about 60–80 min of CPU (the 2B took 37 min at K = 3,900) and
the gate about 5 min under the exclusive lock. Gate 3 about 25 min in-process and 10 min served. By day; nothing past 21:30 PST.

## Gate 1, 2026-11-22 17:18–17:23 PST (nobara, RTX 2070 SUPER, driver 595.112.07): PASS

- **The kernels' code:** `fa_partial_96`, `fa_partial_rows_96_g1` to `g8` and `fa_partial_ragged_96`, at `0d4843a18`. The PTX was rebuilt at
  the ambient NVRTC 12.9.86 (`build_ptx.sh decode_fa`).
- **`TestAttnKernelDigests` (G1-K):** 2,152 digests equal base's on this card and driver. Every existing kernel is unchanged
  ([`gate1/gate1.log`](gate1/gate1.log)).
- **`TestFlashDecodeHD96_againstFloat64`:** all 288 cases are inside the bar (cosine ≥ 0.9999, max abs ≤ 1e-2), at f32 and f16 KV:
  - the M = 1 lane at 4 and 16 splits;
  - 704 verify rows, each also bit for bit against the M = 1 lane at its depth;
  - 32 ragged rows, bit for bit against the M = 1 lane.
- **The plant:** the P·V load dropped one of each lane's three dims (`v[c] = 0` at c = 2). 896 comparisons fell outside the bar
  (cosine about 0.77; `gate1/plant-red.log.gz`). The bit-identity checks stay green under it, as they should, since both sides carry
  it. Restored, the PTX rebuilt to the committed bytes (sha256 `74627d34…`).

## Addendum, 2026-11-22 18:49 PST (commit `8c4b42ab7`; first typed as 18:52, corrected from the commit time), before gate 2's first reading: the instrument seeds Phi-3 by sequential prefill

- **What happened.** Gate 2's first launch (18:49 PST) stopped at its first prompt, before scoring anything:
  `PrefillLast (lane split=0): cuda prefill: per-32 activation quantization has no batched kernels yet`. The gate seeds each prompt
  through the resident's batched prefill, and CUDA has none for q4k's per-32 activations. Serve takes the sequential path for Phi-3 at
  q4k too ("prefill path: sequential" in its banner).
- **The change, the instrument only.** For the `P3M` cell, `laneArm` prefills the K prompt positions one `Forward` at a time, on the
  exact path in both arms: `faSplit` is 0 during the prefill and the arm's split for the scored positions only. So precondition 3 (the
  seed row identical across arms) still tests what it did. The other cells keep `PrefillLast`.
- **Unchanged:** the cell, K = 3,900, set B, the reference, the preconditions, criteria (a), (b) and (c) and their ambiguous band.
- **Cost:** about 21 sequential prefills of 3,900 positions (10 prompts × 2 arms, plus the A/A), about 25 min under the exclusive lock.

## Gate 2, 2026-11-22 18:50–19:06 PST (nobara, RTX 2070 SUPER, driver 595.112.07): **PASS** by the registered rule

- **Run.** The CPU f32 reference (`TestPrefillGateReference` P3M, K = 3,900, set B) held the lock shared, 17:19–18:23 PST. The gate,
  `TestFlashDecodeGateVsReference` `P3M:3900` at `8c4b42ab7` with the addendum's sequential seed, held it exclusive, 18:50:06–19:06:14 PST.
  - It waited behind a night-runner job, `cuda-device-suite`, which held the lock from 18:24 to 18:49.
  - It started at a 1-minute load of 0.43 with 61,302 MiB available, with kwin_wayland holding 19 MiB of the card.
  - Logs: [`gate2/gate2.log`](gate2/gate2.log), `gate2/p3m-refs.log.gz`, and the first launch, `gate2-attempt1-batched-prefill.log`
    (the addendum's cause).
- **Preconditions held:** A/A bit-identical over 64 rows; the lane launched; the seed rows were identical. All 630 continuation rows
  differ from exact's.

| | exact | lane (hd-96 flash-decode) |
|---|---|---|
| mean teacher-forced agreement | 98.59% | 97.81% |
| hard flips | 0 / 640 | 0 / 640 |
| mean KL(reference ‖ arm) | 0.003148 | 0.002999 |

- **(a) hard flips: PASS** on the strict reading too, 0 against 0.
- **(b) agreement: PASS.** The lane is −0.78 pt from exact, inside the −1.0 bar and 0.02 pt short of the parked zone (−0.80 to
  −1.0), so close to the line. Paired wins are 7 of 10.
- **(c) KL: PASS.** The ratio is 0.9529, under 1.10 and outside the 1.05–1.10 band.
- **Verdict: PASSES.** Gate 3 runs next.

## Gate 3, in-process half, 2026-11-22 19:06–19:55 PST: the direction resolves, faster at depth 3,900 and flat at 128

- **Run.** [`run_gate3_inproc.py`](run_gate3_inproc.py) under the exclusive lock. One test binary from `29e674092`, with the f16-KV scratch
  init. The arms:
  - A: the lane off, `GOINFER_CUDA_FLASH_DECODE=0`;
  - B: on, `=16` and `GOINFER_BENCH_FA_HD96=1`.

  The design was A B B A × 5 blocks at depth 3,900, then at 128: 40 cells, none void. Each cell started under load 1.0 after its
  settle; the highest start reading was 1.0, under the void rule's 8.0. Available memory was at least 61,373 MiB, and no foreign process
  was present. Records: [`gate3/inproc-cells.tsv`](gate3/inproc-cells.tsv), `inproc-run.log`, `inproc-cell-logs.tar.gz`.

| depth | pairs | B faster | median B ÷ A | range | A (lane off), tok/s | B (lane on), tok/s |
|--:|--:|--:|--:|--:|---|---|
| 3,900 | 10 | 10 / 10 | **1.1598** | 1.1555 – 1.1622 | 62.03 – 62.27 | 71.92 – 72.15 |
| 128 | 10 | 3 / 10 | **0.9996** | 0.9992 – 1.0008 | 126.3 – 126.6 | 126.2 – 126.7 |

- At 3,900 the token goes from 16.1 to 13.9 ms: the lane recovers 2.2 ms of the depth-dependent 8.2 ms. Split-KV recovered 0.79 ms there
  (the split-KV record).
- At 128 the lane does not engage (its floor is 2,048 attended keys), so 1.000 is the noise between two identical paths.
- By the registration, the in-process result gives the direction; the served cell below decides SHIP.

## Addendum, 2026-11-22 19:59 PST, before the served cell's first reading: the arm probe counts launches, not text

- **What happened.** The served cell's untimed arm probe (19:55 PST; `gate3/served-attempt1-text-probe.log`) stopped the cell before any
  timed run. Both arms' 256-token greedy replies to the depth-2,048 prompt were the same degenerate "planeplane…" loop. A distribution
  that peaked does not move under the lane's small numeric differences, so equal text could not tell "the lane did not engage" from
  "it engaged and flipped no token".
- **The change.** [`probe_lane_cost.py`](probe_lane_cost.py) replaces it in `run_gate3_served.sh`. Each arm takes the same depth-2,048
  request with goinfer's cost report, and the decode phase's kernel launches decide:
  - the lane arm must launch `faCombine` and no `bAttn`;
  - the exact arm must launch `bAttn` and no `faCombine`.

  Its first run (`gate3/probe-cost-first-filter.log`) showed exactly that: the lane arm had `faCombine` 2,048 (32 layers × 64 tokens)
  and its partial kernels, which `pipeName` reports as `?`, and the exact arm had `bAttn` 2,048. But it exited 1, because its filter
  also matched `fArg`, the fused argmax. The filter now names the two kernels.
- **And a settle after the probe.** The first launch with the new probe (20:00 PST) passed it, but bench_peer's idle preflight then
  refused at load 1.39, raised by the probe's two loads, and measured nothing. The script now waits for load under 0.9 after the probe
  too, as it does before it.
- **Unchanged:** the cell, its depths, runs, order, rule and grader.

### Gate 3, served cell: not yet read; owed tomorrow by day

- **Attempt 4, 20:02–20:35 PST: VOID, measured nothing.** The probe passed: the lane arm launched `faCombine`, the exact arm `bAttn`.
  Then every cell's server refused to start (`gate3/served-attempt4/`). The script, copied from OP1-o2's, set `BENCH_REQUIRE_BACKEND=1`,
  and serve's `--require-backend` refuses a load that declines any path. Phi-3 at q4k declines the batched prefill by design ("prefill
  path: sequential"). The script now drops it. The harness's decode-path gate still voids a cell that does not decode `cuda-resident
  (q4k)`.
- **Attempts 2 and 3 (20:00 and 20:01 PST)** were refused before timing anything: bench_peer's idle preflight at load 1.39, then no `go`
  on PATH for its memory reading (`served-attempt2-*.log`, `served-attempt3-*.log`).
- **Why not tonight.** At 20:08 PST Francis moved the C12 runs to start as soon as nobara's lock is free, so this session does not take
  the lock again tonight. The served cell runs tomorrow by day: `run_gate3_served.sh` as it stands, with its registered rule (SHIP at a
  served median ≥ 1.02 at 3,900 and ≥ 0.98 at 128).
- **Where gate 3 stands:** in-process, 1.160× at 3,900 (10/10 pairs) and flat at 128. The registered SHIP decision waits for the served
  cell.

## Gate 3, served cell, 2026-11-23 06:40–06:49 PST (nobara, RTX 2070 SUPER, driver 595.112.07): **SHIP** by the registered rule

- **Run.** `run_gate3_served.sh` as fixed on 2026-11-22: the launch-count probe, a settle after it, and no `--require-backend`. It ran
  from the Mac's `night.py` under nobara's exclusive lock, on serve `29e674092` (sha256 `583b937a…`): `goinfer` with `-flash-decode-hd96`,
  `goinfer_old` with it off. Phi-3 mini at q4k, `BENCH_RUNS=2`, `BENCH_ABBA=1`.
  - All 8 cells are valid: each decoded `cuda-resident (q4k)`, the token gate passed, and swap grew 0.0 MB.
  - Each cell started at a 1-minute load of 0.83–0.98, with at least 61,681 MiB available.
  - Records: [`gate3/served/`](gate3/served/) (`verdict.txt`, `decode-cuda-all.json`, the log and the serve logs).

| depth | lane ÷ exact, per block | median | lane, tok/s | exact, tok/s |
|--:|---|--:|--:|--:|
| 128 | 0.9991, 1.0011 | **1.0001** | 114.52 | 114.50 |
| 3,900 | 1.1489, 1.1492 | **1.1491** | 67.93 | 59.12 |

- **Verdict: SHIP.** The median is at least 1.02 at 3,900 (1.149) and at least 0.98 at 128 (1.000). By the registration, the lane lands
  behind `--flash-decode-hd96`. Whether it becomes Phi-3's default is Francis's separate call.
- **The baseline these figures carry.** This binary predates the CUDA q4k sidecar fix (#1236). Both arms decoded Phi-3's Q5_K/Q6_K
  tensors from int8 copies, the same on both sides, so the ratio compares like with like. Since #1236, served decode reads the native
  blocks, which saves about 0.84 ms a token on every token. That cost is not attention, so the lane's saving at depth (about 2.2 ms a
  token in-process at 3,900) should stand, and the ratio at depth should be at least this one. That is a statement about mechanism, not
  a measurement.

## Default, 2026-11-23: on at a context of at most 4,096 (Francis)

- **The decision.** Francis, 2026-11-23, about 08:25 PST, relayed by the coordinator, option (a) of the default-on case: the lane is on by
  default only for a head-dim-96 load whose context is at most 4,096, which is what the gates read. `--flash-decode-hd96` reaches a longer
  context as an opt-in, and `=false` turns it off.
- **Its shape.** When `--flash-decode-hd96` is not given, goinfer-chat and serve set `decoder.Options.FlashDecodeHD96` with
  `FlashDecodeHD96Default`. CUDA's `faEligible` then takes an hd-96 layer only while the resident's context cap is at most
  `flashDecodeHD96DefaultMaxCtx` (4,096). The cap, not the model's maximum, bounds the span the lane attends, so a 128k model loaded with
  `--ctx 4096` takes it too. A given flag is that choice at any cap. A library caller's zero `Options` leave the lane off.
  `TestResolve_flashDecodeHD96Default` and `TestFlashDecodeHD96_defaultByContext` hold the rule.
- **What it rests on:** gate 2 PASS at K = 3,900, served SHIP (1.149 at 3,900, 1.000 at 128), and no memory or context change (4 loads,
  flag on and off, device 5,922 MiB and host RSS about 2.90 GB on each).
