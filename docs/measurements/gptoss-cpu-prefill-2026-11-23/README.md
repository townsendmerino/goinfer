# gpt-oss batched CPU prefill (perf review finding 11, CPU half): registration

Registered 2026-11-23, before any timed run or fidelity reading.

## The change

gpt-oss's prompts ran one token at a time on the CPU: `Program.BatchEligible` refused its FFN form (`FfnGptOss`). The batched walker
now takes it. Its FFN runs expert-major in chunks (`gptOssMoEBatch`, new: the gpt-oss form of
`internal/runtime/cpuengine/mlp.go:moeMLPBatchWith`), its attention is the batched kernel with the sink term (KS-O4), and its window layers go through the KV ring.
With exact attention the batched walker is bit-identical to the sequential one (tier 0, `TestGptOssPrefill_batchedEqualsSequential`, new).
At the default it takes CPU fast attention (f32) from 512 prompt rows, as every batched family does, so outputs change by float order.
Batched as gpt-oss's CPU default is Francis's call; this records what it is decided on.

## Arms and instrument

`TestGptOssCPUPrefillGate` (new, in `decoder/`; it lands with the code), one CPU load of `~/models/gpt-oss-20b-MXFP4.gguf` at serve's default
quant (int4), nobara (Ryzen 7 3700X, 16 threads). S = today's sequential prefill. E = the batched walker with exact attention
(`ExactPrefill`). B = the batched walker at the default (the candidate). Prompts: prefill-gate prose set A, truncated to K.

## Gates

| gate | question | metric | bar |
|---|---|---|---|
| G1, tier 0 | is the batched walker exact? | the tiny fixture's logits, batched against sequential, exact attention, past twice the window, 8 greedy steps, expert-major on and off | bit-identical (any difference: a bug, stop) |
| G2, tier 1 | is E the sequential path on the real model? | E's first logits against S's, on every S cell of G4 | bit-identical on every prompt (any difference: stop, a bug) |
| G3, tier 1 | does the default's f32 attention keep the exact outputs? | B against E, 8 prompts at K = 1,024 and 3,900, each scored at 33 positions (the first logits, then 32 teacher-forced on E's greedy tokens): 528 positions pooled. Hard flip: an argmax disagreement whose gap is over 3% of E's logit range (`NearTieArgmaxForTest`) | **PASS:** 0 hard flips and top-1 agreement ≥ 99.0%. **AMBIGUOUS (parked, DECISION NEEDED):** at most 2 hard flips and agreement ≥ 97.0%. **FAIL:** otherwise. Mean and max KL(E ‖ B) reported, not graded |
| G4, tier 1 | how much faster is the first token? | TTFT in process, S against B, A B B A, 2 blocks per K (4 pairs), K = 1,024 and 3,900, block b on prompt b | **WORTH IT:** every pair ≥ 2.0× at both K. **SMALL:** every pair ≥ 1.2×. **FAIL:** any pair under 1.2×. The Go heap's peak per arm is reported beside it (host bytes) |

**Void:** a G4 cell that starts with the 1-minute load above 8.0 or with less than 31,107 MiB available (the fit plan's need below
plus 2 GiB; `GPTOSS_PF_MIN_AVAIL_MIB=31107`) is VOID and listed; a void cell is re-run in a follow-up invocation of the same block. **Stopping:** fixed sizes; no early stop. **Order:** G1, then G3 (it decides first, since the default changes
output), then G4 with G2 inside it. G3 and G4 hold the timing lock (G3 shared, G4 exclusive), by day, done before 21:30.

**What each outcome means:** G1 or G2 red: a bug, fixed before anything else. G3 PASS and G4 WORTH IT: the default change goes to
Francis with these readings, recommended. G3 AMBIGUOUS: DECISION NEEDED with the flips listed. G3 FAIL: batched stays exact-only (an
`ExactPrefill`-style default for gpt-oss) or parked, Francis's call. G4 SMALL or FAIL: reported; the change is not worth its risk.

**Cost (estimated before running):** S runs about 0.1 s a token (9.8 tok/s decode at depth 128), so S is about 2 min at 1,024 and 7 min at
3,900. G3: 16 prompt-cells of E and B, each a batched prefill plus 64 decode steps, about 30 min. G4: 8 S cells (about 36 min), 8 B
cells, 8 E checks, about 50 min. **Fit check** (`GPTOSS_PF_MODE=fit`, 2026-11-23 12:45 PST): the CPU plan at int4 fits, needing
30,470,999,151 bytes (29,059 MiB) at the full 131,072-position context against 63.4 GB available; after the load VmHWM is 21.9 GB and
VmSwap 0.

## Amendment, 2026-11-23 12:41 PST, before any G4 data

G3's first two prompts (K = 1,024) read 59 of 66 positions agreeing and one hard flip, so G3 is likely to FAIL and its registered
outcome makes the exact-only option (the batched walker with exact attention, arm E) the live one. G4 as registered timed only B
against S, which leaves that option's speed unmeasured. **G4 now times all three arms**, S B E E B S per block (S–B pairs at cells
1–2 and 5–6, S–E pairs at 1–3 and 4–6), 2 blocks per K, the same K, prompts, void rule and stopping rule. **Bars, each read on its
own pairs:** S–B and S–E each WORTH IT when every pair is at least 2.0× at both K, SMALL at 1.2×, FAIL under it. G2 is unchanged (E
against S on every S cell). G3 runs to its fixed size, ungraded until it ends. **Cost:** 24 cells, about 60 min.

## G1 result: PASS (2026-11-23, nobara CPU, tier 0)

`TestGptOssPrefill_batchedEqualsSequential` (expert-major on and off), `TestGptOssMoEBatch_expertMajorEqualsPerRow` and
`TestGptOssPrefill_fastAttentionRuns` green, with the KS-O4 ring test, `TestMoEExpertMajor_bitIdentical` and the gpt-oss GGUF parity test
beside them. Red first: with `BatchEligible` as on main, the prefill test fails at `canBatchN`. A plant (the expert-major fold in
reverse rank order) turns the synthetic-layer test red; the tiny fixture's prefill test stays green under it, so the fold order is
pinned by the synthetic layer (top 4), not by the fixture.

## G3 result: FAIL (2026-11-23 12:36–12:42 PST, nobara CPU), stopped at 3 of 16 prompt-cells once FAIL was certain

Binary `decoder-2a349dff6`, gpt-oss-20b at int4, K = 1,024, records in `g3/`. Load at the start: 6.9 (16 cores).

| prompt (set a) | agree of 33 | hard flips | KL at position 0 |
|---|---|---|---|
| `audit-2026-09-02.md` | 32 | 0 | 0.025 |
| `QUEUE.md` | 27 | 1 | 0.049 |
| `queue-engineering.md` | 28 | 3 | 0.065 |
| **pooled, 99 positions** | **87 (87.9%)** | **4** | mean KL 0.169, max 1.7 over all positions |

**FAIL by the registered bar:** 4 hard flips is past the AMBIGUOUS band's cap of 2, so no reading of the 13 cells left could change
the verdict. **Deviation from the registration, stated:** it said fixed sizes and no early stop; the run was stopped here under
CLAUDE.md's "stop as soon as the answer is clear", because the verdict was fixed and the run held nobara's timing lock against every
timed run on a contended day. The K = 3,900 cells were not read.

**What it means:** at the default, the batched prefill's f32 attention moves gpt-oss-20b's first-token distribution by 0.03–0.07 nats
and flips argmaxes outside the near-tie band. The mechanism is the one `attendBatchedHeadsP`'s comment names (an f32 QKᵀ
reassociation flips a top-k expert at a near-tie and cascades through 24 MoE layers); the tiny fixture shows no difference at all at
512 and 1,024 rows, so a tier-0 test cannot see it. By the registered outcomes the live option is batched with exact attention (arm E,
bit-identical to the sequential path), which G4 times since its amendment.

## Amendment 2, 2026-11-23 13:24 PST, before any valid B or E reading

G4's first attempt (13:14–13:19 PST, records `g4-attempt1/`) read one cell, S at K = 1,024 on prompt 1: **246.2 s** (0.24 s a token,
not the 0.1 s the cost estimate assumed), with E's first logits equal to S's (a G2 reading, PASS). Every later cell started at load
9.26 and is VOID: the harness had no settle wait, so the 1-minute average still carried the previous cell's own 16 threads on an
otherwise idle box. Two changes:

- **Settle:** each cell waits (up to 10 min) for the 1-minute load to fall under 4.0 before its start state is read; the void rule
  (above 8.0, or under 31,107 MiB available) is unchanged.
- **Size:** at 0.24 s a token an S cell at K = 3,900 runs about 20 minutes, so the registered 2 blocks there (4 S cells) put G4 near 3
  hours, past the day's bound, and nobara's night is taken (C12 from 22:00). **K = 3,900 runs 1 block** (S B E E B S: one S–B pair
  and one S–E pair in each order); K = 1,024 keeps 2 blocks. The bars are unchanged and are read on the pairs that run. **Cost:** about
  95 minutes.

## G4 result: WORTH IT for both batched arms; G2 PASS (2026-11-23, nobara CPU, the exclusive lock)

Binary `decoder-f553040b3` (amendment 2), gpt-oss-20b at int4. Records: `g4/` (13:24–14:56 PST) and `g4/rerun-b1/` (15:39–15:54).
**Void and re-run:** the first cell of K = 1,024 block 1 in `g4/` (S, 13:24:51–13:28:54) overlapped three 1–2 s `go vet` runs that
another session made without the timing lock (13:25:57, 13:26:12, 13:26:56; that session's overlap table), so the whole block is
VOID and was re-run in place as `g4/rerun-b1/`. nobara then crashed at about 15:16 and rebooted, before the re-run had started (it
was waiting for the lock), so the crash voids nothing here. No other cell overlaps a recorded unlocked run. Every cell started at
load 2.06–3.98 with at least 48.5 GB available: none void by the registered rule.

| K | block | S s | B s | E s | S ÷ B (two pairs) | S ÷ E (two pairs) |
|---|---|---|---|---|---|---|
| 1,024 | 1 (re-run) | 240.6, 237.9 | 37.7, 37.7 | 42.1, 42.2 | 6.39, 6.31 | 5.71, 5.64 |
| 1,024 | 2 | 243.6, 241.2 | 38.3, 38.3 | 42.9, 42.8 | 6.36, 6.30 | 5.68, 5.64 |
| 3,900 | 1 | 977.3, 975.9 | 206.4, 207.0 | 220.7, 221.0 | 4.73, 4.71 | 4.43, 4.42 |

- **S–B and S–E: WORTH IT** (every pair at least 2.0× at both K). E, the exact batched arm, is 7–12% behind B.
- **G2: PASS.** E's first logits equal S's bit for bit on every S cell that ran (8 cells, including the voided block's, whose
  identity check is not a timing).
- **Memory (the Go heap's peak in use, sampled every 20 ms):** S 11,853 MiB at 1,024 and 11,988 at 3,900; B +213 and +595 MiB; E
  +214 and +380 MiB. The batched pass's scratch is the difference; the fit plan (29,059 MiB need) is unchanged.

## What it decides (Francis's call)

By the registered outcomes, G3 FAIL leaves the default's f32 attention out, and the exact batched prefill (E) is the live option:
bit-identical to today's sequential prefill (G1, G2), first token 5.6–5.7× sooner at 1,024 prompt tokens and 4.4× at 3,900, for
+214–380 MiB of peak heap.

## Decision: batched with exact attention (Francis, 2026-11-23 about 16:40 PST, relayed by the coordinator)

Option 1: gpt-oss's CPU prefill takes the batched walker with exact attention by default. `Model.cpuFastAttention` reports false for an
architecture with attention sinks, so gpt-oss's batched prefill always runs the exact kernel (`TestGptOssPrefill_defaultIsExact`).
Output bit-identical to the sequential path (G1, G2); first token 5.6–5.7× sooner at 1,024 prompt tokens and 4.4× at 3,900; Go heap
peak +214 MiB at 1,024 and +380 MiB at 3,900; the fit plan unchanged. The f32-attention variant stays off, recorded as failing G3.
From this commit `TestGptOssCPUPrefillGate`'s B arm is the exact kernel too (the knob no longer reaches gpt-oss); the B readings above
are `decoder-2a349dff6`'s and `decoder-f553040b3`'s.
