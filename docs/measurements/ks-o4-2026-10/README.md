# KS-O4 — gpt-oss onto the batched acc64 attention with a sink term, then its window layers through the KV ring (registered 2026-10-25, before any code)

KS-O4 of the kernel survey (`docs/tasks/task-kernel-survey-2026-10.md` § "KS-F2", its option and its "Step 2, memory"). **Francis's decision,
2026-10-25 (6:33–6:38 am, relayed by the coordinator): B, approved gated** — "move gpt-oss onto the batched acc64 attention with a sink term;
register a fidelity gate (KL and agreement against the current gpt-oss path, what each outcome means) before any code". This record is that
registration, committed before any code. A separate PR from KS-F2's.

## The change

- **Step 1, the attention (changes output):** gpt-oss's attention order becomes `attnBatchedAcc64`, with the per-head learned sink joining
  the max-shift and the denominator, as `decoder/forward_gptoss.go:attendSinksToken` does today. It joins in every softmax the batched
  path takes for gpt-oss.
  - **What moves:** each score's dot is rounded to float32 before the float64 scale (`MatmulQKAcc64`), where today the float64 dot is
    scaled. AV accumulates in float64 (`MatmulAVAcc64`), where today it accumulates in float32. So the candidate's numbers differ from
    today's in the last bits of each score and of the context.
  - **What it unblocks** (KS-O4, §2): the KV ring, f16/int8 KV, the batched walker and MC3c for gpt-oss on the CPU.
  - **Unchanged for every other family:** a model without sinks takes the batched path exactly as before. That is checked by bit-identity
    (G-ID below), not assumed.
- **Step 2, memory:** gpt-oss's 12 window-128 layers read and write the KV ring (`Architecture.kvRingsOK` admits them).
  - **Today:** they keep append-forever f32 KV, 49,152 B per position (1.6 GB at 32k, 6.4 GB at 131k), to serve a 6.3 MB window
    (survey §3, note 1).
  - **Bit-identical to Step 1:** the ring reads the same rows in the same order (G-RING).
- `attnSinksF64` stays in the tree as the exact arm of the fidelity gate. Removing it is a later, separate change.
- **Parity:** the change moves gpt-oss's numbers.
  - **Every other family:** its `deps_hash` refresh is non-numeric, proven by G-ID.
  - **gpt-oss:** its line in `testdata/parity_manifest.json` is refreshed only so the branch can be pushed, and is provisional. Its T3
    parity (`go run ./cmd/gate parity` for gpt-oss, then `-update`) is owed before merge, after G-FID.

## Gates (registered before the code)

### G-FID — the fidelity gate (`internal/fidelity`, TE12's protocol)

- **Box and model:** nobara's CPU, `gpt-oss-20b-MXFP4.gguf`.
- **Reference:** the same checkpoint loaded weight-only int8 with f32 activations (`Quant: "int8"`), on today's attention
  (`attnSinksF64`).
  - This is the nearest load to the checkpoint's own MXFP4 weights that fits the box. A float32 load needs about 84 GB, and nobara has 62.
  - goinfer dequantizes MXFP4 at load and has no native MXFP4 matmul.
- **Arms:** both int4 (`Quant: "int4"`), the load a CPU user runs.
  - The **exact arm** is today's path (`attnSinksF64`).
  - The **candidate** is Step 1.
- **Scoring:** both arms are teacher-forced on the reference's own greedy tokens.
  - A prompt's 32 positions are the one after the prompt and the ones after each of the reference's first 31 greedy tokens.
  - Each position records top-1 agreement with the reference, a hard flip (outside the near-tie band, `decoder.NearTieArgmaxForTest`),
    and KL(reference ‖ arm).
- **Cells:**
  - **short:** prompts of about 150 tokens.
  - **long:** prompts of about 600 tokens, more than four times the 128-key window, so the window layers' sliding read and the ring are
    exercised.
- **Prompts:** passages and instructions in the shape of `gpu/attn_fidelity_test.go`, with a pilot set that shares no passage with the gate
  set. `fidelity.PromptSetHash` is recorded.
  - **N, per cell:** from the pilot (2 prompts a cell): `python3 scripts/power.py fidelity --from-positions <the pilot's positions log>`.
    It gives a pooled count, so N a cell is half of it, rounded up.
  - N is at least 10 and at most 24 a cell. If power.py asks for more than 24, the gate is "cannot resolve" and stops with DECISION NEEDED
    before the gate set runs.
- **Criteria:** `fidelity.NonInferior`, pooled over the two cells, `AgreeMarginPts = 1`, `KLMargin = 0.10`, one-sided alpha 0.05, and hard
  flips under the pooled Poisson bound. Per-cell values are reported, with no per-cell veto.
- **Reported beside it, in Francis's words "against the current gpt-oss path":** the candidate against the exact arm directly. That is
  their top-1 agreement over the same positions, and the mean and maximum KL(exact ‖ candidate).

**Outcomes, and what each means:**

| reading | condition | meaning, and the next step |
|---|---|---|
| **PASS** | non-inferior on agreement, KL and hard flips | Step 1 leaves gpt-oss no further from the model than today's attention. Step 2 and the remaining gates follow |
| **AMBIGUOUS** | not PASS, but the agreement difference's lower bound is above −2 points, the KL ratio's upper bound is at most 1.25, and the flips are under their bound | within twice the registered margins: the evidence neither clears nor condemns it. **Stop, DECISION NEEDED** |
| **FAIL** | anything else | Step 1 moves gpt-oss measurably away from the model. **Stop, DECISION NEEDED**; nothing past Step 1 is built |

### The other gates

| gate | tier | what | bar |
|---|---|---|---|
| G-ID | 0, the Mac and nobara CPU | `go run ./cmd/gate identity origin/main <branch> -backend cpu -assets tiny -steps 64` | every family except gpt-oss IDENTICAL; gpt-oss is expected to differ (Step 1 changes its output) and is graded by G-FID |
| G-SINK | 0, the Mac | `decoder/ks_o4_sink_test.go`, over the shapes of KS-F2's G0: (a) the batched path with every sink at −∞ against the same path with no sinks; (b) with finite sinks, the batched path against `attendSinksToken` | (a) every word equal: the sink term adds only to the max and the denominator. (b) max \|diff\| at most 1e-5 of the context's largest magnitude, every cell: the two differ only in AV's accumulation and the score's rounding |
| G-RING | 0, the Mac | Step 2 against Step 1 on the tiny gpt-oss fixture, at a context past its window: full logits over 64 greedy steps, and the window rows read | byte-identical |
| G-MEM | 0, nobara | the KV bytes per position, and the fit guard's plan for gpt-oss-20b at 32k, before and after Step 2 | the window layers' KV is stated before and after, from the plan's own pricing; the bar is that it falls by the window layers' share less the ring's fixed bytes |
| G-SPEED | timed, nobara CPU, the timing lock, by day | KS-F2's harness (`TestKSF2_cpuDecodeSpeed`), A B B A over 10 pairs, with today's build against Step 1 + 2, gpt-oss-20b at depth 128 and 2048 | reported. **FAIL** if a cell has at most 1 of 10 pairs faster (a slowdown beyond doubt); otherwise it passes with the ratio stated. Not a speed claim: KS-O4's purpose is what it unblocks |

**Amended before any G-MEM or G-SPEED data (2026-10-26), after G-FID's acceptance:**
- **G-MEM, why its registered form cannot answer:**
  - The fit guard prices a sliding-window layer at its window plus a mirror (`Architecture.kvPositionsAt`, `kvBytesForCtx`), whether or not
    the layer keeps a ring. So the plan's own figure for gpt-oss is the same before and after KS-O4.
  - On `main`, gpt-oss's window layers keep append-forever KV, so the plan under-prices the memory actually held. KS-O4 makes it true.
- **G-MEM, as amended (tier 0, the Mac):** `decoder/ks_o4_mem_test.go:TestKSO4_kvHeldBytes` counts the bytes the cache actually holds
  (every layer's K and V arrays and ring buffers) on the tiny fixture, past its window, with the ring (KS-O4) and with every layer
  append-forever (`main`'s layout). It checks them against the per-layer arithmetic.
  - The same arithmetic, from gpt-oss-20b's config (24 layers, 12 window-128; 8 KV heads × 64), is stated for 32k and 131k.
  - The fit plan's figure is reported beside it.
  - **Bar:** the held bytes fall by exactly the window layers' append-forever bytes, less the ring's 2 × window positions, and match the
    plan's pricing after.
- **G-SPEED:** depth 128 only. KS-F2 measured gpt-oss-20b's CPU prefill on nobara at about 0.58 s a token, so a depth-2048 cell is about 8
  hours. A is `main` (which now carries KS-F2), B is KS-O4 merged with it.

**Amended before any data (2026-10-25 10:20, Francis: "KS-O4 A"): G-FID's budget.**

KS-F2's first nobara run measured gpt-oss-20b's CPU path there at about 0.58 s a token (prefill and decode), not the 0.13–0.15 s assumed
below, so the gate as registered would run past 5 hours.

| | was | now |
|---|---|---|
| short cell | about 150 tokens | **about 100:** the first two sentences of a passage, with an instruction |
| long cell | about 600 tokens | **about 300:** two passages, still more than twice the 128-key window |
| scored positions | 32 a prompt | **16:** the one after the prompt and after each of the reference's first 15 greedy tokens |
| N a cell | 10–24 from power.py | **at most 12** |
| run | by day | **one nobara night job,** `--not-before 22:00`, at most 3 hours |

- **How N is set:** the smallest of 12, power.py's ask (at least 10 a cell), and the number of gate prompts a cell that the 3 hours admit,
  using the pilot's own measured wall per prompt.
- **If even 10 a cell does not fit** the budget, the gate stops as "cannot resolve within the budget" (exit 3) before the gate set runs.
- **If power.py asks for more than 12,** the gate runs at 12, and the record states what power.py asked for.
- **Unchanged:** the criteria, the bands and what each outcome means.
- **Still owed before merge:** the real-checkpoint gpt-oss parity run.
- **Measured lengths:** the job prints the prompts' token lengths, and they go in the record.

**Superseded by the amendment above:** G-FID's cell lengths and cost. gpt-oss's CPU prefill runs token by token: its FFN form is not
batch-eligible (`program.batchEligible`). At about 0.13–0.15 s a token on nobara, a 1,500-token cell costs about 4 min a pass, so three
passes over 24 prompts run past 2 hours. The cells move to about 150 and 600 tokens, and the cost is restated below. No reading existed.

**Amended before any code (2026-10-25): G-SINK.** As first written it compared the batched path, word for word, with a hand-written
reference of its arithmetic. That reference would have to reproduce aikit's `MatmulQKAcc64` and `MatmulAVAcc64` contraction exactly,
which a test cannot promise. The amended G-SINK checks the two things it can:
- (a) the sink plumbing, by identity;
- (b) the closeness to today's path, at float32 rounding.

The fidelity gate, G-FID, is unchanged.

**Amended before any data (2026-10-25), by Francis's standing merge rule of 8:46–8:48 am (relayed by the coordinator):**
- **Order of decision:** G-FID decides first, because KS-O4 changes output. A change inside a registered fidelity gate is lossless under
  the rule.
- **Why it needs more:** KS-O4 changes gpt-oss's default path, so it is a default change, which needs the gain judged against its memory
  trade-off, and Francis's call.
- **How G-SPEED is graded:** also against the merge bar (median ≥ 1.05× and at least 9 of 10 pairs faster, every cell). Its registered
  reading is kept as recorded.
- **Memory, for the PR:** G-MEM's bytes per position before and after, device and host (this is the CPU path, so host), and the longest
  context the fit plan admits before and after.

## Pre-registration (`docs/measurements/noise-registry.md` §8)

- **Tier:** day.
- **Testing tiers:**
  - G-ID, G-SINK and G-RING are tier 0.
  - G-FID is tier 2. A change of output on a MoE model is decided only on the real checkpoint (tiny fixtures cannot carry the routing
    near-ties). Its wall clock is stated below.
  - G-SPEED is tier 1.
- **Interleaving:** G-FID's arms run in one process on one load, in alternating order per prompt (exact then candidate, then candidate then
  exact). G-SPEED is A B B A.
- **Cell validity:** each run records the load average and `MemAvailable` at its start. A run started above 8.0 load, or below the fit
  plan's need plus 2 GB, is void and re-run.
- **Reused readings:** none.
- **Fit check:** a load-only start of each load (int8 and int4) through the gate's binary before any scoring. A refusal exits 75. The two
  loads run one after the other, never at once (the reference's logits are kept, about 25 MB a prompt).
- **Never-swap:** the test process's `VmHWM` and peak `VmSwap`, from `/proc/<pid>/status`.
- **Instrument:** `internal/fidelity`, the house protocol for a decode change that moves output (CLAUDE.md, "Gate tiers").
- **Noise:** the spreads come from the pilot's positions log. There is no prior gpt-oss positions log.
- **power.py N:** from the pilot, as above.
- **Stopping rule:** fixed N. No early stop.
- **Ambiguous band:** as tabled. The owner decides.
- **Cost (G-FID, estimated, restated before any data):**
  - **Loads:** the reference int8 load is about 21 GB, held in the heap (`DirectLoad`, so no sidecar is written); the int4 load is about
    11 GB. About 1 min each.
  - **A pass** is the prompt prefilled token by token, then 31 forced steps, at about 0.13–0.15 s a token (the reference somewhat slower).
    A short prompt is about 25 s, a long one about 90 s.
  - **Three passes a prompt.** The pilot (2 + 2 prompts) is about 12 min. The gate at N = 12 a cell is about 14 min short and 55 min
    long.
  - **Total:** about 1.5–2 hours, run by day. The start and expected finish are stated when it is queued.

## G-FID result: AMBIGUOUS (2026-10-26, nobara, 00:31:05–02:54:21 PDT)

The night job `ks-o4-gfid` ran on nobara's `night.py serve` under the timing lock, at `b593e258` (KS-O4's code and the amended harness),
on `gpt-oss-20b-MXFP4.gguf`, CPU. Records: `raw/` here (copied from `~/goinfer-logs/ks-o4-2026-10/gfid-2026-10-26-0031/` on nobara).

- **Pilot** (00:31–00:51, 4 prompts, prompt-set hash `63fb95aa…`):
  - the per-prompt agreement spread was 5.98 points, and the KL residual's coefficient of variation 0.161;
  - `power.py` asked for 223 prompts pooled (112 a cell) to resolve the agreement margin of 1 point, and 18 for the KL margin;
  - the budget admitted 14 a cell, so the gate ran at the registered cap, **12 a cell**, as the amendment says.
- **Gate** (00:51–02:54, 24 prompts, prompt-set hash `746fec76aa18c9e2ffc61837fb1f0a39d4564c076cf31a05804118350d3a015f`):
  - it started at load 6.31 with 59,765 MiB available (valid);
  - the process peaked at VmHWM 31,280 MiB and **628 MiB of swap**. That moves time, not numbers, and G-FID grades no time.

| cell | prompts | agreement, candidate / exact | difference (lower bound) | mean KL, candidate / exact | ratio (upper bound) | hard flips, candidate / exact |
|---|---|---|---|---|---|---|
| short | 12 | 94.27% / 93.75% | +0.52 (−1.46) | 0.03177 / 0.03624 | 0.877 (0.943) | 0 / 0 |
| long | 12 | 94.27% / 94.79% | −0.52 (−3.19) | 0.03604 / 0.03918 | 0.920 (1.140) | 0 / 3 |
| **pooled** | **24** | **94.27% / 94.27%** | **+0.00 (−1.64)** | **0.03391 / 0.03771** | **0.899 (1.014)** | **0 / 3** |

**Graded against the registered rule** (`fidelity.NonInferior`: `AgreeMarginPts` 1, `KLMargin` 0.10, alpha 0.05, pooled):

| criterion | bar | reading | |
|---|---|---|---|
| agreement | lower bound ≥ −1.00 pt | −1.64 pt | not met |
| KL | ratio upper bound ≤ 1.10 | 1.014 | met |
| hard flips | candidate ≤ exact + 2√exact | 0 ≤ 3 + 3.46 | met |
| AMBIGUOUS band | not PASS; agreement bound > −2, KL bound ≤ 1.25, flips met | −1.64, 1.014, met | **inside** |

**VERDICT: AMBIGUOUS**, as the job graded it. Under the registration, KS-O4 stops here with DECISION NEEDED.

**What the reading says, beside the verdict:**
- **The point estimates all favour or equal the candidate.**
  - Agreement is identical (94.27% both).
  - The KL is 10% lower (ratio 0.899; the KL criterion passes).
  - Hard flips are 0 against 3.
- **Why agreement misses:** only its interval is too wide. With a per-prompt spread of about 6 points, 24 prompts give a one-sided
  bound near −1.6 to −2.0 points around a difference of zero. The 1-point margin needs about 100 prompts pooled.
- **The candidate against today's path, directly** (Francis's "against the current gpt-oss path"):
  - top-1 agreement is 365 of 384 positions (95.05%), with mean KL(exact ‖ candidate) 0.0136.
  - The largest KL, 1.72, is one position of one long prompt. The next largest is 0.13.
  - At the 19 positions where the two arms differ, the reference sides with the candidate at 8, with today's path at 8, and with neither
    at 3. That is no sign of a systematic shift.
  - Disagreement this large for a change in the last bits fits gpt-oss's discontinuous MoE routing: a rounding difference can flip an
    expert pick, and teacher forcing does not stop that within a position.

### Decision: accepted as non-inferior (Francis, 2026-10-26 07:56, relayed by the coordinator)

"A on KS-O4 G-FID — accept it as non-inferior and continue."
- **What was not met:** the registered agreement bound. The lower bound read −1.64 points against the −1.00 margin.
- **What he accepted it on, the point estimates:**
  - an agreement difference of 0.00 points (94.27% both);
  - a KL ratio of 0.899, whose bound (1.014) passes;
  - hard flips 0 against 3.
- **Next:** KS-O4 goes on to its remaining gates (G-ID on amd64, G-MEM, G-SPEED), then the owed real-checkpoint gpt-oss parity run, then
  its PR. The PR has a Gemini review before merge.

## G-MEM result: PASS (2026-10-26, the Mac, tier 0)

`TestKSO4_kvHeldBytes` on the tiny fixture (window 4), 19 positions:

| layout | KV in use | arithmetic |
|---|---|---|
| append-forever (`main`'s layout) | 9,728 B | equal |
| ring (KS-O4) | 6,912 B | equal |
| what the fit plan prices (`kvBytesForCtx`) | 6,912 B | equal to the ring layout |

**gpt-oss-20b, from its config** (24 layers, 12 window-128 and 12 full, 8 KV heads × 64, f32: 4,096 B a layer a position):

| | full layers | window layers | KV held | what the fit plan prices |
|---|---|---|---|---|
| **32k, before** | 1.61 GB | 1.61 GB | **3.22 GB** | 1.62 GB |
| **32k, after** | 1.61 GB | 12.6 MB | **1.62 GB** | 1.62 GB |
| **131k, before** | 6.44 GB | 6.44 GB | **12.9 GB** | 6.45 GB |
| **131k, after** | 6.44 GB | 12.6 MB | **6.45 GB** | 6.45 GB |

- **The saving:** 1.6 GB of host memory at 32k and 6.4 GB at 131k. Device memory: none, on the CPU path.
- **The pricing gap it closes:** on `main` the fit guard under-prices gpt-oss's CPU KV by exactly that much, so the longest context it
  admits is longer than the memory actually allows. After KS-O4 the plan and the memory agree, and the longest context the plan admits is
  unchanged.

## G-ID result: PASS (2026-10-26, the Mac and nobara CPU)

`go run ./cmd/gate identity 2abd032c2 <branch> -backend cpu -assets tiny -steps 64`, with the gitignored granite fixture copied in:

| box | revision | IDENTICAL | DIFFERENT | NOT RUN | record |
|---|---|---|---|---|---|
| the Mac (arm64), 08:00 | `b593e258` merged with `main` | 32 | 1, gpt-oss | 10 | `raw/g-id-mac-identity.log` |
| nobara (amd64), 09:09 | `dc378f6b5` | 32 | 1, gpt-oss | 10 | `raw/remaining/g-id-identity.log` |

- **Bar met:** every family that ran other than gpt-oss is IDENTICAL on both arches, including granitemoehybrid, lfm2, llama4_text,
  nemotron_h and qwen3_5_moe, whose cells ran in full. gpt-oss differs as expected and is graded by G-FID.
- **NOT RUN, on both boxes:** deepseek_v2, gpt2, kimi_k2, mellum, qwen2, qwen2_moe, qwen3, qwen3_asr, qwen3_vl and voxtral have no tiny
  identity asset on either box. They are reported, not counted as a pass.

## G-SPEED result: passes its registered reading; the merge bar is not met (2026-10-26, nobara CPU, 09:11–09:50 PDT)

The night job `ks-o4-remaining` ran on nobara under the timing lock, with KS-F2's harness. A is `2abd032c2` (`main` with KS-F2) and B is
`dc378f6b5` (KS-O4). gpt-oss-20b-MXFP4.gguf from `~/models`, int4, depth 128, 32 decode tokens; A B B A, 2 warm runs, then 10 pairs.
Records: `raw/remaining/`.

- **Validity:** no run was void. Each run started at a load of at most 7.37 (bar 8.0), with at least 58,721 MiB available against a need of
  16,184 MiB + 2,048, and none swapped.
- **Decode:** median 585.9 ms a token (A) against 567.2 ms (B). Pair ratios (B / A): 0.986, 0.971, 0.955, 0.970, 0.955, 0.886, 1.027,
  0.985, 1.006, 0.967.

| reading | bar | result |
|---|---|---|
| registered: FAIL if a cell has at most 1 of 10 pairs faster | | 8 of 10 faster: **passes**, at a median 1.030× |
| the merge bar (median ≥ 1.05× and at least 9 of 10 pairs faster) | | **not met** (1.030×, 8 of 10) |

- **Not a speed claim:** `grade_speed.py` printed PARK, which is KS-F2's small-gains label. KS-O4 registered G-SPEED as "reported, FAIL only on
  a slowdown beyond doubt", because its purpose is the KV saving.
- **Peak memory:** VmHWM 23,383 MiB (A) and 23,384 MiB (B). At 160 positions the window layers' saving is under 8 MB, so the peak does not move.

## Real-checkpoint parity: PASS (2026-10-26, nobara CPU, 09:55–09:56 PDT)

The night job `ks-o4-parity` ran `TestGptOssReal_gate` and `TestGptOssReal_logitParity` (`-tags realckpt`, int8) at A (`2abd032c2`) and
at KS-O4 (`960ef65a7`), against the HF bf16 golden from `scripts/gptoss_hf_oracle.py` (sha256 `75c2e662…`, copied from the box's main
checkout). Records: `raw/parity/`.

| arm | gate | logit parity: argmax | cosine | VmHWM | swap |
|---|---|---|---|---|---|
| A | PASS | 12650 = 12650 | 0.99848 | 31,638 MiB | 0 |
| KS-O4 | PASS | 12650 = 12650 | 0.99848 | 31,636 MiB | 0 |

- **Bar met:** argmax exact and cosine ≥ 0.99 at KS-O4, so the manifest's gpt-oss row is re-validated at `960ef65a7`: cosine_min goes from
  0.99843 (`25a4711`) to 0.99848. `docs/capability-matrix.md`, its JSON and `pull/capability-matrix.json` are regenerated from it.
- **The two arms agree to five digits** on this one prompt. KS-O4's difference is in the last bits, so it does not show at this precision.

## Status (2026-10-26): every registered gate has run; whether it lands is Francis's call

- **G-FID:** AMBIGUOUS, accepted by Francis (07:56).
- **G-SINK, G-RING, G-ID, G-MEM:** PASS.
- **G-SPEED:** passes its registered reading at 1.030×, and the merge bar is not met.
- **Real-checkpoint parity:** PASS.
- **Decision posted:** whether KS-O4 lands today, posted through the coordinator during the 2026-10-26 wind-down.
