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
  - **short:** prompts of about 200 tokens.
  - **long:** prompts of about 1,500 tokens, past the 128-key window, so the window layers' sliding read is exercised.
- **Prompts:** passages and instructions in the shape of `gpu/attn_fidelity_test.go`, with a pilot set that shares no passage with the gate
  set. `fidelity.PromptSetHash` is recorded.
  - **N, per cell:** from the pilot (2 prompts a cell): `python3 scripts/power.py fidelity --from-positions <the pilot's positions log>`.
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
| G-SINK | 0, the Mac | a unit test: the batched path with a sink against `attendSinksToken` with float64 AV substituted (the reference form of Step 1's arithmetic), over the shapes of KS-F2's G0 | every word equal |
| G-RING | 0, the Mac | Step 2 against Step 1 on the tiny gpt-oss fixture, at a context past its window: full logits over 64 greedy steps, and the window rows read | byte-identical |
| G-MEM | 0, nobara | the KV bytes per position, and the fit guard's plan for gpt-oss-20b at 32k, before and after Step 2 | the window layers' KV is stated before and after, from the plan's own pricing; the bar is that it falls by the window layers' share less the ring's fixed bytes |
| G-SPEED | timed, nobara CPU, the timing lock, by day | KS-F2's harness (`TestKSF2_cpuDecodeSpeed`), A B B A over 10 pairs, with today's build against Step 1 + 2, gpt-oss-20b at depth 128 and 2048 | reported. **FAIL** if a cell has at most 1 of 10 pairs faster (a slowdown beyond doubt); otherwise it passes with the ratio stated. Not a speed claim: KS-O4's purpose is what it unblocks |

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
- **Cost (G-FID, estimated):**
  - The reference int8 load is about 21 GB, about 1 min.
  - A short prompt is about 10 s per arm, a long one about 45 s, on CPU.
  - Three passes a prompt (the reference, then two arms), with N = 12 a cell plus the pilot's 4 prompts: about 40–50 min.
  - The start and expected finish are stated when it is queued.
