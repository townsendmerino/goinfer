# CUDA batched prefill on the q4k lane (Phi-3 mini): registration (2026-11-23)

**Status: REGISTERED 2026-11-23 08:20 PST, before any tier-1 or timed run.** Gates in order, each passing before the next; fixed N,
no early stop. If it ships, the batched pass becomes the default route for Phi-3's default load only with Francis's word.

**Question.** Does Phi-3 mini's default CUDA load (`--quant q4k`: Q5_K q/k/v, Q4_K o/gate/up, Q4_K or Q6_K down, Q6_K head, all
per-32 activations) prefill faster through a batched pass than through the route main takes, one decode forward per prompt
token (`PrefillPath`: "sequential: batched prefill needs int4 or int8 projections"), with output unchanged?

**The change.** Row-batched twins of the per-32 decode kernels (`actgroup.cu`): the three quantizers one block per activation row,
and the Q4_K, Q5_K, Q6_K and int8 GEMVs looping an M tile of activation rows over weights held in registers. Each (output row,
activation row) pair runs decode's arithmetic in decode's lane order, so the pass is designed to be bit-identical to the
sequential path. The fast levers do not apply (attention at head dim 96 and block-quant weights are outside L2 and L3), so the
pass is the exact path at every length. The per-row activation scales grow from 1 float to 2·K/32; `PrefillScratchBytes` prices them.

**Gate 1, tier 0 (defect detector, not a decision).** On small shapes, every rows kernel bit for bit against its decode kernel
(M = 1..33, odd N, K = 256 to 8192), each quantizer against its decode twin, a planted defect red; `TestAttnKernelDigests`
unchanged; the tiny per-32 model admitted to the pass (`PrefillPath` reads batched) and its `PrefillLast` logits bit-equal to
the sequential loop's.

**Gate 2, bit-identity on the real model** (`TestOP6C_q4kPassBitIdentity`, a new real-checkpoint test in the CUDA backend,
`~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`, `q4k`, serve's f16 KV, exclusive lock). Prompt set A's first three
prompts at K = 128 and 3,900: the seed logits after the prompt, then 16 greedy decode steps, each step's full logits, from the
batched arm against the sequential arm. **PASS:** every value bit-equal. **Not bit-equal:** a defect against the design; it is
debugged, not graded by a looser bar. Only if the difference traces to an attention kernel that is exact but ordered differently
from decode's, the record says so, and `fidelity.NonInferior` (AgreeMarginPts 1, KLMargin 0.10, alpha 0.05) against the
sequential arm on set A at K = 128 and 3,900 decides, registered here before any reading.

**Gate 3, speed** (`TestOP6C_q4kPassSpeed`, exclusive lock, after gate 2). A = the sequential loop (main's route, the do-nothing
arm), B = `PrefillLast`, one load, P = 128 and 3,900 from position 0 on set A's first prompt. Per P one untimed call of each arm,
then 5 pairs alternating A B / B A (10 per P in all). Each cell records the load average and available memory at its start; a cell
started under heavy load (load above 8.0, or available memory below the fit need plus 2 GB) is void, re-run in order, at most 4 a
P, and listed. Metric: the median of the per-pair A ÷ B TTFT (wall time to the seed logits, the device synchronised).
- **ADOPT:** median ≥ 3.0 at 3,900 and ≥ 2.0 at 128, each with at least 9 of its 10 pairs above 1.
- **KILL:** median under 1.5 at 3,900.
- **AMBIGUOUS, parked to Francis:** anything between.

The PR states the pass's device scratch at the width the build chooses and the load's fit verdict; the standing device bytes and
the longest context that fits are unchanged, since the kernels add module code and no standing buffer.

**Stopping rule.** One run of each gate decides; a run is repeated only when a P exhausts its void re-runs.

**What each outcome means.** ADOPT: DECISION NEEDED to Francis with the readings (the default route for Phi-3's default load).
AMBIGUOUS: the branch stays open for Francis. KILL, or a gate 2 failure that the fallback does not clear: the record says why and
the lane keeps the sequential route; a tensor-core q4k loader is then its own registration.

**Cost.** Gate 1 about 10 minutes of card tests. Gate 2 about 6 minutes (the sequential arm is about 40 s a 3,900-token prompt).
Gate 3 about 10 minutes. By day; nothing on the card after 21:30 PST.
