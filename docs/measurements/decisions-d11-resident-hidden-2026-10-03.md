# D11 follow-up — a resident all-positions hidden state, so the Clef route can run on the GPU (2026-10-03)

D11 gave the Clef joint head its input on the CPU: `Model.PromptHiddenAll` returns the final-norm hidden state of EVERY prompt position. It said so plainly: it never asked a resident backend, because none exposed every position. This adds that, for CUDA.

## What was built

- **`decoder.ResidentResidualAll`**, an optional resident hook: ingest a sequence, return every row's residual stream after the last layer and BEFORE the final norm. `PromptHiddenAll` tries it first (same exclusive claim and fall-through as `PromptHidden`) and applies the final norm on the host in f32, exactly as its CPU path does. A decline, a wrong row count or a wrong width falls back to the CPU; a cancellation returns.
- **Why before the norm, on the host.** `ResidentHiddenLast` returns the last row only, and on CUDA that vector has been through the int8 activation quantization the LM head reads (the audit's F-D03 note). A head that consumes the hidden state itself, at every position, should not inherit that. The batched prefill already downloads the whole f32 `[M, hidden]` residual block for its tails, so the new tail (`tailResidualAll`) returns its rows; no kernel was added. Every tail but an ordinary single-row prefill runs the exact kernels, and a DeltaNet model always does.
- **Chunking:** a long prompt runs in passes of at most 512 rows (OOM halving as before); every pass keeps its rows, in order.
- **Metal and WebGPU do not implement it** and take the CPU path. The Metal side is the Mac's job: its headless forward is per-token (`forwardHiddenNoHead`), which would collect each token's residual.

## Gates

- **`TestPromptHiddenAllResidentCUDA`** (`cuda goinfer_testhooks`, needs a device): on the tiny Qwen3.5 dense (random final-norm weight) and MoE, int4, three prompt lengths. The answer must equal the runner's own residual rows with the host's final norm applied (so the dispatch took the resident), and track the CPU per row. **Dense: worst row cosine 0.99996, bar 0.998.** The new path's last row is closer to the CPU (0.99999) than the existing int8-requantized `HiddenLast` path's (0.99996). **MoE: mean row cosine 0.9996, last row 0.99999, the worst row 0.993 reported and not asserted:** a random-init router has near-tied top-k scores, so a few positions pick a different expert on the device and only those rows move (the lesson of the MoE router-flip noise floor: floor the mean, not the min). A defect confined to one MoE position would not fail that bar; the dense case, on the same DeltaNet and attention kernels, pins per-position correctness.
- **`TestPromptHiddenAllResidentCUDA_chunked`:** passes of 16 rows give rows BIT-IDENTICAL to one pass.
- **Able to fail:** skipping the host final norm fails the first; keeping only each chunk's last row fails the second (sources restored byte for byte).
- **`TestPromptHiddenAll_residentRowsGetTheHostFinalNorm`** (decoder, no GPU): a stub resident's rows come back normalised on the host; a decline, a wrong row count and a wrong width fall back to the CPU's rows; the claim is released each time; a cancelled context returns.

## On the real weights (EXPLORATORY: three records, one run by day, never quoted as a result)

`cuda/cmd/serve` built with `-tags cuda`, `-model clef=~/models/clef-flash -backend cuda -quant int4`, nobara's RTX 2070 SUPER (8 GB), `POST /v1/systemone` with the first three D10 records. Load took 48 s; the decode path read `cuda-resident (int4)`.

| record | tokens | time | per token | P(true) on the GPU | CPU int4 | CPU int8int8 | CPU q4k |
|---|---|---|---|---|---|---|---|
| 1 | 589 | 1.88 s | 3.2 ms | 0.8765 | 0.9217 | 0.9278 | 0.9189 |
| 2 | 238 | 0.76 s | 3.2 ms | 0.9506 | 0.9424 | 0.9179 | 0.9394 |
| 3 | 549 | 1.75 s | 3.2 ms | 0.1254 | | | |

The CPU arms ran about 54 to 68 ms per token, so the whole request is roughly 17 to 21 times faster on the device at these prompt lengths. **The fidelity is the open question:** the three CPU arms agree with each other to within 0.01 on record 1, the GPU is 0.045 away, and on record 2 it is 0.008 from the CPU int4. This is the same shape as D6b's open item for JEV ("the GPU's int4 kernels are not the CPU's, so the two answers differ: slightly on most items, and on one of the 150 by a lot"). Two records cannot say which this is.

## Not done

- **No graded GPU arm.** D13's harness cannot link the CUDA backend (the root module must not depend on the cuda module), so a graded run goes through the server and a client; the probabilities there are rounded to four decimals, which the grader would need to account for. Owed: a GPU int4 arm graded against the same f32 reference rows as D13's CPU arms, before anything is said about this path's accuracy.
- **Metal and WebGPU:** not implemented (above).
- **The head still runs on the CPU, in f32**, and the per-token figure above is the whole request (encoder, device backbone and CPU head together); the head's own cost is unmeasured and D14 owns it.
