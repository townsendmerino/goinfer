# D11 follow-up — a resident all-positions hidden state, so the Clef route can run on the GPU (2026-10-03)

D11 gave the Clef joint head its input on the CPU: `Model.PromptHiddenAll` returns the final-norm hidden state of EVERY prompt position. It said so plainly: it never asked a resident backend, because none exposed every position. This adds that, for CUDA.

## What was built

- **`decoder.ResidentResidualAll`**, an optional resident hook: ingest a sequence, return every row's residual stream after the last layer and BEFORE the final norm. `PromptHiddenAll` tries it first (same exclusive claim and fall-through as `PromptHidden`) and applies the final norm on the host in f32, exactly as its CPU path does. A decline, a wrong row count or a wrong width falls back to the CPU; a cancellation returns.
- **Why before the norm, on the host.** `ResidentHiddenLast` returns the last row only, and on CUDA that vector has been through the int8 activation quantization the LM head reads (the audit's F-D03 note). A head that consumes the hidden state itself, at every position, should not inherit that. The batched prefill already downloads the whole f32 `[M, hidden]` residual block for its tails, so the new tail (`tailResidualAll`) returns its rows; no kernel was added. Every tail but an ordinary single-row prefill runs the exact kernels, and a DeltaNet model always does.
- **Chunking:** a long prompt runs in passes of at most 512 rows (OOM halving as before); every pass keeps its rows, in order.
- **Metal implements it per token (added 2026-10-03, below); WebGPU does not**, and takes the CPU path.

## Metal (2026-10-03)

- **Where the pre-norm residual lives.** Metal fuses the final norm into the trunk: `encodeTrunkWith` runs the layers, then `encodeNorm(r.x → r.aq/r.aSc)`, so the normed
  value only ever exists as the int8 `r.aq/r.aSc` the LM head reads. That is the vector this seam exists to avoid, so it is never used.
  - Every layer writes its residual into `r.x`: `encodeLayerWith` passes `r.x` as the residual buffer.
  - The final norm's kernels (`rmsnorm_quant`, `layernorm_quant`) take `x` as `device const float*`, so they only read it.
  - So once the trunk has run, `r.x` holds the last layer's output untouched by the norm. There is no ping-pong buffer and no in-place norm.
- **`metalResident.ResidualAll`** (`metal/backend.go`) is `HiddenLast`'s loop. Per position it runs one `forwardHiddenNoHead(emb, startPos+i, false)` (one command-buffer
  submit), then copies `r.x[:H]` into a fresh row.
  - The guards are `HiddenLast`'s exactly: a paged MoE declined, `checkCap`, `Reset` at `startPos == 0` (DeltaNet), `ctx.Err()` per token, `takeExecErr` per token, and the
    embedding-width check. The rows come back only after every token succeeded.
  - **Cost: the same as `HiddenLast`, one submit per token.** That measured 7.2 to 7.6 ms per token on the Qwen3.5-0.8B on the M1 Pro (below). A batched version, collecting
    the residual block from `PrefillLast`'s dispatch graph, is the speed lever and its own parity-gated job. Not done.
- **`TestPromptHiddenAllResidentMetal`** (`darwin goinfer_testhooks`; run with `-tags goinfer_testhooks`, or it reports no tests): the CUDA test's checks, int4, three prompt
  lengths run back to back.
  - **Dense (`qwen3_5-tiny-normw`):** worst row cosine 0.99996049, mean 0.99997797 to 0.99998482, last row 0.99996851 to 0.99999008, bar 0.998. As on CUDA, the new path's
    last row is closer to the CPU than the existing int8 `HiddenLast` path's (0.99999008 against 0.99996245 at 40 tokens).
  - **MoE (`qwen3_5_moe-tiny`):** mean 0.99942073 to 0.99996288, last row 0.99994446 to 0.99998488. The worst row, 0.99300233, is reported and not asserted, for the router
    near-tie reason above; CUDA saw the same 0.993.
  - **Able to fail:** skipping the host final norm (`decoder/prompt_hidden_all.go`) fails the bit-identity check on both fixtures. The source was restored byte for byte.
- **Real weights, EXPLORATORY, by day, not graded** (`TestPromptHiddenAllResidentMetal_realWeights`, `GOINFER_METAL_QWEN35_REAL=~/models/qwen3.5-0.8b`, int4 on both
  sides, every row of 300- and 33-token prompts): per-row cosine against the CPU is mean 0.993, last row 0.9946 (300) and 0.9922 (33).
  - **The worst row is row 0**, the attention sink, at 0.9665. After row 0 the worst is 0.985 (300) and 0.991 (33).
  - **This is the decode kernels' agreement on these weights, not this seam's.** The existing shipped last-row path (`HiddenLast`) reads 0.9945 and 0.9921 against the CPU on
    the same prompts; the new path's last row reads 0.9946 and 0.9922.
  - The resident run took 2.3 s for 300 tokens and the CPU 2.75 s.
  - **The same check on CUDA** (nobara, RTX 2070 SUPER, 2026-10-03, EXPLORATORY, a temporary test that was not committed): the same model, int4 on both sides, prompts of 300 and 33 token ids `(i*37+11) mod vocab` (random ids, not natural text; the Mac's exact prompts may differ, so the comparison is approximate). Per-row cosine against the CPU: **mean 0.9962 (300) and 0.9950 (33); the worst row is row 0, the sink, at 0.9724; after row 0 the worst is 0.9911 in both; the last row 0.9958 and 0.9977; the existing `HiddenLast` path 0.9956 and 0.9973.** So the shortfall is **not Metal's alone**: two independent backends sit at 0.993 to 0.996 mean against the CPU on real 0.8B weights, against 0.9999 on the tiny random fixtures, and on both the new path's last row matches the old path's to the third decimal. The 0.998 bar belongs to the tiny fixtures and is not what to expect on a real model; the graded measure of a real device path is D13's `cuda-int4` arms (KL and top-1 against the reference rows), not a per-row cosine on random ids.
- **What this does NOT give.** This 16 GB Mac cannot hold the 9B Clef-flash resident: Metal's guard allows about 5.4 to 5.8 GB against the 6.04 GB it needs. So no
  real-weights Clef check ran here, D14 has no Metal cell, and the Clef route's backbone on a Mac stays on the CPU for the 9B. What this buys is the Clef route on a Mac with
  more memory, and the 0.8B and tiny fixtures as its only resident gates.

## Gates

- **`TestPromptHiddenAllResidentCUDA`** (`cuda goinfer_testhooks`, needs a device): on the tiny Qwen3.5 dense (random final-norm weight) and MoE, int4, three prompt lengths. The answer must equal the runner's own residual rows with the host's final norm applied (so the dispatch took the resident), and track the CPU per row. **Dense: worst row cosine 0.99996, bar 0.998.** The new path's last row is closer to the CPU (0.99999) than the existing int8-requantized `HiddenLast` path's (0.99996). **MoE: mean row cosine 0.9996, last row 0.99999, the worst row 0.993 reported and not asserted:** a random-init router has near-tied top-k scores, so a few positions pick a different expert on the device and only those rows move (the lesson of the MoE router-flip noise floor: floor the mean, not the min). A defect confined to one MoE position would not fail that bar; the dense case, on the same DeltaNet and attention kernels, pins per-position correctness.
- **`TestPromptHiddenAllResidentCUDA_chunked`:** passes of 16 rows give rows BIT-IDENTICAL to one pass.
- **Able to fail:** skipping the host final norm fails the first; keeping only each chunk's last row fails the second (sources restored byte for byte).
- **`TestPromptHiddenAll_residentRowsGetTheHostFinalNorm`** (decoder, no GPU): a stub resident's rows come back normalised on the host; a decline, a wrong row count and a wrong width fall back to the CPU's rows; the claim is released each time; a cancelled context returns.

## On the real weights (EXPLORATORY: three records, one run by day, never quoted as a result)

`cuda/cmd/serve` built with `-tags cuda`, `-model clef=~/models/clef-flash -backend cuda -quant int4`, nobara's RTX 2070 SUPER (8 GB), `POST /v1/systemone` with the first three D10 records. Load took 48 s; the decode path read `cuda-resident (int4)`. The same three records were then run through the full-precision harness in the cuda module (`cuda/clef_fidelity_test.go`), twice: with the token-embedding/LM-head table stored at int4 (`--embed-int4`, which serve and the CLIs turn ON by default with `-quant int4`) and at the int8 pin (the flag off).

| record | tokens | time | per token | GPU, table int4 (as served) | GPU, table int8 pin | CPU int4, int8 pin | CPU int8int8 | CPU q4k |
|---|---|---|---|---|---|---|---|---|
| 1 | 589 | 1.88 s | 3.2 ms | 0.8765 | 0.9186 | 0.9217 | 0.9278 | 0.9189 |
| 2 | 238 | 0.76 s | 3.2 ms | 0.9506 | 0.9474 | 0.9424 | 0.9179 | 0.9394 |
| 3 | 549 | 1.75 s | 3.2 ms | 0.1254 | 0.1085 | | | |

(P(true) in each cell.) The whole request is roughly 17 to 21 times faster on the device than the CPU arms' 54 to 68 ms per token at these prompt lengths.

**A correction to what this record first said.** It first reported that the GPU sat 0.045 from the CPU int4 on record 1 and read that as the shape of D6b's open GPU-int4 discrepancy for JEV. That was a mistake in the comparison, not a finding about the device: the server run had the embedding/LM-head table at int4 (the served default) while the CPU int4 smoke had the int8 pin. Run the same way, the device and the CPU agree: with the pin, the GPU reads 0.9186 and 0.9474 against the CPU's 0.9217 and 0.9424 (0.003 and 0.005 apart, inside the 0.01 the CPU arms sit from each other), and with the table at int4 the GPU reproduces the server's numbers to four decimals. So on these records **the flag moves P(true) by up to 0.04 (record 1: 0.9186 to 0.8765) and the device does not**, and this says nothing either way about D6b's JEV result, which is a different model and set. Two or three records cannot establish that the device path is faithful; they say a gap I reported was an artifact of mismatched settings. The graded GPU arms are D13's amendment 5b and 5c.

The flag matters beyond this record: any int4 Clef number, CPU or GPU, has to say which table it used. D13's int4 arms run as served (table at int4) and a diagnostic GPU arm runs the pin.

**Graded since (D13, 2026-10-03, 150 records; `decisions-d13-clef-fidelity-2026-10-03.md` §7).** CUDA int4 as served read mean KL 0.0440 and top-1 0.840 against the f32 reference, the CPU int4 as served 0.0512 and 0.840: the device is not worse than the CPU (paired mean KL difference -0.0071, 95% bootstrap [-0.0270, +0.0098], unresolved). The flag on the device (served minus pin) is -0.0011 [-0.0193, +0.0147], also unresolved on average, though it moved single records by up to 0.53 in P(true). The 3.2 ms per token held over all 150 records (decode path `cuda-resident (int4)` on every row).

## Not done

- **The graded GPU arms are queued, not run** (`d13-clef-cuda-int4` and `d13-clef-cuda-int4-pin`, D13 amendments 5b and 5c): the harness in the cuda module writes full-precision rows, which the root module's harness cannot (it may not link the cuda module). Nothing is said about this path's accuracy until they are graded against the same f32 reference rows as the CPU arms.
- **Metal:** implemented per token (above). The batched version is not built.
- **WebGPU:** not implemented.
- **The head still runs on the CPU, in f32**, and the per-token figure above is the whole request (encoder, device backbone and CPU head together); the head's own cost is unmeasured and D14 owns it.
