# Prefill-gate reference files vs their prompts — identity audit (2026-09-26)

**Finding.** Every set-A reference file at **K = 512, 1024 and 3900**, on both machines and both models, is invalid for
1 to 4 of its 10 prompts. Each such file holds the logits of different text from the prompt the gates now feed.
Set A's K = 64, 128, 256 and 8000 cells and every set-B cell audited are valid.

## Why this happened

The gates tokenise their prompts from frozen snapshots (`testdata/prefill-gate-prose-a/`, `-b/`) added in `b0bdf43d`
on **2026-09-09**. Before that they read the *live* docs. Set A's references at K = 256/512/1024/3900 (S and D7) were
generated on **2026-09-05**, from the live docs of that day: at 14:35–15:35 on the Mac, and at 17:10–20:02 on
`nobara`. `QUEUE.md`, `benchmarks.md`, `legacy-benchmarks.md` and `audit-2026-09-02.md` changed early in their text
between 09-05 and 09-09.

The reference files carry no prompt identity, so nothing noticed for three weeks. The run that generated a file and
scored it in the same process (the 2026-09-05 CUDA §3 gate) was self-consistent. Every later run on those files was
not. Found 2026-09-25 by R17 ([`metal-decode-attn-r17-2026-09-25.md`](metal-decode-attn-r17-2026-09-25.md)), where the
exact arm's pooled KL of 0.485 on set A turned out to be four mismatched prompts.

## Method

`TestPrefillRefIdentity` (`metal/prefill_ref_identity_test.go`):

- For every `S|D7-K<K>-p<i>.bin` file in a set's reference directory, prefill the snapshot prompt's first K tokens on
  Metal and compute KL(the reference's prompt-final logits ‖ the resident's).
- S cells use qwen2.5-coder-1.5b; D7 cells use qwen2.5-7b from its `.int4.metal.giw` sidecar.
- A reference from the same text lands at the W4A8-vs-CPU level. Over the 150 valid (cell, prompt) pairs below, the
  largest is 0.80. A reference from different text lands at 3.1–18.1.
- `nobara`'s files were audited on the Mac from copies of each file's header and prompt-final logits (the first
  8 + 4·vocab bytes, which is all the check reads), through `GOINFER_REF_IDENTITY_DIR`.

Logs: [`ref-identity.log`](metal-decode-attn-r17-2026-09-25/ref-identity.log) (Mac, sets A and B),
[`ref-identity-nobara-setA.log`](metal-decode-attn-r17-2026-09-25/ref-identity-nobara-setA.log),
[`ref-identity-nobara-setB.log`](metal-decode-attn-r17-2026-09-25/ref-identity-nobara-setB.log).

## Results — prompts whose reference does not match (KL > 1.0)

| cell | written | Mac set A | `nobara` set A |
|---|---|---|---|
| S-K64, S-K128 | 09-20 (Mac only) | none | — |
| S-K256, D7-K256 | 09-05 | none | none |
| **S-K512** | 09-05 (`nobara` only) | — | **5** (8.91) |
| **D7-K512** | 09-05 (`nobara` only) | — | **5** (8.79) |
| **S-K1024** | 09-05 | **2, 8** (14.97, 13.22) | **2, 8** (identical values) |
| **D7-K1024** | 09-05 | **2, 8** (12.39, 11.67) | **2, 8** (12.44, 11.79) |
| **S-K3900** | 09-05 | **1, 2, 5, 8** (4.84, 3.11, 18.08, 6.46) | **1, 2, 5, 8** (identical values) |
| S-K8000, D7-K8000 | 09-13 (`nobara` only) | — | none |

The affected prompts are prompt 1 `completed/audit-2026-09-02.md`, 2 `QUEUE.md`, 5 `benchmarks.md` and 8
`legacy-benchmarks.md`. The first changed byte (snapshot vs `29d40c2d`) is 9,172, 2,894, 1,356 and 2,180
respectively, which matches the K each first appears at.

**Set B is valid everywhere audited.** Mac: S-K256/512/1024/3900 and D7-K256/512/1024, 70 pairs, KL
0.0000–0.80. `nobara`: S-K3900 (identical values to the Mac's, so the same files) and D7-K8000, 20 pairs, KL 0.0000–0.80.

## Verdicts that rest on the invalid cells

| record | cells | status |
|---|---|---|
| R2 gate (3), 2026-09-21 (`r2-attn-fa-rootcause-2026-09-21.md`) | Mac set A S-K3900 | **void**. Re-gated on set B 2026-09-25: PASSES. Retraction done. |
| R1 gate (3), 2026-09-21 (red-october R1) | Mac set A S-K3900 | **void** as a record (also contaminated by the executor leak); moot, R1 was killed on speed. |
| R17 set-A runs, 2026-09-25 | Mac set A S-K3900 | calibration only; the decision ran on set B. |
| CUDA chunk demotion, 2026-09-21 (`prefill-chunk-demotion-2026-09-21.md`) | `nobara` set A S-K3900 (confirmation cell) | **that cell is void**. The decision stands on its own bit-identity result (chunked fast == single-pass fast, 0 of 151,936 logits differ), which inherits the single-pass fast path's gate. |
| R16 §3.2 gate (2026-09-25) | Mac set A S-K256/1024 | gave no verdict (K=512 missing). S-K1024 is also invalid, so set A cannot decide it as is. |
| CUDA §3 (L2/L3 Phase 3), 2026-09-05 | `nobara` set A K=512/1024 | self-consistent: generated and scored in one run from the same docs. |
| CUDA vsum (2026-09-13), R6 flash-decode (2026-09-20) | K=8000 (set A), S-K3900 / D7-K8000 (set B) | valid. |
| Metal L1 fast-prefill decision, 2026-09-09 | set B K=256/512/1024 | valid. |

## What changed so this cannot recur silently

The audit test itself prints a per-prompt heartbeat. Its first version printed one line per cell only after all 10
prompts, so the 7B at K=8000 (about 1.5 minutes per prompt) ran silently for about 14 minutes; the GPU was at 100%
throughout.

Every gate now checks prompt identity per prompt and refuses a mismatched reference:

- the decode gate (`runDecodeFidelityGate`, `metal/r2_gate_test.go`): verdict VOID;
- the Metal prefill gate (`runPrefillGateSet`, `metal/prefill_gate_ref_test.go`): cell and pooled verdict VOID plus
  a test error;
- the CUDA prefill gate (`TestPrefillGateVsReferenceCUDA`, `cuda/prefill_gate_ref_test.go`): cell VOID plus a test
  error.

The check is KL(reference prompt-final logits ‖ exact arm's) > 1.0.

## Open

Set A's invalid cells should be regenerated from the snapshot (`TestPrefillGateReference`, `GOINFER_PREFILL_GATE_PROMPTS=a`,
`GOINFER_CPU_REF_KS=512,1024,3900`), keeping the 2026-09-05 files aside for provenance, before set A is used as a
decision set again. On this 16 GB Mac an f32 S build needs a fit-guard bypass, so `nobara` is the place to do it and
copy from.
