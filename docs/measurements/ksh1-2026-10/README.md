# KS-H1: Metal blocks the exact-attention and E-model prompts, 2026-10-22

Gates: `docs/tasks/task-kernel-survey-2026-10.md`, "KS-H1 gates" (KH1-G), registered before the code (8dd54168), after
re-checking the finding on main 0e620108. Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. Built on MEM6's branch.

**The build.** `metal/backend.go:metalResident.PrefillBlockRows` no longer excludes exact-attention models and Gemma 4 E-models.
- It blocks whenever the widest exact-attention head dim is at most 256, where the exact and tiled kernels a block may choose are
  bit-identical.
- An E-model blocks on its batched pass.
- A model with a 512-wide exact-attention layer (Gemma 4's global layers) stays unblocked: its tile choice would have to key on
  the whole prompt's length, which the resident does not see (asked, not built).
- The decoder now splits these prompts, and admission prices one pass (`residentPrefillScratch`) instead of the CPU formula.
- The function's comment, which still gave the 4,096-key decline as the reason (KS-DOC-code note 6), now says what the code does.

| gate | reading | result |
|---|---|---|
| KH1-G1 identity, llama | `TestKSH1_exactAttentionBlocks`: synthetic llama-tiny, exact kernel forced, 32-key tile, 300 rows whole against 64-row blocks: state hash (last logits + 16 steps) equal (`0x880d640a…`), 0 of 4 K/V buffers differ, every pass inside its scratch figure | **PASS** |
| KH1-G2 identity, qwen35 and E-model | `TestKSH1_qwen35AndEModelBlocks`: 600 rows whole against 128-row blocks, Metal, batched route: qwen35-tiny equal (0 of 2 K/V buffers differ), gemma4-emodel-tiny equal (0 of 8). On the real Qwen3.5-0.8B (head dim 256, exact kernel), `TestI8_exactKernelModelIsSplit` (formerly `…IsNotSplit`): blocks of 64 and 256 give one pass's tokens, log-probabilities and state | **PASS** |
| KH1-G3 admission | the exact-forced llama-tiny resident at 20,000 tokens: `residentPrefillScratch` ok, equal to `PrefillScratchBytes` of the longest pass plus its embeddings (main: not ok, so the CPU formula) | **PASS** |
| KH1-G4 unchanged | the whole `metal` package: every test passes but three that fail identically on main f3fc2cbf (`TestOlmoHybridResidentSmokeMetal`, `TestQwen35ResidentDecodeRateMetal`, `TestResidentSelfTest_passesOnHealthyMetal`: a Gated-DeltaNet fixture declines "resident at int4 only"; not this change). The one test that asserted the old exclusion now asserts the split | **PASS** |
| KH1-G5 measure | Qwen3.5-9B at 20K, `mem0_darwin.py` | not run: registered by day, and this ran in the evening |

**Plant** (a block pass attending from position 0): red on all three fixtures.

**Open:**
- **G5**, by day.
- **The head-dim-512 models** (E2B's global layers, about −2.1 GiB at 21K in the survey's estimate): the tile choice must key on
  the whole prompt's length, by a new decoder-to-resident hint or by keying every block's tile on a length both sides agree on.
