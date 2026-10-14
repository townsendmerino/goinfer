# OP1 on Metal: the day gates (2026-10-13)

**Status: step 1 and step 2's mechanism pass their day gates. The softcap is NOT declared: its tiny parity gate (G2-T)
failed as registered.** The gates are registered in `docs/tasks/task-forward-architecture-2026-10.md` §4.6. Everything
here ran on the MacBook (Apple M1 Pro, 16 GB, macOS Darwin 25.6.0, go1.27.2) on 2026-10-13 between 20:28 and 20:53 PDT.
Nothing was timed. The night gates (identity on real checkpoints, the device gate, the speed guard, the peer cell) have
not run.

| commit | what |
|---|---|
| `10279588` | base: the digest instrument, kernels as on main `d03da21f` |
| `a83a75ed` | step 1: every attention kernel but `attention_fa_blk` passes its score through `attn_score` |
| `984797bf` | step 2: the softcap as a variant of the kernel source; not declared |

## Results

| gate | result | evidence |
|---|---|---|
| G1-K, kernel digests, step 1 | **PASS**: all 1,095 digests equal base's | `attn-digest-base.tsv`; 15 recording tests, 0 skipped |
| G1-M, model identity, tiny, step 1 | **PASS** for the 25 families that ran; 18 not run | `g1m-tiny-step1.log` |
| G1-S, the suite by name | **PASS**: 140 passed, 91 skipped (below) | `g1s-named-step2.verdicts.log` |
| G2-I, kernels, step 2 | **PASS**: 1,095 digests equal base's; the two kernel source files are unchanged since step 1 | the same digest file |
| G2-I, model identity, tiny, step 2 | **PASS** for the same 25 families | `g2i-tiny-step2.log` |
| G2-K, the capped kernels against float64 | **PASS**: 112 decode cases and 40 exact-prefill cases inside cosine 0.9999 and max abs 1e-2 | `TestAttnSoftcap_decodeKernels`, `TestAttnSoftcap_prefillKernel` |
| G2-D, a dropped cap is seen | **PASS** at kernel level: every one of the 152 cases misses the bar with the cap dropped. At model level the tiny fixtures fall to 0.9785 and 0.9799 | below |
| **G2-T, Gemma 2 tiny against the CPU** | **FAIL as registered**: worst logit cosine 0.998997 and 0.998657 against a bar of 0.999 | below |
| G2-R, Gemma 2 real | not run: it needs the declaration G2-T withholds, and the checkpoint is not on the Mac or in the archive (it is in nobara's local models only) | |
| G2-M, the matrix | not done: Gemma 2 still reads CPU on Metal | |

## G1-K: what the digest instrument covers, and how sensitive it is

- **It records** a sha256 of the raw output each kernel test reads back. Two runs at base were byte-identical, so the
  outputs are deterministic on this box.
- **Every attention kernel entry point has a line** (`TestAttnDigest_coversEveryKernel`): 22 of the sources' 32 names ran
  directly. The other ten are four template names and six group-size instantiations of the block kernel's rows form,
  which is the same template text as the one that ran (G = 6).
- **Planted defect, one ulp:** `attn_score` returning `x * 1.00000012f` changed digests under every routed kernel's
  label. It changed none under `attention_fa_blk`'s, as expected: that kernel is not routed.
- **`attention_prefill_steel` did not move under the one-ulp plant** (0 of 240 lines). INFERRED: its score is the dot
  product times a folded constant, the compiler folds the planted factor into that constant, and for the one scale the
  test uses the product rounds to the same float. A 1% plant changed 228 of its 240 lines and failed its own test. So
  the digest sees a 1% change in that kernel and is not shown to see a one-ulp one.

## G1-M and G2-I: the 18 families not run

A skip is not a pass. None of the 18 is a softmax family on Metal's exact or split-KV kernels that ran elsewhere at the
tiny tier:
- **Metal declines them** (so no Metal kernel runs): `bailing_hybrid`, `deepseek_v3`, `kimi_k2`, `gemma2`, `granitemoehybrid`,
  `laguna`, `lfm2`, `llama4_text`, `nemotron_h`, `phi3`, `spark2_5`, and at these fixtures' widths `qwen2_moe` and one
  cell each of `gemma4` and `glm_ocr`.
- **No tiny asset on this Mac:** `qwen2` and `qwen3` (their assets are small real checkpoints, absent here), `mellum` (a
  4.2 GB slice), `deepseek_v2`, `qwen3_asr`, `voxtral`.
- The plain dense Qwen shape did run at model level through the batched step's identity tests, on three generated
  models (`model-decode` lines in the digest file). The real checkpoints are the night run's.

## G1-S: the skips and three fixture links

- The run was `go test -tags goinfer_testhooks ./metal -run 'Attn|Attention|MC3|Prefill|Gemma|Softcap|KVI8|GptOss|Executor|Resident|Chain|Verify|Window|Sink'`.
- 91 tests skipped: they need `GOINFER_HEAVY_TESTS=1` and real checkpoints, and belong to the night's device gate.
- Three tests failed in the first run for want of gitignored fixtures a fresh worktree does not have
  (`TestGPT2ResidentParityMetal`, `TestHiddenLastResidentParityMetal`, `TestS16DeepstackPrefill_tiny`). With the two
  fixture directories linked from the main checkout all three passed. The verdicts file is the first run; the count
  above includes the rerun.

## G2-T: the failure, and what was measured around it

The registered rule is `TestGemma1ResidentParityMetal`'s shape on both tiny fixtures, 16 positions, both sides int4,
worst logit cosine ≥ 0.999. Both fixtures have cap 5, window 4, head dimension 32.

| fixture, int4 | worst cosine | against 0.999 |
|---|---|---|
| `gemma2-tiny`, with the cap | 0.998997 | fails by 0.000003 |
| `gemma2-hd-tiny`, with the cap | 0.998657 | fails by 0.000343 |

Diagnostics, taken after the failure with a scratch test that was not committed. They are not gates and move no bar.

| run | `gemma2-tiny` | `gemma2-hd-tiny` |
|---|---|---|
| the cap removed from the model on both sides (config edited) | 0.999116 | 0.999296 |
| the cap dropped from the Metal build only (the planted defect) | 0.978495 | 0.979910 |
| int8int8 on both sides, with the cap | 0.999255 | 0.999334 |
| `precise::tanh` in the capped function | 0.999044 | 0.998657 |
| for scale: `gemma1-tiny` 0.999995, `gemma3-vl-tiny` 0.999847 | | |

**What these say, and what they do not.**
- The softcap is in effect: dropping it from the build costs two points of cosine on these fixtures.
- The fixtures' own agreement with the cap removed everywhere is 0.9991 and 0.9993. The registered bar sits within
  0.0003 of that, so the gate had almost no margin before any softcap was added.
- The capped runs are 0.0001 and 0.0006 under their own no-cap controls. That is not shown to be the softcap's error:
  a model with a cap is a different function from one without, at position 0 (where no softmax runs) every variant
  reads 0.99989, and a precise tanh moves one fixture and not the other.
- **It is still a failure of the gate as registered.** The bar was taken from Gemma 1's test without measuring this
  fixture's control first, which is the mistake `metal/gemma_parity_test.go` warns about ("the bar is read off the
  committed control, not assumed").

**G2-T: FAILED.** The owner's ruling (OP1-M1, 2026-10-13): the bar stays at 0.999, and it is not amended against the
numbers above.
- **The cause:** the 0.999 bar was copied from Gemma 1's test without first measuring this fixture's no-softcap baseline
  at int4, which is 0.9991 to 0.9993.
- **The softcap's own contribution** is shown by the planted defect: with the cap dropped from the Metal build the
  fixtures read 0.978 and 0.980.
- The declaration stays on the branch `op1-metal-declare-2026-10`, unmerged. What decides it is in the task doc's §4.6:
  G2-T2, then the real Gemma 2 2B gate.

## Deviations from the registration

- **`attention_fa_blk` is not routed** in step 1. Registered as an amendment before any Metal code (`d03da21f`).
- **G1-S** needed two gitignored fixtures linked into the worktree (above).
- **G2-T's diagnostics** used a scratch test and config-edited copies of the fixture in a temporary directory.
