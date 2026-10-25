# SP2 — speculation on a recurrent hybrid: Qwen3.5-9B with DFlash on nobara (2026-10-24)

SP2 of track SP. The gates were registered before any code, in `docs/tasks/task-forward-architecture-2026-10.md` §6a, "SP2 on
nobara" (commit `7d5a96cf`, pushed before the first line of SP2 code). This page records each reading against that rule.

- **The mechanism:** `cuda/recurrent_spec.go` (the arena, `VerifyLastNArgmax`, `FoldVerified`), the twins in `cuda/deltanet.cu`
  (`delta_conv_rows_spec`, `delta_rule_rows_128_spec`, `delta_rule_rows_spec`), and the loop in `decoder/blockspec.go`
  (`BlockSpec.generate`, `Model.specRollbackSafeVia`).
- **The gates:** tier 0 in `cuda/recurrent_spec_test.go` and `decoder/recurrent_spec_test.go`; the 9B in
  `cuda/recurrent_spec_real_test.go`, driven by `run-sp2.sh` (`MODE=identity|cost|speed`).
- **Logs:** `~/goinfer-logs/sp2-2026-10/` on nobara; the JSON lines each 9B run writes are copied under `raw/`.

## The kernels: what changed in `deltanet.ptx`

Rebuilt with `cuda/build_ptx.sh deltanet` (NVRTC 12.9.86; the committed PTX first rebuilt byte-identically at main `d9a1cb4d`,
so the toolchain reproduces it). Every pre-existing kernel's floating-point instruction sequence is unchanged; the PTX of the
row-batched kernels differs only in labels, symbol names and address arithmetic order. Each verify twin's floating-point
sequence equals its committing kernel's (`delta_conv_rows_spec` = `delta_conv_rows`, `delta_rule_rows_128_spec` =
`delta_rule_rows_128`), and each shares one function body with it. The generic twin, `delta_rule_rows_spec`, is a separate body:
it holds the state row in a predicated 128-float register array where `delta_rule_rows` walks it in global memory, with the same
ascending-kd FMA chains; G1a shows it bit-identical on the two tiny fixtures that take it. None of the three declares local memory in its PTX; at the driver's JIT the 128-wide twin spills 80
B/thread, under the 96 B its committing kernel already spills (below).

## Readings

### Tier 0 (G1a, G1b, G5, G3's pricing): PASS on all three models, both plants caught (2026-10-24, 23:39–23:55 PDT)

Run on nobara under the timing lock (shared), driver 595.91.07: `go test -tags 'cuda goinfer_testhooks' -run TestRecurrentSpec
./cuda/` with `GOINFER_HEAVY_TESTS=1`, and the decoder's `TestSpecRollbackSafeVia_admitsOnlyTheFoldPath`,
`TestRecurrentSpecArenaBytes_formula` and `TestBlockSpecGenerate_foldsTheEmittedRows`. Final run at `f3d4b1b2`; log
`~/goinfer-logs/sp2-2026-10/tier0-1024-2339.log` and the run after the lint fixes.

| gate | qwen35-tiny (generic scan, MoE FFN) | qwen3next-tiny (generic scan) | qwen3.5-0.8b (128-wide scan) |
|---|---|---|---|
| G1a: verify leaves the state; fold n = prefill n, byte for byte, w ∈ {2, 4, 8, 16}, every n | PASS (30 cases) | PASS | PASS |
| G1a under P1 (fold skips the conv window) | caught, 30 of 30 cases | caught, 30 of 30 | caught, 30 of 30 |
| G1a under P2 (verify writes the state) | caught, 56 failures | caught, 56 | caught, 56 |
| G1b: scripted drafter, tokens and end state equal plain greedy, w ∈ {2, 4, 8, 16}, 2 prompts | PASS | PASS | PASS (128 tokens) |
| G1b with the guard tripped (plain steps after folds) | PASS | PASS | PASS |
| G1b under P1 / P2 | caught, 8 of 8 / 8 of 8 | caught, 8 / 8 | caught, 8 / 8 |
| G5: checkpoint untouched; kvmgr reuses all of A; state after A = cold prefill; B warm = cold, plain and speculative | PASS | PASS | PASS |
| G3: the reserve grows by the arena's price, which is the allocated bytes rounded up to 2 MiB | PASS | PASS | PASS |
| a seed longer than one pass keeps every captured row, bit-equal to one pass | — | PASS | — |

**Two defects that predate SP2, found by these gates and fixed on this branch:**
- **BlockSpec at the context cap.** On qwen35-tiny's 64-position context, a speculative turn emitted one token more than plain
  decode (a round's last, never forwarded), and with the guard tripped its fallback stepped past the cap and failed where plain
  decode finishes. `BlockSpec.generate` now takes plain decode's token budget (the context left after the prompt).
- **A long prompt's seed went out of device memory** (G1c's first run, below): with the drafter's capture armed, the CUDA prefill
  ran the whole prompt as one pass. It now takes `PrefillLast`'s passes and joins each pass's captured rows, so `serve --drafter`
  on a long prompt no longer needs one 3k-row pass beside the drafter.

**The source-scan gates:** `touched.py` green at `f3d4b1b2` (gofmt, build and vet of both modules, the cuda kernel lints, the
gate's heavy partition, which now classifies Qwen3.5-9B). The local-memory census pins the new `delta_rule_rows_128_spec` at 80
B/thread, under the 96 B of the scan it twins. staticcheck (CI's v0.8.1 build, its unused-field canary red first) is clean on
`./decoder/` and on `./cuda/` under `cuda goinfer_testhooks`.

### G1c, first run (2026-10-24, 23:41–23:45 PDT): stopped by an out-of-memory in the seed, not a divergence

All 36 cells it reached passed: 18 prompts (SP0's 12 thinking-off, its 6 thinking-on) at w = 4 and w = 16, tokens and every
DeltaNet layer's state byte-equal to plain greedy. The 19th prompt (2,924 tokens) went out of device memory in BlockSpec's seed
prefill, the defect above. The run is kept as recorded (`raw/identity-2026-10-24-2340.jsonl`); G1c is graded on the re-run.

### G1c, the re-run: PASS (2026-10-24 23:54 – 2026-10-25 00:02 PDT, at `f3d4b1b2`)

`MODE=identity bash run-sp2.sh`, the test binary run directly under the timing lock (shared), at load 1.6 or below with 58 GB
available. Peak resident 15.3 GB (`VmHWM`), peak swap 0. Raw: `raw/identity-2026-10-24-2353.jsonl`.

- **The load:** Qwen3.5-9B's `.giw`, resident context 4,608 (SP0's, held with the arena priced in), f32 KV, the drafter
  attached through `NewBlockSpec`. The arena allocates 31,817,728 B (30.3 MiB: 16 rows × 24 DeltaNet layers, the registered
  formula), and the fit guard prices it at 33,554,432 B (32 MiB, rounded up to 2 MiB as `cuMemAlloc` rounds it). The companion
  reserve: 1,326,350,336 B. Free VRAM with everything attached: 1.15 GB of 8.16.

| cell | prompts | w = 4 | w = 16 |
|---|---|---|---|
| code, thinking off | 6 | 6 of 6 | 6 of 6 |
| math, thinking off | 6 | 6 of 6 | 6 of 6 |
| math, thinking on | 6 | 6 of 6 | 6 of 6 |
| long (2,924 and 2,922 tokens) | 2 | 2 of 2 | 2 of 2 |

- **The bar:** every cell's speculative tokens equal plain greedy's (`Model.Generate` under the exact-attention scope), to the
  end, and every DeltaNet layer's conv window and state after the turn are byte-equal to plain's. **40 of 40.**
- **The guard's fallback** (tripped at its window, 4 code prompts at w = 4): 4 of 4.
- **The plants** at w = 4 on the 12 thinking-off prompts: P1 (fold skips the conv window) diverged on 12 of 12; P2 (verify
  writes the state) on 12 of 12. Registered expectation: at least one each. Met.
- **The served default's plain decode** (no exact-attention scope; recorded, no bar): equal to the speculative tokens on all 40.
- **G5 on the 9B:** the checkpoint's 50.2 MiB untouched by a speculative turn; the next turn reused 119 of 119 positions (A's
  prompt and output); the state after A byte-equal to a cold prefill of A's prompt and output; warm B equals cold B.

**Verdict: G1 passes (G1a, G1b, G1c), and G5 passes.** The path is correct on the real pairing; the PR opens on it.

**An observation, not a gate:** on the two long prompts the acceptance guard stopped speculating after its 6-round window (the
turns finished on plain steps, and identically). A diagnostic on two prompts of about 1,000 tokens (a source file and a question)
showed the same, and rules out the chunked seed: seeding in 512-row and in 256-row passes gave the same rounds and tokens. The
drafter accepted 1–4 tokens a round behind 1,000 tokens of code (mean about 2) against about 4 on the bare question. So DFlash's
acceptance on this pairing falls with context; G4's long cell reads what that costs.

### Review fixes before G2 and G4 (2026-10-25, from the PR's second-opinion review)

- **The fold path is prepared before the drafter attaches.** `NewBlockSpec` now asks the resident to prepare the arena
  (`PrepareRecurrentSpec`) and checks admission before `AttachBlockDrafter`, so a recurrent model the path refuses (Mamba-2,
  heads past 128, a sliding window, an arena that does not fit) gets no drafter weights on its device. The arena's price
  (`RecurrentSpecArenaBytes`) is 0 for each model the path refuses, so the fit guard reserves nothing for them.
  `TestPrepareRecurrentSpec_refusesBeforeAttach` pins each refusal.
- **G4's window and void rule.** Both arms time from the first token out to the end of the turn. BlockSpec folds the prompt into
  its drafter's context before its first token (`BlockSpec.generate`), so that work is outside the window in both arms; the last
  forward and the commit are inside it in both, since plain decode forwards its last token before its channel closes. The
  registered void rule is enforced in G2 and G4: a cell or pair whose start reads load above 8.0, or available memory under 15.3
  GB (the identity run's peak resident memory for this load) plus 2 GB, is recorded as void and re-run in order.
- **Plants and the guard trip** are armed through helpers that disarm them with `defer`, so a failing helper cannot leave a later
  test planted.
