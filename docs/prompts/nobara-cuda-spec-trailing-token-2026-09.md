# Prompt: the CUDA check of the speculation trailing-token fixes (nobara)

For a Claude Code session on `nobara-pc` (RTX 2070 SUPER 8 GB), repo `~/mycode/goinfer`. `git pull` first.

## What changed, and why it needs CUDA

Both fixes are in `docs/measurements/spec-vs-batching-metal-2026-09-27.md` (§4 and its two "Update" sections).

- **`0e579400`: the block drafter** (`--drafter`, `BlockSpec.generate`, CUDA only).
  - **The bug.** At a completed exit it recorded prompt + every emitted token as held in the resident KV. But the
    last emitted token, the next round's anchor, was never forwarded. So the next turn reused that position as it
    stood: a rejected draft's K/V, or nothing at `max_tokens` 1.
  - **The fix.** Forward the trailing token at that exit with the guard fallback's M = 1 step:
    `host.PrefillLastNArgmax([emb], pos)`, with its capture fused while the seam is armed, or `ForwardArgmax` when it
    is off. If the forward cannot run, commit only the positions that were written.
  - **What is proven so far:** only against stubs, on the Mac (`TestBlockSpecGenerate_commitsOnlyWrittenPositions`).
    The real drafter and kernels exist only on CUDA.
- **`97615930`: the n-gram loop** (`serve --spec ngram`) had the same gap. It left the trailing token unforwarded,
  one short.
  - It was measured and fixed on Metal, where the re-prefilled position changed every later turn.
  - On CUDA its trailing forward is `ForwardArgmax`, the one-row verify.
  - CUDA is where n-gram spec pays (2.14× on copy-heavy prompts), so it needs the same two-turn check.

## Before anything

- **Do not run GPU work while another session is timing.** This afternoon nobara was fully loaded by another
  session's run (load 16.5 on 16 threads). Check `uptime`, `nvidia-smi` and `ps` for its `go test` / `serve` / bench
  processes, and wait until it is done.
- **Assets:**
  - `GOINFER_CUDA_MODEL` defaults to `~/models/qwen3-4b`;
  - `GOINFER_DFLASH_F32` resolves to `~/models/qwen3-4b-dflash-f32` (`testdata/assets.json`);
  - never read a model from `/srv/models`; if an asset is missing, `models-pull` it.
- Record the NVIDIA driver version.

## 1. The existing CUDA gates, with the fixes in

```
GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 60m -v ./cuda/ \
  -run '^(TestGenerateBlockSpec_production|TestBlockSpecStream|TestFlashDecodeBlockSpecLane|TestFlashDecodeSpeculativeScope|TestFlashDecodeTwoModelSpecLane)$'
```

- Read the output for `--- PASS`, not `ok`, and `tee` it to a durable log.
- These check one generation each. A failure here is a regression in the new trailing forward, for example
  `PrefillLastNArgmax` at M = 1, or the one-row capture fuse.
- `TestDrafterVsOff_perSuite` is a timing comparison, and optional. If you run it, the box must be idle; record its
  numbers.

## 2. New: two-turn tests on CUDA (the property the fixes are for)

Write `cuda/spec_twoturn_test.go` (`//go:build cuda && goinfer_testhooks`, heavy-gated). Port the shape of
`metal/spec_multiturn_test.go`.

- **`TestBlockSpec_twoTurnsMatchPlain`**: qwen3-4b int4 on CUDA plus the DFlash drafter. Each arm gets a fresh load.
  - *plain:* turn 1 is `Generate` (greedy), ending at `max_tokens` N; assert it emitted N, so no EOS. Turn 2 is a
    strict extension (prompt1 + out1 + a ChatML user turn), also through `Generate`.
  - *spec:* turn 1 is the block drafter's `GenerateStream`; turn 2 is the same extension through `Generate`. A
    variant runs turn 2 through `GenerateStream` again.
  - Assert, per arm pair:
    - turn-1 ids are equal (losslessness);
    - turn-2 `PrefillReused` is equal, and equals len(prompt1) + len(out1);
    - turn-2 ids are equal (report the first differing token).
  - Use N ∈ {1, 17, 48}. N = 1 is the seed-only exit.
- **`TestNgramSpec_twoTurnsMatchPlain`**: the same shape with `GenerateNgramSpeculativeAdaptive` (serve's resident
  call) on the target alone, no drafter.
- **Run each test twice**, keeping the new tests both times:
  - at HEAD, with the fixes in;
  - with only the decoder file reverted: `decoder/blockspec.go` from `0e579400^`, `decoder/spec_ngram.go` from
    `97615930^`.
- **Expected:** reverted, turn 2 reuses one position less (n-gram) or a stale one (block), and its ids differ from
  plain's. With the fix, everything is equal.
- **If turn 2 still differs WITH the fix, that is a finding to chase before anything else.** It would mean CUDA's
  verify writes K/V that is not bit-identical to decode's, so speculation on CUDA is not token-identical across turns,
  for a reason other than this gap. Metal's `TestSpecVerify_forwardNMatchesForward` is the template for isolating it.

## 3. Record

- In the record's last "Update" section, replace the "Owed: the CUDA check" bullet with the results. Put the logs
  under `docs/measurements/spec-vs-batching-metal-2026-09-27/`.
- Update the task doc line in `docs/tasks/task-concurrency-2026-09.md` that says the block drafter's CUDA check is
  owed.
- Commit the new test and the docs with explicit paths. Then `git fetch` + rebase, push (the pre-push citation lint
  must be green), and check `gh run list`.
