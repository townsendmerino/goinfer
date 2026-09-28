# Prompt: decisions D6a — Route A on Qwen3.5-9B, graded against JEV-9B (nobara, CUDA)

For a Claude Code session on `nobara-pc` (RTX 2070 SUPER 8 GB), repo `~/mycode/goinfer`. `git pull` first.

## The job

Run D6a exactly as pre-registered in
[`docs/tasks/task-constrained-confidence.md`](../tasks/task-constrained-confidence.md), in the section "D6a
pre-registration (2026-09-28)". Read that section first; do not restate or change its numbers.

D6a asks whether label scoring on the untrained base model (Route A) is good enough, or whether JEV's trained
decision head (Route B, D2–D4) has to be built. It moved here from the Mac because the 9B could not be GPU-resident
there: first an auto-pinned context over Metal's ceiling, then Metal's memory guard. The CPU path would take about 60
hours.

The artifacts are in `docs/measurements/decisions-d6a-2026-09-28/`:
- `select.py` (the deterministic samples);
- `run-d6a.sh` (the three arms through the production `goinfer-chat decisions-calibrate` / `decide`);
- `analyze.py` (the metrics and the decision).

The owner chose chat-v1 as the graded template (arm A). Arm B (bare-v1) is the control that must reproduce the
authors' published B0.

## Steps

1. **Inputs**, all into `~/models` and the bench dir, never `/srv/models`:
   - `curl -L -o ~/models/Qwen3.5-9B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf`.
     It must be exactly 5,680,522,464 B; record its sha256.
   - `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` is already in `~/models` (the MC1 grading used it).
   - The corpus: `calibration.jsonl` and `ood.jsonl` from
     `https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3/resolve/fc99c6357a9f89f7512c4a987314352addead049/<file>`,
     into `~/goinfer-bench/decisions-d6a/data/`. The sha256s must match the pre-registration.
2. **Samples:** `python3 docs/measurements/decisions-d6a-2026-09-28/select.py ~/goinfer-bench/decisions-d6a/data
   ~/goinfer-bench/decisions-d6a/samples`. Both sha256s must match the pre-registration. Stop if they do not.
3. **The binary,** once: `CGO_ENABLED=0 go -C cuda build -tags cuda -o ~/goinfer-bench/decisions-d6a/goinfer-chat-cuda-<rev>
   ./cmd/chat` at the commit that holds the pre-registration or later, named by its short hash.
4. **A resident check before the run.** Run `decide` on 3 rows *outside* both samples: the highest `sha256(id)`
   rows of `ood.jsonl`, as the Mac smoke run took them. Pass `--backend cuda --ctx 4096 --template chat-v1`.
   - Its stderr must show a CUDA-resident decode path.
   - If the resident build declines (a VRAM fit, or a dense-`qwen3_5` gap on CUDA), **stop and report the reason**.
     Do not grade a CPU run.
   - Record the per-row latency. The rows are not graded; say they were seen.
5. **The run,** detached, with start and expected finish times in local and UTC, and the log archived out of `/tmp`:
   `REV=<rev> BACKEND=cuda BIN=~/goinfer-bench/decisions-d6a/goinfer-chat-cuda-<rev> bash docs/measurements/decisions-d6a-2026-09-28/run-d6a.sh`.
   - Each arm is 3,572 prefills. Estimate from step 4's latency.
   - Check every arm's stderr for the resident path.
6. **Analysis:** `python3 docs/measurements/decisions-d6a-2026-09-28/analyze.py A=<eval-A.jsonl>:<cal-A.json>
   B=<eval-B.jsonl>:<cal-B.json> C=<eval-C.jsonl>:<cal-C.json>`.
   - Read the CONTROL line first: if arm B does not reproduce B0, arm A's verdict is not trusted until the
     difference is explained. Look at tokenization first: goinfer encodes a state as plain text, while HF parses
     special tokens.
   - Then read the DECISION line.
7. **Optional, reported only:** if the D0 fixture has landed (`testdata/decisions/`,
   `docs/prompts/nobara-decisions-d0-fixture-2026-09.md`), run arm A's `decide` on its items too, and report the
   mean KL to `jev9b_ref_bf16.jsonl`.

## Record

- **The record:** `docs/measurements/decisions-d6a-<date>.md`, in the house shape. Include:
  - the result first: the verdict, arm A's top-1 and ECE against the bars, and the control;
  - the setup, with every pin;
  - the per-kind table for every arm, raw and calibrated;
  - each arm's fitted T, flagging any at the search bound;
  - the deviations.
- **Raw data** under `docs/measurements/decisions-d6a-2026-09-28/`: the eval outputs, the calibrations, the stderr
  logs, the run log and `analyze.py`'s output. Write any file over 1 MB as `.gz`.
- **The task doc's D6a section:** add a "Result" block. If the verdict is "build D2–D4", or it lands in between, the
  next step is D2–D4, or the owner's call.

## Rules that apply

- **Pushing:** push with every outstanding file, staged by explicit path, never `git add -A`. Run `git fetch` and
  rebase first, and use `gh auth switch --user townsendmerino`.
- **Lints:** run the citation lint and read its exit code directly. Check CI after the push.
- **The bench set:** the model path must be under `~/models`. A path under `/srv/models` voids the run.
