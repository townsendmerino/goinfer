# v0.21.0 §C1: the T3 parity sweep (nobara-pc, REV 7b50947a, aikit v1.57.0)

The §C1 numeric gate for v0.21.0, as a full sweep plus a scoped re-run of the four gates the sweep never reached. Same scoped form
the owner accepted for v0.20.0 (`../release-v0.20.0/scoped-revalidation-2026-10-01.md`). Both runs are pinned to 7b50947a (main moved only by
docs commits afterwards; no code file changed).

## Full sweep, 2026-10-05 20:05-22:42 PDT (`run-parity-sweep.sh`, log `sweep/parity-sweep.log`)

Verdict as printed: **5 BLOCKERs**, none numeric. The `realckpt real-model gates` cell hit its 120 min `-timeout` (progress line at 2h0m0s:
1788 tests finished, `TestQwen3MoeReal_oracle` in flight), so the cell exited rc=1 with zero `--- FAIL` lines, and the four gates after the
panic point were reported DID NOT RUN. v0.20.0's same cell took about 1h46; this run's loads from the `/srv/models` archive (qwen3.6-35b,
Laguna 63 GB, Llama 4 37 GB) were slower. The panic text itself was in `/tmp/gate_parity_realckpt_real-model_gates.json`, which the scoped
re-run overwrote; the log above holds the progress lines and the rc=1 verdict.

## Scoped re-run, 2026-10-06 07:06-07:07 PDT (`run-scoped-unreached.sh`, log `scoped-unreached/scoped.log`)

Pass rule written before the run (in the script header, committed 5f965949). The four unreached gates, checkpoints from `~/models`:

| gate | result |
|---|---|
| `TestQwen3MoeReal_oracle` | pass, row recorded (`qwen3_moe`) |
| `TestSmolLM3_3bReal_gate` | pass, row recorded (`smollm3`) |
| `TestSpark25Real_gate` | pass, row recorded (`spark2_5`) |
| `TestQwen3NextReal_oracle` | skip, the permitted one: neverConfirmed, needs ~59.3 GB against the ~43.4 GB fit guard |

Printed verdict: `7b50947a: ALL REQUIRED GATES GREEN (+1 coverage gap)` (the Qwen3Next skip), with the gate's own "SCOPED RE-RUN, NOT A
RELEASE-READY VERDICT" banner. 1m31s: the checkpoints were warm in the page cache from the night before, which the 40 min estimate did not
assume. The cell was in flight for 55 s at its first progress line, so the tests ran.

## What was merged into the manifest, and what was not

The emitter writes into each run's pinned worktree, not the repo. The two runs' manifests cannot be taken as they are: in the full sweep the
two real gates that never ran left `qwen3_moe` and `smollm3` at their tiny-golden rows (status `experimental`), a downgrade from the committed
`validated` real-oracle rows. So the committed manifest is the full sweep's emitted text with the `qwen3_moe`, `smollm3` and `spark2_5` blocks
taken from the scoped run. Result against the previous manifest:

- 33 families: `date` 2026-10-0x to 2026-10-05/06 and `validated_at` to 7b50947a. No other field changed.
- **`olmo3`: `experimental`/`tiny-golden` to `validated`/`full-forward-oracle`** (reference HF f32 `allenai/Olmo-3-7B-Think`). Its real gate
  `TestOlmo3Real_gate` (f32 against f32, exact argmax and continuation, last-logit cosine bar 0.9999) passed in this sweep. This is the one
  claim change. `TestParityManifest_methodTier` and `_fresh` pass on the merged manifest, and `docs/capability-matrix.*` were re-rendered.

## Ledger and warnings

- `TestInt4_forwardParity` re-confirmed at 7b50947a (`gate ledger promote --force`). It had been confirmed at 70be7081, before 4ad711d0
  changed its assertion to closeness bounds; it passed in this sweep.
- FIRST-RUN gates (`TestNemotron35LightningReal_oracle`, `TestQwen3NextReal_oracle`): both skipped, so nothing to promote.
- **Open: `TestGemma12Real_gate`** passed but emitted no row for gemma+gemma2 ("emitter missing?"). Those families keep their previous rows, restamped
  only by the tiny parity gates. Not investigated yet.
