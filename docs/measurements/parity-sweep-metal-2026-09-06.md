# Parity sweep — the Metal/arm64 half (2026-09-06)

**Machine:** MacBook, `darwin/arm64`, 16 GB RAM. **Tree:** goinfer `d72400e`, aikit v1.37.0.
**Command:** `EMIT_MANIFEST=1 GOINFER_MANIFEST_MACHINE=mac go run ./cmd/gate parity -logdir
~/gate-logs/parity-metal-2026-09-06-final`. **Wall clock: 8m10s.** Companion to
[`parity-sweep-aikit-rearm-2026-09-06.md`](parity-sweep-aikit-rearm-2026-09-06.md), the amd64
half — read that one first; this is the arm64 half of exactly the same re-arm.

Three runs total: the first two are why two real bugs below exist and are fixed, not just
reported. Log archived outside the worktree and outside `/tmp`, per `docs/prompts/
macbook-metal-revalidation.md`'s own trap list — hit none of its traps this time (logdir made
first, `launchctl submit` used with self-removal, `cd <repo>` remembered on the third attempt
after failing the first two silently on it).

## Result: the pin cost nothing measurable on arm64 either

| cell | pass | skip | fail | wall |
|---|---|---|---|---|
| `./decoder/ ./tokenizer/` | 440 | 106 | 0 | 8m5s |
| realckpt real-model gates | 0 | 38 | 1 | 3s |

Dramatically faster than amd64's 2h51m — not because arm64 is faster per test, but because only
**4 of 39 registered assets resolve on this Mac** (vs. amd64's fuller `~/models`), so most
realckpt-tagged gates skip on a missing asset in milliseconds instead of loading a real
checkpoint. The pass/skip split is not comparable between the two runs; the **cosines on the
families that DID run** are the comparable number, and they agree with amd64.

**11 rows recorded, all on `machine: mac`:**

| family | method | argmax | cosine (min/mean) |
|---|---|---|---|
| bailing_hybrid, glm4_moe, gpt2, granite, lfm2, mistral3, olmo3, olmo_hybrid, qwen3_moe, smollm3 | tiny-golden / full-forward-oracle | 100.0% | **1.00000 / 1.00000** |
| qwen2_5_vl | tiny-golden | 100.0% | 0.99998 / 0.99998 |

Zero families disagreed with amd64's characterization of any shared family. **arm64 did not
disagree — that is itself the finding the brief asked to watch for**, and it means the Metal/NEON
code path is not silently drifting from what the amd64 sweep already re-armed.

## Two real bugs found running this, both fixed (not just reported)

Neither is arm64-specific in cause — both were latent in the merge tooling and only surfaced
because this was the first time anyone ran it from a machine that is not `nobara-pc`.

**1. `emitParityRow` let a legitimately-NaN cosine corrupt the whole merge.**
`cosineToFull` (`decoder/forward_test.go`) documents returning `NaN` when its full-logit-dump
reference is absent — a gitignored, locally-regenerated fixture, not present in this checkout.
`TestMixtral_forwardParity`'s tiny-golden checks (argmax/sample/top-k) all passed against the
checked-in golden, but the full-cosine comparison came back NaN, and `emitParityRow` formatted
that straight into JSON with `%.5f` — a literal `NaN` token, which is not valid JSON and failed
`TestParityManifest_merge` for the run, not just for mixtral. First run here: `❌ merge failed:
exit status 1`. Fixed in `decoder/parity_emit_test.go`: a NaN metric now logs and skips emitting
a row (same shape as an already-established "SKIP, no row expected" real-model case) instead of
being recorded as if it were measured. Verified directly:
`GOINFER_MANIFEST_EMIT=1 go test ./decoder -run TestMixtral_forwardParity -v` now passes with
"not emitting a row" logged, no `PARITY_ROW` line.

**2. The merge's default machine stamp was a specific box's name, not a real default.**
`TestParityManifest_merge` fell back to the literal string `"linux-62gb"` when
`GOINFER_MANIFEST_MACHINE` was unset — accurate by coincidence every time it had ever run
(always from `nobara-pc`), and silently wrong the first time it ran from anywhere else. Second
run here re-merged five already-`mac`-validated rows (`granite`, `mistral3`, `olmo3`,
`qwen3_moe`, `smollm3`) as `"linux-62gb"` — a real machine this Mac is not. Fixed in
`decoder/parity_manifest_test.go`: the default is now `runtime.GOOS + "-" + runtime.GOARCH`
(`darwin-arm64`, `linux-amd64`, …), which can be less pretty than a remembered hostname but can
never silently claim to be a *different* one. This run's own manifest rows still say `mac` — that
came from passing `GOINFER_MANIFEST_MACHINE=mac` explicitly on the third run, matching the
existing convention for this box, not from the new default.

## The one failure, dispositioned — and NOT overridden

**`TestGptOssReal_gate` — the fit guard refused it, correctly, and the override was deliberately
not taken.** `gpt-oss-20b-MXFP4.gguf` needs ~19.8 GB resident at quant int8; this Mac has 16 GB
(70% budget = 11.2 GB). Unlike amd64's `TestLlama4Real_gate` (refused, then re-run with
`GOINFER_NO_FIT_GUARD=1` to confirm the guard's arithmetic, accepting a 31 GB swap excursion on a
62 GB box), this one was **not** re-run past the guard: this exact model has a real prior
incident on this exact machine (documented separately — a CPU-staged run of a comparably-sized
checkpoint drove swap to within ~1 GB of the swap file's own ceiling). The guard's number and the
prior incident agree, so this is treated as "the model genuinely does not fit" rather than "the
estimate might be wrong," and left refused rather than forced through.

## What this sweep did NOT validate — read this before trusting the green

**35 of 39 assets are unresolved on this Mac** (vs. amd64's 4) — every family whose T3 evidence
depends on a real HF/GGUF checkpoint not present in `~/models` here rode the aikit-version-bump
`deps_hash` refresh with **zero fresh evidence from this run**. That is most of the manifest.
This sweep's actual contribution is narrow and stated precisely: the 11 families above that
*do* have a local tiny-golden/synthetic fixture, confirming their Metal/arm64 forward path agrees
with the amd64-validated numerics after the aikit v1.37.0 pin. It is not a claim that the other
~40 families were re-checked on arm64 — they were not, and their manifest rows still say
`linux-62gb` or carry no machine at all.

Same emitter-coverage caveat amd64's own doc raised still applies and was not re-investigated
here: a gate that passes without calling `emitParityRow` looks identical, after a merge, to a
gate that was never re-run. `TestQwen3MoeReal_oracle` did call it and recorded a row (see above);
whether every other real-model gate on this Mac's own asset list does the same was not audited
in this pass.

## Items 2–4 from the brief

- **Item 2 (C3, the Metal consumer manual device gate)** — **done, green**:
  `go run ./cmd/gate gpu` (run sequentially after Item 1's sweep finished, not concurrently, given
  this Mac's RAM headroom). `PASS — metal on Darwin @ eb113cd (2026-09-07T02:55:49Z)`: 9/9 declared
  check groups reported, 10 pass / 2 skip / 0 fail, both skips legitimate (no `nvidia-smi` on a
  Mac; 17 linux-only CI hygiene steps). Log archived at `~/gate-logs/gpu-metal-2026-09-06/`,
  outside the worktree and outside `/tmp`. Found `RELEASING.md` §C1-M's own "known-red" warning
  for `TestMetalSnapshotGolden` was stale — it passed here, and the re-bake it said was owed
  (`160dc3f`, 2026-08-06) had already happened and already matched the doc's own stated validation
  shape; fixed the doc rather than re-investigating a non-issue.
- **Item 3 (agent CLI)** — done: `opencode` 1.18.29 installed into a contained prefix
  (`~/.local/opt/opencode`), matching `nobara-pc`'s own install, recorded in
  `docs/task-first-hour.md` §1.
- **Item 4 (cold run 2)** — nice-to-have here per the brief (Run 2 is specified for the Linux
  box); not attempted.
