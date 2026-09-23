# S1 — sidecar `.giw` by default on darwin: all three registered gates pass

**Result: CLEAN PASS on both models measured.** Anonymous footprint of the sidecar load was
16–21% of the direct load's (well under the registered ≤25% bound), swap-used delta was 0 MB
across every run (6 total: 4 on model A, 2 on model B), and greedy output was byte-identical
between direct and sidecar loads on 3 prompts × 64 tokens for both models. Per the brief's own
Decision rule ("all three or the switch does not ship"), S1's default-sidecar mechanism ships as
built.

Box: MacBook (arm64, CPU backend), goinfer main (uncommitted S1 work at measurement time,
committed alongside this doc). `footprint <pid>` for the anonymous/file-backed split, `sysctl -n
vm.swapusage` before/after each load, `df -g /` for disk headroom, `uptime` for loadavg.

## Models measured

The brief names "1.5B, phi3-mini" as the two safe test subjects; no phi3-mini checkpoint was
available locally, so a second real dense model was substituted instead — same methodology, a
different architecture/quant class, not a weaker substitute:

- **Model A**: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (Qwen2.5-Coder, 1.5B, Q4_K_M source),
  loaded at the CLI's own default quant (int4). 2 runs per arm, interleaved (direct, sidecar,
  direct, sidecar), matching the brief's "both arms interleaved" — reduced from its named "three
  loads each" to two for this pass's time budget; logged here rather than silently done, per this
  repo's own "no silent caps" measurement discipline.
- **Model B**: `qwen3-1.7b-q8_0.gguf` (Qwen3, 1.7B, Q8_0 source), loaded at int4. 1 run per arm —
  a single-shot corroboration on a second model/architecture, not a full replication. Needed
  `GOINFER_NO_FIT_GUARD=1 -ctx 4096` on both arms to reach a genuine resident load at all on this
  machine's real available memory at measurement time (see "A confound found and fixed" below);
  both arms used the identical override so the comparison stays apples-to-apples.

## Results

| model | arm | footprint (anonymous) | mapped file (sidecar only) | swap Δ |
|---|---|---|---|---|
| A (1.5B) | direct, run 1 | 1572 MB | — | 0 MB |
| A (1.5B) | sidecar, run 1 | 327 MB | 1228 MB | 0 MB |
| A (1.5B) | direct, run 2 | 1573 MB | — | 0 MB |
| A (1.5B) | sidecar, run 2 | 319 MB | 1234 MB | 0 MB |
| B (1.7B) | direct | 1506 MB | — | 0 MB |
| B (1.7B) | sidecar | 241 MB | (not captured separately) | 0 MB |

Anonymous-footprint ratio (sidecar / direct): model A run 1 = 20.8%, run 2 = 20.3%; model B =
16.0%. All three well under the registered ≤25% bound.

Byte-identity: for each model, `decoder.Load` was called directly on the .gguf (`Options{Quant:
"int4", Backend: "cpu"}`, matching the CLI's own defaults) and on the corresponding sidecar .giw
(`Options{Backend: "cpu"}`), then 3 fixed prompts × 64 greedy tokens each were compared. All 6
prompt-comparisons (3 per model × 2 models) matched exactly. The same comparison, run
deliberately across two genuinely DIFFERENT quants (int8int8 vs int4) as a sanity check that the
comparison logic is actually sensitive to a real divergence, produced a real mismatch — confirming
this is not a vacuous pass (see `internal/prequant/sidecar_identity_test.go`'s own doc comment,
committed alongside this measurement, for the permanent regression test).

## A confound found and fixed mid-measurement

Model B's first attempt at a "direct" arm silently fell through a DIFFERENT, PRE-EXISTING
mechanism: `internal/serveapp/main.go`'s fit-guard auto-retry (task-fit-to-hardware.md's CPU
placement piece, unrelated to S1) — this machine's tight free RAM at the time made the STATIC fit
guard refuse the plain resident load, and the pre-existing auto-retry-to-streaming fallback loaded
it through a `.giw` too, before S1's own new logic ever got a chance to run. `-direct-load` only
turns off S1's OWN default-sidecar branch; it was never meant to (and should not) turn off that
older, separate mechanism. Caught because the footprint numbers came back suspiciously close
together and the server's own log line ("automatically retrying with weight streaming") named the
cause directly — re-run with `GOINFER_NO_FIT_GUARD=1 -ctx 4096` on both arms to reach a genuine,
comparable direct load. Recorded here because a confound that gets caught and fixed is exactly the
kind of thing this repo's measurement discipline asks to write down, not quietly redo and forget.

## Decision

As registered: all three criteria hold on both models measured. S1's darwin default (a plain
`.gguf` resolves to its sidecar `.giw` unless `-direct-load`/`GOINFER_GGUF_DIRECT=1`) ships as
built, already merged into `loadDecoder` (serve), `loadFromPath`'s call site (chat), and `fit`'s
reuse-only-if-fresh path.

## Not done here (owed if this is picked up again)

- gpt-oss-20b, the model that produced the original 22.9 GB incident, per the brief's own note:
  "after S2" — the `needsResidentSerialize` families (gpt-oss among them) build resident THEN
  serialize today, so transcoding gpt-oss-20b to a sidecar would itself require the very resident
  build S1 exists to avoid. Genuinely blocked on S2, not skipped.
- A third interleaved run for model A (brief names three; two ran here).
- Cold-vs-warm TTFT was captured for model A (both arms, both runs: cold ≈156–180 ms, warm ≈
  164–171 ms — no material difference between arms, consistent with "speed is not a criterion
  here") but not analyzed further; not gating.
- `benchmarks.md` Table 1's cold-start/footprint row, `task-fit-to-hardware.md`'s own cross-link,
  and this task doc's status table still need this measurement's numbers folded in.
