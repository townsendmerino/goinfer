# MacBook task: verify the Metal half of `gate gpu` against the script it replaced

> **STATUS: DONE 2026-08-21. Do not re-run this as a task.** The Metal half agrees with the retired
> script to the CUDA half's standard — identical verdict and exit code, 7 declared → 7 reported,
> 6 pass / 2 skip / 1 fail, every verdict line matching including the evidence lines. The Homebrew-bash
> dependency is confirmed gone. One bug was found IN THE RETIRED SCRIPT (`-darwin$` parsed as grep
> flags, so its CI-hygiene block reported a green having reproduced 0 of 8 checks); the port is
> correct at 8 pass / 14 skip. Full record: `docs/task-gate-runner.md` §12.
>
> **One loose end, deliberately not closed:** the INCONCLUSIVE verdict arm was read from source but
> never demonstrated live, because every fresh-clone attempt hit the pre-existing Metal `fault 0x10`
> crash (6 of 7 attempts that day). It needs no GPU and no dedicated run — `touch` a tracked file
> before any future `gate gpu` invocation and check for `INCONCLUSIVE` and rc 1.

## Why this is owed

E8 replaced six tallying shell/Python gates with one Go runner, `cmd/gate`. The last and largest was
`scripts/gpu_gate.sh` (715 lines), now `go run ./cmd/gate gpu`. **The CUDA half was verified on the
Linux box; the Metal half has never run.** No machine here has a Metal device, so its four groups —
`suite`, `cgofree`, `lifecycle`, `prefill` — are code review and not evidence. Until this task is
done, the release ritual's Mac step rests on a gate nobody has executed.

The standard the CUDA half met, so you know what "agreed" means:

| | shell | Go |
|---|---|---|
| check groups | 9 declared → 9 reported | identical |
| verdicts | 9 pass, 1 skip, 2 fail | identical |
| the 12 per-check verdict lines | — | identical |
| which tests failed | 2, by name | identical |
| final verdict + exit code | FAIL, rc 1 | identical |

## The method — and the one way to get it wrong

**Do NOT check out the old commit and run the script there.** Seven `metal/` commits have landed
since the script was deleted, so a two-checkout comparison measures those, not the gate. Extract the
script and run BOTH against the SAME (current) tree:

```sh
cd <goinfer>
git pull
git show 16a8084^:scripts/gpu_gate.sh > /tmp/gpu_gate.sh && chmod +x /tmp/gpu_gate.sh
```

`16a8084` is the commit that deleted it, so `16a8084^` is its last living version.

Then, on a quiet machine (close anything GPU-heavy — the gate's own header records a run poisoned by
leaked processes):

```sh
/tmp/gpu_gate.sh                    > /tmp/metal_sh.txt 2>&1 ; echo "rc=$?" >> /tmp/metal_sh.txt
go run ./cmd/gate gpu -logdir /tmp  > /tmp/metal_go.txt 2>&1 ; echo "rc=$?" >> /tmp/metal_go.txt
```

Run them **sequentially**, never concurrently: two Metal suites contending for the GPU is exactly the
condition that produces bogus numerics rather than an honest failure.

## Three prerequisites, each of which will otherwise waste your afternoon

1. **The old script needs Homebrew bash.** macOS ships `/bin/bash` 3.2, which cannot run `declare -A`
   — the script uses an associative array for its group accounting. If `bash --version` says 3.2,
   run it as `/opt/homebrew/bin/bash /tmp/gpu_gate.sh`. *This dependency is one of the things the
   migration removes: the Go binary has no such requirement. Worth confirming that claim while you
   are here.*
2. **`go.work` must include `./metal`.** It is untracked and per-machine, so yours may differ from
   the Linux box's (which lists `.`, `./cuda`, `./demo/agent`, `./gpu` — no `./metal`). Both the
   script and the runner do `go test ./metal/`, which needs the workspace to resolve a sibling
   module. Check with `go list ./metal/` before starting.
3. **One checkpoint, used by two groups:** `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`.
   Without it, `lifecycle` and `prefill` SKIP — and a skip is not a pass, so a run missing it does
   not verify those two. If it is absent, say so rather than reporting 7/7 green.

## What the four Metal groups assert

- **`suite`** — `go test -p 1 ./metal/ -count=1 -short`. The broad one.
- **`cgofree`** — `CGO_ENABLED=0 go build ./cmd/serve`. The premise: Metal is dlopen'd via
  purego-objc, so the binary must build with cgo off.
- **`lifecycle`** — `TestMetal_CloseFreesMemory|TestMetal_CloseWithSecondModelAlive`, run WITHOUT
  `-short` because they load a real model. Metal had the same hole CUDA did: `Close()` froze a
  channel and freed nothing, leaking ~267 MB per Load+Close on a 0.5B (`aacec89`). Both conditions
  matter — the sequential sawtooth, and a second model alive, which is the case that made CUDA's
  first fix look correct when it was not.
- **`prefill`** — `TestPrefillParity|TestPrefillNoNaN` under `GOINFER_METAL_MODEL`. `PrefillLast`,
  the f16 simdgroup_matrix TTFT path, emitted NaN logits at EVERY prompt length after the LM head was
  pinned to int8, because prefill still ran int8 head weights through the int4 `gemm_w4f16`
  (`19ef47d`). It hit the DENSE control, a model Metal ships, and nothing caught it until a hand-run.

## What agreement means, and what may legitimately differ

Compare, in this order: **the final verdict and exit code**, then **`N declared → M reported`**, then
**the `pass/skip/fail` counts**, then **each `PASS`/`FAIL`/`SKIP` line**, then the **skip-notes block**.

Expect these to differ and do not chase them:

- **Wall-clock figures** anywhere they appear.
- **Log paths** — the runner takes `-logdir` and writes `go test -json` streams; the script used
  `mktemp`. This is deliberate: the v1.0 gate's C1a note says a `mktemp` log is a verdict nobody can
  re-check.
- **Anything under a FAILING group.** The runner holds structured results, so its failure excerpts are
  attributable per test where the script grepped one flat log. Richer, not different.

Anything else that differs is a finding. In particular, on the CUDA half **one difference was my bug
and the shell was right**: `go test -v` does not indent a subtest's `=== RUN` line, so
`grep -cE '^=== RUN'` counts subtests and my first version did not. If you see a count that looks
low, suspect the runner before the script.

## If they disagree

Send the two transcripts and the diff. Do **not** "fix" the runner to match without saying which
side you believe is right — E8's rule is that the migration changes the SUBSTRATE, not what a gate
decides, so a disagreement is either a port bug (fix the Go) or a latent script bug the port exposed
(fix nothing, record it). The mutation checker's migration already produced one of the latter: the
script's `sed -i` was BSD-incompatible and had been broken on macOS its whole life, invisible because
nobody hand-ran it there.

## Also worth reporting, briefly

- Does `go run ./cmd/gate gpu` work under macOS's stock `/bin/bash` — i.e. is the Homebrew dependency
  genuinely gone?
- Is the verdict **PASS**, or **INCONCLUSIVE** (all green but a dirty tree)? The three-state verdict
  is new-ish and its dirty-tree arm should fire on an uncommitted checkout. Confirm both arms if you
  can — that is a two-minute check with a scratch edit.
- `go run ./cmd/gate -h` lists six other subcommands (`census`, `heavy`, `parity`, `composition`,
  `selector`, `mutation`). `census` and `composition` are cheap and Mac-relevant; if you have five
  spare minutes, `go run ./cmd/gate census` on the Mac would be the first Metal-side census since the
  migration.

## Declining is a valid outcome

If the Metal box is busy with the kernel-optimization campaign, say so and this waits — the CUDA half
is verified and the Metal half is unchanged behaviour in a new binary, not new behaviour. What is NOT
acceptable is leaving it ambiguous: the plan doc currently says, in its status line, that this half is
"code review, not evidence", and that sentence should either be deleted because you ran it or left
standing because nobody has.
