# Task (goinfer): promote `qwen3_next` from tiny-oracle to a real-checkpoint T3 row

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.

> **CLOSED 2026-09-13.** The work order ran, on `nobara-pc`, starting the same day it was written
> (`3b7facd` .. `8f003f2`, 2026-08-26/27). `qwen3_next` cleared T3: `testdata/parity_manifest.json`
> shows `status: "validated"`, `method: "real-model-oracle"`, real `Qwen3-Next-80B-A3B-Instruct`
> (int4 weights, f32 activations) against a full HF bf16 reference via `accelerate` disk offload —
> argmax 100.0%, logit cosine 0.98988. `docs/capability-matrix.md`'s Qwen3-Next row reads
> `real-oracle 100.0%/0.98988`, not `experimental: tiny-oracle`.
>
> The shared-path proxy was checked first and correctly rejected (own-set diverges from
> `qwen3_5_moe` by six files). The first run (`dc35a31`) measured cosine 0.989876 and initially
> read as a fail against the 0.99 gate — but that gate was calibrated for int8 and this was the
> repo's first int4 T3 row. Two separate defects, not one: the realckpt gate's `-run` filter
> couldn't reach the test at all (`20fc6e6`, `...Real_oracle` matched neither `Qwen35` nor
> `Real_gate`), and the bar itself needed a per-precision split (`c62f2b7`: int8/int8int8 keep 0.99,
> int4 takes 0.98, pre-registered before the fix was applied). With both fixed, the measured
> 0.989876 clears 0.98 and the row was written by the gate (`8f003f2`), never by hand. A broader
> question — whether the bar also needs a sparsity axis, not just precision — was filed as `G25` in
> `docs/QUEUE.md` and parked; it does not block this family's validated status. Full record:
> `docs/measurements/qwen3next-t3-int4-2026-08-26.md` and the archived sweep logs beside it.

> **For:** Claude Code, in `~/mycode/goinfer` on **`nobara-pc`** (the linux/CUDA box — note the
> path is `mycode`, not `tmcode`; that mistake has already been made once). Written 2026-08-26.
> **Box: linux, and it must be.** T3 is the "on the big box" tier, the reference needs host RAM the
> mac does not have, and the resident half needs the RTX 2070 SUPER.
>
> **This is not implementation work. Qwen3-Next already runs.** It is registered
> (`decoder/registry.go`), it has its own forward, and it is GPU-resident on WebGPU and CUDA as of
> v0.15.0. The deliverable is *evidence*, not a feature.

## Why this is worth a box-day

Community interest in Qwen3-Next is high, and a launch that claims support draws exactly the
audience that downloads it and checks. Our current claim rests on a **tiny synthetic oracle**:

```
qwen3_next  status: experimental   method: tiny-golden   machine: mac-m1pro
            reference: HF f32 tiny golden (transformers 5.15.0) + a REAL-WEIGHT 4-layer slice
            metrics: argmax 100.0%, cosine_min 1.0, cosine_mean 1.0
```

Those numbers are perfect and they are honest — for what they cover. What they do **not** cover is
a real released checkpoint end to end, which is what `docs/parity-coverage-policy.md` requires of a
family claimed as supported. Shipping a headline in front of a tiny-oracle row is the shape of claim
this repo has retracted before.

It is also **invisible**: zero mentions in `README.md`, zero rows in `docs/benchmarks.md`. Fixing
the evidence first means the visibility work can then quote something real.

## What T3 means here (read the policy, do not infer it)

`docs/parity-coverage-policy.md` § "The three tiers": T3 is *"full argmax-exact + logit-cosine vs
the HF bf16 reference on the real checkpoint; when model + reference won't co-reside (the 35B-A3B
case), `weightDiff` GGUF-vs-safetensors is the substitute proof."*

**Valid `method` values — pick the one the run actually earns, not the strongest-sounding:**
`full-forward-oracle`, `real-model-oracle` (int8-resident vs bf16 reference), `weightDiff`
(+ optional layer-slice), or `shared-path (via <family>)`.

**`shared-path` is worth checking BEFORE downloading anything.** `qwen3_next`'s manifest entry says
`uses: [core, loaders, quant, moe, deltanet]` and its `own:` list includes
`decoder/forward_qwen35.go` — shared with `qwen3_5_moe`. If its numerics surface is genuinely
covered by a family that already has a real-checkpoint T3, the policy's own § on proxies may let it
clear T3 as `shared-path (via qwen3_5_moe)` without a separate 80B download. **Read that section and
decide deliberately** — if it qualifies, say so and stop; if `own:` files put it outside the proxy
rule, proceed. Either outcome is a result worth recording; only an unexamined download is waste.

## Steps, if a real run is needed

0. **Check the box is idle and record it.** `docs/benchmarks.md`'s methodology now requires machine
   state beside any timing-sensitive number, after a contaminated measurement survived into three
   documents this week.
1. **Get the checkpoint into `~/models` on the box.** `models-pull <name>`. **Never measure or
   validate from `/srv/models`** — it is local, which is exactly why the prohibition is easy to
   rationalise past, and it is a 5400 rpm SMR disk. A row whose model path starts `/srv/models` is
   void.
2. **Pick the largest variant that fits the proof you can actually run.** If model + bf16 reference
   cannot co-reside, that is not a failure — it is the documented `weightDiff` case, and the 35B-A3B
   precedent shows how it was handled.
3. **Run the T3 gate**: `go run ./cmd/gate parity` (see its env: `REALCKPT`, `EMIT_MANIFEST`,
   `TIMEOUT`). Read the SKIP list — *a skip is not a pass*, and `ok` in 0.02 s means nothing ran.
4. **Record the result in the manifest by MEASUREMENT, not by hand.** `EMIT_MANIFEST=1` folds
   measured `PARITY_ROW` lines in. **Never write metrics you did not measure** — the manifest's
   pending rows exist precisely because someone must run them.
5. **If it fails, that is the deliverable.** A family that cannot clear T3 gets demoted or scoped,
   with the measured reason. `docs/queue-release.md`'s E2 is the precedent: four "demotion
   judgments" turned out to be two real loader bugs, found only because a released checkpoint was
   actually loaded. A negative here is worth more than a green tiny-oracle.

## Gates

- The manifest row for `qwen3_next` shows `status: validated`, a T3-valid `method`, `machine:` the
  linux box, and metrics that came from the run.
- `go test ./decoder/ -run TestParityManifest_fresh` green afterwards.
- `docs/capability-matrix.md`'s Qwen3-Next row no longer reads `experimental: tiny-oracle`.
- The run's log archived under `docs/measurements/` — a verdict nobody can re-read is not evidence.

## Explicitly NOT in scope

- Benchmark tok/s rows (that is the next task, and it needs the re-anchor settled first — the box's
  peer rows were re-measured on 2026-08-26 after a Nobara 43→44 upgrade moved driver, kernel and
  libc, and ~20 STALE markers remain).
- README/launch copy. Write the evidence; the copy quotes it afterwards.
- Any change to `qwen3_next`'s forward. If the T3 run finds a numerics bug, STOP and file it — a fix
  and its own validation are separate work, and this task's whole point is that the evidence came
  before the claim.
