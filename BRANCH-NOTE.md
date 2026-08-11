# `flag-pair-moe-cache` — parked, waiting on the `6edd1ca` core-numerics freeze

## What is here

`--moe-cache-experts` and `--moe-cache-slots` on `serve` (decisions 2 and 3): the env-var-only
expert-cache controls promoted to real CLI flags, wired `decoder.Options` → `Model` accessor →
CUDA backend, following the `KVPrecision` pattern rather than adding more `os.Getenv` to the
backend. Both env vars stay honoured, so nothing existing breaks.

## What it is waiting on

The `Options` fields and accessors touch **`decoder/model.go`** and **`decoder/gguf.go`**, both
inside the `deps_hash` scope, so `TestParityManifest_fresh` reports **19 families stale**. The
change adds struct fields and accessor methods and alters no forward-path arithmetic — but a
`deps_hash` refresh is not a re-validation, and the freeze is in force.

Precedent exists for a goldens-gated refresh on a metadata-only addition (`9e5f8fa`, where a
metadata field re-staled `weights.go` and the refresh ran 19 goldens). It was deliberately not
spent here: the user-visible half of this work — the slot default, which was costing ~3× decode
rate to anyone who enabled expert caching — was CUDA-only and landed on `main` at `7ccec1e`
without touching the frozen surface. Spending the precedent on flag ergonomics when the defect
could be fixed without it is the wrong trade.

## To pick this up

1. Confirm the freeze has lifted, or that a goldens-gated refresh is authorised.
2. `git rebase main`. Expect a conflict in `cuda/backend.go`: `main` has the same slot-default
   hunk reading the env var directly; this branch replaces it with `m.MoECacheSlotsRequest()` /
   `m.MoECacheExperts()`. Keep the branch's accessor calls, keep `main`'s comment.
3. `go test ./decoder -run ParityManifest -update`, then **run the goldens** — not just the hash
   refresh. That is the whole point of the tripwire.
4. Re-run the Gemma-4 MoE gates (`-run Gemma4MoE`); they were green here before parking.
5. Delete this file in the merge commit.

Do NOT refresh `deps_hash` to quiet the gate without running the goldens behind it.
