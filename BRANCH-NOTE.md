# `flag-pair-moe-cache` — SHIPPED (N-40)

> **This note is a historical record, not open work.** `--moe-cache-experts` and
> `--moe-cache-slots` are real `serve` flags today (see `internal/serveapp/main.go`). The note
> was left at the root reading as an in-flight branch, which is what audit N-40 flagged: a
> stale process artifact in the most visible place in the repo. Kept because the design read it
> records is still the reference for the cap derivation; retitled so nobody picks it up as a
> TODO.

## Original title

`flag-pair-moe-cache` — unblocked; re-derived against the corrected cap (2026-08-12)

## What is here

`--moe-cache-experts` and `--moe-cache-slots` on `serve` (decisions 2 and 3): the env-var-only
expert-cache controls promoted to real CLI flags, wired `decoder.Options` → `Model` accessor →
CUDA backend, following the `KVPrecision` pattern rather than adding more `os.Getenv` to the
backend. Both env vars stay honoured, so nothing existing breaks.

## What it is waiting on

**Nothing.** The `6edd1ca` freeze was re-declared as a **proof requirement** (`cda8cfe`): a change to
a path in `testdata/parity_manifest.json` needs a goldens run whose axis composition is printed with
the result. The `Options` fields and accessors touch `decoder/model.go` and `decoder/gguf.go`, so this
re-stales 19 families — that is the price, not a prohibition, and `scripts/refresh_parity_hashes.sh`
is the authorised path. The goldens now span f32, int4, int8 and int8int8 across both loaders.

The old paragraph here cited `9e5f8fa` as the precedent for a goldens-gated refresh. **That was the
wrong commit** — it is `fix(quant): reject --quant that conflicts with a prequant .giw` and touches
the manifest not at all. The real precedents are nine commits between 2026-07-26 and 2026-08-09.

## What CHANGED under it while this branch was parked

`main` fixed the thing this branch was written beside, and the flag's meaning moved with it:

- **A5 (`6091e7a`)** made the cap a search over the driver's 2 MiB allocation granularity. The old
  cap was a division over a raw byte sum and returned a value that allocated and then could not
  launch.
- **A9-FIX (`0103b49`)** pays the deferred first-launch reservation before the cap is computed.
- `7ccec1e`, which this branch is based on, was **reverted** at `97ee663`.

**So `--moe-cache-slots` no longer means "how you get a working cache". It means "request no more
than N", and the cap may still lower it** — the `C′ cache: … capping to N` log line says what was
chosen. Write the flag help that way; the old framing described a workaround for a cap that could not
be trusted, and that cap is gone.

## To pick this up

1. `git rebase main`. **The conflict guidance that used to be here is stale** — it said to keep
   `main`'s comment in the slot-default hunk of `cuda/backend.go`, and that hunk has been rewritten
   three times since (`7ccec1e` reverted at `97ee663`, then A5 `6091e7a`, then A9-FIX `0103b49`).
   Re-read the current `cuda/backend.go` and keep the branch's accessor calls over whatever is there.
   `decoder/model.go` and `decoder/gguf.go` were untouched by A5/A9-FIX/P6/P7, so they should be clean.
2. Rewrite the flag help for the corrected cap (see above) — this is the substantive part, not the
   rebase.
3. `bash scripts/refresh_parity_hashes.sh` — it runs the goldens and refreshes `deps_hash` only,
   refusing on any golden failure and on a vacuous all-skipped run. Do not `-update` the manifest by
   hand.
4. `python3 scripts/queue_citation_lint.py` before pushing: this file now carries checked citations.
