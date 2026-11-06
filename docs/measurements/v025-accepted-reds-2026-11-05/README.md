# The accepted-reds file for `gate release` (2026-11-05, nobara-sb01)

Asked for by the coordinator (Francis approved it the morning of 2026-11-05): v0.24.0 shipped on reds he had read and accepted, and `gate release` had no way to say so, so its GPU-gate,
`quick -full` and parity rows read OWED for a release he had cleared.

## What it does

`docs/releases/<tag>-accepted.json` (schema 1, described in `RELEASING.md`): each entry names a failing gate record by file name, an item, a class, the evidence and who accepted it and when.
A row whose failing record has valid entries prints `accepted` with the entries, does not block the tag, and is counted apart from `done`. Everything else stays OWED, with the reason when an entry
was refused: another record, `accepted_by` other than Francis, a class that is not one, no evidence, no date, a file for another release, or a file that is not JSON. The rows are the GPU gate (per backend),
`quick -full` (per system) and parity. The replay of an old `gate release` record answers "no accepted file", so recorded reports do not change.

## Not in it

- No v0.24.0 file: that release is tagged, and the acceptance was relayed to me, not given to me; an entry carries Francis's name, so he or a commit citing his word adds it.
- No match of items to failures: the records carry counts, not names. The gate compares counts: a record is accepted only when its entries name at least as many distinct items as it has reds (see below), and the report shows the entries so the reader can match them.

## Red, then green

- `red.log`: `TestRowsAccepted` and `TestReportRelease_acceptedDoesNotBlock`, the rows as they were: OWED with the entry present, and exit 1.
- `green-new-tests.log`: the three new tests pass; `go test ./cmd/gate/` passes; `touched.py`: see `green-touched.log`.
- The coverage functions return the record that failed to count (`gpuCoverageR`, `parityCoverageR`, `quickCoverageR`); the old names are wrappers, so no caller changed.

## Review: one entry must not accept a record with two reds (coordinator, 2026-11-05 16:35)

The first version made a record `accepted` as soon as one valid entry named it, so a record with an accepted red and a new one would have read accepted. `coversReds` now requires as many distinct items as the
record's reds (GPU: failed groups + vacuous + empty cells; quick: failed tests; parity: blockers; one for a non-PASS record with none of those), and the row names how many are unlisted. Red first
(`red-two-reds.log`: one entry accepted records with 2, 4 and 7 reds; fewer-than-reds and a repeated item too), green after (`green-two-reds.log`, `go test ./cmd/gate/`).
**The limit:** the quick-full record counts failed tests (v0.24.0's read 4) while its red cells were 7 (lint steps and timeouts are not tests), so the count under-states those reds; the gate cannot see what the record does
not hold. A future `quick -full` record that lists its failing cells would close that; it is not done here.

