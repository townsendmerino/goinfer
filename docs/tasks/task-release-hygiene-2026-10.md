# Release hygiene, 2026-10: three changes that remove a manual read from a tag

Status 2026-10-11. Item 1 is built and merged with this file. Items 2 and 3 are planned here and not started.
The three came out of the library-surface review's release-readiness notes. They share one purpose: a drift
between a thing that changed and the sentence that says so is found by a command, not by someone reading.

## 1. The sidecar's file name carries the format version (BUILT)

**The cost it removes.** A sidecar was `<model>.<quant>[.e4h].<target>.giw`, one name for every build. A build that
reads only weights format vN refuses a v(N+1) file as too new and replaces it; the other build then does the same. A
release binary and a dev build on one box (the normal state of a developer, and of anyone trying a release candidate)
therefore transcoded the model each time they alternated: about 12 minutes and 17 GB written for a 30B.

**The change.** `internal/prequant.streamCachePath` names the file `<model>.<quant>[.e4h].<target>.v<N>.giw`, N being
`decoder.GIWFormatVersion` (a new exported constant for `giwVersion`). `EnsureCachedGIW` and `SidecarPathIfFresh` (which
`fit` uses) go through `freshCache`:

- the versioned file is fresh: use it;
- else an unversioned file under the old name that is newer than the source and loads as this build's format is
  **hard-linked** to the versioned name (`adoptLegacy`): no copy, no transcode, the verified-marker linked beside it;
  the old name stays, because a build from before the change still reads and rewrites it (by temp and rename, which
  leaves the link alone);
- else transcode to the versioned name, and say that an unversioned file is being left behind.

**What it deliberately does not do: delete.** An old-version sidecar may be the other build's only copy, and deleting it
brings back the ping-pong. The price is disk: one extra model-sized file per model per format bump. The CHANGELOG and
`docs/giw-bundles.md` ("One version") say so. A sweep that lists the stale ones is item 1b below.

**Tests** (`internal/prequant`): the name carries the version and the file holds that version; an unversioned file in
this format is adopted as a hard link with no transcode, and `SidecarPathIfFresh` finds it; with another build's file
sitting under the old name, this build neither transcodes nor touches it; an unversioned file in another format is left
alone, one transcode happens, and the note appears. Removing the link makes the adoption test fail. `fit`'s and
`modelload`'s tests moved to the versioned name. The queued night script `run-gs10q-d-night.sh` accepts either name.

**Not covered here.** Windows (hard links on NTFS work for files in one volume; untested). A sidecar on a filesystem
that refuses links (some network mounts) falls back to a transcode, silently, as before.

**1b, not built, owner's call: list the sidecars of versions this build does not read.** `pull`'s cache listing already
sums sidecars per model; add a "stale format" column (read the version from each file's header with
`giw.WeightsVersionFile`) and a `goinfer-pull prune-sidecars` that deletes the unreadable ones after listing them. About
half a day. Worth it once a second format bump has left a second copy on someone's disk.

## 2. A CHANGELOG drift check (BUILT 2026-10-11, aikit's first)

**The problem.** The release notes are written by hand from memory of the diff. The docs already have checks that fail
when a table disagrees with the code (`TestFlagsDoc`, the support table, the env-var list); the CHANGELOG has none.
Four kinds of change are user-visible, enumerable, and have been missed: a route, a flag, a family, and the aikit pin.

**The command.** `go run ./cmd/gate changelog [-base vX.Y.Z]`, exit 0 clean, 1 with a list of what has no line, 2 refused
(no tag, no CHANGELOG section). `-base` defaults to the newest tag reachable from HEAD. It compares the tree at `-base`
with the working tree, reading the base side with `git show <tag>:<path>` (no checkout), and checks the text of the
CHANGELOG's `[Unreleased]` section, or the `[vX]` section when the tag is being cut (`-section`).

| kind | read from, at both revs | owed when |
|---|---|---|
| route | the route literals `serverBanner` can print and the `mux` registrations in `internal/serveapp` (extracted by a small `go/ast` pass, not a regex over text, so a commented-out route does not count) | a route exists at HEAD and not at base, and its path is not a substring of the section |
| flag | the names in `docs/flags.meta.json` | a name is new, or removed, and `--name` is not a substring of the section |
| family | the `model_type` keys of `docs/capability-matrix.json` | a family is new and its name is not a substring of the section |
| aikit pin | the `aikit` require in `go.mod` | the version string differs and the section does not contain the new one |

**Design decisions to make before building**
1. *Substring match is the whole test.* It cannot tell that a mention is about the right thing, only that the string is
   there. That is the right strength: it catches the omission (which is what happens) and not the careless sentence
   (which no tool catches). A flag mentioned only in a table of removed flags would pass; accepted.
2. *Waivers.* Some pin bumps and some internal routes (`/admin/...`, health) are not release-note material. A
   `docs/changelog-waivers.txt`, one `kind name reason` per line, read by the check and printed by it, so a waiver is a
   committed sentence and not a flag. The reason is mandatory.
3. *Where it runs.* By hand and inside `gate release` (item 3); in CI as a job on tag pushes only (`release-assets.yml`
   already runs `TestParity_noPendingGateOutlivesARelease` as a backstop and this is the same shape). Not on every push:
   between releases `[Unreleased]` is allowed to be behind.

**Tests.** A throwaway git repo built in the test (two commits, a tag between them, a route, a flag, a family and a pin
each added in the second): all four owed; add the CHANGELOG line for each and the owed set shrinks by exactly that one;
a waiver clears one and prints its reason. **A gate that cannot go red proves nothing** (`cmd/gate/mutation.go`): the test
that removes one CHANGELOG line and expects red is the mutation proof, run through `gate mutation`.

**Cost.** About a day: the `go/ast` route pass is the only part with real work, and the base-side read is a few lines
over `git show`. **Risk.** A route registered through a helper the pass does not follow reads as "no routes", so the first
step is to assert the pass finds a known count at HEAD (today's), the same guard `TestPipelineLint_boundKernelsAreLaunched`
uses against an empty scan.

**Built as planned, with these particulars.** `gate changelog [-base TAG] [-head REV] [-section S]` in `cmd/gate/changelog.go`. Routes are the
first string literal of every `HandleFunc` or `Handle` call in the non-test files of `internal/serveapp`, found with `go/parser`
(a commented-out route and a registration in a test file do not count; the scan refuses to run if it finds fewer than 10 routes in
either tree, and a test asserts at least 30 at HEAD, so a refactor cannot make it read "nothing changed"). A route is covered by
its own path, or by a new route that is a path prefix of it. Flags come from `docs/flags.meta.json` (new and removed), families
from `docs/capability-matrix.json` (covered by the name, the display name or a model type), the pin from `go.mod`. Waivers in
`docs/changelog-waivers.txt`, reason mandatory. **On the first real run against v0.22.0 it found four omissions in `[Unreleased]`:
the `POST /v1/audio/transcriptions` route, `--windowed-kv` (a default flipped on for CUDA with no line), and the `qwen3_asr` and
`qwen3_vl_moe` families.** Three CHANGELOG entries were written for them (speech to text, Qwen3-VL MoE with its unread CUDA FAIL
stated, windowed KV with its agent-loop caveat) and the check reads clean. Backtest on v0.22.0 against v0.21.0: clean (older tags
predate `docs/flags.meta.json`, so there is nothing to compare). **aikit first:** the same rule for exported API was built into
aikit's `releasegate` as a sixth check ("changelog-api", `tools/releasegate/apiadditions.go`), run against v1.60.0 to v1.68.0: one real
omission (`VoxtralDefectCountForTest`, a test seam, now exempt by its `ForTest` name), every other release clean, and removing
`SetDevice` from v1.69.0's section turns the gate red.

## 3. `go run ./cmd/gate release vX.Y.Z` (PLANNED)

**The problem.** `RELEASING.md`'s pre-flight is eight reads, in prose, each pointing at a different place. v0.21.0 and
v0.22.0 shipped without a recorded CUDA gate run: read 8 was prose, nothing counted it.

**The command.** One run, one line per read: `done` or `OWED` with the record path that proves it or that is missing.
Exit 0 only when every line is done, 1 when anything is owed, 2 when refused (not on `main`, tree dirty, no tag argument,
tag exists). It runs nothing long: it reads records. Where a read already has a function (the pending-gate check, the pin
agreement test, the hardware matrix test) it calls that function, so the command and the test cannot disagree.

| # | read | record or source | done when |
|---|---|---|---|
| 1 | no required gate pending since before the previous release | `cmd/gate` ledger and `awaitingFirstConfirmation` | the existing check passes for the tag |
| 2 | the five modules agree on `aikit`, and `cuda`/`metal` on `aikit/gpu` | each module's `go.mod` | the same agreement `TestAikitPinsAgree` asserts |
| 3 | a cold-user record for the **previous** tag | `docs/measurements/cold-user-<date>.md`, whose first lines name the tag it read | one exists and names it |
| 4 | the GPU gate, **both** backends, at a qualifying commit | a record each for cuda and metal (below) | each record's commit differs from HEAD by nothing under `metal/`, `cuda/`, `gpu/`, `decoder/` |
| 5 | the never-executed hardware list read | `docs/hardware-matrix.md`'s generated "Verified on" section against `docs/hardware-coverage.json` | the generated section is current and the diff against the previous tag's list is printed (new entries with no record, and entries that lost theirs) |
| 6 | no unmerged branch carries a fix | `git branch -r --no-merged origin/main` and local branches, minus patch-equal ones (the `get-status` skill's rule) | none has a commit whose subject starts with a fix marker, or the branch is named in a waiver |
| 7 | the CHANGELOG has a line for each user-visible change | item 2's `gate changelog` | exit 0 |
| 8 | the citation lint, `gofmt`, CI at HEAD | `scripts/queue_citation_lint.py`, `gh run list` at the full SHA, run-level `completed` | each green |

**The prerequisite, BUILT 2026-10-11: row 4's record.** `gate gpu` writes `docs/measurements/gpu-gate/<backend>-<time>-<commit>.json` on every run (`-record DIR`, empty for none): backend, full commit, verdict (PASS, FAIL, INCONCLUSIVE for green on a dirty tree, NO_GATE, VACUOUS, EMPTY), dirty, whether the heavy tier was skipped, the group and test counts, and the git tree id of `metal/`, `cuda/`, `gpu/` and `decoder/` at that commit. `gate gpu-records [-head REV]` reads them: the newest record made at the same code (those four trees identical to HEAD's) decides, whatever its verdict, and it counts only if it is a PASS on a clean tree with the heavy tier run; exit 0 only when cuda and metal both do. Tested against a git repository built in the test (a docs change keeps a run valid, a change under any of the four voids it, a later FAIL at the same code is not hidden by an earlier PASS, each non-qualifying kind of run is named); dropping `decoder` from the rule fails a test. **Not exercised end to end:** no real `gate gpu` run has written one yet, so the first CUDA night run is also the writer's first real use. `go.mod` pins are not in the qualifying set (RELEASING.md's rule names the four directories); an aikit bump between the run and the tag is read by row 2, not here. Both backends read OWED until each box has run the gate and its record is committed.

**Design decisions to make before building**
1. *Branch rule (row 6).* "Carries a fix" is a heuristic on commit subjects. It will miss a fix called "change". It
   errs toward listing, the output is a list for a person, and a wrong inclusion costs a waiver line.
2. *Offline.* Rows 8 (CI) and 6 (remote branches) need the network; without it they print `UNKNOWN`, never `done`, and the
   exit code is 1. A read that could not be made is owed.
3. *Records stay where they are.* The command reads the paths the prose already names; it does not move them.

**Tests.** Each row against a temp repo: a record present and absent; the GPU record at a commit that differs by a
`docs/` file only (qualifies) and by a `decoder/` file (does not); a cold-user record naming the wrong tag; a fix branch
that is patch-equal to main (not listed) and one that is not. The mutation proof as for item 2. **Cost.** A day and a
half with the GPU record; two rows (1, 2) are one-line calls into existing functions.

## Order

1 (done) → the GPU gate's record file (done) → 2 → 3, because 3 calls 2. Items
2 and 3 change nothing a user runs; they change what a release costs the person cutting it.
