# Post-tag gate fixes, PR 1: the Metal lint targets and the release rows' module list (2026-11-05, nobara-sb01)

Asked for by the coordinator after v0.24.0 was tagged: the four Metal lint specs name `darwin/arm64`, and `gate release`'s rows 2 (module pins) and 8
(gofmt) read the backend/ paths.

## What was wrong

- `cmd/gate/quick_select.go`: backend/metal's four lint steps (build, vet, vet with `goinfer_testhooks`, staticcheck) had Target `""`, the host's. On
  Linux they built a darwin-only module natively and failed: `backend/metal/internal/gpudev/copy_coalesce.go:32:24: undefined: DeviceCopy`
  (`metal-lint-by-hand.log`, last command). They failed in every Linux `quick -full`, including v0.24.0's (nobara, 2026-11-04, accepted for that tag).
- `cmd/gate/grade_release.go`: `relModules` still listed `gpu`, `cuda` and `metal`, which the layout moved to `backend/`. Rows 2 and 8 would have read
  directories that do not exist. `TestRowPins` and `TestRowGofmtAndLint` build their fixtures from `relModules`, so they passed on the stale list.
  The list also lacked `backend/webgpu` under its new name.

## Red, then green

- `red.log`: the two new tests at 7f19e161c, both failing for the reasons above. `TestQuickModules_nativeOnlyModulesLintOnTheirTarget` fails on the four
  Metal steps; `TestRelModules_areTheTaggedModulesOfTheTree` reads the tracked go.mod files and compares them with `relModules`.
- The fix: the four Metal specs take `"darwin/arm64"`; `relModules` is `., backend/webgpu, backend/cuda, backend/metal, demo/agent`; the row 2 message
  names `backend/{cuda,metal}/internal/gpudev`.
- A recorded release run replays through the new list: `testdata/gate/goldens/release-replay` was captured before the layout move and holds the modules'
  go.mod under `gpu/`, `cuda/` and `metal/`. `readMoved` and `cmdMoved` (grade_release.go) read the pre-move name once when the backend/ path is absent
  from a record, the way `objectAt` already does for git objects. One line of that golden's `live.txt` follows (row 2's record line names the new
  paths); `case.json`'s note says so. `TestReleaseReplay_plants` expects the new names.
- `green.log`: `python3 scripts/touched.py`, 12 checks passed, including the whole `cmd/gate` package.

## Evidence the Metal steps now pass on Linux

`metal-lint-by-hand.log`: the four steps run as `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0` (what the quick cells now run) on linux/amd64 in a worktree with
`go.work`: build, vet, vet with the testhooks tag and `staticcheck -tags goinfer_testhooks` (staticcheck 2026.2.1 (0.8.1), the pinned build) all rc=0.
The same build with no target: rc=1, the error above. The `gate quick` cell logs of a run with `-files backend/metal/alias.go` name the four cells
with `(darwin/arm64)` and show no output.

## Not done here

PR 2 (test fixes) and PR 3 (`build_ptx.sh`) are separate, as asked. The `Mirrors` text of the lint specs still says `./metal/...` and `./cuda/...`;
that is a label in the report, not in scope.
