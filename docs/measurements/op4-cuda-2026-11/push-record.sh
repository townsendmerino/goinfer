#!/usr/bin/env bash
# OP4 on CUDA's push record (docs/tasks/task-forward-architecture-2026-10.md §4.9a, each step's U8): the checks one push of a
# step's branch runs, written to docs/measurements/op4-cuda-2026-11/<step>-push-record.log. It merges origin/main, and it pushes
# only when every check is green. Run it under nobara's timing lock, shared, from the step's worktree:
#
#   python3 scripts/timing_lock.py run --shared --label <step>-push -- bash docs/measurements/op4-cuda-2026-11/push-record.sh <step> <branch>
#
# FIXTURE_SRC (default ~/mycode/goinfer) is the checkout whose gitignored fixtures are placed in this worktree
# (scripts/worktree_fixtures.sh) before TestLoadedSublayers_matchTheProgram runs with OP4CU_REQUIRE_FIXTURES=1. So that test's
# PLE, Gemma 4 MoE and layer-scale plants, which reach only gitignored fixtures, run at each push of this series. CI has no
# device, and a checkout without the fixtures lists them NOT RUN.
set -u
step=$1 branch=$2
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
cd "$(git rev-parse --show-toplevel)"
M=docs/measurements/op4-cuda-2026-11
log=$M/$step-push-record.log
tmp=$(mktemp -d)
attr=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01C3vNEedTzt53SzZLQiNUmi'
trap 'rm -rf "$tmp"' EXIT
git fetch -q && git merge -q --no-edit origin/main -m "Merge origin/main into $branch$attr" || { echo "MERGE FAILED"; exit 1; }
H=$(git rev-parse --short=9 HEAD)
fail=0
{
	echo "# $step push record at $H (origin/main $(git rev-parse --short=9 origin/main) merged), $(date '+%Y-%m-%d %H:%M:%S %Z')"
	CW=$tmp/clean
	git worktree add -q --detach "$CW" "$H" && cp go.work "$CW/"
	echo "# touched.py --base origin/main, in a clean detached worktree of $H (no placed fixtures, as CI sees the tree)"
	(cd "$CW" && python3 scripts/touched.py --base origin/main 2>&1 | grep -vE '^\s+(cuda: |\[resident\] )' | tail -25; exit "${PIPESTATUS[0]}")
	rc=$?; echo "exit $rc"; [ $rc = 0 ] || fail=1
	git worktree remove --force "$CW"
	echo "# the gitignored fixtures placed from ${FIXTURE_SRC:-$HOME/mycode/goinfer}"
	bash scripts/worktree_fixtures.sh "${FIXTURE_SRC:-$HOME/mycode/goinfer}" "$PWD" > "$tmp/fixtures.log" 2>&1
	rc=$?; tail -2 "$tmp/fixtures.log"; echo "exit $rc"; [ $rc = 0 ] || fail=1
	echo "# TestLoadedSublayers_matchTheProgram and D2-1 with those fixtures required (OP4CU_REQUIRE_FIXTURES=1)"
	OP4CU_REQUIRE_FIXTURES=1 go test -tags 'cuda goinfer_testhooks' -run '^(TestLoadedSublayers_matchTheProgram|TestCUDAStep_tableEqualsTheFlagList)$' -count=1 -v ./cuda/ 2>&1 |
		grep -E '^(--- |ok|FAIL)|NOT RUN|plant |sb2_census_test.go:'
	rc=${PIPESTATUS[0]}; echo "exit $rc"; [ $rc = 0 ] || fail=1
	mkdir "$tmp/canary"
	printf 'module canary\n\ngo 1.27\n' > "$tmp/canary/go.mod"
	printf 'package canary\n\nimport "strings"\n\ntype s struct{ deadField int }\n\nfunc F() string { _ = s{}; return strings.ToUpper("x") }\n' > "$tmp/canary/c.go"
	echo "# staticcheck $(staticcheck -version): the canary must report U1000"
	(cd "$tmp/canary" && staticcheck ./... 2>&1) | tee "$tmp/canary.out"
	grep -q 'deadField is unused (U1000)' "$tmp/canary.out" || { echo "CANARY SILENT: staticcheck analysed nothing"; fail=1; }
	for args in "-tags cuda,goinfer_testhooks ./cuda/..." "./internal/program/ ./decoder/" "-tags goinfer_testhooks ./internal/program/ ./decoder/"; do
		echo "# staticcheck $args"; staticcheck $args 2>&1; rc=$?; echo "exit $rc"; [ $rc = 0 ] || fail=1
	done
	echo "# GOOS=darwin vet ./metal/... (untagged, then goinfer_testhooks)"
	GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go vet ./metal/... 2>&1; rc=$?; echo "exit $rc"; [ $rc = 0 ] || fail=1
	GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go vet -tags goinfer_testhooks ./metal/... 2>&1; rc=$?; echo "exit $rc"; [ $rc = 0 ] || fail=1
	echo "# citation lint"
	python3 scripts/queue_citation_lint.py > "$tmp/lint.txt" 2>&1; rc=$?; echo "exit $rc"; [ $rc = 0 ] || { tail -20 "$tmp/lint.txt"; fail=1; }
	echo "# fail=$fail"
} > "$tmp/record.log" 2>&1
cp "$tmp/record.log" "$log"
grep -E '^#|exit|VERDICT|FAIL' "$log"
[ $fail = 0 ] || { echo "NOT PUSHING"; exit 1; }
git add "$log" && git commit -q -m "$step: the push record (touched, the fixtures-required loaded-sublayer test, staticcheck, darwin vet, the citation lint), green$attr" &&
	git push -u origin "$branch" 2>&1 | tail -3
