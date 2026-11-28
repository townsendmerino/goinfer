#!/usr/bin/env bash
# The one red of v0.26.0's darwin `gate quick -full` record at bfce8996: the backend/webgpu/internal/towers cell died with a
# SIGSEGV in the garbage collector while TestGridVisionWebGPU_matchesCPU/glm-ocr was inside the test helper makeWritable.
# Three readings at the same commit and toolchain. The tree is not edited: `go test -overlay` supplies the two test files.
#   rerun   the cell as the gate ran it, RERUNS times
#   plant   the Metal backend's regression test for this helper (TestMakeWritable_heapSlicesUnderGC, verbatim), added to
#           this package and run against this package's helper as it is
#   fixed   the same test, and then the cell's own tests, with the helper in the Metal backend's form (mprotect by integers)
#   WT=<a clean tree at the commit, with a go.work and the fixtures> OUT=<durable dir> [RERUNS=10] bash run-classify.sh
set -uo pipefail
: "${WT:?WT: the tree}" "${OUT:?OUT: a durable directory}"
RERUNS=${RERUNS:-10}
export PATH=/usr/local/go/bin:$HOME/go/bin:/usr/local/bin:$PATH
PKG=github.com/townsendmerino/goinfer/backend/webgpu/internal/towers
DIR=$WT/backend/webgpu/internal/towers
mkdir -p "$OUT/overlay"
cd "$WT/backend/webgpu" || exit 2
GOTOOLCHAIN=$(awk '$1=="toolchain"{print $2}' "$WT/go.mod"); export GOTOOLCHAIN
{ echo "tree:    $(git -C "$WT" rev-parse HEAD) ($(git -C "$WT" status --porcelain | wc -l | tr -d ' ') git status entries)"
  echo "go:      $(go version)"; echo "machine: $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion), page size $(pagesize)"
  echo "started: $(date '+%F %T %Z')"; } | tee "$OUT/provenance.txt"

# The planted test: the Metal backend's, verbatim from its comment to the end of the file, under this package's build line.
{ printf '//go:build gpu && goinfer_testhooks && unix\n\npackage towers\n\nimport (\n\t"runtime"\n\t"sync/atomic"\n\t"testing"\n)\n\n'
  sed -n '/^\/\/ TestMakeWritable_heapSlicesUnderGC/,$p' "$WT/backend/metal/internal/towers/writable_test.go"; } > "$OUT/overlay/zz_writable_gc_test.go"
# The helper in the Metal backend's form: its function, verbatim, under this package's build line.
{ printf '//go:build gpu && goinfer_testhooks && unix\n\npackage towers\n\nimport (\n\t"fmt"\n\t"runtime"\n\t"syscall"\n\t"unsafe"\n)\n\n'
  sed -n '/^func makeWritable/,/^}/p' "$WT/backend/metal/internal/towers/writable_test.go"; } > "$OUT/overlay/writable_unix_test.go"
gofmt -l "$OUT/overlay" | sed 's/^/gofmt would change: /'
printf '{"Replace": {"%s": "%s"}}\n' "$DIR/zz_writable_gc_test.go" "$OUT/overlay/zz_writable_gc_test.go" > "$OUT/overlay/plant.json"
printf '{"Replace": {"%s": "%s", "%s": "%s"}}\n' "$DIR/zz_writable_gc_test.go" "$OUT/overlay/zz_writable_gc_test.go" \
  "$DIR/writable_unix_test.go" "$OUT/overlay/writable_unix_test.go" > "$OUT/overlay/fixed.json"

verdict() { # <log>: what one `go test -v` run ended as
  if grep -q -E '^(ok|PASS)' "$1" && ! grep -q -E '^--- FAIL|^FAIL|SIGSEGV|fatal error' "$1"; then echo "ok ($(grep -c -E '^\s*--- PASS' "$1") PASS lines, $(grep -c -E '^\s*--- SKIP' "$1") SKIP)"
  else echo "DIED: $(grep -m1 -E 'SIGSEGV|fatal error|^--- FAIL|panic:' "$1" | cut -c1-110)"; fi; }

echo "== rerun: the cell as the gate ran it, $RERUNS times"
ok=0
for i in $(seq 1 "$RERUNS"); do
  go test -v -count=1 -tags 'gpu goinfer_testhooks' -timeout 45m -short "$PKG" > "$OUT/rerun-$i.log" 2>&1
  v=$(verdict "$OUT/rerun-$i.log"); echo "  rerun $i ($(date '+%T')): $v"; case "$v" in ok*) ok=$((ok+1));; esac
done
echo "  rerun: $ok of $RERUNS runs ok"

echo "== plant: the Metal backend's GC test against this package's helper as it is"
go test -v -count=1 -tags 'gpu goinfer_testhooks' -timeout 10m -overlay "$OUT/overlay/plant.json" -run '^TestMakeWritable_heapSlicesUnderGC$' "$PKG" > "$OUT/plant.log" 2>&1
echo "  plant ($(date '+%T')): $(verdict "$OUT/plant.log")"

echo "== fixed: the same test with the helper in the Metal backend's form, then the cell's own tests with it"
go test -v -count=1 -tags 'gpu goinfer_testhooks' -timeout 10m -overlay "$OUT/overlay/fixed.json" -run '^TestMakeWritable_heapSlicesUnderGC$' "$PKG" > "$OUT/fixed-gc.log" 2>&1
echo "  fixed, the GC test ($(date '+%T')): $(verdict "$OUT/fixed-gc.log")"
go test -v -count=1 -tags 'gpu goinfer_testhooks' -timeout 45m -short -overlay "$OUT/overlay/fixed.json" "$PKG" > "$OUT/fixed-cell.log" 2>&1
echo "  fixed, the cell ($(date '+%T')): $(verdict "$OUT/fixed-cell.log")"
echo "finished: $(date '+%F %T %Z'); tree after: $(git -C "$WT" status --porcelain | wc -l | tr -d ' ') git status entries" | tee -a "$OUT/provenance.txt"
