#!/usr/bin/env bash
# The second page-size probe: long-lived small slices (probe_settled_test.go.txt says why the first probe cannot show them).
# Each test runs in its own process, RUNS times, for one second or until it dies, from one test binary per architecture:
# natively (darwin/arm64, 16 KiB pages) and as amd64 under Rosetta (4 KiB pages).
# It keeps each test's first log and the log of its first death, and counts the rest.
#   WORK=<scratch dir, outside any Go workspace> OUT=<this folder> [RUNS=10] GOTOOLCHAIN=<go.mod's toolchain> bash run-probe-settled.sh
set -uo pipefail
: "${WORK:?WORK: a scratch directory}" "${OUT:?OUT: where the logs go}" "${GOTOOLCHAIN:?GOTOOLCHAIN: the pinned toolchain}"
RUNS=${RUNS:-10}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/usr/local/go/bin:$HOME/go/bin:/usr/local/bin:$PATH GOWORK=off GOFLAGS=
mkdir -p "$WORK" "$OUT" && cd "$WORK" || exit 2
cp "$HERE/probe_test.go.txt" probe_test.go
cp "$HERE/probe_settled_test.go.txt" probe_settled_test.go
printf 'module probe\n\ngo 1.27\n' > go.mod
gofmt -l . | sed 's/^/gofmt would change: /'
{ echo "go:      $(go version)"
  echo "machine: $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion), native page size $(pagesize)"
  echo "started: $(date '+%F %T %Z')"; } | tee "$OUT/settled.provenance.txt"

for arch in arm64 amd64; do
  GOARCH=$arch go vet . || exit 2
  GOARCH=$arch go test -c -o "probe.$arch.test" . || exit 2
done

for arch in arm64 amd64; do
  echo "== $arch"
  for form in SliceForm IntForm; do
    for size in 96 400 4096 6000; do
      t="Test${form}Settled_$size"; ok=0; first=""; kinds=""
      for i in $(seq 1 "$RUNS"); do
        log="$WORK/$arch.$t.$i.log"
        "./probe.$arch.test" -test.v -test.count=1 -test.timeout 5m -test.run "^$t\$" > "$log" 2>&1; rc=$?
        [ "$i" -eq 1 ] && cp "$log" "$OUT/$arch.$t.first.log"
        if [ $rc -eq 0 ] && grep -q -E "^--- PASS: $t" "$log"; then ok=$((ok+1)); continue; fi
        kind=$(grep -m1 -E 'SIGSEGV|SIGBUS|fatal error|panic:|^--- FAIL' "$log" | cut -c1-100)
        kinds="$kinds$kind"$'\n'
        if [ -z "$first" ]; then first=$kind; cp "$log" "$OUT/$arch.$t.death.log"; fi
      done
      page=$(grep -m1 -o 'OS page [0-9]* bytes' "$OUT/$arch.$t.first.log")
      echo "  $t ($page, $(date '+%T')): $ok of $RUNS survive"
      [ -n "$kinds" ] && printf '%s' "$kinds" | sort | uniq -c | sed 's/^ */      deaths: /'
    done
  done
done
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/settled.provenance.txt"
