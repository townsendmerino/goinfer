#!/usr/bin/env bash
# check_asset_vcs.sh <binary>... — fail unless every binary's embedded build info says vcs.modified=false.
#   check_asset_vcs.sh --selftest   — prove this check can go red (a dirty and a clean build, both read back).
#
# Why: the v0.17.0 to v0.19.0 release assets printed `vX.Y.Z (<rev>-dirty)` (R6, then R18 of docs/tasks/task-first-hour.md).
# The build step's own `go mod edit -replace` was an uncommitted edit to a tracked file, and the embedded job's downloaded
# licenses/ an untracked one; either marks the whole tree modified in the VCS stamp. v0.20.0 still shipped it (`go version -m` on the published darwin-arm64 and linux-amd64 assets
# read vcs.modified=true), because nothing read the stamp back. This reads it, on the built file, before it is published.
#
# A binary with NO vcs.modified line fails too: that is a build with no VCS information, where "not dirty" is unknown, and a
# check that passes on "nothing found" is indistinguishable from one that never ran. Works on any GOOS/GOARCH's binary
# (`go version -m` reads the build info, it does not run the file), and on a downloaded release asset.
set -uo pipefail

check() {
  local f rc=0 info mod rev
  for f in "$@"; do
    if ! info="$(go version -m "$f" 2>&1)"; then
      echo "FAIL $f: go version -m could not read it: $(echo "$info" | head -1)"; rc=1; continue
    fi
    mod="$(echo "$info" | awk '$1=="build" && $2 ~ /^vcs\.modified=/ { sub(/^vcs\.modified=/, "", $2); print $2; exit }')"
    rev="$(echo "$info" | awk '$1=="build" && $2 ~ /^vcs\.revision=/ { sub(/^vcs\.revision=/, "", $2); print substr($2, 1, 8); exit }')"
    case "$mod" in
      false) echo "ok   $f (vcs.revision=${rev:-?}, vcs.modified=false)" ;;
      true)  echo "FAIL $f: vcs.modified=true (revision ${rev:-?}) — the checkout it was built from had a changed or untracked file"; rc=1 ;;
      *)     echo "FAIL $f: no vcs.modified in its build info — built without VCS stamping, so 'not dirty' is unknown"; rc=1 ;;
    esac
  done
  return $rc
}

selftest() {
  local d; d="$(mktemp -d)" || return 2
  trap 'rm -rf "$d"' RETURN
  ( cd "$d" && git init -q . && printf 'module r18probe\n\ngo 1.21\n' > go.mod && printf 'package main\n\nfunc main() {}\n' > main.go \
      && git add go.mod main.go && git -c user.email=t@t -c user.name=t commit -q -m probe \
      && go build -o clean . \
      && printf '\n// edit\n' >> go.mod && go build -o dirty . ) || { echo "selftest: could not build the probes"; return 2; }
  local ok=0
  check "$d/clean" >/dev/null || { echo "selftest FAILED: a clean build was rejected"; ok=1; }
  check "$d/dirty" >/dev/null && { echo "selftest FAILED: a build from a tree with an edited go.mod was accepted"; ok=1; }
  ( cd "$d" && go build -buildvcs=false -o novcs . ) && { check "$d/novcs" >/dev/null && { echo "selftest FAILED: a build with no VCS stamp was accepted"; ok=1; }; }
  [ "$ok" = 0 ] && echo "selftest ok: clean accepted; edited-go.mod and unstamped builds rejected"
  return $ok
}

[ "$#" -gt 0 ] || { echo "usage: $0 <binary>... | --selftest" >&2; exit 2; }
if [ "$1" = "--selftest" ]; then selftest; exit $?; fi
check "$@"
