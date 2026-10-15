#!/usr/bin/env bash
# Put the main checkout's gitignored test fixtures into a fresh worktree, for the release-night scripts that run a gate in a
# detached worktree of a pinned commit (docs/measurements/**/run-*.sh). A fresh worktree has none of them, and most
# tiny-fixture tests then skip.
#
#   bash scripts/worktree_fixtures.sh SRC WT [EXCLUDE_FILE]
#
# The rule, per entry `git ls-files --others --ignored --directory` lists under testdata/ and decoder/testdata/:
#   - an entry that is a symlink in SRC (testdata/mellum2-tokenizer -> ~/models/...) stays a symlink: the same link is made in WT;
#   - a real file or directory of at most FIXTURE_COPY_MAX_MB (default 64) is COPIED (cp -a --reflink=auto; cp -cpR clones on
#     macOS): real files, because tests that copy a fixture directory keep only regular files (DirEntry.Type().IsRegular()) and
#     silently drop a symlinked config.json or model.safetensors;
#   - a larger real entry (a model slice) is CLONED (cp -a --reflink=always; cp -c on macOS): real files sharing the originals'
#     blocks, so no time and no space on a filesystem that clones (btrfs, XFS, APFS). Only where the clone fails is it linked:
#     a file as a symlink, a directory as a real directory of per-child symlinks (so a directory-only ignore rule still
#     matches it), with a WARNING on stderr naming it, since the tests that copy a fixture directory may then fail on it.
# Every entry is the same kind of object (file, directory, symlink) as in SRC, so the same .gitignore ignores it and the worktree's
# `git status --porcelain` stays empty.
#
# Untracked fixture directories that are NOT ignored (only part of them is gitignored) are placed by the same rule only when
# EXCLUDE_FILE is given: their paths are written to it, for the caller to pass as core.excludesFile so the gate does not read the
# tree as dirty. Without EXCLUDE_FILE they are left out and named on stderr.
#
# Prints one summary line on stdout for the caller's provenance block. Exits non-zero when a copy fails or the worktree's
# git status is not empty afterwards.
set -uo pipefail
SRC=${1:?usage: worktree_fixtures.sh SRC WT [EXCLUDE_FILE]}
WT=${2:?usage: worktree_fixtures.sh SRC WT [EXCLUDE_FILE]}
EXC=${3:-}
MAXKB=$(( ${FIXTURE_COPY_MAX_MB:-64} * 1024 ))
ROOTS="testdata decoder/testdata"
copied=0; copiedKB=0; cloned=0; clonedKB=0; linked=0; kept=0; fail=0; untracked=""

copyOne() { # $1 SRC path, $2 WT path
  if [ "$(uname -s)" = Darwin ]; then
    cp -cpR "$1" "$2" 2>/dev/null || { rm -rf "$2"; cp -pR "$1" "$2"; }
  else
    cp -a --reflink=auto "$1" "$2"
  fi
}

cloneOne() { # $1 SRC path, $2 WT path; fails, leaving nothing at $2, where the filesystem cannot clone
  if [ "$(uname -s)" = Darwin ]; then cp -cpR "$1" "$2" 2>/dev/null; else cp -a --reflink=always "$1" "$2" 2>/dev/null; fi && return 0
  rm -rf "$2"; return 1
}

place() { # $1: a path relative to the repo root
  local p=${1%/} s d kb c
  s=$SRC/$p d=$WT/$p
  [ -e "$d" ] || [ -L "$d" ] && return 0
  mkdir -p "$(dirname "$d")"
  if [ -L "$s" ]; then
    ln -s "$(readlink "$s")" "$d" || { fail=1; return; }
    kept=$((kept + 1)); return
  fi
  kb=$(du -sk "$s" | cut -f1)
  if [ "$kb" -le "$MAXKB" ]; then
    if copyOne "$s" "$d"; then copied=$((copied + 1)); copiedKB=$((copiedKB + kb)); else echo "worktree_fixtures: copy failed: $p" >&2; fail=1; fi
  elif cloneOne "$s" "$d"; then
    cloned=$((cloned + 1)); clonedKB=$((clonedKB + kb))
  elif [ -d "$s" ]; then
    echo "worktree_fixtures: WARNING: $p ($((kb / 1024)) MB) could not be cloned and is LINKED; a test that copies this fixture directory may fail on it" >&2
    mkdir -p "$d"
    for c in "$s"/* "$s"/.[!.]*; do { [ -e "$c" ] || [ -L "$c" ]; } && ln -s "$c" "$d/"; done
    linked=$((linked + 1))
  else
    echo "worktree_fixtures: WARNING: $p ($((kb / 1024)) MB) could not be cloned and is LINKED; a test that copies it may fail on it" >&2
    ln -s "$s" "$d" || fail=1
    linked=$((linked + 1))
  fi
}

cd "$SRC" || exit 2
[ -n "$EXC" ] && : > "$EXC"
# Untracked, not-ignored directories first: placing only their ignored files would make a directory holding model.safetensors
# and no config.json, which a directory-existence check takes for a whole fixture.
while IFS= read -r p; do
  case "$p" in */) ;; *) continue;; esac
  p=${p%/}
  if [ -n "$EXC" ]; then place "$p"; echo "$p" >> "$EXC"; untracked="$untracked $p"; else echo "worktree_fixtures: untracked, not ignored, left out: $p" >&2; fi
done < <(git ls-files --others --exclude-standard --directory -- $ROOTS)
while IFS= read -r p; do place "$p"; done < <(git ls-files --others --ignored --exclude-standard --directory -- $ROOTS)

dirty=$(if [ -n "$EXC" ]; then git -C "$WT" -c core.excludesFile="$EXC" status --porcelain; else git -C "$WT" status --porcelain; fi)
echo "fixtures from $SRC: $copied copied ($((copiedKB / 1024)) MB, entries <= ${FIXTURE_COPY_MAX_MB:-64} MB), $cloned cloned ($((clonedKB / 1024)) MB, larger), $linked linked (larger, could not clone), $kept symlinks kept as symlinks${untracked:+; untracked dirs placed and excluded:$untracked}; worktree git status: $(printf '%s' "$dirty" | grep -c .) entries"
[ -n "$dirty" ] && { echo "worktree_fixtures: the worktree is not clean after placing fixtures:" >&2; printf '%s\n' "$dirty" | head -5 >&2; fail=1; }
exit $fail
