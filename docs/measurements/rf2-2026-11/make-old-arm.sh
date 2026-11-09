#!/usr/bin/env bash
# G-RF2-S's and G-RF2-I's old arm: <main> with every RF2 merge reverted, newest first, so old -> new differs by exactly RF2's code.
#   make-old-arm.sh <main-sha> <worktree-dir> [--push <branch>]
# RF2's merges are found among all of <main>'s merges (not only its first-parent line: a branch merged into main can put a PR's
# merge on a second parent) by their PR numbers (RF2_PRS). A PR with no merge of its own (it came in through a later PR's branch)
# is checked with gh: its head must be inside another listed PR's merge, whose revert then removes it. A revert that conflicts only in docs/ keeps
# <main>'s side (no binary reads a doc); a conflict in any other path stops the script. It then checks that no RF2 symbol is left in
# backend/ or decoder/ and prints the old arm's sha. --push pushes that sha to <branch> (a new branch; never a force-push).
set -euo pipefail
MAIN=${1:?main sha}; WT=${2:?worktree dir}; PUSH=${3:-}; BRANCH=${4:-}
RF2_PRS=${RF2_PRS:-"787 799 804 811 823 825 826"}
git worktree add --detach "$WT" "$MAIN"
cd "$WT"
merges=()
merge_of() { git log --merges --format='%H %s' "$MAIN" | awk -v p="#$1 " '!f && $0 ~ "Merge pull request " p {print $1; f = 1}'; }
absorbed=()
for pr in $RF2_PRS; do
  c=$(merge_of "$pr")
  if [ -n "$c" ]; then merges+=("$c"); else absorbed+=("$pr"); fi
done
for pr in "${absorbed[@]}"; do
  head=$(gh pr view "$pr" --json state,headRefOid -q 'select(.state == "MERGED") | .headRefOid') || head=
  [ -n "$head" ] || { echo "make-old-arm: PR #$pr has no merge on $MAIN and gh does not show it merged" >&2; exit 2; }
  via=
  for c in "${merges[@]}"; do
    if git merge-base --is-ancestor "$head" "$c^2" && ! git merge-base --is-ancestor "$head" "$c^1"; then via=$c; break; fi
  done
  [ -n "$via" ] || { echo "make-old-arm: PR #$pr has no merge on $MAIN and no listed PR's merge brought its head $head in" >&2; exit 2; }
  echo "PR #$pr has no merge of its own: it came in with $(git log -1 --format=%s "$via" | cut -c1-40), whose revert removes it"
done
# newest first: sort the merges in topological order
mapfile -t ordered < <(git log --topo-order --format=%H "$MAIN" | grep -F -x -f <(printf '%s\n' "${merges[@]}"))
for c in "${ordered[@]}"; do
  if git revert --no-edit -m 1 "$c" >/dev/null 2>&1; then
    echo "reverted $c ($(git log -1 --format=%s "$c" | cut -c1-60))"
    continue
  fi
  conflicted=$(git diff --name-only --diff-filter=U)
  for f in $conflicted; do
    case $f in
      docs/*) if git cat-file -e HEAD:"$f" 2>/dev/null; then git checkout --ours -- "$f"; git add -- "$f"; else git rm -q -- "$f"; fi ;;
      *) echo "make-old-arm: reverting $c conflicts in $f (not a doc): build this arm by hand" >&2; exit 3 ;;
    esac
  done
  GIT_EDITOR=true git revert --continue >/dev/null
  echo "reverted $c (doc conflicts kept as $MAIN's: $(echo $conflicted | tr '\n' ' '))"
done
if left=$(git grep -l -E 'StepRagged|attn_ragged|ResidentRaggedStepper|soloRagged|fa_partial_ragged|splitkv_scores_ragged' -- backend decoder); then
  echo "make-old-arm: RF2 symbols are left in: $left" >&2; exit 4
fi
OLD=$(git rev-parse HEAD)
echo "OLD ARM $OLD (= $MAIN with RF2 reverted)"
if [ "$PUSH" = "--push" ]; then
  [ -n "$BRANCH" ] || { echo "make-old-arm: --push needs a branch name" >&2; exit 2; }
  git push origin "$OLD:refs/heads/$BRANCH"
fi
