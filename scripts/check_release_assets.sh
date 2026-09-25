#!/usr/bin/env bash
# check_release_assets.sh <tag> — fail if the GitHub Release for <tag> carries no assets, or fewer
# than the nearest earlier release that carries any.
#
# Why this checks the SYMPTOM and not a cause: v0.18.0 went out with 0 assets while v0.17.x had 27.
# The asset workflow had failed (a Windows cross-compile break), so its `publish` job never ran —
# and the Release was created and published anyway by RELEASING.md's final step, which never looked
# at what was attached. The run was red; the releases page looked normal. A compile break is one
# way to get there; a dropped matrix leg, a failed upload or a Release created before a re-run are
# others. The one thing they share is the count on the published Release, so that is what this reads.
#
# Run from two places, because either one alone misses a case:
#   - release-assets.yml's last job, `if: always()` — runs even when the build jobs failed;
#   - RELEASING.md's GitHub Release step, after the human creates or edits the Release — the moment
#     it becomes something users see.
#
# The bar is the nearest EARLIER release with a nonzero count, so an empty release (v0.18.0) does not
# lower it for the next one. ALLOW_FEWER_ASSETS=1 accepts a deliberate drop (e.g. a workflow_dispatch
# with the 1.5B tier opted out); zero assets is never accepted.
#
# Exit 0: OK, or no Release exists for <tag> yet (nothing is published, so nothing misleads).
# Exit 1: a published Release is empty or short. Exit 2: usage / lookup failure.
set -euo pipefail

tag="${1:-}"
[ -n "$tag" ] || { echo "usage: $0 <tag>" >&2; exit 2; }
repo="${GH_REPO:-townsendmerino/goinfer}"

if ! gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
  echo "no GitHub Release for $tag yet — nothing published, nothing to check"
  exit 0
fi

count() { gh release view "$1" --repo "$repo" --json assets --jq '.assets | length'; }
have=$(count "$tag")

# Earlier releases in version order (not publish order: a patch on an older line can publish later).
# Drafts and pre-releases are not a bar.
earlier=$(gh release list --repo "$repo" --limit 200 --exclude-drafts --exclude-pre-releases \
            --json tagName --jq '.[].tagName' | grep -E '^v[0-9]' | sort -V \
          | awk -v t="$tag" '$0 == t { exit } { print }')
bar=0 bar_tag=""
for t in $(printf '%s\n' "$earlier" | sort -rV); do
  n=$(count "$t")
  if [ "$n" -gt 0 ]; then bar=$n bar_tag=$t; break; fi
done

echo "$tag: $have asset(s); nearest earlier release with assets: ${bar_tag:-none} ($bar)"
if [ "$have" -eq 0 ]; then
  echo "::error::$tag is published with ZERO assets. Users arriving at this tag find nothing to download."
  echo "Check this tag's release-assets run; re-run it (workflow_dispatch with tag=$tag) once the cause is fixed."
  exit 1
fi
if [ "$have" -lt "$bar" ]; then
  if [ "${ALLOW_FEWER_ASSETS:-}" = "1" ]; then
    echo "WARNING: $have < $bar ($bar_tag), accepted because ALLOW_FEWER_ASSETS=1"
    exit 0
  fi
  echo "::error::$tag carries $have assets, fewer than $bar_tag's $bar. Missing assets are silent on the releases page."
  echo "If the drop is deliberate, re-run with ALLOW_FEWER_ASSETS=1 and say why in the release notes."
  exit 1
fi
echo "OK"
