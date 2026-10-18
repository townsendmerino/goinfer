#!/usr/bin/env python3
"""Refuse a push to main while main's latest CI run is still going (owner, 2026-10-18; CLAUDE.md, "Push finished work").

"Latest" is every push-triggered workflow run for the newest commit on main. If any of them has not completed, a push now
would land on top of a run nobody has read, and a docs-only push would cancel it. The script names each such run and
exits 1; it exits 0 when every one has completed.

The pre-push hook (scripts/install-git-hooks.sh) runs it only for a push to refs/heads/main, with the push's range. A
docs-only range goes straight through: scripts/ci_docs_only.sh decides, the same script CI's gate runs (docs/**,
RELEASING.md and .claude/**, none read by Go code or tests), so the guard and CI agree on what is product code. It reads
the runs with `gh run list`; MAIN_CI_GUARD_RUNS names a JSON file of the same shape instead, and MAIN_CI_GUARD_DOCS_ONLY
(true or false) stands in for the classifier, for the tests. A failed read refuses rather than passing silently. Bypass
for a deliberate push: git push --no-verify.

Usage: python3 scripts/main_ci_guard.py [BASE HEAD]"""
import json
import os
import subprocess
import sys

FIELDS = "databaseId,status,conclusion,headSha,event,workflowName,displayTitle,url,createdAt"
DONE = {"completed"}


def runs():
    """main's recent workflow runs, newest first, as gh returns them."""
    fake = os.environ.get("MAIN_CI_GUARD_RUNS")
    if fake:
        with open(fake) as f:
            return json.load(f)
    out = subprocess.run(["gh", "run", "list", "--branch", "main", "--limit", "30", "--json", FIELDS],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"gh exited {out.returncode}")
    return json.loads(out.stdout)


def unfinished(rs):
    """The push-triggered runs for the newest pushed commit on main that have not completed."""
    pushes = [r for r in rs if r.get("event") == "push"]
    if not pushes:
        return []
    newest = max(pushes, key=lambda r: r.get("createdAt", ""))["headSha"]
    return [r for r in pushes if r["headSha"] == newest and r.get("status") not in DONE]


def docs_only(base, head):
    """Whether the push base..head changes only non-code paths, by scripts/ci_docs_only.sh (which prints "false" on any
    doubt, so an unclassifiable push is held like product code)."""
    forced = os.environ.get("MAIN_CI_GUARD_DOCS_ONLY")
    if forced in ("true", "false"):
        return forced == "true"
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ci_docs_only.sh")
    out = subprocess.run(["bash", script, base, head], capture_output=True, text=True, timeout=120)
    return out.stdout.strip() == "true"


def main(argv):
    if len(argv) == 2 and docs_only(*argv):
        print("main-ci-guard: a docs-only push (scripts/ci_docs_only.sh); main's CI is not waited for", file=sys.stderr)
        return 0
    try:
        rs = runs()
    except Exception as e:  # noqa: BLE001 — any failure to read is a refusal, never a silent pass
        print(f"main-ci-guard REFUSED: could not read main's CI runs ({e}).", file=sys.stderr)
        print("  Read them yourself (gh run list --branch main --limit 5); a deliberate push: git push --no-verify", file=sys.stderr)
        return 1
    busy = unfinished(rs)
    if not busy:
        return 0
    print(f"main-ci-guard REFUSED: main's latest CI is still running for {busy[0]['headSha'][:8]} "
          f"({busy[0].get('displayTitle', '')[:70]}):", file=sys.stderr)
    for r in busy:
        print(f"  run {r['databaseId']} {r.get('workflowName', '')}: {r.get('status')} — {r.get('url', '')}", file=sys.stderr)
    print("  Wait for it to finish and read it, then push. A deliberate push: git push --no-verify", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
