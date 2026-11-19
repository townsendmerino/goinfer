#!/usr/bin/env python3
"""ci_pr_covered.py: does a green pull-request ci run already cover this push to main, on the identical tree?

On a push to main, ci.yml's `changes` job runs this and exports `pr_covered`. Lanes whose only main-run failures have been
ones their PR run also saw skip when it is "true"; `changes`, `lint`, `root-windows`, `root-linux-arm64` and `gpu-darwin`
always run, since each has gone red on main on a tree its PR run passed (the 2026-11-19 CI review, finding 1).

"true" needs every one of these, and anything else, including any error, prints "false" (fail open: the full matrix runs):
  1. the push is a two-parent merge commit on main (a squash, a rebase merge or a direct push is not covered);
  2. a merged pull request names this commit as its merge commit, and its head is the merge's second parent;
  3. the newest `pull_request` ci run on that head concluded "success";
  4. that run's `ci-tree` artifact (written by `changes` in a PR run) records docs_only=false and the same tree id as this
     push's HEAD^{tree}.

Usage: ci_pr_covered.py --repo <owner/name> --sha <full sha> [--ref refs/heads/main] [--event push]
Writes pr_covered=<v> to $GITHUB_OUTPUT and the reason to $GITHUB_STEP_SUMMARY when set; always exits 0.
Tests: python3 -B scripts/test_ci_pr_covered.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ARTIFACT = "ci-tree"
ARTIFACT_FILE = "ci-tree.txt"


def evaluate(sha, ref, event, git, api, download):
    """Returns (covered, reason). git(*args) -> stdout, api(path) -> parsed JSON and download(run_id) -> the artifact's
    text (None when absent) are injected so the decision can be tested without git history or the network. Any exception
    they raise is a "false"."""
    try:
        return _evaluate(sha, ref, event, git, api, download)
    except Exception as e:  # fail open on anything unexpected
        return False, f"error: {type(e).__name__}: {e}"


def _evaluate(sha, ref, event, git, api, download):
    if event != "push" or ref != "refs/heads/main":
        return False, f"not a push to main (event={event}, ref={ref})"
    parents = git("rev-list", "--parents", "-n", "1", sha).split()[1:]
    if len(parents) != 2:
        return False, f"not a two-parent merge commit ({len(parents)} parent(s): a squash, rebase merge or direct push)"
    head_tree = git("rev-parse", f"{sha}^{{tree}}")

    prs = api(f"commits/{sha}/pulls") or []
    pr = next((p for p in prs if p.get("merged_at") and p.get("merge_commit_sha") == sha
               and (p.get("head") or {}).get("sha") == parents[1]), None)
    if pr is None:
        return False, "no merged pull request has this commit as its merge commit with the second parent as its head"
    head = parents[1]

    runs = (api(f"actions/workflows/ci.yml/runs?event=pull_request&head_sha={head}&per_page=50") or {}).get("workflow_runs") or []
    runs = [r for r in runs if r.get("head_sha") == head]
    if not runs:
        return False, f"PR #{pr['number']}: no pull_request ci run on its head {head[:12]}"
    run = max(runs, key=lambda r: (r.get("created_at") or "", r.get("id") or 0))
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return False, (f"PR #{pr['number']}: its last ci run {run.get('id')} is "
                       f"{run.get('status')}/{run.get('conclusion')}, not completed/success")

    text = download(run["id"])
    if text is None:
        return False, f"PR #{pr['number']}: run {run['id']} has no {ARTIFACT} artifact"
    rec = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    if rec.get("docs_only") != "false":
        return False, f"PR #{pr['number']}: run {run['id']} did not run the full matrix (docs_only={rec.get('docs_only')})"
    if rec.get("tree") != head_tree:
        return False, f"PR #{pr['number']}: run {run['id']} tested tree {rec.get('tree')}, this push is tree {head_tree}"
    return True, f"PR #{pr['number']}: run {run['id']} passed on tree {head_tree}, the tree of this push"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _api(repo):
    def call(path):
        out = subprocess.run(["gh", "api", f"repos/{repo}/{path}"], capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    return call


def _download(repo):
    def call(run_id):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(["gh", "run", "download", str(run_id), "-R", repo, "-n", ARTIFACT, "-D", d],
                               capture_output=True, text=True)
            p = os.path.join(d, ARTIFACT_FILE)
            if r.returncode != 0 or not os.path.exists(p):
                return None
            with open(p) as f:
                return f.read()
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    a = ap.parse_args()
    covered, reason = evaluate(a.sha, a.ref, a.event, _git, _api(a.repo), _download(a.repo))
    v = "true" if covered else "false"
    print(f"pr_covered={v}: {reason}")
    for var, line in (("GITHUB_OUTPUT", f"pr_covered={v}\n"),
                      ("GITHUB_STEP_SUMMARY", f"**pr_covered={v}**: {reason}\n")):
        if os.environ.get(var):
            with open(os.environ[var], "a") as f:
                f.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
