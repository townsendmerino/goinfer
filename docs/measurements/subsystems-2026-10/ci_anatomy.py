#!/usr/bin/env python3
"""ci_anatomy.py — how long goinfer's `ci` runs take, and, for the slow ones, where the time went.

Read-only (it calls `gh` with the account's read access). For each of the last N completed `ci` runs it prints the
wall time from creation to the last update; for each run over --slow minutes it prints every job's start offset from
the run's creation and its own duration, so queueing (a late start) and work (a long job) can be told apart.

  usage: ci_anatomy.py [--repo townsendmerino/goinfer] [--limit 60] [--slow 20]
"""
import argparse
import datetime as dt
import json
import subprocess


def gh(*args):
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout)


def t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="townsendmerino/goinfer")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--slow", type=float, default=20.0)
    a = ap.parse_args()
    runs = gh("run", "list", "--repo", a.repo, "--workflow", "ci.yml", "--limit", str(a.limit),
              "--json", "databaseId,event,status,conclusion,createdAt,updatedAt")
    done = [r for r in runs if r["status"] == "completed"]
    walls = sorted((t(r["updatedAt"]) - t(r["createdAt"])).total_seconds() / 60 for r in done)
    if not walls:
        return
    print(f"# {len(done)} completed ci runs: median {walls[len(walls) // 2]:.1f} min, "
          f"p90 {walls[int(.9 * len(walls))]:.1f} min, max {walls[-1]:.1f} min\n")
    for r in done:
        wall = (t(r["updatedAt"]) - t(r["createdAt"])).total_seconds() / 60
        print(f"{r['databaseId']} {r['event']:12s} {r['conclusion']:10s} {r['createdAt']} {wall:5.1f} min")
        if wall < a.slow:
            continue
        jobs = gh("run", "view", str(r["databaseId"]), "--repo", a.repo, "--json", "jobs")["jobs"]
        c0 = t(r["createdAt"])
        for j in sorted(jobs, key=lambda j: j.get("startedAt") or ""):
            if not j.get("startedAt") or not j.get("completedAt"):
                continue
            s, e = t(j["startedAt"]), t(j["completedAt"])
            print(f"    starts +{(s - c0).total_seconds() / 60:5.1f} min, runs {(e - s).total_seconds() / 60:5.1f} min  {j['name']}")


if __name__ == "__main__":
    main()
