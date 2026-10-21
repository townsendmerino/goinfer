#!/usr/bin/env python3
"""Run a `go test -json` command and judge it as a DEVICE run: a green line from a package whose device tests all skipped is not a pass.

The night job docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh runs ./cuda/... on nobara with the GPU present, through this
runner, so the verdict is not `go test`'s exit status alone:

  - a failed test, a failed package (a build error included) or a non-zero exit is RED;
  - a skip whose reason says there is no device (the cuda tests' own "no device" / "no CUDA device" / "did not go resident") is RED, because on this box a
    device skip means the device was unusable and the test proved nothing;
  - no test passing at all is RED;
  - other skips (a missing asset, GOINFER_HEAVY_TESTS unset) are counted and named but are not failures here.

The last lines of the output are the verdict and the failing names, because night.py's SUMMARY.md quotes the last 25 lines of a job that is not ok. A progress line
(elapsed, tests finished, the test in flight) is printed at least every PROGRESS_S seconds.

  python3 scripts/cuda_device_suite.py --cwd cuda --raw /path/raw.jsonl -- go test -json -tags 'cuda goinfer_testhooks' -timeout 25m ./...
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time

# A skip that means "the device was not there". The skip messages are the cuda tests' own wording (t.Skipf("no device: %v"), "no CUDA device", "no cuda device",
# "the model did not go resident on this device").
DEVICE_SKIP = re.compile(r"(?i)\bno (?:cuda )?device\b|did not go resident")
PROGRESS_S = 60


def judge(events, exit_code, tail):
    """events: decoded `go test -json` records. Returns (red, lines)."""
    out, status, flight = {}, {}, {}
    pkg_failed = []
    for e in events:
        t = (e.get("Package", ""), e.get("Test", ""))
        a = e.get("Action")
        if a == "output" and t[1]:
            out.setdefault(t, []).append(e.get("Output", ""))
        elif a in ("pass", "fail", "skip"):
            if t[1]:
                status[t] = a
            elif a == "fail":
                pkg_failed.append(t[0])
    failed = sorted(f"{p}.{n}" for (p, n), s in status.items() if s == "fail")
    passed = sum(1 for s in status.values() if s == "pass")
    device_skips, other_skips = [], []
    for (p, n), s in status.items():
        if s != "skip":
            continue
        (device_skips if DEVICE_SKIP.search("".join(out.get((p, n), []))) else other_skips).append(f"{p}.{n}")
    device_skips.sort()
    red = bool(failed or pkg_failed or device_skips or exit_code != 0 or passed == 0)
    lines = []
    if exit_code != 0 and not failed and not pkg_failed:
        lines += ["go test exited %d with no failed test in its output; its last lines:" % exit_code, *tail[-10:]]
    lines.append(f"cuda-device-suite: {passed} passed, {len(failed)} failed, {len(device_skips)} device skips, {len(other_skips)} other skips, "
                 f"{len(pkg_failed)} failed package(s)")
    if passed == 0:
        lines.append("RED: no test passed, so nothing ran")
    for title, names in (("FAILED TESTS", failed), ("DEVICE SKIPS (a skip is a failure here)", device_skips), ("FAILED PACKAGES", sorted(set(pkg_failed)))):
        if names:
            lines.append(f"{title} ({len(names)}):")
            lines += [f"  {n}" for n in names[:40]] + ([f"  ... and {len(names) - 40} more (the raw log has them)"] if len(names) > 40 else [])
    lines.append("VERDICT: " + ("RED" if red else "GREEN"))
    return red, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--raw", help="write the command's output here, verbatim")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=a.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    events, tail, raw = [], [], open(a.raw, "w") if a.raw else None
    state = {"done": 0, "last": ""}
    stop = threading.Event()

    def ticker():
        while not stop.wait(PROGRESS_S):
            print(f"... cuda-device-suite: {time.time() - t0:.0f}s elapsed, {state['done']} tests finished, last: {state['last']}", flush=True)

    threading.Thread(target=ticker, daemon=True).start()
    for line in p.stdout:
        if raw:
            raw.write(line)
        try:
            e = json.loads(line)
        except ValueError:
            tail.append(line.rstrip())
            tail[:] = tail[-50:]
            continue
        events.append(e)
        if e.get("Action") in ("pass", "fail", "skip") and e.get("Test"):
            state["done"] += 1
            state["last"] = e["Test"]
            if e["Action"] == "fail":
                print(f"--- FAIL: {e['Package']}.{e['Test']}", flush=True)
    rc = p.wait()
    stop.set()
    if raw:
        raw.close()
    red, lines = judge(events, rc, tail)
    print(f"finished in {time.time() - t0:.0f}s")
    print("\n".join(lines), flush=True)
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
