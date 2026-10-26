#!/usr/bin/env python3
"""aikit_sweep.py: internal/kernelprint's fingerprints at each aikit version goinfer pinned in VK0's window.

    python3 scripts/timing_lock.py run --shared --label vk1-aikit-sweep -- \
        python3 docs/measurements/vk1-2026-10/aikit_sweep.py OUT.jsonl v1.75.0 v1.74.0 ...

Run from the repository root. For each version, newest first, and each configuration (arm64 as built, and amd64 under
Rosetta with AVX advertised), it builds TestCompute_child against that aikit through a temporary -modfile and records
the report, or the build error when the suite does not compile against that version's API. One JSON object per line.
A heartbeat goes to stderr after every build.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

CONFIGS = [
    ("arm64", {}),
    ("amd64-rosetta", {"GOARCH": "amd64", "ROSETTA_ADVERTISE_AVX": "1"}),
]


def main(out_path, versions):
    root = os.getcwd()
    gomod = open(os.path.join(root, "go.mod")).read()
    if not re.search(r"(?m)^\tgithub\.com/townsendmerino/aikit v\S+$", gomod):
        sys.exit("go.mod has no tab-indented aikit require line")
    t0 = time.time()
    total = len(versions) * len(CONFIGS)
    done = 0
    with open(out_path, "w") as out:
        for v in versions:
            tmp = tempfile.mkdtemp(prefix="vk1-sweep-")
            try:
                mod = re.sub(r"(?m)^(\tgithub\.com/townsendmerino/aikit) v\S+$", r"\1 " + v, gomod)
                open(os.path.join(tmp, "go.mod"), "w").write(mod)
                shutil.copy(os.path.join(root, "go.sum"), os.path.join(tmp, "go.sum"))
                for name, extra in CONFIGS:
                    report = os.path.join(tmp, "report.json")
                    env = dict(os.environ, GOWORK="off", GOINFER_KERNELPRINT_OUT=report, **extra)
                    cmd = ["go", "test", "-modfile=" + os.path.join(tmp, "go.mod"), "-mod=mod", "-count=1",
                           "-run", "^TestCompute_child$", "./internal/kernelprint/"]
                    t1 = time.time()
                    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    rec = {"aikit": v, "config": name, "ok": p.returncode == 0, "seconds": round(time.time() - t1, 1)}
                    if p.returncode == 0:
                        rec["report"] = json.load(open(report))
                        os.remove(report)
                    else:
                        rec["error"] = (p.stdout + p.stderr).strip().splitlines()[:6]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    done += 1
                    el = time.time() - t0
                    print(f"[{time.strftime('%H:%M:%S')}] {done}/{total} {v} {name}: {'ok' if rec['ok'] else 'build failed'}"
                          f" in {rec['seconds']}s; elapsed {el:.0f}s", file=sys.stderr, flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
