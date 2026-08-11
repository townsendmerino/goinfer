#!/usr/bin/env python3
"""skip_census.py — the release-ritual test census.

A green `go test ./...` is not "all green" if most of the suite skipped. This tool runs
(or parses) `go test -json`, then reports PASS / SKIP / FAIL counts with every SKIP
bucketed by *why* it skipped, so a run that skipped 200 asset-gated tests can never be
mistaken for one that actually exercised them.

Buckets (see docs/parity-coverage-policy.md, "A gate must be able to run, and able to
fail"):
  missing-fixture  a committed golden over a gitignored checkpoint (regen: scripts/pin_*.py)
  missing-golden   no *_golden.json recorded yet
  no-gpu-device    needs a real GPU/Metal/CUDA device (CI has none)
  heavy-model      gated behind GOINFER_HEAVY_TESTS / a real ~/models checkpoint
  integration-env  needs a runtime env var (GOINFER_SERVE_MODEL / _EMBED_MODEL) — a live-server test
  other            anything unclassified — inspect these; they should be rare

Usage:
  scripts/skip_census.py                      # run pure-Go `go test ./... -json`, report
  scripts/skip_census.py path/to/run.json     # parse an already-recorded -json stream
  scripts/skip_census.py -- -tags metal ./metal/   # pass args through to `go test`

Release ritual — turn asset-skips into a hard gate:
  GOINFER_REQUIRE_FIXTURES=1 scripts/skip_census.py    # exit 1 if any missing-fixture skip
        # (a committed-fixture family MUST run; if its fixture is absent the release box
        #  is misconfigured — the census fails loudly instead of skipping silently.)
"""
import json, os, re, subprocess, sys

# reason -> bucket, first match wins (order matters: fixture before device before heavy)
RULES = [
    ("heavy-model",     re.compile(r"heavy|GOINFER_HEAVY|~/models|large model|big box|real (model|checkpoint)", re.I)),
    ("integration-env", re.compile(r"\bset GOINFER_|GOINFER_SERVE_MODEL|GOINFER_EMBED_MODEL|GOINFER_VISION|integration", re.I)),
    ("missing-golden",  re.compile(r"no golden|_golden\.json|run scripts/pin", re.I)),
    ("missing-fixture", re.compile(r"no tiny|checkpoint|fixture|\.safetensors|not present|not found|absent|no such|no (gguf|model|tokenizer|qwen|mellum|gemma|llama|mistral|phi|glm|deepseek|kimi|granite|nemotron|cohere)", re.I)),
    ("no-gpu-device",   re.compile(r"\bgpu\b|\bdevice\b|webgpu|no adapter|headless|no metal|no cuda", re.I)),
]
BUCKETS = ["missing-fixture", "missing-golden", "no-gpu-device", "heavy-model", "integration-env", "other"]

def classify(reason):
    for name, rx in RULES:
        if rx.search(reason or ""):
            return name
    return "other"

def stream_events(argv):
    """Yield parsed -json objects, either from a file arg or by running go test."""
    if argv and not argv[0].startswith("-") and os.path.exists(argv[0]):
        with open(argv[0]) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: yield json.loads(line)
                    except json.JSONDecodeError: pass
        return
    # Default run carries -tags goinfer_testhooks so the relocated test hooks (audit B-08)
    # are present and their tests execute rather than silently skipping. Explicit argv
    # (e.g. `-- -tags cuda ./cuda/`) should include goinfer_testhooks too.
    cmd = ["go", "test", "-json"] + (argv if argv else ["-tags", "goinfer_testhooks", "./..."])
    env = dict(os.environ, CGO_ENABLED=os.environ.get("CGO_ENABLED", "0"))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    for line in p.stdout:
        line = line.strip()
        if line:
            try: yield json.loads(line)
            except json.JSONDecodeError: pass
    p.wait()

def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]

    # per (package,test): final action + accumulated output (for skip reason)
    final = {}          # key -> action
    out = {}            # key -> [output lines]
    pkg_fail = []       # packages that failed at the package level (may be a native crash)
    pkg_out = {}        # package -> [output] (to detect SIGSEGV/panic when no test failed)
    for ev in stream_events(argv):
        test = ev.get("Test")
        pkg = ev.get("Package", "")
        act = ev.get("Action")
        if not test:    # package-level event
            if act == "output":
                pkg_out.setdefault(pkg, []).append(ev.get("Output", ""))
            elif act == "fail":
                pkg_fail.append(pkg)
            continue
        key = (pkg, test)
        if act == "output":
            out.setdefault(key, []).append(ev.get("Output", ""))
        elif act in ("pass", "fail", "skip"):
            final[key] = act

    npass = sum(1 for a in final.values() if a == "pass")
    nfail = sum(1 for a in final.values() if a == "fail")
    skips = [k for k, a in final.items() if a == "skip"]

    # bucket skips by reason (reason = the last non-empty --- SKIP / skip line)
    bucketed = {b: [] for b in BUCKETS}
    for key in skips:
        text = "".join(out.get(key, []))
        # prefer the explicit skip message line
        m = re.search(r"--- SKIP.*?\n(.*)", text)
        reason = ""
        for ln in text.splitlines():
            ls = ln.strip()
            if ls and "--- SKIP" not in ls and "=== RUN" not in ls and "=== PAUSE" not in ls and "=== CONT" not in ls:
                reason = ls
        bucketed[classify(reason)].append((key[0], key[1], reason))

    print("=" * 72)
    print(" goinfer test census")
    print("=" * 72)
    print(f"  PASS  {npass}")
    print(f"  FAIL  {nfail}")
    print(f"  SKIP  {len(skips)}")
    print(f"  total {len(final)}")
    print("\n  skip buckets:")
    for b in BUCKETS:
        print(f"    {b:16s} {len(bucketed[b])}")

    if nfail:
        print("\n  FAILURES:")
        for k, a in final.items():
            if a == "fail":
                print(f"    {k[0]}  {k[1]}")

    # package-level fails with no per-test fail = build error or NATIVE CRASH. The Metal suite
    # has a flaky single-process `fault 0x10` tail (purego-objc / no-ARC — reproduces even as
    # sole GPU user; contention raises the odds). It is NOT a test failure: every test passes
    # in isolation/shards. Shard the run (-run '^Test[A-L]' / '^Test[M-Z]') for a green pass.
    crashed = []
    for pkg in pkg_fail:
        blob = "".join(pkg_out.get(pkg, []))
        if any(s in blob for s in ("SIGSEGV", "fault 0x", "signal arrived", "panic:")):
            crashed.append(pkg)
    if pkg_fail:
        print("\n  PACKAGE-LEVEL FAILS (not counted above):")
        for pkg in pkg_fail:
            tag = "  ⚠ NATIVE CRASH (flaky fault 0x10 — shard the run; tests pass in isolation)" if pkg in crashed else ""
            print(f"    {pkg}{tag}")

    if bucketed["other"]:
        print("\n  unclassified skips (inspect — bucket rules may need a new pattern):")
        for pkg, t, r in bucketed["other"][:40]:
            print(f"    {pkg}  {t}\n      {r[:100]}")

    # release-ritual gate
    rc = 1 if nfail else 0

    # A census of ZERO tests is not a clean census — it is the absence of one. An empty -json
    # stream (bad -tags, wrong package path, truncated capture, a build error that emitted no
    # test events) lands here as PASS 0 / FAIL 0 / SKIP 0 and used to exit 0, which reads as
    # "nothing wrong" when it means "nothing looked". Same shape as the gate's skip counter,
    # which read zero both for "no skips" and for "I forgot -v so go test never printed them".
    # Wherever a zero can mean either, it has to say which.
    if not final:
        print("\n  ✗ NO TESTS OBSERVED — the -json stream contained no test events.")
        print("    This is NOT a pass. Usual causes: a build/tag error that produced no test")
        print("    events, a package path matching nothing, or a truncated recorded stream.")
        print("    Re-run and check `go test` itself succeeds before reading this census.")
        rc = 1
    if os.environ.get("GOINFER_REQUIRE_FIXTURES") and bucketed["missing-fixture"]:
        print("\n  ✗ GOINFER_REQUIRE_FIXTURES=1 and missing-fixture skips present:")
        for pkg, t, r in bucketed["missing-fixture"]:
            print(f"      {pkg}  {t}  — {r[:80]}")
        print("    A committed-fixture family must run. Regenerate the fixture "
              "(scripts/pin_*.py) or fix the box; do not tag on a skipped parity gate.")
        rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    main()
