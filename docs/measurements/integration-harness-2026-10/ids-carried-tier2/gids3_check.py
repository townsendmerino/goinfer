#!/usr/bin/env python3
"""G-ids-3's count (id-roundtrip-step0-2026-10-15.md, §(c) and Amendment 1; plan in ids-carried-2026-10-15.md): per arm, every
Codex follow-up classified by what serve reused, and the on arm's carried-id fallbacks by reason. Reads only a run's output
directory; no server, no GPU.

  python3 -I gids3_check.py OUT                 # the table
  python3 -I gids3_check.py OUT --stop-check    # exit 10 when the registered early stop holds, 0 when it does not
  python3 -I gids3_check.py --after ARCHIVE_DIR # the same count on the 2026-10-15 after run (its check: 15 cold, 20 compactions)

A follow-up is each response after a session's first, classified from Codex's own session files (cached_input_tokens), as
diagnose_after_compactions.py classifies the after run: `compaction` (the response Codex compacted after), `post-compaction`
(the first after it), else `full` (reused the previous prompt plus its output), `cold` (reused nothing) or `partial`. A cold
plain continuation is a `cold` follow-up. Fallbacks are read from serve's -log-requests lines (ids_fallback=).

Early stop (registered in Amendment 1): when the on arm's first 5 runs, in run order, show no cold plain continuation and no
fallback other than decode, the answer is clear and the remaining reps are not run.
"""
import glob, gzip, json, os, re, sys, tarfile, tempfile


def load_session(path):
    out = []
    for line in open(path):
        o = json.loads(line)
        if o["type"] == "token_usage_record":
            u = o["payload"]["usage"]
            out.append({"inp": u["input_tokens"], "out": u["output_tokens"], "cached": u["cached_input_tokens"], "comp": False})
        elif o["type"] == "compacted" and out:
            out[-1]["comp"] = True
    return out


def kind(L, n):
    a, b = L[n - 1], L[n]
    if b["comp"]:
        return "compaction"
    if a["comp"]:
        return "post-compaction"
    if b["cached"] >= a["inp"] + a["out"]:
        return "full"
    return "cold" if b["cached"] == 0 else "partial"


def sessions_of(rundir):
    fs = sorted(glob.glob(os.path.join(rundir, "home", ".codex", "sessions", "*", "*", "*", "*.jsonl")))
    return [load_session(f) for f in fs]


FB = re.compile(r"ids_fallback=(\S+)")


def fallbacks(logpath):
    out = {}
    if not os.path.exists(logpath):
        return out
    for line in open(logpath, errors="replace"):
        if line.startswith("request: POST /v1/responses"):
            m = FB.search(line)
            if m and m.group(1) != "-":
                for r in m.group(1).split(","):
                    out[r] = out.get(r, 0) + 1
    return out


def count(out_dir):
    recs = []
    p = os.path.join(out_dir, "records.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("harness_error") or r.get("void_halted"):
                continue
            recs.append(r)
    arms = {}
    for r in recs:
        lab = r.get("labels") or {}
        arm = str(lab.get("arm", "?"))
        a = arms.setdefault(arm, {"runs": [], "kinds": {}, "passed": 0, "compactions": 0, "fallbacks": {}})
        kinds = {}
        for L in sessions_of(os.path.join(out_dir, "runs", r["id"])):
            for n in range(1, len(L)):
                k = kind(L, n)
                kinds[k] = kinds.get(k, 0) + 1
        fb = fallbacks(os.path.join(out_dir, "logs", f"serve-{lab.get('config')}.log"))
        a["runs"].append({"id": r["id"], "config": lab.get("config"), "kinds": kinds, "fallbacks": fb, "passed": bool(r.get("passed"))})
        for k, v in kinds.items():
            a["kinds"][k] = a["kinds"].get(k, 0) + v
        for k, v in fb.items():
            a["fallbacks"][k] = a["fallbacks"].get(k, 0) + v
        a["passed"] += bool(r.get("passed"))
        a["compactions"] += int(r.get("compactions") or 0)
    return arms


def show(arms):
    for arm in sorted(arms):
        a = arms[arm]
        print(f"arm {arm}: {len(a['runs'])} runs, passed {a['passed']}, compactions {a['compactions']}, "
              f"follow-ups {a['kinds']}, cold plain continuations {a['kinds'].get('cold', 0)}, fallbacks {a['fallbacks'] or '-'}")
        for r in a["runs"]:
            print(f"    {r['config']:24s} {r['id']:28s} passed={r['passed']!s:5s} {r['kinds']} fallbacks={r['fallbacks'] or '-'}")


def stop_clear(arms):
    on = (arms.get("on") or {}).get("runs", [])[:5]
    if len(on) < 5:
        return False, f"the on arm has {len(on)} of the 5 runs the early stop reads"
    cold = sum(r["kinds"].get("cold", 0) for r in on)
    bad = {k: v for r in on for k, v in r["fallbacks"].items() if k != "decode"}
    if cold == 0 and not bad:
        return True, "the on arm's first 5 runs: no cold plain continuation, no fallback other than decode"
    return False, f"the on arm's first 5 runs: {cold} cold plain continuations, fallbacks other than decode {bad or '-'}"


def after(archive):
    tmp = tempfile.mkdtemp()
    with tarfile.open(os.path.join(archive, "codex-rollouts.tar.gz")) as tf:
        tf.extractall(tmp, filter="data")
    kinds, comps = {}, 0
    for f in sorted(glob.glob(tmp + "/runs/*/home/.codex/sessions/*/*/*/*.jsonl")):
        L = load_session(f)
        comps += sum(x["comp"] for x in L)
        for n in range(1, len(L)):
            k = kind(L, n)
            kinds[k] = kinds.get(k, 0) + 1
    print(f"after run: follow-ups {kinds}, compactions {comps}")
    return kinds, comps


def main():
    a = sys.argv[1:]
    if a and a[0] == "--after":
        kinds, comps = after(a[1])
        return 0 if kinds.get("cold") == 15 and comps == 20 else 1
    out_dir = a[0]
    arms = count(out_dir)
    show(arms)
    if "--stop-check" in a:
        clear, why = stop_clear(arms)
        print(("STOP: " if clear else "CONTINUE: ") + why)
        return 10 if clear else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
