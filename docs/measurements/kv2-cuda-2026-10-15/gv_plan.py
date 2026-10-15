#!/usr/bin/env python3
"""GV-C(a) (docs/measurements/kv2-cuda-2026-10-15.md): KV2 leaves the CUDA plan unchanged. For each load, start serve from
the base binary and from the branch's, read the banner until the server answers, and compare every line before
"goinfer serving" (the CUDA backend's plan lines, serve's context, fit, KV plan, session reuse and concurrency lines), with
free-VRAM, host-memory and timing figures blanked and the page-cache read notice dropped: none of them is the plan. Run under the timing lock:

  python3 scripts/timing_lock.py run --label kv2-gv -- python3 gv_plan.py OUT.txt BASE_BIN REV_BIN LOAD...

Each LOAD is NAME=PATH[,extra serve flags separated by commas]. It prints one verdict per load and exits 1 if any differs
or is void (a server that does not come up, or a banner without the CUDA plan)."""
import os, re, signal, socket, subprocess, sys, time

PORT = 18871
DROP = re.compile(r"is not in memory: reading|^swap guard")
NOISE = [
    (re.compile(r"budget [\d.]+ GB, [\d.]+ GB left"), "budget <host>"),
    (re.compile(r"\d+(\.\d+)? ?(MB|MiB|GB|GiB) (of [\d.]+ ?(MB|MiB|GB|GiB) )?free"), "<free>"),
    (re.compile(r"free[^,;)]*?\d+(\.\d+)? ?(MB|MiB|GB|GiB)"), "free <n>"),
    (re.compile(r"\d+(\.\d+)? ?(ms|s)\b"), "<t>"),
    (re.compile(r"127\.0\.0\.1:\d+"), "<addr>"),
]


def up(proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            socket.create_connection(("127.0.0.1", PORT), 1).close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def banner(binary, path, extra, log):
    argv = [binary, "-model", f"bench={path}", "-backend", "cuda", "-require-backend", "-addr", f"127.0.0.1:{PORT}"] + extra
    with open(log, "w") as sink:
        sink.write("argv: " + " ".join(argv) + "\n")
        sink.flush()
        proc = subprocess.Popen(argv, stdout=sink, stderr=sink, preexec_fn=os.setsid)
        try:
            ok = up(proc, 900)
            time.sleep(1)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            time.sleep(2)
    lines = []
    for ln in open(log).read().splitlines()[1:]:
        if ln.startswith("goinfer serving"):
            break
        if DROP.search(ln):
            continue
        for rx, rep in NOISE:
            ln = rx.sub(rep, ln)
        lines.append(ln)
    return ok, lines


def main():
    out, base, rev, loads = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    bad = False
    with open(out, "w") as f:
        def say(s):
            print(s, flush=True)
            f.write(s + "\n")
        say(f"# GV-C(a), {time.strftime('%F %T %Z')}; base {base}, rev {rev}")
        for ld in loads:
            name, spec = ld.split("=", 1)
            path, *extra = spec.split(",")
            logs = {k: f"{out}.{name}.{k}.log" for k in ("base", "rev")}
            okb, lb = banner(base, path, extra, logs["base"])
            okr, lr = banner(rev, path, extra, logs["rev"])
            plan = [x for x in lb if "KV plan:" in x or "context:" in x]
            if not (okb and okr) or not plan:
                say(f"== {name}: VOID (base up {okb}, rev up {okr}, {len(plan)} context/KV plan lines)")
                bad = True
            elif lb == lr:
                say(f"== {name}: SAME ({len(lb)} plan lines)")
            else:
                say(f"== {name}: DIFFERS")
                bad = True
            for x in lb:
                say(("   " if x in lr else " - ") + x)
            for x in lr:
                if x not in lb:
                    say(" + " + x)
    sys.exit(1 if bad else 0)


main()
