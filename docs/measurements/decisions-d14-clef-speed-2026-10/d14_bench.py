#!/usr/bin/env python3
"""D14: Clef-flash against JEV-9B, speed, 1 and 5 questions about one state. The harness for the pre-registration in ../decisions-d14-clef-speed-2026-10-03.md.
Stdlib only. Reuses D7's frozen prompts and question wording (../decisions-d7-2026-09-28/), so every arm reads the same states and the same questions.

  run      for each registered pass, start the pinned server on one model at a time, warm it up, run the schedule, append raw rows
  analyze  the registered ratios (per-state paired, geometric mean, t-interval on the logs), the band comparison, and the D8 rule

  GOINFER_SERVE_BIN=<pinned serve> D14_JEV=~/models/JEV-9B D14_CLEF=~/models/clef-flash python3 d14_bench.py run --raw raw.jsonl --log servers.log
  python3 d14_bench.py analyze --raw raw.jsonl

ONE 9B model is resident at a time (two do not fit a 16 GB machine), so the two arms cannot be interleaved request by request. They are blocked and the order is reversed in the
second pass (pass 1: JEV block then Clef block; pass 2: Clef block then JEV block), which cancels a drift that runs one way across the night; analyze reports the two passes' ratios
side by side so a drift the order did not cancel is visible. Every request is a POST /v1/systemone with the same body to both models; only the route differs (JEV: the head route, one
prefill per question; Clef: one backbone pass for the whole record).

--smoke: K=256 only, 1 state, one pass. EXPLORATORY: it tests the machinery and is never quoted. A run with --max-busy raised, a non-CPU decode path or an unexpected route is VOID.
"""
import argparse, hashlib, json, math, os, signal, socket, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("D7_PORT", os.environ.get("D14_PORT", "18181"))   # d7_bench reads its port at import
sys.path.insert(0, os.path.join(HERE, "..", "decisions-d7-2026-09-28"))
import d7_bench as d7  # noqa: E402  (Q, FIVE, KIND_ROT, post, BASE: the same wording and transport as D7)

PORT = int(os.environ["D7_PORT"])
PROMPTS = os.path.join(HERE, "..", "decisions-d7-2026-09-28", "prompts.json")
PROMPTS_SHA = "2d56d87f800e2b7235bbeb9e1148039daee1e3d22c53a503ea26711ba0910dd9"
EXPECT_ROUTE = {"jev": "head", "clef": "clef"}


# ---- idle gate -------------------------------------------------------------------------------------------------------------------------------
def busy_percent():
    """Share of all CPUs busy right now: top on darwin (the instant gate, TE1), 1-minute load per CPU on Linux."""
    if sys.platform == "darwin":
        out = subprocess.run(["top", "-l", "2", "-s", "1", "-n", "0"], capture_output=True, text=True, timeout=30).stdout
        line = [l for l in out.splitlines() if l.startswith("CPU usage")][-1]          # the second sample: the first is since boot
        idle = float(line.split("idle")[0].split(",")[-1].strip().rstrip("% ").strip())
        return 100.0 - idle
    return 100.0 * float(open("/proc/loadavg").read().split()[0]) / (os.cpu_count() or 1)


def idle_gate(max_busy, waited_max=900):
    t0 = time.time()
    while True:
        b = busy_percent()
        if b <= max_busy:
            return round(b, 1)
        if time.time() - t0 > waited_max:
            raise SystemExit(f"NOT IDLE after {waited_max}s (busy {b:.0f}% > {max_busy}%): stopping, the numbers would be worthless")
        time.sleep(5)


# ---- the server ------------------------------------------------------------------------------------------------------------------------------
class Server:
    def __init__(self, binary, name, entry, log, backend, quant):
        self.binary, self.name, self.entry, self.log, self.backend, self.quant = binary, name, entry, log, backend, quant
        self.proc, self.path_line = None, ""

    def __enter__(self):
        argv = [self.binary, "-model", f"{self.name}={self.entry}", "-backend", self.backend, "-quant", self.quant, "-addr", f"127.0.0.1:{PORT}", "-ctx", "8192"]
        sink = open(self.log, "a")
        sink.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n"); sink.flush()
        self.proc = subprocess.Popen(argv, stdout=sink, stderr=sink, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 1800:
            if self.proc.poll() is not None:
                raise SystemExit(f"VOID: the server for {self.name} exited during load; see {self.log}")
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        text = open(self.log).read().split("==== ")[-1]
        self.path_line = next((l.strip() for l in text.splitlines() if "decode path" in l), "")
        if self.backend == "cpu" and any(w in self.path_line for w in ("resident", "metal", "cuda")) and "cpu" not in self.path_line:
            raise SystemExit(f"VOID: {self.name} did not run on the CPU path: {self.path_line!r}")
        return self

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        time.sleep(3)


def ask(model, state, names):
    body = {"model": model, "state": state, "questions": {n: d7.Q[n] for n in names}}
    t, resp, err = d7.post("/v1/systemone", body)
    if err or resp is None:
        return {"t": round(t, 4), "ok": False, "err": err}
    a = resp.get("answers") or {}
    route = (resp.get("goinfer") or {}).get("route")
    return {"t": round(t, 4), "in": (resp.get("usage") or {}).get("input_tokens"), "route": route,
            "ok": all(n in a for n in names) and route == EXPECT_ROUTE[model]}


def cmd_run(a):
    P = json.load(open(PROMPTS))
    sha = hashlib.sha256(open(PROMPTS, "rb").read()).hexdigest()
    if sha != PROMPTS_SHA:
        raise SystemExit(f"prompts.json sha256 {sha} is not the registered {PROMPTS_SHA}")
    ks = [256] if a.smoke else [int(k) for k in a.ks.split(",")]       # --smoke is K=256 only, as the docstring says (the first smoke ran both cells and took 19 minutes)
    n_states = 1 if a.smoke else a.states
    if a.max_busy > 10 and not a.smoke:
        print(f"[d14] NOTE: --max-busy {a.max_busy} is above the registered 10: this run is VOID as a result", flush=True)
    models = {"jev": (a.jev_dir and f"{a.jev_dir},head={a.jev_dir}"), "clef": a.clef_dir}
    passes = [["jev", "clef"]] if a.smoke else [["jev", "clef"], ["clef", "jev"]]
    out = open(a.raw, "a")
    total = len(passes) * 2 * len(ks) * n_states * 2
    done, t_all = 0, time.time()
    for pi, order in enumerate(passes, 1):
        for name in order:
            idle_gate(a.max_busy)
            with Server(a.serve, name, models[name], a.log, a.backend, a.quant) as srv:
                out.write(json.dumps({"meta": {"pass": pi, "model": name, "decode_path": srv.path_line, "serve": os.path.basename(a.serve), "backend": a.backend, "quant": a.quant,
                                               "prompts_sha256": sha, "host": os.uname().nodename, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke,
                                               "max_busy": a.max_busy, "states": n_states, "ks": ks}}) + "\n"); out.flush()
                w = P["states"][str(ks[0])][0]["text"]
                for _ in range(2):                                              # warm-up, discarded
                    ask(name, w, [d7.KIND_ROT[0]])
                for K in ks:
                    for si, st in enumerate(P["states"][str(K)][:n_states]):
                        busy = idle_gate(a.max_busy)
                        kind = d7.KIND_ROT[si % 3]
                        for shape, names in (("1", [kind]), ("5", d7.FIVE)):
                            r = ask(name, st["text"], names)
                            r.update({"pass": pi, "arm": name, "shape": shape, "K": K, "state": si, "kind": kind, "state_tokens": st["state_tokens"], "busy": busy})
                            out.write(json.dumps(r) + "\n"); out.flush()
                            done += 1
                            print(f"[d14] pass {pi} {name:4s} K={K} state {si} {shape}q {r['t']:8.2f} s in={r.get('in')} ok={r['ok']}  ({done}/{total}, {time.time()-t_all:.0f} s elapsed)", flush=True)
    print("[d14] done")


# ---- the analysis ----------------------------------------------------------------------------------------------------------------------------
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def gm_interval(ratios):
    logs = [math.log(r) for r in ratios]
    n, m = len(logs), statistics.mean(logs)
    if n < 2:
        return math.exp(m), None, None
    se = statistics.stdev(logs) / math.sqrt(n)
    t = TCRIT.get(n - 1, 1.96)
    return math.exp(m), math.exp(m - t * se), math.exp(m + t * se)


def cmd_analyze(a):
    rows = [json.loads(l) for l in open(a.raw)]
    metas = [r["meta"] for r in rows if "meta" in r]
    rows = [r for r in rows if "arm" in r]
    bad = [(r["pass"], r["arm"], r["K"], r["state"], r["shape"]) for r in rows if not r["ok"]]
    print(f"{len(rows)} rows from {len(metas)} block(s); smoke={sorted({m.get('smoke') for m in metas})}; invalid or failed: {len(bad)} {bad[:6]}")
    if any(m.get("smoke") for m in metas):
        print("EXPLORATORY (smoke): not quotable")
    paths = sorted({(m['model'], m['decode_path']) for m in metas})
    print("decode paths:", paths)
    cell = {}
    for r in rows:
        if r["ok"]:
            cell.setdefault((r["K"], r["shape"], r["arm"], r["state"]), {})[r["pass"]] = r["t"]
    Ks = sorted({k[0] for k in cell})
    # BANDS: JEV time / Clef time, from the registered projection (docs/measurements/decisions-d14-clef-speed-2026-10-03.md section 2), widened -20% / +10%.
    # The 5-question values are section 2a's (D8: JEV's same-kind questions share their prefill, 3 prefills for 5), which
    # replaced section 2's 1.89 / 3.37 / 4.44 before any graded run; the 1-question values are unchanged.
    PROJ = {("1", 256): 0.74, ("1", 1024): 0.91, ("1", 4096): 0.97, ("5", 256): 1.12, ("5", 1024): 2.03, ("5", 4096): 2.67}
    print("\nratio = JEV time / Clef time, per state (the geometric mean of that state's times over the passes), geometric mean over states, 95% t-interval;")
    print("> 1 means Clef is faster. band = projected x0.80 .. x1.10.")
    print(f"{'cell':12s} {'n':>2s} {'ratio':>7s} {'95% interval':>17s}  {'band':>13s}  {'pass 1':>7s} {'pass 2':>7s}  verdict")
    growth = {}
    for K in Ks:
        for shape in ("1", "5"):
            rs, per_pass = [], {1: [], 2: []}
            for (k, s, arm, st), tp in cell.items():
                if k != K or s != shape or arm != "jev":
                    continue
                other = cell.get((K, shape, "clef", st))
                if not other:
                    continue
                gm = lambda d: math.exp(statistics.mean(math.log(v) for v in d.values()))
                rs.append(gm(tp) / gm(other))
                for p in (1, 2):
                    if p in tp and p in other:
                        per_pass[p].append(tp[p] / other[p])
            if not rs:
                continue
            g, lo, hi = gm_interval(rs)
            b0, b1 = PROJ[(shape, K)] * 0.80, PROJ[(shape, K)] * 1.10
            verdict = "n<2" if lo is None else ("as projected" if (hi >= b0 and lo <= b1) else "OFF PROJECTION: name the wrong input before quoting")
            pp = [format(math.exp(statistics.mean(math.log(x) for x in per_pass[p])), ".2f") if per_pass[p] else "-" for p in (1, 2)]
            print(f"{shape+'q K='+str(K):12s} {len(rs):2d} {g:7.2f} {('['+format(lo,'.2f')+', '+format(hi,'.2f')+']') if lo else '-':>17s}  {format(b0,'.2f')+'..'+format(b1,'.2f'):>13s}  {pp[0]:>7s} {pp[1]:>7s}  {verdict}")
    print("\nwhat five questions cost over one (the D8 question): time(5q) / time(1q), per state, geometric mean over states")
    for K in Ks:
        for arm in ("jev", "clef"):
            rs = []
            for (k, s, ar, st), tp in cell.items():
                if k == K and s == "5" and ar == arm and (K, "1", arm, st) in cell:
                    gm = lambda d: math.exp(statistics.mean(math.log(v) for v in d.values()))
                    rs.append(gm(tp) / gm(cell[(K, "1", arm, st)]))
            if rs:
                g, lo, hi = gm_interval(rs)
                growth[(K, arm)] = g
                print(f"  K={K:5d} {arm:5s} 5q/1q = {g:5.2f}" + (f"  [{lo:.2f}, {hi:.2f}]" if lo else ""))
    clef_g = [v for (k, ar), v in growth.items() if ar == "clef"]
    if clef_g:
        worst = max(clef_g)
        verdict = "D8 UNNECESSARY for this shape on Clef" if worst <= 2.0 else "AMBIGUOUS: goes to the owner" if worst <= 3.0 else "D8 STILL WANTED"
        print(f"  registered D8 rule (Clef 5q/1q at every K: <= 2.0 unnecessary, <= 3.0 ambiguous, above that wanted): worst {worst:.2f} -> {verdict}")
    print()
    for K in Ks:
        for arm in ("jev", "clef"):
            for shape in ("1", "5"):
                ts = [r["t"] for r in rows if r["ok"] and r["K"] == K and r["arm"] == arm and r["shape"] == shape]
                toks = [r.get("in") for r in rows if r["ok"] and r["K"] == K and r["arm"] == arm and r["shape"] == shape]
                if ts:
                    print(f"  K={K:5d} {arm:5s} {shape}q n={len(ts):2d} mean {statistics.mean(ts):8.2f} s  input tokens ~{statistics.mean(toks):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--serve", default=os.environ.get("GOINFER_SERVE_BIN", ""))
    r.add_argument("--jev-dir", default=os.path.expanduser(os.environ.get("D14_JEV", "~/models/JEV-9B")))
    r.add_argument("--clef-dir", default=os.path.expanduser(os.environ.get("D14_CLEF", "~/models/clef-flash")))
    r.add_argument("--raw", required=True)
    r.add_argument("--log", default="d14-servers.log")
    r.add_argument("--ks", default=os.environ.get("D14_KS", "256,1024"))
    r.add_argument("--states", type=int, default=int(os.environ.get("D14_STATES", "6")))
    r.add_argument("--backend", default="cpu")
    r.add_argument("--quant", default="int8int8")
    r.add_argument("--max-busy", type=float, default=10.0)
    r.add_argument("--smoke", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("--raw", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
