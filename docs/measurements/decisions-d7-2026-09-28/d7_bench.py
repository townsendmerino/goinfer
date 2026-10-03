#!/usr/bin/env python3
"""D7: decisions against constrained generation, speed. The harness for the pre-registration in ../decisions-d7-2026-09-28.md §4.
Stdlib only. Three subcommands:

  build    start the pinned server, build the frozen prompt file (8 states per K, calibrated to K state tokens with goinfer's own count)
  run      start the pinned server, check the decode path is resident, run the registered schedule, append raw rows
  analyze  the registered ratios (per-state paired, geometric mean, t-interval on the logs), the band comparison, D8's trigger share

  GOINFER_SERVE_BIN=<pinned serve-cuda> D7_MODEL=~/models/Qwen3.5-9B-Q4_K_M.gguf python3 d7_bench.py build --out prompts.json
  ... python3 d7_bench.py run --prompts prompts.json --raw raw.jsonl
  python3 d7_bench.py analyze --raw raw.jsonl

--smoke (build and run): K=256 only, 2 states. EXPLORATORY: it tests the machinery and is never quoted. The registered run uses neither --smoke nor
--max-load above its default; a run with --max-load raised or a CPU decode path is void by the pre-registration and the harness says so.
"""
import argparse, hashlib, json, math, os, random, signal, socket, statistics, subprocess, sys, time, urllib.request, urllib.error

PORT = int(os.environ.get("D7_PORT", "18171"))
BASE = f"http://127.0.0.1:{PORT}"
KS = [256, 1024, 4096]
K_SHORT = 32          # D8's trigger cell
N_STATES = 8
SINGLE_ARMS = ["decision", "schema", "tool"]
KIND_ROT = ["noul", "score", "choice"]   # one question per state, rotating

# ---- the questions (fixed text; the same wording reaches every arm) -------------------------------------------------------------------------
Q = {
    "noul": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "score": {"type": "score", "instructions": "How frustrated is the customer?",
              "criteria": ["calm", "mildly annoyed", "annoyed", "frustrated", "very frustrated", "furious"]},   # an ordered ARRAY (the 422 on an object)
    "choice": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": "New purchases", "account": "Logins, profile changes"}},
    "noul2": {"type": "noul", "instructions": "Does the customer mention a deadline?"},
    "choice2": {"type": "choice", "instructions": "How should the customer be contacted?",
                "criteria": {"email": "Written reply", "phone": "A call", "chat": "Live chat", "none": "No reply needed"}},
}
FIVE = ["noul", "score", "choice", "noul2", "choice2"]


def q_text(name):
    q = Q[name]
    t = q["instructions"]
    if isinstance(q.get("criteria"), list):    # score: levels 0..n-1, in order
        t += " Levels: " + "; ".join(f"{i} ({v})" for i, v in enumerate(q["criteria"])) + "."
    elif "criteria" in q:
        t += " Options: " + "; ".join(f"{k} ({v})" for k, v in q["criteria"].items()) + "."
    elif q["type"] == "noul":
        t += " Answer true or false."
    return t


def q_schema(name):
    q = Q[name]
    if q["type"] == "noul":
        return {"type": "boolean"}
    if q["type"] == "score":
        return {"type": "integer", "enum": list(range(len(q["criteria"])))}
    return {"type": "string", "enum": list(q["criteria"])}


# ---- requests --------------------------------------------------------------------------------------------------------------------------------
def req_decision(state, names):
    return "/v1/systemone", {"model": "bench", "state": state, "questions": {n: Q[n] for n in names}}


def req_schema(state, names):
    props = {n: q_schema(n) for n in names}
    body = {"model": "bench", "temperature": 0, "max_tokens": 48 + 40 * len(names),
            "messages": [{"role": "user", "content": state + "\n\n" + "\n".join(f"{i+1}. {q_text(n)}" for i, n in enumerate(names)) if len(names) > 1 else state + "\n\n" + q_text(names[0])}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "d7", "strict": True, "schema": {
                "type": "object", "properties": props if len(names) > 1 else {"answer": props[names[0]]},
                "required": list(props) if len(names) > 1 else ["answer"], "additionalProperties": False}}}}
    return "/v1/chat/completions", body


def req_tool(state, name):
    body = {"model": "bench", "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": state + "\n\n" + q_text(name)}],
            "tools": [{"type": "function", "function": {"name": "decide", "description": "Record the answer to the question.",
                       "parameters": {"type": "object", "properties": {"answer": q_schema(name)}, "required": ["answer"]}}}],
            "tool_choice": "required"}
    return "/v1/chat/completions", body


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return time.perf_counter() - t0, json.loads(raw), None
    except urllib.error.HTTPError as e:
        return time.perf_counter() - t0, None, f"HTTP {e.code}: {e.read()[:200]!r}"


def usage_of(path, resp):
    u = resp.get("usage") or {}
    return (u.get("input_tokens"), u.get("output_tokens")) if path == "/v1/systemone" else (u.get("prompt_tokens"), u.get("completion_tokens"))


def valid(arm, resp, names):
    """Did the arm actually answer every question in a form that can be read? An invalid answer is recorded, never silently dropped."""
    try:
        if arm.startswith("decision"):
            a = resp["answers"]
            return all(n in a for n in names)
        if arm.startswith("schema"):
            d = json.loads(resp["choices"][0]["message"]["content"])
            return all(n in d for n in names) if len(names) > 1 else "answer" in d
        d = json.loads(resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        return "answer" in d
    except Exception:
        return False


# ---- the server ------------------------------------------------------------------------------------------------------------------------------
class Server:
    def __init__(self, binary, model, log, ctxs=(8192, 5120)):
        self.binary, self.model, self.log, self.ctxs = binary, model, log, list(ctxs)
        self.proc, self.ctx, self.declines = None, None, []

    def __enter__(self):
        for ctx in self.ctxs:
            argv = [self.binary, "-model", f"bench={self.model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-ctx", str(ctx)]
            sink = open(self.log, "a")
            sink.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n"); sink.flush()
            self.proc = subprocess.Popen(argv, stdout=sink, stderr=sink, start_new_session=True)
            t0 = time.time()
            while time.time() - t0 < 900:
                if self.proc.poll() is not None:
                    break
                try:
                    with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                        break
                except OSError:
                    time.sleep(0.5)
            text = open(self.log).read().split(f"==== ")[-1]
            if self.proc.poll() is None and "decode path: cuda-resident" in text:
                self.ctx = ctx
                return self
            self.declines.append((ctx, "exited" if self.proc.poll() is not None else "not resident"))
            self.__exit__()
        raise SystemExit(f"VOID: no resident decode path at any of ctx {self.ctxs}: {self.declines} (a CPU fallback voids the run)")

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        time.sleep(2)


def idle_gate(max_load, waited_max=600):
    t0 = time.time()
    while True:
        l1 = float(open("/proc/loadavg").read().split()[0])
        if l1 <= max_load:
            return l1
        if time.time() - t0 > waited_max:
            raise SystemExit(f"NOT IDLE after {waited_max}s (load1 {l1} > {max_load}): stopping, the numbers would be worthless")
        time.sleep(5)


# ---- the states ------------------------------------------------------------------------------------------------------------------------------
PRODUCTS = ["Orbit Pay", "Lumen Billing", "Harbor CRM", "Pixel Sync", "Cobalt Ledger", "Fable Desk"]
ISSUES = ["the payout to my bank account failed again", "I was charged twice for the same invoice", "the dashboard shows an error when I export data",
          "my login link expires before I can use it", "the mobile app crashes when I open the reports tab", "the monthly statement does not match the transactions",
          "I cannot change the email address on my profile", "the integration stopped syncing contacts overnight"]
DETAILS = ["It started last {day} and has happened {n} times since.", "I already tried clearing the cache and signing in again.", "My account number ends in {num}.",
           "Our finance team needs this sorted before the {day} close.", "I spoke to someone on the phone on {day} but nothing changed.",
           "The error code on the screen was E-{num}.", "This affects {n} of our team members.", "I have attached the screenshots to the previous message.",
           "Please let me know what you need from my side.", "We are on the {plan} plan and have been customers for {n} years."]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PLANS = ["Starter", "Team", "Business", "Enterprise"]


def gen_sentence(rnd):
    return rnd.choice(DETAILS).format(day=rnd.choice(DAYS), n=rnd.randint(2, 9), num=rnd.randint(1000, 9999), plan=rnd.choice(PLANS))


def make_state(seed, nsent):
    """Support-ticket-like prose, deterministic in (seed, nsent), distinct per seed from its first words on (no shared prefix)."""
    rnd = random.Random(seed)
    head = f"Ticket {rnd.randint(100000, 999999)} about {rnd.choice(PRODUCTS)}: {rnd.choice(ISSUES)}."
    return head + " " + " ".join(gen_sentence(rnd) for _ in range(nsent))


def state_tokens(measure, state, base):
    return measure(state) - base + 1


def calibrate(K, seed, measure, base):
    """Add or remove sentences, then trailing words, until the state is within 2 percent of K state tokens (at least 3) by goinfer's own count."""
    tol = max(3, int(0.02 * K))
    n = max(1, K // 17)
    for _ in range(6):
        s = make_state(seed, n)
        cur = state_tokens(measure, s, base)
        if abs(cur - K) <= tol:
            return s, cur
        per = max(1.0, cur / max(n, 1))
        n = max(1, n + round((K - cur) / per))
    words = s.split(" ")
    for _ in range(6):
        cur = state_tokens(measure, " ".join(words), base)
        if abs(cur - K) <= tol:
            return " ".join(words), cur
        tpw = max(0.5, cur / len(words))
        step = round((K - cur) / tpw)
        words = words[:max(5, len(words) + step)] if step < 0 else words + make_state(seed + 7919, 40).split(" ")[:step]
    return " ".join(words), state_tokens(measure, " ".join(words), base)


def cmd_build(a):
    ks = [256] if a.smoke else KS
    nst = 2 if a.smoke else N_STATES
    with Server(a.serve, a.model, a.log, ctxs=(8192, 5120)) as srv:
        def measure(state):
            _, r, err = post("/v1/systemone", {"model": "bench", "state": state, "questions": {"q": Q["noul"]}})
            if err:
                raise SystemExit("calibration request failed: " + err)
            return r["usage"]["input_tokens"]
        base = measure(".")
        out = {"model": os.path.basename(a.model), "serve": os.path.basename(a.serve), "ctx": srv.ctx, "tolerance": "2 percent of K, at least 3 tokens",
               "built": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke, "states": {}}
        for K in ks + ([] if a.smoke else [K_SHORT]):
            out["states"][str(K)] = []
            for i in range(nst):
                s, got = calibrate(K, 1000 * K + i, measure, base)
                out["states"][str(K)].append({"seed": 1000 * K + i, "state_tokens": got, "text": s})
                print(f"[d7 build] K={K} state {i}: {got} state tokens", flush=True)
    body = json.dumps(out, indent=1, sort_keys=True)
    open(a.out, "w").write(body)
    print(f"[d7 build] wrote {a.out}  sha256 {hashlib.sha256(body.encode()).hexdigest()}")


# ---- the schedule ----------------------------------------------------------------------------------------------------------------------------
def one(arm, state, names):
    if arm == "decision":
        path, body = req_decision(state, names)
    elif arm == "schema":
        path, body = req_schema(state, names)
    elif arm == "tool":
        path, body = req_tool(state, names[0])
    elif arm == "decision5":
        path, body = req_decision(state, FIVE)
    else:
        path, body = req_schema(state, FIVE)
    t, resp, err = post(path, body)
    if err or resp is None:
        return {"arm": arm, "t": round(t, 4), "ok": False, "err": err}
    i, o = usage_of(path, resp)
    return {"arm": arm, "t": round(t, 4), "in": i, "out": o, "ok": valid(arm, resp, FIVE if arm.endswith("5") else names)}


def cmd_run(a):
    P = json.load(open(a.prompts))
    ks = [int(k) for k in P["states"] if int(k) != K_SHORT]
    if a.max_load > 1.0 and not a.smoke:
        print(f"[d7] NOTE: --max-load {a.max_load} is above the registered 1.0: this run is VOID as a result", flush=True)
    with Server(a.serve, a.model, a.log) as srv:
        meta = {"serve": os.path.basename(a.serve), "ctx": srv.ctx, "declines": srv.declines, "prompts_sha256": hashlib.sha256(open(a.prompts, "rb").read()).hexdigest(),
                "host": os.uname().nodename, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke, "max_load": a.max_load,
                "load_at_start": open("/proc/loadavg").read().split()[:3]}
        out = open(a.raw, "a")
        out.write(json.dumps({"meta": meta}) + "\n"); out.flush()
        # warm-up: 2 requests per arm, discarded
        w = P["states"][str(ks[0])][0]["text"]
        for arm in SINGLE_ARMS + ["decision5", "schema5"]:
            for _ in range(2):
                one(arm, w, [KIND_ROT[0]])
        t_all = time.time(); done = 0
        total = sum(len(P["states"][str(k)]) * 5 for k in ks) + len(P["states"].get(str(K_SHORT), [])) * 2
        cells = [(k, False) for k in ks] + ([(K_SHORT, True)] if str(K_SHORT) in P["states"] else [])
        for K, short in cells:
            for si, st in enumerate(P["states"][str(K)]):
                l1 = idle_gate(a.max_load)
                kind = KIND_ROT[si % 3]
                names = [kind]
                rot = SINGLE_ARMS[si % 3:] + SINGLE_ARMS[:si % 3]            # 3x3 Latin square over consecutive states
                five = ["decision5", "schema5"] if si % 2 == 0 else ["schema5", "decision5"]  # AB / BA by state
                plan = [] if short else [(arm, names) for arm in rot]
                plan += [(arm, FIVE) for arm in five] if not short else [("decision5", FIVE)]
                for arm, nm in plan:
                    r = one(arm, st["text"], nm)
                    r.update({"K": K, "state": si, "kind": kind, "state_tokens": st["state_tokens"], "load1": l1})
                    out.write(json.dumps(r) + "\n"); out.flush()
                    done += 1
                    print(f"[d7] K={K} state {si} {arm:10s} {r['t']:8.2f} s in={r.get('in')} out={r.get('out')} ok={r['ok']}  ({done}/{total}, {time.time()-t_all:.0f} s elapsed)", flush=True)
    print("[d7] done")


# ---- the analysis ----------------------------------------------------------------------------------------------------------------------------
# Projected bands (docs/measurements/decisions-d7-2026-09-28.md §3): ratio = arm time / decision time for single questions, decision x5 / schema x5 for five.
BANDS = {("schema", 256): (1.11, 1.11), ("schema", 1024): (1.03, 1.03), ("schema", 4096): (1.01, 1.01),
         ("tool", 256): (1.88, 2.00), ("tool", 1024): (1.22, 1.25), ("tool", 4096): (1.06, 1.06),
         ("five", 256): (3.2, 3.4), ("five", 1024): (4.4, 4.5), ("five", 4096): (4.8, 4.9)}
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def gm_interval(ratios):
    logs = [math.log(r) for r in ratios]
    n = len(logs)
    m = statistics.mean(logs)
    if n < 2:
        return math.exp(m), None, None
    se = statistics.stdev(logs) / math.sqrt(n)
    t = TCRIT.get(n - 1, 1.96)
    return math.exp(m), math.exp(m - t * se), math.exp(m + t * se)


def cmd_analyze(a):
    rows = [json.loads(l) for l in open(a.raw)]
    metas = [r["meta"] for r in rows if "meta" in r]
    rows = [r for r in rows if "arm" in r]
    by = {}
    for r in rows:
        by.setdefault((r["K"], r["state"]), {})[r["arm"]] = r
    invalid = [(r["K"], r["state"], r["arm"]) for r in rows if not r["ok"]]
    print(f"{len(rows)} rows from {len(metas)} run(s); smoke={[m.get('smoke') for m in metas]}; invalid or failed: {len(invalid)} {invalid[:6]}")
    if any(m.get("smoke") for m in metas):
        print("EXPLORATORY (smoke): not quotable")
    print("\nratio = arm time / decision time (single) or decision x5 / schema x5 (five); geometric mean over states, 95% t-interval; band = projected, widened +/-10%")
    print(f"{'cell':14s} {'n':>2s} {'ratio':>7s} {'95% interval':>17s}  {'band (x0.9..x1.1)':>19s}  verdict")
    for K in sorted({r["K"] for r in rows if r["K"] != K_SHORT}):
        for kind, num, den in (("schema", "schema", "decision"), ("tool", "tool", "decision"), ("five", "decision5", "schema5")):
            rs = [c[num]["t"] / c[den]["t"] for (k, s), c in sorted(by.items()) if k == K and num in c and den in c and c[num]["ok"] and c[den]["ok"]]
            if not rs:
                continue
            g, lo, hi = gm_interval(rs)
            band = BANDS[(kind, K)]
            b0, b1 = band[0] * 0.9, band[1] * 1.1
            verdict = "n<2" if lo is None else ("as projected" if (hi >= b0 and lo <= b1) else "OFF PROJECTION: name the wrong input before quoting")
            print(f"{kind+' K='+str(K):14s} {len(rs):2d} {g:7.2f} {('['+format(lo,'.2f')+', '+format(hi,'.2f')+']') if lo else '-':>17s}  {format(b0,'.2f')+'..'+format(b1,'.2f'):>19s}  {verdict}")
    t32 = [c["decision5"]["t"] for (k, s), c in by.items() if k == K_SHORT and "decision5" in c and c["decision5"]["ok"]]
    t1024 = [c["decision5"]["t"] for (k, s), c in by.items() if k == 1024 and "decision5" in c and c["decision5"]["ok"]]
    if t32 and t1024:
        share = 1 - statistics.mean(t32) / statistics.mean(t1024)
        print(f"\nD8 trigger: state-prefill share of a five-question decision request at K=1024 = 1 - {statistics.mean(t32):.2f} s / {statistics.mean(t1024):.2f} s = {share:.3f}; fires if >= 0.70: {'FIRES' if share >= 0.70 else 'does not fire'}")
    for K in sorted({r["K"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["K"] == K}):
            ts = [r["t"] for r in rows if r["K"] == K and r["arm"] == arm and r["ok"]]
            toks = [r.get("in") for r in rows if r["K"] == K and r["arm"] == arm and r["ok"]]
            if ts:
                print(f"  K={K:5d} {arm:10s} n={len(ts)} mean {statistics.mean(ts):7.2f} s  input tokens ~{statistics.mean(toks):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run"):
        p = sub.add_parser(name)
        p.add_argument("--serve", default=os.environ.get("GOINFER_SERVE_BIN", ""))
        p.add_argument("--model", default=os.environ.get("D7_MODEL", os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf")))
        p.add_argument("--log", default="d7-servers.log")
        p.add_argument("--smoke", action="store_true")
    sub.choices["build"].add_argument("--out", required=True)
    r = sub.choices["run"]
    r.add_argument("--prompts", required=True)
    r.add_argument("--raw", required=True)
    r.add_argument("--max-load", type=float, default=float(os.environ.get("BENCH_MAX_LOADAVG", "1.0")))
    an = sub.add_parser("analyze")
    an.add_argument("--raw", required=True)
    a = ap.parse_args()
    if a.cmd in ("build", "run"):
        if not a.serve or not os.access(a.serve, os.X_OK):
            sys.exit("set GOINFER_SERVE_BIN (or --serve) to the pinned serve-cuda binary")
        if a.model.startswith(("/srv/models", "/Volumes")):
            sys.exit("the model is on the archive, not the bench set")
    {"build": cmd_build, "run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
