#!/usr/bin/env python3
"""Metal KV8a gate M5 with M4(b) (README.md here; Francis 2026-10-21 10:11 PT: the paired two-build replay and its bar). For each of the
16 opencode and Codex pilot pairs, on each build in turn, a fresh server: turn 1 (greedy, its recorded first token forbidden, as M2(b)),
then turn 2, whose time to first token is serve's own request-log `ttft`. Pass 1 runs before-KV8a then KV8a per pair; pass 2 the
reverse. The reading is the median over pairs and passes of (KV8a turn 2 ttft) / (before turn 2 ttft): PASS <= 0.5, PARKED 0.5 to 0.7,
FAIL above 0.7. A pair whose KV8a turn 2 did not restore at turn 1's end, or whose before turn 2 reused anything, is void. M4(b): the
p95 of `ckpt_save_ms` over every KV8a generating request, bar 12 ms.

  python3 m5.py --before <serve> --kv8a <serve> --model <giw> --key <ids key> --bodies <bodies-pilot> --ids <ids.jsonl> --out <dir> [--pass 1|2]

--pass runs one pass and records it as <out>/m5-pass<N>.json (two night jobs, each under the 3-hour cap); the run that finds both passes'
records in <out> grades them. Exit 0 PASS (or one pass recorded), 1 FAIL, 3 PARKED, 2 VOID.
"""
import argparse, atexit, glob, json, os, signal, statistics, subprocess, sys, time, urllib.request

PORT = 18209
BASE = f"http://127.0.0.1:{PORT}"
SESSIONS = ["pilot-qwen35-9b-20k-opencode-chain-r1", "pilot-qwen35-9b-20k-codex-chain-r1"]
CTX, MAX_NEW, MIN_PAIRS = 20480, 24, 10


_LIVE = []  # servers started and not yet closed: stopped at exit, so a crash cannot leave one holding the model and the port


@atexit.register
def _stop_live():
    for p in _LIVE:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kv(line):
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


class Server:
    def __init__(self, binary, a, tag):
        self.logpath = os.path.join(a.out, f"serve-{tag}.log")
        self.log = open(self.logpath, "a")
        argv = [binary, "--model", a.model, "--backend", "metal", "--ctx", str(CTX), "--addr", f"127.0.0.1:{PORT}", "--kv-sessions", "1",
                "--ids-key-file", a.key, "--log-requests"]
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        _LIVE.append(self.proc)
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                self.log.flush()
                path = [ln for ln in open(self.logpath, errors="replace") if "decode path:" in ln][-1:]
                if not path or "metal-resident" not in path[0]:
                    self.close()
                    sys.exit(f"VOID: {tag} is not Metal-resident: {path[0].strip() if path else 'no decode path line'}")
                self.load_s = time.time() - t0
                return
            except Exception:
                if self.proc.poll() is not None:
                    sys.exit(f"VOID: server exited during load ({tag})")
                time.sleep(0.5)
        sys.exit(f"VOID: server not up in 600 s ({tag})")

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            os.killpg(self.proc.pid, signal.SIGKILL)
        if self.proc in _LIVE:
            _LIVE.remove(self.proc)
        self.log.close()
        time.sleep(3)

    def send(self, rec, ban=None):
        """One recorded body, greedy, MAX_NEW tokens, not streamed; returns serve's request-log fields for it."""
        body = json.loads(rec["body"]) if isinstance(rec["body"], str) else dict(rec["body"])
        path = rec["path"]
        body["stream"] = False
        body.pop("stream_options", None)
        body["temperature"] = 0
        body.pop("logit_bias", None)
        if ban is not None:
            body["logit_bias"] = {str(ban): -100}
        body["max_output_tokens" if path.endswith("/responses") else "max_tokens"] = MAX_NEW
        n0 = sum(1 for _ in open(self.logpath, errors="replace"))
        urllib.request.urlopen(urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=1800).read()
        self.log.flush()
        time.sleep(0.2)
        lines = [ln for ln in list(open(self.logpath, errors="replace"))[n0:] if ln.startswith("request: POST") and " status=200 " in ln]
        if len(lines) != 1:
            sys.exit(f"VOID: {len(lines)} request lines logged for one request on {path}")
        return kv(lines[0])


def ms(v):
    return float(v.rstrip("ms")) if v and v not in ("-",) else None


def main():
    ap = argparse.ArgumentParser()
    for k in ("before", "kv8a", "model", "key", "bodies", "ids", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--pass", dest="only", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = {os.path.realpath(r["file"]): r["ids"] for r in map(json.loads, open(a.ids)) if r.get("ids")}
    pairs = []
    for s in SESSIONS:
        fs = sorted(glob.glob(os.path.join(a.bodies, s, "*.json")))
        pairs += [(s, i, fs[i], fs[i + 1]) for i in range(len(fs) - 1)]
    pairs = [p for p in pairs if not (p[0].endswith("opencode-chain-r1") and p[1] == 0)]  # opencode #0 is the title request's pair
    say(f"M5: {len(pairs)} pairs x 2 builds x 2 passes")
    rows, saves, t_all = [], [], time.time()
    for pas, order in ((1, ("before", "kv8a")), (2, ("kv8a", "before"))):
        if a.only and pas != a.only:
            continue
        for s, i, f1, f2 in pairs:
            t1rec, t2rec = json.load(open(f1)), json.load(open(f2))
            i1, i2 = ids.get(os.path.realpath(f1)), ids.get(os.path.realpath(f2))
            ban = i2[len(i1)] if i1 and i2 and len(i2) > len(i1) and i2[:len(i1)] == i1 else None
            r = {"pass": pas, "session": s, "pair": i, "ban": ban}
            for b in order:
                srv = Server(getattr(a, b), a, f"p{pas}-{s[-14:]}-{i}-{b}")
                l1 = srv.send(t1rec, ban)
                l2 = srv.send(t2rec)
                srv.close()
                r[b] = {"load_s": round(srv.load_s, 1), "t1_tokens": int(l1["prompt_tokens"]), "t1_ttft_ms": ms(l1.get("ttft")),
                        "t2_tokens": int(l2["prompt_tokens"]), "t2_ttft_ms": ms(l2.get("ttft")), "t2_reused": int(l2.get("reused", "0")),
                        "t2_restore": l2.get("ckpt_restore")}
                if b == "kv8a":
                    saves += [float(x["ckpt_save_ms"]) for x in (l1, l2) if "ckpt_save_ms" in x]
            k, o = r["kv8a"], r["before"]
            r["void"] = (None if k["t2_restore"] == str(k["t1_tokens"]) and o["t2_reused"] == 0
                         else f"kv8a restore {k['t2_restore']} at turn 1 {k['t1_tokens']}; before reused {o['t2_reused']}")
            r["ratio"] = round(k["t2_ttft_ms"] / o["t2_ttft_ms"], 4) if not r["void"] and k["t2_ttft_ms"] and o["t2_ttft_ms"] else None
            rows.append(r)
            say(f"  pass {pas} {s[-14:]}#{i}: before {o['t2_ttft_ms']} ms, kv8a {k['t2_ttft_ms']} ms (restore {k['t2_restore']}), ratio {r['ratio']}"
                + (f", VOID: {r['void']}" if r["void"] else ""))
            json.dump({"rows": [r for r in rows if r["pass"] == pas], "saves_ms": saves},
                      open(os.path.join(a.out, f"m5-pass{pas}.json"), "w"), indent=1)
    rows, saves = [], []
    for pas in (1, 2):
        f = os.path.join(a.out, f"m5-pass{pas}.json")
        if not os.path.exists(f):
            say(f"pass {pas} not recorded yet in {a.out}: grading waits for it")
            return 0
        d = json.load(open(f))
        rows, saves = rows + d["rows"], saves + d["saves_ms"]
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    say(f"done in {time.time() - t_all:.0f} s: {len(ratios)} graded of {len(rows)}")
    if len(ratios) < 2 * MIN_PAIRS or not saves:
        print(f"VERDICT VOID ({len(ratios)} graded ratios, {len(saves)} saves)")
        return 2
    med = statistics.median(ratios)
    per = {p: statistics.median([r["ratio"] for r in rows if r["ratio"] is not None and r["pass"] == p]) for p in (1, 2)}
    p95 = statistics.quantiles(saves, n=20)[-1] if len(saves) >= 20 else max(saves)
    print(f"M5: median paired ratio {med:.3f} over {len(ratios)} (pass 1 {per[1]:.3f}, pass 2 {per[2]:.3f}); bar <= 0.5, parked 0.5-0.7")
    print(f"M4(b): save p95 {p95:.3f} ms over {len(saves)} saves (median {statistics.median(saves):.3f}, max {max(saves):.3f}); bar 12 ms")
    v5 = "PASS" if med <= 0.5 else ("PARKED" if med <= 0.7 else "FAIL")
    v4 = "PASS" if p95 <= 12 else "FAIL"
    print(f"VERDICT M5 {v5}, M4(b) {v4}")
    return 1 if "FAIL" in (v5, v4) else (3 if v5 == "PARKED" else 0)


if __name__ == "__main__":
    sys.exit(main())
