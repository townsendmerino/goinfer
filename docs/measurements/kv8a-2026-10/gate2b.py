#!/usr/bin/env python3
"""KV8a gate 2(b) (README.md here): every consecutive pair of the KV0-H pilot's recorded bodies, turn 1 then turn 2 on a warm server (turn 2
restores the checkpoint at turn 1's end), and turn 2 again on a server with resident reuse off (cold). Equal tokens, and on
/v1/chat/completions equal logprob text, or the gate fails. Writes <out>/gate2b.json (statistics and token ids, no prompt text) and
prints a VERDICT line; exit 0 PASS, 1 FAIL, 2 VOID.

  python3 gate2b.py --serve <bin> --model <gguf> --key <ids key file> --bodies <bodies-pilot dir> --out <dir>
"""
import argparse, glob, json, os, signal, subprocess, sys, time, urllib.request

PORT = 18208
BASE = f"http://127.0.0.1:{PORT}"
CONFIGS = [  # (name, ctx, session dirs under --bodies)
    ("h20", 20480, ["pilot-qwen35-9b-20k-opencode-chain-r1", "pilot-qwen35-9b-20k-codex-chain-r1"]),
    ("cc", 32768, ["pilot-qwen35-9b-28k-cc-claude-code-chain-r1"]),
]
MAX_NEW = 24
MIN_TESTED = 15


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Server:
    def __init__(self, a, ctx, tag, cold):
        self.tokfile = os.path.join(a.out, f"tokens-{tag}.jsonl")
        self.log = open(os.path.join(a.out, f"serve-{tag}.log"), "a")
        env = dict(os.environ)
        if cold:
            env["GOINFER_NO_RESIDENT_REUSE"] = "1"
        argv = [a.serve, "--model", a.model, "--backend", "cuda", "--ctx", str(ctx), "--addr", f"127.0.0.1:{PORT}", "--kv-sessions", "1",
                "--ids-key-file", a.key, "--log-requests", "--log-token-ids", self.tokfile]
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
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
        self.log.close()
        time.sleep(3)

    def send(self, rec, ban=None):
        """Sends one recorded body, greedy, MAX_NEW tokens, not streamed; ban is a token id given logit_bias -100 (forbidden). Returns
        (the token-log line of its generation, logprobs or None)."""
        body = json.loads(rec["body"]) if isinstance(rec["body"], str) else dict(rec["body"])
        path = rec["path"]
        body["stream"] = False
        body.pop("stream_options", None)
        body["temperature"] = 0
        body.pop("logit_bias", None)
        if ban is not None:
            body["logit_bias"] = {str(ban): -100}
        if path.endswith("/responses"):
            body["max_output_tokens"] = MAX_NEW
        else:
            body["max_tokens"] = MAX_NEW
        if path.endswith("/chat/completions"):
            body["logprobs"] = True
        before = os.path.getsize(self.tokfile) if os.path.exists(self.tokfile) else 0
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        raw = urllib.request.urlopen(req, timeout=900).read().decode()
        lp = None
        if path.endswith("/chat/completions"):
            d = json.loads(raw, parse_float=str)
            lp = [[t["token"], t["logprob"]] for t in (d["choices"][0].get("logprobs") or {}).get("content") or []]
        with open(self.tokfile) as f:
            f.seek(before)
            lines = [json.loads(x) for x in f if x.strip()]
        if len(lines) != 1:
            sys.exit(f"VOID: {len(lines)} generations logged for one request on {path}")
        return lines[0], lp


def main():
    ap = argparse.ArgumentParser()
    for k in ("serve", "model", "key", "bodies", "ids", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    ids = {os.path.realpath(r["file"]): r["ids"] for r in map(json.loads, open(a.ids)) if r.get("ids")}  # KV0-H's rendering of each body
    os.makedirs(a.out, exist_ok=True)
    pairs = []
    t_all = time.time()
    for name, ctx, sessions in CONFIGS:
        recs = []
        for s in sessions:
            fs = sorted(glob.glob(os.path.join(a.bodies, s, "*.json")))
            recs += [(s, i, fs[i], fs[i + 1]) for i in range(len(fs) - 1)]
        say(f"config {name} (-ctx {ctx}): {len(recs)} pairs; warm arm")
        srv = Server(a, ctx, f"{name}-warm", cold=False)
        mine = []
        for s, i, f1, f2 in recs:
            t1rec, t2rec = json.load(open(f1)), json.load(open(f2))
            # Turn 1 may not write the recorded reply's first token (turn 2's id at turn 1's length), so turn 2 diverges at the
            # checkpoint and takes the restore. Only where turn 2 extends turn 1: a pair that diverges before it cannot test it.
            i1, i2 = ids.get(os.path.realpath(f1)), ids.get(os.path.realpath(f2))
            ban = i2[len(i1)] if i1 and i2 and len(i2) > len(i1) and i2[:len(i1)] == i1 else None
            t1, _ = srv.send(t1rec, ban)
            if t1.get("reused", 0) != 0:  # turn 1 must be a cold prefill: restart the server and send it again
                say(f"  {s}#{i}: turn 1 reused {t1.get('reused')}; restarting the warm server")
                srv.close()
                srv = Server(a, ctx, f"{name}-warm", cold=False)
                t1, _ = srv.send(t1rec, ban)
            if ban is not None and t1["gen"][:1] == [ban]:
                sys.exit(f"VOID: {s}#{i}: turn 1 wrote the forbidden token {ban}; the ban did not reach the sampler")
            t2, lp2 = srv.send(t2rec)
            p = {"config": name, "session": s, "pair": i, "route": t2rec["path"], "t1_len": t1["prompt_len"], "t2_len": t2["prompt_len"],
                 "t1_reused": t1.get("reused"), "warm_reused": t2.get("reused"), "warm_sha": t2["prompt_sha256"], "warm_gen": t2["gen"], "warm_lp": lp2,
                 "ban": ban, "ids_len_agrees": i1 is not None and len(i1) == t1["prompt_len"]}
            p["tested"] = p["warm_reused"] == p["t1_len"] and p["t1_reused"] == 0
            mine.append((p, t2rec))
            say(f"  {s}#{i} {t2rec['path']}: turn 1 {p['t1_len']} tokens, turn 2 {p['t2_len']}, reused {p['warm_reused']}, ban {ban} -> tested {p['tested']}")
        srv.close()
        say(f"config {name}: cold arm (resident reuse off)")
        srv = Server(a, ctx, f"{name}-cold", cold=True)
        for p, t2rec in mine:
            t2, lp2 = srv.send(t2rec)
            p.update(cold_reused=t2.get("reused"), cold_sha=t2["prompt_sha256"], cold_gen=t2["gen"], cold_lp=lp2)
            p["same"] = p["warm_sha"] == p["cold_sha"] and p["warm_gen"] == p["cold_gen"] and p["warm_lp"] == p["cold_lp"]
            pairs.append(p)
            say(f"  {p['session']}#{p['pair']}: cold reused {p['cold_reused']}, identical {p['same']}")
        srv.close()
        json.dump({"pairs": pairs}, open(os.path.join(a.out, "gate2b.json"), "w"), indent=1)
    tested = [p for p in pairs if p["tested"]]
    bad = [p for p in tested if not p["same"] or p["cold_reused"] != 0]
    say(f"done in {time.time() - t_all:.0f}s: {len(pairs)} pairs, {len(tested)} tested, {len(bad)} differ")
    for p in bad:
        print(f"DIFF {p['config']} {p['session']}#{p['pair']} {p['route']}: sha {p['warm_sha'] == p['cold_sha']}, gen {p['warm_gen'] == p['cold_gen']}, "
              f"logprobs {p['warm_lp'] == p['cold_lp']}, cold reused {p['cold_reused']}")
    if len(tested) < MIN_TESTED:
        print(f"VERDICT VOID ({len(tested)} tested, under {MIN_TESTED})")
        return 2
    print(f"VERDICT {'FAIL' if bad else 'PASS'} ({len(tested)} tested, {len(bad)} differ)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
