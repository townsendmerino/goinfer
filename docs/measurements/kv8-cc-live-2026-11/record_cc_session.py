#!/usr/bin/env python3
"""Record a short real Claude Code session against a local goinfer, through a recording proxy (README.md here).

  python3 record_cc_session.py --serve <bin> --model <path> --out <dir outside the repository> [--ctx 32768]

Starts serve (--log-requests --log-token-ids), a proxy that saves every POST /v1/messages body in order as <out>/bodies/NNNN.json
({"path", "body"}) and forwards it, then runs three headless Claude Code turns of one session in a fresh empty HOME and a temporary toy
project. The only credential value is the placeholder ANTHROPIC_AUTH_TOKEN=goinfer, which the local server ignores; the HOME holds none.
"""
import argparse, http.client, json, os, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVE_PORT, PROXY_PORT = 18231, 18232
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection", "upgrade", "te", "trailers"}


def make_proxy(bodies):
    counter = {"n": 0}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the body ends when the connection closes: a stream passes through untouched

        def log_message(self, *a):
            pass

        def forward(self, method):
            n = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(n) if n else b""
            path = self.path
            if method == "POST" and path.split("?")[0] == "/v1/messages":
                with lock:
                    counter["n"] += 1
                    i = counter["n"]
                with open(os.path.join(bodies, f"{i:04d}.json"), "w") as f:
                    json.dump({"path": path, "body": json.loads(raw)}, f)
            c = http.client.HTTPConnection("127.0.0.1", SERVE_PORT, timeout=900)
            hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
            c.request(method, path, raw if raw else None, hdrs)
            r = c.getresponse()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            c.close()

        def do_POST(self):
            self.forward("POST")

        def do_GET(self):
            self.forward("GET")

    return ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), H), counter


PROMPTS = [
    "Read calc.py and tell me in one sentence what add does.",
    "add has a bug. Fix it with the smallest possible edit so that test_calc.py passes.",
    "Run python3 test_calc.py and tell me whether it passes.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--extra-prompt", action="append", default=[])
    a = p.parse_args()
    bodies = os.path.join(a.out, "bodies")
    os.makedirs(bodies, exist_ok=True)
    log = open(os.path.join(a.out, "serve-live.log"), "w")
    srv = subprocess.Popen([a.serve, "--model", a.model, "--backend", "cuda", "--ctx", str(a.ctx), "--addr", f"127.0.0.1:{SERVE_PORT}",
                            "--kv-sessions", "1", "--log-requests", "--log-token-ids", os.path.join(a.out, "tokenids-live.jsonl")],
                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(300):
            try:
                mid = json.load(urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/v1/models", timeout=1))["data"][0]["id"]
                break
            except OSError:
                if srv.poll() is not None:
                    sys.exit("serve exited")
                time.sleep(1)
        else:
            sys.exit("serve did not come up")
        proxy, counter = make_proxy(bodies)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        home, work = tempfile.mkdtemp(prefix="cc-home-"), tempfile.mkdtemp(prefix="cc-proj-")
        open(os.path.join(work, "calc.py"), "w").write("def add(a, b):\n    return a - b\n")
        open(os.path.join(work, "test_calc.py"), "w").write("from calc import add\nassert add(2, 3) == 5, add(2, 3)\nprint('ok')\n")
        env = {"HOME": home, "PATH": os.environ["PATH"], "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PROXY_PORT}", "ANTHROPIC_AUTH_TOKEN": "goinfer",
               "ANTHROPIC_MODEL": mid, "ANTHROPIC_DEFAULT_MODEL": mid, "ANTHROPIC_DEFAULT_FABLE_MODEL": mid, "ANTHROPIC_DEFAULT_OPUS_MODEL": mid,
               "ANTHROPIC_DEFAULT_SONNET_MODEL": mid, "ANTHROPIC_DEFAULT_HAIKU_MODEL": mid + "-fast", "ANTHROPIC_SMALL_FAST_MODEL": mid + "-fast",
               "CLAUDE_CODE_SUBAGENT_MODEL": mid, "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
               "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(a.ctx), "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096", "DISABLE_TELEMETRY": "1",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"}
        print(f"model {mid}; home {home}; project {work}", flush=True)
        for i, prompt in enumerate(PROMPTS + a.extra_prompt):
            argv = [os.path.expanduser("~/.local/bin/claude"), "-p", prompt, "--max-turns", "8", "--permission-mode", "acceptEdits",
                    "--allowedTools", "Read,Edit,Bash(python3 test_calc.py)", "--output-format", "json"] + (["--continue"] if i else [])
            before, t0 = counter["n"], time.time()
            r = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=1500, stdin=subprocess.DEVNULL)
            try:
                j = json.loads(r.stdout)
                res = {k: j.get(k) for k in ("subtype", "is_error", "num_turns")}
            except ValueError:
                res = {"stdout": r.stdout[:200], "stderr": r.stderr[:200]}
            print(f"turn {i + 1}: exit {r.returncode}, {time.time() - t0:.0f} s, {counter['n'] - before} requests, {res}", flush=True)
        print("calc.py now:", repr(open(os.path.join(work, "calc.py")).read()), flush=True)
        print(f"recorded {counter['n']} /v1/messages requests in {bodies}", flush=True)
    finally:
        srv.terminate()
        srv.wait(30)
        log.close()


if __name__ == "__main__":
    main()
