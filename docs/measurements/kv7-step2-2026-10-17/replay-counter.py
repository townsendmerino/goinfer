#!/usr/bin/env python3
"""replay-counter.py BODIES_DIR PORT SOCKET OUT.json — post the recorded request bodies in order (non-streamed, max_tokens 1) and save the server's /admin/status.
Prefill is what the snapshot machinery sees, so one token is enough; the night's own replies are already inside the later bodies."""
import glob, gzip, http.client, json, socket, sys, time
bodies, port, sock, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
files = sorted(glob.glob(bodies + "/*.req.json.gz")); t0 = time.time()
for i, f in enumerate(files, 1):
    b = json.loads(gzip.open(f).read()); b["max_tokens"] = 1; b["stream"] = False
    t = time.time(); c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/messages", json.dumps(b), {"Content-Type": "application/json", "x-api-key": "x", "anthropic-version": "2023-06-01"})
    r = c.getresponse(); d = r.read()
    u = json.loads(d).get("usage", {}) if r.status == 200 else d[:120]
    print(f"[{time.time()-t0:6.0f}s] {i}/{len(files)} status {r.status} {time.time()-t:6.1f}s usage {u}", flush=True)
class U(http.client.HTTPConnection):
    def connect(self): self.sock = socket.socket(socket.AF_UNIX); self.sock.connect(sock)
c = U("localhost"); c.request("GET", "/admin/status"); st = json.loads(c.getresponse().read())
json.dump(st, open(out, "w"), indent=1); print(json.dumps(st)[:1500])
