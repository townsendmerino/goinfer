#!/usr/bin/env python3
"""Recording pass-through proxy: python3 proxy.py LISTEN_PORT UPSTREAM_PORT OUTDIR. Streams SSE through, saves req-NNN.json and resp-NNN.txt."""
import http.server, http.client, socketserver, sys, os, threading
lp, up, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
n = [0]; lock = threading.Lock()
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with lock:
            n[0] += 1; k = n[0]
        open(f"{out}/req-{k:03d}.json", "wb").write(body)
        c = http.client.HTTPConnection("127.0.0.1", up, timeout=600)
        c.request("POST", self.path, body, {"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "")})
        r = c.getresponse()
        self.send_response(r.status)
        self.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
        self.send_header("Connection", "close")
        self.end_headers()
        with open(f"{out}/resp-{k:03d}.txt", "wb") as f:
            while True:
                b = r.read1(65536)
                if not b: break
                f.write(b); f.flush()
                self.wfile.write(b); self.wfile.flush()
        self.close_connection = True
    def do_GET(self):
        c = http.client.HTTPConnection("127.0.0.1", up, timeout=60)
        c.request("GET", self.path); r = c.getresponse(); d = r.read()
        self.send_response(r.status); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True
S(("127.0.0.1", lp), H).serve_forever()
