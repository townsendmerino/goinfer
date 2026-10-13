#!/usr/bin/env python3
"""Record what a client posts: every request's method, path, headers of interest and body go to <outdir>/req-N.json; replies 400."""
import http.server, json, os, sys
out = sys.argv[1]; port = int(sys.argv[2]); os.makedirs(out, exist_ok=True); n = [0]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _rec(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        n[0] += 1
        hdr = {k: (v if k.lower() not in ("authorization", "x-api-key") else "<set>") for k, v in self.headers.items()}
        try: parsed = json.loads(body) if body else None
        except Exception: parsed = body.decode("utf-8", "replace")
        json.dump({"method": self.command, "path": self.path, "headers": hdr, "body": parsed}, open(os.path.join(out, f"req-{n[0]}.json"), "w"), indent=1)
        self.send_response(400); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"type":"error","error":{"type":"invalid_request_error","message":"capture only"}}')
    do_POST = do_GET = _rec
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
