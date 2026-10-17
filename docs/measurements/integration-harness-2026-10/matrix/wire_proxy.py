#!/usr/bin/env python3
"""A logging relay between the sandbox's socket and goinfer, so a harness run's wire facts are recorded where they happen:
for every request, its path, the model and output cap it asked for, how many tools it sent, goinfer's status, the
response's X-Goinfer-* headers, the time to the first body byte and to the end, the size in and out, the token counts
from the response's own usage, and which SSE event types came back (response.failed, error, ...). One JSON line per
request goes to the log file; the runner reads the lines a run added.

  wire_proxy.py <listen port> <upstream port> <log file> [<bodies dir>]

It copies bytes and does not interpret them: a request is forwarded as received and a response streamed as it arrives.
With a bodies dir, each request's body and the response's whole byte stream are also written there, gzipped, as
<seq>.req.json.gz and <seq>.resp.gz, and the log line names them (`bodies`). The bodies hold the harness's prompts and the
model's replies: they are what the id round trip needs (the echoed text against what serve emitted, and every prompt to
re-render), and they stay with the run's outputs, not in the repository unless a grade commits them.
"""
import gzip, http.client, itertools, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, UPSTREAM, LOG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
BODIES = sys.argv[4] if len(sys.argv) > 4 else None
if BODIES:
    os.makedirs(BODIES, exist_ok=True)
SEQ = itertools.count(1)
lock = threading.Lock()
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host"}


def log(rec):
    with lock, open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = b""
            while True:
                n = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if n == 0:
                    self.rfile.readline()
                    return out
                out += self.rfile.read(n)
                self.rfile.read(2)
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def handle_any(self):
        t0 = time.time()
        data = self.body()
        rec = {"t": t0, "method": self.command, "path": self.path, "req_bytes": len(data)}
        resp_file = None
        if BODIES and self.command == "POST":
            seq = next(SEQ)
            with gzip.open(os.path.join(BODIES, f"{seq:05d}.req.json.gz"), "wb") as f:
                f.write(data)
            resp_file = gzip.open(os.path.join(BODIES, f"{seq:05d}.resp.gz"), "wb")
            rec["bodies"] = f"{seq:05d}"
        try:
            j = json.loads(data) if data else {}
            if isinstance(j, dict):
                rec.update(model=j.get("model"), stream=j.get("stream"),
                           max_out=j.get("max_tokens") or j.get("max_output_tokens") or j.get("max_completion_tokens"),
                           tools=len(j.get("tools") or []))
        except ValueError:
            pass
        try:
            c = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=3600)
            hdr = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            hdr["Content-Length"] = str(len(data))
            c.request(self.command, self.path, body=data, headers=hdr)
            r = c.getresponse()
        except Exception as e:
            rec.update(status=502, error=str(e)[:200], t_end=time.time() - t0)
            log(rec)
            self.send_error(502, "upstream unreachable")
            return
        rec["status"] = r.status
        rec["goinfer_headers"] = {k: v for k, v in r.getheaders() if k.lower().startswith("x-goinfer")}
        chunked = (r.getheader("Transfer-Encoding") or "").lower() == "chunked"
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP and not (chunked and k.lower() == "content-length"):
                self.send_header(k, v)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        head, tail, events, first, n = "", "", {}, None, 0
        try:
            while True:
                b = r.read1(65536)
                if not b:
                    break
                if first is None:
                    first = time.time() - t0
                n += len(b)
                if resp_file:
                    resp_file.write(b)
                txt = b.decode("utf-8", "replace")
                if len(head) < 6000:
                    head = (head + txt)[:6000]
                tail = (tail + txt)[-6000:]
                for m in re.finditer(r"^event: (\S+)", txt, re.M):
                    events[m.group(1)] = events.get(m.group(1), 0) + 1
                if '"type":"error"' in txt or "event: error" in txt:
                    events["error"] = events.get("error", 0) + 1
                if chunked:
                    self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
                else:
                    self.wfile.write(b)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            rec["client_gone"] = True
        finally:
            c.close()
            if resp_file:
                resp_file.close()
        # The tail holds the final usage on every route. On Messages the prompt is input_tokens (prefilled) plus
        # cache_read_input_tokens (reused); a stream's message_delta has the split, its message_start the whole prompt.
        usage = {}
        for key in ("prompt_tokens", "input_tokens", "cache_read_input_tokens", "completion_tokens", "output_tokens"):
            m = re.findall(r'"%s":\s*(\d+)' % key, tail) or re.findall(r'"%s":\s*(\d+)' % key, head)
            if m:
                usage[key] = int(m[-1])
        rec.update(ttfb=round(first, 3) if first is not None else None, t_end=round(time.time() - t0, 3), resp_bytes=n, events=events, usage=usage)
        if r.status >= 400:
            rec["error_body"] = tail[-400:]
        log(rec)

    do_GET = do_POST = do_PUT = do_DELETE = handle_any


class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64


if __name__ == "__main__":
    S(("127.0.0.1", LISTEN), H).serve_forever()
