#!/usr/bin/env python3
"""A stand-in Anthropic Messages server for watching a client's request pattern (item 3a of the carried-ids follow-ups): it
records every request and answers at once. A main-loop request (one with tools) gets a Read call the first time and a text
answer once a tool_result comes back; anything else gets a short text. No model.

  python3 stub.py PORT OUTDIR MODEL WORKFILE
"""
import gzip, http.server, json, os, sys, threading, time

PORT, OUT, MODEL, WORKFILE = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True)
lock, seq = threading.Lock(), [0]


def text_of(c):
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c or [])


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self.send_json({"object": "list", "data": [{"id": MODEL, "object": "model", "context_window": 32768, "max_positions": 131072}]})
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        with lock:
            seq[0] += 1
            n = seq[0]
        with gzip.open(os.path.join(OUT, f"{n:05d}.req.json.gz"), "wb") as f:
            f.write(raw)
        msgs = body.get("messages") or []
        last = msgs[-1] if msgs else {}
        lastc = last.get("content")
        has_result = any(isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
                         for m in msgs[-2:])
        sysp = body.get("system")
        sys_head = text_of(sysp)[:160] if sysp else ""
        rec = {"seq": n, "t": time.time(), "path": self.path, "model": body.get("model"), "stream": bool(body.get("stream")),
               "max_tokens": body.get("max_tokens"), "tools": len(body.get("tools") or []), "messages": len(msgs), "bytes": len(raw),
               "system_head": sys_head, "last_head": text_of(lastc)[:160] if lastc is not None else "", "thinking": body.get("thinking")}
        with lock, open(os.path.join(OUT, "requests.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if "count_tokens" in self.path:
            return self.send_json({"input_tokens": max(1, len(raw) // 4)})
        if rec["tools"] and not has_result:
            content = [{"type": "tool_use", "id": f"toolu_{n:04d}", "name": "Read", "input": {"file_path": WORKFILE}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "Add returns a + b." if rec["tools"] else "Reading calc.go"}]
            stop = "end_turn"
        mid = f"msg_stub{n:04d}"
        usage = {"input_tokens": max(1, len(raw) // 4), "output_tokens": 8}
        if not rec["stream"]:
            return self.send_json({"id": mid, "type": "message", "role": "assistant", "model": MODEL, "content": content,
                                   "stop_reason": stop, "stop_sequence": None, "usage": usage})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def ev(name, obj):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode())

        ev("message_start", {"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": MODEL,
                                                                  "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0}}})
        for i, b in enumerate(content):
            if b["type"] == "text":
                ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
                ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}})
            else:
                ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
                ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})
            ev("content_block_stop", {"type": "content_block_stop", "index": i})
        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 8}})
        ev("message_stop", {"type": "message_stop"})
        self.wfile.flush()


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
