#!/usr/bin/env python3
"""Tests for mem_darwin.py, the Mac memory instrument: python3 -B scripts/test_mem_darwin.py (stdlib only, seconds, any OS).

The parsers read real macOS output. The cell runs against a fake `serve` (a small HTTP server that prints guard lines, answers
the first request and refuses the second) and a fake machine, so it needs no model, no GPU and no macOS."""
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mem_darwin as md  # noqa: E402

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    21329.
Pages active:                                 291538.
Pages inactive:                               272009.
Pages speculative:                             19195.
Pages throttled:                                   0.
Pages wired down:                             152356.
Pages purgeable:                               62696.
File-backed pages:                            211908.
Anonymous pages:                              370834.
Pages stored in compressor:                   596961.
Pages occupied by compressor:                 252621.
"""

# A Metal serve log's lines around a swap-guard trip (a 1.5B at -ctx 32768).
SERVE_LOG = """swap guard (load): armed for /m.giw: system swap +512 MB over baseline (the per-process reader is not armed: macOS has no per-process swap figure)
loading /m.giw
  fit: 2 KV slots at a 32768-token cap hold 1.8 GB KV (weights 1.1 GB, budget 5.0 GB, 3.3 GB left for a request's own prefill)
  KV plan: 2 conversations x 32768 positions (4 asked for; --ctx 32768)
  session reuse: on the GPU cache, 2 conversations kept resident; a memory guard may keep fewer
swap guard: armed — system swap +512 MB over baseline (the per-process reader is not armed: macOS has no per-process swap figure)
listening on 127.0.0.1:18297
swap guard: baseline 1.23 GB (system swap)
swap guard: TRIPPED — system swap grew 0.95 GB over baseline (2.18 GB used now); refusing new requests until it recovers
""".splitlines()

TEXT = "".join(f"line {i} of the prompt\n" for i in range(400))  # not periodic: swapping its halves changes its prefix

FAKE_SERVE = r'''#!/usr/bin/env python3
import http.server, json, os, sys, time
a = sys.argv[1:]
host, port = a[a.index("-addr") + 1].rsplit(":", 1)
rec = os.environ["FAKE_SERVE_RECORD"]
open(rec, "a").write(json.dumps({"started": a}) + "\n")
if os.environ.get("FAKE_SERVE_EXIT"):
    print("fit: refused: this machine currently has 1.0 GB of memory available", flush=True)
    sys.exit(3)
print("swap guard (load): armed for m: system swap +512 MB over baseline", flush=True)
print("  fit: 2 KV slots at a 32768-token cap hold 1.8 GB KV (weights 1.1 GB, budget 5.0 GB)", flush=True)
print("  KV plan: 2 conversations x 32768 positions", flush=True)
n = [0]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        self.reply(200, {"data": [{"id": "m"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        c = body["messages"][0]["content"]
        c = c if isinstance(c, str) else json.dumps(c)
        open(rec, "a").write(json.dumps({"len": len(c), "head": c[:80], "hash": hash(c)}) + "\n")
        n[0] += 1
        if n[0] == 1:
            print("swap guard: TRIPPED — system swap grew 0.95 GB over baseline (2.18 GB used now)", flush=True)
            time.sleep(float(os.environ.get("FAKE_SERVE_REQ_S", "0.8")))
            self.reply(200, {"choices": [{"message": {"content": "goinfer"}}], "usage": {"prompt_tokens": 20189}})
        else:
            self.reply(503, {"error": "halted", "reason": "swap guard tripped"})
http.server.ThreadingHTTPServer((host, int(port)), H).serve_forever()
'''


class FakeBox:
    """A machine whose wired pages grow 10 MiB and file-backed pages 1 MiB at every reading."""

    def __init__(self, swap=1200.0, load=1.5):
        self.swap, self.load, self.n = swap, load, 0

    def loadavg(self):
        return [self.load, 2.0, 2.5]

    def swap_used_mib(self):
        return self.swap

    def pages(self):
        self.n += 1
        return {"wired_MiB": 2000.0 + 10 * self.n, "file_backed_MiB": 3000.0 + self.n, "free_MiB": 500.0, "compressor_MiB": 100.0}

    def hostmem(self, *args):
        if args[:1] == ("-pid",):
            return f"resident 100 internal 50 compressed {1 << 20} compressed_peak {2 << 20} phys_footprint {2453 << 20}"
        return {("-v",): "physical 16.00 GiB, available 7.51 (level 50%); reclaimable 2.57 (free 0.07, file-backed 2.48)",
                ("-mb",): "7690", ("-reclaimable", "-mb"): "2631"}[args]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Parse(unittest.TestCase):
    def test_vm_stat_pages_as_mib(self):
        got = md.parse_vm_stat(VM_STAT)
        self.assertEqual(got, {"wired_MiB": round(152356 * 16384 / md.MiB, 1), "file_backed_MiB": round(211908 * 16384 / md.MiB, 1),
                               "free_MiB": round(21329 * 16384 / md.MiB, 1), "compressor_MiB": round(252621 * 16384 / md.MiB, 1)})
        self.assertIsNone(md.parse_vm_stat(VM_STAT.replace("File-backed pages", "File pages")))
        self.assertIsNone(md.parse_vm_stat(""))

    def test_sysctl_and_hostmem_lines(self):
        self.assertEqual(md.parse_loadavg("{ 3.66 4.16 4.05 }\n"), [3.66, 4.16, 4.05])
        self.assertIsNone(md.parse_loadavg(""))
        self.assertEqual(md.parse_swap_used_mib("total = 2048.00M  used = 733.25M  free = 1314.75M  (encrypted)"), 733.25)
        self.assertIsNone(md.parse_swap_used_mib(""))
        self.assertEqual(md.parse_task("resident 1 internal 2 compressed 3 compressed_peak 4 phys_footprint 5"),
                         {"resident": 1, "internal": 2, "compressed": 3, "compressed_peak": 4, "phys_footprint": 5})
        self.assertEqual(md.int_or_none("7690\n"), 7690)
        self.assertIsNone(md.int_or_none(""))

    def test_guard_lines_are_the_guards_decisions(self):
        got = md.guard_lines(SERVE_LOG)
        self.assertEqual(len(got), 6, got)
        self.assertTrue(got[0].startswith("swap guard (load): armed"))
        self.assertTrue(got[1].lstrip().startswith("fit: 2 KV slots"))
        self.assertTrue(got[2].lstrip().startswith("KV plan:"))
        self.assertTrue(got[-1].startswith("swap guard: TRIPPED"))
        self.assertFalse(any("session reuse" in l or "listening" in l for l in got))

    def test_the_second_prompt_has_the_same_length_and_another_prefix(self):
        t = TEXT
        s = md.second_prompt(t)
        self.assertEqual(len(s), len(t) + 2)
        self.assertNotEqual(s[:100], t[:100])

    def test_start_rule(self):
        self.assertIsNone(md.start_refusal(FakeBox(swap=1200, load=1.5), 2048, 4))
        self.assertIn("system swap 3000", md.start_refusal(FakeBox(swap=3000), 2048, 4))
        self.assertIn("load1 6.6", md.start_refusal(FakeBox(load=6.6), 2048, 4))
        self.assertIsNone(md.start_refusal(FakeBox(swap=3000, load=6.6), -1, -1))
        self.assertIn("system swap None", md.start_refusal(FakeBox(swap=None), 2048, 4))


class Cell(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mem-darwin-test-"))
        self.serve = self.dir / "serve"
        self.serve.write_text(FAKE_SERVE)
        self.serve.chmod(0o755)
        self.record = self.dir / "record.jsonl"
        self.env = {k: os.environ.get(k) for k in ("FAKE_SERVE_RECORD", "FAKE_SERVE_REQ_S", "FAKE_SERVE_EXIT")}
        os.environ["FAKE_SERVE_RECORD"] = str(self.record)
        self.out = self.dir / "out"
        self.out.mkdir()

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def run_cell(self, req_timeout=30.0, name="metal:1.5b"):
        return md.cell(name, str(self.serve), FakeBox(), TEXT, free_port(), str(self.out), gap=0.3, interval=0.1,
                       req_timeout=req_timeout, settle=0.1, path="/m.giw", extra=["-ctx", "32768"])

    def bodies(self):
        return [json.loads(l) for l in self.record.read_text().splitlines() if '"len"' in l]

    def test_a_cell_records_both_requests_the_guard_lines_and_the_page_deltas(self):
        rec = self.run_cell()
        self.assertEqual(rec["request1"]["status"], 200, rec)
        self.assertEqual(rec["request1"]["usage"]["prompt_tokens"], 20189)
        self.assertEqual(rec["request2"]["status"], 503)
        self.assertIn("halted", rec["request2"]["body"])
        self.assertTrue(rec["server_alive_after"])
        b = self.bodies()
        self.assertEqual(len(b), 2)
        self.assertEqual(b[0]["len"] + 2, b[1]["len"], "request 2 is the same text with its halves swapped")
        self.assertNotEqual(b[0]["head"], b[1]["head"], "request 2 must not share request 1's prefix")
        self.assertEqual([p["label"] for p in rec["points"]], ["before", "loaded", "after request 1", "0.3 s after request 1", "after request 2"])
        before = rec["points"][0]
        for p in rec["points"]:
            self.assertEqual((p["available_MiB"], p["reclaimable_MiB"]), (7690, 2631))
            self.assertIn("reclaimable 2.57", p["hostmem_v"])
            self.assertEqual(p["loadavg"], [1.5, 2.0, 2.5])
            self.assertEqual(p["swap_used_MiB"], 1200.0)
        for p in rec["points"][1:]:
            self.assertEqual(p["wired_delta_MiB"], round(p["wired_MiB"] - before["wired_MiB"], 1))
            self.assertEqual(p["file_backed_delta_MiB"], round(p["file_backed_MiB"] - before["file_backed_MiB"], 1))
            self.assertGreater(p["wired_delta_MiB"], 0)
        self.assertGreater(len(rec["samples"]), 3)
        self.assertEqual({s["phase"] for s in rec["samples"]} - {"request 1", "gap", "request 2"}, set())
        self.assertEqual(rec["peaks"]["phys_footprint_MiB"], 2453)
        self.assertEqual(rec["peaks"]["compressed_peak_MiB"], 2)
        self.assertEqual(rec["peaks"]["wired_MiB"], max(s["wired_MiB"] for s in rec["samples"]))
        self.assertEqual(rec["peaks"]["wired_MiB_from_before"], round(rec["peaks"]["wired_MiB"] - before["wired_MiB"], 1))
        lines = [g["line"] for g in rec["guard_lines"]]
        self.assertTrue(lines[0].startswith("swap guard (load): armed"), lines)
        self.assertTrue(any(l.lstrip().startswith("fit:") for l in lines))
        self.assertTrue(any(l.lstrip().startswith("KV plan:") for l in lines))
        trip = [g for g in rec["guard_lines"] if "TRIPPED" in g["line"]]
        self.assertEqual(len(trip), 1)
        self.assertRegex(trip[0]["t"], r"^\d\d:\d\d:\d\d$", "a guard line carries the time it was first seen")

    def test_a_request_past_the_timeout_is_recorded_as_a_hang(self):
        os.environ["FAKE_SERVE_REQ_S"] = "3"
        rec = self.run_cell(req_timeout=0.5)
        self.assertIsNone(rec["request1"]["status"])
        self.assertIn("timed out", rec["request1"]["error"])
        self.assertIsNone(rec["request1"]["server_exit"])
        self.assertEqual(rec["request2"]["status"], 503)

    def test_a_server_that_does_not_come_up_is_void_with_its_log(self):
        os.environ["FAKE_SERVE_EXIT"] = "1"
        rec = self.run_cell()
        self.assertIn("did not come up (exit 3)", rec["void"])
        self.assertIn("1.0 GB of memory available", "\n".join(rec["log_tail"]))
        self.assertNotIn("request1", rec)

    def test_a_sample_taken_as_the_sampler_starts_is_request_1s(self):
        """The thread may take its first sample before start() returns to the cell. Forcing that pass to happen inside start(),
        the first sample must already carry request 1's phase."""
        real = md.Sampler.start

        def eager(sampler):
            sampler.sample_once()
            real(sampler)

        md.Sampler.start = eager
        try:
            rec = self.run_cell()
        finally:
            md.Sampler.start = real
        self.assertEqual(rec["samples"][0]["phase"], "request 1")

    def test_a_tower_cell_sends_two_different_images(self):
        rec = self.run_cell(name="tower:qwen3vl-2b")
        self.assertEqual((rec["request1"]["status"], rec["request2"]["status"]), (200, 503))
        b = self.bodies()
        self.assertNotEqual(b[0]["hash"], b[1]["hash"])
        self.assertIn("image_url", b[0]["head"])


class Main(unittest.TestCase):
    def test_the_start_rule_refuses_before_anything_loads(self):
        d = Path(tempfile.mkdtemp(prefix="mem-darwin-test-"))
        marker = d / "started"
        serve = d / "serve"
        serve.write_text(f"#!/bin/sh\ntouch {marker}\n")
        serve.chmod(0o755)
        rc = md.main(["--serve", str(serve), "--out", str(d / "out"), "--repo", md.REPO, "--cell", "metal:1.5b"], box=FakeBox(swap=3000))
        self.assertEqual(rc, md.RETRY_RC)
        self.assertFalse(marker.exists(), "no server may start")
        rec = json.loads((d / "out" / "cell-metal-1.5b.json").read_text())
        self.assertIn("system swap 3000", rec["refused"])
        self.assertEqual(rec["start_rule"], {"max_swap_mib": 2048, "max_load": 4})


if __name__ == "__main__":
    unittest.main()
