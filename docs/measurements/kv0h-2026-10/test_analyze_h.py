#!/usr/bin/env python3
"""Tier-0 test of analyze_h.py on synthetic server logs: one session, two threads, requests built to be N, A, N (other thread), C-A, B, D, with the
server's slot= and reused= consistent, then a plant (one slot= off by one) that the invariant must report and a plant (one rendering changed) that the
SHA-256 check must exclude. Run: python3 test_analyze_h.py"""
import json, os, subprocess, sys, tempfile, unittest, hashlib, struct

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(ids):
    return hashlib.sha256(struct.pack("<%dI" % len(ids), *ids)).hexdigest()


X = list(range(1000, 1200)); Y = list(range(2000, 2200))


def build(tmp, slot_plant=None, render_plant=None):
    # thread X: r1 N, r2 A, (Y title r3 N), r4 C-A, r5 B, r6 D; every reply is 10 ids
    seq = []  # (thread, prompt ids, gen ids)
    p1 = X + [1, 2, 3, 4, 5]; g1 = list(range(500, 510)); seq.append(("X", p1, g1))
    p2 = p1 + g1 + [6, 7, 8]; g2 = list(range(510, 520)); seq.append(("X", p2, g2))
    p3 = Y + [9, 9]; g3 = list(range(520, 530)); seq.append(("Y", p3, g3))
    p4 = p2 + g2 + [10, 11]; g4 = list(range(530, 540)); seq.append(("X", p4, g4))            # extends C_p2 but Y ran between: C-A
    p5 = p4 + [777, 778, 779, 780] ; g5 = list(range(540, 550)); seq.append(("X", p5, g5))     # extends prompt(p4), not its reply: B (no thread between)
    p6 = X + [1, 2, 99] + list(range(60, 90)); g6 = list(range(550, 560)); seq.append(("X", p6, g6))   # diverges inside the previous prompt: D
    sid = "s1"
    os.makedirs(os.path.join(tmp, "wire")); os.makedirs(os.path.join(tmp, "bodies", sid))
    json.dump(dict(id=sid, config="c", agent="opencode", task="chain", rep=1, t_start=1.0, model="m", ctx=1000), open(os.path.join(tmp, "s.json"), "w"))
    open(os.path.join(tmp, "sessions.jsonl"), "w").write(open(os.path.join(tmp, "s.json")).read() + "\n")
    wire, ids_lines, tids, slog = [], [], [], []
    prev_len = 0
    for i, (th, p, g) in enumerate(seq, 1):
        wire.append(dict(method="POST", seq=i, path="/v1/chat/completions", t_start=float(i), t_end=float(i) + 0.5))
        shown = list(p)
        if render_plant == i:
            shown = p[:-1] + [4242]          # a rendering that is not the server's prompt
        ids_lines.append(dict(file=os.path.join(tmp, "bodies", sid, f"{i:04d}.json"), ids=shown, pieces=["<|im_start|>", "user"] + ["x"] * (len(shown) - 2)))
        slot = prev_len if i > 1 else 0
        if slot_plant == i:
            slot += 1
        tids.append(dict(t="2026-10-20T00:00:%02dZ" % i, route="/v1/chat/completions", id=f"r{i}", prompt_len=len(p), prompt_sha256=sha(p), gen=g, canon=1, reused=0, slot=slot))
        slog.append(f"request: POST /v1/chat/completions model=m status=200 prompt_tokens={len(p)} completion_tokens={len(g)} ttft=1000ms total=2000ms reused=0 slot={slot} div=- canon=1")
        prev_len = len(p) + len(g)
    open(os.path.join(tmp, "wire", f"wire-{sid}.jsonl"), "w").write("\n".join(json.dumps(w) for w in wire) + "\n")
    open(os.path.join(tmp, "ids.jsonl"), "w").write("\n".join(json.dumps(x) for x in ids_lines) + "\n")
    open(os.path.join(tmp, "tokenids.jsonl"), "w").write("\n".join(json.dumps(x) for x in tids) + "\n")
    open(os.path.join(tmp, "serve.log"), "w").write("\n".join(slog) + "\n")


def run(tmp):
    out = os.path.join(tmp, "out.json")
    subprocess.run([sys.executable, os.path.join(HERE, "analyze_h.py"), "--sessions", os.path.join(tmp, "sessions.jsonl"), "--wire", os.path.join(tmp, "wire"),
                    "--ids", os.path.join(tmp, "ids.jsonl"), "--tokenids", os.path.join(tmp, "tokenids.jsonl"), "--serve-log", os.path.join(tmp, "serve.log"), "--out", out],
                   check=True, capture_output=True)
    return json.load(open(out))


class T(unittest.TestCase):
    def test_categories_and_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            build(tmp)
            d = run(tmp)
            self.assertEqual([r["cat"] for r in d["requests"]], ["N", "A", "N", "C-A", "B", "D"])
            self.assertEqual(d["slot_invariant"]["failures"], [])
            self.assertEqual(d["excluded"], [])
            b = d["requests"][4]
            self.assertEqual(b["reprocessed_under_checkpoint"], len(b and [0] * (b["n"] - b["prev_prompt"])))
            self.assertEqual(b["skip_tokens_if_served"], b["prev_prompt"])

    def test_plant_slot_off_by_one_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            build(tmp, slot_plant=4)
            d = run(tmp)
            self.assertEqual(len(d["slot_invariant"]["failures"]), 1)
            self.assertEqual(d["slot_invariant"]["offsets"], {"1": 1})

    def test_plant_wrong_rendering_is_excluded_by_the_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            build(tmp, render_plant=3)
            d = run(tmp)
            self.assertEqual(len(d["excluded"]), 1)
            self.assertEqual(d["accepted"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
