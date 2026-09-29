#!/usr/bin/env python3
"""MC3 per-pass prefill: how much of a 4-client served W7 cell is the resident's prefill passes (mc3-prefill-attr-
2026-09-28.md). Reads the time counters serve prints at shutdown (ResidentBatchStats, 2bdd3f3b); times nothing
else itself beyond the cell's wall clock.

    python3 mc3_prefill_attr.py <job dir> [--reps 3] [--max-tokens 128] [--clients 4] [--models 1.5B,7B]

<job dir> holds serve-metal-<rev> and a copy of bench_w7_plain.py. Output: <job dir>/results.json and one server log per
cell (server-<model>-<rep>.log).
"""
import argparse, glob, http.client, json, os, re, socket, sys, time

MODELS = {"1.5B": "~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "7B": "~/models/qwen2.5-7b-instruct-q4_k_m.gguf"}
TIME_RE = re.compile(r'resident batch "bench" time: runs ([\d.]+) s, exclusive ([\d.]+) s \(prefill (\d+) passes ([\d.]+) s, '
                     r'bookkeeping ([\d.]+) s\)')
BATCH_RE = re.compile(r'resident batch "bench": (\d+) runs, (\d+) batched steps \((\d+) tokens\), (\d+) solo tokens')


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("admin-socket")
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)


def batch_stats(sock):
    """The resident batcher's counters for "bench", from /admin/status over serve's admin socket."""
    c = UnixHTTP(sock)
    c.request("GET", "/admin/status")
    st = json.loads(c.getresponse().read())
    c.close()
    return (st.get("resident_batch") or {}).get("bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--models", default="1.5B,7B")
    a = ap.parse_args()
    J = os.path.abspath(a.job)
    sys.path.insert(0, J)
    import bench_w7_plain as w7
    (w7.SERVE_CPU_METAL,) = glob.glob(os.path.join(J, "serve-metal-*"))
    out = {"header": w7.machine_header(), "binary": os.path.basename(w7.SERVE_CPU_METAL), "args": vars(a), "cells": []}
    for model in a.models.split(","):
        w7.MODEL_PATH = os.path.expanduser(MODELS[model])
        for rep in range(1, a.reps + 1):
            log = os.path.join(J, f"server-{model}-{rep}.log")
            open(log, "w").close()
            # -embed-int4=false: since 9ccf7fb1 the int4 embedding default makes the Metal resident decline to the CPU.
            sock = os.path.join("/tmp", f"mc3attr-{os.getpid()}-{model}-{rep}.sock")  # a Unix socket path must be short
            with w7.GoinferServer("metal", f"-embed-int4=false -admin-socket {sock}", log) as srv:
                w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4,
                                  "temperature": 0}, timeout=300)
                before = batch_stats(sock)
                r = w7.run_concurrent(srv.url, a.max_tokens, 0.0, a.clients, fixed_nonce=True)
                after = batch_stats(sock)
            time.sleep(1)  # the shutdown lines are written as the server exits
            text = open(log).read()
            m, bm = TIME_RE.search(text), BATCH_RE.search(text)
            if "decode path: metal-resident" not in text:
                m = None  # not the resident: the cell is invalid, whatever the counters say
            cell = {"model": model, "rep": rep, "wall_s": r["wall_s"], "aggregate_tok_s": r["aggregate_tok_s"],
                    "completion_tokens": r["total_completion_tokens"],
                    "prompt_tokens": sum((t["prompt_tokens"] or 0) for c in r["per_client"] for t in c),
                    "reused_tokens": sum((t["prefill_reused_tokens"] or 0) for c in r["per_client"] for t in c),
                    "turn_latency_ms": r["per_client_latency_ms"]}
            if m:  # the server's whole life, load and warm-up included: a cross-check, not the reading
                cell["lifetime"] = {"runs_s": float(m[1]), "exclusive_s": float(m[2]), "prefill_passes": int(m[3]),
                                    "prefill_s": float(m[4]), "bookkeeping_s": float(m[5])}
            if m and before and after:  # THE READING: the counters' change across the cell alone
                d = {k: after[k] - before[k] for k in ("RunNs", "ExclusiveNs", "PrefillNs", "PrefillPasses", "Runs", "Steps")}
                runs, excl, pre = d["RunNs"] / 1e9, d["ExclusiveNs"] / 1e9, d["PrefillNs"] / 1e9
                cell.update({"runs_s": round(runs, 3), "exclusive_s": round(excl, 3), "prefill_passes": d["PrefillPasses"],
                             "prefill_s": round(pre, 3), "bookkeeping_s": round(excl - pre, 3),
                             "prefill_share": round(pre / r["wall_s"], 4),
                             "idle_s": round(r["wall_s"] - runs - excl, 3)})
            else:
                cell["error"] = "no resident-batch counters (not on the Metal resident, or the admin socket gave none)"
            if bm:
                cell.update({"batch_runs": int(bm[1]), "steps": int(bm[2]), "step_tokens": int(bm[3]), "solo_tokens": int(bm[4])})
            out["cells"].append(cell)
            json.dump(out, open(os.path.join(J, "results.json"), "w"), indent=1)
            print(f"[mc3-attr] {model} rep {rep}: wall {cell['wall_s']} s, prefill {cell.get('prefill_s')} s in "
                  f"{cell.get('prefill_passes')} passes, runs {cell.get('runs_s')} s, share {cell.get('prefill_share')}"
                  f"{'  ' + cell['error'] if 'error' in cell else ''}", flush=True)
    # The pre-registered reading (mc3-prefill-attr-2026-09-28.md §3): the median share on the 1.5B.
    shares = sorted(c["prefill_share"] for c in out["cells"] if c["model"] == "1.5B" and "prefill_share" in c)
    if len(shares) < 2:
        verdict = f"INCONCLUSIVE: {len(shares)} valid 1.5B cell(s)"
    else:
        med = shares[len(shares) // 2]
        verdict = (f"CLOSE the cut: median prefill share {med:.3f} < 0.15" if med < 0.15 else
                   f"REGISTER the cut: median prefill share {med:.3f} >= 0.25" if med >= 0.25 else
                   f"OWNER'S CALL: median prefill share {med:.3f} in [0.15, 0.25)")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(J, "results.json"), "w"), indent=1)
    print("[mc3-attr] VERDICT:", verdict, flush=True)


if __name__ == "__main__":
    main()
