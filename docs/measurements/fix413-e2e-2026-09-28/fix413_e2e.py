#!/usr/bin/env python3
"""The 413 prefill-share fix (5a92348a), end to end on the 7B, Metal, 4 clients, with live memory held in the band
where the two builds disagree. The design and decision rule are in fix413-e2e-2026-09-28.md, written before this
ran.

    python3 fix413_e2e.py <job dir>

<job dir> holds the pinned serve binaries (serve-metal-fixed-*, serve-metal-unfixed-*), copies of bench_spec_copy.py
and bench_w7_plain.py, and ballast.py. Results go to <job dir>/results.json, with the servers' stderr in servers.log
and the ballast's per-second status in ballast-<round>-<arm>.log.
"""
import glob, json, os, subprocess, sys, time, types

J = os.path.abspath(sys.argv[1])
os.environ["BENCH_W7_MODEL"] = os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf")
sys.path.insert(0, J)
import bench_w7_plain as w7  # noqa: E402  (reads BENCH_W7_MODEL at import)
import bench_spec_copy as sc  # noqa: E402

BAND = (1.0, 1.6)                   # GB of vm_stat "available"; te pre-registration §2
ROUNDS = [("unfixed", "fixed"), ("fixed", "unfixed"), ("unfixed", "fixed")]
ARGS = types.SimpleNamespace(clients=4, rounds=2, chars=3500, max_tokens=256, rev="cc5f8c2c")
SETTLE_TIMEOUT = 240


def binary(arm):
    (b,) = glob.glob(os.path.join(J, f"serve-metal-{arm}-*"))
    return b


def ballast_status(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def run_round(k, arm, secs):
    w7.SERVE_CPU_METAL = binary(arm)
    rec = {"round": k, "arm": arm, "binary": os.path.basename(w7.SERVE_CPU_METAL), "started": time.strftime("%H:%M:%S")}
    status = os.path.join(J, f"ballast-{k}-{arm}.status")
    blog = open(os.path.join(J, f"ballast-{k}-{arm}.log"), "w")
    with w7.GoinferServer("metal", "", os.path.join(J, "servers.log")) as srv:
        w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4,
                          "temperature": 0}, timeout=300)
        bp = subprocess.Popen([sys.executable, os.path.join(J, "ballast.py"), "--lo-gb", str(BAND[0]), "--hi-gb",
                               str(BAND[1]), "--status", status, "--max-s", "900"], stdout=blog, stderr=subprocess.STDOUT)
        t0 = time.time()
        st = {}
        while time.time() - t0 < SETTLE_TIMEOUT:
            st = ballast_status(status)
            if st.get("state") in ("settled", "ABORT", "EXPIRED") or bp.poll() is not None:
                break
            time.sleep(1)
        rec["ballast_at_start"] = st
        if st.get("state") != "settled":
            rec["valid"] = False
            rec["invalid_why"] = f"ballast did not settle in {SETTLE_TIMEOUT}s: {st.get('state')} {st.get('reason', '')}"
        else:
            samples = []
            done = [False]

            def sample():
                while not done[0]:
                    s = ballast_status(status)
                    if s:
                        samples.append((s.get("available"), s.get("state"), s.get("swap_growth")))
                    time.sleep(1)
            import threading
            th = threading.Thread(target=sample, daemon=True)
            th.start()
            rec["cell"] = sc.cell(srv.url, ARGS, secs)
            done[0] = True
            th.join()
            av = [s[0] for s in samples if s[0]]
            rec["available_during_cell_gb"] = [round(min(av) / 2**30, 3), round(max(av) / 2**30, 3)] if av else None
            rec["ballast_states_during_cell"] = sorted({s[1] for s in samples})
            rec["swap_growth_max_mb"] = round(max((s[2] or 0) for s in samples) / 2**20, 1) if samples else None
            aborted = any(s[1] in ("ABORT", "EXPIRED") for s in samples)
            in_band = bool(av) and min(av) >= 0.8 * 2**30 and max(av) <= 2.0 * 2**30
            swap_trip = 503 in rec["cell"]["http_errors"]
            rec["valid"] = (not aborted) and in_band and not swap_trip
            rec["invalid_why"] = None if rec["valid"] else \
                f"aborted={aborted} in_band={in_band} swap_guard_503={swap_trip}"
        bp.terminate()
        try:
            bp.wait(timeout=30)
        except Exception:
            bp.kill()
    rec["ended"] = time.strftime("%H:%M:%S")
    return rec


def main():
    secs = json.load(open(os.path.join(J, "sections.json")))  # frozen at queue time from decoder/model.go @ cc5f8c2c
    out = {"header": w7.machine_header(), "band_gb": BAND, "args": vars(ARGS), "rounds": []}
    for k, order in enumerate(ROUNDS, 1):
        for arm in order:
            r = run_round(k, arm, secs)
            out["rounds"].append(r)
            json.dump(out, open(os.path.join(J, "results.json"), "w"), indent=1)
            c = r.get("cell") or {}
            print(f"[fix413-e2e] round {k} {arm:8s} valid={r['valid']} errors={c.get('http_errors')} "
                  f"completed={c.get('completion_tokens')} avail={r.get('available_during_cell_gb')} "
                  f"{r.get('invalid_why') or ''}", flush=True)
    # The pre-registered verdict (fix413-e2e-2026-09-28.md §3).
    v = {arm: [r for r in out["rounds"] if r["arm"] == arm and r["valid"]] for arm in ("fixed", "unfixed")}
    fixed_413 = sum(r["cell"]["http_errors"].count(413) for r in v["fixed"])
    unfixed_413 = sum(r["cell"]["http_errors"].count(413) for r in v["unfixed"])
    if len(v["fixed"]) < 2 or len(v["unfixed"]) < 2:
        verdict = f"INCONCLUSIVE: valid rounds fixed {len(v['fixed'])}, unfixed {len(v['unfixed'])} (need 2 each)"
    elif fixed_413 > 0:
        verdict = f"FAIL: the fixed build refused {fixed_413} request(s) with 413 inside the band"
    elif unfixed_413 == 0:
        verdict = "NOT DISCRIMINATING: the unfixed build refused nothing either, so the band did not reproduce §5"
    else:
        verdict = f"PASS: fixed 0 refusals; unfixed {unfixed_413} refusal(s) (413) in the same band"
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(J, "results.json"), "w"), indent=1)
    print("[fix413-e2e] VERDICT:", verdict, flush=True)


if __name__ == "__main__":
    main()
