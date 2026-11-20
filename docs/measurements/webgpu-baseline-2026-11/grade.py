#!/usr/bin/env python3
"""WG0's grader: WebGPU as a fraction of the native backend, per cell, from A B B A pairs.

  grade.py <results dir>   reads decode-*.json (bench_peer.py, engines goinfer = native, goinfer_old = WebGPU) and ttft.json
                           (wg0_ttft.py); prints one row per cell and writes grade.json beside them.

A pair is one ABBA block: both arms' cells in it. Decode: fraction = WebGPU tok/s / native tok/s (each cell's mean over its runs).
TTFT: fraction = native TTFT / WebGPU TTFT (each cell's median), so above 1 means WebGPU is faster in both. A cell is valid only if
it has no error, its token gate is ok (decode), no void on swap, the decode path is the arm's resident path, and every block of the
cell is complete. A cell that fails any check is INVALID with the reason; there is no pass or fail on the fraction (G-WG0).
"""
import glob, json, os, statistics, sys

WANT = {"native": ("cuda-resident", "metal"), "webgpu": ("webgpu:",)}


def path_ok(arm, dp):
    if not dp:
        return False
    if arm == "webgpu":
        return dp.startswith("webgpu:") and "-resident" in dp.split()[0]
    return dp.split()[0] in ("cuda-resident", "metal-resident") or dp.startswith("metal")


def summarize(fr):
    return {"median": round(statistics.median(fr), 4), "min": round(min(fr), 4), "max": round(max(fr), 4), "n": len(fr)}


def decode(d):
    rows, problems = {}, []
    for f in sorted(glob.glob(os.path.join(d, "decode-*.json"))):
        for r in json.load(open(f)):
            if r.get("kind") == "provenance" or r.get("phase") not in ("A", "B"):
                continue
            arm = "native" if r["engine"] == "goinfer" else "webgpu"
            key = (r["model"], r["depth"])
            c = r.get("counts") or {}
            why = None
            if r.get("error") or not r.get("runs"):
                why = f"error {r.get('error')}"
            elif (c.get("token_gate") or {}).get("verdict") != "ok":
                why = f"token gate {(c.get('token_gate') or {}).get('verdict')}"
            elif (c.get("swap") or {}).get("swap_void"):
                why = f"swap {(c.get('swap') or {}).get('swap_void')}"
            elif not path_ok(arm, c.get("decode_path")):
                why = f"decode path {c.get('decode_path')}"
            rows.setdefault(key, {}).setdefault(r["block"], {})[arm] = (r.get("mean"), why, c.get("decode_path"),
                                                                       r.get("void_starts"), r.get("swap_voids"))
    out = []
    for (mk, depth), blocks in sorted(rows.items(), key=lambda kv: (["0.5B", "1.5B", "7B"].index(kv[0][0]), kv[0][1])):
        fr, bad, nat, web, voids = [], [], [], [], []
        for b, arms in sorted(blocks.items()):
            if set(arms) != {"native", "webgpu"}:
                bad.append(f"block {b} incomplete")
                continue
            for a, v in arms.items():
                if v[1]:
                    bad.append(f"block {b} {a}: {v[1]}")
                if v[3] or v[4]:
                    voids.append(f"block {b} {a}: {v[3] or ''} {v[4] or ''}".strip())
            if not arms["native"][1] and not arms["webgpu"][1]:
                fr.append(arms["webgpu"][0] / arms["native"][0])
                nat.append(arms["native"][0]); web.append(arms["webgpu"][0])
        row = {"cell": f"decode {mk} @{depth}", "valid": not bad, "problems": bad, "voids_rerun": voids,
               "native_tok_s_median": round(statistics.median(nat), 1) if nat else None,
               "webgpu_tok_s_median": round(statistics.median(web), 1) if web else None,
               "fraction": summarize(fr) if fr else None,
               "decode_paths": sorted({v[2] for arms in blocks.values() for v in arms.values() if v[2]})}
        out.append(row)
    return out


def ttft(d):
    f = os.path.join(d, "ttft.json")
    if not os.path.exists(f):
        return []
    rows = {}
    for r in json.load(open(f)):
        why = r.get("error") or (None if path_ok(r["arm"] if r["arm"] == "webgpu" else "native", r.get("decode_path"))
                                 else f"decode path {r.get('decode_path')}")
        rows.setdefault(r["model"], {}).setdefault(r["block"], {})["webgpu" if r["arm"] == "webgpu" else "native"] = (
            r.get("ttft_ms_median"), why, r.get("decode_path"), r.get("void_attempts"), r.get("repeat_over_fresh"),
            r.get("prompt_tokens"))
    out = []
    for mk, blocks in rows.items():
        fr, bad, nat, web, voids, cache, pt = [], [], [], [], [], [], set()
        for b, arms in sorted(blocks.items()):
            if set(arms) != {"native", "webgpu"}:
                bad.append(f"block {b} incomplete")
                continue
            for a, v in arms.items():
                if v[1]:
                    bad.append(f"block {b} {a}: {v[1]}")
                if v[3]:
                    voids.append(f"block {b} {a}: {v[3]}")
                cache.append((a, v[4]))
                pt.update(v[5] or [])
            if not arms["native"][1] and not arms["webgpu"][1]:
                fr.append(arms["native"][0] / arms["webgpu"][0])
                nat.append(arms["native"][0]); web.append(arms["webgpu"][0])
        out.append({"cell": f"TTFT {mk} @1024", "valid": not bad, "problems": bad, "voids_rerun": voids,
                    "native_ttft_ms_median": round(statistics.median(nat), 1) if nat else None,
                    "webgpu_ttft_ms_median": round(statistics.median(web), 1) if web else None,
                    "fraction": summarize(fr) if fr else None, "prompt_tokens": sorted(pt),
                    "repeat_over_fresh": cache})
    return out


def main():
    d = sys.argv[1]
    rows = decode(d) + ttft(d)
    for r in rows:
        f = r["fraction"]
        fs = f"{f['median']:.3f} ({f['min']:.3f}-{f['max']:.3f}, {f['n']} pairs)" if f else "-"
        a = r.get("native_tok_s_median", r.get("native_ttft_ms_median"))
        b = r.get("webgpu_tok_s_median", r.get("webgpu_ttft_ms_median"))
        print(f"{r['cell']:<20} {'VALID' if r['valid'] else 'INVALID'}  native {a}  webgpu {b}  fraction {fs}"
              + (f"  problems: {r['problems']}" if r["problems"] else "") + (f"  voids re-run: {r['voids_rerun']}" if r["voids_rerun"] else ""))
    json.dump(rows, open(os.path.join(d, "grade.json"), "w"), indent=1)
    print("G-WG0:", "every cell valid" if rows and all(r["valid"] for r in rows) else "NOT every cell valid")


if __name__ == "__main__":
    main()
