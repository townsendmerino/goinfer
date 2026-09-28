#!/usr/bin/env python3
"""te2b_variance.py: TE2(b) of docs/tasks/task-test-efficiency-2026-09.md. Where does the variance of a served cell's
decode tok/s live: between server restarts, or inside one server lifetime?

Read-only and stdlib-only. It reads every bench_peer-shaped JSON under docs/measurements (recursive) and prints tables.
It runs no benchmark, server or test.

  usage: python3 docs/measurements/test-efficiency-2026-09/te2b_variance.py [--list-skipped] [--boot N]

WHAT bench_peer.py RESTARTS, read from its own code (scripts/bench_peer.py, run_cell() and main()):
  - main() calls run_cell() once per planned cell (`rates, err, counts = run_cell(engine, mk, depth, cfg, backend)`).
    run_cell() is documented as "Restart the server, do NRUNS runs of NCOMP completions". It Popen()s a fresh server,
    waits for the port, sends ONE discarded warm-up completion (`warm = post_stream(url, mk(), parse)`), then runs
    `for _ in range(nruns): for _ in range(ncomp): post_stream(...)`, and killpg()s the server in `finally`, followed by
    `time.sleep(3)`. So ONE CELL RECORD IS ONE SERVER LIFETIME, which this script calls a "restart".
  - A "run" is a block of NCOMP consecutive timed completions on that same server: NCOMP = 8, or 2 under DEEP_CTX
    (gen_params()), and NRUNS = 2 unless BENCH_RUNS overrides it. runs[] holds each block's mean. Nothing separates two
    runs: no restart, no pause, no re-warm.
  - counts.completion_rates holds every timed completion in order, NRUNS*NCOMP of them.
  Hierarchy: [session >] restart (cell record) > run (block of NCOMP back-to-back completions) > completion.

A REPLICATED CELL (a "group") is one set of the same host, serve binary, model, backend, depth, config, prompt
(prompt_tokens + prompt_format), ctx pin, GPU driver and NGEN x NCOMP shape, measured in more than one restart.
  - prompt_format is part of the key because the harness changed prompts at the same token count ("essay-v2", with
    the engine-reported token numerator, from 2026-09-25). A cell before and after that change is not the same cell.
  - goinfer binaries are identified by provenance serve_binaries / serve_binaries_old path + mtime. The identity is
    ENGINE-AGNOSTIC: a binary run as "goinfer" in one pass and "goinfer_old" in another (the L1 reversed pass) is the
    same build. A wrapper script (*.sh), a missing mtime or an absent entry leaves the build UNIDENTIFIED, and such a
    cell is grouped only with restarts in its own file.
  - peers: Ollama by bin path + version; llama-server by version string (it carries the build commit). MLX records no
    version, so it stays file-local.
  - CURATED EXCLUSIONS, each with the reason in VOID_FILES / ARM_FILES below. A void file is dropped. In an arm file,
    the named engines' treatment was set by something the record does not carry (an env var, a quant override or a
    different checkpoint), so those engines' cells are file-local. Section 7 re-runs the headline numbers with the
    curation and the prompt key removed, to show how much they matter. An unrecorded arm left in the pool can only
    INFLATE the between-restart and between-session estimates. When new data arrives, add to these lists.

MODEL: y = ln(tok/s), a nested random-effects model per group, estimated by method-of-moments. This is a general
unbalanced nested ANOVA: for each level L, E[SS_L] = sum over M finer-or-equal of c_LM * s2_M, and c_LM comes from the
unit sizes. It is pooled across the groups of a stratum by summing SS and c. Negative estimates are shown as 0 with *.
On the log scale, 100*sd reads as a CV in percent, and a component is additive for a ratio, which is how gates read.
  - 3-level (sections 2, 3): restart > run > completion. Here "restart" includes whatever differs between the two
    restarts, so for restarts in different sessions it includes session drift.
  - 4-level (section 4): session > restart > run > completion. A session is a chain of this host's bench_peer files
    whose start times are < SESSION_GAP_H apart. Here "restart" is restart-to-restart variance inside one session.
    That within-session drift, such as the Mac's pass-to-pass order effect, still sits in it: the records cannot
    separate the two.
  - Section 5 is a model-free cross-check: pairwise semivariance of restart means by separation (same pass / same
    session / other session), net of each group's own within-server noise.

THE KILL LINE (task doc, TE2 Kill (b)): "between-restart variance is >= half the total". The doc says to split
"the recorded runs[]", so the PRIMARY total is the variance of one recorded run mean:
      V_run = s2_R + s2_B + s2_C / n_c        (plus s2_S in the 4-level table)
  and the restart share is s2_R / V_run. Two more bases are printed. The per-completion basis is the most lenient.
  The cell-mean basis at BENCH_RUNS=3 is the strictest, because it is the variance of the number a gate reads.

LIMITS: the records carry no per-cell timestamp, so sessions come from each file's start time (the earliest provenance
utc, resumes included). The cost model (section 6) brackets a restart's cost between a floor and a ceiling taken from
the cell wall `secs`, because the split of non-decode time into per-start and per-completion is not recorded (TE0 adds
it). `secs` excludes the idle-gate wait, which bench_peer.py spends before t0 and which is paid once per restart.
"""
import argparse
import ast
import collections
import datetime as dt
import glob
import gzip
import json
import math
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
MEAS = REPO / "docs" / "measurements"

# ---- curation. Every entry carries its reason, and every reason was read from the record's own writeup or script. ----
VOID_FILES = {
    "peer-claim-2026-09-25/void-attempt1-a-e-dense.json":
        "declared void by peer-claim-2026-09-25.md ('nothing from them is used'; LD_LIBRARY_PATH broke llama-server)",
    "r13-cpu-depth-row-2026-09-19.json":
        "declared thermally contaminated from its midpoint by r13-cpu-depth-row-2026-09-19.md; ran at "
        "BENCH_MAX_LOADAVG=4.0",
}
_OPTFWD = "GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record"
ARM_FILES = {  # path relative to docs/measurements -> (engines whose cells stay file-local, reason)
    "q4k-peer-2026-09-26/s1-q4k.json": (("goinfer",), "q4k checkpoint vs s2's int4: same binary, different weights"),
    "q4k-peer-2026-09-26/s2-int4.json": (("goinfer",), "int4 checkpoint vs s1's q4k: same binary, different weights"),
    "f16kv-baseline-2026-09-26/a-int8int8-g32.json": (("goinfer",), "BENCH_QUANT_OVERRIDE phi3-mini=int8int8 (run.sh)"),
    "f16kv-baseline-2026-09-26/b-int4.json": (("goinfer",), "BENCH_QUANT_OVERRIDE phi3-mini=int4 (run.sh)"),
    "peer-matrix-2026-09/mac-m1pro-quant-int8int8-2026-09-04.json": (("goinfer",), "a quant-override pass (int8int8)"),
    "metal-depth-r2-2026-09-18-raw.json":
        (("ollama",), "Ollama ran with OLLAMA_KV_CACHE_TYPE=q8_0 (metal-depth-r2-2026-09-18.md)"),
    "metal-depth-r2-kvcache-isolation-2026-09-18-raw.json":
        (("ollama",), "Ollama isolation arm, OLLAMA_FLASH_ATTENTION=1 (r2-kvcache-isolation-run.log)"),
}
for _p in glob.glob(str(MEAS / "g2[678]-*.json")):
    _n = os.path.basename(_p)
    if "optfwd" in _n or _n.endswith("-on.json") or _n.endswith("-off.json"):
        ARM_FILES[_n] = (("goinfer",), _OPTFWD)

SESSION_GAP_H = 3.0  # this host's files whose start times chain within this gap form one session
DESIGN_RUNS = 3      # BENCH_RUNS of the current gates (L1, peer-claim): the cell-mean basis and the design table
SETTLE_S = 3.0       # run_cell's `time.sleep(3)` after teardown, inside `secs`


def rel(p):
    return os.path.relpath(p, MEAS)


def load(p):
    """-> (obj, form) or (None, error). Handles .gz, a whole-file JSON value and JSON-lines."""
    try:
        if p.endswith(".gz"):
            with gzip.open(p, "rt", errors="replace") as f:
                t = f.read()
        else:
            with open(p, errors="replace") as f:
                t = f.read()
    except Exception as e:
        return None, f"read error: {e}"
    try:
        return json.loads(t), "json"
    except Exception as e1:
        try:
            return [json.loads(l) for l in t.splitlines() if l.strip()], "jsonl"
        except Exception as e2:
            return None, f"not JSON ({str(e1)[:50]}) nor JSON-lines ({str(e2)[:50]})"


def parse_utc(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def hostclass(h):
    h = (h or "").lower()
    return "mac" if "macbook" in h else ("nobara" if "nobara" in h else None)


def binary_id(prov, engine, backend):
    """-> (identity string or None, reason when None)."""
    if engine in ("goinfer", "goinfer_old"):
        table = prov.get("serve_binaries" if engine == "goinfer" else "serve_binaries_old") or {}
        b = table.get(backend) or {}
        path, mtime = b.get("path"), b.get("mtime")
        if not path:
            return None, "no serve binary recorded"
        if path.endswith(".sh"):
            return None, "serve path is a wrapper script (the build it execs is unrecorded)"
        if not mtime:
            return None, "serve binary has no mtime (absent at provenance time)"
        return f"{os.path.basename(path)}@{mtime[5:16]}", None
    if engine == "ollama":
        pe = prov.get("peer") or {}
        v, b = pe.get("version"), pe.get("ollama_bin") or pe.get("bin")
        return (f"ollama {v} {b}", None) if v and b else (None, "Ollama version not recorded")
    if engine == "llamacpp":
        v = (prov.get("peer_llamacpp") or {}).get("version")
        return (f"llama-server {v}", None) if v else (None, "llama-server version not recorded")
    return None, f"{engine}: no version recorded"


def short_bin(bid, engine):
    if not bid:
        return f"{engine}(file-local)"
    if bid.startswith("ollama"):
        return "ollama " + bid.split()[1]
    if bid.startswith("llama-server"):
        return "llama-server " + bid.split("commit ")[-1].rstrip(")")[:7]
    return bid.split("@")[0]


def engclass(e):
    return "goinfer" if e in ("goinfer", "goinfer_old", "goinfer*") else "peer"


def discover():
    paths = sorted(set(glob.glob(str(MEAS / "**" / "*.json"), recursive=True)
                       + glob.glob(str(MEAS / "**" / "*.json.gz"), recursive=True)
                       + glob.glob(str(MEAS / "**" / "*.jsonl"), recursive=True)
                       + glob.glob(str(MEAS / "**" / "*.jsonl.gz"), recursive=True)))
    files, skipped = [], collections.defaultdict(list)
    for p in paths:
        obj, form = load(p)
        if obj is None:
            skipped["UNPARSEABLE: " + form].append(rel(p))
            continue
        if isinstance(obj, list) and obj and isinstance(obj[0], dict) and obj[0].get("kind") == "provenance":
            files.append((rel(p), form, obj))
        elif isinstance(obj, dict) and "provenance" in obj:
            skipped["not bench_peer-shaped: a dict with its own provenance/cells (another harness)"].append(rel(p))
        elif isinstance(obj, list):
            skipped["not bench_peer-shaped: a list without a provenance header"].append(rel(p))
        else:
            skipped["not bench_peer-shaped: other JSON"].append(rel(p))
    return paths, files, skipped


def sessions_of(files):
    """file path -> session id: this host's files chained by start time with gaps < SESSION_GAP_H."""
    by_host = collections.defaultdict(list)
    for path, _, obj in files:
        prov = obj[0]
        ts = [parse_utc(prov.get("utc") or "")] + [parse_utc(s.get("utc") or "") for s in
                                                    (prov.get("resumed_from") or []) if isinstance(s, dict)]
        ts = [t for t in ts if t]
        by_host[hostclass(prov.get("host"))].append((min(ts) if ts else None, path))
    out, t0 = {}, {}
    for h, lst in by_host.items():
        sid, prev, first = 0, None, None
        for t, path in sorted(lst, key=lambda x: (x[0] is None, x[0] or dt.datetime.min, x[1])):
            if t is None:
                out[path] = f"{h}:undated:{path}"
                continue
            if prev is None or (t - prev).total_seconds() >= SESSION_GAP_H * 3600:
                sid, first = sid + 1, t
            out[path], prev, t0[path] = f"{h}:s{sid:02d} {first:%m-%d %H:%M}Z", t, t
    return out, t0


def extract(files, curate=True, prompt_key=True):
    """-> (cells, coverage counter). A cell is one restart with its runs of log-rates."""
    sess, t0s = sessions_of(files)
    cells, cov = [], collections.Counter()
    for path, form, obj in files:
        prov = obj[0]
        hc = hostclass(prov.get("host"))
        driver = (prov.get("gpu") or {}).get("driver") if isinstance(prov.get("gpu"), dict) else None
        ctx = (prov.get("ctx_pin") or 0, prov.get("deep_ctx") or 0)
        void = curate and path in VOID_FILES
        arm_engines = ARM_FILES.get(path, ((), ""))[0] if curate else ()
        for i, r in enumerate(obj[1:]):
            cov["records"] += 1
            if not isinstance(r, dict) or not r.get("engine"):
                cov["dropped: no engine field (not a run_cell record)"] += 1
                continue
            if not r.get("runs"):
                cov["dropped: errored cell (no runs)"] += 1
                continue
            if hc is None:
                cov["dropped: host not recorded"] += 1
                continue
            c = r.get("counts")
            if isinstance(c, str):
                try:
                    c = ast.literal_eval(c)
                except Exception:
                    c = None
            cr = (c or {}).get("completion_rates")
            runs = r["runs"]
            if not cr:
                cov["dropped: no completion_rates (harness before per-completion rates were kept)"] += 1
                continue
            if len(cr) % len(runs):
                cov["dropped: completion_rates do not split into runs[] blocks"] += 1
                continue
            k = len(cr) // len(runs)
            blocks = [cr[j * k:(j + 1) * k] for j in range(len(runs))]
            if any(abs(sum(b) / k - m) > 1e-9 * max(1.0, abs(m)) for b, m in zip(blocks, runs)) or \
                    any(x <= 0 for x in cr):
                cov["dropped: completion_rates blocks do not reproduce runs[]"] += 1
                continue
            if void:
                cov["dropped: void file (curated)"] += 1
                continue
            eng = r["engine"]
            bid, why = binary_id(prov, eng, r.get("backend"))
            local = None
            if bid is None:
                local = f"unidentified: {why}"
            elif eng in arm_engines:
                local = "arm file (curated)"
            ekey = "goinfer*" if eng in ("goinfer", "goinfer_old") else eng
            prompt = (r.get("prompt_tokens"), r.get("prompt_format") or "pre-essay-v2") if prompt_key else None
            key = (hc, ekey, bid if local is None else f"LOCAL:{path}:{eng}", r.get("model"), r.get("backend"),
                   r.get("depth"), r.get("config"), prompt, ctx, driver, c.get("ngen"), k)
            # decode seconds actually timed: per completion (tokens - 1) / rate; tokens from the engine where recorded
            ctoks = c.get("completion_tokens")
            per = [((ctoks[j] if ctoks and j < len(ctoks) and ctoks[j] else
                     (c.get("tokens") / len(cr) if c.get("tokens") else c.get("ngen") or 64)) - 1) / x
                   for j, x in enumerate(cr)]
            cells.append({"key": key, "path": path, "session": sess.get(path), "t0": t0s.get(path), "idx": i,
                          "engine": eng, "bin": bid, "local": local, "host": hc, "model": r.get("model"),
                          "backend": r.get("backend"), "depth": r.get("depth"), "config": r.get("config"),
                          "secs": r.get("secs"), "ngen": c.get("ngen"), "ncomp": k, "decode_s": sum(per),
                          "runs": [[math.log(x) for x in b] for b in blocks]})
            cov["used: cell with consistent completion_rates"] += 1
    return cells, cov


# ---- general unbalanced nested ANOVA (method of moments), additive across groups so strata can pool ----
def nested(obs, depth):
    """obs: list of (path tuple of length `depth`, y). Levels 1..depth are the path prefixes; level depth+1 is the
    single observation. Returns {"SS": [..], "df": [..], "c": [[..]], "N": n, "units": [..]} for levels 1..depth+1,
    with E[SS_l] = sum_{m >= l} c[l][m] * s2_m (0-based lists, index 0 = level 1)."""
    L = depth + 1
    N = len(obs)
    units = [collections.defaultdict(lambda: [0, 0.0]) for _ in range(L + 1)]  # level 0 = grand
    for idx, (path, y) in enumerate(obs):
        for l in range(L + 1):
            key = path[:l] if l <= depth else path + (idx,)
            u = units[l][key]
            u[0] += 1
            u[1] += y
    def sq(l):
        return sum(s * s / n for n, s in units[l].values())
    SS = [sq(l) - sq(l - 1) for l in range(1, L + 1)]
    df = [len(units[l]) - len(units[l - 1]) for l in range(1, L + 1)]
    # A[l][m] = sum over units u at level l of (N_u if m <= l else sum_{v at m, v in u} N_v^2 / N_u)
    child = [collections.defaultdict(float) for _ in range(L + 1)]
    A = [[0.0] * (L + 1) for _ in range(L + 1)]
    for l in range(L + 1):
        for m in range(1, L + 1):
            if m <= l:
                A[l][m] = float(N)
            else:
                acc = collections.defaultdict(float)
                for vkey, (nv, _) in units[m].items():
                    acc[vkey[:l]] += nv * nv
                A[l][m] = sum(acc[ukey] / nu for ukey, (nu, _) in units[l].items())
    c = [[(A[l][m] - A[l - 1][m]) if m >= l else 0.0 for m in range(1, L + 1)] for l in range(1, L + 1)]
    return {"SS": SS, "df": df, "c": c, "N": N, "groups": 1,
            "units": [len(units[l]) for l in range(1, L + 1)]}


def pool(ms):
    ms = list(ms)
    if not ms:
        return None
    L = len(ms[0]["SS"])
    return {"SS": [sum(m["SS"][l] for m in ms) for l in range(L)],
            "df": [sum(m["df"][l] for m in ms) for l in range(L)],
            "c": [[sum(m["c"][l][k] for m in ms) for k in range(L)] for l in range(L)],
            "N": sum(m["N"] for m in ms), "groups": sum(m["groups"] for m in ms),
            "units": [sum(m["units"][l] for m in ms) for l in range(L)]}


def solve(m):
    """-> raw component estimates s2 for levels 1..L (may be negative; nan where a level has no replication)."""
    L = len(m["SS"])
    s2 = [float("nan")] * L
    for l in range(L - 1, -1, -1):
        rest = m["SS"][l] - sum(m["c"][l][k] * s2[k] for k in range(l + 1, L) if s2[k] == s2[k])
        s2[l] = rest / m["c"][l][l] if m["c"][l][l] > 1e-9 else float("nan")
    return s2


def pos(x):
    return max(x, 0.0) if x == x else 0.0


def sd(x):
    if x != x:
        return "   n/a"
    return ("%5.2f" % (100 * math.sqrt(x))) if x >= 0 else " 0.00*"


def shares3(s2, nc):
    R, Bv, C = (pos(x) for x in s2)
    tot = R + Bv + C
    v_run = R + Bv + C / nc
    v_cell = R + Bv / DESIGN_RUNS + C / (DESIGN_RUNS * nc)
    f = lambda x, t: 100 * x / t if t > 0 else float("nan")
    return {"cR": f(R, tot), "cB": f(Bv, tot), "cC": f(C, tot), "run": f(R, v_run), "cell": f(R, v_cell)}


def boot(glist, reps, stat, seed=1):
    """90% interval of stat(pooled) under resampling of groups."""
    if len(glist) < 3 or reps <= 0:
        return None
    rnd = random.Random(seed)
    vals = sorted(v for v in (stat(pool(rnd.choice(glist) for _ in range(len(glist)))) for _ in range(reps)) if v == v)
    return (vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals)) - 1]) if vals else None


def fmt_iv(iv):
    return f"{iv[0]:4.0f}-{iv[1]:3.0f}" if iv else "     -"


def groups_of(cells):
    g = collections.defaultdict(list)
    for c in cells:
        g[c["key"]].append(c)
    return {k: v for k, v in g.items() if len(v) >= 2}


def obs3(v):
    return [((ri, bi), y) for ri, c in enumerate(v) for bi, b in enumerate(c["runs"]) for y in b]


def obs4(v):
    return [((c["session"], ri, bi), y) for ri, c in enumerate(v) for bi, b in enumerate(c["runs"]) for y in b]


def pair_class(c1, c2):
    if c1["path"] == c2["path"]:
        return "P same pass"
    if c1["session"] == c2["session"]:
        return "S same session"
    return "O other session"


def label(k, members):
    hc, ek, bid, model, be, depth, cfg, prompt, ctx, drv, ngen, nc = k
    b = short_bin(members[0]["bin"], members[0]["engine"]) if not str(bid).startswith("LOCAL:") \
        else short_bin(None, members[0]["engine"])
    extra = "" if (ctx == (0, 0) and ngen == 64 and nc == 8) else f" ctx{ctx[0] or ctx[1]} {ngen}x{nc}"
    era = " v2" if prompt and prompt[1] == "essay-v2" else ""
    return hc, b, f"{model}", f"{be} d{depth} {cfg}{era}{extra}"


def robust(vals):
    """median, % of groups >= 25 and >= 50, of per-group shares (nan dropped)."""
    v = sorted(x for x in vals if x == x)
    if not v:
        return "     -"
    return f"{v[len(v) // 2]:4.0f} {100 * sum(x >= 25 for x in v) / len(v):4.0f} {100 * sum(x >= 50 for x in v) / len(v):4.0f}" \
           f" (n={len(v)})"


def table3(title, strata, reps):
    print(f"\n{title}\n")
    print(f"  {'stratum':<22} {'grp':>4} {'rst':>4} {'runs':>5} {'comps':>6}   {'sdR%':>6} {'sdB%':>6} {'sdC%':>6}   "
          f"{'R/B/C % of 1 comp':>18}   {'R % run mean':>12} {'90% boot':>9}   {'R % cell mean':>13}   "
          f"{'per-group R % run: median, % >=25, % >=50':>41}")
    for name, glist in strata:
        m = pool(glist)
        s2 = solve(m)
        nc = m["N"] / m["units"][1]
        s = shares3(s2, nc)
        iv = boot(glist, reps, lambda p: shares3(solve(p), p["N"] / p["units"][1])["run"])
        per = [shares3(solve(g), g["N"] / g["units"][1])["run"] for g in glist]
        print(f"  {name:<22} {m['groups']:>4} {m['units'][0]:>4} {m['units'][1]:>5} {m['N']:>6}   "
              f"{sd(s2[0]):>6} {sd(s2[1]):>6} {sd(s2[2]):>6}   {s['cR']:5.0f} /{s['cB']:4.0f} /{s['cC']:4.0f}   "
              f"{s['run']:12.0f} {fmt_iv(iv):>9}   {s['cell']:13.0f}   {robust(per):>41}")


def shares4(s2, nc):
    S, R, Bv, C = (pos(x) for x in s2)
    v_run = S + R + Bv + C / nc
    v_cell = S + R + Bv / DESIGN_RUNS + C / (DESIGN_RUNS * nc)
    f = lambda x, t: 100 * x / t if t > 0 else float("nan")
    return {"S": f(S, v_run), "R": f(R, v_run), "W": f(Bv + C / nc, v_run),
            "Rw": f(R, R + Bv + C / nc), "Rcell": f(R, v_cell - S), "SR": f(S + R, v_run)}


def table4(title, strata, reps):
    print(f"\n{title}\n")
    print(f"  {'stratum':<22} {'grp':>4} {'sess':>5} {'rst':>4} {'runs':>5}   {'sdS%':>6} {'sdR%':>6} {'sdB%':>6} "
          f"{'sdC%':>6}   {'share of one run mean: S / R / within':>37}   {'R % in-session':>14} {'90% boot':>9}   "
          f"{'(S+R) %':>7} {'90% boot':>9}   {'in-session restart pairs':>24}")
    for name, glist in strata:
        m = pool(glist)
        s2 = solve(m)
        nc = m["N"] / m["units"][2]
        s = shares4(s2, nc)
        iv_r = boot(glist, reps, lambda p: shares4(solve(p), p["N"] / p["units"][2])["Rw"])
        iv_sr = boot(glist, reps, lambda p: shares4(solve(p), p["N"] / p["units"][2])["SR"])
        # restarts that share a session with another restart of the same cell: the only data s2_R rests on
        insess = m["units"][1] - m["units"][0]
        print(f"  {name:<22} {m['groups']:>4} {m['units'][0]:>5} {m['units'][1]:>4} {m['units'][2]:>5}   "
              f"{sd(s2[0]):>6} {sd(s2[1]):>6} {sd(s2[2]):>6} {sd(s2[3]):>6}   "
              f"{s['S']:15.0f} /{s['R']:5.0f} /{s['W']:6.0f}         {s['Rw']:14.0f} {fmt_iv(iv_r):>9}   "
              f"{s['SR']:7.0f} {fmt_iv(iv_sr):>9}   {insess:>18} df")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-skipped", action="store_true", help="name every non-bench_peer JSON, not just counts")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates for the pooled shares (0 = off)")
    args = ap.parse_args()

    paths, files, skipped = discover()
    cells, cov = extract(files)
    G = groups_of(cells)
    M3 = {k: nested(obs3(v), 2) for k, v in G.items()}
    M4 = {k: nested(obs4(v), 3) for k, v in G.items()}

    def strata(M, keyf):
        s = collections.defaultdict(list)
        for k, m in M.items():
            s[keyf(k)].append(m)
        return sorted(s.items())

    # ---------------- 1. coverage ----------------
    print("## 1. data coverage\n")
    print(f"JSON / JSON-lines files under docs/measurements: {len(paths)}")
    print(f"  bench_peer-shaped (list, element 0 kind=provenance): {len(files)} "
          f"({sum(1 for f in files if f[1] == 'json')} whole-file JSON, {sum(1 for f in files if f[1] == 'jsonl')} JSON-lines)")
    for why, ps in sorted(skipped.items()):
        print(f"  skipped, {why}: {len(ps)}")
        if why.startswith("UNPARSEABLE") or "provenance/cells" in why or args.list_skipped:
            for p in ps:
                print(f"      {p}")
    print("\ncell records in the bench_peer-shaped files:")
    for k in sorted(cov, key=lambda k: (not k.startswith("records"), k)):
        print(f"  {cov[k]:>5}  {k}")
    nloc = collections.Counter(c["local"] for c in cells if c["local"])
    for k, v in sorted(nloc.items()):
        print(f"         of the used cells, file-local because {k}: {v}")
    print("\ncurated exclusions (reasons from each record's own writeup or script):")
    for p, why in VOID_FILES.items():
        print(f"  VOID  {p}: {why}")
    for p, (eng, why) in sorted(ARM_FILES.items()):
        print(f"  ARM   {p} [{','.join(eng)} file-local]: {why}")
    inrep = sum(len(v) for v in G.values())
    ns = len({c["session"] for v in G.values() for c in v})
    print(f"\nreplicated groups (>= 2 restarts of one cell): {len(G)} groups, {inrep} restarts, "
          f"{sum(m['units'][1] for m in M3.values())} runs, {sum(m['N'] for m in M3.values())} completions, "
          f"from {len({c['path'] for v in G.values() for c in v})} files in {ns} sessions")
    per_file = collections.Counter(c["path"] for v in G.values() for c in v)
    sess_of = {c["path"]: c["session"] for c in cells}
    print("files contributing restarts to replicated groups, by session:")
    for p, n in sorted(per_file.items(), key=lambda kv: (sess_of[kv[0]], kv[0])):
        print(f"  {sess_of[p]:<24} {n:>3}  {p}")

    # ---------------- 2. per group ----------------
    print("\n## 2. per replicated cell: 3-level components of ln(tok/s) (restart > run > completion)\n")
    print("sd = 100*sqrt(component), a CV in %; * = negative estimate shown as 0. 'R%run' = restart share of one run")
    print(f"mean's variance (the kill-line basis); 'R%cell' = share for a {DESIGN_RUNS}-run cell mean. 'pairs' counts "
          "restart pairs: P same pass, S same session, O other session. 'v2' = essay-v2 prompts.\n")
    print(f"  {'host':<6} {'binary':<21} {'model':<9} {'cell':<27} {'rst':>3} {'runs':>4} {'comp':>4} {'mean':>7}  "
          f"{'sdR%':>6} {'sdB%':>6} {'sdC%':>6}  {'R/B/C % 1-comp':>14}  {'R%run':>5} {'R%cell':>6}  pairs")
    rows = []
    for k, v in G.items():
        m = M3[k]
        s2 = solve(m)
        s = shares3(s2, m["N"] / m["units"][1])
        pc = collections.Counter(pair_class(v[i], v[j])[0] for i in range(len(v)) for j in range(i + 1, len(v)))
        mean = math.exp(sum(y for _, y in obs3(v)) / m["N"])
        rows.append((label(k, v), m, s2, s, pc, mean))
    for (lab, m, s2, s, pc, mean) in sorted(rows, key=lambda r: r[0]):
        pcs = " ".join(f"{c}{n}" for c, n in sorted(pc.items()))
        print(f"  {lab[0]:<6} {lab[1][:21]:<21} {lab[2]:<9} {lab[3][:27]:<27} {m['units'][0]:>3} {m['units'][1]:>4} "
              f"{m['N']:>4} {mean:7.1f}  {sd(s2[0]):>6} {sd(s2[1]):>6} {sd(s2[2]):>6}  "
              f"{s['cR']:4.0f}/{s['cB']:3.0f}/{s['cC']:3.0f}     {s['run']:5.0f} {s['cell']:6.0f}  {pcs}")
    nhalf = sum(1 for r in rows if r[3]["run"] >= 50)
    print(f"\n  groups whose restart share of a run mean is >= 50%: {nhalf} of {len(rows)}. A single group has 2-13 "
          "restarts, so one group's estimate is very noisy; read the pooled tables.")

    # ---------------- 3. pooled 3-level ----------------
    table3("## 3. pooled 3-level, per machine (restart here includes any drift between the restarts' sessions)",
           strata(M3, lambda k: k[0]), args.boot)
    table3("## 3a. pooled 3-level, per machine x model", strata(M3, lambda k: f"{k[0]} {k[3]}"), args.boot)
    table3("## 3b. pooled 3-level, per machine x engine class", strata(M3, lambda k: f"{k[0]} {engclass(k[1])}"),
           args.boot)
    table3("## 3c. pooled 3-level, per machine x backend", strata(M3, lambda k: f"{k[0]} {k[4]}"), args.boot)

    # ---------------- 3d. within-server components from every used cell ----------------
    print("\n## 3d. within-server components from EVERY used cell (one restart is enough to estimate run and completion)\n")
    print("Pooled 2-level ANOVA inside each restart (run > completion). 'W' = within-server variance of one run mean,")
    print("s2_B + s2_C/n_c. This is the noise that more completions per restart would average down.\n")
    WS = collections.defaultdict(list)
    for c in cells:
        m = nested([((bi,), y) for bi, b in enumerate(c["runs"]) for y in b], 1)
        WS[(c["host"], engclass(c["engine"]), c["model"], c["backend"])].append(m)
    wcomp = {}
    print(f"  {'machine':<7} {'engine':<8} {'model':<9} {'backend':<7} {'cells':>5} {'runs':>5}   {'sdB%':>6} {'sdC%':>6} "
          f"{'sdW%':>6}   {'B share of W':>12}")
    for k, gl in sorted(WS.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        m = pool(gl)
        if m["df"][0] <= 0:
            continue
        s2B, s2C = solve(m)
        nc = m["N"] / m["units"][0]
        W = pos(s2B) + pos(s2C) / nc
        wcomp[k] = (s2B, s2C)
        print(f"  {k[0]:<7} {k[1]:<8} {str(k[2]):<9} {str(k[3]):<7} {len(gl):>5} {m['units'][0]:>5}   {sd(s2B):>6} "
              f"{sd(s2C):>6} {sd(W):>6}   {100 * pos(s2B) / W if W > 0 else float('nan'):11.0f}%")

    # ---------------- 4. pooled 4-level ----------------
    print(f"\n(4-level tables: a session = this host's files chained by start time with gaps < {SESSION_GAP_H:g} h. "
          "'R % in-session' = s2_R / (s2_R + within-server variance of a run mean): the kill line with session drift "
          "taken out. '(S+R) %' = the same with session drift left in.)")
    table4("## 4. pooled 4-level (session > restart > run > completion), per machine", strata(M4, lambda k: k[0]),
           args.boot)
    table4("## 4a. pooled 4-level, per machine x model", strata(M4, lambda k: f"{k[0]} {k[3]}"), args.boot)
    table4("## 4b. pooled 4-level, per machine x engine class", strata(M4, lambda k: f"{k[0]} {engclass(k[1])}"),
           args.boot)
    table4("## 4c. pooled 4-level, per machine x backend", strata(M4, lambda k: f"{k[0]} {k[4]}"), args.boot)

    # ---------------- 5. pairwise semivariance by separation ----------------
    print("\n## 5. cross-check: restart-to-restart semivariance by separation, net of each cell's own within-server "
          "noise\n")
    print("For each pair of restarts of one cell: (ybar_i - ybar_j)^2 / 2 - (w_i + w_j) / 2, where w is the within-server")
    print("variance of that restart's mean from the cell's OWN 3-level s2_B and s2_C (raw, so the average is unbiased).")
    print("Its expectation is the restart component plus any drift between the two restarts. The mean is over pairs,")
    print("and 'sd%' = 100*sqrt(mean). The median is printed too, because a few pairs carry most of the sum.\n")
    acc = collections.defaultdict(list)
    for k, v in G.items():
        _, s2B, s2C = solve(M3[k])
        s2B = 0.0 if s2B != s2B else s2B
        w, mu = [], []
        for c in v:
            Ni = sum(len(r) for r in c["runs"])
            w.append(s2B * sum(len(r) ** 2 for r in c["runs"]) / Ni ** 2 + s2C / Ni)
            mu.append(sum(sum(r) for r in c["runs"]) / Ni)
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                cl = pair_class(v[i], v[j])
                g = (mu[i] - mu[j]) ** 2 / 2 - (w[i] + w[j]) / 2
                d = 100 * (math.exp(abs(mu[i] - mu[j])) - 1)
                for st in (k[0], f"{k[0]} {engclass(k[1])}", f"{k[0]} {k[4]}"):
                    acc[(st, cl)].append((g, k, d))
    print(f"  {'stratum':<16} {'separation':<16} {'pairs':>6} {'cells':>6}   {'mean (sd%)':>10} {'median (sd%)':>13}   "
          f"{'|restart-mean diff| median / p90 / max %':>40}")
    for (st, cl), lst in sorted(acc.items()):
        e = sum(g for g, _, _ in lst) / len(lst)
        med = sorted(g for g, _, _ in lst)[len(lst) // 2]
        ds = sorted(d for _, _, d in lst)
        print(f"  {st:<16} {cl:<16} {len(lst):>6} {len({k for _, k, _ in lst}):>6}   {sd(e):>10} {sd(med):>13}   "
              f"{ds[len(ds) // 2]:17.2f} / {ds[min(len(ds) - 1, int(0.9 * len(ds)))]:5.2f} / {ds[-1]:5.2f}")

    # ---------------- 5b. within-server trend ----------------
    print("\n## 5b. within-server trend: last run vs first run inside one server lifetime (every used restart with >= 2 "
          "runs)\n")
    print("A run level that only adds noise averages away with more runs. A TREND does not: if the rate drifts over a")
    print("server's lifetime, a cell with more completions per restart measures a different mean, not a sharper one.")
    print("d = 100*(exp(mean ln rate of last run - of first run) - 1) per restart; 't' = mean(d) / (sd(d)/sqrt(n)).\n")
    tr = collections.defaultdict(list)
    for c in cells:
        if len(c["runs"]) >= 2:
            f, l = c["runs"][0], c["runs"][-1]
            d = 100 * (math.exp(sum(l) / len(l) - sum(f) / len(f)) - 1)
            tr[(c["host"], c["backend"], engclass(c["engine"]))].append((d, c))
    print(f"  {'stratum':<24} {'restarts':>8}   {'mean d%':>7} {'median d%':>9} {'t':>6}   {'% d<-2%':>7} {'% d>+2%':>7}")
    worst, trend_flag = [], {}
    trm = collections.defaultdict(list)
    for lst in tr.values():
        for d, c in lst:
            trm[(c["host"], c["backend"], engclass(c["engine"]), c["model"])].append(d)
    # flag, per model: a stratum where a quarter or more of its restarts move > 2% between first and last run
    for k, ds in trm.items():
        trend_flag[k] = (sum(abs(x) > 2 for x in ds) / len(ds)) >= 0.25
    for k, lst in sorted(tr.items()):
        ds = [d for d, _ in lst]
        n = len(ds)
        mu = sum(ds) / n
        sdv = math.sqrt(sum((x - mu) ** 2 for x in ds) / (n - 1)) if n > 1 else float("nan")
        t = mu / (sdv / math.sqrt(n)) if n > 1 and sdv > 0 else float("nan")
        print(f"  {' '.join(k):<24} {n:>8}   {mu:7.2f} {sorted(ds)[n // 2]:9.2f} {t:6.1f}   "
              f"{100 * sum(x < -2 for x in ds) / n:6.0f}% {100 * sum(x > 2 for x in ds) / n:6.0f}%")
        worst += lst
    print("\n  per-model strata flagged (>= 25% of restarts move > 2% first run -> last run):")
    for k, ds in sorted(trm.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if trend_flag[k]:
            print(f"    {' '.join(str(x) for x in k):<32} restarts {len(ds):>3}  moved >2%: "
                  f"{sum(abs(x) > 2 for x in ds):>3}  median d {sorted(ds)[len(ds) // 2]:6.1f}%  "
                  f"min {min(ds):6.1f}%  max {max(ds):6.1f}%")
    print("\n  the largest |d|, with the cell (how often a trend reproduces across restarts is the tell):")
    for d, c in sorted(worst, key=lambda x: -abs(x[0]))[:12]:
        print(f"    {d:7.1f}%  {c['host']:<6} {c['engine']:<11} {c['model']:<9} {c['backend']:<6} d{c['depth']:<5} "
              f"{c['config']:<16} {c['path']}")

    # ---------------- 6. cost and design ----------------
    print("\n## 6. what a restart costs, and what that buys\n")
    print("Per cell: T = secs (start, load, warm-up, all timed completions with their prefill, teardown, the 3 s settle)")
    print("and D = timed decode seconds, summed over its completions as (tokens - 1) / rate. Overhead O = T - D. How O")
    print("splits between per-start and per-completion work (each completion's prompt prefill) is not recorded, so a")
    print("restart's cost a is bracketed: a_hi = O (all overhead is per-start), a_lo = 3 s settle + one warm-up")
    print("completion's decode (D/n). The per-completion cost b runs the other way: b_lo = D/n, b_hi = D/n + (O - a_lo)/n.")
    print("The idle-gate wait is NOT included, and it is paid once per restart (section 1.2 of the task doc: 44% of")
    print("gated wall).\n")
    cost = collections.defaultdict(list)
    for c in cells:
        if c["config"] == "greedy" and c["depth"] == 128 and c["ngen"] == 64 and c["secs"]:
            n = sum(len(r) for r in c["runs"])
            cost[(c["host"], engclass(c["engine"]), c["model"], c["backend"])].append((c["secs"], c["decode_s"], n))
    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(f"  {'machine':<7} {'engine':<8} {'model':<9} {'backend':<7} {'cells':>5}   {'T med s':>7} {'D med s':>7} "
          f"{'O med s':>7}   {'a_lo':>5} {'a_hi':>6}   {'b_lo':>5} {'b_hi':>5}   {'O share of T':>12}")
    costs = {}
    for k, pts in sorted(cost.items()):
        if len(pts) < 2:
            continue
        T = med([p[0] for p in pts])
        D = med([p[1] for p in pts])
        n = med([p[2] for p in pts])
        O = med([p[0] - p[1] for p in pts])
        a_lo, a_hi = SETTLE_S + D / n, max(O, SETTLE_S + D / n)
        b_lo, b_hi = D / n, D / n + max(O - a_lo, 0.0) / n
        costs[k] = (a_lo, a_hi, b_lo, b_hi)
        print(f"  {k[0]:<7} {k[1]:<8} {k[2]:<9} {k[3]:<7} {len(pts):>5}   {T:7.1f} {D:7.1f} {O:7.1f}   {a_lo:5.1f} "
              f"{a_hi:6.1f}   {b_lo:5.2f} {b_hi:5.2f}   {100 * O / T:11.0f}%")

    print("\nDesign comparison for one goinfer cell mean inside one session (a gate's paired reading), per machine x")
    print("model x backend. A design is R restarts x r runs x 8 completions in the same session:")
    print("  Var = s2_R/R + s2_B/(R r) + s2_C/(8 R r),   wall = R (a + 8 r b)")
    print("s2_R = the machine's pooled in-session restart component for GOINFER cells (section 4b), at its point estimate,")
    print("at the 95th percentile of its group bootstrap, and at zero (the best case for fewer restarts).")
    print("s2_B and s2_C = that stratum's within-server components")
    print(f"from every used cell (section 3d); W = s2_B + s2_C/8. D0 = today's two-pass gate: R=2 (pass 1 + reversed")
    print(f"pass 2) x r={DESIGN_RUNS}. BREAK-EVEN: one restart with any r can match D0 only if s2_R < W/3, i.e. only if the")
    print("restart share of a run mean is < 25%. That bar is tighter than the task doc's 50% kill line. 'best' = the")
    print("cheapest (R, r), R 1-4 and r 1-60, with Var <= Var(D0), at the dear-restart end (a_hi, b_lo) and the")
    print("cheap-restart end (a_lo, b_hi) of the cost bracket. 'trend' = section 5b flags this stratum: a quarter or")
    print("more of its restarts move > 2% between first and last run, so more runs per restart shift the mean and the")
    print("row does not apply. Session drift s2_S is not in Var: nothing inside one session reduces it.\n")
    m4ec = {name: solve(pool(gl)) for name, gl in strata(M4, lambda k: f"{k[0]} {engclass(k[1])}")}
    rup = {}
    for name, gl in strata(M4, lambda k: f"{k[0]} {engclass(k[1])}"):
        if args.boot > 0 and len(gl) >= 3:
            rnd = random.Random(2)
            vals = sorted(pos(solve(pool(rnd.choice(gl) for _ in range(len(gl))))[1]) for _ in range(args.boot))
            rup[name] = vals[int(0.95 * len(vals)) - 1]
    print(f"  {'stratum':<22} {'sdW%':>5} {'trend':>5}  {'s2_R':<5} {'sdR%':>5} {'R%run':>5} {'R=1 ok':>6}  "
          f"{'sd(D0)%':>7} {'sd(1x6)%':>8}   {'best, dear restart':>26}   {'best, cheap restart':>26}")
    for (hc, eng, model, be), (a_lo, a_hi, b_lo, b_hi) in sorted(costs.items()):
        if eng != "goinfer" or (hc, eng, model, be) not in wcomp or f"{hc} goinfer" not in m4ec:
            continue
        B2, C2 = (pos(x) for x in wcomp[(hc, eng, model, be)])
        W = B2 + C2 / 8
        tf = "YES" if trend_flag.get((hc, be, "goinfer", model)) else "-"
        for tag, R2 in (("point", pos(m4ec[f"{hc} goinfer"][1])), ("p95", rup.get(f"{hc} goinfer", float("nan"))),
                        ("zero", 0.0)):
            if R2 != R2:
                continue
            var = lambda R, r: R2 / R + B2 / (R * r) + C2 / (8 * R * r)
            v0 = var(2, DESIGN_RUNS)
            out = []
            for a, b in ((a_hi, b_lo), (a_lo, b_hi)):
                wall = lambda R, r: R * (a + 8 * r * b)
                w0 = wall(2, DESIGN_RUNS)
                best = min((wall(R, r), R, r) for R in range(1, 5) for r in range(1, 61)
                           if var(R, r) <= v0 * (1 + 1e-12))
                out.append(f"{best[1]} x {best[2]:<2} {best[0]:5.0f}s/{w0:4.0f}s {100 * (1 - best[0] / w0):4.0f}%")
            share = 100 * R2 / (R2 + W) if R2 + W > 0 else float("nan")
            first = tag == "point"
            print(f"  {(hc + ' ' + model + ' ' + be) if first else '':<22} "
                  f"{(('%5.2f' % (100 * math.sqrt(W))) if first else ''):>5} {(tf if first else ''):>5}  {tag:<5} "
                  f"{100 * math.sqrt(R2):5.2f} {share:5.0f} {('yes' if R2 < W / 3 else 'no'):>6}  "
                  f"{100 * math.sqrt(v0):7.2f} {100 * math.sqrt(var(1, 2 * DESIGN_RUNS)):8.2f}   "
                  f"{out[0]:>26}   {out[1]:>26}")
    print("\n  Read with section 6's O share: where a restart is cheap next to its completions (a_hi small against 48 b),")
    print("  dropping one saves little wall whatever the variance says.")

    # ---------------- 7. sensitivity ----------------
    print("\n## 7. sensitivity: the per-machine headline with the curation and the prompt key removed\n")
    for name, kw in (("as analysed (curated, prompt in key)", {}),
                     ("no curation (void + arm files pooled)", {"curate": False}),
                     ("no prompt key (essay-v2 pooled with older prompts)", {"prompt_key": False}),
                     ("neither", {"curate": False, "prompt_key": False})):
        cu, _ = extract(files, **kw)
        Gu = groups_of(cu)
        s3 = collections.defaultdict(list)
        s4 = collections.defaultdict(list)
        for k, v in Gu.items():
            s3[k[0]].append(nested(obs3(v), 2))
            s4[k[0]].append(nested(obs4(v), 3))
        for h in sorted(s3):
            p3, p4 = pool(s3[h]), pool(s4[h])
            a3 = shares3(solve(p3), p3["N"] / p3["units"][1])
            a4 = shares4(solve(p4), p4["N"] / p4["units"][2])
            print(f"  {name:<52} {h:<7} groups {len(s3[h]):>3}   3-level R % run mean {a3['run']:4.0f}   "
                  f"4-level: R in-session {a4['Rw']:4.0f}%  (S+R) {a4['SR']:4.0f}%")


if __name__ == "__main__":
    main()
