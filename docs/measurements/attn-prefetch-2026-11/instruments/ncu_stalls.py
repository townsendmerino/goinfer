#!/usr/bin/env python3
"""Per-launch stall shares from an attention-prefetch ncu report (docs/measurements/attn-prefetch-2026-11/README.md).

Usage: ncu_stalls.py <report.ncu-rep>. A share is a stall reason's warps-stalled-per-issue ratio over the sum of every
reason's, `selected` included, as the kernel survey's KS-B2 read computed it."""
import csv, io, subprocess, sys

NCU = "/opt/nvidia/nsight-compute/2026.1.1/ncu"
out = subprocess.run([NCU, "-i", sys.argv[1], "--page", "raw", "--csv"], capture_output=True, text=True, check=True).stdout
rows = list(csv.reader(io.StringIO(out)))
h, units, data = rows[0], rows[1], rows[2:]
pre, suf = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
col = {n: i for i, n in enumerate(h)}
reasons = [n[len(pre):-len(suf)] for n in h if n.startswith(pre) and n.endswith(suf)]

SCALE = {"": 1, "%": 1, "inst": 1, "ratio": 1, "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
         "ns": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1, "msecond": 1, "nsecond": 1e-6, "register/thread": 1}


def val(r, name):
    """The metric in base units: bytes, and milliseconds for a duration."""
    try:
        i = col[name]
        return float(r[i].replace(",", "")) * SCALE.get(units[i], float("nan"))
    except (KeyError, ValueError):
        return float("nan")

print("launch grid  dur_ms  ld_stall%  long_sb%  lg_thr%  barrier%  tensor%  B/ld  ld_inst   warps_active%  regs  top other")
for r in data:
    st = {k: val(r, pre + k + suf) for k in reasons}
    tot = sum(v for v in st.values() if v == v)
    sh = {k: 100 * v / tot for k, v in st.items()}
    ld = val(r, "smsp__sass_inst_executed_op_global_ld.sum")
    by = val(r, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum")
    top = sorted((v, k) for k, v in sh.items() if k not in ("long_scoreboard", "lg_throttle", "barrier"))[::-1][:5]
    print(f'{r[col["ID"]]:>6} {r[col["Grid Size"]]:>9} {val(r,"gpu__time_duration.sum"):7.3f} '
          f'{sh["long_scoreboard"]+sh["lg_throttle"]:9.1f} {sh["long_scoreboard"]:8.1f} {sh["lg_throttle"]:8.1f} {sh["barrier"]:9.1f} '
          f'{val(r,"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"):8.2f} {by/(ld*32):5.2f} {ld:9.0f} '
          f'{val(r,"sm__warps_active.avg.pct_of_peak_sustained_active"):13.1f} {val(r,"launch__registers_per_thread"):5.0f}  '
          + " ".join(f"{k} {v:.1f}" for v, k in top))
