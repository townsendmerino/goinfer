#!/usr/bin/env python3
"""The A/A of docs/tasks/task-cuda-windowed-kv-2026-10.md, section 10: does the windowed-KV branch change the DEFAULT path (the option off)?

    python3 aa_default_path.py <serve-main> <serve-branch> <model-dir> <out-dir> [--smoke]

Two arms at ctx 2048 on Mellum2.1 (the model whose decode the branch's per-layer view touches most), neither passing --windowed-kv:
  M  serve-main    built at the branch's merge base with main (the code before any windowed-KV change)
  B  serve-branch  built at the branch tip
Sessions M B B M (ABBA), one cold server each, one discarded warm request, then per session the workloads of windowed_kv_gate.py: 3 short prompts x 3 reps (128 tokens) and one 1500+ token prompt x 3 reps
(384 tokens). Greedy, streamed, rate = (n-1)/(t_last-t_first). The session driver is windowed_kv_gate.py's.
Pre-registered: texts of one prompt identical across both arms and all four sessions (the default path's output must not change: any difference FAILS); per class the ratio B/M of the medians in [0.98, 1.02]
PASS, below 0.97 FAIL, 0.97-0.98 AMBIGUOUS -> parked, above 1.02 reported as faster and passes; the verdict is the worse class. A session whose server is not cuda-resident, or one that engaged windowed KV,
VOIDS the job."""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import windowed_kv_gate as g  # noqa: E402

main_bin, branch_bin, model_dir, outdir = sys.argv[1:5]
smoke = "--smoke" in sys.argv
os.makedirs(outdir, exist_ok=True)
lp = g.long_prompt(27)
work = [("short", g.SHORT[:1], 2, 32), ("long", [lp], 1, 64)] if smoke else [("short", g.SHORT, 3, 128), ("long", [lp], 3, 384)]
order = [("M", main_bin, "s1M"), ("B", branch_bin, "s2B")] if smoke else [("M", main_bin, "s1M"), ("B", branch_bin, "s2B"), ("B", branch_bin, "s3B"), ("M", main_bin, "s4M")]
rows, meta = [], []
for arm, binary, tag in order:
    r, m = g.session(binary, model_dir, arm, outdir, tag, 2048, False, work)
    rows += r
    meta.append(m)
out, ratios, identical = [], {}, True
for cls in ("short", "long"):
    for pi in sorted({x["prompt"] for x in rows if x["cls"] == cls}):
        texts = {x["text"] for x in rows if x["cls"] == cls and x["prompt"] == pi}
        identical &= len(texts) == 1
        M = [x["rate"] for x in rows if x["arm"] == "M" and x["cls"] == cls and x["prompt"] == pi]
        B = [x["rate"] for x in rows if x["arm"] == "B" and x["cls"] == cls and x["prompt"] == pi]
        out.append(f"{cls} prompt {pi}: main median {statistics.median(M):.2f} tok/s (min {min(M):.2f} max {max(M):.2f}); branch median {statistics.median(B):.2f} (min {min(B):.2f} max {max(B):.2f}); texts identical across arms and sessions: {len(texts) == 1}")
    M = [x["rate"] for x in rows if x["arm"] == "M" and x["cls"] == cls]
    B = [x["rate"] for x in rows if x["arm"] == "B" and x["cls"] == cls]
    ratios[cls] = statistics.median(B) / statistics.median(M)
    out.append(f"{cls}: branch/main ratio of medians = {ratios[cls]:.3f}")
for arm in ("M", "B"):
    ss = sorted({x["tag"] for x in rows if x["arm"] == arm})
    if len(ss) == 2:
        a, b = (statistics.median([x["rate"] for x in rows if x["tag"] == t]) for t in ss)
        out.append(f"arm {arm} session medians {a:.2f} vs {b:.2f} tok/s (drift {100 * (b / a - 1):+.1f}%)")
worst = min(ratios.values())
verdict = "PASS" if worst >= 0.98 else ("FAIL" if worst < 0.97 else "AMBIGUOUS -> parked")
out.append(f"A/A texts identical across arms and sessions: {'PASS' if identical else 'FAIL'}")
out.append(f"A/A OVERALL worst class ratio {worst:.3f} -> {verdict}" + ("  [SMOKE, not a result]" if smoke else ""))
text = "\n".join(out)
print(text)
open(os.path.join(outdir, "result.txt"), "w").write(text + "\n")
