#!/usr/bin/env python3
"""B9's split (docs/QUEUE.md B9): how much of D6a arm B's divergence from the f32 reference is the CUDA path, and how
much is int4. Registered 2026-10-01, before either arm ran.

Arms: goinfer Route A (bare-v1, raw, --ctx 4096) on Qwen3.5-9B-Q4_K_M.gguf, on the CPU at int4 and at int8int8, on the
100 rows of noul-100-ids.txt (select100.py). Comparators on the same rows: the transformers f32 reference
(../decisions-d6a-2026-09-28/results/b0-ood-noul-f32.jsonl) and D6a's arm B, the same model and route CUDA-resident at
int4 (../decisions-d6a-2026-09-28/results/eval-B.jsonl).

Measured before either arm ran, on these 100 rows: reference top-1 against gold 0.540, arm B 0.380 (gap 0.160, against
0.240 on all 400), arm B's agreement with the reference 0.660.

Validity: an arm answers all 100 rows by route "label", each with the reference's prompt token count, or it is void.
Primary reading, per arm: the share of arm B's disagreement with the reference that the arm removes,
  s = (agreement_arm - 0.660) / (1 - 0.660).
For the CPU int4 arm:
  s >= 0.50       the CUDA path is the main part of arm B's divergence: B9's per-layer CUDA-against-CPU-int4 work is next
  s <= 0.25       the CUDA path is a minor part: the divergence is the Q4_K file and goinfer's int4, which the int8int8
                  arm's s then splits
  between         both contribute, reported as such
Also reported: top-1 against gold, mean P(true) shift against the reference, and the paired top-1 disagreements with
arm B.

    python3 grade.py RESULTS_DIR      (holding cpu-int4.jsonl and cpu-int8int8.jsonl)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D6A = os.path.join(HERE, "..", "decisions-d6a-2026-09-28", "results")
AGREE_B, ACC_REF, ACC_B = 0.660, 0.540, 0.380  # measured on the 100 rows before either arm ran


def am(p):
    return max(range(len(p)), key=lambda k: p[k])


def main(rdir):
    ids = [l.strip() for l in open(os.path.join(HERE, "noul-100-ids.txt")) if l.strip()]
    ref = {r["id"]: r for r in map(json.loads, open(os.path.join(D6A, "b0-ood-noul-f32.jsonl")))}
    armb = {r["id"]: r for r in map(json.loads, open(os.path.join(D6A, "eval-B.jsonl"))) if r.get("kind") == "noul"}
    gold = {i: am(ref[i]["target"]) for i in ids}
    agree_b = sum(am(ref[i]["p"]) == am(armb[i]["distribution"]) for i in ids) / len(ids)
    assert abs(agree_b - AGREE_B) < 1e-9, agree_b
    print(f"reference top-1 {ACC_REF:.3f}, arm B (CUDA int4) top-1 {ACC_B:.3f}, arm B agreement with the reference {AGREE_B:.3f}")
    for arm in ("cpu-int4", "cpu-int8int8"):
        path = os.path.join(rdir, arm + ".jsonl")
        if not os.path.exists(path):
            print(f"\n## {arm}: MISSING")
            continue
        rows = {r["id"]: r for r in map(json.loads, open(path)) if r}
        bad = [i for i in ids if i not in rows or rows[i].get("route") != "label"
               or rows[i].get("prompt_tokens") != ref[i]["n_tokens"]]
        print(f"\n## {arm}: {len(rows)} rows")
        if len(rows) != len(ids) or bad:
            print(f"  VOID: {len(bad)} row(s) missing, off route, or with a different prompt token count")
            continue
        agree = sum(am(rows[i]["distribution"]) == am(ref[i]["p"]) for i in ids) / len(ids)
        acc = sum(am(rows[i]["distribution"]) == gold[i] for i in ids) / len(ids)
        shift = sum(rows[i]["distribution"][1] - ref[i]["p"][1] for i in ids) / len(ids)
        vs_b = sum(am(rows[i]["distribution"]) != am(armb[i]["distribution"]) for i in ids)
        s = (agree - AGREE_B) / (1 - AGREE_B)
        print(f"  agreement with the reference {agree:.3f}   s = {s:+.2f}   top-1 against gold {acc:.3f}   "
              f"mean P(true) shift {shift:+.3f}   top-1 differs from arm B on {vs_b} of {len(ids)}")
        if arm == "cpu-int4":
            band = ("the CUDA path is the main part" if s >= 0.50 else
                    "the CUDA path is a minor part" if s <= 0.25 else "both contribute")
            print(f"  READING (pre-registered): {band}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results"))
