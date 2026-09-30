#!/usr/bin/env python3
"""D6b (docs/tasks/task-constrained-confidence.md): goinfer's Route B on autotrust's JEV-9B against the D0 reference,
graded by the pre-registered rule and its 2026-09-30 amendment.

    python3 grade.py results/          (the dir holding f32.jsonl, int8int8.jsonl, int4.jsonl from run-d6b.sh)

Reference: testdata/decisions/jev9b_ref_f32.jsonl (transformers f32, the adapter unmerged, the head, the per-kind T),
on testdata/decisions/items.jsonl's 150 items.

Per arm:
  valid       150 rows, no errors, route "head", and every prompt's token count equal to the reference's (a template
              or tokenizer drift would otherwise grade a different prompt)
  KL          mean KL(reference || goinfer) over the 150 items
  top-1       argmax agreement with the reference
  ECE         top-label ECE against gold on the 84 gold rows (15 equal-width bins), with the reference's own ECE on the
              same rows beside it, and a paired bootstrap of (arm ECE - reference ECE) over those rows

Rule (registered before any run):
  f32       PASS if KL <= 0.01 and top-1 >= 0.98
  int4      PASS if KL <= 0.03 and top-1 >= 0.98
  int8int8  graded on int4's band (amendment: it is the fallback default)
  calibration (amendment): an arm FAILS CALIBRATION if the bootstrap 95% interval of (arm ECE - reference ECE) lies
            wholly above 0 (resolvably worse calibrated than the reference); an interval that reaches 0 is UNRESOLVED,
            not failed. If int4 fails calibration but passes top-1, the default for decision models is int8.
"""
import json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "..", "..", "testdata", "decisions")
BANDS = {"f32": 0.01, "int8int8": 0.03, "int4": 0.03}
TOP1 = 0.98


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kl(p, q):
    return sum(a * math.log(a / max(b, 1e-300)) for a, b in zip(p, q) if a > 0)


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def ece(rows):
    """rows: (confidence, correct) pairs; top-label ECE over 15 equal-width bins."""
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    for c, ok in rows:
        b = bins[min(14, int(c * 15))]
        b[0] += 1
        b[1] += ok
        b[2] += c
    return sum(abs(b[1] - b[2]) for b in bins if b[0]) / len(rows)


def main(resdir):
    items = {x["id"]: x for x in load(os.path.join(TD, "items.jsonl"))}
    ref = {x["id"]: x for x in load(os.path.join(TD, "jev9b_ref_f32.jsonl"))}
    gold = [i for i, x in items.items() if x["gold"]]
    gold_idx = {i: argmax(items[i]["target"]) for i in gold}
    ref_rows = {i: (max(ref[i]["p"]), argmax(ref[i]["p"]) == gold_idx[i]) for i in gold}
    print(f"reference: {len(ref)} items, {len(gold)} gold rows; reference ECE on them {ece(list(ref_rows.values())):.4f}")
    verdicts, facts = {}, {}
    for arm in ("f32", "int8int8", "int4"):
        p = os.path.join(resdir, arm + ".jsonl")
        if not os.path.exists(p):
            print(f"\n## {arm}: MISSING ({p})")
            continue
        out = {x["id"]: x for x in load(p)}
        bad = [i for i in items if i not in out or out[i].get("error") or out[i].get("route") != "head"
               or out[i].get("prompt_tokens") != ref[i]["n_tokens"]]
        print(f"\n## {arm}: {len(out)} rows")
        if bad:
            print(f"  INVALID: {len(bad)} rows missing, failed, not route head, or with a different prompt length, e.g. {bad[:3]}")
            verdicts[arm] = "INVALID"
            continue
        kls = {i: kl(ref[i]["p"], out[i]["distribution"]) for i in items}
        agree = {i: argmax(ref[i]["p"]) == argmax(out[i]["distribution"]) for i in items}
        mkl, t1 = sum(kls.values()) / len(kls), sum(agree.values()) / len(agree)
        per = {}
        for i, x in items.items():
            per.setdefault(x["kind"], []).append(i)
        print(f"  mean KL(ref || goinfer) {mkl:.5f}   top-1 agreement {t1:.4f}   worst item KL {max(kls.values()):.4f}")
        for k, ids in sorted(per.items()):
            print(f"    {k:6s} n={len(ids):3d}  KL {sum(kls[i] for i in ids)/len(ids):.5f}  top-1 {sum(agree[i] for i in ids)/len(ids):.4f}")
        arm_rows = {i: (max(out[i]["distribution"]), argmax(out[i]["distribution"]) == gold_idx[i]) for i in gold}
        e_arm, e_ref = ece(list(arm_rows.values())), ece(list(ref_rows.values()))
        rng = random.Random(0)
        diffs = []
        for _ in range(2000):
            s = [rng.choice(gold) for _ in gold]
            diffs.append(ece([arm_rows[i] for i in s]) - ece([ref_rows[i] for i in s]))
        diffs.sort()
        lo, hi = diffs[50], diffs[1949]
        calib = "FAILS" if lo > 0 else "unresolved (interval reaches 0)" if hi > 0 else "not worse"
        print(f"  ECE on gold rows {e_arm:.4f} (reference {e_ref:.4f}); arm - reference {e_arm-e_ref:+.4f}, "
              f"95% paired bootstrap [{lo:+.4f}, {hi:+.4f}] -> calibration {calib}")
        passed = mkl <= BANDS[arm] and t1 >= TOP1
        facts[arm] = {"top1": t1, "calib_fails": lo > 0}
        verdicts[arm] = ("PASS" if passed else "FAIL") + f" (KL {mkl:.5f} vs {BANDS[arm]}, top-1 {t1:.4f} vs {TOP1}); calibration {calib}"
    print("\nVERDICTS")
    for arm, v in verdicts.items():
        print(f"  {arm:9s} {v}")
    f = facts.get("int4")
    if f and f["top1"] >= TOP1 and f["calib_fails"]:
        print("  -> int4 passes top-1 but fails calibration: the default for decision models is int8 (record it in the capability matrix)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results"))
