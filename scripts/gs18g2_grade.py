#!/usr/bin/env python3
"""Grade G-S18g2 (docs/tasks/task-multimodal-support-2026-10.md, "G-S18g2"): Gemma 3's int8 Metal tower against the f16
Metal tower, each measured against the exact CPU tower by decoder/gemma3_tower_sensitivity_real_test.go.

Input: the test's GOINFER_G3_OUT file, one JSON line per unit (an image and a prompt). Per unit the test teacher-forces
the int4 decoder along the CPU tower's own greedy reply and reports, per arm, the mean KL(CPU tower || arm) over the
graded positions and how many positions keep the CPU tower's argmax.

    D = mean over units of (KL mean, int8 tower) - (KL mean, f16 tower)          nats per position
    A = (agreement, f16 tower) - (agreement, int8 tower), positions pooled      percentage points

Both with a 95% cluster bootstrap over images (10,000 resamples, seed 20261010). The verdict reads each interval's
UPPER bound against the registered bands and takes the worse of the two.

Controls, any failure is VOID: every unit's "CPU tower again" arm reads exactly 0 with full agreement, and every unit's
"features negated" arm reads at least 10 times that unit's f16 arm (and at least 0.5 nats).

    python3 scripts/gs18g2_grade.py <units.jsonl> [--expect-units N]
"""
import json
import random
import sys

PASS_D, PARK_D = 0.05, 0.20
PASS_A, PARK_A = 5.0, 15.0
F16, I8 = "Metal f16 tower", "Metal int8 tower"
ZERO, REV = "control: CPU tower again", "control: features negated"


def band(x, lo, hi):
    return "PASS" if x <= lo else ("PARKED" if x <= hi else "FAIL")


def stats(units):
    d = sum(u[I8]["kl_mean"] - u[F16]["kl_mean"] for u in units) / len(units)
    steps = sum(u["steps"] for u in units)
    a = 100.0 * (sum(u[F16]["agree"] for u in units) - sum(u[I8]["agree"] for u in units)) / steps
    return d, a


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    expect = None
    if "--expect-units" in sys.argv:
        expect = int(sys.argv[sys.argv.index("--expect-units") + 1])
        args = [a for a in args if a != str(expect)]
    units = []
    for line in open(args[0]):
        if not line.strip():
            continue
        r = json.loads(line)
        u = {"image": r["image"], "prompt": r["prompt"], "steps": r["steps"]}
        for a in r["arms"]:
            u[a["name"]] = a
        units.append(u)
    if not units:
        sys.exit("G-S18g2: VOID (no units)")
    void = []
    if expect is not None and len(units) != expect:
        void.append(f"{len(units)} units, {expect} registered")
    seen = set()
    for u in units:
        key = (u["image"], u["prompt"])
        tag = f"{u['image']} / {u['prompt'][:28]!r}"
        if key in seen:
            void.append(f"{tag}: a duplicate unit")
        seen.add(key)
        missing = [k for k in (F16, I8, ZERO, REV) if k not in u]
        if missing:
            void.append(f"{tag}: no arm {missing}")
            continue
        if u[ZERO]["kl_mean"] != 0 or u[ZERO]["agree"] != u["steps"]:
            void.append(f"{tag}: the CPU tower again reads KL {u[ZERO]['kl_mean']:.3g}, agreement {u[ZERO]['agree']}/{u['steps']}")
        if u[REV]["kl_mean"] < max(0.5, 10 * u[F16]["kl_mean"]):
            void.append(f"{tag}: features negated reads {u[REV]['kl_mean']:.3f}, the f16 tower {u[F16]['kl_mean']:.3f}")

    print(f"{len(units)} units over {len({u['image'] for u in units})} images, {sum(u['steps'] for u in units)} graded positions")
    print(f"{'image':34s} {'prompt':22s} steps   f16 KL  int8 KL   excess   agree f16/int8   noise at int8's size")
    for u in units:
        if F16 not in u or I8 not in u:
            continue
        noise = sorted(v["kl_mean"] for k, v in u.items() if isinstance(v, dict) and k.startswith(f"noise at {I8}"))
        nz = f"{noise[0]:.3f}-{noise[-1]:.3f}" if noise else "-"
        print(f"{u['image']:34s} {u['prompt'][:20]!r:22s} {u['steps']:5d}  {u[F16]['kl_mean']:7.4f}  {u[I8]['kl_mean']:7.4f}  "
              f"{u[I8]['kl_mean'] - u[F16]['kl_mean']:+7.4f}   {u[F16]['agree']:3d}/{u[I8]['agree']:3d}          {nz}")
    if void:
        print("\nG-S18g2: VOID")
        for v in void:
            print("  " + v)
        sys.exit(1)

    by_image = {}
    for u in units:
        by_image.setdefault(u["image"], []).append(u)
    images = sorted(by_image)
    rng = random.Random(20261010)
    ds, as_ = [], []
    for _ in range(10000):
        sample = [u for im in (rng.choice(images) for _ in images) for u in by_image[im]]
        d, a = stats(sample)
        ds.append(d)
        as_.append(a)
    ds.sort()
    as_.sort()
    d, a = stats(units)
    dlo, dhi = ds[249], ds[9749]
    alo, ahi = as_[249], as_[9749]
    f16 = sum(u[F16]["kl_mean"] for u in units) / len(units)
    i8 = sum(u[I8]["kl_mean"] for u in units) / len(units)
    print(f"\nmean KL to the CPU tower: f16 {f16:.4f}, int8 {i8:.4f} nats per position")
    print(f"D (int8 minus f16)   {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}]   bands on the upper bound: PASS <= {PASS_D}, PARKED <= {PARK_D} -> {band(dhi, PASS_D, PARK_D)}")
    print(f"A (f16 minus int8)   {a:+.2f} points [{alo:+.2f}, {ahi:+.2f}]   bands on the upper bound: PASS <= {PASS_A}, PARKED <= {PARK_A} -> {band(ahi, PASS_A, PARK_A)}")
    order = ["PASS", "PARKED", "FAIL"]
    verdict = max(band(dhi, PASS_D, PARK_D), band(ahi, PASS_A, PARK_A), key=order.index)
    print(f"\nG-S18g2: {verdict}")
    sys.exit(0 if verdict == "PASS" else 3 if verdict == "PARKED" else 1)


if __name__ == "__main__":
    main()
