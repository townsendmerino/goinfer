#!/usr/bin/env python3
"""D13: goinfer's Clef pipeline against the reference, graded by the pre-registered rule
(docs/measurements/decisions-d13-clef-fidelity-2026-10-03.md, "Pre-registration", committed before any graded run).

    python3 grade.py --ref REF.jsonl --arms f32=PATH[:EVERY] int8int8=PATH[:EVERY] int4=PATH[:EVERY] [--bf16 PATH] [--exploratory]

REF is scripts/pin_clef_d10.py's probs_f32.jsonl: Cloudflare's own code at f32 on the 150 D10 records. Each arm file is internal/clef's TestFidelityArm_run output. EVERY is the
registered stride for an arm that runs a subset (every Nth record by position in records.jsonl); the arm must hold exactly that subset or it is INVALID.

Per arm:
  valid       exactly the registered records, no duplicates, every option list equal to the reference's, every probability finite and the row summing to 1,
              and the prompt token count equal to the encoder dump's (a tokenizer or template drift would otherwise grade a different prompt)
  KL          mean KL(reference || arm) over the arm's records
  top-1       argmax agreement with the reference
  ECE         top-label ECE against gold on the gold rows among the arm's records (15 equal-width bins), the reference's own ECE on the same rows beside it, and a
              paired bootstrap (2000 resamples, seed 0) of (arm ECE - reference ECE)

Rule (registered before any graded run):
  f32       PASS if KL <= 0.01 and top-1 >= 0.98
  int8int8  PASS if KL <= 0.03 and top-1 >= 0.98     } the band D6b graded JEV on
  int4      PASS if KL <= 0.03 and top-1 >= 0.98     }
  cuda-int4, cuda-int4-pin  (the second is the diagnostic arm with the embedding table at the int8 pin, amendment 5c)
            PASS if KL <= 0.03 and top-1 >= 0.98 (amendment 5b, 2026-10-03: the backbone resident on the CUDA device, int4); VOID unless the mean request time is under 20 ms per input token (the
            device measured ~3, the CPU 54-68), so a CPU fallback cannot pass as a GPU result. Its verdict is its own and does not change the CPU arms'.
  q4k       PASS if KL <= 0.03 and top-1 >= 0.98 (amendment 2026-10-03: a third-party Q4_K_M GGUF of the backbone, graded on int4's band, all 150 records)
  AMBIGUOUS (reported, never a pass, goes to the owner): KL within 2x the band, or top-1 in [0.95, 0.98), with the other criterion passing; anything past that FAILS.
  calibration: an arm FAILS CALIBRATION if the bootstrap 95% interval of (arm ECE - reference ECE) lies wholly above 0; an interval that reaches 0 is UNRESOLVED, not failed.
  If the f32 arm does not PASS, the port is suspect before any quantization finding is read.

Beside the arms: the reference at bf16 (--bf16, the release's own dtype, run with f32 GEMMs rounded to bf16) against the f32 reference on the same metrics, as the floor of numeric
noise a correct port is held against. And Clef against JEV-9B on gold: top-1 accuracy and ECE of the Clef reference (f32) against the JEV reference (f32) on the same gold rows.
"""
import argparse, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "..", "..", "testdata", "decisions")
BANDS = {"f32": 0.01, "int8int8": 0.03, "int4": 0.03, "q4k": 0.03, "cuda-int4": 0.03, "cuda-int4-pin": 0.03}   # q4k (added by the 2026-10-03 amendment): int4's band
TOP1, TOP1_AMBIG = 0.98, 0.95
GPU_MS_PER_TOKEN = 20.0   # cuda-int4 validity (D6b's amendment 2): the device measured ~3 ms per token, the CPU 54-68; above this it did not run resident


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kl(p, q):
    return sum(a * math.log(a / max(b, 1e-300)) for a, b in zip(p, q) if a > 0)


def am(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def ece(rows):
    """rows: (confidence, correct) pairs; top-label ECE over 15 equal-width bins (D6b's)."""
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    for c, ok in rows:
        b = bins[min(14, int(c * 15))]
        b[0] += 1; b[1] += ok; b[2] += c
    return sum(abs(b[1] - b[2]) for b in bins if b[0]) / len(rows)


def boot(diff_fn, ids, n=2000, seed=0):
    rng = random.Random(seed)
    d = sorted(diff_fn([rng.choice(ids) for _ in ids]) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--bf16")
    ap.add_argument("--exploratory", action="store_true", help="grade whatever rows exist; the output says it is not a verdict")
    a = ap.parse_args()
    recs = load(os.path.join(TD, "clef", "records.jsonl"))
    order = [r["id"] for r in recs]
    R = {r["id"]: r for r in recs}
    enc = {r["id"]: r for r in load(os.path.join(TD, "clef", "encoder.jsonl"))}
    ref = {r["id"]: r for r in load(a.ref)}
    if len(ref) != 150:
        print(f"REFERENCE INCOMPLETE: {len(ref)} rows, want 150"); sys.exit(1)
    # Everything is compared in the ITEM's label order (the order of target and labels), mapping Clef's option ids (its alphabetical order for choice) by label.
    def dist(row, rec):
        by = dict(zip(row["option_ids"], row["probs"]))
        return [by[l] for l in rec["labels"]]
    gold_idx = {i: am(R[i]["target"]) for i in order if R[i]["gold"]}

    def metrics(rows, ids, tag):
        kls = {i: kl(dist(ref[i], R[i]), dist(rows[i], R[i])) for i in ids}
        agree = {i: am(dist(ref[i], R[i])) == am(dist(rows[i], R[i])) for i in ids}
        g = [i for i in ids if i in gold_idx]
        arm_g = {i: (max(rows[i]["probs"]), am(dist(rows[i], R[i])) == gold_idx[i]) for i in g}
        ref_g = {i: (max(ref[i]["probs"]), am(dist(ref[i], R[i])) == gold_idx[i]) for i in g}
        out = dict(kl=sum(kls.values()) / len(ids), top1=sum(agree.values()) / len(ids), worst=max(kls.values()), n=len(ids), gold=len(g))
        if g:
            out["ece"], out["ece_ref"] = ece(list(arm_g.values())), ece(list(ref_g.values()))
            out["lo"], out["hi"] = boot(lambda s: ece([arm_g[i] for i in s]) - ece([ref_g[i] for i in s]), g)
            out["acc"], out["acc_ref"] = sum(v[1] for v in arm_g.values()) / len(g), sum(v[1] for v in ref_g.values()) / len(g)
        return out, kls, agree

    def valid(rows, ids):
        bad = []
        for i in ids:
            r = rows.get(i)
            if (r is None or r["option_ids"] != ref[i]["option_ids"] or len(r["probs"]) != len(ref[i]["probs"])
                    or not all(math.isfinite(x) for x in r["probs"]) or abs(sum(r["probs"]) - 1) > 1e-4 or r["n_tokens"] != enc[i]["n_tokens"]):
                bad.append(i)
        return bad

    print(f"reference: {len(ref)} rows; {len(gold_idx)} gold rows")
    if a.exploratory:
        print("EXPLORATORY: whatever rows exist are graded; this is not a verdict")
    verdicts = {}
    for spec in a.arms:
        name, _, rest = spec.partition("=")
        path, _, every = rest.partition(":")
        every = int(every) if every else 1
        want = [i for k, i in enumerate(order) if k % every == 0]
        if not os.path.exists(path):
            print(f"\n## {name}: MISSING ({path})"); verdicts[name] = "MISSING"; continue
        raw = load(path)
        rows = {r["id"]: r for r in raw}
        ids = [i for i in want if i in rows] if a.exploratory else want
        print(f"\n## {name}: {len(raw)} rows, registered set {len(want)} (every {every})")
        dups = len(raw) - len(rows)
        extra = set(rows) - set(want)
        if dups or (extra and not a.exploratory) or (not ids):
            print(f"  INVALID: {dups} duplicate rows, {len(extra)} rows outside the registered set"); verdicts[name] = "INVALID"; continue
        bad = valid(rows, ids)
        if bad and not a.exploratory:
            print(f"  INVALID: {len(bad)} rows missing, malformed, or with a different option list or prompt length, e.g. {bad[:3]}"); verdicts[name] = "INVALID"; continue
        ids = [i for i in ids if i not in bad]
        if name.startswith("cuda"):
            ms = 1000 * sum(rows[i]["seconds"] for i in ids) / sum(rows[i]["n_tokens"] for i in ids)
            print(f"  {ms:.1f} ms per input token (validity: under {GPU_MS_PER_TOKEN:.0f}, i.e. it ran on the device)")
            if ms > GPU_MS_PER_TOKEN:
                print("  INVALID: this arm did not run resident; it measured a CPU fallback")
                verdicts[name] = "INVALID (not resident)"
                continue
        m, kls, agree = metrics(rows, ids, name)
        per = {}
        for i in ids:
            per.setdefault(R[i]["kind"], []).append(i)
        print(f"  n={m['n']}  mean KL(ref || goinfer) {m['kl']:.5f}   top-1 agreement {m['top1']:.4f}   worst item KL {m['worst']:.4f}")
        for k, kid in sorted(per.items()):
            print(f"    {k:6s} n={len(kid):3d}  KL {sum(kls[i] for i in kid)/len(kid):.5f}  top-1 {sum(agree[i] for i in kid)/len(kid):.4f}")
        calib = "no gold rows in the set"
        if m["gold"]:
            calib = "FAILS" if m["lo"] > 0 else "unresolved (interval reaches 0)" if m["hi"] > 0 else "not worse"
            print(f"  gold rows {m['gold']}: ECE {m['ece']:.4f} (reference {m['ece_ref']:.4f}); arm - reference {m['ece']-m['ece_ref']:+.4f}, 95% paired bootstrap [{m['lo']:+.4f}, {m['hi']:+.4f}] -> calibration {calib}")
            print(f"            top-1 vs gold {m['acc']:.3f} (reference {m['acc_ref']:.3f})")
        band = BANDS.get(name)
        if band is None:
            verdicts[name] = "no band registered"; continue
        kl_ok, kl_amb = m["kl"] <= band, m["kl"] <= 2 * band
        t_ok, t_amb = m["top1"] >= TOP1, m["top1"] >= TOP1_AMBIG
        v = "PASS" if kl_ok and t_ok else "AMBIGUOUS" if kl_amb and t_amb else "FAIL"
        verdicts[name] = f"{v} (KL {m['kl']:.5f} vs {band}, top-1 {m['top1']:.4f} vs {TOP1}; ambiguous band KL <= {2*band}, top-1 >= {TOP1_AMBIG}); calibration {calib}" + (" [EXPLORATORY]" if a.exploratory else "")

    if a.bf16 and os.path.exists(a.bf16):
        b = {r["id"]: r for r in load(a.bf16)}
        ids = [i for i in order if i in b]
        if ids:
            m, _, _ = metrics(b, ids, "bf16")
            print(f"\n## reference at bf16 (the release's dtype) against the f32 reference: n={m['n']}  mean KL {m['kl']:.5f}  top-1 agreement {m['top1']:.4f}  (the noise floor of a correct port)")

    jp = os.path.join(TD, "jev9b_ref_f32.jsonl")
    if os.path.exists(jp) and gold_idx:
        jev = {r["id"]: r for r in load(jp)}
        g = [i for i in gold_idx if i in jev]
        cl = {i: (max(ref[i]["probs"]), am(dist(ref[i], R[i])) == gold_idx[i]) for i in g}
        jv = {i: (max(jev[i]["p"]), am(jev[i]["p"]) == gold_idx[i]) for i in g}
        acc = lambda d, s: sum(d[i][1] for i in s) / len(s)
        da = acc(cl, g) - acc(jv, g)
        dlo, dhi = boot(lambda s: acc(cl, s) - acc(jv, s), g)
        de = ece(list(cl.values())) - ece(list(jv.values()))
        elo, ehi = boot(lambda s: ece([cl[i] for i in s]) - ece([jv[i] for i in s]), g)
        print(f"\n## Clef (reference, f32) against JEV-9B (reference, f32) on the same {len(g)} gold rows")
        print(f"  top-1 vs gold: Clef {acc(cl, g):.3f}, JEV {acc(jv, g):.3f}; Clef - JEV {da:+.3f}, 95% paired bootstrap [{dlo:+.3f}, {dhi:+.3f}]")
        print(f"  ECE: Clef {ece(list(cl.values())):.4f}, JEV {ece(list(jv.values())):.4f}; Clef - JEV {de:+.4f}, 95% paired bootstrap [{elo:+.4f}, {ehi:+.4f}]")
        level = da >= 0 and de <= 0
        print(f"  INFORMATIONAL (the owner keeps and extends both routes equally, 2026-10-03; no rule hangs on it). Point estimates, accuracy Clef >= JEV and ECE Clef <= JEV: {'YES' if level else 'NO'}"
              f"{' -- both intervals reach 0, so this is a point-estimate reading and unresolved' if dlo <= 0 <= dhi and elo <= 0 <= ehi else ''}")

    print("\nVERDICTS")
    for n, v in verdicts.items():
        print(f"  {n:9s} {v}")


if __name__ == "__main__":
    main()
