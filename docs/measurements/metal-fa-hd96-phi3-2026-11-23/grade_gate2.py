#!/usr/bin/env python3
"""Grade gate 2 (README beside this file) from TestR2_decodeFidelityGate's output by CUDA's criteria
(attn-decode-fa-fidelity-PREREGISTERED.md): (a) hard flips: lane <= exact + 2*sqrt(exact); (b) mean agreement: lane >= exact - 1.0 pt
and lane >= exact on at least half the prompts; (c) mean KL: lane <= 1.10 x exact. PARKED: (b) within its last 0.2 pt (lane - exact
in [-1.0, -0.8)), or (c) in 1.05-1.10x. VOID: a prompt failed the reference-identity check.

  grade_gate2.py <gate log>
"""
import math, re, sys

txt = open(sys.argv[1]).read()
per = re.findall(r"prompt\s+(\d+)/\s*\d+ shipped\(agree=([\d.]+)% HF=(\d+)/(\d+) KL=([\d.]+)\) \S+\(agree=([\d.]+)% HF=(\d+)/\d+ KL=([\d.]+)\)", txt)
pooled = re.search(r"shippedAgree=([\d.]+)% faAgree=([\d.]+)%", txt)
kl = re.search(r"shippedMeanKL=([\d.]+) faMeanKL=([\d.]+)", txt)
hf = re.search(r"\(shipped=(\d+) fa=(\d+)\)", txt)
if not (per and pooled and kl and hf):
    sys.exit("VERDICT VOID: the log has no complete gate output")
void = "REFERENCE/PROMPT MISMATCH SUSPECTED" in txt
ea, la = float(pooled.group(1)), float(pooled.group(2))
ek, lk = float(kl.group(1)), float(kl.group(2))
eh, lh = int(hf.group(1)), int(hf.group(2))
wins = sum(float(p[5]) >= float(p[1]) for p in per)
a = lh <= eh + 2 * math.sqrt(eh)
d = la - ea
b = d >= -1.0 and wins * 2 >= len(per)
ratio = lk / ek
c = ratio <= 1.10
print(f"prompts {len(per)}; exact agree {ea:.2f}% lane {la:.2f}% (diff {d:+.2f} pt), lane >= exact on {wins}/{len(per)}")
print(f"hard flips exact {eh} lane {lh}; mean KL exact {ek:.6f} lane {lk:.6f} (ratio {ratio:.4f})")
print(f"(a) {'PASS' if a else 'FAIL'}  (b) {'PASS' if b else 'FAIL'}  (c) {'PASS' if c else 'FAIL'}")
if void:
    print("VERDICT VOID (reference identity check failed)"); sys.exit(2)
if not (a and b and c):
    print("VERDICT FAIL"); sys.exit(1)
if d < -0.8 or ratio > 1.05:
    print("VERDICT PARKED (inside the ambiguous band)"); sys.exit(1)
print("VERDICT PASS")
