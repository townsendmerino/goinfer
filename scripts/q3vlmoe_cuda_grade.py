#!/usr/bin/env python3
"""G-S10q-e graded as registered (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE"): is the CUDA path
(paged experts, CPU prefill + UploadKV + resident decode) as close to Hugging Face float32 as the CPU int4 path is, on the
same teacher-forced positions G-S10q-c graded?
For every position: agreement = argmax equals HF's, and KL(HF || arm). The statistic is the PAIRED difference per position,
CUDA minus CPU int4, with a cluster bootstrap over the prompts (10,000 resamples, seed 20261012, 95%).
PASS: agreement difference >= -5.0 points (G-S10q-c's margin) and every logit finite. PARKED: -5.0 down to -15.0. FAIL: worse,
or any non-finite logit. The CPU arm is the baseline G-S10q-c already measured (92.5% on the 30B); it is recomputed here from
the same files so the two arms are compared on exactly the same positions.
Usage: q3vlmoe_cuda_grade.py <prefix>  where <prefix>.seqs.json, <prefix>.int4.f32 (the CPU arm), <prefix>.cuda.f32 and
<hfprefix>.f32/.json exist (the HF prefix defaults to <prefix>.hf; pass it as argv[2] to override)."""
import json, sys
import numpy as np

prefix = sys.argv[1]
hfp = sys.argv[2] if len(sys.argv) > 2 else prefix + ".hf"
rng = np.random.default_rng(20261012)
seqs = json.load(open(prefix + ".seqs.json"))
lens = [len(s["path"]) for s in seqs]
meta = json.load(open(hfp + ".json"))
V, N = meta["vocab"], sum(lens)
assert meta["positions"] == lens, (meta["positions"], lens)
hf = np.fromfile(hfp + ".f32", dtype="<f4").reshape(N, V)
cpu = np.fromfile(prefix + ".int4.f32", dtype="<f4").reshape(N, V)
cuda = np.fromfile(prefix + ".cuda.f32", dtype="<f4").reshape(N, V)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def per_pos(arm):
    agree = (hf.argmax(-1) == arm.argmax(-1)).astype(float)
    lh, la = logsm(hf), logsm(arm)
    return agree, (np.exp(lh) * (lh - la)).sum(-1)


def clusters(vals):
    idx = np.cumsum([0] + lens)
    return [vals[idx[i]:idx[i + 1]] for i in range(len(lens))]


def boot(vals):
    cl = clusters(vals)
    pt = float(np.concatenate(cl).mean())
    bs = [float(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]).mean()) for _ in range(10000)]
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


fin = bool(np.isfinite(cuda).all())
ag_cpu, kl_cpu = per_pos(cpu)
ag_cuda, kl_cuda = per_pos(cuda)
print(f"{N} positions over {len(lens)} prompts; every CUDA logit finite: {fin}")
for name, ag, kl in (("CPU int4 ", ag_cpu, kl_cpu), ("CUDA int4", ag_cuda, kl_cuda)):
    a, k = boot(ag), boot(kl)
    print(f"  {name} agreement with HF {100 * a[0]:6.2f}% [{100 * a[1]:.1f}, {100 * a[2]:.1f}]   mean KL(HF||arm) {k[0]:.4f} [{k[1]:.4f}, {k[2]:.4f}]")
d_ag, d_kl = boot(ag_cuda - ag_cpu), boot(kl_cuda - kl_cpu)
print(f"  paired difference (CUDA - CPU): agreement {100 * d_ag[0]:+.2f} points [{100 * d_ag[1]:+.1f}, {100 * d_ag[2]:+.1f}]; KL {d_kl[0]:+.4f} [{d_kl[1]:+.4f}, {d_kl[2]:+.4f}]")
print(f"  CUDA argmax equals the CPU arm's at {100 * float((cuda.argmax(-1) == cpu.argmax(-1)).mean()):.2f}% of positions (for the record, not graded)")
gap = 100 * d_ag[0]
verdict = "FAIL (non-finite logits)" if not fin else ("PASS" if gap >= -5.0 else ("PARKED" if gap >= -15.0 else "FAIL"))
print(f"\nG-S10q-e: CUDA minus CPU = {gap:+.2f} points (bar: PASS at >= -5.0, PARKED to -15.0) -> {verdict}")
json.dump({"n": N, "finite": fin, "agree_cpu": boot(ag_cpu), "agree_cuda": boot(ag_cuda), "diff_agree": d_ag, "diff_kl": d_kl, "verdict": verdict},
          open(prefix + ".cuda.grade.json", "w"), indent=1)
