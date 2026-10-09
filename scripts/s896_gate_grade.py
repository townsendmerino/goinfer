#!/usr/bin/env python3
"""G-S10k's Hugging Face half and grader (docs/tasks/task-multimodal-support-2026-10.md, "G-S10k", registered before this code).
Run 1 dump (OFF = serve's CPU prefill + upload, ON = the resident DeepStack prefill, ONX = ON with the planted defect) and run 2 dump (the CPU arm at the exact attention kernel = OFFX; its ON must equal run 1's).
Per unit: HF float32 logits at the 4 positions (cached beside the dump as <unit>.hf.f32); c(arm) = mean over the 4 positions of cos(arm, HF); d = c(ON) - c(OFF), a = c(OFFX) - c(OFF), x = c(ONX) - c(OFF).
Margin M = max(0.005, -mean(a)). Stratified cluster bootstrap (units within image, images equally weighted), 10,000 resamples, seed 20261009, 95%.
PASS: LB(mean d) >= -M and no image's mean d < -2M. FAIL: UB(mean d) < -M or any image's mean d < -2M. PARKED otherwise. NO READING if SE(mean a) > M/2.
Usage: ~/g4venv/bin/python -I scripts/s896_gate_grade.py <model dir> <run1 dump> <run2 dump> [G-S10j round-3 dump dir for the reproduction control]"""
import hashlib, json, math, os, sys, time
import numpy as np
model_dir, A, B = sys.argv[1:4]
R3 = sys.argv[4] if len(sys.argv) > 4 else None
T0 = time.time()
def hb(m): print(f"[s896g grade {time.time() - T0:6.0f}s] {m}", flush=True)
IMAGES = ["gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "formula.png", "table.png"]
units = sorted({f[:-len(".meta.json")] for f in os.listdir(A) if f.endswith(".meta.json") and ".p" in f})
hb(f"{len(units)} units in {A}")
def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))
def logsm(x):
    x = x.astype(np.float64); m = x.max(); return x - m - math.log(np.exp(x - m).sum())
def kl(hf, arm): p = np.exp(logsm(hf)); return float((p * (logsm(hf) - logsm(arm))).sum())
# ---- the Hugging Face side
need = [u for u in units if not os.path.exists(f"{A}/{u}.hf.f32")]
tmpl_ok, top3_ok = {}, {}
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
tok = AutoTokenizer.from_pretrained(model_dir)
if need:
    hb(f"loading float32 model for {len(need)} units without a cached reference")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32); model.eval()
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    pix_cache = {}
    for i, u in enumerate(need):
        meta = json.load(open(f"{A}/{u}.meta.json")); img = u.rsplit(".p", 1)[0]; grid = meta["grid"]
        if img not in pix_cache:
            px = np.fromfile(f"{A}/{img}.px.f32", dtype="<f4"); pix_cache[img] = torch.from_numpy(px.reshape(-1, px.size // (grid[0] * grid[1] * grid[2])).copy())
        ids = torch.tensor([meta["ids"] + meta["teacher"]]); mm = (ids == pad).long()
        t = time.time()
        with torch.no_grad():
            lg = model(input_ids=ids, mm_token_type_ids=mm, pixel_values=pix_cache[img], image_grid_thw=torch.tensor([grid])).logits[0].float().numpy()
        n, V, steps = meta["n"], meta["vocab"], meta["steps"]
        lg[n - 1 : n - 1 + steps][:, :V].astype("<f4").tofile(f"{A}/{u}.hf.f32")
        hb(f"HF unit {i + 1}/{len(need)} {u}: {ids.shape[1]} tokens in {time.time() - t:.0f}s")
    del model
# ---- per unit statistics
rows = []
for u in units:
    meta = json.load(open(f"{A}/{u}.meta.json")); V, S = meta["vocab"], meta["steps"]; img = u.rsplit(".p", 1)[0]
    ld = lambda d, s: np.fromfile(f"{d}/{u}.{s}.f32", dtype="<f4").reshape(S, V)
    hf = ld(A, "hf"); off, on = ld(A, "off"), ld(A, "on")
    offx = ld(B, "off") if os.path.exists(f"{B}/{u}.off.f32") else None
    planted = {nm: ld(A, "onx_" + nm) for nm in ("late", "notadded", "textrows") if os.path.exists(f"{A}/{u}.onx_{nm}.f32")}
    c = lambda arm: float(np.mean([cos(arm[k], hf[k]) for k in range(S)]))
    rec = dict(unit=u, image=img, c_off=c(off), c_on=c(on), kl_off=float(np.mean([kl(hf[k], off[k]) for k in range(S)])), kl_on=float(np.mean([kl(hf[k], on[k]) for k in range(S)])),
               agree_off=float(np.mean([hf[k].argmax() == off[k].argmax() for k in range(S)])), agree_on=float(np.mean([hf[k].argmax() == on[k].argmax() for k in range(S)])))
    if offx is not None: rec["c_offx"] = c(offx)
    for nm, arr in planted.items(): rec["c_onx_" + nm] = c(arr)
    rec["on_equal_run2"] = (hashlib.sha256(np.ascontiguousarray(on).tobytes()).hexdigest() == hashlib.sha256(np.ascontiguousarray(ld(B, "on")).tobytes()).hexdigest()) if os.path.exists(f"{B}/{u}.on.f32") else None
    top3 = lambda x: set(np.argsort(-x[0])[:3].tolist()); rec["hf_argmax_in_top3"] = int(hf[0].argmax()) in (top3(off) | top3(on))
    txt = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": meta["prompt"]}]}], tokenize=False, add_generation_prompt=True).replace("<|image_pad|>", "<|image_pad|>" * meta["nImg"])
    rec["template_ids_equal"] = tok.encode(txt, add_special_tokens=False) == meta["ids"]
    rows.append(rec)
by = {i: [r for r in rows if r["image"] == i] for i in IMAGES if any(r["image"] == i for r in rows)}
rng = np.random.default_rng(20261009)
def stat(vals_by_img, idx_by_img=None):
    return float(np.mean([np.mean(v if idx_by_img is None else v[idx_by_img[i]]) for i, v in vals_by_img.items()]))
def boot(key):
    v = {i: np.array([r[key] for r in rs]) for i, rs in by.items()}
    pt = stat(v); bs = np.array([stat(v, {i: rng.integers(0, len(a), len(a)) for i, a in v.items()}) for _ in range(10000)])
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(bs.std(ddof=1)), {i: float(a.mean()) for i, a in v.items()}
for r in rows:
    r["d"] = r["c_on"] - r["c_off"]
    if "c_offx" in r: r["a"] = r["c_offx"] - r["c_off"]
    for k in [k for k in r if k.startswith("c_onx_")]: r["x_" + k[6:]] = r[k] - r["c_off"]
d_pt, d_lo, d_hi, d_se, d_img = boot("d")
out = {"units": len(rows), "d": dict(mean=d_pt, lo=d_lo, hi=d_hi, se=d_se, per_image=d_img)}
print(f"\n{len(rows)} units; per image {({i: len(rs) for i, rs in by.items()})}")
print(f"d = c(ON) - c(OFF): mean {d_pt:+.4f}  95% [{d_lo:+.4f}, {d_hi:+.4f}]  se {d_se:.4f}   per image { {i: round(v, 4) for i, v in d_img.items()} }")
M = None
if all("a" in r for r in rows):
    a_pt, a_lo, a_hi, a_se, a_img = boot("a"); M = max(0.005, -a_pt)
    out["a"] = dict(mean=a_pt, lo=a_lo, hi=a_hi, se=a_se, per_image=a_img); out["M"] = M
    print(f"a = c(OFFX) - c(OFF) (the A/A): mean {a_pt:+.4f}  95% [{a_lo:+.4f}, {a_hi:+.4f}]  se {a_se:.4f}   per image { {i: round(v, 4) for i, v in a_img.items()} }; unit-level sd {np.std([r['a'] for r in rows], ddof=1):.4f}")
    print(f"margin M = max(0.005, -mean a) = {M:.4f}; instrument sufficiency: se(mean a) {a_se:.4f} <= M/2 = {M / 2:.4f}: {'HELD' if a_se <= M / 2 else 'NOT HELD'}")
    def verdict(pt, lo, hi, per):
        if hi < -M or min(per.values()) < -2 * M: return "FAIL"
        if lo >= -M and min(per.values()) >= -2 * M: return "PASS"
        return "PARKED"
    suff = a_se <= M / 2
    v = verdict(d_pt, d_lo, d_hi, d_img) if suff else "NO READING (instrument cannot resolve M)"
    out["verdict"] = v; out["sufficient"] = suff
    print(f"\nVERDICT (ON against OFF at margin M): {v}")
    for nm in ("late", "notadded", "textrows"):
        if all(("x_" + nm) in r for r in rows):
            x_pt, x_lo, x_hi, x_se, x_img = boot("x_" + nm); xv = verdict(x_pt, x_lo, x_hi, x_img)
            out.setdefault("planted", {})[nm] = dict(mean=x_pt, lo=x_lo, hi=x_hi, verdict=xv, per_image=x_img)
            print(f"control 3, planted defect '{nm}': mean x {x_pt:+.4f} [{x_lo:+.4f}, {x_hi:+.4f}] per image { {i: round(v, 4) for i, v in x_img.items()} } -> {xv}" + ("  (must be FAIL)" if nm == "notadded" else "  (reported)"))
# ---- secondary, never in the verdict
print(f"\nsecondary: mean KL(HF||arm) OFF {np.mean([r['kl_off'] for r in rows]):.4f}  ON {np.mean([r['kl_on'] for r in rows]):.4f};  argmax agreement with HF OFF {np.mean([r['agree_off'] for r in rows]):.3f}  ON {np.mean([r['agree_on'] for r in rows]):.3f}")
# ---- controls 1, 2, 4
c4 = np.mean([r["hf_argmax_in_top3"] for r in rows]); c4t = all(r["template_ids_equal"] for r in rows)
c2 = [r["on_equal_run2"] for r in rows if r["on_equal_run2"] is not None]
print(f"control 2, ON byte-identical across the two runs: {sum(c2)}/{len(c2)}")
print(f"control 4, template ids equal on every unit: {c4t}; HF step-0 argmax in an arm's top 3 on {100 * c4:.0f}% of units (needs >= 90%)")
if R3:
    ok = []
    for img in by:
        for arm in ("off", "on"):
            a = open(f"{A}/{img}.p00.{arm}.f32", "rb").read(); b = open(f"{R3}/{img}.{arm}.f32", "rb").read()[: len(a)]
            ok.append((img, arm, a == b))
    print(f"control 1, the served-prompt unit of each image byte-identical to G-S10j round 3 (first 4 steps): {sum(o[2] for o in ok)}/{len(ok)}" + ("" if all(o[2] for o in ok) else f"  MISMATCH {[o for o in ok if not o[2]]}"))
    out["reproduction"] = ok
json.dump({"summary": out, "units": rows}, open(f"{A}/gate-result.json", "w"), indent=1, default=str)
hb("done")
