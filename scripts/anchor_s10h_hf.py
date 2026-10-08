#!/usr/bin/env python3
"""G-S10h's Hugging Face float32 side (docs/tasks/task-multimodal-support-2026-10.md, "G-S10h", registered before this code).

Reads the dump written by cuda TestS10DeepstackPrefillCUDA_anchorDump (ids, pixels, grid, teacher tokens, and the 9 logit vectors of the "off" and "on" arms per image),
runs Qwen3-VL-2B in float32 on the CPU over the SAME ids and pixels with the teacher tokens appended, and prints the registered reading:
  delta = cos(on, HF) - cos(off, HF) per (image, step); FAIL if mean < -0.005 or any step-0 delta < -0.02; PASS if mean >= -0.002 and every step-0 delta >= -0.01; else AMBIGUOUS.
Usage: ~/g4venv/bin/python scripts/anchor_s10h_hf.py <model dir> <dump dir> [out json]
"""
import json, math, sys, time
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

model_dir, dump = sys.argv[1], sys.argv[2]
out_json = sys.argv[3] if len(sys.argv) > 3 else dump + "/anchor-summary.json"
IMAGES = [("gemma3_preprocess_image.png", "gemma3_preprocess_image.png"), ("qwen25vl_preprocess_image.png", "qwen25vl_preprocess_image.png"),
          ("formula.png", "glm_ocr/formula.png"), ("table.png", "glm_ocr/table.png")]
T0 = time.time()
def hb(msg): print(f"[G-S10h hf {time.time() - T0:6.0f}s] {msg}", flush=True)

tok = AutoTokenizer.from_pretrained(model_dir)
hb("loading float32 model")
model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32)
model.eval()
hb(f"loaded; torch threads {torch.get_num_threads()}")
try:
    proc = AutoProcessor.from_pretrained(model_dir)
except Exception as e:
    proc = None; hb(f"no processor: {e}")

def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))
def logsm(x):
    x = x.astype(np.float64); m = x.max(); return x - m - math.log(np.exp(x - m).sum())

rows, summary = [], {}
for tag, rel in IMAGES:
    meta = json.load(open(f"{dump}/{tag}.meta.json"))
    n, steps, vocab, grid = meta["n"], meta["steps"], meta["vocab"], meta["grid"]
    px = np.fromfile(f"{dump}/{tag}.px.f32", dtype="<f4")
    off = np.fromfile(f"{dump}/{tag}.off.f32", dtype="<f4").reshape(steps, vocab)
    on = np.fromfile(f"{dump}/{tag}.on.f32", dtype="<f4").reshape(steps, vocab)
    pdim = px.size // (grid[0] * grid[1] * grid[2])
    pix = torch.from_numpy(px.reshape(-1, pdim).copy())
    info = {"rows": n, "patch_dim": pdim}
    # instrument check (ii), informational: HF's own preprocessing of the same image at the same cap
    if proc is not None:
        try:
            from PIL import Image
            img = Image.open(f"testdata/{rel}").convert("RGB")
            r = proc.image_processor(images=img, size={"shortest_edge": proc.image_processor.size["shortest_edge"], "longest_edge": 1048576}, return_tensors="pt")
            same_grid = r["image_grid_thw"][0].tolist() == grid
            md = float((r["pixel_values"] - pix).abs().max()) if r["pixel_values"].shape == pix.shape else None
            info.update(hf_grid=r["image_grid_thw"][0].tolist(), same_grid=same_grid, pixel_max_abs_diff=md)
            hb(f"{tag}: (ii) HF processor grid {info['hf_grid']} against dump {grid}: same={same_grid}; pixel max|diff| = {md}")
        except Exception as e:
            hb(f"{tag}: (ii) processor check failed: {e}")
    # instrument check (iv), G-S10i: the hand-built ids are the checkpoint's own chat template's rendering of one image then the prompt, image-pad expanded
    try:
        txt = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": meta["prompt"]}]}], tokenize=False, add_generation_prompt=True)
        txt = txt.replace("<|image_pad|>", "<|image_pad|>" * meta["nImg"])
        tmpl_ids = tok.encode(txt, add_special_tokens=False)
        info["iv_template_ids_equal"] = (tmpl_ids == meta["ids"])
        hb(f"{tag}: (iv) chat-template ids equal the dump's ids: {info['iv_template_ids_equal']} ({len(tmpl_ids)} vs {len(meta['ids'])})")
    except Exception as e:
        info["iv_template_ids_equal"] = None; hb(f"{tag}: (iv) could not run: {e}")
    ids = torch.tensor([meta["ids"] + meta["teacher"][: steps - 1]])
    mm = (ids == tok.convert_tokens_to_ids("<|image_pad|>")).long()  # transformers 5 wants the image-token mask for M-RoPE
    hb(f"{tag}: HF float32 forward, {ids.shape[1]} tokens")
    t = time.time()
    with torch.no_grad():
        lg = model(input_ids=ids, mm_token_type_ids=mm, pixel_values=pix, image_grid_thw=torch.tensor([grid])).logits[0].float().numpy()
    hf = lg[n - 1 : n - 1 + steps][:, :vocab]
    hb(f"{tag}: done in {time.time() - t:.0f}s, logits {lg.shape}")
    c_off = [cos(off[k], hf[k]) for k in range(steps)]
    c_on = [cos(on[k], hf[k]) for k in range(steps)]
    d = [b - a for a, b in zip(c_off, c_on)]
    kl = lambda arm: float((np.exp(logsm(hf[0])) * (logsm(hf[0]) - logsm(arm[0]))).sum())
    info.update(cos_off=c_off, cos_on=c_on, delta=d, step0_delta=d[0], kl_off=kl(off), kl_on=kl(on),
                argmax_hf=int(hf[0].argmax()), argmax_off=int(off[0].argmax()), argmax_on=int(on[0].argmax()),
                off_vs_on_worst=min(cos(off[k], on[k]) for k in range(steps)))
    top3 = lambda x: set(np.argsort(-x[0])[:3].tolist())
    info["sanity_iii_hf_argmax_in_top3_of_an_arm"] = info["argmax_hf"] in (top3(off) | top3(on))
    if tag == "table.png":
        lo = lambda x: float(logsm(x[0])[tok.encode("Quarter", add_special_tokens=False)[0]] - logsm(x[0])[tok.encode("Table", add_special_tokens=False)[0]])
        info["table_step0_logodds_quarter_over_table"] = {"hf": lo(hf), "off": lo(off), "on": lo(on)}
        p = lambda x, w: float(np.exp(logsm(x[0]))[tok.encode(w, add_special_tokens=False)[0]])
        info["table_step0_probs"] = {w: {"hf": p(hf, w), "off": p(off, w), "on": p(on, w)} for w in ("Quarter", "Table")}
    summary[tag] = info; rows += d
    hb(f"{tag}: cos(off,HF) {' '.join(f'{x:.4f}' for x in c_off)}")
    hb(f"{tag}: cos(on ,HF) {' '.join(f'{x:.4f}' for x in c_on)}  step-0 delta {d[0]:+.4f}  KL off {info['kl_off']:.4f} on {info['kl_on']:.4f}  off-vs-on worst {info['off_vs_on_worst']:.4f}")
mean = float(np.mean(rows)); s0 = [summary[t]["step0_delta"] for t, _ in IMAGES]
verdict = "FAIL" if (mean < -0.005 or min(s0) < -0.02) else ("PASS" if (mean >= -0.002 and min(s0) >= -0.01) else "AMBIGUOUS")
summary["_reading"] = {"pairs": len(rows), "mean_delta": mean, "step0_deltas": s0, "verdict": verdict}
json.dump(summary, open(out_json, "w"), indent=1)
hb(f"READING: {len(rows)} pairs, mean delta {mean:+.5f}, step-0 deltas {[round(x, 4) for x in s0]} -> {verdict}")
if "table_step0_probs" in summary["table.png"]:
    hb(f"table.png step-0 probabilities: {json.dumps(summary['table.png']['table_step0_probs'])}")
hb("table.png step-0 ln(p(Quarter)/p(Table)): " + json.dumps(summary["table.png"]["table_step0_logodds_quarter_over_table"]))
hb("instrument (iv) template ids: " + str({t: summary[t]["iv_template_ids_equal"] for t, _ in IMAGES}))
hb("instrument (iii) sanity: " + str({t: summary[t]["sanity_iii_hf_argmax_in_top3_of_an_arm"] for t, _ in IMAGES}))
