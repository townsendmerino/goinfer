#!/usr/bin/env python
"""Real-checkpoint G2 golden for Qwen3.5+ images (P8a, docs/measurements/p8a-qwen35-vl-2026-09/
preregistration.md): HF f32 greedy, N_NEW tokens, on the three grid-aligned procedural images.

    ~/g4venv/bin/python scripts/pin_qwen35_vl_preprocess.py --g2 ~/models/qwen35vl_g2   # images first
    ~/g4venv/bin/python scripts/pin_qwen35_vl_real.py ~/models/qwen3.5-0.8b ~/models/qwen35vl_g2

writes DIR/golden_{A,B,C}.json. The prompt is the model's REAL chat template applied to
[{"type":"image"},{"type":"text", Q}] with the single <|image_pad|> expanded to the merged-token count
(what Qwen3VLProcessor does), so goinfer's prompt-construction is checked by G4, not assumed here.
pixel_values come from the numpy transcription (scripts/qwen35vl_images.py) — no PIL/torchvision here.
Every step's top-1/top-2 logit gap is recorded: the pre-registered near-tie rule needs it.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

Q = "Describe this image in detail."
N_NEW = 32


def main():
    ckpt, d = (os.path.expanduser(a) for a in sys.argv[1:3])
    meta = json.load(open(os.path.join(d, "images.json")))
    tok = AutoTokenizer.from_pretrained(ckpt)
    m = Qwen3_5ForConditionalGeneration.from_pretrained(ckpt, dtype=torch.float32).eval()
    cfg = m.config
    img_id = cfg.image_token_id
    pad = tok.decode([img_id])
    for k, im in meta.items():
        t0 = time.time()
        grid = torch.tensor([im["grid_thw"]])
        n_merged = im["n_merged"]
        pv = torch.from_numpy(np.fromfile(os.path.join(d, f"img_{k}.pv.f32"), dtype="<f4").reshape(-1, 3 * 2 * 16 * 16))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        assert text.count(pad) == 1, "expected exactly one <|image_pad|> in the rendered template"
        text = text.replace(pad, pad * n_merged)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert ids.count(img_id) == n_merged
        start = ids.index(img_id)
        assert ids[start:start + n_merged] == [img_id] * n_merged
        inp = torch.tensor([ids])
        mm = torch.tensor([[1 if t == img_id else 0 for t in ids]], dtype=torch.int32)
        am = torch.ones_like(inp)
        with torch.no_grad():
            feats = m.model.get_image_features(pv, grid, return_dict=True).pooler_output[0]
            pos, delta = m.model.get_rope_index(inp, mm, image_grid_thw=grid, attention_mask=am)
            gen = m.generate(input_ids=inp, attention_mask=am, pixel_values=pv, image_grid_thw=grid,
                             mm_token_type_ids=mm, max_new_tokens=N_NEW, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
        toks = gen.sequences[0, len(ids):].tolist()
        gaps = [float(sc[0].topk(2).values.diff().abs()) for sc in gen.scores]
        json.dump(dict(image=k, question=Q, prompt_text=text[:200] + " ...", input_ids=ids,
                       image_token_id=img_id, image_token_start=start, n_image_tokens=n_merged,
                       grid_thw=[im["grid_thw"]], image_features=feats.reshape(-1).tolist(),
                       position_ids=pos[:, 0].tolist(), rope_delta=int(delta[0, 0]),
                       hf_tokens=toks, hf_gaps=gaps, hf_text=tok.decode(toks),
                       eos_ids=[int(x) for x in ([cfg.text_config.eos_token_id] if isinstance(cfg.text_config.eos_token_id, int) else cfg.text_config.eos_token_id)]),
                  open(os.path.join(d, f"golden_{k}.json"), "w"))
        print(f"{k}: {len(ids)} tokens ({n_merged} image), {len(toks)} generated in {time.time()-t0:.0f}s, "
              f"min gap {min(gaps):.3f}\n   {tok.decode(toks)!r}")


if __name__ == "__main__":
    main()
