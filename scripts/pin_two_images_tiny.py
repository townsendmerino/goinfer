#!/usr/bin/env python
"""Two-image goldens for S11 (docs/tasks/task-multimodal-support-2026-10.md, "S11, plan and gates"): every tiny VL
fixture with TWO images in one prompt, in two layouts:

  interleaved  prefix, image A, text, image B, text   (Qwen: A and B different sizes)
  adjacent     prefix, image A, image B, text         (the same size; nothing but each block's own wrapper between them)

Each image is its own block (Qwen: its own t = 1 grid): HF never pairs separate images, and neither may goinfer
(docs/multimodal.md, "Do not pair images"). Loads the EXISTING checkpoints under --td (never re-creates them: a re-pin
under another torch is not the same fixture), so pin from the same fixture bytes the Go tests read.

Per layout: input_ids, the image spans [[start, n], ...], each image's pixel_values and tower output, the logits at
EVERY position, and a 4-token greedy continuation. Qwen adds grid_thw, the m-RoPE position_ids and rope_delta. Gemma 3
and Gemma 4 add second_causal_logits: the same forward with only the FIRST block marked bidirectional, so the planted
defect "the second block left causal" is measured in HF itself (a fixture whose two forwards barely differ is blind to
it, and the Go test says so instead of passing vacuously).

    ~/g4venv/bin/python scripts/pin_two_images_tiny.py --td <testdata dir> --out <dir> --only gemma3_vl,qwen25vl,qwen35vl
    ~/.venv-vl/bin/python scripts/pin_two_images_tiny.py --td <testdata dir> --out <dir> --only gemma4_vl_bidir
    -> <out>/{gemma3_vl,qwen25vl,qwen35vl,gemma4_vl_bidir}_tiny_two_images_golden.json

Gemma 4 is pinned under transformers 5.12 (~/.venv-vl), not 5.15 (~/g4venv), ON PURPOSE. 5.15 changed a "vision"
Gemma 4's bidirectional mask: global layers causal only, and on sliding layers the window applied after the block OR.
5.12 applied Gemma 3's mask on every layer, which is what goinfer's gemma4AttendRange implements. Measured 2026-10-09:
the single-image golden re-pinned under 5.15 reads cosine 0.99982 against the committed (5.12) one, with identical image
features and causal-only logits. Which is right for the real checkpoints is a separate question
(docs/tasks/task-multimodal-support-2026-10.md, S11 results); S11's golden measures multi-image handling against the
semantics goinfer has.
"""
import argparse
import json
import os

import torch

N_CONT = 4


def greedy(fwd, ids):
    cur, cont = list(ids), []
    for _ in range(N_CONT):
        cont.append(int(fwd(cur).logits[0, -1].argmax()))
        cur.append(cont[-1])
    return cont


def spans_of(ids, tok):
    out, i = [], 0
    while i < len(ids):
        if ids[i] == tok:
            j = i
            while j < len(ids) and ids[j] == tok:
                j += 1
            out.append([i, j - i])
            i = j
        else:
            i += 1
    return out


def gemma3(td):
    from transformers import Gemma3ForConditionalGeneration
    m = Gemma3ForConditionalGeneration.from_pretrained(os.path.join(td, "gemma3-vl-tiny"), dtype=torch.float32).eval()
    IMG, N = 260, 4
    B, E = 254, 253  # in-vocab stand-ins for <start_of_image>/<end_of_image> (the tiny config's are out of its vocab)
    blk = [B] + [IMG] * N + [E]
    cfg = m.config.vision_config
    g = torch.Generator().manual_seed(11)
    pv = torch.randn(2, cfg.num_channels, cfg.image_size, cfg.image_size, generator=g)
    layouts = {"interleaved": [2, 7, 42] + blk + [13, 88] + blk + [21, 99],
               "adjacent": [2, 7, 42] + blk + blk + [13, 88]}
    out = {}
    with torch.no_grad():
        vis = m.model.vision_tower(pixel_values=pv).last_hidden_state
        feats = m.model.multi_modal_projector(vis)  # [2, N, hidden]
        for name, ids in layouts.items():
            sp = spans_of(ids, IMG)
            assert len(sp) == 2 and all(n == N for _, n in sp), sp
            t = torch.tensor([ids])
            tt = (t == IMG).long()
            tt2 = tt.clone()
            tt2[0, sp[1][0]:sp[1][0] + N] = 0  # the second block not marked: causal there
            full = m(input_ids=t, pixel_values=pv, token_type_ids=tt, use_cache=False).logits[0]
            # Only the mask may change: the splice still needs the second block's placeholders, so the defect run keeps
            # input_ids and pixel_values and moves only token_type_ids.
            bad = m(input_ids=t, pixel_values=pv, token_type_ids=tt2, use_cache=False).logits[0]
            cont = greedy(lambda c: m(input_ids=torch.tensor([c]), pixel_values=pv,
                                      token_type_ids=(torch.tensor([c]) == IMG).long(), use_cache=False), ids)
            out[name] = dict(input_ids=ids, spans=sp, logits=full.tolist(), second_causal_logits=bad.tolist(),
                             second_causal_max_abs_diff=float((full - bad).abs().max()), continuation_ids=cont)
            print(f"gemma3 {name}: spans {sp}, second-block-causal max|diff| {out[name]['second_causal_max_abs_diff']:.4f}")
    return dict(note="tiny Gemma 3 VL, two images; CPU fp32; transformers " + __import__("transformers").__version__,
                image_token_index=IMG, mm_tokens_per_image=N, boi_standin=B, eoi_standin=E,
                pixel_values_shape=list(pv.shape), pixel_values=[pv[i].flatten().tolist() for i in range(2)],
                image_features=[feats[i].reshape(-1).tolist() for i in range(2)], layouts=out)


def qwen(td, which):
    if which == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as C
        ck, vend = "qwen25vl-tiny", None  # its vision_end id is outside the tiny vocab; the single-image golden omits it too
    else:
        from transformers import Qwen3_5ForConditionalGeneration as C
        ck, vend = "qwen35vl-tiny", 252
    m = C.from_pretrained(os.path.join(td, ck), dtype=torch.float32).eval()
    IMG, VS = m.config.image_token_id, m.config.vision_start_token_id
    v = m.config.vision_config
    merge = v.spatial_merge_size
    patch_dim = v.in_chans * v.temporal_patch_size * v.patch_size ** 2 if hasattr(v, "in_chans") else \
        v.in_channels * v.temporal_patch_size * v.patch_size ** 2

    def image(grid, seed):
        n = grid[0] * grid[1] * grid[2]
        return torch.randn(n, patch_dim, generator=torch.Generator().manual_seed(seed))

    def block(grid):
        n = grid[0] * grid[1] * grid[2] // merge ** 2
        return [VS] + [IMG] * n + ([vend] if vend is not None else [])

    plans = {"interleaved": ([[1, 4, 6], [1, 4, 4]], lambda a, b: [2, 7, 42] + a + [13, 88] + b + [5, 100]),
             "adjacent": ([[1, 4, 6], [1, 4, 6]], lambda a, b: [2, 7, 42] + a + b + [13, 88])}
    out = {}
    with torch.no_grad():
        for name, (grids, lay) in plans.items():
            pvs = [image(gr, 21 + k) for k, gr in enumerate(grids)]
            pv = torch.cat(pvs)
            grid = torch.tensor(grids)
            ids = lay(block(grids[0]), block(grids[1]))
            t = torch.tensor([ids])
            mm = (t == IMG).int()
            vo = m.model.get_image_features(pv, grid, return_dict=True).pooler_output
            feats = [x.reshape(-1).tolist() for x in vo] if isinstance(vo, (list, tuple)) else None
            if feats is None or len(feats) != 2:  # some versions return one concatenated tensor
                cat = torch.cat(list(vo)) if isinstance(vo, (list, tuple)) else vo
                n0 = grids[0][0] * grids[0][1] * grids[0][2] // merge ** 2
                feats = [cat[:n0].reshape(-1).tolist(), cat[n0:].reshape(-1).tolist()]
            pos, delta = m.model.get_rope_index(t, mm, image_grid_thw=grid, attention_mask=torch.ones_like(t))
            full = m(input_ids=t, attention_mask=torch.ones_like(t), pixel_values=pv, image_grid_thw=grid,
                     mm_token_type_ids=mm, use_cache=False).logits[0]
            cont = greedy(lambda c: m(input_ids=torch.tensor([c]), attention_mask=torch.ones(1, len(c), dtype=torch.long),
                                      pixel_values=pv, image_grid_thw=grid,
                                      mm_token_type_ids=(torch.tensor([c]) == IMG).int(), use_cache=False), ids)
            sp = spans_of(ids, IMG)
            assert len(sp) == 2, sp
            out[name] = dict(input_ids=ids, spans=sp, grid_thw=grids, pixel_values=[x.flatten().tolist() for x in pvs],
                             image_features=feats, position_ids=pos[:, 0].tolist(), rope_delta=int(delta.reshape(-1)[0]),
                             logits=full.tolist(), continuation_ids=cont)
            print(f"{which} {name}: spans {sp}, grids {grids}, rope_delta {out[name]['rope_delta']}")
    return dict(note=f"tiny {which}, two images, each its own t=1 grid; CPU fp32; transformers " +
                __import__("transformers").__version__, image_token_id=IMG, vision_start_token_id=VS,
                vision_end_token_id=vend, layouts=out)


def gemma4(td):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    m = Gemma4ForConditionalGeneration.from_pretrained(os.path.join(td, "gemma4-vl-bidir-tiny"), dtype=torch.float32).eval()
    IMG, B, E = 250, 251, 252  # 250 as pin_gemma4_vl_bidir_image.py; 251/252 in-vocab begin/end stand-ins
    m.config.image_token_id = IMG
    m.config.text_config.image_token_id = IMG
    v = m.config.vision_config
    grid, pk = 9, v.pooling_kernel_size
    n = (grid // pk) ** 2
    patch_dim = 3 * v.patch_size ** 2
    pv = torch.rand(2, grid * grid, patch_dim, generator=torch.Generator().manual_seed(31))
    xs, ys = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="xy")
    pos = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).unsqueeze(0).repeat(2, 1, 1).long()
    blk = [B] + [IMG] * n + [E]
    layouts = {"interleaved": [2, 7, 42] + blk + [13, 88] + blk + [5, 100],
               "adjacent": [2, 7, 42] + blk + blk + [13, 88]}
    out = {}
    with torch.no_grad():
        feats = m.model.get_image_features(pv, pos, return_dict=True).pooler_output
        feats = [f.reshape(-1).tolist() for f in (feats if isinstance(feats, (list, tuple)) else feats.reshape(2, n, -1))]
        for name, ids in layouts.items():
            sp = spans_of(ids, IMG)
            assert len(sp) == 2 and all(k == n for _, k in sp), sp
            t = torch.tensor([ids])
            mm = (t == IMG).long()
            mm2 = mm.clone()
            mm2[0, sp[1][0]:sp[1][0] + n] = 0
            run = lambda c, mmx: m(input_ids=torch.tensor([c]) if not torch.is_tensor(c) else c, pixel_values=pv,
                                   image_position_ids=pos, mm_token_type_ids=mmx, use_cache=False)
            full = run(t, mm).logits[0]
            bad = run(t, mm2).logits[0]
            cont = greedy(lambda c: run(c, (torch.tensor([c]) == IMG).long()), ids)
            out[name] = dict(input_ids=ids, spans=sp, logits=full.tolist(), second_causal_logits=bad.tolist(),
                             second_causal_max_abs_diff=float((full - bad).abs().max()), continuation_ids=cont)
            print(f"gemma4 {name}: spans {sp}, second-block-causal max|diff| {out[name]['second_causal_max_abs_diff']:.4f}")
    return dict(note="tiny Gemma 4 VL (bidirectional vision), two images; CPU fp32; transformers " +
                __import__("transformers").__version__, image_token_id=IMG, boi_standin=B, eoi_standin=E,
                n_image_tokens=n, pixel_values_shape=list(pv.shape), pixel_values=[pv[i].flatten().tolist() for i in range(2)],
                image_features=feats, layouts=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--td", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    jobs = {"gemma3_vl": lambda: gemma3(a.td), "qwen25vl": lambda: qwen(a.td, "qwen25vl"),
            "qwen35vl": lambda: qwen(a.td, "qwen35vl"), "gemma4_vl_bidir": lambda: gemma4(a.td)}
    for k, f in jobs.items():
        if a.only and k not in a.only.split(","):
            continue
        g = f()
        p = os.path.join(a.out, f"{k}_tiny_two_images_golden.json")
        json.dump(g, open(p, "w"))
        print("wrote", p)


if __name__ == "__main__":
    main()
