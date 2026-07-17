#!/usr/bin/env python
"""Tiny-random Mistral (MistralForCausalLM) with a SMALL SLIDING WINDOW + text golden.

WHY A SEPARATE, DELIBERATELY SMALL-WINDOW FIXTURE. Sliding-window residency had no
window-ENGAGED gate. Every parity test ran short prompts, where winStart = max(0, nKeys-W)
= 0 and the window is inert — they proved only that passing a window doesn't break full
causal attention. The one test that tried (cuda/window_longctx_test.go) drove the real
phi3-mini-4k, whose window is 2047: it needed win+40 = 2087 forwards on BOTH the CPU and
the GPU of a 3.8B model, so it took 15-25 minutes and never actually completed. A gate that
cannot finish is not a gate.

The window's SPAN is not a numerical property worth scaling — winStart = max(pos-W+1, 0) is
the same arithmetic at W=16 as at W=4096. So a tiny checkpoint with sliding_window=16 tests
the identical logic in ~50 forwards instead of ~2087: deterministic, ~1s, no 7 GB download,
and it runs anywhere.

Mistral-arch on purpose (rather than adding a window to phi3-tiny): it also closes the
second gap — the README claims Mistral runs GPU-resident, but no Mistral checkpoint was ever
run resident. Mistral needs exactly one feature beyond plain dense (sliding-window), so this
fixture gates both the window math and the Mistral admission path. It is a NEW fixture, so
the existing phi3-tiny golden is untouched.

    ~/.venv-vl/bin/python scripts/pin_mistral_tiny_window.py
    -> testdata/mistral_tiny_window_golden.json + testdata/mistral-tiny-window/
"""
import json, os, torch
from transformers import MistralConfig, MistralForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "mistral_tiny_window_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "mistral-tiny-window")

# hidden_size 128 (not 64): the resident W4A8 GEMV packs K/8 words with a 32-word stride and
# int4 group scales every 32 elements, so a too-small K leaves nothing for the tail path to
# exercise. 128 keeps the fixture tiny while staying representative of the real packing.
# sliding_window 16 with a 48-token prompt puts ~32 positions PAST the window, so winStart
# is > 0 and moving for most of the run — which is the whole point.
WINDOW = 16
CFG = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
           sliding_window=WINDOW, rope_theta=10000.0, rms_norm_eps=1e-5,
           tie_word_embeddings=False, pad_token_id=0, eos_token_id=1, bos_token_id=2)
# Long enough that the tail positions are well past the window.
PROMPT = [2, 7, 42, 100, 5, 88, 13, 19, 61, 3, 77, 21, 9, 54, 33, 8,
          91, 12, 45, 6, 70, 29, 17, 83, 4, 38, 66, 11, 95, 23, 50, 15,
          72, 30, 87, 2, 41, 68, 19, 55, 26, 99, 7, 34, 63, 10, 48, 81]
N_NEW = 6


def main():
    torch.manual_seed(0)
    c = MistralConfig(**CFG)
    print("=== config of interest ===")
    cd = c.to_dict()
    for k in ["sliding_window", "hidden_act", "head_dim", "rope_scaling", "attention_bias"]:
        print(f"  {k} = {cd.get(k, '<absent>')}")
    if cd.get("sliding_window") != WINDOW:
        raise SystemExit(f"transformers dropped sliding_window (got {cd.get('sliding_window')}) — "
                         "the whole point of this fixture is that the window survives into config.json")
    m = MistralForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        last = m(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = dict(note=f"tiny Mistral text fwd fp32, sliding_window={WINDOW} (window ENGAGED past pos {WINDOW})",
             config=CFG, window=WINDOW, prompt_ids=PROMPT,
             argmax=int(torch.tensor(last).argmax()), last_logits=last,
             n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT, safe_serialization=True)
    print("saved", CKPT)
    # The released Mistral/Phi GGUFs DROP sliding_window, so only safetensors can test this.
    # Assert it landed rather than trusting save_pretrained.
    saved = json.load(open(os.path.join(CKPT, "config.json")))
    if saved.get("sliding_window") != WINDOW:
        raise SystemExit(f"config.json lost sliding_window (got {saved.get('sliding_window')})")
    print(f"config.json carries sliding_window={WINDOW}")


if __name__ == "__main__":
    main()
