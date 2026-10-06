#!/usr/bin/env python3
"""Pin the tiny EmbeddingGemma 2 fixture (docs/tasks/task-embeddinggemma2.md, Gate 1).

Builds a random-weight EmbeddingGemma2TextModel with transformers' own modeling code (>= 5.19.0, which added
`embedding_gemma2`), in float32, and writes:

  testdata/embeddinggemma2-tiny/config.json        the composite config's shape, text half only, as the real
                                                   google/embeddinggemma-2 config.json lays it out
  testdata/embeddinggemma2-tiny/model.safetensors  the text model's weights under the real checkpoint's
                                                   `language_model.` prefix
  testdata/embeddinggemma2-tiny/golden.json        for each input-id sequence: every layer's hidden state, the
                                                   projected last_hidden_state, the mean-pooled vector and its
                                                   L2-normalised form (the sentence-transformers pipeline:
                                                   Transformer -> Pooling(mean, prompt included) -> Normalize);
                                                   the same embedding from the same weights run in float64 (the
                                                   accuracy reference: float32 rounding in the reference itself is
                                                   of the same size as goinfer's, so a bar on the float32 output
                                                   alone cannot tell them apart); plus the bidirectional
                                                   sliding-window mask the reference builds

The shape is chosen so the defects that would stay silent cannot: layer_scalar and every norm weight are random, not
1; sliding and full layers have different head dims and KV-head counts; the sliding radius (3) is smaller than the
longest sequence (17), so the window cuts; and there are 1-, 2- and multi-token sequences (a 1-token case alone makes
every softmax the identity).

Run with a transformers that has embedding_gemma2, e.g. a venv: pip install 'transformers==5.19.0'.
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

import transformers
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel
from transformers import masking_utils

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "embeddinggemma2-tiny")
SEQS = [
    [2, 17],
    [2],
    [2, 45, 9, 130, 77, 5, 201, 33, 1],
    [2, 11, 250, 7, 7, 64, 180, 3, 99, 140, 22, 41, 8, 255, 61, 90, 1],
]


def main():
    torch.manual_seed(0x2026)
    cfg = EmbeddingGemma2TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=3,
        hidden_size_per_layer_input=16,
        embedding_dim=48,
        sliding_window_pattern=3,
        global_head_dim=32,
        num_global_key_value_heads=1,
    )
    cfg._attn_implementation = "eager"
    model = EmbeddingGemma2TextModel(cfg).eval().float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            else:
                p.copy_(0.15 * torch.randn_like(p))
        for i, layer in enumerate(model.layers):
            layer.layer_scalar.copy_(torch.tensor([0.6 + 0.15 * i]))

    os.makedirs(OUT, exist_ok=True)
    state = {"language_model." + k: v.contiguous() for k, v in model.state_dict().items()
             if "inv_freq" not in k and "embed_scale" not in k}
    save_file(state, os.path.join(OUT, "model.safetensors"))

    text = cfg.to_dict()
    config = {"architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2", "text_config": text,
              "transformers_version": transformers.__version__}
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    golden = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": []}
    m64 = EmbeddingGemma2TextModel(cfg).eval()
    m64.load_state_dict(model.state_dict())
    m64 = m64.double()
    for ids in SEQS:
        x = torch.tensor([ids])
        with torch.no_grad():
            out = model(input_ids=x, output_hidden_states=True)
        last = out.last_hidden_state[0]
        pooled = last.mean(dim=0)
        normed = torch.nn.functional.normalize(pooled, dim=0)
        with torch.no_grad():
            normed64 = torch.nn.functional.normalize(m64(input_ids=x).last_hidden_state[0].mean(dim=0), dim=0)
        golden["cases"].append({
            "ids": ids,
            "layers": [h[0].tolist() for h in out.hidden_states],
            "last_hidden_state": last.tolist(),
            "pooled": pooled.tolist(),
            "normalized": normed.tolist(),
            "normalized_f64": normed64.tolist(),
        })
    # The sliding mask the reference builds for the longest case, read back as allowed[q][k].
    T = len(SEQS[-1])
    emb = torch.zeros(1, T, cfg.hidden_size)
    m = masking_utils.create_bidirectional_sliding_window_mask(config=cfg, inputs_embeds=emb, attention_mask=None)
    if m is None:
        allowed = [[True] * T for _ in range(T)]
    else:
        m = m[0, 0]
        allowed = (m == 0).tolist() if m.dtype != torch.bool else m.tolist()
    golden["sliding_mask"] = allowed
    with open(os.path.join(OUT, "golden.json"), "w") as f:
        json.dump(golden, f)
    print("wrote", OUT, "with", len(SEQS), "cases; transformers", transformers.__version__, file=sys.stderr)


if __name__ == "__main__":
    main()
