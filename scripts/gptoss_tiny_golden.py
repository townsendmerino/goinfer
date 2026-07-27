#!/usr/bin/env python3
"""Build a TINY gpt-oss model, run the HF reference forward, and emit both a GGUF
goinfer can load and a golden JSON (input ids + reference next-token logits) — the
forward-parity gate for the gpt-oss family (decoder/gptoss_test.go).

Non-MXFP4 on purpose: the experts are written as plain F32 GGUF tensors so the gate
runs today (aikit does not yet dequant ggml type 39). It exercises everything the
real path does EXCEPT the MXFP4 dequant, which the committed bit-exact unpacker test
covers separately. Uses a small sliding_window with seq_len > window so the
alternating sliding/full attention is actually exercised.

Run in the transformers venv:
  ~/.venv-vl/bin/python3 scripts/gptoss_tiny_golden.py \
      decoder/testdata/gptoss_tiny.gguf decoder/testdata/gptoss_tiny_golden.json
"""
import json
import sys

import numpy as np
import torch
import gguf
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

# --- tiny dims (fast, but exercise GQA + MoE top-k + sliding/full interleave + YaRN) ---
HIDDEN = 64
LAYERS = 4
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 8
INTER = 32          # expert intermediate size
EXPERTS = 4
TOPK = 2
VOCAB = 128
WINDOW = 4          # small so seq_len (8) > window ⇒ sliding layers actually clip
ROPE_THETA = 150000.0
EPS = 1e-5
SEQ = [3, 14, 7, 42, 1, 99, 5, 60]  # fixed input token ids (len 8 > WINDOW)


def build_config():
    return GptOssConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_local_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        sliding_window=WINDOW,
        rms_norm_eps=EPS,
        rope_theta=ROPE_THETA,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
            "beta_slow": 1.0, "truncate": False,
            "original_max_position_embeddings": 4096, "rope_theta": ROPE_THETA,
        },
        attn_implementation="eager",  # so the per-head sink softmax runs (sdpa path differs)
    )


def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else "decoder/testdata/gptoss_tiny.gguf"
    json_path = sys.argv[2] if len(sys.argv) > 2 else "decoder/testdata/gptoss_tiny_golden.json"

    torch.manual_seed(0)
    cfg = build_config()
    model = GptOssForCausalLM(cfg).eval()
    # Random but non-trivial weights (default init is near-zero for some params).
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)

    ids = torch.tensor([SEQ], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    logits = out.logits[0, -1].float().numpy()  # next-token logits at the last position
    argmax = int(logits.argmax())
    sys.stderr.write(f"reference argmax={argmax} logits[:4]={logits[:4]}\n")

    sd = model.state_dict()

    def t(name):
        return sd[name].detach().float().numpy()

    # --- write the GGUF (F32 tensors, gpt-oss metadata) ---
    w = gguf.GGUFWriter(gguf_path, "gpt-oss")
    w.add_uint32("gpt-oss.block_count", LAYERS)
    w.add_uint32("gpt-oss.context_length", 4096)
    w.add_uint32("gpt-oss.embedding_length", HIDDEN)
    w.add_uint32("gpt-oss.feed_forward_length", INTER)
    w.add_uint32("gpt-oss.attention.head_count", HEADS)
    w.add_uint32("gpt-oss.attention.head_count_kv", KV_HEADS)
    w.add_uint32("gpt-oss.attention.key_length", HEAD_DIM)
    w.add_uint32("gpt-oss.attention.value_length", HEAD_DIM)
    w.add_uint32("gpt-oss.attention.sliding_window", WINDOW)
    w.add_float32("gpt-oss.attention.layer_norm_rms_epsilon", EPS)
    w.add_uint32("gpt-oss.expert_count", EXPERTS)
    w.add_uint32("gpt-oss.expert_used_count", TOPK)
    w.add_uint32("gpt-oss.expert_feed_forward_length", INTER)
    w.add_float32("gpt-oss.rope.freq_base", ROPE_THETA)
    w.add_string("gpt-oss.rope.scaling.type", "yarn")
    w.add_float32("gpt-oss.rope.scaling.factor", 32.0)
    w.add_uint32("gpt-oss.rope.scaling.original_context_length", 4096)
    w.add_float32("gpt-oss.rope.scaling.yarn_beta_fast", 32.0)
    w.add_float32("gpt-oss.rope.scaling.yarn_beta_slow", 1.0)

    def add(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))

    # embeddings / head / final norm
    add("token_embd.weight", t("model.embed_tokens.weight"))       # [vocab, hidden] -> gguf [hidden, vocab]
    add("output_norm.weight", t("model.norm.weight"))              # [hidden]
    add("output.weight", t("lm_head.weight"))                      # [vocab, hidden]

    for i in range(LAYERS):
        hf = f"model.layers.{i}."
        p = f"blk.{i}."
        add(p + "attn_norm.weight", t(hf + "input_layernorm.weight"))
        add(p + "post_attention_norm.weight", t(hf + "post_attention_layernorm.weight"))
        # attention projections + biases (nn.Linear weight is [out, in] -> gguf [in, out])
        add(p + "attn_q.weight", t(hf + "self_attn.q_proj.weight"))
        add(p + "attn_k.weight", t(hf + "self_attn.k_proj.weight"))
        add(p + "attn_v.weight", t(hf + "self_attn.v_proj.weight"))
        add(p + "attn_output.weight", t(hf + "self_attn.o_proj.weight"))
        add(p + "attn_q.bias", t(hf + "self_attn.q_proj.bias"))
        add(p + "attn_k.bias", t(hf + "self_attn.k_proj.bias"))
        add(p + "attn_v.bias", t(hf + "self_attn.v_proj.bias"))
        add(p + "attn_output.bias", t(hf + "self_attn.o_proj.bias"))
        add(p + "attn_sinks.weight", t(hf + "self_attn.sinks"))     # [n_head]
        # router
        add(p + "ffn_gate_inp.weight", t(hf + "mlp.router.weight"))  # [n_expert, hidden]
        add(p + "ffn_gate_inp.bias", t(hf + "mlp.router.bias"))      # [n_expert]
        # experts: HF gate_up_proj [n_expert, hidden, 2*inter] (interleaved gate=even/up=odd),
        # down_proj [n_expert, inter, hidden]. De-interleave + transpose to gguf's
        # per-expert [out, in] stacked as (n_expert, out, in) -> gguf [in, out, n_expert].
        gate_up = t(hf + "mlp.experts.gate_up_proj")     # [E, hidden, 2*inter]
        gate_up_b = t(hf + "mlp.experts.gate_up_proj_bias")  # [E, 2*inter]
        down = t(hf + "mlp.experts.down_proj")           # [E, inter, hidden]
        down_b = t(hf + "mlp.experts.down_proj_bias")    # [E, hidden]
        gate = gate_up[:, :, 0::2]   # [E, hidden, inter]
        up = gate_up[:, :, 1::2]     # [E, hidden, inter]
        gate_b = gate_up_b[:, 0::2]  # [E, inter]
        up_b = gate_up_b[:, 1::2]    # [E, inter]
        add(p + "ffn_gate_exps.weight", np.transpose(gate, (0, 2, 1)))  # [E, inter, hidden]
        add(p + "ffn_up_exps.weight", np.transpose(up, (0, 2, 1)))      # [E, inter, hidden]
        add(p + "ffn_down_exps.weight", np.transpose(down, (0, 2, 1)))  # [E, hidden, inter]
        add(p + "ffn_gate_exps.bias", gate_b)  # [E, inter] -> gguf [inter, E]
        add(p + "ffn_up_exps.bias", up_b)      # [E, inter]
        add(p + "ffn_down_exps.bias", down_b)  # [E, hidden]

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sys.stderr.write(f"wrote GGUF: {gguf_path}\n")

    # self-check: read back a couple of dims so a layout mistake fails HERE, not in Go.
    r = gguf.GGUFReader(gguf_path)
    for tt in r.tensors:
        if tt.name == "ffn_gate_exps.weight" or tt.name == "blk.0.ffn_gate_exps.weight":
            sys.stderr.write(f"check {tt.name} dims={list(map(int,tt.shape))} (want [hidden,inter,E]=[{HIDDEN},{INTER},{EXPERTS}])\n")

    golden = {
        "input_ids": SEQ,
        "argmax": argmax,
        "logits": logits.tolist(),
        "dims": {"hidden": HIDDEN, "layers": LAYERS, "heads": HEADS, "kv_heads": KV_HEADS,
                 "head_dim": HEAD_DIM, "inter": INTER, "experts": EXPERTS, "topk": TOPK,
                 "vocab": VOCAB, "window": WINDOW},
    }
    with open(json_path, "w") as f:
        json.dump(golden, f)
    sys.stderr.write(f"wrote golden: {json_path} ({len(logits)} logits)\n")


if __name__ == "__main__":
    main()
