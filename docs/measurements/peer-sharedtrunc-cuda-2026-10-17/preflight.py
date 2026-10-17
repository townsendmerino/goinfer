#!/usr/bin/env python3
"""Before the first half: each arm's request fields (seed, context, min_p, the sampling fields) and quant, from the harness's own functions. Prints markdown."""
import json, os, sys
sys.path.insert(0, "scripts"); import bench_peer as b
cfg = b.CONFIGS["temp0.8_topk40_topp0.9"]
engines = ["goinfer", "ollama", "llamacpp"]
print(f"context every arm: cell_ctx() = {b.cell_ctx()}\n")
for m in ("0.5B", "gemma3-1b", "phi3-mini"):
    path, tag = b.MODELS[m]
    gp = b.goinfer_payload(path, "<prompt>", cfg); op = b.ollama_payload(tag, "<prompt>", cfg, "cuda"); lp = b.llamacpp_payload("<prompt>", cfg)
    strip = lambda p: {k: v for k, v in p.items() if k != "messages"}
    print(f"### {m}\n- goinfer `{json.dumps(strip(gp), sort_keys=True)}`\n  cmd `{' '.join(b.goinfer_cmd('<serve>', path, 'cuda', []))}`")
    print(f"- Ollama `{json.dumps(strip(op), sort_keys=True)}`")
    print(f"- llama.cpp `{json.dumps(strip(lp), sort_keys=True)}`\n  cmd `{' '.join(b.llamacpp_cmd(path, ['-ngl', '99']))}`")
    for e in ("ollama", "llamacpp"):
        print(f"- {e} quant: {b.peer_quant(e, m)}")
    print(f"- goinfer quant asked: {b.goinfer_quant(m)}  (override table {b.GOINFER_QUANT_OVERRIDE})")
    print(f"- quant_mismatch: {b.quant_mismatch(m, engines)!r}\n")
need = ("temperature", "top_p", "top_k", "seed")
for name, body in (("goinfer", b.goinfer_payload("p", "x", cfg)), ("ollama", b.ollama_payload("t", "x", cfg)["options"]), ("llamacpp", b.llamacpp_payload("x", cfg))):
    assert all(k in body for k in need), (name, body)
assert "min_p" in b.ollama_payload("t", "x", cfg)["options"] and "min_p" in b.llamacpp_payload("x", cfg), "min_p missing on a peer arm"
print("CHECK OK: temperature, top_p, top_k and seed on every arm; min_p on Ollama and llama.cpp (goinfer has no min_p field)")
