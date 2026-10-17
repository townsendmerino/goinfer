#!/usr/bin/env python3
"""make_record.py RESULTS.json REPO — the new sampled-CUDA ratios beside the published ones, and each arm's exact request fields built by the harness's own payload functions."""
import hashlib, json, os, statistics, sys
res, repo = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "b51"  # "b51": the eight published cells; "trunc": the shared truncated preset
TRUNC = "temp1.0_topk40_topp0.9"
sys.path.insert(0, os.path.join(repo, "scripts"))
import bench_peer as bp
data = json.load(open(res)); prov = data[0]; cells = [r for r in data[1:] if r.get("phase") in ("A", "C")]
# the published rows (docs/benchmarks.md, "Sampled configurations, depth 128, CUDA"): goinfer tok/s, Ollama tok/s
OLD = {("temp1.0_notrunc", "0.5B"): (237.7, 268.7), ("temp1.0_notrunc", "gemma3-1b"): (143.7, 148.2), ("temp1.0_notrunc", "phi3-mini"): (109.8, 125.6),
       ("temp0.8_topp0.95", "0.5B"): (227.2, 265.2), ("temp0.8_topp0.95", "gemma3-1b"): (124.7, 148.2), ("temp0.8_topp0.95", "phi3-mini"): (102.1, 125.7),
       ("temp0.8_topk40", "gemma3-1b"): (156.4, 148.0), ("temp0.8_topk40", "phi3-mini"): (112.1, 125.7)}
by = {}
for r in cells:
    by[(r["config"], r["model"], r["engine"])] = r
def row(cfg, m):
    g, o = by.get((cfg, m, "goinfer")), by.get((cfg, m, "ollama"))
    if not g or not o or not g.get("runs") or not o.get("runs"):
        return None
    gv = lambda r: (r.get("counts") or {}).get("token_gate", {}).get("verdict", "?")
    return dict(cfg=cfg, model=m, g=g["mean"], o=o["mean"], gs=g["spread"], os=o["spread"], gr=g["runs"], orr=o["runs"], gate=(gv(g), gv(o)), ratio=g["mean"] / o["mean"],
                gtok=(g.get("counts") or {}).get("tokens"), otok=(o.get("counts") or {}).get("tokens"))
print("## New readings against the published rows (ratio = goinfer / Ollama; below 1 means Ollama is faster)\n")
print("| config | model | goinfer tok/s (runs) | Ollama tok/s (runs) | **new ratio** | token gate (goinfer / Ollama) | published goinfer / Ollama | **published ratio** |")
print("|---|---|---|---|---|---|---|---|")
for (cfg, m) in (list(OLD) + [("temp0.8_topk40", "0.5B")] if mode == "b51" else [(TRUNC, m) for m in ("0.5B", "gemma3-1b", "phi3-mini")]):
    r = row(cfg, m)
    old = OLD.get((cfg, m))
    ot = f"{old[0]} / {old[1]}" if old else "— (not a published row: a control)"
    orat = f"{old[0]/old[1]:.3f}" if old else "—"
    if not r:
        print(f"| {cfg} | {m} | missing | missing | — | — | {ot} | {orat} |"); continue
    print(f"| {cfg} | {m} | {r['g']} ({', '.join(map(str, r['gr']))}) | {r['o']} ({', '.join(map(str, r['orr']))}) | **{r['ratio']:.3f}** | {r['gate'][0]} / {r['gate'][1]} | {ot} | {orat} |")
print("\n## Greedy depth 128, same session (phase A; controls)\n")
print("| model | goinfer tok/s | Ollama tok/s | ratio |\n|---|---|---|---|")
for m in ("0.5B", "gemma3-1b", "phi3-mini"):
    r = row("greedy", m)
    if r: print(f"| {m} | {r['g']} | {r['o']} | {r['ratio']:.3f} |")
print("\n## Each arm's exact request fields (built by bench_peer.py's own payload functions; the prompt is named by its SHA-256)\n")
for m in ("0.5B", "gemma3-1b", "phi3-mini"):
    path, tag = bp.MODELS[m]
    prompt = bp.prompt_for_depth(128, m)
    print(f"### {m} (prompt {len(prompt)} chars, sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:16]}, {bp.prompt_tokens(128, m)} tokens, format {bp.prompt_format(128, m)}; GGUF {path}, Ollama tag {tag})\n")
    for cfg in (("greedy", "temp1.0_notrunc", "temp0.8_topp0.95", "temp0.8_topk40") if mode == "b51" else ("greedy", TRUNC)):
        c = bp.CONFIGS[cfg]
        gp = bp.goinfer_payload(path, "<prompt>", c); op = bp.ollama_payload(tag, "<prompt>", c, "cuda")
        print(f"- `{cfg}` goinfer `{json.dumps({k: v for k, v in gp.items() if k != 'messages'}, sort_keys=True)}`")
        print(f"  Ollama `{json.dumps({k: v for k, v in op.items() if k != 'messages'}, sort_keys=True)}`")
print("\n## Provenance header (as the harness stamped it)\n\n```json\n" + json.dumps(prov, indent=1, default=str)[:3500] + "\n```")
