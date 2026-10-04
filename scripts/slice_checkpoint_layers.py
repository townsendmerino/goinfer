#!/usr/bin/env python3
"""Cut a safetensors checkpoint down to its first N decoder layers, as a real-weights fixture.

    python3 scripts/slice_checkpoint_layers.py <src dir> <dst dir> <N>

Keeps every tensor that is not under `model.layers.<i>.` for i >= N (embeddings, final norm, LM head and layers
0..N-1), writes them to one `model.safetensors` with their original dtype and bytes, copies config.json with
`num_hidden_layers = N` (and a per-layer list such as `mlp_only_layers` or `layer_types` cut to match), and copies the
tokenizer and generation files. No torch: a safetensors file is an 8-byte little-endian header length, a JSON header
mapping names to {dtype, shape, data_offsets}, then the raw bytes.

What a slice is for: the real router, experts and weight distributions of a model too large to hold whole, for a
FIDELITY question about one layer's computation (D-G01: expert-major MoE prefill against the sequential path, per
layer). It is not the model: its output is not the model's output, and a speed read off it is a slice's speed
(CLAUDE.md: quoting a slice as a model). Its README says which checkpoint and which layers it holds.
"""
import json
import os
import re
import shutil
import struct
import sys

LAYER = re.compile(r"^model\.layers\.(\d+)\.")
COPY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json",
        "special_tokens_map.json", "tokenizer.model"]


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cfg = json.load(open(os.path.join(src, "config.json")))
    total = cfg["num_hidden_layers"]
    if not 0 < n < total:
        sys.exit(f"N must be in 1..{total - 1} (the checkpoint has {total} layers)")
    idx = os.path.join(src, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]

    keep = []  # (name, dtype, shape, file, absolute start, length)
    for fn in files:
        hdr, base = read_header(os.path.join(src, fn))
        for name, t in hdr.items():
            if name == "__metadata__":
                continue
            m = LAYER.match(name)
            if m and int(m.group(1)) >= n:
                continue
            s, e = t["data_offsets"]
            keep.append((name, t["dtype"], t["shape"], fn, base + s, e - s))
    keep.sort(key=lambda k: k[0])

    header, off = {"__metadata__": {"format": "pt", "sliced_from": os.path.basename(os.path.normpath(src)),
                                    "layers_kept": str(n), "layers_total": str(total)}}, 0
    for name, dt, shape, _, _, ln in keep:
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + ln]}
        off += ln
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)  # the data section starts 8-byte aligned

    os.makedirs(dst, exist_ok=True)
    out = os.path.join(dst, "model.safetensors")
    with open(out + ".tmp", "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        handles = {}
        for name, _, _, fn, start, ln in keep:
            f = handles.setdefault(fn, open(os.path.join(src, fn), "rb"))
            f.seek(start)
            left = ln
            while left:
                chunk = f.read(min(left, 64 << 20))
                if not chunk:
                    sys.exit(f"short read in {fn} for {name}")
                w.write(chunk)
                left -= len(chunk)
        for f in handles.values():
            f.close()
    os.replace(out + ".tmp", out)

    cfg["num_hidden_layers"] = n
    for k in ("layer_types", "mlp_only_layers"):
        if isinstance(cfg.get(k), list):
            v = cfg[k]
            cfg[k] = v[:n] if k == "layer_types" else [i for i in v if i < n]
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for fn in COPY:
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    with open(os.path.join(dst, "README.md"), "w") as f:
        f.write(f"Layer slice of {os.path.basename(os.path.normpath(src))}: decoder layers 0..{n - 1} of {total}, plus the "
                f"embeddings, final norm and LM head, bytes unchanged (scripts/slice_checkpoint_layers.py). A fidelity "
                f"fixture, not the model: never quote its output or speed as the model's.\n")
    print(f"{len(keep)} tensors, {off / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
