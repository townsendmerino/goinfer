#!/usr/bin/env python3
"""Extract a bit-exactness golden fixture for the MXFP4 unpacker (goinfer/decoder/mxfp4_test.go).

Reads a REAL MXFP4 tensor from the gpt-oss:20b GGUF (reached from the local Ollama store, no
recopy) and emits, for the first N blocks: the raw block bytes + the reference `gguf`-library
dequantized values as their float32 BIT patterns. The Go test replays the unpack on the same
raw bytes and must match every bit — the exact discipline (verify unpack vs the reference lib
before writing the forward) that de-risks the whole task.

Usage: python3 scripts/extract_mxfp4_golden.py [gguf_path] > testdata/mxfp4_golden.json
If gguf_path is omitted, resolves gpt-oss:20b from the Ollama manifest.
"""
import json, os, sys, glob
import numpy as np
import gguf


def resolve_gguf_from_ollama() -> str:
    root = os.path.expanduser("~/.ollama/models")
    # manifest: manifests/registry.ollama.ai/library/gpt-oss/20b (JSON of layers → blob digests)
    man = None
    for p in glob.glob(os.path.join(root, "manifests", "**", "gpt-oss", "*"), recursive=True):
        if os.path.isfile(p):
            man = p
            break
    if not man:
        sys.exit("could not find a gpt-oss manifest under ~/.ollama/models/manifests")
    m = json.load(open(man))
    # the model weights layer holds the GGUF; pick the largest blob referenced.
    digs = [l["digest"] for l in m.get("layers", []) if "digest" in l]
    if "config" in m and "digest" in m["config"]:
        digs.append(m["config"]["digest"])
    best, bestsz = None, -1
    for d in digs:
        blob = os.path.join(root, "blobs", d.replace(":", "-"))
        if os.path.isfile(blob) and os.path.getsize(blob) > bestsz:
            best, bestsz = blob, os.path.getsize(blob)
    if not best:
        sys.exit("no blob files found for the gpt-oss manifest")
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else resolve_gguf_from_ollama()
    sys.stderr.write(f"reading GGUF: {path}\n")
    r = gguf.GGUFReader(path)
    MXFP4 = gguf.GGMLQuantizationType.MXFP4

    # first tensor stored as MXFP4 (an expert / ffn weight in gpt-oss)
    t = next((t for t in r.tensors if t.tensor_type == MXFP4), None)
    if t is None:
        sys.exit("no MXFP4 tensor found in this GGUF")
    name = t.name
    row = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else int(t.shape[0])
    sys.stderr.write(f"tensor {name} shape={list(t.shape)} type=MXFP4\n")

    # t.data is the raw quantized bytes as uint8 (gguf exposes it flat). 17 bytes/block.
    raw = np.asarray(t.data, dtype=np.uint8).reshape(-1)
    n_blocks = min(128, raw.size // 17)
    if n_blocks < 1:
        sys.exit(f"tensor too small: {raw.size} bytes")
    raw_blocks = raw[: n_blocks * 17].copy()

    # reference dequant via the gguf library on exactly those blocks.
    blocks = raw_blocks.reshape(n_blocks, 17)
    deq = gguf.quants.MXFP4.dequantize_blocks(blocks).reshape(-1).astype(np.float32)

    want_bits = deq.view(np.uint32).tolist()  # float32 bit patterns → exact comparison in Go
    fixture = {
        "tensor": name,
        "block_bytes": 17,
        "block_elems": 32,
        "n_blocks": int(n_blocks),
        "raw_hex": raw_blocks.tobytes().hex(),
        "want_bits": want_bits,
    }
    json.dump(fixture, sys.stdout)
    sys.stderr.write(f"wrote fixture: {n_blocks} blocks, {len(want_bits)} float32 values\n")


if __name__ == "__main__":
    main()
