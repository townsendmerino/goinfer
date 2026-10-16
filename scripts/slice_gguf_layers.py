#!/usr/bin/env python3
"""Cut a GGUF checkpoint down to its first N layers, as a real-weights fixture.

    python3 scripts/slice_gguf_layers.py <src.gguf> <dst.gguf> <N | i,j,k...>

The GGUF twin of slice_checkpoint_layers.py. N keeps layers 0..N-1; a comma list keeps those layers, in that order,
renumbered from 0 (a hybrid's slice can then hold one block of each kind without the layers between). Every tensor that
is not `blk.<i>.` is kept, and each kept tensor keeps its type and bytes. `<arch>.block_count` becomes the kept count and
every `<arch>.*` array whose length is the old layer count (per-layer lists such as nemotron_h's attention.head_count_kv
and feed_forward_length) keeps the kept layers' entries. Every other key is copied unchanged. No dependencies: GGUF v2/v3 is a header, the key/value pairs, the tensor infos, then the tensor data at
`general.alignment`.

What a slice is for: the real weights of a model too large to hold whole, for a FIDELITY question about its layers'
computation on two backends. It is not the model: its output is not the model's output, and a speed read off it is a
slice's speed (CLAUDE.md: quoting a slice as a model).
"""
import mmap
import struct
import sys

# GGUF value types -> struct format of one scalar (8 string, 9 array are handled apart).
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, buf):
        self.b, self.o = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("Q")
        s = bytes(self.b[self.o:self.o + n]).decode("utf-8")
        self.o += n
        return s

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            return (et, [self.value(et) for _ in range(n)])
        return self.take(SCALAR[t])


def put_string(out, s):
    b = s.encode("utf-8")
    out += struct.pack("<Q", len(b)) + b


def put_value(out, t, v):
    if t == 8:
        put_string(out, v)
    elif t == 9:
        et, items = v
        out += struct.pack("<IQ", et, len(items))
        for x in items:
            put_value(out, et, x)
    else:
        out += struct.pack("<" + SCALAR[t], v)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, spec = sys.argv[1], sys.argv[2], sys.argv[3]
    fsrc = open(src, "rb")
    buf = memoryview(mmap.mmap(fsrc.fileno(), 0, access=mmap.ACCESS_READ))
    r = Reader(buf)
    if bytes(buf[:4]) != b"GGUF":
        sys.exit(f"{src}: not a GGUF file")
    r.o = 4
    version, n_tensors, n_kv = r.take("I"), r.take("Q"), r.take("Q")
    if version not in (2, 3):
        sys.exit(f"{src}: GGUF version {version}, want 2 or 3")
    kvs = []
    for _ in range(n_kv):
        k = r.string()
        t = r.take("I")
        kvs.append([k, t, r.value(t)])
    kv = {k: (t, v) for k, t, v in kvs}
    arch = kv["general.architecture"][1]
    total = kv[f"{arch}.block_count"][1]
    layers = [int(x) for x in spec.split(",")] if "," in spec else list(range(int(spec)))
    if not layers or len(set(layers)) != len(layers) or not all(0 <= i < total for i in layers) or len(layers) >= total:
        sys.exit(f"{src}: layers {spec} must be distinct, in 0..{total - 1}, and fewer than {total}")
    new = {old: j for j, old in enumerate(layers)}
    n = len(layers)
    align = kv.get("general.alignment", (4, 32))[1]
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        typ, off = r.take("I"), r.take("Q")
        tensors.append((name, dims, typ, off))
    data0 = (r.o + align - 1) // align * align
    # Each tensor's bytes, padding included, run to the next tensor's offset (or the file's end).
    order = sorted(tensors, key=lambda x: x[3])
    size = {}
    for i, (name, _, _, off) in enumerate(order):
        end = order[i + 1][3] if i + 1 < len(order) else len(buf) - data0
        size[name] = end - off

    def renamed(name):  # the kept tensor's name in the slice, or None
        if not name.startswith("blk."):
            return name
        parts = name.split(".")
        j = new.get(int(parts[1]))
        return None if j is None else ".".join(["blk", str(j)] + parts[2:])

    cut, out_kvs = [], []
    for k, t, v in kvs:
        if k == f"{arch}.block_count":
            v = n
        elif k.startswith(arch + ".") and t == 9 and len(v[1]) == total:
            v = (v[0], [v[1][i] for i in layers])
            cut.append(k)
        out_kvs.append((k, t, v))
    kvs = out_kvs
    kept = [x for x in tensors if renamed(x[0]) is not None]
    kept.sort(key=lambda x: (not x[0].startswith("blk."), new.get(int(x[0].split(".")[1]), 0) if x[0].startswith("blk.") else 0))

    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, len(kept), len(kvs))
    for k, t, v in kvs:
        put_string(out, k)
        out += struct.pack("<I", t)
        put_value(out, t, v)
    off = 0
    for name, dims, typ, _ in kept:
        put_string(out, renamed(name))
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", typ, off)
        off += (size[name] + align - 1) // align * align
    out += b"\0" * ((-len(out)) % align)
    with open(dst, "wb") as f:
        f.write(out)
        for name, _, _, src_off in kept:
            b = buf[data0 + src_off:data0 + src_off + size[name]]
            f.write(b)
            f.write(b"\0" * ((-len(b)) % align))
    print(f"{dst}: {arch}, layers {layers} of {total} (renumbered 0..{n - 1}), {len(kept)} of {len(tensors)} tensors; per-layer arrays cut: {', '.join(cut) or 'none'}")


if __name__ == "__main__":
    main()
