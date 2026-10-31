#!/usr/bin/env python3
"""G-VK1's grading (SB2-vk, docs/tasks/task-subsystems-2026-10.md): each model's predicted Vulkan device use against nvidia-smi's rise
over the load, per load. usage: score_vk1.py <dir> <pass>. M1's rule lives in m1_predict(), fixed before the run."""
import glob, os, re, sys

MiB = 1 << 20
TOL_MIB = 64  # the registered bar


def loads(d, p):
    out = []
    for f in sorted(glob.glob(os.path.join(d, f"{p}-*-probe.txt"))):
        tag = os.path.basename(f)[len(p) + 1:-len("-probe.txt")]
        line = open(f).read()
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        before, loaded = (int(x) for x in re.search(r"before=(\d+) loaded=(\d+)", line).groups())
        bufs = []
        for row in open(f[:-len(".txt")] + "-buffers.tsv").read().splitlines()[1:]:
            _, n, u, ln = row.split("\t")
            bufs.append((int(n), int(u, 16), ln))
        out.append(dict(tag=tag, rise=(loaded - before) * MiB, B=int(kv["B"]), ledger=int(kv["ledger_list"]), bufs=bufs))
    return out


def m0(L):
    return L["ledger"]


# M1: gpu-allocator 0.28.0 as wgpu-hal 29.0.3 drives it under MemoryHints::Performance (wgpu-native v29.0.1.1 always passes it),
# fixed from the source before the run (README, "M1, fixed before the run"). Nothing in it is fitted.
MAP_READ, MAP_WRITE = 0x1, 0x2
ALIGN = 256          # the driver's req.alignment and req.size rounding for a buffer: assumed, not read (README)
INTERNAL = (512 << 10) + 256 + 256  # wgpu-core's per-device zero buffer, external-texture params and indirect-validation buffer


def _pack(sizes, first, grow, cap):
    """Device memory gpu-allocator opens for sizes, placed in order: a personal block of exactly the size above the current
    block size; else best fit in the open general blocks, newest first; else a new general block. Returns bytes allocated."""
    blocks = []  # free bytes per general block (no fragmentation inside a block: sizes are aligned and live)
    total = 0
    for r in sizes:
        mb = min(first << min(len(blocks), 7), cap) if grow else first
        if r > mb:
            total += r
            continue
        fit = [i for i in range(len(blocks)) if blocks[i] >= r]
        if fit:
            i = min(fit, key=lambda i: (blocks[i], -i))
            blocks[i] -= r
            continue
        blocks.append(mb - r)
        total += mb
    return total


def _al(n):
    n = max(n, 4)
    n = (n + 3) // 4 * 4
    return (n + ALIGN - 1) // ALIGN * ALIGN


def m1_predict(bufs, bar_type, host_term):
    dev = [INTERNAL] + [_al(n) for n, u, _ in bufs if not (u & (MAP_READ | MAP_WRITE))]
    total = _pack(dev, 128 * MiB, True, 256 * MiB)
    if bar_type:
        host = [_al(n) for n, u, _ in bufs if (u & MAP_WRITE) and not (u & MAP_READ)]
        total += max(_pack(host, 64 * MiB, True, 128 * MiB), host_term)
    return total


def main():
    d, p = sys.argv[1], sys.argv[2]
    Ls = loads(d, p)
    # A DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT memory type (the BAR heap, or VRAM under ReBAR) takes wgpu's CpuToGpu staging into
    # VRAM; gpu-allocator keeps a type's last general block, so at least one 64 MiB host block stays open after the uploads.
    vk = open(os.path.join(d, f"{p}-vulkaninfo-memory.txt")).read()
    bar_type = any("MEMORY_PROPERTY_DEVICE_LOCAL_BIT" in t and "MEMORY_PROPERTY_HOST_VISIBLE_BIT" in t and "MEMORY_PROPERTY_HOST_COHERENT_BIT" in t
                   for t in re.split(r"memoryTypes\[\d+\]", vk)[1:])
    print(f"vulkaninfo: a device-local host-visible coherent memory type {'exists' if bar_type else 'does not exist'}")
    by = {L["tag"]: L for L in Ls}
    # M2: a + b * buffers, fitted on the 0.5B and the Qwen 7B, judged on the other four.
    f1, f2 = by["qwen0.5b"], by["qwen7b"]
    r1, r2 = f1["rise"] - f1["ledger"], f2["rise"] - f2["ledger"]
    n1, n2 = len(f1["bufs"]), len(f2["bufs"])
    b = (r2 - r1) / (n2 - n1) if n2 != n1 else 0.0
    a = r1 - b * n1
    print(f"M2 fit: a={a / MiB:.1f} MiB, b={b / 1024:.1f} KiB per buffer (on qwen0.5b, qwen7b)")
    print(f"{'load':18} {'buffers':>7} {'rise MiB':>9} {'M0 err':>8} {'M1 err':>8} {'M2 err':>8} {'M1-nohost':>9}")
    verdict = {"M0": True, "M1": True, "M2": True}
    for L in Ls:
        e0 = (m0(L) - L["rise"]) / MiB
        try:
            e1 = (m1_predict(L["bufs"], bar_type, 64 * MiB) - L["rise"]) / MiB
        except NotImplementedError:
            e1 = float("nan")
        e1n = (m1_predict(L["bufs"], bar_type, 0) - L["rise"]) / MiB  # reported, not graded: M1 without the kept host block
        e2 = (L["ledger"] + a + b * len(L["bufs"]) - L["rise"]) / MiB
        held = L["tag"] not in ("qwen0.5b", "qwen7b")
        verdict["M0"] &= abs(e0) <= TOL_MIB
        verdict["M1"] &= abs(e1) <= TOL_MIB
        if held:
            verdict["M2"] &= abs(e2) <= TOL_MIB
        print(f"{L['tag']:18} {len(L['bufs']):7d} {L['rise'] / MiB:9.0f} {e0:8.1f} {e1:8.1f} {e2:8.1f} {e1n:9.1f}{'' if held else '  (M2 fit)'}")
    for k, v in verdict.items():
        print(f"{k}: {'HOLDS' if v else 'does not hold'} (within {TOL_MIB} MiB on every load it is judged on)")


if __name__ == "__main__":
    main()
