#!/usr/bin/env python3
"""KV2's CUDA planted defects (docs/measurements/kv2-cuda-2026-10-15.md): each plant is a text edit of a production file,
applied, its gate test run, and the file restored. A plant passes when its test goes red. Run from the repository root
under the timing lock:

  python3 scripts/timing_lock.py run --label kv2-cuda-plants -- python3 docs/measurements/kv2-cuda-2026-10-15/plant.py [name ...]

It prints each plant's verdict and the test's failure lines, and exits 1 if any plant stayed green."""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
COPY = "cuda/kvcopy.go"

PLANTS = [
    # name, file, old, new, test regexp
    ("short", COPY, "srcOff, dstOff, rows := start*row, start*row, end-start",
     "srcOff, dstOff, rows := start*row, start*row, max(1, end-start-64)", "^TestKV2_copyExact_tiny$"),
    ("layer-stride", COPY, "{{from.kc[l], to.kc[l]}, {from.vc[l], to.vc[l]}}",
     "{{from.kc[(l+1)%len(r.layers)], to.kc[l]}, {from.vc[(l+1)%len(r.layers)], to.vc[l]}}", "^TestKV2_copyExact_tiny$"),
    ("f16-as-f32", COPY, "elem := 4\n\t\tif Ly.kvHalf {\n\t\t\telem = 2\n\t\t}\n\t\trow := Ly.kvDim * elem // bytes",
     "elem := 4\n\t\trow := Ly.kvDim * elem // bytes", "^TestKV2_copyExact_tiny$"),
    ("window-base-not-set", COPY, "\t\tr.kvBases[dst] = lo\n\t\tif dst == r.kvSlot {\n\t\t\tr.kvBase = lo\n\t\t}\n",
     "\t\t_ = lo\n", "^TestKV2_copyWindowed$"),
    ("window-unshifted-read", COPY, "srcOff, dstOff, rows = (lo-r.kvBases[src])*row, 0, end-lo",
     "srcOff, dstOff, rows = lo*row, 0, end-lo", "^TestKV2_copyWindowed$"),
    ("window-check-removed", COPY, "if !kvWindowReusable(r.kvBases[src], end, r.kvWindow) {",
     "if false && !kvWindowReusable(r.kvBases[src], end, r.kvWindow) {", "^TestKV2_copyWindowed$"),
    ("chunker-short-tail", "cuda/prefill.go", "if rem := M - (i + n); rem > 0 && rem < attnFusedMinRows {",
     "if rem := M - (i + n); false && rem > 0 && rem < attnFusedMinRows {", "^TestKV2_recordedClass_tiny$"),
    ("copy-does-nothing", COPY, "\treturn r.do(func() error { return r.copyKVSlots(src, dst, start, end) })",
     "\treturn nil", "^TestKV2_scheduleIndependence_tiny$"),
    ("hit-one-block-long", "decoder/resident_reuse.go", "err := cp.CopyKV(src, dst, start, hit)",
     "hit = min(hit+64, len(prompt)-16)\n\terr := cp.CopyKV(src, dst, start, hit)", "^TestKV2_scheduleIndependence_tiny$"),
    ("cap-at-n-1", "internal/kvmgr/share.go", "\tcold := c.Pass(n, n)\n", "\treturn hit\n\tcold := c.Pass(n, n)\n",
     "^TestKV2_hitCap_tiny$"),
    ("passat-ignores-windowed", "internal/kvmgr/key.go", "if cls == ClassFast && c.Windowed && start%BlockRows != 0 {",
     "if false && cls == ClassFast && c.Windowed && start%BlockRows != 0 {", "^TestKV2_windowedUnalignedNotShared$"),
    ("staging-buffer-kept", COPY, "\t\tfor _, p := range [2][2]Buffer{",
     "\t\t_ = r.af(rows * row / 4)\n\t\tfor _, p := range [2][2]Buffer{", "^TestKV2_copyAllocatesNothing_tiny$"),
]

def run(name, path, old, new, test):
    f = ROOT / path
    orig = f.read_text()
    if orig.count(old) != 1:
        return f"{name}: VOID, the edit does not match {path} exactly once", False
    f.write_text(orig.replace(old, new))
    try:
        p = subprocess.run(["go", "test", "-count=1", "-timeout", "10m", "-tags", "cuda goinfer_testhooks", "-run", test, "-v", "."],
                           cwd=ROOT / "cuda", capture_output=True, text=True)
    finally:
        f.write_text(orig)
    out = p.stdout + p.stderr
    lines = [l for l in out.splitlines() if l.startswith(("--- FAIL", "--- PASS", "--- SKIP", "FAIL", "ok", "panic")) or "_test.go:" in l]
    red = p.returncode != 0 and "--- FAIL" in out
    head = f"== {name} ({path}; {test}): {'RED' if red else 'STAYED GREEN'}"
    return "\n".join([head] + ["    " + l for l in lines[:25]]), red

names = sys.argv[1:]
ok = True
for pl in PLANTS:
    if names and pl[0] not in names:
        continue
    text, red = run(*pl)
    print(text, flush=True)
    ok = ok and red
sys.exit(0 if ok else 1)
