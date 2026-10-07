import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
src = os.path.expanduser("~/models/gemma-3-4b-it")
pfx = "vision_tower.vision_model."
cfg = json.load(open(src + "/config.json"))["vision_config"]
f = safe_open(src + "/model-00001-of-00002.safetensors", "pt")
keys = [k for k in f.keys() if k.startswith(pfx)]
for n in map(int, sys.argv[1:]):
    t = {}
    for k in keys:
        s = k[len(pfx):]
        if s.startswith("encoder.layers.") and int(s.split(".")[2]) >= n:
            continue
        t[s] = f.get_tensor(k).float().contiguous()
    out = f"real-L{n}"
    os.makedirs(out, exist_ok=True)
    save_file(t, out + "/model.safetensors")
    json.dump({**cfg, "num_hidden_layers": n}, open(out + "/config.json", "w"))
    print(out, len(t))
