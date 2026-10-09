import torch, time, sys
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
d = sys.argv[1]
proc = AutoProcessor.from_pretrained(d)
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(d, dtype=torch.float32).eval()
a, b = Image.open("/home/francis/s11pin/table.png").convert("RGB"), Image.open("/home/francis/s11pin/formula.png").convert("RGB")
msgs = [{"role": "user", "content": [{"type": "text", "text": "Here are two images. "}, {"type": "image"}, {"type": "text", "text": " and "}, {"type": "image"},
        {"type": "text", "text": " What does each image show? Answer briefly, one line per image."}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inp = proc(text=[text], images=[a, b], return_tensors="pt")
print("prompt tokens", inp["input_ids"].shape[1], "grids", inp["image_grid_thw"].tolist(), flush=True)
t = time.time()
with torch.no_grad():
    out = m.generate(**inp, max_new_tokens=96, do_sample=False)
print("%.0fs" % (time.time() - t), repr(proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)))
