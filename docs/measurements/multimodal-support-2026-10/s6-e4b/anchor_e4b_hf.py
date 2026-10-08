#!/usr/bin/env python3
"""HF float32 anchor for the Gemma 4 E4B G3 disagreement (docs/tasks/task-multimodal-support-2026-10.md, "S6 on nobara").

Reads the dump written by cuda's TestGemma4EModel_realE4BAnchorDump (the CPU's greedy sequence on each of G3's prompts, with goinfer's CPU and CUDA top-8 at every position), teacher-forces the SAME ids through
transformers in float32, and at every position where goinfer's CPU and CUDA argmax differ says which of them HF sides with. Also prints each arm's overall argmax agreement with HF.

    ~/g4venv/bin/python scripts/anchor_e4b_hf.py [dump.json] [model dir]
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM

dump = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/goinfer-logs/e4b-anchor/dump.json")
mdir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/gemma-4-E4B-it")
data = json.load(open(dump))
torch.set_num_threads(os.cpu_count())
print("loading", mdir, flush=True)
model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32)
model.eval()
tot = dict(pos=0, cpu_eq_cuda=0, hf_cpu=0, hf_cuda=0)
dis = dict(n=0, hf_with_cpu=0, hf_with_cuda=0, hf_with_neither=0)
for pi, d in enumerate(data):
    ids = torch.tensor([d["ids"]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()  # [n, vocab]
    n = len(d["cpu"])
    pa = {"cpu": 0, "cuda": 0, "cc": 0, "dis": 0}
    for i in range(n):
        hf_top = int(logits[i].argmax())
        c, g = d["cpu"][i]["ids"][0], d["cuda"][i]["ids"][0]
        tot["pos"] += 1
        tot["hf_cpu"] += hf_top == c
        tot["hf_cuda"] += hf_top == g
        tot["cpu_eq_cuda"] += c == g
        if c != g:
            dis["n"] += 1
            pa["dis"] += 1
            if hf_top == c: dis["hf_with_cpu"] += 1
            elif hf_top == g: dis["hf_with_cuda"] += 1
            else: dis["hf_with_neither"] += 1
    print(f"prompt {pi+1}: {n} positions, goinfer cpu!=cuda at {pa['dis']}", flush=True)
print("\nOVERALL over", tot["pos"], "positions")
print(f"  cpu==cuda          {tot['cpu_eq_cuda']}/{tot['pos']} = {100*tot['cpu_eq_cuda']/tot['pos']:.2f}%")
print(f"  hf==goinfer cpu    {tot['hf_cpu']}/{tot['pos']} = {100*tot['hf_cpu']/tot['pos']:.2f}%")
print(f"  hf==goinfer cuda   {tot['hf_cuda']}/{tot['pos']} = {100*tot['hf_cuda']/tot['pos']:.2f}%")
print(f"WHERE cpu!=cuda ({dis['n']} positions): HF sides with cpu {dis['hf_with_cpu']}, with cuda {dis['hf_with_cuda']}, with neither {dis['hf_with_neither']}")
