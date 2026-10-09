#!/usr/bin/env python3
"""Option D of the --embed-int4 decision (docs/tasks/task-multimodal-support-2026-10.md, "option D", registered before this code): the Hugging Face float32 side.
For each model: 32 fixed chat prompts through the checkpoint's own chat template, a greedy 32-token continuation (EOS suppressed so every prompt has 32 positions), and the float32
logits at those 32 positions, written for the Go test (decoder TestHeadPrecision_real) to teacher-force. A model that fails is recorded in its meta.json and the rest still run.
Usage: ~/g4venv/bin/python -I scripts/head_precision_hf.py <out root> [model name ...]   (models under ~/models)"""
import json, os, sys, time, traceback
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["qwen2.5-0.5b-instruct", "qwen3-1.7b-bf16", "qwen25vl-3b-instruct", "gemma-3-4b-it", "tinyllama-1.1b-chat", "phi3-mini-4k", "olmo3-7b-think"]
PROMPTS = [
    "Explain in two or three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number, with a short explanation.",
    "What is the capital of Australia, and what is one interesting fact about it?",
    "Summarize the plot of Romeo and Juliet in a short paragraph.",
    "Give me three tips for learning a new language quickly.",
    "Translate 'Where is the nearest train station?' into French, Spanish and German.",
    "If a train travels 120 km in 1.5 hours, what is its average speed? Show the calculation.",
    "List four differences between a process and a thread in an operating system.",
    "Write a haiku about autumn rain.",
    "Describe how a binary search works and state its time complexity.",
    "What are the main causes of inflation? Answer in a short paragraph.",
    "Write a polite email asking a colleague to reschedule a meeting to next Tuesday.",
    "Why do we have leap years? Explain briefly.",
    "Give a short recipe for a vegetable omelette.",
    "What is the difference between a virus and a bacterium?",
    "Explain what a hash table is and when you would use one.",
    "Write a short story opening about a lighthouse keeper who finds a message in a bottle.",
    "What is the difference between weather and climate?",
    "Explain how photosynthesis works in simple terms.",
    "Write a SQL query that returns the three highest-paid employees from a table named employees.",
    "Give two arguments for and two against remote work.",
    "How do vaccines train the immune system? Answer briefly.",
    "Convert 72 degrees Fahrenheit to Celsius and show the formula.",
    "Suggest a name for a small bakery and explain the choice in one sentence.",
    "What is recursion? Give a simple example in JavaScript.",
    "Describe the water cycle in four short steps.",
    "Why is the Roman Empire often said to have fallen? Give a brief answer.",
    "Write a limerick about a cat who loves keyboards.",
    "Explain the difference between TCP and UDP.",
    "What are three good habits for a healthy sleep schedule?",
    "Rewrite this sentence to be more formal: 'Hey, can you send me that file asap?'",
    "Explain what a prime number is and list the first eight.",
]
STEPS = 32
out_root = sys.argv[1]
names = sys.argv[2:] or MODELS
home = os.path.expanduser("~")
T0 = time.time()
def hb(msg): print(f"[head-precision hf {time.time() - T0:6.0f}s] {msg}", flush=True)

for name in names:
    d = f"{home}/models/{name}"; od = f"{out_root}/{name}"; os.makedirs(od, exist_ok=True)
    try:
        tok = AutoTokenizer.from_pretrained(d)
        hb(f"{name}: loading float32")
        model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32); model.eval()
        prompts, conts, p1 = [], [], []
        with open(f"{od}/logits.f32", "wb") as f:
            for i, p in enumerate(PROMPTS):
                ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
                if not isinstance(ids, list): ids = ids["input_ids"]
                if ids and isinstance(ids[0], list): ids = ids[0]
                x = torch.tensor([ids])
                with torch.no_grad():
                    g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=STEPS, min_new_tokens=STEPS, do_sample=False)
                    cont = g[0, len(ids):].tolist()
                    assert len(cont) == STEPS, (len(cont), STEPS)
                    full = torch.tensor([ids + cont])
                    lg = model(input_ids=full).logits[0].float()[len(ids) - 1: len(ids) - 1 + STEPS]
                p1 += torch.softmax(lg, -1).max(-1).values.tolist()
                f.write(lg.numpy().astype("<f4").tobytes())
                prompts.append(ids); conts.append(cont)
                hb(f"{name}: prompt {i + 1}/{len(PROMPTS)} ({len(ids)} prompt tokens)")
        json.dump({"name": name, "dir": d, "vocab": int(lg.shape[1]), "steps": STEPS, "prompts": prompts, "conts": conts,
                   "hf_mean_top1_prob": float(np.mean(p1)), "transformers": __import__("transformers").__version__}, open(f"{od}/meta.json", "w"))
        del model
    except Exception as e:
        traceback.print_exc()
        json.dump({"name": name, "dir": d, "error": f"{type(e).__name__}: {e}"}, open(f"{od}/meta.json", "w"))
        hb(f"{name}: FAILED {type(e).__name__}: {e}")
hb("done")
