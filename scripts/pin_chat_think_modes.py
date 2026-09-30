#!/usr/bin/env python
"""Golden for chat/reasoning.go: what each real chat template renders for every thinking setting.

    ~/g4venv/bin/python scripts/pin_chat_think_modes.py ~/models testdata/chat_think_goldens/think_modes.json

For each checkpoint it stores the checkpoint's FULL chat_template string (so chat.Detect runs on the real text, not a
paraphrase) and HuggingFace's apply_chat_template(add_generation_prompt=True) output, plus its token ids, for
enable_thinking unset / False / True over a few conversations. Qwen's own tool syntax (XML) is not goinfer's Hermes JSON,
so tools are pinned only for Gemma 4, whose tool rendering is byte-exact by design.
The checkpoints are the ones whose templates differ in DEFAULT, which is the whole point: Qwen3 (default on, nothing
written), Qwen3.5 0.8B (default off, closed block), Qwen3.5 9B (default on, open block), Gemma 4 (default off).
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

CKPTS = ["qwen3-4b", "qwen3.5-0.8b", "qwen3.5-9b", "gemma-4-26b-a4b-it"]
CONVS = {
    "user": [{"role": "user", "content": "Hi"}],
    "sys_user": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    "multi": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
              {"role": "user", "content": "And?"}],
}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
MODES = {"unset": {}, "false": {"enable_thinking": False}, "true": {"enable_thinking": True}}


def main():
    root, out = os.path.expanduser(sys.argv[1]), sys.argv[2]
    res = []
    for name in CKPTS:
        tok = AutoTokenizer.from_pretrained(os.path.join(root, name))
        entry = {"checkpoint": name, "chat_template": tok.chat_template, "cases": []}
        for cname, msgs in CONVS.items():
            for mname, kw in MODES.items():
                p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
                entry["cases"].append({"conv": cname, "mode": mname, "tools": False, "prompt": p,
                                       "ids": tok(p, add_special_tokens=False)["input_ids"]})
        if name.startswith("gemma-4"):
            for cname in ("user", "sys_user"):
                for mname, kw in MODES.items():
                    p = tok.apply_chat_template(CONVS[cname], tools=TOOLS, add_generation_prompt=True, tokenize=False, **kw)
                    entry["cases"].append({"conv": cname, "mode": mname, "tools": True, "prompt": p,
                                           "ids": tok(p, add_special_tokens=False)["input_ids"]})
        res.append(entry)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, sum(len(e["cases"]) for e in res), "cases")


if __name__ == "__main__":
    main()
