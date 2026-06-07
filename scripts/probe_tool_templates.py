#!/usr/bin/env python3
"""Probe each family's tool-calling template: how tools are declared, how the
model emits a tool *call*, and how a tool *result* is fed back. Renders a fixed
conversation (with tools + an assistant tool_call + a tool result) through HF
apply_chat_template so we can design goinfer's per-family render + parser.

    ~/g4venv/bin/python scripts/probe_tool_templates.py
"""
from transformers import AutoTokenizer

FAMILIES = {
    "chatml(qwen)": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma4": "google/gemma-4-12B-it-qat-q4_0-unquantized",
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}]

# A conversation that exercises declare → call → result → final.
CONV = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather",
         "arguments": {"location": "Paris", "unit": "celsius"}}}]},
    {"role": "tool", "name": "get_weather", "content": "18C, sunny"},
]

for fam, repo in FAMILIES.items():
    print("\n" + "=" * 70 + f"\n{fam}  ({repo})\n" + "=" * 70)
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        print("SKIP:", e)
        continue
    for label, conv in [("declare+gen", CONV[:1]), ("full call+result", CONV)]:
        try:
            s = tok.apply_chat_template(conv, tools=TOOLS, tokenize=False, add_generation_prompt=True)
            print(f"\n--- {label} ---\n{s!r}")
        except Exception as e:
            print(f"\n--- {label}: ERROR {e}")
