#!/usr/bin/env python3
"""Generate byte-exact chat-template goldens from HF apply_chat_template, one
file per family, for the goinfer `chat` package's renderer tests. Small tokenizer
configs only (no model weights).

    ~/g4venv/bin/python scripts/gen_chat_goldens.py
"""
import json, os, datetime
from transformers import AutoTokenizer

OUT = os.path.expanduser("~/mycode/goinfer/testdata/chat_goldens")
os.makedirs(OUT, exist_ok=True)

# Pin dynamic dates (Llama-3's template injects "Today Date: <today>") to a fixed
# value so the committed fixture is stable; the Go renderer's clock is pinned to
# the same date in the test.
PINNED_DATE = "01 Jan 2025"
TODAY = datetime.datetime.now().strftime("%d %b %Y")

# family -> an ungated HF repo whose chat_template represents it.
FAMILIES = {
    "gemma3": "unsloth/gemma-3-1b-it",
    "gemma4": "google/gemma-4-12B-it-qat-q4_0-unquantized",
    "chatml": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

# N-22: pin each repo to the commit SHA the committed goldens were built from, so an upstream
# chat_template edit can't silently change a byte-exact fixture on regeneration. None = track the
# repo's main branch (the drift-prone default) — the loop warns loudly when a repo is unpinned.
REVISIONS = {repo: None for repo in FAMILIES.values()}

# Every case carries an EXPLICIT system message so no family injects its own
# default (e.g. Qwen's "You are Qwen…") — the renderer takes system from the
# caller. add_generation_prompt=True (we're building a prompt to continue).
SYS = "You are a terse assistant."
CASES = [
    ("sys_user", [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "What is the capital of France?"},
    ]),
    ("sys_multi", [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Name a primary color."},
    ]),
]

for fam, repo in FAMILIES.items():
    rev = REVISIONS.get(repo)
    if rev is None:
        print(f"WARNING {fam} ({repo}): unpinned revision — goldens may drift; set REVISIONS[{repo!r}] to a commit SHA")
    try:
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    except Exception as e:
        print(f"SKIP {fam} ({repo}): {e}")
        continue
    out = {"family": fam, "repo": repo, "revision": rev, "chat_template": (tok.chat_template or ""), "cases": []}
    for name, msgs in CASES:
        try:
            rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            rendered = rendered.replace(TODAY, PINNED_DATE) # stabilize Llama-3's dynamic date
            out["cases"].append({"name": name, "messages": msgs, "rendered": rendered})
        except Exception as e:
            out["cases"].append({"name": name, "messages": msgs, "error": str(e)})
    path = f"{OUT}/{fam}.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {path}  ({len(out['cases'])} cases, template {len(out['chat_template'])} chars)")
    for c in out["cases"]:
        if "rendered" in c:
            print(f"  --- {fam}/{c['name']} ---\n{c['rendered']!r}")
        else:
            print(f"  --- {fam}/{c['name']}: ERROR {c['error'][:80]}")
