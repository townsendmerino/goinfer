# Usage: python3 scripts/inkling2/pin_tokenizer.py <tokenizer.json of thinkingmachines/Inkling-2-Mini> testdata/inkling2_tokenizer_golden.json
# The oracle is the release's own tokenizer.json through the tokenizers library (no special tokens added).
import json, sys
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1])
texts = [
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Hello, world! It's 2026; don't you think 1234567 is big?",
    "  leading spaces\tand\ttabs\n\n\nnewlines   ",
    "func search(xs []int, target int) int {\n\treturn -1\n}",
    "naïve café — Ünïcödé, 日本語のテキスト, 中文文本, emoji 🙂👍",
    "URL: https://example.com/a/b?c=1&d=2 // path/to/file.go",
    "CamelCaseIdentifier snake_case_name SCREAMING_CASE x2y3",
    "<|message_user|> is a special token in text",
]
cases = []
for s in texts:
    e = t.encode(s, add_special_tokens=False)
    cases.append({"text": s, "ids": e.ids, "tokens": e.tokens, "ids_bos": e.ids, "decode": t.decode(e.ids, skip_special_tokens=False), "decode_bos": t.decode(e.ids, skip_special_tokens=False)})
json.dump({"model_id": "thinkingmachines/Inkling-2-Mini",
           "note": "Byte-level BPE, o200k Split regex, no normalizer, ignore_merges=true (GGUF tokenizer.ggml.pre=inkling2). ids are without special tokens.",
           "cases": cases}, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(cases))
