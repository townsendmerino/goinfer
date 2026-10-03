#!/usr/bin/env python3
"""Pin HF's tokenization of GLM-OCR strings as a goinfer tokenizer golden (O1,
docs/tasks/task-glm-ocr-2026-10.md): tokenizer/testdata/glm_ocr_tokenizer_golden.json.

Reads the REAL tokenizer (zai-org/GLM-OCR tokenizer.json, ~6.8 MB, not committed) from
$GOINFER_MODELS_DIR/glm-ocr (default ~/models/glm-ocr) with transformers' AutoTokenizer and records
the ids for: digit runs of 1..5 digits (the Split regex caps digits at 3 per piece), newlines and
newline runs, leading/trailing/multiple spaces, CJK and mixed text, contractions, and EVERY added
token 59246..59281 alone and embedded in text — including the ones with special=false in
tokenizer.json (<think>, </think>, the tool tags, <|begin_of_video|>, the box tags, <|image|>,
<|video|>). add_special_tokens=False throughout: this pins the text->ids function, not a template.

Run:  ~/.venv-vl/bin/python scripts/pin_glm_ocr_tokenizer.py
"""
import json
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tokenizer", "testdata", "glm_ocr_tokenizer_golden.json")
MODELS = os.environ.get("GOINFER_MODELS_DIR") or os.path.expanduser("~/models")
DIR = os.path.join(MODELS, "glm-ocr")

tok = AutoTokenizer.from_pretrained(DIR)

# every added token, by id, straight from the tokenizer
added = sorted(
    ((t.content, i) for i, t in tok.added_tokens_decoder.items()), key=lambda x: x[1]
)
assert added[0][1] == 59246 and added[-1][1] == 59281 and len(added) == 36, added

TEXTS = [
    "",
    "Hello, world!",
    "1", "12", "123", "1234", "12345", "123456", "1234567",
    "Invoice No. 20240131-0042, total $1,234,567.89 (12.5%)",
    "x=1;y=22;z=333;w=4444;v=55555",
    "a\nb", "a\n\nb", "a\r\nb", "\n", "\n\n\n", " \n ", "line1\nline2\n",
    " leading", "trailing ", "  two leading", "two trailing  ", "a  b   c", " ", "  ", "\t", "a\tb",
    "don't I'll we've she'd they're it's THEY'RE I'M",
    "中文文本识别", "请按下列JSON格式输出图中信息:", "日本語のテキスト、数字123と英語mixed", "한국어 텍스트",
    "Text Recognition:", "Formula Recognition:", "Table Recognition:",
    "$$\\int_0^1 x^2 dx = \\frac{1}{3}$$", "| a | b |\n|---|---|\n| 1 | 2 |",
    "{\"name\": \"\", \"amount\": \"\", \"date\": \"\"}",
    "<html><body>x</body></html>", "<unk> <pad> </s> <s>", "emoji \U0001F600 and éè café",
    "The quick brown fox jumps over the lazy dog. " * 3,
]
# the chat template's real shape, rendered as text (O0's probe)
TEXTS.append("[gMASK]<sop><|user|>\n<|begin_of_image|>" + "<|image|>" * 5 + "<|end_of_image|>Text Recognition:<|assistant|>\n")
TEXTS.append("[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>Text Recognition:<|assistant|>\n<think></think>\n")
TEXTS.append("<|user|>\nhi<|assistant|>\n<think>reasoning 123</think>\nanswer<|endoftext|>")
TEXTS.append("<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call><tool_response>r</tool_response>")
TEXTS.append("<|begin_of_box|>1234<|end_of_box|><|begin_of_video|><|video|><|end_of_video|>")
for content, _ in added:  # each added token alone, flanked, and doubled
    TEXTS += [content, "a" + content + "b", " " + content + " ", content + content, "12" + content + "345\n" + content]

cases = []
for s in TEXTS:
    ids = tok.encode(s, add_special_tokens=False)
    cases.append({"text": s, "ids": ids})

golden = {
    "tokenizer": "zai-org/GLM-OCR tokenizer.json (revision 2e85a62840ccac27daa451df36c736c4636b8628)",
    "transformers": __import__("transformers").__version__,
    "added": [{"content": c, "id": i} for c, i in added],
    "cases": cases,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(golden, f, ensure_ascii=False, indent=0)
print("wrote", os.path.relpath(OUT), len(cases), "cases,", len(added), "added tokens")
