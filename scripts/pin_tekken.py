#!/usr/bin/env python3
"""Golden for tokenizer/tekken.go (G-S14e1 of docs/tasks/task-multimodal-support-2026-10.md): Mistral's Tekken tokenizer against mistral_common itself.

    ~/goinfer-bench/s14e/venv/bin/python -W ignore -I scripts/pin_tekken.py ~/models/voxtral-mini-3b-2507/tekken.json testdata/tekken_golden.json.gz

mistral_common 1.12.0 is the reference (it is what transformers' VoxtralProcessor.apply_transcription_request calls). Two parts:
  strings:  plain-text ids from the tokenizer's own encode (no BOS/EOS; a special token written in the text is ordinary text) and the text decode(ids) gives back
  requests: the full transcription-request ids (encode_transcription) for 1, 2 and 3 thirty-second chunks, with and without a language
"""
import gzip
import hashlib
import json
import sys

import numpy as np
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

STRINGS = [
    "", " ", "a", "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading and trailing  ", "tabs\tand\nnewlines\r\nand\n\n\nrunsofnewlines",
    "don't can't it's we're they've I'm you'll he'd DON'T", "CamelCaseWordsAndHTTPServerIDs", "snake_case_names and kebab-case-names", "x = 3 + 4 * (5 - 2) / 7 ** 2",
    "def f(a, b):\n    return a + b  # adds\n", "for (int i = 0; i < n; i++) { sum += a[i]; }", "https://example.com/a/b/c?x=1&y=2#frag", "path/to/some/file.txt and C:\\Windows\\System32",
    "2026", "1234567890", "3.14159 and 1,000,000 and -42 and 0x1F", "١٢٣ ४५६ ๗๘๙ (digits from other scripts)", "年 2026 年 10 月 9 日",
    "Bonjour le monde, ça va très bien aujourd'hui ?", "Straße, Fußgängerübergang, Müller, Köln", "Привет, мир! Как дела?", "你好，世界！今天天气很好。", "こんにちは世界。今日は良い天気です。",
    "안녕하세요 세계", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก", "Γειά σου Κόσμε", "Xin chào thế giới",
    "emoji 😀 and 🎉🎉 and 👨‍👩‍👧‍👦 and 🇫🇷", "cafe\u0301 (decomposed) vs café (precomposed)", "a\u0300\u0301\u0302 stacked marks", "\u200b zero width \u200d joiners \ufeff bom",
    "[INST] this is plain text, not a control token [/INST]", "<s>not the BOS</s>", "[AUDIO][BEGIN_AUDIO][TRANSCRIBE] written out", "<unk> <pad> [TOOL_CALLS] [IMG]",
    "the " * 40, "ab" * 100, "z" * 300, "!" * 20 + "?" * 20, "...---===___", "a  b   c    d     e", "   ", "\n\n", "\n", " \n \n ", "end.   ",
    "It's 5 o'clock; the CEO's iPhone 15 Pro costs $1,199.99!", "SELECT * FROM users WHERE id = 42 AND name LIKE '%x%';", "{\"key\": [1, 2, {\"nested\": true}], \"s\": \"v\"}",
    "<html><body class=\"x\">hi</body></html>", "# Heading\n\n- item one\n- item two\n\n```python\nprint('x')\n```\n", "lang:en", "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
    "Ünïcödé wörds wïth dïàcrïtïcs", "ALL CAPS WORDS AND MiXeD cAsE", "ǅ ǈ ǋ titlecase digraphs", "ʰ ʲ modifier letters ꙮ ꝰ",
]
# A seeded differential-fuzz set: glued fragments (a contraction directly after a word, digits against letters, slashes, marks) that natural prose never produces and on which the
# o200k and Tekken pre-tokenizers give different ids (found 2026-10-09: 68 of 4,000 such strings differ). Pinned from mistral_common like the rest.
import random
_BITS = ["don't", "it's", "we're", "I'll", "2026", "12345", "1,000", "a/b", "http://x", "/", "x", "Hello", "WORLD", " ", "\n", "  ", "'", "3.14", "He'd", "THEY'RE", "abc123", "123abc", "\u0301", "\u00e9", "-", "--"]
_rng = random.Random(20261009)
for _ in range(400):
    STRINGS.append("".join(_rng.choice(_BITS) for _ in range(1 + _rng.randrange(6))))
STRINGS.append(("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 30)[:3000])


def main():
    path, out = sys.argv[1], sys.argv[2]
    raw = open(path, "rb").read()
    tok = MistralTokenizer.from_file(path)
    inner = tok.instruct_tokenizer.tokenizer
    rec = {"tekken_sha256": hashlib.sha256(raw).hexdigest(), "mistral_common": __import__("mistral_common").__version__, "n_vocab": inner.n_words,
           "bos": inner.bos_id, "eos": inner.eos_id, "strings": [], "requests": []}
    for s in STRINGS:
        ids = inner.encode(s, bos=False, eos=False)
        back = inner.decode(ids)
        rec["strings"].append({"text": s, "ids": [int(i) for i in ids], "decoded": back})
    sr = 16000
    for seconds in (2.0, 35.0, 65.0):
        x = (np.sin(np.arange(int(seconds * sr)) * 0.05) * 0.1).astype(np.float32)
        raw_audio = RawAudio.from_audio(Audio(audio_array=x, sampling_rate=sr, format="wav"))
        for lang in ("en", "fr", None):
            o = tok.encode_transcription(TranscriptionRequest(model="voxtral", audio=raw_audio, language=lang))
            rec["requests"].append({"seconds": seconds, "language": lang, "ids": [int(i) for i in o.tokens], "audio_samples": [int(a.audio_array.shape[0]) for a in o.audios]})
    with gzip.open(out, "wt") as f:
        json.dump(rec, f)
    print("wrote", out, len(rec["strings"]), "strings,", len(rec["requests"]), "requests; n_vocab", rec["n_vocab"], "bos", rec["bos"], "eos", rec["eos"])


if __name__ == "__main__":
    main()
