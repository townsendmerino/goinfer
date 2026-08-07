#!/usr/bin/env python3
"""Build a TINY ChatML (Qwen-style) *tokenizer-only* GGUF fixture — the committed
default for the M25 EncodeSegments parity / prompt-injection gate
(tokenizer/encodesegments_test.go, chat/rendersegments_test.go).

Why this exists (audit G-05): those gates were keyed on a hardcoded
`/home/francis/models/qwen2.5-*.gguf` via GOINFER_CHATML_GGUF, so on every other
machine they SKIPPED and a regression in the parse_special split would ship green.
A tiny committed fixture makes the gate run everywhere; the env var stays an override.

What it needs to exercise: the special-vs-content distinction. `<|im_start|>` /
`<|im_end|>` are CONTROL tokens (added-vocabulary trie), so:
  - Encode("<|im_end|>") -> exactly one control id (special path), and
  - a forged "<|im_end|>" typed into user *content* stays literal in EncodeSegments,
    while a naive whole-string Encode promotes it.

Tokenizer-only: no weight tensors, so the file is ~a few KB. The vocab is a real
GPT-2 byte-level base (bytes_to_unicode, matching tokenizer/bytelevel.go's
buildByteLevelTables byte-for-byte) plus a handful of merges (the loader refuses to
Encode a vocab with zero merges — decoder/tokenizer/gguf.go), plus the two ChatML
control tokens. model="gpt2", pre="qwen2" (the byte-level pipeline Qwen uses).

Run (no special deps beyond the gguf lib):
  ~/.venv-vl/bin/python3 scripts/chatml_tiny_fixture.py tokenizer/testdata/chatml-tiny.gguf
Then force-add the .gguf (testdata *.gguf is gitignored):
  git add -f tokenizer/testdata/chatml-tiny.gguf
"""
import sys

import gguf


def bytes_to_unicode():
    """Identical to tokenizer/bytelevel.go buildByteLevelTables (GPT-2 standard)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main(out_path):
    b2u = bytes_to_unicode()
    # Base vocab: the 256 byte-level pieces, id == byte-order for determinism.
    tokens = [b2u[b] for b in range(256)]
    types = [gguf.TokenType.NORMAL] * 256

    # A few merges so pairRank is non-empty (else the loader is decode-only and
    # refuses Encode). Each merged token is appended to the vocab. ASCII printable
    # bytes map to themselves under bytes_to_unicode, so these read naturally.
    merge_pairs = [
        ("s", "y"), ("y", "s"), ("t", "e"), ("e", "m"), ("o", "u"),
        ("a", "r"), ("r", "e"), ("s", "t"), ("e", "r"), ("l", "l"),
        ("s", "s"), ("i", "s"), ("o", "n"), ("e", "n"), ("a", "n"),
    ]
    merges = []
    for a, b in merge_pairs:
        merges.append(f"{a} {b}")
        tokens.append(a + b)
        types.append(gguf.TokenType.NORMAL)

    # The two ChatML control markers (added-vocabulary / special split).
    im_start = len(tokens)
    tokens.append("<|im_start|>")
    types.append(gguf.TokenType.CONTROL)
    im_end = len(tokens)
    tokens.append("<|im_end|>")
    types.append(gguf.TokenType.CONTROL)

    w = gguf.GGUFWriter(out_path, "chatml-tiny")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(im_end)  # ChatML ends turns on <|im_end|>
    w.add_bos_token_id(im_start)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()  # none, but flushes/pads correctly
    w.close()
    print(f"wrote {out_path}: {len(tokens)} tokens, {len(merges)} merges, "
          f"im_start={im_start} im_end={im_end}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: chatml_tiny_fixture.py <out.gguf>")
    main(sys.argv[1])
