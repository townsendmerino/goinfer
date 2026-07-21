#!/usr/bin/env python3
"""pin_qwen3_embedding.py — golden fixture for Qwen/Qwen3-Embedding-0.6B, the parity oracle for
goinfer's DECODER-as-embedder path (docs/task-decoder-as-embedder.md).

Qwen3-Embedding is a causal decoder used as an embedder: sentence-transformers wraps it as
`Transformer -> Pooling(last-token) -> Normalize`. goinfer reproduces that with
decoder.HiddenLast (the final hidden state of the last token, post-final-norm) plus an
instruction prefix on queries. This script dumps the reference so the Go side can be certified
against it instead of against its own output.

Conventions verified from the model's own config files (NOT by reputation — the task doc records
how that assumption already bit this project once):
  - config_sentence_transformers.json: query prompt = "Instruct: ...\\nQuery:", document = "".
  - 1_Pooling/config.json: pooling_mode_lasttoken=true, include_prompt=true, so the query prompt
    is part of the pooled input and is NOT stripped before pooling.
  - tokenizer_config.json: add_bos_token=false, add_eos_token absent, and the model card appends
    no EOS/EOD in either usage example -> the model eats exactly the (prompt+text) tokens.
  - there is no sentence_bert_config.json (404), so no truncation shorter than the tokenizer's
    model_max_length (131072) applies.

For a few short curated cases this dumps, from the real model:
  - input_ids : the token ids the model ate (prompt+text, no BOS/EOS). Fed directly to the Go
                forward, so tokenizer parity is out of scope for this gate.
  - embedding : the last-token-pooled, L2-normalized sentence embedding [1024].
Plus a small retrieval set (queries x documents) with the reference top-1 per query, because a
high cosine can coexist with wrong ranking.

Run from the repo root:
    .venv/bin/python scripts/pin_qwen3_embedding.py
"""
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
OUT = REPO_ROOT / "testdata" / "qwen3_embedding_golden.json"

# Short, varied cases. (is_query selects the Instruct prefix.)
CASES = [
    ("what is the capital of France", True),
    ("Paris is the capital and most populous city of France.", False),
    ("how do i parse json in go", True),
    ("Use encoding/json: json.Unmarshal(data, &v) decodes into v.", False),
    ("def add(a, b):\n    return a + b", False),
]

# Retrieval gate: each query's correct document is at the same index in DOCUMENTS.
QUERIES = [
    "what is the capital of France",
    "how do i parse json in go",
]
DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "Use encoding/json: json.Unmarshal(data, &v) decodes into v.",
    "The mitochondrion is the powerhouse of the cell.",
    "Rust's borrow checker enforces ownership at compile time.",
]


def main() -> None:
    model = SentenceTransformer(MODEL_ID)
    tok = model.tokenizer
    hidden = model.get_sentence_embedding_dimension()

    # The exact prompts sentence-transformers uses, read off the loaded model.
    prompts = getattr(model, "prompts", {}) or {}
    query_prompt = prompts.get("query", "")
    doc_prompt = prompts.get("document", "")

    def encode(text: str, is_query: bool):
        # prompt_name="query" makes ST prepend the Instruct preamble; documents get nothing.
        if is_query:
            vec = model.encode([text], prompt_name="query", normalize_embeddings=True)[0]
        else:
            vec = model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

    def ids_for(text: str, is_query: bool):
        prefix = query_prompt if is_query else doc_prompt
        return tok(prefix + text, add_special_tokens=False)["input_ids"]

    cases = []
    for text, is_query in CASES:
        cases.append(
            {
                "text": text,
                "is_query": is_query,
                "input_ids": ids_for(text, is_query),
                "embedding": encode(text, is_query),
            }
        )

    # Reference ranking: cosine over the L2-normalized vectors is a dot product.
    qvecs = model.encode(QUERIES, prompt_name="query", normalize_embeddings=True)
    dvecs = model.encode(DOCUMENTS, normalize_embeddings=True)
    sims = torch.tensor(qvecs) @ torch.tensor(dvecs).T
    top1 = [int(i) for i in sims.argmax(dim=1)]
    ranking = [[int(j) for j in row.argsort(descending=True)] for row in sims]

    out = {
        "note": f"{MODEL_ID}; sentence-transformers last-token pooling + L2 normalize",
        "model": MODEL_ID,
        "hidden": int(hidden),
        "query_prompt": query_prompt,
        "doc_prompt": doc_prompt,
        "cases": cases,
        "retrieval": {
            "queries": QUERIES,
            "documents": DOCUMENTS,
            "top1": top1,
            "ranking": ranking,
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases, hidden {hidden}, top1 {top1})")


if __name__ == "__main__":
    main()
