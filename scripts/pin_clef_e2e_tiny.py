#!/usr/bin/env python3
"""D12/D13: a TINY end-to-end fixture for internal/clef (docs/measurements/decisions-d12-clef-encoder-2026-10-02.md).

Cloudflare's own code (joint_schema_model.py from the pinned release directory, sha256 checked, imported unmodified) chained the way ClefModel.forward chains it,
at f32, on a tiny backbone and a tiny head, so one Go test can run the WHOLE pipeline (encoder -> backbone -> head) against a reference on any checkout, with no
19 GB asset:

  backbone   decoder/testdata/qwen3_5-tiny-normw  (committed; hidden 64, vocab 256, UNTIED lm_head, random final-norm weight so a missing or doubled final norm shows)
  head       JointSchemaHead(hidden_size=64, width=32, routing_layers=2, layers=2, heads=4, feedforward=64), seeded random weights in every parameter (the
             release's init leaves the three scales at 0 and the norms at identity, which would let whole paths drop out unseen)
  tokenizer  a stand-in: each fragment's UTF-8 bytes are cut into chunks of 4 and each chunk becomes one id, v = fold(v*131 + byte) mod 256 (the same few lines
             in the Go test), so the prompt TEXT is what is gated, the ids fit the backbone's vocabulary (256) and its 512 positions; the real tokenizer is gated
             by internal/clef's TestEncode_realTokenizer
  requests   one per question type and one with three questions (self-attention across questions), kept under the backbone's 512 positions

    python3 scripts/pin_clef_e2e_tiny.py     ->  internal/clef/testdata/tiny/{joint_head.safetensors, joint_head_config.json, golden.json}

CLEF_MODEL is the pinned release directory (default /srv/models/clef-flash; offline fixture generation, not a timed measurement). The weights are seeded, so a
re-run reproduces them byte for byte (checked by the script writing twice into memory and comparing).
"""
import hashlib, importlib.util, json, os, sys
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
MODEL = os.path.expanduser(os.environ.get("CLEF_MODEL", "/srv/models/clef-flash"))
MODULE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"   # joint_schema_model.py at the pinned revision (scripts/pin_clef_d10.py)
OUT = os.path.join(REPO, "internal", "clef", "testdata", "tiny")
BACKBONE = os.path.join(REPO, "decoder", "testdata", "qwen3_5-tiny-normw")
CFG = dict(hidden_size=64, width=32, routing_layers=2, layers=2, heads=4, feedforward=64)

src = os.path.join(MODEL, "joint_schema_model.py")
got = hashlib.sha256(open(src, "rb").read()).hexdigest()
if got != MODULE_SHA256:
    sys.exit(f"{src}: sha256 {got} is not the pinned {MODULE_SHA256}")
spec = importlib.util.spec_from_file_location("jsm", src); jsm = importlib.util.module_from_spec(spec); sys.modules["jsm"] = jsm; spec.loader.exec_module(jsm)


def make_head():
    torch.manual_seed(20261002)
    head = jsm.JointSchemaHead(**CFG)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if name.endswith("norm.weight") or ".norm" in name and name.endswith(".weight") or name.endswith("_norm.weight"):
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))
            elif p.ndim == 0:
                p.copy_(torch.tensor({"prior_logit_scale": 1.2, "joint_logit_scale": 1.5, "residual_gate": 0.7}[name]))
            elif name.endswith(".bias"):
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / p.shape[-1] ** 0.5) * 1.5)
    return head.float().eval()


head = make_head()
sd = {k: v.detach().clone().contiguous() for k, v in head.state_dict().items()}
assert all(torch.equal(sd[k], v) for k, v in make_head().state_dict().items()), "the seeded head is not reproducible"

def chunk_ids(text):             # keep in step with chunkTokenize in internal/clef/pipeline_test.go
    b, out = text.encode("utf-8"), []
    for i in range(0, len(b), 4):
        v = 0
        for c in b[i:i + 4]:
            v = (v * 131 + c) % 256
        out.append(v)
    return out

class Tok:                       # tokenizer(text, add_special_tokens=False).input_ids
    def __call__(self, text, add_special_tokens=False):
        class R: pass
        r = R(); r.input_ids = chunk_ids(text); return r
tok = Tok()

lm = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32).eval()
assert not lm.config.tie_word_embeddings, "the fixture must have an UNTIED lm_head, or the embedding-vs-lm_head mistake is invisible"
assert not torch.equal(lm.lm_head.weight, lm.get_input_embeddings().weight)

REQUESTS = {
    "noul": {"state": "Customer asks for a refund of $12.50.", "questions": {"q": {"type": "noul", "instructions": "Refund?"}}},
    "choice": {"state": {"mood": "calm", "n": 3}, "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"red": "r", "blue": None, "amber": "a"}}}},
    "score": {"state": "ok", "questions": {"q": {"type": "score", "instructions": "Risk 0-3", "criteria": ["none", "low", "mid", "high"]}}},
    "multi": {"state": {"t": "x", "v": [1, 2.5]}, "questions": {
        "a": {"type": "noul", "instructions": "A?", "criteria": {"true": "yes"}},
        "b": {"type": "choice", "instructions": "B", "criteria": {"z": "zed", "m": None, "e": "eh"}},
        "c": {"type": "score", "criteria": ["lo", "hi"]}}},
}
items = []
for name, req in REQUESTS.items():
    rec = jsm.encode_record(tok, dict(req, id=name))
    ids = list(rec.input_ids)
    assert len(ids) <= 500 and max(ids) < 256, (name, len(ids))
    x = torch.tensor([ids])
    with torch.no_grad():
        hidden = lm.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True).last_hidden_state
        logits = head(hidden, x, torch.ones_like(x), [rec], lm.get_output_embeddings().weight)[0]
    probs = [torch.softmax(l.float(), -1).tolist() for l in logits]
    spread = max(max(p) - min(p) for p in probs)
    assert spread > 0.05, (name, "probabilities are near-uniform: the fixture would not see a wrong path", probs)
    items.append(dict(name=name, request=req, input_ids=ids, n_tokens=len(ids), logits=[l.tolist() for l in logits], probs=probs,
                      questions=[dict(id=q.question_id, type=q.question_type, option_ids=list(q.option_ids), question_span=list(q.question_span),
                                      option_spans=[list(s) for s in q.option_spans]) for q in rec.questions]))
    print(f"{name:6} {len(ids):3} tokens, questions {[(q.question_id, q.question_type, len(q.option_ids)) for q in rec.questions]}, max prob {[round(max(p), 3) for p in probs]}")

os.makedirs(OUT, exist_ok=True)
save_file(sd, os.path.join(OUT, "joint_head.safetensors"))
json.dump(CFG, open(os.path.join(OUT, "joint_head_config.json"), "w"), indent=1)
json.dump(dict(module_sha256=MODULE_SHA256, backbone="decoder/testdata/qwen3_5-tiny-normw", torch=torch.__version__, transformers=__import__("transformers").__version__,
               head=CFG, items=items), open(os.path.join(OUT, "golden.json"), "w"))
print("wrote", OUT, os.path.getsize(os.path.join(OUT, "joint_head.safetensors")), "bytes of head weights")
