#!/usr/bin/env python3
"""D12 (head half): a SYNTHETIC development golden for internal/clef's TestHead_matchesReference and TestEncode_multiQuestionMatchesReference.

Cloudflare's own JointSchemaHead (joint_schema_model.py, imported unmodified from the pinned release directory, f32) run on seeded random hidden states
and seeded random lm_head rows, for the shortest noul, choice and score fixture items plus one 4-question record encoded by the official encode_record.
It isolates the head from the backbone, so the Go head can be gated before the backbone's real last_hidden_state exists. It is NOT committed (about 8 MB of
incompressible floats) and is not the D12 gate of record: that is the same test on the D10 f32 fixture's real hidden states (the d10-clef-f32 night job).

  CLEF_MODEL   the pinned release dir (default /srv/models/clef-flash: offline fixture generation, not a timed measurement)
  D12_HEAD_OUT output directory (default ~/goinfer-bench/d12-head-dev); then run the Go tests with
               CLEF_HEAD_GOLDEN=$D12_HEAD_OUT CLEF_HEAD=$CLEF_MODEL CLEF_TOKENIZER=$CLEF_MODEL go test ./internal/clef/
Run from the repo root with a venv that has torch, transformers and safetensors.
"""
import json, sys, os, importlib.util, numpy as np, torch
from safetensors.torch import load_file
M=os.path.expanduser(os.environ.get("CLEF_MODEL","/srv/models/clef-flash")); OUT=os.path.expanduser(os.environ.get("D12_HEAD_OUT","~/goinfer-bench/d12-head-dev")); REPO=os.getcwd()
os.makedirs(OUT,exist_ok=True)
spec=importlib.util.spec_from_file_location("jsm", f"{M}/joint_schema_model.py"); jsm=importlib.util.module_from_spec(spec); sys.modules["jsm"]=jsm; spec.loader.exec_module(jsm)
cfg=json.load(open(f"{M}/joint_head_config.json")); head=jsm.JointSchemaHead(**cfg)
sd={k:v.float() for k,v in load_file(f"{M}/joint_head.safetensors").items()}
print(head.load_state_dict(sd, strict=True)); head=head.float().eval()
enc=[json.loads(l) for l in open(f"{REPO}/testdata/decisions/clef/encoder.jsonl")]
kinds={}
for e in enc:
    k=e["questions"][0]["type"]; kinds.setdefault(k,[]).append(e)
picks=[min(v,key=lambda e:e["n_tokens"]) for k,v in sorted(kinds.items())]   # shortest noul, choice, score
class Rows:                      # synthetic lm_head: row i is a seeded random vector
    def __getitem__(self, ids):
        out=[]
        for i in ids.tolist():
            g=torch.Generator().manual_seed(1000+i); out.append(torch.randn(4096,generator=g)*0.05)
        return torch.stack(out)
rows=Rows(); items=[]
for n,e in enumerate(picks):
    L=e["n_tokens"]; g=torch.Generator().manual_seed(77+n)
    hidden=torch.randn(1,L,4096,generator=g)*1.5
    ids=torch.tensor([e["input_ids"]])
    qs=tuple(jsm.EncodedQuestion(q["id"],q["type"],tuple(q["question_span"]),tuple(tuple(s) for s in q["option_spans"]),tuple(q["option_ids"])) for q in e["questions"])
    rec=jsm.EncodedRecord(tuple(e["input_ids"]),qs,e["id"])
    with torch.no_grad():
        logits=head(hidden, ids, torch.ones(1,L,dtype=torch.long), [rec], rows)[0]
    probs=[torch.softmax(l,0).tolist() for l in logits]
    tok=sorted({t for q in e["questions"] for s in q["option_spans"] for t in e["input_ids"][s[0]:s[1]]})
    tag=f"item{n}"
    hidden[0].numpy().astype("<f4").tofile(f"{OUT}/{tag}.hidden.f32")
    rows[torch.tensor(tok)].numpy().astype("<f4").tofile(f"{OUT}/{tag}.rows.f32")
    items.append(dict(tag=tag,id=e["id"],n=L,ids=e["input_ids"],questions=e["questions"],row_tokens=tok,logits=[l.tolist() for l in logits],probs=probs))
    print(tag, e["questions"][0]["type"], L, "options", [len(q["option_ids"]) for q in e["questions"]], "p", [round(max(p),4) for p in probs])

# A multi-question record (the fixture's records are all single-question): one state, a noul + a choice + a score + a second choice,
# encoded by the official encode_record with the real tokenizer, so the head's cross-question attention and the encoder's FIELD numbering are covered.
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained(M)
recs=[json.loads(l) for l in open(f"{REPO}/testdata/decisions/clef/records.jsonl")]
by={}
for r in recs: by.setdefault(r["kind"],[]).append(r)
state=by["noul"][0]["request"]["state"]
qq={}
for name,kind,idx in (("a","noul",0),("b","choice",3),("c","score",1),("d","choice",7)):
    qq[name]=list(by[kind][idx]["request"]["questions"].values())[0]
# "é" and a quote in an instruction so the renderer's escaping is on the path too
qq["c"]=dict(qq["c"],instructions=qq["c"]["instructions"]+' Say "é" if unsure.')
request={"state":state,"questions":qq}
rec=jsm.encode_record(tk, request)
L=len(rec.input_ids); g=torch.Generator().manual_seed(999)
hidden=torch.randn(1,L,4096,generator=g)*1.5
with torch.no_grad():
    logits=head(hidden, torch.tensor([list(rec.input_ids)]), torch.ones(1,L,dtype=torch.long), [rec], rows)[0]
tok=sorted({t for q in rec.questions for s in q.option_spans for t in rec.input_ids[s[0]:s[1]]})
hidden[0].numpy().astype("<f4").tofile(f"{OUT}/multi.hidden.f32"); rows[torch.tensor(tok)].numpy().astype("<f4").tofile(f"{OUT}/multi.rows.f32")
items.append(dict(tag="multi",id="multi",n=L,ids=list(rec.input_ids),request=request,
  questions=[dict(id=q.question_id,type=q.question_type,question_span=list(q.question_span),option_spans=[list(x) for x in q.option_spans],option_ids=list(q.option_ids)) for q in rec.questions],
  row_tokens=tok,logits=[l.tolist() for l in logits],probs=[torch.softmax(l,0).tolist() for l in logits]))
print("multi",L,"tokens, questions",[(q.question_id,q.question_type,len(q.option_ids)) for q in rec.questions],"p",[round(float(torch.softmax(l,0).max()),4) for l in logits])
json.dump(items,open(f"{OUT}/golden.json","w"))
