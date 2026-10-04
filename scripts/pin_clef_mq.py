#!/usr/bin/env python3
"""D13 addendum: the multi-question check. Cloudflare's own code (the pinned joint_schema_model.py, via scripts/pin_clef_d10.py's loader) on Cloudflare/clef-flash at f32,
over FIVE-question records: the same state asked noul, score, choice, noul, choice (D7's wording, d7_bench.Q / FIVE, as D14 sent them), so the multi-question path (the
decoder's self-attention over the questions, the per-question option split) gets a real-weights reference. D10's 150 records have one question each.

  pin_clef_mq.py records            the records  -> OUT/records.jsonl     (states: D7's frozen prompts.json, 3 at K=256 and 2 at K=1024)
  pin_clef_mq.py model --dtype f32  the reference per-question probabilities -> OUT/probs_f32.jsonl   (resumable: ids already written are skipped)
  pin_clef_mq.py records --single   the same 25 questions, each in a record of its own -> OUT/records_single.jsonl (ids "<record id>::<question>")
  pin_clef_mq.py model --dtype f32 --single   the reference on those        -> OUT/probs_single_f32.jsonl  (joint versus single, a characterization)

Environment: CLEF_MODEL (default ~/models/clef-flash, the bench set), MQ_OUT (default ~/goinfer-bench/decisions-d13-mq).
"""
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CLEF_MODEL", os.path.expanduser("~/models/clef-flash"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "docs", "measurements", "decisions-d7-2026-09-28"))
os.environ.setdefault("D7_PORT", "18171")
import pin_clef_d10 as d10  # noqa: E402  (load, tokenizer, env_record, the pins)
import d7_bench as d7  # noqa: E402  (Q, FIVE)

OUT = os.path.expanduser(os.environ.get("MQ_OUT", "~/goinfer-bench/decisions-d13-mq"))
PROMPTS = os.path.join(HERE, "..", "docs", "measurements", "decisions-d7-2026-09-28", "prompts.json")
PROMPTS_SHA = "2d56d87f800e2b7235bbeb9e1148039daee1e3d22c53a503ea26711ba0910dd9"
CELLS = [(256, 0), (256, 1), (256, 2), (1024, 0), (1024, 1)]   # (K, state index)


def records():
    sha = hashlib.sha256(open(PROMPTS, "rb").read()).hexdigest()
    if sha != PROMPTS_SHA:
        sys.exit(f"prompts.json sha256 {sha} is not the registered {PROMPTS_SHA}")
    P = json.load(open(PROMPTS))
    out = []
    for K, si in CELLS:
        out.append({"id": f"mq-K{K}-s{si}", "request": {"model": "clef-flash", "state": P["states"][str(K)][si]["text"], "questions": {n: d7.Q[n] for n in d7.FIVE}}})
    return out


def single_records():
    """Each question of each five-question record, alone: the same state, the same question object, nothing else."""
    out = []
    for r in records():
        for name, q in r["request"]["questions"].items():
            out.append({"id": f"{r['id']}::{name}", "request": {"model": r["request"]["model"], "state": r["request"]["state"], "questions": {name: q}}})
    return out


def cmd_records(a):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "records_single.jsonl" if a.single else "records.jsonl")
    with open(p, "w") as f:
        for r in (single_records() if a.single else records()):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {p} sha256 {d10.sha256_file(p)}")


def cmd_model(a):
    import torch
    jsm, model = d10.load(a.dtype)
    tok = d10.tokenizer()
    recs = [json.loads(l) for l in open(os.path.join(OUT, "records_single.jsonl" if a.single else "records.jsonl"))]
    outp = os.path.join(OUT, f"probs_single_{a.dtype}.jsonl" if a.single else f"probs_{a.dtype}.jsonl")
    done = {json.loads(l)["id"] for l in open(outp)} if os.path.exists(outp) else set()
    json.dump(d10.env_record(a.dtype), open(os.path.join(OUT, f"clef_env_{a.dtype}.json"), "w"), indent=1, sort_keys=True)
    with open(outp, "a") as out:
        for k, rec in enumerate(recs, 1):
            if rec["id"] in done:
                continue
            t0 = time.time()
            enc = jsm.encode_record(tok, rec["request"])
            batch = jsm.collate_records([enc], tok.pad_token_id, torch.device("cpu"))
            with torch.inference_mode():
                logits = model(batch)[0]
            qs = []
            for q, lg in zip(enc.questions, logits):
                lg = lg.float()
                qs.append({"id": q.question_id, "option_ids": list(q.option_ids), "logits": lg.tolist(), "probs": lg.softmax(-1).tolist()})
            out.write(json.dumps({"id": rec["id"], "n_tokens": len(enc.input_ids), "questions": qs, "seconds": round(time.time() - t0, 2)}) + "\n")
            out.flush()
            d10.log(f"[{k}/{len(recs)}] {rec['id']} {len(enc.input_ids)} tok, {len(qs)} questions, {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("--single", action="store_true")
    r.set_defaults(fn=cmd_records)
    m = sub.add_parser("model")
    m.add_argument("--dtype", choices=("f32",), required=True)
    m.add_argument("--single", action="store_true")
    m.set_defaults(fn=cmd_model)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
