#!/usr/bin/env python3
"""Decisions D10: the Clef-flash reference fixture (docs/tasks/task-constrained-confidence.md, Route C).

Cloudflare's own code (joint_schema_model.py, read from the pinned release directory and imported unmodified) run on Cloudflare/clef-flash, over the
150 items of the D0 fixture (testdata/decisions/items.jsonl) re-expressed as Clef records, so goinfer's Clef encoder (D12) and head (D12) and the
fidelity gate (D13) have a reference: the record encoder's output, the backbone's last_hidden_state, and the per-option probabilities.

How an item becomes a Clef record (a D10 decision, recorded in the task doc): one question per record, id "q"; "instructions" is the item's question text;
state is the item's state parsed as JSON when it parses, else the raw string (Clef's render() writes a string as is and anything else as sorted compact JSON);
noul: no criteria; choice: criteria {label: None} (the item's own labels are the option ids, no invented descriptions); score: criteria ["0".."5"].

Phases, each its own process (a bf16 load is ~19 GB, f32 ~38 GB; the two never share RAM):

  pin_clef_d10.py records              the 150 Clef records                         -> clef/records.jsonl
  pin_clef_d10.py encode               the official encoder on every record         -> clef/encoder.jsonl(.gz)
  pin_clef_d10.py model --dtype bf16   the release model, per-option probabilities  -> clef/probs_bf16.jsonl
        (bf16 uses the f32-GEMM emulation by default, ~3x faster on this CPU: see install_bf16_gemm_emulation; --native-bf16 for PyTorch's own)
  pin_clef_d10.py model --dtype f32    the same at f32 (the exact reference)        -> clef/probs_f32.jsonl
        [--limit N]   stop after N new items (the probe that sizes a full run)
        [--hidden K]  also save last_hidden_state (f32) for the K items with the fewest tokens -> clef/hidden/*.npy
  pin_clef_d10.py check                sanity: sums to 1, finite, top-1 against gold, bf16 against f32, encoder against the official one

Model phases append one line per item and skip ids already written, so an interrupted run resumes.

Environment: CLEF_MODEL (the pinned release dir, default /srv/models/clef-flash, the archive: this is offline fixture generation, not a timed measurement),
D10_OUT (default testdata/decisions/clef). Paths in the goldens are never absolute.
"""
import argparse, gzip, hashlib, json, os, platform, sys, time

# The pinned release. Clef-flash by default; CLEF_REPO, CLEF_REV, CLEF_MODULE_SHA256 and CLEF_HEAD_SHA256 pin another release of the same
# module (Clef 27B, decisions-d13-clef27b-2026-10-03.md), all four together or none.
MODEL_REPO = os.environ.get("CLEF_REPO", "Cloudflare/clef-flash")
MODEL_REV = os.environ.get("CLEF_REV", "17f0b0ad64efb65d273590632833508766b2aae6")
MODULE_SHA256 = os.environ.get("CLEF_MODULE_SHA256", "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3")   # joint_schema_model.py at MODEL_REV
HEAD_SHA256 = os.environ.get("CLEF_HEAD_SHA256", "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0")     # joint_head.safetensors at MODEL_REV
_PINS = [k for k in ("CLEF_REPO", "CLEF_REV", "CLEF_MODULE_SHA256", "CLEF_HEAD_SHA256") if k in os.environ]
if _PINS and len(_PINS) != 4:
    sys.exit(f"pin another release with all four of CLEF_REPO, CLEF_REV, CLEF_MODULE_SHA256, CLEF_HEAD_SHA256 (got {_PINS})")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser(os.environ.get("CLEF_MODEL", "/srv/models/clef-flash"))
OUT = os.environ.get("D10_OUT", os.path.join(HERE, "..", "testdata", "decisions", "clef"))
ITEMS = os.path.join(HERE, "..", "testdata", "decisions", "items.jsonl")
GZ_OVER = 1 << 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def path(name):
    os.makedirs(os.path.dirname(os.path.join(OUT, name)), exist_ok=True)
    return os.path.join(OUT, name)


def read_jsonl(name):
    for p in (os.path.join(OUT, name), os.path.join(OUT, name + ".gz")):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return [json.loads(l) for l in f if l.strip()]
    return []


def check_pins():
    """The module and the head must be the pinned release's, byte for byte, or the fixture grades something else."""
    for f, want in (("joint_schema_model.py", MODULE_SHA256), ("joint_head.safetensors", HEAD_SHA256)):
        got = sha256_file(os.path.join(MODEL, f))
        if got != want:
            sys.exit(f"{f}: sha256 {got} is not the pinned {want} ({MODEL_REPO} @ {MODEL_REV[:8]})")


def import_module():
    check_pins()
    sys.path.insert(0, MODEL)
    import joint_schema_model as jsm
    return jsm


def tokenizer():
    from transformers import AutoTokenizer          # text-only: the release's processor.tokenizer is this tokenizer; AutoProcessor also needs PIL
    return AutoTokenizer.from_pretrained(MODEL)


# ---- phase: records ----------------------------------------------------------------------------------------------
def to_record(it):
    try:
        state = json.loads(it["state"])
    except (ValueError, TypeError):
        state = it["state"]
    kind, labels = it["kind"], it["labels"]
    q = {"type": kind, "instructions": it["question"]}
    if kind == "choice":
        q["criteria"] = {l: None for l in labels}
    elif kind == "score":
        q["criteria"] = [str(i) for i in range(len(labels))]
    return {"id": it["id"], "request": {"model": "clef-flash", "state": state, "questions": {"q": q}},
            "kind": kind, "labels": labels, "target": it["target"], "gold": bool(it["gold"])}


def cmd_records(_):
    items = [json.loads(l) for l in open(ITEMS) if l.strip()]
    recs = [to_record(i) for i in items]
    with open(path("records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    log(f"wrote {len(recs)} records; records.jsonl sha256 {sha256_file(path('records.jsonl'))}; items.jsonl sha256 {sha256_file(ITEMS)}")


# ---- phase: encode -----------------------------------------------------------------------------------------------
def encode_one(jsm, tok, rec):
    """The official encode_record, with every fragment it tokenizes recorded in the order it asks (the per-fragment tokenization is part of the contract)."""
    frags = []
    orig = jsm._tokens

    def spy(tokenizer, text):
        ids = orig(tokenizer, text)
        frags.append({"text": text, "ids": list(ids)})   # a COPY: the module extends the returned list in place (schema_ids = _tokens(...); schema_ids.extend(...))
        return ids
    jsm._tokens = spy
    try:
        enc = jsm.encode_record(tok, rec["request"])
    finally:
        jsm._tokens = orig
    return enc, frags


def cmd_encode(_):
    jsm = import_module()
    tok = tokenizer()
    recs = read_jsonl("records.jsonl")
    rows = []
    for rec in recs:
        enc, frags = encode_one(jsm, tok, rec)
        rows.append({"id": rec["id"], "n_tokens": len(enc.input_ids), "input_ids": list(enc.input_ids),
                     "input_ids_sha256": hashlib.sha256(json.dumps(list(enc.input_ids)).encode()).hexdigest(),
                     "questions": [{"id": q.question_id, "type": q.question_type, "question_span": list(q.question_span),
                                    "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)} for q in enc.questions],
                     "fragments": [{"text": f["text"], "n_tokens": len(f["ids"])} for f in frags]})
    body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n"
    p = path("encoder.jsonl")
    if len(body) > GZ_OVER:
        p += ".gz"
        with gzip.open(p, "wt") as f:
            f.write(body)
    else:
        open(p, "w").write(body)
    log(f"encoded {len(rows)} records, {sum(r['n_tokens'] for r in rows)} tokens (mean {sum(r['n_tokens'] for r in rows)/len(rows):.0f}); wrote {os.path.basename(p)} ({len(body)} bytes raw)")


# ---- phase: model ------------------------------------------------------------------------------------------------
def install_bf16_gemm_emulation():
    """bf16 matmuls as f32 GEMMs, rounded back to bf16. This CPU (Ryzen 7 3700X, Zen 2: AVX2 and FMA, no AVX-512 or BF16 instructions) runs PyTorch's native bf16 GEMM at
    53 GFLOP/s against f32's 337 (measured, 589 x 4096 x 12288; oneDNN on or off makes no difference). Converting the bf16 operands to f32, doing the f32 GEMM and rounding the
    result to bf16 is the SAME ARITHMETIC a bf16 GEMM unit performs (bf16 inputs, exact products, f32 accumulation, a bf16-rounded output), differing only in summation order, and ran
    3.2x faster on the 238-token probe item (15.7 s against 49.9 s; the remaining time is the f32 GEMMs themselves, then the conversions). Validated against the native-bf16 probe rows:
    the logits differ by at most 0.031, one or two bf16 rounding steps. F.linear is replaced at module level, which also covers nn.MultiheadAttention (it looks `linear` up as a global of that module)."""
    import torch, torch.nn.functional as F
    orig = F.linear

    def linear(x, w, b=None):
        if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            return orig(x.float(), w.float(), None if b is None else b.float()).to(torch.bfloat16)
        return orig(x, w, b)
    F.linear = linear


def load(dtype_name):
    """load_release_model's behaviour (backbone, strict head load) without device_map, so it needs no GPU; the module's own head class and config."""
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration
    jsm = import_module()
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[dtype_name]
    t0 = time.time()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=dtype)
    backbone.config.use_cache = False
    head = jsm.JointSchemaHead(**json.load(open(os.path.join(MODEL, "joint_head_config.json"))))
    head.load_state_dict(load_file(os.path.join(MODEL, "joint_head.safetensors")), strict=True)
    head = head.to(dtype=dtype)
    model = jsm.ClefModel(backbone, head).eval()
    log(f"loaded {dtype_name} in {time.time()-t0:.0f}s")
    return jsm, model


def env_record(dtype_name, native_bf16=False):
    import torch, transformers
    return {"model_repo": MODEL_REPO, "model_rev": MODEL_REV, "module_sha256": MODULE_SHA256, "head_sha256": HEAD_SHA256, "dtype": dtype_name,
            "torch": torch.__version__, "transformers": transformers.__version__, "release_tested_with": {"torch": "2.11", "transformers": "5.10.2"},
            "python": platform.python_version(), "machine": platform.machine(), "host": platform.node(), "cpu_threads": torch.get_num_threads(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "text_only": True,
            "bf16_matmul": ("native torch bf16 GEMM" if dtype_name == "bf16" and native_bf16 else "f32 GEMM, rounded to bf16" if dtype_name == "bf16" else "n/a (f32)")}


def cmd_model(a):
    import torch
    if a.dtype == "bf16" and not a.native_bf16:
        install_bf16_gemm_emulation()
    jsm, model = load(a.dtype)
    tok = tokenizer()
    recs = read_jsonl("records.jsonl")
    if a.every > 1:  # a registered subset: every Nth record by position, the stride the grader checks an arm against
        recs = recs[::a.every]
    out_name = f"probs_{a.dtype}.jsonl"
    done = {r["id"] for r in read_jsonl(out_name)}
    json.dump(env_record(a.dtype, a.native_bf16), open(path(f"clef_env_{a.dtype}.json"), "w"), indent=1, sort_keys=True)
    hidden_ids = set()
    if a.hidden:
        lens = sorted((len(jsm.encode_record(tok, r["request"]).input_ids), r["id"]) for r in recs)
        hidden_ids = {i for _, i in lens[:a.hidden]}
    captured = {}
    model.head.register_forward_pre_hook(lambda m, args: captured.__setitem__("h", args[0].detach().float().cpu().numpy()))
    n_new = 0
    with open(path(out_name), "a") as out:
        for rec in recs:
            if rec["id"] in done:
                continue
            if a.limit and n_new >= a.limit:
                break
            t0 = time.time()
            enc = jsm.encode_record(tok, rec["request"])
            batch = jsm.collate_records([enc], tok.pad_token_id, torch.device("cpu"))
            with torch.inference_mode():
                logits = model(batch)[0]
            q, lg = enc.questions[0], logits[0].float()
            probs = lg.softmax(-1).tolist()
            row = {"id": rec["id"], "kind": rec["kind"], "n_tokens": len(enc.input_ids), "option_ids": list(q.option_ids),
                   "logits": lg.tolist(), "probs": probs, "seconds": round(time.time() - t0, 2)}
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            if rec["id"] in hidden_ids and a.dtype == "f32":
                import numpy as np
                np.save(path(f"hidden/{hashlib.sha256(rec['id'].encode()).hexdigest()[:16]}.npy"), captured["h"][0])
            n_new += 1
            log(f"[{len(done)+n_new}/{len(recs)}] {rec['id'][:48]:48s} {row['n_tokens']:5d} tok  {row['seconds']:6.1f}s  top {q.option_ids[max(range(len(probs)), key=probs.__getitem__)][:20]!r}")
    log(f"done: {n_new} new rows in {out_name}")


# ---- phase: check ------------------------------------------------------------------------------------------------
def cmd_check(_):
    recs = {r["id"]: r for r in read_jsonl("records.jsonl")}
    enc = {r["id"]: r for r in read_jsonl("encoder.jsonl")}
    am = lambda v: max(range(len(v)), key=lambda i: v[i])
    ok = True
    for dt in ("bf16", "f32"):
        rows = read_jsonl(f"probs_{dt}.jsonl")
        if not rows:
            log(f"{dt}: no rows"); continue
        bad = [r["id"] for r in rows if not all(map(lambda x: x == x and abs(x) != float("inf"), r["logits"])) or abs(sum(r["probs"]) - 1) > 1e-5]
        gold = [r for r in rows if recs[r["id"]]["gold"]]
        hit = 0
        for r in gold:
            rec = recs[r["id"]]
            tgt = rec["target"]
            want = rec["labels"][am(tgt)]                       # the gold label, in the item's own labels
            hit += r["option_ids"][am(r["probs"])] == (want if rec["kind"] != "score" else want)
        log(f"{dt}: {len(rows)} rows; non-finite or not summing to 1: {len(bad)}; top-1 against gold {hit}/{len(gold)}")
        ok &= not bad
        mism = [r["id"] for r in rows if r["n_tokens"] != enc[r["id"]]["n_tokens"]] if enc else []
        log(f"{dt}: prompt token count differs from encoder.jsonl on {len(mism)} rows")
        ok &= not mism
    a, b = {r["id"]: r for r in read_jsonl("probs_bf16.jsonl")}, {r["id"]: r for r in read_jsonl("probs_f32.jsonl")}
    both = sorted(set(a) & set(b))
    if both:
        agree = sum(am(a[i]["probs"]) == am(b[i]["probs"]) for i in both)
        log(f"bf16 against f32: top-1 agreement {agree}/{len(both)}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("records"); sub.add_parser("encode"); sub.add_parser("check")
    m = sub.add_parser("model")
    m.add_argument("--dtype", choices=("bf16", "f32"), required=True)
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--hidden", type=int, default=0)
    m.add_argument("--every", type=int, default=1, help="only every Nth record by position (a registered subset)")
    m.add_argument("--native-bf16", action="store_true", help="use PyTorch's own bf16 GEMM (about 3x slower on this CPU) instead of the f32-GEMM emulation")
    a = ap.parse_args()
    {"records": cmd_records, "encode": cmd_encode, "model": cmd_model, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
