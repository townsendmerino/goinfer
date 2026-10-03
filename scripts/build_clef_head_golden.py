#!/usr/bin/env python3
"""D12's head gate of record: build the golden internal/clef's TestHead_matchesReference reads, from the D10 f32 run's REAL backbone hidden states.

    python3 scripts/build_clef_head_golden.py [--d10 DIR] [--model DIR] [--out DIR] [--no-verify]

Inputs (all produced by scripts/pin_clef_d10.py's `model --dtype f32 --hidden 3`, the night job d10-clef-f32):
  DIR/probs_f32.jsonl            one row per record: id, n_tokens, option_ids, logits, probs
  DIR/hidden/<sha256(id)[:16]>.npy[.gz]   the text model's last_hidden_state (post final norm), f32 [n_tokens, 4096], saved for the 3 shortest records
  testdata/decisions/clef/encoder.jsonl   the official encoder's ids and spans for every record
  MODEL/model-*.safetensors + index       the backbone, for the raw lm_head rows the head averages over each option's tokens

Output, in the format TestHead_matchesReference reads (CLEF_HEAD_GOLDEN=OUT CLEF_HEAD=MODEL):
  OUT/golden.json                items: tag, id, n, ids, row_tokens, logits, probs, questions[id, type, question_span, option_spans, option_ids]
  OUT/<tag>.hidden.f32           [n, 4096] little-endian f32: the real hidden states, as the head received them
  OUT/<tag>.rows.f32             [len(row_tokens), 4096] f32: lm_head.weight rows of the tokens the options use (bf16 widened, which is exactly what the f32 reference's lm_head holds)
  OUT/MANIFEST.json              sha256 of every input and output, and what was checked

THE SELF-CHECK (--verify, on by default when torch is importable): the official JointSchemaHead (joint_schema_model.py from MODEL, sha256-checked against the pinned one) is
re-run on the stored hidden states and the extracted rows, and its logits must match the ones D10 recorded from the full model. That is what proves the three things the Go test
cannot: that the hidden file belongs to the record it is filed under, that the extracted lm_head rows are the ones the reference used, and that nothing was transposed or
mis-sliced on the way. A mismatch above 1e-2 is a hard failure (the golden would fail the Go head for a reason that is not the head); above 1e-4 it is a warning.
"""
import argparse, gzip, hashlib, importlib.util, io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
MODULE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"   # joint_schema_model.py at the pinned revision (scripts/pin_clef_d10.py)
HIDDEN = 4096


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return [json.loads(l) for l in f if l.strip()]
    sys.exit(f"missing {path}[.gz]")


def load_npy(d, rec_id):
    name = hashlib.sha256(rec_id.encode()).hexdigest()[:16]
    for p in (os.path.join(d, "hidden", name + ".npy"), os.path.join(d, "hidden", name + ".npy.gz")):
        if os.path.exists(p):
            if p.endswith(".gz"):
                with gzip.open(p, "rb") as f:
                    return np.load(io.BytesIO(f.read())), p
            return np.load(p), p
    return None, None


def lm_head_rows(model_dir, tokens):
    """The raw lm_head.weight rows of the given token ids, as f32 [len(tokens), 4096]. Reads only those rows from the shard that holds the tensor."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    shard = os.path.join(model_dir, idx["lm_head.weight"])
    out = np.empty((len(tokens), HIDDEN), dtype="<f4")
    with safe_open(shard, framework="pt") as f:
        sl = f.get_slice("lm_head.weight")
        shape = sl.get_shape()
        if shape[1] != HIDDEN:
            sys.exit(f"lm_head.weight is {shape}, want [vocab, {HIDDEN}]")
        for i, t in enumerate(tokens):
            if not 0 <= t < shape[0]:
                sys.exit(f"token {t} is outside the vocabulary {shape[0]}")
            out[i] = sl[t:t + 1][0].float().numpy()
    return out, shard


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d10", default=os.path.expanduser("~/goinfer-bench/decisions-d10/out"))
    ap.add_argument("--model", default=os.path.expanduser("~/models/clef-flash"))
    ap.add_argument("--out", default=os.path.expanduser("~/goinfer-bench/decisions-d12-head-gate"))
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--expect", type=int, default=3, help="how many records must have a hidden file (the D10 job saves the 3 shortest)")
    a = ap.parse_args()
    if a.model.startswith("/srv/models"):
        sys.exit("--model is the archive; use the local copy under ~/models")

    probs = {r["id"]: r for r in read_jsonl(os.path.join(a.d10, "probs_f32.jsonl"))}
    enc = {r["id"]: r for r in read_jsonl(os.path.join(REPO, "testdata", "decisions", "clef", "encoder.jsonl"))}
    if len(probs) < 150:
        print(f"WARNING: probs_f32.jsonl has {len(probs)} rows, not 150: the D10 run did not finish", file=sys.stderr)
    items = []
    for rid, row in probs.items():
        h, path = load_npy(a.d10, rid)
        if h is None:
            continue
        e = enc[rid]
        n = e["n_tokens"]
        if h.dtype != np.float32 or h.shape != (n, HIDDEN):
            sys.exit(f"{rid}: hidden file {path} is {h.dtype} {h.shape}, want float32 ({n}, {HIDDEN})")
        if not np.isfinite(h).all():
            sys.exit(f"{rid}: non-finite values in the hidden states")
        if row["n_tokens"] != n or len(e["input_ids"]) != n:
            sys.exit(f"{rid}: token counts disagree (D10 row {row['n_tokens']}, encoder {n}, ids {len(e['input_ids'])})")
        if [q["option_ids"] for q in e["questions"]] != [row["option_ids"]]:
            sys.exit(f"{rid}: option ids differ between the D10 row and the encoder dump")
        items.append((n, rid, row, e, h, path))
    if len(items) != a.expect:
        sys.exit(f"found {len(items)} records with a hidden file, expected {a.expect} (looked in {a.d10}/hidden)")
    items.sort(key=lambda x: (x[0], x[1]))

    os.makedirs(a.out, exist_ok=True)
    manifest = {"d10_dir": "(not recorded: a local path)", "inputs": {}, "outputs": {}, "items": []}
    golden = []
    for k, (n, rid, row, e, h, hpath) in enumerate(items):
        tag = f"item{k}"
        tokens = sorted({t for q in e["questions"] for s in q["option_spans"] for t in e["input_ids"][s[0]:s[1]]})
        rows, shard = lm_head_rows(a.model, tokens)
        h.astype("<f4").tofile(os.path.join(a.out, tag + ".hidden.f32"))
        rows.tofile(os.path.join(a.out, tag + ".rows.f32"))
        golden.append(dict(tag=tag, id=rid, n=n, ids=e["input_ids"], row_tokens=tokens, logits=[row["logits"]], probs=[row["probs"]],
                           questions=[dict(id=q["id"], type=q["type"], question_span=q["question_span"], option_spans=q["option_spans"], option_ids=q["option_ids"]) for q in e["questions"]]))
        manifest["items"].append({"tag": tag, "id": rid, "n_tokens": n, "n_row_tokens": len(tokens)})
        manifest["inputs"][tag + ".hidden"] = sha256(hpath)
    manifest["inputs"]["lm_head shard"] = {"file": os.path.basename(shard), "sha256": sha256(shard)}
    manifest["inputs"]["probs_f32.jsonl"] = sha256(os.path.join(a.d10, "probs_f32.jsonl"))
    json.dump(golden, open(os.path.join(a.out, "golden.json"), "w"))

    if not a.no_verify:
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError:
            print("torch not importable: the self-check was SKIPPED (the golden is unverified)", file=sys.stderr)
            manifest["verified"] = "skipped (no torch)"
        else:
            src = os.path.join(a.model, "joint_schema_model.py")
            if sha256(src) != MODULE_SHA256:
                sys.exit(f"{src}: sha256 differs from the pinned reference module")
            spec = importlib.util.spec_from_file_location("jsm", src)
            jsm = importlib.util.module_from_spec(spec)
            sys.modules["jsm"] = jsm
            spec.loader.exec_module(jsm)
            head = jsm.JointSchemaHead(**json.load(open(os.path.join(a.model, "joint_head_config.json"))))
            head.load_state_dict({k: v.float() for k, v in load_file(os.path.join(a.model, "joint_head.safetensors")).items()}, strict=True)
            head = head.float().eval()
            worst = 0.0
            for g, (n, rid, row, e, h, hpath) in zip(golden, items):
                tokens = g["row_tokens"]
                rows = np.fromfile(os.path.join(a.out, g["tag"] + ".rows.f32"), dtype="<f4").reshape(len(tokens), HIDDEN)
                where = {t: i for i, t in enumerate(tokens)}

                class Rows:                                  # output_embedding_weight[token_ids] from the EXTRACTED rows only
                    def __getitem__(self, ids):
                        return torch.from_numpy(np.stack([rows[where[int(t)]] for t in ids.tolist()]))
                qs = tuple(jsm.EncodedQuestion(q["id"], q["type"], tuple(q["question_span"]), tuple(tuple(s) for s in q["option_spans"]), tuple(q["option_ids"])) for q in e["questions"])
                rec = jsm.EncodedRecord(tuple(e["input_ids"]), qs, rid)
                with torch.no_grad():
                    out = head(torch.from_numpy(h).unsqueeze(0), torch.tensor([e["input_ids"]]), torch.ones(1, n, dtype=torch.long), [rec], Rows())[0]
                d = max(abs(float(x) - float(y)) for x, y in zip(out[0].tolist(), row["logits"]))
                worst = max(worst, d)
                print(f"{g['tag']} {rid[:48]:48s} n={n:4d}: head re-run on the stored hidden states vs D10's logits, max |diff| {d:.3g}")
                if d > 1e-2:
                    sys.exit(f"{rid}: the re-run head disagrees with D10's logits by {d:.3g}: the hidden file, the lm_head rows or the ids do not belong together; NOT a head defect")
            manifest["verified"] = {"head_rerun_vs_d10_logits_max_abs_diff": worst, "warn_above": 1e-4, "fail_above": 1e-2}
            if worst > 1e-4:
                print(f"WARNING: max |diff| {worst:.3g} is above 1e-4; look before trusting the gate", file=sys.stderr)
    for f in sorted(os.listdir(a.out)):
        if f != "MANIFEST.json":
            manifest["outputs"][f] = sha256(os.path.join(a.out, f))
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1, sort_keys=True)
    print(f"wrote {len(golden)} items to {a.out}; run:\n  CLEF_HEAD_GOLDEN={a.out} CLEF_HEAD={a.model} GOINFER_HEAVY_TESTS=1 go test -count=1 -v -run TestHead_matchesReference ./internal/clef/")


if __name__ == "__main__":
    main()
