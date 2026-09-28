#!/usr/bin/env python3
"""Decisions D0: the JEV-9B reference fixture (docs/prompts/nobara-decisions-d0-fixture-2026-09.md).

autotrust/JEV-9B's own distributions on 150 held-out items, so goinfer's Route A (D1/D6a) and Route B (D2-D4/D6b)
can be graded against the reference. autotrust's server is unpublished (docs/measurements/
decisions-d0-prior-art-2026-09-27.md §2); the reference is the model card's decide(), as the authors' demo Space
ports it (jev_core.py, pinned below): the backbone with the LoRA UNMERGED, the post-final-norm hidden state of the
last token, the fp32 24-slot head, z / T per kind, softmax over the kind's active slots.

Phases, each its own process (a 9B f32 load is ~36 GB; the two precisions never share RAM):

  pin_decisions_d0.py select            the 150 items, rendered bare-v1 prompts and token ids -> items.jsonl
  pin_decisions_d0.py bf16 [--limit N]  Route B in bf16 (the Space's dtype)            -> jev9b_ref_bf16.jsonl
  pin_decisions_d0.py f32  [--limit N]  Route B in f32, and Route A B0 (adapter off)   -> jev9b_ref_f32.jsonl,
                                                                                          route_a_b0.jsonl
  pin_decisions_d0.py check             the sanity checks: verbalizer ids, top-1 vs gold, bf16 vs f32

Model phases append one line per item and skip ids already written, so an interrupted run resumes. --limit N stops
after N new items: the probe that sizes a full run before it is launched.

Environment: JEV_MODEL_PATH (default ~/models/JEV-9B), D0_DATA (the corpus files), D0_OUT (default
testdata/decisions). Paths in the goldens are never absolute.
"""
import argparse, gzip, hashlib, json, os, platform, sys, time

# ---- pins -----------------------------------------------------------------------------------------------------
MODEL_REPO = "autotrust/JEV-9B"
MODEL_REV = "4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35"
DATA_REPO = "SargeDev/jev-distill-corpus-v3"
DATA_REV = "fc99c6357a9f89f7512c4a987314352addead049"
DATA_SHA256 = {
    "calibration.jsonl": "c5e232a0f8efdf9fafc4a829ad40917a6b3ace6ce4bc276b6210c947fb068155",
    "ood.jsonl": "70d0f01742c2d7bbae8546794ce2e12d45ec86eedbd48e7afd31922b4f885781",
}
SPACE_REPO = "autotrust/jev-9b-decision-demo"  # jev_core.py: decide(), build_decision_prompt(), truncate_state()
SPACE_REV = "54a96723bd136ab401bf069b7bb7aab03a1b343f"
LIBS = {"transformers": "5.16.1", "peft": "0.21.0"}

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_STATE_TOKENS = 1024  # jev_core.MAX_STATE_TOKENS: the state alone, head 60% / tail 40%

# (split, kind, source or None for "any gold row", n). Within a stratum: the lowest sha256(id).
STRATA = [
    ("calibration", "noul", "openjev_v2", 17),
    ("calibration", "noul", "yuri_v3", 17),
    ("calibration", "choice", "openjev_v2", 17),
    ("calibration", "choice", "yuri_v3", 16),
    ("calibration", "score", "yuri_v3", 33),  # the split has no gold score rows
    ("ood", "noul", "openjev_v2", 17),
    ("ood", "choice", "openjev_v2", 17),
    ("ood", "score", "openjev_v2", 16),
]
ROW_FIELDS = ("id", "kind", "options", "target", "state", "question", "source", "domain", "family")

MODEL = os.path.expanduser(os.environ.get("JEV_MODEL_PATH", "~/models/JEV-9B"))
DATA = os.path.expanduser(os.environ.get("D0_DATA", "~/goinfer-bench/decisions-d6a/data"))
OUT = os.environ.get("D0_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "decisions"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def open_out(name, mode="r"):
    """name.jsonl, or name.jsonl.gz when that is the one on disk (files over 1 MB are committed gzipped)."""
    p = os.path.join(OUT, name)
    if "r" in mode and not os.path.exists(p) and os.path.exists(p + ".gz"):
        return gzip.open(p + ".gz", mode + "t" if "t" not in mode else mode)
    return open(p, mode)


def read_jsonl(name):
    try:
        with open_out(name) as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


# ---- jev_core.py, verbatim in behaviour ----------------------------------------------------------------------
def build_decision_prompt(kind, state, question, options):
    if kind == "noul":
        options = ["false", "true"]
    elif kind == "score":
        options = [str(i) for i in range(6)]
    options = list(options)
    lines = [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)] if kind == "choice" else options
    prompt = (f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines)
              + "\n[decision]:")
    return prompt, options


def truncate_state(tok, state):
    ids = tok(state, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_STATE_TOKENS:
        return state, False, len(ids)
    keep_head = int(MAX_STATE_TOKENS * 0.6)
    keep_tail = MAX_STATE_TOKENS - keep_head
    return tok.decode(ids[:keep_head] + ids[-keep_tail:], skip_special_tokens=True), True, len(ids)


# ---- select ---------------------------------------------------------------------------------------------------
def cmd_select(_):
    for fn, want in DATA_SHA256.items():
        got = sha256_file(os.path.join(DATA, fn))
        if got != want:
            sys.exit(f"{fn}: sha256 {got} != pinned {want} — not the pinned corpus revision")
    rows = {s: [json.loads(l) for l in open(os.path.join(DATA, s + ".jsonl"))] for s in ("calibration", "ood")}
    key = lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    tok = tokenizer()
    items = []
    for split, kind, source, n in STRATA:
        pool = [r for r in rows[split] if r["kind"] == kind and r["source"] == source]
        if kind == "choice":
            pool = [r for r in pool if 2 <= len(r["options"]) <= 16]  # decide()'s own bound
        pick = sorted(pool, key=key)[:n]
        if len(pick) < n:
            sys.exit(f"{split}/{kind}/{source}: only {len(pick)} rows, want {n}")
        for r in pick:
            state, truncated, n_state = truncate_state(tok, r["state"])
            prompt, labels = build_decision_prompt(r["kind"], state, r["question"], r["options"])
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            it = {f: r.get(f) for f in ROW_FIELDS}
            it.update(split=split, gold=r["source"] == "openjev_v2", labels=labels,
                      state_tokens=n_state, state_truncated=truncated,
                      prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      token_ids=ids, n_tokens=len(ids))
            items.append(it)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    nt = [it["n_tokens"] for it in items]
    n16 = sum(1 for it in items if it["kind"] == "choice" and len(it["labels"]) == 16)
    log(f"items: {len(items)}; prompt tokens total {sum(nt)}, min {min(nt)}, median {sorted(nt)[len(nt)//2]}, "
        f"max {max(nt)}; state truncated: {sum(it['state_truncated'] for it in items)}; 16-option choice rows: {n16}")


# ---- the model phases -----------------------------------------------------------------------------------------
def load(dtype_name):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[dtype_name]
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    peft = PeftModel.from_pretrained(base, MODEL, subfolder="adapter", torch_device="cpu")  # UNMERGED, as the Space
    peft.eval()
    head_sd = load_file(os.path.join(MODEL, "head.safetensors"))
    W, b = head_sd["proj.weight"].float(), head_sd["proj.bias"].float()
    causal = peft.base_model.model
    lora_dtype = next(p.dtype for n, p in peft.named_parameters() if "lora_A" in n)
    log(f"loaded {dtype_name} in {time.time()-t0:.0f}s; LoRA weights {lora_dtype}")
    return peft, causal, W, b, str(lora_dtype).replace("torch.", "")


def env_record(dtype_name, lora_dtype):
    import torch, transformers, peft
    try:
        import fla  # noqa: F401
        has_fla = True
    except ImportError:
        has_fla = False
    return {"precision": dtype_name, "lora_mode": "unmerged", "lora_dtype": lora_dtype, "device": "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "flash_linear_attention": has_fla, "threads": torch.get_num_threads(), "machine": platform.node(),
            "cpu": platform.processor() or platform.machine()}


def softmax(z):
    import torch
    return torch.softmax(z, dim=0).tolist()


def cmd_model(args):
    import torch
    for lib, want in LIBS.items():
        have = __import__(lib).__version__
        if have != want:
            sys.exit(f"{lib} {have} != pinned {want}")
    items = read_jsonl("items.jsonl")
    if not items:
        sys.exit("no items.jsonl — run `select` first")
    judge = json.load(open(os.path.join(MODEL, "judge_config.json")))
    temps = json.load(open(os.path.join(MODEL, "calibration.json")))["per_kind"]
    ranges = judge["slots"]["ranges"]
    verb = judge["verbalizer_ids"]
    outs = {"ref": f"jev9b_ref_{args.phase}.jsonl"}
    if args.phase == "f32":
        outs["b0"] = "route_a_b0.jsonl"
    done = {k: {r["id"] for r in read_jsonl(v)} for k, v in outs.items()}
    todo = [it for it in items if any(it["id"] not in done[k] for k in outs)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        log("nothing to do: every item is already written")
        return
    peft, causal, W, b, lora_dtype = load(args.phase)
    env = env_record(args.phase, lora_dtype)
    files = {k: open(os.path.join(OUT, v), "a") for k, v in outs.items()}
    tok_total = sum(it["n_tokens"] for it in todo)
    tok_done, t_start, last_beat = 0, time.time(), 0.0
    for i, it in enumerate(todo):
        ids = torch.tensor([it["token_ids"]])
        kind, n = it["kind"], len(it["labels"])
        lo = ranges[kind][0]
        if it["id"] not in done["ref"]:
            t0 = time.time()
            h = causal.model(input_ids=ids).last_hidden_state[0, -1].float()  # LoRA active: System 1's backbone
            z = (W @ h + b)[lo:lo + n]
            rec = {"id": it["id"], "kind": kind, "labels": it["labels"], "logits": z.tolist(),
                   "p_raw": softmax(z), "T": temps[kind], "p": softmax(z / temps[kind]),
                   "hidden_first8": h[:8].tolist(), "hidden_l2": float(h.norm()),
                   "n_tokens": it["n_tokens"], "seconds": round(time.time() - t0, 3), **env}
            files["ref"].write(json.dumps(rec) + "\n")
            files["ref"].flush()
        if "b0" in outs and it["id"] not in done["b0"]:
            t0 = time.time()
            with peft.disable_adapter():  # pristine Qwen3.5-9B: Route A as goinfer's D1 computes it
                h0 = causal.model(input_ids=ids).last_hidden_state[0, -1]
                z0 = causal.lm_head.weight[verb[lo:lo + n]].float() @ h0.float()
            rec = {"id": it["id"], "kind": kind, "labels": it["labels"], "verbalizer_ids": verb[lo:lo + n],
                   "logits": z0.tolist(), "p": softmax(z0),
                   "hidden_first8": h0[:8].float().tolist(), "hidden_l2": float(h0.float().norm()),
                   "n_tokens": it["n_tokens"], "seconds": round(time.time() - t0, 3), **env}
            files["b0"].write(json.dumps(rec) + "\n")
            files["b0"].flush()
        tok_done += it["n_tokens"]
        now = time.time()
        if now - last_beat >= 30 or i == len(todo) - 1:
            el = now - t_start
            eta = el / tok_done * (tok_total - tok_done)  # per token: prompt length is what an item costs
            log(f"{args.phase}: {i+1}/{len(todo)} items, {tok_done}/{tok_total} tokens, elapsed {el/60:.1f} min, "
                f"eta {eta/60:.1f} min")
            last_beat = now
    for f in files.values():
        f.close()


# ---- check ----------------------------------------------------------------------------------------------------
def cmd_check(_):
    judge = json.load(open(os.path.join(MODEL, "judge_config.json")))
    tok = tokenizer()
    got = [tok.convert_tokens_to_ids(v) for v in judge["slots"]["verbalizers"]]
    print(f"verbalizer ids from the tokenizer {'EQUAL' if got == judge['verbalizer_ids'] else 'DIFFER'} "
          f"judge_config.json's" + ("" if got == judge["verbalizer_ids"] else f": {got}"))
    items = {it["id"]: it for it in read_jsonl("items.jsonl")}
    argmax = lambda p: max(range(len(p)), key=p.__getitem__)
    runs = {"bf16": read_jsonl("jev9b_ref_bf16.jsonl"), "f32": read_jsonl("jev9b_ref_f32.jsonl"),
            "B0": read_jsonl("route_a_b0.jsonl")}
    for name, rows in runs.items():
        if not rows:
            continue
        for kind in ("noul", "choice", "score", None):
            g = [r for r in rows if items[r["id"]]["gold"] and (kind is None or r["kind"] == kind)]
            if g:
                hit = sum(argmax(r["p"]) == argmax(items[r["id"]]["target"]) for r in g)
                print(f"{name}: top-1 vs gold, {kind or 'all'}: {hit}/{len(g)} = {hit/len(g):.3f}")
    a, b = {r["id"]: r for r in runs["bf16"]}, {r["id"]: r for r in runs["f32"]}
    both = sorted(set(a) & set(b))
    if both:
        dp = max(abs(x - y) for i in both for x, y in zip(a[i]["p"], b[i]["p"]))
        agree = sum(argmax(a[i]["p"]) == argmax(b[i]["p"]) for i in both)
        print(f"bf16 vs f32 on {len(both)} items: argmax agreement {agree/len(both):.4f}, max |dp| {dp:.4f} "
              f"(authors' own vLLM vs HF floor: 0.9912, 0.258)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "bf16", "f32", "check"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"select": cmd_select, "bf16": cmd_model, "f32": cmd_model, "check": cmd_check}[a.phase](a)


if __name__ == "__main__":
    main()
