#!/usr/bin/env python3
"""GLM-OCR O5: field-level accuracy of schema-constrained invoice extraction on the 15 SYNTHETIC invoices (NOT real scans).

Sends each testdata/glm_ocr/invoices/invNN.png to a running goinfer-serve (chat completions, image_url part, NO text part,
response_format json_schema = testdata/glm_ocr/invoice.schema.json, temperature 0, goinfer_confidence on) and scores the reply
against labels.json. Reported, not gated: nothing here fails a build.

    python3 scripts/o5_eval_invoices.py --url http://127.0.0.1:18081 --tag int4-cuda --out docs/measurements/glm-ocr-o5-2026-10 [--docs 1,2,3]

WHAT IS NORMALISED (and nothing else):
  * strings (invoice_number, date, due_date, vendor, bill_to, line item description): runs of whitespace collapsed to one space,
    ends trimmed. Case, punctuation and digits must match exactly.
  * currency: a symbol the model wrote is mapped to the ISO code ($ -> USD, EUR sign -> EUR, pound sign -> GBP), then compared as
    an exact string. (The schema types currency as a free string, as a Go struct has no enum.)
  * money and quantities are JSON numbers in the reply (the grammar cannot emit a currency symbol or a thousands separator in a
    number), so there is nothing to strip; they are compared as numbers to the cent (|diff| < 0.005). The ground truth has its
    separators and symbols removed the same way (scripts/gen_glm_ocr_invoices.py).
  * paid: exact boolean.
Line items are compared POSITIONALLY (item i against truth item i). A reply with fewer items than the truth counts the missing
items' fields as wrong; extra items are ignored for field scoring and flagged in line_item_count.
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "testdata", "glm_ocr")
SYM = {"$": "USD", "€": "EUR", "£": "GBP"}
SCALAR_STR = ["invoice_number", "date", "due_date", "vendor", "bill_to"]
SCALAR_NUM = ["subtotal", "tax", "total"]
ITEM_STR, ITEM_NUM = ["description"], ["quantity", "unit_price", "amount"]


def ws(s):
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else s


def cur(s):
    s = ws(s)
    return SYM.get(s, s)


def num_eq(a, b):
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) < 0.005


def score(truth, got):
    """-> dict field -> (correct, total) over leaf fields, plus list of (path, want, got, ok) rows."""
    rows = []

    def put(path, want, have, ok):
        rows.append((path, want, have, bool(ok)))

    g = got if isinstance(got, dict) else {}
    for k in SCALAR_STR:
        put(k, truth[k], g.get(k), ws(g.get(k)) == ws(truth[k]))
    put("currency", truth["currency"], g.get("currency"), cur(g.get("currency")) == truth["currency"])
    put("paid", truth["paid"], g.get("paid"), g.get("paid") is truth["paid"])
    for k in SCALAR_NUM:
        put(k, truth[k], g.get(k), num_eq(g.get(k), truth[k]))
    items = g.get("line_items") if isinstance(g.get("line_items"), list) else []
    put("line_item_count", len(truth["line_items"]), len(items), len(items) == len(truth["line_items"]))
    for i, t in enumerate(truth["line_items"]):
        it = items[i] if i < len(items) and isinstance(items[i], dict) else {}
        for k in ITEM_STR:
            put(f"line_items[{i}].{k}", t[k], it.get(k), ws(it.get(k)) == ws(t[k]))
        for k in ITEM_NUM:
            put(f"line_items[{i}].{k}", t[k], it.get(k), num_eq(it.get(k), t[k]))
    return rows


def field_kind(path):
    m = re.match(r"line_items\[\d+\]\.(\w+)", path)
    return "line_items." + m.group(1) if m else path


def ask(url, png, schema, max_tokens, conf=True):
    body = {"model": "glm", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "invoice", "schema": schema}}}
    if conf:
        body["goinfer_confidence"] = True
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", default="", help="comma list of 1-based document numbers (default: all 15)")
    ap.add_argument("--max-tokens", type=int, default=2500)
    a = ap.parse_args()
    labels = json.load(open(os.path.join(DOCS, "invoices", "labels.json")))
    schema = json.load(open(os.path.join(DOCS, "invoice.schema.json")))
    pick = {int(x) for x in a.docs.split(",") if x} or set(range(1, len(labels) + 1))
    os.makedirs(a.out, exist_ok=True)
    results = []
    t_all = time.time()
    for n, lab in enumerate(labels, 1):
        if n not in pick:
            continue
        png = open(os.path.join(DOCS, "invoices", lab["file"]), "rb").read()
        d, secs = ask(a.url, png, schema, a.max_tokens)
        ch = d["choices"][0]
        content = ch["message"]["content"]
        parsed, perr = None, None
        try:
            parsed = json.loads(content)
        except ValueError as e:
            perr = str(e)
        rows = score(lab["truth"], parsed)
        res = {"file": lab["file"], "seconds": round(secs, 1), "finish": ch["finish_reason"], "usage": d["usage"], "parsed": parsed is not None,
               "parse_error": perr, "content": content, "rows": rows, "confidence": d.get("goinfer_confidence", [])}
        results.append(res)
        ok = sum(r[3] for r in rows)
        print(f"[{n:2d}/{len(labels)}] {lab['file']} {secs:5.1f}s finish={ch['finish_reason']} completion_tokens={d['usage']['completion_tokens']} parsed={parsed is not None} "
              f"fields {ok}/{len(rows)}  elapsed {time.time() - t_all:.0f}s", flush=True)
        json.dump({"tag": a.tag, "url": a.url, "results": results}, open(os.path.join(a.out, f"results_{a.tag}.json"), "w"), indent=1, ensure_ascii=False)
    report(a.tag, results)


def report(tag, results):
    tot = {}
    for r in results:
        for path, want, have, ok in r["rows"]:
            c = tot.setdefault(field_kind(path), [0, 0])
            c[0] += ok
            c[1] += 1
    print(f"\n## {tag}: {len(results)} documents (synthetic invoices, not real scans)")
    print(f"parsed: {sum(r['parsed'] for r in results)}/{len(results)}; finish=stop: {sum(r['finish'] == 'stop' for r in results)}/{len(results)}")
    print("\n| field | correct / n | exact-match |\n|---|---|---|")
    allc = [0, 0]
    for k, (c, n) in tot.items():
        print(f"| {k} | {c} / {n} | {100 * c / n:.1f}% |")
        allc[0] += c
        allc[1] += n
    print(f"| **all fields** | {allc[0]} / {allc[1]} | {100 * allc[0] / allc[1]:.1f}% |")
    print("\n| document | fields correct | parsed | finish | seconds | wrong fields |\n|---|---|---|---|---|---|")
    for r in results:
        wrong = "; ".join(f"{p}: {w!r} -> {h!r}" for p, w, h, ok in r["rows"] if not ok)
        print(f"| {r['file']} | {sum(x[3] for x in r['rows'])} / {len(r['rows'])} | {r['parsed']} | {r['finish']} | {r['seconds']} | {wrong or '-'} |")


if __name__ == "__main__":
    sys.exit(main())
