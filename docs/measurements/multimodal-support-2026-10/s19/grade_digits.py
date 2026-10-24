#!/usr/bin/env python3
"""G-S19g's every-digit grade (docs/tasks/task-multimodal-support-2026-10.md, "S19 follow-up step 2"). For each form the
arm transcribed (<out>/<arm>-<name>.txt), every number in its truth must appear in the reply. Invoices
(testdata/glm_ocr/invoices/labels.json): the invoice number and both dates as strings; each line's quantity, unit price and
amount, the subtotal, tax and total as numbers (a reply number matches within 0.005, with thousands separators removed).
The table form (table_truth.json beside this script): its 30 cell values as numbers.
Usage: grade_digits.py <out dir> <arm> [<arm> ...]"""
import json, os, re, sys
here = os.path.dirname(os.path.abspath(__file__))
labels = {x["file"].rsplit(".", 1)[0]: x["truth"] for x in json.load(open("testdata/glm_ocr/invoices/labels.json"))}
table = json.load(open(os.path.join(here, "table_truth.json")))["values"]
num = re.compile(r"\d[\d,]*(?:\.\d+)?")

def numbers(text):
    out = []
    for m in num.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out

def grade(name, text):
    got = numbers(text)
    missing = []
    def need(v):
        if not any(abs(g - v) < 0.005 for g in got):
            missing.append(v)
    if name == "table":
        for v in table:
            need(v)
        return len(table), missing
    t = labels[name]
    want = 0
    for key in ("invoice_number", "date", "due_date"):
        want += 1
        if t[key] not in text:
            missing.append(t[key])
    for li in t["line_items"]:
        for key in ("quantity", "unit_price", "amount"):
            want += 1
            need(float(li[key]))
    for key in ("subtotal", "tax", "total"):
        want += 1
        need(float(t[key]))
    return want, missing

out, arms = sys.argv[1], sys.argv[2:]
ok = True
for arm in arms:
    forms = sorted(f[len(arm) + 1:-4] for f in os.listdir(out) if f.startswith(arm + "-") and f.endswith(".txt"))
    total_want = total_miss = 0
    bad = []
    for name in forms:
        want, missing = grade(name, open(os.path.join(out, f"{arm}-{name}.txt")).read())
        total_want += want
        total_miss += len(missing)
        if missing:
            bad.append(f"{name}: missing {missing}")
    print(f"{arm}: {len(forms)} forms, {total_want - total_miss}/{total_want} values present" + ("" if not bad else "; " + "; ".join(bad)))
    ok = ok and not bad and len(forms) == 16
    print(f"  {arm}: {'PASS' if not bad and len(forms) == 16 else 'FAIL'}")
sys.exit(0 if ok else 1)
