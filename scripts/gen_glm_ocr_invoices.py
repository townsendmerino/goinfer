#!/usr/bin/env python3
"""Render 15 SYNTHETIC INVOICES with known ground truth, for GLM-OCR O5's field-level accuracy report.

THESE ARE PROCEDURALLY RENDERED DOCUMENTS, NOT REAL SCANS: no skew, noise, JPEG artefacts or paper
texture, drawn with three fixed layouts and three font families. A model reading them says little
about real-invoice accuracy; what they give is a labelled set whose answers are exact because we drew them.

    ~/.venv-vl/bin/python scripts/gen_glm_ocr_invoices.py [OUT_DIR]     # default testdata/glm_ocr/invoices

Writes inv01.png .. inv15.png, labels.json (the ground truth of every document, the same JSON the O5
extraction schema asks for) and manifest.json (sha256 and size of each PNG, the fonts, PIL version).
Deterministic for a fixed PIL + font set: one seeded RNG, no timestamps.

Ground-truth conventions (they decide what "exact" means in the report, see docs/measurements/glm-ocr-o5-2026-10/):
  * strings are exactly as printed (a date stays "09/18/2026" if that is how the page shows it);
  * money and quantities are numbers (what the page shows, thousands separators and currency symbols removed);
  * currency is the ISO code of the symbol printed beside the total ($ USD, EUR, GBP pounds);
  * paid is true iff the page carries a PAID stamp.
"""
import hashlib
import json
import os
import random
import sys
from decimal import ROUND_HALF_UP, Decimal

import PIL
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "testdata", "glm_ocr", "invoices")
FD = "/usr/share/fonts"
FONTS = {
    "sans": (f"{FD}/liberation-sans-fonts/LiberationSans-Regular.ttf", f"{FD}/liberation-sans-fonts/LiberationSans-Bold.ttf"),
    "serif": (f"{FD}/liberation-serif-fonts/LiberationSerif-Regular.ttf", f"{FD}/liberation-serif-fonts/LiberationSerif-Bold.ttf"),
    "mono": (f"{FD}/liberation-mono-fonts/LiberationMono-Regular.ttf", f"{FD}/liberation-mono-fonts/LiberationMono-Bold.ttf"),
}
INK, RULE, GREY, RED = (20, 20, 20), (70, 70, 70), (228, 228, 228), (190, 30, 30)

VENDORS = ["Harbor Light Supply Co.", "Brightwater Logistics Ltd", "Cedar & Pine Joinery", "Kestrel Analytics GmbH",
           "Northgate Office Products", "Alder Street Bakery", "Meridian Cloud Services Inc.", "Tamarack Surveying LLC",
           "Quill & Ledger Stationers", "Summit Peak Outfitters", "Lantern Row Printing", "Foxglove Garden Centre",
           "Ironbridge Fabrication", "Marlow Dental Group", "Paloma Linens Co."]
CLIENTS = ["Northwind Printing LLC", "Oakline Hotels plc", "Vega Robotics Inc.", "Thistle & Rowan Ltd", "Bluebird Pediatrics",
           "Granite State Credit Union", "Pioneer Ridge School District", "Sable Marine Services", "Lumen Photography",
           "Redcliff Estates", "Orchid Bay Hospitality", "Atlas Freight Partners", "Willow Creek Farms", "Juniper Legal LLP",
           "Copperfield Motors"]
STREETS = ["88 Quay Street", "1204 Alder Avenue, Suite 5", "17 Mill Lane", "402 Harbour Road", "9 Foundry Way", "3300 Cascade Blvd",
           "56 Orchard Close", "711 Ridge Park Drive"]
CITIES = ["Portland, OR 97209", "Seattle, WA 98101", "Leeds LS1 4AP", "Austin, TX 78701", "Berlin 10115", "Boise, ID 83702",
          "Bristol BS1 5TR", "Denver, CO 80202"]
ITEMS = ["Thermal label roll 100mm", "Ink cartridge, cyan", "Ink cartridge, black", "A4 paper, 80gsm (ream)", "Binding spine strips",
         "Shipping and handling", "Site survey, per hour", "Oak shelf bracket", "Cloud hosting, monthly", "Support retainer",
         "Pastry box, assorted", "Design review", "Laminated menu cards", "Steel angle 40x40x3m", "Dental hygiene kit",
         "Cotton napkins, set of 12", "Courier fee", "Hardwood flooring, per m2", "Data export licence", "Pallet wrap"]
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£"}
TAX = [Decimal("0"), Decimal("0.05"), Decimal("0.085"), Decimal("0.10"), Decimal("0.20")]


def q2(d):
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def money(d, thousands):
    s = f"{q2(d):,.2f}"
    return s if thousands else s.replace(",", "")


def make_doc(rng, i):
    """One invoice's content: the ground truth (a dict) plus the presentation choices that are not part of it."""
    n_items = rng.randint(2, 6)
    items, sub = [], Decimal(0)
    for desc in rng.sample(ITEMS, n_items):
        qty = rng.choice([1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 24, 40])
        unit = q2(Decimal(rng.randint(250, 95000)) / 100)
        amt = q2(unit * qty)
        sub += amt
        items.append({"description": desc, "quantity": qty, "unit_price": float(unit), "amount": float(amt)})
    rate = rng.choice(TAX)
    tax = q2(sub * rate)
    total = sub + tax
    cur = rng.choice(["USD", "USD", "EUR", "GBP"])
    datefmt = rng.choice(["iso", "us", "long"])
    y, m, d = 2026, rng.randint(1, 12), rng.randint(1, 28)
    due_m, due_y = (m + 1 - 1) % 12 + 1, 2026 + (m + 1 - 1) // 12
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def fmt(yy, mm, dd):
        return {"iso": f"{yy}-{mm:02d}-{dd:02d}", "us": f"{mm:02d}/{dd:02d}/{yy}", "long": f"{dd} {mon[mm - 1]} {yy}"}[datefmt]

    numfmt = rng.choice(["INV-2026-%04d", "%05d", "A-%05d", "2026/%04d"])
    truth = {
        "invoice_number": numfmt % rng.randint(1, 99999 if "%05d" in numfmt else 9999),
        "date": fmt(y, m, d),
        "due_date": fmt(due_y, due_m, d),
        "vendor": VENDORS[i % len(VENDORS)],
        "bill_to": CLIENTS[(i * 7 + 3) % len(CLIENTS)],
        "currency": cur,
        "paid": i % 3 == 1,
        "line_items": items,
        "subtotal": float(sub),
        "tax": float(tax),
        "total": float(total),
    }
    style = {
        "layout": ["classic", "band", "compact"][i % 3],
        "font": ["sans", "serif", "mono", "sans", "serif"][i % 5] if i % 3 != 0 else ["sans", "serif", "sans"][(i // 3) % 3],
        "thousands": rng.random() < 0.6,
        "rate": rate,
        "vendor_street": rng.choice(STREETS), "vendor_city": rng.choice(CITIES),
        "client_street": rng.choice(STREETS), "client_city": rng.choice(CITIES),
        "size": rng.choice([(900, 1150), (1000, 1200), (1050, 1250), (950, 1100)]),
    }
    return truth, style


class Page:
    def __init__(self, size, family, scale):
        self.im = Image.new("RGB", size, "white")
        self.d = ImageDraw.Draw(self.im)
        self.W, self.H = size
        self.reg, self.bold = FONTS[family]
        self.scale = scale
        self.cache = {}

    def f(self, px, bold=False):
        k = (px, bold)
        if k not in self.cache:
            self.cache[k] = ImageFont.truetype(self.bold if bold else self.reg, int(px * self.scale))
        return self.cache[k]

    def t(self, xy, s, px, bold=False, fill=INK, anchor="la"):
        self.d.text(xy, s, font=self.f(px, bold), fill=fill, anchor=anchor)


def stamp(pg, y):
    """A rotated red PAID stamp, drawn on a transparent layer and pasted at the left margin beside the totals (it must not
    cover a value: the ground truth is only as good as the page is legible)."""
    layer = Image.new("RGBA", (340, 120), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle((4, 4, 336, 116), radius=14, outline=RED + (255,), width=6)
    ld.text((170, 62), "PAID", font=ImageFont.truetype(FONTS["sans"][1], 80), fill=RED + (255,), anchor="mm")
    layer = layer.rotate(10, expand=True, resample=Image.BICUBIC)
    pg.im.paste(layer, (60, y), layer)


def draw_table(pg, x0, x1, y, truth, style, grid, hdr_fill):
    cols = [("No", 0.0, 0.07, "l"), ("Description", 0.07, 0.50, "l"), ("Qty", 0.50, 0.62, "r"), ("Unit Price", 0.62, 0.81, "r"), ("Amount", 0.81, 1.0, "r")]
    w = x1 - x0
    rh = int(46 * pg.scale)
    pg.d.rectangle((x0, y, x1, y + rh), fill=hdr_fill)
    for name, a, b, al in cols:
        xx = x0 + (a * w + 10 if al == "l" else b * w - 10)
        pg.t((xx, y + rh // 2), name, 22, bold=True, anchor="lm" if al == "l" else "rm")
    y += rh
    for i, it in enumerate(truth["line_items"], 1):
        vals = [str(i), it["description"], str(it["quantity"]), money(Decimal(str(it["unit_price"])), style["thousands"]),
                money(Decimal(str(it["amount"])), style["thousands"])]
        for (name, a, b, al), v in zip(cols, vals):
            xx = x0 + (a * w + 10 if al == "l" else b * w - 10)
            pg.t((xx, y + rh // 2), v, 22, anchor="lm" if al == "l" else "rm")
        y += rh
        if grid == "rows":
            pg.d.line((x0, y, x1, y), fill=GREY, width=2)
    if grid == "box":
        pg.d.rectangle((x0, y - rh * (len(truth["line_items"]) + 1), x1, y), outline=RULE, width=3)
        for _, a, _, _ in cols[1:]:
            pg.d.line((x0 + a * w, y - rh * (len(truth["line_items"]) + 1), x0 + a * w, y), fill=RULE, width=2)
        for k in range(1, len(truth["line_items"]) + 1):
            pg.d.line((x0, y - rh * k, x1, y - rh * k), fill=RULE, width=2)
    return y


def draw_totals(pg, x1, y, truth, style):
    sym = SYMBOL[truth["currency"]]
    rate = style["rate"]
    label = f"Tax ({(rate * 100).normalize():f}%)" if rate else "Tax (0%)"
    rows = [("Subtotal", truth["subtotal"], False), (label, truth["tax"], False), ("Total Due", truth["total"], True)]
    for name, v, b in rows:
        pg.t((x1 - int(300 * pg.scale), y), name, 24, bold=b)
        s = money(Decimal(str(v)), style["thousands"])
        pg.t((x1 - 10, y), (sym + s) if b else s, 24, bold=b, anchor="ra")
        y += int(40 * pg.scale)
    return y


def render(truth, style, path):
    pg = Page(style["size"], style["font"], 1.0)
    W, H, d = pg.W, pg.H, pg.d
    lay = style["layout"]
    m = 60
    if lay == "classic":
        pg.t((m, 50), "INVOICE", 44, bold=True)
        for k, (lab, v) in enumerate([("Invoice No", truth["invoice_number"]), ("Date", truth["date"]), ("Due", truth["due_date"])]):
            pg.t((W - m, 55 + 34 * k), f"{lab}: {v}", 24, anchor="ra")
        d.line((m, 170, W - m, 170), fill=RULE, width=4)
        pg.t((m, 195), "From:", 26, bold=True)
        pg.t((W // 2 + 20, 195), "Bill To:", 26, bold=True)
        for k, s in enumerate([truth["vendor"], style["vendor_street"], style["vendor_city"]]):
            pg.t((m, 234 + 34 * k), s, 24)
        for k, s in enumerate([truth["bill_to"], style["client_street"], style["client_city"]]):
            pg.t((W // 2 + 20, 234 + 34 * k), s, 24)
        y = draw_table(pg, m, W - m, 380, truth, style, "box", GREY) + 40
        ty, y = y, draw_totals(pg, W - m, y, truth, style)
    elif lay == "band":
        d.rectangle((0, 0, W, 150), fill=(36, 52, 84))
        pg.t((m, 75), truth["vendor"], 38, bold=True, fill=(255, 255, 255), anchor="lm")
        pg.t((W - m, 75), "TAX INVOICE", 30, bold=True, fill=(255, 255, 255), anchor="rm")
        pg.t((m, 185), style["vendor_street"] + ", " + style["vendor_city"], 20, fill=RULE)
        pg.t((m, 240), "Billed to", 22, bold=True)
        pg.t((m, 272), truth["bill_to"], 26)
        pg.t((m, 306), style["client_street"] + ", " + style["client_city"], 20, fill=RULE)
        for k, (lab, v) in enumerate([("Invoice number", truth["invoice_number"]), ("Issued", truth["date"]), ("Payment due", truth["due_date"])]):
            pg.t((W - 330, 240 + 34 * k), lab, 22, bold=True)
            pg.t((W - m, 240 + 34 * k), v, 22, anchor="ra")
        y = draw_table(pg, m, W - m, 400, truth, style, "rows", (214, 222, 240)) + 40
        ty, y = y, draw_totals(pg, W - m, y, truth, style)
    else:  # compact: one text block per line, no vertical rules, smaller type
        pg.t((m, 45), f"{truth['vendor']}", 34, bold=True)
        pg.t((m, 92), f"{style['vendor_street']} - {style['vendor_city']}", 20, fill=RULE)
        d.line((m, 128, W - m, 128), fill=RULE, width=2)
        pg.t((m, 150), "Invoice #", 22, bold=True)
        pg.t((m + 150, 150), truth["invoice_number"], 22)
        pg.t((W // 2, 150), "Date", 22, bold=True)
        pg.t((W // 2 + 90, 150), truth["date"], 22)
        pg.t((m, 186), "Customer", 22, bold=True)
        pg.t((m + 150, 186), truth["bill_to"], 22)
        pg.t((W // 2, 186), "Due date", 22, bold=True)
        pg.t((W // 2 + 130, 186), truth["due_date"], 22)
        pg.t((m + 150, 220), style["client_street"] + ", " + style["client_city"], 20, fill=RULE)
        y = draw_table(pg, m, W - m, 290, truth, style, "rows", GREY) + 36
        ty, y = y, draw_totals(pg, W - m, y, truth, style)
    pg.t((m, H - 110), "Payment terms: net 30 days. Thank you for your business.", 20, fill=RULE)
    if truth["paid"]:
        stamp(pg, ty)
    pg.im.convert("L").save(path, optimize=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(20261002)
    labels, manifest = [], {}
    for i in range(15):
        truth, style = make_doc(rng, i)
        name = f"inv{i + 1:02d}.png"
        path = os.path.join(OUT, name)
        render(truth, style, path)
        raw = open(path, "rb").read()
        with Image.open(path) as im:
            w, h = im.size
        manifest[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "size": [w, h], "megapixels": round(w * h / 1e6, 2),
                          "layout": style["layout"], "font": style["font"], "thousands_separators": style["thousands"]}
        labels.append({"file": name, "truth": truth})
    json.dump(labels, open(os.path.join(OUT, "labels.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({"generator": "scripts/gen_glm_ocr_invoices.py", "seed": 20261002, "pil": PIL.__version__,
               "fonts": {k: list(v) for k, v in FONTS.items()}, "documents": manifest},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    tot = sum(v["bytes"] for v in manifest.values())
    print(f"wrote {len(labels)} invoices to {OUT}: {tot / 1e6:.2f} MB of PNG")


if __name__ == "__main__":
    main()
