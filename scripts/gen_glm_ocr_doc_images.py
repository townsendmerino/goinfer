#!/usr/bin/env python3
"""Render the three GLM-OCR O3 gate documents: an invoice, a ruled table, a page with a formula.

THESE ARE PROCEDURALLY RENDERED DOCUMENTS, NOT REAL SCANS. They have no scanner noise, skew,
JPEG artefacts or paper texture, so a model reading them says little about real-document accuracy.
They exist to give the O3 gate three deterministic images of the right shape (text lines, a
numeric grid, a typeset equation) at 1.0-1.5 MP. The ground truth printed here is exact because
we drew it.

    ~/.venv-vl/bin/python scripts/gen_glm_ocr_doc_images.py [OUT_DIR]     # default testdata/glm_ocr

Writes invoice.png, table.png, formula.png and manifest.json (sha256 of each PNG, size, font
files with their packaged versions, PIL version, and the ground-truth text of every image).
Deterministic for a fixed PIL + font set: no timestamps, no randomness.
"""
import hashlib
import json
import os
import subprocess
import sys

import PIL
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "testdata", "glm_ocr")
SANS = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"
SANS_B = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
SERIF = "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Regular.ttf"
SERIF_I = "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Italic.ttf"
MONO = "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf"
MATH = "/usr/share/fonts/stix-fonts/STIXTwoMath-Regular.otf"
FONTS = {"sans": SANS, "sans_bold": SANS_B, "serif": SERIF, "serif_italic": SERIF_I, "mono": MONO, "math": MATH}
INK = (20, 20, 20)
RULE = (60, 60, 60)


def font(path, size):
    return ImageFont.truetype(path, size)


def pkg_versions():
    out = {}
    for pkg in ("liberation-sans-fonts", "liberation-serif-fonts", "liberation-mono-fonts", "stix-fonts"):
        try:
            out[pkg] = subprocess.run(["rpm", "-q", pkg], capture_output=True, text=True).stdout.strip()
        except OSError:
            out[pkg] = "unknown"
    return out


def invoice():
    W, H = 1000, 1300
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    lines = []  # ground truth, reading order

    def text(xy, s, f, fill=INK, anchor="la"):
        d.text(xy, s, font=f, fill=fill, anchor=anchor)

    h1, h2, body, small = font(SANS_B, 44), font(SANS_B, 26), font(SANS, 24), font(SANS, 22)
    text((60, 50), "INVOICE", h1)
    text((W - 60, 56), "Invoice No: INV-2026-0417", body, anchor="ra")
    text((W - 60, 90), "Date: 2026-09-18", body, anchor="ra")
    text((W - 60, 124), "Due: 2026-10-18", body, anchor="ra")
    lines += ["INVOICE", "Invoice No: INV-2026-0417", "Date: 2026-09-18", "Due: 2026-10-18"]
    d.line([(60, 170), (W - 60, 170)], fill=RULE, width=3)

    text((60, 195), "From:", h2)
    for i, s in enumerate(["Harbor Light Supply Co.", "88 Quay Street", "Portland, OR 97209"]):
        text((60, 235 + 34 * i), s, body)
    text((540, 195), "Bill To:", h2)
    for i, s in enumerate(["Northwind Printing LLC", "1204 Alder Avenue, Suite 5", "Seattle, WA 98101"]):
        text((540, 235 + 34 * i), s, body)
    lines += ["From:", "Harbor Light Supply Co.", "88 Quay Street", "Portland, OR 97209",
              "Bill To:", "Northwind Printing LLC", "1204 Alder Avenue, Suite 5", "Seattle, WA 98101"]

    # Line-item grid.
    top = 390
    cols = [60, 120, 520, 650, 800, W - 60]  # No | Description | Qty | Unit | Amount
    heads = ["No", "Description", "Qty", "Unit Price", "Amount"]
    rowh = 52
    items = [
        ("1", "Thermal label roll 100mm", "12", "18.40", "220.80"),
        ("2", "Ink cartridge, cyan", "6", "42.15", "252.90"),
        ("3", "Ink cartridge, black", "6", "39.95", "239.70"),
        ("4", "A4 paper, 80gsm (ream)", "40", "5.25", "210.00"),
        ("5", "Binding spine strips", "8", "11.60", "92.80"),
        ("6", "Shipping and handling", "1", "35.00", "35.00"),
    ]
    nrows = 1 + len(items)
    d.rectangle([cols[0], top, cols[-1], top + rowh * nrows], outline=RULE, width=3)
    d.rectangle([cols[0], top, cols[-1], top + rowh], fill=(228, 228, 228), outline=RULE, width=3)
    for c in cols[1:-1]:
        d.line([(c, top), (c, top + rowh * nrows)], fill=RULE, width=2)
    for r in range(1, nrows):
        d.line([(cols[0], top + rowh * r), (cols[-1], top + rowh * r)], fill=RULE, width=2)
    for ci, hd in enumerate(heads):
        text((cols[ci] + 12, top + 12), hd, h2)
    lines.append(" | ".join(heads))
    for ri, row in enumerate(items):
        y = top + rowh * (ri + 1) + 13
        for ci, cell in enumerate(row):
            if ci >= 2:  # numbers right-aligned
                text((cols[ci + 1] - 14, y), cell, body, anchor="ra")
            else:
                text((cols[ci] + 12, y), cell, body)
        lines.append(" | ".join(row))

    yb = top + rowh * nrows + 40
    totals = [("Subtotal", "1051.20"), ("Tax (8.5%)", "89.35"), ("Total Due", "1140.55")]
    for i, (k, v) in enumerate(totals):
        f = h2 if k == "Total Due" else body
        text((650, yb + 40 * i), k, f)
        text((W - 74, yb + 40 * i), v, f, anchor="ra")
        lines.append(f"{k}: {v}")
    yn = yb + 40 * 3 + 50
    notes = ["Payment terms: net 30 days.", "Please quote the invoice number on your bank transfer.",
             "Account: 4471 0093 2216    Bank: First Harbor Credit Union"]
    for i, s in enumerate(notes):
        text((60, yn + 36 * i), s, small)
    lines += notes
    return im, lines


def table():
    W, H = 1200, 900
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    cap, head, body = font(SERIF_I, 30), font(SANS_B, 28), font(MONO, 28)
    d.text((70, 50), "Table 2. Quarterly unit sales by region (thousands)", font=cap, fill=INK)
    heads = ["Region", "Q1", "Q2", "Q3", "Q4", "Total"]
    rows = [
        ["North", "128.4", "135.9", "142.2", "151.0", "557.5"],
        ["South", "97.1", "101.6", "99.8", "108.3", "406.8"],
        ["East", "210.5", "224.0", "231.7", "246.9", "913.1"],
        ["West", "176.2", "181.4", "190.6", "203.8", "752.0"],
        ["Central", "64.9", "70.3", "73.5", "79.2", "287.9"],
        ["All regions", "677.1", "713.2", "737.8", "789.2", "2917.3"],
    ]
    cols = [70, 310, 480, 650, 820, 960, W - 70]
    top, rowh = 130, 76
    nrows = 1 + len(rows)
    d.rectangle([cols[0], top, cols[-1], top + rowh * nrows], outline=RULE, width=4)
    d.rectangle([cols[0], top, cols[-1], top + rowh], fill=(226, 232, 240), outline=RULE, width=4)
    for c in cols[1:-1]:
        d.line([(c, top), (c, top + rowh * nrows)], fill=RULE, width=3)
    for r in range(1, nrows):
        w = 5 if r in (1, nrows - 1) else 2
        d.line([(cols[0], top + rowh * r), (cols[-1], top + rowh * r)], fill=RULE, width=w)
    for ci, hd in enumerate(heads):
        if ci == 0:
            d.text((cols[0] + 16, top + 20), hd, font=head, fill=INK)
        else:
            d.text((cols[ci + 1] - 16, top + 20), hd, font=head, fill=INK, anchor="ra")
    for ri, row in enumerate(rows):
        y = top + rowh * (ri + 1) + 20
        for ci, cell in enumerate(row):
            if ci == 0:
                d.text((cols[0] + 16, y), cell, font=body, fill=INK)
            else:
                d.text((cols[ci + 1] - 16, y), cell, font=body, fill=INK, anchor="ra")
    foot = font(SERIF, 24)
    d.text((70, top + rowh * nrows + 30), "Source: internal sales ledger, FY2025. Totals may not sum due to rounding.", font=foot, fill=INK)
    truth = ["Table 2. Quarterly unit sales by region (thousands)", " | ".join(heads)]
    truth += [" | ".join(r) for r in rows]
    truth.append("Source: internal sales ledger, FY2025. Totals may not sum due to rounding.")
    return im, truth


def formula():
    W, H = 1000, 1200
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title, body, bi = font(SERIF, 40), font(SERIF, 30), font(SERIF_I, 30)
    math = font(MATH, 44)
    msub = font(MATH, 28)
    d.text((W // 2, 60), "A Note on the Gaussian Integral", font=title, fill=INK, anchor="ma")
    para1 = [
        "The Gaussian integral appears throughout probability and",
        "statistics. Its value is classical and can be obtained by",
        "squaring the integral and passing to polar coordinates.",
    ]
    y = 150
    for s in para1:
        d.text((80, y), s, font=body, fill=INK)
        y += 46
    # Equation 1:  integral_{-inf}^{inf} e^{-x^2} dx = sqrt(pi)
    y += 40
    x = 200
    d.text((x, y), "∫", font=font(MATH, 90), fill=INK)
    d.text((x + 8, y + 100), "−∞", font=msub, fill=INK)
    d.text((x + 8, y - 14), "∞", font=msub, fill=INK)
    x2 = x + 70
    d.text((x2, y + 30), "e", font=math, fill=INK)
    d.text((x2 + 28, y + 12), "−x²", font=msub, fill=INK)
    d.text((x2 + 120, y + 30), "dx  =  √π", font=math, fill=INK)
    d.text((W - 90, y + 36), "(1)", font=body, fill=INK, anchor="ra")
    y += 170
    para2 = [
        "More generally, for a > 0 the same argument gives the",
        "following result, which defines the normal density.",
    ]
    for s in para2:
        d.text((80, y), s, font=body, fill=INK)
        y += 46
    # Equation 2: f(x) = 1/(sigma sqrt(2 pi)) exp(-(x-mu)^2 / (2 sigma^2))
    y += 50
    d.text((120, y + 18), "f(x)  =", font=math, fill=INK)
    fx = 250
    d.text((fx + 60, y - 6), "1", font=math, fill=INK)
    d.line([(fx, y + 52), (fx + 190, y + 52)], fill=INK, width=3)
    d.text((fx, y + 56), "σ√2π", font=math, fill=INK)
    d.text((fx + 215, y + 18), "exp", font=math, fill=INK)
    d.text((fx + 300, y + 4), "(", font=font(MATH, 80), fill=INK)
    d.text((fx + 345, y - 4), "−(x − μ)²", font=font(MATH, 36), fill=INK)
    d.line([(fx + 345, y + 52), (fx + 520, y + 52)], fill=INK, width=3)
    d.text((fx + 380, y + 58), "2σ²", font=font(MATH, 36), fill=INK)
    d.text((fx + 535, y + 4), ")", font=font(MATH, 80), fill=INK)
    d.text((W - 90, y + 36), "(2)", font=body, fill=INK, anchor="ra")
    y += 190
    para3 = ["Here μ is the mean and σ² the variance of the distribution."]
    for s in para3:
        d.text((80, y), s, font=body, fill=INK)
        y += 46
    truth = ["A Note on the Gaussian Integral"] + para1 + [
        "integral from -infinity to infinity of e^{-x^2} dx = sqrt(pi)   (1)"] + para2 + [
        "f(x) = 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)^2 / (2*sigma^2))   (2)"] + para3
    return im, truth


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {
        "note": "Procedurally rendered documents, NOT real scans. Deterministic for this PIL and these fonts.",
        "pil": PIL.__version__,
        "fonts": FONTS,
        "font_packages": pkg_versions(),
        "images": {},
    }
    for name, fn in (("invoice", invoice), ("table", table), ("formula", formula)):
        im, truth = fn()
        path = os.path.join(OUT, name + ".png")
        im.save(path, optimize=True)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        manifest["images"][name] = {"file": name + ".png", "sha256": sha, "width": im.width, "height": im.height,
                                    "megapixels": round(im.width * im.height / 1e6, 3), "truth": truth}
        print(f"{name}: {im.width}x{im.height} {os.path.getsize(path)} bytes sha256 {sha[:16]}")
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
