#!/usr/bin/env python3
"""Parse the saved ollama.com/library?sort=popular page (library.html.gz, read 2026-10-09 19:25 PDT) into
[tag, capability tags, pulls as printed] for the first 60 entries, in page order. Size chips (8b, 70b, ...) and the
"cloud" chip are left out; the pull count is the page's own string ("93.3M", "120M").

    python3 parse.py            # prints the 60 as JSON
"""
import gzip, json, os, re

h = gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "library.html.gz"), "rt").read()
rows = []
for it in re.findall(r'<li\s+class="flex items-baseline border-b[^"]*">(.*?)</li>', h, re.S)[:60]:
    tag = re.search(r'href="/library/([^"]+)"', it).group(1)
    chips = re.findall(r'<span\s+class="inline-flex items-center rounded-md ([^"]*)">([^<]+)</span>', it)
    caps = [t.strip() for c, t in chips if "bg-[#ddf4ff]" not in c and t.strip() != "cloud"]
    pulls = re.search(r'<span\s*>([\d.]+[KMB])</span>\s*<span class="hidden sm:flex">&nbsp;Pulls', it).group(1)
    rows.append([tag, caps, pulls])
print(json.dumps(rows))
