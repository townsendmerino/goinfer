#!/usr/bin/env python3
"""Write the release facts the goinfer.dev Download page is built from.

    python3 scripts/site_release_data.py [TAG|latest] OUT.json

Reads the GitHub release with `gh` (public data; GH_TOKEN only raises the rate limit): its tag, date, every asset's name,
size and URL, and the per-asset sha256 from the release's own checksums.txt. It refuses a release that looks incomplete,
so a Download page can never be built from a half-attached release (release-assets.yml attaches 27 assets from v0.17.0).
"""
import json, subprocess, sys

MIN_ASSETS = 20  # the release workflow attaches 27; well under that means the upload is still running or failed

def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "latest"
    out = sys.argv[2] if len(sys.argv) > 2 else "release.json"
    repo = "townsendmerino/goinfer"
    if tag == "latest":
        tag = json.loads(gh("release", "view", "--repo", repo, "--json", "tagName"))["tagName"]
    rel = json.loads(gh("release", "view", tag, "--repo", repo, "--json", "tagName,publishedAt,url,assets"))
    assets = [{"name": a["name"], "size": a["size"], "url": a["url"]} for a in rel["assets"]]
    if len(assets) < MIN_ASSETS:
        sys.exit(f"{tag}: only {len(assets)} assets attached (expected at least {MIN_ASSETS}); the upload is not finished")
    sums = {}
    for ln in gh("release", "download", tag, "--repo", repo, "-p", "checksums.txt", "-O", "-").splitlines():
        parts = ln.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1].lstrip("*")] = parts[0]
    missing = [a["name"] for a in assets if a["name"] != "checksums.txt" and a["name"] not in sums]
    if missing:
        sys.exit(f"{tag}: no checksum listed for {missing[:5]}")
    json.dump({"tag": rel["tagName"], "published": rel["publishedAt"][:10], "url": rel["url"],
               "assets": sorted(assets, key=lambda a: a["name"]), "checksums": sums}, open(out, "w"), indent=1)
    print(f"{tag}: {len(assets)} assets, {len(sums)} checksums -> {out}")

main()
