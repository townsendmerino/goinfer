# site — goinfer.dev, generated from the repo

A Go program that writes plain static files. Its own module (`github.com/townsendmerino/goinfer/site`), so nothing it
needs ever enters goinfer's own dependency graph. Plan and decisions: `docs/tasks/task-site-2026-09.md`.

```sh
cd site
GOWORK=off go run ./cmd/build -repo .. -out _site     # the build; also runs every gate
GOWORK=off go test ./...                              # the tests
python3 -m http.server -d _site 8765                  # look at it
```

`GOWORK=off` because a checkout's `go.work` (gitignored, and mandatory for the other modules) does not list `site/`.

## What it reads

| input | what it gives |
|---|---|
| `docs/capability-matrix.json` | the families: architecture, parity, loaders, and each family's plain-language `summary` and `tasks`. Generated from the decoder registry, so **never hand-edit it** (`go test ./decoder -run CapabilityMatrix -update`) |
| the matrix's `checkpoint` objects | the vetted checkpoints (label, repo, file, quant, size, sha256, good for, needs, tool calling) |
| `pull/curated.json` | the release tiers, attached to the family that shares their model line |
| `site/data/machines.json` | the machines speeds were measured on, and what the fit verdict knows about them |
| `site/data/claims.json` | every measured figure the site shows, and where it came from |
| `site/different/*.md` | the "What's different" writeups (below) |
| `site/data/docs.json` | which repo documents are rendered under `/docs/`; their relative links are rewritten to docs pages or to GitHub |
| `docs/book/` | the book's chapter titles, for Home (the book itself is built by Jekyll into `/book/` by the workflow) |
| the release data (a generated file next to the other data, not committed) | `scripts/site_release_data.py` writes the release's tag, assets, sizes and checksums for the Download page. A build without it says so on that page |

## The gates

A build fails, naming the cause, when:

- the inputs cannot make pages (a family with no summary, a duplicate, a claim on an unlisted machine or with no date);
- a curated tier fits no family, or a claim is about a checkpoint no page carries;
- **the claims check** finds a figure that is not in the file it cites, under the heading it cites. A figure that is
  a run's median must list its three runs exactly as the record prints them, and be their median. The mockup's own
  1.5B speed (253.1, the first run) failed this and became 252.9;
- **the writeup check** finds a listed figure missing from the page or from its record, or a source that does not exist;
- **the output check** finds a family with no page, a family the Models page does not link, or a figure that is not on
  its page with its date; a doc with no page or a relative link left in it; a Download page missing a binary, its URL or
  its checksum; a writeup with no page, or an unreviewed draft in a deploy build.

## Writeups

`site/different/NN-slug.md`: YAML front matter (every string double-quoted; unknown keys are errors), then Markdown, or HTML
with `raw: true`. The front matter carries the title, standfirst, facts strip, the "It doesn't..." list, the `figures` it
quotes (each must appear in the page **and** in the record it names) and its `sources`. The body has the problem, what
goinfer does, how it works, what was measured and how to use it; the template adds "What it doesn't do".

A writeup is **published only when `reviewed:` holds a date**, which is the owner's. An empty one is a draft: it is built
only by `go run ./cmd/build -drafts` (a preview, marked `noindex`, never linked) and the output check fails a deploy build
that contains one. `NN-slug.css` and `NN-slug.js` beside a writeup are loaded on its page only (the Confidence example).

## Adding a speed

Add it to `claims.json` from a record, never from memory: `runs` / `peer_runs` exactly as printed, `value` / `peer` the
medians, `ratio_raw` the record's own ratio, `date`, and `source` (file and heading). Run the build.

## Not built yet

The remaining writeups (the twenty are listed in the task doc, section 4a; the rest are drafts for review). Deploy happens
only when a release is cut (S8d): `.github/workflows/site.yml`.
