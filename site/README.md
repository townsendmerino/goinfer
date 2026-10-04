# site — goinfer.dev, generated from the repo

A Go program that writes plain static files. Its own module (`github.com/townsendmerino/goinfer/site`), so nothing it
needs ever enters goinfer's own dependency graph. Rules: "Policy" below. Plan, design and record: `docs/completed/task-site-2026-09.md`.

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

The remaining writeups (the twenty are listed in the archived plan, `docs/completed/task-site-2026-09.md` section 4a; the rest are drafts for review). Deploy happens
only when a release is cut (S8d): `.github/workflows/site.yml`.

## Policy — what the site may and may not do

These rules came out of the site's planning doc (closed 2026-10-04; its record, with the design, the mockups' rationale and what building it
found, is [`docs/completed/task-site-2026-09.md`](../docs/completed/task-site-2026-09.md)). They are live. The build enforces what it can.

- **The site changes only when a release is cut**, or when `docs/capability-matrix.json` on `main` changes (owner decisions 2026-09-29 and 2026-09-30).
  A deploy builds the **release tag's commit**, not `main`, so the site shows exactly what was released. `.github/workflows/site.yml` deploys
  on a successful `release assets` run for a `v*` tag, on a matrix change, and by hand (`workflow_dispatch`, for the first launch and an emergency fix).
  A push or pull request touching `site/**`, the matrix, `pull/*.json` or the workflow runs the build and every gate but never publishes. The deploy
  job needs the gates, so a failed gate leaves the last good site up. Numbers measured after the last release do not appear until the next one; that is intended.
- **Every number the site shows is a claim with a source.** `site/data/claims.json` holds one entry per figure (value, unit, machine, date, a repo path
  and a heading, the peer value where there is one). The claims check fails the build unless each value appears verbatim in its cited source under its cited
  heading, so a remeasured number fails the build until `claims.json` is updated. Speeds go in `claims.json`, never in templates.
- **Gates (S7, enforced by `Verify` and the build):** every family in the capability matrix has a page; every number names its machine and date; no claim that
  is not in the repo; the book's URLs resolve under `/book/`; the site builds from a clean clone with no manual steps; muted text meets WCAG AA on every surface
  it sits on (light theme `#5c6672`; the earlier `#69737f` measured 4.2:1 and 3.9:1).
- **The front page must not claim a speed parity that does not hold, and it has no speed-versus-competitors hero.** Lead with what is true and plain: one static binary, parity-gated
  numerics, output a Go struct cannot violate; say where the gaps are.
- **What not to build:** no pricing, sign-in, accounts or anything hosted (there is no service); no blog until there is something to put in it; no site-wide search (the book has its own);
  no newsletter, chat-server badge or star counter.
- **What stays hand-written:** the "What's different" writeups and each family's `summary` field. Each writeup carries a `reviewed:` date and its figures go through the claims check.
  Tone is plain and humble.
- **Not built:** preview deploys for pull requests; a `www.goinfer.dev` redirect (a Cloudflare Single Redirect rule set in the dashboard, optional); a decision-model tag for Clef-flash
  (approved 2026-10-03, waits for a release cut and a registry entry; see `docs/tasks/task-constrained-confidence.md`, "Route C — what is still open").
