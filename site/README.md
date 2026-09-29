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

## The gates

A build fails, naming the cause, when:

- the inputs cannot make pages (a family with no summary, a duplicate, a claim on an unlisted machine or with no date);
- a curated tier fits no family, or a claim is about a checkpoint no page carries;
- **the claims check** finds a figure that is not in the file it cites, under the heading it cites. A figure that is
  a run's median must list its three runs exactly as the record prints them, and be their median. The mockup's own
  1.5B speed (253.1, the first run) failed this and became 252.9;
- **the output check** finds a family with no page, a family the Models page does not link, or a figure that is not on
  its page with its date.

## Adding a speed

Add it to `claims.json` from a record, never from memory: `runs` / `peer_runs` exactly as printed, `value` / `peer` the
medians, `ratio_raw` the record's own ratio, `date`, and `source` (file and heading). Run the build.

## Not built yet

Home, Docs, Download and What's different (S1, S3, S4, S8c). Deploy: it happens only when a release is cut
(S8d), and waits on the owner's Cloudflare steps (S8e).
