# Task: a site for goinfer — a browsable library, generated from the repo (S0–S7) — 2026-09

> **Status: SCOPED 2026-09-18, unstarted. Domain registered 2026-09-21: `goinfer.dev` (§5).** Filed after the owner looked at ollama.com and wanted
> something similar in organisation — "not a perfect copy".
>
> **The decision this doc makes:** copy their *models library*, not their site. ollama.com is now a
> SaaS funnel; goinfer has no service to funnel anyone into. What is worth taking is the one part
> goinfer already has better raw material for, sitting generated and unused in this repo.

---

## 1. Why a site, given the promotion position

The owner's standing position is that goinfer will not be promoted until it is at least as fast as
Ollama. **This doc does not argue with that, and a site is not promotion in that sense.** It is:

- **Legibility.** The README is doing a site's job and cannot. Someone deciding whether to try this
  needs to browse — which model, will it fit, is that family proven — and a README is read
  top-to-bottom or not at all.
- **The fix for the name collision.** Four other repos are called goinfer. A domain is the only
  durable answer to that, and it is needed whether or not anything is ever promoted.
- **Where the book already lives.** `.github/workflows/book-pages.yml` already deploys
  `docs/book/` to GitHub Pages. Half a site exists; it just has no front door.

**The constraint that follows from the position:** the front page must not claim a speed parity
that does not hold. No benchmark-versus-competitors hero. Lead with what nothing else has — one
static binary, no toolchain of any kind, parity-gated numerics, and output a Go struct cannot
violate — and be plain about where the gaps are.

## 2. What ollama.com actually is now, so the model is chosen rather than absorbed

Read 2026-09-18. Nav: **Models / Docs / Pricing / Sign in / Download**. The homepage leads with
"Run open models. Get more usage", then social proof (9M developers, company logos), then sections
on throughput, frontier hosted models, integrations ("Keep your setup"), privacy, and a $20/month
Pro tier with usage included.

That is a product site for a hosted service. **Take the library. Leave the funnel.**

## 3. The architectural decision: the site generates from the repo

This is the item that decides whether the site is alive in a year.

Everything the good part of such a site needs is already generated and maintained here:

- **`docs/capability-matrix.json`** — 36 families, each with `loaders`, `modality`, `parity`,
  `gpu_residency_eligible`, `moe`, `description`. Generated from the `decoder` registry.
- **`pull/curated.json` and `pull/registry.go`'s `Checkpoint`** — per-recommended-checkpoint
  `GoodFor`, `Needs`, `Tools` (from a recorded `serve check` run, never guessed), `Quant`, `Bytes`,
  `SHA256`, plus `Family` and `Parity` copied off the matrix row.
- **`docs/benchmarks.md`** — current claims only, provenance-gated: machine, checkpoint, quant,
  date, thermal note.
- **`docs/hardware-matrix.md`**, **`docs/quantization.md`**, **`docs/what-parity-gated-means.md`**.

**So the Models section is a build step, not a content-writing job.** A hand-written model page goes
stale the week a family lands; a generated one cannot. This is the same reasoning the capability
matrix itself already carries ("the registry is the source of truth; do not hand-edit").

## 4. The sections

### S1 — Home
One line on what it is. The demo GIF that already exists. Install. Then the four things that are
true and unusual: pure Go with no toolchain, one file that can embed its own model, parity-gated
numerics against HuggingFace, and schema-constrained output. A short, plain statement of what it is
*not* (`docs/positioning.md` already writes this well — it is not a serving engine).

### S2 — Models — the reason to build this at all
Browsable and filterable over the 36 families and the recommended checkpoints. Filters that match
how someone actually chooses: **what hardware I have**, **what I want it to do** (code / chat /
vision / embedding), **what is proven** (parity tier).

A model page carries what the repo knows and ollama.com's equivalent pages do not:

- what it is good for, and what it costs to run (`GoodFor`, `Needs`)
- the quants, with sizes, and **whether it fits the visitor's machine** — the same coarse
  fits / tight / won't-fit verdict W33 added to the web UI
- **the parity row** — `full-oracle 100.0%/1.00000`, `real-oracle 100.0%/0.98988`,
  `experimental: tiny-oracle` — with a link to what that means. Nobody else publishes this.
- **the tools row** — whether this checkpoint held up under a real agent's tool schema, measured
- measured throughput, with the machine and date named
- the exact `goinfer-chat pull` line, copyable

### S3 — Docs
The book (already built and deployed) plus `server.md`, `api-tiers.md`, `integrations/`,
`quantization.md`, `giw-bundles.md`. Mostly a move and a navigation shell, not new writing.

### S4 — Download
Per-platform release binaries, which now ship with every tag: `goinfer-serve` (each with the backend
that platform can use), `goinfer-chat`, and the model-embedded `goinfer-chat-0.5b`. Checksums
beside each. This is the promise the README makes, given a page.

## 5. S5 — Hosting and the domain

- **Where the book is now:** GitHub Pages via `book-pages.yml`, at `townsendmerino.github.io/goinfer`.
- **Two options, decide once:** keep Pages and point a domain at it with a CNAME, or move to
  Cloudflare Pages (the owner already has the account that serves Numeratica's docs site). Either
  works; splitting the site across both does not.
- **The domain: `goinfer.dev` — registered 2026-09-21.** Chosen over `.io` (a country-code TLD
  whose long-term future has been uncertain since the 2024 Chagos agreement) and `.ai` (several
  times the price, and it brands an engine as an AI product). `goinfer.com` was already taken —
  registered December 2023, not by this project. `.dev` echoes `go.dev`, which is the right
  neighbourhood for an audience of Go engineers, and the whole TLD is HSTS-preloaded, so the site is
  HTTPS-only by construction. That is no constraint on either Pages option, both of which issue
  certificates automatically.
- **Put the book on a subdomain now — `book.goinfer.dev` — rather than the apex.** Pointing the apex
  at today's book would make the domain useful immediately, but it would move every chapter's URL a
  second time when the real site arrives and claims the apex. A subdomain moves the book exactly
  once, forever, and leaves the apex free for S1. GitHub Pages redirects the old
  `townsendmerino.github.io/goinfer` URLs to a configured custom domain on its own, which satisfies
  the URL rule below for that move.
- **Existing URLs must keep working.** The primer is linked from the README, from release notes, and
  from chapter to chapter. Redirects are a requirement of this item, not a nicety.

## 6. S6 — What not to build, stated

- **No pricing, sign-in, accounts or anything hosted.** There is no service. Adding the shape of one
  to the site would be the clearest possible way to mislead.
- **No speed-versus-competitors hero**, per §1.
- **No blog until there is something to put in it.** An empty blog dates a site faster than no blog.
- **No site-wide search in v1.** The book already has `docs/book-search/`.
- **No newsletter, no Discord badge, no star counter.**

## 7. S7 — Gates

- **Every family in `capability-matrix.json` has a page, or the site build fails.** This is what
  makes "generated from the repo" true rather than aspirational.
- **Every number on the site names its machine and date**, per `docs/benchmarks.md`'s own
  provenance rule.
- **No claim on the site that is not in the repo** — a check in the spirit of
  `scripts/queue_citation_lint.py`, verifying the site's figures against their source files.
- The book's existing URLs still resolve after the move.
- The site builds from a clean clone with no manual steps, like everything else here.

## Sources

`docs/capability-matrix.json` (36 families, the `loaders`/`modality`/`parity` columns S2 renders) ·
`pull/registry.go` (`Checkpoint`: `GoodFor`, `Needs`, `Tools`, `Quant`, `Bytes`, `Family`,
`Parity`) · `pull/curated.json` · `docs/benchmarks.md` (provenance rule) ·
`docs/what-parity-gated-means.md` · `docs/positioning.md` (what it is not) ·
`.github/workflows/book-pages.yml` (the existing Pages deploy) ·
`.github/workflows/release-assets.yml` (what S4 lists) ·
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W33 (the fit verdict S2 reuses) ·
ollama.com, read 2026-09-18 (the model being selectively copied)

<!-- doc-reviewed: 2026-09-18 -->
