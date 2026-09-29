# Task: a site for goinfer — a browsable library, generated from the repo (S1–S8) — 2026-09

> **Status: SCOPED 2026-09-18, unstarted. Domain registered 2026-09-21: `goinfer.dev` (§5). Designed
> 2026-09-29: three mockups in `site/mockups/` are the reference (§4a). Hosting and the build/deploy
> pipeline decided 2026-09-29 (§5, §8).** Filed after the owner looked at ollama.com and wanted
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

- **`docs/capability-matrix.json`** — 37 families, each with `loaders`, `modality`, `parity`,
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
Browsable and filterable over the 37 families and the recommended checkpoints. Filters that match
how someone actually chooses: **what hardware I have**, **what I want it to do** (code / chat /
vision / embedding), **what is proven** (parity tier).

A model page carries what the repo knows and ollama.com's equivalent pages do not:

- what it is good for, and what it costs to run (`GoodFor`, `Needs`)
- the quants, with sizes, and **whether it fits the visitor's machine** — the same coarse
  fits / tight / won't-fit verdict W33 added to the web UI
- **the parity row** — `full-oracle 100.0%/1.00000`, `real-oracle 100.0%/0.98988`,
  `experimental: tiny-oracle` — with a link to what that means. Nobody else publishes this.
- **the tools row** — whether this checkpoint held up under a real agent's tool schema, measured
- **the decisions row** (added 2026-09-28, D9 of `task-constrained-confidence.md`). It shows this checkpoint's measured
  label-scoring figures (top-1, ECE) where a D6-style run exists, and "label scoring, unmeasured" otherwise. A
  **decision model** tag goes only on checkpoints with a trained decision head (Route B, D2–D4), and none exist yet.
- measured throughput, with the machine and date named
- the exact `goinfer-chat pull` line, copyable

### S3 — Docs
The book (already built and deployed) plus `server.md`, `api-tiers.md`, `integrations/`,
`quantization.md`, `giw-bundles.md`. Mostly a move and a navigation shell, not new writing.

### S4 — Download
Per-platform release binaries, which now ship with every tag: `goinfer-serve` (each with the backend
that platform can use), `goinfer-chat`, and the model-embedded `goinfer-chat-0.5b`. Checksums
beside each. This is the promise the README makes, given a page.

### 4a. The design reference — `site/mockups/` (2026-09-29)

Three hand-built pages, reviewed by the owner and approved as the direction. They are a reference
for the generator's templates, not files to ship. Open them straight from disk; they need nothing
but the Google Fonts request.

- **`home.html` — S1.** Nav is Models / Docs / Download / What's different. The hero is
  "What will run well on your machine?": pick one of the three measured machines (M1 Pro 16 GB /
  RTX 2070 SUPER 8 GB / Ryzen 7 CPU) or "something else" (a fit estimate, no speeds). Each model row
  shows fit, parity tier, and **two labelled bars, goinfer and Ollama** (a single bar with a tick
  was tried and read as ambiguous). Then four doors (Try it / Embed it / Connect your tools / Run
  bigger models), six "What's different" cards each with an "It doesn't…" line, "What didn't work",
  and the book.
- **`models.html` — S2.** Proof first: a counts strip (families, checked against released weights,
  against a test model only, with no check of their own, vetted checkpoints, and **0 embedding
  models**), the vetted checkpoints, then all families sorted by strength of check. **There is no
  popularity sort**, because goinfer does not count downloads. Filters: task (chat / code / images /
  embeddings, which is disabled with "none yet"), how it was checked, hide experimental, GGUF-pullable,
  GPU, experts, has a vetted checkpoint. A model page shows get-it (fit on your machine, pull line,
  repo/file/quant/size/sha), good-for/needs, tools, **how sure we are** (the two parity numbers
  explained in words), measured speed per machine, and a **"What hasn't been shown" list generated
  from the data**, not written by hand. Architecture sits in a collapsed section.
- **`different-confidence.html` — one of the twelve writeups.** Every writeup follows the same
  order: the problem → what goinfer does (with a worked example) → how it works → what was
  measured → use it (HTTP and Go) → **what it doesn't do**. A left rail lists all twelve. Its
  example tickets are marked illustrative on the page; everything else is from the repo.

**Tokens.** The colours, fonts and spacing are the `:root` block at the top of each file, the same
in all three, with light and dark themes. Fonts: Schibsted Grotesk (display), Instrument Sans (body),
Martian Mono (labels, numbers). The muted grey is `#5c6672` (see S7).

**The twelve writeups** (titles are the mockup's; the owner may reword): Stop means stop · How sure
was it? · Tool calls that can't come out malformed · A Go struct the model can't break · Decisions
without generating · Checked against the reference · It refuses rather than swaps · A 26B model on
an 8 GB card · Turn nine in under half a second · Work that survives a disconnect · One file, model
inside · No toolchain of any kind.

**What the mockups found in the repo's data**, to fix before or during S2:

- **No family has a plain-language summary.** The matrix `description` is written for engineers.
  The mockup's one-line summaries were invented for it. Add a `summary` field (and `tasks` tags:
  chat / code / vision) to the decoder registry, so the matrix carries them. Do not hand-edit the
  JSON.
- **The Gemma 4 26B-A4B checkpoint's `needs` text quotes 16.12 tok/s** on the RTX 2070 SUPER at 30
  cached expert slots. `docs/benchmarks.md` now reports 40.2 tok/s on the same card. They may be
  different configurations, but side by side on one page they read as a contradiction. Reconcile or
  date the registry text.
- **The Qwen2.5-Coder 1.5B release tier has no `GoodFor` / `Needs` / `Tools`.** It is pinned in
  `pull/curated.json`, not in the registry, so its model page is thin.
- The `qwen3_5` family displays as "Qwen3.8". Confirm that is intended.

## 5. S5 — Hosting and the domain

- **Where the book is now:** GitHub Pages via `book-pages.yml`, at `townsendmerino.github.io/goinfer`.
- **Decided 2026-09-29: the apex site on Cloudflare Workers (static assets); the book stays on GitHub
  Pages at `book.goinfer.dev`.**
  - **Workers, not Cloudflare Pages.** Pages is still maintained, but Cloudflare now ships new
    features to Workers only. Static-asset requests on Workers are free and unlimited, and
    `_redirects` / `_headers` files work the same way they do on Pages.
  - **Two hosts, split by subdomain, on purpose.** This replaces the earlier "splitting the site
    across both does not work" line. GitHub redirecting the old `github.io` URLs by itself is what
    keeps the book's existing links alive, and nothing on the Cloudflare side would reproduce that for
    free. Moving the book to Cloudflare later is its own item, and it would need redirect pages left
    on GitHub Pages.
- **The domain: `goinfer.dev` — registered 2026-09-21.** Chosen over `.io` (a country-code TLD
  whose long-term future has been uncertain since the 2024 Chagos agreement) and `.ai` (several
  times the price, and it brands an engine as an AI product). `goinfer.com` was already taken —
  registered December 2023, not by this project. `.dev` echoes `go.dev`, which is the right
  neighbourhood for an audience of Go engineers, and the whole TLD is HSTS-preloaded, so the site is
  HTTPS-only by construction. Both hosts issue certificates automatically.
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
- **Muted text meets WCAG AA on every surface it sits on.** The 2026-09-29 mockups used `#69737f`,
  which is 4.2:1 on the page ground and 3.9:1 on the sunk panels; an `impeccable detect` pass caught it.
  Use `#5c6672` in the light theme (5.1:1 on ground, 4.7:1 on sunk and the tinted fills). The dark
  theme's `#8a95a2` already passes (4.8:1 at worst). Nothing else from that pass is adopted.

## 8. S8 — Build, deploy, and keep it current (decided 2026-09-29)

### S8a — The generator
- **A Go program in `site/`, with its own `go.mod`** (`github.com/townsendmerino/goinfer/site`). A
  markdown library for the writeups then never enters goinfer's own dependency graph.
- `cd site && go run ./cmd/build -out _site` writes plain static files. Templates are
  `html/template`, built from `site/mockups/`. Pages keep the mockups' small vanilla scripts (the
  machine picker, filters, the confidence example); no JS framework, no Node in the build.
- **Inputs:** `docs/capability-matrix.json`, `pull/capability-matrix.json`, `pull/curated.json`,
  `site/claims.json` (S8b), `site/different/*.md` (S8c), and the release assets for S4. Read the
  assets from `.github/workflows/release-assets.yml`'s matrix, or from the GitHub API at build
  time.
- The book is not built here. It stays on `book-pages.yml` (§5).

### S8b — `site/claims.json`: every number the site shows, with its source
- One entry per displayed figure: `id`, `value`, `unit`, `machine`, `date`, `source` (a repo path
  and a heading), and the peer value where there is one. Speeds go here, not in templates. Parsing
  `docs/benchmarks.md` prose is too fragile to trust.
- **The claims check** is S7's "no claim not in the repo" made concrete: each value must appear
  verbatim in its cited source file, under its cited heading. When a number is remeasured, the build
  fails until `claims.json` is updated. That is the mechanism that would have caught the Gemma
  discrepancy in §4a.

### S8c — The writeups
- `site/different/NN-slug.md`, one per writeup, with front matter: `title`, `area`, `measured`
  (date), `facts` (the strip under the title), `doesnt` (a list of title + one line), and the claim
  ids it cites.
- The body follows the §4a order. Each writeup carries a `reviewed:` date, and its figures go
  through the claims check.
- Tone: plain and humble, per the owner's standing style rules for public writing.

### S8d — Deploy: `.github/workflows/site.yml`
- **Triggers:** a push to `main` touching `site/**`, `docs/capability-matrix.json`, `pull/*.json`
  or the workflow itself; `release: published` (so S4 lists the new tag); `workflow_dispatch`.
- **Steps:** checkout → setup-go → build → **every S7 gate** → deploy with
  `cloudflare/wrangler-action`. Actions are pinned by SHA, like the repo's other workflows.
  `concurrency: site`, no cancel-in-progress. **The deploy job needs the gates**, so a failed gate
  leaves the last good site up.
- `site/wrangler.jsonc`: `assets.directory` → the build output, `not_found_handling: "404-page"`,
  `goinfer.dev` attached as a custom domain. `_redirects` sends `www.goinfer.dev` to the apex.
- Secrets: `CLOUDFLARE_API_TOKEN` (from the "Edit Cloudflare Workers" token template) and
  `CLOUDFLARE_ACCOUNT_ID`.
- Preview deploys for pull requests (`wrangler versions upload`) are not in v1.

### S8e — The owner's one-time steps
1. Put `goinfer.dev`'s DNS on Cloudflare (already there if it was registered through Cloudflare).
2. Create the API token and add both repo secrets.
3. `book.goinfer.dev`: add a DNS-only CNAME to `townsendmerino.github.io`, set it as the custom
   domain in the repo's Pages settings, and enforce HTTPS.
4. After the first deploy, check with `curl -sI`: the apex is 200; `www` redirects; an old
   `townsendmerino.github.io/goinfer/...` chapter URL gives a 301 to `book.goinfer.dev`.

### S8f — Keeping it current
- **Generated pages update themselves.** A new family, a new checkpoint or a new release rebuilds
  the site on the next push. S7 fails the build if a family has no page.
- **Numbers cannot silently go stale.** The claims check fails the build when a cited source
  changes under it.
- **What stays hand-written:** the twelve writeups and the family `summary` field. Each carries a
  reviewed date.
- **Build order:** S8a skeleton + S2 Models first (the reason for the site), then S1 Home, then the
  writeups (confidence first, since its mockup exists), then S4 Download, then the S3 docs shell.
  Deploy from the first working S2, with the other nav items marked "coming".

## Sources

`docs/capability-matrix.json` (37 families, the `loaders`/`modality`/`parity` columns S2 renders) ·
`pull/registry.go` (`Checkpoint`: `GoodFor`, `Needs`, `Tools`, `Quant`, `Bytes`, `Family`,
`Parity`) · `pull/curated.json` · `docs/benchmarks.md` (provenance rule) ·
`docs/what-parity-gated-means.md` · `docs/positioning.md` (what it is not) ·
`.github/workflows/book-pages.yml` (the existing Pages deploy) ·
`.github/workflows/release-assets.yml` (what S4 lists) ·
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W33 (the fit verdict S2 reuses) ·
ollama.com, read 2026-09-18 (the model being selectively copied) · `site/mockups/` (the design
reference, 2026-09-29) · Cloudflare's Workers static-assets docs, read 2026-09-29 (redirects;
billing and limits)

<!-- doc-reviewed: 2026-09-29 -->
