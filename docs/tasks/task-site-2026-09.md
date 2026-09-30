# Task: a site for goinfer — a browsable library, generated from the repo (S1–S8) — 2026-09

> **Current, 2026-09-29. LIVE at https://goinfer.dev (first launch 2026-09-29, a manual run of the `site` workflow). Built: the
> generator, Models (S2), Home (S1), Download (S4), the Docs shell (S3), the writeup machinery (S8c) with the first writeup, and the
> book at `/book/`. The old GitHub Pages address is a redirect stub.**
> - **Owner decisions:**
>   - the site changes **only when a release is cut** (S8d);
>   - the 26B figure is **40.2 tok/s** (done);
>   - **twenty** "What's different" writeups (§4a);
>   - one host: the book lives at `goinfer.dev/book/` (§5).
> - **Built:**
>   - `site/`, its own module: 37 family pages, the Models index, **Home** (the machine picker, generated from the
>     vetted checkpoints and `claims.json`; the doors; published writeups; the book's chapters), **Download** (from the
>     release's own checksums, `scripts/site_release_data.py`), **Docs** (seven repo documents, links rewritten), and
>     **What's different** (each writeup a page; a writeup is published only when its front matter has a `reviewed` date).
>   - the deploy job and `site/wrangler.jsonc`; the repo secrets are set.
>   - "How sure was it?" (writeup #2) is the first published writeup, carried over from the approved mockup.
> - **For the owner to review:** the first batch of drafts (#13, #14, #16, #18), in `site/different/`, with notes in
>   `REVIEW-batch1.md`. Setting `reviewed:` publishes one at the next release.
> - **Left out of Home on purpose:** the mockup's "What didn't work" cards. Their figures are not traced to records yet,
>   and no page in this plan sits behind them.
> - **Launch:** done (S8e steps 1–4). It needed a new Cloudflare token from the "Edit Cloudflare Workers" template: a first token with
>   only Pages and DNS permissions failed with `Authentication error [code: 10000]`. Checked live afterwards: `/`, `/models/`,
>   `/download/`, `/docs/`, `/different/` and `/book/` are 200, a chapter's `.html` URL 307s to its extensionless page, an unknown path
>   is the 404 page, and no draft is linked. `www.goinfer.dev` is not attached (the optional step 5).
> - **Open:**
>   - writeups #17 and #20 wait on the embed-int4 Metal issue;
>   - the writeups #1, #3–#12, #15, #17, #19, #20 are not written;
>   - the `qwen3_5` family displays as "Qwen3.8";
>   - the owner reviews the 37 summaries and the checkpoint labels (they are public copy, now in the registry);
>   - the fit-rule differences from the mockup (§8g).

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
- **`different-confidence.html` — one of the writeups (the mockup shows the first twelve; twenty are planned, below).** Every writeup follows the same
  order: the problem → what goinfer does (with a worked example) → how it works → what was
  measured → use it (HTTP and Go) → **what it doesn't do**. A left rail lists them all (the mockup's lists twelve). Its
  example tickets are marked illustrative on the page; everything else is from the repo.

**Tokens.** The colours, fonts and spacing are the `:root` block at the top of each file, the same
in all three, with light and dark themes. Fonts: Schibsted Grotesk (display), Instrument Sans (body),
Martian Mono (labels, numbers). The muted grey is `#5c6672` (see S7).

**The first twelve writeups** (titles are the mockup's; the owner may reword): Stop means stop · How sure
was it? · Tool calls that can't come out malformed · A Go struct the model can't break · Decisions
without generating · Checked against the reference · It refuses rather than swaps · A 26B model on
an 8 GB card · Turn nine in under half a second · Work that survives a disconnect · One file, model
inside · No toolchain of any kind.

**Eight more, twenty in all (the owner approved the list, 2026-09-29).** Titles are proposals, and each follows the
same order, with a real "It doesn't…" line and its figures in `claims.json`.

| # | Title | The claim, and where the evidence is | It doesn't… |
|---|---|---|---|
| 13 | Batching that doesn't change the answer | Several conversations share GPU steps, and each reply is bit-identical to serving it alone. Metal 1.59× at 4 clients, CUDA 1.38× (1.5B) and 1.83× (7B), CPU 2.19× the workers on the 7B (`measurements/concurrency-mc3-2026-09-26.md`, `-cuda-`, `-mc3c-step2-`) | do continuous batching or paged attention. It runs 4 at once, the rest queue, a lone request is unchanged, not faster, and hybrid families, adapters and speculation stay one at a time |
| 14 | Faster, with the same words | Lossless n-gram speculation: on Metal 2.08× on copy-heavy traffic and 1.07× on chat, every reply identical to plain decode (`measurements/metal-spec-step-verify-2026-09-27.md`). Carries a "what we got wrong": an earlier build measured 0.98× and later turns differed, found and fixed | help ordinary chat much, and today it gives up batching under load |
| 15 | An upgrade that can't change your answers | `gate identity <old> <new>` compares full logits across two builds and reports IDENTICAL or names the difference: 120 CPU, 38 Metal and 80 WebGPU cells byte-identical run to run (`tasks/task-test-efficiency-2026-09.md`, TE6(b), "Determinism holds"; the census record `measurements/test-efficiency-2026-09.md` does not carry them) | hold across machines or OS versions, cover CUDA from a Mac, or say the answers are right (that is "Checked against the reference"). The most developer-facing of the twenty |
| 16 | Numbers with their receipts | Every benchmark row names its machine, checkpoint and quant, versions, date and thermal state. Both engines are driven over their own HTTP, interleaved, with restarts. The retired "1.78×" claim and why, and "a cross-session ratio is not a ratio" (`benchmarks.md` Methodology, `measurements/noise-registry.md`) | cover more than three machines, or anything but single-user, single-stream speed |
| 17 | Runs on the GPU you have | CUDA, Metal, and WebGPU over Vulkan, Metal or DX12 (so AMD and Intel too), from one cgo-free Go codebase (`hardware-matrix.md`, the real-Vulkan check in `measurements/mc1-webgpu-nobara-2026-09-28/`) | match CUDA or Metal speed on WebGPU, or support ROCm |
| 18 | Starts in a hundredth of a second | A prebuilt sidecar is mapped, not re-quantized: load in 0.01 s against 5.6–16.5 s, about zero heap against 1.3–5 GB (`measurements/cpu-giw-vs-direct-2026-09-24.md`) | skip the first-load conversion, which takes minutes and about the model's size in disk space |
| 19 | Find out before your agent does | `serve check` runs a dozen-tool schema shaped like opencode's before you configure anything, and `fit` answers "does it fit, and how would it be placed" without starting a chat (it loads the checkpoint: a header-only fit is still open, `internal/fitcmd/fit.go`). Its answer matched every real opencode outcome measured (`integrations/opencode.md`) | guarantee a long agent session. `skip` means pick another checkpoint, and the `Tools` row is measured, never guessed |
| 20 | What 4-bit costs | The quality cost of each quantization mode, stated: `--embed-int4` at about 2.3 points of top-1, and which families are int4-hostile (`quantization.md`) | measure against each source's own f16. The 2.3 figure is last measured before the 2026-09-28 default flip and not re-verified |

- **Two of them wait on an open issue.** #17 and #20 both touch the embed-int4 default, which currently makes a
  default Metal load run on the CPU (`quantization.md`, "Known issue"). Each is written after that is fixed, or says
  it plainly.
- **Proposal, not decided: ship a subset first.** With release-only deploys, the first release could carry the
  clearest-numbers writeups (16, 13, 14, 18) beside the first twelve, and add the rest at later releases. That
  keeps the review load manageable, and the owner decides.
- **Reserves, if any is swapped out:** "a Go library, not a server" (overlaps the "Embed it" door) and "vision with
  no Python" (31 s per image on CPU, so weak).

**What the mockups found in the repo's data**, to fix before or during S2:

- **No family has a plain-language summary.** The matrix `description` is written for engineers.
  The mockup's one-line summaries were invented for it. Add a `summary` field (and `tasks` tags:
  chat / code / vision) to the decoder registry, so the matrix carries them. Do not hand-edit the
  JSON.
- **The Gemma 4 26B-A4B checkpoint's `needs` text quoted 16.12 tok/s** (the pre-DMA-overlap slot sweep, 2026-08-27)
  while `docs/benchmarks.md` reports 40.2 tok/s on the same card (2026-09-25, peer-claim cell c). **Resolved
  2026-09-29 (owner: "go with 40.2"):** the registry text, README, the `benchmarks.md` §B4 banner and the mockup now
  say 40.2, with the configuration named. The 16.12 / 17.62 rows stay in §B4.1 as the pre-overlap record. The
  claims check (S8b) is what would have caught this.
- **The Qwen2.5-Coder 1.5B release tier has no `GoodFor` / `Needs` / `Tools`.** It is pinned in
  `pull/curated.json`, not in the registry, so its model page is thin.
- The `qwen3_5` family displays as "Qwen3.8". Confirm that is intended.

## 5. S5 — Hosting and the domain

- **Where the book is now:** GitHub Pages via `book-pages.yml`, at `townsendmerino.github.io/goinfer`. It stays
  there until the launch step below.
- **Decided 2026-09-29, revised the same day: the whole site on Cloudflare Workers (static assets), the book
  included, at `goinfer.dev/book/`.** The first plan kept the book on GitHub Pages at `book.goinfer.dev`. The owner
  preferred one host: one deploy, one certificate, no subdomain or second custom-domain step, and the book can share the
  site's nav and stylesheet.
  - **Workers, not Cloudflare Pages.** Pages is still maintained, but Cloudflare now ships new
    features to Workers only. Static-asset requests on Workers are free and unlimited.
  - **The book's Jekyll build runs in the site workflow** (`scripts/build_book_src.sh` plus `actions/jekyll-build-pages`,
    the same steps `book-pages.yml` uses) with `baseurl: /book`. Its output goes into the artifact under `book/`.
    Consequence: the book follows the release-only rule. An edit to `docs/book/` reaches the public site at the next
    release, not on push.
  - **GitHub Pages cannot simply be switched off.** The old address is baked into things that cannot be edited:
    every released binary's web UI links to `townsendmerino.github.io/goinfer/` (`internal/serveapp/webui/index.html`),
    and the published v0.19.0 and v0.20.0 release notes quote it. So Pages stays, reduced to a **redirect stub**: a
    `404.html` that sends any old path to the same path under `goinfer.dev/book/`. It is a meta refresh plus script, not a
    301. It is a small permanent obligation. The stub goes live only after `goinfer.dev/book/` exists, or every old link would
    redirect to a 404.
- **The domain: `goinfer.dev` — registered 2026-09-21.** Chosen over `.io` (a country-code TLD
  whose long-term future has been uncertain since the 2024 Chagos agreement) and `.ai` (several
  times the price, and it brands an engine as an AI product). `goinfer.com` was already taken —
  registered December 2023, not by this project. `.dev` echoes `go.dev`, which is the right
  neighbourhood for an audience of Go engineers, and the whole TLD is HSTS-preloaded, so the site is
  HTTPS-only by construction. Both hosts issue certificates automatically.
- **Existing URLs must keep working.** The primer is linked from the README, from release notes, from every released
  binary's web UI, and from chapter to chapter. The stub above is the requirement met, not a nicety.

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

**Owner decision, 2026-09-29: the site changes only when a release is cut.** The owner's concern was churn: a public site
that moves with every push to `main`. So a push never deploys. It builds and runs the gates, and the deploy waits for a
release. The consequences are spelled out in S8d and S8f.

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
- **Deploy triggers (release-only):**
  - `workflow_run` of `release assets` completing successfully for a `v*` tag. `release-assets.yml` runs on the tag push
    and attaches the binaries, so waiting for it means S4 lists assets that exist. A plain `release: published` would
    race it.
  - `workflow_dispatch`, for the owner's first launch (S8e) and an emergency fix.
- **Validation triggers (no deploy):** a push or pull request touching `site/**`, `docs/capability-matrix.json`,
  `pull/*.json` or this workflow. It runs the build, every S7 gate and the claims check, so breakage is caught
  before a release, and it never publishes.
- **What gets built:** a deploy checks out the **release tag's commit**, not `main`. The site shows exactly what was
  released, and the claims check runs against the sources as they were at that tag. The artifact is the generator's pages
  plus the book under `book/`, and the build gates the book too (13 pages or more, the search index present, the theme
  stylesheet under `/book/`, no link to the old path, no root-relative link that escapes `/book/`).
- **Steps:** checkout → setup-go → build → **every S7 gate** → deploy with
  `cloudflare/wrangler-action`. Actions are pinned by SHA, like the repo's other workflows.
  `concurrency: site`, no cancel-in-progress. **The deploy job needs the gates**, so a failed gate
  leaves the last good site up.
- `site/wrangler.jsonc`: `assets.directory` → the build output, `not_found_handling: "404-page"`,
  `goinfer.dev` attached as a custom domain. `www.goinfer.dev` is not attached: `_redirects` sources are paths only, so a
  host redirect is a Cloudflare Single Redirect rule, set in the dashboard (S8e).
- Secrets: `CLOUDFLARE_API_TOKEN` (from the "Edit Cloudflare Workers" token template) and
  `CLOUDFLARE_ACCOUNT_ID`.
- Preview deploys for pull requests (`wrangler versions upload`) are not in v1.

### S8e — The owner's one-time steps
1. Put `goinfer.dev`'s DNS on Cloudflare. **Done** (the nameservers are Cloudflare's).
2. Create the API token and add both repo secrets. **Done 2026-09-29** (`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`).
3. **First launch:** run the `site` workflow by hand (Actions → site → Run workflow, on `main`). The deploy job attaches
   `goinfer.dev` as a custom domain, so Cloudflare creates the DNS record and certificate. Then check with `curl -sI`:
   `https://goinfer.dev/` and `https://goinfer.dev/book/` are 200, and `https://goinfer.dev/models/` too.
4. **Then, and only then, the stub** (Claude does this in one commit): `book-pages.yml` publishes the redirect stub instead
   of the book, and the README, `docs/README.md` and the web UI link move to `goinfer.dev/book/`. Check that an old
   `townsendmerino.github.io/goinfer/<chapter>.html` lands on the same chapter under `/book/`.
5. Optional: `www.goinfer.dev` → the apex, as a Cloudflare Single Redirect rule (the `www` DNS record must be proxied).

### S8f — Keeping it current
- **Generated pages update themselves, at a release.** A new family, a new checkpoint or a new release rebuilds the
  site when the next release is cut, and only then. S7 fails the build if a family has no page. Numbers measured after
  the last release do not appear until the next one, which is intended.
- **Numbers cannot silently go stale.** The claims check fails the build when a cited source
  changes under it.
- **What stays hand-written:** the twenty writeups and the family `summary` field. Each carries a
  reviewed date.
- **Build order:** S8a skeleton + S2 Models first (the reason for the site), then S1 Home, then the
  writeups (confidence first, since its mockup exists), then S4 Download, then the S3 docs shell.
  The first deploy is the owner's manual dispatch once S8e is done, with the other nav items marked "coming". After that,
releases only.

### S8g — What was built, and what building it found (2026-09-29)

**Built.**
- `site/`, its own module (`GOWORK=off`). `cd site && GOWORK=off go run ./cmd/build -repo .. -out _site`. Inputs are
  `docs/capability-matrix.json`, `pull/curated.json` and `site/data/{machines,claims}.json`. (S8b's `claims.json` lives
  in `site/data/`.)
- Pages: `/models/` and `/models/<family>/` for all 37 families, plus a redirect at `/` and a 404. The pages are server-rendered.
  The machine picker, ledger filters and copy buttons are progressive enhancement.
- **Gates.** The build fails unless: every family has a page that names it and carries "What hasn't been shown"; the Models page
  links every family; every claim's speed and date appears on its page; the claims check passes.
- **The claims check** requires each value and its peer, the raw ratio and the date to be present in the cited section of the cited
  record. For a median-of-3 claim it also requires the printed runs to have that median, so a run-list member cannot pass as the
  result. It refuses drift (test `TestCheckClaims_refusesDrift`).
- `js_test.go` runs 1,200 cases of the JS fit rule against the Go one under Node. `staticcheck` (CI's pinned version) is clean, and a
  canary showed it can go red. Three code mutations and a JS-rule mutation were each caught by a test.

**Findings: the mockup was wrong in these places, and the generator does not carry them over.**
1. **Mac 0.5B "109 tok/s"** was the CPU cell labelled Metal. The Metal cell is 171.2 against Ollama's 144.5 (1.18x).
2. **Gemma 4 26B on the Mac, "16.98"**, is a CUDA figure (§B4). That claim is dropped, so there is no Mac speed for it.
3. **1.5B on CUDA "253.1"** appears in the record's run list. The median, which is the claim, is 252.9. The first version of the
   check let 253.1 through because it is present in the section; the median rule is the fix.
4. **Embeddings.** The mockup says "0 embedding models" and shows embeddings as "none yet". `docs/server.md` documents
   `--embed-model`, which serves `/v1/embeddings` from Qwen3-Embedding-0.6B (a GGUF; cosine 1.0000000 against the
   sentence-transformers reference on five cases, 2026-07-20, `docs/completed/task-decoder-as-embedder.md`) or CodeRankEmbed
   (an HF directory; served and timed against nomic-embed-text, no reference comparison recorded in this repo). The registry has
   no embedding family, so the generator drops the count and the filter cell and carries a hand-written "Embeddings" section on
   the Models page that says only what those records support. `Verify` fails the build if the section goes missing.
   embeddinggemma is left off: it was never verified (the checkpoint is gated).
5. Phi-3 mini 4k has a CUDA speed in the record (143.2 against 125.9) and none in the mockup. The generator shows it.

**Decision for the owner: fit verdicts are now derived by rule** (`FitFor`, mirrored in `site.js` and tested against it), where the
mockup hand-wrote them. Five cells differ, all towards "fits", because `machines.json` gives the CPU box 62 GB and the mockup assumed
less:

| checkpoint | machine | mockup | derived |
|---|---|---|---|
| Granite 4.0-H Tiny (7.4 GB) | Mac | tight, "half your memory" | fits, on the CPU |
| gpt-oss 20B (12.1 GB) | CPU box | tight | fits |
| Gemma 4 26B-A4B (14.4 GB) | CPU box | tight | fits |
| Granite 4.0-H Tiny | CUDA | fits, "on the CPU, not the card" | fits, "on the CPU" |
| gpt-oss 20B | CUDA | fits, "with -moe-cache-experts" | fits, "experts streamed to the card" |

The two CUDA rows differ in wording only. A fit says nothing about speed. If the owner wants the old hand verdicts, the
change is a per-checkpoint override in `pull/curated.json`, not a rule change.

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
