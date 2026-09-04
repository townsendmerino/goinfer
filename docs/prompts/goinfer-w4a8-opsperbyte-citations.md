# goinfer: the W4A8 ops-per-byte answer cites three aikit artifacts that were never committed

> Written 2026-08-22 against goinfer `9cb2c73` and aikit `aebf27e` (aikit freshly pulled, working
> tree clean). **Scope is goinfer-side only, and docs-only.** The measurement's numbers and its
> conclusions are NOT in question — do not re-litigate them, do not re-run anything, do not touch
> aikit. What is wrong is *provenance*: `docs/measurements/aikit-w4a8-opsperbyte.md` (commit
> `f410fe2`) points a reader at aikit files that do not exist at any commit, so none of it can be
> reproduced or checked from either repo.

## What was already verified — do not re-derive this

In `/home/francis/mycode/aikit/aikit` at `aebf27e`, `git status --porcelain` empty:

- `git log --all -S'w4a8_opsperbyte'` → **empty**. No commit on any ref has ever contained that string.
- No `linalg/w4a8_opsperbyte_bench_test.go` — not tracked, not untracked, not present.
- No `*fold4*` file anywhere under the repo, and `dotW4A8Fold4AVX2` is not a symbol in `linalg/`.
- aikit's internal perf dead-ends log is **tracked and committed there** (not gitignored), and its
  Group 8 ends at **§8.8** (MarshalBinary wrapper). There is no §8.9.
- grepping `ops-per-byte`, `opsperbyte` and `ops/byte` across aikit's internal docs → nothing.

**These citations DO resolve — leave them exactly as they are:** `linalg/dot_w4a8_amd64.s` /
`dotW4A8FoldAVX2`; `linalg/dot_amd64.s` / `dotI8AVX2`; aikit's internal microgpt-c priors note §1
(which is indeed "Instrument: marginal-FMA injection as an issue-width probe");
`QuantizeActivationsInto` and `MatmulBTW4A8Into` in `linalg/quant.go`.

## Rule this out first (five minutes, and it changes the whole task)

Check whether the bench and the two experimental kernels survive in *any* clone or on the other box
before you erase a pointer to them. If they exist somewhere, **the right fix is to land them in
aikit and leave this doc's citations intact** — items 1–3 below become moot and you should say so
rather than doing them. Only proceed with the edits if the work is genuinely gone.

## Item 1 — the method citation names a file that has never existed

`docs/measurements/aikit-w4a8-opsperbyte.md:6-7`:

> Method and code: `linalg/w4a8_opsperbyte_bench_test.go` (`TestW4A8OpsPerByte`,
> `TestW4A8IssueWidthProbe`) in the aikit repo.

This is the doc's only statement of method. As written it tells a reader "go read the bench" and
the bench is not there — which is worse than saying nothing, because it reads as verifiable.

## Item 2 — two experiments are reported as "Built" with no surviving artifact

- `:42` — "Built the 4-independent-accumulator variant (`dotW4A8Fold4AVX2`)". No such symbol, no
  such file, in aikit or anywhere under `~/mycode`.
- `:64` — "Built the 'pre-unpacked-weights, still-scaled' micro-benchmark". Same.

Both carry specific numbers (`17.36`/`16.15 GMAC/s`; the 295/188/106 ns table). Keep the numbers.
The problem is that "Built" implies something a reader could go look at.

## Item 3 — the §8.9 reference is wrong twice over

`:58-60`:

> Recorded as a measured dead end, not a re-triable one, in aikit's own internal (gitignored,
> machine-local) dead-ends log, §8.9 — not cited as a concrete path here since a cross-repo
> gitignored file is unverifiable from any other clone or from CI.

Two independent errors: (a) that dead-ends log is **tracked in aikit**, not
gitignored and not machine-local — so the stated *reason* for hedging the citation is false; and
(b) **§8.9 does not exist** — the entry was never written, so the hedge was hiding a dangling
reference rather than a merely-unverifiable one. Note the irony to fix, not to editorialize about:
the sentence explains why an unverifiable cross-repo pointer shouldn't be trusted, and is itself
one.

`:114` back-references "the accumulator dependency chain, §8.9" in the Recommendation — fix that
too, or it re-dangles from the section people actually read.

## Item 4 — small misattribution

`:96` credits "goinfer's own `QuantizeActivationsInto`". It is aikit's —
`linalg/quant.go:283` (shifted from `:206` by the 2026-09-03 aikit v1.33.0 bump). goinfer calls it;
it doesn't own it.

## What "fixed" looks like

Make the doc's provenance honest without weakening its findings. The true story is: this was
written and run against a working tree that was never committed, on the Ryzen box, and the code did
not survive. That is a real limitation and stating it plainly is the fix — the numbers stand as
reported by whoever ran them, but they are **not reproducible from any committed artifact**, and a
reader deserves to know that up front rather than after a failed `git log`.

Constraints:

- **Do not delete the findings, the tables, or the recommendation.** The conclusion (don't start the
  Q4_K-style format redesign; VNNI is the remaining untested lever) is unaffected by any of this.
- **Do not invent** a commit hash, a file path, or a section number to make a citation look
  resolvable.
- Put the caveat where it can't be missed — near the top, with the method statement in item 1, not
  in a footnote.
- `docs/prompts/aikit-w4a8-ops-per-byte.md` **stays ANSWERED**. This is a correction to the answer's
  provenance, not a reopening of the question. If you add a line there, say that and nothing more.
- One docs-only commit. Nothing in `linalg/`, nothing in aikit.

## Worth knowing

`~/claude-mailbox/to-cowork/2026-08-13-linux-AK6-committed-to-uncommitted-citations.md` is the same
failure class from nine days earlier: a committed doc in one repo citing uncommitted state in
another. If the fix here suggests a cheap check that would have caught both, mention it in the
commit message — but don't build tooling for it under this prompt.
