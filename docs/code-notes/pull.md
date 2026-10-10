# pull: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `pull`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## Package pull

Moved from `pull/pull.go` (the comment the package comment of `pull/pull.go`) on 2026-10-09.

```text
Package pull fetches a GGUF checkpoint from HuggingFace onto local disk.

EXPERIMENTAL (docs/api-tiers.md): supported and used by every goinfer front end, but it may
change in any release. It was `internal/modelpull` until 2026-09-02; it is exported because a
library caller embedding goinfer needs the same first step the CLIs and the web UI take, and
re-deriving it is the one part of "get a model" that has no other owner.

It is deliberately the ONLY new capability in the model-pull work: goinfer already
loads a .gguf or .giw (--model), already transcodes a bare .gguf to a sidecar .giw
cache on first use, and already converts one offline (cmd/prequant). The single step
that existed nowhere in the repo was getting the bytes onto disk — before this, a
`grep` for outbound HTTP in Go code returned nothing, and the only fetch anywhere was
a curl in the release workflow. So this package gets bytes onto disk and stops; it
deliberately does NOT load, convert, quantize, or embed, because each of those already
has a tested owner and a second copy would drift from it.

No new module dependency: net/http, crypto/sha256 and encoding/json only, which keeps
the cgo-free single-static-binary property the rest of the project is built around.
```

## CheckAccess

Moved from `pull/pull.go` (the comment above `CheckAccess`) on 2026-10-09.

```text
CheckAccess reports whether the repo is reachable anonymously, naming the reason when it
is not. Called BEFORE any download so a gated repo fails in a second with an actionable
message instead of after a multi-GB 401.

Worth knowing when reading a failure here: gating is far less of a wall for GGUF than it
looks. Upstream originals are frequently gated (google/gemma-3-4b-it and
meta-llama/Llama-3.2-3B-Instruct both report gated="manual"), but the community GGUF
re-uploads that this command actually targets are not — bartowski/google_gemma-3-4b-it-GGUF
and unsloth/Llama-3.2-3B-Instruct-GGUF both report gated=false. So the usual fix is to
point at a GGUF repo, which is what you wanted anyway.
```

## Download

Moved from `pull/pull.go` (the comment above `Download`) on 2026-10-09.

```text
Download streams f from repo into dir, verifying the sha256 HF declared for it, and
returns the final path.

The digest is VERIFIED, not merely printed, because HF hands it over before the transfer:
the tree API's LFS oid is the file's sha256 (confirmed against the digest
.github/workflows/release-assets.yml already pins for the same file). A hash nobody
compares is self-documentation, not a check.

Writes to a .part file and renames only after the digest matches, so an interrupted or
corrupted pull can never leave something at the final path that later looks loadable.
```

## Download: a Range request at EOF

Moved from `pull/pull.go` (the comment in `Download`) on 2026-10-09.

```text
N-70 (docs/audit-2026-09-10.md): resumeFrom landed at or past EOF — the .part file
already covers everything the server has. The common cause is f.Size <= 0 (a non-LFS
file with no declared size): the resume guard above admits ANY existing .part
regardless of completeness when there's no size to compare against, so a prior run
that fetched every byte but was interrupted before this digest-check-and-rename
re-requests a Range starting exactly at EOF next time — 416, not 206/200. Falling into
the default case below used to report a confusing "HuggingFace returned 416" AND leave
the .part in place, so every retry hit the identical 416 forever. h already covers the
whole .part (hashPrefix, above); verify it directly instead of copying a body a 416
response doesn't have.
```

## recordVerifiedDigest

Moved from `pull/pull.go` (the comment above `recordVerifiedDigest`) on 2026-10-09.

```text
recordVerifiedDigest writes path's digest sidecar from a digest Download has JUST verified against the streamed bytes (audit R-18).
Without it the first Resolve after a pull, via cachedIntact, read the whole file again to arrive at the digest the download had already
computed: several seconds for a 5-20 GB model. A no-op when no digest was declared (sum == ""), so nothing unverified is ever recorded.
Best-effort like cachedFileSHA256's own write: a failure only costs the next call a re-hash.
```

## cachedFileSHA256

Moved from `pull/pull.go` (the comment above `cachedFileSHA256`) on 2026-10-09.

```text
cachedFileSHA256 is fileSHA256 behind a sidecar cache keyed on (size, mtime) — P-12
(audit-2026-09-10): cachedIntact re-hashes the WHOLE checkpoint on every serve start (Resolve's
offline check, Download's own cache check), even when the file has not changed since the last
run verified it. A cache hit here means "the file's size and mtime are exactly what they were
when this SHA-256 was computed" — the same staleness signal `make`/rsync use, cheap to check
(one stat) against the cost it avoids (a full read of a multi-GB checkpoint). A mismatch (or a
missing/corrupt sidecar) falls through to the real hash and rewrites the sidecar; a sidecar
write failure is not fatal — it only means the NEXT call re-hashes too, same as today.
```

## resolveOffline

Moved from `pull/resolve.go` (the comment above `resolveOffline`) on 2026-10-09.

```text

V-16 (docs/review-2026-09-04.md): Resolve used to call CheckAccess+List unconditionally before
any cache check at all, so `serve --model demo:1.5b` with the file already cached still failed
to start offline — the opposite of what Resolve's own doc comment promised.
```

## Checkpoint pull: file header

Moved from `pull/checkpoint.go` (the comment at the top of `pull/checkpoint.go`) on 2026-10-09.

```text
A safetensors checkpoint as a pull target (docs/tasks/task-checkpoint-fetch-2026-09.md, P1-P3 and P5). GGUF is one
file; a safetensors checkpoint is a SET: config.json, the tokenizer files, and either model.safetensors or an index
plus the shards it names. decoder.Load already opens such a directory, so this only gets the set onto disk, with the
guarantees a single file has: every file digest-verified, resumable, and nothing at the final path until the whole set
is there. Anonymous only (the doc's §2, option (c), owner 2026-10-03): a gated original is declined by CheckAccess
before any of this runs.
```

## The recommended-checkpoint registry

Moved from `pull/registry.go` (the comment at the top of `pull/registry.go`) on 2026-10-09.

```text
The recommended-checkpoint registry: a short name a person can type, mapped to a checkpoint this
project has actually run.

THE PROBLEM IT SOLVES. `pull owner/repo:quant` works and requires the user to already know three
things — that a GGUF conversion exists, who published it, and which quantization to ask for.
That is knowledge from having spent time on Hugging Face, which is exactly what a first-time
user does not have.

IT DERIVES FROM THE CAPABILITY MATRIX, and that is the whole design. docs/capability-matrix.json
already records which families this project supports and at what parity status; a hand-kept
second list would drift from it, and a registry claiming support the matrix does not back is
worse than no registry. So a checkpoint entry lives ON its family's matrix row, and
TestRegistry_everyEntryTracesToItsFamily fails if one ever names a family that is not there.

NOT A DOWNLOAD SERVICE. Every entry points at Hugging Face and carries a sha256 the fetch
verifies. This project hosts no weights: it costs money, creates an availability obligation
nobody here can meet, and redistribution carries licence questions worth avoiding.

DISTINCT FROM `demo:` TIERS, deliberately. curated.json pins the models that are EMBEDDED in
release binaries and is checked against the release workflow; its own comment says it is "not a
name registry to grow". That is a different question from "which checkpoints do we recommend",
so this does not extend it.
```

## Checkpoint.Kind

Moved from `pull/registry.go` (the comment in `Checkpoint`) on 2026-10-09.

```text
Kind is "" for a single-file GGUF (every entry until 2026-10-08) and KindDirectory for a safetensors checkpoint
fetched as a set (`pull <name>` then runs the same plan-and-verify path as `pull owner/repo:safetensors`, and
refuses when the repo's files no longer match the tree digest this build pins). It is how a vision-language
checkpoint can be recommended at all: the GGUF loader reads no image projector, the safetensors directory
carries the tower (docs/multimodal.md, P9(d)).
```

## Checkpoint.Tools

Moved from `pull/registry.go` (the comment in `Checkpoint`) on 2026-10-09.

```text
Tools records what `internal/servecheck`'s two tools rows measured for this checkpoint
(R11, docs/measurements/cold-user-2026-09-06-nobara-pc.md): "tools, OpenAI" is a
one-function schema, "tools, harness-scale" a dozen-tool schema shaped like a real agent's
— the shape that broke under opencode with a server whose minimal-schema row was green.
From a RECORDED `serve check` run, never guessed (TestRegistry_toolsColumnIsNonEmpty).
```

## TestSelect_realWorldNaming

Moved from `pull/pull_test.go` (the comment above `TestSelect_realWorldNaming`) on 2026-10-09.

```text
TestSelect_realWorldNaming pins the two naming facts that were MEASURED against the live
HF API on 2026-09-02, not assumed — the model-pull design doc explicitly refused to commit
to a matching scheme without checking, and these are what the check found.
```

## TestSelect_exactShardFilenameIsRefusedToo

Moved from `pull/pull_test.go` (the comment above `TestSelect_exactShardFilenameIsRefusedToo`) on 2026-10-09.

```text
An EXACT shard filename took a different branch from the quant selector above and skipped its split
refusal: `pull owner/repo:big-Q4_K_M-00001-of-00003.gguf` downloaded one piece of a checkpoint no loader
can assemble (a cancelled one left a 4.7 GB .part in the cache). Same refusal, same shard list.
```

## TestIntegrationsDoc_everyHarnessTheReadmeNamesHasAPage

Moved from `pull/integrations_doc_test.go` (the comment above `TestIntegrationsDoc_everyHarnessTheReadmeNamesHasAPage`) on 2026-10-09.

```text
R14 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the README named opencode
alongside Claude Code as "a real agent" target for three releases before a recipe for it
existed anywhere in the tree — a cold user had to reconstruct opencode's provider config from
outside knowledge, and that reconstruction cost the whole run's only safety incident. The
README's own claim ("Pointing a real agent (Claude Code, opencode) at it: docs/integrations/")
is the thing that silently went stale; this reads it back and checks the promise against the
directory it names, so a THIRD harness added to that sentence without a matching page fails
here instead of waiting for the next cold-user run to find it.
```

## TestReadme_familyCountMatchesTheMatrix

Moved from `pull/readme_family_count_test.go` (the comment above `TestReadme_familyCountMatchesTheMatrix`) on 2026-10-09.

```text
TestReadme_familyCountMatchesTheMatrix: the README's "N model families" is a claim the generated
capability matrix can check, and it did drift — the README said 35 while docs/capability-matrix.json
(and the site built from it) said 37. The matrix is the source of truth; every README occurrence must
equal its length.
```

## TestRegistry_toolsColumnIsNonEmpty

Moved from `pull/registry_test.go` (the comment above `TestRegistry_toolsColumnIsNonEmpty`) on 2026-10-09.

```text
R11 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): `serve check`'s minimal-schema
tools row passed against a server that a real agent (opencode) then broke under its own,
larger tool schema — the registry recommended a checkpoint with no signal that this could
happen. The bar here is deliberately low (non-empty, not "measured"): recommendedCheckpoints'
own comments require an honest "not yet measured" placeholder rather than a guess for an
entry nobody has run `serve check` against yet, and this only catches the entry that forgot
the field entirely (an empty string) — it cannot tell a real measurement from a placeholder,
which is the one thing a human filling this in has to get right.
```

## TestRegistry_digestsMatchLocalFiles

Moved from `pull/registry_test.go` (the comment above `TestRegistry_digestsMatchLocalFiles`) on 2026-10-09.

```text
TestRegistry_digestsMatchLocalFiles verifies each entry's sha256 and size against a real file,
when one is present. It is the only gate that can catch the failure that actually happened.

MEASURED 2026-09-06: two of the three entries shipped with a FABRICATED digest and a wrong byte
count. Both were hand-typed from a truncated display — the first 16 hex characters were right,
because that is what had been printed, and the remaining 48 were invented. Every format check in
this file passed on them, because a fabricated digest is still well-formed lowercase hex of the
correct length. A digest can only be checked against the bytes it claims to describe.

Skipped when the file is absent, which is most machines — set GOINFER_MODELS_DIR (or keep
checkpoints under ~/models) to run it. A skip here is not a pass and the log says so.
```

## TestRegistry_directoryEntriesAreVisionLanguage

Moved from `pull/registry_test.go` (the comment above `TestRegistry_directoryEntriesAreVisionLanguage`) on 2026-10-09.

```text
A directory entry is a recommendation that carries a vision tower, so it is only worth having if it can be told apart from a
GGUF one everywhere a list is printed, and if at least one exists for each box class the registry promises (P9(d), 2026-10-08).
```

## registry_tokenizer_test.header

Moved from `pull/registry_tokenizer_test.go` (the comment at the top of the file) on 2026-10-09.

```text
R8 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): granite-4.0-h-tiny — a
registry-recommended checkpoint — loaded with "tokenizer.ggml.pre=\"dbrx\" is not a known
pre-tokenizer; falling back to cl100k" on every pull. A registry entry backed by parity gates
cannot ship with a tokenizer this build declines to walk: recommending it to a first-time user
means recommending a checkpoint whose token ids may differ from HF and llama.cpp, silently,
unless they happen to read a startup warning.
```

## TestResolveOffline_verifiedCacheHitNeedsNoNetwork

Moved from `pull/resolve_test.go` (the comment above `TestResolveOffline_verifiedCacheHitNeedsNoNetwork`) on 2026-10-09.

```text
TestResolveOffline_verifiedCacheHitNeedsNoNetwork pins V-16 (docs/review-2026-09-04.md):
when a ref's exact file, size AND digest are all known WITHOUT asking HuggingFace anything
(a demo: ref's shape — File+Pin+Bytes all set from curated.json), an already-cached, matching
copy resolves offline. resolveOffline only ever touches CacheDir + the filesystem, so this
needs no network stub to prove the property — the function has nothing in it that COULD reach
the network.
```

## TestResolve_callsResolveOfflineBeforeCheckAccess

Moved from `pull/resolve_test.go` (the comment above `TestResolve_callsResolveOfflineBeforeCheckAccess`) on 2026-10-09.

```text
TestResolve_callsResolveOfflineBeforeCheckAccess is the wiring guard: resolveOffline works in
isolation (the tests above), but that proves nothing about whether Resolve's actual request
path reaches it BEFORE the network call it exists to skip — the exact shape of gap this
session's audit keeps finding (a helper with a test, and a call site nobody checked).
```

## TestDownload_rangeNotSatisfiableOnAnAlreadyCompletePart

Moved from `pull/resume_test.go` (the comment above `TestDownload_rangeNotSatisfiableOnAnAlreadyCompletePart`) on 2026-10-09.

```text
TestDownload_rangeNotSatisfiableOnAnAlreadyCompletePart is N-70 (docs/audit-2026-09-10.md):
f.Size <= 0 (a non-LFS file with no declared size) admits ANY existing .part into the resume
path regardless of whether it is actually complete — there is no size to compare against. If
a prior run fetched every byte but was interrupted before the digest-check-and-rename, the
next run's Range request starts exactly at EOF and HuggingFace answers 416, not 206/200.
Before this fix that fell into the catch-all default case: a confusing "HuggingFace returned
416" error, AND the .part was never cleared — so every subsequent retry hit the identical 416
forever. This drives that exact shape and asserts it now succeeds instead.
```
