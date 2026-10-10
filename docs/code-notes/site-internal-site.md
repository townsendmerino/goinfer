# site/internal/site: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `site/internal/site`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestBuildDownload_sampleIsDarwinArm64

Moved from `site/internal/site/docs_download_test.go` (the comment above `TestBuildDownload_sampleIsDarwinArm64`) on 2026-10-09.

```text
The live page's example used darwin-amd64 because that asset came first in the release. The sample is the Apple
silicon file whenever the release has one, wherever it is listed.
```

## TestCheckOllama_theRealDataHolds

Moved from `site/internal/site/ollama_test.go` (the comment above `TestCheckOllama_theRealDataHolds`) on 2026-10-09.

```text
The real data holds: every row transcribes the snapshot, every derived status equals the snapshot's, the shares
recompute to its Result table, and the headline is a cited Fact. The three vision families the rule depends on come
out supported, which is why the rule reads modality and not only tasks (gemma3's tasks do not say vision).
No real row is T since the 2026-10-09 snapshot (mistral-small3.2 gained its images); the T derivation is pinned by
TestCheckOllama_refuses' "the matrix moved (qwen3_5 loses its images)" case.
```

## TestCheckOllama_refuses.headline

Moved from `site/internal/site/ollama_test.go` (the comment above `TestCheckOllama_refuses.headline`) on 2026-10-09.

```text
the first percentage in the headline is the supported share, whatever the current snapshot says: a refresh
must not break this case by changing the number (it hard-coded 92.4% until the 2026-10-02 snapshot)
```

## TestCheckClaims_refusesDrift

Moved from `site/internal/site/site_test.go` (the comment above `TestCheckClaims_refusesDrift`) on 2026-10-09.

```text
The claims check has to be able to fail. Each case corrupts one claim and must be refused, naming it. The first is
the mockup's own error: it showed the 1.5B on CUDA at 253.1 tok/s, the first of three runs, where the record's median
is 252.9.
```

## TestVerify_acceptsATitleWithAnApostrophe

Moved from `site/internal/site/writeup_test.go` (the comment above `TestVerify_acceptsATitleWithAnApostrophe`) on 2026-10-09.

```text
The templates escape an apostrophe as &#39;, so the output check must compare escaped titles. A title like "Batching that
doesn't change the answer" failed the build before this (found while drafting the first batch).
```
