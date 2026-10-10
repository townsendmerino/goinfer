# internal/hwcensus: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/hwcensus`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## hardwareCensus.header

Moved from `internal/hwcensus/census_test.go` (the comment above `censusPath`) on 2026-10-09.

```text
H0, the hardware census: a census, deliberately not a verdict, in the shape of decoder/dispatch_census_test.go.

The class: a code path selected by hardware our machines lack. aikit v1.47.1 fixed an AVX-512 VNNI kernel that was
3.2e-3 per logit off the AVX2 path; none of our machines has AVX-512, and it surfaced only because GitHub's runner
pool mixes CPU models. Nothing here can tell whether such a path is correct. What it can do is make sure no
hardware-gated branch lands unseen: every predicate that selects code by hardware must have an entry in
docs/hardware-coverage.json, which says where that path last executed (or that it never has).

So this detects CHANGE. A green result means "every hardware predicate in the source is in the census, and every
census predicate is still in the source". It does NOT mean the paths are tested; the entries' last_executed records
say that, and the never-executed list this logs is the honest summary.

When this goes red: a new predicate needs an entry (what it selects, and where it has run, or [] for never); or an
entry names a predicate the source no longer has, and the entry is stale. Adding an entry with a made-up record
defeats the census: a record names a machine, a date, how it ran, both commits, and the gate that ran.

WHAT IS SCANNED. goinfer's whole tree (all five modules) and, at the versions goinfer's go.mod files pin, aikit's root
module and aikit/gpu: the CPU feature predicates live in aikit's linalg, and the CUDA and Metal device queries partly
in aikit/gpu. A pinned module that cannot be found FAILS the test rather than skipping it: a census that silently left
aikit out would cover nothing, the citation lint's rule for the same situation. go/scanner tokenizes each file, so a
comment never counts and a string literal (an objc selector, a sysctl name, a /proc path) does.

THE PREDICATES ARE DERIVED, NOT LISTED, so a new one enters the census without anyone remembering to add it here:
  - cpu:<name>   a package-level `var has<X> = ...` (aikit's dispatch flags: hasAVX2, hasDotProd, hasF16C, ...)
  - cuda:<name>  any DeviceAttribute<X> identifier (a CUDA device-attribute query)
  - metal:<sel>  an objc selector registered with RegisterName whose name reads as a device capability or limit
    (max*, supports*, recommended*, has*, threadExecutionWidth, currentAllocatedSize, registryID)
  - mem:<probe>@<file>  a memory probe: a sysctl or /proc or cgroup name, or a driver call (MemGetInfo, MemInfo, ...),
    keyed with its file, because two files reading the same name are two different gates; and every backend's
    registered fit-guard probe, mem:probe:<backend>@<file> (decoder.RegisterMemoryProbe)

Keys are by name, never by line, for the dispatch census's reason: a census that cries wolf on every reformat is a
census someone disables.
```
