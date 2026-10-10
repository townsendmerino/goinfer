# internal/giw: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/giw`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestBundle_hostileLength_noPanic

Moved from `internal/giw/bundle_test.go` (the comment above `TestBundle_hostileLength_noPanic`) on 2026-10-09.

```text
TestBundle_hostileLength_noPanic is the regression for the cur.take overflow
FuzzGIWRead found: a v2 weights length near maxint64 made c.off+n wrap negative,
slipping past the bound check and panicking the slice. Read must return a typed
error instead.
```

## FuzzGIWRead.header

Moved from `internal/giw/fuzz_test.go` (the comment above `FuzzGIWRead.header`) on 2026-10-09.

```text
Track 2.3 (testing campaign): the .giw bundle frame is goinfer's own untrusted
binary input (cmd/prequant writes it, the demo embeds it). Read's magic /
version / CRC-style guards exist; these targets fuzz PAST them — valid header,
hostile body, truncation at every boundary, and the u64 v2 length path — to
hold the "typed error, never a panic" bar.
```
