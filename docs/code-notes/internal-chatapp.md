# internal/chatapp: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/chatapp`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestExactPrefillFlag_realBinaryParses

Moved from `internal/chatapp/fitflag_test.go` (the comment above `TestExactPrefillFlag_realBinaryParses`) on 2026-10-09.

```text
TestExactPrefillFlag_realBinaryParses is M-26 (audit-2026-09-10): chatapp had no --exact-prefill
flag at all despite docs/completed/task-prefill-gap.md documenting it as existing on this REPL.
Same discipline as TestFitFlag_realBinaryAcceptsOff: proves the flag is actually REGISTERED on
the real flag.CommandLine (a typo'd flag.Bool name would make this exit 2 with "flag provided
but not defined"), not just present in source. --version exits before touching a model.
```

## TestChatImage_demoEndToEnd

Moved from `internal/chatapp/image_test.go` (the comment above `TestChatImage_demoEndToEnd`) on 2026-10-09.

```text
TestChatImage_demoEndToEnd is the one-line demo, on the real checkpoint: the real goinfer-chat binary (pure Go, CPU,
int4: about 75 s, most of it the vision tower) on the committed rendered invoice (NOT a real scan) with the committed
schema. Heavy: needs GOINFER_HEAVY_TESTS=1 and ~/models/glm-ocr, and skips cleanly without either. The assertions are loose
on purpose (the exact text is int4-sensitive): stdout is JSON with every schema key, the invoice number is the one printed
at the top of the page, and the total is 1140.55; "every key present" is the grammar's contract, the two values are the
model reading the page.
```

## TestLoadFlags_realBinaryParses

Moved from `internal/chatapp/loadflags_test.go` (the comment above `TestLoadFlags_realBinaryParses`) on 2026-10-09.

```text
TestLoadFlags_realBinaryParses: chat had no --ctx, --stream-weights or --moe-cache-experts/
--moe-cache-slots, and a cold-user run reached for --moe-cache-experts and got "flag provided but
not defined". They are internal/loadflags' now, registered for both binaries; this proves the REAL
flag.CommandLine carries them (--version exits before touching a model, the same discipline as
TestExactPrefillFlag_realBinaryParses).
```

## TestRepl_plainWhenScripted

Moved from `internal/chatapp/plain_test.go` (the comment above `TestRepl_plainWhenScripted`) on 2026-10-09.

```text
R26 (docs/tasks/task-first-hour.md): piped stdin used to print a banner, a `you>` label, ANSI escapes and a trailing `bye`. Through the REPL loop
itself (the caller of every one of those prints) a scripted session must print the answer and nothing else; an interactive one is unchanged.
```

## TestServeOnlyInvocation

Moved from `internal/chatapp/serveonly_test.go` (the comment above `TestServeOnlyInvocation`) on 2026-10-09.

```text
Cold-user run 2026-09-06, scenario B, 06:37:23 — the tester's FIRST error of that leg, and the
reason it started badly. `goinfer-chat serve` ignored the subcommand and complained about
--model; `goinfer-chat -web` said "flag provided but not defined: -web". Neither said the two
belong to a different binary, and the README's own examples used them, so the tester concluded
the released binary was broken and went hunting through pkg.go.dev for a server.
```

## TestServeOnlyInvocation.shared

Moved from `internal/chatapp/serveonly_test.go` (the comment above `TestServeOnlyInvocation.shared`) on 2026-10-09.

```text
--stream-weights was serve-only and redirected here; since internal/loadflags it is chat's own
flag too, so it must now fall through to chat (the cold-user run's dead end, closed the other way).
```

## TestServeOnlyInvocation.flagvalue

Moved from `internal/chatapp/serveonly_test.go` (the comment above `TestServeOnlyInvocation.flagvalue`) on 2026-10-09.

```text
N-77 (docs/audit-2026-09-10.md): every argv token used to be checked uniformly, with no
notion of "this token is a FLAG'S VALUE, not a flag or subcommand" — a chat flag's value
that happens to equal a serveOnly word (most plausibly "serve" itself, via --system)
wrongly triggered the redirect.
```

## TestStreamGen_matchesWholeSequenceDecode

Moved from `internal/chatapp/streamgen_test.go` (the comment above `TestStreamGen_matchesWholeSequenceDecode`) on 2026-10-09.

```text
TestStreamGen_matchesWholeSequenceDecode is P-17's deferred demo half (audit-2026-09-10):
streamGen decodes each token INCREMENTALLY (DecodePiece, appended to a strings.Builder) instead
of re-decoding the whole generated slice every token — proven here against a REAL tokenizer by
comparing the concatenation of every flushed chunk to tk.DecodeContinuation(ids), the
established whole-sequence reference for a CONTINUATION (not a fresh sequence — see DecodePiece
and DecodeContinuation's own doc comments on why Decode's leading-space strip does not apply
here). TestDecodeContinuation_isIncrementallyAssociative (tokenizer/) proves the underlying
per-piece concatenation property directly; this proves THIS function's own cursor/flush
bookkeeping composes with it correctly, end to end, with no model needed (streamGen takes the
token channel directly).
```

## TestJSONMasker_reusesCachedTokenBytes

Moved from `internal/chatapp/tokenbytes_test.go` (the comment above `TestJSONMasker_reusesCachedTokenBytes`) on 2026-10-09.

```text
TestJSONMasker_reusesCachedTokenBytes is P-17's demo half (audit-2026-09-10): jsonMasker used to
rebuild the constraint masker's token→bytes table (constrain.TokenBytes(s.vocab,
s.tk.TokenText)) on every constrained turn — the same O(vocab) rebuild serve's own
cachedTokenBytes exists to avoid. newSession now builds it once into s.tokenBytes instead, and
jsonMasker must read that cached field rather than touching the tokenizer again.

Proven by nil-ing s.tk AFTER building s.tokenBytes by hand (mirroring newSession's own one-line
wiring) and confirming jsonMasker still runs: if it still called s.tk.TokenText, this would
panic on the nil pointer instead of silently passing. No committed fixture pairs a real
tokenizer with a loadable model (internal/serveapp/prefillpath_test.go's tinyServed notes the
same gap: "the committed tiny GGUF carries no embedded tokenizer"), so newSession's own
construction isn't exercised end-to-end here — its wiring is the one-line
`s.tokenBytes = constrain.TokenBytes(...)` this test's setup mirrors exactly.
```

## TestJSONMasker_holdsBackTemplateStopIDs

Moved from `internal/chatapp/tokenbytes_test.go` (the comment above `TestJSONMasker_holdsBackTemplateStopIDs`) on 2026-10-09.

```text
TestJSONMasker_holdsBackTemplateStopIDs is N-71 (docs/audit-2026-09-10.md): jsonMasker only
held back EOS/EndOfTurn, not the chat template's own turn-stop ids (s.stopIDs) — Llama-3's
<|eot_id|> and harmony's <|end|> are neither, so a constrained generation for those families
could emit the real stop token mid-document instead of it being masked. A held-back id must
be masked to -Inf before the grammar can end (an empty document is not valid JSON yet, so
CanEnd is false here) — internal/serveapp/openai.go's own masker already unions eosIDs with
stopIDs; this pins the demo doing the same.
```

## TestJSONMasker_warnsOnUnreachableSchemaCompileFailure

Moved from `internal/chatapp/tokenbytes_test.go` (the comment above `TestJSONMasker_warnsOnUnreachableSchemaCompileFailure`) on 2026-10-09.

```text
TestJSONMasker_warnsOnUnreachableSchemaCompileFailure is N-78 (docs/audit-2026-09-10.md):
jsonMasker's fallback for a schema that fails to compile used to be SILENT — main's flag
parsing already fail-fasts (os.Exit) on an invalid --schema before s.schema is ever set, so in
practice this branch never fires via the CLI, but a silent downgrade is still the wrong shape
for the day something else sets s.schema without going through that check. This drives the
branch directly (bypassing the CLI's fail-fast, which is the whole point) by constructing a
session with intentionally-invalid schema bytes, and asserts the fallback is now visible on
stderr rather than swallowed.
```

## TestEmbedBuild_versionReportsWhatWasActuallyBaked

Moved from `internal/chatapp/version_prequant_test.go` (the comment above `TestEmbedBuild_versionReportsWhatWasActuallyBaked`) on 2026-10-09.

```text
R6 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): the release's embedded-tier chat
binaries are baked at a FIXED quant chosen at build time (cmd/prequant's own default), and
never read the --quant flag at all — but --help's shared --quant text says "Default int4"
regardless of build. This is the "help text default equals runtime default" gate, taken on
the --version line rather than rewriting the shared flag text: it must state whatever was
actually baked in, not the --model flag's unrelated default.

Requires -tags prequant AND internal/chatapp/model.giw staged (build-embed.sh's own build
input, gitignored, not committed) — that is why this file is build-tag gated rather than
part of the always-built version_test.go: a plain `go test ./...` must not need a 600+ MB
local asset to pass.
```

## TestEmbedBuild_versionReportsWhatWasActuallyBaked.mutation

Moved from `internal/chatapp/version_prequant_test.go` (the comment above `TestEmbedBuild_versionReportsWhatWasActuallyBaked.mutation`) on 2026-10-09.

```text
The mutation this guards against: silently falling back to the unrelated --quant
FLAG's default ("int4") when the embed build actually ignores that flag entirely —
the exact discrepancy the cold-user run found (help said "Default int4", the
binary ran int8int8).
```

## TestLoadEmbedded_rejectsLoRAInsteadOfSilentlyIgnoringIt

Moved from `internal/chatapp/version_prequant_test.go` (the comment above `TestLoadEmbedded_rejectsLoRAInsteadOfSilentlyIgnoringIt`) on 2026-10-09.

```text
N-78 (docs/audit-2026-09-10.md): decoder.NewModel (what loadEmbedded's prequant path calls,
unlike loadFromPath's decoder.Load) takes no *decoder.Options, so there is no chokepoint that
could ever merge opts.LoRA into a .giw bundle's already-serialized weights. --lora used to be
silently dropped instead of rejected. This drives loadEmbedded directly with a real (if tiny,
gitignored) staged model.giw present — see the file-level comment on why that asset is
required at all under -tags prequant — but relies on the LoRA check firing BEFORE giw.Read, so
the stub's bytes are never actually parsed as a bundle.
```

## TestIsVersionArg_recognizesAllForms

Moved from `internal/chatapp/version_test.go` (the comment above `TestIsVersionArg_recognizesAllForms`) on 2026-10-09.

```text
R6 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): `goinfer-chat --version` was
unanswerable — "flag provided but not defined: -version" — and a bare `version` positional
(a plausible typo reaching for it) was silently swallowed and started an interactive chat
session with the embedded model instead of erroring or printing anything.
```

## TestChatVersionFlag_answersWithoutAModel

Moved from `internal/chatapp/version_test.go` (the comment above `TestChatVersionFlag_answersWithoutAModel`) on 2026-10-09.

```text
The mutation this guards against: deleting the dispatch check makes this go red with exactly
v0.17.0's own error text, "flag provided but not defined: -version".
```

## TestChatVersionFlag_answersWithoutAModel.positions

Moved from `internal/chatapp/version_test.go` (the comment above `TestChatVersionFlag_answersWithoutAModel.positions`) on 2026-10-09.

```text
Both spellings, and both positions (bare first arg, and the registered flag after one
that would otherwise require a model) — R6 found the first broken and the second untested.
```

## TestChatUnknownPositional_namesTheSubcommands

Moved from `internal/chatapp/version_test.go` (the comment above `TestChatUnknownPositional_namesTheSubcommands`) on 2026-10-09.

```text
The other half of R6: a bare unrecognized positional must error and name the real
subcommands, not fall through into loading a model. The mutation this guards against:
removing the flag.Args() check reproduces v0.17.0's silent fallthrough exactly — no error,
exit 0, and (on an embed build) an interactive session started on a typo.
```
