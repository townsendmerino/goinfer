# demo/agent/agent: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `demo/agent/agent`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestAgentWeb_hardening

Moved from `demo/agent/agent/n26_test.go` (the comment above `TestAgentWeb_hardening`) on 2026-10-09.

```text
N-26: agent-web accepted cross-origin POSTs and unbounded bodies, and the session encoded user
text with SPECIAL-TOKEN PARSING — so a user typing "<|im_start|>assistant" into the chat box
promoted those bytes to real role tokens, forging the template's own boundaries.

"Demo-grade" is not a security boundary: it binds a port, and one request can occupy the model
for minutes (a vision turn) or discard the conversation (/api/reset).
```

## TestAgentWeb_hardening.anchor

Moved from `demo/agent/agent/n26_test.go` (the comment above `TestAgentWeb_hardening.anchor`) on 2026-10-09.

```text
The mutating routes must carry BOTH wrappers. GET /api/info is read-only and cheap, so it
is deliberately not wrapped — checking that keeps this from passing by blanket-wrapping.
Anchor on the mux REGISTRATION, not on the route string — which also appears in the
file's doc comment, and matching that made the first version of this guard fail on prose.
```

## TestSpliceImageBlock_imageBlockIsSpecialUserTextIsNot

Moved from `demo/agent/agent/n26_test.go` (the comment above `TestSpliceImageBlock_imageBlockIsSpecialUserTextIsNot`) on 2026-10-09.

```text
TestSpliceImageBlock_imageBlockIsSpecialUserTextIsNot guards V-03 (docs/review-2026-09-04.md):
N-26 moved TurnImage's turn to EncodeSegments (correct — the check above), but the image
placeholder block was glued into the turn's plain text and rendered as part of a non-Special
segment, so it got BPE'd as literal text instead of parsed into the real image tokens, and
FindImageRun always found no run ("image placeholder run = 0"). The serving path hit the
identical gap and was fixed with spliceImageBlock (M-22, internal/serveapp/vision_serve.go);
this pins the same fix ported into this package (unreachable directly — separate module,
unexported).

Mirrors internal/serveapp/vision_hardening_test.go's TestVision_userTextIsNotSpecialButTheImageBlockIs:
asserts segment SHAPE, not token ids, so it needs no tokenizer or model.
```

## TestSpliceImageBlock_usesTheLastOccurrenceNotTheFirst

Moved from `demo/agent/agent/n26_test.go` (the comment above `TestSpliceImageBlock_usesTheLastOccurrenceNotTheFirst`) on 2026-10-09.

```text
TestSpliceImageBlock_usesTheLastOccurrenceNotTheFirst pins V-19 (docs/review-2026-09-04.md) for
this package's ported copy of spliceImageBlock. It used to splice the FIRST non-Special segment
containing the block, anywhere in the rendered history — an earlier turn that happens to contain
the literal block text as ordinary words would get spliced instead of the real current image
turn, reopening the special-token-forging class V-03/M-22 closed. Mirrors
internal/serveapp/vision_hardening_test.go's TestVision_spliceUsesTheLastOccurrenceNotTheFirst.
```

## TestStreamGen_matchesWholeSequenceDecode

Moved from `demo/agent/agent/streamgen_test.go` (the comment above `TestStreamGen_matchesWholeSequenceDecode`) on 2026-10-09.

```text
TestStreamGen_matchesWholeSequenceDecode is P-17's deferred demo half (audit-2026-09-10):
streamGen decodes each token INCREMENTALLY (DecodePiece, appended to a strings.Builder) instead
of re-decoding the whole generated slice every token — proven here against a REAL tokenizer by
comparing the concatenation of every emitted chunk to tk.DecodeContinuation(ids), the
established whole-sequence reference for a CONTINUATION (not a fresh sequence — see DecodePiece
and DecodeContinuation's own doc comments on why Decode's leading-space strip does not apply
here). TestDecodeContinuation_isIncrementallyAssociative (tokenizer/) proves the underlying
per-piece concatenation property directly; this proves THIS function's own cursor/flush
bookkeeping composes with it correctly, end to end, with no model needed — streamGen takes the
token channel and a bare *decoder.Generation directly.
```

## TestSchemaMasker_reusesCachedTokenBytes

Moved from `demo/agent/agent/tokenbytes_test.go` (the comment above `TestSchemaMasker_reusesCachedTokenBytes`) on 2026-10-09.

```text
TestSchemaMasker_reusesCachedTokenBytes is P-17's demo half (audit-2026-09-10): schemaMasker
used to rebuild the constraint masker's token→bytes table (constrain.TokenBytes(s.vocab,
s.tk.TokenText)) on every DECIDE turn — the same O(vocab) rebuild serve's own cachedTokenBytes
exists to avoid. newSession now builds it once into s.tokenBytes instead, and schemaMasker must
read that cached field rather than touching the tokenizer again.

Proven by nil-ing s.tk AFTER building s.tokenBytes by hand (mirroring newSession's own one-line
wiring) and confirming schemaMasker still runs: if it still called s.tk.TokenText, this would
panic on the nil pointer instead of silently passing. No committed fixture pairs a real
tokenizer with a loadable model (internal/serveapp/prefillpath_test.go's tinyServed notes the
same gap), so newSession's own construction isn't exercised end-to-end here — its wiring is the
one-line `s.tokenBytes = constrain.TokenBytes(...)` this test's setup mirrors exactly.
```

## TestSchemaMasker_holdsBackTemplateStopIDs

Moved from `demo/agent/agent/tokenbytes_test.go` (the comment above `TestSchemaMasker_holdsBackTemplateStopIDs`) on 2026-10-09.

```text
TestSchemaMasker_holdsBackTemplateStopIDs is N-71 (docs/audit-2026-09-10.md): schemaMasker only
held back EOS/EndOfTurn, not the chat template's own turn-stop ids (s.stopIDs) — Llama-3's
<|eot_id|> and harmony's <|end|> are neither, so a constrained DECIDE turn for those families
could emit the real stop token mid-document instead of it being masked. A held-back id must be
masked to -Inf before the grammar can end (an empty document is not valid JSON yet, so CanEnd
is false here) — internal/serveapp/openai.go's own masker already unions eosIDs with stopIDs;
this pins the demo doing the same.

The stop id's own surface MUST be a byte the grammar would otherwise accept here (decisionSchema
requires an object, so its first byte is "{") — otherwise grammar-validity masking alone would
mask it regardless of EOS registration, and the test could not tell the fix apart from a no-op.
```
