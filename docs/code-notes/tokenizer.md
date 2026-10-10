# tokenizer: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `tokenizer`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## Tokenizer.PreTokenizerDecline

Moved from `tokenizer/sentencepiece.go` (the comment above `Tokenizer.PreTokenizerDecline`) on 2026-10-10.

```text
PreTokenizerDecline reports why this tokenizer's pre-tokenizer is NOT the alternation the walker
implements, or "" when it is (or when the family is not byte-level at all).

It exists because the answer used to be nothing. splitGPT2 is exactly the cl100k/Llama-3
alternation, and every byte-level family was walked with it: a `Split` regex of a different shape,
or a GGUF `tokenizer.ggml.pre` outside a four-name switch, produced a DIFFERENT id stream from
HF and from llama.cpp with no error and no log — `count_tokens` and usage drifting by the same
amount (audit-2026-09-02 C-10). Measured on this machine's own assets: gpt-oss's GGUF is
pre="gpt-4o" and Qwen3.5's is pre="qwen35", and neither was in the switch.

A caller that cares — serve's startup line, a gate — can now say so out loud. Nothing here
changes what the walker does: naming the divergence is separable from fixing it, and shipping a
walker for a pattern this repo cannot yet check against a reference would be the worse half.
```

## Chat-template selection

Moved from `tokenizer/sentencepiece.go` (the comment above `tokenizerJSON`) on 2026-10-10.

```text
Chat-template selection moved to the chat package: chat.Detect fingerprints
ChatTemplate() (or falls back to Has() vocab markers) and returns a native
renderer. (The old ChatStyle enum/heuristic lived here.)
```

## parseTokenizerJSON: the SentencePiece chat template

Moved from `tokenizer/sentencepiece.go` (the comment in `parseTokenizerJSON`) on 2026-10-10.

```text
The chat template lives in tokenizer_config.json (or chat_template.jinja) beside
tokenizer.json, as for the byte-level families. This path never read it, so every
SentencePiece checkpoint loaded from a directory reached chat.Detect with no template:
Gemma survived on the vocab heuristic, but Phi-3 and Mistral fell to raw completion
(found 2026-09-25). siblingDir == "" is the blob load, which reads no siblings (M-14).
```

## initGemma: BOS/EOS spellings

Moved from `tokenizer/sentencepiece.go` (the comment in `initGemma`) on 2026-10-10.

```text
BOS/EOS are required under either SentencePiece spelling: Gemma's "<bos>"/"<eos>", or
Llama-2's "<s>"/"</s>", which Phi-3 and Mistral use. Requiring the Gemma names alone made
every Llama-style tokenizer.json unloadable (Phi-3 safetensors could not tokenize at all,
found 2026-09-25). Pad is optional: Llama-style vocabs have none (-1).
```

## Tokenizer.mergeSymbols

Moved from `tokenizer/sentencepiece.go` (the comment above `Tokenizer.mergeSymbols`) on 2026-10-10.

```text
mergeSymbols is the shared BPE core: repeatedly merge the lowest-rank
adjacent pair (leftmost on ties) until no adjacent pair has a known rank.
Both families call it; only the initial symbol construction (per-rune +
byte-fallback vs byte-level) and the id mapping around it differ. The merge
table itself is identical HF data, so the merge loop is too.
mergeSymbols applies BPE merges to syms in ascending rank order, leftmost first
on a tie. Behaviorally identical to the naive "rescan for the globally best pair
each step" (the golden-parity tests gate this), but O(n log n) via a min-heap over
a doubly-linked list instead of O(n²): Gemma has no pretokenizer, so a whole
inter-added-token gap arrives here as ONE unit, and the old rescan turned a few
hundred KB of client text into minutes of CPU (M28).
```

## buildScoreRank

Moved from `tokenizer/sentencepiece.go` (the comment above `buildScoreRank`) on 2026-10-10.

```text
buildScoreRank turns per-id SentencePiece scores into a merge-priority rank: rank 0 is the
highest-scoring token, so a lower rank merges first (matching mergeSymbols' min-heap). Tokens with
the SAME score share a rank, so on a score tie the heap key's leftIndex — not the token id —
decides, and two equal-score merges at different positions fire left-to-right, matching llama.cpp's
leftmost-position SPM order. (Previously every id got a distinct rank via a lower-id tiebreak, so a
same-score merge on a lower id fired ahead of a leftward one — a silent divergence from the
reference tokenization on a vocab with equal-score competing merges: audit R-29.)
```

## Tokenizer.DecodeContinuation

Moved from `tokenizer/sentencepiece.go` (the comment above `Tokenizer.DecodeContinuation`) on 2026-10-10.

```text
DecodeContinuation decodes ids that CONTINUE an existing sequence rather than
forming one, so the SentencePiece dummy-prefix strip does NOT apply: that space
belongs to the first token of the whole sequence, and these ids are not it.

M-25: the serving loop decoded generated ids with Decode, which strips. On a
dummy-prefix family (Llama-2/Mistral) a generation whose first token is `▁Paris`
reached the client as "Paris" where OpenAI and llama.cpp both return " Paris".
Decoding prompt+generation together gives the right answer too, by re-decoding the
prompt every token; this is the same correction without that cost.

Byte-level tokenizers never strip, so this is identical to Decode for them.
```

## Tokenizer.TokenText

Moved from `tokenizer/sentencepiece.go` (the comment above `Tokenizer.TokenText`) on 2026-10-10.

```text
TokenText returns the raw surface bytes a single token id contributes when
decoded — the per-token building block for byte-level constrained decoding
(mapping the vocab onto a grammar). Unlike Decode it does NO whole-sequence
post-processing: no SentencePiece leading-space strip, and no fusing of
adjacent byte-fallback pieces. A byte-level piece is mapped through the byte
decoder; a SentencePiece byte-fallback token yields its single raw byte; a
normal SentencePiece piece has ▁ mapped to a space. Special tokens render as
their literal surface form (so a grammar that forbids them masks them out).

That parenthetical is only true where the surface is grammar-illegal in context — e.g.
outside a string, where a literal '<' breaks JSON syntax. INSIDE a JSON string value, a
special token's surface is typically ordinary printable text (no '"' or backslash), which is
plain-string-legal content, so it is NOT masked out there. See N-79
(docs/audit-2026-09-10.md, investigated 2026-09-16, deferred) and constrain/constrain.go's
maskID for the fuller explanation and why fixing it needs new plumbing, not a local patch.
An out-of-range id returns nil.
```

## Tokenizer.initByteLevel: pre-tokenizer shape

Moved from `tokenizer/bytelevel.go` (the comment in `Tokenizer.initByteLevel`) on 2026-10-10.

```text
C-10: compare the declared regex against the shape the walker actually implements. A Digits
pre-tokenizer (Mellum2) carries no Split regex and is handled by splitDigits, so an empty
regex is not a mismatch — only a regex that IS present and is a different alternation.

V-15 (docs/review-2026-09-04.md): a bare (non-Sequence) `{"type":"ByteLevel",
"use_regex":true}` pre_tokenizer — a real HF `gpt2` export's actual shape
(testdata/gpt2/onnx/tokenizer.json), no separate Split node at all — also carries an empty
splitRegex, but "empty regex" here does NOT mean "no opinion" the way Mellum2's Digits case
does: use_regex:true on a bare ByteLevel is HF's own way of saying "use my built-in GPT-2
regex". Treating it as an unclassified no-op silently walked it with the cl100k alternation
instead — no error, no PreTokenizerDecline, just wrong ids on exactly the inputs (` 2020`,
a `\r\n` after punctuation) where the two shapes diverge.
```

## Tokenizer.splitPre

Moved from `tokenizer/bytelevel.go` (the comment above `Tokenizer.splitPre`) on 2026-10-10.

```text
splitPre applies the pre-tokenizer alternation this tokenizer actually declares.

It used to be splitGPT2 unconditionally — one alternation for every byte-level family, which is
audit-2026-09-02 C-10. The shape is classified from the model's own `Split` regex at load, so
this dispatches on what the file says rather than on which family someone assumed it was.

A shape with no walker keeps the cl100k one AND sets PreTokenizerDecline, because a wrong split
that says so is strictly better than a wrong split that does not — and refusing to load a model
that works today, imperfectly, would be a worse trade than reporting it.
```

## isBareByteLevelUseRegex

Moved from `tokenizer/bytelevel.go` (the comment above `isBareByteLevelUseRegex`) on 2026-10-10.

```text
isBareByteLevelUseRegex reports whether pre_tokenizer is a TOP-LEVEL (not Sequence-wrapped)
`{"type":"ByteLevel", "use_regex":true}` node with no separate Split component — HF's own way
of saying "use my built-in GPT-2 regex" rather than "no opinion". A real HF `gpt2` export takes
exactly this shape (testdata/gpt2/onnx/tokenizer.json carries no `pattern.Regex` anywhere), so
splitRegex above returns "" for it — the same empty result a genuinely regex-agnostic
pre-tokenizer (Mellum2's Digits+ByteLevel Sequence) produces. The `type` check is what tells
them apart: a Sequence's own top-level type is "Sequence", not "ByteLevel", so this is false for
Mellum2's shape without needing to inspect its sub-pretokenizers at all (V-15,
docs/review-2026-09-04.md).
```

## readTokenizerConfig: chat_template.jinja

Moved from `tokenizer/bytelevel.go` (the comment in `readTokenizerConfig`) on 2026-10-10.

```text
Recent transformers save the chat template as its own file beside tokenizer_config.json
(and drop the key from it). Without this fallback such a checkpoint reached chat.Detect with
no template at all: it fell through to the vocab heuristic (mellum2 → generic chatml) or to
raw completion (Granite-4.0-H, Laguna XS.2), and a released SmolLM3/Olmo 3 shipped that way
would bypass the M-36 fingerprints and render as plain chatml (docs/tool-call-coverage.md,
finding 1). The JSON key still wins when both are present, as transformers resolves it.
```

## GGUF tokenizer: file header

Moved from `tokenizer/gguf.go` (the comment at the top of `tokenizer/gguf.go`) on 2026-10-10.

```text
GGUF tokenizer (G7 follow-up). A .gguf checkpoint carries its tokenizer in
metadata — the vocab (tokenizer.ggml.tokens, id == array index), the BPE
merges (tokenizer.ggml.merges, space-joined), per-token types, and the
special-token ids — so LoadGGUF builds a Tokenizer with no sidecar
tokenizer.json, letting a bare .gguf tokenize and chat end-to-end.

Scope: the SentencePiece-style byte-fallback family (tokenizer.ggml.model ==
"llama" — Llama-2/Mistral/TinyLlama and friends), which maps onto the same
modeGemma merge-rank core with the ▁ dummy prefix. The merges live in
metadata, so tokenization is merge-rank (not score-based) and reuses the
shared BPE loop verbatim. The byte-level family ("gpt2" — Llama-3/Qwen/GPT-2)
is a follow-up: same machinery as modeByteLevel, but its pretokenizer knobs
(digit-run cap, NFC) come from tokenizer.ggml.pre and want a committed
byte-level GGUF to parity-gate, which testdata doesn't have yet.

Metadata reference: the GGUF spec (ggml-org/ggml).
```

## setupGGUFByteLevel: unknown pre

Moved from `tokenizer/gguf.go` (the comment in `setupGGUFByteLevel`) on 2026-10-10.

```text
C-10: an unrecognised `pre` used to fall silently to GPT-2-like defaults, and the two
families this repo ships that do that are not hypothetical — measured on the local
assets, gpt-oss-20b-MXFP4.gguf is pre="gpt-4o" and Qwen3.5-35B-A3B-Q4_K_M.gguf is
pre="qwen35". A GGUF carries no Split regex, so there is nothing to classify here; the
name is all the evidence there is, and an unknown name means unknown knobs.
```

## byteLevelKnobs: gpt-4o

Moved from `tokenizer/gguf.go` (the comment in `byteLevelKnobs`) on 2026-10-10.

```text
o200k. A GGUF carries no Split regex, so a name→shape mapping is exactly the guess that
produced C-10 — this one is not a guess. Measured: gpt-oss-20b-MXFP4.gguf declares
pre="gpt-4o", and the SAME MODEL's HF tokenizer.json (~/models/gpt-oss-20b-hf) declares a
Split regex byte-identical to the o200k pattern splitO200k implements — sha256 2d1b8dc1…
on both that file's regex and this repo's test constant for it. Digits cap at 3 in that
pattern, and it ships no normalizer.
```

## byteLevelKnobs: qwen35

Moved from `tokenizer/gguf.go` (the comment in `byteLevelKnobs`) on 2026-10-10.

```text
Qwen3.5's GGUFs. Same family and the same tokenizer.json pipeline as qwen2 — NFC on, one
digit — and it was NOT in this switch, so it fell to the default whose only difference is
NFC OFF. That diverges on exactly the inputs needing normalisation and on nothing else,
which is why it went unnoticed. Measured: Qwen3.5-35B-A3B-Q4_K_M.gguf is pre="qwen35".
```

## byteLevelKnobs: dbrx

Moved from `tokenizer/gguf.go` (the comment in `byteLevelKnobs`) on 2026-10-10.

```text
R8 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): granite-4.0-h-tiny — a
registry-recommended checkpoint (pull/registry.go) — declared pre="dbrx" and fell
through to this switch's unknown-pre default, so every `pull granite-4.0-h-tiny` user
got a PreTokenizerDecline warning on a checkpoint this project was actively
recommending. Measured, not guessed, same discipline as C-10: ibm-granite/
granite-4.0-h-tiny's HF tokenizer.json (the safetensors source for this project's own
GGUF pin) declares a Split regex byte-identical to the cl100k pattern splitGPT2/
shapeCl100k implements, "\p{N}{1,3}" digit runs (Llama-3's cap, not Qwen's single
digit): "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" (fetched 2026-09-07, HF repo sha
791e0d3d28c86e106c9b6e0b4cecdee0375b6124). Its `normalizer` is null (normOn=false,
matching llama-bpe) and `model.ignore_merges` is `false` — UNLIKE llama-bpe
(ignoreMerges=true), which is otherwise the same shape+digit-cap; that is the one knob
that makes this its own case rather than an alias for "llama-bpe".
```

## byteLevelKnobs: gpt-2 and default

Moved from `tokenizer/gguf.go` (the comment in `byteLevelKnobs`) on 2026-10-10.

```text
GPT-2's OWN alternation is not the cl100k one either (no contraction clause, ` ?\p{N}+`
rather than a capped run) — shapeGPT2Original implements it directly (split_gpt2orig.go,
its own golden-tested file), not a fallback of unknown correctness (N-72,
docs/audit-2026-09-10.md corrects an earlier, more hedged version of this comment).

"default" is grouped in here too, but it should NOT be: llama.cpp's own source
(llama-vocab.cpp, fetched directly 2026-09-16) shows LLAMA_VOCAB_PRE_TYPE_DEFAULT falls
to that switch's OWN default: case, whose regex_exprs is FOUR separate patterns —
leading punctuation (`[\p{P}\$\+<=>\^~\|]+`), THEN the same alternation GPT-2 uses, THEN
unbounded digit runs (`\p{N}+`), THEN a fixed 3-digit grouping (`[0-9][0-9][0-9]`) — a
materially different, multi-pass shape from GPT-2's single pattern. A pre="default" (or
absent) GGUF is walked with GPT-2's shape here, a shape the file never actually declared.
Confirmed, not "plausible, from llama.cpp as remembered" as the earlier version of this
finding put it — but implementing the real DEFAULT shape (a new splitShape, its own
goldens) is real new work, deferred rather than rushed into this comment fix.
```

## The pre-tokenizer shape problem

Moved from `tokenizer/splitshape.go` (the comment at the top of `tokenizer/splitshape.go`) on 2026-10-10.

```text
The pre-tokenizer SHAPE problem (audit-2026-09-02 C-10).

splitGPT2 is exactly one regex: the cl100k / Llama-3 alternation, parameterised only by the
digit-run cap. Nothing ever compared a tokenizer's ACTUAL `Split` regex against that shape, and a
GGUF's `tokenizer.ggml.pre` outside a four-name switch fell to a default that is still that
walker. So a family whose pre-tokenizer differs was tokenized by the wrong one, silently: no
error, no log, and `count_tokens` and usage drift by the same amount.

Measured on this machine's own assets, which is what turned the audit's "medium confidence on the
exact pre strings" into a fact:

	gpt-oss-20b-MXFP4.gguf       pre="gpt-4o"   -> not in the switch -> default
	Qwen3.5-35B-A3B-Q4_K_M.gguf  pre="qwen35"   -> not in the switch -> default
	Qwen3 / qwen2.5-coder        pre="qwen2"    -> recognised

Both unrecognised ones are families this repo ships AND gates. The gpt-oss case is the audit's;
the qwen35 case is not in the audit, and it is the quieter of the two — "qwen35" falls to a
default that differs from the "qwen2" branch only in NFC being OFF, so it diverges on exactly the
inputs that need normalising and on nothing else.

This file does not guess a walker for an unknown shape. It NAMES the shape, so a mismatch can be
reported instead of silently mis-tokenized.
```

## splitO200kVariant: the optional prefix backtracks

Moved from `tokenizer/split_o200k.go` (the comment in `splitO200kVariant`) on 2026-10-10.

```text
Alts 1 and 2: an optional leading non-(CRLF/letter/number), then a word. Alt 1 is
upper* lower+ ; alt 2 is upper+ lower*. Trying them in order is what splits a case
transition: `CamelCase` matches alt 1 as `Camel`, leaving `Case`.

THE OPTIONAL PREFIX BACKTRACKS. `X?` is greedy but the engine gives it up when the rest
of the alternative cannot match, so each alternative is tried WITH the prefix and then
WITHOUT it before moving on. A first cut committed to the prefix once and shared it
between both alternatives, which mis-split a combining mark followed by an uppercase
letter: it consumed the mark as the prefix, failed to find a lower run after `É`, and ran
on into alt 2 with the prefix already spent, emitting `◌́É` where the pattern gives
`◌́` then `É`. The differential oracle found it; no hand-written case would have.
```

## splitO200kVariant: alt1 backtracking

Moved from `tokenizer/split_o200k.go` (the comment in `splitO200kVariant`) on 2026-10-10.

```text
alt1 matches [Lu Lt Lm Lo M]* [Ll Lm Lo M]+ at j — and the two classes OVERLAP on
{Lm, Lo, M}, so the greedy star has to be able to give runes back. Take the longest
upper-ish run, then shrink it until the lower-ish run can match at least one, which is
what a backtracking engine does and is the first success in ITS order.

A shortcut here (stopping the star at the first rune that is also lower-ish) was wrong on
exactly the overlap: a combining mark before uppercase, `◌́ΩÉéé`, matched only the mark
where the pattern matches the whole word. Found by the differential oracle, twice — this
is the second time the shortcut looked obviously right.
```
