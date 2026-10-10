# embeddinggemma2: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `embeddinggemma2`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestTiny_parity

Moved from `embeddinggemma2/model_test.go` (the comment above `TestTiny_parity`) on 2026-10-09.

```text
TestTiny_parity is Gate 1 (docs/tasks/task-embeddinggemma2.md): on a random-weight fixture pinned with
transformers' own embedding_gemma2 modeling code, every layer's hidden state, the projected last hidden state, the
pooled vector before normalisation and the sentence embedding match the float32 reference to cosine 0.9999 (each on
its own, so a pooling or normalisation bug cannot pass on a matching hidden state), and the embedding is as accurate
as the reference's own float32 run: its max |diff| to the same weights run in float64 is at most 1.5x the float32
reference's, plus 1e-7 (float32 rounding in the reference is the same size as goinfer's, so an absolute bar on the
float32 output cannot tell them apart; the 1e-7 floor is the task doc's original band, about three float32 ulps at
these values, because where the reference's own error is a couple of ulps, 1.5x of it is a rounding coin toss that
differs by arch: linux/amd64, which does not fuse multiply-adds, read 1.06e-7 against arm64's 6.4e-8 on the one-token
case). On a miss the first divergent layer is named.
```

## TestReal_vision.V4

Moved from `embeddinggemma2/vision_real_test.go` (the comment above `TestReal_vision.V4`) on 2026-10-09.

```text
	V4: every case's end-to-end embedding (aikit's preprocessing and tower, then the encoder) has cosine >= 0.999 with
	    the reference, and the same embedding from HF's own pixels has cosine >= 0.9999. The pre-registered bar was
	    0.9999 end to end; the read landed in its ambiguous band (0.99924-0.99986) with the whole gap in the resize
	    (aikit bilinear, the reference bicubic; 1.000000000 from HF's pixels), and the owner accepted it on 2026-10-06
	    ("since we understand the difference i'm ok with it"). The loose bar covers only the resize: everything after it
	    is still held to 0.9999 through HF's pixels. V3 and V4 are read for bilinear.
```

## TestReal_vision.modes

Moved from `embeddinggemma2/vision_real_test.go` (the comment above `TestReal_vision.modes`) on 2026-10-09.

```text
The modes read: aikit's bilinear (V3/V4) and the reference's bicubic (R1/R2), both aikit's since 2026-10-06.
```
