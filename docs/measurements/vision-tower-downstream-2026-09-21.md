# R8 gate 4: served downstream check — FAILS the registered rule; the fused vision kernel stays opt-in

Pre-registration: `vision-tower-downstream-PREREGISTERED.md` (+ its addendum, written after the first launch showed the served vision path returns no `logprobs`, before any comparison was computed — the metric was switched from token-level to word-level there, before this run). `serve --backend cuda`, real `gemma-3-4b-it`, greedy, `max_tokens=64`, 8 real images (a repo fixture, a JPEG sample, two desktop assets, a wallpaper, three flag PNGs). Raw: `vision-tower-downstream-2026-09-21.json` / `.log`.

## Preconditions: all held
1. **E vs E2** (repeat of arm E on 3 images): word-for-word identical on all 3 — greedy through this pipeline has a real floor.
2. Every arm returned >= 40 words on every image; prompt_tokens = 275 on every request (the tower ran).
3. **Speed**: N (fused) 5.8-6.3 s/image vs E (exact) 27.9-28.5 s — **4.7x on img0**, comfortably past the >=3x precondition.

## Result

| | f_N (differ within first 12 words, of 8) | A_N (mean prefix agreement, words) | A_P (jitter control) | A_N / A_P | whole-text identical |
|---|---:|---:|---:|---:|---:|
| N vs E | **6/8** | 9.75 | 13.00 | 0.750 | 0/8 |

**Registered rule: f_N >= 2 -> FAILS. The fused kernel stays opt-in; nothing defaults.** (A_N/A_P alone, 0.75, would have landed in the ambiguous band — it is the f_N clause that decides it.)

## Reading the texts (a reading, not a metric; it does not rescue the rule)
The eight N descriptions are coherent, on-topic, and materially accurate for every image (correctly identifies: an anime-style illustration of a person at a computer setup, a 3x3 grid of color-channel test images, the Australian flag rendered as a glossy icon, etc.) — no garbled or off-topic output, no repeated tokens, nothing that looks like a wiring defect. The E and N texts for img4/img1/img7 (excerpted in the log) describe the same content in different words from the first sentence on: this looks like ordinary greedy-decoding sensitivity to a small numeric perturbation — a rephrased opening compounds into a different word at every step after — the same shape the P (jitter) control shows against E (A_P = 13, also well short of whole-text identity on 0/8 images). **N's A_N (9.75) is below P's A_P (13.00)**: the fused kernel perturbs greedy generation somewhat MORE than a 1-LSB pixel jitter does, which is the fact the pre-registered ratio (0.75, just under the 0.8 support line) was designed to catch, and it did.

## What this does and does not establish
It does not find a defect: no garbled text, no off-image content, no crash, and it is weaker interference than a random jitter is not measured (0.75x, not e.g. 5x). It is also not a demonstration of safety: n=8 images, greedy only, one prompt, and the metric (word-level, not token-level, because the served vision path has no logprobs) is coarser than gate 3's per-row cosine. **Under the rule fixed before this run, the fused kernel does not clear the bar for a served-generation-based default flip.** Gate 3 (tower-level cosine, 0.96, "ambiguous -> parked") and gate 4 now agree in direction: both land short of their pass line without finding a wiring defect.

## State (unchanged)
`GOINFER_CUDA_VISION_ATTN=bm128` stays opt-in. Default is `attn_img_batched` (exact). If a default flip is wanted despite this, it is the owner's call against this record, not a rule the record itself clears — the same shape as R2's "shipped anyway" decision on the Mac.
