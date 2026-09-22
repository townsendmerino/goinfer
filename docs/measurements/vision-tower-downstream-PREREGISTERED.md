# PRE-REGISTERED — R8 gate 4: does the fused vision-tower attention change what the model says? (served, real images)

**Written 2026-09-21 BEFORE the run. Not edited after any result was seen.** Record: `vision-tower-downstream-2026-09-21.md`. This is gate 4 of `vision-tower-attn-PREREGISTERED.md`, taken because gate 3 read 0.96 (ambiguous) and the owner leans to default-on but asked for one downstream test first.

## Design
`serve --backend cuda` on the real `gemma-3-4b-it` checkpoint, fresh process per arm, real chat-template + preprocessing + tower + projector + resident decode, greedy (temperature 0), `max_tokens=64`, `logprobs` with top-5, prompt "Describe this image in detail." **8 real images**: `testdata/gemma3_preprocess_image.png`, four PNG/JPGs from this box's system dirs (gutenprint profile.jpg, inkscape start-welcome.png, plasma "Office Worker Konqi.png", a Nobara wallpaper) and three flag PNGs (`_earth_pernefeldt`, `ad`, `au`). Arms, per image:
- **E** `GOINFER_CUDA_VISION_ATTN=exact` (shipped tower attention);
- **N** `=bm128` (the fused kernel);
- **E2** = a second E run on 3 of the images (determinism check);
- **P** = the CONTROL, arm E on the same image with a 1-LSB jitter (each pixel channel +-1 with probability 5%, fixed seed, PNG re-encoded). P is what "a benign, tiny input perturbation" does to greedy generation through this exact pipeline; N is judged against P, not against an imagined zero.

## Preconditions (a failure voids the run, it is not a fail of N)
1. E vs E2 identical on all 3 repeated images (greedy is deterministic here); else the comparison has no floor.
2. All 4 arms return >= 32 tokens on every image and the image tower actually ran in each request (usage prompt tokens > 256).
3. `serve` logs (or the arm's tower time) confirm the arms differ: N's request is >= 3x faster than E's on the first image (proves the env knob reached the tower).

## Metrics, per image, N vs E and P vs E
first-token identical (top-1 token id); first-token top-1 logprob difference; **prefix agreement** = number of leading tokens identical before the first divergence (of 64); whole-text identical.

## Decision rule (fixed now)
Let f_N = images where N's first token differs from E's (of 8), and A_N / A_P = mean prefix agreement of N-vs-E / P-vs-E.
- **Supports default-on:** f_N <= 1 AND A_N >= 0.8 x A_P. (The fused kernel disturbs generation no more than a 1-LSB input jitter, within 20%.)
- **Fails (stays opt-in, mechanism investigated):** f_N >= 2 OR A_N < 0.5 x A_P.
- **Ambiguous (stays opt-in, recorded):** everything between.
I also READ the eight N and E texts and report whether N's descriptions are coherent and about the image; that is a reading, stated as one, not a metric, and it cannot rescue a rule failure.

## What this can and cannot show
n = 8 images is small: it can catch a gross defect (garbled or off-topic description, first-token flips) and place N against the jitter control; it cannot resolve a few-percent quality difference. Gemma 3's vision path only (Qwen2.5-VL's tower is not this kernel). Greedy decoding on one prompt.

---

## ADDENDUM — written 2026-09-21 after launching the first run, BEFORE any E-vs-N or P-vs-E comparison had been computed

The first launch showed the served **vision path returns no `logprobs`** (the text path does; the VL response had `0` logprob entries), so token-id metrics are unavailable. Only N's first ~70 characters were seen before that run was stopped (all eight begin "Here's a detailed description of the image:" / "Okay, ..."), which is why first-token identity would have been uninformative anyway. **Deviation, fixed now:** all token-level metrics are replaced by WORD-level ones on the returned text (whitespace-split words):
- f_N := number of images (of 8) where N and E differ within the **first 12 words** (stricter than "first token": the boilerplate opener is ~8-10 words, so a difference inside 12 words means a different opening);
- prefix agreement := number of leading words identical before the first differing word; A_N / A_P as before; whole-text identity unchanged.
The decision rule keeps its numbers (f_N <= 1 and A_N >= 0.8 x A_P supports; f_N >= 2 or A_N < 0.5 x A_P fails; between = ambiguous). Preconditions 1 and 3 are unchanged; precondition 2's ">= 32 tokens" becomes ">= 30 words". The logprob request fields stay in the driver but are unused.
