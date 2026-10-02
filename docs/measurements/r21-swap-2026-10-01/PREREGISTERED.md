# R21 — is the swap growth under scenario D goinfer's? (2026-10-01, nobara-pc; written before the run)

Arms, one pass each, sampled every 2 s from /proc/meminfo (swap used, MemAvailable, Cached):
- idle: 30 s of nothing, before and after each arm (the box's own swap slope; other sessions run here).
- B (control, no goinfer): `cat ~/models/gemma4-26b-q4_k_m.gguf > /dev/null` (16.8 GB: a larger page-cache fill than A's 14.2 GB file).
- A (scenario D): the serve binary at this tree, `-backend cuda`, the 26B q4_0, no -moe-cache-experts (declines to the CPU), until `decode path:` plus 30 s.

Rule. Let dA, dB be the peak swap-used increase over each arm's own starting value, and s the idle slope x arm duration.
- GOINFER's footprint: dA - s >= 0.5 GB AND dA >= 2 x dB  ->  a fix is justified (the swap guard should have refused: its +512 MB threshold).
- THE KERNEL's: dB >= 0.5 x dA  ->  nothing to fix; record and close R21 as an observation.
- Anything else: AMBIGUOUS, parked, no change.
Confounds stated up front: the 26B's files are in the page cache from earlier today (A reads warm, B cold-ish), swap may be moved by other sessions
(the idle slope measures that), one pass each. A single pass cannot settle a close call; that is what the AMBIGUOUS band is for.
