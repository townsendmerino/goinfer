# The five "known unrelated" decoder failures were fixture mismatches, not regressions — 2026-09-24

**Status: fixed on the Mac; a guard now names the cause if it recurs on any box.**

## What was failing

On the MacBook, `go test ./decoder/` had been red since at least 2026-09-18 on five tests, which
`scripts/refresh_parity_hashes.sh` carried in `KNOWN_UNRELATED_FAILURES` from 2026-09-20 as "pre-existing,
unrelated to R13" — confirmed pre-existing by `git stash`, never root-caused:

| test | symptom |
|---|---|
| `TestGemma3VL_textParity` | argmax 154 vs golden 246, logit cosine **−0.043** |
| `TestGemma3VL_imageParity` | argmax 144 vs 179, cosine −0.004 |
| `TestGenerateVL_streams` | first token 144 vs 179 |
| `TestInt4_forwardParity/gemma3-vl-tiny` | argmax 144 vs 198, samples off by up to 0.30 |
| `TestBailingHybrid_forwardParity` | argmax right, sampled logits off by up to 0.15 (tolerance 5e-3) — but cosine against the gitignored full-vector reference **0.99999999999994** |

## Cause

Both fixtures (`testdata/gemma3-vl-tiny/`, `testdata/bailing_hybrid-tiny/`) are **gitignored and regenerated
per machine** by `scripts/pin_*.py`, while the goldens recorded from them are **committed**. The pin scripts
random-initialise the weights, and the result is not reproducible across torch/transformers versions — commit
`2dca08c4` already noted a fixed `torch.manual_seed(0)` giving a different checkpoint on a newer torch, and
`83f4c64b` noted `pin_bailing_hybrid_tiny.py` differs even run to run. On 2026-09-18 both fixtures were re-pinned
on nobara-pc and their goldens committed. The Mac kept its older checkpoints: `gemma3-vl-tiny` re-pinned locally
under transformers 5.15.0 (sha256 `c1b2c0fe…`, vs nobara's `639dcf85…`), `bailing_hybrid-tiny` from 2026-09-06
(`b2d41412…` vs `db6c9ad4…`). Every test then compared the committed golden against different random weights.

goinfer was correct throughout: the Mac's Bailing checkpoint reproduces the **pre-2026-09-18** golden's values
exactly (sample id 225: −1.26722 then and now), and the gitignored `bailing_hybrid_forward_full.json` — regenerated
on the Mac together with its checkpoint — matched at cosine 0.99999999999994 the whole time.

## Fix

1. Copied nobara's 2026-09-18 checkpoints (and Bailing's gitignored full reference) to the Mac; the Mac's old ones
   are kept in the session scratchpad. All five pass, at the numbers the regeneration commits recorded: gemma3 text
   cosine 1.000000, image 0.994641; Bailing max sample Δ 0.00000, cosine 0.99999999999991. `refresh_parity_hashes.sh`'s
   forward goldens: **38 passed / 0 failed** (was 35 / 3 with three exclusions).
2. `testdata/fixture_identity.json` (tracked) records the sha256 of the checkpoint each committed golden was recorded
   from; `requireFixtureIdentity` (`decoder/fixture_identity_test.go`) runs first in the six tests that use these
   fixtures and fails with "FIXTURE mismatch, not a forward-pass regression: copy that box's checkpoint, or re-pin and
   commit goldens + hash together". Verified by putting the Mac's old gemma3 checkpoint back: both affected tests fail
   with that message instead of a cosine. CI has no fixtures, so these tests still skip there, before the check.
3. The three `KNOWN_UNRELATED_FAILURES` entries are removed, per that array's own rule.

## What this does not fix

- The pin scripts are still non-deterministic. The durable fixes are either committing the checkpoints (0.5 MB and
  1.4 MB) or initialising weights with a version-independent RNG (e.g. numpy `default_rng(0)` written into the state
  dict) — neither done. Until then, whoever re-pins must update `fixture_identity.json` in the same commit.
- Only these two fixtures are in the manifest. The other 20 tiny fixtures `TestInt4_forwardParity` compares have the
  same exposure; they pass on this box today, so their current hashes could be added the same way.
- The exclusion-list entries were "confirmed pre-existing" by `git stash`, which is true and says nothing about cause.
  A pre-existing red is still a red; this one cost six days of a red decoder suite on the development box.
