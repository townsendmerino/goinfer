# The red CUDA device suite, 2026-10-25: cause and fix

The nightly `cuda-device-suite` (`docs/measurements/cuda-device-suite-2026-10-18/`, issue #229) has been red since 2026-10-24 on two
tests, `TestKEqVCopy_bitIdenticalToTheDoubleProjection` and `TestPrefillTails_bitIdenticalToTheRecordedBaseline`. The other 868 tests
pass. The last green run was 2026-10-23 at 5275808c.

- **What differs:** only the `gemma4-moe-scaled` fixture. Its decode, prefill and tail hashes no longer match the baselines the two
  tests recorded (at 3c1a368b and 2d9ce186). Every other fixture still matches.
- **The cause: #200** (c0acef3b0, Gemma 4 MoE's float-activation decode on CUDA, `cuda/floatact.go`). Its contract change says that
  Gemma 4 MoE decode is no longer bit-identical to the W4A8 path, and that the tests holding W4A8 decode to W4A8 references build with
  `floatActOffForTest`. Four tests were changed that way. These two compare against W4A8 baselines too, and were not.
- **The bisect** (`run-bisect.sh`, the device suite's own runner with `REV` and `TESTARGS`, through `night.py`, 2026-10-25 19:52):

  | revision | the two tests |
  |---|---|
  | d0a64bcc6 (#200's parent on main) | 2 passed: **GREEN** |
  | c0acef3b0 (#200) | 2 failed: **RED** |
  | 2a39592bc (the fix) | 2 passed: **GREEN** |

- **The fix:** both tests build with `floatActOffForTest`, as #200's contract does for the others, so they hold Gemma 4 MoE's W4A8
  decode to the baselines they recorded. No baseline is re-recorded and no bar moves. The float-activation decode is held to the
  CPU float path by `TestGemma4MoE_floatParity` (cosine 1.0000000 at all 16 positions in #200's record).
- **Logs:** `verdict-<rev>.log` here; the raw output is under `~/goinfer-logs/cuda-suite-bisect-2026-10-25/`.
