# G5 range survey for Flint on the real checkpoint, nobara, 2026-11-13

`TestFlint_rangeSurvey` (`backend/cuda/kv4b_groups_test.go`, heavy: `GOINFER_HEAVY_TESTS=1`): the real Flint checkpoint on CUDA, int4 from `~/models/Flint-Q4_K_M.int4.e4h.cuda.v17.giw` (the tokenizer from `~/models/Flint-Q4_K_M.gguf`), at an f32 cache
with the routed experts paged (`MoECacheExperts`), one KV slot, context 4,028. Each of the ten held-out set-B gate prompts is prefilled at 3,900 tokens, and every K/V-keeping layer's max |K| and max |V| are read back from the cache. The rule
(`docs/measurements/kv4-cuda-2026-10-21.md` §2, G5): the worst under 65504/2 = 32,752, or an f16 cache could overflow.

- **Result: PASS.** 49 K/V layers, 10 prompts x 3,900 tokens: **worst max |V| 129.80 (layer 48); worst max |K| 49.85 (layer 48)**, both far under 32,752, no NaN. Per layer in `survey.log` (the `[flint-survey]` lines); max |V| rises with depth, from 0.7 to 3 in the first twenty layers to 55.4 at layer 46 and 129.8 at layer 48.
  No per-(position, head) scale is needed for an f16 cache on this checkpoint at these prompts.
- Machine nobara, RTX 2070 SUPER 8 GB, driver 595.112.07, 09:27 to 09:32 PST, test wall 259 s (load about 42 s). Load average 0.42 at the start, 62 GB host RAM. Shared timing lock (a correctness read, not timed).
- **Fit check first**, serve's own plan (`fit.log`: serve-cuda-new at the same flags, context 4,028, `-kv f32 -moe-cache-experts -kv-sessions 1`): loads and plans resident (`decode path: cuda-resident (int4)`), weights 20.9 GB, KV 0.4 GB at the serve default windowed layout, budget 26.2 GB. The test loads without windowed KV (the library default), so
  its f32 cache is the unwindowed one; it loaded and ran without refusal.
- **Never-swap:** the test process's `VmSwap` read 0 kB at all 26 samples, 10 seconds apart (`procmem.log`; `VmHWM` was 43.1 GB, which presumably counts the mapped weights file). System swap was 299 MB before and 479 MB after, which no process of this run is shown to own (context only).
- Limits: ten prompts of one prose set at one depth (3,900), the Q4_K_M checkpoint's int4 transcode, an f32 cache. The rule asks for exactly this, and a deeper context or other prompts could read higher; V grows with depth in the last layers.
- G5's remaining items are not done here: `auto` stays f32 until a G4 passes (D7), and `docs/kv-precision-matrix.md` has no flint-tiny row.
