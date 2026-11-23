# WG1, the Mac half: where a WebGPU token goes now (MacBook Pro M1 Pro, 2026-11-23)

**Read 2026-11-23 against [`registration-mac.md`](registration-mac.md)** (registered at `ccc7bba2a`, on main before any run).
**Status: decode run 1 read; decode run 2 VOID, and its re-run VOID too; prefill NOT RUN (the instrument declines on Metal).** Only
one valid decode run exists, so **the registered stopping rule (two runs) is not met, and the ranking and the `attnKeysPipeline`
lever below are provisional.** Closed so by the coordinator (2026-11-23 about 13:45 PST): a third attempt would be machine time past
Francis's approval. The second valid run comes with the later prefill registration (§ "How this completes").

**Provenance:** goinfer `ccc7bba2a`, one test binary (`go test -c -tags 'gpu goinfer_testhooks' ./backend/webgpu`, sha256 `d333efd3…`),
WebGPU over Metal (wgpu-native) on an Apple M1 Pro (14-core GPU), macOS 26.7.3 · qwen2.5-coder 1.5B Q4_K_M GGUF from `~/models`,
loaded `int8int8` by the test · the Mac's exclusive timing lock. Raw: [`results-mac/`](results-mac/) (`provenance.txt`, the four
test logs) · scripts: [`build-mac.sh`](build-mac.sh), [`run-mac.sh`](run-mac.sh), [`rerun-mac.sh`](rerun-mac.sh).

## Decode, run 1 (11:04:45–11:48:04 PST, on AC, no sleep)

`TestDecode_dispatchProfile`, ablation: what omitting the class from the token's plan saves. Rooflines are nominal, against the M1
Pro's ~200 GB/s: attention reads the f32 KV, 57,344 bytes a key (28 layers × K and V × 256 floats × 4 B); the projections read
about 1.54 GB of `int8` weights a token (1.31 G for the layers, 0.23 G for the LM head).

| depth | token | 1st class | 2nd class | attention vs its roofline | gemv vs its roofline |
|--:|--:|---|---|---|---|
| 128 | 20.5 ms | `gemvPipeline` 12.9 ms (63%) | `attnKeysPipeline` 1.8 ms (8.7%) | 1.78 against 0.037 ms (48×) | 12.9 against 7.7 ms (1.7×) |
| 2,048 | 48.5 ms | `attnKeysPipeline` 29.7 ms (61%) | `gemvPipeline` 13.0 ms (27%) | 29.7 against 0.59 ms (51×) | 13.0 against 7.7 ms (1.7×) |
| 8,192 | 143.7 ms | `attnKeysPipeline` 125.2 ms (87%) | `gemvPipeline` 13.0 ms (9.1%) | 125.2 against 2.35 ms (53×) | 13.0 against 7.7 ms (1.7×) |
| 16,384 | 259.1 ms | `attnKeysPipeline` 240.7 ms (93%) | `gemvPipeline` 13.1 ms (5.1%) | 240.7 against 4.70 ms (51×) | 13.1 against 7.7 ms (1.7×) |

Every other class is under 5% of the token at every depth, and several read negative (an omitted class that the rest overlapped);
the full per-class tables are in `results-mac/decode-run1.log`. The sums of the parts run 84–105% of the token.

**Provisional reading, by the registered outcomes (one valid run; the two void runs agree, as context only):**
- **`attnKeysPipeline` (the key-split decode attention) is a WG2 lever on the Mac:** the largest share from depth 2,048 on (61% →
  93%), and about 50× its bandwidth roofline at every depth, so the share is not bandwidth-bound work. It is KS-E5's long-context
  decode attention, the same class WG0's 2,048 cells pointed at (WebGPU loses 51–55% of its depth-128 rate by 2,048 on this box).
- **`gemvPipeline` is not ranked a lever:** the largest share at depth 128 (63%) but within 1.7× of its nominal roofline, and flat at
  about 13 ms across depths.

## Decode, run 2 (11:48:04–12:31:55 PST): VOID, re-run in place

At 11:48:49 the lid was closed and the Mac went on battery; it slept from 11:49:24 to 11:49:44 (`pmset -g log`), woke, and finished
on battery. A run with a sleep in it is void (`noise-registry.md` §8). It is listed, not used. For context only, its shares match run
1's: `attnKeysPipeline` 9.0 / 61.0 / 86.6 / 93.1% at the four depths against 8.7 / 61.3 / 87.2 / 92.9%; `gemvPipeline` read 13.6 /
12.5 / 7.2 / 9.7 ms. Log: `results-mac/decode-run2.log`. The re-run waits for AC with the lid open (latest start 20:00 PST), under
`caffeinate -i`, and checks `pmset`'s log for any sleep, wake or battery during it.

## Decode, run 2 re-run (12:58:21–13:36:55 PST, on AC, no sleep): VOID

Run in place under `caffeinate -i` once the Mac was on AC with the lid open (`results-mac/decode-run2-rerun.log`,
`results-mac/rerun-job.log`). No sleep, wake or battery event in the window (`pmset -g log`'s event types: assertions only). The job's
own automatic check printed VOID, but that was a false positive: its pattern matched "Idle Sleep Preventer" assertion lines. The
committed `rerun-mac.sh` reads `pmset`'s event-type column instead, and finds 0 events in this window and 10 in run 2's. The run
is void on another ground: **system swap grew from 787 to 1,307 MB while it ran** (it started at load 5.52, with the Mac in use), and
the registration voids a run on swap growth (`noise-registry.md` §8). Listed, not used. Its shares match run 1's: `attnKeysPipeline`
10.8 / 62.0 / 85.1 / 92.8% at the four depths.

## Prefill: NOT RUN, the instrument declines on Metal

Both prefill runs stopped at the warm-up, 6 s each: `gpu: PrefillLast declines — q/k/v bias on backend "metal" is not yet verified
bit-exact (Vulkan only for now)` (`results-mac/prefill-run1.log`). Qwen2.5 carries q/k/v biases, so `TestPrefill_dispatchProfile`
cannot profile it on the Mac. A Mac prefill profile needs a model without the bias, or the bias path verified on Metal, under its own
registration.

## Cost, against the registration

Registered: about 30–45 min in total. Decode run 1 alone took 43 min: at depth 16,384 the ablation re-records a 259 ms token R times
for each of the plan's classes, and the registration's estimate did not price that. Francis approved letting the run finish as
registered past the estimate (option (a), 2026-11-23 about 11:55 PST, via the coordinator), and the re-run of the void run 2 within
the same approval (the coordinator, 12:40 PST: prefill used none of it). In all: about 2 h 6 min under the exclusive lock (run 1 43 min, run 2 44
min, the two prefill attempts 13 s, the re-run 39 min).

## How this completes

The prefill half needs its own registration anyway: a model without q/k/v biases, or the bias path verified on Metal. That
registration also takes **one more decode run of this registration's cell**, unchanged (`TestDecode_dispatchProfile`, the 1.5B at
`int8int8`, depths 128 / 2,048 / 8,192 / 16,384), with the Mac on AC, the lid open and the box quiet. Its cost is now measured at
about 43 min. If that run is valid, the ranking stands on two runs and the provisional marks above are lifted in a dated amendment; if
its ranks differ from run 1's, the registered rule reports those classes as unranked.

