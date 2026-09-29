# MC3 per-pass prefill: how much of a 4-client served cell is prefill passes — pre-registration, 2026-09-28

The first step of the "per-pass prefill cost cut" follow-on in
[`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md) (MC3 follow-ons).
- **Why it is owed:** the item rests on an estimate. Newcomer suffix prefills were put at "~⅓ of a 4-client 1.5B
  cell", from W7's own numbers, before chunked prefill shipped (`-prefill-chunk 512`, default since 2026-09-27).
- **What this measures:** that share, directly. Nothing is built unless the share is large. **Written before the run.**

## 1. The instrument

- **2bdd3f3b:** `ResidentBatchStats` gains wall time with the resident held. `RunNs` is time inside decode runs,
  `ExclusiveNs` inside exclusive sections, and `PrefillNs` / `PrefillPasses` inside the sections that are prefill
  passes (chunked `PrefillLast` and the final seed). `TestMC3_longPrefillChunksWhileOthersDecode` checks the pass
  count against the resident's calls and that the durations nest; zeroing the counter turns it red.
- **666561ab:** `GET /admin/status` reports the counters per MC3 model. The driver reads them over serve's admin socket
  immediately before and after the cell, and differences them. So load, the warm request and first-use pipeline
  compiles are excluded.
  - An exploratory smoke showed why that matters: the server's lifetime counters carried 1.49 s of pre-cell prefill.
  - The differenced numbers add up: prefill + runs + bookkeeping + 0.06 s idle = the cell's wall clock.
  - The smoke used 16-token replies. Its share is not quoted.

## 2. The run

- **Binary:** `serve-metal-666561ab`, Metal resident, int4, `-embed-int4=false`. Since 9ccf7fb1 the int4-embedding
  default makes the Metal resident decline to the CPU; a cell whose log does not show `decode path: metal-resident`
  is void. Otherwise serve's defaults: 4 generations, `-prefill-chunk 512`.
- **Workload:** `bench_w7_plain.py`'s W7, 4 clients × 4 growing chat turns, 128 max tokens, greedy, fixed nonces.
  One warm request, then the cell.
- **Models:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M (graded) and Qwen2.5-7B-Instruct Q4_K_M (reported), both from
  `~/models`. 3 repetitions each, each on a fresh server.
- **Mac night queue**, ~30 min: [`run-mc3-prefill-attr.sh`](mc3-prefill-attr-2026-09-28/run-mc3-prefill-attr.sh),
  driver [`mc3_prefill_attr.py`](mc3-prefill-attr-2026-09-28/mc3_prefill_attr.py).

## 3. The decision rule, on the 1.5B's median prefill share (ΔPrefillNs ÷ the cell's wall clock)

- **< 0.15 → close the cut.** The estimated ~⅓ does not hold with chunked prefill, and the follow-on is struck with
  this record.
- **≥ 0.25 → register the cut** as its own item. It gets a per-pass cost breakdown (passes, prompt tokens prefilled,
  time per pass and per token) and a bar, before any code.
- **0.15–0.25 → the owner's call**, with the numbers.
- **Fewer than 2 valid 1.5B cells → inconclusive**; the run is fixed and re-queued.
- **Reported, not graded:** the 7B's share, runs and bookkeeping time, idle time, passes per cell, and the lifetime
  counters beside the differenced ones.

## 4. Results

*(After the run.)*
