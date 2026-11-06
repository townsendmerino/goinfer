# MEM6 on CUDA: resident prefill embeds per pass, and the memory comes back (2026-11, nobara-sb01)

The ledger row (`docs/tasks/task-memory-campaign-2026-10.md`, MEM6): the Metal routes were done 2026-10-22 (`docs/measurements/mem6-2026-10/README.md`); left are the CUDA cells and the `RssAnon` gate. MEM0's nobara half had
measured the host embeddings of a 20K prompt **held after the request**: +218 MiB on the 1.5B and +476 MiB on the 7B, still there 10 s later (`docs/measurements/mem0-2026-10/README.md`, "MEM6").

## The change

CUDA's resident implements `decoder.ResidentPrefillRows` (`PrefillLastRows`): the decoder embeds a range of the suffix into one reused slab each time the chunked prefill asks for a pass,
so the host holds one pass of embedding rows, not the whole suffix. `prefillChunkedMarked` is the same loop over a rows source (`prefillChunkedRows`); the slice entry point passes `embeddings[lo:hi]`.
A pass retried at half the width asks for its range again. The marked, image, span and m-RoPE routes are unchanged.

## Gates

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| MEM6-C1 rows | 0, the card | `TestMEM6_cudaPrefillHostRows`: `llama-tiny` with its window raised to 4,096, CUDA-resident, the pass set to 256 rows (the learned chunk cap stands in for a real model's pass), prefills of 1,024 and 2,048 rows; the most embedding rows the decoder holds at once, through the test observer | at most one pass (256). **Red first on main:** 1,024 and 2,048 held |
| MEM6-C2 identity | 0, the card | the same test: the last logits and 32 decode steps, hashed, through `ResidentPrefillSeedForTest` against the slice entry point (`PrefillLast` on every row embedded up front) | byte-equal |
| MEM6-C3 neighbours | 0, the card | the prefill, KV3 and resident tests of `backend/cuda` | pass as on main |
| MEM6-C4 `RssAnon` returns | 1, the card | `mem0_linux.py` (MEM0's cell, unchanged): one fresh `serve` per cell at `-ctx 32768`, one cold 20K-token request, `RssAnon` read every 250 ms; `cuda:1.5b` and `cuda:7b` (`~/models` NVMe), serve built from main (`serve-mem6-before`, `6ab62e5b2`) and from the branch (`serve-mem6-after`); one run each, before first; the exclusive timing lock | on both models the branch's `RssAnon` 10 s after the request is within **64 MiB** of its value after load. CUDA keeps no session state on the host beyond the id record (20K ints, 160 KB), so nothing is added for it. Reported: the peak, and main's own figures |
| MEM6-C5 identity of replies | 1, the card | the cell's reply text (`max_tokens` 8, greedy) on main and on the branch | equal |
| MEM6-C6 identity | 1, the card | `go run ./cmd/gate identity -backend cuda <main> <branch>` | every family that ran IDENTICAL |

**PARKED:** within 128 MiB, the others holding. **FAIL:** beyond that, a different reply, or a DIFFERENT family.

- **Void:** a run with the one-minute load above 8.0 at its start, or the lock not held; re-run once in order and listed. The prompt must tokenize to about 20K on the Qwen tokenizer (`mem0_linux.py` prints it); a rejected request (a 400) measures nothing and is void.
- **Not the question:** speed, the peak during the request beyond what is reported, other models, the load's own transient (about 4 GB on the 7B, MEM0's observation, a different item).
- **Cost and times (from `date`):** tier 0 minutes; MEM6-C4 four cells at about 1 minute each; MEM6-C6 about 6 minutes.
- **Void if:** a card run outside the lock, a NOT RUN counted as a pass, a bar moved after a reading.

Registered 2026-11-05 16:24 PST (the time from `date`), after MEM6-C1 to C3 were written and read (`red.log`, `green.log`) and before MEM6-C4 ran.
