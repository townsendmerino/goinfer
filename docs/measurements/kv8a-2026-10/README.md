# KV8a gates — registration (2026-10-20, before any code)

The spec is `docs/tasks/task-kv-architecture-2026-10.md`, "KV8a — An end-of-prompt checkpoint on today's single slot". This turns
its five sketch gates into registered gates. **Gate 2's bar is a proposal awaiting the owner's answer; nothing is built until then.**
CUDA only; Metal waits for CUDA to pass. Readings stay unmerged for the owner's review. The rules below do not move after data.

## Where the checkpoint lives on CUDA

**Device memory, charged to the load's memory plan; when the plan cannot admit it, KV8a declines and the slot behaves as today.**
No host fallback is built.

- **Why device:** the save runs on every request (gate 4), so its cost is paid on every A turn. A device-to-device copy of the
  state (`gpu.CopyDeviceBatch`) runs at device bandwidth. spec/09 measured snapshot plus restore at 623 µs for the 0.8B
  (`task-recompute-audit.md`, R-01 phase 1 correction). A host copy crosses PCIe on every save, milliseconds each way at
  tens of MiB.
- **Why no host fallback:** a second storage path doubles the restore code and its gates, for a case the 8 GB card does not hit
  at the 9B's size. A load the plan cannot fit declines, which is what the spec requires.
- **Bytes for the 9B:** measured in gate 1 from the plan's charge line. From the state spec: every DeltaNet layer's
  recurrent matrix plus its conv window, at f32. §5.7 gives 62.8 MiB for Qwen3.6-35B-A3B; the 9B's number is entered here once
  read.

## Gates

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| **1. Bit-exact save and restore** (tier 0, CUDA) | On `qwen3next-tiny` (DeltaNet) and a Mamba hybrid tiny fixture: prefill a prompt; save; run 8 decode steps, which move the live state; restore. Every recurrent layer's state and conv window, downloaded after the restore, equals the download taken right after the save, byte for byte. Plant: a restore that skips the conv window must differ. Then on the 9B (tier 1, one server start), the plan's charge line gives the checkpoint's bytes. | A fixture is missing (a skip counts as a fail); the plant does not differ. | Every layer of both fixtures compared once. | Tiny fixtures: about 1 min on nobara. The 9B bytes: one load, about 15 s (2026-10-18 probe: 12 s). |
| **2. Agreement with a cold prefill** | **PROPOSED, see below.** | | | |
| **3. The cache-state grid** (tier 0) | `decoder/cachestate.go:cacheStateGrid` gains the save (end of prefill) and restore (at `c`) paths, with a declared cell for every kind of state: attention KV (f32 and f16, windowed), DeltaNet, Mamba, image spans. The recurrent `lcShare` cell's reason changes from "no checkpoint to restore". An undeclared cell fails the test, so the grid stays fail-closed. Plant: drop one declared cell, and the test goes red. | | Every cell checked. | Seconds. |
| **4. The cost on every turn** | (a) Tier 0: the save is ordered before the first chained decode step (`decoder/residency.go:ResidentGreedyChain`, `ResidentSampleChain`), checked through the caller with a recording fake; plant: the save moved after the first step goes red. (b) Tier 1, inside gate 5's run: each request logs the save's own device time; **p95 ≤ 12 ms** (2% of Codex's 0.6 s A median). (c) A turns' median time to first token is reported beside KV0-H's (2.2 s opencode, 0.6 s Codex) as context, not graded as a ratio, since the two runs are different sessions (`CLAUDE.md`, TE3). | The save time is not logged on every request. | (b) every request of gate 5. | Inside gate 5. |
| **5. The measured win** (tier 1, nobara, day run, detached) | KV0-H's opencode and Codex configuration exactly: `plan-study-h20.json` (10 sessions, KV0's tasks and repetitions), Qwen3.5-9B Q4_K_M, `-ctx 20480`, `--kv-sessions 1`, f32 KV, the ids key, the default swap guard, the same KV0-H driver commit (`2272c3cf9`), with the serve build swapped for KV8a's. **Bar: the pooled median time to first token of B turns is at least 17.1 s (half of the projected 34.2 s) below KV0-H's pooled B median.** A turns as gate 4(c). KV0-H's readings are the do-nothing arm. | A session that KV0-H's own rules would void (the length and SHA-256 check, the swap guard, a 503 `halted`); the timing lock shared with another timed run; the 2026-10-19 heavy-load rule at a request's start; a different driver than KV0-H's 595.91.07. | KV0-H's plan once. If fewer than 10 B turns occur, add one `chain` session per harness (KV0-H's own stopping rule), once. | KV0-H took 26.5 min, with B turns cold at 27–38 s; with KV8a, about 15–20 min. Queued at 30, timeout 60. |

## Gate 2: the bar (PROPOSED; the owner answers before any code)

**Nothing is on record.** R-01 phase 0 (commit `44a935eb`) was accepted on `decoder/resident_reuse_test.go:TestResidentReuseLen_recurrentExactExtensionOnly`,
a unit test of the reuse-length rule. Its registered hardware gate, "a two-turn exact extension equals the cold prefill of the
concatenation token for token", is recorded as unrun in that commit and in `docs/tasks/task-recompute-audit.md`, R-01 ("CUDA-hardware
scenarios unrun").

**Bit-identity is not available here.** `c` (the end of the previous prompt) is almost never a pass boundary of the cold prefill of the
next prompt, and the suffix after a restore runs as a short pass with the exact kernels where cold runs it inside a fused pass
(`PrefillTailMinRows`). B35 got bit-identity only by keeping its state on the pass grid.

**Proposal:**
- **(a) Tier 0, CUDA, tiny fixtures:**
  - **A/A reference:** the same prompt prefilled cold at two pass widths (512 and 256), measured in the same test. That is the deviation today's code already accepts from a change of pass shape.
  - **Bar:** for the restored path against cold, the last prompt row's max |Δlogit| and KL are each no larger than 1.5 × that A/A, and greedy decode agrees for 32 tokens.
- **(b) Tier 1, the 9B, through serve, about 20 B-shaped requests.** The previous reply is re-rendered 15 to 123 tokens in, as KV0-H's are; built from the pilot's recorded bodies, never from the study's.
  - **Arms:** the KV8a restore and a cold prefill of the same request on a fresh server. The strict-extension path (phase 0) runs through the same comparison as a reference arm, since its own bar was never measured.
  - **PASS:** greedy tokens equal cold for the first 24 tokens on every request, except where the first difference falls at a near-tie: a top-2 log-probability gap under 0.1 at that position, both arms recorded.
  - **KV8a's agreement is at least the strict-extension path's** (non-inferior, same requests).
  - **PARKED for the owner:** one disagreement that is not a near-tie.
  - **FAIL:** more than one.

**Cost:** (a) seconds. (b) about 40 server requests plus 20 fresh loads, about 10 minutes on nobara.

## Build estimate

S–M on CUDA, about a day of session time:
- the save after prefill and before the first decode;
- the restore, which copies the checkpoint in, truncates the attention KV to `c`, and prefills from `c`;
- `internal/kvmgr/slots.go:scoreRecurrent` returning `c`;
- the plan charge and the decline;
- the grid cells;
- gates 1, 3 and 4(a).

Tier 0 runs on nobara (CUDA) in minutes. Gates 2(b) and 5 then run about 40 minutes of nobara time between them.
