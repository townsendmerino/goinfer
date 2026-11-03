# KV8 on CUDA (2026-11)

The plan, its gates and the first slice's readings are in `docs/tasks/task-kv-architecture-2026-10.md`, KV8. The logs are `state-slots.log` and `state-slots-push.log`.

## The served reading: two clients on Qwen3.5-9B, registered 2026-11-03 14:20 PST before any run (nobara-sb01)

**The question:** when two clients take turns on a hybrid, does a second KV slot give each its own conversation back? On one slot each turn evicts the other client's state, so every turn prefills from 0.

- **Tier:** tier 1 on the card, by day. Tiers 0 and 1 on tiny fixtures show the state is kept apart; they cannot show the time saved on a real prompt.
- **Instrument:** `served_two_clients.py`, under the exclusive timing lock, with serve built at `214878e5c` (#646's head), `~/models/Qwen3.5-9B-Q4_K_M.gguf` from local NVMe, `--ctx 8192`, greedy, 24 tokens per reply.
- **Design:** one slot (`--kv-sessions 1`) against two (`--kv-sessions 2`), A B B A, one server per cell. Each cell: two clients with about 2,000 tokens of their own text take 3 turns each in turn, on `/v1/completions`, each follow-up being its whole prompt, its reply and a new question. Then both send at once. One slot is the do-nothing arm.
- **The numbers:** the follow-ups' (rounds 1 and 2) median time to first token, two slots over one, in each A B pair.

**The decision, worst of the two pairs:**

| | |
|---|---|
| **PASS** | ratio at most 0.5, every follow-up on two slots reuses rows, each conversation's replies equal the one-slot arm's, and peak `VmSwap` of the server process at most 512 MB |
| **PARKED** | ratio above 0.5 and at most 0.7, the others holding |
| **FAIL** | ratio above 0.7, a follow-up that reuses nothing, a reply that differs, or `VmSwap` over 512 MB |

Replies are compared turn by turn up to a conversation's first difference, because past it the prompts differ. A reply that differs is a FAIL to look into, not a verdict on the state: a reused prompt prefills in two parts and a cold one in one.

- **Not the question:** batching. CUDA's batched step declines a recurrent mixer, so at once means in turn. No peer, and no row for `docs/benchmarks.md`.
- **Void cell:** one-minute load above 8.0 at its start. It is re-run once in order and listed. The free-memory term is read from `MemAvailable`, and the fit check below is the load's own.
- **Fit check first:** the same binary's banner at one and two slots. A refusal is NOT RUN, named.
- **Stopping rule:** the four cells, no early stop.
- **Cost:** the fit check about 1 minute. Cold about 11 s a turn by KV0-H's rate, so about 2.5 minutes for a one-slot cell, about 1 for two slots: about 7 minutes in all. Started by 14:23 PST, finished by about 14:34 PST.
- **Driver:** 595.112.07 on this box today, not the `595.91.07` of KV0-H's rows, so KV0-H's times are not compared.
- **Void if:** a cell outside the lock, the model read from `/srv/models`, a NOT RUN counted as a pass, a bar moved after a reading.
