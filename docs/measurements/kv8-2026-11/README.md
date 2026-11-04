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

### The reading, 2026-11-03 14:21 to 14:25 PST: FAIL by the registered rule

`served-2026-11-03/run.log`, `turns.jsonl`. Serve at `214878e5c`, driver 595.112.07, load at most 1.14 at a cell's start, about 61 GB available, no void cell, and the fit check in `fit-check.log`. A first attempt timed nothing: the script named the wrong model (`attempts.txt`).

| | one slot | two slots |
|---|---|---|
| follow-up median first token, each pair | 6.93 s | 0.05 s (ratio **0.008**) |
| on the card once loaded | 5,606 MiB | 6,168 MiB (+562) |
| peak `VmSwap` | 0 MB | 0 MB |
| follow-ups that reused rows | 0 of 8 | **6 of 8** |
| conversations whose replies equal the one-slot arm's | | **2 of 4** |

The ratio and the swap clear their bars. **Two clauses fail, so the verdict is FAIL, and no bar moves.** Both causes were found after the run, by a control that was not registered: each client alone on one slot, the same binary and prompts (`control.json`, `control-compare.txt`).

**1. Two follow-ups reused nothing: a load of several slots never scores a checkpoint.** Client A's first reply does not re-tokenize as generated (the record says 1,931 rows, a divergence at 1,930), so its follow-up is a restore from the checkpoint at its prompt's end, not a strict extension. Alone, on one slot, it restores 1,907 rows and starts in 0.14 s. With two slots, `decoder/resident_reuse.go:Model.residentAcquireSlotAll` asks every slot's score without the checkpoint, so the request scores 0 and no restore is attempted, and A prefills from 0 in 6.58 s. **The one checkpoint copy would have refused it next** if it belonged to the other client, so **serving it takes both: the score and a checkpoint per slot.**

*Corrected the same day, before this record merged, by reading that code while building the next slice: this paragraph first said the restore was refused because B saved last. No restore was attempted; the request log records no decline, and I had not checked it. The control and the numbers stand.*

**2. Two conversations' replies differ: reuse against cold, not the slots.** Client B's follow-up reuses 2,071 rows on two slots and is prefilled whole on one, and the replies differ (` 10` against ` 0`). **Alone on one slot it reuses the same rows and gives the same ` 10`.** Every one of the 12 turns on two slots, in both pairs, equals the same client's alone on one slot, including client A's turn where the rows reused differ. So the registered comparison, against the one-slot arm's cold prefill, was the wrong one for what it meant to test: it reads the strict-extension path's known kernel-class difference (`docs/measurements/kv8a-2026-10/README.md`), which the first turn after a reuse shows with no second slot at all. That is my registration's flaw, recorded here. It does not change this verdict.

**What this does and does not show.**
- **Shown, on a real model:** the second slot gives a client's own conversation back at 0.05 s against 6.93 s, without swap, for 562 MiB of card, and its replies are the ones that client gets alone.
- **Not shown:** the both-at-once round was compared only through the arms' script, which stopped at the first difference, so at-once replies were checked only between the two two-slot cells, where they match, and against no alone run, since the control ran no such round.
- **Not read:** the one-slot control ran this branch's binary. That reuse differs from cold on main's binary too is by reading, since the one-slot path did not change, and was not run.

**Next, if wanted:** a checkpoint per slot, then this reading again, registered against each client alone on one slot.

## The served reading again, against each client alone, registered 2026-11-03 15:00 PST before any run (nobara-sb01)

Francis's answer of 14:58 PST, relayed by the coordinator: keep the FAIL above on record, give each slot its own checkpoint, and register the corrected comparison before rerunning. **This does not replace the FAIL above.** It is a new reading, of a new build.

- **What is corrected.** A two-slot client's replies are compared with **that client's alone on one slot**, the control of the reading above, not with the cold one-slot arm. Alone is what a reused prompt should give; cold is another kernel class on this path (`docs/measurements/kv8a-2026-10/README.md`).
- **Tier and instrument:** as above, with serve built at this slice's pushed head, named in the record. `~/models/Qwen3.5-9B-Q4_K_M.gguf` from local NVMe, `--ctx 8192`, greedy, 24 tokens per reply, the exclusive timing lock.
- **Design:** one slot two clients, two slots two clients, each client alone on one slot, two slots two clients, one slot two clients. The same prompts' script as above, with 3 turns each in turn and then both at once. Alone, the at-once round is each client's fourth turn, sent on its own, so that round is compared too.

**The decision, worst of the two pairs:**

| | |
|---|---|
| **PASS** | follow-up median first token, two slots over one, at most 0.5; **every** follow-up on two slots reuses rows, the turn that restores its checkpoint included; **every** turn of a two-slot client, the at-once round included, has the reply that client gets alone; peak `VmSwap` at most 512 MB |
| **PARKED** | ratio above 0.5 and at most 0.7, the others holding |
| **FAIL** | ratio above 0.7, a follow-up that reuses nothing, a reply that differs from that client's alone, or `VmSwap` over 512 MB |

- **Reported, not graded:** each turn against the cold one-slot arm, and the first token of a restore.
- **Void cell, fit check, stopping rule and void conditions:** as above. The fit check also reads the copies kept.
- **Cost:** the fit check about 1 minute, the five cells about 4. The start and finish times go in the record from `date` when it runs.

### The reading again, 2026-11-03 17:23 to 17:27 PST: PASS by the corrected registration

`served-again-2026-11-03/run.log`, `turns.jsonl`. Serve at `37e1411bd`, driver 595.112.07, load at most 1.33 at a cell's start, about 61 GB available, no void cell, and `fit-check.log`, which keeps two checkpoints (100.5 MiB) at two slots.

| | one slot | two slots | bar |
|---|---|---|---|
| follow-up median first token, each pair | 6.93 s, 6.94 s | 0.05 s (ratio **0.008**) | at most 0.5 |
| follow-ups that reused rows, the at-once round included | | **12 of 12** | all |
| turns equal to that client's alone on one slot | | **16 of 16**, the at-once round included | all |
| on the card once loaded | 5,606 MiB | 6,218 MiB (+612) | none |
| peak `VmSwap` | 0 MB | 0 MB | at most 512 MB |

**Client A's follow-up, the one that reused nothing in the FAIL above, now restores.** On two slots it reuses 1,907 rows and starts in 0.14 s, where it took 6.58 s; its request log reads `ckpt_restore=1907`. The script's own line for restores printed an empty list: its filter was wrong, so the log is the reading.

**Reported, not graded:**
- 3 of 8 two-slot turns differ from the cold one-slot arm's. **So a reused prompt's reply on this path still differs from a cold prefill's, as it does with one slot:** a property of strict extension, and a real one, which this reading leaves as it found it.
- The at-once round runs in turn, so its second client waits: 0.51 s here, and 14.7 s on one slot.
- The second checkpoint costs 50 MiB of the 612.

**What this does not cover:** one prompt shape of about 2,000 tokens, two clients, one card and driver, and no agent's real traffic. It is not a row for `docs/benchmarks.md`. Comparing with each client alone says the slots keep conversations apart; it does not say a reused reply is as good as a cold one.

## The served reading a third time, restoring on both slots, registered 2026-11-03 18:19 PST before any run (nobara-sb01)

**Why:** #658's review read the logs of the reading above. Client A's one follow-up restored (0.14 s). Every other follow-up, all of client B's among them, was a strict extension. **So the second slot never restored, and the 0.05 s median is mostly strict extensions.** The reading passed its registration, and that registration could not see this. It stands as it is.

**What it adds:** after the three turns, an edit round, before the at-once round: each client sends its last prompt, then its reply with a word put in front, as a client that re-renders a reply does, then a new question. That diverges at the prompt's end on both slots, so each must restore its own checkpoint. Alone, each client sends the same turn.

**The decision, added to the registration above, whose other clauses hold, over both two-slot cells:**

| | |
|---|---|
| **PASS** | both clients' edit turns reuse exactly their last prompt's rows, on two slots and alone; those turns' median first token, two slots over the one-slot arm's, at most 0.5; and each reply equals that client's alone |
| **PARKED** | that ratio above 0.5 and at most 0.7, the others holding |
| **FAIL** | either client's edit turn restoring nothing, a reply that differs from that client's alone, or that ratio above 0.7 |

- **The restores are read from each turn's reuse,** which replaces the filter that printed an empty list above. The request logs' `ckpt_restore` are the check on it.
- **Instrument, void rules, stopping rule:** as above, with serve built at this PR's head. Cost about 5 minutes.

