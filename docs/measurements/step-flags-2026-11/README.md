# Step-flag memory: should `--recurrent-step` and `--gemma4-step` default on?

Release v0.24.0, item 3d. Registered 2026-11-03, before any run; runs Wednesday 2026-11-04 by day, on the Mac, under the shared timing lock. No speed is measured: the gains are on record (1.5799× on Qwen3.5 0.8B, 1.3794× on Gemma 4 E2B, `docs/tasks/task-forward-architecture-2026-10.md`).

**Question:** what does each flag cost in memory, so Francis can decide its default for this release?

**Models, at int4, from `~/models`:** with `--recurrent-step`: Qwen3.5 0.8B, Qwen3.5 9B, Nemotron-H 4B, Nemotron-H 9B. With `--gemma4-step`: Gemma 4 E2B, Gemma 4 E4B. Which file stands for each Nemotron-H goes in the record. A load the fit guard refuses, or a file this Mac can't take, is named in the record, not substituted.

**Arms:** flag off and on, at 4 and at 2 KV sessions, each at `-ctx 4096`: four fresh loads per model. Plus flag on at 1 session, for what `goinfer-chat` pays.

**What it records, per arm:**
- **Priced:** the step's scratch (`metalStepScratchReserve`) and each session's state (`metalRecurrentSlotBytes`).
- **Allocated:** the ledger's bytes by line after a build (`metalLedgerOf(...).Usage().ByLine`), and whether a batched step actually built.
- **Longest context that fits:** the largest whole-block context up to `MetalCtxCeiling` whose weights, KV for that session count, state and step scratch fit each of the 16 GB M1 Pro's two memory profiles in the census, `m1pro-near-idle` and `m1pro-apps-open`, less `MetalFitMarginBytes`. It is arithmetic from the priced figures, checked at one point against the serve banner of a real load.
- **`goinfer-chat`:** what it pays with the flag on, and whether it can use the flag at one session at all.

**Metric:**
- **Extra memory at 4 sessions:** allocated bytes on minus off, as a share of the off arm's allocated total. Each session's recurrent state counts as extra only where the flag adds it, which the off arm's ledger shows.
- **Context shrink:** 1 minus longest-on over longest-off, at 4 sessions, on `m1pro-apps-open`.

**Threshold, a placeholder Francis sets:** on by default for a flag if, on every model it serves, extra memory at 4 sessions is under 5% of resident size **and** the longest fitting context shrinks by under 10%. A near miss, within one point of the 5% or two of the 10%, goes to him. Otherwise off. If one family of a flag passes and another fails, that goes to him too, since a flag is one default.

**Stopping rule:** one pass per arm. The readings are byte counts, so nothing repeats. A priced figure more than 16 KiB per lease from its allocation stops that model's reading until it is understood, since the fit would then be pricing the wrong number, and the memory gate's list is where that fix belongs.

**What each outcome means:** under the rule, a one-line default change, in its own PR, to Francis before merge, like any default. Over it, the flag stays opt-in and the record says what it costs. Either way this record is the number `docs/flags.md` quotes.

## Read, 2026-11-04, by day, untimed, shared lock

MacBookPro18,3, 16 GB, int4, `-ctx 4096`, instrument `stepflags_test.go.txt` (copy it into `backend/metal/` to run it), raw `readings.json.gz`. Nothing here flips a default: **Francis decides.** The rule is his placeholder.

| Flag | Model | Extra at 4 sessions, MiB | Share of resident size | Longest context on `m1pro-apps-open`, off → on | Shrink | Against the placeholder |
|---|---|---|---|---|---|---|
| `--recurrent-step` | Qwen3.5 0.8B | 8.16 | 0.996% | 32,768 → 32,768 | 0% | under both |
| `--recurrent-step` | Qwen3.5 9B | 9.30 | 0.170% | 6,976 → 6,912 | 0.92% | under both |
| `--recurrent-step` | Nemotron-H 9B | 12.92, priced | 0.112% of its need, priced | fits at neither | on `m1pro-near-idle`, 12,224 → 12,032: 1.57% | under both, priced only |
| `--recurrent-step` | Nemotron-H 4B | 9.74 | 0.256% | 20,544 → 20,352 | 0.93% | under both (read 2026-11-07, below) |
| `--gemma4-step` | Gemma 4 E2B | 9.62 | 0.566% | 32,768 → 32,768 | 0% | under both |
| `--gemma4-step` | Gemma 4 E4B | 9.68 | 0.263% | 11,392 → 11,328 | 0.56% | under both |

**By the rule as written:** `--gemma4-step` reads under both bars on both its models. `--recurrent-step` reads under both on all four of its models, the 4B since 2026-11-07. No reading is a near miss. Everything read is well inside the bars.

**What the extra is.** It is the step's scratch alone: 8 to 10 MiB allocated and about the same whatever the model's size, so it weighs least on the largest. KV and recurrent state are equal between the arms, because an explicit `--kv-sessions` already holds that many slots with the flag off. So this doesn't price what a **default** session count costs, which is a separate question.

**At 2 sessions** the extra is the same bytes: 1.193%, 0.182%, 0.619% and 0.299% on the four models with allocations on 2026-11-04, and 0.278% on the 4B, and the longest context shrinks at most 0.82%.

**`goinfer-chat`:** `--kv-sessions` is a serve flag, so chat holds one session. With the flag on at one session the step builds and takes the same scratch, 8 to 10 MiB. It has only one row to run, so chat pays that for no batching; the batched step's gain needs two or more sessions. An on-by-default flag that skips the step at one session would save it. That was not measured.

**What was not read:**
- **Nemotron-H 4B:** on 2026-11-04 only its flag-off arm at 4 sessions, 3,797 MiB allocated; the fit guard refused the rest by day. Every arm read on 2026-11-07, below.
- **Nemotron-H 9B:** the file standing for it is Nemotron Nano 9B v2, `nemotron-nano-9b-v2-int4-metal.v17.giw`. The guard refused every arm, needing 10.68 GB against a live budget near 7 GB, so its figures are priced from the model with no allocation behind them. The 4B's file is `nemotron-h-4b-instruct-128k`.

**Checks.** Where a step built, the ledger's scratch moved by the step's list to within one 16 KiB rounding (0.8B: 8,540,936 B asked, 8,557,320 B allocated; the 9B's Qwen3.5 exactly). The priced figure sits above that by design, each buffer rounded to a page (`stepShape.scratchBytes`), so the longest-context column errs short. KV allocated equals KV priced in every arm. The serve banner of a real load, E4B at 2 slots, reads 0.2 GB of KV a conversation and 4.2 GB of weights, against 0.237 GB and 4.19 GiB here.

**A cost of this measurement.** The 4B's direct safetensors load quantizes in memory, and at 07:45 PST it pushed 1.5 GB of swap-outs while Francis was on the Mac, taking swap in use from about 3.2 to 7.1 GB. It came back down, and no timed run was under way. E4B read 29 MB of swap-outs, with `mac-layout2` building beside it. Neither moves a figure above, which are byte counts. **Load big checkpoints from a bundle,** as `run-4b.sh` now does.

## Francis's decision, 2026-11-04 09:00, relayed by the coordinator: "a seems like a no-brainer"

**In v0.24.0, `--gemma4-step` and `--recurrent-step` default on in serve, on any load with two or more KV sessions, and off in single-session chat, where the scratch buys nothing. The explicit flags stay, so either can be turned on or off.** He decided from the table above: at most 0.996% extra at 4 sessions and at most 1.57% off the longest context, with the 1.38× on Gemma 4 E2B and the 1.58× on Qwen3.5 0.8B on the other side.

**Nemotron-H 4B, covered by his decision:** `--recurrent-step` stays on for it. Its arms read under both bars on 2026-11-07 (below), so nothing goes back to him.

**How "two or more sessions" lands:** a recurrent model keeps one slot unless `--kv-sessions` asks for more, so `--recurrent-step` follows a given count of 2 or more, and plain serve does not batch those models. A Gemma 4 dense model keeps Metal's default slots, so `--gemma4-step` also follows serve's default count. The build decides on the slots it was **granted**: if the memory guard grants one where two were asked, a defaulted step is not built, and a given flag still builds it. The price read before the build still counts the step in that case, since the granted count depends on that price.

**What serve's own banner reads, on this change's build:**

| Load | Flags | Concurrency line |
|---|---|---|
| Qwen3.5 tiny, no tower | none | one generation |
| Qwen3.5 tiny, no tower | `--kv-sessions 2` | 2 at once, batched |
| Qwen3.5 tiny, no tower | `--kv-sessions 2 --recurrent-step=false` | one generation |
| Nemotron-H tiny | `--kv-sessions 4` | 4 at once, batched |
| Nemotron-H tiny | none | one generation |
| Gemma 4 E2B | none | 2 at once, batched |
| Gemma 4 E2B | `--gemma4-step=false` | one generation |
| **Qwen3.5 0.8B, its checkpoint directory** | `--kv-sessions 2`, and with `--recurrent-step` given | **one generation** |

**That last row is older than this change, and it narrows what the default buys.** `serve` runs a model with a vision tower attached one generation at a time, and Qwen3.5 0.8B's checkpoint ships its tower. So the step builds, takes its scratch and does not batch there. The 1.58× is an in-process reading (D4b-4, `ForwardBatch` on four slots), and it does not reach that load through serve. Text-only Qwen3.5 loads, such as a `.giw`, and Nemotron-H do batch. The same holds for a Gemma 4 load with its tower attached. Dropping the step where a tower attaches was not done, because it is more than small: towers attach after the build, and on those loads the step also serves short-prompt prefill, so serve would have to release it from a built resident or know of the tower before the load.

## Nemotron-H 4B's arms, 2026-11-07 10:10 to 10:11 PST, by day, exclusive lock, on AC

`run-4b.sh` (beside this record) ran by day once the Mac had room: 8,331 MB reclaimable, load 1.79. It was queued as night job `step-flags-4b`, which missed three nights on memory, and its night copy was dropped once it ran. It loaded from the Metal bundle `nemotron-h-4b-instruct-128k.int4.metal.v17.giw`, with 0 swap-outs in the first load and in the arms (63 pages swapped in). `TestStepFlagsMemory` passed. Raw: `readings.json.gz`, key `nemotron-4b-2026-11-07`.

- At 4 sessions: off 3,983,643,084 B, on 3,993,855,192 B, so 9.74 MiB extra, 0.256% of its resident size. The longest context on `m1pro-apps-open` goes from 20,544 to 20,352, a 0.93% shrink. Under both bars, and not a near miss.
- At 2 sessions: 9.74 MiB, 0.278%. The longest context is 32,768 in both arms.
- At 1 session, the flag on builds the step (20,425,912 B of scratch).
- Checks: the ledger's scratch moved by 10,212,108 B against the step's list of 10,195,724 B, one 16 KiB rounding. KV allocated equals KV priced in every arm.
- The script removed the 4B's safetensors after the bundle verified; nobara holds the checkpoint.
