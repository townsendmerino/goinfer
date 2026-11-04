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
| `--recurrent-step` | Nemotron-H 4B | **not read** | **not read** | **not read** | **not read** | **open** |
| `--gemma4-step` | Gemma 4 E2B | 9.62 | 0.566% | 32,768 → 32,768 | 0% | under both |
| `--gemma4-step` | Gemma 4 E4B | 9.68 | 0.263% | 11,392 → 11,328 | 0.56% | under both |

**By the rule as written:** `--gemma4-step` reads under both bars on both its models. `--recurrent-step` reads under both on three of its four models, and the rule asks for every model, so it can't be graded whole until the 4B reads. No reading is a near miss. Everything read is well inside the bars.

**What the extra is.** It is the step's scratch alone: 8 to 10 MiB allocated and about the same whatever the model's size, so it weighs least on the largest. KV and recurrent state are equal between the arms, because an explicit `--kv-sessions` already holds that many slots with the flag off. So this doesn't price what a **default** session count costs, which is a separate question.

**At 2 sessions** the extra is the same bytes: 1.193%, 0.182%, 0.619% and 0.299% on the four models with allocations, and the longest context shrinks at most 0.82%.

**`goinfer-chat`:** `--kv-sessions` is a serve flag, so chat holds one session. With the flag on at one session the step builds and takes the same scratch, 8 to 10 MiB. It has only one row to run, so chat pays that for no batching; the batched step's gain needs two or more sessions. An on-by-default flag that skips the step at one session would save it. That was not measured.

**What was not read:**
- **Nemotron-H 4B:** only its flag-off arm at 4 sessions, 3,797 MiB allocated. The fit guard refused the rest by day. Night job `step-flags-4b` (`run-4b.sh`) reads them from a Metal bundle.
- **Nemotron-H 9B:** the file standing for it is Nemotron Nano 9B v2, `nemotron-nano-9b-v2-int4-metal.v17.giw`. The guard refused every arm, needing 10.68 GB against a live budget near 7 GB, so its figures are priced from the model with no allocation behind them. The 4B's file is `nemotron-h-4b-instruct-128k`.

**Checks.** Where a step built, the ledger's scratch moved by the step's list to within one 16 KiB rounding (0.8B: 8,540,936 B asked, 8,557,320 B allocated; the 9B's Qwen3.5 exactly). The priced figure sits above that by design, each buffer rounded to a page (`stepShape.scratchBytes`), so the longest-context column errs short. KV allocated equals KV priced in every arm. The serve banner of a real load, E4B at 2 slots, reads 0.2 GB of KV a conversation and 4.2 GB of weights, against 0.237 GB and 4.19 GiB here.

**A cost of this measurement.** The 4B's direct safetensors load quantizes in memory, and at 07:45 PST it pushed 1.5 GB of swap-outs while Francis was on the Mac, taking swap in use from about 3.2 to 7.1 GB. It came back down, and no timed run was under way. E4B read 29 MB of swap-outs, with `mac-layout2` building beside it. Neither moves a figure above, which are byte counts. **Load big checkpoints from a bundle,** as `run-4b.sh` now does.

## Francis's decision, 2026-11-04 09:00, relayed by the coordinator: "a seems like a no-brainer"

**In v0.24.0, `--gemma4-step` and `--recurrent-step` default on in serve, on any load with two or more KV sessions, and off in single-session chat, where the scratch buys nothing. The explicit flags stay, so either can be turned on or off.** He decided from the table above: at most 0.996% extra at 4 sessions and at most 1.57% off the longest context, with the 1.38× on Gemma 4 E2B and the 1.58× on Qwen3.5 0.8B on the other side.

**Still open:** Nemotron-H 4B's flag-on arms, night job `step-flags-4b`. If it fails the bar, that comes back to him.

**How "two or more sessions" lands:** a recurrent model keeps one slot unless `--kv-sessions` asks for more, so `--recurrent-step` follows a given count of 2 or more, and plain serve does not batch those models. A Gemma 4 dense model keeps Metal's default slots, so `--gemma4-step` also follows serve's default count. If the memory guard grants one slot where two were asked, the step still builds and takes its scratch.

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

**That last row is older than this change, and it narrows what the default buys.** `serve` runs a model with a vision tower attached one generation at a time, and Qwen3.5 0.8B's checkpoint ships its tower. So the step builds, takes its scratch and does not batch there. The 1.58× is an in-process reading (D4b-4, `ForwardBatch` on four slots), and it does not reach that load through serve. Text-only Qwen3.5 loads, such as a `.giw`, and Nemotron-H do batch. The same holds for a Gemma 4 load with its tower attached. Dropping the step where a tower attaches was not done: serve decides concurrency after the towers load, which is after the build.

