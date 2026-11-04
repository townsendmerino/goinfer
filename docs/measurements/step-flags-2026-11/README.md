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
