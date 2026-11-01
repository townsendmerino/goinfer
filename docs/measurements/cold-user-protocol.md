# The cold-user protocol — how to run the first-hour test again

This is the protocol and its amendments, moved on 2026-10-31 from §1 of the first-hour task doc, which is now archived as
the findings record at `docs/completed/task-first-hour.md` (what each run found, fixed and gated, R1 to R28). The run
records are the `cold-user-*.md` files in this directory.

## The protocol

**Who may run it.** A window with no access to the source tree, `docs/`, or git history, working in
an empty directory outside the repo, installing the **published tag** the way a stranger would.
Prior contact with the project is not disqualifying but **must be declared in the report**: the
2026-09-06 tester had edited the README in earlier sessions and said so in the first paragraph,
and named the three findings they were most confident were uncontaminated (D1, B1, E2 — each
discovered by failing, not by knowing). The next run uses a tester who has not.

**Allowed sources.** The GitHub README, the release API and its assets, `--help`, error text, and
pkg.go.dev. Nothing else. Note that pkg.go.dev renders doc comments lifted from source, so it *is*
the source's comments — the 2026-09-06 run flagged this itself rather than letting it pass.

**The six scenarios**, each time-boxed at **25 minutes**:

| | scenario | the question |
|---|---|---|
| A | Try it | from nothing to an answer |
| B | Point my tools at it | an OpenAI-speaking client, and an agent CLI |
| C | Embed it | a ≤40-line Go program that prints a completion |
| D | Run bigger than my hardware | a model that does not fit |
| E | Control | the same question through a peer tool, recorded to the same standard |
| F | Show it a screenshot | attach an image, ask about it, then resend the SAME image next turn |

**Rules.**

1. Two attempts per obstacle, then stop and record a dead end. A dead end is a finding, not a
   failure of the run.
2. **The swap-safety rule: stop at the first pageouts.** Scenario D obeyed it and the run survived;
   without it the machine does, and there is no report. **Watch the banner's swap line too**: serve
   prints `swap guard: baseline N GB swap-used` at start (and chat/serve arm a load-time guard for a
   direct `.gguf` build and for a `.giw` load) — record the baseline it printed, and whether the guard
   or the tester noticed swap growth first (S3, `docs/completed/task-never-swap-2026-09.md`).
   **Amended 2026-10-06: the trigger is pageouts, or swap-used growth past the guard's threshold — not
   any rise in swap-used.** On nobara (zram swap) the 2026-10-05 run's watcher killed scenario D twice
   on swap-used rises of +18.9 MB and +14.5 MB with `pswpin`/`pswpout` +0 and `vmstat` si/so 0 throughout
   (`docs/measurements/cold-user-2026-10-05-nobara-pc.md`), so neither attempt reached model output and
   the scenario was recorded as a dead end by the tester's own tripwire, not by the product. Stop on
   `pswpout` (Linux) / Swapouts (macOS) increasing, or on swap-used growing more than the guard's
   default, 512 MB, over the baseline (`GOINFER_SWAP_GUARD`, the same figure goinfer's own watch uses).
   Record any smaller swap-used movement in the friction log; do not stop on it. Note too that goinfer's
   load-time guard covers a direct `.gguf` build and a `.giw` load (including the sidecar a `.gguf` is
   transcoded to, B33), but not a streamed load or a safetensors directory loaded without a sidecar, and
   it says so on stderr as such a load starts (`swap guard (load): not armed`). A watcher you run
   yourself is the only protection for those loads, which is why it must not trip on noise.
3. Record a friction log with **timestamps**, tagged `Guessed` / `Wanted and absent` / `Error`,
   plus a numbers table per scenario. The timestamps are what make "half the time if…" checkable.
4. Record `--version` output for every binary used. (This exists *because* of R2; the 2026-09-06
   run could not, which is how the Metal gap survived to the release.)
5. The time box is a **stop rule, not a target**. Nobody came near it in 2026-09-06 — the longest
   scenario was D at 3.5 minutes — and the box stays at 25 anyway.
6. Deliverable: one Markdown file into `docs/measurements/cold-user-<date>.md`, **verbatim**,
   including its contamination note. It is a measurement record; it is not edited to look better
   afterwards, and a later fix does not rewrite it.

**Per-scenario assertions the next run must record**, so a repeat measures the fixes rather than
re-discovering the findings:

- **Scenario D — "the tool told me before the machine did."** The run passes this leg only if a
  refusal or a budget warning appeared BEFORE any swap growth. On 2026-09-06 the machine told the
  tester and the tool never did; R3's guard exists precisely to invert that, and a cold run is the
  only place it gets tested the way a user meets it.
- **Every scenario — record `--version` for each binary used** (R2). The 2026-09-06 run could not,
  which is how a Mac asset with no Metal in it reached a release.
- **Scenario A — the cold-start total**, which is the number goinfer wins and worth tracking.
- **Scenario F — "the second look is fast."** Attach an image, ask a question about it, and time
  to first token (TTFT) for that turn. Then, in the SAME conversation, resend the byte-identical
  image (not a re-encode or re-crop — the literal file again) with a different question, and time
  TTFT for that turn too. The leg passes only if the second (resent-image) TTFT is visibly faster
  than the first — the resident image-block reuse this exists to check (`docs/multimodal.md`'s
  P9(a)) skips the vision tower and the CPU prefill entirely on a verified-identical image, so the
  second turn should start generating close to as fast as a plain text turn does. As a negative
  control, follow with a THIRD turn attaching a *different* image at the same point in the
  conversation: its TTFT must look like the FIRST (cold) turn, not the fast second one — if it is
  also fast, the reuse is being applied when it should not be, which would mean the server is
  silently answering about the wrong image (see this document's own contamination-declaration
  rule: a false negative here is a finding, not something to explain away). Record all three TTFTs
  in the scenario's numbers table.

**Scenario D stays one class above the brief.** The 2026-09-06 run substituted a 35B-A3B on 16 GB
for the brief's "20–30B", because the README named no MoE and the tester would not download 15–20
GB onto a disk with 23 GB free. That is a *harder* test, and it is the one that found the real
thing. Keep it there.

**Scenario B needs an agent CLI installed.** The 2026-09-06 run could not test one: aider,
opencode, cline, `llm` and codex were all absent, and `continue` matched a shell builtin — a false
positive worth knowing about.

**Scenario F needs a vision-capable checkpoint loaded** (the server started with `--vision <dir>`,
per `serve check`'s own `vision` row). If none is available on the box under test, the tester
records the gap the same way Scenario B records an absent agent CLI — as a dead end for that leg,
not a silently skipped one — rather than substituting a text-only model and reporting a leg that
was never actually exercised.

**`nobara-pc`: `opencode` 1.18.29**, installed into a contained prefix
(`npm install --prefix ~/.local/opt/opencode opencode-ai`; the binary is at
`~/.local/opt/opencode/node_modules/.bin/opencode`). A contained prefix rather than a global
install because npm's global prefix here is `/usr/local` and needs root; nothing on `PATH` and
nothing in `npm config` was changed, so the next tester must use the full path or add it
themselves — which is a step to note in their report, not a hidden one.

**`aider` was tried first and does not install on this box:** the system Python is 3.14.7, and
pip only offers `aider-chat` up to **0.16.0** there — every current version declares an upper
Python bound that excludes 3.14, and pip backtracks to a 2023 release whose pinned `multidict`
6.0.4 then fails to build. This is a fact about aider and 3.14, not about goinfer; if the next
run wants aider specifically it needs an older interpreter.

**The MacBook is still unprovisioned** — it cannot be reached from `nobara-pc`, so whoever runs
scenario B there installs one first and names it.

**`macbook-arm64`: `opencode` 1.18.29 too**, installed 2026-09-06 the same way
(`npm install --prefix ~/.local/opt/opencode opencode-ai`; binary at
`~/.local/opt/opencode/node_modules/.bin/opencode`), so both machines now have scenario B
provisioned identically — same tool, same version, same contained-prefix path. Same caveat
applies: nothing on `PATH`, use the full path or add it in the run's own report.

**Next run:** the **Linux box**, after the R1–R3 tags land. The CUDA path makes "bigger than my
hardware" a different story (expert streaming vs `-stream-weights`), so it is not a repeat. Its
results go in as run 2 and the first-hour record's §4 table gains a column.

### Protocol amendments, after run 2

Run 2 found gaps in the protocol itself, not only in the product. Recorded here as amendments
rather than silently fixed, per this document's own rule that R5 (the protocol) is run, not
shipped — a protocol that changes without a record of why is not reproducible either.

- **The tester window is created with NO working directories granted.** Run 2's session had
  `goinfer/{chat,decoder,multimodal,cmd/serve,cuda,metal,internal/chatapp,internal/gemmaapp,
  scripts,docs/measurements}` pre-granted as "additional working directories" — package-level
  structural knowledge (backends, internal app layout, a docs/measurements directory even
  existing) that a genuinely first-contact window would not carry. The tester declared this
  before starting and the run proceeded anyway, on explicit instruction, with none of those
  directories opened — but a window that finds repo paths pre-granted at all is evidence of
  prior contact by construction, and the correct response is to declare it and count the run
  contaminated, not to proceed carefully around it. Whoever provisions the next tester window
  must confirm zero working-directory grants into this tree before the run starts.
- **"Install the latest release" — the tag is recorded from the release page, never given in the
  prompt.** Run 2 was told to install v0.16.1; `GET
  /repos/townsendmerino/goinfer/releases/tags/v0.16.1` returned 404 — that release never existed.
  A prompt-supplied tag can be stale, mistyped, or (as here) simply wrong, and the tester has no
  way to distinguish "this tag doesn't exist" from "something about my install is broken" without
  checking the release page directly — which is what the README's own install instructions tell a
  real user to do anyway. The tag a run installs should always be discovered from the release
  page at run time, never dictated in advance.
- **Scenario A records download and run as two legs**, because they have different bottlenecks
  and a single combined number hides which one dominates. Run 2's clean-path total (56.5 s) was
  ~98% download time (55.3 s for 1.71 GiB at this run's ~31.5 MB/s) and ~2% launch-to-first-token
  (1.22 s) — a number that would read as "goinfer is slow to start" is actually "this network was
  slow," and only splitting the legs makes that visible. **Scenario E starts Ollama with an empty
  `OLLAMA_MODELS`** for the same reason from the peer side: run 2's Ollama already had 8 models
  cached from unrelated prior work on the same machine, including the exact model class needed,
  so its "cold start" leg measured a warm cache and had to be flagged as non-comparable rather
  than fixed — recording its download leg the same way scenario A does is what makes the two
  sides actually comparable next time, rather than caveated after the fact.
- **Provision the `openai` Python package alongside opencode on both boxes.** Scenario B's
  protocol asks for three clients in order — `curl`, the `openai` Python package, then an agent
  CLI — and run 2's box had the CLI provisioned (per the existing note below) but not the Python
  package, so that leg was skipped rather than run. Symmetric with opencode's own contained-prefix
  provisioning note: install it ahead of the run, not discovered as missing during it.
- **Rule 5 (write only in this directory and `~/models`) permits the tool's own scratchpad.** Run
  2's tester kept working notes in the harness-provided session scratchpad (outside both the run
  directory and `~/models`) before assembling the final report, and flagged it as a rule breach
  because the rule as written does not carve out that space. It should: a scratchpad the tooling
  itself provides and manages is not writing into the project tree or leaving artifacts behind,
  which is what the rule exists to prevent, and forbidding it only pushes a tester toward keeping
  notes in-memory (worse for a long run) or writing them into the one directory the rule is
  actually protecting.

### Protocol amendments, after run 2b

- **A run that skips a scenario for machine safety gets a targeted two-scenario re-run after the
  fix, not a full repeat.** Run 2b correctly stopped short of Scenario D (rule 4) after Scenario
  B's opencode leg alone drove the machine into heavy swap — deliberately loading something 2–5×
  larger next was judged an unacceptable risk to a shared machine, and the run said so rather than
  manufacturing a second, bigger instance of the same failure. Once R13 ships, the next run on
  that machine repeats exactly Scenario D (the skipped one — first line: told-me-first, yes/no)
  and Scenario B's opencode leg (the one that triggered the skip), not Scenarios A/C/E again,
  which nothing about R13 changes.
- **Scenario B records the prompt token count of the agent's first request** (the server's own
  `usage` field on the response, or `count_tokens` if the request never completes) so a memory
  event can be priced afterwards without re-running the request. Run 2b's swap event had no
  recorded prompt size — R13's fix had to reconstruct an equivalent size from `serve check`'s
  harness-scale row shape instead of the run's own numbers, which is strictly worse evidence than
  the run recording it directly would have been.
- **The tester writes its report in its own directory; the goinfer window, never the tester,
  copies it into `docs/measurements/`.** Run 2b's assigned working directory was itself inside the
  excluded tree (`~/tmcode/eval2`, under `~/tmcode`) — a sharper version of run 2's scratchpad
  finding, and independent confirmation that a tester should not be the one placing its own report
  inside the repository at all. The window running this document is the one with a legitimate
  reason to write into `docs/measurements/`; the tester's job ends at producing the report
  somewhere outside the excluded tree.
- **Provisioning gaps do not self-heal between runs — verify, don't just amend the doc.** Run 2's
  amendment ("provision the `openai` Python package alongside opencode on both boxes") was written
  after run 2's box was missing it; run 2b's box was missing it too (`ModuleNotFoundError`,
  installed ad hoc mid-run). Writing the amendment did not provision the second machine. Before
  the next run, confirm the package is actually present on the box in question — do not rely on a
  prior run's amendment having been carried out.

### Protocol amendments, after run 3 (2026-10-01, v0.19.0, nobara-pc)

- **The handoff starts with the launch command, before the prompt:** `mkdir ~/gi-cold-<date> && cd ~/gi-cold-<date>`,
  then open the session there. Run 3's prompt named the directory, but the session was opened in
  `~/mycode/gocudrv-fork`, with `~/mycode/goinfer/...` directories granted and a goinfer skill loaded. The tester
  declared all of it, and the owner accepted the contamination for that run
  ([`cold-user-2026-10-01-nobara-pc.md`](cold-user-2026-10-01-nobara-pc.md)).
- **Scenario F's resend leg conflicts with the server's one-image-per-request limit.** Sending the same image again
  in the same conversation fails with `400: v1 supports 1 image per request, got 2` while the first image is still in
  history, so run 3 had to drop it to text first, and the leg then measured a new request, not reuse. Rewrite the leg,
  or keep it, once a conversation can carry more than one image.
