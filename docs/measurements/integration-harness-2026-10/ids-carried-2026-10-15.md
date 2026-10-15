# Carried token ids (proposal 2): build, tier-0 and tier-1 gates, tier-2 plan (2026-10-15)

nobara-pc, RTX 2070 SUPER, driver 595.91.07. Branch `ids-carried-2026-10` from `origin/main` 772b1fb2 (KV2 merged). The design and
the gates are id-roundtrip-step0-2026-10-15.md's §(a) to §(c), as its §(c) Amendment 1 tiers them; the logging is its §(e). This
record holds the results and the readings the build needed. Code: `internal/serveapp/carried_ids.go`,
`internal/serveapp/carried_ids_render.go`, `internal/serveapp/carried_ids_mint.go`; user-facing description in `docs/server.md`
("Carried token ids on /v1/responses").

## What was built

- **The blob.** `goinfer-reasoning-v2:` then base64 of: version, a flags byte (whole turn or reasoning only), key id, model and
  tokenizer identity, namespace, the reasoning text, the varint-packed ids, and a length-prefixed HMAC-SHA256 over everything but
  the text (the flags byte included, a detail the design did not name). v1 blobs are still read. The identity is a hash of the
  tokenizer's vocabulary and of the weights' fingerprint less its mtime, plus the adapter. The namespace is the single local one
  (`requestNamespace`), inside the MAC as designed.
- **The key.** 32 bytes from `crypto/rand` per start, or `--ids-key-file` (32 bytes or more, mode 0600, refused otherwise).
  `--carry-ids=false` turns the feature off (no key: v1 blobs, every turn re-tokenized); it exists for G-ids-3's off arm.
- **Use.** `loadedModel.encodePrompt` renders as the routes did. For each assistant turn that carries a blob it checks, in order,
  key id, MAC, model, namespace, then locates the turn's bytes by rendering once more with that turn emptied, and uses the ids
  only if they decode to a run of the rendered text that holds every difference and lines up with the template's special tokens.
  Everything outside the carried ranges is tokenized exactly as before (`encodeRegions`).
- **Carriers.** The reasoning item carries the whole turn's ids. A reply with no reasoning gets a carrier-only reasoning item
  (empty `content`) when the request has `include: ["reasoning.encrypted_content"]`. Streamed, a reasoning item closes when the
  answer starts, so a streamed answer's item carries the ids up to the end of the reasoning block (flags: not whole); a call is
  emitted after the turn ends, so the item before it carries the whole turn. A turn a stop string ended carries v1.
- **Counting (condition b).** `-log-requests` ends with `ids=<used>/<carried> ids_fallback=<reasons>`; `GET /admin/status`
  reports `carried_ids` totals by reason.
- **§(e), the serve side.** After `total=` the `-log-requests` line carries `reused`, `slot`, `div`, `canon`, `ids`,
  `ids_fallback`, in that order. `slot` and `div` read two new `decoder.Generation` fields, `PrefillSlot` and `PrefillMatch`,
  set on the CPU session path and the resident path (`kvAcquired`). `--log-token-ids <file>` writes one JSON line per generation:
  `t`, `route`, `id`, `prompt_len`, `prompt_sha256` (of the ids as little-endian uint32s, `idsSHA256`), `reused`, `slot`, `gen`,
  `canon`. `matrix.py` takes `"token_ids": true` beside `"bodies": true`.

## Readings of the registration the build needed

1. **G-ids-0 rule 1's `ids=1/1` is read as G-ids-1 reads archived history.** Every request after a session's first carries the
   night's earlier replies with their v1 blobs. Each of those is a carried turn that falls back as `v1`, so the line reads
   `ids=1/(1+h) ids_fallback=v1,…` with h such turns. The test asserts exactly that: the reply's own turn uses its ids, and the
   only other fallbacks are the h `v1`. A defect's line is `ids=0/(1+h)` with the h `v1` and then its named reason.
2. **"Today's tokenization"** is the same request rendered with no ids used. The reasoning text is read as main reads a v1 blob:
   from a blob that parses (whatever its MAC), else from `content`. So defect (a), which edits the text inside a signed blob,
   renders the edited text, as an edited v1 blob would today.
3. **A reason the registration did not list: `history`.** Mellum2.1's and the Qwen templates drop the reasoning of turns before
   the last user query. Such a turn's ids cannot match what is rendered, and that is not a client edit, so it is counted as
   `history`, not `decode`. It fires in neither gate below; it will in live runs after a new user message.
4. **Defect (h)** is planted as a `<tool_call>` special id at the end of the reasoning run, with the blob's text carrying the
   literal `<tool_call>` there: the ids decode to the blob's text exactly, and the template renders that text as content.
   **Defect (e)'s** client split re-encodes each run one character at a time, under the server's key id with no MAC.
5. **Not covered:** a reply with several calls that the client echoes as separate assistant turns (it re-renders as several turns
   today and falls back as `decode`); `/v1/chat/completions` and `/v1/messages` (no carrier, per the design).

## Tier 0 (CPU, seconds): red before, green at 12bf54d7

**Red before.** The tests, run on `origin/main` 772b1fb2 in a throwaway worktree with an inert stub (the blob format and key code,
and the identifiers the tests name, with main's behaviour: no ids carried or used):
- `TestIDsCarried_G0`: the first reply's next request read `ids=0/0`, where `ids=1/1` is required. Defect (a) then voided,
  because main does not read a v2 blob's text.
- `TestIDsCarried_tinyFallbackAndKV`: the `-log-requests` line had no reuse or ids fields.
- `TestIDsCarried_carrierItems`: no v2 blob in either form, and no carrier-only item.
- `TestIDsBlob_roundTrip`: `reasoningItemText` returned no text for a v2 blob.
- `TestIDsKeyFile` passed: it tests the key code the stub shares.

**Green (12bf54d7).**

| test | what it ran | result |
|---|---|---|
| `TestIDsCarried_G0` (G-ids-0) | 45 replies; cases: untouched 45, visible-content edit 45, (a) 45, (b) 44, (c) 2, (d) 1, (e) to (m) 45 each | the next prompt starts with the slot's id record in **45 of 45**, the 4 non-canonical replies included; every defect gives today's tokenization, its reason and no error; 28.7 s (registered basis about 15 s) |
| `TestIDsCarried_tinyFallbackAndKV` (G-ids-2's tier-0 half) | arms A-warm, A-cold, B, C, D1, D2 on the tiny committed fixture, one KV session | see below; 0.08 s |
| `TestIDsCarried_carrierItems`, `_streamedAnswer`, `_flag`, `TestIDsBlob_roundTrip`, `TestIDsKeyFile` | carriers streamed and not, a streamed answer's reasoning-only blob, `history` after a new query, the flag, the format, the key file | pass |

The tiny fixture: `tinyChatCheckpoint`'s weights with a tokenizer filled to the model's 256 ids by two-character pieces no merge
reaches, so a generated filler id is a split the BPE never produces. Turn 1 generated 12 ids from a 21-token prompt, with
`canon=0` (the planted split is present; the test voids otherwise).

| arm | reused rows (of the prompt) | ids line | prompt ids | output |
|---|---|---|---|---|
| A-warm | 33 of 58 (the 21-token prompt and all 12 generated ids) | `ids=1/1` | start with turn 1's prompt and ids | `Sf[KfCySf[Kf` |
| A-cold | 6 (the preamble shared with the other conversation) | `ids=1/1` | identical to A-warm's | identical to A-warm's |
| B (forged, no MAC) | 6 | `ids=0/1 ids_fallback=mac` | identical to D's | identical to D's |
| C (another key) | 6 | `ids=0/1 ids_fallback=key` | identical to D's | identical to D's |
| D1, D2 (no carrier) | 6, 6 | `ids=0/0` | D1 = D2 (not void); first differ from A's inside turn 1's reply | `Sf[AbPbYfPKbYd` both |

`/admin/status` after the arms: 4 carried, 2 used, `mac` 1, `key` 1. The test reads the prompt ids twice: from `--log-token-ids`
(the handler's) and from the route's renderer, and requires them equal.

`python3 scripts/touched.py`: green (serveapp 235 s, its whole package; tokenizer 13 s; decoder's beside-file tests 6 s). Tagged
vet (`realckpt`; `cuda goinfer_testhooks` on serveapp, decoder and cuda) and CI-pinned staticcheck (untagged, `cuda
goinfer_testhooks`, `realckpt`; canary red first) clean.

## Tier 1 (by day, GPU, under the timing lock): G-ids-1 trimmed, PASS

Cells and order exactly as Amendment 1 registers them: the 4 misses and 8 controls from the 41 that held, in `select.txt`'s order
(`ids-carried-tier1/cells.txt`). Load as step 0: `--model ~/models/mellum2.1-thinking --ctx 16384 --moe-cache-experts
--windowed-kv=false`, KV f32 (banner), C′ cache 45 slots. Harness `ids-carried-tier1/idstier1_test.go.txt` (step 0's, extended),
binary built at 12bf54d7 plus the harness. Per miss, the next request was also sent twice (`max_output_tokens` 1): warm, right
after the reply; cold, after another session's first request (`search` r2 #0). The no-carrier arm's prompt ids are the same body
rendered with the reasoning item's `encrypted_content` removed.

**Cost.** Estimated before the run at 11 minutes, not the amendment's "under 6": its basis, 13.6 s a request, came from step 0's
consecutive replay, where each request reused its predecessor. Run: **512 s with a 25.5 s load**.

| cell | prompt (night −18) | generated | canonical | next request with ids | starts with the slot |
|---|---|---|---|---|---|
| `write` r2 #0 | 6,090 | 205 | yes | `ids=1/1` | yes |
| `write` r2 #2 | 6,687 | 232 | yes | `ids=1/3`, v1 ×2 | yes |
| `write` r2 #3 | 7,131 | 545 | yes | `ids=1/4`, v1 ×3 | yes |
| **`write` r2 #4** | 7,858 | 723 | **no** | `ids=1/5`, v1 ×4 | yes |
| **`write` r2 #5** | 8,548 | 2,238 | **no** | `ids=1/6`, v1 ×5 | yes |
| `write` r2 #8 | 10,317 | 124 | yes | `ids=1/9`, v1 ×8 | yes |
| `write` r2 #10 | 10,812 | 936 | **no** | `ids=1/11`, v1 ×10 | yes |
| **`fix` r3 #1** | 6,375 | 287 | yes | `ids=1/2`, v1 ×1 | yes |
| `fix` r3 #7 | 8,826 | 85 | yes | `ids=1/8`, v1 ×7 | yes |
| **`fix` r3 #9** | 10,501 | 1,133 | **no** | `ids=1/10`, v1 ×9 | yes |
| `write` r3 #8 | 9,417 | 145 | yes | `ids=1/9`, v1 ×8 | yes |
| `write` r3 #10 | 10,000 | 365 | yes | `ids=1/11`, v1 ×10 | yes |

The misses (bold), sent:

| miss | warm: reused / prompt | cold: reused / prompt | warm ids = cold ids | first difference from the no-carrier arm |
|---|---|---|---|---|
| `write` r2 #4 | 8,581 / 8,656 (the whole slot), 1.0 s | 6,072 / 8,656, 26.6 s | yes | +508 into a 723-token reply |
| `write` r2 #5 | 10,786 / 10,839, 0.8 s | 6,072 / 10,839, 49.4 s | yes | +1,430 into 2,238 |
| `fix` r3 #1 | 6,662 / 7,091, 4.3 s | 6,072 / 7,091, 10.3 s | yes | none: this build's reply is canonical |
| `fix` r3 #9 | 11,634 / 11,887, 2.8 s | 6,072 / 11,887, 60.1 s | yes | +532 into 1,133 |

The cold arms' 6,072 is exactly the common prefix of their prompt with the other session's (the shared developer message and
tools), which is what cold means here.

**Rule:** all 12 re-render to their generated ids with the feature on (12 of 12), with no fallback but `v1` on archived history
(105 `v1` and nothing else in `/admin/status` over the run). The 4 misses' prompt ids are equal warm and cold (4 of 4). They differ
from the no-carrier arm's inside the reply in the 3 misses whose regenerated reply is non-canonical. **`fix` r3 #1's greedy reply
on this build is canonical** (287 tokens, against step 0's 370), so its carrier and no-carrier prompts coincide. That sub-check has
nothing to check for this cell; it is not a void condition. Every non-canonical reply this run produced (4 of 12, one of them a
control) re-rendered to its own ids. **Void conditions:** none fired: offset −18 in 12 of 12; the slot started with the route's
render in 12 of 12; no render error; flags as registered; every arm's `cached_tokens` matches its stated state. **Verdict: PASS.**

The greedy replies differ from step 0's (other lengths in 11 of 12): a different build and a different reuse history before each
cell. Archived: `ids-carried-tier1/run.jsonl.gz` (per cell, no text), `token-ids.jsonl.gz` (the `--log-token-ids` file: the
prompts' hashes and the generated ids, the content class of step 0's committed `run.jsonl.gz`), `run.log`.

## Owner decisions (2026-10-15), before G-ids-3 runs

- **`history` is accepted as a disclosed amendment, as a label only.** For grading, a `history` fallback counts exactly like
  any other non-`v1` fallback: G-ids-3's rule reads it as a fallback, and nothing in any gate is excused because a fallback is
  labelled `history`. The label says why the ids were not used; it does not change what the gate counts.
- **G-ids-3 is queued tonight**, pinned at `2a0d11bb`, three jobs with the early stop as registered. It is tier 2 because its
  outcome (cold continuations and compactions in real Codex runs) cannot be measured by day.
- **PR #31 merges only after G-ids-3 is graded**, by hand on green CI. No code is pushed to the branch before then.
- **Out of scope, unchanged:** streamed plain answers, tool calls echoed back as separate turns, Chat Completions and
  `/v1/messages`. **When G-ids-3 is graded, every cold continuation it shows is attributed**, and the report says how many come
  from those uncovered cases, against the rest.

## Tier 2 (night): G-ids-3, prepared, NOT queued

**The off arm is re-run.** Amendment 1 cites the 2026-10-15 after run (serve at 695ce16c) as the off arm only if nothing on the
reuse path changed since. KV2's merge (772b1fb2) is on that path, and so is this build. So both arms run, from one pinned binary: off is `--carry-ids=false`, on is the default. The budget doubles, as registered.

**Design** (`ids-carried-tier2/plan-ids3.json`): `mellum-16k-codex-f` (Mellum2.1, `--ctx 16384 --moe-cache-experts`, the default
windowed KV, as the after run), Codex, 5 tasks × 3 repetitions × 2 arms = 30 runs, `"bodies": true` and `"token_ids": true` on
both arms. Interleaved by task, repetition-major, so the on arm's first 5 runs are one of each task; within each task the arm that
goes first alternates. **One night job per repetition** (about 110 minutes each, under the 3-hour cap), all writing one output
directory. Jobs 2 and 3 first run the registered early stop (`gids3_check.py --stop-check`): if the on arm's first 5 runs show no
cold plain continuation and no fallback other than `decode`, they record "NOT RUN" and exit.

**The count** (`gids3_check.py`): follow-ups classified from Codex's own session files exactly as `diagnose_after_compactions.py`
classifies the after run, per arm; fallbacks from serve's `ids_fallback`. Its check: on the after run's archive it gives **15
cold, 15 partial, 213 full, 20 compactions**, the record's own figures.

**Fit check, by day** (`ids-carried-tier2/fitcheck.sh`, under the timing lock, the pinned binary with the job's flags, both arms):
loads; C′ cache **57 slots** in both arms (the after run had 62; the void condition is the cap differing between arms, and it does
not); windowed KV 1,536 positions per sliding layer; KV f32; one conversation × 16,384. `kwin_wayland` held 17 MiB of the card.

**Dry run** into empty directories (`DRY_RUN=1`, reps 1 and 2): every pinned file found, the sha256 checked, the configuration
list built, the early-stop check run on nothing ("CONTINUE: the on arm has 0 of the 5 runs").

**Pinned** by `pin-ids3.sh` at 2a0d11bb (code identical to this PR's head) into a directory outside the repository, as the I3 jobs
were. **To queue** (from the main checkout; three jobs, about 110 minutes each, so start by 00:30 for all three to fit before
06:30, or about 110 minutes in all if the early stop holds):

```sh
python3 scripts/night.py add ids3-r1 --est 120 --by "carried ids, G-ids-3" --doc docs/measurements/integration-harness-2026-10/ids-carried-2026-10-15.md -- env REP=1 B=$HOME/goinfer-bench/ids3-2026-10 bash $HOME/goinfer-bench/ids3-2026-10/run-ids3.sh
python3 scripts/night.py add ids3-r2 --est 120 --by "carried ids, G-ids-3" --doc docs/measurements/integration-harness-2026-10/ids-carried-2026-10-15.md -- env REP=2 B=$HOME/goinfer-bench/ids3-2026-10 bash $HOME/goinfer-bench/ids3-2026-10/run-ids3.sh
python3 scripts/night.py add ids3-r3 --est 120 --by "carried ids, G-ids-3" --doc docs/measurements/integration-harness-2026-10/ids-carried-2026-10-15.md -- env REP=3 B=$HOME/goinfer-bench/ids3-2026-10 bash $HOME/goinfer-bench/ids3-2026-10/run-ids3.sh
```

**Estimate:** the after run's 2 h 28 min for 15 runs is 9.9 minutes a run; 10 runs a job plus 10 server starts (about 30 s each)
and the sandbox self-test is about 110 minutes. Re-pin before queueing if the branch's code changes after 2a0d11bb.
