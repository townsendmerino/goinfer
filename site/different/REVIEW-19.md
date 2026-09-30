# REVIEW-19: "Find out before your agent does"

Draft: `site/different/19-find-out-before-your-agent-does.md` (body ~890 words counting code blocks). `reviewed:` left empty. Builds clean with `-drafts`.

## Claims and sources

| Claim | Source |
|---|---|
| `goinfer-serve check [url]`, default http://127.0.0.1:8080, starts nothing, flags `-api-key` (env GOINFER_API_KEY), `-model`, `-long-prompt` (2000, 0 skips), `-timeout` | `internal/servecheck/cmd.go` (`Run`, usage string) |
| Dispatched as `check` first arg of goinfer-serve | `internal/serveapp/main.go` ~line 319 |
| Row `tools, harness-scale`: 12 tools, nested params, temp 0, max_tokens 96, first turn only (no result round trip); `ok` line text `call get_weather(...) among 12 tools` | `internal/servecheck/check.go` `ToolsHarness`, `harnessScaleTools`, `toolCall` |
| Skip text (`model did not call the tool under a harness-scale (12-tool) schema — ...`), skip is not a fail; only FAIL exits non-zero | `check.go` `toolCall`, `cmd.go` end of `Run` |
| Ok/skip sample output (7B) | `docs/integrations/opencode.md` ("Run serve check before opencode") |
| Skip sample output (1.5B, Mac CPU, 2026-09-07, v0.17.1) | `docs/measurements/cold-user-2026-09-07-macbook-arm64.md` Scenario B |
| 7B q4_k_m, CUDA, ctx 16384, nobara-pc, 2026-09-08, v0.17.2: all 8 checks, load 26.2 s, 6,824 / 8,192 MiB, two turns 7,165 / 7,399 input tokens, Read + Glob calls | `docs/integrations/opencode.md` "The working configuration, measured"; also `docs/tasks/task-first-hour.md` R14 |
| 1.5B, opencode printed fake JSON call as prose twice, ~173 s each, 2026-09-06, opencode 1.18.29 | `docs/measurements/cold-user-2026-09-06-nobara-pc.md` (opencode leg); row added as a consequence: `docs/tasks/task-first-hour.md` R11 |
| 0.5B harness-scale skip measured 2026-09-07 on nobara-pc; other four registry checkpoints "not yet measured" | `docs/capability-matrix.json` ("tools" fields, line ~693 and four others); `pull/registry.go` `Checkpoint.Tools` |
| 0.5B opencode 2026-09-18: first request 9,917 tokens, never completed | `docs/measurements/cold-user-2026-09-18-nobara-pc.md` Scenario B |
| Mac 1.5B skip; Mac 3B skip, "not a memory problem" | 09-07 Mac record; `task-first-hour.md` R13-follow-on "fourth live data point" |
| Mac 7B q3_k_m: load check said "79% of budget", opencode then swapped ~9.7 GB | 09-07 Mac record; the guard was later rewritten (`task-first-hour.md` R13-follow-on) |
| `fit`: `goinfer-chat fit <path>` (chat binary only), per-backend PLACEMENT + reason, default ctx 8192, `-ctx` refused not shrunk, `-measure`, accepts `hf:` refs | `internal/fitcmd/fit.go`; dispatch in `internal/chatapp/main.go` ~line 137 |
| `fit` loads the checkpoint (32.4s on gpt-oss-20b), output sample lines | `cold-user-2026-09-18-nobara-pc.md` Scenario D |
| fit said "fits" but the real load swapped; load-time guard changed afterwards | `CHANGELOG.md` Unreleased entry "Loading a large .gguf ... drove real, incremental swap growth" |
| Models page tool line is measured, never guessed | `pull/registry.go` comment on `Tools`; `site/internal/site/model.go` ~line 351 |
| Claude Code page measured 2026-09-02, before the row existed | `docs/integrations/claude-code.md` |
| Tool grammar constraint since 2026-09-24 | `docs/integrations/opencode.md`, `docs/tool-call-coverage.md` |

## Conflicts between records, and how I resolved them

1. **"fit answers before loading" (your brief, and the home page card "before loading it")** vs `fit.go`, whose own header and usage say it DOES load the checkpoint (header-only version "still open"). I wrote "before serving" and stated that fit loads (32.4s). Please decide whether the home-page wording needs a fix.
2. **Nobara 1.5B, 2026-09-06: CUDA or CPU?** opencode.md's table says "nobara-pc (CUDA)". The cold-user record says the server was launched with no `--backend` (CPU). I wrote only "nobara-pc, Qwen2.5-Coder 1.5B" without naming a backend.
3. **"Matched every outcome"**: opencode.md says in every run the row's answer matched opencode. The records show only ONE case where an `ok` came first and opencode confirmed it (7B), and the 1.5B `skip` was run AFTER the opencode failure (row didn't exist on 2026-09-06). The Mac 1.5B and 3B skips had no opencode run on that model (the Mac record's "correctly predicted what opencode later did" refers to opencode run on a different model, the 7B q3_k_m, which was killed for memory). I described the claim narrowly and put this in the table.
4. **Mac 7B load check:** opencode.md calls the load-time message "comfortable-sounding 79% of budget"; the Mac record shows the actual message was `fit is tight ... (79% of budget)`. I quoted only "79% of budget".

## Left out as not verified

- `docs/server.md` does not mention `serve check` (grep found nothing), so it is not cited.
- `pull/curated.json` has no tools field; the Tools row lives in `docs/capability-matrix.json` (identical copy in `pull/`).
- Whether `fit`'s own report now prices the transient .gguf-load peak (only the load-time guard is documented as changed). Stated as "not stated in the records".
- Date of the 1.5B harness-scale skip on nobara (task-first-hour R11 says verified on real hardware, no date). Date of the 3B run (only "after the tag", 2026-09-08).
- Whether the 0.5B/1.5B skips still hold after the 2026-09-24 grammar-constrained tool calls: stated only as "not re-measured".
- Mac 7B q3_k_m tag "would swap" mechanism details were left to the records.

## Questions for the owner

1. Is "a handful of runs, one Linux box" the right framing, or should the page wait for a fresh 1.5B/3B check-then-opencode pair (the check on both, opencode on the same model) so "matched" has more than one clean case?
2. Should the check's `ok` on Qwen2.5-7B be re-run now that tool calls are grammar-constrained (2026-09-24)? A `skip` for the 0.5B may no longer hold.
3. Fix the home-page "before loading" wording for `fit`, given that it loads?
