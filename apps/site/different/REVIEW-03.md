# REVIEW-03: "Tool calls that can't come out malformed"

Draft: `03-tool-calls-that-cant-be-malformed.md` (body 863 words incl. code block). `reviewed:` left empty. Build passes.

## Claims and sources

| Claim | Source |
|---|---|
| Baseline: Qwen2.5-7B, 12 tools, auto: 14/111 unusable (12.6%), 10 of 14 invented names, 8x `git_commit`, plus 10 bad-args parses = 24 of 111 | `docs/measurements/tool-call-failure-t0-2026-09-23.md`, Results table and "What the failures are" |
| nobara-pc, RTX 2070 SUPER, driver 595.91.07, q4_k_m, 10 greedy + 300 sampled | same, Results; `tool-union-2026-09-24.md` Results |
| With the union: 0 unknown name, 0 unparsed, 0 args invalid, 1 truncated reported apart | `tool-union-2026-09-24.md`, "Gate C" table |
| Prose turns byte-identical 310/310 (1.5B), 310/310 (llama3-1B); 630/630 total | `tool-union-2026-09-24.md` "Gate A'"; `CHANGELOG.md` ("630/630 outputs byte-identical") |
| Prose-turn decode 0.999 to 1.001; ungated first build 0.807 (1.5B, T=0.7) | `tool-union-2026-09-24.md`, Gate B and B' tables |
| llama3-1B under auto: 12/185 to 4/185; 188 armed calls all clean | `tool-union-2026-09-24.md`, "llama3 option (c) - results" |
| Coder 0.5B/1.5B: never write opener, parser fix 0 to 236 / 219 of 300; on-intended 9-11% vs 27% (7B); 0 wrapped in 1,200 | `tool-call-failure-t0-2026-09-23.md`, "What the failures are" and "Follow-up A - results" |
| Modes: named / required / auto; 400 for a named tool not in `tools`; llama3 arms on `{"name": "`; each call in a turn independent; spec servers leave auto unconstrained | `docs/server.md` "What `tool_choice` actually constrains"; code: `internal/serveapp/tools.go` (`constrainForcedTool`, `constrainToolUnion`), `constrain/lazy.go` |
| Union grammar = parallel branches, narrows at the name; tests | `constrain/tools_union.go`; `constrain/tools_union_test.go` (`TestToolCallsGrammar_acceptsExactlyTheUnionOfSingleToolGrammars`, cases `read_fil`, wrong-tool args); `constrain/tool_grammar_test.go` (`TestToolGrammar_property`: 100 seeds x chatml/llama3/mistral) |
| Token mask to -inf | `constrain/constrain.go` package comment |
| Families constrained / parsed only / none; mistral v0.3 not run ("no v0.3 checkpoint here") | `docs/tool-call-coverage.md` table and "What can be claimed" |
| Schema subset; unsupported keyword, freeform object, >64 props fail compile; named choice gets 400, otherwise silently unconstrained | `constrain/schema.go` (`schemaKeywords`, `compileObject`), `constrain/tools_union.go` doc comment, `internal/serveapp/tools.go` (`constrainToolUnion` returns on error; `constrainForcedTool` 400) |
| `GOINFER_TOOL_UNION=0` opts out; single forced tool still constrained | `docs/env-vars.md` row; `testdata/env_reads.txt`; code |
| Not measured: MoE, one transcript, CUDA only; a grammar cannot prevent a length stop | `tool-union-2026-09-24.md` "Not covered" / "Not established"; Gate C rule |
| `goinfer-serve check <url>` "tools, harness-scale" row | `docs/integrations/opencode.md`; `internal/servecheck/check.go` |
| Server start line, default addr 127.0.0.1:8080 | `docs/integrations/claude-code.md`; `internal/serveapp/main.go` (`-addr`) |
| Response shape (`content` null, `tool_calls[].function.arguments` a JSON string, `finish_reason` "tool_calls") | code: `toAPICalls` and the non-stream path in `internal/serveapp/tools.go`. The example call values are illustrative and were NOT run; the page says so |

## Conflicts between records

1. **Brief says Llama 3 is left unconstrained. The records say otherwise.** Newest record (`tool-union-2026-09-24.md` option (c), `CHANGELOG.md`, `docs/tool-call-coverage.md`, `docs/env-vars.md`): llama3 is constrained under `required`/named from token 1, and under `auto` when the reply begins `{"name": "`. Only a call after prose, or a non-`name`-first shape, is parsed only. `docs/tasks/task-tool-grammar-union-2026-09.md` still says "llama3 left unarmed" in places (older status line). I followed the newest.
2. **T0's llama3-1B baseline is 9/157 (5.7%) in one reading and 6.5% in another; the archived run did not reproduce from its own binary.** I used only the same-run comparison in the option (c) record (12/185 to 4/185) and did not quote 5.7%.
3. **Older docs said 2+ tools were unconstrained** (`docs/tasks/task-tool-grammar-union-2026-09.md` section 1 quote of server.md). Superseded by T1-T3 on 2026-09-24; server.md itself notes it.
4. **Task doc claims neither Ollama nor llama.cpp constrains; it carries a 2026-09-24 correction: llama.cpp builds a similar lazy grammar (read from source).** I did not mention peers in the page, per the tone rule. Question 2 below.
5. **Code vs. code comment (unresolved, left out of the page).** `forcedTool` in `internal/serveapp/tools.go` returns the lone tool when there is exactly one tool, including under OpenAI `auto` (only `none` and named differ). `constrainForcedTool` then builds a grammar that starts at token 1. On my reading that makes a one-tool OpenAI `auto` request a forced call with no prose option, while the comment on `constrainForcedTool` says the model "is still free to answer in prose". The Anthropic path (`anthropicForcedTool`) deliberately does not force under `auto`. I found no test for the OpenAI one-tool case. I did not run it, so I left it out of the page.

## Left out as unverified

- Any claim about Metal or CPU backends: every measurement is CUDA. The `--embed-int4` Metal default (`docs/quantization.md` "Known issue") is not mentioned since the page makes no Metal claim.
- Any end-to-end run through a real agent showing zero malformed calls (opencode/Claude Code pages record successful runs but the opencode page says its run predates the grammar).
- Mistral v0.3 behaviour on a real checkpoint; MoE rates; other transcripts.
- A serveapp-level test of the union (grep found none referencing `constrainToolUnion`); the page cites only `constrain/` tests. Not run by me (no tests were run).
- Speculative-server behaviour beyond "auto is left unconstrained" (unit-tested per the record; no live run).

## Questions for the owner

1. Item 5 above: is OpenAI `auto` with a single tool meant to force the call? If yes, should the page (and `docs/server.md`) say so? If no, that is a bug to look at.
2. Should the page say plainly that llama.cpp also builds a lazy grammar for the same problem (task doc's 2026-09-24 correction)? I left it out to stay on goinfer's own evidence.
3. The headline "14/111 to 0" is one transcript in which 9 of the 14 failures fell on a single turn (T0 caveat). Is the Facts line acceptable as written, or should it carry that caveat?
