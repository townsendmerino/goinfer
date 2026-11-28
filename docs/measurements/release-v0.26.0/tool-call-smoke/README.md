# v0.26.0: a tool-call smoke at the candidate, against v0.25.0

<!-- citations-at: bfce89962 -->

A release smoke, not a measurement: nothing here was registered and no figure is a rate. v0.26.0 makes a tool's parameters
follow `$ref`, `type` lists, `anyOf` and `oneOf`, where v0.25.0 left such a value free to be any JSON value. The smoke asks
whether a served model's tool calls still work at the candidate, and whether the new keywords are held where the docs say.

**Answer.** With a named tool or `required`, every call the candidate returned satisfies its tool's schema, on Metal and on the
CPU; v0.25.0 breaks the new keywords in 8 of its 40 named replies and in 1 of its 9 calls under `required`. The agent
transcript replays as it did on v0.25.0. Under
`auto`, one reply in the third run came back as a call that breaks its schema, the same reply on the candidate and on
v0.25.0: a call the model writes without its `<tool_call>` wrapper is parsed and never constrained. That is not a regression,
and it is the one place the new keywords do not reach.

## What ran

`toolcall_smoke.py` starts one `serve` on Qwen2.5-Coder-1.5B (`q4_k_m` `.gguf`, loaded at `int4`) and checks every reply from
the HTTP body against the tool's schema with its own validator. Three arms, one after another, on an Apple M1 Pro with 16 GB
(macOS 26.7.3), each under the shared timing lock, 2026-11-28:

| arm | binary (`go version -m`) |
|---|---|
| candidate, Metal | `backend/metal/cmd/serve` at `bfce89962`, go1.27.2, not modified |
| candidate, CPU | `cmd/serve` at `bfce89962`, go1.27.4, not modified |
| v0.25.0, CPU | `cmd/serve` at the `v0.25.0` tag, go1.27.4, not modified |

The Metal binary was built in a workspace whose `go` line selects go1.27.2; the other two were built with `GOWORK=off`.

| group | requests | what it sends |
|---|---|---|
| `replay` | 10 turns | the 12-tool agent transcript (`scripts/w4_transcript_base.json`), a named tool each turn, temperature 0; these tools use none of the new keywords: the control |
| `keywords` | 10 | five tools that use the new keywords (a `$ref` into `$defs`, a nullable `anyOf`, a `type` list, a `oneOf` told apart by a `const`, a recursive `$ref`), two prompts each, named, temperature 0 |
| `sampled` | 30 | the `keywords` requests at temperature 0.8, three seeds each |
| `auto` | 5 | all 17 tools offered, `tool_choice` `auto`, one prompt per keyword tool, temperature 0 |
| `required` | 10 | all 17 tools offered, `tool_choice` `required`: the model picks the tool and the grammar is the union of every tool's |
| `auto5` | 10 | the five keyword tools alone, `tool_choice` `auto` |

| run | time (PST) | groups | script |
|---|---|---|---|
| 1 | 06:22:30 to 06:35:30 | `replay`, `keywords`, `sampled`, `auto` | `run1/toolcall_smoke.py`: a replay turn had 256 tokens and a cut reply was counted as prose |
| 2 | 06:36:18 to 06:45:04 | the same | 1,024 tokens for a replay turn, and a cut reply counted apart; that intermediate version of the script was not kept |
| 3 | 08:00:14 to 08:03:29 | `required`, `auto5` | `toolcall_smoke.py`, which adds those two groups and `--groups` and leaves the other four as run 2 had them |

## Results

Replies that pass (status 200, `finish_reason` `tool_calls`, the chosen tool, arguments that satisfy its schema):

| group | candidate, Metal | candidate, CPU | v0.25.0, CPU |
|---|---|---|---|
| `replay` (runs 1 and 2) | 10 of 10 | 7 of 8, then the replay stops | 7 of 8, then the replay stops |
| `keywords` (runs 1 and 2) | 10 of 10 | 10 of 10 | **8 of 10** |
| `sampled` (runs 1 and 2) | 30 of 30 | 30 of 30 | **24 of 30** |
| `required` (run 3) | 9 calls, **9 pass**, 1 cut | 9 calls, **9 pass**, 1 cut | 9 calls, **8 pass**, 1 cut |
| `auto` (runs 1 and 2) | 0 calls of 5 | 0 calls of 5 | 0 calls of 5 |
| `auto5` (run 3) | 0 calls of 10 | 1 call of 10, **which fails** | 1 call of 10, **which fails** |

Runs 1 and 2 agree on every row.

- **v0.25.0's failures are the new keywords, unheld** (`run2/smoke-cpu-old.log`): `create_event` with an attendee's `email`
  `null` where the schema says a string, or an attendee written as a string; `duration_minutes`, `threshold` and `placed`
  matching no `anyOf` alternative. Eight in the two named groups and one under `required` (`set_threshold` with a string for
  `threshold`). The candidate has none.
- **The replay's eighth turn is the model, not the grammar.** On the CPU, at temperature 0, the reply runs to the token budget
  on the candidate and on v0.25.0, at the same turn. `turn8-cpu-new.log` replays the candidate's with 3,000 tokens:
  `write_file`'s `content` argument is in a repetition loop (8,664 characters, one 60-character window six times in the last
  2,000). A cut reply has no call to carry the history on, so the replay stops there. On Metal the same turn completes, and all
  ten pass.
- **The one cut reply under `required`** is a `create_event` prompt whose reply ran to the 400-token budget on all three arms;
  a cut reply carries no call and is counted apart. Of each arm's nine calls, seven went to the tool the prompt was written
  for (two `apply_change` prompts drew `edit_file` and `write_file`); the choice of tool is the model's and is not what this
  checks.
- **`auto` drew almost no call.** With 17 tools, none on any arm; with five, one on each CPU arm. Every other reply is the
  call written inside a fenced `json` block, which is prose unless the server runs with `-lenient-tool-calls`.

## The `auto5` call that breaks its schema

On both CPU arms one `create_event` reply came back with `finish_reason` `tool_calls` and these arguments
(`run3/result-cpu-new-run3.json`):

```json
{"title": "Quarterly planning", "start": "2026-12-10T09:00", "duration_minutes": null, "location": "Dee Park",
 "attendees": [{"name": "Dee Park", "email": null, "role": "required"}]}
```

`email` is declared a string. With a named tool or `required` the candidate cannot emit that (the table above), so this call
was not decoded under the grammar. The code says why:

- Under `auto` the grammar is lazy (`chat/toolgrammar.go:Template.ToolGrammar`): it masks nothing until the template's call
  opener, `<tool_call>` on this family, and holds the call to the tools' schemas from there.
- A reply that is a call object with no wrapper is still returned as a call (`chat/tools.go:Template.ParseToolCallsFor`,
  through `internal/chat/chatcore/jsoncall.go:BareToolCall`): its name must be an offered tool and its arguments a JSON object.
  Nothing checks the arguments against the tool's schema.
- Qwen2.5-Coder is the model that rule exists for: `docs/measurements/tool-call-failure-t0-2026-09-23.md` found no wrapped
  call in 1,200 samples under `auto`.

The reply's text is not in the HTTP body, so the missing wrapper is read from the code and that record, not seen here.

**So:** on a `<tool_call>` family, under `auto`, a model that leaves the wrapper off gets its call parsed and nothing more.
The new keywords, and every older guarantee about a call's arguments, hold for a named tool, for `required`, and under `auto`
for a call that opens with its wrapper. v0.25.0 behaves the same, so nothing regressed; `docs/server.md` now says so where it
describes the unwrapped call. Holding an unwrapped call to its schema is product work for after this release: the lazy grammar
could arm when a reply begins as a call, as it already does for the family whose call has no opener, or the parser could check
the arguments as the fenced-call path does.

## Files

- `toolcall_smoke.py`, `run-smoke.sh`: the smoke and the three-arm runner (run 3's form).
- `run1/`, `run2/`, `run3/`: each arm's `result-*.json` (every row), its summary (`smoke-*.log`), the server's log, and the
  runner's log with each arm's start, end and load.
- `turn8_probe.py`, `turn8-cpu-new.log`, `turn8-cpu.serve.log`: the eighth replay turn with a 3,000-token budget.
