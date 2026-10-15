# Id round trip, step 0: do generated ids survive decode, echo and re-render? (2026-10-15)

## Method (written and committed before the run)

**The question** (owner, step 0 of proposal 2 in `i3-night-2026-10-13.md` § "After run: the compactions and the cold
continuations, diagnosed", part 3): replay archived Codex prompts greedily, capture the generated ids, and count how often the
reply, decoded and echoed back the way Codex echoes it, re-renders to the same ids. Break it down by JSON escapes, `./...`,
non-ASCII and code, and by part (reasoning, answer, tool-call JSON).

**What serve exposes today.** Nothing that carries generated ids on `/v1/responses`. `-log-requests` writes one line per request
(token counts and times, no ids, no reuse figure). `/v1/chat/completions` with `logprobs` returns each token's text, not its id,
and renders a different prompt. `usage.input_tokens_details.cached_tokens` gives the reused length only. So the replay runs
serve's own handler in process and reads the slot's id record after each request.

**Instrument.** `id-roundtrip-step0/idstep0_test.go.txt`, a throwaway test copied into `internal/serveapp/` for the run and not
committed there. It builds the server as `Main` does from serve's flags (`newServer`, `routes`) and POSTs each request to
`/v1/responses` through the real mux, streamed, as Codex sends it. After the request it reads the decoder's committed id record
(`resIDs`, the prompt followed by the generated ids, which is what prefix reuse compares) and checks that it starts with the
prompt as the route renders it. It then builds the next request as Codex would: the same input, then this reply's output items
less `status` (what Codex echoes; checked on the 2026-10-14 live capture, where the echoed reasoning and function_call items equal
the emitted ones key for key apart from `status`), then the archived items Codex appended before its next request (tool output,
inserted messages), with their call ids moved onto this reply's calls. It renders that through the route's own renderer
(`kv0Render`, KV0's) and compares token ids with the slot.

**Prompts.** `id-roundtrip-step0/prep.py` rebuilds each request from `codex-rollouts.tar.gz`: the first developer message and
the tool list from the live 2026-10-14 capture (same Codex 0.162.1, same base instructions in all 15 sessions), then the
rollout's items. Its check: rebuilding the live session's third request gives that request's captured input exactly. Requests
before each session's first compaction only (129; a compaction request's reply is not echoed as an assistant turn). **Render
check (CPU, done before the run):** all 129 render to the night's `input_tokens` less a constant 18, so every token after the
fixed preamble renders as on the night; the 18 are in the tool list or first developer message, which the night did not record.

**Sample and cost.** Greedy (`temperature` 0), `max_output_tokens` 4,096 as a guard (Codex sends none; the night's largest reply
was 3,584). Every pre-compaction request of five sessions, in order (`select.txt`): `write` r2, `fix` r3, `write` r3, `edit` r2,
`search` r2, 52 requests, chosen for the categories their night replies carried (dots 8 of 9, non-ASCII 8 of 11, code 23 of 44,
escapes 13 of 30, of the 129). `scripts/power.py binomial --p 0.15 --half-width 0.10` gives N = 49 for a 95% half-width of 10
points; intervals are reported as Wilson intervals because n·p < 10. Load as the night (`--ctx 16384 --moe-cache-experts`,
KV f32) **with `--windowed-kv=false`**: the greedy reply differs from the archived one the next prompt carries, so under the ring
nearly every request would roll back past 448 and re-prefill 6K to 12K tokens cold; without it each request prefills its
growth. The ring changes which positions are kept, not the ids a request generates. **Estimate:** one cold 6.1K-token prefill
(about 60 s), then per request about 560 tokens of prefill (about 6 s) and about 430 generated (about 8 s): about 13 minutes for
52. **Stopping rule:** no request starts after 600 s of replay wall; the order is fixed above, so the cut does not depend on any
outcome. Under the timing lock (label `step0`).

**Per reply, recorded:** match (the next prompt's ids start with the slot's whole id record), first divergent position and the
tokens there, whether each run of generated ids between special tokens is the tokenizer's own (canonical) encoding of its text,
and whether the re-rendered text equals the decoded text. **Causes:** non-canonical ids (texts equal, ids differ: fixable by
reusing ids), a rendering difference (the re-rendered text differs from the decoded text: template or serve's parse and emit),
and client edits (Codex's echo differs from what serve emitted; read from the live capture and the archive, not the replay).
