# Review notes: writeup 05, "Decisions without generating" (drafted 2026-09-29)

Draft, `reviewed:` empty. Body about 917 words counting code blocks and table markup (about 850 of prose). Site build with `-drafts` passes. Nothing was run; the request/response example is docs/server.md's, labelled illustrative.

## The big conflict with the brief

The brief asks for D6a ECE and the bias-adjusted bar as measured quality. **D6a has no graded result anywhere in the tree.** `docs/tasks/task-constrained-confidence.md` (status, 2026-09-28) says "D6a queued"; `docs/README.md` line ~93, `README.md` "Decisions", `docs/server.md` ("goinfer's own measurement ... (D6a) is pending") and `docs/integrations/typesafe-jevx.md` all say pending. `docs/measurements/decisions-d6a-2026-09-28/` holds only `run-d6a.sh`, `select.py`, `analyze.py` (no result file). No D6a log in `~/goinfer-logs/night`. The page therefore says "not yet measured", gives the pre-registered bar as "not yet run", and does not invent an ECE. If a nobara result exists off-repo, the table and the first `doesnt` item need rewriting.

## Claims and sources

| Claim | Source |
|---|---|
| `POST /v1/systemone`, TypeSafe wire shape; jevx and SDKs work by base-URL | `docs/server.md` "Decisions"; `internal/serveapp/systemone.go` header; `docs/measurements/decisions-d0-prior-art-2026-09-27.md` §3 |
| Request/response example; `usage.output_tokens` 0; `goinfer` block fields (route, template, calibrated map) | `docs/server.md` example; `serveSystemOne` in `systemone.go` |
| Kinds: noul P(true), choice 2..16 (A..P), score 2..10 levels with expected level | `decide.MaxChoiceOptions`, `MaxScoreLevels` in `internal/decide/decide.go`; `systemOneRequest` |
| confidence = `(n·p − 1)/(n − 1)` | `typesafeConfidence` in `systemone.go`; `docs/server.md` "confidence" |
| 422 before any prefill for a rule-breaking question | `handleSystemOne`: validation runs before `withModel`; empty `instructions`, unknown type, bad criteria, over-limit questions all 422 |
| `/v1/models` lists `decisions` (endpoint, route, kinds, template, calibrated) | `decisionsField` in `internal/serveapp/openai.go` |
| Mechanism: prompt with lettered options, chat-v1 asks for one letter, one prefill, restricted softmax at last position, per-kind temperature, labels must be single tokens | `internal/decide/decide.go` (`RenderChat`, `score`, `New`); D1 in the task doc |
| Adapter entry refused; encoder/non-generative entries lack `decisions` | `serveSystemOne` (400 "not supported on a compute-time adapter entry"); `decisionsField` (nil) |
| Flags `-decisions-template` (default chat-v1), `-decisions-calibration`; startup refusal on template mismatch | `internal/serveapp/main.go` lines ~398-399, ~505-516 |
| `goinfer-chat decide` / `decisions-calibrate`, flags `--model --template --calibration --permute -o`, input row fields, default template `bare-v1` | `internal/decidecmd/decidecmd.go`; dispatch in `internal/chatapp/main.go` ~142 |
| B0 numbers 0.5323 / 0.898 / 0.5180 / 0.9181 / 0.0720 / 0.0396 (authors' run, bare-v1, bf16) | D0 record §1 "Published numbers" table |
| Pre-registered bar 0.8881 / 0.8181, bias-adjusted ECE bar sqrt(0.05² + floor²), 872-row eval sample, arm A chat-v1, nobara CUDA, Qwen3.5-9B Q4_K_M | task doc "D6a pre-registration", "amendment", "amendment 2" |
| Smoke 3/6 vs 1/6, "decides nothing" | task doc, D6a pre-registration "Machine" bullet |
| C0 enum AUROC 0.847, bar 0.65, 1.5B on Metal | `confidence-c0-2026-09-27.md` (table, Setup); the 0.65 bar is in the task doc, as writeup 02 already cites |
| 9B fell back to CPU on the Mac, so D6a is on CUDA | task doc, D6a pre-registration |
| Metal `--embed-int4=false` note | `docs/quantization.md` "Known issue"; flag in `internal/loadflags/loadflags.go` |
| Many questions re-prefill; no prefix reuse on Gated-DeltaNet families; D7 is a projection only | `docs/server.md`; `decisions-d7-2026-09-28.md` header and §4.5 |
| `--permute` cost and position bias | `decidecmd.go`; task doc D1 ("permute 3 flattened a 3-way choice", 0.5B, anecdotal) |

## Conflicts and how I resolved them

- **Adapter refusal status.** Brief and `docs/server.md` group "Refused" loosely; the code returns 400 for an adapter entry, 422 only for a malformed question. I said 400 and 422 respectively.
- **Example arithmetic.** `docs/server.md` prints `"confidence": 0.77` for probabilities 0.85/0.12/0.03; the documented formula gives 0.775. My example prints 0.775. A one-character doc nit in `server.md`, not fixed here.
- **CLI vs server default template.** The CLI defaults to `bare-v1`, the server to `chat-v1`. Stated on the page because a calibration fitted under one is refused under the other.
- **Bias-adjusted bar form.** The task doc says the owner chose "0.05 + the floor", then records that a simulation showed the linear form over-corrects, and adopts quadrature. I wrote quadrature, the recorded bar, and said "adopted".
- **Brief said `goinfer-chat decide` exists**: confirmed. "label scoring, unmeasured" is not in the repo verbatim; it comes from `task-site-2026-09.md` line ~112 (the planned Models-page wording). Used as the meaning, not quoted.

## Left out as unverified

- Any D6a top-1 or ECE for goinfer (none exists); any D6a result on the 1.5B (arm C); the D7 ratios (projection, and the D7 record says never to quote them).
- That jevx or the SDKs work against a live goinfer: `typesafe-jevx.md` says "not yet run end to end". The page says clients "can be pointed at" goinfer, from the wire shape, and does not claim a verified run. `TestSystemOne_jevxShape` exists but I did not run it.
- Latency or prefill throughput for decisions on any machine.
- Whether the server test `TestSystemOne_realModel` ran on any checkpoint (not opened).

## Questions for the owner

1. Is publishing this before D6a lands what you want? The page is honest but its headline quality claim is "unmeasured". Should it wait, or should the card wording say more plainly that no accuracy figure exists yet?
2. Quoting the JEV authors' bf16 B0 numbers (third-party, bare template) on a goinfer page: fine as context, or drop them until goinfer has its own?
3. The C0 row: keep it as adjacent evidence about the same readout, or remove it to avoid readers taking it as a decisions result?
