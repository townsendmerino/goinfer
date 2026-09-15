# scripts/ — what's here and whether to run it

141 files, most of them one-shot fixture/golden generators tied to a specific model family or a
specific debugging session. This page exists so "is this still useful, can I delete it, should I
run it" doesn't require opening each one. Grouped by what they're *for*, not alphabetically —
alphabetical is what `ls` already gives you.

Two tags used throughout:
- **[repeatable]** — rerun this whenever its trigger condition recurs (a new family, a drifted
  fixture, a routine sweep). Safe and expected to be run again.
- **[one-off]** — built for one specific debugging/decision session. Left in the tree as a record
  of how a question was answered, and as a template for the next similar question, not as
  something to rerun as-is. Several hardcode a session-specific path or a checkpoint that isn't
  committed.

## Doc & repo hygiene

The tools this session added, plus the two they lean on:

| script | what it checks | repeatable? |
|---|---|---|
| [`queue_citation_lint.py`](queue_citation_lint.py) | every `path:line` citation into `docs/QUEUE.md`'s generated index still resolves, cross-repo (aikit) citations against the **pinned** go.mod version. Runs as the **pre-push hook** — see root `CLAUDE.md` §"Citations and the pre-push hook" for its traps. | yes, every push |
| [`doc_id_collisions.py`](doc_id_collisions.py) | a queue-item ID (`P24`, `G23`, `B8`, ...) defined twice in `docs/queue-*.md`/`docs/QUEUE.md`. Found real collisions 2026-09-13 (G23, B8, P24). Read-only, not a push gate. | run before filing a new item, or periodically |
| [`doc_review_staleness.py`](doc_review_staleness.py) | whether a `docs/tasks/`\*/`docs/prompts/*` doc's `<!-- doc-reviewed: YYYY-MM-DD -->` footer predates a change to the doc itself, a goinfer file it cites, or (via `git log -G aikit` on the four `go.mod` files) the aikit pin. Read-only. | run before a `/doc-review` sweep, to see what's actually worth reviewing |
| [`readme_counts_check.py`](readme_counts_check.py) | whether `docs/README.md`'s stated file counts (design records, `measurements/`, `completed/`, `prompts/`, ...) match what's actually on disk. Found real drift 2026-09-13 (`measurements/` claimed 151, actually 75). Read-only. | run whenever `docs/README.md` is touched, or periodically |
| [`book_link_lint.py`](book_link_lint.py) | every GitHub-blob and relative-image link in `docs/book/` resolves to a real tracked path. | **CI-wired** (`.github/workflows/ci.yml:110`) |
| [`readme_smoke.sh`](readme_smoke.sh) | every root-`README.md` command marked `<!-- smoke -->` actually runs, from a throwaway module outside the repo (`GOWORK=off`) — simulates a cold first-time reader. Written after a real cold-user run found an undocumented-but-required build step. | run before a release / when the README's Quick Start changes |
| [`ci_checks.py`](ci_checks.py) | parses `.github/workflows/ci.yml` itself and derives the repo-hygiene command set (gofmt, build, vet, staticcheck, module-boundary guard) with correct per-job env, so a local pre-push gate can't silently drift from what CI actually runs. | consult when writing/updating local gate tooling |
| [`apidiff_check.sh`](apidiff_check.sh) | the v1.0 public-API-compatibility gate — diffs decoder/tokenizer/chat/constrain against a baseline tag, fails only on a **hard-tier** (`testdata/apidiff/hard_tier.txt`) break. | CI / pre-release |
| [`audit_gate_bars.py`](audit_gate_bars.py) | scans `*_test.go` for loose numeric tolerance bars (cosine/rel-err thresholds) that lack an explanatory comment. Purely informational, never fails. Motivated by a real incident (G25, an undocumented tolerance that hid a bug). | periodic inventory sweep |
| [`install-git-hooks.sh`](install-git-hooks.sh) | installs the local opt-in pre-push hook (citation lint + `TestParityManifest_fresh`), calling the same entry points CI uses. | once per fresh clone |

## Parity fixtures & goldens (non-`pin_*` generators)

Each of these regenerates one committed golden/fixture consumed by a specific Go test. Rerun only
when that fixture needs to change (a new family, a changed reference implementation, a bit-packing
format update) — the output is what's committed, not the script's own execution.

- [`gen_chat_goldens.py`](gen_chat_goldens.py), [`gen_tool_goldens.py`](gen_tool_goldens.py) — byte-exact chat-template / tool-calling render goldens per family, from HF's real `apply_chat_template`, for the `chat` package's renderer tests.
- [`gptoss_hf_oracle.py`](gptoss_hf_oracle.py) — real `openai/gpt-oss-20b` HF reference golden for the `-tags realckpt` logit-parity gate.
- [`gptoss_tiny_golden.py`](gptoss_tiny_golden.py) — synthetic tiny gpt-oss GGUF + golden for the fast forward-parity gate (experts kept F32 on purpose — aikit doesn't dequant MXFP4 GGUF yet).
- [`extract_mxfp4_golden.py`](extract_mxfp4_golden.py) — bit-exact MXFP4 unpacker golden, extracted from a real locally-installed gpt-oss:20b GGUF via the Ollama blob store.
- [`chatml_tiny_fixture.py`](chatml_tiny_fixture.py) — tiny tokenizer-only ChatML/Qwen GGUF fixture for the M25 EncodeSegments gate (replaced a hardcoded personal-machine path).
- [`shard_checkpoint.py`](shard_checkpoint.py) — splits a single-file safetensors checkpoint into an HF-shaped multi-shard layout, to exercise the sharded loader without a real multi-GB sharded download. Referenced directly from `decoder/sharded_test.go`.
- [`make_gemma4_moe_unified.py`](make_gemma4_moe_unified.py) — re-keys the tiny gemma4-MoE checkpoint under the real 26B-A4B "unified" `model.language_model.*` layout, weights byte-identical, to test the loader without the 51GB download.
- [`gguf_same_weights.py`](gguf_same_weights.py) — proves two GGUF files carry bit-identical per-tensor weights even when whole-file MD5 differs (Ollama's import reorders tensors). `python3 scripts/gguf_same_weights.py A.gguf B.gguf`.
- [`kda_oracle.py`](kda_oracle.py) — NumPy reference for the Kimi Delta Attention recurrence, validated against `fla`'s own implementation; the durable deliverable for a future KDA parity gate, same pattern as `pin_qwen35_deltanet.py`.
- [`asset_registry.py`](asset_registry.py) — the one canonical "is this test asset present" implementation (`testdata/assets.json`); subcommands `preflight`/`check --env`/`list`/`verdicts`/`census`. Cross-checked against a parallel Go implementation in `decoder/asset_registry_test.go`.
- [`gate_ledger.py`](gate_ledger.py) — maintains the gate system's "confirmed baseline" ledger (B14), keyed by a hash of the gate function's body so a renamed/edited gate reverts to unconfirmed. Wired into `cmd/gate`.
- [`remap_gate_citations.py`](remap_gate_citations.py) (+ [`test_remap_gate_citations.py`](test_remap_gate_citations.py)) — rewrites doc citation line numbers after a source edit, using `git diff -U0` hunks (more precise than the citation lint's own suggestions). Run once per base commit — a second pass over the same base drifts further. Its test pins a real historical bug (V-26, pure-insertion hunks); passes 4/4 as of 2026-09-13.
- [`refresh_parity_hashes.sh`](refresh_parity_hashes.sh) — refreshes the provably-non-numeric `deps_hash` in `testdata/parity_manifest.json` after a comment-only/non-semantic decoder edit re-stales `TestParityManifest_fresh`, without the multi-hour full T3 sweep. Commit with a `Deps-Hash-Refresh: <sha> goldens=<N> arch=<arch>` trailer.

## Benchmark harnesses

- [`bench_peer.py`](bench_peer.py) — the real goinfer-vs-peer (Ollama) harness: interleaved, server-restarted, decode-only timing from first streamed token, full provenance stamped. This is what backs `docs/benchmarks.md`.
- [`bench_peer_prefill.py`](bench_peer_prefill.py) — companion for prefill/TTFT: reports both `ttft_tok_s` (with request overhead) and `marginal_tok_s` (overhead-free, least-squares fit) since they can disagree by multiples.
- [`bench_peer_transcript.py`](bench_peer_transcript.py) — W4 harness replaying a scripted multi-turn agent tool-calling conversation as growing prefixes, measuring per-turn TTFT and prompt-cache reuse — models an agent loop (Claude Code, opencode), not single-shot throughput.
- [`bench_splitkv.py`](bench_splitkv.py) — split-KV decode attention ON vs OFF, paired per (geometry, depth) cell. Imports `bench_peer.py` for primitives but drives no external peer.
- [`bench_prompts_calibrate.py`](bench_prompts_calibrate.py) — (re)builds `scripts/prompts.json` (the token-depth prompt fixtures the harnesses above consume) by measuring real `usage.prompt_tokens` against a running server, since word count doesn't map reliably to tokens across tokenizers.
- [`bench_compare.sh`](bench_compare.sh) — goinfer-only in-process Go benchmarks. **Not** for peer comparisons — its own header says so; that mixing produced the retired "0.5B 1.78×" claim (see root `CLAUDE.md`).

## Site / book build

- [`build_search_index.py`](build_search_index.py) — builds the client-side heading-level search index for the published book site (GitHub Pages' Jekyll safe mode can't run a search plugin). **CI-wired** (`.github/workflows/book-pages.yml:112`).

`book_link_lint.py` is listed above under hygiene since it's a lint, not a build step.

## The `pin_*.py` family (90 files)

Every `pin_*.py` script builds or loads **one specific model/config** in HuggingFace
`transformers`, runs a fixed short prompt through it (float32/bf16, CPU), and dumps a golden JSON
(last-token logits, argmax, usually a short greedy continuation) that a corresponding Go test
compares goinfer's own forward pass against. They are the project's per-architecture logit-parity
fixtures — one per family, sometimes several per family (dense/MoE/vision/tiny/real variants).

Two shapes, distinguishable by name:
- **`_tiny` variants** (e.g. `pin_deepseek_tiny.py`, `pin_glm_tiny.py`) construct a small
  random-weight checkpoint from scratch with a hand-picked minimal config that still exercises
  every structurally interesting code path (MLA, MoE routing, sliding-window attention, ...).
  Fast, offline, safe for CI — both the synthetic checkpoint and its golden are committed to
  `testdata/`.
- **`_real` variants** (or bare names, e.g. `pin_cohere2_r7b.py`, `pin_gemma3_4b_text.py`) require
  a genuine multi-GB HF checkpoint on disk, path given via an env var (e.g. `GEMMA3_4B`). Slow,
  best run when the machine is idle, gated behind `-tags realckpt`. Only the resulting golden JSON
  is committed — the real weights stay gitignored.

Both shapes are **[repeatable]** golden generators, not one-offs: they're the backbone of the
model-family parity gate system, rerun whenever a family's reference implementation or fixture
needs regenerating. Individual files aren't enumerated here — the naming convention
(`pin_<family>[_variant]_{tiny,real}.py`) is the index; `grep -l <family>` finds the right one.

## One-off investigation scripts

Each of these was built to answer one specific question during one specific debugging or
architecture-decision session. They're kept as a record of how the question was answered — and as
a template for the next similar one — not as routine tools. Several hardcode a session-specific
path or depend on a checkpoint/trace file that was never committed (noted below where that makes
the script currently non-runnable as-is, not just historical).

- [`diff_gemma4_12b.py`](diff_gemma4_12b.py), [`dump_gemma4_12b_trace.py`](dump_gemma4_12b_trace.py) — paired scripts that localized a Gemma-4 12B forward-pass bug to a specific layer via per-layer cosine diffing against an HF trace. **Currently non-runnable**: both hardcode `~/mycode/goinfer/testdata` (not this checkout's `~/tmcode/goinfer`) and their input trace JSONs were never committed. The underlying bug is resolved (`docs/completed/task-gemma4-moe.md`).
- [`g33_replay.py`](g33_replay.py), [`g34_blockverify_replay.py`](g34_blockverify_replay.py) — replay a captured MoE expert-routing trace (`docs/measurements/g33-routing-trace.json`, still present) through a simulated LRU cache to answer two queue items (G33, G34; both have RESULT entries in `docs/QUEUE.md`). Runnable as committed.
- [`gemma4_int8_restore_probe.py`](gemma4_int8_restore_probe.py), [`gemma4_quant_recon.py`](gemma4_quant_recon.py), [`gemma4_scale_probe.py`](gemma4_scale_probe.py) — a three-part NumPy investigation into int4/int8 quantization reconstruction quality on real gemma-4-26b-a4b-it weights (symmetric vs affine, zero-point vs scale). Hardcoded checkpoint paths.
- [`probe_tool_templates.py`](probe_tool_templates.py) — rendered a fixed tool-calling conversation through each family's HF chat template to design goinfer's tool-call rendering. No golden output, just prints; rerunnable if a new family needs the same design pass.
- [`ref_dflash_accept.py`](ref_dflash_accept.py), [`ref_dspark_accept.py`](ref_dspark_accept.py) — ran the upstream (z-lab DFlash / DeepSpec DSpark) speculative-decoding drafter's own verbatim acceptance-rate loop against goinfer's benchmark suite, to attribute a low acceptance rate to environment vs a genuine gap (P10 kill-gate decisions). `ref_dspark_accept.py`'s default `DEEPSPEC_DIR` points at a session-scoped `/tmp/claude-.../scratchpad/DeepSpec` path that no longer exists — override via env var and re-clone DeepSpec to rerun.
- [`convert_dflash_f32.py`](convert_dflash_f32.py) — converted a z-lab DFlash drafter checkpoint bf16→f32 safetensors so the Go loader can read it. Speculative decoding is still live in the tree (`cuda/drafter.go`), so rerun if a new DFlash checkpoint needs the same conversion.

## Web UI

| script | what it checks | repeatable? |
|---|---|---|
| [`webui_md_gate.mjs`](webui_md_gate.mjs) (+ [`webui-gate/md-harness.html`](webui-gate/md-harness.html)) | the web UI's Markdown renderer (`internal/serveapp/webui/ui/markdown.js`, W1) in headless Chrome: hostile Markdown rendered into live DOM with a canary that must never fire, pathological inputs under a render-time budget (ReDoS), structural correctness. Runs from `go test` as `TestWebUI_markdownGateInBrowser`. | yes — after any change to `ui/markdown.js` |
| [`webui_app_gate.mjs`](webui_app_gate.mjs) | the shipped web UI page end to end, driven through its own `send()` with a fake SSE stream: streamed Markdown (W1); Copy on both clipboard paths, a stopped answer, and a re-render (W2); and persistence across real page reloads — restore, a mid-stream reload, unreadable storage, quota failure, New chat, tab sync, hostile stored content (W3); the system prompt as actually sent, its persistence and tab sync (W4); loading a pulled model from the page — the path posted, heartbeat, selection, failures, already-loaded (W5); thinking folded above the answer, including replayed real serve output from `webui-gate/captured-thinking.json` (W6). Each W-item adds its checks here. Runs from `go test` as `TestWebUI_appGateInBrowser`. | yes — after any change to the web UI |
| [`webui-gate/cdp.mjs`](webui-gate/cdp.mjs) | shared headless-Chrome driver for both gates. Loads pages from `file://`; exit 0 pass / 1 fail (including a page that navigates mid-gate, or a gate program that throws because the page is in the wrong state) / 2 ONLY when Chrome could not be launched or reached — `go test` skips on 2, so nothing page-side may produce it. | library, not run directly |

## Other

- `prompts.json`, `w4_transcript_base.json`, `w4_transcript_edited_turn6.json` — **data, not scripts.** `prompts.json` is `bench_prompts_calibrate.py`'s output, consumed by the `bench_peer*` harnesses. The two `w4_transcript_*.json` files are the scripted conversation `bench_peer_transcript.py` replays. Left in `scripts/` rather than `testdata/` because they're benchmark inputs, not test fixtures.
