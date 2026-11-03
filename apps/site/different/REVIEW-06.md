# Review notes: writeup 06 "Checked against the reference" (drafted 2026-09-29)

Draft, `reviewed:` empty. Builds with `-drafts` (site build: 37 families, no error). Body about 890 words counting table pipes. Nothing was run or measured; every figure is quoted from a record.

## Claims and sources

| Claim | Source (file, heading) |
|---|---|
| Compared with HF `transformers`, same weights; Python reference, goinfer under test; argmax, logit cosine, spot trace | `docs/what-parity-gated-means.md`, "The short version", "What gets compared" |
| Four labels and their wording; the two numbers; "Checked" filter labels; "Not recorded" | `site/internal/site/templates/models.html` ("What the checks mean", filters); `site/internal/site/model.go` (TierText, Gaps: "not all of them") |
| 37 families; 30 released (19 full-oracle + 11 real-oracle); 6 tiny-oracle; 1 shared-path (Kimi K2 via deepseek_v3); 0 unrecorded | DERIVED: counted from `docs/capability-matrix.json` `parity` strings (not in `figures`) |
| LFM2 real 5 GB checkpoint, argmax matched, cosine 0.897, 2026-08-31, MacBook | `docs/completed/queue-correctness.md`, "G1 . LFM2.5-2.6B" |
| LFM2 bugs: norm epsilon zero (first divergence layer 0), attention scale zero (5-token bisect, layer 2, invisible at one token) | same, table under G1; the softmax-of-one-element point also in CLAUDE.md "A MINIMAL REPRO..." |
| Per-layer differencing with `output_hidden_states` | same (G1); CLAUDE.md "PREFER DIFFERENCING PER LAYER" |
| Manifest rows hold commit, date, machine, reference, method, metrics; "parity stale" test hashes source files; shared-path goes stale with its source | `docs/parity-coverage-policy.md`, "The validation manifest is the source of truth", "Shared-path validation"; `decoder/parity_manifest_test.go` header |
| Bars from measured failures: two dispatch bugs passed the 3% near-tie rule at 0.9887 and 0.9977; a gate is not trusted until seen to fail | `docs/parity-coverage-policy.md`, "Falsifiable: prove the gate red" |
| internlm2: failed 2026-09-07 at 0.873148; cause was `transformers` uninitialised `inv_freq`; bisect; patched matches goinfer to 7 significant figures; then 1.000000 | `docs/parity-coverage-policy.md`, "Timing: batch attempt (2026-09-07 ...)" and "RESOLVED 2026-09-07" |
| Table rows (Llama, InternLM2, Gemma 4, Qwen3-Next, Qwen3.5-MoE): reference, machine label, date, metrics | `testdata/parity_manifest.json` (families); matrix rows in `docs/capability-matrix.md`; E2B 0.99924 is only in the manifest reference text |
| Qwen3.5-MoE 77.5% is "a deliberate bandwidth trade" bisected to a commit | `docs/measurements/parity-sweep-aikit-rearm-2026-09-06.md` (row note) |
| 29 (MacBook) vs 49 (nobara-pc) passed, same selector and commit, 2026-08-31; MacBook skips are missing assets | `docs/parity-coverage-policy.md`, "Scoped: a goldens green names the quantizations that actually RAN" |
| CPU reference bit-identical within an architecture only | same file, "The pure-Go CPU reference is bit-identical WITHIN an architecture" |
| "Two families promoted past tiny-oracle had real defects behind a passing fixture" | `docs/what-parity-gated-means.md`, "tiny-oracle is a weaker claim" (also `models.html` Gaps text) |
| Not a quality benchmark; misses wrong chat template, stop token, tokenizer edge case | `docs/what-parity-gated-means.md`, "What this does not tell you" |
| Release sweep 2h51m on nobara-pc, 2026-09-06 | `docs/measurements/parity-sweep-aikit-rearm-2026-09-06.md` header |
| Commands: `go run ./cmd/gate parity`, `REALCKPT=0`, `go run ./cmd/gate census`; `TestParityManifest_fresh` | `cmd/gate/main.go` usage; `decoder/parity_manifest_test.go` |

## Conflicts between records

- **Olmo 3.** `docs/parity-coverage-policy.md` (batch 3, 2026-09-07) says Olmo 3 was fixed and is "now `full-oracle 100.0%/1.00000` in the manifest". The manifest and capability matrix as they stand today list it as `experimental: tiny-oracle` (manifest date 2026-09-18, method tiny-golden). `docs/what-parity-gated-means.md` still tells the Olmo 3 story (argmax exact, cosine 0.992789) as a released-weights promotion. I resolved it by leaving Olmo 3 out of the page entirely and using the LFM2 case for "matching argmax alone means nothing". Someone should check whether the promotion row was lost from the manifest.
- **Family count.** `docs/what-parity-gated-means.md` says the README lists 36 families; the matrix has 37. I used 37 (newest, and it is what the Models page counts).
- **InternLM2 cosine.** CLAUDE.md says 0.87; the record prints 0.873148. I used the record's figure.
- **MacBook goldens count.** The same section prints "28 ran, 20 skipped" and, from the later comparison table, "29 passed, 20 skipped". I quoted only the second (the direct MacBook-vs-nobara comparison).
- **Label wording.** The brief says "none on file"; the Models page says "Not recorded". I used the page's.
- **Machine labels.** The manifest uses `linux-amd64` and `linux-62gb` for what the sweep record calls nobara-pc. I printed the manifest's labels in the table and did not equate them, except the one sentence naming nobara-pc for the 2h51m sweep.

## Left out as not verified

- Any claim that the bars are 0.9999 (f32) or 0.99 (int8) in general: the 0.9999 appears only inside the Olmo 3 record; the 0.99 floor is described only as historical for two families, and Qwen3-Next passes at 0.98931.
- Which prompt set or how many positions the "next token the same" percentage is taken over. The docs say "at every position" but no single record states the count per family.
- Whether Gemma 4 and Qwen3-Next's cosines below 1 are caused by quantization. The manifest names the quantization but no record ties the two, so the page states only what ran.
- Whether the Models page's "Not recorded" tier will ever be populated; today the count is 0.
- No pre-registered decision band exists for these gates; the page says a failing gate is reported red, not forced green (policy, "Outcome: gate FAILS ... NOT forced green").

## Questions for the owner

1. Olmo 3 (see conflicts): is the manifest wrong, or was the promotion reverted? If the promotion stands, the strongest "argmax alone" example on the repo (cosine 0.992789 with exact argmax and an exact 8-token continuation) can go in the page.
2. Is a table of five hand-picked rows fine, or should the page link the full matrix instead? I chose rows that show the range: exact, quantized, and a 77.5% row.
3. The Qwen3.5-MoE note relies on a sweep record dated 2026-09-06 that says "known deliberate bandwidth trade". Is that still your reading, or has the row been re-run?
