---
name: model-bring-up
description: "Lessons for bringing up a model family or a real-checkpoint golden in goinfer, and for chasing a divergence from the HF reference: gzip goldens, minimal repros that hide the bug, per-layer differencing, hand-built per-family struct literals, a bug in the reference itself, and tests that vouch for behavior the system cannot produce. Read it when adding a family, pinning a golden, or debugging a parity red."
---

# Model bring-up: the Tests rules and the stories behind them

`CLAUDE.md` § "Tests" keeps each of these rules as a sentence or two. This is the full section, moved from `CLAUDE.md`
unchanged, with the bring-up incidents that taught each one.

## The rules, with their evidence

**A SKIP IS NOT A PASS.** `go test` prints `ok` for a package whose tests all skipped, so a
green line in 0.02s usually means "no assets, nothing ran". Confirm with `-v` and read for
`--- PASS`, not `ok`.

**A NEW FIXED-RESOLUTION REAL-CHECKPOINT GOLDEN GETS WRITTEN GZIP-COMPRESSED.** A family whose
vision input has no small-test-image option (SigLIP: 896×896 for every image, unlike Qwen2.5-VL's
dynamic resolution) produces a golden dominated by one giant `pixel_values` float array — measured
2026-09-08, `gemma3_real_golden.json` was 52.72 MB uncompressed, over GitHub's 50 MB
recommendation; gzip took it to 2.58 MB (~20×, JSON float-array text compresses hard). Convention:
write it `.json.gz` (Python: `gzip.open(OUT, "wt")` instead of `open(OUT, "w")`), read it via
`decoder.ReadGoldenJSONForTest` (gunzips transparently on a `.gz` suffix, reads a bare `.json`
unchanged) rather than a raw `os.ReadFile`+`json.Unmarshal` (in-package tests that want the bytes use
the `readGolden` test helper, same rule). Every golden that was over 1 MB was migrated 2026-09-25
(24 files, 56.5 → 20.3 MB); small `.json` goldens stay as they are.

**THE SAME GOES FOR LARGE MEASUREMENT RAW DATA (ncu CSVs, sweep dumps): commit it `.gz`.** The eleven
`docs/measurements/` CSVs over 1 MB were compressed 2026-09-25 (64.2 → 0.8 MB; ncu output is extremely
repetitive). The originals are still in history, so that shrank a checkout, not a clone — which is why
anything over ~1 MB of text goes in as `.gz` from the start, and the record names the `.gz` path.

**A UNIT TEST THAT SUPPLIES ITS OWN CALLING CONVENTION PROVES THE UNIT WORKS WHEN CALLED THAT WAY —
NOT THAT ANYTHING CALLS IT THAT WAY.** It is the microbenchmark trap one level up: same failure, in
composition rather than in cost. Measured here (G27): `optFwdGate` documents a two-way hysteresis
band, and `TestOptFwdGate_hysteresis` confirms it by driving `Observe` in an unconditional loop. But
production calls `Observe` only from inside the branch that `Should()` guards, so once the gate turns
off nothing is observed again, the estimate freezes, and the re-enable half of the band is
unreachable. The component test passes, is correct, and vouches for behaviour the system cannot
produce. **When a component's contract depends on HOW OFTEN or UNDER WHAT CONDITION it is called,
test it through its caller, or the test is asserting your assumption back at you.**

**A DOC COMMENT CLAIMING COVERAGE IS NOT COVERAGE, AND IT IS WORSE THAN SILENCE.** This is
NOT the rule above — that one is about a calling convention the test supplies for itself.
This one is about a comment asserting something the assertions never touch, which turns the
test into a trap for whoever audits the claim later: they read the promise, match it to a
test name that fits, and stop. Measured 2026-08-28: `a3_divergence_test.go`'s
`TestA3FastAttentionDivergence` opens by saying it pins *"MoE excluded: the flag cannot turn
f32 attention on for a MoE arch at all"*; the body loads the DENSE bench checkpoint and
asserts nothing about MoE anywhere. The `--cpu-fast-attention` MoE exclusion it advertised as
pinned had in fact never been measured on a MoE — no MoE appears in its kernel-ratio record
either — while the excluded term was 97.1% of an 8k MoE prefill (a 4-layer-slice figure;
the full model is lower, and the win it was blocking measured 1.52x, not the slice's 3.11x —
which is its own lesson about quoting a slice as a model). **The check: for any doc
comment that asserts coverage, the body must contain an assertion naming the thing.** Both
this and the rule above defeat the same defence — reading the test NAME instead of the test
BODY — which is why neither is caught by running the suite.

**`go test -v ./a/ ./b/` prints nothing until `./a/` finishes.** Output is buffered per package.
Run one package per invocation and `tee` if you need to watch progress.

**A MINIMAL REPRO CAN BE MINIMAL IN EXACTLY THE DIMENSION THAT HIDES THE BUG.** Shrinking an
input to isolate a defect is right, but each thing you remove can also remove a defect's only
symptom — and the repro then reports GREEN on broken code, which is worse than not having run
it. Measured 2026-08-31 bringing up LFM2: a single-token prompt was used deliberately to
eliminate conv-window history, and it worked — it isolated a zeroed `NormEps`. But **softmax
over one element is 1.0 at any scale**, so that same repro certified an attention path whose
`AttnScale` was 0 and which returned a uniform average of the whole context. It took ≥2 tokens
to see at all. Before trusting a shrunk case, ask which effects are IDENTICALLY ZERO at that
size — n=1 kills every interaction term, any softmax/normalisation over the reduced axis
becomes the identity, and any per-position table (RoPE at pos 0) becomes trivial.

**PREFER DIFFERENCING PER LAYER OVER REASONING FROM FINAL LOGITS.** HF's
`output_hidden_states=True` gives a per-layer reference; comparing against it names the exact
first divergent layer in ONE run. In the same bring-up, guessing from the final logits had
already burned several hypotheses — the conv split order and tap indexing were both suspected
and both were correct, measuring cosine 1.00000000 / max|diff| 0.000000 the first time they
were tested directly. Two cautions: HF appends the last hidden state AFTER the final norm, so
the last row compares normed against un-normed unless you account for it (that artifact read as
a catastrophic 0.50 cosine and was NOT a bug); and a matching argmax means nothing — both LFM2
bugs held argmax while the logit cosine was 0.897.

**A STRUCT LITERAL BUILT BY HAND PER FAMILY WILL EVENTUALLY OMIT A FIELD, AND ZERO USUALLY
LOOKS LEGAL.** Both LFM2 bugs were that one shape (`NormEps` read from the wrong JSON key,
`AttnScale` simply absent). The fix that generalises is a check at the single chokepoint every
family passes through — `resolveArchitecture` → `validateResolved()` — not a per-adapter fix,
because the point is to cover the families nobody has written yet. Note the family validator may
fire first for any given family, so test the chokepoint guard DIRECTLY or its branch is never
exercised.

**A TIGHT REAL-CHECKPOINT BAR CAN CATCH A BUG IN THE REFERENCE, NOT JUST IN GOINFER — DO
NOT ASSUME THE DIVERGENCE IS YOURS.** internlm2's real-checkpoint promotion
(`internlm/internlm2_5-1_8b-chat`) first failed at cosine 0.87 against HF. The bug was in
`transformers`, not goinfer: `InternLM2RotaryEmbedding.__init__` computes `inv_freq` as a
`persistent=False` buffer, and the installed version's `from_pretrained` fast-init path
never re-runs that formula for buffers absent from the checkpoint's state dict — so
`inv_freq` came back as uninitialized memory (denormals, sometimes literal NaN) instead of
the real frequency table. Found by bisecting the REAL loaded `nn.Module`'s forward line by
line (same discipline as "prefer differencing per layer," above) until the NaN's first
appearance pinpointed `rotary_emb`; confirmed by patching the buffer post-load and matching
goinfer's own output to 7 significant figures. goinfer was correct the entire time this
gate reported red. **The lesson generalises beyond this one bug**: any
`trust_remote_code=True` model with a `persistent=False` buffer computed in `__init__`
(RoPE tables are the common case, not the only one) is exposed to this class of
`from_pretrained` fast-init defect — sanity-check such buffers (non-NaN, roughly the
expected magnitude) before trusting a reference disagreement as goinfer's bug. This is also
why the T3 real-checkpoint tier exists at all: no amount of code review or a tiny golden
would have found this, because the defect isn't in goinfer's code — the tight bar has to
actually run against real, independently-produced numbers before it can catch something it
doesn't control.

Heavy tests need `GOINFER_HEAVY_TESTS=1` and their assets; `testdata/assets.json` is the
registry and `go run ./cmd/gate` is the runner (`census`, `heavy`, `parity`, `composition`,
`selector`, `gpu`, `mutation`).
