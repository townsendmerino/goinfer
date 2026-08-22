# Upstream issue draft — missing LICENSE on the `dspark_*_block7` drafter checkpoints

> **STATUS: STILL DRAFTED, STILL NOT FILED — and that is the current decision, not an oversight.**
> Re-checked 2026-08-21. P15 step (0) re-ran the licence audit and reached the same place: filing
> matters for eventual distribution/ship, not for local research use, and `license=None` was
> accepted for the latter. Leave drafted until something actually ships that depends on it.


**Status: DRAFTED, not filed.** Written 2026-08-15 (Claude app, P10 investigation). Filing is
for eventual **distribution/ship**, not to unblock work: Francis accepted `license=None` for
exploration the same day, so the block7 drafters are already in scope for the DSpark spike.
See [`docs/spec/08-dspark-dflash.md`](../spec/08-dspark-dflash.md) §"Performance levers and
the DSpark pivot".

**Target:** the DeepSeek / DeepSpec org that publishes the DSpark drafter checkpoints on
Hugging Face (`dspark_qwen3_8b_block7`, `dspark_qwen3_14b_block7`, and the two
originally-named block7 drafters). File as a discussion/issue on one of those model repos, or
on the DeepSpec GitHub repo if it accepts issues.

---

**Title:** Missing LICENSE on the `dspark_*_block7` drafter checkpoints — MIT intended?

Hi — thanks for releasing the DSpark drafters; they're a clean fit for lossless block
speculative decoding and we'd like to use them.

The standalone drafter checkpoints — e.g. `dspark_qwen3_8b_block7`,
`dspark_qwen3_14b_block7` — currently ship only `.gitattributes`, `config.json`, and
`model.safetensors`, with no LICENSE file and no model card. By contrast, the full
`DeepSeek-V4-{Flash,Pro}-DSpark` repos do carry an MIT LICENSE, and the DeepSpec project
itself is MIT and lists these drafter checkpoints among its released artifacts.

That combination reads like an oversight rather than an intentional restriction, but we can't
assume it — the absence of a license is legally distinct from a permissive one.

Could you confirm the intended license for the standalone `dspark_*_block7` drafter repos,
and (if MIT, as the parent project and the full-model repos suggest) add a LICENSE file to
them?

One note that may help others auditing this: the Hugging Face list/search API reports
`license=None` for **every** repo in the org, including the ones that genuinely do carry an
MIT LICENSE — only per-repo detail queries surface the real license. So an automated
enumeration can't tell the licensed and unlicensed repos apart, which is probably how the
drafters' missing license went unnoticed.

Thanks!

---

## Notes for Francis (not part of the issue)

- Filed for eventual distribution/ship, **not** to unblock exploration.
- If a clean-license path is wanted without waiting on a reply, the **apache-2.0 PARO family**
  in the same org is the fallback lead (unexamined method, but properly licensed).
- Tone kept collaborative and specific — names the exact repos, the exact missing file, and
  hands them the HF-list-endpoint root cause — so it is a two-minute confirm-and-add for a
  maintainer rather than a demand.
- The HF-list-endpoint detail is also the root cause of **this repo's own** mis-audit: P10
  increment 1 recorded the 8b/14b as MIT on an enumeration pass. Worth carrying into any
  future licence audit here, not just upstream.
