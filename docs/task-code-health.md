# Task: goinfer code health — act on the repowise findings, ignore the noise

> Scoping doc. Opened 2026-08-26 from the repowise trial (`aikit/docs/prompts/repowise-trial-results.md`),
> the first complete index+health run over this repo — the earlier sandboxed pass never identified
> the worst file. **Status: SCOPED, nothing started.** The findings are recorded and triaged here;
> only §3 item 1 is unambiguously worth doing on this evidence. Everything else is either
> constrained by bit-identity (§3.2), mechanical-but-cosmetic (§3.3), or noise (§4). Do not treat
> the 20-item tool output as a work list — §4 explains why most of it is not one.

Tooling: repowise 0.45.0, `apple-m1pro`, 2026-08-26, `--no-prose` (structural only, no LLM).
Reproduce with §5.

## 1 · Where goinfer actually stands

| | goinfer | fin | aikit | ken |
|---|--:|--:|--:|--:|
| Health average | **7.92** *(Warning)* | 9.21 | 8.04 | 7.43 |
| Hotspot score | 4.68 | 6.04 | 4.79 | 4.57 |
| Worst file | **1.4** | 2.15 | 1.95 | 2.05 |
| Alert-tier files | 36 (10.4%) | 10 (1.2%) | 31 (10.5%) | 17 (13.2%) |
| Healthy by volume | 61.9% | 87.1% | 64.5% | 53.5% |
| Unreachable files | 84 | 88 | 31 | 12 |
| Unused exports | 7 | 164 | 33 | 1 |
| Perf-risk / 10K LOC | 14.93 | 5.66 | 18.26 | 44.61 |

Index: 1,227 files · 5,000 symbols · graph 7,501 nodes / 33,114 edges · 930 files with git
history · 150 hotspots · 63 architectural decisions extracted.

**Read this as "mid-pack, not alarming."** goinfer sits between aikit and ken and well below fin —
but fin is a calculation library with a fundamentally simpler shape, and is not the right
comparison. Against aikit, the closest sibling in kind, goinfer is within noise on every column
except unreachable files (84 vs 31), which §4.3 explains.

**The score is not arbitrary.** repowise self-reports that 18 of goinfer's 20 lowest-health files
had a bug fix in the last ~6 months — **4.25× the 21% repo baseline**. It is the tool grading its
own homework, but the effect is strong and consistent across all four repos indexed (2.9×–5.01×),
so the ranking is worth taking seriously even where individual findings are not.

## 2 · The two worst files, and why they are the way they are

| file | score | CCN | nest | NLOC | paired test |
|---|--:|--:|--:|--:|:--:|
| `decoder/gguf.go` | **1.4** | **306** | 5 | 2,174 | yes |
| `decoder/serialize.go` | **1.4** | 47 | 6 | 981 | yes |
| `cuda/resident.go` | 1.7 | 43 | 5 | 1,590 | yes |
| `gpu/moe_w4a8.go` | 1.8 | 12 | 2 | 151 | **no** |
| `decoder/weights.go` | 1.9 | 105 | 4 | 1,569 | yes |

`gguf.go` at CCN **306** is the single largest complexity concentration in the repo. That is a
format parser, and format parsers earn high CCN honestly — a branch per tensor kind, per quant
type, per metadata variant. High CCN here is not automatically a defect.

**But `gguf.go` and `serialize.go` are also the two files the W4A8 work already flagged as
load-bearing for the `.giw` on-disk format** (`docs/task-w4a8-neon-bandwidth.md` § Gate 1
correction: `serialize.go` kind=3 writes packed nibbles that are zero-copy mmap-aliased back with
no version tag). The health score reached the same two files from pure structure, with no
knowledge of that. Two independent signals landing on the same code is the finding worth keeping
from this whole exercise.

That convergence cuts **both ways**: it makes them the most interesting files to improve and the
most dangerous ones to touch. Any change here is a bit-identity change until proven otherwise.

## 3 · The actionable items

### 3.1 · `gpu/moe_w4a8.go` is genuinely untested — the one clear gap

repowise flagged two "untested hotspots". **Only one survives verification**, and it is not the
one the tool ranked higher:

| file | NLOC | dependents | paired `_test.go` | symbols referenced from any test |
|---|--:|--:|:--:|--:|
| `decoder/routercapture.go` | 52 | 187 | none | **9 files** |
| `gpu/moe_w4a8.go` | 175 | 18 | none | **0 files** |

`routercapture.go` trips the finding on a **paired-filename heuristic** — there is no
`routercapture_test.go`, so the tool calls it untested, but nine test files exercise its symbols.
Its 187 dependents make it look like the scarier of the two; it is not. Downgrade it.

`gpu/moe_w4a8.go` has no paired test **and zero references from any `_test.go` in the repo** —
175 lines of MoE W4A8 path with 18 dependents and no test exercising it, on the same quant path
the W4A8 campaign is actively changing. **This is the item to do.** Not a refactor: a test.

### 3.2 · The duplication clusters in `gguf.go` — mechanical, but gated

Nine extract-helper clusters, concentrated in two files:

- `decoder/gguf.go` — **17 lines across 5 sites** (1114-1130, 1551-1557, 1765-1771, 1984-1990,
  2107-2123); 11 lines across 3 sites; 10 lines across 6 sites; plus 20-line and 15-line pairs.
- `decoder/model.go` — four 14-to-22-line pairs (518-532/657-664, 576-589/601-608,
  604-618/621-635, 686-707/758-770).

The `model.go` clusters are ordinary and safe. The `gguf.go` ones are in the `.giw`-adjacent
parser, so they inherit §2's constraint: **extract-helper is only safe here behind a byte-identity
gate on a real `.giw` fixture** — the 638 MB bundle the embedded chat demo carries is the artifact
that would silently misdecode. (Described rather than cited: it lives under a gitignored directory,
so a path here would resolve on one machine and nowhere else, and an uncommitted target has no
history against which anyone could later establish what it contained.) Cheap to do, cheap to prove,
but not a drive-by.

One extract-method (`generate` in `decoder/blockspec.go`, 7 lines, −2 CCN) and one move-method
(`server.serveChatText` → `loadedModel`, both in `internal/serveapp/openai.go`, uses 7 foreign vs
0 own members) are genuinely trivial and unconstrained.

### 3.3 · `decoder/kvsnapshot.go` — the strongest predictor in the set

Flagged on `prior_defect`: **3 bug fixes touched it in the last ~6 months**, and repowise's own
text calls recent defect history "the strongest cost-effective predictor of further defects." It
scores 2.1/10 over only 334 lines, so unlike §2's files it is small enough to actually read
end-to-end in one sitting. If anything here deserves a review pass on judgment rather than metrics,
it is this.

## 4 · What is NOT a work list, and why

**4.1 · 14 of the 20 "refactoring targets" are test files.** `eagle_accept_test.go`,
`gemma4_moe_scaled_parity_test.go`, `gemma_parity_test.go`, `e2e_int4_test.go`,
`eagle_alpha_test.go`, `int4direct_test.go`, `model_test.go`, `w4a8_fast_test.go`,
`qwen35moe_35b_cache_test.go`, `gen_test.go`, `specdecode_test.go`, `ssm_w8a16_test.go`,
`gemma4_admission_test.go`, `quant_noise_floor_test.go` — 70% of the list. A parity test with
CCN 25 and a censusing loop nested 5 deep is doing its job; complexity in a test that pins
numerical behaviour is not the same defect as complexity in the code under test. **Do not
refactor these to move a score.** Only six production files appear at all: `routercapture.go`,
`cmd/serve/main.go`, `demo/chat/main.go`, `forward_gemma4_moe.go`, `gpu/moe_w4a8.go`,
`kvsnapshot.go`.

**4.2 · `change_entropy` findings are an artifact of being actively developed.**
`cmd/serve/main.go` (top 1%), `demo/chat/main.go` (top 5%), `eagle_accept_test.go` (top 7%),
`forward_gemma4_moe.go` and `e2e_int4_test.go` (top 9%) are flagged for having changes scattered
across noisy commits. That describes this repo's last six months accurately and is not a code
defect. Useful as a *where to look* signal; useless as a *what to fix* list.

**4.3 · The 84 unreachable files are mostly a Go-shape artifact, and the count is confirmed real.**
84 unreachable / 7 unused exports reproduced under a standalone `repowise dead-code .` run — worth
stating because a known repowise bug (`offset-naive and offset-aware datetimes`) can crash the
detector and then report a *clean* "0 unreachable", silently. It did not reproduce here, and these
numbers match the pre-crash figures from the sandboxed pass. **But** the same detector calls
`cmd/`-style entrypoints and per-backend files "unreachable" purely on `in_degree=0`, which is
wrong for Go main packages and build-tagged backends. Before acting on any of the 84, filter that
class out; the residue is likely small.

**4.4 · Perf-risk findings were not triaged.** 189 findings at 14.93/10K covered LOC — between
aikit (18.26) and fin (5.66), so not an outlier. repowise scopes this to I/O-in-loop / N+1,
resource and defer-in-loop, blocking-in-async; it explicitly does *not* cover algorithmic blowups
or GC pressure, which is where this repo's actual performance work lives. Low expected value here;
`docs/task-w4a8-neon-bandwidth.md` and `docs/task-attention-decode-cost.md` are where the real
levers are.

**4.5 · "0 architectural decisions" is retired.** The sandboxed pass concluded repowise's ADR-miner
does not match this repo's `docs/task-*.md` + `CLAUDE.md` convention. Run locally with real git
history it extracted **63 decisions**. The earlier zero was an artifact of indexing *copies*
without history. No action, no finding.

## 5 · Reproducing

```
python3 -m venv /tmp/rwenv && /tmp/rwenv/bin/pip install repowise   # needs Python 3.11+
/tmp/rwenv/bin/repowise telemetry disable                          # ON by default
cd ~/tmcode/goinfer
repowise init --no-prose --no-editor-setup -y .                    # ~2:49
repowise health --format md --refactoring-targets .                # ~1:50, targets + plans
repowise health .                                                  # the aggregate — NOT in the md
repowise dead-code .                                               # confirm counts standalone (§4.3)
```

`--no-editor-setup` matters: without it, `init` writes MCP config and hooks machine-wide plus
`.mcp.json` / `.claude/CLAUDE.md` / `.vscode/mcp.json` into the repo. Adopting that is a separate
decision, deliberately not taken.

`.repowise/` is now gitignored here (`13bea28`). It had previously been committed by accident —
390 files / 87.6 MB, swept in by `git add -A` in `ef887cc` and `160c09b`. Untracked now, but the
blobs remain in pushed history (`.git` is 216 MB vs 21–54 MB for the sibling repos); purging needs
a `git filter-repo` rewrite plus a force-push, reviewed 2026-08-26 and **explicitly declined for
now** — recorded here so it is a known state rather than a lurking surprise.

## 6 · If this gets picked up

Suggested order, cheapest-first, and each independently droppable:

1. **Test `gpu/moe_w4a8.go`** (§3.1) — the only unambiguous gap; a test, not a refactor.
2. **Read `decoder/kvsnapshot.go`** (§3.3) — 334 lines, 3 recent bug fixes, highest-signal predictor.
3. **`model.go` duplication + the two trivial method plans** (§3.2) — unconstrained, mechanical.
4. **`gguf.go` duplication** (§3.2) — only behind a `.giw` byte-identity gate.

Items 1–3 are small enough not to need a gate. Item 4 does. Nothing here is urgent, and none of it
should preempt `task-w4a8-neon-bandwidth.md` or `task-attention-decode-cost.md`.
