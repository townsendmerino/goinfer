# Classification rubric — goinfer audit findings, option-path admission step 1

You are classifying audit findings from goinfer (a pure-Go LLM inference engine with CPU, CUDA, Metal and WebGPU
backends). Each finding gets exactly ONE class. The rules below were fixed before classification began; apply them
literally. Do not try to steer the totals in any direction — nobody wants a particular answer.

## The classes, tested in this order: R, B, N, G, O. The FIRST that fits wins.

- **R (registration)** — the defect is in PRODUCT code (not a test), and an option value, a model family, or a kind
  of state that exists is not accounted for by a guard, predicate, dispatch, serializer, loader branch or lifecycle
  path that needed to know about it. The path treats the unregistered thing as if it were the default case. The fix
  is a registration: add it to a list, predicate or switch, or give it a named decline/refusal.
  - Panics, corruption or wrong output CAUSED by such a miss are still R (e.g. a path admits a family whose layer
    type it has no code for, and panics).
- **B (bound)** — a specific limit in PRODUCT code (an array or threadgroup-memory size, a kernel capacity, a context
  or position cap, a model's max positions, a byte budget) that a reachable input exceeds, with nothing checking it.
  The fix is a bound check or a decline. **Unbounded resource use where the code holds no limit at all is O, not B**
  (e.g. an unbounded request field that allocates proportional memory is O).
- **N (numerics)** — a path that DOES handle the input (it knows about the option/family/state) computes it wrong:
  arithmetic, indexing, rounding, layout, wrong buffer offset, wrong sizing arithmetic not tied to a fixed limit.
- **G (gate)** — the defect is in a test, gate, CI job, release check, or a claim in docs/comments/help text that
  does not check or say what it claims. A test whose hand-written family/option list is incomplete is G, and you set
  `r_shaped_gate: true` on it.
- **O (other)** — everything else: performance, API shape (exported types, naming, breaking-change risks), release
  and module mechanics, leaks, races and panics NOT caused by an R or B miss, request validation, security, resource
  exhaustion, missing features, documentation organisation.

## Synthetic examples (not from the audits)

| finding | class |
|---|---|
| A cache-reset predicate lists Mamba and DeltaNet state but not a newly added conv-window state; reused sessions read stale state | R (state × lifecycle path) |
| The batched path's admission checks model features but never the `--foo` load option, so `--foo` models run it wrong | R (option × batched path) |
| A kernel keeps scores in a fixed 2048-entry array and nothing compares the context against 2048 | B |
| A softmax subtracts the max of the wrong row | N |
| A parity test checks cosine ≥ 0.9 on a path documented as bit-identical | G |
| A test's hand-written list of recurrent families omits one family | G, r_shaped_gate=true |
| README says a flag value that the parser rejects | G |
| A kernel is slow / a dispatch could be fused / a copy is redundant | O, perf=true |
| A request field is unbounded and allocates proportional memory | O |
| A buffer leaks on an error path | O |
| An exported method returns an unexported type | O |

## How to work

1. For each ID you are given, find its FULL entry in the audit file (search for the ID; the full entry is the heading
   or bold line plus its body up to the next entry). Some audits also have short table rows or later disposition
   notes — read the full entry, and its DISPOSITION/Status lines if present, before classifying.
2. Classify by the defect the entry describes as filed. A later fix does not change the class.
3. **Withdrawn:** only if the audit itself records that this entry was NOT a defect (wrong, does not reproduce, a
   deliberate design that was misread). Then set `withdrawn` to a short quote. "Fixed" is not withdrawn. When in doubt,
   it is not withdrawn.
4. **Re-filings:** if the entry says it re-files, re-opens or repeats a finding from an EARLIER audit (e.g. "prior
   audit M-23, open", "R-10 reopened"), put that earlier ID in `refiles`.
5. Severity: copy the tier as the audit files it (e.g. "Critical", "Major", "Major (gate)", "Gate", "Correctness",
   "Release blocker", "Performance — Major").

## Output

Write ONE JSON object per line (JSONL) to the output path you are given, one line per ID, in the order given:

```json
{"id": "C-03", "severity": "Critical", "title": "<= 12 words>", "class": "R", "reason": "<one line, <= 30 words, why this class and not an earlier one>", "rb_kind": "option|family|state|limit|null", "option_or_family": "<the option, family or state involved; R/B only, else null>", "path_missed": "<the guard/path that missed it; R/B only, else null>", "r_shaped_gate": false, "perf": false, "refiles": null, "withdrawn": null, "confidence": "high|medium|low"}
```

- `perf` is true when the finding is fundamentally about speed or memory footprint rather than correctness.
- `rb_kind`: for R use option|family|state; for B use limit. null otherwise.
- `confidence`: low when two classes fit nearly equally and the order rule did not settle it cleanly; say which two
  in `reason`.

After writing the file, check it: the line count must equal the number of IDs you were given, every line must parse
as JSON, and every `class` must be one of R, B, N, G, O. Then reply with ONLY: the output path, the line count, and the
count per class. Do not paste the classifications into your reply.
