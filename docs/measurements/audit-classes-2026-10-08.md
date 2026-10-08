# Audit findings by class — option-path admission, step 1 (2026-10-08)

Task: [`task-option-path-admission-2026-10.md`](../tasks/task-option-path-admission-2026-10.md) §2–§3. Counting rules
fixed in that doc's §2.1 and committed (`69684e7a`) before any finding was classified. Raw data, the rubric the
reviewers worked from, both passes and the review are in [`audit-classes-2026-10-08/`](audit-classes-2026-10-08/).

## Verdict: build step 2, with a thin margin

**71 of 261 counted Critical and Major findings are R or B: 27.2%.** §3 says build at 25% or more with at least three
distinct options or families among them. Both conditions hold: the 71 name more than thirty distinct options, families,
kinds of state and limits.

The margin is six findings (65.25 would be exactly 25%). Every registered sensitivity also clears 25%, the narrowest, (c), by 1.7 points; the floor in (c′) by 0.6:

| | entries | R | B | N | G | O | R+B | share |
|---|---|---|---|---|---|---|---|---|
| **registered count** (four withdrawn entries out) | 261 | 52 | 19 | 30 | 62 | 98 | **71** | **27.2%** |
| withdrawn entries kept in | 265 | 52 | 19 | 30 | 63 | 101 | **71** | **26.8%** |
| (a) without the 37 performance findings | 224 | 48 | 19 | 29 | 60 | 68 | **67** | **29.9%** |
| (b) the 7 R-shaped gate findings counted as R | 261 | 59 | 19 | 30 | 55 | 98 | **78** | **29.9%** |
| (c) the three "prior audit … open" re-filings counted once | 258 | 51 | 18 | 30 | 61 | 98 | **69** | **26.7%** |
| (c′) bound: every entry that points at an earlier one dropped, new instances included | 242 | 47 | 15 | 30 | 58 | 92 | **62** | **25.6%** |

(c′) is not a registered sensitivity. It drops 19 later entries that cite an earlier finding, including ones that are a
new instance of the class rather than the same defect (Bailing Hybrid's KDA state, 09-10 C-03, cites LFM2's conv window,
09-02 C-02: one family later, same miss). It is here as the floor.

Before the review in §5 the two passes put the share at 31.0%. The review moved nine R calls to N or O and none the other
way; the 27.2% is the stricter reading and the one this doc stands on.

## 1. By audit

| | entries | R | B | N | G | O | R+B | share |
|---|---|---|---|---|---|---|---|---|
| 2026-08-05 | 74 | 6 | 8 | 10 | 9 | 41 | **14** | **18.9%** |
| 2026-09-02 | 57 | 14 | 5 | 10 | 12 | 16 | **19** | **33.3%** |
| 2026-09-10 | 80 | 28 | 3 | 9 | 28 | 12 | **31** | **38.8%** |
| Metal 2026-09-12 | 27 | 2 | 2 | 1 | 9 | 13 | **4** | **14.8%** |
| Metal 2026-09-30 | 23 | 2 | 1 | 0 | 4 | 16 | **3** | **13.0%** |

The three whole-repo audits run 18.9% → 33.3% → 38.8%; the two Metal audits were performance-led by their own brief, and
their registers are mostly performance items (O). Three audits are three data points, not a trend anyone should fit a line
to, but they do not suggest the class is fading as the audits catch it.

## 2. What the 71 are

| kind | R or B findings |
|---|---|
| load or request **option** × a path | 22 |
| model **family** × a path | 20 |
| **limit** a reachable input exceeds (B) | 19 |
| kind of **state** × a lifecycle path | 10 |

**Options that recur** (the set §3's ambiguous band would have restricted step 2 to; listed because it sizes the
matrix either way):

| option | findings |
|---|---|
| LoRA adapter on a resident path | 09-10 C-01, C-02, C-07, M-01 (R); 09-10 M-05 and its Metal re-filing 09-12 C-03 (B, adapter shape vs base) |
| quantisation mode (`int4`, `int4mix`, `int8`, `-embed-int4`) | 09-10 C-06, M-09, M-25, M-31 |
| prefill lever selection (`--cpu-fast-attention`, the fused/MMA levers, `--exact-prefill`) | 09-02 M-07; 09-10 M-10, M-11, M-48 |
| paged MoE slots (`--moe-cache-slots`, the Metal slot setting) | 09-02 M-02; 09-10 C-08; Metal 09-12 C-02 |
| KV precision (`-kv i8` / `f16`) | 09-02 M-32; Metal 09-30 A-C01 |

**Paths that miss most** (a finding can sit under more than one):

| path | findings |
|---|---|
| batched / fast prefill admission | 09-02 C-01, M-07; 09-10 C-01, C-08, M-10, M-11, M-23, M-48; Metal 09-12 M-06; Metal 09-30 A-C01, F-C02 |
| speculative decoding (EAGLE, block-spec, n-gram, verify) | 08-05 C-02, M-07; 09-02 C-11, M-07, M-13, M-14, M-15; 09-10 M-03, M-09, M-11 |
| state lifecycle (truncate, reset, reuse, snapshot, rollback) | 08-05 C-01, C-05, C-17; 09-02 C-02; 09-10 C-02, C-03, C-07, M-07 |
| fit guard and memory accounting | 09-02 M-01, M-02; 09-10 M-24, M-28, M-29, M-30; Metal 09-30 C-C01 |
| kernel limits at dispatch (threadgroup memory, fixed arrays, head dim, alignment) | 08-05 C-12, C-13, M-10, M-11, M-12; 09-02 M-16; Metal 09-12 C-06; Metal 09-30 F-C02 |

**One limit was missed twice.** The exact prefill attention kernel's `threadgroup float sc[4096]` is 08-05 C-12 (the
exported Metal entry points took `pos` unchecked) and, eight weeks later, Metal 09-30 F-C02 (contexts above 4096 when the
fused kernel cannot run). The first fix guarded the entry points; nothing tied the array to every route that reaches it.

## 3. What this means for step 2 — for the owner

The registered rule says build, and nothing here argues against that. Two things the data says that §4 did not know when
it was written:

1. **Options are under a third of the class.** §4 scopes step 2 as load options × paths. That is 22 of the 71. Families
   (20) already have a registry for backends (`decoder/features.go`), and the family misses here are mostly on paths it
   does not cover: loaders and serializers, speculative decoding, the fit guard, chat-template detection. State kinds
   (10) have a partial registry (the recurrent bit) and still produced four recurrent-state misses across three audits (08-05 C-01 and C-17, 09-02 C-02,
   09-10 C-03).
   Limits (19) are a different mechanism: a bound checked where the kernel is dispatched, not a registration.
2. **The rows of the matrix are where the misses land.** Batched/fast prefill admission and the speculative paths hold
   19 of the 71 between them (09-02 M-07 and 09-10 M-11 sit in both). A matrix whose rows start with those two, and whose columns take the recurring options
   above plus state kinds, covers the bulk of what recurred.

Choices, none made here: build §4 as written (options × paths); widen the columns to state kinds; or build §4 as written
and file the limit findings as their own small task (each kernel declares its bounds, each dispatch site checks them).

## 4. Withdrawn entries (out of the count and the total)

| audit | ID | the audit's own words |
|---|---|---|
| Metal 2026-09-12 | M-09 | "CLOSED 2026-09-13, NEGATIVE — materially SLOWER, not faster." |
| Metal 2026-09-12 | M-10 | "the isolated 68 GB/s vs 96 GB/s gap this finding started from does not survive contact with the actual decode token" |
| Metal 2026-09-12 | M-16 | "CLOSED 2026-09-13, NEGATIVE — not a lever." |
| Metal 2026-09-12 | G-10 | "It carries no independent **Fix:** or **Confidence:** line because it isn't a defect" |

All four are performance or gate entries; none was R or B, so withdrawing them raises the share slightly (26.8% with
them kept).

## 5. Method, and how far to trust it

- **Pass A and pass B**, seven reviewers each, one per audit slice, working from one written rubric
  ([`rubric.md`](audit-classes-2026-10-08/rubric.md)) and blind to each other. They agreed on **253 of 265 (95.5%)**.
  The twelve disagreements were settled by re-reading the entries (the `settled` rows in `final.json`).
- **A sample check** of twelve R/B calls both passes agreed on (seeded, 20261008) found two that fail a strict reading
  of the R test: the fix honoured a return value or changed a comparison, it registered nothing (09-02 C-05, 09-10
  M-06). Two in twelve, spread over 78 agreed calls, could have taken the share near the line, so:
- **A full review** by six fresh reviewers of all 261 classifications, in both directions, under the strict test
  ([`review_brief.md`](audit-classes-2026-10-08/review_brief.md)). It changed nine R calls to N or O and moved nothing
  into R or B. All nine were accepted on reading; 09-10 M-06, which the review left as R, was moved to N on the same
  test. The final classification agrees with pass A on 246 of 261 and with pass B on 247.
- **The asymmetry is worth naming.** The review found only false R's, no false N/G/O. Either the passes leaned toward R
  and nothing leaned the other way, or the strict brief made a reviewer slower to call R than the rubric requires. The
  share under the passes' own reading (31.0%) and under the review's (27.2%) both clear 25%, so the verdict does not
  depend on which.
- **Limits.** Reviewers read the audit entries, not the code; a finding is classified by what its entry says the defect
  and the fix were. The R/B boundary is a judgement where a fix both registers something and changes arithmetic
  (09-10 M-22, the drafter's K/V, is the clearest such case and ended as N). The counted set leans on how each audit
  tiered its findings (§2.1 of the task doc).

## 6. Every counted entry

Class after settlement and review; `A`/`B` are the two passes; `option, family, state or limit` and `path` are filled
for R and B only. Withdrawn entries are marked and not counted.

| audit | ID | severity | class | A | B | why | option, family, state or limit | path that missed it |
|---|---|---|---|---|---|---|---|---|
| 2026-08-05 | B-01 | Critical (release blocker) | **O** | O | O | Module/release mechanics: go.mod require on a pseudo placeholder; no registration, bound, numerics or gate defect. |  |  |
| 2026-08-05 | B-02 | Critical (release blocker) | **O** | O | O | Module mechanics: missing require in go.mod breaks standalone build; not a product-code registration or limit. |  |  |
| 2026-08-05 | B-03 | Critical (release blocker) | **O** | O | O | Module mechanics: incomplete go.sum breaks clean-clone build; release/module class. |  |  |
| 2026-08-05 | B-04 | Critical (release blocker) | **O** | O | O | Module mechanics: stale go.sum in an unbuilt committed module; CI absence is secondary to the module defect. |  |  |
| 2026-08-05 | B-05 | Critical (release blocker) | **O** | O | O | Release mechanics: clean clone links stale published backends; module graph, not product logic. |  |  |
| 2026-08-05 | B-06 | Critical (release blocker) | **O** | O | O | Release ordering/tagging mechanics across modules. |  |  |
| 2026-08-05 | B-07 | Critical (release blocker) | **O** | O | O | Dependency/module skew; release mechanics class. |  |  |
| 2026-08-05 | B-08 | Critical (release blocker) | **O** | O | O | API shape: exported test hooks and global mutators; breaking-change risk, not R/B/N/G. |  |  |
| 2026-08-05 | B-09 | Critical (release blocker) | **O** | O | O | API shape/security/race on an exported global map; nothing unregistered, no limit, no numerics. |  |  |
| 2026-08-05 | B-10 | Critical (release blocker) | **O** | O | O | Panic from recover placement on exported API; deliberate internal panics, not caused by an R or B miss. |  |  |
| 2026-08-05 | B-11 | Critical (release blocker) | **O** | O | O | API shape: exported mutable fields and error-less method signatures. |  |  |
| 2026-08-05 | B-12 | Critical (release blocker) | **O** | O | O | API shape/naming consistency before semver tag. |  |  |
| 2026-08-05 | B-13 | Critical (release blocker) | **O** | O | O | API shape: exported signatures exposing unexported types. |  |  |
| 2026-08-05 | B-14 | Critical (release blocker) | **O** | O | O | Security/default exposure; not a registration, bound, numerics or gate defect. |  |  |
| 2026-08-05 | C-01 | Critical | **R** | R | R | TruncateTo rewinds KV, rings, mlaLatent but omits delta/mamba state; resident reset unwired. State missed by lifecycle path. | Mamba-2 / Gated DeltaNet recurrent state (c.mamba, c.delta, resident {win,ssm}) | KVCache.TruncateTo(0) / session reset and resident Forward at pos 0 |
| 2026-08-05 | C-02 | Critical | **R** | R | R | Recurrent families reach the one speculative entry point lacking the rollback-safety decline; fix gives them a named refusal. | recurrent targets (granite-4.0-h, nemotron-h, qwen3_5_moe) | GenerateSpeculative entry-point specRollbackSafe check |
| 2026-08-05 | C-03 | Critical | **O** | O | O | Concurrency race on shared resident KV; not a registration miss, limit, or numerics error. |  |  |
| 2026-08-05 | C-04 | Critical | **N** | R | R | Predicate knows windowed families but is keyed on the wrong condition; filed fix passes the taken path and honours discarded exact returns, not a registration. |  |  |
| 2026-08-05 | C-05 | Critical | **R** | R | R | Per-layer stride state is not handled by the session loader, so restored caches mis-slice. R (state x loader) vs N (sizing). | KVCache per-layer stride[] state | LoadSession snapshot restore path |
| 2026-08-05 | C-06 | Critical | **B** | B | B | Config-sized scratch buffers overrun by blob-declared dims with no check; fix is a bound check. B vs O (input validation). | config-sized scratch (NumExperts logits, q/k/v, logits) vs .giw tensor dims | LoadSerializedWeights shape validation (validateShapes) |
| 2026-08-05 | C-07 | Critical | **N** | N | R | Capture is skipped by control flow in a branch that handles MoE; the fix moves code, it registers nothing. |  |  |
| 2026-08-05 | C-08 | Critical | **O** | O | O | Dropped errors at GPU boundary; no unregistered value, limit, or arithmetic fault. |  |  |
| 2026-08-05 | C-09 | Critical | **O** | O | O | Dropped GPU errors leading to stale output; error handling, not R/B/N. |  |  |
| 2026-08-05 | C-10 | Critical | **N** | N | N | Row index derived from reduced tail threadgroup size: indexing error. N vs B (no fixed capacity exceeded). |  |  |
| 2026-08-05 | C-11 | Critical | **N** | N | N | Floor vs ceil sizing arithmetic not tied to a fixed limit; explicit N case. |  |  |
| 2026-08-05 | C-12 | Critical | **B** | B | B | metalCtxCap and threadgroup sc[4096] exceeded via exported methods with no check; fix is a bound check. | metalCtxCap / threadgroup sc[4096] score array | exported Resident.Forward/ForwardEmb/ForwardArgmax/PrefillLast |
| 2026-08-05 | C-13 | Critical | **B** | B | B | Fixed 16-element kernel accumulator exceeded by M>16 with no check; fix rejects M > gemmRowMaxM. | array<i32,16> accumulator (gemmRowMaxM) | DecodeTokenFusedBatched dispatch |
| 2026-08-05 | C-14 | Critical | **N** | N | N | Reduction handles input but tie-break computes wrong index; numerics. |  |  |
| 2026-08-05 | C-15 | Critical | **N** | N | N | Wrong rounding/underflow/overflow arithmetic in conversion; numerics. |  |  |
| 2026-08-05 | C-16 | Critical | **G** | G | G | Lint test's hand-written file list omitted a production kernel; kernel itself later found clean. G vs N for the kernel half. |  |  |
| 2026-08-05 | C-17 | Critical | **R** | R | R | Non-positional Mamba state not accounted for by resident truncate/verify path; fix declines ForwardN with mamba. | Mamba-2 state in resident runner (mambaWin/mambaSSM) | resident TruncateTo no-op / ForwardN batched verify |
| 2026-08-05 | C-18 | Critical | **O** | O | O | Unbounded request field allocating proportional memory: rubric's explicit O case. O vs B (model context). |  |  |
| 2026-08-05 | C-19 | Critical | **O** | O | O | Request validation gap causing panic; not R/B/N. |  |  |
| 2026-08-05 | C-20 | Critical | **B** | B | B | Arch.MaxPositions loaded but never compared to prompt length; reachable input exceeds model max positions. | Arch.MaxPositions (model context window) | serve prompt admission in openai.go before worker |
| 2026-08-05 | C-21 | Critical | **O** | O | O | Unbounded resource use with no limit held; resource exhaustion. |  |  |
| 2026-08-05 | C-22 | Critical | **O** | O | O | Shutdown/lifecycle hang in server mechanics; not R/B/N/G. |  |  |
| 2026-08-05 | C-23 | Critical | **B** | B | B | Loop over hidden_size overruns shorter norm tensor; fix is a load-time length check. B vs O (input validation). | mm_soft_emb_norm.weight length vs vision hidden_size | LoadProjector load-time validation |
| 2026-08-05 | C-24 | Critical | **O** | O | O | Panic boundary missing in product code; comment claims secondary. O vs G. |  |  |
| 2026-08-05 | C-25 | Critical | **N** | N | R | Indexes layer 0 in a path that knows the MoE layer list; an indexing error. |  |  |
| 2026-08-05 | C-26 | Critical | **O** | O | O | Leak plus non-idempotent close causing use-after-free; O vs R (incomplete hand-kept release list). |  |  |
| 2026-08-05 | C-27 | Critical | **O** | O | O | Panic and dropped errors on allocation failure; not caused by R/B miss. |  |  |
| 2026-08-05 | C-28 | Critical | **O** | O | O | Error-path cleanup omission leaving resource state poisoned; leak/error-handling class. |  |  |
| 2026-08-05 | C-29 | Critical | **O** | O | O | Use-after-free and data race; concurrency class. |  |  |
| 2026-08-05 | C-30 | Critical | **O** | O | O | Data race on per-model pager state; concurrency class. |  |  |
| 2026-08-05 | C-31 | Critical | **O** | O | O | Unbounded allocation where code holds no limit; O vs B (file size bound). |  |  |
| 2026-08-05 | G-01 | Gate | **G** | G | G | Defect is a test whose guarded check is unreachable; it stays green if the feature table is emptied. Not product code, so R/B/N do not apply. |  |  |
| 2026-08-05 | G-02 | Gate | **G** | G | G | Filed as a golden hashing a computation the shipped path never runs; the product embed-scale miss is cross-referenced to B-11, not this entry (R considered). |  |  |
| 2026-08-05 | G-03 | Gate | **G** | G | G | Freshness test and -update output depend on process environment; a gate defect in test code, fixed with t.Setenv. |  |  |
| 2026-08-05 | G-04 | Gate | **G** | G | G | Filed as a disabled gate: slots arm is dispatched and pins correctly, only the record the test reads is missing; not treated-as-default, so not R. |  |  |
| 2026-08-05 | G-05 | Gate | **G** | G | G | Gate skips on every other machine because its fixture path is hardcoded; a test/gate defect with no product-code miss. |  |  |
| 2026-08-05 | G-06 | Gate | **G** | G | G | Test fixture paths hardcoded to two machines make 29% of tests skip; purely a test/gate defect. |  |  |
| 2026-08-05 | M-01 | Major | **O** | O | O | Unbounded concurrency and slow-body pinning on pre-queue work with no limit held; rubric puts no-limit resource exhaustion in O, not B. |  |  |
| 2026-08-05 | M-02 | Major | **R** | R | R | Explicit top_p=0 falls into the sampler's TopP>0 'disabled' default; fix gives it named greedy handling plus refusal. R vs O (request validation). | explicit top_p: 0 (and negative values) | prepare top_p mapping / sampler TopP > 0 predicate |
| 2026-08-05 | M-03 | Major | **O** | O | O | The absent case is the default itself and its chosen behaviour breaks the OpenAI contract; no unregistered value or arithmetic error, so API semantics O. |  |  |
| 2026-08-05 | M-04 | Major | **N** | N | N | finish_reason logic handles max_tokens but compares against requested, not effective, budget; fix plumbs the value rather than registering anything (R considered). |  |  |
| 2026-08-05 | M-05 | Major | **O** | R | R | Error-handling: grammar build error and ok=false are returned but discarded; filed fix honours that return with a 400, which the strict R test excludes. |  |  |
| 2026-08-05 | M-06 | Major | **O** | O | O | API compatibility and request-parsing shape; no unregistered option on a dispatch path and no numeric error. |  |  |
| 2026-08-05 | M-07 | Major | **R** | R | R | Families the capture seam rejects (DeepSeek/MLA etc.) are admitted by EAGLE entry points and panic in fuseAt; fix applies the allow-list/canBatchN guard. | families without capture seam (mla/DeepSeek, gemma4, qwen35, granite, nemotron, llama4, gptoss) | GenerateEagleSpeculative / GenerateEagleSpeculativeTree arch guard |
| 2026-08-05 | M-08 | Major | **O** | O | O | Resource leak on error paths; rubric places leaks in O. |  |  |
| 2026-08-05 | M-09 | Major | **N** | N | N | Prefill claims sliding-window support but binds the wrong (model-level) buffers per layer; a wrong-binding computation fixed by rebinding, not registration (R considered). |  |  |
| 2026-08-05 | M-10 | Major | **B** | B | B | Kernels hard-assume K multiple of 32 with nothing checking; fix is a decline. A shape constraint, not an option/family/state, so B not R. | K % 32 == 0 int4 group alignment | int4 pack entry points (int4Buf/int4Concat) / BuildResident |
| 2026-08-05 | M-11 | Major | **B** | B | B | Threadgroup memory sized from model dims can exceed maxThreadgroupMemoryLength with no check; fix is a decline in BuildResident. | maxThreadgroupMemoryLength (32 KB threadgroup budget) | metal BuildResident (moe.go/model.go/gemma4_moe.go dispatch sizing) |
| 2026-08-05 | M-12 | Major | **B** | B | B | Kernel capacity of 128 (workgroup_size, red array) exceeded by admitted archs with no check; fix adds a decline. | head_dim <= 128 (workgroup_size(128), red: array<f32,128>) | BuildResident / newDecodeRunner / decodeRunnerEligible |
| 2026-08-05 | M-13 | Major | **N** | N | N | Decoder knows the dummy-prefix strip but applies it at the wrong granularity, producing wrong text; nothing unregistered (O for exported-API break considered). |  |  |
| 2026-08-05 | M-14 | Major | **O** | R | O | Sibling-file path derived from a literal name; a loader path bug, not an option, family or state. |  |  |
| 2026-08-05 | M-15 | Major | **O** | O | O | Unbounded decoded dimensions allocate proportional memory; MaxPixels caps output only, so no input limit is held — rubric's unbounded-allocation O case. |  |  |
| 2026-08-05 | M-16 | Major | **O** | O | O | Buffer leaks on error paths; rubric places leaks in O. |  |  |
| 2026-08-05 | M-17 | Major | **G** | G | G | Wrong constant only feeds the published capability matrix whose comment claims runtime parity; runtime admits correctly. Not B (no unchecked overflow); N considered. |  |  |
| 2026-08-05 | M-18 | Major | **G** | G | G | CI jobs' hand-written build lists omit wired serve for the gpu and metal tags that the cuda job covers; a gate gap, R-shaped. |  |  |
| 2026-08-05 | M-19 | Major | **O** | O | O | Module-graph mechanics; the CI guard gap is secondary to the module-structure defect. |  |  |
| 2026-08-05 | M-20 | Major | **O** | O | O | License-attribution completeness is release mechanics; an omission rather than a false claim, so O over G. |  |  |
| 2026-08-05 | M-21 | Major | **O** | O | O | Missing cancellation in an exported signature; API shape / breaking-change risk. |  |  |
| 2026-08-05 | M-22 | Major | **O** | O | O | Exported type naming collision across modules; API shape. |  |  |
| 2026-08-05 | M-23 | Major | **O** | O | O | Exported API returning mutable internal pointers; API shape, not an unregistered value or arithmetic error. |  |  |
| 2026-09-02 | C-01 | Critical | **R** | R | R | canBatchN's hand list excludes every own-forward family except lfm2; path treats LFM2 as default dense stack and panics. Registration miss. | lfm2 | canBatchN (batched prefill admission) |
| 2026-09-02 | C-02 | Critical | **R** | R | R | Conv-window recurrent state absent from TruncateTo/reconcile/specRollbackSafe/Snapshot/capture guards; treated as positional default. State x lifecycle registration miss. | lfm2 shortConv conv window (c.conv) | TruncateTo/reconcile/specRollbackSafe/Snapshot/ForwardCapture/HiddenLast/layer pager |
| 2026-09-02 | C-03 | Critical | **R** | R | R | The .giw writer has no branch for the shortConv field; family serialized as if it had none. Serializer registration miss (regression of R3). | lfm2 (shortConv weights) | .giw writer v6 tail / validateShapes |
| 2026-09-02 | C-04 | Critical | **N** | N | N | Path handles ring layers but sizes slots from global keys and tiles from per-layer keys; computed, not fixed, buffer. B also plausible (clamp fix). |  |  |
| 2026-09-02 | C-05 | Critical | **N** | R | R | Fix is consuming TruncateTo's ignored exact bool in reconcile (honouring a return value); ring caches are handled, result discarded. Not a registration. |  |  |
| 2026-09-02 | C-06 | Critical | **O** | O | O | Data race between two writers; not caused by any registration or bound miss. |  |  |
| 2026-09-02 | C-07 | Critical | **B** | B | B | Model's max positions is the limit; a 1 MiB input exceeds it and nothing checks tokens. Incomplete closure of prior C-21. | Config.MaxPositions (token count) | loadDecoderEmbedder maxTokens=0 / HiddenLast |
| 2026-09-02 | C-08 | Critical | **O** | O | O | Unbounded request field allocating proportional memory; code holds no limit at all, so O not B. |  |  |
| 2026-09-02 | C-09 | Critical | **N** | N | N | Paged path knows gpt-oss biases but indexes bias table with idxZeros; wrong buffer index, not a registration miss. |  |  |
| 2026-09-02 | C-10 | Critical | **R** | R | R | byteLevelKnobs switch and walker silently default unrecognized pre/regex (gpt-oss, GPT-2, Llama-4) to cl100k shape; fix registers walkers and loud decline. | gpt-oss o200k, GPT-2, Llama-4/Kimi/DeepSeek/GLM4 pre-tokenizers | byteLevelKnobs / Split-regex dispatch |
| 2026-09-02 | C-11 | Critical | **R** | R | R | Block-spec stop set never reads SamplingParams.StopIDs or merged eosIDs; option treated as absent default. N plausible (stop set computed wrong). | SamplingParams.StopIDs / generation_config EOS | BlockSpec.generate stop set |
| 2026-09-02 | C-12 | Critical | **O** | O | O | Wasted M x vocab head work and grow-only buffer leaking its predecessor: performance and leak, not a registration or bound miss. |  |  |
| 2026-09-02 | G-01 | Gate | **G** | G | G | Gate prints PASS while every test skips (env unset, skip counted as run). |  |  |
| 2026-09-02 | G-02 | Gate | **G** | G | G | Release ritual and CI never build the tagged Metal parity tests they vouch for. |  |  |
| 2026-09-02 | G-03 | Gate | **G** | G | G | Release check builds the wrong binary, so it passes forever. |  |  |
| 2026-09-02 | G-04 | Gate | **G** | G | G | Ledger never promoted; regressions in required gates are non-blocking items. |  |  |
| 2026-09-02 | G-05 | Gate | **G** | G | G | Sweep ignores cell FAILs; hand-written 55-name checkset and realckpt -run pattern omit many families' parity gates. |  |  |
| 2026-09-02 | G-06 | Gate | **G** | G | G | Test tolerance far weaker than the bit-identical contract it gates; stale comment. |  |  |
| 2026-09-02 | G-07 | Gate | **G** | G | G | Test's stated configuration is not what runs; nothing asserts effective slot count. |  |  |
| 2026-09-02 | G-08 | Gate | **G** | G | G | Evidence test binds wrong buffers so it cannot test what it claims. |  |  |
| 2026-09-02 | G-09 | Gate | **G** | G | G | Gate's backend detection lists cuda\|metal\|none only; WebGPU parity never runs in the ritual. |  |  |
| 2026-09-02 | G-10 | Gate | **G** | G | G | Dominant: TestAPITiers substring match cannot fail; also TestTokenText zero-assertion PASS and Log-only measurement Test* harnesses. |  |  |
| 2026-09-02 | M-01 | Major | **R** | R | R | Byte accountant hand-enumerates fields and skips gemma4moe, delta/qattn, mamba, mla, shortConv, PLE; those families counted as dense default. | gemma4 MoE, qwen3.5, mamba-2, MLA, lfm2 conv, PLE | ResidentWeightBytes / Metal memory-fit guard |
| 2026-09-02 | M-02 | Major | **R** | R | R | Guard never reads GOINFER_METAL_MOE_SLOTS, judging paged models as unpaged default. Second half (one copy counted) is sizing N. | GOINFER_METAL_MOE_SLOTS (paged MoE) | residentFitsMemory before buildResident |
| 2026-09-02 | M-03 | Major | **O** | O | O | Per-token allocation on a path that never uses it: performance/memory churn. |  |  |
| 2026-09-02 | M-04 | Major | **O** | B | O | Untrusted snapshot-blob validation; the bound exists but the defect is unvalidated input. |  |  |
| 2026-09-02 | M-05 | Major | **N** | B | R | Llama-4 path handles the family but applies the wrong mask past chunk size; fix is chunk-start arithmetic, not a list/switch registration. |  |  |
| 2026-09-02 | M-06 | Major | **O** | R | R | Config validation of zero/NaN/Inf values and zero dims at the chokepoint; RoPE exemptions are incidental. Not a registration, bound or numerics miss. |  |  |
| 2026-09-02 | M-07 | Major | **R** | R | R | EAGLE prefill never consults the cpuFastAttention option, running default exact kernel; doc/missing-test claims are secondary G. | cpuFastAttention (--cpu-fast-attention default ON) | EAGLE prefill (captureN/forwardN) |
| 2026-09-02 | M-08 | Major | **N** | N | N | Sampler handles MinP but empty candidate set indexes -1; O (parameter validation) plausible. |  |  |
| 2026-09-02 | M-09 | Major | **R** | R | R | Five GGUF family branches never drive the sink and nothing refuses them; fix is a named needsResidentSerialize list plus refusal. | gpt-oss, laguna, granitehybrid, nemotron_h, llama4 | StreamTranscodeGGUF sink path / canSerialize |
| 2026-09-02 | M-10 | Major | **B** | B | B | Layer-count ceilings exist but laguna path and resolveArchitecture precede them; fatal OOM. Reopened M16 gap. | block_count / num_hidden_layers ceilings | ggufLagunaConfig, resolveArchitecture before maxSerializedLayers |
| 2026-09-02 | M-11 | Major | **R** | R | R | Validator lacks gpt-oss/etc tail fields and treats absent as allowed default; fix registers required fields. B (short vector) plausible. | gpt-oss AttnSinks/expert biases, gemma4 PLE, tail tensors | validateShapes |
| 2026-09-02 | M-12 | Major | **O** | O | O | Non-atomic write and weak cache freshness: robustness mechanics, no registration or bound. |  |  |
| 2026-09-02 | M-13 | Major | **B** | B | B | Burst width never clamped against MaxTokens or context cap; reachable input exceeds both. Gate blindness is secondary. | MaxTokens and resident context cap | BlockSpec.generate round width |
| 2026-09-02 | M-14 | Major | **R** | R | R | verifyTheta keys on backend name only, treating MoE/K=V/non-int4 models as batched default; perf only. O plausible. | MoE / K=V / non-uniform / non-int4 models on CUDA | verifyTheta |
| 2026-09-02 | M-15 | Major | **B** | B | B | Fixed +512 drafter KV capacity exceeded by longer generations with no admission check; N (sizing) plausible. | drafter kvCap = len(prompt)+512 | ExtendContext / DraftBlock |
| 2026-09-02 | M-16 | Major | **B** | B | B | Dynamic shared-memory request grows with keys past the 48 KB device limit; nothing checks it. | 48 KB dynamic shared memory per block (~12,160 keys) | single-block attention launch / split-KV admission |
| 2026-09-02 | M-17 | Major | **O** | O | O | No per-write deadline: resource exhaustion, not a registration or bound miss. |  |  |
| 2026-09-02 | M-18 | Major | **R** | R | R | Item decoder ignores type values function_call/_output, treating them as default user message. O (missing feature) plausible. | responses input item types function_call / function_call_output | /v1/responses item decoder |
| 2026-09-02 | M-19 | Major | **O** | O | O | Heartbeat feature missing on one route; a route is not an option, family or state. |  |  |
| 2026-09-02 | M-20 | Major | **N** | N | N | Path handles Gemma 4 tools but renders turn scaffolding and nested args wrong; nested default arm is R-like, O plausible. |  |  |
| 2026-09-02 | M-21 | Major | **O** | O | O | Resource exhaustion from late validation on unguarded routes; B unlikely since prompts are still rejected. |  |  |
| 2026-09-02 | M-22 | Major | **O** | O | O | Security/prompt-injection exposure; not a registration or bound miss. |  |  |
| 2026-09-02 | M-23 | Major | **O** | R | O | Shutdown is a cancel source, not an option, family or state; a lifecycle-semantics bug. |  |  |
| 2026-09-02 | M-24 | Major | **O** | O | O | Resource leak on the 409 path. |  |  |
| 2026-09-02 | M-25 | Major | **N** | N | N | Decoder knows the dummy-prefix rule but applies it to a continuation; R (continuation state unhandled) plausible. |  |  |
| 2026-09-02 | M-26 | Major | **R** | R | R | Option honoured on one stream path; tool/vision/completions paths treat it as default off. O plausible. | stream_options.include_usage | tool, vision and /v1/completions stream paths |
| 2026-09-02 | M-27 | Major | **N** | N | N | Masker handles scalar schemas but reads may-end as must-end; logic computed wrong. O plausible. |  |  |
| 2026-09-02 | M-28 | Major | **N** | O | N | Reflection builds wrong schemas for struct shapes it does handle. |  |  |
| 2026-09-02 | M-29 | Major | **N** | N | N | Path handles literals but encodes them wrong: HTML escaping and float64 precision loss. |  |  |
| 2026-09-02 | M-30 | Major | **O** | O | O | Request/API handling gap for a schema shape; not option/family/state. R plausible. |  |  |
| 2026-09-02 | M-31 | Major | **O** | O | B | A restated constant over-declines to CPU; the input is checked, nothing runs wrong. |  |  |
| 2026-09-02 | M-32 | Major | **R** | R | R | Family-specific KV allocation branches never consult --kv i8/f16 (and -ctx), allocating f32 default. Option x family branch. | --kv i8/f16 (and -ctx) | Nemotron/Qwen3.5/MLA KV allocation in BuildResident |
| 2026-09-02 | M-33 | Major | **O** | O | O | Release mechanics and licensing; NOTICE false claim (G) is secondary. |  |  |
| 2026-09-02 | M-34 | Major | **G** | G | G | Release doc claims a CI check that does not exist; demo/agent has no CI. |  |  |
| 2026-09-02 | M-35 | Major | **G** | G | G | Comments, REGEN.md and FMA-lint exemption claim a frozen audited artifact that no longer exists. O (toolchain process) plausible. |  |  |
| 2026-09-10 | C-01 | Critical | **R** | R | R | Batched-prefill admission never checked the bound LoRA adapter, running adapter prompts as the base default; fix adds a !hasAdapter decline at residentPrefillSeed. | LoRA adapter (--adapter) bound to a resident session | residentPrefillSeed Prefiller branch / backend prefillStaticDecline |
| 2026-09-10 | C-02 | Critical | **R** | R | R | Reuse predicate matched token ids only; the adapter that built the KV was unaccounted for and treated as base; fix records it and declines on mismatch. | adapter identity of the resident KV prefix | residentReuseLen / residentCommitIDs reuse rule |
| 2026-09-10 | C-03 | Critical | **R** | R | R | A new recurrent state kind (KDA) missing from the registry bit, hasRecurrentState and resetRecurrent; reuse, reset and rollback treat it as non-recurrent default. | KDA recurrent state (bailing_hybrid) | Recurrent registry bit, KVCache.hasRecurrentState, resetRecurrent |
| 2026-09-10 | C-04 | Critical | **R** | R | R | CUDA tails hard-code RMS and prefillStaticDecline omits layerNorm/parallelBlock, so the Cohere family runs as the default RMS path; fix adds dispatch and decline. | Cohere/Command-R (layerNorm, parallelBlock, logit_scale) | CUDA decode/prefill tail norm dispatch, prefillStaticDecline, prefill-tail logit scale |
| 2026-09-10 | C-05 | Critical | **R** | R | R | ggufQKPermuted name list lacked "granite", so its permuted q/k were loaded as the default unpermuted case; fix adds the family to the list. | granite (dense Granite 4.2 GGUF) | ggufQKPermuted family list |
| 2026-09-10 | C-06 | Critical | **R** | R | R | moeExpertGptOssDown had no s.w4 branch, so the int4 quant was dispatched as the int8 default; fix branches on w4 or declines. | int4 quant (W4A8 stacked experts) for gpt-oss on WebGPU | moeExpertGptOssDown dispatch (no s.w4 branch) |
| 2026-09-10 | C-07 | Critical | **R** | R | R | SetAdapter lifecycle never invalidated the executor's pre-encoded command buffer, so stale state ran; R (state x lifecycle path) over O (ordering hazard), but close. | pre-encoded Metal executor command buffer across adapter changes | SetAdapter (no stopExec/invalidation of execLoop's cur) |
| 2026-09-10 | C-08 | Critical | **R** | R | R | prefillOK admission checks features and geometry but never MoE paging, so --moe-cache-slots models take batched prefill; fix adds one predicate term. | paged generic MoE (--moe-cache-slots / MoE slot env vars) | Metal prefillOK batched-prefill admission |
| 2026-09-10 | C-09 | Critical | **N** | N | N | The i8 rope-store path handles m-RoPE but indexes the scale write by rope position instead of true position: an indexing error, not a missed registration. |  |  |
| 2026-09-10 | G-01 | Gate | **G** | G | G | The parity staleness gate hashes a hand-typed aikit_version that drifted from go.mod, so it stays green across aikit bumps. |  |  |
| 2026-09-10 | G-02 | Gate | **G** | G | G | A validated family's deps_hash covers no files, so the parity freshness gate can never restale it; the defect is in the gate's manifest. |  |  |
| 2026-09-10 | G-03 | Gate | **G** | G | G | CI skips gitignored fixtures, the sweep files FAILs as FIRST-RUN and the ledger is unpromoted, so the parity gates cannot block anything. |  |  |
| 2026-09-10 | G-04 | Gate | **G** | G | G | The refresh refusal is bypassable and -update has none, so deps_hash is refreshed while required goldens are red; evidence-chain gate defect. |  |  |
| 2026-09-10 | G-05 | Gate | **G** | G | G | The tests compare the registry bit against hand-written family lists, so a family missing from both passes; incomplete hand list, r-shaped. |  |  |
| 2026-09-10 | G-06 | Gate | **G** | G | G | The gates drive per-token Forward, never PrefillLast, use a 3-token fake prompt, and bind Metal on a fresh executor, so they cannot fail. |  |  |
| 2026-09-10 | G-07 | Gate | **G** | G | G | A bias-dominated fixture and 0.95 cosine floor sit below the damage of a dropped expert matmul term, so the gate cannot fail on C-06. |  |  |
| 2026-09-10 | G-08 | Gate | **G** | G | G | The decision gate drops missing-reference cells and turns a build decline into SKIP yet prints SHIPS; only the first half reproduced. |  |  |
| 2026-09-10 | G-09 | Gate | **G** | G | G | The only PTX freshness check is NVRTC-patch-bound and its absence is a counted skip on a PASS; CI has no step. |  |  |
| 2026-09-10 | G-10 | Gate | **G** | G | G | The webgpu-parity cell's selector reaches only ResidentParity, so seven gate-shaped WebGPU tests run in no gate cell or CI runner. |  |  |
| 2026-09-10 | G-11 | Gate | **G** | G | G | The lint regex misses c + a*b, so a bare float MAC ships in a linted kernel; filed against the lint, not the kernel. |  |  |
| 2026-09-10 | G-12 | Gate | **G** | G | G | The doctor's tools row passes turn two on any 200 and the stop row passes when the stop never fires; checks don't check their claim. |  |  |
| 2026-09-10 | G-13 | Gate | **G** | G | G | All eleven are vacuous tests/checks. Dominant (b) hand-lists 15/22 kernel modules; (c) hand-counts GGUF family cases (r-shaped); rest are stale claims, vacuous skips, grep checks. |  |  |
| 2026-09-10 | M-01 | Major | **R** | R | R | The cap guard's residentPath keyed adapters (and VL) as staged, so resident routes skipped ResidentContextCap; R (route unregistered) vs B (cap unchecked) nearly equal. | LoRA adapter route and reused-image VL route running resident | prepare residentPath predicate (lm.adapter == "") / P9a full-reuse branch |
| 2026-09-10 | M-02 | Major | **O** | R | R | VL clamp sites simply fail to set Budget/BudgetClamped; fix is two field writes, not a list/predicate registration. Missing propagation. |  |  |
| 2026-09-10 | M-03 | Major | **B** | B | B | Draft width is not clamped to the resident context cap before ForwardN, so a reachable round exceeds it and errors; the fix is a clamp. | resident context cap (ResidentContextCap) | n-gram and draft-model spec loops' draft width |
| 2026-09-10 | M-04 | Major | **O** | R | R | Resolved EOS set never propagated into Cfg before serialization; fix is a value write-back, not a registration. Data-propagation defect. |  |  |
| 2026-09-10 | M-05 | Major | **B** | B | B | A delta's In/Out can exceed the base projection's buffers with no check, giving OOB reads/writes or panics; fix is a load-time bound refusal. B over O. | base projection In/Out dimensions vs adapter delta shape | validateTargets / LoadAdapter / applyLoRA (names only, no shape check) |
| 2026-09-10 | M-06 | Major | **N** | R | R | A 0 == 0 sentinel comparison in a path that handles image-hash reuse; the fix changes a comparison, it registers nothing. |  |  |
| 2026-09-10 | M-07 | Major | **R** | R | R | The Gemma 4 VL commit omits the image-block record its siblings register, so the reuse scan treats image positions as plain text. | Gemma 4 VL image block | GenerateGemma4VL residentCommitIDs (nil image block) |
| 2026-09-10 | M-08 | Major | **R** | R | R | Plan omits ResidentEligible's checks, so ineligible families and unrecognised backends fall through to the default 'resident' verdict. | families/backends ResidentEligible declines (Llama-4/cuda, Gemma 4/webgpu, Kimi-K2/metal, unknown backend) | Model.Plan admission (MissingResidentFeatures only) |
| 2026-09-10 | M-09 | Major | **R** | R | R | Spec admission did not know staged webgpu int4 splits decode and verify kernels, breaking verify==decode; R vs G (false bit-identity comment) close. | staged WebGPU int4/int4mix | SpecDecodeConflict / spec-decode admission |
| 2026-09-10 | M-10 | Major | **R** | R | R | The fast-lever gate didn't know the HiddenLast tail needs exact kernels, so long prompts took the non-bit-identical path; R vs G (no gate). | tailHiddenLast prefill tail (/v1/embeddings) | useAttnFused/useGemmMMA fast-lever gate |
| 2026-09-10 | M-11 | Major | **R** | R | R | The fast-lever dispatch has no exact-only case for verify tails; exactness relied on M<16. R over B: 16 is a perf floor, not a capacity. | spec-verify tails (tailAllLogits/tailAllArgmax) | useGemmMMA/useAttnFused fast-lever gate |
| 2026-09-10 | M-12 | Major | **N** | N | N | The planner's sizing arithmetic omits checkKVFits' margin, so interior ctx choices always fail the build; the build check exists, so not B; N vs O(perf). |  |  |
| 2026-09-10 | M-13 | Major | **O** | O | O | A missing write deadline lets a stalled client pin the worker and decode mutex: a hang/resource issue, not registration, bound or numerics. |  |  |
| 2026-09-10 | M-14 | Major | **G** | G | G | Server message and task doc promise --fit=off, which the bool parser rejects: the rubric's flag-value claim example; chat lacking --fit is secondary. |  |  |
| 2026-09-10 | M-15 | Major | **B** | R | B | The pre-tokenize byte budget exists; tool schemas and arguments reach BPE unpriced. | tools schemas and replayed tool_calls arguments | pre-tokenize size guard (chatInputBytes/anthropicInputBytes) |
| 2026-09-10 | M-16 | Major | **R** | R | R | /v1/responses lacked the named 400 decline its siblings have, silently treating tools as the absent default; R vs O (request validation). | tools on a template with no tool form | serveResponsesWith tools guard |
| 2026-09-10 | M-17 | Major | **O** | R | R | Loader called with empty Options; fix passes cfg.backend and a quant mapping through: missing wiring, not list/predicate registration. |  |  |
| 2026-09-10 | M-18 | Major | **O** | O | O | Dominant defect is a per-image device-scratch and stream leak; the cuda-not-enabled wiring (R-shaped) and stale DONE doc are secondary. |  |  |
| 2026-09-10 | M-19 | Major | **N** | N | N | The free-RAM budget is measured after the model loads, so the CPU fit is priced against a consumed budget: sizing error, not a missing check. |  |  |
| 2026-09-10 | M-20 | Major | **N** | N | O | The printed rate divides by prefill+decode time (first fit in order); the double load is the O half. |  |  |
| 2026-09-10 | M-21 | Major | **O** | O | O | Device-buffer leak on regrow; no unregistered option/family/state and no fixed limit, so not R/B; leaks are O. |  |  |
| 2026-09-10 | M-22 | Major | **N** | R | N | Reservation already handles the drafter (prices weights); omitting K/V term is sizing arithmetic, fix adds a multiplication term, not a registration. N. |  |  |
| 2026-09-10 | M-23 | Major | **R** | R | R | FeatPerLayerRoPE (every mixed local/global Gemma) missing from prefillFeatures allowlist; fix adds it. Effect is lost batched TTFT only. R vs O(perf). | Gemma 3 (FeatPerLayerRoPE feature) | metal prefillFeatures / prefillOK batched-prefill admission |
| 2026-09-10 | M-24 | Major | **R** | R | R | Guard treats mmap-aliased .giw weights as the default heap-copied case; fix is an exclusion predicate on MmapByteOffset. R vs N (sizing arithmetic). | .giw mmap-aliased weight payloads | ResidentHostCopyBytes / Metal memory-fit guard |
| 2026-09-10 | M-25 | Major | **R** | R | R | residentQuantLabel predicate lists only int8int8; int8/int4mix on Metal fall to the default raw label though also requantized. Fix widens the predicate. | --quant int8, int4mix on Metal | residentQuantLabel (and chat/gemma load banners) |
| 2026-09-10 | M-26 | Major | **O** | O | O | Missing flag and library option (feature gap); no existing option value is mishandled, so not R. Task-doc claim is secondary. G vs O. |  |  |
| 2026-09-10 | M-27 | Major | **R** | R | R | Router must stay f32, but generic GGUF MoE loader branches and matmulQuant's ffn_ predicate treat it as an ordinary FFN matrix; fix registers the exception. R vs N. | GGUF MoE router (ffn_gate_inp) on GLM/Mellum/Qwen3-MoE/DeepSeek/Laguna/Granite-H/Nemotron/Llama-4 | generic GGUF MoE loader branches (mat vs streamMat quantNone) and matmulQuant |
| 2026-09-10 | M-28 | Major | **R** | R | R | KV pricing treats hybrid/sliding-window/MLA families as the default full-attention case; fix makes the guard know per-layer kinds. | hybrid (DeltaNet/conv/Mamba), sliding-window and MLA families | fit guard kvBytesPerPosition / AdmitPrefillMemory |
| 2026-09-10 | M-29 | Major | **R** | R | R | Tensors loaders keep f32 or packed (Mamba-2, MLA, GPTQ/AWQ, vision tower) are priced as default quantized weights; fix adds name predicates and pack factor. | f32-pinned Mamba-2/MLA mixers, GPTQ/AWQ qweight, vision tower | fit guard weight estimators (estimateGGUF/SafetensorsWeightBytes) |
| 2026-09-10 | M-30 | Major | **R** | R | R | remedy() assumes only .gguf reaches a refusal; safetensors-directory source gets the default gguf remedy. Fix branches on source. R vs G (wrong message). | safetensors directory model source | fitguard remedy() / -stream-weights directory handling |
| 2026-09-10 | M-31 | Major | **R** | R | R | GGUF transcode branch hardcodes false, treating -embed-int4 as default; an option x path miss per rubric. | -embed-int4 | internal/prequant Transcode GGUF branch |
| 2026-09-10 | M-32 | Major | **O** | O | O | Integrity pin dropped by a ref round-trip: security/integrity plumbing, not an option/family/state registration miss. R vs O. |  |  |
| 2026-09-10 | M-33 | Major | **O** | O | O | Non-atomic write / crash robustness on one branch; no option/family/state guard miss or fixed limit. |  |  |
| 2026-09-10 | M-34 | Major | **R** | R | R | Families build a pager but their MoE loops are absent from the touch hook list, so the RAM bound is unenforced; fix registers touch or a nil-pager decline. | gpt-oss, Llama 4, Nemotron 3 Nano | expert pager touch hooks in MoE forward loops |
| 2026-09-10 | M-35 | Major | **R** | R | R | n_shared_experts=0 is accepted elsewhere but the WebGPU builder assumes a shared expert (default case); fix gates on shInter>0. | Nemotron-H MoE with n_shared_experts=0 | gpu/residency.go Nemotron-H MoE BuildResident branch |
| 2026-09-10 | M-36 | Major | **R** | R | R | chat.Detect has no branch for these families and routes them to default Mistral/ChatML renderers silently; fix adds routing or a decline. | Ministral 3, SmolLM3, Olmo 3 | chat.Detect template dispatch |
| 2026-09-10 | M-37 | Major | **N** | N | N | Harmony renderer handles gpt-oss but renders multi-turn history wrong; the doc-claim and missing-golden parts are secondary. N vs G. |  |  |
| 2026-09-10 | M-38 | Major | **N** | N | N | Per-family vision prompt builders know each family but splice the wrong newline layout around the image block; output computed wrong. |  |  |
| 2026-09-10 | M-39 | Major | **N** | N | N | Path renders the template then encodes with addBOS=true unconditionally, a wrong token sequence on a handled path. N vs R. |  |  |
| 2026-09-10 | M-40 | Major | **O** | O | O | Code holds no depth limit at all; unbounded request-driven resource use is O, not B, per rubric. |  |  |
| 2026-09-10 | M-41 | Major | **R** | R | R | Declined-tokenizer state is unhandled by chat/agent/gemma-web/embed load paths and treated as normal; fix adds the named decline. R vs O. | PreTokenizerDecline (unrecognised GGUF pre-tokenizer) | load/newSession paths in goinfer-chat, demo/agent, gemma-web, examples/embed |
| 2026-09-10 | M-42 | Major | **G** | G | G | Release workflow gate and post-publish check do not verify the promised embedded assets. G (release check) vs O (release mechanics). |  |  |
| 2026-09-10 | M-43 | Major | **O** | O | O | Release procedure ordering is release mechanics, not a check failing to check. G vs O. |  |  |
| 2026-09-10 | M-44 | Major | **G** | G | G | CI job's scan does not check what its header claims (the shipped GPU artifacts). |  |  |
| 2026-09-10 | M-45 | Major | **G** | G | G | CI gate compares README at HEAD with the @latest binary, so it does not check what it claims. |  |  |
| 2026-09-10 | M-46 | Major | **O** | O | O | Dependency provenance / module mechanics; no product path miss, limit, numerics or gate claim. |  |  |
| 2026-09-10 | M-47 | Major | **G** | G | G | README gives a flag value the parser rejects, the rubric's direct G example. |  |  |
| 2026-09-10 | M-48 | Major | **R** | R | R | exact-prefill handler sets Metal/CPU knobs but not CUDA's, so CUDA runs the default fast path; fix adds CUDA. R vs G (help claim). | --exact-prefill | serve exact-prefill env-setting block (CUDA knob missing) |
| 2026-09-10 | M-49 | Major | **G** | G | G | Help text and server.md state the opposite of the code; a docs/help claim that is false. |  |  |
| 2026-09-10 | M-50 | Major | **G** | G | G | server.md, -quant help and book ch. 5 make a false RAM claim; docs/help text claim. |  |  |
| 2026-09-10 | M-51 | Major | **G** | G | G | Help text on both binaries makes false capability claims. |  |  |
| 2026-09-10 | M-52 | Major | **G** | G | G | Help text states a default the binary no longer uses, and docs do not record the change. |  |  |
| 2026-09-10 | M-53 | Major | **G** | G | O | CHANGELOG, the release contract, omits default flips and flags it exists to record. |  |  |
| 2026-09-10 | M-54 | Major | **G** | G | G | Doc status header and server.md paragraph make stale, false claims contradicted by the body and code. |  |  |
| 2026-09-10 | M-55 | Major | **G** | G | G | Doc's exactness promise is false for CUDA/Metal resident reuse; docs-only claim fix. |  |  |
| 2026-09-10 | M-56 | Major | **G** | G | G | Doc claims a no-op knob is operative, and the phantom-check test counts a test-only reference; both G. |  |  |
| 2026-09-10 | M-57 | Major | **G** | G | G | A 'current claims' doc states a stale count; docs claim that is false. |  |  |
| 2026-09-10 | M-58 | Major | **G** | G | G | Published book chapters quote withdrawn or superseded figures; docs claim that is false. |  |  |
| Metal 2026-09-12 | M-01 | Performance — Major | **O** | O | O | Missing feature: interface never implemented, so prefill tokens pay unneeded head/readback; speed only, no registration miss or wrong output. |  |  |
| Metal 2026-09-12 | M-02 | Performance — Major | **O** | O | O | Defect is a too-high perf threshold constant; its stale comment is secondary, so O not G (O vs G). |  |  |
| Metal 2026-09-12 | M-03 | Performance — Major | **O** | O | O | Kernel is slow (lane occupancy, barrier ratio, no operand staging); pure performance. |  |  |
| Metal 2026-09-12 | M-04 | Performance — Major | **O** | O | O | Barrier-heavy scalar rescale and redundant K/V reads make the kernel slow; pure performance. |  |  |
| Metal 2026-09-12 | M-05 | Performance — Major | **O** | O | O | Row-serial expert dispatch costs bandwidth/dispatches; performance plus missing documentation, no wrong output. |  |  |
| Metal 2026-09-12 | M-06 | Performance — Major | **R** | R | R | Supported feature absent from admission map, so Gemma 3 falls to default decline; fix adds one map entry. Consequence is perf only (R vs O). | Gemma 3 (FeatPerLayerRoPE) | metal prefillFeatures batched-prefill admission map |
| Metal 2026-09-12 | M-07 | Performance — Major | **O** | O | O | Redundant host row4/canonical copies waste memory footprint; matches the 'copy is redundant' O example rather than a registration miss. |  |  |
| Metal 2026-09-12 | M-08 | Performance — Major | **O** | O | O | Single-threadgroup serial rank loop is slow; performance, re-opening prior P-11's Metal disposition. |  |  |
| Metal 2026-09-12 | M-09 | Performance — Major | **O (withdrawn)** | O | O | Proposed load-pattern lever is a performance question; probe found staged variant slower. |  |  |
| Metal 2026-09-12 | M-10 | Performance — Major | **O (withdrawn)** | O | O | Dispatch-site kernel swap for speed; end-to-end probe showed no win. |  |  |
| Metal 2026-09-12 | M-11 | Performance — Major | **O** | O | O | Per-command-buffer submit overhead on the paged path; performance (merged pread-staging body also perf). |  |  |
| Metal 2026-09-12 | M-13 | Performance — Major | **O** | O | O | Missing auto-sizing default (option documented CUDA-only), not a pure registration; R also arguable since Metal never reads MoECacheExperts (R vs O). |  |  |
| Metal 2026-09-12 | M-14 | Performance — Major | **O** | O | O | Missing binding selector forces queue-wide residency attach costing time per token; performance. |  |  |
| Metal 2026-09-12 | M-15 | Performance — Major | **O** | O | O | Missing/slow GPU vision path; not wiring it is deliberate and correct today; performance and missing feature. |  |  |
| Metal 2026-09-12 | M-16 | Performance — Major | **O (withdrawn)** | O | O | Hypothesised per-dispatch overhead lever; performance; probe showed no effect. |  |  |
| Metal 2026-09-12 | C-01 | Correctness | **N** | N | N | Kernel handles the input but full 8-row tile reads overrun a cache sized without tile round-up: sizing/indexing arithmetic (N vs B). |  |  |
| Metal 2026-09-12 | C-02 | Correctness | **R** | R | R | Paged-MoE mode unaccounted for on three entry points; encodeLayer treats it as non-paged default, yielding garbage; fix is a named decline. | paged MoE residency (--moe-cache-slots) | HiddenLast/Forward/ForwardArgmax via encodeLayer (no paged branch) |
| Metal 2026-09-12 | C-03 | Correctness | **B** | B | B | Projection dims are a size limit a mismatched adapter exceeds unchecked, writing past Q into K/V; fix is a bound check (B before O validation). | adapter In/Out vs base projection K/N | metal SetAdapter conv / decoder LoadAdapter (rank-only check) |
| Metal 2026-09-12 | C-04 | Correctness | **O** | O | O | Resource leak on an error path; matches the O leak example. |  |  |
| Metal 2026-09-12 | C-05 | Correctness | **O** | O | O | Missing thread pin risks crash/race; not caused by an option/family/state or bound miss. |  |  |
| Metal 2026-09-12 | C-06 | Correctness | **B** | B | B | Patch count above 7,680 exceeds 32 KiB threadgroup memory with nothing checking; fix is a bound check/decline. | 32 KiB threadgroup memory vs SigLIP patch count (np > 7,680) | visionmetal newEncoder attention dispatch (no attnThreadgroupBytes guard) |
| Metal 2026-09-12 | G-01 | Gate | **G** | G | G | Test fails before reaching the MoE check it exists for; a permanently red gate gates nothing. |  |  |
| Metal 2026-09-12 | G-02 | Gate | **G** | G | G | Release gate silently narrows its decision set and can pass without its decision cells. |  |  |
| Metal 2026-09-12 | G-03 | Gate | **G** | G | G | Gate fixture omits the global layer, so it never covers the FeatPerLayerRoPE variant real Gemma 3 has. |  |  |
| Metal 2026-09-12 | G-04 | Gate | **G** | G | G | Golden regenerated by the code under test cannot detect drift, and its audit trail is missing. |  |  |
| Metal 2026-09-12 | G-05 | Gate | **G** | G | G | Measurement verdict carried as settled in comments was taken on a shape where the cost does not exist. |  |  |
| Metal 2026-09-12 | G-06 | Gate | **G** | G | G | LedgerLen is emptied whether or not native release happens, so the leak test cannot fail. |  |  |
| Metal 2026-09-12 | G-07 | Gate | **G** | G | G | Superseded gate cannot run and three files cite a test that does not exist. |  |  |
| Metal 2026-09-12 | G-08 | Gate | **G** | G | G | Coverage gap in the fidelity gate; later coverage found no product defect but the gap was real. |  |  |
| Metal 2026-09-12 | G-09 | Gate | **G** | G | G | Measurement harness never run and declines the shape that matters; MoE default-ON claim rests on dense cells. |  |  |
| Metal 2026-09-12 | G-10 | Gate | **G (withdrawn)** | G | G | Status-latch gate cannot fire on this hardware; documentation observation about a gate. |  |  |
| Metal 2026-09-30 | A-C01 | Critical (non-default `-kv i8`) | **R** | R | R | Product admission guard prefillOK/PrefillLast never checks the -kv i8 option, treating int8 KV as default f16; overrun and wrong K/V are caused by that miss. | -kv i8 (r.kvI8, int8 KV cache layout) | prefillOK / PrefillLast batched-prefill admission |
| Metal 2026-09-30 | F-C02 | Critical (non-default: ctx > 4096 with hd > 128, or fused attention off) | **B** | B | B | No unregistered option/family; a fixed 4096-float threadgroup score array is exceeded by reachable contexts up to 32768 with no check. | threadgroup float sc[4096] in exact attention_prefill (nKeys > 4096) | PrefillLast bounds only startPos+len <= ctxCap, not the exact kernel's 4096-key array |
| Metal 2026-09-30 | A-P01 | Major | **O** | O | O | Pure performance: GEMM tile shape and occupancy at small M; no registration miss, limit or wrong numerics. |  |  |
| Metal 2026-09-30 | A-P02 | Major | **O** | O | O | Performance: short prompts take the slower exact path because no fidelity cell was gated; correct output, no miss or bound. |  |  |
| Metal 2026-09-30 | A-P03 | Major if confirmed | **O** | O | O | Performance/occupancy question of the steel kernel at small M; the stale table is evidence, not the defect. |  |  |
| Metal 2026-09-30 | A-P04 | Major (owner call) | **O** | O | O | Performance of the embeddings path (no pipelining); output is correct, no registration miss or bound. |  |  |
| Metal 2026-09-30 | B-P01 | Major | **O** | O | O | canUseAttnFA explicitly declines hd!=128 by recorded scope and the legacy path is correct; fix is a new templated kernel, so performance. |  |  |
| Metal 2026-09-30 | B-P02 | Major | **O** | O | O | Other G fall to the correct but slower attention_fa by deliberate scope; performance, not a correctness registration miss. R vs O considered. |  |  |
| Metal 2026-09-30 | B-P03 | Major (REVISIT) | **O** | O | O | Stale performance crossover threshold; both kernels compute correctly, no limit exceeded, so performance tuning. |  |  |
| Metal 2026-09-30 | B-P04 | Major | **O** | O | O | Kernel lane-utilisation performance at K=1536/3584; no wrong output or bound. |  |  |
| Metal 2026-09-30 | B-P05 | Major (REVISIT of M-10) | **O** | O | O | Revisit of a performance negative: down GEMV bandwidth; correctness unaffected. |  |  |
| Metal 2026-09-30 | C-P01 | Major | **O** | O | O | Memory footprint duplication with no wrong output and no fixed limit exceeded; resource use, so O. |  |  |
| Metal 2026-09-30 | C-C01 | Major (conditional) | **R** | R | R | Planner's per-backend ceiling exists for webgpu but not metal, treating Metal as default (no ceiling, f32 KV); fix registers it. R vs B: Metal itself checks the cap. | -backend metal (32768 context ceiling, f16 KV) | Plan("metal") / guardGIWFit / guardFit ceiling and KV-precision pricing |
| Metal 2026-09-30 | D-P01 | Major | **O** | O | O | prefillOK declines paged MoE by name, pinned as intended; the cost is performance/missing feature, not a registration miss. |  |  |
| Metal 2026-09-30 | D-B01 | Major | **O** | O | O | DeltaNet is explicitly declined from PrefillLast and the sequential path is correct; missing feature and performance. |  |  |
| Metal 2026-09-30 | D-B02 | Major | **O** | O | O | Dispatch fan-out, per-layer sync and tile padding are performance; no correctness miss or bound. |  |  |
| Metal 2026-09-30 | D-P04 | Major | **O** | O | O | Single-thread routing is correct but slow; performance revisit of a deliberate design. |  |  |
| Metal 2026-09-30 | D-B03 | Major | **O** | O | O | Deliberate lossy requant; severity rests on 0.6 GB memory plus unquantified fidelity, a missing native feature. N vs O: not an arithmetic bug. |  |  |
| Metal 2026-09-30 | E-P01 | Major | **O** | O | O | Performance: an existing exact kernel could serve short prompts faster; no miss, bound or wrong numerics. |  |  |
| Metal 2026-09-30 | D-G01 | Major (gate) | **G** | G | G | Defect is in the gate: dense-model fidelity gate and weak tiny-fixture cosine tests cannot see the MoE path; no hand-written list omission. |  |  |
| Metal 2026-09-30 | F-G01 | Major (gate) | **G** | G | G | Test/gate coverage defect: only log-only, env-gated harnesses exercise attention_prefill_steel. |  |  |
| Metal 2026-09-30 | F-G02 | Major (gate) | **G** | G | G | Gates skip by default and the cited invariance record measured a different kernel; a gate/record defect. |  |  |
| Metal 2026-09-30 | F-G03 | Major (gate) | **G** | G | G | A test documented as guaranteeing the ceiling checks constants only and cannot see F-C02; claim/test mismatch. |  |  |
